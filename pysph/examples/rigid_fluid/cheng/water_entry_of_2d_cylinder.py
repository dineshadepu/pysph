from __future__ import print_function
import os
import numpy as np
from pysph.examples._db_geometry import DamBreak2DGeometry

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application

from pysph.sph.equation import Group
from pysph.sph.cheng import (
    ContinuityEquationFluid, ContinuityEquationSolid, MomentumEquationFluid,
    MomentumEquationSolid, StateEquation, get_particle_array_fluid_cheng,
    RigidFluidForce, SourceNumberDensity, SolidWallPressureBC,
    SetFreeSlipWallVelocity, SetNoSlipWallVelocity, SetNoSlipWallVelocityAdami,
    RK2ChengFluidStep)
from pysph.base.kernels import QuinticSpline
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import EPECIntegrator
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.tools.geometry_rigid_fluid import (get_2d_hydrostatic_tank)

# rigid body imports
from pysph.dem.discontinuous_dem.dem_nonlinear import (
    EPECIntegratorMultiStage, EulerIntegratorMultiStage)
from pysph.sph.equation import Group, MultiStageEquations
from pysph.solver.application import Application

from pysph.sph.rigid_body import (BodyForce)

from pysph.sph.rigid_body_cundall_2d import (SumUpExternalForces)
from pysph.sph.rigid_body_cundall_3d import (
    get_particle_array_rigid_body_cundall_dem_3d,
    RigidBodyCollision3DCundallParticleParticleStage1,
    RigidBodyCollision3DCundallParticleParticleStage2,
    UpdateTangentialContactsCundall3dPaticleParticle,
    RK2StepRigidBodyQuaternionsDEMCundall3d)

from pysph.sph.wall_normal import (ComputeNormals, SmoothNormals)


def create_circle(diameter=1, spacing=0.05, center=None):
    dx = spacing
    x = [0.0]
    y = [0.0]
    r = spacing
    nt = 0
    while r < diameter / 2:
        nnew = int(np.pi * r**2 / dx**2 + 0.5)
        tomake = nnew - nt
        theta = np.linspace(0., 2. * np.pi, tomake + 1)
        for t in theta[:-1]:
            x.append(r * np.cos(t))
            y.append(r * np.sin(t))
        nt = nnew
        r = r + dx
    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


class RigidFluidCoupling(Application):
    def initialize(self):
        # dimensions
        self.tank_height = 0.5
        self.tank_length = 0.3
        self.fluid_height = 0.3
        self.spacing = 0.01
        self.layers = 4

        self.cylinder_radius = 0.05
        self.cylinder_diameter = 2. * self.cylinder_radius
        self.cylinder_spacing = self.spacing / 2.
        self.cylinder_rho = 1. * 1e3

        self.Umax = np.sqrt(2 * 9.81 * self.fluid_height)
        self.c0 = 10. * self.Umax
        self.dx = self.spacing
        self.hdx = 1.2
        self.rho0 = 1000
        self.m = 1000 * self.dx * self.dx
        self.alpha = 0.2
        self.beta = 0.0
        self.eps = 0.5
        self.gamma = 7
        self.p0 = self.c0 * self.c0 * self.rho0
        self.nu = 1. / 500
        self.gy = -9.81
        self.dim = 2

        h0 = self.hdx * self.dx
        self.tf = 5.

        dt_cfl = 0.25 * h0/(self.c0 + self.Umax)
        dt_viscous = 0.125 * h0**2/self.nu

        self.dt = 0.5 * min(dt_cfl, dt_viscous)
        print("time step is :", self.dt)

    def create_particles(self):
        xt, yt, xf, yf = get_2d_hydrostatic_tank(
            ht_length=self.tank_height, ht_height=self.tank_height,
            fluid_height=self.fluid_height, spacing=self.spacing,
            layers=self.layers)
        # import matplotlib.pyplot as plt
        # plt.scatter(xt, yt)
        # plt.scatter(xf, yf)
        m = self.rho0 * self.dx * self.dx
        rho = self.rho0
        h = self.hdx * self.dx
        fluid = get_particle_array_fluid_cheng(x=xf, y=yf, h=h, m=m, rho=rho,
                                               name="fluid")

        m = self.rho0 * self.dx * self.dx
        rho = self.rho0
        h = self.hdx * self.dx
        tank = get_particle_array_fluid_cheng(x=xt, y=yt, h=h, m=m, rho=rho,
                                              name="tank")

        tank.add_property('normal_x')
        tank.add_property('normal_y')
        tank.add_property('normal_z')
        tank.add_property('normal_tmp_x')
        tank.add_property('normal_tmp_y')
        tank.add_property('normal_tmp_z')
        # add properties to tank for Adami boundary boundary condition
        for prop in ('ug', 'vg', 'wg', 'uf', 'vf', 'wf', 'wij'):
            tank.add_property(name=prop)

        # create rigid body
        xc, yc = create_circle(self.cylinder_diameter, self.cylinder_spacing)
        yc = yc + self.fluid_height + self.cylinder_diameter
        xc = xc + self.tank_length / 2.
        # plt.scatter(xc, yc)
        # plt.show()
        m = self.cylinder_rho * self.cylinder_spacing**2
        h = self.hdx * self.cylinder_spacing
        rad_s = self.cylinder_spacing / 2.
        cylinder = get_particle_array_rigid_body_cundall_dem_3d(
            x=xc, y=yc, h=h, m=m, rho=self.cylinder_rho, rad_s=rad_s, dem_id=1,
            name="cylinder")

        cylinder.add_property('normal_x')
        cylinder.add_property('normal_y')
        cylinder.add_property('normal_z')
        cylinder.add_property('normal_tmp_x')
        cylinder.add_property('normal_tmp_y')
        cylinder.add_property('normal_tmp_z')

        cylinder.set_output_arrays([
            'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au',
            'av', 'aw', 'tag', 'gid', 'fx', 'fy', 'fz', 'body_id',
            'normal_x', 'normal_y', 'normal_z'
        ])

        # add properties to boundary for Adami boundary boundary condition
        for prop in ('ug', 'vg', 'wg', 'uf', 'vf', 'wf', 'wij'):
            cylinder.add_property(name=prop)

        return [fluid, tank, cylinder]

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegratorMultiStage(
            fluid=RK2ChengFluidStep(),
            cylinder=RK2StepRigidBodyQuaternionsDEMCundall3d())

        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
                        dt=self.dt, tf=self.tf)

        return solver

    def create_equations(self):
        stage1 = [
            # compute the normals
            # Group(equations=[
            #     ComputeNormals(dest='cylinder', sources=['cylinder'])
            # ]),
            # Group(equations=[
            #     ComputeNormals(dest='tank', sources=['tank'])
            # ]),
            # Group(equations=[
            #     SmoothNormals(dest='cylinder', sources=['cylinder'])
            # ]),
            # Group(equations=[
            #     SmoothNormals(dest='tank', sources=['tank'])
            # ]),

            Group(equations=[
                SourceNumberDensity(dest='tank', sources=['fluid']),
                SolidWallPressureBC(dest='tank', sources=['fluid'], gy=self.gy,
                                    c0=self.c0, rho0=self.rho0),
                # SetFreeSlipWallVelocity(dest='tank', sources=['fluid']),
                # SetNoSlipWallVelocity(dest='tank', sources=['fluid']),
                SetNoSlipWallVelocityAdami(dest='tank', sources=['fluid']),

                # --------- Set the pressure and velocity of dummy particles
                # of rigid body ------------#
                SourceNumberDensity(dest='cylinder', sources=['fluid']),
                SolidWallPressureBC(dest='cylinder', sources=['fluid'],
                                    gy=self.gy, c0=self.c0, rho0=self.rho0),
                # SetFreeSlipWallVelocity(dest='cylinder', sources=['fluid']),
                # SetNoSlipWallVelocity(dest='cylinder', sources=['fluid']),
                SetNoSlipWallVelocityAdami(dest='cylinder', sources=['fluid']),
            ]),
            Group(equations=[
                # apply gravity to rigid body
                BodyForce(dest='cylinder', sources=None, gy=-9.81),
                StateEquation(dest='fluid', sources=None, c0=self.c0,
                              rho0=self.rho0),
                ContinuityEquationFluid(dest='fluid', sources=['fluid'],
                                        c0=self.c0, alpha=0.2),
                # ContinuityEquationFluid(dest='fluid', sources=[
                #     'tank', 'cylinder'
                # ], c0=self.c0, alpha=0.2),
                ContinuityEquationSolid(dest='fluid', sources=[
                    'tank', 'cylinder'
                ], c0=self.c0, alpha=0.2),
                MomentumEquationFluid(dest='fluid', sources=['fluid'], c0=self.
                                      c0, rho0=self.rho0, dim=self.dim,
                                      alpha=0.2, nu=self.nu, gy=self.gy),
                MomentumEquationSolid(
                    dest='fluid', sources=['tank', 'cylinder'], c0=self.c0,
                    rho0=self.rho0, dim=self.dim, alpha=0.2, nu=self.nu),
                RigidFluidForce(dest='cylinder', sources=['fluid']),
                SumUpExternalForces(dest='cylinder', sources=None)
            ])
        ]

        stage2 = [
            # compute the normals
            # Group(equations=[
            #     ComputeNormals(dest='cylinder', sources=['cylinder'])
            # ]),
            # Group(equations=[
            #     ComputeNormals(dest='tank', sources=['tank'])
            # ]),
            # Group(equations=[
            #     SmoothNormals(dest='cylinder', sources=['cylinder'])
            # ]),
            # Group(equations=[
            #     SmoothNormals(dest='tank', sources=['tank'])
            # ]),
            Group(equations=[
                SourceNumberDensity(dest='tank', sources=['fluid']),
                SolidWallPressureBC(dest='tank', sources=['fluid'], gy=self.gy,
                                    c0=self.c0, rho0=self.rho0),
                # SetFreeSlipWallVelocity(dest='tank', sources=['fluid']),
                # SetNoSlipWallVelocity(dest='tank', sources=['fluid']),
                SetNoSlipWallVelocityAdami(dest='tank', sources=['fluid']),

                # --------- Set the pressure and velocity of dummy particles
                # of rigid body ------------#
                SourceNumberDensity(dest='cylinder', sources=['fluid']),
                SolidWallPressureBC(dest='cylinder', sources=['fluid'],
                                    gy=self.gy, c0=self.c0, rho0=self.rho0),
                # SetFreeSlipWallVelocity(dest='cylinder', sources=['fluid']),
                # SetNoSlipWallVelocity(dest='cylinder', sources=['fluid']),
                SetNoSlipWallVelocityAdami(dest='cylinder', sources=['fluid']),
            ]),
            Group(equations=[
                # apply gravity to rigid body
                BodyForce(dest='cylinder', sources=None, gy=-9.81),
                StateEquation(dest='fluid', sources=None, c0=self.c0,
                              rho0=self.rho0),
                ContinuityEquationFluid(dest='fluid', sources=['fluid'],
                                        c0=self.c0, alpha=0.2),
                # ContinuityEquationFluid(dest='fluid', sources=[
                #     'tank', 'cylinder'
                # ], c0=self.c0, alpha=0.2),
                ContinuityEquationSolid(dest='fluid', sources=[
                    'tank', 'cylinder'
                ], c0=self.c0, alpha=0.2),
                MomentumEquationFluid(dest='fluid', sources=['fluid'], c0=self.
                                      c0, rho0=self.rho0, dim=self.dim,
                                      alpha=0.2, nu=self.nu, gy=self.gy),
                MomentumEquationSolid(
                    dest='fluid', sources=['tank', 'cylinder'], c0=self.c0,
                    rho0=self.rho0, dim=self.dim, alpha=0.2, nu=self.nu),
                RigidFluidForce(dest='cylinder', sources=['fluid']),
                SumUpExternalForces(dest='cylinder', sources=None)

            ])
        ]

        return MultiStageEquations([stage1, stage2])


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
