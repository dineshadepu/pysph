"""A sphere of density 500 falling into a hydrostatic tank (15 minutes)

Check basic equations of SPH to throw a ball inside the vessel
"""
from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.utils import (get_particle_array_wcsph,
                              get_particle_array_rigid_body)
from pysph.sph.rigid_body import (
    BodyForce, SummationDensityBoundary, RigidBodyCollision, RigidBodyMoments,
    RigidBodyMotion, AkinciRigidFluidCoupling, RK2StepRigidBody)
from pysph.tools.geometry_rigid_fluid import (get_2d_hydrostatic_tank)
import os

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application

from pysph.sph.equation import Group
from pysph.sph.pengnan_sun import (
    ContinuityEquationFluid, ContinuityEquationSolid, MomentumEquationFluid,
    MomentumEquationSolid, StateEquation, get_particle_array_fluid_pengnan,
    SourceNumberDensity, SolidWallPressureBC,
    SetWallVelocity, RK2PengwanFluidStep)
from pysph.base.kernels import QuinticSpline
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import EPECIntegrator
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver


class RigidFluidCoupling(Application):
    def initialize(self):
        # dimensions
        self.tank_height = 1.
        self.tank_length = 0.5
        self.fluid_height = 0.75
        self.spacing = 0.01
        self.layers = 2

        self.c0 = 2 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.dx = self.spacing
        self.hdx = 1.2
        self.rho0 = 1000
        self.solid_rho = 500
        self.m = 1000 * self.dx * self.dx
        self.alpha = 0.1
        self.beta = 0.0
        self.eps = 0.5
        self.gamma = 7
        self.p0 = self.c0 * self.c0 * self.rho0
        self.nu = 1. / 500
        self.gy = -9.81
        self.dim = 2

        h0 = self.hdx * self.dx
        dt_cfl = 0.25 * h0 / (self.c0)
        dt_force = 1.0
        self.tf = 1.0
        self.dt = min(dt_cfl, dt_force)
        self.dt = 1e-4

    def create_particles(self):
        xt, yt, xf, yf = get_2d_hydrostatic_tank(
            ht_length=self.tank_height, ht_height=self.tank_height,
            fluid_height=self.fluid_height, spacing=self.spacing,
            layers=self.layers)
        m = self.rho0 * self.dx * self.dx
        rho = self.rho0
        h = self.hdx * self.dx
        fluid = get_particle_array_fluid_pengnan(x=xf, y=yf, h=h, m=m, rho=rho,
                                                 V=0, name="fluid")

        m = self.rho0 * self.dx * self.dx
        rho = self.rho0
        h = self.hdx * self.dx
        V = self.dx**2.
        tank = get_particle_array_fluid_pengnan(x=xt, y=yt, h=h, m=m, rho=rho,
                                                V=V, name="tank")

        dx = self.spacing
        volume = dx * dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * self.rho0
        tank.m[:] = volume * self.rho0
        # Set a reference rho also, some schemes will overwrite this with a
        # summation density.
        fluid.rho[:] = self.rho0
        tank.rho[:] = self.rho0

        # volume is set as dx^2
        fluid.V[:] = 1. / volume
        tank.V[:] = 1. / volume

        # smoothing lengths
        fluid.h[:] = self.hdx * dx
        tank.h[:] = self.hdx * dx

        # add properties to tank for Adami tank boundary condition
        for prop in ('ug', 'vg', 'wg', 'uf', 'vf', 'wf', 'wij'):
            tank.add_property(name=prop)

        return [fluid, tank]

    def create_solver(self):
        kernel = QuinticSpline(dim=2)

        integrator = EPECIntegrator(fluid=RK2PengwanFluidStep())

        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        dt=self.dt, tf=self.tf)

        return solver

    def create_equations(self):
        eq = [
            Group(equations=[
                # The following two equations are used to
                # find the pressure of the ghost particle belonging to
                # boundary
                SourceNumberDensity(dest='tank', sources=['fluid']),
                SolidWallPressureBC(dest='tank', sources=['fluid'], gy=self.gy,
                                    c0=self.c0, rho0=self.rho0),
                # and also set the velocity of the ghost particle to
                # satisfy the no slip or free slip conditions
                # this velocity is different from the velocity the
                # particle moves.
                SetWallVelocity(dest='tank', sources=['fluid'])
            ]),
            Group(equations=[
                ContinuityEquationFluid(dest='fluid', sources=['fluid'],
                                        c0=self.c0, delta=0.2),
                ContinuityEquationSolid(dest='fluid', sources=['tank'],
                                        c0=self.c0, delta=0.2),
                StateEquation(dest='fluid', sources=None, c0=self.c0,
                              rho0=self.rho0),
                MomentumEquationFluid(dest='fluid', sources=['fluid'],
                                      c0=self.c0, rho0=self.rho0, dim=self.dim,
                                      alpha=0.2, nu=self.nu),
                MomentumEquationSolid(dest='fluid', sources=['tank'],
                                      c0=self.c0, rho0=self.rho0, dim=self.dim,
                                      alpha=0.2, nu=self.nu)
            ])
        ]
        return eq


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
