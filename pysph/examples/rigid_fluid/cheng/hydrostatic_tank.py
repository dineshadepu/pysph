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
    SourceNumberDensity, SolidWallPressureBC, SetFreeSlipWallVelocity,
    SetNoSlipWallVelocity, RK2ChengFluidStep)
from pysph.base.kernels import QuinticSpline
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import EPECIntegrator
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.tools.geometry_rigid_fluid import (get_2d_hydrostatic_tank)


class RigidFluidCoupling(Application):
    def initialize(self):
        # dimensions
        self.boundary_height = 1.
        self.boundary_length = 0.5
        self.fluid_height = 0.75
        self.spacing = 0.01
        self.layers = 2

        self.c0 = 2 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.dx = self.spacing
        self.hdx = 1.2
        self.rho0 = 1000
        self.solid_rho = 500
        self.m = 1000 * self.dx * self.dx
        self.alpha = 0.2
        self.beta = 0.0
        self.eps = 0.5
        self.gamma = 7
        self.p0 = self.c0 * self.c0 * self.rho0
        self.nu = 1. / 500
        self.gy = -9.81
        self.dim = 2

        self.tf = 1
        self.dt = 1e-4

    def create_particles(self):
        xt, yt, xf, yf = get_2d_hydrostatic_tank(
            ht_length=self.boundary_height, ht_height=self.boundary_height,
            fluid_height=self.fluid_height, spacing=self.spacing,
            layers=self.layers)
        m = self.rho0 * self.dx * self.dx
        rho = self.rho0
        h = self.hdx * self.dx
        fluid = get_particle_array_fluid_cheng(x=xf, y=yf, h=h, m=m, rho=rho,
                                               name="fluid")

        m = self.rho0 * self.dx * self.dx
        rho = self.rho0
        h = self.hdx * self.dx
        boundary = get_particle_array_fluid_cheng(x=xt, y=yt, h=h, m=m,
                                                  rho=rho, name="boundary")

        # add properties to boundary for Adami boundary boundary condition
        for prop in ('ug', 'vg', 'wg', 'uf', 'vf', 'wf', 'wij'):
            boundary.add_property(name=prop)

        return [fluid, boundary]

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegrator(fluid=RK2ChengFluidStep())

        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
                        dt=self.dt, tf=self.tf)

        return solver

    def create_equations(self):
        eq = [
            Group(equations=[
                # The following two equations are used to
                # find the pressure of the ghost particle belonging to
                # boundary
                SourceNumberDensity(dest='boundary', sources=['fluid']),
                SolidWallPressureBC(dest='boundary', sources=['fluid'],
                                    gy=self.gy, c0=self.c0, rho0=self.rho0),
                # and also set the velocity of the ghost particle to
                # satisfy the no slip or free slip conditions
                # this velocity is different from the velocity the
                # particle moves.
                # SetFreeSlipWallVelocity(dest='boundary', sources=['fluid']),
                SetNoSlipWallVelocity(dest='boundary', sources=['fluid'])
            ]),
            Group(equations=[
                ContinuityEquationFluid(dest='fluid', sources=['fluid'],
                                        c0=self.c0, alpha=0.2),
                ContinuityEquationSolid(dest='fluid', sources=['boundary'],
                                        c0=self.c0, alpha=0.2),
                StateEquation(dest='fluid', sources=None, c0=self.c0,
                              rho0=self.rho0),
                MomentumEquationFluid(dest='fluid', sources=['fluid'], c0=self.
                                      c0, rho0=self.rho0, dim=self.dim,
                                      alpha=0.2, nu=self.nu, gy=self.gy),
                MomentumEquationSolid(dest='fluid', sources=['boundary'],
                                      c0=self.c0, rho0=self.rho0, dim=self.dim,
                                      alpha=0.2, nu=self.nu)
            ])
        ]
        return eq


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
