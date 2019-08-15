"""Two-dimensional dam break over a dry bed.  (30 minutes)

The case is described in "State of the art classical SPH for free surface
flows", Moncho Gomez-Gesteira, Benedict D Rogers, Robert A, Dalrymple and Alex
J.C Crespo, Journal of Hydraulic Research, Vol 48, Extra Issue (2010), pp
6-27. DOI:10.1080/00221686.2010.9641242

"""

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
    SourceNumberDensity, SolidWallPressureBC,
    RK2ChengFluidStep)
from pysph.base.kernels import QuinticSpline
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import EPECIntegrator
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver


class DamBreak2D(Application):
    def initialize(self):
        self.fluid_column_height = 2.0
        self.fluid_column_width = 1.0
        self.container_height = 4.0
        self.container_width = 4.0
        self.nboundary_layers = 2
        self.nu = 0.0
        self.dx = 0.03
        self.rho0 = 1000.0
        self.c0 = 10.0 * np.sqrt(2 * 9.81 * self.fluid_column_height)
        self.gamma = 7.0
        self.alpha = 0.1
        self.beta = 0.0
        self.B = self.c0 * self.c0 * self.rho0 / self.gamma
        self.p0 = 1000.0
        self.hdx = 1.5
        self.h = self.hdx * self.dx
        self.dim = 2
        self.tf = 2
        self.dt = 1e-4
        self.gy = -9.81

    def add_user_options(self, group):
        group.add_argument(
            '--staggered-grid',
            action="store_true",
            dest='staggered_grid',
            default=False,
            help="Use a staggered grid for particles.",
        )

    def create_particles(self):
        if self.options.staggered_grid:
            nboundary_layers = 2
            nfluid_offset = 2
            wall_hex_pack = True
        else:
            nboundary_layers = 4
            nfluid_offset = 1
            wall_hex_pack = False
        geom = DamBreak2DGeometry(
            container_width=self.container_width,
            container_height=self.container_height,
            fluid_column_height=self.fluid_column_height,
            fluid_column_width=self.fluid_column_width, dx=self.dx, dy=self.dx,
            nboundary_layers=1, ro=self.rho0, co=self.c0, with_obstacle=False,
            wall_hex_pack=wall_hex_pack, beta=1.0, nfluid_offset=1,
            hdx=self.hdx)
        fluid, boundary = geom.create_particles(
            nboundary_layers=nboundary_layers,
            hdx=self.hdx,
            nfluid_offset=nfluid_offset,
        )

        # create pengwang particle array
        fluid = get_particle_array_fluid_cheng(
            x=fluid.x, y=fluid.y, h=self.h, m=fluid.m, rho=fluid.rho, V=0,
            name="fluid")

        # fluid.V[:] = self.dx**self.dim

        boundary = get_particle_array_fluid_cheng(
            x=boundary.x, y=boundary.y, h=self.h, m=boundary.m,
            rho=boundary.rho, V=0, name="boundary")
        # boundary.V[:] = self.dx**self.dim

        # add properties to tank for Adami tank boundary condition
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
                # SetWallVelocity(dest='boundary', sources=['fluid'])
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
    app = DamBreak2D()
    app.run()
