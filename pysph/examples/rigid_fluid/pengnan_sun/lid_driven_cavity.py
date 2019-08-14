"""Lid driven cavity using the Transport Velocity formulation. (10 minutes)
"""

import os

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application

from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import (
    SummationDensity, StateEquation, MomentumEquationPressureGradient,
    MomentumEquationArtificialViscosity, MomentumEquationViscosity,
    MomentumEquationArtificialStress, SolidWallPressureBC, SolidWallNoSlipBC,
    SetWallVelocity)
from pysph.base.kernels import QuinticSpline
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import EPECIntegrator
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver

# numpy
import numpy as np


class LidDrivenCavity(Application):
    def initialize(self):
        # domain and reference values
        self.L = 1.0
        self.Umax = 1.0
        self.c0 = 10 * self.Umax
        self.rho0 = 1.0
        self.p0 = self.c0 * self.c0 * self.rho0

        # Numerical setup
        self.hdx = 1.0

        nx = 50
        self.re = 100
        self.n_avg = 5
        self.dx = self.L / nx
        h0 = self.hdx * self.dx
        self.nu = self.Umax * self.L / self.re
        dt_cfl = 0.25 * h0 / (self.c0 + self.Umax)
        dt_viscous = 0.125 * h0**2 / self.nu
        dt_force = 1.0
        self.tf = 10.0
        self.dt = min(dt_cfl, dt_viscous, dt_force)

    def create_particles(self):
        dx = self.dx
        ghost_extent = 5 * dx
        # create all the particles
        _x = np.arange(-ghost_extent - dx / 2, self.L + ghost_extent + dx / 2, dx)
        x, y = np.meshgrid(_x, _x)
        x = x.ravel()
        y = y.ravel()

        # sort out the fluid and the solid
        indices = []
        for i in range(x.size):
            if ((x[i] > 0.0) and (x[i] < self.L)):
                if ((y[i] > 0.0) and (y[i] < self.L)):
                    indices.append(i)

        # create the arrays
        solid = get_particle_array(name='solid', x=x, y=y, V=0.)

        # remove the fluid particles from the solid
        fluid = solid.extract_particles(indices)
        fluid.set_name('fluid')
        solid.remove_particles(indices)

        print("Lid driven cavity :: Re = %d, dt = %g" % (self.re, self.dt))

        # setup the particle properties
        volume = dx * dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * self.rho0
        solid.m[:] = volume * self.rho0
        # Set a reference rho also, some schemes will overwrite this with a
        # summation density.
        fluid.rho[:] = self.rho0
        solid.rho[:] = self.rho0

        # volume is set as dx^2
        fluid.V[:] = 1. / volume
        solid.V[:] = 1. / volume

        # smoothing lengths
        fluid.h[:] = self.hdx * dx
        solid.h[:] = self.hdx * dx

        # imposed horizontal velocity on the lid
        solid.u[:] = 0.0
        solid.v[:] = 0.0
        for i in range(solid.get_number_of_particles()):
            if solid.y[i] > self.L:
                solid.u[i] = self.Umax

        # add properties to solid for Adami solid boundary condition
        for prop in ('ug', 'vg', 'wg', 'uf', 'vf', 'wf', 'wij'):
            solid.add_property(name=prop)

        # add properties to fluid for TVF stepping scheme
        for prop in ('auhat', 'avhat', 'awhat', 'uhat', 'vhat', 'what', 'vmag2'):
            fluid.add_property(name=prop)
        return [fluid, solid]

    def create_equations(self):
        return [
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid', 'solid'])
            ]),
            Group(equations=[
                StateEquation(dest='fluid', sources=None, p0=100.0, rho0=1.0,
                              b=1.0),
                SetWallVelocity(dest='solid', sources=['fluid'])
            ]),
            Group(equations=[
                SolidWallPressureBC(dest='solid', sources=['fluid'], rho0=1.0,
                                    p0=100.0, b=1.0, gx=0.0, gy=0.0, gz=0.0)
            ]),
            Group(equations=[
                MomentumEquationPressureGradient(dest='fluid', sources=[
                    'fluid', 'solid'
                ], pb=100.0, gx=0.0, gy=0.0, gz=0.0, tdamp=0.0),
                MomentumEquationViscosity(dest='fluid', sources=['fluid'],
                                          nu=0.01),
                SolidWallNoSlipBC(dest='fluid', sources=['solid'], nu=0.01),
                MomentumEquationArtificialStress(dest='fluid',
                                                 sources=['fluid'])
            ])
        ]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(fluid=TransportVelocityStep())

        dt = self.dt
        tf = 1.
        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        dt=dt, tf=tf)
        return solver


if __name__ == '__main__':
    app = LidDrivenCavity()
    app.run()
    # app.post_process(app.info_filename)
