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
        fluid = get_particle_array(x=xf, y=yf, h=h, m=m, rho=rho, V=0,
                                   name="fluid")

        m = self.rho0 * self.dx * self.dx
        rho = self.rho0
        h = self.hdx * self.dx
        V = self.dx**2.
        tank = get_particle_array(x=xt, y=yt, h=h, m=m, rho=rho, V=V,
                                  name="tank")

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

        # add properties to fluid for TVF stepping scheme
        for prop in ('auhat', 'avhat', 'awhat', 'uhat', 'vhat', 'what',
                     'vmag2'):
            fluid.add_property(name=prop)

        return [fluid, tank]

    def create_solver(self):
        kernel = QuinticSpline(dim=2)

        integrator = EPECIntegrator(fluid=TransportVelocityStep())

        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        dt=self.dt, tf=self.tf)

        return solver

    def create_equations(self):
        eq = [
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid', 'tank'])
            ]),
            Group(equations=[
                StateEquation(dest='fluid', sources=None, p0=self.p0,
                              rho0=self.rho0, b=1.0),
                SetWallVelocity(dest='tank', sources=['fluid'])
            ]),
            Group(equations=[
                SolidWallPressureBC(dest='tank', sources=['fluid'], rho0=self.
                                    rho0, p0=self.p0, b=1.0, gy=-9.81)
            ]),
            Group(equations=[
                MomentumEquationPressureGradient(dest='fluid', sources=[
                    'fluid', 'tank'
                ], pb=self.p0, gy=-9.81),
                MomentumEquationViscosity(dest='fluid', sources=['fluid'],
                                          nu=self.nu),
                SolidWallNoSlipBC(dest='fluid', sources=['tank'], nu=self.nu),
                MomentumEquationArtificialStress(dest='fluid', sources=['fluid'])
            ])
        ]
        return eq


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
