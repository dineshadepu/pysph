"""A sphere of density 500 falling into a hydrostatic tank (15 minutes)

Check basic equations of SPH to throw a ball inside the vessel
"""
from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.utils import (get_particle_array_wcsph,
                              get_particle_array_rigid_body)
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import (XSPHCorrection, SummationDensity)
from pysph.sph.wc.basic import TaitEOSHGCorrection, MomentumEquation
from pysph.solver.application import Application
from pysph.tools.geometry import (get_2d_block_from_limits,
                                  get_2d_tank_from_limits)
from dem import (get_particle_array_dem, RK2DEMStep, BodyForce,
                 LinearSpringForceParticleParticle)


class RigidFluidCoupling(Application):
    def initialize(self):
        self.grains_spacing = 1 * 1e-2
        self.grains_length = 1.
        self.grains_height = 1
        self.grain_rho = 1000

        self.tank_spacing = 1 * 1e-2
        self.tank_length = 4.
        self.tank_height = 4
        self.tank_layers = 2
        self.tank_rho = 1000

        self.hdx = 1.2

    def create_particles(self):
        xf, yf = get_2d_block_from_limits(
            1.5, 1.5 + self.grains_length, self.grains_spacing, 1.5,
            1.5 + self.grains_height, self.grains_spacing)

        m = self.grain_rho * self.grains_spacing**2.
        rho = self.grain_rho
        h = self.hdx * self.grains_spacing
        grains = get_particle_array_dem(x=xf, y=yf, h=h, m=m, rho=rho,
                                        name="grains")

        xt, yt = get_2d_tank_from_limits(
            0., self.tank_length, self.tank_spacing, 0., self.tank_height,
            self.tank_spacing, self.tank_length, True)
        m = self.tank_rho * self.tank_spacing**2.
        rho = self.tank_rho
        h = self.hdx * self.tank_spacing
        tank = get_particle_array_dem(x=xt, y=yt, h=h, m=m, rho=rho,
                                      name="tank")
        return [grains, tank]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(grains=RK2DEMStep())

        dt = 1e-4
        print("DT: %s" % dt)
        tf = 0.5
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            dt=dt,
            tf=tf,
            adaptive_timestep=False,
        )

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='grains', sources=None, gy=-9.81),
                LinearSpringForceParticleParticle(
                    dest='grains', sources=['grains', 'tank'], gy=-9.81),
            ]),
        ]
        return equations


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
