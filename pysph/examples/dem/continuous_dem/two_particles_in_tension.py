"""An example test to verify the bonded DEM model. In this benchmark we analyze
two particles in tension.

This benchmark is taken from "Code Implementation Of Particle Based Discrete
Element Method For Concrete Viscoelastic Modelling" thesis. Please see section
4.2.2.

Time: -

"""

from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator

from pysph.sph.equation import Group, MultiStageEquations
from pysph.solver.application import Application
from pysph.dem.continuous_dem.spring_dem import (
    get_particle_array_continuous_dem_spring_model, BondedDEMPotyndyForce)


class TensionTest(Application):
    def initialize(self):
        self.radius = 1.
        self.clearence = 0.1

    def create_particles(self):
        x, y = np.array([0., 2. * self.radius]), np.array([0., 0.])
        beam = get_particle_array_continuous_dem_spring_model(
            x=x, y=y, dim=2, h=2. * self.radius, rad_s=self.radius,
            clrnc=self.clearence, name='beam')

        return [beam]

    def create_equations(self):
        Group(equations=[
            BondedDEMPotyndyForce(dest='beam', sources=None),
        ])


if __name__ == '__main__':
    app = TensionTest()
    [beam] = app.create_particles()
