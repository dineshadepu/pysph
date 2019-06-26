"""Particles motion in a rotating drum. This is validation test of existing DEM
equations.

This example is taken from [1] paper. See section 3.

[1] Numerical simulation of particle dynamics in different flow regimes in a
rotating drum
"""

from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.dem.discontinuous_dem.dem_nonlinear import (
    get_particle_array_dem_nonlinear, EPECIntegratorMultiStage,
    UpdateTangentialContacts)
from pysph.dem.discontinuous_dem.dem_linear import (
    get_particle_array_dem_linear, LinearDEMScheme)
from pysph.base.utils import get_particle_array_rigid_body
from pysph.sph.scheme import SchemeChooser
from pysph.sph.equation import Group


def create_circle(points=100, radius=50e-3):
    theta = np.linspace(0., 2. * np.pi, points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


class RotatingDrum(Application):
    def __init__(self, theta=0.):
        self.theta = theta
        super(RotatingDrum, self).__init__()

    def initialize(self):
        self.drum_radius = 50e-3
        self.dt = 1e-4
        self.dim = 2
        self.en = 0.8
        self.kn = 1e4
        # friction coefficient
        self.mu = 0.5
        self.gy = -9.81

    def create_particles(self):
        # create a rotating drum
        x_d, y_d = create_circle(200, self.drum_radius)
        self.drum_spacing = (
            (x_d[2] - x_d[1])**2. + (y_d[2] - y_d[1])**2.)**0.5

        yng_m = 7. * 1e10
        poissons_ratio = 0.33
        drum = get_particle_array_rigid_body(x=x_d, y=y_d, m=1,
                                             rad_s=self.drum_spacing / 2.,
                                             dem_id=0, name="drum")
        drum.omega[2] = 3.

        # create a bunch of particles inside the drum
        x, y = np.mgrid[-20e-3:20e-3:self.drum_spacing /
                        2., -20e-3:20e-3:self.drum_spacing / 2.]
        x = x.ravel()
        y = y.ravel()

        sand = get_particle_array_dem_linear(
            x=x, y=y, m=1, rad_s=self.drum_spacing / 2., dem_id=1, name="sand")

        return [drum, sand]

    def create_scheme(self):
        ldems = LinearDEMScheme(dem_bodies=['sand'], rigid_bodies=['drum'],
                                solids=None, dim=self.dim, kn=self.kn,
                                mu=self.mu, en=self.en, gy=self.gy)
        s = SchemeChooser(default='ldems', ldems=ldems)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        tf = 5.
        dt = 1e-3
        scheme.configure()
        scheme.configure_solver(kernel=kernel,
                                integrator_cls=EPECIntegratorMultiStage, dt=dt,
                                tf=tf)

    def _make_accel_eval(self, equations, pa_arrays):
        from pysph.tools.sph_evaluator import SPHEvaluator
        kernel = CubicSpline(dim=self.dim)
        seval = SPHEvaluator(arrays=pa_arrays, equations=equations,
                             dim=self.dim, kernel=kernel)
        return seval

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        eqs1 = [
            Group(equations=[
                UpdateTangentialContacts(dest='spheres', sources=["spheres"])
            ])
        ]
        arrays = self.particles
        a_eval = self._make_accel_eval(eqs1, arrays)

        # When
        a_eval.evaluate(t, dt)


if __name__ == '__main__':
    app = RotatingDrum()
    app.run()
