from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver

from pysph.solver.application import Application
from pysph.dem.discontinuous_dem.dem_nonlinear import (EPECIntegratorMultiStage, EulerIntegratorMultiStage)
from pysph.sph.scheme import SchemeChooser
from pysph.dem.discontinuous_dem.dem_3d_linear_cundall import (
    get_particle_array_dem_3d_linear_cundall, BodyForce, Cundall3dForceStage1,
    Cundall3dForceStage2, UpdateTangentialContactsCundall3d,
    Cundall3dForceEuler, RK2StepDEM3dCundall,
    UpdateTangentialContactsCundall3d, EulerStepDEM3dCundall)
from pysph.dem.discontinuous_dem.dem_3d_linear_cundall import (
    get_particle_array_dem_3d_linear_cundall, Dem3dCundallScheme,
    UpdateTangentialContactsCundall3d)
from pysph.sph.equation import Group, MultiStageEquations
from pysph.tools.geometry import rotate
from pysph.tools.geometry import (get_2d_tank, get_2d_block)


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class ParticlesinGlass2d(Application):
    def __init__(self):
        super(ParticlesinGlass2d, self).__init__()

    def initialize(self):
        self.dx = 0.05
        self.dt = 1e-4
        self.tf = 3
        self.dim = 2
        self.en = 0.1
        self.kn = 1e5
        self.wall_time = 1.5
        # friction coefficient
        self.mu = 0.5
        self.gy = -9.81
        self.seval = None

        self.dam_length = 4
        self.dam_height = 4
        self.dam_spacing = self.dx
        self.dam_layers = 2
        self.dam_rho = 2000.

        self.sand_length = 1
        self.sand_height = 3
        self.sand_spacing = self.dx

        self.pfreq = 100

    def create_particles(self):
        # create tank
        dx = self.dx
        xt, yt = get_2d_tank(self.dx, length=self.dam_length,
                             height=self.dam_height,
                             num_layers=self.dam_layers, outside=True)
        m = self.dam_rho * self.dam_spacing**2
        tank = get_particle_array_dem_3d_linear_cundall(
            x=xt, y=yt, m=m, rad_s=self.dx / 2., dem_id=0,
            h=1.2 * self.dx / 2., name="tank")

        # create bunch of particle
        xp, yp = get_2d_block(self.dx, self.sand_length, self.sand_height)
        yp = yp + self.sand_height / 2. + self.dx
        xp = xp - self.sand_length / 2. + self.dx

        rad_s = np.ones_like(xp) * self.dx / 2.
        rho = 2500
        m = rho * rad_s**3.
        inertia = m * 2. * rad_s**2. / 10.
        m_inverse = 1. / m
        I_inverse = 1. / inertia
        sand = get_particle_array_dem_3d_linear_cundall(
            x=xp, y=yp, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=rad_s, dem_id=2, h=1.2 * self.dx / 2., name="sand")

        return [tank, sand]

    def create_scheme(self):
        dem3drk2 = Dem3dCundallScheme(
            bodies=['sand'], solids=['tank'], dim=self.dim, kn=self.kn,
            mu=self.mu, en=self.en, gy=-9.81, integrator="rk2")
        dem3deuler = Dem3dCundallScheme(
            bodies=['sand'], solids=['tank'], dim=self.dim, kn=self.kn,
            mu=self.mu, en=self.en, gy=-9.81, integrator="euler")
        s = SchemeChooser(default='dem3drk2', dem3drk2=dem3drk2,
                          dem3deuler=dem3deuler)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        scheme.configure()
        scheme.configure_solver(kernel=kernel, dt=self.dt, tf=self.tf,
                                pfreq=self.pfreq)

    def _make_accel_eval(self, equations, pa_arrays):
        from pysph.tools.sph_evaluator import SPHEvaluator
        if self.seval is None:
            kernel = CubicSpline(dim=self.dim)
            seval = SPHEvaluator(arrays=pa_arrays, equations=equations,
                                 dim=self.dim, kernel=kernel)
            self.seval = seval
            return self.seval
        else:
            return self.seval

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        eqs1 = [
            Group(equations=[
                UpdateTangentialContactsCundall3d(dest='sand',
                                                  sources=['tank', 'sand']),
            ])
        ]
        arrays = self.particles
        a_eval = self._make_accel_eval(eqs1, arrays)

        # When
        a_eval.evaluate(t, dt)


if __name__ == '__main__':
    app = ParticlesinGlass2d()
    app.run()
