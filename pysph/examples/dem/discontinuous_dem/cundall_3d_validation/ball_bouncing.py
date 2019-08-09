from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser
from pysph.dem.discontinuous_dem.dem_3d_linear_cundall import (
    get_particle_array_dem_3d_linear_cundall, Dem3dCundallScheme,
    UpdateTangentialContactsCundall3dPaticleParticle)
from pysph.sph.equation import Group


class BouncingBall(Application):
    def __init__(self):
        super(BouncingBall, self).__init__()

    def initialize(self):
        self.dx = 0.05
        self.dt = 1e-4
        self.tf = 2
        self.dim = 2
        self.en = 0.5
        self.kn = 50000
        # friction coefficient
        self.mu = 0.5
        self.gy = -9.81
        self.seval = None
        self.radius = 0.1
        self.pfreq = 100

    def create_particles(self):
        # create a particle
        xp = np.array([0.])
        yp = np.array([0.5])
        rho = 2600
        m = rho * 4. / 3. * np.pi * self.radius**3
        I = 2. / 5. * m * self.radius**2.
        m_inverse = 1. / m
        I_inverse = 1. / I
        sand = get_particle_array_dem_3d_linear_cundall(
            x=xp, y=yp, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=self.radius, dem_id=1, h=1.2 * self.dx / 2., name="sand")

        # create a particle
        xp = np.array([0.])
        yp = np.array([0.0])
        rho = 2600
        m = rho * 4. / 3. * np.pi * self.radius**3
        I = 2. / 5. * m * self.radius**2.
        m_inverse = 1. / m
        I_inverse = 1. / I
        wall = get_particle_array_dem_3d_linear_cundall(
            x=xp, y=yp, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=self.radius, dem_id=1, h=1.2 * self.dx / 2., name="wall")
        return [wall, sand]

    def create_scheme(self):
        dem3drk2 = Dem3dCundallScheme(
            bodies=['sand'], solids=['wall'], dim=self.dim, kn=self.kn,
            mu=self.mu, en=self.en, gy=-9.81, integrator="rk2")
        dem3deuler = Dem3dCundallScheme(
            bodies=['sand'], solids=['wall'], dim=self.dim, kn=self.kn,
            mu=self.mu, en=self.en, gy=-9.81, integrator="euler")
        s = SchemeChooser(default='dem3drk2', dem3drk2=dem3drk2,
                          dem3deuler=dem3deuler)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        scheme.configure()
        scheme.configure_solver(kernel=kernel, dt=self.dt, tf=self.tf, pfreq=self.pfreq)

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
                UpdateTangentialContactsCundall3dPaticleParticle(dest='sand',
                                                  sources=['wall']),
            ])
        ]
        arrays = self.particles
        a_eval = self._make_accel_eval(eqs1, arrays)

        # When
        a_eval.evaluate(t, dt)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['sand']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {s_rad}
        b.scalar = 'y'
        '''.format(s_rad=self.radius))


if __name__ == '__main__':
    app = BouncingBall()
    app.run()
