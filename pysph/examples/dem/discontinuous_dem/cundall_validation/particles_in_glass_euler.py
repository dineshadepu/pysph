from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver

from pysph.solver.application import Application
from pysph.dem.discontinuous_dem.dem_nonlinear import EPECIntegratorMultiStage, EulerIntegratorMultiStage
from pysph.dem.discontinuous_dem.dem_2d_linear_cundall import (
    get_particle_array_dem_2d_linear_cundall, BodyForce, Cundall2dForceStage1,
    Cundall2dForceStage2, Cundall2dForceEuler,
    UpdateTangentialContactsCundall2d, EulerStepDEM2dCundall,
    RK2StepDEM2dCundall, UpdateTangentialContactsCundall2d)
from pysph.sph.equation import Group, MultiStageEquations
from pysph.tools.geometry import rotate


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class ParticlesinGlass2d(Application):
    def __init__(self):
        super(ParticlesinGlass2d, self).__init__()

    def initialize(self):
        self.dx = 0.05
        self.dt = 1e-4
        self.tf = 3.
        self.dim = 2
        self.en = 0.1
        self.kn = 1e5
        self.wall_time = 1
        # friction coefficient
        self.mu = 0.5
        self.gy = -9.81
        self.seval = None

    def create_particles(self):
        # create 2d wall with particles
        dx = self.dx
        width = 80
        xw, yw = np.mgrid[-width * dx:width * dx:dx, -2 * dx:0. + dx / 2.:dx]
        xw = xw.ravel()
        yw = yw.ravel()
        rho = 2699.
        wall = get_particle_array_dem_2d_linear_cundall(
            x=xw, y=yw, rho=rho, name="wall", rad_s=self.dx / 2)
        wall.add_property('dem_id', type='int')
        wall.add_property('dem_id', type='int')
        wall.dem_id[:] = 0
        add_properties(wall, 'wx', 'wy', 'wz')
        wall.y -= 0.77

        # create 2d glass with particles
        height = 40
        xg1, yg1 = np.mgrid[-15. * dx:-13 * dx:dx, dx:height * dx:dx]
        xg2, yg2 = np.mgrid[13. * dx:15 * dx:dx, dx:height * dx:dx]
        xg3, yg3 = np.mgrid[-15. * dx:-13 * dx:dx, dx:20 * dx:dx]
        xg4, yg4 = np.mgrid[13. * dx:15 * dx:dx, dx:20 * dx:dx]
        xg1 = xg1.ravel()
        yg1 = yg1.ravel()
        xg2 = xg2.ravel()
        yg2 = yg2.ravel()
        xg3 = xg3.ravel()
        yg3 = yg3.ravel()
        xg4 = xg4.ravel()
        yg4 = yg4.ravel()

        xg3, yg3, zg3 = rotate(xg3, yg3, np.zeros_like(xg3), angle=30)
        xg4, yg4, zg4 = rotate(xg4, yg4, np.zeros_like(xg4), angle=-30)
        xg3 += 0.38
        xg4 -= 0.38
        yg3 -= 0.42
        yg4 -= 0.42

        xg = np.concatenate((xg1, xg2, xg3, xg4))
        yg = np.concatenate((yg1, yg2, yg3, yg4))

        rho = 2699.
        glass = get_particle_array_dem_2d_linear_cundall(
            x=xg, y=yg, rho=rho, name="glass", rad_s=self.dx / 2)
        glass.add_property('dem_id', type='int')
        glass.dem_id[:] = 2
        add_properties(glass, 'wx', 'wy', 'wz')

        # create bunch of particle
        height = 15
        xp, yp = np.mgrid[-10 * dx:10 * dx:dx, dx:height * dx:dx]

        rad_s = np.ones_like(xp) * self.dx / 2.
        rho = 2500
        m = rho * rad_s**3.
        inertia = m * 2. * rad_s**2. / 10.
        m_inverse = 1. / m
        I_inverse = 1. / inertia
        sand = get_particle_array_dem_2d_linear_cundall(
            x=xp, y=yp, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=rad_s, dem_id=1, h=1.2 * self.dx / 2., name="sand")

        return [wall, sand, glass]

    def create_equations(self):
        eq1 = [
            Group(
                equations=[
                    BodyForce(dest='sand', sources=None, gy=-9.81),
                    Cundall2dForceEuler(dest='sand',
                                        sources=['wall', 'sand', 'glass'],
                                        kn=self.kn, mu=0.5, en=self.en),
                ])
        ]
        return eq1

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EulerIntegratorMultiStage(sand=EulerStepDEM2dCundall())

        dt = self.dt
        tf = self.tf
        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
                        dt=dt, tf=tf)
        solver.set_disable_output(True)
        return solver

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
        T = self.wall_time
        if (T - dt / 2) < t < (T + dt / 2):
            for pa in self.particles:
                if pa.name == 'glass':
                    pa.y += self.dx * 40

        eqs1 = [
            Group(equations=[
                UpdateTangentialContactsCundall2d(dest='sand',
                                                  sources=['wall', 'sand', 'glass']),
            ])
        ]
        arrays = self.particles
        a_eval = self._make_accel_eval(eqs1, arrays)

        # When
        a_eval.evaluate(t, dt)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['sand']
        b.vectors = 'fx, fy, fz'
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {s_rad}
        b.scalar = 'u'
        b = particle_arrays['glass']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {g_rad}
        b.scalar = 'u'
        '''.format(s_rad=self.dx/2., g_rad=self.dx/2.))


if __name__ == '__main__':
    app = ParticlesinGlass2d()
    app.run()
