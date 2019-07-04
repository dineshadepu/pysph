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
from pysph.dem.discontinuous_dem.dem_nonlinear import EPECIntegratorMultiStage
from pysph.sph.rigid_body import BodyForce
from pysph.dem.discontinuous_dem.dem_linear import (
    get_particle_array_dem_linear, LinearDEMNoRotationScheme,
    UpdateTangentialContactsNoRotation, LinearPWFDEMNoRotationStage1,
    LinearPWFDEMNoRotationStage2, RK2StepLinearDEMNoRotation,
    UpdateTangentialContactsWallNoRotation, LinearPPFDEMNoRotationStage1,
    LinearPPFDEMNoRotationStage2)
from pysph.sph.scheme import SchemeChooser
from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.sph.equation import Group, MultiStageEquations


class SettlingParticles(Application):
    def __init__(self, theta=0.):
        self.theta = theta
        super(SettlingParticles, self).__init__()

    def initialize(self):
        self.dx = 0.05
        self.dt = 1e-4
        self.tf = 5
        self.dim = 2
        self.en = 0.1
        self.kn = 1e5
        # friction coefficient
        self.mu = 0.5
        self.gy = -9.81
        self.seval = None

    def create_particles(self):
        # single wall
        # xw_a = np.array([0.])
        # yw_a = np.array([0.])
        # nxw_a = np.array([0])
        # nyw_a = np.array([1.])
        # 3 wall
        xw_a = np.array([0., 1.2, -0.2])
        yw_a = np.array([0., 0., 0])
        nxw_a = np.array([0, -1, 1.])
        nyw_a = np.array([1., 0, 0])
        rho = 2699.
        wall = get_particle_array(x=xw_a, y=yw_a, nx=nxw_a, ny=nyw_a, nz=0.,
                                  rho=rho, constants={'np':
                                                      len(xw_a)}, name="wall")
        wall.add_property('dem_id', type='int')
        wall.dem_id[:] = 0

        # create bunch of particle
        xp, yp = np.mgrid[0:1.:self.dx, self.dx/2.:1.:self.dx]
        xp[0] -= self.dx / 8.
        u = 0.

        # create 4 particles
        # dx = self.dx
        # xp = np.array([0., 2.*dx, 4.*dx, 6.*dx])
        # yp = np.ones_like(xp) * 2. * dx
        # u = np.array([1., -1., 1, -1])
        rad_s = np.ones_like(xp) * self.dx / 2.
        rho = 2500
        m = rho * rad_s**3.
        inertia = m * 2. * rad_s**2. / 10.
        m_inverse = 1. / m
        I_inverse = 1. / inertia
        sand = get_particle_array_dem_linear(
            x=xp, y=yp, u=u, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=rad_s, dem_id=1, h=1.2 * self.dx / 2., name="sand")

        return [wall, sand]

    # def create_scheme(self):
    #     ldems = LinearDEMNoRotationScheme(
    #         dem_bodies=['sand'], rigid_bodies=None, solids=None, walls=[
    #             'wall'
    #         ], dim=self.dim, kn=self.kn, mu=self.mu, en=self.en, gy=self.gy)
    #     s = SchemeChooser(default='ldems', ldems=ldems)
    #     return s

    # def configure_scheme(self):
    #     scheme = self.scheme
    #     kernel = CubicSpline(dim=self.dim)
    #     tf = self.tf
    #     dt = self.dt
    #     scheme.configure()
    #     scheme.configure_solver(kernel=kernel,
    #                             integrator_cls=EPECIntegratorMultiStage, dt=dt,
    #                             tf=tf)

    def create_equations(self):
        eq1 = [
            Group(
                equations=[
                    BodyForce(dest='sand', sources=None, gx=0.0, gy=-9.81,
                              gz=0.0),
                    LinearPWFDEMNoRotationStage1(dest='sand', sources=['wall'],
                                                 kn=self.kn, mu=0.5, en=0.5),
                    LinearPPFDEMNoRotationStage1(dest='sand', sources=['sand'],
                                                 kn=self.kn, mu=0.5, en=0.5)
                ], real=False, update_nnps=False, iterate=False,
                max_iterations=1, min_iterations=0, pre=None, post=None)
        ]
        eq2 = [
            Group(
                equations=[
                    BodyForce(dest='sand', sources=None, gx=0.0, gy=-9.81,
                              gz=0.0),
                    LinearPWFDEMNoRotationStage2(dest='sand', sources=['wall'],
                                                 kn=self.kn, mu=0.5, en=0.5),
                    LinearPPFDEMNoRotationStage2(dest='sand', sources=['sand'],
                                                 kn=self.kn, mu=0.5, en=0.5)
                ], real=False, update_nnps=False, iterate=False,
                max_iterations=1, min_iterations=0, pre=None, post=None)
        ]

        return MultiStageEquations([eq1, eq2])

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegratorMultiStage(
            sand=RK2StepLinearDEMNoRotation())

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
        eqs1 = [
            Group(equations=[
                UpdateTangentialContactsWallNoRotation(dest='sand',
                                                       sources=['wall']),
                UpdateTangentialContactsNoRotation(dest='sand',
                                                   sources=['sand'])
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
        b.scalar = 'fy'
        '''.format(s_rad=self.dx/2.))


if __name__ == '__main__':
    app = SettlingParticles()
    app.run()
