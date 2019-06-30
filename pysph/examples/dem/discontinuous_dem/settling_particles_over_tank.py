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

from pysph.solver.application import Application
from pysph.dem.discontinuous_dem.dem_nonlinear import EPECIntegratorMultiStage
from pysph.dem.discontinuous_dem.dem_linear import (
    get_particle_array_dem_linear, LinearDEMNoRotationScheme,
    UpdateTangentialContactsNoRotation)
from pysph.sph.scheme import SchemeChooser
from pysph.sph.equation import Group
from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.tools.geometry import get_2d_tank, get_2d_block


class SettlingParticles(Application):
    def __init__(self, theta=0.):
        self.theta = theta
        super(SettlingParticles, self).__init__()

    def initialize(self):
        self.dx = 0.05
        self.dt = 1e-4
        self.tf = 5.
        self.dim = 2
        self.en = 0.1
        self.kn = 1e6
        # friction coefficient
        self.mu = 0.5
        self.gy = -9.81
        self.seval = None

    def create_particles(self):
        # create a tank
        xt, yt = get_2d_tank(self.dx, [0., 0.], 0.4, 1., 2)
        rad_s = np.ones_like(xt) * self.dx / 2.
        rho = 2500
        m = rho * np.pi * rad_s**2.
        tank = get_particle_array(x=xt, y=yt, m=m, rad_s=rad_s, name="tank",
                                  wx=0., wy=0., wz=0.)
        tank.add_property('dem_id', type='int')
        tank.dem_id[:] = 0

        tank.set_output_arrays([
            'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag',
            'rad_s', 'dem_id'
        ])
        # drum.omega[2] = 30.

        # create a bunch of particles inside the drum
        xp, yp = get_2d_block(self.dx, 0.3, 0.3, [0., 0.2])
        rad_s = np.ones_like(xp) * self.dx / 2.
        rho = 2500
        m = rho * np.pi * rad_s**2.
        inertia = m * 2. * rad_s / 10.
        m_inverse = 1. / m
        I_inverse = 1. / inertia
        sand = get_particle_array_dem_linear(
            x=xp, y=yp, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=rad_s, dem_id=1, h=1.2 * self.dx / 2., name="sand")

        return [tank, sand]

    def create_scheme(self):
        ldemnrs = LinearDEMNoRotationScheme(
            dem_bodies=['sand'], rigid_bodies=None, solids=['tank'],
            walls=None, dim=self.dim, kn=self.kn, mu=self.mu, en=self.en,
            gy=self.gy)
        s = SchemeChooser(default='ldemnrs', ldemnrs=ldemnrs)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        tf = self.tf
        dt = self.dt
        scheme.configure()
        scheme.configure_solver(kernel=kernel,
                                integrator_cls=EPECIntegratorMultiStage, dt=dt,
                                tf=tf)

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
                UpdateTangentialContactsNoRotation(dest='sand',
                                                   sources=['sand', 'tank'])
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
        b = particle_arrays['tank']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {t_rad}
        b.scalar = 'm'
        '''.format(s_rad=self.dx / 2., t_rad=self.dx / 2.))


if __name__ == '__main__':
    app = SettlingParticles()
    app.run()
