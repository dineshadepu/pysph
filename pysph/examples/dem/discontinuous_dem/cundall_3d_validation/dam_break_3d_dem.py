from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver

from pysph.solver.application import Application
from pysph.dem.discontinuous_dem.dem_nonlinear import EPECIntegratorMultiStage
from pysph.sph.equation import Group, MultiStageEquations
from pysph.tools.geometry import rotate
from pysph.tools.geometry import (get_3d_block)
from pysph.dem.discontinuous_dem.dem_3d_linear_cundall import (
    get_particle_array_dem_3d_linear_cundall, Dem3dCundallScheme,
    UpdateTangentialContactsCundall3dPaticleParticle)
from pysph.sph.scheme import SchemeChooser
from pysph.examples._db_geometry import DamBreak3DGeometry


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class DemDamBreak3d(Application):
    def consume_user_options(self):
        self.dx = 0.1
        self.nboundary_layers = 1
        self.hdx = 1.3
        self.rho = 1000

        self.geom = DamBreak3DGeometry(
            dx=self.dx, nboundary_layers=self.nboundary_layers, hdx=self.hdx,
            fluid_column_height=2,
        )

    def initialize(self):
        self.kn = 50000
        self.en = 0.3
        self.mu = 0.5
        self.dt = 1e-4
        self.tf = 3
        self.pfreq = 100
        self.dim = 3
        self.seval = None

    def create_particles(self):
        sand, tank, obstcl = self.geom.create_particles()

        # create tank
        dx = self.dx
        m = self.rho * self.dx**3
        tank = get_particle_array_dem_3d_linear_cundall(
            x=tank.x, y=tank.y, z=tank.z, m=m, rad_s=self.dx / 2., dem_id=0,
            h=1.2 * self.dx, name="tank")

        # create bunch of particle
        rad_s = self.dx / 2
        rho = 2500
        m = rho * rad_s**3.
        inertia = m * 2. * rad_s**2. / 10.
        m_inverse = 1. / m
        I_inverse = 1. / inertia
        sand = get_particle_array_dem_3d_linear_cundall(
            x=sand.x, y=sand.y, z=sand.z, m=m, I_inverse=I_inverse,
            m_inverse=m_inverse, rad_s=self.dx/2, dem_id=1, h=1.2 * self.dx,
            name="sand")
        fltr = sand.x < 0.2
        sand.x[fltr] += self.dx/3.

        return [tank, sand]

    def create_scheme(self):
        dem3drk2 = Dem3dCundallScheme(
            bodies=['sand'], solids=['tank'], dim=self.dim, kn=self.kn,
            mu=self.mu, en=self.en, gz=-9.81, integrator="rk2")
        dem3deuler = Dem3dCundallScheme(
            bodies=['sand'], solids=['tank'], dim=self.dim, kn=self.kn,
            mu=self.mu, en=self.en, gz=-9.81, integrator="euler")
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
                UpdateTangentialContactsCundall3dPaticleParticle(dest='sand',
                                                  sources=['tank', 'sand']),
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
        '''.format(s_rad=self.dx/2.))


if __name__ == '__main__':
    app = DemDamBreak3d()
    app.run()
