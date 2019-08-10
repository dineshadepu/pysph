from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver

from pysph.solver.application import Application
from pysph.dem.discontinuous_dem.dem_nonlinear import (
    EPECIntegratorMultiStage, EulerIntegratorMultiStage)
from pysph.sph.scheme import SchemeChooser
from pysph.dem.discontinuous_dem.dem_2d_linear_cundall import (
    get_particle_array_dem_2d_linear_cundall, BodyForce,
    Cundall2dForceParticleParticleStage1, Cundall2dForceParticleParticleStage2,
    UpdateTangentialContactsCundall2dPaticleParticle,
    Cundall2dForceParticleParticleEuler, RK2StepDEM2dCundall,
    UpdateTangentialContactsCundall2dPaticleParticle, EulerStepDEM2dCundall)
from pysph.dem.discontinuous_dem.dem_2d_linear_cundall import (
    Dem2dCundallScheme, UpdateTangentialContactsCundall2dPaticleParticle,
    UpdateTangentialContactsCundall2dPaticleWall)

from pysph.base.utils import get_particle_array
from pysph.sph.equation import Group, MultiStageEquations
from pysph.tools.geometry import rotate
from pysph.tools.geometry import (get_2d_tank, get_2d_block)
from pysph.tools.geometry_rigid_fluid import (get_2d_dam_break)


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class BouncingBall(Application):
    def __init__(self):
        super(BouncingBall, self).__init__()

    def initialize(self):
        self.dx = 0.05
        self.dt = 1e-4
        self.tf = 5
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
        sand = get_particle_array_dem_2d_linear_cundall(
            x=xp, y=yp, u=1.0, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=self.radius, dem_id=1, h=1.2 * self.radius, name="sand")

        # create a particle
        xw_a = np.array([0.])
        yw_a = np.array([0.])
        nxw_a = np.array([0])
        nyw_a = np.array([1.])
        wall = get_particle_array(x=xw_a, y=yw_a, nx=nxw_a, ny=nyw_a, nz=0.,
                                  constants={'np': len(xw_a)}, name="wall")
        wall.add_property('dem_id', type='int')
        wall.dem_id[:] = 0
        return [wall, sand]

    def create_scheme(self):
        dem2drk2 = Dem2dCundallScheme(
            bodies=['sand'], solids=None, walls=['wall'], dim=self.dim,
            kn=self.kn, mu=self.mu, en=self.en, gy=-9.81, integrator="rk2")
        dem2deuler = Dem2dCundallScheme(
            bodies=['sand'], solids=None, walls=['wall'], dim=self.dim,
            kn=self.kn, mu=self.mu, en=self.en, gy=-9.81, integrator="euler")
        s = SchemeChooser(default='dem2drk2', dem2drk2=dem2drk2,
                          dem2deuler=dem2deuler)
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
                UpdateTangentialContactsCundall2dPaticleWall(
                    dest='sand', sources=['wall'])
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
        b.show_vectors = True
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {s_rad}
        b.scalar = 'fy'
        '''.format(s_rad=self.radius))


if __name__ == '__main__':
    app = BouncingBall()
    app.run()
