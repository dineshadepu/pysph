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
    Dem2dCundallScheme, UpdateTangentialContactsCundall2dPaticleParticle)

from pysph.sph.equation import Group, MultiStageEquations
from pysph.tools.geometry import rotate
from pysph.tools.geometry import (get_2d_tank, get_2d_block)
from pysph.tools.geometry_rigid_fluid import (get_2d_dam_break)


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class ParticlesinGlass2d(Application):
    def __init__(self):
        super(ParticlesinGlass2d, self).__init__()

    def initialize(self):
        self.dx = 0.05
        self.dt = 1e-4
        self.tf = 2.5 + 0.51
        self.dim = 2
        self.en = 0.5
        self.kn = 1e5
        # friction coefficient
        self.mu = 0.2
        self.gy = -9.81
        self.seval = None

        self.dam_length = 20
        self.dam_height = 5
        self.dam_spacing = self.dx
        self.dam_layers = 2
        self.dam_rho = 1850.

        self.sand_length = 4
        self.sand_height = 2
        self.sand_spacing = self.dx
        self.sand_rho = 1850.

        self.pfreq = 1000

    def create_particles(self):
        # create tank
        xt, yt, xp, yp = get_2d_dam_break(self.dam_length, self.dam_height,
                                          self.sand_height, self.sand_length,
                                          self.dx, self.dam_layers)
        m = self.dam_rho * self.dam_spacing**2
        tank = get_particle_array_dem_2d_linear_cundall(
            x=xt, y=yt, m=m, rad_s=self.dam_spacing / 2., dem_id=0,
            h=1.2 * self.dx / 2., name="tank")

        # create sand of particle
        rad_s = np.ones_like(xp) * self.dx / 2.
        rho = self.sand_rho
        m = rho * rad_s**3.
        inertia = m * 2. * rad_s**2. / 10.
        m_inverse = 1. / m
        I_inverse = 1. / inertia
        sand = get_particle_array_dem_2d_linear_cundall(
            x=xp, y=yp, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=rad_s, dem_id=2, h=1.2 * self.sand_spacing / 2., name="sand")

        return [tank, sand]

    def create_scheme(self):
        dem2drk2 = Dem2dCundallScheme(bodies=['sand'], solids=['tank'],
                                      dim=self.dim, kn=self.kn, mu=self.mu,
                                      en=self.en, gy=-9.81, integrator="rk2")
        dem2deuler = Dem2dCundallScheme(
            bodies=['sand'], solids=['tank'], dim=self.dim, kn=self.kn,
            mu=self.mu, en=self.en, gy=-9.81, integrator="euler")
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
                UpdateTangentialContactsCundall2dPaticleParticle(
                    dest='sand', sources=['tank', 'sand']),
            ])
        ]
        arrays = self.particles
        a_eval = self._make_accel_eval(eqs1, arrays)

        # When
        a_eval.evaluate(t, dt)


if __name__ == '__main__':
    app = ParticlesinGlass2d()
    app.run()
