"""
This example is a benchmark of DEM numerical method. This is a test case from
[1] paper. In this test case we simulate two spherical particles having a
head on collision with different impact angles. Section 4. of [1]

[1] Using distributed contacts in DEM, Sharen J.Cummins, Paul W.Cleary

"""
from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.dem.discontinuous_dem.dem_nonlinear import EPECIntegratorMultiStage
from pysph.sph.integrator import EPECIntegrator
from pysph.dem.discontinuous_dem.dem_linear import (
    get_particle_array_dem_linear, LinearDEMScheme,
    UpdateTangentialContactsNoRotation, LinearPPFDEMStage1, LinearPPFDEMStage2,
    RK2StepLinearDEM, LinearDEMNoRotationScheme, LinearPPFDEMNoRotationStage1,
    LinearPPFDEMNoRotationStage2)
from pysph.dem.discontinuous_dem.dem_linear import (BodyForce)
from pysph.sph.scheme import SchemeChooser
from pysph.sph.equation import Group
from pysph.sph.equation import Group, MultiStageEquations


class BinaryImpact2d(Application):
    def __init__(self, theta=0.):
        self.theta = theta
        super(BinaryImpact2d, self).__init__()

    def initialize(self):
        self.tf = 0.05
        # self.tf = 0.01
        self.dt = 1e-4
        self.dim = 2
        self.en = 1.
        # self.en = 0.8
        self.mu = 0.5
        self.kn = 1e6
        self.radius = 0.05
        self.diameter = 2. * self.radius
        self.gx = 0.
        self.gy = 0.
        self.gz = 0.
        # self.tf = 0.000012

    def create_particles(self):
        scale = 0.05
        # scale = 0.001
        rad_s = np.array([self.radius, self.radius])
        dia_s = 2. * np.array([self.radius, self.radius])
        x = np.array([-self.radius, self.radius + scale])
        y = np.array([0., 0.])
        u = np.array([1., -1])
        rho = np.array([2700., 2700.])
        m = rho * np.pi * rad_s**2.
        inertia = m * dia_s**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres = get_particle_array_dem_linear(
            x=x, y=y, u=u, h=h, m=m, rho=rho, rad_s=rad_s, dem_id=0,
            m_inverse=m_inv, I_inverse=I_inv, name="spheres")

        return [spheres]

    # def create_equations(self):
    #     stage1 = [
    #         Group(equations=[
    #             BodyForce(dest='spheres', sources=None, gx=0.0, gy=0.0,
    #                       gz=0.0),
    #             LinearPPFDEMNoRotationStage1(dest='spheres', sources=[
    #                 'spheres'
    #             ], kn=self.kn, mu=0.5, en=1.0)
    #         ]),
    #     ]
    #     return stage1
    #     stage2 = [
    #         Group(equations=[
    #             BodyForce(dest='spheres', sources=None, gx=0.0, gy=0.0,
    #                       gz=0.0),
    #             LinearPPFDEMNoRotationStage2(dest='spheres', sources=[
    #                 'spheres'
    #             ], kn=self.kn, mu=0.5, en=1.0)
    #         ]),
    #     ]
    #     return MultiStageEquations([stage1, stage2])

    # def create_solver(self):
    #     kernel = CubicSpline(dim=self.dim)

    #     # integrator = EPECIntegratorMultiStage(spheres=RK2StepLinearDEM())
    #     integrator = EPECIntegrator(spheres=RK2StepLinearDEM())

    #     dt = self.dt
    #     tf = self.tf
    #     solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
    #                     dt=dt, tf=tf, pfreq=10)
    #     solver.set_disable_output(True)
    #     return solver

    def create_scheme(self):
        ldems = LinearDEMScheme(dem_bodies=['spheres'], rigid_bodies=None,
                                solids=None, dim=self.dim, kn=self.kn,
                                mu=self.mu, en=self.en, gx=self.gx, gy=self.gy,
                                gz=self.gz)
        ldemnrs = LinearDEMNoRotationScheme(
            dem_bodies=['spheres'], rigid_bodies=None, solids=None,
            dim=self.dim, kn=self.kn, mu=self.mu, en=self.en, gx=self.gx,
            gy=self.gy, gz=self.gz)
        s = SchemeChooser(
            default='ldemnrs',
            ldemnrs=ldemnrs,
            ldems=ldems,
        )
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        tf = 0.1
        dt = self.dt
        scheme.configure()
        scheme.configure_solver(kernel=kernel,
                                integrator_cls=EPECIntegratorMultiStage, dt=dt,
                                tf=tf, pfreq=10)

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
                UpdateTangentialContactsNoRotation(dest='spheres',
                                                   sources=["spheres"])
            ])
        ]
        arrays = self.particles
        a_eval = self._make_accel_eval(eqs1, arrays)

        # When
        a_eval.evaluate(t, dt)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['spheres']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {radius}
        b.scalar = 'm'
        '''.format(radius=self.radius))


if __name__ == '__main__':
    app = BinaryImpact2d(theta=0.)
    app.run()
