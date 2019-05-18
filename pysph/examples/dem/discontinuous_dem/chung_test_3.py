"""
This example is a benchmark of DEM numerical method.
This is test three, where one spheres will have an elastic normal
impact with a wall with different restitution coefficients.

Link: https://link.springer.com/article/10.1007/s10035-011-0277-0
"""
from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator

from pysph.sph.equation import Group, MultiStageEquations
from pysph.solver.application import Application
from pysph.dem.dem_nonlinear import (get_particle_array_dem,
                                     RK2StepNonLinearDEM, ResetForces,
                                     TsuijiNonLinearParticleWallForceStage1,
                                     TsuijiNonLinearParticleWallForceStage2)


class Test3(Application):
    def __init__(self, fname, en=1.0):
        self.en = en
        super(Test3, self).__init__(fname)

    def initialize(self):
        self.radius = 0.0025
        self.diameter = 2. * self.radius
        self.tf = 0.00003
        self.dt = 1e-6
        self.dim = 3

    def create_particles(self):
        # al oxide (ao) particle positions
        scale = 0.000001
        xao = np.array([-self.radius - scale])
        yao = np.array([0.])
        u = np.array([3.9])
        rad_s = self.radius
        rho = 4000.
        yng_m = 3.8 * 1e11
        poissons_ratio = 0.23
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        m = rho * 4. / 3. * np.pi * self.radius**3.
        inertia = m * self.diameter**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_ao = get_particle_array_dem(
            x=xao, y=yao, u=u, h=h, m=m, rho=rho, rad_s=rad_s, yng_m=yng_m,
            poissons_ratio=poissons_ratio, shear_m=shear_m, dem_id=0,
            m_inv=m_inv, I_inv=I_inv, name="spheres_ao")

        # wall al oxide
        xw_ao = np.array([0.])
        yw_ao = np.array([0.])
        nxw_ao = np.array([-1.])
        nyw_ao = np.array([0.])
        rho = 4000.
        yng_m = 3.8 * 1e11
        poissons_ratio = 0.23
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        wall_ao = get_particle_array_dem(
            x=xw_ao, y=yw_ao, nx=nxw_ao, ny=nyw_ao, nz=0., rho=rho,
            yng_m=yng_m, poissons_ratio=poissons_ratio, shear_m=shear_m,
            dem_id=1, constants={'np': len(xw_ao)}, name="wall_ao")
        wall_ao.set_output_arrays([
            'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au',
            'av', 'aw', 'tag', 'gid', 'nx', 'ny', 'nz'
        ])

        # cast iron (ci) particle positions
        xci = np.array([-self.radius - scale])
        yci = np.array([3. * self.radius])
        u = np.array([3.9])
        rad_s = self.radius
        rho = 7000.
        yng_m = 1. * 1e11
        poissons_ratio = 0.25
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        m = rho * 4. / 3. * np.pi * self.radius**3.
        inertia = m * self.diameter**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_ci = get_particle_array_dem(
            x=xci, y=yci, u=u, h=h, m=m, rho=rho, rad_s=rad_s, yng_m=yng_m,
            poissons_ratio=poissons_ratio, shear_m=shear_m, dem_id=0,
            m_inv=m_inv, I_inv=I_inv, name="spheres_ci")

        # wall cast iron
        xw_ci = np.array([0.])
        yw_ci = np.array([3. * self.radius])
        nxw_ci = np.array([-1.])
        nyw_ci = np.array([0.])
        rho = 7000.
        yng_m = 1. * 1e11
        poissons_ratio = 0.25
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        wall_ci = get_particle_array_dem(
            x=xw_ci, y=yw_ci, nx=nxw_ci, ny=nyw_ci, nz=0., rho=rho,
            yng_m=yng_m, poissons_ratio=poissons_ratio, shear_m=shear_m,
            dem_id=1, constants={'np': len(xw_ci)}, name="wall_ci")

        wall_ci.set_output_arrays([
            'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au',
            'av', 'aw', 'tag', 'gid', 'nx', 'ny', 'nz'
        ])
        return [spheres_ao, spheres_ci, wall_ao, wall_ci]

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegrator(spheres_ao=RK2StepNonLinearDEM(),
                                    spheres_ci=RK2StepNonLinearDEM())

        dt = self.dt
        tf = self.tf
        if dt < 1e-5:
            pfreq = 1000
        else:
            pfreq = 100

        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
                        dt=dt, tf=tf, pfreq=pfreq)

        return solver

    def create_equations(self):
        stage1 = [
            Group(equations=[
                ResetForces(dest='spheres_ao', sources=None),
                ResetForces(dest='spheres_ci', sources=None),
                TsuijiNonLinearParticleWallForceStage1(
                    dest='spheres_ao', sources=["wall_ao"], en=self.en, mu=0.),
                TsuijiNonLinearParticleWallForceStage1(
                    dest='spheres_ci', sources=["wall_ci"], en=self.en, mu=0.)
            ]),
        ]

        stage2 = [
            Group(equations=[
                ResetForces(dest='spheres_ao', sources=None),
                ResetForces(dest='spheres_ci', sources=None),
                TsuijiNonLinearParticleWallForceStage2(
                    dest='spheres_ao', sources=["wall_ao"
                                                ], en=self.en, mu=0.0),
                TsuijiNonLinearParticleWallForceStage2(
                    dest='spheres_ci', sources=["wall_ci"], en=self.en, mu=0.0)
            ]),
        ]
        return MultiStageEquations([stage1, stage2])

    def _make_accel_eval(self, equations, pa_arrays):
        from pysph.tools.sph_evaluator import SPHEvaluator
        kernel = CubicSpline(dim=self.dim)
        seval = SPHEvaluator(arrays=pa_arrays, equations=equations,
                             dim=self.dim, kernel=kernel)
        return seval

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['spheres_ao']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {radius}
        b.scalar = 'u'
        b.show_legend = True
        b = particle_arrays['spheres_ci']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {radius}
        b.scalar = 'u'
        b.show_legend = True
        '''.format(radius=self.radius))


if __name__ == '__main__':
    en = [0.2, 0.4, 0.6, 0.8, 1.0]
    for i in en:
        app = Test3(fname="chung_test_3_" + str(i), en=i)
        app.run()
        particles = app.particles[1]
        print("en is " + str(i))
        print(particles.u[0] / -3.9)
