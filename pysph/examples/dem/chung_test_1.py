"""
This example is a benchmark one of the benchmarks of DEM numerical method.
This is a first test, where two identical spheres will have an elastic normal
impact.

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
from pysph.dem.dem_nonlinear import (
    get_particle_array_dem, RK2StepNonLinearDEM,
    ResetForces,
    TsuijiNonLinearParticleParticleForceStage1,
    TsuijiNonLinearParticleParticleForceStage2)


class Test1(Application):
    def initialize(self):
        self.radius = 0.010
        self.diameter = 2. * self.radius
        self.tf = 0.00007
        self.dt = 1e-6
        self.dim = 3

    def create_particles(self):
        # glass particle positions
        scale = 0.000001
        # scale = self.radius / 30.
        xg = np.array([-self.radius - scale, self.radius + scale])
        yg = np.array([0., 0.])
        u = np.array([10., -10.])
        rad_s = self.radius
        rho = 2800.
        yng_m = 4.8 * 1e10
        poissons_ratio = 0.20
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        m = rho * 4. / 3. * np.pi * self.radius**3.
        inertia = m * self.diameter**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_glass = get_particle_array_dem(
            x=xg, y=yg, u=u, h=h, m=m, rho=rho, rad_s=rad_s, yng_m=yng_m,
            poissons_ratio=poissons_ratio, shear_m=shear_m, dem_id=0,
            m_inv=m_inv, I_inv=I_inv, name="spheres_glass")

        # limestone particle positions
        xl = np.array([-self.radius - scale, self.radius + scale])
        yl = np.array([3. * self.radius, 3. * self.radius])
        u = np.array([10., -10.])
        rad_s = self.radius
        rho = 2500.
        yng_m = 2. * 1e10
        poissons_ratio = 0.25
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        m = rho * 4. / 3. * np.pi * self.radius**3.
        inertia = m * self.diameter**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_limestone = get_particle_array_dem(
            x=xl, y=yl, u=u, h=h, m=m, rho=rho, rad_s=rad_s, yng_m=yng_m,
            poissons_ratio=poissons_ratio, shear_m=shear_m, dem_id=0,
            m_inv=m_inv, I_inv=I_inv, name="spheres_limestone")
        return [spheres_glass, spheres_limestone]

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegrator(spheres_glass=RK2StepNonLinearDEM(),
                                    spheres_limestone=RK2StepNonLinearDEM())

        dt = self.dt
        tf = self.tf
        if dt < 1e-5:
            pfreq = 1000
        else:
            pfreq = 100

        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator, dt=dt,
                        tf=tf, pfreq=pfreq)

        return solver

    def create_equations(self):
        stage1 = [
            Group(equations=[
                ResetForces(dest='spheres_glass', sources=None),
                ResetForces(dest='spheres_limestone', sources=None),
                TsuijiNonLinearParticleParticleForceStage1(
                    dest='spheres_glass', sources=["spheres_glass"], en=0.1,
                    mu=0.3),
                TsuijiNonLinearParticleParticleForceStage1(
                    dest='spheres_limestone', sources=["spheres_limestone"],
                    en=0.1, mu=0.3)
            ]),
        ]

        stage2 = [
            Group(equations=[
                ResetForces(dest='spheres_glass', sources=None),
                ResetForces(dest='spheres_limestone', sources=None),
                TsuijiNonLinearParticleParticleForceStage2(
                    dest='spheres_glass', sources=["spheres_glass"], en=0.1,
                    mu=0.3),
                TsuijiNonLinearParticleParticleForceStage2(
                    dest='spheres_limestone', sources=["spheres_limestone"],
                    en=0.1, mu=0.3)
            ]),
        ]
        return MultiStageEquations([stage1, stage2])

    def post_process(self):
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        files = self.output_files
        t, fn_g, fn_l = [], [], []
        for sd, arrays in iter_output(files):
            glass, limestone = arrays['spheres_glass'], arrays[
                'spheres_limestone'],
            t.append(sd['t'])
            fn_g.append(glass.fx[1])
            fn_l.append(limestone.fx[1])
        t = np.asarray(t)
        fn_g = np.asarray(fn_g)
        fn_l = np.asarray(fn_l)
        t = t * 1e6
        fn_g = fn_g * 1e-3
        fn_l = fn_l * 1e-3

        # Limestone data
        # real data
        data = np.loadtxt('chung_test_1_limestone.csv', delimiter=',')
        tl_r, fn_l_r = data[:, 0], data[:, 1]

        # glass data
        # real data
        data = np.loadtxt('chung_test_1_glass.csv', delimiter=',')
        tg_r, fn_g_r = data[:, 0], data[:, 1]

        import matplotlib.pyplot as plt
        plt.scatter(t, fn_g, label='glass')
        plt.scatter(t, fn_l, label='limestone')
        plt.plot(tg_r, fn_g_r, label='glass_data')
        plt.plot(tl_r, fn_l_r, label='limestone_data')
        plt.legend()
        plt.xlim([0.0, 60])
        plt.ylim([0.0, 12])
        import os
        fig = os.path.join(self.output_dir, "force_vs_time.png")
        plt.show()
        plt.savefig(fig, dpi=300)

    def _make_accel_eval(self, equations, pa_arrays):
        from pysph.tools.sph_evaluator import SPHEvaluator
        kernel = CubicSpline(dim=self.dim)
        seval = SPHEvaluator(
            arrays=pa_arrays, equations=equations, dim=self.dim,
            kernel=kernel)
        return seval

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        # eqs1 = [
        #     Group(equations=[
        #         UpdateTangentialContacts(dest='spheres_glass',
        #                                  sources=["spheres_glass"]),
        #         UpdateTangentialContacts(dest='spheres_limestone',
        #                                  sources=["spheres_limestone"])
        #     ]),
        # ]
        # arrays = self.particles
        # a_eval = self._make_accel_eval(eqs1, arrays)

        # # When
        # a_eval.evaluate(t, dt)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['spheres_glass']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = 0.010
        b.scalar = 'fx'
        b = particle_arrays['spheres_limestone']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = 0.010
        b.scalar = 'fx'
        ''')


if __name__ == '__main__':
    app = Test1()
    app.run()
    app.post_process()
