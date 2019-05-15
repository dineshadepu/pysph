"""
This example is a benchmark DEM numerical method.
This is a test two, where one spheres will have an elastic normal
impact with a wall.

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


class Test2(Application):
    def initialize(self):
        self.radius = 0.10
        self.diameter = 2. * self.radius
        self.tf = 0.00105
        self.dt = 1e-6
        self.dim = 3

    def create_particles(self):
        # al alloy particle positions
        scale = 0.000001
        xa = np.array([-self.radius - scale])
        ya = np.array([0.])
        u = np.array([0.2])
        rad_s = self.radius
        rho = 2699.
        yng_m = 7. * 1e10
        poissons_ratio = 0.30
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        m = rho * 4. / 3. * np.pi * self.radius**3.
        inertia = m * self.diameter**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_al = get_particle_array_dem(
            x=xa, y=ya, u=u, h=h, m=m, rho=rho, rad_s=rad_s, yng_m=yng_m,
            poissons_ratio=poissons_ratio, shear_m=shear_m, dem_id=0,
            m_inv=m_inv, I_inv=I_inv, name="spheres_al")

        # wall al alloy
        xw_a = np.array([0.])
        yw_a = np.array([0.])
        nxw_a = np.array([-1.])
        nyw_a = np.array([0.])
        rho = 2699.
        yng_m = 7. * 1e10
        poissons_ratio = 0.30
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        wall_al = get_particle_array_dem(
            x=xw_a, y=yw_a, nx=nxw_a, ny=nyw_a, nz=0., rho=rho, yng_m=yng_m,
            poissons_ratio=poissons_ratio, shear_m=shear_m, dem_id=1,
            constants={'np': len(xw_a)}, name="wall_al")

        # mg alloy particle positions
        xm = np.array([-self.radius - scale])
        ym = np.array([3. * self.radius])
        u = np.array([0.2])
        rad_s = self.radius
        rho = 1800.
        yng_m = 4. * 1e10
        poissons_ratio = 0.35
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        m = rho * 4. / 3. * np.pi * self.radius**3.
        inertia = m * self.diameter**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_mg = get_particle_array_dem(
            x=xm, y=ym, u=u, h=h, m=m, rho=rho, rad_s=rad_s, yng_m=yng_m,
            poissons_ratio=poissons_ratio, shear_m=shear_m, dem_id=0,
            m_inv=m_inv, I_inv=I_inv, name="spheres_mg")

        # wall mg alloy
        xw_m = np.array([0.])
        yw_m = np.array([3. * self.radius])
        nxw_m = np.array([-1.])
        nyw_m = np.array([0.])
        rho = 1800.
        yng_m = 4. * 1e10
        poissons_ratio = 0.35
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        wall_mg = get_particle_array_dem(
            x=xw_m, y=yw_m, nx=nxw_m, ny=nyw_m, nz=0., rho=rho, yng_m=yng_m,
            poissons_ratio=poissons_ratio, shear_m=shear_m, dem_id=1,
            constants={'np': len(xw_a)}, name="wall_mg")
        return [spheres_al, spheres_mg, wall_al, wall_mg]

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegrator(spheres_al=RK2StepNonLinearDEM(),
                                    spheres_mg=RK2StepNonLinearDEM())

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
                ResetForces(dest='spheres_al', sources=None),
                ResetForces(dest='spheres_mg', sources=None),
                TsuijiNonLinearParticleWallForceStage1(
                    dest='spheres_al', sources=["wall_al"], en=1, mu=0.),
                TsuijiNonLinearParticleWallForceStage1(
                    dest='spheres_mg', sources=["wall_mg"], en=1, mu=0.)
            ]),
        ]

        stage2 = [
            Group(equations=[
                ResetForces(dest='spheres_al', sources=None),
                ResetForces(dest='spheres_mg', sources=None),
                TsuijiNonLinearParticleWallForceStage2(
                    dest='spheres_al', sources=["wall_al"], en=1., mu=0.0),
                TsuijiNonLinearParticleWallForceStage2(
                    dest='spheres_mg', sources=["wall_mg"], en=1, mu=0.0)
            ]),
        ]
        return MultiStageEquations([stage1, stage2])

    def post_process(self):
        # Mg data
        # real data
        data = np.loadtxt('./chung_test_2_mg.csv', delimiter=',')
        tl_r, fn_l_r = data[:, 0], data[:, 1]

        # al data
        # real data
        data = np.loadtxt('./chung_test_2_al.csv', delimiter=',')
        tg_r, fn_g_r = data[:, 0], data[:, 1]

        if len(self.output_files) == 0:
            return
        from pysph.solver.utils import iter_output
        files = self.output_files
        t, fn_g, fn_l = [], [], []
        for sd, arrays in iter_output(files):
            al, mg = arrays['spheres_al'], arrays['spheres_mg'],
            t.append(sd['t'])
            fn_g.append(-al.fx[0])
            fn_l.append(-mg.fx[0])
        t = np.asarray(t)
        fn_g = np.asarray(fn_g)
        fn_l = np.asarray(fn_l)
        t = t * 1e6
        fn_g = fn_g * 1e-3
        fn_l = fn_l * 1e-3

        import matplotlib.pyplot as plt
        plt.plot(t, fn_g, label='al')
        plt.plot(t, fn_l, label='mg')
        plt.scatter(tg_r, fn_g_r, label='al_data')
        plt.scatter(tl_r, fn_l_r, label='mg_data')
        plt.legend()
        plt.xlim([0.0, 1000])
        plt.ylim([0.0, 12])
        import os
        fig = os.path.join(self.output_dir, "force_vs_time.png")
        plt.show()
        plt.savefig(fig, dpi=300)

    def _make_accel_eval(self, equations, pa_arrays):
        from pysph.tools.sph_evaluator import SPHEvaluator
        kernel = CubicSpline(dim=self.dim)
        seval = SPHEvaluator(arrays=pa_arrays, equations=equations,
                             dim=self.dim, kernel=kernel)
        return seval

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        # eqs1 = [
        #     Group(equations=[
        #         UpdateTangentialContacts(dest='spheres_al',
        #                                  sources=["spheres_al"]),
        #         UpdateTangentialContacts(dest='spheres_mg',
        #                                  sources=["spheres_mg"])
        #     ]),
        # ]
        # arrays = self.particles
        # a_eval = self._make_accel_eval(eqs1, arrays)

        # # When
        # a_eval.evaluate(t, dt)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['spheres_al']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = 0.010
        b.scalar = 'fx'
        b = particle_arrays['spheres_mg']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = 0.010
        b.scalar = 'fx'
        ''')


if __name__ == '__main__':
    app = Test2()
    app.run()
    app.post_process()
