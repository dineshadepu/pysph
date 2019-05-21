"""
This example is a benchmark of DEM numerical method. This is a test six,
where impact of two identical spheres with a constant normal velocity and
varying angular velocities is studied.

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
from pysph.dem.discontinuous_dem.dem_nonlinear import (
    get_particle_array_dem, RK2StepNonLinearDEM, ResetForces,
    TsuijiNonLinearParticleParticleForceStage1,
    TsuijiNonLinearParticleParticleForceStage2, UpdateTangentialContacts)


class Test7(Application):
    def __init__(self, ang_vel):
        self.ang_vel = ang_vel
        self.norm_vel = 0.2
        super(Test7, self).__init__()

    def initialize(self):
        self.radius = 0.1
        self.diameter = 2. * self.radius
        self.tf = 0.000012
        self.dt = 1e-6
        self.dim = 3
        # for both particles restitution
        self.en = 0.5
        # friction coefficient
        self.mu = 0.4

    def create_particles(self):
        # al alloy (aa) particle positions
        scale = 0.000001
        xaa = np.array([-self.radius - scale, self.radius + scale])
        yaa = np.array([0., 0.])
        u = np.array([0.2, -0.2])
        wz = np.array([self.ang_vel, -self.ang_vel])
        rad_s = self.radius
        rho = 2700.
        yng_m = 7. * 1e11
        poissons_ratio = 0.33
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        m = rho * 4. / 3. * np.pi * self.radius**3.
        inertia = m * self.diameter**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_aa = get_particle_array_dem(
            x=xaa, y=yaa, u=u, wz=wz, h=h, m=m, rho=rho, rad_s=rad_s,
            yng_m=yng_m, poissons_ratio=poissons_ratio, shear_m=shear_m,
            dem_id=0, m_inv=m_inv, I_inv=I_inv, name="spheres_aa")

        # Copper (c) particle positions
        scale = 0.000001
        xc = np.array([-self.radius - scale, self.radius + scale])
        yc = np.array([0., 0.])
        u = np.array([0.2, -0.2])
        wz = np.array([self.ang_vel, -self.ang_vel])
        rad_s = self.radius
        rho = 8900
        yng_m = 1.2 * 1e11
        poissons_ratio = 0.35
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        m = rho * 4. / 3. * np.pi * self.radius**3.
        inertia = m * self.diameter**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_c = get_particle_array_dem(
            x=xc, y=yc, u=u, wz=wz, h=h, m=m, rho=rho, rad_s=rad_s,
            yng_m=yng_m, poissons_ratio=poissons_ratio, shear_m=shear_m,
            dem_id=0, m_inv=m_inv, I_inv=I_inv, name="spheres_c")
        return [spheres_aa, spheres_c]

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegrator(spheres_aa=RK2StepNonLinearDEM(),
                                    spheres_c=RK2StepNonLinearDEM())

        dt = self.dt
        tf = self.tf
        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
                        dt=dt, tf=tf)
        solver.set_disable_output(True)
        return solver

    def create_equations(self):
        stage1 = [
            Group(equations=[
                ResetForces(dest='spheres_aa', sources=None),
                ResetForces(dest='spheres_c', sources=None),
                TsuijiNonLinearParticleParticleForceStage1(
                    dest='spheres_aa', sources=["spheres_aa"], en=self.en,
                    mu=self.mu),
                TsuijiNonLinearParticleParticleForceStage1(
                    dest='spheres_c', sources=["spheres_c"], en=self.en,
                    mu=self.mu),
            ]),
        ]

        stage2 = [
            Group(equations=[
                ResetForces(dest='spheres_aa', sources=None),
                ResetForces(dest='spheres_c', sources=None),
                TsuijiNonLinearParticleParticleForceStage2(
                    dest='spheres_aa', sources=["spheres_aa"], en=self.en,
                    mu=self.mu),
                TsuijiNonLinearParticleParticleForceStage2(
                    dest='spheres_c', sources=["spheres_c"], en=self.en,
                    mu=self.mu),
            ]),
        ]
        return MultiStageEquations([stage1, stage2])

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
                UpdateTangentialContacts(dest='spheres_aa',
                                         sources=["spheres_aa"]),
                UpdateTangentialContacts(dest='spheres_c',
                                         sources=["spheres_c"]),
            ]),
        ]
        arrays = self.particles
        a_eval = self._make_accel_eval(eqs1, arrays)

        # When
        a_eval.evaluate(t, dt)

    def post_process(self, sim_data):
        import matplotlib.pyplot as plt
        #####################
        # angular velocity #
        #####################
        # al alloy data
        # real data
        data = np.loadtxt(
            './chung_test_7_pre_c_ang_vel_vs_post_c_ang_vel_al_alloy.csv',
            delimiter=',')
        pre_c_ang_vel_scraped_aa, post_c_ang_vel_scraped_aa = (data[:, 0],
                                                               data[:, 1])
        # copper (c) data
        # real data
        data = np.loadtxt(
            './chung_test_7_pre_c_ang_vel_vs_post_c_ang_vel_copper.csv',
            delimiter=',')
        pre_c_ang_vel_scraped_c, post_c_ang_vel_scraped_c = (data[:, 0],
                                                             data[:, 1])

        # simulated data
        pre_c_ang_vel_sim_aa, post_c_ang_vel_sim_aa = (
            sim_data['pre_c_ang_vel_sim_aa'],
            sim_data['post_c_ang_vel_sim_aa'])
        pre_c_ang_vel_sim_c, post_c_ang_vel_sim_c = (
            sim_data['pre_c_ang_vel_sim_c'], sim_data['post_c_ang_vel_sim_c'])

        plt.plot(pre_c_ang_vel_scraped_aa, post_c_ang_vel_scraped_aa,
                 label='Chung Data of Al. alloy')
        plt.plot(pre_c_ang_vel_sim_aa, post_c_ang_vel_sim_aa,
                 label='PySPH simulation of Al. alloy')

        plt.plot(pre_c_ang_vel_scraped_c, post_c_ang_vel_scraped_c,
                 label='Chung Data of Copper')
        plt.plot(pre_c_ang_vel_sim_c, post_c_ang_vel_sim_c,
                 label='PySPH simulation of Copper')

        plt.legend()
        plt.xlabel(r'\omega_1')
        plt.ylabel(r'V^{\'}_{ct, 1}')
        plt.xlim([0., 25.])
        plt.ylim([-10., 10.])
        import os
        fig = os.path.join("test_7_pre_ang_vel_vs_post_ang_vel.png")
        plt.savefig(fig, dpi=300)
        # plt.show()
        ##########################
        # angular velocity ends #
        ##########################

        #######################
        # tangential velocity #
        #######################
        # al alloy data
        # real data
        plt.clf()
        data = np.loadtxt(
            './chung_test_7_pre_c_ang_vel_vs_post_c_tng_vel_al_alloy.csv',
            delimiter=',')
        pre_c_ang_vel_scraped_aa, post_c_tng_vel_scraped_aa = (data[:, 0],
                                                               data[:, 1])
        # copper
        # real data
        data = np.loadtxt(
            './chung_test_7_pre_c_ang_vel_vs_post_c_tng_vel_copper.csv',
            delimiter=',')
        pre_c_ang_vel_scraped_c, post_c_tng_vel_scraped_c = (data[:, 0],
                                                             data[:, 1])

        # simulated data
        pre_c_ang_vel_sim_aa, post_c_tng_vel_sim_aa = (
            sim_data['pre_c_ang_vel_sim_aa'],
            sim_data['post_c_tng_vel_sim_aa'])
        pre_c_ang_vel_sim_c, post_c_tng_vel_sim_c = (
            sim_data['pre_c_ang_vel_sim_c'], sim_data['post_c_tng_vel_sim_c'])

        plt.plot(pre_c_ang_vel_scraped_aa, post_c_tng_vel_scraped_aa,
                 label='Chung Data of Al. alloy')
        plt.plot(pre_c_ang_vel_sim_aa, post_c_tng_vel_sim_aa,
                 label='PySPH simulation of Al. alloy')

        plt.plot(pre_c_ang_vel_scraped_c, post_c_tng_vel_scraped_c,
                 label='Chung Data of Copper')
        plt.plot(pre_c_ang_vel_sim_c, post_c_tng_vel_sim_c,
                 label='PySPH simulation of Copper')

        plt.legend()
        plt.xlabel(r'\omega_1')
        plt.ylabel(r'V^{\'}_{ct, 1}')
        plt.xlim([0., 25.])
        plt.ylim([-10., 10.])
        import os
        fig = os.path.join("test_7_pre_ang_vel_vs_post_tng_vel.png")
        plt.savefig(fig, dpi=300)
        # plt.show()
        #############################
        # tangential velocity ends #
        #############################

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['spheres_aa']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {radius}
        b.scalar = 'v'
        b = particle_arrays['spheres_c']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {radius}
        b.scalar = 'v'
        '''.format(radius=self.radius))


if __name__ == '__main__':
    ang_vel = [
        0.175, 0.2, 0.3, 0.5, 0.7, 0.9, 1.3, 2., 3., 4., 5., 7., 9., 12., 22.
    ]

    # for al alloy (aa) (pre-collision (pre_c))
    pre_c_ang_vel_aa = []
    post_c_ang_vel_aa = []
    post_c_tng_vel_aa = []

    # for copper (c)
    pre_c_ang_vel_c = []
    post_c_ang_vel_c = []
    post_c_tng_vel_c = []

    for i in ang_vel:
        app = Test7(ang_vel=i)
        import tempfile
        tmpdir = tempfile.mkdtemp()
        app.run(['--disable-output', '-d', tmpdir])
        import shutil
        shutil.rmtree(tmpdir)

        particles = app.particles
        pre_c_ang_vel_aa.append(i)
        post_c_ang_vel_aa.append(particles[0].wz[0])
        post_c_tng_vel_aa.append(particles[0].v[0])

        pre_c_ang_vel_c.append(i)
        post_c_ang_vel_c.append(particles[1].wz[0])
        post_c_tng_vel_c.append(particles[1].v[0])

    # save the simulation data for the post processing
    data = {}
    data['pre_c_ang_vel_aa'] = pre_c_ang_vel_aa
    data['post_c_ang_vel_aa'] = post_c_ang_vel_aa
    data['post_c_tng_vel_aa'] = post_c_tng_vel_aa

    data['pre_c_ang_vel_c'] = pre_c_ang_vel_c
    data['post_c_ang_vel_c'] = post_c_ang_vel_c
    data['post_c_tng_vel_c'] = post_c_tng_vel_c

    app.post_process(data)
