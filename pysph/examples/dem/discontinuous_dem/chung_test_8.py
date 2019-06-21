"""
This example is a benchmark of DEM numerical method. This is a test eight,
where impact of two differently sized spheres with a constant normal velocity
and varying angular velocities is studied.

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


class Test8(Application):
    def __init__(self, ang_vel=0.1, case_a=True, case_b=False):
        self.ang_vel = ang_vel
        self.norm_vel = 0.2
        self.case_a = case_a
        self.case_b = case_b
        super(Test8, self).__init__()

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
        rad_s = np.array([self.radius, 5. * self.radius])
        dia_s = 2. * np.array([self.radius, 5. * self.radius])
        xaa = np.array([-rad_s[0] - scale, rad_s[1] + scale])
        yaa = np.array([0., 0.])
        u = np.array([self.norm_vel, 0.0])
        wz = np.array([self.ang_vel, 0.])
        rho = np.array([2700., 1000. * 2700.])
        yng_m = 7. * 1e10
        poissons_ratio = 0.33
        if self.case_a:
            shear_m = 1000. * yng_m / (2. * (1. + poissons_ratio))
        if self.case_b:
            shear_m = yng_m / (2. * (1. + poissons_ratio))

        m = rho * 4. / 3. * np.pi * rad_s**3.
        inertia = m * dia_s**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_aa = get_particle_array_dem(
            x=xaa, y=yaa, u=u, wz=wz, h=h, m=m, rho=rho, rad_s=rad_s,
            yng_m=yng_m, poissons_ratio=poissons_ratio, shear_m=shear_m,
            dem_id=0, m_inv=m_inv, I_inv=I_inv, name="spheres_aa")

        # nylon (n) particle positions
        scale = 0.000001
        rad_s = np.array([self.radius, 5. * self.radius])
        dia_s = 2. * np.array([self.radius, 5. * self.radius])
        xn = np.array([-rad_s[0] - scale, rad_s[1] + scale])
        yn = np.array([0., 0.])
        u = np.array([self.norm_vel, 0.0])
        wz = np.array([self.ang_vel, 0.])
        rho = np.array([1000, 1000. * 1000.])
        yng_m = 2.5 * 1e9
        poissons_ratio = 0.4
        if self.case_a:
            shear_m = 1000. * yng_m / (2. * (1. + poissons_ratio))
        if self.case_b:
            shear_m = yng_m / (2. * (1. + poissons_ratio))

        m = rho * 4. / 3. * np.pi * rad_s**3.
        inertia = m * dia_s**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_n = get_particle_array_dem(
            x=xn, y=yn, u=u, wz=wz, h=h, m=m, rho=rho, rad_s=rad_s,
            yng_m=yng_m, poissons_ratio=poissons_ratio, shear_m=shear_m,
            dem_id=0, m_inv=m_inv, I_inv=I_inv, name="spheres_n")
        return [spheres_aa, spheres_n]

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegrator(spheres_aa=RK2StepNonLinearDEM(),
                                    spheres_n=RK2StepNonLinearDEM())

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
                ResetForces(dest='spheres_n', sources=None),
                TsuijiNonLinearParticleParticleForceStage1(
                    dest='spheres_aa', sources=["spheres_aa"], en=self.en,
                    mu=self.mu),
                TsuijiNonLinearParticleParticleForceStage1(
                    dest='spheres_n', sources=["spheres_n"], en=self.en,
                    mu=self.mu),
            ]),
        ]

        stage2 = [
            Group(equations=[
                ResetForces(dest='spheres_aa', sources=None),
                ResetForces(dest='spheres_n', sources=None),
                TsuijiNonLinearParticleParticleForceStage2(
                    dest='spheres_aa', sources=["spheres_aa"], en=self.en,
                    mu=self.mu),
                TsuijiNonLinearParticleParticleForceStage2(
                    dest='spheres_n', sources=["spheres_n"], en=self.en,
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
                UpdateTangentialContacts(dest='spheres_n',
                                         sources=["spheres_n"]),
            ]),
        ]
        arrays = self.particles
        a_eval = self._make_accel_eval(eqs1, arrays)

        # When
        a_eval.evaluate(t, dt)

    def post_process(self, sim_data):
        import matplotlib.pyplot as plt
        ################
        # recoil angle #
        ################
        # al alloy data
        # real data
        data = np.loadtxt('./chung_test_8_recoil_case_a_aa.csv', delimiter=',')
        incident_angle_scraped_case_a_aa, recoil_scraped_case_a_aa = (
            data[:, 0], data[:, 1])

        data = np.loadtxt('./chung_test_8_recoil_case_b_aa.csv', delimiter=',')
        incident_angle_scraped_case_b_aa, recoil_scraped_case_b_aa = (
            data[:, 0], data[:, 1])

        # simulated data
        incident_angle_sim_case_a_aa, recoil_sim_case_a_aa = (
            sim_data['incident_angle_sim_case_a_aa'],
            sim_data['recoil_sim_case_a_aa'])
        incident_angle_sim_case_b_aa, recoil_sim_case_b_aa = (
            sim_data['incident_angle_sim_case_b_aa'],
            sim_data['recoil_sim_case_b_aa'])

        plt.plot(incident_angle_scraped_case_a_aa, recoil_scraped_case_a_aa,
                 label='Chung Data of Al. alloy (case A)')
        plt.plot(incident_angle_sim_case_a_aa, recoil_sim_case_a_aa,
                 label='PySPH simulation of Al. alloy (case A)')

        plt.plot(incident_angle_scraped_case_b_aa, recoil_scraped_case_b_aa,
                 label='Chung Data of Al. alloy (case B)')
        plt.plot(incident_angle_sim_case_b_aa, recoil_sim_case_b_aa,
                 label='PySPH simulation of Al. alloy (case B)')

        plt.legend()
        plt.xlabel(r"$Vs/Vn$")
        plt.ylabel(r"$V'_{st}/V'_{cn}$")
        plt.xlim([0., 8.])
        plt.ylim([-2., 10.])
        import os
        fig = os.path.join("test_8_incident_vs_recoil.png")
        plt.savefig(fig, dpi=300)
        plt.show()
        #####################
        # recoil angle ends #
        #####################

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
    # application run imports
    import tempfile
    import shutil

    # ang_vel = [
    #     0.175, 0.2, 0.3, 0.5, 0.7, 0.9, 1.3, 2., 3., 4., 5., 7., 9., 12., 22.
    # ]
    ang_vel = [
        0.175,
    ]

    # case A
    # for al alloy (aa)
    incident_angle_case_a_aa = []
    recoil_case_a_aa = []

    for i in ang_vel:
        app = Test8(ang_vel=i, case_a=True, case_b=False)
        tmpdir = tempfile.mkdtemp()
        app.run(['--disable-output', '-d', tmpdir])
        shutil.rmtree(tmpdir)

        particles = app.particles
        incident_angle_case_a_aa.append(
            app.ang_vel * app.radius / app.norm_vel)
        recoil_case_a_aa.append(particles[0].v[0] / particles[0].u[0])

    # save the simulation data for the post processing
    data = {}
    data['incident_angle_sim_case_a_aa'] = incident_angle_case_a_aa
    data['recoil_sim_case_a_aa'] = recoil_case_a_aa

    # case B
    # for al alloy (aa)
    incident_angle_case_b_aa = []
    recoil_case_b_aa = []
    for i in ang_vel:
        app = Test8(ang_vel=i, case_a=True, case_b=False)
        tmpdir = tempfile.mkdtemp()
        app.run(['--disable-output', '-d', tmpdir])
        shutil.rmtree(tmpdir)

        particles = app.particles
        incident_angle_case_b_aa.append(
            app.ang_vel * app.radius / app.norm_vel)
        recoil_case_b_aa.append(particles[0].v[0] / particles[0].u[0])

    # save the simulation data for the post processing
    data['incident_angle_sim_case_b_aa'] = incident_angle_case_b_aa
    data['recoil_sim_case_b_aa'] = recoil_case_b_aa

    app.post_process(data)
