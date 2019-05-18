"""
This example is a benchmark of DEM numerical method. This is test four,
where oblique impact of a sphere with a rigid plane with a constant resultant
velocity but at different incident angles is studied


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
    get_particle_array_dem, RK2StepNonLinearDEM, ResetForces,
    TsuijiNonLinearParticleWallForceStage1,
    TsuijiNonLinearParticleWallForceStage2, UpdateTangentialContacts)


class Test4(Application):
    def __init__(self, fname, theta=30.):
        self.incident_angle = theta
        super(Test4, self).__init__(fname)

    def initialize(self):
        self.radius = 0.0025
        self.diameter = 2. * self.radius
        self.tf = 0.0003
        self.dt = 1e-6
        self.dim = 3
        # for both particles restitution
        self.en = 0.98
        # friction coefficient
        self.mu = 0.092

    def create_particles(self):
        # al oxide (ao) particle positions
        scale = 0.0001
        xao = np.array([0.])
        yao = np.array([self.radius + scale])
        yng_m = 3.8 * 1e11
        poissons_ratio = 0.23
        rho = 4000.
        rad_s = self.radius
        # u, v varies depending on the angle of impact
        velocity = 3.9
        radians = self.incident_angle * np.pi / 180.
        u = np.asarray([velocity * np.sin(radians)])
        v = -np.asarray([velocity * np.cos(radians)])
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        m = rho * 4. / 3. * np.pi * self.radius**3.
        inertia = m * self.diameter**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_ao = get_particle_array_dem(
            x=xao, y=yao, u=u, v=v, h=h, m=m, rho=rho, rad_s=rad_s,
            yng_m=yng_m, poissons_ratio=poissons_ratio, shear_m=shear_m,
            dem_id=0, m_inv=m_inv, I_inv=I_inv, name="spheres_ao")

        # wall al oxide
        xw_ao = np.array([0.])
        yw_ao = np.array([0.])
        nxw_ao = np.array([0.])
        nyw_ao = np.array([1.])
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

        # al alloy (aa) particle positions
        xaa = np.array([3. * self.radius + scale])
        yaa = np.array([self.radius + scale])
        yng_m = 7. * 1e10
        poissons_ratio = 0.33
        rho = 2700.
        rad_s = self.radius
        # u, v varies depending on the angle of impact
        velocity = 3.9
        radians = self.incident_angle * np.pi / 180.
        u = np.asarray([velocity * np.sin(radians)])
        v = -np.asarray([velocity * np.cos(radians)])
        print("u is " + str(u))
        print("v is " + str(v))

        shear_m = yng_m / (2. * (1. + poissons_ratio))
        m = rho * 4. / 3. * np.pi * self.radius**3.
        inertia = m * self.diameter**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_aa = get_particle_array_dem(
            x=xaa, y=yaa, u=u, v=v, h=h, m=m, rho=rho, rad_s=rad_s,
            yng_m=yng_m, poissons_ratio=poissons_ratio, shear_m=shear_m,
            dem_id=0, m_inv=m_inv, I_inv=I_inv, name="spheres_aa")

        # wall al alloy
        xw_aa = np.array([3. * self.radius + scale])
        yw_aa = np.array([0.])
        nxw_aa = np.array([0.])
        nyw_aa = np.array([1.])
        rho = 4000.
        yng_m = 7. * 1e10
        poissons_ratio = 0.33
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        wall_aa = get_particle_array_dem(
            x=xw_aa, y=yw_aa, nx=nxw_aa, ny=nyw_aa, nz=0., rho=rho,
            yng_m=yng_m, poissons_ratio=poissons_ratio, shear_m=shear_m,
            dem_id=1, constants={'np': len(xw_ao)}, name="wall_aa")
        wall_ao.set_output_arrays([
            'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au',
            'av', 'aw', 'tag', 'gid', 'nx', 'ny', 'nz'
        ])

        return [spheres_ao, spheres_aa, wall_ao, wall_aa]

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegrator(spheres_ao=RK2StepNonLinearDEM(),
                                    spheres_aa=RK2StepNonLinearDEM())

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
                ResetForces(dest='spheres_aa', sources=None),
                TsuijiNonLinearParticleWallForceStage1(
                    dest='spheres_ao', sources=["wall_ao"], en=self.en, mu=0.),
                TsuijiNonLinearParticleWallForceStage1(
                    dest='spheres_aa', sources=["wall_aa"], en=self.en, mu=0.)
            ]),
        ]

        stage2 = [
            Group(equations=[
                ResetForces(dest='spheres_ao', sources=None),
                ResetForces(dest='spheres_aa', sources=None),
                TsuijiNonLinearParticleWallForceStage2(
                    dest='spheres_ao', sources=["wall_ao"
                                                ], en=self.en, mu=0.0),
                TsuijiNonLinearParticleWallForceStage2(
                    dest='spheres_aa', sources=["wall_aa"], en=self.en, mu=0.0)
            ]),
        ]
        return MultiStageEquations([stage1, stage2])

    def post_process(self):
        # ao data
        # real data
        data = np.loadtxt(
            './chung_test_4_incident_angle_vs_post_collision_angular_velocity.csv',
            delimiter=',')
        incident_angle_ao, ang_vel_ao = data[:, 0], data[:, 1]

        # aa data
        # real data
        data = np.loadtxt(
            './chung_test_4_incident_angle_vs_post_collision_angular_velocity.csv',
            delimiter=',')
        incident_angle_aa, ang_vel_aa = data[:, 0], data[:, 1]

        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        files = self.output_files
        incident_angle, ang_vel_ao_simulated, ang_vel_aa_simulated = [], [], []
        for sd, arrays in iter_output(files):
            ao, aa = arrays['spheres_ao'], arrays['spheres_aa'],
            incident_angle.append()
            ang_vel_ao_simulated.append(-ao.wz[0])
            ang_vel_ao_simulated.append(-aa.wz[0])
        incident_angle = np.asarray(incident_angle)
        ang_vel_ao_simulated = np.asarray(ang_vel_ao)
        ang_vel_aa_simulated = np.asarray(ang_vel_aa)

        import matplotlib.pyplot as plt
        plt.plot(incident_angle, ang_vel_ao_simulated, label='ao')
        plt.plot(incident_angle, ang_vel_aa_simulated, label='aa')
        # plt.scatter(tg_r, fn_g_r, label='al_data')
        # plt.scatter(tl_r, fn_l_r, label='mg_data')
        plt.legend()
        # plt.xlim([0.0, 1000])
        # plt.ylim([0.0, 12])
        import os
        fig = os.path.join(self.output_dir, "incident_angle_vs_ang_vel.png")
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
        eqs1 = [
            Group(equations=[
                UpdateTangentialContacts(dest='spheres_ao',
                                         sources=["spheres_ao"]),
                UpdateTangentialContacts(dest='spheres_aa',
                                         sources=["spheres_aa"])
            ]),
        ]
        arrays = self.particles
        a_eval = self._make_accel_eval(eqs1, arrays)

        # When
        a_eval.evaluate(t, dt)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['spheres_ao']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {radius}
        b.scalar = 'wz'
        b.show_legend = True
        b = particle_arrays['spheres_aa']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {radius}
        b.scalar = 'wz'
        b.show_legend = True
        '''.format(radius=self.radius))


if __name__ == '__main__':
    # incident_angle = [5., 10., 20., 25., 30., 35., 40., 50., 60., 70., 80., 85.]
    incident_angle = [30.]
    for i in incident_angle:
        app = Test4(fname="chung_test_4_" + str(i), theta=i)
        app.run()
        # particles = app.particles[1]
        # print("en is " + str(i))
        # print(particles.u[0] / -3.9)
    # app.post_process()
