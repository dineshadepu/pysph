"""
This example is a benchmark of DEM numerical method. This is test five,
where oblique impact of a sphere with a rigid plane with a constant normal
velocity but at different tangential velocities is studied


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
    TsuijiNonLinearParticleWallForceStage1,
    TsuijiNonLinearParticleWallForceStage2, UpdateTangentialContacts)


class Test5(Application):
    def __init__(self, tng_vel):
        self.tng_vel = tng_vel
        self.norm_vel = 5
        super(Test5, self).__init__()

    def initialize(self):
        self.radius = 1e-5
        self.diameter = 2. * self.radius
        self.tf = 0.000012
        self.dt = 1e-6
        self.dim = 3
        # for both particles restitution
        self.en = 1.
        # friction coefficient
        self.mu = 0.3

    def create_particles(self):
        # steel particle positions
        scale = 0.000001
        xs = np.array([0.])
        ys = np.array([self.radius + scale])
        yng_m = 2.08 * 1e11
        poissons_ratio = 0.3
        rho = 7850
        rad_s = self.radius
        u = np.asarray([self.tng_vel])
        v = -np.asarray([self.norm_vel])
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        m = rho * 4. / 3. * np.pi * self.radius**3.
        inertia = m * self.diameter**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_s = get_particle_array_dem(
            x=xs, y=ys, u=u, v=v, h=h, m=m, rho=rho, rad_s=rad_s, yng_m=yng_m,
            poissons_ratio=poissons_ratio, shear_m=shear_m, dem_id=0,
            m_inv=m_inv, I_inv=I_inv, name="spheres_s")

        # wall sphere
        xw_s = np.array([0.])
        yw_s = np.array([0.])
        nxw_s = np.array([0.])
        nyw_s = np.array([1.])
        yng_m = 2.08 * 1e11
        poissons_ratio = 0.3
        rho = 7850
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        wall_s = get_particle_array_dem(
            x=xw_s, y=yw_s, nx=nxw_s, ny=nyw_s, nz=0., rho=rho, yng_m=yng_m,
            poissons_ratio=poissons_ratio, shear_m=shear_m, dem_id=1,
            constants={'np': len(xw_s)}, name="wall_s")
        wall_s.set_output_arrays([
            'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au',
            'av', 'aw', 'tag', 'gid', 'nx', 'ny', 'nz'
        ])

        # polyethylene (pe) particle positions
        scale = 0.000001
        xpe = np.array([0.])
        ype = np.array([self.radius + scale])
        yng_m = 1e9
        poissons_ratio = 0.4
        rho = 1400.
        rad_s = self.radius
        u = np.asarray([self.tng_vel])
        v = -np.asarray([self.norm_vel])
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        m = rho * 4. / 3. * np.pi * self.radius**3.
        inertia = m * self.diameter**2. / 10.
        m_inv = 1. / m
        I_inv = 1. / inertia
        h = 1.2 * rad_s
        spheres_pe = get_particle_array_dem(
            x=xpe, y=ype, u=u, v=v, h=h, m=m, rho=rho, rad_s=rad_s,
            yng_m=yng_m, poissons_ratio=poissons_ratio, shear_m=shear_m,
            dem_id=0, m_inv=m_inv, I_inv=I_inv, name="spheres_pe")

        # wall polyethylene
        xw_pe = np.array([0.])
        yw_pe = np.array([0.])
        nxw_pe = np.array([0.])
        nyw_pe = np.array([1.])
        yng_m = 1e9
        poissons_ratio = 0.4
        rho = 1400.
        shear_m = yng_m / (2. * (1. + poissons_ratio))
        wall_pe = get_particle_array_dem(
            x=xw_pe, y=yw_pe, nx=nxw_pe, ny=nyw_pe, nz=0., rho=rho,
            yng_m=yng_m, poissons_ratio=poissons_ratio, shear_m=shear_m,
            dem_id=1, constants={'np': len(xw_pe)}, name="wall_pe")
        wall_pe.set_output_arrays([
            'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au',
            'av', 'aw', 'tag', 'gid', 'nx', 'ny', 'nz'
        ])

        return [spheres_s, spheres_pe, wall_s, wall_pe]

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegrator(spheres_s=RK2StepNonLinearDEM(),
                                    spheres_pe=RK2StepNonLinearDEM())

        dt = self.dt
        tf = self.tf
        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
                        dt=dt, tf=tf)
        solver.set_disable_output(True)
        return solver

    def create_equations(self):
        stage1 = [
            Group(equations=[
                ResetForces(dest='spheres_s', sources=None),
                ResetForces(dest='spheres_pe', sources=None),
                TsuijiNonLinearParticleWallForceStage1(
                    dest='spheres_s', sources=["wall_s"
                                               ], en=self.en, mu=self.mu),
                TsuijiNonLinearParticleWallForceStage1(
                    dest='spheres_pe', sources=["wall_pe"
                                                ], en=self.en, mu=self.mu)
            ]),
        ]

        stage2 = [
            Group(equations=[
                ResetForces(dest='spheres_s', sources=None),
                ResetForces(dest='spheres_pe', sources=None),
                TsuijiNonLinearParticleWallForceStage2(
                    dest='spheres_s', sources=["wall_s"
                                               ], en=self.en, mu=self.mu),
                TsuijiNonLinearParticleWallForceStage2(
                    dest='spheres_pe', sources=["wall_pe"
                                                ], en=self.en, mu=self.mu)
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
                UpdateTangentialContacts(dest='spheres_s', sources=["wall_s"]),
                UpdateTangentialContacts(dest='spheres_pe',
                                         sources=["wall_pe"])
            ]),
        ]
        arrays = self.particles
        a_eval = self._make_accel_eval(eqs1, arrays)

        # When
        a_eval.evaluate(t, dt)

    def post_process(self, sim_data):
        import matplotlib.pyplot as plt
        ###########################
        # recoil angle comparison #
        ###########################
        # sphere data
        # real data
        data = np.loadtxt(
            './chung_test_5_normalized_recoil_angle_vs_normalized_incident_angle_steel.csv',
            delimiter=',')
        n_incident_angle_scraped_s, n_recoil_angle_scraped_s = (data[:, 0],
                                                                data[:, 1])
        # polyethylene (pe) data
        # real data
        data = np.loadtxt(
            './chung_test_5_normalized_recoil_angle_vs_normalized_incident_angle_polyethylene.csv',
            delimiter=',')
        n_incident_angle_scraped_pe, n_recoil_angle_scraped_pe = (data[:, 0],
                                                                  data[:, 1])

        # simulated data
        n_incident_angle_sim_s, n_recoil_angle_sim_s = sim_data[
            'n_incident_angle_sim_s'], sim_data['n_recoil_angle_sim_s']
        n_incident_angle_sim_pe, n_recoil_angle_sim_pe = sim_data[
            'n_incident_angle_sim_pe'], sim_data['n_recoil_angle_sim_pe']

        plt.plot(n_incident_angle_scraped_s, n_recoil_angle_scraped_s,
                 label='Chung Data of steel')
        plt.plot(n_incident_angle_scraped_pe, n_recoil_angle_scraped_pe,
                 label='Chung Data of polyethylene')

        plt.plot(n_incident_angle_sim_s, n_recoil_angle_sim_s,
                 label='DEM simulation of steel')
        plt.plot(n_incident_angle_sim_pe, n_recoil_angle_sim_pe,
                 label='DEM simulation of polyethylene')

        plt.legend()
        plt.xlabel(r'Incident angle Vt/\mu Vn')
        plt.ylabel(r'recoil angle V\'st/\mu V\'cn ')
        plt.xlim([0., 14.])
        plt.ylim([-4., 6.])
        import os
        fig = os.path.join("test_5_n_incident_angle_vs_n_recoil_angle.png")
        plt.savefig(fig, dpi=300)
        # plt.show()
        #####################
        # recoil angle ends #
        #####################

        ###################################
        # post collision angular velocity #
        ###################################
        # sphere data
        # real data
        data = np.loadtxt(
            './chung_test_5_normalized_post_collision_ang_vel_vs_normalized_incident_angle_steel.csv',
            delimiter=',')
        n_incident_angle_scraped_s, n_post_collision_ang_vel_scraped_s = (
            data[:, 0], data[:, 1])
        # polyethylene (pe) data
        # real data
        data = np.loadtxt(
            './chung_test_5_normalized_post_collision_ang_vel_vs_normalized_incident_angle_polyethylene.csv',
            delimiter=',')
        n_incident_angle_scraped_pe, n_post_collision_ang_vel_scraped_pe = (
            data[:, 0], data[:, 1])

        # simulated data
        n_incident_angle_sim_s, n_post_collision_ang_vel_sim_s = sim_data[
            'n_incident_angle_sim_s'], sim_data[
                'n_post_collision_ang_vel_sim_s']
        n_incident_angle_sim_pe, n_post_collision_ang_vel_sim_pe = sim_data[
            'n_incident_angle_sim_pe'], sim_data[
                'n_post_collision_ang_vel_sim_pe']

        plt.plot(n_incident_angle_scraped_s,
                 n_post_collision_ang_vel_scraped_s,
                 label='Chung Data of steel')
        plt.plot(n_incident_angle_scraped_pe,
                 n_post_collision_ang_vel_scraped_pe,
                 label='Chung Data of polyethylene')

        plt.plot(n_incident_angle_sim_s, n_post_collision_ang_vel_sim_s,
                 label='DEM simulation of steel')
        plt.plot(n_incident_angle_sim_pe, n_post_collision_ang_vel_sim_pe,
                 label='DEM simulation of polyethylene')

        plt.legend()
        plt.xlabel(r'Incident angle Vt/\mu Vn')
        plt.ylabel(r'ang vel r \omega_1\' \'st/\mu Vn')
        plt.xlim([0, 20])
        plt.ylim([-6, 0])
        import os
        fig = os.path.join("test_5_n_incident_angle_vs_n_ang_vel.png")
        plt.savefig(fig, dpi=300)
        # plt.show()
        ###################################
        # post collision angular velocity #
        ###################################

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['spheres_s']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {radius}
        b.scalar = 'v'
        b = particle_arrays['spheres_pe']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {radius}
        b.scalar = 'v'
        '''.format(radius=self.radius))


if __name__ == '__main__':
    tng_vel = [0.1, 0.3, 0.5, 1.0, 5., 10, 15., 20., 30., 40., 50., 70.]

    # for steel (s)
    # n is for normalized
    n_incident_angle_s = []
    n_recoil_angle_s = []
    n_post_collision_ang_vel_s = []

    # for polyethylene (pe)
    n_incident_angle_pe = []
    n_recoil_angle_pe = []
    n_post_collision_ang_vel_pe = []

    for i in tng_vel:
        app = Test5(theta=i)
        import tempfile
        tmpdir = tempfile.mkdtemp()
        app.run(['--disable-output', '-d', tmpdir])
        import shutil
        shutil.rmtree(tmpdir)

        particles = app.particles
        n_incident_angle_s.append(i / (app.mu * app.norm_vel))
        n_recoil_angle_s.append(
            particles[0].u[0] / (app.mu * particles[0].v[0]))
        n_post_collision_ang_vel_s.append(
            particles[0].rad_s[0] * particles[0].wz[0] /
            (app.mu * app.norm_vel))

        n_incident_angle_pe.append(i / (app.mu * app.norm_vel))
        n_recoil_angle_pe.append(
            particles[1].u[0] / (app.mu * particles[1].v[0]))
        n_post_collision_ang_vel_pe.append(
            particles[1].rad_s[0] * particles[1].wz[0] /
            (app.mu * app.norm_vel))

    # save the simulation data for the post processing
    data = {}
    data['n_incident_angle_s'] = n_incident_angle_s
    data['n_recoil_angle_s'] = n_recoil_angle_s
    data['n_post_collision_ang_vel_s'] = n_post_collision_ang_vel_s

    data['n_incident_angle_pe'] = n_incident_angle_pe
    data['n_recoil_angle_pe'] = n_recoil_angle_pe
    data['n_post_collision_ang_vel_pe'] = n_post_collision_ang_vel_pe
    app.post_process(data)
