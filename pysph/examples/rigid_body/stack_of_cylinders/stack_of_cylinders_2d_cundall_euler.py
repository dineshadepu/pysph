"""
Simulation of solid-fluid mixture flow using moving particle methods
Shuai Zhang

link: https://www.sciencedirect.com/science/article/pii/S0021999108006499

In case 1 we deal the collision forces by Potyondy model.

Time: 7 minutes
"""
from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.tools.sph_evaluator import SPHEvaluator

from pysph.sph.equation import Group, MultiStageEquations
from pysph.solver.application import Application
from pysph.dem.discontinuous_dem.dem_nonlinear import EuleIntegratorMultiStage
from pysph.sph.rigid_body import (RigidBodyMoments, RigidBodyMotion,
                                  RK2StepRigidBodyDEM, BodyForce)
from pysph.sph.rigid_body_cundall_2d import (
    get_particle_array_rigid_body_cundall, RigidBodyCollision2DCundallParticleParticleEuler,
    RigidBodyCollision2DCundallParticleParticleStage1, RigidBodyCollision2DCundallParticleParticleStage2,
    UpdateTangentialContactsCundall2dPaticleParticle)
from pysph.tools.geometry import (get_2d_tank)


def create_circle(diameter=1, spacing=0.05, center=None):
    dx = spacing
    x = [0.0]
    y = [0.0]
    r = spacing
    nt = 0
    while r < diameter / 2:
        nnew = int(np.pi * r**2 / dx**2 + 0.5)
        tomake = nnew - nt
        theta = np.linspace(0., 2. * np.pi, tomake + 1)
        for t in theta[:-1]:
            x.append(r * np.cos(t))
            y.append(r * np.sin(t))
        nt = nnew
        r = r + dx
    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


class ZhangStackOfCylinders(Application):
    def initialize(self):
        self.dam_length = 26 * 1e-2
        self.dam_height = 26 * 1e-2
        self.dam_spacing = 1e-3
        self.dam_layers = 2
        self.dam_rho = 2000.

        self.cylinder_radius = 1. / 2. * 1e-2
        self.cylinder_diameter = 1. * 1e-2
        self.cylinder_spacing = 1e-3
        self.cylinder_rho = 2.7 * 1e3

        self.wall_height = 20 * 1e-2
        self.wall_spacing = 1e-3
        self.wall_layers = 2
        # self.wall_time = 0.01
        self.wall_time = 0.2
        self.wall_rho = 2000.

        # simulation properties
        self.hdx = 1.2
        self.alpha = 0.1

        # solver data
        self.tf = 0.5 + self.wall_time
        self.dt = 5e-5
        self.dim = 2
        self.seval = None

    def create_dam(self):
        xt, yt = get_2d_tank(self.dam_spacing,
                             np.array([self.dam_length / 2., 0.]),
                             length=self.dam_length, height=self.dam_height,
                             num_layers=self.dam_layers, outside=True)
        return xt, yt

    def create_wall(self):
        x = np.arange(0.054 + 2. * self.wall_spacing,
                      0.056 + 2 * self.wall_spacing, self.wall_spacing)
        y = np.arange(0., self.wall_height, self.wall_spacing)
        xw, yw = np.meshgrid(x, y)
        return xw.ravel(), yw.ravel()

    def create_cylinders_stack(self):
        # create a row of six cylinders
        x_six = np.array([])
        y_six = np.array([])
        x_tmp1, y_tmp1 = create_circle(
            self.cylinder_diameter, self.cylinder_spacing, [
                self.cylinder_radius,
                self.cylinder_radius + self.cylinder_spacing
            ])
        for i in range(6):
            x_tmp = x_tmp1 + i * (
                self.cylinder_diameter - self.cylinder_spacing / 2.)
            x_six = np.concatenate((x_six, x_tmp))
            y_six = np.concatenate((y_six, y_tmp1))

        # create three layers of six cylinder rows
        y_six_three = np.array([])
        x_six_three = np.array([])
        for i in range(3):
            x_six_three = np.concatenate((x_six_three, x_six))
            y_six_1 = y_six + 1.6 * i * self.cylinder_diameter
            y_six_three = np.concatenate((y_six_three, y_six_1))

        # create a row of five cylinders
        x_five = np.array([])
        y_five = np.array([])
        x_tmp1, y_tmp1 = create_circle(
            self.cylinder_diameter, self.cylinder_spacing, [
                2. * self.cylinder_radius, self.cylinder_radius +
                self.cylinder_spacing + self.cylinder_spacing / 2.
            ])

        for i in range(5):
            x_tmp = x_tmp1 + i * (
                self.cylinder_diameter - self.cylinder_spacing / 2.)
            x_five = np.concatenate((x_five, x_tmp))
            y_five = np.concatenate((y_five, y_tmp1))

        y_five = y_five + 0.75 * self.cylinder_diameter
        x_five = x_five

        # create three layers of five cylinder rows
        y_five_three = np.array([])
        x_five_three = np.array([])
        for i in range(3):
            x_five_three = np.concatenate((x_five_three, x_five))
            y_five_1 = y_five + 1.6 * i * self.cylinder_diameter
            y_five_three = np.concatenate((y_five_three, y_five_1))

        x = np.concatenate((x_six_three, x_five_three))
        y = np.concatenate((y_six_three, y_five_three))

        # create body_id
        no_particles_one_cylinder = len(x_tmp)
        total_bodies = 3 * 5 + 3 * 6

        body_id = np.array([], dtype=int)
        for i in range(total_bodies):
            b_id = np.ones(no_particles_one_cylinder, dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, body_id

    def geometry(self):
        from matplotlib import pyplot as plt

        # please run this function to know how
        # geometry looks like
        xc, yc, body_id = self.create_cylinders_stack()
        xt, yt = self.create_dam()
        xw, yw = self.create_wall()

        plt.scatter(xc, yc)
        plt.scatter(xt, yt)
        plt.scatter(xw, yw)
        plt.axes().set_aspect('equal', 'datalim')
        print("done")
        plt.show()

    def create_particles(self):
        # get bodyid for each cylinder
        xc, yc, body_id = self.create_cylinders_stack()
        m = self.cylinder_rho * self.cylinder_spacing**2
        h = self.hdx * self.cylinder_radius
        rad_s = self.cylinder_spacing / 2.
        V = self.cylinder_spacing**2
        cylinders = get_particle_array_rigid_body_cundall(
            x=xc, y=yc, h=h, m=m, rho=self.cylinder_rho, rad_s=rad_s, V=V,
            body_id=body_id, dem_id=body_id, name="cylinders")

        xd, yd = self.create_dam()
        dam = get_particle_array_rigid_body_cundall(
            x=xd, y=yd, h=h, m=m, rho=self.dam_rho,
            rad_s=self.dam_spacing / 2., V=self.dam_spacing**2, name="dam",
            dem_id=max(body_id) + 1)

        xw, yw = self.create_wall()
        wall = get_particle_array_rigid_body_cundall(
            x=xw, y=yw, h=h, m=m, rho=self.wall_rho,
            rad_s=self.wall_spacing / 2., V=self.wall_spacing**2, name="wall",
            dem_id=max(body_id) + 2)
        wall.x += 0.0015

        # please run this function to know how
        # geometry looks like
        # from matplotlib import pyplot as plt
        # plt.scatter(cylinders.x, cylinders.y)
        # plt.scatter(dam.x, dam.y)
        # plt.scatter(wall.x, wall.y)
        # plt.axes().set_aspect('equal', 'datalim')
        # print("done")
        # plt.show()
        return [cylinders, dam, wall]

    def create_equations(self):
        stage1 = [
            Group(
                equations=[
                    BodyForce(dest='cylinders', sources=None, gy=-9.81),
                ], real=False),
            Group(equations=[
                RigidBodyCollision2DCundallParticleParticleEuler(
                    dest='cylinders', sources=['dam', 'wall', 'cylinders'],
                    kn=1e7, en=0.5, mu=0.3),
            ]),
            Group(
                equations=[RigidBodyMoments(dest='cylinders', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='cylinders', sources=None)]),
        ]
        return stage1

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EuleIntegratorMultiStage(cylinders=RK2StepRigidBodyDEM())

        dt = self.dt
        print("DT: %s" % dt)
        tf = self.tf
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf)

        return solver

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
        T = self.wall_time
        if (T - dt / 2) < t < (T + dt / 2):
            for pa in self.particles:
                if pa.name == 'wall':
                    pa.y += 14 * 1e-2

        eqs1 = [
            Group(equations=[
                UpdateTangentialContactsCundall2dPaticleParticle(dest='cylinders',
                                                  sources=['cylinders', 'dam', 'wall']),
            ])
        ]
        arrays = self.particles
        a_eval = self._make_accel_eval(eqs1, arrays)

        # When
        a_eval.evaluate(t, dt)

    def post_process(self):
        """This function will run once per time step after the time step is
        executed. For some time (self.wall_time), we will keep the wall near
        the cylinders such that they settle down to equilibrium and replicate
        the experiment.

        By running the example it becomes much clear.
        """
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        files = self.output_files
        print(len(files))
        t = []
        system_x = []
        system_y = []
        for sd, array in iter_output(files, 'cylinders'):
            _t = sd['t']
            if _t > self.wall_time:
                t.append(_t)
                # get the system center
                cm_x = 0
                cm_y = 0
                for i in range(array.n_body[0]):
                    cm_x += array.xcm[3 * i] * array.total_mass[i]
                    cm_y += array.xcm[3 * i + 1] * array.total_mass[i]
                cm_x = cm_x / np.sum(array.total_mass)
                cm_y = cm_y / np.sum(array.total_mass)

                system_x.append(cm_x / self.dam_length)
                system_y.append(cm_y / self.dam_length)

        import matplotlib.pyplot as plt
        t = np.asarray(t)
        t = t - np.min(t)

        plt.plot(t, system_x, label='system com x')
        plt.plot(t, system_y, label='system com y')
        plt.legend()
        plt.show()

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['cylinders']
        b.plot.actor.property.point_size = 2.
        ''')


if __name__ == '__main__':
    app = ZhangStackOfCylinders()
    app.run()
