"""A 2d cube bouncing inside a box. (5 seconds)

This is used to test the rigid body equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_rigid_body

from pysph.sph.integrator import EPECIntegrator

from pysph.sph.scheme import SchemeChooser
from pysph.solver.application import Application
from pysph.sph.rigid_body import (
    RigidBodySimpleScheme, RigidBodyRotationMatricesScheme,
    RigidBodyQuaternionScheme, get_particle_array_rigid_body_rotation_matrix,
    get_particle_array_rigid_body_quaternion)
from pysph.examples.solid_mech.impact import add_properties
from pysph.tools.geometry import get_2d_tank


def create_four_bodies():
    nx, ny = 10, 10
    x, y = np.mgrid[0:1:nx * 1j, 0:1:ny * 1j]
    x = x.ravel()
    y = y.ravel()
    bid = np.ones_like(x, dtype=int)
    x4 = np.concatenate((x, x+2, x+4, x+6))
    y4 = np.concatenate((y, y, y, y))
    b4 = np.concatenate((bid*0, bid*1, bid*2, bid*3))
    return x4, y4, b4


class Case1(Application):
    def initialize(self):
        self.rho0 = 10.0
        self.hdx = 1.3
        self.dx = 0.02
        self.dy = 0.02
        self.kn = 1e4
        self.gy = -9.81
        self.kn = 0.
        self.mu = 0.5
        self.en = 1.0
        self.dim = 2

    def create_particles(self):
        nx = 10
        dx = 1.0 / (nx - 1)
        x4, y4, b4 = create_four_bodies()
        m = np.ones_like(x4) * dx * dx * self.rho0
        h = np.ones_like(x4) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x4) * dx
        body = get_particle_array_rigid_body(name='body', x=x4, y=y4, h=h, m=m,
                                             rad_s=rad_s, body_id=b4)

        if self.options.scheme == 'rbrms':
            body = get_particle_array_rigid_body_rotation_matrix(
                name='body', x=body.x, y=body.y, h=body.h, m=body.m,
                rad_s=body.rad_s, body_id=body.body_id)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        elif self.options.scheme == 'rbqs':
            body = get_particle_array_rigid_body_quaternion(
                name='body', x=body.x, y=body.y, h=body.h, m=body.m,
                rad_s=body.rad_s, body_id=body.body_id)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        # setup initial conditions
        # print(body.R)
        body.vc[0] = 0.3
        body.vc[1] = 0.3
        body.omega[2] = 1.0

        body.vc[3] = 0.3
        body.vc[4] = 0.3
        body.omega[5] = 1.0

        body.vc[6] = 0.3
        body.vc[7] = 0.3
        body.omega[8] = 1.0

        body.vc[9] = 0.3
        body.vc[10] = 0.3
        body.omega[11] = 1.0
        return [body]

    def create_scheme(self):
        # rbss = RigidBodySimpleScheme
        rbss = RigidBodySimpleScheme(bodies=['body'], solids=None,
                                     dim=self.dim, rho0=self.rho0, kn=self.kn,
                                     mu=self.mu, en=self.en, gy=self.gy)
        rbrms = RigidBodyRotationMatricesScheme(
            bodies=['body'], solids=None, dim=self.dim, rho0=self.rho0,
            kn=self.kn, mu=self.mu, en=self.en, gy=self.gy)
        rbqs = RigidBodyQuaternionScheme(
            bodies=['body'], solids=['tank'], dim=self.dim, rho0=self.rho0,
            kn=self.kn, mu=self.mu, en=self.en, gy=self.gy)
        s = SchemeChooser(default='rbss', rbss=rbss, rbrms=rbrms, rbqs=rbqs)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        dt = 5e-4
        tf = 2.
        scheme.configure()
        scheme.configure_solver(kernel=kernel, integrator_cls=EPECIntegrator,
                                dt=dt, tf=tf)

    def customize_output(self):
        #
        self._mayavi_config('''
        b = particle_arrays['body']
        b.formula = "np.sum(0.5*(u*u+v*v)+9.81*y)*m"
        ''')

    def post_process(self):
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        t, y = [], []
        files = self.output_files
        for sd, array in iter_output(files, 'body'):
            t.append(sd['t'])
            y.append(array.cm[1])
        import os
        import matplotlib.pyplot as plt

        plt.plot(t, y)
        fig = os.path.join(self.output_dir, 't_vs_comy.png')
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = Case1()
    app.run()