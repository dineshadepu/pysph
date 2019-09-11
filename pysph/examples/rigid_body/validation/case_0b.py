"""A cube translating and rotating freely without the influence of gravity.

This is used to test the rigid body dynamics equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_rigid_body
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser
from pysph.sph.rigid_body import (
    RigidBodySimpleScheme, RigidBodyRotationMatricesScheme,
    RigidBodyQuaternionScheme,
    get_particle_array_rigid_body_rotation_matrix,
    get_particle_array_rigid_body_quaternion)
from pysph.examples.solid_mech.impact import add_properties


class Case0(Application):
    def initialize(self):
        self.rho0 = 10.0
        self.hdx = 1.3
        self.dx = 0.02
        self.dy = 0.02
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.dim = 2

    def create_scheme(self):
        rbss = RigidBodySimpleScheme(bodies=['body'], solids=None, dim=self.dim,
                                     rho0=self.rho0, kn=self.kn, mu=self.mu,
                                     en=self.en)
        rbrms = RigidBodyRotationMatricesScheme(
            bodies=['body'], solids=None, dim=self.dim, rho0=self.rho0, kn=self.kn,
            mu=self.mu, en=self.en)
        rbqs = RigidBodyQuaternionScheme(
            bodies=['body'], solids=None, dim=self.dim, rho0=self.rho0, kn=self.kn,
            mu=self.mu, en=self.en)
        s = SchemeChooser(default='rbss', rbss=rbss, rbrms=rbrms, rbqs=rbqs)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        dt = 1e-3
        tf = 1.
        scheme.configure()
        scheme.configure_solver(kernel=kernel, integrator_cls=EPECIntegrator,
                                dt=dt, tf=tf)

    def create_particles(self):
        nx, ny, nz = 10, 10, 10
        dx = self.dx
        x, y = np.mgrid[0:1:nx * 1j, 0:1:ny * 1j]
        x = x.flat
        y = y.flat
        fltr = ((x > 0.7) & (x < 0.9) & (y < 0.4))
        x = x[~fltr]
        y = y[~fltr]
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        body = get_particle_array_rigid_body(name='body', x=x, y=y, h=h,
                                             m=m, rad_s=rad_s)

        if self.options.scheme == 'rbrms':
            body = get_particle_array_rigid_body_rotation_matrix(
                name='body', x=x, y=y, h=h, m=m, rad_s=rad_s)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        elif self.options.scheme == 'rbqs':
            body = get_particle_array_rigid_body_quaternion(
                name='body', x=x, y=y, h=h, m=m, rad_s=rad_s)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        body.vc[0] = 0.5
        body.vc[1] = 0.5
        body.omega[2] = 1.
        print(body.mib.reshape(3, 3))
        print(np.cross(
            body.omega, np.matmul(body.mib.reshape(3, 3), body.omega)))
        return [body]


if __name__ == '__main__':
    app = Case0()
    app.run()
