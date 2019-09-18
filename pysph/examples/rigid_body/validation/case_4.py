"""Two cubes having a head on collision (case 1). (10 seconds)

This is used to test the rigid body dynamics equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_rigid_body
from pysph.sph.equation import Group

from pysph.sph.integrator import EPECIntegrator

from pysph.sph.scheme import SchemeChooser
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.rigid_body import (
    RigidBodySimpleScheme, RigidBodyRotationMatricesScheme,
    RigidBodyQuaternionScheme, get_particle_array_rigid_body_rotation_matrix,
    get_particle_array_rigid_body_quaternion)
from pysph.examples.solid_mech.impact import add_properties
from pysph.tools.geometry import get_2d_tank, get_2d_block


def make_cube(lx, ly, lz, dx):
    """Return points x, y, z for a cube centered at origin with given lengths.
    """
    # Convert to floats to be safe with integer division.
    lx, ly, lz = list(map(float, (lx, ly, lz)))
    x, y, z = np.mgrid[-lx / 2:lx / 2 + dx:dx, -ly / 2:ly / 2 + dx:dx, -lz /
                       2:lz / 2 + dx:dx]
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    return x, y, z


class Case1(Application):
    def initialize(self):
        self.rho0 = 10.0
        self.hdx = 1.3
        self.dx = 0.02
        self.dy = 0.02
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.dim = 2

    def create_particles(self):
        # body1 tank
        dx = 1.0 / 9.0
        x, y = get_2d_tank(dx, [0.0, -0.5], 3., 3., 3, top=True)
        m = np.ones_like(x) * dx * dx * self.rho0 / 1000.
        h = np.ones_like(x) * self.hdx * dx
        rad_s = np.ones_like(x) * dx / 2.
        body1 = get_particle_array_rigid_body(name='body1', x=x, y=y, h=h, m=m,
                                              rad_s=rad_s)
        # create body 2
        dx = 1.0 / 9.0
        x, y = get_2d_block(dx, 0.5, 0.5)
        y = y + 1
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        rad_s = np.ones_like(x) * dx / 2.
        body2 = get_particle_array_rigid_body(name='body2', x=x, y=y, h=h, m=m,
                                              rad_s=rad_s)
        if self.options.scheme == 'rbrms':
            body1 = get_particle_array_rigid_body_rotation_matrix(
                name='body1', x=body1.x, y=body1.y, h=body1.h, m=body1.m,
                rad_s=body1.rad_s)
            add_properties(body1, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')
            body2 = get_particle_array_rigid_body_rotation_matrix(
                name='body2', x=body2.x, y=body2.y, h=body2.h, m=body2.m,
                rad_s=body2.rad_s)
            add_properties(body2, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        elif self.options.scheme == 'rbqs':
            body1 = get_particle_array_rigid_body_quaternion(
                name='body1', x=body1.x, y=body1.y, h=body1.h, m=body1.m,
                rad_s=body1.rad_s)

            add_properties(body1, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

            body2 = get_particle_array_rigid_body_quaternion(
                name='body2', x=body2.x, y=body2.y, h=body2.h, m=body2.m,
                rad_s=body2.rad_s)

            add_properties(body2, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        body2.vc[0] = 5.
        body2.vc[1] = 5.
        body2.omega[2] = 10.
        return [body1, body2]

    def create_scheme(self):
        rbss = RigidBodySimpleScheme(bodies=['body1', 'body2'], solids=None,
                                     dim=self.dim, kn=self.kn, mu=self.mu,
                                     en=self.en)
        rbrms = RigidBodyRotationMatricesScheme(bodies=['body1',
                                                        'body2'], solids=None,
                                                dim=self.dim, kn=self.kn,
                                                mu=self.mu, en=self.en)
        rbrms = RigidBodyRotationMatricesScheme(bodies=['body1',
                                                        'body2'], solids=None,
                                                dim=self.dim, kn=self.kn,
                                                mu=self.mu, en=self.en)
        rbqs = RigidBodyQuaternionScheme(bodies=['body1', 'body2'],
                                         solids=None, dim=self.dim, kn=self.kn,
                                         mu=self.mu, en=self.en)
        s = SchemeChooser(default='rbss', rbss=rbss, rbrms=rbrms, rbqs=rbqs)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        tf = 10.
        dt = 1e-4
        scheme.configure()
        scheme.configure_solver(kernel=kernel, integrator_cls=EPECIntegrator,
                                dt=dt, tf=tf)


if __name__ == '__main__':
    app = Case1()
    app.run()
