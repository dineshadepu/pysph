"""A cube bouncing inside a box. (5 seconds)

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


class Case2a(Application):
    def initialize(self):
        self.rho0 = 10.0
        self.hdx = 1.3
        self.dx = 0.02
        self.dy = 0.02
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.dim = 3

    def create_particles(self):
        nx, ny, nz = 10, 10, 10
        dx = 1.0 / (nx - 1)
        x, y, z = np.mgrid[0:1:nx * 1j, 0:1:ny * 1j, 0:1:nz * 1j]
        x = x.flat
        y = y.flat
        z = (z - 1).flat
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        body = get_particle_array_rigid_body(name='body', x=x, y=y, z=z, h=h,
                                             m=m, rad_s=rad_s)

        body.vc[0] = -5.0
        body.vc[2] = -5.0

        # Create the tank.
        nx, ny, nz = 40, 40, 40
        dx = 1.0 / (nx - 1)
        xmin, xmax, ymin, ymax, zmin, zmax = -2, 2, -2, 2, -2, 2
        x, y, z = np.mgrid[xmin:xmax:nx * 1j, ymin:ymax:ny * 1j, zmin:zmax:nz *
                           1j]
        interior = ((x < 1.8) & (x > -1.8)) & ((y < 1.8) &
                                               (y > -1.8)) & ((z > -1.8) &
                                                              (z <= 2))
        tank = np.logical_not(interior)
        x = x[tank].flat
        y = y[tank].flat
        z = z[tank].flat
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx

        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        tank = get_particle_array_rigid_body(name='tank', x=x, y=y, z=z, h=h,
                                             m=m, rad_s=rad_s)
        tank.total_mass[0] = np.sum(m)

        if self.options.scheme == 'rbrms':
            body = get_particle_array_rigid_body_rotation_matrix(
                name='body', x=body.x, y=body.y, z=body.z, h=body.h, m=body.m,
                rad_s=body.rad_s)
            body.vc[0] = -5.0
            body.vc[2] = -5.0
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        elif self.options.scheme == 'rbqs':
            body = get_particle_array_rigid_body_quaternion(
                name='body', x=body.x, y=body.y, z=body.z, h=body.h, m=body.m,
                rad_s=body.rad_s)
            body.vc[0] = -5.0
            body.vc[2] = -5.0
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')
        return [body, tank]

    def create_scheme(self):
        # rbss = RigidBodySimpleScheme
        rbss = RigidBodySimpleScheme(bodies=['body'], solids=['tank'], dim=3,
                                     rho0=self.rho0, kn=self.kn, mu=self.mu,
                                     en=self.en, gz=-9.81)
        rbrms = RigidBodyRotationMatricesScheme(
            bodies=['body'], solids=['tank'], dim=self.dim, rho0=self.rho0,
            kn=self.kn, mu=self.mu, en=self.en, gz=-9.81)
        rbqs = RigidBodyQuaternionScheme(bodies=['body'], solids=['tank'],
                                         dim=3, rho0=self.rho0, kn=self.kn,
                                         mu=self.mu, en=self.en, gz=-9.81)
        s = SchemeChooser(default='rbss', rbss=rbss, rbrms=rbrms, rbqs=rbqs)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        tf = 3.
        dt = 5e-4
        scheme.configure()
        scheme.configure_solver(kernel=kernel, integrator_cls=EPECIntegrator,
                                dt=dt, tf=tf)

    def customize_output(self):
        #
        self._mayavi_config('''
        viewer.scalar = 'u'
        b = particle_arrays['tank']
        b.plot.actor.mapper.scalar_visibility = False
        b.plot.actor.property.opacity = 0.1
        viewer.scalar = 'u'

        b = particle_arrays['body']
        b.formula = "np.sum(0.5*(u*u+v*v+w*w)+9.81*z)*u/u"
        ''')


if __name__ == '__main__':
    app = Case2a()
    app.run()
