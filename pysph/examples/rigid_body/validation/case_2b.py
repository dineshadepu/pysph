"""A sphere bouncing inside a box. (5 seconds)

This is used to test the rigid body equations.
"""
import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_rigid_body

from pysph.sph.integrator import EPECIntegrator

from pysph.sph.scheme import SchemeChooser
from pysph.solver.application import Application
from pysph.tools.geometry import get_3d_sphere
from pysph.sph.rigid_body import (
    RigidBodySimpleScheme, RigidBodyRotationMatricesScheme,
    get_particle_array_rigid_body_rotation_matrix)
from pysph.examples.solid_mech.impact import add_properties


class Case2b(Application):
    def initialize(self):
        self.rho0 = 10.0
        self.hdx = 1.3
        self.dx = 0.02
        self.dy = 0.02
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.dim = 3
        self.radius = 0.5

    def create_particles(self):
        nx, ny, nz = 10, 10, 10
        dx = 1.0 / (nx - 1)
        x, y, z = get_3d_sphere(dx, self.radius,
                                center=np.array([0.5, 0.5, 0.5]))
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
        return [body, tank]

    def create_scheme(self):
        # rbss = RigidBodySimpleScheme
        rbss = RigidBodySimpleScheme(bodies=['body'], solids=['tank'], dim=3,
                                     rho0=self.rho0, kn=self.kn, mu=self.mu,
                                     en=self.en, gz=-9.81)
        rbrms = RigidBodyRotationMatricesScheme(
            bodies=['body'], solids=['tank'], dim=self.dim, rho0=self.rho0,
            kn=self.kn, mu=self.mu, en=self.en, gz=-9.81)
        s = SchemeChooser(default='rbss', rbss=rbss, rbrms=rbrms)
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
        self._mayavi_config('''
        viewer.scalar = 'u'
        b = particle_arrays['tank']
        b.plot.actor.mapper.scalar_visibility = False
        b.plot.actor.property.opacity = 0.1
        ''')

    def post_process(self):
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output

        files = self.output_files
        t = []
        tot_ene_b1, ang_mom_b1, lin_mom_b1 = [], [], []
        tot_ene_b2, ang_mom_b2, lin_mom_b2 = [], [], []
        for sd, array in iter_output(files, 'body1', 'body2'):
            _t = sd['t']
            t.append(_t)

        import matplotlib
        import os
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt
        plt.clf()
        plt.plot(t, amplitude)
        plt.xlabel('t')
        plt.ylabel('Amplitude')
        plt.legend()
        fig = os.path.join(self.output_dir, "amplitude.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = Case2b()
    app.run()
