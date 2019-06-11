"""A cube translating and rotating freely without the influence of gravity.

This is used to test the rigid body dynamics equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_rigid_body
from pysph.sph.equation import Group

from pysph.sph.integrator import EPECIntegrator

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision,
                                  RigidBodyMoments, RigidBodyMotion,
                                  RK2StepRigidBody)

dim = 3

dt = 5e-3
tf = 5.0
gz = -9.81

hdx = 1.0
dx = dy = 0.02
rho0 = 10.0


class Case0(Application):
    def create_particles(self):
        nx, ny, nz = 10, 10, 10
        dx = 1.0 / (nx - 1)
        x, y, z = np.mgrid[0:1:nx * 1j, 0:1:ny * 1j, 0:1:nz * 1j]
        x = x.flat
        y = y.flat
        z = (z - 1).flat
        m = np.ones_like(x) * dx * dx * rho0
        h = np.ones_like(x) * hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        body = get_particle_array_rigid_body(name='body', x=x, y=y, z=z, h=h,
                                             m=m, rad_s=rad_s)

        print("here")
        body.vc[0] = 0.1
        body.vc[1] = 0.1
        body.omega[2] = 1.
        return [body]

    def create_solver(self):
        kernel = CubicSpline(dim=dim)

        integrator = EPECIntegrator(body=RK2StepRigidBody())

        solver = Solver(kernel=kernel, dim=dim, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)
        solver.set_print_freq(10)
        return solver

    def create_equations(self):
        equations = [
            Group(equations=[RigidBodyMoments(dest='body', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='body', sources=None)]),
        ]
        return equations


if __name__ == '__main__':
    app = Case0()
    app.run()
