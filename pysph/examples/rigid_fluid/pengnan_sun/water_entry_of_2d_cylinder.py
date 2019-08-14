"""Water entry of 2-D cylinder.

This is taken from the "Numerical simulation of interactions between free
surface and rigid body using a robust SPH method", from section =3.1.3=.

"""
from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.utils import (get_particle_array_wcsph,
                              get_particle_array_rigid_body)
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import (XSPHCorrection, SummationDensity)
from pysph.sph.wc.basic import TaitEOSHGCorrection, MomentumEquation
from pysph.solver.application import Application
from pysph.sph.rigid_body import (
    BodyForce, SummationDensityBoundary, RigidBodyCollision, RigidBodyMoments,
    RigidBodyMotion, AkinciRigidFluidCoupling, RK2StepRigidBody)
from pysph.tools.geometry_rigid_fluid import get_2d_hydrostatic_tank


def create_sphere(dx=1):
    x = np.arange(0, 100, dx)
    y = np.arange(151, 251, dx)
    x, y = np.meshgrid(x, y)
    x = x.ravel()
    y = y.ravel()

    p = ((x - 50)**2 + (y - 200)**2) < 20**2
    x = x[p]
    y = y[p]

    # lower sphere a little
    y = y - 20
    return x * 1e-3, y * 1e-3


def geometry():
    import matplotlib.pyplot as plt
    # please run this function to know how
    # geometry looks like
    xt, yt, xf, yf = get_2d_hydrostatic_tank()
    x_cube, y_cube = create_sphere()
    plt.scatter(xf, yf)
    plt.scatter(xf, yt)
    plt.scatter(x_cube, y_cube)
    plt.axes().set_aspect('equal', 'datalim')
    print("done")
    plt.show()


class WaterEntry2dCylinder(Application):
    def initialize(self):
        self.dx = 2 * 1e-3
        self.hdx = 1.2
        self.ro = 1000
        self.solid_rho = 500
        self.m = 1000 * self.dx * self.dx
        self.co = 2 * np.sqrt(2 * 9.81 * 150 * 1e-3)
        self.alpha = 0.1

    def create_particles(self):
        xt, yt, xf, yf = get_2d_hydrostatic_tank()

        # create fluid
        m = self.ro * self.dx * self.dx
        rho = self.ro
        h = self.hdx * self.dx
        fluid = get_particle_array_wcsph(x=xf, y=yf, h=h, m=m, rho=rho,
                                         name="fluid")

        m = 1000 * self.dx * self.dx
        rho = 1000
        rad_s = 2 / 2. * 1e-3
        h = self.hdx * self.dx
        V = self.dx**2.
        tank = get_particle_array_wcsph(x=xt, y=yt, h=h, m=m, rho=rho,
                                        rad_s=rad_s, V=V, name="tank")
        for name in ['fx', 'fy', 'fz']:
            tank.add_property(name)

        dx = 1
        xc, yc = create_sphere(1)
        m = self.solid_rho * dx * 1e-3 * dx * 1e-3
        rho = self.solid_rho
        h = self.hdx * self.dx
        rad_s = dx / 2. * 1e-3
        V = dx**2.
        cs = 0.0
        cube = get_particle_array_rigid_body(x=xc, y=yc, h=h, m=m, rho=rho,
                                             rad_s=rad_s, V=V, cs=cs,
                                             name="cube")

        return [fluid, tank, cube]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(fluid=WCSPHStep(), tank=WCSPHStep(),
                                    cube=RK2StepRigidBody())

        dt = 0.125 * self.dx * self.hdx / (self.co * 1.1) / 2.
        # dt = 1e-4
        print("DT: %s" % dt)
        tf = 0.5
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='cube', sources=None, gy=-9.81),
            ], real=False),
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid']),
                SummationDensityBoundary(
                    dest='fluid', sources=['tank', 'cube'], fluid_rho=1000.0)
            ]),

            # Tait equation of state
            Group(
                equations=[
                    TaitEOSHGCorrection(dest='fluid', sources=None,
                                        rho0=self.ro, c0=self.co, gamma=7.0),
                ], real=False),
            Group(equations=[
                MomentumEquation(dest='fluid', sources=['fluid'], alpha=self.
                                 alpha, beta=0.0, c0=self.co, gy=-9.81),
                AkinciRigidFluidCoupling(dest='fluid',
                                         sources=['cube', 'tank']),
                XSPHCorrection(dest='fluid', sources=['fluid', 'tank']),
            ]),
            Group(equations=[
                RigidBodyCollision(dest='cube', sources=['tank'], kn=1e5)
            ]),
            Group(equations=[RigidBodyMoments(dest='cube', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='cube', sources=None)])
        ]
        return equations


if __name__ == '__main__':
    app = WaterEntry2dCylinder()
    app.run()
