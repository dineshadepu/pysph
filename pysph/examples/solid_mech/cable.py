"""
Numerical example of wave propagation, section 8.1 from
https://www.sciencedirect.com/science/article/pii/S0045782516304182#br000170
"""
import numpy as np
from numpy import cos, sin, cosh, sinh

# SPH equations
from pysph.sph.solid_mech.basic import (ElasticSolidsScheme,
                                        get_particle_array_elastic_dynamics)

from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application


class Cable(Application):
    def initialize(self):
        # the dimension of the cable is 10m X 0.2m X 0.2m
        # x axis length
        self.L = 10
        # y axis length
        self.B = 0.2
        # z axis length
        self.H = 0.2

        # wave number K
        self.KL = 1.875
        self.K = 1.875 / self.L

        # particle spacing
        self.dx = 0.1
        self.h = 1.3 * self.dx
        self.rho0 = 8000.
        self.E = 2. * 1e11
        self.nu = 0.0

        self.inside_wall_length = self.L / 4.
        self.wall_layers = 3

        self.alpha = 1.0
        self.beta = 1.0
        self.xsph_eps = 0.5
        self.artificial_stress_eps = 0.3

        self.tf = 1.0
        self.dt = 1e-5
        self.dim = 3

    def create_cable(self):
        dx = self.dx
        xp, yp, zp = np.mgrid[0.:self.L + dx / 2.:dx, 0:self.B +
                              dx / 2.:dx, 0:self.H + dx / 2.:dx]
        xp = xp.ravel()
        yp = yp.ravel()
        zp = zp.ravel()
        return xp, yp, zp

    def create_particles(self):
        xp, yp, zp = self.create_cable()
        m = self.rho0 * self.dx**3.

        kernel = CubicSpline(dim=self.dim)
        self.wdeltap = kernel.kernel(rij=self.dx, h=self.h)
        cable = get_particle_array_elastic_dynamics(
            x=xp, y=yp, m=m, h=self.h,
            rho=self.rho0, name="cable", constants=dict(
                wdeltap=self.wdeltap, n=4, rho_ref=self.rho0,
                E=self.E, nu=self.nu))

        return [cable]

    def create_equations(self):
        pass


if __name__ == '__main__':
    app = Cable()
    app.run()
    app.post_process()
