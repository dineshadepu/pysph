import numpy as np
from math import cos, sin, cosh, sinh

# SPH equations
from pysph.sph.equation import Group
from pysph.sph.basic_equations import IsothermalEOS

from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.base.kernels import (CubicSpline, WendlandQuintic)
# from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import SolidMechStep

from oscillating_plate_gray import (OscillatingPlate)

from gtvf import (GTVFSolidRK2Step, add_gtvf_solid_properties, CorrectDensity,
                  DensityEvolutionUhat, DensityEvolutionU,
                  MomentumEquationSolidGTVF, VelocityGradientHat,
                  VelocityGradient, DeviatoricStressRate, StateEquationGTVF,
                  remove_gray_solid_properties, get_particle_array_gtvf,
                  GTVFEPECIntegrator)


class Rings(Application):
    def initialize(self):
        # constants
        self.E = 1e7
        self.nu = 0.3975
        self.rho0 = 1.

        self.dx = 0.001
        self.hdx = 1.5
        self.h = self.hdx * self.dx

        # geometry
        self.ri = 0.03
        self.ro = 0.04

        self.spacing = 0.041

        self.dt = 1e-8
        self.tf = 6.0000000000007015e-05

    def create_particles(self):
        spacing = self.spacing  # spacing = 2*5cm

        x, y = np.mgrid[-self.ro:self.ro:self.dx, -self.ro:self.ro:self.dx]
        x = x.ravel()
        y = y.ravel()

        d = (x * x + y * y)
        ro = self.ro
        ri = self.ri
        keep = np.flatnonzero((ri * ri <= d) * (d < ro * ro))
        x = x[keep]
        y = y[keep]

        x = np.concatenate([x - spacing, x + spacing])
        y = np.concatenate([y, y])

        dx = self.dx
        hdx = self.hdx
        m = self.rho0 * dx * dx
        h = np.ones_like(x) * hdx * dx
        rho = self.rho0

        solid = get_particle_array_gtvf(name="solid", x=x + spacing, y=y, m=m,
                                        rho=rho, h=h, constants=dict(
                                            rho_ref=self.rho0, E=self.E,
                                            nu=self.nu))
        print(solid.b_mod)

        print('Ellastic Collision with %d particles' % (x.size))
        print(
            "Shear modulus G = %g, Young's modulus = %g, Poisson's ratio =%g" %
            (solid.G, solid.E, solid.nu))

        solid.add_property('cs')
        solid.cs[:] = solid.c0[0]

        # u_f = 0.12
        u_f = 0.059
        solid.u = solid.cs * u_f * (2 * (x < 0) - 1)

        return [solid]

    def create_solver(self):
        # kernel = CubicSpline(dim=2)
        kernel = WendlandQuintic(dim=2)

        integrator = GTVFEPECIntegrator(solid=GTVFSolidRK2Step())

        # time step
        print("time step has to be")
        dt_stable = 0.25 * (self.h / (np.sqrt(self.E / self.rho0) + 14))
        print(dt_stable)
        dt = self.dt
        tf = self.tf

        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False, pfreq=500)
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    # correct the density
                    CorrectDensity(dest='solid', sources=['solid']),
                ], ),
            Group(equations=[
                StateEquationGTVF(dest='solid', sources=None),
            ], ),
            Group(
                equations=[
                    # p
                    # arho
                    DensityEvolutionUhat(dest='solid', sources=['solid']),

                    # au, av, aw, auhat, avhat, awhat
                    MomentumEquationSolidGTVF(dest='solid', sources=['solid']),

                    # ads,
                    VelocityGradientHat(dest='solid', sources=['solid']),
                    DeviatoricStressRate(dest='solid', sources=['solid']),
                ], ),
        ]
        return equations

    def _get_sph_evaluator(self, arrays):
        if not hasattr(self, '_sph_eval'):
            from pysph.tools.sph_evaluator import SPHEvaluator
            equations = [CorrectDensity(dest='solid', sources=['solid'])]
            sph_eval = SPHEvaluator(arrays=arrays, equations=equations, dim=2,
                                    kernel=WendlandQuintic(dim=2))
            self._sph_eval = sph_eval
        return self._sph_eval

    def post_step(self, solver):
        # pass
        self._get_sph_evaluator(solver.particles).evaluate()


if __name__ == '__main__':
    app = Rings()
    app.run()
    app = Rings()
