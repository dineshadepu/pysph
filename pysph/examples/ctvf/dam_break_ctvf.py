"""Dam break 2d with CTVF
"""

import numpy as np

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application

from pysph.sph.scheme import TVFScheme, SchemeChooser
from pysph.sph.wc.edac import EDACScheme

from pysph.base.kernels import QuinticSpline

# PySPH imports
from pysph.solver.application import Application
from pysph.solver.solver import Solver

from pysph.sph.equation import Group
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.utils import (get_particle_array_tvf_fluid,
                              get_particle_array_tvf_solid)
from pysph.sph.ctvf import (SummationDensityTmp, GradientRhoTmp,
                            PsuedoForceOnFreeSurface, add_ctvf_properties,
                            MinNeighbourRho)

from pysph.base.kernels import QuinticSpline
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import PECIntegrator

# imports for TVF
from pysph.sph.wc.transport_velocity import (
    SummationDensity, StateEquation, MomentumEquationPressureGradient,
    MomentumEquationArtificialViscosity, MomentumEquationViscosity,
    MomentumEquationArtificialStress, SolidWallPressureBC, SolidWallNoSlipBC,
    SetWallVelocity)

# imports for CTVF
from pysph.sph.ctvf import (SummationDensityTmp, GradientRhoTmp,
                            PsuedoForceOnFreeSurface, add_ctvf_properties,
                            MinNeighbourRho)

from pysph.tools.geometry import get_2d_block
from pysph.tools.geometry import get_2d_tank, get_2d_block


class DamBreak2D(Application):
    def initialize(self):
        self.Umax = (4. * 9.81)**0.5
        self.c0 = 10 * self.Umax
        self.rho0 = 1000.0
        self.p0 = self.c0 * self.c0 * self.rho0
        print(self.p0)
        self.pb = 10
        self.dx = 0.1
        self.volume = self.dx**2.

    def create_particles(self):
        dx = self.dx
        tank_length = 4.
        tank_height = 4.
        # center = [0.0, 0.0]
        # x, y = create_circle(2. * rad, dx)
        rho = 1000.
        m = rho * dx**2.
        h = 1.2 * dx

        fluid_length = 2.
        fluid_height = 2.

        xt, yt = get_2d_tank(dx, length=tank_length, height=tank_height,
                             base_center=[2, 0], num_layers=2)

        xf, yf = get_2d_block(dx=dx, length=fluid_length, height=fluid_height,
                              center=[0.5, 1])

        xf += 6. * dx
        yf += 1. * dx

        fluid = get_particle_array_tvf_fluid(x=xf, y=yf, m=m, h=h, rho=rho,
                                             name="fluid")
        tank = get_particle_array_tvf_solid(x=xt, y=yt, m=m, h=h, rho=rho,
                                            name="tank")

        add_ctvf_properties(fluid)
        add_ctvf_properties(tank)

        # volume is set as dx^2
        fluid.V[:] = 1. / dx**2.
        tank.V[:] = 1. / dx**2.

        return [fluid, tank]

    def create_solver(self):
        kernel = QuinticSpline(dim=2)

        integrator = PECIntegrator(fluid=TransportVelocityStep())

        # dt = 5e-6
        dt = 1e-5
        tf = 0.0076
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=True, cfl=0.3, n_damp=50)

        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    SummationDensity(dest='fluid', sources=['fluid', 'tank']),

                    # Added for CTVF
                    SummationDensityTmp(dest='fluid',
                                        sources=['fluid', 'tank']),
                    SummationDensityTmp(dest='tank', sources=['fluid',
                                                              'tank']),
                ], ),
            Group(
                equations=[
                    StateEquation(dest='fluid', sources=None, p0=self.p0,
                                  rho0=self.rho0, b=1.0),
                    SetWallVelocity(dest='tank', sources=['fluid'])
                ], ),
            Group(
                equations=[
                    SolidWallPressureBC(dest='tank', sources=['fluid'],
                                        rho0=self.rho0, p0=self.p0, b=1.0,
                                        gy=-9.81)
                ], ),
            Group(
                equations=[
                    MomentumEquationPressureGradient(dest='fluid',
                                                     sources=['fluid', 'tank'],
                                                     pb=self.p0, gy=-9.81,
                                                     tdamp=0.0),
                    MomentumEquationViscosity(dest='fluid', sources=['fluid'],
                                              nu=0.01),
                    SolidWallNoSlipBC(dest='fluid', sources=['tank'], nu=0.01),
                    MomentumEquationArtificialStress(dest='fluid',
                                                     sources=['fluid']),

                    # Added for CTVF
                    MinNeighbourRho(dest='fluid', sources=['fluid']),
                    GradientRhoTmp(dest='fluid', sources=['fluid', 'tank'])
                ], ),
            Group(
                equations=[
                    PsuedoForceOnFreeSurface(
                        dest='fluid', sources=['fluid'], dx=self.dx, m0=self.m0,
                        pb=self.pb, rho=self.rho
),
                ], ),
        ]
        return equations


if __name__ == '__main__':
    app = DamBreak2D()
    app.run()
