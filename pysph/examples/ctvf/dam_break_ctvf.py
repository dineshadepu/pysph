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

from pysph.sph.equation import Group, MultiStageEquations
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.utils import (get_particle_array_tvf_fluid,
                              get_particle_array_tvf_solid)
from pysph.sph.ctvf import (SummationDensityTmp, GradientRhoTmp,
                            PsuedoForceOnFreeSurface, add_ctvf_properties,
                            MinNeighbourRho)

from pysph.base.kernels import QuinticSpline

# tvf
from pysph.sph.wc.transport_velocity import (
    StateEquation, SetWallVelocity, SolidWallPressureBC, VolumeSummation,
    SolidWallNoSlipBC, MomentumEquationArtificialViscosity, ContinuitySolid)

# gtvf
from pysph.sph.wc.gtvf import (get_particle_array_gtvf, GTVFIntegrator,
                               GTVFStep, ContinuityEquationGTVF,
                               CorrectDensity,
                               MomentumEquationArtificialStress,
                               MomentumEquationViscosity)

from pysph.sph.ctvf import (SummationDensityTmp, GradientRhoTmp,
                            PsuedoForceOnFreeSurface, MinNeighbourRho,
                            add_ctvf_properties,
                            MomentumEquationPressureGradientCTVF)

# for normals
from pysph.sph.isph.wall_normal import ComputeNormals, SmoothNormals

# geometry
from pysph.tools.geometry import get_2d_tank, get_2d_block


class DamBreak2D(Application):
    def initialize(self):
        self.fluid_column_height = 2.0
        self.fluid_column_width = 1.0
        self.container_height = 4.0
        self.container_width = 4.0
        self.ntank_layers = 4
        self.nu = 0.0
        self.dx = 0.03
        self.g = 9.81
        self.rho = 1000.0
        self.vref = np.sqrt(2 * 9.81 * self.fluid_column_height)
        self.co = 10.0 * self.vref
        self.gamma = 7.0
        self.alpha = 0.1
        self.beta = 0.0
        self.B = self.co * self.co * self.rho / self.gamma
        self.p0 = self.rho * self.co**2. / self.gamma
        self.b = 1
        self.hdx = 1.3
        self.h = self.hdx * self.dx
        self.m = self.dx**2 * self.rho

    def create_particles(self):
        xt, yt = get_2d_tank(dx=self.dx, length=self.container_width,
                             height=self.container_height, base_center=[2, 0],
                             num_layers=self.ntank_layers)
        xf, yf = get_2d_block(dx=self.dx, length=self.fluid_column_width,
                              height=self.fluid_column_height, center=[0.5, 1])

        xf += self.dx
        yf += self.dx

        fluid = get_particle_array_gtvf(name='fluid', x=xf, y=yf,
                                        h=self.h, m=self.m, rho=self.rho)
        tank = get_particle_array_tvf_solid(name='tank', x=xt, y=yt,
                                            h=self.h, m=self.m,
                                            rho=self.rho)
        tank.add_property('rho0')

        add_ctvf_properties(fluid)
        add_ctvf_properties(tank)

        return [fluid, tank]

    def create_solver(self):
        kernel = QuinticSpline(dim=2)

        integrator = GTVFIntegrator(fluid=GTVFStep())

        # dt = 5e-6
        dt = 1e-4
        tf = 1.
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf)

        return solver

    def create_equations(self):
        stage1 = [
            Group(equations=[SetWallVelocity(dest='tank',
                                             sources=['fluid'])], ),
            Group(
                equations=[
                    ContinuityEquationGTVF(dest='fluid', sources=['fluid']),
                    ContinuitySolid(dest='fluid', sources=['tank']),

                    # For CTVF
                    ComputeNormals(dest='fluid', sources=['tank', 'fluid'])
                ], ),

            Group(
                equations=[
                    # For CTVF
                    SmoothNormals(dest='fluid', sources=['fluid'])
                ],
            )
        ]

        stage2 = [
            Group(
                equations=[
                    CorrectDensity(dest='fluid', sources=['fluid', 'tank'])
                ], ),
            Group(
                equations=[
                    StateEquation(dest='fluid', sources=None,
                                  p0=self.p0, rho0=self.rho, b=self.b)
                ], ),
            Group(
                equations=[
                    VolumeSummation(dest='tank', sources=['fluid', 'tank']),
                    SolidWallPressureBC(dest='tank', sources=['fluid'],
                                        rho0=self.rho, p0=self.p0,
                                        b=self.b, gy=-9.81)
                ], ),
            Group(
                equations=[
                    MomentumEquationPressureGradientCTVF(
                        dest='fluid', sources=['fluid', 'tank'],
                        pb=self.p0, rho=self.rho, gy=-9.81),
                    MomentumEquationArtificialStress(dest='fluid',
                                                     sources=['fluid'], dim=2)
                ], )
        ]
        return MultiStageEquations([stage1, stage2])


if __name__ == '__main__':
    app = DamBreak2D()
    app.run()
