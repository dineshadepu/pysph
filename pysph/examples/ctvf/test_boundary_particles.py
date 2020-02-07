import os
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
from pysph.solver.utils import (load)


from pysph.sph.ctvf import (SummationDensityTmp, GradientRhoTmp,
                            PsuedoForceOnFreeSurface, add_ctvf_properties,
                            MinNeighbourRho)

from pysph.base.kernels import QuinticSpline
from pysph.solver.output import dump
from pysph.sph.tests.test_ctvf import (create_circle, create_sin)

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
                            MomentumEquationPressureGradientCTVF,
                            IdentifyBoundaryParticle1,
                            IdentifyBoundaryParticle2)

# for normals
from pysph.sph.isph.wall_normal import ComputeNormals, SmoothNormals

# geometry
from pysph.tools.geometry import get_2d_tank, get_2d_block


class TestBoundaryParticles():
    def _make_accel_eval(self, equations):
        kernel = QuinticSpline(dim=self.dim)
        seval = SPHEvaluator(arrays=self.pa_arrays, equations=equations,
                             dim=self.dim, kernel=kernel)
        return seval

    def dump(self, filename):
        os.makedirs(self.folder_name, exist_ok=True)
        dump(os.path.join(self.folder_name, filename), self.pa_arrays,
             dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=False)

    def create_equations_single_fluid(self, fluid_name):
        # Given
        # pa = self.pa
        dest = fluid_name
        sources = [fluid_name]
        eqs = [
            Group(equations=[
                ComputeNormals(dest=dest, sources=sources),
            ]),
            Group(equations=[SmoothNormals(dest=dest, sources=sources)]),

            Group(equations=[IdentifyBoundaryParticle1(dest=dest, sources=sources, fac=self.fac)]),
            Group(equations=[IdentifyBoundaryParticle2(dest=dest, sources=sources, fac=self.fac)]),
        ]
        return eqs

    def create_equations_single_fluid_with_boundary(self, fluid_name, boundary_name):
        # Given
        # pa = self.pa
        dest = fluid_name
        sources = [fluid_name, boundary_name]
        eqs = [
            Group(equations=[
                ComputeNormals(dest=dest, sources=sources),
            ]),
            Group(equations=[SmoothNormals(dest=dest, sources=sources)]),

            # Group(equations=[IdentifyBoundaryParticle1(dest=dest, sources=sources, fac=self.fac)]),
            # Group(equations=[IdentifyBoundaryParticle2(dest=dest, sources=sources, fac=self.fac)]),
            Group(equations=[IdentifyBoundaryParticle1(dest=dest, sources=[fluid_name], fac=self.fac)]),
            Group(equations=[IdentifyBoundaryParticle2(dest=dest, sources=[fluid_name], fac=self.fac)]),
        ]
        return eqs

    def test_circle_geometry(self):
        self.folder_name = "test_circle_geometry_output"

        self.dim = 2
        dx = 0.1
        self.fac = dx / 10
        rho = 1000.
        m = rho * dx**2.
        h = 1.3 * dx
        x, y = create_circle()
        u = x + y
        fluid = get_particle_array_gtvf(name='fluid', x=x, y=y, h=h, m=m,
                                        rho=rho, u=u, V=1.0)
        add_ctvf_properties(fluid)

        self.pa_arrays = [fluid]

        # dump the particle array
        self.dump("test_circle_geometry_0")

        a_eval = self._make_accel_eval(
            self.create_equations_single_fluid(fluid.name))

        # When
        a_eval.evaluate(0.0, 0.1)

        # dump the particle array after evaluation
        self.dump("test_circle_geometry_1")

    def test_sin_geometry(self):
        self.folder_name = "test_sin_geometry_output"

        self.dim = 2
        dx = 0.005
        self.fac = dx / 2
        rho = 1000.
        m = rho * dx**2.
        h = 1.3 * dx
        x, y = create_sin(dx, np.pi)
        u = x + y
        fluid = get_particle_array_gtvf(name='fluid', x=x, y=y, h=h, m=m,
                                        rho=rho, u=u, V=1.0)
        add_ctvf_properties(fluid)

        self.pa_arrays = [fluid]

        # dump the particle array
        self.dump("test_sin_geometry_0")

        a_eval = self._make_accel_eval(
            self.create_equations_single_fluid(fluid.name))

        # When
        a_eval.evaluate(0.0, 0.1)

        # dump the particle array after evaluation
        self.dump("test_sin_geometry_1")

    def test_dam_break_geometry(self):
        self.folder_name = "test_dam_break_geometry_output"
        self.dim = 2
        dx = 0.03
        self.fac = dx / 2
        rho = 1000.
        m = rho * dx**2.
        h = 1.3 * dx
        # load x, y from file
        data = load("../dam_break_2d_output/dam_break_2d_0.hdf5")
        arrays = data['arrays']
        fluid = arrays['fluid']
        boundary = arrays['boundary']
        xf, yf = fluid.x, fluid.y
        xb, yb = boundary.x, boundary.y
        fluid = get_particle_array_gtvf(name='fluid', x=xf, y=yf, h=h, m=m,
                                        rho=rho)
        boundary = get_particle_array_gtvf(name='boundary', x=xb, y=yb, h=h, m=m,
                                           rho=rho)
        add_ctvf_properties(fluid)
        add_ctvf_properties(boundary)

        self.pa_arrays = [fluid, boundary]

        # dump the particle array
        self.dump("test_dam_break_geometry_0")

        a_eval = self._make_accel_eval(
            self.create_equations_single_fluid_with_boundary(fluid.name, boundary.name))

        # When
        a_eval.evaluate(0.0, 0.1)

        # dump the particle array after evaluation
        self.dump("test_dam_break_geometry_1")

    def test_dam_break_geometry_later(self):
        self.folder_name = "test_dam_break_geometry_later_output"
        self.dim = 2
        dx = 0.03
        self.fac = dx / 2
        rho = 1000.
        m = rho * dx**2.
        h = 1.3 * dx
        # load x, y from file
        data = load("../dam_break_2d_output/dam_break_2d_27600.hdf5")
        arrays = data['arrays']
        fluid = arrays['fluid']
        boundary = arrays['boundary']
        xf, yf = fluid.x, fluid.y
        xb, yb = boundary.x, boundary.y
        fluid = get_particle_array_gtvf(name='fluid', x=xf, y=yf, h=h, m=m,
                                        rho=rho)
        boundary = get_particle_array_gtvf(name='boundary', x=xb, y=yb, h=h, m=m,
                                           rho=rho)
        add_ctvf_properties(fluid)
        add_ctvf_properties(boundary)

        self.pa_arrays = [fluid, boundary]

        # dump the particle array
        self.dump("test_dam_break_geometry_0")

        a_eval = self._make_accel_eval(
            self.create_equations_single_fluid_with_boundary(fluid.name, boundary.name))

        # When
        a_eval.evaluate(0.0, 0.1)

        # dump the particle array after evaluation
        self.dump("test_dam_break_geometry_1")


if __name__ == '__main__':
    test = TestBoundaryParticles()
    # test.test_circle_geometry()
    # test.test_sin_geometry()
    # test.test_dam_break_geometry()
    test.test_dam_break_geometry_later()

# normal[::3], normal[1::3], normal[2::3]
# grad_rho_x, grad_rho_y, grad_rho_z
# auhat, avhat, awhat
