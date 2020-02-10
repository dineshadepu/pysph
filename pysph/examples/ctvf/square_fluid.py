"""Square fluid

This shows how one can explicitly setup equations and the solver instead of
using a scheme.
"""
from __future__ import print_function
from numpy import ones_like, mgrid
import numpy as np
import os

# PySPH base and carray imports
# from pysph.base.utils import get_particle_array_wcsph
from pysph.base.kernels import Gaussian, QuinticSpline

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import XSPHCorrection, ContinuityEquation
from pysph.sph.wc.basic import TaitEOS, MomentumEquation
from pysph.sph.equation import Group, MultiStageEquations

from pysph.examples.elliptical_drop import EllipticalDrop as EDScheme

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

from pysph.sph.ctvf import (IdentifyBoundaryParticle2,
                            IdentifyBoundaryParticleCosAngle,
                            SetHIJForInsideParticles)
# for normals
from pysph.sph.isph.wall_normal import ComputeNormals, SmoothNormals

# for benchmarking
from pysph.examples.elliptical_drop import (exact_solution)


class EllipticalDrop(Application):
    def initialize(self):
        self.co = 1400.0
        self.rho = 1.0
        self.rho0 = 1.0
        self.p0 = self.co**2. * self.rho
        self.nu = 0.01
        self.hdx = 1.3
        self.dx = 0.025
        self.fac = self.dx / 2.
        self.h = self.hdx * self.dx
        self.kernel_factor = 3
        self.m0 = self.rho * self.dx**2.
        self.alpha = 0.1

        self.pb = self.p0

    def create_particles(self):
        """Create the circular patch of fluid."""
        dx = self.dx
        hdx = self.hdx
        rho = self.rho
        name = 'fluid'
        x, y = mgrid[-1.05:1.05 + 1e-4:dx, -1.05:1.05 + 1e-4:dx]
        # Get the particles inside the circle.
        # condition = ~((x * x + y * y - 1.0) > 1e-10)
        # x = x[condition].ravel()
        # y = y[condition].ravel()

        m = ones_like(x) * dx * dx * rho
        h = ones_like(x) * hdx * dx
        rho = ones_like(x) * rho
        u = 0.
        v = 0.

        pa = get_particle_array_gtvf(x=x, y=y, m=m, rho=rho, h=h, u=u, v=v,
                                     name=name)
        add_ctvf_properties(pa)

        print("Stationary Square fluid :: %d particles" %
              (pa.get_number_of_particles()))

        # add requisite variables needed for this formulation
        for name in ('arho', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'rho0', 'u0',
                     'v0', 'w0', 'x0', 'y0', 'z0'):
            pa.add_property(name)

        # set the output property arrays
        pa.set_output_arrays(
            ['x', 'y', 'u', 'v', 'rho', 'm', 'h', 'p', 'pid', 'tag', 'gid'])

        return [pa]

    def create_solver(self):
        kernel = QuinticSpline(dim=2)

        integrator = GTVFIntegrator(fluid=GTVFStep())

        dt = 5e-6
        tf = 0.0076
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf)

        return solver

    def create_equations(self):

        stage1 = [
            Group(
                equations=[
                    ContinuityEquationGTVF(dest='fluid', sources=['fluid']),

                    # For CTVF
                    ComputeNormals(dest='fluid', sources=['fluid'])
                ], ),
            Group(
                equations=[
                    # For CTVF
                    SmoothNormals(dest='fluid', sources=['fluid'])
                ], ),

            Group(equations=[
                # For CTVF
                IdentifyBoundaryParticleCosAngle(dest='fluid',
                                                 sources=['fluid'])
            ]),

            # don't use this.
            # Group(equations=[IdentifyBoundaryParticle2(dest='fluid', sources=['fluid'], fac=self.fac)]),

            Group(equations=[
                SetHIJForInsideParticles(dest='fluid', sources=[
                    'fluid'
                ], h=self.h, kernel_factor=self.kernel_factor)
            ]),
        ]

        stage2 = [
            Group(
                equations=[
                    CorrectDensity(dest='fluid', sources=['fluid']),

                ], ),
            Group(
                equations=[
                    StateEquation(dest='fluid', sources=None, p0=self.p0,
                                  rho0=self.rho0, b=1.0)
                ], ),
            Group(
                equations=[
                    MomentumEquationPressureGradientCTVF(
                        dest='fluid', sources=['fluid'], pb=self.pb,
                        rho=self.rho),
                    MomentumEquationViscosity(dest='fluid', sources=['fluid'],
                                              nu=self.nu),
                    MomentumEquationArtificialStress(dest='fluid',
                                                     sources=['fluid'], dim=2)
                ], ),

            # for CTVF
            # Group(
            #     equations=[
            #         PsuedoForceOnFreeSurface(dest='fluid', sources=['fluid'],
            #                                  dx=self.dx, m0=self.m0,
            #                                  pb=self.pb, rho=self.rho0)
            #     ], ),
        ]
        return MultiStageEquations([stage1, stage2])

    def _make_final_plot(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
        except ImportError:
            print("Post processing requires matplotlib.")
            return
        last_output = self.output_files[-1]
        from pysph.solver.utils import load
        data = load(last_output)
        pa = data['arrays']['fluid']
        tf = data['solver_data']['t']
        a, A, po, xe, ye = exact_solution(tf)
        print("At tf=%s" % tf)
        print("Semi-major axis length (exact, computed) = %s, %s" %
              (1.0 / a, max(pa.y)))
        plt.plot(xe, ye)
        plt.scatter(pa.x, pa.y, marker='.')
        plt.ylim(-2, 2)
        plt.xlim(plt.ylim())
        plt.title("Particles at %s secs" % tf)
        plt.xlabel('x')
        plt.ylabel('y')
        fig = os.path.join(self.output_dir, "comparison.png")
        plt.savefig(fig, dpi=300)
        print("Figure written to %s." % fig)

    def _compute_results(self):
        from pysph.solver.utils import iter_output
        from collections import defaultdict
        data = defaultdict(list)
        for sd, array in iter_output(self.output_files, 'fluid'):
            _t = sd['t']
            data['t'].append(_t)
            m, u, v, x, y = array.get('m', 'u', 'v', 'x', 'y')
            vmag2 = u**2 + v**2
            data['ke'].append(0.5 * np.sum(m * vmag2))
            data['xmax'].append(x.max())
            data['ymax'].append(y.max())
            a, A, po, _xe, _ye = exact_solution(_t, n=0)
            data['minor'].append(a)
            data['major'].append(1.0 / a)

        for key in data:
            data[key] = np.asarray(data[key])
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(fname, **data)

    def post_process(self, info_file_or_dir):
        if self.rank > 0:
            return
        self.read_info(info_file_or_dir)
        if len(self.output_files) == 0:
            return
        self._compute_results()
        self._make_final_plot()


if __name__ == '__main__':
    app = EllipticalDrop()
    app.run()
    # app.post_process(app.info_filename)


# Semi-major axis length (exact, computed) = 1.9445172415576217, 1.9567269643753342
