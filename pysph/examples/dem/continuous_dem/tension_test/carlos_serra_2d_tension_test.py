from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.sph.equation import Equation

from pysph.solver.application import Application
from pysph.sph.rigid_body import BodyForce
from pysph.dem.common import (EPECIntegratorMultiStage)
from pysph.dem.continuous_dem.carlos_serra_2d import (
    get_particle_array_bonded_dem_carlos_serra_2d, setup_bc_contacts,
    CarlosSeraIPForceStage1, CarlosSeraIPForceStage2, RK2StepCarlosSera)

from pysph.sph.scheme import SchemeChooser
from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.sph.equation import Group, MultiStageEquations


class MakeForcesZero(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        d_fx[d_idx] = 0.0
        d_fy[d_idx] = 0.0
        d_fz[d_idx] = 0.0


class ResetForce(Equation):
    def __init__(self, dest, sources, idx):
        self.idx = idx
        super(ResetForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        if d_idx == self.idx:
            d_fx[d_idx] = 0.0
            d_fy[d_idx] = 0.0
            d_fz[d_idx] = 0.0


class ApplyTensionForce(Equation):
    def __init__(self, dest, sources, fx, idx):
        self.fx = fx
        self.idx = idx
        super(ApplyTensionForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        if d_idx == self.idx:
            d_fx[d_idx] = self.fx


class TensionTest(Application):
    def initialize(self):
        self.dt = 1e-4
        self.pfreq = 100
        self.wall_time = 3
        self.debug_time = 0.0
        self.tf = self.wall_time + self.debug_time
        self.dim = 2
        self.en = 0.5
        self.kn = 5000
        # friction coefficient
        self.mu = 0.5
        self.gy = -9.81
        self.seval = None
        self.radius = 0.1
        self.slow_pfreq = 1
        self.slow_dt = 1e-4
        self.post_u = None

    def create_particles(self):
        # create a particle
        xp = np.array([0., 2. * self.radius])
        yp = np.array([0., 0.])
        rho = 2600
        m = rho * 4. / 3. * np.pi * self.radius**3
        I = 2. / 5. * m * self.radius**2.
        m_inverse = 1. / m
        I_inverse = 1. / I
        beam = get_particle_array_bonded_dem_carlos_serra_2d(
            x=xp, y=yp, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=self.radius, dem_id=1, h=1.2 * self.radius, name="beam")
        setup_bc_contacts(2, beam, 0.2)
        print(beam.bc_idx)

        return [beam]

    def create_equations(self):
        eq1 = [
            Group(equations=[
                # BodyForce(dest='sand', sources=None, gx=0.0, gy=-9.81,
                #           gz=0.0),
                MakeForcesZero(dest='beam', sources=None),
                ApplyTensionForce(dest='beam', sources=None, fx=50., idx=1),
                CarlosSeraIPForceStage1(dest='beam', sources=None, kn=self.kn),
                ResetForce(dest='beam', sources=None, idx=0),
            ])
        ]
        eq2 = [
            Group(equations=[
                # BodyForce(dest='sand', sources=None, gx=0.0, gy=-9.81,
                #           gz=0.0),
                MakeForcesZero(dest='beam', sources=None),
                ApplyTensionForce(dest='beam', sources=None, fx=50., idx=1),
                CarlosSeraIPForceStage2(dest='beam', sources=None, kn=self.kn),
                ResetForce(dest='beam', sources=None, idx=0),
            ])
        ]

        return MultiStageEquations([eq1, eq2])

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegratorMultiStage(beam=RK2StepCarlosSera())

        dt = self.dt
        tf = self.tf
        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
                        dt=dt, tf=tf)
        solver.set_disable_output(True)
        return solver

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['beam']
        b.vectors = 'fx, fy, fz'
        b.show_vectors = True
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {s_rad}
        b.scalar = 'fy'
        '''.format(s_rad=self.radius))

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        import matplotlib.pyplot as plt

        files = self.output_files
        # simulated data
        t, y, v = [], [], []
        for sd, arrays in iter_output(files):
            sand = arrays['sand']
            t.append(sd['t'])
            y.append(sand.y[0])
            v.append(sand.v[0])

        data = np.loadtxt('ffpw_y.csv', delimiter=',')
        ta = data[:, 0]
        ya = data[:, 1]
        plt.plot(ta, ya)
        plt.scatter(t, y)
        plt.savefig('t_vs_y.png')
        plt.clf()

        data = np.loadtxt('ffpw_v.csv', delimiter=',')
        ta = data[:, 0]
        va = data[:, 1]
        plt.plot(ta, va)
        plt.scatter(t, v)
        plt.savefig('t_vs_v.png')


if __name__ == '__main__':
    app = TensionTest()
    app.run()
