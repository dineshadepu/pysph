from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver

from pysph.solver.application import Application
from pysph.dem.discontinuous_dem.dem_nonlinear import EPECIntegratorMultiStage
from pysph.sph.rigid_body import BodyForce
from pysph.dem.discontinuous_dem.dem_linear import (
    get_particle_array_dem_linear, LinearDEMNoRotationScheme,
    UpdateTangentialContactsNoRotation, LinearPWFDEMNoRotationStage1,
    LinearPWFDEMNoRotationStage2, RK2StepLinearDEMNoRotation,
    UpdateTangentialContactsWallNoRotation, LinearPPFDEMNoRotationStage1,
    LinearPPFDEMNoRotationStage2)
from pysph.sph.scheme import SchemeChooser
from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.sph.equation import Group, MultiStageEquations


class BallSlipping(Application):
    def __init__(self, theta=0.):
        self.theta = theta
        super(BallSlipping, self).__init__()

    def initialize(self):
        self.dx = 0.05
        self.dt = 1e-4
        self.tf = 0.5
        self.dim = 2
        self.en = 0.9
        self.kn = 50000
        # friction coefficient
        self.mu = 0.5
        self.gy = -9.81
        self.seval = None
        self.radius = 0.1
        self.wall_time = 0.1

    def create_particles(self):
        # wall
        xw_a = np.array([0.])
        yw_a = np.array([0.])
        nxw_a = np.array([0])
        nyw_a = np.array([1.])
        wall = get_particle_array(x=xw_a, y=yw_a, nx=nxw_a, ny=nyw_a, nz=0.,
                                  constants={'np': len(xw_a)}, name="wall")
        wall.add_property('dem_id', type='int')
        wall.dem_id[:] = 0

        # create a particle
        xp = np.array([0.])
        yp = np.array([self.radius])
        rho = 2600
        m = rho * 4. / 3. * np.pi * self.radius**3
        I = 2. / 5. * m * self.radius**2.
        m_inverse = 1. / m
        I_inverse = 1. / I
        sand = get_particle_array_dem_linear(
            x=xp, y=yp, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=self.radius, dem_id=1, h=1.2 * self.dx / 2., name="sand")

        return [wall, sand]

    def create_scheme(self):
        ldems = LinearDEMNoRotationScheme(
            dem_bodies=['sand'], rigid_bodies=None, solids=None, walls=[
                'wall'
            ], dim=self.dim, kn=self.kn, mu=self.mu, en=self.en, gy=self.gy)
        s = SchemeChooser(default='ldems', ldems=ldems)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        tf = self.tf
        dt = self.dt
        scheme.configure()
        scheme.configure_solver(kernel=kernel,
                                integrator_cls=EPECIntegratorMultiStage, dt=dt,
                                tf=tf)

    # def create_equations(self):
    #     eq1 = [
    #         Group(
    #             equations=[
    #                 BodyForce(dest='sand', sources=None, gx=0.0, gy=-9.81,
    #                           gz=0.0),
    #                 LinearPWFDEMNoRotationStage1(dest='sand', sources=['wall'],
    #                                              kn=self.kn, mu=0.5, en=self.en),
    #             ], real=False, update_nnps=False, iterate=False,
    #             max_iterations=1, min_iterations=0, pre=None, post=None)
    #     ]
    #     eq2 = [
    #         Group(
    #             equations=[
    #                 BodyForce(dest='sand', sources=None, gx=0.0, gy=-9.81,
    #                           gz=0.0),
    #                 LinearPWFDEMNoRotationStage2(dest='sand', sources=['wall'],
    #                                              kn=self.kn, mu=0.5, en=self.en),
    #             ], real=False, update_nnps=False, iterate=False,
    #             max_iterations=1, min_iterations=0, pre=None, post=None)
    #     ]

    #     return MultiStageEquations([eq1, eq2])

    # def create_solver(self):
    #     kernel = CubicSpline(dim=self.dim)

    #     integrator = EPECIntegratorMultiStage(
    #         sand=RK2StepLinearDEMNoRotation())

    #     dt = self.dt
    #     tf = self.tf
    #     solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
    #                     dt=dt, tf=tf)
    #     solver.set_disable_output(True)
    #     return solver

    def _make_accel_eval(self, equations, pa_arrays):
        from pysph.tools.sph_evaluator import SPHEvaluator
        if self.seval is None:
            kernel = CubicSpline(dim=self.dim)
            seval = SPHEvaluator(arrays=pa_arrays, equations=equations,
                                 dim=self.dim, kernel=kernel)
            self.seval = seval
            return self.seval
        else:
            return self.seval

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        eqs1 = [
            Group(equations=[
                UpdateTangentialContactsWallNoRotation(dest='sand',
                                                       sources=['wall']),
            ])
        ]
        arrays = self.particles
        a_eval = self._make_accel_eval(eqs1, arrays)

        # When
        a_eval.evaluate(t, dt)

        # give a tangential velocity to the particles
        # once it settles down
        T = self.wall_time
        if (T - dt / 2) < t < (T + dt / 2):
            for pa in self.particles:
                if pa.name == 'sand':
                    pa.u[0] = 1.

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['sand']
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
    app = BallSlipping()
    app.run()
