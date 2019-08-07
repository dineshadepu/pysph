"""An example test to verify the bonded DEM model. In this benchmark we analyze
two particles in tension.

Two particles in tension. Both the particles will a constant force acting on
them

"""

from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.dem.discontinuous_dem.dem_nonlinear import EPECIntegratorMultiStage

from pysph.sph.equation import Equation
from pysph.sph.equation import Group, MultiStageEquations
from pysph.solver.application import Application
from pysph.dem.continuous_dem.onate import (
    get_particle_array_bonded_dem_onate, OnateBDEMForceStage1,
    OnateBDEMForceStage2, GlobalDampingForce, RK2StepOnate)


class MakeForcesZero(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz):
        d_fx[d_idx] = 0.0
        d_fy[d_idx] = 0.0
        d_fz[d_idx] = 0.0
        d_torx[d_idx] = 0.0
        d_tory[d_idx] = 0.0
        d_torz[d_idx] = 0.0


class ResetForce(Equation):
    def __init__(self, dest, sources, x):
        self.x = x
        super(ResetForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_x, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz):
        if d_x[d_idx] < self.x:
            d_fx[d_idx] = 0.0
            d_fy[d_idx] = 0.0
            d_fz[d_idx] = 0.0
            d_torx[d_idx] = 0.0
            d_tory[d_idx] = 0.0
            d_torz[d_idx] = 0.0


class ApplyTensionForceCustom(Equation):
    def __init__(self, dest, sources, idx, fx, fy, fz):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        super(ApplyTensionForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        i = declare('int')
        if d_idx == 0:
            for i in range(self.length):
                d_fx[self.idx[i]] = self.fx[self.idx[i]]
                d_fy[self.idx[i]] = self.fy[self.idx[i]]
                d_fz[self.idx[i]] = self.fz[self.idx[i]]


class ApplyTensionForce(Equation):
    def __init__(self, dest, sources, idx, fx, fy, fz):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.idx = idx
        self.length = len(self.idx)
        super(ApplyTensionForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        i = declare('int')
        if d_idx == 0:
            for i in range(self.length):
                d_fx[self.idx[i]] = self.fx[i]
                d_fy[self.idx[i]] = self.fy[i]
                d_fz[self.idx[i]] = self.fz[i]


class TensionTest(Application):
    def initialize(self):
        self.radius = 1.
        self.clearence = 0.1
        self.dim = 2
        self.dt = 1e-4
        self.tf = 1.
        # self.tf = 5.*self.dt
        self.kn = 1e5
        self.cn = 100.
        self.frc = 5000

    def create_particles(self):
        x, y = np.array([0., 2. * self.radius]), np.array([0., 0.])
        m = 2000. * self.radius**2.
        I = 2. / 5. * m * self.radius**2.
        beam = get_particle_array_bonded_dem_onate(
            x=x, y=y, dim=2, m=m, m_inverse=1 / m, I_inverse=1 / I,
            h=2. * self.radius, rad_s=self.radius, clrnc=self.clearence,
            name='beam')

        print(beam.bc_total_contacts)
        print(beam.bc_idx)
        print(beam.bc_rest_len)
        return [beam]

    def create_equations(self):
        eq1 = [
            Group(equations=[
                ApplyTensionForce(dest='beam', sources=None, fx=np.array(
                    [-self.frc, self.frc]), fy=np.array([0., 0.]), fz=np.array(
                        [0., 0.]), idx=np.array([0, 1])),
            ]),
            Group(equations=[
                OnateBDEMForceStage1(dest='beam', sources=None, kn=self.kn,
                                     cn=self.cn),
            ]),
            Group(equations=[
                GlobalDampingForce(dest='beam', sources=None)
            ])
        ]
        eq2 = [
            Group(equations=[
                ApplyTensionForce(dest='beam', sources=None, fx=np.array(
                    [-self.frc, self.frc]), fy=np.array([0., 0.]), fz=np.array(
                        [0., 0.]), idx=np.array([0, 1])),
            ]),
            Group(equations=[
                OnateBDEMForceStage2(dest='beam', sources=None, kn=self.kn,
                                     cn=self.cn),
            ]),
            Group(equations=[
                GlobalDampingForce(dest='beam', sources=None)
            ])
        ]

        return MultiStageEquations([eq1, eq2])

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegratorMultiStage(beam=RK2StepOnate())

        dt = self.dt
        tf = self.tf
        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
                        dt=dt, tf=tf)
        solver.set_disable_output(True)
        return solver

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['beam']
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {radius}
        b.scalar = 'fx'
        '''.format(radius=self.radius))


if __name__ == '__main__':
    app = TensionTest()
    # [beam] = app.create_particles()
    app.run()
