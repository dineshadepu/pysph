from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.sph.equation import Equation

from pysph.solver.application import Application
from pysph.dem.common import (EPECIntegratorMultiStage)

from pysph.sph.equation import Group, MultiStageEquations

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

    def initialize(self, d_idx, d_x, d_fx, d_fy, d_fz):
        if d_x[d_idx] < self.x:
            d_fx[d_idx] = 0.0
            d_fy[d_idx] = 0.0
            d_fz[d_idx] = 0.0


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


class CantileverBeam(Application):
    def initialize(self):
        self.dt = 1e-4
        self.pfreq = 100
        self.wall_time = 1.
        self.debug_time = 0.0
        self.tf = self.wall_time + self.debug_time
        self.dim = 2
        self.en = 0.5
        self.kn = 10000
        self.frc = 10000
        self.cn = 0.
        self.alpha_t = 0.1
        # friction coefficient
        self.mu = 0.5
        self.gy = -9.81
        self.seval = None
        self.radius = 0.1
        self.slow_pfreq = 1
        self.slow_dt = 1e-4
        self.post_u = None
        self.scale_factor = 1.1

    def create_particles(self):
        # create a particle
        xp, yp = np.mgrid[0.:20 * self.radius:2. * self.radius, 0.:10 *
                          self.radius:2. * self.radius]
        # xp, yp = np.mgrid[0.:20*self.radius:2.*self.radius, 0.:2*self.radius:2.*self.radius]
        # xp, yp = np.mgrid[0.:4*self.radius:2.*self.radius, 0.:2*self.radius:2.*self.radius]
        xp = xp.ravel()
        yp = yp.ravel()
        rho = 2600
        m = rho * 4. / 3. * np.pi * self.radius**3
        I = 2. / 5. * m * self.radius**2.
        m_inverse = 1. / m
        I_inverse = 1. / I
        beam = get_particle_array_bonded_dem_onate(
            x=xp, y=yp, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=self.radius / self.scale_factor, dem_id=1,
            h=1.2 * self.radius, name="beam", clrnc=0.2, dim=2)
        print(beam.bc_idx)

        return [beam]

    def create_equations(self):
        eq1 = [
            Group(equations=[
                ApplyTensionForce(dest='beam', sources=None, fx=np.array(
                    [0.]), fy=np.array([self.frc]), fz=np.array([0.]), idx=np.
                                  array([49]))
            ]),
            Group(equations=[
                OnateBDEMForceStage1(dest='beam', sources=None, kn=self.kn,
                                     cn=self.cn),
            ]),
            Group(equations=[
                GlobalDampingForce(dest='beam', sources=None, alpha_t=self.
                                   alpha_t)
            ]),
            Group(equations=[
                ResetForce(dest='beam', sources=None, x=self.radius +
                           self.radius / 100.),
            ])
        ]
        eq2 = [
            Group(equations=[
                ApplyTensionForce(dest='beam', sources=None, fx=np.array(
                    [0.]), fy=np.array([self.frc]), fz=np.array([0.]), idx=np.
                                  array([1]))
            ]),
            Group(equations=[
                OnateBDEMForceStage2(dest='beam', sources=None, kn=self.kn,
                                     cn=self.cn),
            ]),
            Group(equations=[
                GlobalDampingForce(dest='beam', sources=None, alpha_t=self.
                                   alpha_t)
            ]),
            Group(equations=[
                ResetForce(dest='beam', sources=None, x=self.radius +
                           self.radius / 100.),
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
        b.vectors = 'fx, fy, fz'
        b.show_vectors = True
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {s_rad}
        b.scalar = 'fy'
        '''.format(s_rad=self.radius / self.scale_factor))


if __name__ == '__main__':
    app = CantileverBeam()
    app.run()
