from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.sph.equation import Equation

from pysph.solver.application import Application
from pysph.sph.rigid_body import BodyForce
from pysph.dem.common import (EPECIntegratorMultiStage)
from pysph.dem.continuous_dem.potyondy_3d import (
    get_particle_array_bonded_dem_potyondy_3d, setup_bc_contacts,
    Potyondy3dIPForceStage1, Potyondy3dIPForceStage2,
    DampingForce,
    RK2StepPotyondy3d)

from pysph.sph.equation import Group, MultiStageEquations


class MakeForcesZero(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz):
        d_fx[d_idx] = 0.0
        d_fy[d_idx] = 0.0
        d_fz[d_idx] = 0.0
        d_torx[d_idx] = 0.0
        d_tory[d_idx] = 0.0
        d_torz[d_idx] = 0.0


class ResetForce(Equation):
    def __init__(self, dest, sources, idx):
        self.idx = idx
        super(ResetForce, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_x, d_fx, d_fy, d_fz, d_wx, d_wy, d_wz, d_torx,
                  d_tory, d_torz):
        if d_idx == self.idx:
            d_fx[d_idx] = 0.0
            d_fy[d_idx] = 0.0
            d_fz[d_idx] = 0.0
            d_wx[d_idx] = 0.0
            d_wy[d_idx] = 0.0
            d_wz[d_idx] = 0.0
            d_torx[d_idx] = 0.0
            d_tory[d_idx] = 0.0
            d_torz[d_idx] = 0.0


class ApplyShearForce(Equation):
    def __init__(self, dest, sources, fy, idx):
        self.fy = fy
        self.idx = idx
        super(ApplyShearForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        if d_idx == self.idx:
            d_fy[d_idx] = self.fy


class TensionTest(Application):
    def initialize(self):
        self.dt = 1e-4
        self.pfreq = 100
        self.wall_time = 10
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
        beam = get_particle_array_bonded_dem_potyondy_3d(
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
                ApplyShearForce(dest='beam', sources=None, fy=50., idx=1),
                Potyondy3dIPForceStage1(dest='beam', sources=None, kn=self.kn,
                                        dim=2),
                ResetForce(dest='beam', sources=None, idx=0),
                DampingForce(dest='beam', sources=None, alpha=0.7)
            ])
        ]
        eq2 = [
            Group(equations=[
                # BodyForce(dest='sand', sources=None, gx=0.0, gy=-9.81,
                #           gz=0.0),
                MakeForcesZero(dest='beam', sources=None),
                ApplyShearForce(dest='beam', sources=None, fy=50., idx=1),
                Potyondy3dIPForceStage2(dest='beam', sources=None, kn=self.kn,
                                        dim=2),
                ResetForce(dest='beam', sources=None, idx=0),
                DampingForce(dest='beam', sources=None, alpha=0.7)
            ])
        ]

        return MultiStageEquations([eq1, eq2])

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegratorMultiStage(beam=RK2StepPotyondy3d())

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
        b.mask_on_ratio = 1
        b.scale_factor = 0.3
        b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
        b.plot.glyph.glyph_source.glyph_source.radius = {s_rad}
        b.scalar = 'fy'
        '''.format(s_rad=self.radius))

    def post_process(self):
        from pysph.solver.utils import iter_output
        import matplotlib.pyplot as plt

        files = self.output_files
        print(len(files))
        # simulated data
        t, fx, fy, torz = [], [], [], []
        for sd, arrays in iter_output(files):
            beam = arrays['beam']
            t.append(sd['t'])
            fx.append(beam.fx[1])
            fy.append(beam.fy[1])
            torz.append(beam.torz[1])

        plt.plot(t, fx, label="force x-direction")
        plt.plot(t, fy, label="force y-direction")
        plt.plot(t, torz, label="torque z-direction")
        plt.xlabel("time")
        plt.ylabel("Net force and torque on particle 1")
        plt.legend()
        plt.savefig('shear_test_potyondy_3d.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    app = TensionTest()
    # app.run()
    app.post_process()
