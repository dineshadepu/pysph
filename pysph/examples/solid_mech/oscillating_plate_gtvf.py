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


class MakeVelocitiesInsideWallZero(Equation):
    def __init__(self, dest, sources, clearance):
        super(MakeVelocitiesInsideWallZero, self).__init__(dest, sources)
        self.clearance = clearance

    def initialize(self, d_idx, d_x, d_au, d_av, d_aw, d_auhat, d_avhat,
                   d_awhat):
        if d_x[d_idx] <= self.clearance:
            d_au[d_idx] = 0.0
            d_av[d_idx] = 0.0
            d_aw[d_idx] = 0.0
            d_auhat[d_idx] = 0.0
            d_avhat[d_idx] = 0.0
            d_awhat[d_idx] = 0.0


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class OscillatingPlateGTVF(Application):
    def initialize(self):
        self.L = 0.2
        self.H = 0.02
        # wave number K
        self.KL = 1.875
        self.K = 1.875 / self.L

        # edge velocity of the plate (m/s)
        self.Vf = 0.05
        self.dx_plate = 0.002
        self.h = 1.3 * self.dx_plate
        self.plate_rho0 = 1000.
        self.plate_E = 2. * 1e6
        self.plate_nu = 0.3975

        # self.plate_bulk_mod = get_bulk_mod(self.plate_G, self.plate_nu)
        self.plate_inside_wall_length = self.L / 4.
        self.wall_layers = 3

    def create_particles(self):
        # plate coordinates
        xp, yp = np.mgrid[-self.plate_inside_wall_length:self.L +
                          self.dx_plate / 2.:self.dx_plate, -self.H /
                          2.:self.H / 2. + self.dx_plate / 2.:self.dx_plate]
        xp = xp.ravel()
        yp = yp.ravel()

        m = self.plate_rho0 * self.dx_plate**2.

        plate = get_particle_array_gtvf(
            x=xp, y=yp, m=m, h=self.h, rho=self.plate_rho0, name="plate",
            constants=dict(E=self.plate_E, nu=self.plate_nu))
        ##################################
        # vertical velocity of the plate #
        ##################################
        # initialize with zero at the beginning
        v = np.zeros_like(xp)
        v = v.ravel()
        # speed of sound
        c0 = plate.c0[0]

        # set the vertical velocity for particles which are only
        # out of the wall
        K = self.K
        # L = self.L
        KL = self.KL
        M = sin(KL) + sinh(KL)
        N = cos(KL) + cosh(KL)
        Q = 2 * (cos(KL) * sinh(KL) - sin(KL) * cosh(KL))
        for i in range(len(v)):
            if xp[i] > 0.:
                # set the velocity
                tmp1 = (cos(K * xp[i]) - cosh(K * xp[i]))
                tmp2 = (sin(K * xp[i]) - sinh(K * xp[i]))
                v[i] = self.Vf * c0 * (M * tmp1 - N * tmp2) / Q
        plate.v = v
        # get the index of the particle which will be used to compute the
        # amplitude
        xp_max = max(xp)
        fltr = np.argwhere(xp == xp_max)
        fltr_idx = int(len(fltr) / 2.)
        amplitude_idx = fltr[fltr_idx][0]

        plate.add_constant("amplitude_idx", amplitude_idx)

        # get the minimum and maximum of the plate
        xp_min = xp.min()
        yp_min = yp.min()
        yp_max = yp.max()
        xw_upper, yw_upper = np.mgrid[-self.plate_inside_wall_length:self.
                                      dx_plate / 2.:self.dx_plate, yp_max +
                                      self.dx_plate:yp_max + self.dx_plate +
                                      (self.wall_layers - 1) * self.dx_plate +
                                      self.dx_plate / 2.:self.dx_plate]
        xw_upper = xw_upper.ravel()
        yw_upper = yw_upper.ravel()

        xw_lower, yw_lower = np.mgrid[-self.plate_inside_wall_length:self.
                                      dx_plate / 2.:self.dx_plate, yp_min -
                                      self.dx_plate:yp_min - self.dx_plate -
                                      (self.wall_layers - 1) * self.dx_plate -
                                      self.dx_plate / 2.:-self.dx_plate]
        xw_lower = xw_lower.ravel()
        yw_lower = yw_lower.ravel()

        xw_left_max = xp_min - self.dx_plate
        xw_left_min = xw_left_max - (
            self.wall_layers - 1) * self.dx_plate - self.dx_plate / 2.
        yw_left_max = yw_upper.max() + self.dx_plate / 2.
        yw_left_min = yw_lower.min()

        xw_left, yw_left = np.mgrid[xw_left_max:xw_left_min:-self.dx_plate,
                                    yw_left_min:yw_left_max:self.dx_plate]
        xw_left = xw_left.ravel()
        yw_left = yw_left.ravel()

        # wall coordinates
        xw, yw = np.concatenate((xw_lower, xw_upper, xw_left)), np.concatenate(
            (yw_lower, yw_upper, yw_left))

        # create the particle array
        wall = get_particle_array_gtvf(
            x=xw, y=yw, m=m, h=self.h, rho=self.plate_rho0, name="wall",
            constants=dict(E=self.plate_E, nu=self.plate_nu))

        wall.set_lb_props(list(wall.properties.keys()))

        return [plate, wall]

    def create_solver(self):
        # kernel = CubicSpline(dim=2)
        kernel = WendlandQuintic(dim=2)

        integrator = GTVFEPECIntegrator(plate=GTVFSolidRK2Step())
        # integrator = GTVFEPECIntegrator(plate=GTVFSolidRK2Step(),
        #                                 wall=GTVFSolidRK2Step())

        dt = 1e-5
        tf = 0.2

        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False, pfreq=10)
        return solver

    def create_equations(self):
        equations = [
            # Group(
            #     equations=[
            #         # correct the density
            #         CorrectDensity(dest='plate', sources=['plate', 'wall']),
            #         # CorrectDensity(dest='wall', sources=['plate', 'wall']),
            #     ], ),
            Group(
                equations=[
                    StateEquationGTVF(dest='plate', sources=None),
                    # StateEquationGTVF(dest='wall', sources=None),
                ], ),
            Group(
                equations=[
                    # p
                    # arho
                    DensityEvolutionUhat(dest='plate', sources=['plate', 'wall']),
                    # DensityEvolution(dest='wall', sources=['plate', 'wall']),

                    # au, av, aw, auhat, avhat, awhat
                    MomentumEquationSolidGTVF(dest='plate',
                                              sources=['plate', 'wall']),

                    # ads,
                    VelocityGradientHat(dest='plate', sources=['plate', 'wall']),
                    # VelocityGradientHat(dest='plate',
                    #                     sources=['plate', 'wall']),
                    DeviatoricStressRate(dest='plate',
                                         sources=['plate', 'wall']),
                ], ),
            # Group(
            #     equations=[
            #         MakeVelocitiesInsideWallZero(dest='plate', sources=None,
            #                                      clearance=self.dx_plate)
            #     ], ),
        ]
        return equations

    def post_process(self):
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output

        files = self.output_files
        t, amplitude = [], []
        for sd, array in iter_output(files, 'plate'):
            _t = sd['t']
            t.append(_t)
            amplitude.append(array.y[array.amplitude_idx[0]])

        import matplotlib
        import os
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt
        plt.clf()
        plt.plot(t, amplitude)
        plt.xlabel('t')
        plt.ylabel('Amplitude')
        plt.legend()
        fig = os.path.join(self.output_dir, "amplitude.png")
        plt.savefig(fig, dpi=300)

    def _get_sph_evaluator(self, arrays):
        if not hasattr(self, '_sph_eval'):
            from pysph.tools.sph_evaluator import SPHEvaluator
            equations = [
                CorrectDensity(dest='plate', sources=['plate', 'wall'])
            ]
            sph_eval = SPHEvaluator(arrays=arrays, equations=equations, dim=2,
                                    kernel=WendlandQuintic(dim=2))
            self._sph_eval = sph_eval
        return self._sph_eval

    def post_step(self, solver):
        pass
        # self._get_sph_evaluator(solver.particles).evaluate()


if __name__ == '__main__':
    app = OscillatingPlateGTVF()
    app.run()
    app.post_process()
