import numpy as np
from math import cos, sin, cosh, sinh

# SPH equations
from pysph.sph.equation import Group
from pysph.sph.basic_equations import (IsothermalEOS, ContinuityEquation,
                                       MonaghanArtificialViscosity,
                                       XSPHCorrection, VelocityGradient2D)
from pysph.sph.solid_mech.basic import (MomentumEquationWithStress,
                                        HookesDeviatoricStressRate,
                                        MonaghanArtificialStress)

from pysph.tools.geometry import (get_2d_circle)

from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import SolidMechStep


def get_bulk_mod(G, nu):
    ''' Get the bulk modulus from shear modulus and Poisson ratio '''
    return 2.0 * G * (1 + nu) / (3 * (1 - 2 * nu))


def get_speed_of_sound(E, nu, rho0):
    return np.sqrt(E / (3 * (1. - 2 * nu) * rho0))


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class MakeAccelerationsZeroForRestPlate(Equation):
    def __init__(self, dest, sources, upper, lower):
        self.upper = upper
        self.lower = lower
        super(MakeAccelerationsZeroForRestPlate, self).__init__(dest, sources)

    def initialize(self, d_idx, d_y, d_ax, d_ay, d_az, d_au, d_av, d_aw, d_u,
                   d_v, d_w, d_p, d_arho):
        if d_y[d_idx] > self.upper or d_y[d_idx] < self.lower:
            d_ax[d_idx] = 0.0
            d_ay[d_idx] = 0.0
            d_az[d_idx] = 0.0
            d_au[d_idx] = 0.0
            d_av[d_idx] = 0.0
            d_aw[d_idx] = 0.0
            d_u[d_idx] = 0.0
            d_v[d_idx] = 0.0
            d_w[d_idx] = 0.0
            d_arho[d_idx] = 0.0
            d_p[d_idx] = 0.0


class RubberImpact(Application):
    def initialize(self):
        self.plate_length = 0.002
        self.plate_height = 0.05
        self.plate_rest_len = self.plate_height / 8.
        self.plate_rest_lower_limit = -self.plate_height / 2.
        self.plate_rest_upper_limit = self.plate_height / 2.
        self.plate_spacing = 0.0003

        self.plate_h = 1.3 * self.plate_spacing
        self.plate_rho0 = 1.2 * 1e3
        self.plate_E = 1e7
        self.plate_nu = 0.49
        self.plate_G = self.plate_E / (2. * (1. + self.plate_nu))
        self.plate_cs = get_speed_of_sound(self.plate_E, self.plate_nu,
                                           self.plate_rho0)

        self.ball_radius = 0.005
        self.ball_spacing = 0.0003
        self.ball_center = np.array(
            [-self.ball_radius - 1. * self.ball_spacing, 0.])

        self.ball_h = 1.3 * self.ball_spacing
        self.ball_rho0 = 1.2 * 1e3
        self.ball_E = 1e7
        self.ball_nu = 0.49
        self.ball_G = self.ball_E / (2. * (1. + self.ball_nu))
        self.ball_cs = get_speed_of_sound(self.ball_E, self.ball_nu,
                                          self.ball_rho0)

    def create_particles(self):
        # create wall coordinates
        xp, yp = np.mgrid[0:self.plate_length + self.plate_spacing / 2.:self.
                          plate_spacing, -self.plate_height / 2. -
                          self.plate_rest_len:self.plate_height / 2. +
                          self.plate_rest_len +
                          self.plate_spacing / 2.:self.plate_spacing]
        xp = xp.ravel()
        yp = yp.ravel()

        m = self.plate_rho0 * self.plate_spacing**2.
        # speed of sound
        cs = get_speed_of_sound(self.plate_E, self.plate_nu, self.plate_rho0)

        plate = get_particle_array(x=xp, y=yp, m=m, h=self.plate_h,
                                   rho=self.plate_rho0, cs=cs, name="plate")
        # add properties
        add_properties(plate, 'cs', 'e', 'v00', 'v01', 'v02', 'v10', 'v11',
                       'v12', 'v20', 'v21', 'v22', 'r00', 'r01', 'r02', 'r11',
                       'r12', 'r22', 's00', 's01', 's02', 's11', 's12', 's22',
                       'as00', 'as01', 'as02', 'as11', 'as12', 'as22', 's000',
                       's010', 's020', 's110', 's120', 's220', 'arho', 'au',
                       'av', 'aw', 'ax', 'ay', 'az', 'ae', 'rho0', 'u0', 'v0',
                       'w0', 'x0', 'y0', 'z0', 'e0')

        # create ball
        xb, yb = get_2d_circle(self.ball_spacing, self.ball_radius,
                               self.ball_center)
        xb = xb.ravel()
        yb = yb.ravel()

        m = self.ball_rho0 * self.ball_spacing**2.
        cs = get_speed_of_sound(self.ball_E, self.ball_nu, self.ball_rho0)
        u = 0.15 * cs

        ball = get_particle_array(x=xb, y=yb, u=u, m=m, h=self.ball_h,
                                  rho=self.ball_rho0, name="ball")

        add_properties(ball, 'cs', 'e', 'v00', 'v01', 'v02', 'v10', 'v11',
                       'v12', 'v20', 'v21', 'v22', 'r00', 'r01', 'r02', 'r11',
                       'r12', 'r22', 's00', 's01', 's02', 's11', 's12', 's22',
                       'as00', 'as01', 'as02', 'as11', 'as12', 'as22', 's000',
                       's010', 's020', 's110', 's120', 's220', 'arho', 'au',
                       'av', 'aw', 'ax', 'ay', 'az', 'ae', 'rho0', 'u0', 'v0',
                       'w0', 'x0', 'y0', 'z0', 'e0')

        return [plate, ball]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        self.plate_wdeltap = kernel.kernel(rij=self.plate_spacing,
                                           h=self.plate_h)
        self.ball_wdeltap = kernel.kernel(rij=self.ball_spacing, h=self.ball_h)

        integrator = EPECIntegrator(plate=SolidMechStep(),
                                    ball=SolidMechStep())

        dt = 1e-7
        tf = 10000 * dt

        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)
        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                # p
                IsothermalEOS(dest='ball', sources=None, rho0=self.ball_rho0,
                              c0=self.ball_cs, p0=0.0),
                IsothermalEOS(dest='plate', sources=None, rho0=self.plate_rho0,
                              c0=self.plate_cs, p0=0.0),

                # vi,j : requires properties v00, v01, v10, v11
                VelocityGradient2D(dest='plate', sources=['plate', 'ball']),
                VelocityGradient2D(dest='ball', sources=['plate', 'ball']),

                # rij : requires properties r00, r01, r02, r11, r12, r22,
                #                           s00, s01, s02, s11, s12, s22
                MonaghanArtificialStress(dest='plate', sources=None, eps=0.3),
                MonaghanArtificialStress(dest='ball', sources=None, eps=0.3),
            ]),

            # Acceleration variables are now computed
            Group(equations=[

                # arho
                ContinuityEquation(dest='plate', sources=['plate', 'ball']),
                ContinuityEquation(dest='ball', sources=['plate', 'ball']),

                # au, av
                MomentumEquationWithStress(dest='plate', sources=[
                    'plate', 'ball'
                ], n=4, wdeltap=self.plate_wdeltap),
                MomentumEquationWithStress(dest='ball', sources=[
                    'plate', 'ball'
                ], n=4, wdeltap=self.ball_wdeltap),

                # au, av
                MonaghanArtificialViscosity(dest='plate', sources=[
                    'plate', 'ball'
                ], alpha=1.0, beta=1.0),
                MonaghanArtificialViscosity(dest='ball', sources=[
                    'plate', 'ball'
                ], alpha=1.0, beta=1.0),

                # a_s00, a_s01, a_s11
                HookesDeviatoricStressRate(dest='plate', sources=None,
                                           shear_mod=self.plate_G),
                HookesDeviatoricStressRate(dest='ball', sources=None,
                                           shear_mod=self.ball_G),

                # ax, ay, az
                XSPHCorrection(dest='plate', sources=[
                    'plate',
                ], eps=0.5),
                XSPHCorrection(dest='ball', sources=[
                    'ball',
                ], eps=0.5),
            ]),
            Group(equations=[
                MakeAccelerationsZeroForRestPlate(
                    dest='plate', sources=None, upper=self.
                    plate_rest_upper_limit, lower=self.plate_rest_lower_limit),
            ])
        ]
        return equations

    # def post_process(self):
    #     if len(self.output_files) == 0:
    #         return

    #     from pysph.solver.utils import iter_output

    #     files = self.output_files
    #     t, amplitude = [], []
    #     for sd, array in iter_output(files, 'plate'):
    #         _t = sd['t']
    #         t.append(_t)
    #         amplitude.append(array.y[array.amplitude_idx[0]])

    #     import matplotlib
    #     import os
    #     matplotlib.use('Agg')

    #     from matplotlib import pyplot as plt
    #     plt.clf()
    #     plt.plot(t, amplitude, label="exact")
    #     plt.xlabel('t')
    #     plt.ylabel('max velocity')
    #     plt.legend()
    #     fig = os.path.join(self.output_dir, "amplitude.png")
    #     plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = RubberImpact()
    app.run()
    # app.post_process()
