"""
Study on coupled dynamics of ship and flooding water based on experimental and
SPH methods

"""
from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep


def get_particle_array_fluid_cheng(constants=None, **props):
    extra_props = [
        'cs', 'ax', 'ay', 'az', 'arho', 'x0', 'y0', 'z0', 'u0', 'v0', 'w0',
        'rho0'
    ]

    pa = get_particle_array(constants=constants, additional_props=extra_props,
                            **props)

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p'
    ])

    return pa


class ContinuityEquationFluid(Equation):
    def __init__(self, dest, sources, c0, alpha=0.2):
        self.c0 = c0
        self.alpha = alpha
        super(ContinuityEquationFluid, self).__init__(dest, sources)

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_arho, s_idx, s_m, s_rho, DWIJ, VIJ, XIJ,
             R2IJ, HIJ):
        tmp1 = d_rho[d_idx] - s_rho[s_idx]
        tmp2 = R2IJ + (0.01 * HIJ)**2.
        tmp = tmp1 / tmp2
        psi_ijdotdwij = tmp * (
            XIJ[0] * DWIJ[0] + XIJ[1] * DWIJ[1] + XIJ[2] * DWIJ[2])

        vijdotdwij = DWIJ[0] * VIJ[0] + DWIJ[1] * VIJ[1] + DWIJ[2] * VIJ[2]

        d_arho[d_idx] += vijdotdwij * s_m[s_idx]

        # diffusive term
        tmp3 = self.alpha * self.c0 * HIJ
        d_arho[d_idx] += tmp3 * psi_ijdotdwij * s_m[s_idx] / s_rho[s_idx]


class ContinuityEquationSolid(Equation):
    def __init__(self, dest, sources, c0, alpha=0.2):
        self.c0 = c0
        self.alpha = alpha
        super(ContinuityEquationSolid, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_arho, d_u, d_v, d_w, s_ug, s_vg, s_wg,
             s_idx, s_m, s_rho, DWIJ, XIJ, R2IJ, HIJ):
        tmp1 = d_rho[d_idx] - s_rho[s_idx]
        tmp2 = R2IJ + (0.01 * HIJ)**2.
        tmp = tmp1 / tmp2
        psi_ijdotdwij = tmp * (
            XIJ[0] * DWIJ[0] + XIJ[1] * DWIJ[1] + XIJ[2] * DWIJ[2])

        uij = d_u[d_idx] - s_ug[s_idx]
        vij = d_v[d_idx] - s_vg[s_idx]
        wij = d_w[d_idx] - s_wg[s_idx]
        vijdotdwij = DWIJ[0] * uij + DWIJ[1] * vij + DWIJ[2] * wij

        d_arho[d_idx] += vijdotdwij * s_m[s_idx]

        # diffusive term
        tmp3 = self.alpha * self.c0 * HIJ
        d_arho[d_idx] += tmp3 * psi_ijdotdwij * s_m[s_idx] / s_rho[s_idx]


class MomentumEquationFluid(Equation):
    def __init__(self, dest, sources, alpha, c0, rho0, dim, nu=0.0, gx=0.,
                 gy=0., gz=0.):
        self.c0 = c0
        self.alpha = alpha
        self.rho0 = rho0
        self.dim = dim
        self.nu0 = nu
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationFluid, self).__init__(dest, sources)

    def initialize(self, d_au, d_av, d_aw, d_idx):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_idx, s_idx, d_rho, d_cs, d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, s_p, VIJ, XIJ, HIJ, R2IJ, DWIJ, WIJ):

        rhoi21 = 1.0 / (d_rho[d_idx] * d_rho[d_idx])
        rhoj21 = 1.0 / (s_rho[s_idx] * s_rho[s_idx])

        vijdotxij = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            piij = vijdotxij / (R2IJ + (0.01 * HIJ**2.))

        tmpi = d_p[d_idx] * rhoi21
        tmpj = s_p[s_idx] * rhoj21

        # gradient and correction terms
        tmp = tmpi + tmpj

        tmp1 = 2. * self.nu0 * (self.dim + 2.) / d_rho[d_idx]
        tmp1 = tmp1 * s_m[s_idx] / s_rho[s_idx]

        d_au[d_idx] += -s_m[s_idx] * (tmp + piij * tmp1) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp + piij * tmp1) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmp + piij * tmp1) * DWIJ[2]


class MomentumEquationSolid(Equation):
    def __init__(self, dest, sources, alpha, c0, rho0, dim, nu=0.0, gx=0.,
                 gy=0., gz=0.):
        self.c0 = c0
        self.alpha = alpha
        self.rho0 = rho0
        self.dim = dim
        self.nu0 = nu
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationSolid, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, d_p, d_au, d_av, d_aw, d_u, d_v, d_w,
             s_ug, s_vg, s_wg, s_m, s_rho, s_p, XIJ, HIJ, R2IJ, EPS, DWIJ,
             WIJ):

        rhoi21 = 1.0 / (d_rho[d_idx] * d_rho[d_idx])
        rhoj21 = 1.0 / (s_rho[s_idx] * s_rho[s_idx])

        uij = d_u[d_idx] - s_ug[s_idx]
        vij = d_v[d_idx] - s_vg[s_idx]
        wij = d_w[d_idx] - s_wg[s_idx]
        vijdotxij = XIJ[0] * uij + XIJ[1] * vij + XIJ[2] * wij

        piij = 0.0
        if vijdotxij < 0:
            piij = vijdotxij / (R2IJ + (0.01 * HIJ**2.))

        tmpi = d_p[d_idx] * rhoi21
        tmpj = s_p[s_idx] * rhoj21

        # gradient and correction terms
        tmp = tmpi + tmpj

        tmp1 = 2. * self.nu0 * (self.dim + 2.) / d_rho[d_idx]
        tmp1 = tmp1 * s_m[s_idx] / s_rho[s_idx]

        d_au[d_idx] += -s_m[s_idx] * (tmp + piij * tmp1) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp + piij * tmp1) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmp + piij * tmp1) * DWIJ[2]


class RigidFluidForce(Equation):
    def loop(self, d_idx, s_idx, d_m, d_rho, d_p, d_fx, d_fy, d_fz, s_m, s_rho,
             s_p, VIJ, XIJ, HIJ, R2IJ, DWIJ, WIJ):

        rhoi21 = 1.0 / (d_rho[d_idx] * d_rho[d_idx])
        rhoj21 = 1.0 / (s_rho[s_idx] * s_rho[s_idx])

        vijdotxij = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            piij = vijdotxij / (R2IJ + (0.01 * HIJ**2.))

        tmpi = d_p[d_idx] * rhoi21
        tmpj = s_p[s_idx] * rhoj21

        # gradient and correction terms
        tmp = tmpi + tmpj

        d_fx[d_idx] += d_m[d_idx] * s_m[s_idx] * (tmp + piij * tmp) * DWIJ[0]
        d_fy[d_idx] += d_m[d_idx] * s_m[s_idx] * (tmp + piij * tmp) * DWIJ[1]
        d_fz[d_idx] += d_m[d_idx] * s_m[s_idx] * (tmp + piij * tmp) * DWIJ[2]


class StateEquation(Equation):
    def __init__(self, dest, sources, c0, rho0):
        self.c0 = c0
        self.c0_2 = c0**2.
        self.rho0 = rho0
        super(StateEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_rho):
        d_p[d_idx] = self.c0_2 * (d_rho[d_idx] - self.rho0)


class SourceNumberDensity(Equation):
    def initialize(self, d_idx, d_wij):
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, d_wij, WIJ):
        d_wij[d_idx] += WIJ


class SolidWallPressureBC(Equation):
    r"""Solid wall pressure boundary condition from Adami and Hu (transport
    velocity formulation).

    """

    def __init__(self, dest, sources, c0, rho0, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.c0_1_2 = 1. / c0**2
        self.rho0 = rho0

        super(SolidWallPressureBC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, s_rho, d_au, d_av, d_aw, WIJ, XIJ):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        d_p[d_idx] += s_p[s_idx] * WIJ + s_rho[s_idx] * gdotxij * WIJ

    def post_loop(self, d_idx, d_wij, d_p, d_rho, d_m):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]

        d_rho[d_idx] = d_p[d_idx] * self.c0_1_2 + self.rho0


class SetNoSlipWallVelocity(Equation):
    def initialize(self, d_idx, d_uf, d_vf, d_wf):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf, s_u, s_v, s_w, WIJ):
        # sum in Eq. (22)
        # this will be normalized in post loop
        d_uf[d_idx] += s_u[s_idx] * WIJ
        d_vf[d_idx] += s_v[s_idx] * WIJ
        d_wf[d_idx] += s_w[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx, d_ug, d_vg, d_wg, d_u,
                  d_v, d_w, d_normal_x, d_normal_y, d_normal_z):

        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface

        # Note that d_wij is already computed for the pressure BC.
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

            # normal component of the dummy velocity
            tmp_u = 2 * d_u[d_idx] - d_uf[d_idx]
            tmp_v = 2 * d_v[d_idx] - d_vf[d_idx]
            tmp_w = 2 * d_w[d_idx] - d_wf[d_idx]

            tmp_u_dot_n = (
                tmp_u * d_normal_x[d_idx] + tmp_v * d_normal_y[d_idx] +
                tmp_w * d_normal_z[d_idx])

            vt_x = tmp_u - tmp_u_dot_n * d_normal_x[d_idx]
            vt_y = tmp_v - tmp_u_dot_n * d_normal_y[d_idx]
            vt_z = tmp_w - tmp_u_dot_n * d_normal_z[d_idx]

            u_ext_dot_n = (d_uf[d_idx] * d_normal_x[d_idx] +
                           d_vf[d_idx] * d_normal_y[d_idx] +
                           d_wf[d_idx] * d_normal_z[d_idx])

            # Dummy velocities at the ghost points using Eq. (23),
            # d_u, d_v, d_w are the prescribed wall velocities.
            d_ug[d_idx] = vt_x + u_ext_dot_n * d_normal_x[d_idx]
            d_vg[d_idx] = vt_y + u_ext_dot_n * d_normal_y[d_idx]
            d_wg[d_idx] = vt_z + u_ext_dot_n * d_normal_z[d_idx]
        else:
            d_ug[d_idx] = d_u[d_idx]
            d_vg[d_idx] = d_v[d_idx]
            d_wg[d_idx] = d_w[d_idx]


class SetNoSlipWallVelocityAdami(Equation):
    def initialize(self, d_idx, d_uf, d_vf, d_wf):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf, s_u, s_v, s_w, WIJ):
        # sum in Eq. (22)
        # this will be normalized in post loop
        d_uf[d_idx] += s_u[s_idx] * WIJ
        d_vf[d_idx] += s_v[s_idx] * WIJ
        d_wf[d_idx] += s_w[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx, d_ug, d_vg, d_wg, d_u,
                  d_v, d_w):

        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface

        # Note that d_wij is already computed for the pressure BC.
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ug[d_idx] = 2*d_u[d_idx] - d_uf[d_idx]
        d_vg[d_idx] = 2*d_v[d_idx] - d_vf[d_idx]
        d_wg[d_idx] = 2*d_w[d_idx] - d_wf[d_idx]


class SetFreeSlipWallVelocity(Equation):
    def post_loop(self, d_idx, d_ug, d_vg, d_wg, d_u, d_v, d_w):
        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ug[d_idx] = d_u[d_idx]
        d_vg[d_idx] = d_v[d_idx]
        d_wg[d_idx] = d_w[d_idx]


class RK2ChengFluidStep(IntegratorStep):
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0,
                   d_w0, d_u, d_v, d_w, d_rho0, d_rho):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av, d_aw, d_arho, dt):
        dtb2 = 0.5 * dt
        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_aw[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av, d_aw, d_arho, dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]
