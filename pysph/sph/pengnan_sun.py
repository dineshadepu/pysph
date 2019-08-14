from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep


def get_particle_array_fluid_pengnan(constants=None, **props):
    extra_props = [
        'cs', 'ax', 'ay', 'az', 'arho', 'x0', 'y0', 'z0', 'u0', 'v0', 'w0',
        'rho0', 'div', 'dt_cfl', 'dt_force'
    ]

    pa = get_particle_array(constants=constants, additional_props=extra_props,
                            **props)

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p'
    ])

    return pa


class ContinuityEquationFluid(Equation):
    def __init__(self, dest, sources, c0, delta=0.2):
        self.c0 = c0
        self.delta = delta
        super(ContinuityEquationFluid, self).__init__(dest, sources)

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_arho, s_idx, s_m, s_rho, s_V, DWIJ, VIJ,
             XIJ, R2IJ, HIJ):
        tmp1 = d_rho[d_idx] - s_rho[s_idx]
        tmp2 = R2IJ + (0.01 * HIJ)**2.
        tmp = tmp1 / tmp2
        psi_ijdotdwij = tmp * (
            XIJ[0] * DWIJ[0] + XIJ[1] * DWIJ[1] + XIJ[2] * DWIJ[2])

        vijdotdwij = DWIJ[0] * VIJ[0] + DWIJ[1] * VIJ[1] + DWIJ[2] * VIJ[2]

        d_arho[d_idx] += d_rho[d_idx] * vijdotdwij * s_V[s_idx]

        # diffusive term
        tmp3 = self.delta * self.c0 * HIJ
        d_arho[d_idx] += tmp3 * psi_ijdotdwij * s_V[s_idx]


class ContinuityEquationSolid(Equation):
    def __init__(self, dest, sources, c0, delta=0.2):
        self.c0 = c0
        self.delta = delta
        super(ContinuityEquationSolid, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_arho, d_u, d_v, d_w, s_idx, s_m, s_rho, s_V,
             s_ug, s_vg, s_wg, DWIJ, XIJ, R2IJ, HIJ):
        tmp1 = d_rho[d_idx] - s_rho[s_idx]
        tmp2 = R2IJ + (0.01 * HIJ)**2.
        tmp = tmp1 / tmp2
        psi_ijdotdwij = tmp * (
            XIJ[0] * DWIJ[0] + XIJ[1] * DWIJ[1] + XIJ[2] * DWIJ[2])

        uij = d_u[d_idx] - s_ug[s_idx]
        vij = d_v[d_idx] - s_vg[s_idx]
        wij = d_w[d_idx] - s_wg[s_idx]
        vijdotdwij = DWIJ[0] * uij + DWIJ[1] * vij + DWIJ[2] * wij

        d_arho[d_idx] += d_rho[d_idx] * vijdotdwij * s_V[s_idx]

        # diffusive term
        tmp3 = self.delta * self.c0 * HIJ
        d_arho[d_idx] += tmp3 * psi_ijdotdwij * s_V[s_idx]


class MomentumEquationFluid(Equation):
    def __init__(self, dest, sources, alpha, c0, rho0, dim, nu=0.0):
        self.c0 = c0
        self.alpha = alpha
        self.rho0 = rho0
        self.dim = dim
        super(MomentumEquationFluid, self).__init__(dest, sources)

    def initialize(self, d_au, d_av, d_aw, d_idx):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_au, d_av, d_aw, s_V, d_p, s_p, DWIJ, s_idx,
             d_m, R2IJ, XIJ, EPS, VIJ, d_rho, s_rho, HIJ):
        p_ij = (s_rho[s_idx] * d_p[d_idx] + d_rho[d_idx] * s_p[s_idx]) / (
            d_rho[d_idx] * s_rho[s_idx])
        uijdotxij = XIJ[0] * VIJ[0] + XIJ[1] * VIJ[1] + XIJ[2] * VIJ[2]
        tmp2 = R2IJ + (0.01 * HIJ)**2.

        # artificial viscosity coefficient
        pi_ij = uijdotxij / tmp2
        mi_1 = 1. / d_m[d_idx]
        tmp1 = self.alpha * HIJ * self.c0 * self.rho0 / d_rho[d_idx] * pi_ij

        factor = (d_V[d_idx]**2. + s_V[s_idx]**2.) * p_ij
        d_au[d_idx] += (-mi_1 * factor * DWIJ[0]) + (
            tmp1 * DWIJ[0] * s_V[s_idx])
        d_av[d_idx] += (-mi_1 * factor * DWIJ[1]) + (
            tmp1 * DWIJ[1] * s_V[s_idx])
        d_aw[d_idx] += (-mi_1 * factor * DWIJ[2]) + (
            tmp1 * DWIJ[2] * s_V[s_idx])


class MomentumEquationSolid(Equation):
    def __init__(self, dest, sources, alpha, c0, rho0, dim, nu=0.0):
        self.c0 = c0
        self.alpha = alpha
        self.rho0 = rho0
        self.dim = dim
        super(MomentumEquationSolid, self).__init__(dest, sources)

    def loop(self, d_idx, d_V, d_u, d_v, d_w, d_au, d_av, d_aw, s_V, d_p, s_p,
             DWIJ, s_idx, d_m, R2IJ, XIJ, EPS, d_rho, s_rho, HIJ,
             s_ug, s_vg, s_wg):
        p_ij = (s_rho[s_idx] * d_p[d_idx] + d_rho[d_idx] * s_p[s_idx]) / (
            d_rho[d_idx] * s_rho[s_idx])

        uij = d_u[d_idx] - s_ug[s_idx]
        vij = d_v[d_idx] - s_vg[s_idx]
        wij = d_w[d_idx] - s_wg[s_idx]
        uijdotxij = XIJ[0] * uij + XIJ[1] * vij + XIJ[2] * wij
        tmp2 = R2IJ + (0.01 * HIJ)**2.

        # artificial viscosity coefficient
        pi_ij = uijdotxij / tmp2
        mi_1 = 1. / d_m[d_idx]
        tmp1 = self.alpha * HIJ * self.c0 * self.rho0 / d_rho[d_idx] * pi_ij

        factor = (d_V[d_idx]**2. + s_V[s_idx]**2.) * p_ij
        d_au[d_idx] += (-mi_1 * factor * DWIJ[0]) + (
            tmp1 * DWIJ[0] * s_V[s_idx])
        d_av[d_idx] += (-mi_1 * factor * DWIJ[1]) + (
            tmp1 * DWIJ[1] * s_V[s_idx])
        d_aw[d_idx] += (-mi_1 * factor * DWIJ[2]) + (
            tmp1 * DWIJ[2] * s_V[s_idx])


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

    def post_loop(self, d_idx, d_wij, d_p, d_rho, d_V, d_m):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]

        d_rho[d_idx] = d_p[d_idx] / self.c0_1_2 + self.rho0
        d_V[d_idx] = d_m[d_idx] / d_rho[d_idx]


class SetWallVelocity(Equation):
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
        d_ug[d_idx] = 2 * d_u[d_idx] - d_uf[d_idx]
        d_vg[d_idx] = 2 * d_v[d_idx] - d_vf[d_idx]
        d_wg[d_idx] = 2 * d_w[d_idx] - d_wf[d_idx]


class PengwanRigidFluidForce(Equation):
    def __init__(self, dest, sources, rho0, dim):
        self.rho0 = rho0
        self.dim = dim

        super(PengwanRigidFluidForce, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, d_p, d_V, s_V, s_p, s_rho, s_u, s_v,
             s_w, d_ug, d_vg, d_wg, d_fx, d_fy, d_fz, DWIJ, WIJ, XIJ, HIJ,
             R2IJ):
        p_ij = (s_rho[s_idx] * d_p[d_idx] + d_rho[d_idx] * s_p[s_idx]) / (
            d_rho[d_idx] * s_rho[s_idx])

        tmp1 = d_V[d_idx]**2. + d_V[d_idx]**2.

        uij = s_u[d_idx] - d_ug[s_idx]
        vij = s_v[d_idx] - d_vg[s_idx]
        wij = s_w[d_idx] - d_wg[s_idx]
        uijdotxij = XIJ[0] * uij + XIJ[1] * vij + XIJ[2] * wij
        tmp2 = R2IJ + (0.01 * HIJ)**2.
        # artificial viscosity coefficient
        pi_ij = uijdotxij / tmp2

        tmp3 = (2. * (self.dim + 2) * self.nu * self.rho0 * pi_ij * d_V[d_idx]
                * s_V[s_idx])

        tmp = -tmp1 * p_ij + tmp3

        # this equation is different from the one given in the paper.
        # we swapped the indices of fluids to rigid body, so that
        # there will be no negative and to avoid data races.
        d_fx[d_idx] += tmp * DWIJ[0]
        d_fy[d_idx] += tmp * DWIJ[1]
        d_fz[d_idx] += tmp * DWIJ[2]


class RK2PengwanFluidStep(IntegratorStep):
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
        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_w[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av, d_aw, d_arho, dt):

        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]
