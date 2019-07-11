from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep
from compyle.api import declare
from pysph.sph.wc.linalg import mat_vec_mult, mat_mult


def get_particle_array_gray(constants=None, **props):
    extra_props = [
        'au', 'av', 'aw', 'V', 'fx', 'fy', 'fz', 'x0', 'y0', 'z0',
        'tang_disp_x', 'tang_disp_y', 'tang_disp_z', 'tang_disp_x0',
        'tang_disp_y0', 'tang_disp_z0', 'tang_velocity_x', 'tang_velocity_y',
        'rad_s', 'tang_velocity_z', 'nx', 'ny', 'nz'
    ]

    # consts = {'total_mass': numpy.zeros(nb, dtype=float),
    #           'cm': numpy.zeros(3*nb, dtype=float),
    #           }
    consts = {}
    if constants:
        consts.update(constants)
    pa = get_particle_array(constants=consts, additional_props=extra_props,
                            **props)
    pa.add_property('sigma', stride=9)
    # deviatoric stress
    pa.add_property('ds', stride=9)
    # acceleration of deviatoric stress
    pa.add_property('ads', stride=9)
    # strain rate tensor
    pa.add_property('epsilon', stride=9)
    # rotation tensor
    pa.add_property('omega', stride=9)
    # velocity gradient
    # where v goes from u, v, w and k goes from x, y, z
    pa.add_property('dvdk', stride=9)

    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au', 'av',
        'aw', 'tag', 'gid', 'V', 'fx', 'fy', 'fz', 'body_id'
    ])
    return pa


class VelocityGradient(Equation):
    def initialize(self, d_idx, d_dvdk):
        i, idx9 = declare('int', 2)

        for i in range(9):
            d_dvdk[idx9 + i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_dvdk, DWIJ, VIJ):
        i, j, idx9 = declare('int', 3)

        tmp = s_m[s_idx] / s_rho[s_idx]

        for i in range(3):
            for j in range(3):
                d_dvdk[idx9 + 3 * i + j] += tmp * -VIJ[i] * DWIJ[j]


class HookesDeviatoricStressRate(Equation):
    def _get_helpers_(self):
        return [mat_vec_mult, mat_mult]

    def initialize(self, d_idx, d_ds, d_ads, d_dvdk, d_omega, d_epsilon, d_G):
        i, j, idx9 = declare('int', 3)
        epsilon, omega, omegaT, ds, dvdk = declare('matrix(9)', 5)

        for i in range(9):
            ds[i] = d_ds[d_idx * 9 + i]
            dvdk[i] = d_dvdk[d_idx * 9 + i]
            d_ads[d_idx * 9 + i] = 0.0

        eps_trace = 0.0
        # strain rate tensor is symmetric
        for i in range(3):
            for j in range(3):
                epsilon[3 * i + j] = (
                    0.5 * (dvdk[3 * i + j] + dvdk[i + 3 * j]))
                omega[3 * i + j] = (0.5 * (dvdk[3 * i + j] - dvdk[i + 3 * j]))
                if i == j:
                    eps_trace += d_epsilon[3 * i + j]

        for i in range(3):
            for j in range(3):
                omegaT[3 * j + i] = d_omega[3 * i + j]

        smo, oms = declare('matrix(9)', 2)
        mat_mult(ds, omegaT, 3, smo)
        mat_mult(omega, ds, 3, oms)
        for i in range(3):
            for j in range(3):
                ind = 3 * i + j
                d_ads[idx9 + ind] = (
                    2 * d_G[0] * epsilon[ind] + smo[ind] + oms[ind])
                if i == j:
                    d_ads[d_idx * 9 + ind] += -2 * d_G[0] * eps_trace / 3.0


class GraySolidMechStep(IntegratorStep):
    """Predictor corrector Integrator for solid mechanics problems"""

    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0,
                   d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_ds0, d_ds, d_e0, d_e):
        i, idx9 = declare('int', 2)
        idx9 = 9 * d_idx
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]
        d_e0[d_idx] = d_e[d_idx]

        for i in range(9):
            d_ds0[idx9 + i] = d_ds[idx9 + i]

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av, d_aw, d_ax, d_ay,
               d_az, d_arho, d_ds, d_ds0, d_ads, d_e, d_e0, d_ae, dt):
        i, idx9 = declare('int', 2)
        idx9 = 9 * d_idx
        dtb2 = 0.5 * dt
        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_ax[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_ay[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_az[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]
        d_e[d_idx] = d_e0[d_idx] + dtb2 * d_ae[d_idx]

        # update deviatoric stress components
        for i in range(9):
            d_ds[idx9 + i] = d_ds0[idx9 + i] + dtb2 * d_ads[idx9 + i]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av, d_aw, d_ax, d_ay,
               d_az, d_arho, d_e, d_ae, d_e0, d_ds, d_ds0, d_ads, dt):
        i, idx9 = declare('int', 2)
        idx9 = 9 * d_idx
        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_ax[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_ay[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_az[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]
        d_e[d_idx] = d_e0[d_idx] + dt * d_ae[d_idx]

        # update deviatoric stress components
        # update deviatoric stress components
        for i in range(9):
            d_ds[idx9 + i] = d_ds0[idx9 + i] + dt * d_ads[idx9 + i]
