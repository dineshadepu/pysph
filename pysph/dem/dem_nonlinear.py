from pysph.base.reduce_array import parallel_reduce_array
from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
import numpy as np
import numpy
from compyle.api import declare
from math import sqrt, asin, sin, cos, pi, log


def get_particle_array_dem(constants=None, **props):
    """Return a particle array for a dem particles
    """
    dim = props.pop('dim', None)

    dem_props = [
        'wx', 'wy', 'wz', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'rad_s',
        'm_inv', 'I_inv', 'u0', 'v0', 'w0', 'wx0', 'wy0', 'wz0', 'x0', 'y0',
        'z0'
    ]

    dem_id = props.pop('dem_id', None)

    pa = get_particle_array(additional_props=dem_props, **props)

    pa.add_property('dem_id', type='int', data=dem_id)
    # create the array to save the tangential interaction particles
    # index and other variables
    if dim == 3:
        limit = 30
    elif dim == 2 or dim is None:
        limit = 6

    pa.add_constant('limit', limit)
    pa.add_constant('tng_idx', [-1] * limit * len(pa.x))
    pa.add_constant('tng_idx_dem_id', [-1] * limit * len(pa.x))
    pa.add_constant('tng_x', [0.] * limit * len(pa.x))
    pa.add_constant('tng_y', [0.] * limit * len(pa.x))
    pa.add_constant('tng_z', [0.] * limit * len(pa.x))
    pa.add_constant('tng_x0', [0.] * limit * len(pa.x))
    pa.add_constant('tng_y0', [0.] * limit * len(pa.x))
    pa.add_constant('tng_z0', [0.] * limit * len(pa.x))
    pa.add_constant('tng_nx', [0.] * limit * len(pa.x))
    pa.add_constant('tng_ny', [0.] * limit * len(pa.x))
    pa.add_constant('tng_nz', [0.] * limit * len(pa.x))
    pa.add_constant('tng_nx0', [0.] * limit * len(pa.x))
    pa.add_constant('tng_ny0', [0.] * limit * len(pa.x))
    pa.add_constant('tng_nz0', [0.] * limit * len(pa.x))
    pa.add_constant('vtx', [0.] * limit * len(pa.x))
    pa.add_constant('vty', [0.] * limit * len(pa.x))
    pa.add_constant('vtz', [0.] * limit * len(pa.x))
    pa.add_constant('total_tng_contacts', [0] * len(pa.x))

    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'wx', 'wy', 'wz', 'm', 'p', 'pid', 'tag',
        'gid', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'I_inv'
    ])

    return pa


class ResetForces(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz,
                   d_torx, d_tory, d_torz):
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_fz[d_idx] = 0.
        d_torx[d_idx] = 0.
        d_tory[d_idx] = 0.
        d_torz[d_idx] = 0.


class TsuijiNonLinearParticleParticleForceStage1(Equation):
    """Force between two spheres is implemented using Nonlinear DEM contact force
    law.

    The force is modelled from reference [1].

    [1] Lagrangian numerical simulation of plug flow of cohesion less particles
    in a horizontal pipe Ye.
    """

    def __init__(self, dest, sources, en=0.1, mu=0.1):
        """the required coefficients for force calculation.


        Keyword arguments:
        kn -- Normal spring stiffness (default 1e3)
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)

        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.

        """
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        super(TsuijiNonLinearParticleParticleForceStage1, self).__init__(
            dest, sources)

    def loop(self, d_idx, d_m, d_fx, d_fy, d_fz, d_tng_x, d_tng_y, d_tng_z,
             d_tng_x0, d_tng_y0, d_tng_z0, d_tng_idx, d_tng_idx_dem_id,
             d_total_tng_contacts, d_dem_id, d_limit, d_vtx, d_vty, d_vtz,
             d_tng_nx, d_tng_ny, d_tng_nz, d_tng_nx0, d_tng_ny0, d_tng_nz0,
             d_wx, d_wy, d_wz, d_yng_m, d_poissons_ratio, d_shear_m, d_torx,
             d_tory, d_torz, VIJ, XIJ, RIJ, d_rad_s, s_idx, s_m, s_rad_s,
             s_dem_id, s_wx, s_wy, s_wz, s_yng_m, s_poissons_ratio, s_shear_m,
             dt):
        overlap = -1.
        # check the particles are not on top of each other.
        if RIJ > 0:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        # ---------- force computation starts ------------
        # if particles are overlapping
        if overlap > 0:
            # equation 2.8
            # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
            rinv = 1.0 / RIJ
            # in PySPH: XIJ[0] = d_x[d_idx] - s_x[s_idx]
            nxc = -XIJ[0] * rinv
            nyc = -XIJ[1] * rinv
            nzc = -XIJ[2] * rinv

            # ---- Relative velocity computation (Eq 2.9) ----
            # relative velocity of particle d_idx w.r.t particle s_idx at
            # contact point. The velocity difference provided by PySPH is
            # only between translational velocities, but we need to
            # consider rotational velocities also.
            # Distance till contact point
            a_i = d_rad_s[d_idx] - overlap / 2.
            a_j = s_rad_s[s_idx] - overlap / 2.
            # TODO: This has to be replaced by a custom cross product
            # function
            # wij = a_i * w_i + a_j * w_j
            wijx = a_i * d_wx[d_idx] + a_j * s_wx[s_idx]
            wijy = a_i * d_wy[d_idx] + a_j * s_wy[s_idx]
            wijz = a_i * d_wz[d_idx] + a_j * s_wz[s_idx]
            # wij \cross nij
            wcn_x = wijy * nzc - wijz * nyc
            wcn_y = wijz * nxc - wijx * nzc
            wcn_z = wijx * nyc - wijy * nxc

            vr_x = VIJ[0] + wcn_x
            vr_y = VIJ[1] + wcn_y
            vr_z = VIJ[2] + wcn_z

            # normal velocity magnitude
            vr_dot_nij = vr_x * nxc + vr_y * nyc + vr_z * nzc
            vn_x = vr_dot_nij * nxc
            vn_y = vr_dot_nij * nyc
            vn_z = vr_dot_nij * nzc

            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y
            vt_z = vr_z - vn_z
            # magnitude of the tangential velocity
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

            # compute the spring stiffness (nonlinear model)
            Ed = d_yng_m[d_idx]
            Es = s_yng_m[s_idx]
            Gd = d_shear_m[d_idx]
            Gs = s_shear_m[s_idx]
            pd = d_poissons_ratio[d_idx]
            ps = s_poissons_ratio[s_idx]
            rd = d_rad_s[d_idx]
            rs = s_rad_s[s_idx]

            E_eff = 4. / 3. * (Ed * Es) / (Ed * (1. - ps**2.) + Es *
                                           (1. - pd**2.))
            G_eff = 16. / 3. * (Gd * Gs) / (Gd * (2. - ps) + Gs * (2. - pd))
            r_eff = (rd * rs) / (rd + rs)

            kn = 4. / 3. * E_eff * sqrt(r_eff)
            kt = 16. / 3. * G_eff * sqrt(r_eff)
            kt_1 = 1. / kt

            # compute the damping constants
            m_eff = d_m[d_idx] * s_m[s_idx] / (d_m[d_idx] + s_m[s_idx])
            log_en = log(self.en)
            eta_n = -2. * log_en * sqrt(
                m_eff * kn) / sqrt(pi**2. + log_en**2.)
            eta_n = 0.

            # normal force
            kn_overlap = kn * overlap**(1.5)
            fn_x = -kn_overlap * nxc - eta_n * vn_x
            fn_y = -kn_overlap * nyc - eta_n * vn_y
            fn_z = -kn_overlap * nzc - eta_n * vn_z

            d_fx[d_idx] += fn_x
            d_fy[d_idx] += fn_y
            d_fz[d_idx] += fn_z


class TsuijiNonLinearParticleParticleForceStage2(Equation):
    """Force between two spheres is implemented using Nonlinear DEM contact force
    law.

    The force is modelled from reference [1].

    [1] Lagrangian numerical simulation of plug flow of cohesion less particles
    in a horizontal pipe Ye.
    """

    def __init__(self, dest, sources, en=0.1, mu=0.1):
        """the required coefficients for force calculation.


        Keyword arguments:
        kn -- Normal spring stiffness (default 1e3)
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)

        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.

        """
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        super(TsuijiNonLinearParticleParticleForceStage2, self).__init__(
            dest, sources)

    def loop(self, d_idx, d_m, d_fx, d_fy, d_fz, d_tng_x, d_tng_y, d_tng_z,
             d_tng_x0, d_tng_y0, d_tng_z0, d_tng_idx, d_tng_idx_dem_id,
             d_total_tng_contacts, d_dem_id, d_limit, d_vtx, d_vty, d_vtz,
             d_tng_nx, d_tng_ny, d_tng_nz, d_tng_nx0, d_tng_ny0, d_tng_nz0,
             d_wx, d_wy, d_wz, d_yng_m, d_poissons_ratio, d_shear_m, d_torx,
             d_tory, d_torz, VIJ, XIJ, RIJ, d_rad_s, s_idx, s_m, s_rad_s,
             s_dem_id, s_wx, s_wy, s_wz, s_yng_m, s_poissons_ratio, s_shear_m,
             dt):
        overlap = -1.
        # check the particles are not on top of each other.
        if RIJ > 0:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        # ---------- force computation starts ------------
        # if particles are overlapping
        if overlap > 0:
            # equation 2.8
            # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
            rinv = 1.0 / RIJ
            # in PySPH: XIJ[0] = d_x[d_idx] - s_x[s_idx]
            nxc = -XIJ[0] * rinv
            nyc = -XIJ[1] * rinv
            nzc = -XIJ[2] * rinv

            # ---- Relative velocity computation (Eq 2.9) ----
            # relative velocity of particle d_idx w.r.t particle s_idx at
            # contact point. The velocity difference provided by PySPH is
            # only between translational velocities, but we need to
            # consider rotational velocities also.
            # Distance till contact point
            a_i = d_rad_s[d_idx] - overlap / 2.
            a_j = s_rad_s[s_idx] - overlap / 2.
            # TODO: This has to be replaced by a custom cross product
            # function
            # wij = a_i * w_i + a_j * w_j
            wijx = a_i * d_wx[d_idx] + a_j * s_wx[s_idx]
            wijy = a_i * d_wy[d_idx] + a_j * s_wy[s_idx]
            wijz = a_i * d_wz[d_idx] + a_j * s_wz[s_idx]
            # wij \cross nij
            wcn_x = wijy * nzc - wijz * nyc
            wcn_y = wijz * nxc - wijx * nzc
            wcn_z = wijx * nyc - wijy * nxc

            vr_x = VIJ[0] + wcn_x
            vr_y = VIJ[1] + wcn_y
            vr_z = VIJ[2] + wcn_z

            # normal velocity magnitude
            vr_dot_nij = vr_x * nxc + vr_y * nyc + vr_z * nzc
            vn_x = vr_dot_nij * nxc
            vn_y = vr_dot_nij * nyc
            vn_z = vr_dot_nij * nzc

            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y
            vt_z = vr_z - vn_z
            # magnitude of the tangential velocity
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

            # compute the spring stiffness (nonlinear model)
            Ed = d_yng_m[d_idx]
            Es = s_yng_m[s_idx]
            Gd = d_shear_m[d_idx]
            Gs = s_shear_m[s_idx]
            pd = d_poissons_ratio[d_idx]
            ps = s_poissons_ratio[s_idx]
            rd = d_rad_s[d_idx]
            rs = s_rad_s[s_idx]

            E_eff = 4. / 3. * (Ed * Es) / (Ed * (1. - ps**2.) + Es *
                                           (1. - pd**2.))
            G_eff = 16. / 3. * (Gd * Gs) / (Gd * (2. - ps) + Gs * (2. - pd))
            r_eff = (rd * rs) / (rd + rs)

            kn = 4. / 3. * E_eff * sqrt(r_eff)
            kt = 16. / 3. * G_eff * sqrt(r_eff)
            kt_1 = 1. / kt

            # compute the damping constants
            m_eff = d_m[d_idx] * s_m[s_idx] / (d_m[d_idx] + s_m[s_idx])
            log_en = log(self.en)
            eta_n = -2. * log_en * sqrt(
                m_eff * kn) / sqrt(pi**2. + log_en**2.)
            eta_n = 0.

            # normal force
            kn_overlap = kn * overlap**(1.5)
            fn_x = -kn_overlap * nxc - eta_n * vn_x
            fn_y = -kn_overlap * nyc - eta_n * vn_y
            fn_z = -kn_overlap * nzc - eta_n * vn_z

            d_fx[d_idx] += fn_x
            d_fy[d_idx] += fn_y
            d_fz[d_idx] += fn_z


class RK2StepNonLinearDEM(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v, d_w,
                   d_u0, d_v0, d_w0, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz,
                   d_wx, d_wy, d_wz, d_wx0, d_wy0, d_wz0, d_m_inv, d_I_inv,
                   d_total_tng_contacts, d_limit, d_tng_x, d_tng_y, d_tng_z,
                   d_tng_x0, d_tng_y0, d_tng_z0, d_tng_nx, d_tng_ny, d_tng_nz,
                   d_tng_nx0, d_tng_ny0, d_tng_nz0, dt):

        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_wx0[d_idx] = d_wx[d_idx]
        d_wy0[d_idx] = d_wy[d_idx]
        d_wz0[d_idx] = d_wz[d_idx]

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_x0, d_y0, d_z0, d_u0, d_v0, d_w0, d_wx0, d_wy0, d_wz0, d_torx,
               d_tory, d_torz, d_wx, d_wy, d_wz, d_m_inv, d_I_inv,
               d_total_tng_contacts, d_limit, d_tng_x,
               d_tng_y, d_tng_z, d_tng_x0, d_tng_y0, d_tng_z0, d_vtx, d_vty,
               d_vtz, dt):
        dtb2 = dt / 2.

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_fx[d_idx] * d_m_inv[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_fy[d_idx] * d_m_inv[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_fz[d_idx] * d_m_inv[d_idx]

        d_wx[d_idx] = d_wx0[d_idx] + dtb2 * d_torx[d_idx] * d_I_inv[d_idx]
        d_wy[d_idx] = d_wy0[d_idx] + dtb2 * d_tory[d_idx] * d_I_inv[d_idx]
        d_wz[d_idx] = d_wz0[d_idx] + dtb2 * d_torz[d_idx] * d_I_inv[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_x0, d_y0, d_z0, d_u0, d_v0, d_w0, d_wx0, d_wy0, d_wz0, d_torx,
               d_tory, d_torz, d_wx, d_wy, d_wz, d_m_inv, d_I_inv,
               d_total_tng_contacts, d_limit, d_tng_x,
               d_tng_y, d_tng_z, d_tng_x0, d_tng_y0, d_tng_z0, d_vtx, d_vty,
               d_vtz, dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt * d_fx[d_idx] * d_m_inv[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_fy[d_idx] * d_m_inv[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_fz[d_idx] * d_m_inv[d_idx]

        d_wx[d_idx] = d_wx0[d_idx] + dt * d_torx[d_idx] * d_I_inv[d_idx]
        d_wy[d_idx] = d_wy0[d_idx] + dt * d_tory[d_idx] * d_I_inv[d_idx]
        d_wz[d_idx] = d_wz0[d_idx] + dt * d_torz[d_idx] * d_I_inv[d_idx]
