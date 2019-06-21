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

    pa = get_particle_array(additional_props=dem_props, constants=constants,
                            **props)

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
    def initialize(self, d_idx, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz):
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
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)

        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.

        """
        self.en = en
        # a constant for used for damping
        log_en = log(self.en)
        self.alpha_1 = -log_en * sqrt(5. / (log_en**2. + pi**2.))
        self.et = 1.0
        log_et = log(self.et)
        self.alpha_2 = -log_et * sqrt(5. / (log_et**2. + pi**2.))
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
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        # total number of contacts of particle i in destination
        tot_ctcs = d_total_tng_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        # check if the particle is in the tracking list
        # if so, then save the location at found_at
        found = 0
        for j in range(p, q1):
            if s_idx == d_tng_idx[j]:
                if s_dem_id[s_idx] == d_tng_idx_dem_id[j]:
                    found_at = j
                    found = 1
                    break
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

            E_eff = (Ed * Es) / (Ed * (1. - ps**2.) + Es * (1. - pd**2.))
            G_eff = (Gd * Gs) / (Gd * (2. - ps) + Gs * (2. - pd))
            r_eff = (rd * rs) / (rd + rs)

            kn = 4. / 3. * E_eff * sqrt(r_eff)
            kt = 16. / 3. * G_eff * sqrt(r_eff)
            kt_1 = 1. / kt

            # damping force is taken from
            # "On the Determination of the Damping Coefficient
            # of Non-linear Spring-dashpot System to Model
            # Hertz Contact for Simulation by Discrete Element
            # Method" paper.
            # compute the damping constants
            m_eff = d_m[d_idx]
            eta_n = self.alpha_1 * sqrt(m_eff * kn) * overlap**0.25

            # normal force
            kn_overlap = kn * overlap**(1.5)
            fn_x = -kn_overlap * nxc - eta_n * vn_x
            fn_y = -kn_overlap * nyc - eta_n * vn_y
            fn_z = -kn_overlap * nzc - eta_n * vn_z

            # ------------- tangential force computation ----------------
            # if the particle is not been tracked then assign an index in
            # tracking history.
            if found == 0:
                found_at = q1
                d_tng_idx[found_at] = s_idx
                d_total_tng_contacts[d_idx] += 1
                d_tng_idx_dem_id[found_at] = s_dem_id[s_idx]
                d_tng_nx[found_at] = nxc
                d_tng_ny[found_at] = nyc
                d_tng_nz[found_at] = nzc
                d_tng_nx0[found_at] = nxc
                d_tng_ny0[found_at] = nyc
                d_tng_nz0[found_at] = nzc

            # compute and set the tangential acceleration for the
            # current time step
            d_vtx[found_at] = vt_x
            d_vty[found_at] = vt_y
            d_vtz[found_at] = vt_z

            # compute the damping constants
            m_eff_t = 2. / 7. * m_eff
            eta_t = self.alpha_2 * sqrt(m_eff_t * kt)

            # find the tangential force from the tangential displacement
            # and tangential velocity (eq 2.11 Thesis Ye)
            ft0_x = -kt * d_tng_x[found_at] - eta_t * vt_x
            ft0_y = -kt * d_tng_y[found_at] - eta_t * vt_y
            ft0_z = -kt * d_tng_z[found_at] - eta_t * vt_z

            # (*) check against Coulomb criterion
            # Tangential force magnitude due to displacement
            ft0_magn = (ft0_x * ft0_x + ft0_y * ft0_y + ft0_z * ft0_z)**(
                0.5)
            fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

            # we have to compare with static friction, so
            # this mu has to be static friction coefficient
            fn_mu = self.mu * fn_magn

            # if the tangential force magnitude is zero, then do nothing,
            # else do following
            if ft0_magn != 0.:
                # compare tangential force with the static friction
                if ft0_magn >= fn_mu:
                    # rescale the tangential displacement
                    # find the unit direction in tangential velocity
                    # TODO: ELIMINATE THE SINGULARITY CASE
                    tx = ft0_x / ft0_magn
                    ty = ft0_y / ft0_magn
                    tz = ft0_z / ft0_magn
                    # this taken from Luding paper [2], eq (21)
                    d_tng_x[found_at] = -kt_1 * (fn_mu * tx + eta_t * vt_x)
                    d_tng_y[found_at] = -kt_1 * (fn_mu * ty + eta_t * vt_y)
                    d_tng_z[found_at] = -kt_1 * (fn_mu * tz + eta_t * vt_z)

                    # and also adjust the spring elongation
                    # at time t, which is used at stage 2 integrator
                    d_tng_x0[found_at] = d_tng_x[found_at]
                    d_tng_y0[found_at] = d_tng_y[found_at]
                    d_tng_z0[found_at] = d_tng_z[found_at]

                    # set the tangential force to static friction
                    # from Coulomb criterion
                    ft0_x = fn_mu * tx
                    ft0_y = fn_mu * ty
                    ft0_z = fn_mu * tz

            d_fx[d_idx] += fn_x + ft0_x
            d_fy[d_idx] += fn_y + ft0_y
            d_fz[d_idx] += fn_z + ft0_z

            # torque = n cross F
            d_torx[d_idx] += (nyc * ft0_z - nzc * ft0_y) * a_i
            d_tory[d_idx] += (nzc * ft0_x - nxc * ft0_z) * a_i
            d_torz[d_idx] += (nxc * ft0_y - nyc * ft0_x) * a_i


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
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)

        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.

        """
        self.en = en
        # a constant for used for damping
        log_en = log(self.en)
        self.alpha_1 = -log_en * sqrt(5. / (log_en**2. + pi**2.))
        self.et = 1.0
        log_et = log(self.et)
        self.alpha_2 = -log_et * sqrt(5. / (log_et**2. + pi**2.))
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
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        # total number of contacts of particle i in destination
        tot_ctcs = d_total_tng_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        # check if the particle is in the tracking list
        # if so, then save the location at found_at
        found = 0
        for j in range(p, q1):
            if s_idx == d_tng_idx[j]:
                if s_dem_id[s_idx] == d_tng_idx_dem_id[j]:
                    found_at = j
                    found = 1
                    break
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

            E_eff = (Ed * Es) / (Ed * (1. - ps**2.) + Es * (1. - pd**2.))
            G_eff = (Gd * Gs) / (Gd * (2. - ps) + Gs * (2. - pd))
            r_eff = (rd * rs) / (rd + rs)

            kn = 4. / 3. * E_eff * sqrt(r_eff)
            kt = 16. / 3. * G_eff * sqrt(r_eff)
            kt_1 = 1. / kt

            # damping force is taken from
            # "On the Determination of the Damping Coefficient
            # of Non-linear Spring-dashpot System to Model
            # Hertz Contact for Simulation by Discrete Element
            # Method" paper.
            # compute the damping constants
            m_eff = d_m[d_idx]
            eta_n = self.alpha_1 * sqrt(m_eff * kn) * overlap**0.25

            # normal force
            kn_overlap = kn * overlap**(1.5)
            fn_x = -kn_overlap * nxc - eta_n * vn_x
            fn_y = -kn_overlap * nyc - eta_n * vn_y
            fn_z = -kn_overlap * nzc - eta_n * vn_z

            # ------------- tangential force computation ----------------
            # if the particle is not been tracked then assign an index in
            # tracking history.
            if found == 0:
                found_at = q1
                d_tng_idx[found_at] = s_idx
                d_total_tng_contacts[d_idx] += 1
                d_tng_idx_dem_id[found_at] = s_dem_id[s_idx]
                d_tng_nx[found_at] = nxc
                d_tng_ny[found_at] = nyc
                d_tng_nz[found_at] = nzc
                d_tng_nx0[found_at] = nxc
                d_tng_ny0[found_at] = nyc
                d_tng_nz0[found_at] = nzc

            # compute and set the tangential acceleration for the
            # current time step
            d_vtx[found_at] = vt_x
            d_vty[found_at] = vt_y
            d_vtz[found_at] = vt_z

            # compute the damping constants
            m_eff_t = 2. / 7. * m_eff
            eta_t = self.alpha_2 * sqrt(m_eff_t * kt)

            # find the tangential force from the tangential displacement
            # and tangential velocity (eq 2.11 Thesis Ye)
            ft0_x = -kt * d_tng_x[found_at] - eta_t * vt_x
            ft0_y = -kt * d_tng_y[found_at] - eta_t * vt_y
            ft0_z = -kt * d_tng_z[found_at] - eta_t * vt_z

            # (*) check against Coulomb criterion
            # Tangential force magnitude due to displacement
            ft0_magn = (ft0_x * ft0_x + ft0_y * ft0_y + ft0_z * ft0_z)**(
                0.5)
            fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

            # we have to compare with static friction, so
            # this mu has to be static friction coefficient
            fn_mu = self.mu * fn_magn

            # if the tangential force magnitude is zero, then do nothing,
            # else do following
            if ft0_magn != 0.:
                # compare tangential force with the static friction
                if ft0_magn >= fn_mu:
                    # rescale the tangential displacement
                    # find the unit direction in tangential velocity
                    # TODO: ELIMINATE THE SINGULARITY CASE
                    tx = ft0_x / ft0_magn
                    ty = ft0_y / ft0_magn
                    tz = ft0_z / ft0_magn
                    # this taken from Luding paper [2], eq (21)
                    d_tng_x[found_at] = -kt_1 * (fn_mu * tx + eta_t * vt_x)
                    d_tng_y[found_at] = -kt_1 * (fn_mu * ty + eta_t * vt_y)
                    d_tng_z[found_at] = -kt_1 * (fn_mu * tz + eta_t * vt_z)

                    # and also adjust the spring elongation
                    # at time t, which is used at stage 2 integrator
                    d_tng_x0[found_at] = d_tng_x[found_at]
                    d_tng_y0[found_at] = d_tng_y[found_at]
                    d_tng_z0[found_at] = d_tng_z[found_at]

                    # set the tangential force to static friction
                    # from Coulomb criterion
                    ft0_x = fn_mu * tx
                    ft0_y = fn_mu * ty
                    ft0_z = fn_mu * tz

            d_fx[d_idx] += fn_x + ft0_x
            d_fy[d_idx] += fn_y + ft0_y
            d_fz[d_idx] += fn_z + ft0_z

            # torque = n cross F
            d_torx[d_idx] += (nyc * ft0_z - nzc * ft0_y) * a_i
            d_tory[d_idx] += (nzc * ft0_x - nxc * ft0_z) * a_i
            d_torz[d_idx] += (nxc * ft0_y - nyc * ft0_x) * a_i


class TsuijiNonLinearParticleWallForceStage1(Equation):
    """Force between two spheres is implemented using Nonlinear DEM contact force
    law.

    The force is modelled from reference [1], [2]. Damping force is taken from
    [2], mainly for constants of damping.

    [1] Lagrangian numerical simulation of plug flow of cohesion less particles
    in a horizontal pipe Ye.

    [2] On the Determination of the Damping Coefficient of Non-linear
    Spring-dashpot System to Model Hertz Contact for Simulation by Discrete
    Element Method

    """

    def __init__(self, dest, sources, mu=0.5, en=0.8):
        """the required coefficients for force calculation.


        Keyword arguments:
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)

        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.

        """
        self.en = en
        # a constant for used for damping
        log_en = log(self.en)
        self.alpha_1 = -log_en * sqrt(5. / (log_en**2. + pi**2.))
        self.et = 1.0
        log_et = log(self.et)
        self.alpha_2 = -log_et * sqrt(5. / (log_et**2. + pi**2.))
        self.mu = mu
        super(TsuijiNonLinearParticleWallForceStage1, self).__init__(
            dest, sources)

    def initialize_pair(
            self, d_idx, d_m, d_u, d_v, d_w, d_x, d_y, d_z, d_fx, d_fy, d_fz,
            d_tng_x, d_tng_y, d_tng_z, d_tng_x0, d_tng_y0, d_tng_z0, d_tng_idx,
            d_tng_idx_dem_id, d_total_tng_contacts, d_dem_id, d_limit, d_vtx,
            d_vty, d_vtz, d_tng_nx, d_tng_ny, d_tng_nz, d_tng_nx0, d_tng_ny0,
            d_tng_nz0, d_wx, d_wy, d_wz, d_yng_m, d_poissons_ratio, d_shear_m,
            d_torx, d_tory, d_torz, d_rad_s, s_idx, s_x, s_y, s_z, s_nx, s_ny,
            s_nz, s_poissons_ratio, s_yng_m, s_shear_m, s_dem_id, s_np):
        i = declare('int')
        xij = declare('matrix(3)')
        vij = declare('matrix(3)')
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        for i in range(s_np[0]):
            # total number of contacts of particle i in destination
            tot_ctcs = d_total_tng_contacts[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_limit[0]
            # ending index is q -1
            q1 = p + tot_ctcs

            # check if the particle is in the tracking list
            # if so, then save the location at found_at
            found = 0
            for j in range(p, q1):
                if i == d_tng_idx[j]:
                    if s_dem_id[i] == d_tng_idx_dem_id[j]:
                        found_at = j
                        found = 1
                        break

            # Force calculation starts
            overlap = -1.
            xij[0] = d_x[d_idx] - s_x[i]
            xij[1] = d_y[d_idx] - s_y[i]
            xij[2] = d_z[d_idx] - s_z[i]
            overlap = d_rad_s[d_idx] - (
                xij[0] * s_nx[i] + xij[1] * s_ny[i] + xij[2] * s_nz[i])

            if overlap > 0:
                # basic variables: normal vector
                nxc = -s_nx[i]
                nyc = -s_ny[i]
                nzc = -s_nz[i]

                # ---- Relative velocity computation (Eq 2.9) ----
                # relative velocity of particle d_idx w.r.t particle i at
                # contact point.
                # Distance till contact point
                a_d = d_rad_s[d_idx] - overlap
                # TODO: This has to be replaced by a custom cross product
                # function
                # wij = a_i * w_i + a_j * w_j
                # since w_j is wall, and angular velocity is zero
                # wij = a_i * w_i
                wijx = a_d * d_wx[d_idx]
                wijy = a_d * d_wy[d_idx]
                wijz = a_d * d_wz[d_idx]
                # wij \cross nij
                wcn_x = wijy * nzc - wijz * nyc
                wcn_y = wijz * nxc - wijx * nzc
                wcn_z = wijx * nyc - wijy * nxc

                vij[0] = d_u[d_idx]
                vij[1] = d_v[d_idx]
                vij[2] = d_w[d_idx]
                vr_x = vij[0] + wcn_x
                vr_y = vij[1] + wcn_y
                vr_z = vij[2] + wcn_z

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
                Es = s_yng_m[i]
                Gd = d_shear_m[d_idx]
                Gs = s_shear_m[i]
                pd = d_poissons_ratio[d_idx]
                ps = s_poissons_ratio[i]
                rd = d_rad_s[d_idx]

                E_eff = Ed / (1. - pd**2.)
                G_eff = Gd / (2. - pd)
                # tsuiji paper gave
                # E_eff = Ed * Es / (Ed * (1. - ps**2.) + Es * (1. - pd**2.))
                # G_eff = Gd * Gs / (Gd * (2. - ps) + Gs * (2. - pd))
                r_eff = rd

                kn = 4. / 3. * E_eff * sqrt(r_eff)
                kt = 16. / 3. * G_eff * sqrt(r_eff * overlap)
                kt_1 = 1. / kt

                # damping force is taken from
                # "On the Determination of the Damping Coefficient
                # of Non-linear Spring-dashpot System to Model
                # Hertz Contact for Simulation by Discrete Element
                # Method" paper.
                # compute the damping constants
                m_eff = d_m[d_idx]
                eta_n = self.alpha_1 * sqrt(m_eff * kn) * overlap**0.25

                # normal force
                kn_overlap = kn * overlap**(1.5)
                fn_x = -kn_overlap * nxc - eta_n * vn_x
                fn_y = -kn_overlap * nyc - eta_n * vn_y
                fn_z = -kn_overlap * nzc - eta_n * vn_z

                # ------------- tangential force computation ----------------
                # if the particle is not been tracked then assign an index in
                # tracking history.
                if found == 0:
                    found_at = q1
                    d_tng_idx[found_at] = i
                    d_total_tng_contacts[d_idx] += 1
                    d_tng_idx_dem_id[found_at] = s_dem_id[i]
                    d_tng_nx[found_at] = nxc
                    d_tng_ny[found_at] = nyc
                    d_tng_nz[found_at] = nzc
                    d_tng_nx0[found_at] = nxc
                    d_tng_ny0[found_at] = nyc
                    d_tng_nz0[found_at] = nzc

                # compute and set the tangential acceleration for the
                # current time step
                d_vtx[found_at] = vt_x
                d_vty[found_at] = vt_y
                d_vtz[found_at] = vt_z

                # compute the damping constants
                m_eff_t = 2. / 7. * m_eff
                eta_t = self.alpha_2 * sqrt(m_eff_t * kt)

                # find the tangential force from the tangential displacement
                # and tangential velocity (eq 2.11 Thesis Ye)
                ft0_x = -kt * d_tng_x[found_at] - eta_t * vt_x
                ft0_y = -kt * d_tng_y[found_at] - eta_t * vt_y
                ft0_z = -kt * d_tng_z[found_at] - eta_t * vt_z

                # (*) check against Coulomb criterion
                # Tangential force magnitude due to displacement
                ft0_magn = (ft0_x * ft0_x + ft0_y * ft0_y + ft0_z * ft0_z)**(
                    0.5)
                fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

                # we have to compare with static friction, so
                # this mu has to be static friction coefficient
                fn_mu = self.mu * fn_magn

                # if the tangential force magnitude is zero, then do nothing,
                # else do following
                if ft0_magn != 0.:
                    # compare tangential force with the static friction
                    if ft0_magn >= fn_mu:
                        # rescale the tangential displacement
                        # find the unit direction in tangential velocity
                        # TODO: ELIMINATE THE SINGULARITY CASE
                        tx = ft0_x / ft0_magn
                        ty = ft0_y / ft0_magn
                        tz = ft0_z / ft0_magn
                        # this taken from Luding paper [2], eq (21)
                        d_tng_x[found_at] = -kt_1 * (fn_mu * tx + eta_t * vt_x)
                        d_tng_y[found_at] = -kt_1 * (fn_mu * ty + eta_t * vt_y)
                        d_tng_z[found_at] = -kt_1 * (fn_mu * tz + eta_t * vt_z)

                        # and also adjust the spring elongation
                        # at time t, which is used at stage 2 integrator
                        d_tng_x0[found_at] = d_tng_x[found_at]
                        d_tng_y0[found_at] = d_tng_y[found_at]
                        d_tng_z0[found_at] = d_tng_z[found_at]

                        # set the tangential force to static friction
                        # from Coulomb criterion
                        ft0_x = fn_mu * tx
                        ft0_y = fn_mu * ty
                        ft0_z = fn_mu * tz

                d_fx[d_idx] += fn_x + ft0_x
                d_fy[d_idx] += fn_y + ft0_y
                d_fz[d_idx] += fn_z + ft0_z

                # torque = n cross F
                d_torx[d_idx] += (nyc * ft0_z - nzc * ft0_y) * a_d
                d_tory[d_idx] += (nzc * ft0_x - nxc * ft0_z) * a_d
                d_torz[d_idx] += (nxc * ft0_y - nyc * ft0_x) * a_d


class TsuijiNonLinearParticleWallForceStage2(Equation):
    """Force between two spheres is implemented using Nonlinear DEM contact force
    law.

    The force is modelled from reference [1], [2]. Damping force is taken from
    [2], mainly for constants of damping.

    [1] Lagrangian numerical simulation of plug flow of cohesion less particles
    in a horizontal pipe Ye.

    [2] On the Determination of the Damping Coefficient of Non-linear
    Spring-dashpot System to Model Hertz Contact for Simulation by Discrete
    Element Method

    """

    def __init__(self, dest, sources, mu=0.5, en=0.8):
        """the required coefficients for force calculation.


        Keyword arguments:
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)

        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.

        """
        self.en = en
        # a constant for used for damping
        log_en = log(self.en)
        self.alpha_1 = -log_en * sqrt(5. / (log_en**2. + pi**2.))
        self.et = 1.0
        log_et = log(self.et)
        self.alpha_2 = -log_et * sqrt(5. / (log_et**2. + pi**2.))
        self.mu = mu
        super(TsuijiNonLinearParticleWallForceStage2, self).__init__(
            dest, sources)

    def initialize_pair(
            self, d_idx, d_m, d_u, d_v, d_w, d_x, d_y, d_z, d_fx, d_fy, d_fz,
            d_tng_x, d_tng_y, d_tng_z, d_tng_x0, d_tng_y0, d_tng_z0, d_tng_idx,
            d_tng_idx_dem_id, d_total_tng_contacts, d_dem_id, d_limit, d_vtx,
            d_vty, d_vtz, d_tng_nx, d_tng_ny, d_tng_nz, d_tng_nx0, d_tng_ny0,
            d_tng_nz0, d_wx, d_wy, d_wz, d_yng_m, d_poissons_ratio, d_shear_m,
            d_torx, d_tory, d_torz, d_rad_s, s_idx, s_x, s_y, s_z, s_nx, s_ny,
            s_nz, s_poissons_ratio, s_yng_m, s_shear_m, s_dem_id, s_np):
        i = declare('int')
        xij = declare('matrix(3)')
        vij = declare('matrix(3)')
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        for i in range(s_np[0]):
            # total number of contacts of particle i in destination
            tot_ctcs = d_total_tng_contacts[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_limit[0]
            # ending index is q -1
            q1 = p + tot_ctcs

            # check if the particle is in the tracking list
            # if so, then save the location at found_at
            found = 0
            for j in range(p, q1):
                if i == d_tng_idx[j]:
                    if s_dem_id[i] == d_tng_idx_dem_id[j]:
                        found_at = j
                        found = 1
                        break

            # Force calculation starts
            overlap = -1.
            xij[0] = d_x[d_idx] - s_x[i]
            xij[1] = d_y[d_idx] - s_y[i]
            xij[2] = d_z[d_idx] - s_z[i]
            overlap = d_rad_s[d_idx] - (
                xij[0] * s_nx[i] + xij[1] * s_ny[i] + xij[2] * s_nz[i])

            if overlap > 0:
                # basic variables: normal vector
                nxc = -s_nx[i]
                nyc = -s_ny[i]
                nzc = -s_nz[i]

                # ---- Relative velocity computation (Eq 2.9) ----
                # relative velocity of particle d_idx w.r.t particle i at
                # contact point.
                # Distance till contact point
                a_d = d_rad_s[d_idx] - overlap
                # TODO: This has to be replaced by a custom cross product
                # function
                # wij = a_i * w_i + a_j * w_j
                # since w_j is wall, and angular velocity is zero
                # wij = a_i * w_i
                wijx = a_d * d_wx[d_idx]
                wijy = a_d * d_wy[d_idx]
                wijz = a_d * d_wz[d_idx]
                # wij \cross nij
                wcn_x = wijy * nzc - wijz * nyc
                wcn_y = wijz * nxc - wijx * nzc
                wcn_z = wijx * nyc - wijy * nxc

                vij[0] = d_u[d_idx]
                vij[1] = d_v[d_idx]
                vij[2] = d_w[d_idx]
                vr_x = vij[0] + wcn_x
                vr_y = vij[1] + wcn_y
                vr_z = vij[2] + wcn_z

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
                Es = s_yng_m[i]
                Gd = d_shear_m[d_idx]
                Gs = s_shear_m[i]
                pd = d_poissons_ratio[d_idx]
                ps = s_poissons_ratio[i]
                rd = d_rad_s[d_idx]

                E_eff = Ed / (1. - pd**2.)
                G_eff = Gd / (2. - pd)
                # tsuiji paper gave
                E_eff = Ed * Es / (Ed * (1. - ps**2.) + Es * (1. - pd**2.))
                G_eff = Gd * Gs / (Gd * (2. - ps) + Gs * (2. - pd))
                r_eff = rd

                kn = 4. / 3. * E_eff * sqrt(r_eff)
                kt = 16. / 3. * G_eff * sqrt(r_eff * overlap)
                kt_1 = 1. / kt

                # damping force is taken from
                # "On the Determination of the Damping Coefficient
                # of Non-linear Spring-dashpot System to Model
                # Hertz Contact for Simulation by Discrete Element
                # Method" paper.
                # compute the damping constants
                m_eff = d_m[d_idx]
                eta_n = self.alpha_1 * sqrt(m_eff * kn) * overlap**0.25

                # normal force
                kn_overlap = kn * overlap**(1.5)
                fn_x = -kn_overlap * nxc - eta_n * vn_x
                fn_y = -kn_overlap * nyc - eta_n * vn_y
                fn_z = -kn_overlap * nzc - eta_n * vn_z

                # ------------- tangential force computation ----------------
                # do not add new particles to the contact list at step
                # t + dt / 2. But normal force will be computed as above.

                # Tangential force is computed if the particle is been tracked
                # already. When the particle is not been tracked (found == 0),
                # then it is an intermediate contact and we don't compute the
                # tangential contact due to it.
                if found == 1:
                    # current normal to the plane is nx, ny, nz
                    # the tangential spring is oriented normal to
                    # nxp, nyp, nzp (p is previous)
                    nxp = d_tng_nx[found_at]
                    nyp = d_tng_ny[found_at]
                    nzp = d_tng_nz[found_at]
                    # in order to compute the tangential force
                    # rotate the spring for current plane
                    # -------------------------
                    # rotation of the spring
                    # -------------------------
                    # rotation matrix
                    # n_current  \cross n_previous
                    tmpx = nyc * nzp - nzc * nyp
                    tmpy = nzc * nxp - nxc * nzp
                    tmpz = nxc * nyp - nyc * nxp
                    tmp_magn = (tmpx**2. + tmpy**2. + tmpz**2.)**0.5
                    # normalized rotation vector
                    hx = tmpx / tmp_magn
                    hy = tmpy / tmp_magn
                    hz = tmpz / tmp_magn

                    phi = asin(tmp_magn)
                    c = cos(phi)
                    s = sin(phi)
                    q = 1. - c

                    # matrix corresponding to the rotation vector
                    H0 = q * hx**2. + c
                    H1 = q * hx * hy - s * hz
                    H2 = q * hx * hz + s * hy

                    H3 = q * hy * hx + s * hz
                    H4 = q * hy**2. + c
                    H5 = q * hy * hz - s * hx

                    H6 = q * hz * hx - s * hy
                    H7 = q * hz * hy + s * hx
                    H8 = q * hz**2. + c

                    # save the tangential displacement temporarily
                    # will be used while rotation
                    tmpx = d_tng_x[found_at]
                    tmpy = d_tng_y[found_at]
                    tmpz = d_tng_z[found_at]

                    d_tng_x[found_at] = H0 * tmpx + H1 * tmpy + H2 * tmpz
                    d_tng_y[found_at] = H3 * tmpx + H4 * tmpy + H5 * tmpz
                    d_tng_z[found_at] = H6 * tmpx + H7 * tmpy + H8 * tmpz

                    # save the current normal of the spring
                    d_tng_nx[found_at] = nxc
                    d_tng_ny[found_at] = nyc
                    d_tng_nz[found_at] = nzc

                    # --------------------------------------
                    # similarly rotate the tangential spring at time t
                    # --------------------------------------
                    # current normal to the plane is nx, ny, nz
                    # the tangential spring is oriented normal to
                    # nxp, nyp, nzp
                    nxp = d_tng_nx0[found_at]
                    nyp = d_tng_ny0[found_at]
                    nzp = d_tng_nz0[found_at]
                    # in order to compute the tangential force
                    # rotate the spring for current plane
                    # -------------------------
                    # rotation of the spring
                    # -------------------------
                    # rotation matrix
                    # n_current  \cross n_previous
                    tmpx = nyc * nzp - nzc * nyp
                    tmpy = nzc * nxp - nxc * nzp
                    tmpz = nxc * nyp - nyc * nxp
                    tmp_magn = (tmpx**2. + tmpy**2. + tmpz**2.)**0.5
                    # normalized rotation vector
                    hx = tmpx / tmp_magn
                    hy = tmpy / tmp_magn
                    hz = tmpz / tmp_magn

                    phi = asin(tmp_magn)
                    c = cos(phi)
                    s = sin(phi)
                    q = 1. - c

                    # matrix corresponding to the rotation vector
                    H0 = q * hx**2. + c
                    H1 = q * hx * hy - s * hz
                    H2 = q * hx * hz + s * hy

                    H3 = q * hy * hx + s * hz
                    H4 = q * hy**2. + c
                    H5 = q * hy * hz - s * hx

                    H6 = q * hz * hx - s * hy
                    H7 = q * hz * hy + s * hx
                    H8 = q * hz**2. + c

                    # save the tangential displacement temporarily
                    # will be used while rotation
                    tmpx = d_tng_x0[found_at]
                    tmpy = d_tng_y0[found_at]
                    tmpz = d_tng_z0[found_at]

                    d_tng_x0[found_at] = H0 * tmpx + H1 * tmpy + H2 * tmpz
                    d_tng_y0[found_at] = H3 * tmpx + H4 * tmpy + H5 * tmpz
                    d_tng_z0[found_at] = H6 * tmpx + H7 * tmpy + H8 * tmpz

                    # save the current normal of the spring
                    d_tng_nx0[found_at] = nxc
                    d_tng_ny0[found_at] = nyc
                    d_tng_nz0[found_at] = nzc

                    # compute and set the tangential acceleration for the
                    # current time step
                    d_vtx[found_at] = vt_x
                    d_vty[found_at] = vt_y
                    d_vtz[found_at] = vt_z

                # compute the damping constants
                m_eff_t = 2. / 7. * m_eff
                eta_t = self.alpha_2 * sqrt(m_eff_t * kt)

                # find the tangential force from the tangential displacement
                # and tangential velocity (eq 2.11 Thesis Ye)
                # here d_tng_x[found_at] is automatically zero if it is
                # an intermediate contact
                ft0_x = -kt * d_tng_x[found_at] - eta_t * vt_x
                ft0_y = -kt * d_tng_y[found_at] - eta_t * vt_y
                ft0_z = -kt * d_tng_z[found_at] - eta_t * vt_z

                # (*) check against Coulomb criterion
                # Tangential force magnitude due to displacement
                ft0_magn = (ft0_x * ft0_x + ft0_y * ft0_y + ft0_z * ft0_z)**(
                    0.5)
                fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

                # we have to compare with static friction, so
                # this mu has to be static friction coefficient
                fn_mu = self.mu * fn_magn

                # if the tangential force magnitude is zero, then do nothing,
                # else do following
                if ft0_magn != 0.:
                    # compare tangential force with the static friction
                    if ft0_magn >= fn_mu:
                        # rescale the tangential displacement
                        # find the unit direction in tangential velocity
                        # TODO: ELIMINATE THE SINGULARITY CASE
                        tx = ft0_x / ft0_magn
                        ty = ft0_y / ft0_magn
                        tz = ft0_z / ft0_magn
                        # this taken from Luding paper [2], eq (21)
                        d_tng_x[found_at] = -kt_1 * (fn_mu * tx + eta_t * vt_x)
                        d_tng_y[found_at] = -kt_1 * (fn_mu * ty + eta_t * vt_y)
                        d_tng_z[found_at] = -kt_1 * (fn_mu * tz + eta_t * vt_z)

                        # and also adjust the spring elongation
                        # at time t, which is used at stage 2 integrator
                        d_tng_x0[found_at] = d_tng_x[found_at]
                        d_tng_y0[found_at] = d_tng_y[found_at]
                        d_tng_z0[found_at] = d_tng_z[found_at]

                        # set the tangential force to static friction
                        # from Coulomb criterion
                        ft0_x = fn_mu * tx
                        ft0_y = fn_mu * ty
                        ft0_z = fn_mu * tz

                d_fx[d_idx] += fn_x + ft0_x
                d_fy[d_idx] += fn_y + ft0_y
                d_fz[d_idx] += fn_z + ft0_z

                # torque = n cross F
                d_torx[d_idx] += (nyc * ft0_z - nzc * ft0_y) * a_d
                d_tory[d_idx] += (nzc * ft0_x - nxc * ft0_z) * a_d
                d_torz[d_idx] += (nxc * ft0_y - nyc * ft0_x) * a_d
            else:
                if found == 1:
                    d_vtx[found_at] = 0.
                    d_vty[found_at] = 0.
                    d_vtz[found_at] = 0.


class UpdateTangentialContacts(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_z, d_rad_s,
                        d_total_tng_contacts, d_tng_idx, d_limit, d_tng_x,
                        d_tng_y, d_tng_z, d_tng_nx, d_tng_ny, d_tng_nz, d_vtx,
                        d_vty, d_vtz, d_tng_idx_dem_id, s_x, s_y, s_z, s_rad_s,
                        s_dem_id):
        p = declare('int')
        count = declare('int')
        k = declare('int')
        xij = declare('matrix(3)')
        last_idx_tmp = declare('int')
        sidx = declare('int')
        dem_id = declare('int')
        rij = 0.0

        idx_total_ctcs = declare('int')
        idx_total_ctcs = d_total_tng_contacts[d_idx]
        # particle idx contacts has range of indices
        # and the first index would be
        p = d_idx * d_limit[0]
        last_idx_tmp = p + idx_total_ctcs - 1
        k = p
        count = 0

        # loop over all the contacts of particle d_idx
        while count < idx_total_ctcs:
            # The index of the particle with which
            # d_idx in contact is
            sidx = d_tng_idx[k]
            # get the dem id of the particle
            dem_id = d_tng_idx_dem_id[k]

            if sidx == -1:
                break
            else:
                if dem_id == s_dem_id[sidx]:
                    xij[0] = d_x[d_idx] - s_x[sidx]
                    xij[1] = d_y[d_idx] - s_y[sidx]
                    xij[2] = d_z[d_idx] - s_z[sidx]
                    rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] +
                               xij[2] * xij[2])
                    rinv = 1. / rij

                    overlap = d_rad_s[d_idx] + s_rad_s[sidx] - rij

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_tng_idx[k] = -1
                            d_tng_idx_dem_id[k] = -1
                            d_tng_x[k] = 0.
                            d_tng_y[k] = 0.
                            d_tng_z[k] = 0.
                            d_tng_nx[k] = 0.
                            d_tng_ny[k] = 0.
                            d_tng_nz[k] = 0.
                            d_vtx[k] = 0.
                            d_vty[k] = 0.
                            d_vtz[k] = 0.
                        else:
                            # swap the current tracking index with the final
                            # contact index
                            d_tng_idx[k] = d_tng_idx[last_idx_tmp]
                            d_tng_idx[last_idx_tmp] = -1

                            # swap tangential x displacement
                            d_tng_x[k] = d_tng_x[last_idx_tmp]
                            d_tng_x[last_idx_tmp] = 0.

                            # swap tangential y displacement
                            d_tng_y[k] = d_tng_x[last_idx_tmp]
                            d_tng_y[last_idx_tmp] = 0.

                            # swap tangential z displacement
                            d_tng_z[k] = d_tng_z[last_idx_tmp]
                            d_tng_z[last_idx_tmp] = 0.

                            # swap tangential nx orientation
                            d_tng_nx[k] = d_tng_nx[last_idx_tmp]
                            d_tng_nx[last_idx_tmp] = 0.

                            # swap tangential ny orientation
                            d_tng_ny[k] = d_tng_nx[last_idx_tmp]
                            d_tng_ny[last_idx_tmp] = 0.

                            # swap tangential nz orientation
                            d_tng_nz[k] = d_tng_nz[last_idx_tmp]
                            d_tng_nz[last_idx_tmp] = 0.

                            # swap tangential nx orientation
                            d_vtx[k] = d_vtx[last_idx_tmp]
                            d_vtx[last_idx_tmp] = 0.

                            # swap tangential ny orientation
                            d_vty[k] = d_vtx[last_idx_tmp]
                            d_vty[last_idx_tmp] = 0.

                            # swap tangential nz orientation
                            d_vtz[k] = d_vtz[last_idx_tmp]
                            d_vtz[last_idx_tmp] = 0.

                            # swap tangential idx dem id
                            d_tng_idx_dem_id[k] = d_tng_idx_dem_id[
                                last_idx_tmp]
                            d_tng_idx_dem_id[last_idx_tmp] = -1

                            # decrease the last_idx_tmp, since we swapped it to
                            # -1
                            last_idx_tmp -= 1

                        # decrement the total contacts of the particle
                        d_total_tng_contacts[d_idx] -= 1
                    else:
                        # ----------------------------------------------------
                        # this implies that the particles are still in contact
                        # now rotate the tangential spring about the new plane
                        # ----------------------------------------------------
                        # current normal to the plane is nx, ny, nz
                        # the tangential spring is oriented normal to
                        # nxp, nyp, nzp
                        nxp = d_tng_nx[k]
                        nyp = d_tng_ny[k]
                        nzp = d_tng_nz[k]
                        # and current normal vector between the particles is
                        nxc = -xij[0] * rinv
                        nyc = -xij[1] * rinv
                        nzc = -xij[2] * rinv

                        # in order to compute the tangential force
                        # rotate the spring for current plane
                        # -------------------------
                        # rotation of the spring
                        # -------------------------
                        # rotation matrix
                        # n_current  \cross n_previous
                        tmpx = nyc * nzp - nzc * nyp
                        tmpy = nzc * nxp - nxc * nzp
                        tmpz = nxc * nyp - nyc * nxp
                        tmp_magn = (tmpx**2. + tmpy**2. + tmpz**2.)**0.5
                        # normalized rotation vector
                        hx = tmpx / tmp_magn
                        hy = tmpy / tmp_magn
                        hz = tmpz / tmp_magn

                        phi = asin(tmp_magn)
                        c = cos(phi)
                        s = sin(phi)
                        q = 1. - c

                        # matrix corresponding to the rotation vector
                        H0 = q * hx**2. + c
                        H1 = q * hx * hy - s * hz
                        H2 = q * hx * hz + s * hy

                        H3 = q * hy * hx + s * hz
                        H4 = q * hy**2. + c
                        H5 = q * hy * hz - s * hx

                        H6 = q * hz * hx - s * hy
                        H7 = q * hz * hy + s * hx
                        H8 = q * hz**2. + c

                        # save the tangential displacement temporarily
                        # will be used while rotation
                        tmpx = d_tng_x[k]
                        tmpy = d_tng_y[k]
                        tmpz = d_tng_z[k]

                        d_tng_x[k] = H0 * tmpx + H1 * tmpy + H2 * tmpz
                        d_tng_y[k] = H3 * tmpx + H4 * tmpy + H5 * tmpz
                        d_tng_z[k] = H6 * tmpx + H7 * tmpy + H8 * tmpz

                        # save the current normal of the spring
                        d_tng_nx[k] = nxc
                        d_tng_ny[k] = nyc
                        d_tng_nz[k] = nzc

                        k = k + 1
                else:
                    k = k + 1
                count += 1


class TsuijiNonLinearParticleWallForceNoRotationStage1(Equation):
    """Force between two spheres is implemented using Nonlinear DEM contact force
    law.

    The force is modelled from reference [1], [2]. Damping force is taken from
    [2], mainly for constants of damping.

    [1] Lagrangian numerical simulation of plug flow of cohesion less particles
    in a horizontal pipe Ye.

    [2] On the Determination of the Damping Coefficient of Non-linear
    Spring-dashpot System to Model Hertz Contact for Simulation by Discrete
    Element Method

    """

    def __init__(self, dest, sources, mu=0.5, en=0.8):
        """the required coefficients for force calculation.


        Keyword arguments:
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)

        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.

        """
        self.en = en
        # a constant for used for damping
        log_en = log(self.en)
        self.alpha_1 = -log_en * sqrt(5. / (log_en**2. + pi**2.))
        self.et = 0.5 * self.en
        log_et = log(self.et)
        self.alpha_2 = -log_et * sqrt(5. / (log_et**2. + pi**2.))
        self.mu = mu
        super(TsuijiNonLinearParticleWallForceNoRotationStage1, self).__init__(
            dest, sources)

    def initialize_pair(
            self, d_idx, d_m, d_u, d_v, d_w, d_x, d_y, d_z, d_fx, d_fy, d_fz,
            d_tng_x, d_tng_y, d_tng_z, d_tng_x0, d_tng_y0, d_tng_z0, d_tng_idx,
            d_tng_idx_dem_id, d_total_tng_contacts, d_dem_id, d_limit, d_vtx,
            d_vty, d_vtz, d_tng_nx, d_tng_ny, d_tng_nz, d_tng_nx0, d_tng_ny0,
            d_tng_nz0, d_wx, d_wy, d_wz, d_yng_m, d_poissons_ratio, d_shear_m,
            d_torx, d_tory, d_torz, d_rad_s, s_idx, s_x, s_y, s_z, s_nx, s_ny,
            s_nz, s_poissons_ratio, s_yng_m, s_shear_m, s_dem_id, s_np):
        i = declare('int')
        xij = declare('matrix(3)')
        vij = declare('matrix(3)')
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        for i in range(s_np[0]):
            # total number of contacts of particle i in destination
            tot_ctcs = d_total_tng_contacts[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_limit[0]
            # ending index is q -1
            q1 = p + tot_ctcs

            # check if the particle is in the tracking list
            # if so, then save the location at found_at
            found = 0
            for j in range(p, q1):
                if i == d_tng_idx[j]:
                    if s_dem_id[i] == d_tng_idx_dem_id[j]:
                        found_at = j
                        found = 1
                        break

            # Force calculation starts
            overlap = -1.
            xij[0] = d_x[d_idx] - s_x[i]
            xij[1] = d_y[d_idx] - s_y[i]
            xij[2] = d_z[d_idx] - s_z[i]
            overlap = d_rad_s[d_idx] - (
                xij[0] * s_nx[i] + xij[1] * s_ny[i] + xij[2] * s_nz[i])

            if overlap > 0:
                # basic variables: normal vector
                nxc = -s_nx[i]
                nyc = -s_ny[i]
                nzc = -s_nz[i]

                # ---- Relative velocity computation (Eq 2.9) ----
                # relative velocity of particle d_idx w.r.t particle i at
                # contact point.
                # Distance till contact point
                a_d = d_rad_s[d_idx] - overlap
                # TODO: This has to be replaced by a custom cross product
                # function
                # wij = a_i * w_i + a_j * w_j
                # since w_j is wall, and angular velocity is zero
                # wij = a_i * w_i
                wijx = a_d * d_wx[d_idx]
                wijy = a_d * d_wy[d_idx]
                wijz = a_d * d_wz[d_idx]
                # wij \cross nij
                wcn_x = wijy * nzc - wijz * nyc
                wcn_y = wijz * nxc - wijx * nzc
                wcn_z = wijx * nyc - wijy * nxc

                vij[0] = d_u[d_idx]
                vij[1] = d_v[d_idx]
                vij[2] = d_w[d_idx]
                vr_x = vij[0] + wcn_x
                vr_y = vij[1] + wcn_y
                vr_z = vij[2] + wcn_z

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
                Es = s_yng_m[i]
                Gd = d_shear_m[d_idx]
                Gs = s_shear_m[i]
                pd = d_poissons_ratio[d_idx]
                ps = s_poissons_ratio[i]
                rd = d_rad_s[d_idx]

                E_eff = Ed / (1. - pd**2.)
                G_eff = Gd / (2. - pd)
                # E_eff = Ed
                # G_eff = Gd
                # tsuiji paper gave
                # E_eff = Ed * Es / (Ed * (1. - ps**2.) + Es * (1. - pd**2.))
                r_eff = rd

                kn = 4. / 3. * E_eff * sqrt(r_eff)
                kt = 16. / 3. * G_eff * sqrt(r_eff)
                kt_1 = 1. / kt

                # damping force is taken from
                # "On the Determination of the Damping Coefficient
                # of Non-linear Spring-dashpot System to Model
                # Hertz Contact for Simulation by Discrete Element
                # Method" paper.
                # compute the damping constants
                m_eff = d_m[d_idx]
                eta_n = self.alpha_1 * sqrt(m_eff * kn) * overlap**0.25

                # normal force
                kn_overlap = kn * overlap**(1.5)
                fn_x = -kn_overlap * nxc - eta_n * vn_x
                fn_y = -kn_overlap * nyc - eta_n * vn_y
                fn_z = -kn_overlap * nzc - eta_n * vn_z

                # ------------- tangential force computation ----------------
                # if the particle is not been tracked then assign an index in
                # tracking history.
                if found == 0:
                    found_at = q1
                    d_tng_idx[found_at] = i
                    d_total_tng_contacts[d_idx] += 1
                    d_tng_idx_dem_id[found_at] = s_dem_id[i]
                    d_tng_nx[found_at] = nxc
                    d_tng_ny[found_at] = nyc
                    d_tng_nz[found_at] = nzc
                    d_tng_nx0[found_at] = nxc
                    d_tng_ny0[found_at] = nyc
                    d_tng_nz0[found_at] = nzc

                # compute and set the tangential acceleration for the
                # current time step
                d_vtx[found_at] = vt_x
                d_vty[found_at] = vt_y
                d_vtz[found_at] = vt_z

                # compute the damping constants
                m_eff_t = 2. / 7. * m_eff
                eta_t = self.alpha_2 * sqrt(m_eff_t * kt)

                # find the tangential force from the tangential displacement
                # and tangential velocity (eq 2.11 Thesis Ye)
                ft0_x = -kt * d_tng_x[found_at] - eta_t * vt_x
                ft0_y = -kt * d_tng_y[found_at] - eta_t * vt_y
                ft0_z = -kt * d_tng_z[found_at] - eta_t * vt_z

                # (*) check against Coulomb criterion
                # Tangential force magnitude due to displacement
                ft0_magn = (ft0_x * ft0_x + ft0_y * ft0_y + ft0_z * ft0_z)**(
                    0.5)
                fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

                # we have to compare with static friction, so
                # this mu has to be static friction coefficient
                fn_mu = self.mu * fn_magn

                # if the tangential force magnitude is zero, then do nothing,
                # else do following
                if ft0_magn != 0.:
                    # compare tangential force with the static friction
                    if ft0_magn >= fn_mu:
                        # rescale the tangential displacement
                        # find the unit direction in tangential velocity
                        # TODO: ELIMINATE THE SINGULARITY CASE
                        tx = ft0_x / ft0_magn
                        ty = ft0_y / ft0_magn
                        tz = ft0_z / ft0_magn
                        # this taken from Luding paper [2], eq (21)
                        d_tng_x[found_at] = -kt_1 * (fn_mu * tx + eta_t * vt_x)
                        d_tng_y[found_at] = -kt_1 * (fn_mu * ty + eta_t * vt_y)
                        d_tng_z[found_at] = -kt_1 * (fn_mu * tz + eta_t * vt_z)

                        # and also adjust the spring elongation
                        # at time t, which is used at stage 2 integrator
                        d_tng_x0[found_at] = d_tng_x[found_at]
                        d_tng_y0[found_at] = d_tng_y[found_at]
                        d_tng_z0[found_at] = d_tng_z[found_at]

                        # set the tangential force to static friction
                        # from Coulomb criterion
                        ft0_x = fn_mu * tx
                        ft0_y = fn_mu * ty
                        ft0_z = fn_mu * tz

                d_fx[d_idx] += fn_x + ft0_x
                d_fy[d_idx] += fn_y + ft0_y
                d_fz[d_idx] += fn_z + ft0_z

                # torque = n cross F
                d_torx[d_idx] += (nyc * ft0_z - nzc * ft0_y) * a_d
                d_tory[d_idx] += (nzc * ft0_x - nxc * ft0_z) * a_d
                d_torz[d_idx] += (nxc * ft0_y - nyc * ft0_x) * a_d


class TsuijiNonLinearParticleWallForceNoRotationStage2(Equation):
    """Force between two spheres is implemented using Nonlinear DEM contact force
    law.

    The force is modelled from reference [1], [2]. Damping force is taken from
    [2], mainly for constants of damping.

    [1] Lagrangian numerical simulation of plug flow of cohesion less particles
    in a horizontal pipe Ye.

    [2] On the Determination of the Damping Coefficient of Non-linear
    Spring-dashpot System to Model Hertz Contact for Simulation by Discrete
    Element Method

    """

    def __init__(self, dest, sources, mu=0.5, en=0.8):
        """the required coefficients for force calculation.


        Keyword arguments:
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)

        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.

        """
        self.en = en
        # a constant for used for damping
        log_en = log(self.en)
        self.alpha_1 = -log_en * sqrt(5. / (log_en**2. + pi**2.))
        self.et = 0.5 * self.en
        self.mu = mu
        super(TsuijiNonLinearParticleWallForceNoRotationStage2, self).__init__(
            dest, sources)

    def initialize_pair(
            self, d_idx, d_m, d_u, d_v, d_w, d_x, d_y, d_z, d_fx, d_fy, d_fz,
            d_tng_x, d_tng_y, d_tng_z, d_tng_x0, d_tng_y0, d_tng_z0, d_tng_idx,
            d_tng_idx_dem_id, d_total_tng_contacts, d_dem_id, d_limit, d_vtx,
            d_vty, d_vtz, d_tng_nx, d_tng_ny, d_tng_nz, d_tng_nx0, d_tng_ny0,
            d_tng_nz0, d_wx, d_wy, d_wz, d_yng_m, d_poissons_ratio, d_shear_m,
            d_torx, d_tory, d_torz, d_rad_s, s_idx, s_x, s_y, s_z, s_nx, s_ny,
            s_nz, s_poissons_ratio, s_yng_m, s_shear_m, s_dem_id, s_np):
        i = declare('int')
        xij = declare('matrix(3)')
        vij = declare('matrix(3)')
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        for i in range(s_np[0]):
            # total number of contacts of particle i in destination
            tot_ctcs = d_total_tng_contacts[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_limit[0]
            # ending index is q -1
            q1 = p + tot_ctcs

            # check if the particle is in the tracking list
            # if so, then save the location at found_at
            found = 0
            for j in range(p, q1):
                if i == d_tng_idx[j]:
                    if s_dem_id[i] == d_tng_idx_dem_id[j]:
                        found_at = j
                        found = 1
                        break

            # Force calculation starts
            overlap = -1.
            xij[0] = d_x[d_idx] - s_x[i]
            xij[1] = d_y[d_idx] - s_y[i]
            xij[2] = d_z[d_idx] - s_z[i]
            overlap = d_rad_s[d_idx] - (
                xij[0] * s_nx[i] + xij[1] * s_ny[i] + xij[2] * s_nz[i])

            if overlap > 0:
                # basic variables: normal vector
                nxc = -s_nx[i]
                nyc = -s_ny[i]
                nzc = -s_nz[i]

                # ---- Relative velocity computation (Eq 2.9) ----
                # relative velocity of particle d_idx w.r.t particle i at
                # contact point.
                # Distance till contact point
                a_d = d_rad_s[d_idx] - overlap
                # TODO: This has to be replaced by a custom cross product
                # function
                # wij = a_i * w_i + a_j * w_j
                # since w_j is wall, and angular velocity is zero
                # wij = a_i * w_i
                wijx = a_d * d_wx[d_idx]
                wijy = a_d * d_wy[d_idx]
                wijz = a_d * d_wz[d_idx]
                # wij \cross nij
                wcn_x = wijy * nzc - wijz * nyc
                wcn_y = wijz * nxc - wijx * nzc
                wcn_z = wijx * nyc - wijy * nxc

                vij[0] = d_u[d_idx]
                vij[1] = d_v[d_idx]
                vij[2] = d_w[d_idx]
                vr_x = vij[0] + wcn_x
                vr_y = vij[1] + wcn_y
                vr_z = vij[2] + wcn_z

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
                vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

                # compute the spring stiffness (nonlinear model)
                Ed = d_yng_m[d_idx]
                Es = s_yng_m[i]
                Gd = d_shear_m[d_idx]
                Gs = s_shear_m[i]
                pd = d_poissons_ratio[d_idx]
                ps = s_poissons_ratio[i]
                rd = d_rad_s[d_idx]

                E_eff = Ed / (1. - pd**2.)
                G_eff = Gd / (2. - pd)
                # E_eff = Ed
                # G_eff = Gd
                # tsuiji paper gave
                # E_eff = Ed * Es / (Ed * (1. - ps**2.) + Es * (1. - pd**2.))
                r_eff = rd

                kn = 4. / 3. * E_eff * sqrt(r_eff)
                kt = 16. / 3. * G_eff * sqrt(r_eff)
                kt_1 = 1. / kt

                # damping force is taken from
                # "On the Determination of the Damping Coefficient
                # of Non-linear Spring-dashpot System to Model
                # Hertz Contact for Simulation by Discrete Element
                # Method" paper.
                # compute the damping constants
                m_eff = d_m[d_idx]
                eta_n = self.alpha_1 * sqrt(m_eff * kn) * overlap**0.25

                # normal force
                kn_overlap = kn * overlap**(1.5)
                fn_x = -kn_overlap * nxc - eta_n * vn_x
                fn_y = -kn_overlap * nyc - eta_n * vn_y
                fn_z = -kn_overlap * nzc - eta_n * vn_z

                # ------------- tangential force computation ----------------
                # do not add new particles to the contact list at step
                # t + dt / 2. But normal force will be computed as above.

                # Tangential force is computed if the particle is been tracked
                # already. When the particle is not been tracked (found == 0),
                # then it is an intermediate contact and we don't compute the
                # tangential contact due to it.
                if found == 1:
                    # current normal to the plane is nx, ny, nz
                    # the tangential spring is oriented normal to
                    # nxp, nyp, nzp (p is previous)
                    nxp = d_tng_nx[found_at]
                    nyp = d_tng_ny[found_at]
                    nzp = d_tng_nz[found_at]
                    # in order to compute the tangential force
                    # rotate the spring for current plane
                    # -------------------------
                    # rotation of the spring
                    # -------------------------
                    # rotation matrix
                    # n_current  \cross n_previous
                    tmpx = nyc * nzp - nzc * nyp
                    tmpy = nzc * nxp - nxc * nzp
                    tmpz = nxc * nyp - nyc * nxp
                    tmp_magn = (tmpx**2. + tmpy**2. + tmpz**2.)**0.5
                    # normalized rotation vector
                    hx = tmpx / tmp_magn
                    hy = tmpy / tmp_magn
                    hz = tmpz / tmp_magn

                    phi = asin(tmp_magn)
                    c = cos(phi)
                    s = sin(phi)
                    q = 1. - c

                    # matrix corresponding to the rotation vector
                    H0 = q * hx**2. + c
                    H1 = q * hx * hy - s * hz
                    H2 = q * hx * hz + s * hy

                    H3 = q * hy * hx + s * hz
                    H4 = q * hy**2. + c
                    H5 = q * hy * hz - s * hx

                    H6 = q * hz * hx - s * hy
                    H7 = q * hz * hy + s * hx
                    H8 = q * hz**2. + c

                    # save the tangential displacement temporarily
                    # will be used while rotation
                    tmpx = d_tng_x[found_at]
                    tmpy = d_tng_y[found_at]
                    tmpz = d_tng_z[found_at]

                    d_tng_x[found_at] = H0 * tmpx + H1 * tmpy + H2 * tmpz
                    d_tng_y[found_at] = H3 * tmpx + H4 * tmpy + H5 * tmpz
                    d_tng_z[found_at] = H6 * tmpx + H7 * tmpy + H8 * tmpz

                    # save the current normal of the spring
                    d_tng_nx[found_at] = nxc
                    d_tng_ny[found_at] = nyc
                    d_tng_nz[found_at] = nzc

                    # --------------------------------------
                    # similarly rotate the tangential spring at time t
                    # --------------------------------------
                    # current normal to the plane is nx, ny, nz
                    # the tangential spring is oriented normal to
                    # nxp, nyp, nzp
                    nxp = d_tng_nx0[found_at]
                    nyp = d_tng_ny0[found_at]
                    nzp = d_tng_nz0[found_at]
                    # in order to compute the tangential force
                    # rotate the spring for current plane
                    # -------------------------
                    # rotation of the spring
                    # -------------------------
                    # rotation matrix
                    # n_current  \cross n_previous
                    tmpx = nyc * nzp - nzc * nyp
                    tmpy = nzc * nxp - nxc * nzp
                    tmpz = nxc * nyp - nyc * nxp
                    tmp_magn = (tmpx**2. + tmpy**2. + tmpz**2.)**0.5
                    # normalized rotation vector
                    hx = tmpx / tmp_magn
                    hy = tmpy / tmp_magn
                    hz = tmpz / tmp_magn

                    phi = asin(tmp_magn)
                    c = cos(phi)
                    s = sin(phi)
                    q = 1. - c

                    # matrix corresponding to the rotation vector
                    H0 = q * hx**2. + c
                    H1 = q * hx * hy - s * hz
                    H2 = q * hx * hz + s * hy

                    H3 = q * hy * hx + s * hz
                    H4 = q * hy**2. + c
                    H5 = q * hy * hz - s * hx

                    H6 = q * hz * hx - s * hy
                    H7 = q * hz * hy + s * hx
                    H8 = q * hz**2. + c

                    # save the tangential displacement temporarily
                    # will be used while rotation
                    tmpx = d_tng_x0[found_at]
                    tmpy = d_tng_y0[found_at]
                    tmpz = d_tng_z0[found_at]

                    d_tng_x0[found_at] = H0 * tmpx + H1 * tmpy + H2 * tmpz
                    d_tng_y0[found_at] = H3 * tmpx + H4 * tmpy + H5 * tmpz
                    d_tng_z0[found_at] = H6 * tmpx + H7 * tmpy + H8 * tmpz

                    # save the current normal of the spring
                    d_tng_nx0[found_at] = nxc
                    d_tng_ny0[found_at] = nyc
                    d_tng_nz0[found_at] = nzc

                    # compute and set the tangential acceleration for the
                    # current time step
                    d_vtx[found_at] = vt_x
                    d_vty[found_at] = vt_y
                    d_vtz[found_at] = vt_z

                # compute the damping constants
                m_eff_t = 2. / 7. * m_eff
                eta_t = self.alpha_2 * sqrt(m_eff_t * kt)

                # find the tangential force from the tangential displacement
                # and tangential velocity (eq 2.11 Thesis Ye)
                # here d_tng_x[found_at] is automatically zero if it is
                # an intermediate contact
                ft0_x = -kt * d_tng_x[found_at] - eta_t * vt_x
                ft0_y = -kt * d_tng_y[found_at] - eta_t * vt_y
                ft0_z = -kt * d_tng_z[found_at] - eta_t * vt_z

                # (*) check against Coulomb criterion
                # Tangential force magnitude due to displacement
                ft0_magn = (ft0_x * ft0_x + ft0_y * ft0_y + ft0_z * ft0_z)**(
                    0.5)
                fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

                # we have to compare with static friction, so
                # this mu has to be static friction coefficient
                fn_mu = self.mu * fn_magn

                # if the tangential force magnitude is zero, then do nothing,
                # else do following
                if ft0_magn != 0.:
                    # compare tangential force with the static friction
                    if ft0_magn >= fn_mu:
                        # rescale the tangential displacement
                        # find the unit direction in tangential velocity
                        # TODO: ELIMINATE THE SINGULARITY CASE
                        tx = ft0_x / ft0_magn
                        ty = ft0_y / ft0_magn
                        tz = ft0_z / ft0_magn
                        # this taken from Luding paper [2], eq (21)
                        d_tng_x[found_at] = -kt_1 * (fn_mu * tx + eta_t * vt_x)
                        d_tng_y[found_at] = -kt_1 * (fn_mu * ty + eta_t * vt_y)
                        d_tng_z[found_at] = -kt_1 * (fn_mu * tz + eta_t * vt_z)

                        # and also adjust the spring elongation
                        # at time t, which is used at stage 2 integrator
                        d_tng_x0[found_at] = d_tng_x[found_at]
                        d_tng_y0[found_at] = d_tng_y[found_at]
                        d_tng_z0[found_at] = d_tng_z[found_at]

                        # set the tangential force to static friction
                        # from Coulomb criterion
                        ft0_x = fn_mu * tx
                        ft0_y = fn_mu * ty
                        ft0_z = fn_mu * tz

                d_fx[d_idx] += fn_x + ft0_x
                d_fy[d_idx] += fn_y + ft0_y
                d_fz[d_idx] += fn_z + ft0_z

                # torque = n cross F
                d_torx[d_idx] += (nyc * ft0_z - nzc * ft0_y) * a_d
                d_tory[d_idx] += (nzc * ft0_x - nxc * ft0_z) * a_d
                d_torz[d_idx] += (nxc * ft0_y - nyc * ft0_x) * a_d
            else:
                if found == 1:
                    d_vtx[found_at] = 0.
                    d_vty[found_at] = 0.
                    d_vtz[found_at] = 0.


class UpdateTangentialContactsNoRotation(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_z, d_rad_s,
                        d_total_tng_contacts, d_tng_idx, d_limit, d_tng_x,
                        d_tng_y, d_tng_z, d_tng_nx, d_tng_ny, d_tng_nz, d_vtx,
                        d_vty, d_vtz, d_tng_idx_dem_id, s_x, s_y, s_z, s_rad_s,
                        s_dem_id):
        p = declare('int')
        count = declare('int')
        k = declare('int')
        xij = declare('matrix(3)')
        last_idx_tmp = declare('int')
        sidx = declare('int')
        dem_id = declare('int')
        rij = 0.0

        idx_total_ctcs = declare('int')
        idx_total_ctcs = d_total_tng_contacts[d_idx]
        # particle idx contacts has range of indices
        # and the first index would be
        p = d_idx * d_limit[0]
        last_idx_tmp = p + idx_total_ctcs - 1
        k = p
        count = 0

        # loop over all the contacts of particle d_idx
        while count < idx_total_ctcs:
            # The index of the particle with which
            # d_idx in contact is
            sidx = d_tng_idx[k]
            # get the dem id of the particle
            dem_id = d_tng_idx_dem_id[k]

            if sidx == -1:
                break
            else:
                if dem_id == s_dem_id[sidx]:
                    xij[0] = d_x[d_idx] - s_x[sidx]
                    xij[1] = d_y[d_idx] - s_y[sidx]
                    xij[2] = d_z[d_idx] - s_z[sidx]
                    rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] +
                               xij[2] * xij[2])
                    rinv = 1. / rij

                    overlap = d_rad_s[d_idx] + s_rad_s[sidx] - rij

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_tng_idx[k] = -1
                            d_tng_idx_dem_id[k] = -1
                            d_tng_x[k] = 0.
                            d_tng_y[k] = 0.
                            d_tng_z[k] = 0.
                            d_tng_nx[k] = 0.
                            d_tng_ny[k] = 0.
                            d_tng_nz[k] = 0.
                            d_vtx[k] = 0.
                            d_vty[k] = 0.
                            d_vtz[k] = 0.
                        else:
                            # swap the current tracking index with the final
                            # contact index
                            d_tng_idx[k] = d_tng_idx[last_idx_tmp]
                            d_tng_idx[last_idx_tmp] = -1

                            # swap tangential x displacement
                            d_tng_x[k] = d_tng_x[last_idx_tmp]
                            d_tng_x[last_idx_tmp] = 0.

                            # swap tangential y displacement
                            d_tng_y[k] = d_tng_x[last_idx_tmp]
                            d_tng_y[last_idx_tmp] = 0.

                            # swap tangential z displacement
                            d_tng_z[k] = d_tng_z[last_idx_tmp]
                            d_tng_z[last_idx_tmp] = 0.

                            # swap tangential nx orientation
                            d_tng_nx[k] = d_tng_nx[last_idx_tmp]
                            d_tng_nx[last_idx_tmp] = 0.

                            # swap tangential ny orientation
                            d_tng_ny[k] = d_tng_nx[last_idx_tmp]
                            d_tng_ny[last_idx_tmp] = 0.

                            # swap tangential nz orientation
                            d_tng_nz[k] = d_tng_nz[last_idx_tmp]
                            d_tng_nz[last_idx_tmp] = 0.

                            # swap tangential nx orientation
                            d_vtx[k] = d_vtx[last_idx_tmp]
                            d_vtx[last_idx_tmp] = 0.

                            # swap tangential ny orientation
                            d_vty[k] = d_vtx[last_idx_tmp]
                            d_vty[last_idx_tmp] = 0.

                            # swap tangential nz orientation
                            d_vtz[k] = d_vtz[last_idx_tmp]
                            d_vtz[last_idx_tmp] = 0.

                            # swap tangential idx dem id
                            d_tng_idx_dem_id[k] = d_tng_idx_dem_id[
                                last_idx_tmp]
                            d_tng_idx_dem_id[last_idx_tmp] = -1

                            # decrease the last_idx_tmp, since we swapped it to
                            # -1
                            last_idx_tmp -= 1

                        # decrement the total contacts of the particle
                        d_total_tng_contacts[d_idx] -= 1
                    else:
                        # ----------------------------------------------------
                        # this implies that the particles are still in contact
                        # now rotate the tangential spring about the new plane
                        # ----------------------------------------------------
                        # current normal to the plane is nx, ny, nz
                        # the tangential spring is oriented normal to
                        # nxp, nyp, nzp
                        nxp = d_tng_nx[k]
                        nyp = d_tng_ny[k]
                        nzp = d_tng_nz[k]
                        # and current normal vector between the particles is
                        nxc = -xij[0] * rinv
                        nyc = -xij[1] * rinv
                        nzc = -xij[2] * rinv

                        # in order to compute the tangential force
                        # rotate the spring for current plane
                        # -------------------------
                        # rotation of the spring
                        # -------------------------
                        # rotation matrix
                        # n_current  \cross n_previous
                        tmpx = nyc * nzp - nzc * nyp
                        tmpy = nzc * nxp - nxc * nzp
                        tmpz = nxc * nyp - nyc * nxp
                        tmp_magn = (tmpx**2. + tmpy**2. + tmpz**2.)**0.5
                        # normalized rotation vector
                        hx = tmpx / tmp_magn
                        hy = tmpy / tmp_magn
                        hz = tmpz / tmp_magn

                        phi = asin(tmp_magn)
                        c = cos(phi)
                        s = sin(phi)
                        q = 1. - c

                        # matrix corresponding to the rotation vector
                        H0 = q * hx**2. + c
                        H1 = q * hx * hy - s * hz
                        H2 = q * hx * hz + s * hy

                        H3 = q * hy * hx + s * hz
                        H4 = q * hy**2. + c
                        H5 = q * hy * hz - s * hx

                        H6 = q * hz * hx - s * hy
                        H7 = q * hz * hy + s * hx
                        H8 = q * hz**2. + c

                        # save the tangential displacement temporarily
                        # will be used while rotation
                        tmpx = d_tng_x[k]
                        tmpy = d_tng_y[k]
                        tmpz = d_tng_z[k]

                        d_tng_x[k] = H0 * tmpx + H1 * tmpy + H2 * tmpz
                        d_tng_y[k] = H3 * tmpx + H4 * tmpy + H5 * tmpz
                        d_tng_z[k] = H6 * tmpx + H7 * tmpy + H8 * tmpz

                        # save the current normal of the spring
                        d_tng_nx[k] = nxc
                        d_tng_ny[k] = nyc
                        d_tng_nz[k] = nzc

                        k = k + 1
                else:
                    k = k + 1
                count += 1


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

        # -----------------------------------------------
        # save the initial tangential contact information
        # -----------------------------------------------
        i = declare('int')
        p = declare('int')
        q = declare('int')
        tot_ctcs = declare('int')
        tot_ctcs = d_total_tng_contacts[d_idx]
        p = d_idx * d_limit[0]
        q = p + tot_ctcs

        for i in range(p, q):
            d_tng_x0[i] = d_tng_x[i]
            d_tng_y0[i] = d_tng_y[i]
            d_tng_z0[i] = d_tng_z[i]
            d_tng_nx0[i] = d_tng_nx[i]
            d_tng_ny0[i] = d_tng_ny[i]
            d_tng_nz0[i] = d_tng_nz[i]

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_x0, d_y0, d_z0, d_u0, d_v0, d_w0, d_wx0, d_wy0, d_wz0, d_torx,
               d_tory, d_torz, d_wx, d_wy, d_wz, d_m_inv, d_I_inv,
               d_total_tng_contacts, d_limit, d_tng_x, d_tng_y, d_tng_z,
               d_tng_x0, d_tng_y0, d_tng_z0, d_vtx, d_vty, d_vtz, dt):
        dtb2 = dt / 2.

        # --------------------------------------
        # increment the tangential displacement
        # --------------------------------------
        i = declare('int')
        p = declare('int')
        q = declare('int')
        tot_ctcs = declare('int')
        tot_ctcs = d_total_tng_contacts[d_idx]
        p = d_idx * d_limit[0]
        q = p + tot_ctcs

        for i in range(p, q):
            d_tng_x[i] += d_vtx[i] * dtb2
            d_tng_y[i] += d_vty[i] * dtb2
            d_tng_z[i] += d_vtz[i] * dtb2

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
               d_total_tng_contacts, d_limit, d_tng_x, d_tng_y, d_tng_z,
               d_tng_x0, d_tng_y0, d_tng_z0, d_vtx, d_vty, d_vtz, dt):
        # --------------------------------------
        # increment the tangential displacement
        # --------------------------------------
        i = declare('int')
        p = declare('int')
        q = declare('int')
        tot_ctcs = declare('int')
        tot_ctcs = d_total_tng_contacts[d_idx]
        p = d_idx * d_limit[0]
        q = p + tot_ctcs

        for i in range(p, q):
            d_tng_x[i] = d_tng_x0[i] + d_vtx[i] * dt
            d_tng_y[i] = d_tng_y0[i] + d_vty[i] * dt
            d_tng_z[i] = d_tng_z0[i] + d_vtz[i] * dt

        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt * d_fx[d_idx] * d_m_inv[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_fy[d_idx] * d_m_inv[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_fz[d_idx] * d_m_inv[d_idx]

        d_wx[d_idx] = d_wx0[d_idx] + dt * d_torx[d_idx] * d_I_inv[d_idx]
        d_wy[d_idx] = d_wy0[d_idx] + dt * d_tory[d_idx] * d_I_inv[d_idx]
        d_wz[d_idx] = d_wz0[d_idx] + dt * d_torz[d_idx] * d_I_inv[d_idx]
