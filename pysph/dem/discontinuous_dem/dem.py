from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from compyle.api import declare
from math import sqrt, log, pi
import numpy as np
from pysph.sph.scheme import Scheme
from pysph.sph.equation import Group, MultiStageEquations
from pysph.dem.discontinuous_dem.dem_nonlinear import EPECIntegratorMultiStage


def get_particle_array_dem(constants=None, **props):
    """Return a particle array for a dem particles
    """
    dim = props.pop('dim', None)

    dem_props = [
        'wx', 'wy', 'wz', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'rad_s',
        'm_inverse', 'I_inverse', 'u0', 'v0', 'w0', 'wx0', 'wy0', 'wz0', 'x0',
        'y0', 'z0'
    ]

    dem_id = props.pop('dem_id', None)

    pa = get_particle_array(additional_props=dem_props, **props,
                            constants=constants)

    pa.add_property('dem_id', type='int', data=dem_id)
    # create the array to save the tangential interaction particles
    # index and other variables
    if dim == 3:
        limit = 30
    elif dim == 2 or dim is None:
        limit = 6

    pa.add_constant('limit', limit)

    pa.add_property('free_cnt_idx', stride=limit, type='int')
    pa.free_cnt_idx[:] = -1
    pa.add_property('free_cnt_idx_dem_id', stride=limit, type='int')
    pa.free_cnt_idx_dem_id[:] = -1

    pa.add_property('free_cnt_tng_disp_x', stride=limit)
    pa.add_property('free_cnt_tng_disp_y', stride=limit)
    pa.add_property('free_cnt_tng_disp_z', stride=limit)
    pa.add_property('free_cnt_tng_disp_x0', stride=limit)
    pa.add_property('free_cnt_tng_disp_y0', stride=limit)
    pa.add_property('free_cnt_tng_disp_z0', stride=limit)

    pa.add_property('total_no_free_cnt', type='int')
    pa.total_no_free_cnt[:] = 0

    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'wx', 'wy', 'wz', 'm', 'pid', 'tag',
        'gid', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'I_inverse',
        'm_inverse', 'rad_s', 'dem_id'
    ])

    return pa


class BodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(BodyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz):
        d_fx[d_idx] = d_m[d_idx] * self.gx
        d_fy[d_idx] = d_m[d_idx] * self.gy
        d_fz[d_idx] = d_m[d_idx] * self.gz
        d_torx[d_idx] = 0.
        d_tory[d_idx] = 0.
        d_torz[d_idx] = 0.


class DEMParticleParticleForceStage1(Equation):
    """
    This equation computes the force between two discrete DEM particles.
    This equation implements both linear and nonlinear DEM. It is written from
    many sources [1]

    [1] Introduction to DEM Luding MS.
    """
    def __init__(self, dest, sources, kn=-1., mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(
            abs(kn)) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(DEMParticleParticleForceStage1, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_yng_mod, d_shear_mod, d_poisson, d_fx, d_fy,
             d_fz, d_free_cnt_tng_disp_x, d_free_cnt_tng_disp_y,
             d_free_cnt_tng_disp_z, d_free_cnt_tng_disp_x0,
             d_free_cnt_tng_disp_y0, d_free_cnt_tng_disp_z0, d_free_cnt_idx,
             d_free_cnt_idx_dem_id, d_total_no_free_cnt, d_dem_id, d_limit,
             d_wx, d_wy, d_wz, d_torx, d_tory, d_torz, VIJ, XIJ, RIJ, d_rad_s,
             s_idx, s_m, s_rad_s, s_dem_id, s_wx, s_wy, s_wz, s_yng_mod,
             s_shear_mod, s_poisson, dt):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        overlap = -1.

        if RIJ > 0:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        # ---------- force computation starts ------------
        # if particles are overlapping
        if overlap > 0:

            # define the normal vector
            # this is different in different papers. The
            # first paper on DEM by Cundall uses normal vector passing from particle
            # i to j (here d_idx to s_idx), but it is implemented in 2d. Here we use
            # implementation of Luding [1]. Where the normal vector taken from
            # particle j to i (here s_idx to d_idx)
            # in PySPH: XIJ[0] = d_x[d_idx] - s_x[s_idx]
            rinv = 1.0 / RIJ
            nx = XIJ[0] * rinv
            ny = XIJ[1] * rinv
            nz = XIJ[2] * rinv

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
            wcn_x = wijy * nz - wijz * ny
            wcn_y = wijz * nx - wijx * nz
            wcn_z = wijx * ny - wijy * nx

            vr_x = VIJ[0] + wcn_x
            vr_y = VIJ[1] + wcn_y
            vr_z = VIJ[2] + wcn_z

            # normal velocity magnitude
            vn = vr_x * nx + vr_y * ny + vr_z * nz
            vn_x = vn * nx
            vn_y = vn * ny
            vn_z = vn * nz

            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y
            vt_z = vr_z - vn_z
            # magnitude of the tangential velocity
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

            # damping force is taken from
            # "On the Determination of the Damping Coefficient
            # of Non-linear Spring-dashpot System to Model
            # Hertz Contact for Simulation by Discrete Element
            # Method" paper.
            # compute the damping constants
            m_eff = d_m[d_idx] * s_m[s_idx] / (d_m[d_idx] + s_m[s_idx])
            eta_n = self.alpha * sqrt(m_eff)

            # compute the stiffness
            if self.kn == -1:
                r_eff = (d_rad_s[d_idx] * s_rad_s[s_idx]) / (d_rad_s[d_idx] +
                                                             s_rad_s[s_idx])
                E_eff = (d_yng_mod[d_idx] * s_yng_mod[s_idx]) / (
                    d_yng_mod[d_idx] *
                    (1. - s_poisson[s_idx]**2.) + s_yng_mod[s_idx] *
                    (1. - d_poisson[d_idx]**2.))

                kn = 4. / 3. * E_eff * r_eff**0.5 * overlap**0.5

                G_eff = (d_shear_mod[d_idx] * s_shear_mod[s_idx]) / (
                    d_shear_mod[d_idx] *
                    (2. - s_poisson[s_idx]) + s_shear_mod[s_idx] *
                    (2. - d_poisson[d_idx]))

                kt = 16. / 3. * G_eff * r_eff**0.5 * overlap**0.5
            else:
                kn = self.kn
                kt = self.kt

            # --------------------------------------------------------------
            # --------------------------------------------------------------
            # Normal force
            kn_overlap = kn * overlap
            fn_x = kn_overlap * nx - eta_n * vn_x
            fn_y = kn_overlap * ny - eta_n * vn_y
            fn_z = kn_overlap * nz - eta_n * vn_z

            # --------------------------------------------------------------
            # --------------------------------------------------------------
            # ------------- tangential force computation ----------------
            # total number of contacts of particle i in destination
            tot_ctcs = d_total_no_free_cnt[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_limit[0]
            # ending index is q -1
            q1 = p + tot_ctcs

            # check if the particle is in the tracking list
            # if so, then save the location at found_at
            found = 0
            for j in range(p, q1):
                if s_idx == d_free_cnt_idx[j]:
                    if s_dem_id[s_idx] == d_free_cnt_idx_dem_id[j]:
                        found_at = j
                        found = 1
                        break
            # if the particle is not been tracked then assign an index in
            # tracking history.
            if found == 0:
                found_at = q1
                d_free_cnt_idx[found_at] = s_idx
                d_total_no_free_cnt[d_idx] += 1
                d_free_cnt_idx_dem_id[found_at] = s_dem_id[s_idx]

            # compute the damping constants
            eta_t = 0.5 * eta_n

            # rotate the tangential spring to the current plane
            tmp = (d_free_cnt_tng_disp_x[found_at] * nx +
                   d_free_cnt_tng_disp_y[found_at] * ny +
                   d_free_cnt_tng_disp_z[found_at] * nz)

            d_free_cnt_tng_disp_x[found_at] = (
                d_free_cnt_tng_disp_x[found_at] - tmp * nx)
            d_free_cnt_tng_disp_y[found_at] = (
                d_free_cnt_tng_disp_y[found_at] - tmp * ny)
            d_free_cnt_tng_disp_z[found_at] = (
                d_free_cnt_tng_disp_z[found_at] - tmp * nz)

            # find the tangential force from the tangential displacement
            # and tangential velocity
            ft0_x = -kt * d_free_cnt_tng_disp_x[found_at] - eta_t * vt_x
            ft0_y = -kt * d_free_cnt_tng_disp_y[found_at] - eta_t * vt_y
            ft0_z = -kt * d_free_cnt_tng_disp_z[found_at] - eta_t * vt_z

            # (*) check against Coulomb criterion
            # Tangential force magnitude due to displacement
            ft0 = (ft0_x * ft0_x + ft0_y * ft0_y + ft0_z * ft0_z)**(0.5)
            fn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

            # we have to compare with static friction, so
            # this mu has to be static friction coefficient
            fn_mu = self.mu * fn

            # if the tangential force magnitude is zero, then do nothing,
            # else do following
            if ft0 > 1e-9:
                # compare tangential force with the static friction
                if ft0 >= fn_mu:
                    # rescale the tangential displacement
                    # find the unit direction in tangential velocity
                    # TODO: ELIMINATE THE SINGULARITY CASE
                    # here you also use tangential spring as direction.
                    tx = ft0_x / ft0
                    ty = ft0_y / ft0
                    tz = ft0_z / ft0

                    # this taken from Luding paper [2], eq (21)
                    d_free_cnt_tng_disp_x[found_at] = -1. / kt * (fn_mu * tx +
                                                                  eta_t * vt_x)
                    d_free_cnt_tng_disp_y[found_at] = -1. / kt * (fn_mu * ty +
                                                                  eta_t * vt_x)
                    d_free_cnt_tng_disp_z[found_at] = -1. / kt * (fn_mu * tz +
                                                                  eta_t * vt_x)

                    # and also adjust the spring elongation
                    # at time t, which is used at stage 2 integrator
                    d_free_cnt_tng_disp_x0[found_at] = d_free_cnt_tng_disp_x[
                        found_at]
                    d_free_cnt_tng_disp_y0[found_at] = d_free_cnt_tng_disp_y[
                        found_at]
                    d_free_cnt_tng_disp_z0[found_at] = d_free_cnt_tng_disp_z[
                        found_at]

                    # set the tangential force to static friction
                    # from Coulomb criterion
                    ft0_x = fn_mu * tx
                    ft0_y = fn_mu * ty
                    ft0_z = fn_mu * tz

            dtb2 = dt / 2.
            d_free_cnt_tng_disp_x[found_at] += self.kt * vt_x * dtb2
            d_free_cnt_tng_disp_y[found_at] += self.kt * vt_y * dtb2
            d_free_cnt_tng_disp_z[found_at] += self.kt * vt_z * dtb2

            d_fx[d_idx] += fn_x + ft0_x
            d_fy[d_idx] += fn_y + ft0_y
            d_fz[d_idx] += fn_z + ft0_z

            # torque = n cross F
            d_torx[d_idx] += (-ny * ft0_z - -nz * ft0_y) * a_i
            d_tory[d_idx] += (-nz * ft0_x - -nx * ft0_z) * a_i
            d_torz[d_idx] += (-nx * ft0_y - -ny * ft0_x) * a_i


class DEMParticleParticleForceStage2(Equation):
    """
    This equation computes the force between two discrete DEM particles.
    This equation implements both linear and nonlinear DEM. It is written from
    many sources [1][2][3].

    [1] Introduction to DEM Luding.
    """
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(
            abs(kn)) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(DEMParticleParticleForceStage2, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_yng_mod, d_shear_mod, d_poisson, d_fx, d_fy,
             d_fz, d_free_cnt_tng_disp_x, d_free_cnt_tng_disp_y,
             d_free_cnt_tng_disp_z, d_free_cnt_tng_disp_x0,
             d_free_cnt_tng_disp_y0, d_free_cnt_tng_disp_z0, d_free_cnt_idx,
             d_free_cnt_idx_dem_id, d_total_no_free_cnt, d_dem_id, d_limit,
             d_wx, d_wy, d_wz, d_torx, d_tory, d_torz, VIJ, XIJ, RIJ, d_rad_s,
             s_idx, s_m, s_rad_s, s_dem_id, s_wx, s_wy, s_wz, s_yng_mod,
             s_shear_mod, s_poisson, dt):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        overlap = -1.

        if RIJ > 0:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        # ---------- force computation starts ------------
        # if particles are overlapping
        if overlap > 0:

            # define the normal vector
            # this is different in different papers. The
            # first paper on DEM by Cundall uses normal vector passing from particle
            # i to j (here d_idx to s_idx), but it is implemented in 2d. Here we use
            # implementation of Luding [1]. Where the normal vector taken from
            # particle j to i (here s_idx to d_idx)
            # in PySPH: XIJ[0] = d_x[d_idx] - s_x[s_idx]
            rinv = 1.0 / RIJ
            nx = XIJ[0] * rinv
            ny = XIJ[1] * rinv
            nz = XIJ[2] * rinv

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
            wcn_x = wijy * nz - wijz * ny
            wcn_y = wijz * nx - wijx * nz
            wcn_z = wijx * ny - wijy * nx

            vr_x = VIJ[0] + wcn_x
            vr_y = VIJ[1] + wcn_y
            vr_z = VIJ[2] + wcn_z

            # normal velocity magnitude
            vn = vr_x * nx + vr_y * ny + vr_z * nz
            vn_x = vn * nx
            vn_y = vn * ny
            vn_z = vn * nz

            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y
            vt_z = vr_z - vn_z
            # magnitude of the tangential velocity
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

            # damping force is taken from
            # "On the Determination of the Damping Coefficient
            # of Non-linear Spring-dashpot System to Model
            # Hertz Contact for Simulation by Discrete Element
            # Method" paper.
            # compute the damping constants
            m_eff = d_m[d_idx] * s_m[s_idx] / (d_m[d_idx] + s_m[s_idx])
            eta_n = self.alpha * sqrt(m_eff)

            # compute the stiffness
            if self.kn == -1:
                r_eff = (d_rad_s[d_idx] * s_rad_s[s_idx]) / (d_rad_s[d_idx] +
                                                             s_rad_s[s_idx])
                E_eff = (d_yng_mod[d_idx] * s_yng_mod[s_idx]) / (
                    d_yng_mod[d_idx] *
                    (1. - s_poisson[s_idx]**2.) + s_yng_mod[s_idx] *
                    (1. - d_poisson[d_idx]**2.))

                kn = 4. / 3. * E_eff * r_eff**0.5 * overlap**0.5

                G_eff = (d_shear_mod[d_idx] * s_shear_mod[s_idx]) / (
                    d_shear_mod[d_idx] *
                    (2. - s_poisson[s_idx]) + s_shear_mod[s_idx] *
                    (2. - d_poisson[d_idx]))

                kt = 16. / 3. * G_eff * r_eff**0.5 * overlap**0.5
            else:
                kn = self.kn
                kt = self.kt

            # --------------------------------------------------------------
            # --------------------------------------------------------------
            # Normal force
            kn_overlap = kn * overlap
            fn_x = kn_overlap * nx - eta_n * vn_x
            fn_y = kn_overlap * ny - eta_n * vn_y
            fn_z = kn_overlap * nz - eta_n * vn_z

            # --------------------------------------------------------------
            # --------------------------------------------------------------
            # ------------- tangential force computation ----------------
            # total number of contacts of particle i in destination
            tot_ctcs = d_total_no_free_cnt[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_limit[0]
            # ending index is q -1
            q1 = p + tot_ctcs

            # check if the particle is in the tracking list
            # if so, then save the location at found_at
            found = 0
            for j in range(p, q1):
                if s_idx == d_free_cnt_idx[j]:
                    if s_dem_id[s_idx] == d_free_cnt_idx_dem_id[j]:
                        found_at = j
                        found = 1
                        break

            ft0_x = 0.
            ft0_y = 0.
            ft0_z = 0.
            # don't compute the tangential force if not tracked
            if found == 1:
                # compute the damping constants
                eta_t = 0.5 * eta_n

                # rotate the tangential spring to the current plane
                tmp = (d_free_cnt_tng_disp_x[found_at] * nx +
                       d_free_cnt_tng_disp_y[found_at] * ny +
                       d_free_cnt_tng_disp_z[found_at] * nz)

                d_free_cnt_tng_disp_x[found_at] = (
                    d_free_cnt_tng_disp_x[found_at] - tmp * nx)
                d_free_cnt_tng_disp_y[found_at] = (
                    d_free_cnt_tng_disp_y[found_at] - tmp * ny)
                d_free_cnt_tng_disp_z[found_at] = (
                    d_free_cnt_tng_disp_z[found_at] - tmp * nz)

                # also rotate the tangential spring at time stage 1 to the
                # current plane
                tmp = (d_free_cnt_tng_disp_x0[found_at] * nx +
                       d_free_cnt_tng_disp_y0[found_at] * ny +
                       d_free_cnt_tng_disp_z0[found_at] * nz)

                d_free_cnt_tng_disp_x0[found_at] = (
                    d_free_cnt_tng_disp_x0[found_at] - tmp * nx)
                d_free_cnt_tng_disp_y0[found_at] = (
                    d_free_cnt_tng_disp_y0[found_at] - tmp * ny)
                d_free_cnt_tng_disp_z0[found_at] = (
                    d_free_cnt_tng_disp_z0[found_at] - tmp * nz)

                # find the tangential force from the tangential displacement
                # and tangential velocity
                ft0_x = -kt * d_free_cnt_tng_disp_x[found_at] - eta_t * vt_x
                ft0_y = -kt * d_free_cnt_tng_disp_y[found_at] - eta_t * vt_y
                ft0_z = -kt * d_free_cnt_tng_disp_z[found_at] - eta_t * vt_z

                # (*) check against Coulomb criterion
                # Tangential force magnitude due to displacement
                ft0 = (ft0_x * ft0_x + ft0_y * ft0_y + ft0_z * ft0_z)**(0.5)
                fn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

                # we have to compare with static friction, so
                # this mu has to be static friction coefficient
                fn_mu = self.mu * fn

                # if the tangential force magnitude is zero, then do nothing,
                # else do following
                if ft0 > 1e-9:
                    # compare tangential force with the static friction
                    if ft0 >= fn_mu:
                        # rescale the tangential displacement
                        # find the unit direction in tangential velocity
                        # TODO: ELIMINATE THE SINGULARITY CASE
                        # here you also use tangential spring as direction.
                        tx = ft0_x / ft0
                        ty = ft0_y / ft0
                        tz = ft0_z / ft0

                        # set the tangential force to static friction
                        # from Coulomb criterion
                        ft0_x = fn_mu * tx
                        ft0_y = fn_mu * ty
                        ft0_z = fn_mu * tz

                d_free_cnt_tng_disp_x[found_at] = (
                    d_free_cnt_tng_disp_x0[found_at] + self.kt * vt_x * dt)
                d_free_cnt_tng_disp_y[found_at] = (
                    d_free_cnt_tng_disp_y0[found_at] + self.kt * vt_y * dt)
                d_free_cnt_tng_disp_z[found_at] = (
                    d_free_cnt_tng_disp_z0[found_at] + self.kt * vt_z * dt)

            d_fx[d_idx] += fn_x + ft0_x
            d_fy[d_idx] += fn_y + ft0_y
            d_fz[d_idx] += fn_z + ft0_z

            # torque = n cross F
            d_torx[d_idx] += (-ny * ft0_z - -nz * ft0_y) * a_i
            d_tory[d_idx] += (-nz * ft0_x - -nx * ft0_z) * a_i
            d_torz[d_idx] += (-nx * ft0_y - -ny * ft0_x) * a_i


class UpdateFreeContactsWithParticles(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_z, d_rad_s,
                        d_total_no_free_cnt, d_free_cnt_idx, d_limit,
                        d_free_cnt_tng_disp_x, d_free_cnt_tng_disp_y,
                        d_free_cnt_tng_disp_z, d_free_cnt_idx_dem_id,
                        d_free_cnt_tng_disp_x0, d_free_cnt_tng_disp_y0,
                        d_free_cnt_tng_disp_z0, s_x, s_y, s_z, s_rad_s,
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
        idx_total_ctcs = d_total_no_free_cnt[d_idx]
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
            sidx = d_free_cnt_idx[k]
            # get the dem id of the particle
            dem_id = d_free_cnt_idx_dem_id[k]

            if sidx == -1:
                break
            else:
                if dem_id == s_dem_id[sidx]:
                    xij[0] = d_x[d_idx] - s_x[sidx]
                    xij[1] = d_y[d_idx] - s_y[sidx]
                    xij[2] = d_z[d_idx] - s_z[sidx]
                    rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] +
                               xij[2] * xij[2])

                    overlap = d_rad_s[d_idx] + s_rad_s[sidx] - rij

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_free_cnt_idx[k] = -1
                            d_free_cnt_idx_dem_id[k] = -1
                            d_free_cnt_tng_disp_x[k] = 0.
                            d_free_cnt_tng_disp_y[k] = 0.
                            d_free_cnt_tng_disp_z[k] = 0.
                            # make tangential0 displacements zero
                            d_free_cnt_tng_disp_x0[k] = 0.
                            d_free_cnt_tng_disp_y0[k] = 0.
                            d_free_cnt_tng_disp_z0[k] = 0.
                        else:
                            # swap the current tracking index with the final
                            # contact index
                            d_free_cnt_idx[k] = d_free_cnt_idx[last_idx_tmp]
                            d_free_cnt_idx[last_idx_tmp] = -1

                            # swap tangential x displacement
                            d_free_cnt_tng_disp_x[k] = d_free_cnt_tng_disp_x[
                                last_idx_tmp]
                            d_free_cnt_tng_disp_x[last_idx_tmp] = 0.

                            # swap tangential y displacement
                            d_free_cnt_tng_disp_y[k] = d_free_cnt_tng_disp_y[
                                last_idx_tmp]
                            d_free_cnt_tng_disp_y[last_idx_tmp] = 0.

                            # swap tangential z displacement
                            d_free_cnt_tng_disp_z[k] = d_free_cnt_tng_disp_z[
                                last_idx_tmp]
                            d_free_cnt_tng_disp_z[last_idx_tmp] = 0.

                            # swap tangential idx dem id
                            d_free_cnt_idx_dem_id[k] = d_free_cnt_idx_dem_id[
                                last_idx_tmp]
                            d_free_cnt_idx_dem_id[last_idx_tmp] = -1

                            # make tangential0 displacements zero
                            d_free_cnt_tng_disp_x0[last_idx_tmp] = 0.
                            d_free_cnt_tng_disp_y0[last_idx_tmp] = 0.
                            d_free_cnt_tng_disp_z0[last_idx_tmp] = 0.

                            # decrease the last_idx_tmp, since we swapped it to
                            # -1
                            last_idx_tmp -= 1

                        # decrement the total contacts of the particle
                        d_total_no_free_cnt[d_idx] -= 1
                    else:
                        k = k + 1
                else:
                    k = k + 1
                count += 1


class DEMParticleInfinityWallForceStage1(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha_1 = - tmp * sqrt(5. / (tmp**2. + pi**2.))

        super(DEMParticleInfinityWallForceStage1, self).__init__(dest, sources)

    def initialize_pair(self, d_idx, d_m, d_u, d_v, d_w, d_x, d_y, d_z, d_fx,
                        d_fy, d_fz, d_free_cnt_tng_disp_x,
                        d_free_cnt_tng_disp_y, d_free_cnt_tng_disp_z,
                        d_free_cnt_tng_disp_x0, d_free_cnt_tng_disp_y0,
                        d_free_cnt_tng_disp_z0, d_free_cnt_idx,
                        d_free_cnt_idx_dem_id, d_total_no_free_cnt, d_dem_id,
                        d_limit, d_wx, d_wy, d_wz, d_torx, d_tory, d_torz,
                        d_rad_s, d_yng_mod, d_shear_mod, d_poisson, s_x, s_y,
                        s_z, s_nx, s_ny, s_nz, s_dem_id, s_np, s_yng_mod,
                        s_shear_mod, s_poisson, dt):
        i, n = declare('int', 2)
        xij = declare('matrix(3)')
        vij = declare('matrix(3)')
        kn = self.kn
        kt = self.kt
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        n = s_np[0]
        for i in range(n):
            # Force calculation starts
            overlap = -1.
            xij[0] = d_x[d_idx] - s_x[i]
            xij[1] = d_y[d_idx] - s_y[i]
            xij[2] = d_z[d_idx] - s_z[i]
            overlap = d_rad_s[d_idx] - (xij[0] * s_nx[i] + xij[1] * s_ny[i] +
                                        xij[2] * s_nz[i])

            if overlap > 0:
                # basic variables: normal vector
                # vector passing from wall to particle
                # j to i (s_idx to d_idx)
                nx = s_nx[i]
                ny = s_ny[i]
                nz = s_nz[i]

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
                wcn_x = wijy * nz - wijz * ny
                wcn_y = wijz * nx - wijx * nz
                wcn_z = wijx * ny - wijy * nx

                vij[0] = d_u[d_idx]
                vij[1] = d_v[d_idx]
                vij[2] = d_w[d_idx]
                vr_x = vij[0] + wcn_x
                vr_y = vij[1] + wcn_y
                vr_z = vij[2] + wcn_z

                # normal velocity magnitude
                vn = vr_x * nx + vr_y * ny + vr_z * nz
                vn_x = vn * nx
                vn_y = vn * ny
                vn_z = vn * nz

                # tangential velocity
                vt_x = vr_x - vn_x
                vt_y = vr_y - vn_y
                vt_z = vr_z - vn_z
                # magnitude of the tangential velocity
                # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

                # compute the stiffness
                Ed = d_yng_mod[d_idx]
                Gd = d_shear_mod[d_idx]
                pd = d_poisson[d_idx]
                rd = d_rad_s[d_idx]
                if self.kn == -1:
                    r_eff = rd

                    E_eff = Ed / (1. - pd**2.)
                    G_eff = Gd / (2. - pd)

                    kn_tmp = 4. / 3. * E_eff * sqrt(r_eff)
                    kn = 4. / 3. * E_eff * sqrt(r_eff * overlap)
                    kt_tmp = 16. / 3. * G_eff * sqrt(r_eff)
                    kt = 16. / 3. * G_eff * sqrt(r_eff * overlap)
                    kt_1 = 1. / kt

                    # tsuiji paper gave
                    # E_eff = Ed * Es / (Ed * (1. - ps**2.) + Es * (1. - pd**2.))
                    # G_eff = Gd * Gs / (Gd * (2. - ps) + Gs * (2. - pd))
                else:
                    kn = self.kn
                    kt = self.kt

                # damping force is taken from
                # https://ieeexplore.ieee.org/document/5571553
                # "On the Determination of the Damping Coefficient
                # of Non-linear Spring-dashpot System to Model
                # Hertz Contact for Simulation by Discrete Element
                # Method" paper.
                # compute the damping constants
                m_eff = d_m[d_idx]
                if self.kn == -1:
                    eta_n = self.alpha_1 * sqrt(m_eff * kn_tmp) * overlap**0.25
                else:
                    eta_n = 1.

                # --------------------------------------------------------------
                # --------------------------------------------------------------
                # Normal force
                kn_overlap = kn * overlap
                fn_x = kn_overlap * nx - eta_n * vn_x
                fn_y = kn_overlap * ny - eta_n * vn_y
                fn_z = kn_overlap * nz - eta_n * vn_z

                # ------------- tangential force computation ----------------
                # total number of contacts of particle i in destination
                tot_ctcs = d_total_no_free_cnt[d_idx]

                # d_idx has a range of tracking indices with sources
                # starting index is p
                p = d_idx * d_limit[0]
                # ending index is q -1
                q1 = p + tot_ctcs

                # check if the particle is in the tracking list
                # if so, then save the location at found_at
                found = 0
                for j in range(p, q1):
                    if i == d_free_cnt_idx[j]:
                        if s_dem_id[i] == d_free_cnt_idx_dem_id[j]:
                            found_at = j
                            found = 1
                            break

                # if the particle is not been tracked then assign an index in
                # tracking history.
                if found == 0:
                    found_at = q1
                    d_free_cnt_idx[found_at] = i
                    d_total_no_free_cnt[d_idx] += 1
                    d_free_cnt_idx_dem_id[found_at] = s_dem_id[i]

                # compute the damping constants
                eta_t = 0.5 * eta_n

                # rotate the tangential spring to the current plane
                tmp = (d_free_cnt_tng_disp_x[found_at] * nx +
                       d_free_cnt_tng_disp_y[found_at] * ny +
                       d_free_cnt_tng_disp_z[found_at] * nz)

                d_free_cnt_tng_disp_x[found_at] = (
                    d_free_cnt_tng_disp_x[found_at] - tmp * nx)
                d_free_cnt_tng_disp_y[found_at] = (
                    d_free_cnt_tng_disp_y[found_at] - tmp * ny)
                d_free_cnt_tng_disp_z[found_at] = (
                    d_free_cnt_tng_disp_z[found_at] - tmp * nz)

                # find the tangential force from the tangential displacement
                # and tangential velocity
                ft0_x = -kt * d_free_cnt_tng_disp_x[found_at] - eta_t * vt_x
                ft0_y = -kt * d_free_cnt_tng_disp_y[found_at] - eta_t * vt_y
                ft0_z = -kt * d_free_cnt_tng_disp_z[found_at] - eta_t * vt_z

                # (*) check against Coulomb criterion
                # Tangential force magnitude due to displacement
                ft0 = (ft0_x * ft0_x + ft0_y * ft0_y + ft0_z * ft0_z)**(0.5)
                fn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

                # we have to compare with static friction, so
                # this mu has to be static friction coefficient
                fn_mu = self.mu * fn

                # if the tangential force magnitude is zero, then do nothing,
                # else do following
                if ft0 > 1e-9:
                    # compare tangential force with the static friction
                    if ft0 >= fn_mu:
                        # rescale the tangential displacement
                        # find the unit direction in tangential velocity
                        # TODO: ELIMINATE THE SINGULARITY CASE
                        # here you also use tangential spring as direction.
                        tx = ft0_x / ft0
                        ty = ft0_y / ft0
                        tz = ft0_z / ft0

                        # this taken from Luding paper [2], eq (21)
                        d_free_cnt_tng_disp_x[found_at] = -1. / kt * (
                            fn_mu * tx + eta_t * vt_x)
                        d_free_cnt_tng_disp_y[found_at] = -1. / kt * (
                            fn_mu * ty + eta_t * vt_x)
                        d_free_cnt_tng_disp_z[found_at] = -1. / kt * (
                            fn_mu * tz + eta_t * vt_x)

                        # and also adjust the spring elongation
                        # at time t, which is used at stage 2 integrator
                        d_free_cnt_tng_disp_x0[
                            found_at] = d_free_cnt_tng_disp_x[found_at]
                        d_free_cnt_tng_disp_y0[
                            found_at] = d_free_cnt_tng_disp_y[found_at]
                        d_free_cnt_tng_disp_z0[
                            found_at] = d_free_cnt_tng_disp_z[found_at]

                        # set the tangential force to static friction
                        # from Coulomb criterion
                        ft0_x = fn_mu * tx
                        ft0_y = fn_mu * ty
                        ft0_z = fn_mu * tz

                dtb2 = dt / 2.
                d_free_cnt_tng_disp_x[found_at] += vt_x * dtb2
                d_free_cnt_tng_disp_y[found_at] += vt_y * dtb2
                d_free_cnt_tng_disp_z[found_at] += vt_z * dtb2

                d_fx[d_idx] += fn_x + ft0_x
                d_fy[d_idx] += fn_y + ft0_y
                d_fz[d_idx] += fn_z + ft0_z

                # frictional torque
                tor_fric_x = 0.
                tor_fric_y = 0.
                tor_fric_z = 0.
                # omega_magn = (
                #     d_wx[d_idx]**2. + d_wy[d_idx]**2. + d_wz[d_idx]**2.)**0.5

                # if omega_magn > 0.:
                #     omega_nx = d_wx[d_idx] / omega_magn
                #     omega_ny = d_wy[d_idx] / omega_magn
                #     omega_nz = d_wz[d_idx] / omega_magn
                #     tor_fric_x = -self.mu * fn_magn * omega_nx
                #     tor_fric_y = -self.mu * fn_magn * omega_ny
                #     tor_fric_z = -self.mu * fn_magn * omega_nz

                # torque = n cross F
                d_torx[d_idx] += (-ny * ft0_z - -nz * ft0_y) * a_d + tor_fric_x
                d_tory[d_idx] += (-nz * ft0_x - -nx * ft0_z) * a_d + tor_fric_y
                d_torz[d_idx] += (-nx * ft0_y - -ny * ft0_x) * a_d + tor_fric_z


class DEMParticleInfinityWallForceStage2(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha_1 = - tmp * sqrt(5. / (tmp**2. + pi**2.))
        super(DEMParticleInfinityWallForceStage2, self).__init__(dest, sources)

    def initialize_pair(self, d_idx, d_m, d_u, d_v, d_w, d_x, d_y, d_z, d_fx,
                        d_fy, d_fz, d_free_cnt_tng_disp_x,
                        d_free_cnt_tng_disp_y, d_free_cnt_tng_disp_z,
                        d_free_cnt_tng_disp_x0, d_free_cnt_tng_disp_y0,
                        d_free_cnt_tng_disp_z0, d_free_cnt_idx,
                        d_free_cnt_idx_dem_id, d_total_no_free_cnt, d_dem_id,
                        d_limit, d_wx, d_wy, d_wz, d_torx, d_tory, d_torz,
                        d_rad_s, d_yng_mod, d_shear_mod, d_poisson, s_x, s_y,
                        s_z, s_nx, s_ny, s_nz, s_dem_id, s_np, s_yng_mod,
                        s_shear_mod, s_poisson, dt):
        i, n = declare('int', 2)
        xij = declare('matrix(3)')
        vij = declare('matrix(3)')
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        n = s_np[0]
        for i in range(n):
            # Force calculation starts
            overlap = -1.
            xij[0] = d_x[d_idx] - s_x[i]
            xij[1] = d_y[d_idx] - s_y[i]
            xij[2] = d_z[d_idx] - s_z[i]
            overlap = d_rad_s[d_idx] - (xij[0] * s_nx[i] + xij[1] * s_ny[i] +
                                        xij[2] * s_nz[i])

            if overlap > 0:
                # basic variables: normal vector
                # vector passing from wall to particle
                # j to i (s_idx to d_idx)
                nx = s_nx[i]
                ny = s_ny[i]
                nz = s_nz[i]

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
                wcn_x = wijy * nz - wijz * ny
                wcn_y = wijz * nx - wijx * nz
                wcn_z = wijx * ny - wijy * nx

                vij[0] = d_u[d_idx]
                vij[1] = d_v[d_idx]
                vij[2] = d_w[d_idx]
                vr_x = vij[0] + wcn_x
                vr_y = vij[1] + wcn_y
                vr_z = vij[2] + wcn_z

                # normal velocity magnitude
                vn = vr_x * nx + vr_y * ny + vr_z * nz
                vn_x = vn * nx
                vn_y = vn * ny
                vn_z = vn * nz

                # tangential velocity
                vt_x = vr_x - vn_x
                vt_y = vr_y - vn_y
                vt_z = vr_z - vn_z
                # magnitude of the tangential velocity
                # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

                # compute the stiffness
                Ed = d_yng_mod[d_idx]
                Gd = d_shear_mod[d_idx]
                pd = d_poisson[d_idx]
                rd = d_rad_s[d_idx]
                if self.kn == -1:
                    r_eff = rd

                    E_eff = Ed / (1. - pd**2.)
                    G_eff = Gd / (2. - pd)

                    kn_tmp = 4. / 3. * E_eff * sqrt(r_eff)
                    kn = 4. / 3. * E_eff * sqrt(r_eff * overlap)
                    kt_tmp = 16. / 3. * G_eff * sqrt(r_eff)
                    kt = 16. / 3. * G_eff * sqrt(r_eff * overlap)
                    kt_1 = 1. / kt

                    # tsuiji paper gave
                    # E_eff = Ed * Es / (Ed * (1. - ps**2.) + Es * (1. - pd**2.))
                    # G_eff = Gd * Gs / (Gd * (2. - ps) + Gs * (2. - pd))
                else:
                    kn = self.kn
                    kt = self.kt

                # damping force is taken from
                # "On the Determination of the Damping Coefficient
                # of Non-linear Spring-dashpot System to Model
                # Hertz Contact for Simulation by Discrete Element
                # Method" paper.
                # compute the damping constants
                m_eff = d_m[d_idx]
                if self.kn == -1:
                    eta_n = self.alpha_1 * sqrt(m_eff * kn_tmp) * overlap**0.25
                else:
                    eta_n = 1.

                # normal force
                kn_overlap = kn * overlap
                fn_x = kn_overlap * nx - eta_n * vn_x
                fn_y = kn_overlap * ny - eta_n * vn_y
                fn_z = kn_overlap * nz - eta_n * vn_z

                # ------------- tangential force computation ----------------
                # total number of contacts of particle i in destination
                tot_ctcs = d_total_no_free_cnt[d_idx]

                # d_idx has a range of tracking indices with sources
                # starting index is p
                p = d_idx * d_limit[0]
                # ending index is q -1
                q1 = p + tot_ctcs

                # check if the particle is in the tracking list
                # if so, then save the location at found_at
                found = 0
                for j in range(p, q1):
                    if i == d_free_cnt_idx[j]:
                        if s_dem_id[i] == d_free_cnt_idx_dem_id[j]:
                            found_at = j
                            found = 1
                            break

                ft0_x = 0.
                ft0_y = 0.
                ft0_z = 0.
                # don't compute the tangential force if not tracked
                if found == 1:
                    # compute the damping constants
                    eta_t = 0.5 * eta_n

                    # rotate the tangential spring to the current plane
                    tmp = (d_free_cnt_tng_disp_x[found_at] * nx +
                           d_free_cnt_tng_disp_y[found_at] * ny +
                           d_free_cnt_tng_disp_z[found_at] * nz)

                    d_free_cnt_tng_disp_x[found_at] = (
                        d_free_cnt_tng_disp_x[found_at] - tmp * nx)
                    d_free_cnt_tng_disp_y[found_at] = (
                        d_free_cnt_tng_disp_y[found_at] - tmp * ny)
                    d_free_cnt_tng_disp_z[found_at] = (
                        d_free_cnt_tng_disp_z[found_at] - tmp * nz)

                    # also rotate the tangential spring at time stage 1 to the
                    # current plane
                    tmp = (d_free_cnt_tng_disp_x0[found_at] * nx +
                           d_free_cnt_tng_disp_y0[found_at] * ny +
                           d_free_cnt_tng_disp_z0[found_at] * nz)

                    d_free_cnt_tng_disp_x0[found_at] = (
                        d_free_cnt_tng_disp_x0[found_at] - tmp * nx)
                    d_free_cnt_tng_disp_y0[found_at] = (
                        d_free_cnt_tng_disp_y0[found_at] - tmp * ny)
                    d_free_cnt_tng_disp_z0[found_at] = (
                        d_free_cnt_tng_disp_z0[found_at] - tmp * nz)

                    # find the tangential force from the tangential displacement
                    # and tangential velocity
                    ft0_x = (-kt * d_free_cnt_tng_disp_x[found_at] -
                             eta_t * vt_x)
                    ft0_y = (-kt * d_free_cnt_tng_disp_y[found_at] -
                             eta_t * vt_y)
                    ft0_z = (-kt * d_free_cnt_tng_disp_z[found_at] -
                             eta_t * vt_z)

                    # (*) check against Coulomb criterion
                    # Tangential force magnitude due to displacement
                    ft0 = (ft0_x * ft0_x + ft0_y * ft0_y +
                           ft0_z * ft0_z)**(0.5)
                    fn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

                    # we have to compare with static friction, so
                    # this mu has to be static friction coefficient
                    fn_mu = self.mu * fn

                    # if the tangential force magnitude is zero, then do nothing,
                    # else do following
                    if ft0 > 1e-9:
                        # compare tangential force with the static friction
                        if ft0 >= fn_mu:
                            # rescale the tangential displacement
                            # find the unit direction in tangential velocity
                            # TODO: ELIMINATE THE SINGULARITY CASE
                            # here you also use tangential spring as direction.
                            tx = ft0_x / ft0
                            ty = ft0_y / ft0
                            tz = ft0_z / ft0

                            # set the tangential force to static friction
                            # from Coulomb criterion
                            ft0_x = fn_mu * tx
                            ft0_y = fn_mu * ty
                            ft0_z = fn_mu * tz

                    d_free_cnt_tng_disp_x[found_at] = (
                        d_free_cnt_tng_disp_x0[found_at] + vt_x * dt)
                    d_free_cnt_tng_disp_y[found_at] = (
                        d_free_cnt_tng_disp_y0[found_at] + vt_y * dt)
                    d_free_cnt_tng_disp_z[found_at] = (
                        d_free_cnt_tng_disp_z0[found_at] + vt_z * dt)

                d_fx[d_idx] += fn_x + ft0_x
                d_fy[d_idx] += fn_y + ft0_y
                d_fz[d_idx] += fn_z + ft0_z

                # frictional torque
                tor_fric_x = 0.
                tor_fric_y = 0.
                tor_fric_z = 0.
                # omega_magn = (
                #     d_wx[d_idx]**2. + d_wy[d_idx]**2. + d_wz[d_idx]**2.)**0.5

                # if omega_magn > 0.:
                #     omega_nx = d_wx[d_idx] / omega_magn
                #     omega_ny = d_wy[d_idx] / omega_magn
                #     omega_nz = d_wz[d_idx] / omega_magn
                #     tor_fric_x = -self.mu * fn_magn * omega_nx
                #     tor_fric_y = -self.mu * fn_magn * omega_ny
                #     tor_fric_z = -self.mu * fn_magn * omega_nz

                # torque = n cross F
                d_torx[d_idx] += (-ny * ft0_z - -nz * ft0_y) * a_d + tor_fric_x
                d_tory[d_idx] += (-nz * ft0_x - -nx * ft0_z) * a_d + tor_fric_y
                d_torz[d_idx] += (-nx * ft0_y - -ny * ft0_x) * a_d + tor_fric_z


class UpdateFreeContactsWithInfinityWall(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_z, d_rad_s,
                        d_total_no_free_cnt, d_free_cnt_idx, d_limit,
                        d_free_cnt_tng_disp_x, d_free_cnt_tng_disp_y,
                        d_free_cnt_tng_disp_z, d_free_cnt_idx_dem_id,
                        d_free_cnt_tng_disp_x0, d_free_cnt_tng_disp_y0,
                        d_free_cnt_tng_disp_z0, s_x, s_y, s_z, s_nx, s_ny,
                        s_nz, s_dem_id):
        p = declare('int')
        count = declare('int')
        k = declare('int')
        xij = declare('matrix(3)')
        last_idx_tmp = declare('int')
        sidx = declare('int')
        dem_id = declare('int')

        idx_total_ctcs = declare('int')
        idx_total_ctcs = d_total_no_free_cnt[d_idx]
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
            sidx = d_free_cnt_idx[k]
            # get the dem id of the particle
            dem_id = d_free_cnt_idx_dem_id[k]

            if sidx == -1:
                break
            else:
                if dem_id == s_dem_id[sidx]:
                    xij[0] = d_x[d_idx] - s_x[sidx]
                    xij[1] = d_y[d_idx] - s_y[sidx]
                    xij[2] = d_z[d_idx] - s_z[sidx]
                    overlap = d_rad_s[d_idx] - (xij[0] * s_nx[sidx] +
                                                xij[1] * s_ny[sidx] +
                                                xij[2] * s_nz[sidx])

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_free_cnt_idx[k] = -1
                            d_free_cnt_idx_dem_id[k] = -1
                            d_free_cnt_tng_disp_x[k] = 0.
                            d_free_cnt_tng_disp_y[k] = 0.
                            d_free_cnt_tng_disp_z[k] = 0.
                            # make tangential0 displacements zero
                            d_free_cnt_tng_disp_x0[k] = 0.
                            d_free_cnt_tng_disp_y0[k] = 0.
                            d_free_cnt_tng_disp_z0[k] = 0.
                        else:
                            # swap the current tracking index with the final
                            # contact index
                            d_free_cnt_idx[k] = d_free_cnt_idx[last_idx_tmp]
                            d_free_cnt_idx[last_idx_tmp] = -1

                            # swap tangential x displacement
                            d_free_cnt_tng_disp_x[k] = d_free_cnt_tng_disp_x[
                                last_idx_tmp]
                            d_free_cnt_tng_disp_x[last_idx_tmp] = 0.

                            # swap tangential y displacement
                            d_free_cnt_tng_disp_y[k] = d_free_cnt_tng_disp_y[
                                last_idx_tmp]
                            d_free_cnt_tng_disp_y[last_idx_tmp] = 0.

                            # swap tangential z displacement
                            d_free_cnt_tng_disp_z[k] = d_free_cnt_tng_disp_z[
                                last_idx_tmp]
                            d_free_cnt_tng_disp_z[last_idx_tmp] = 0.

                            # swap tangential idx dem id
                            d_free_cnt_idx_dem_id[k] = d_free_cnt_idx_dem_id[
                                last_idx_tmp]
                            d_free_cnt_idx_dem_id[last_idx_tmp] = -1

                            # make tangential0 displacements zero
                            d_free_cnt_tng_disp_x0[last_idx_tmp] = 0.
                            d_free_cnt_tng_disp_y0[last_idx_tmp] = 0.
                            d_free_cnt_tng_disp_z0[last_idx_tmp] = 0.

                            # decrease the last_idx_tmp, since we swapped it to
                            # -1
                            last_idx_tmp -= 1

                        # decrement the total contacts of the particle
                        d_total_no_free_cnt[d_idx] -= 1
                    else:
                        k = k + 1
                else:
                    k = k + 1
                count += 1


class RK2StepDEM(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v, d_w,
                   d_u0, d_v0, d_w0, d_wx, d_wy, d_wz, d_wx0, d_wy0, d_wz0,
                   d_total_no_free_cnt, d_limit, d_free_cnt_tng_disp_x,
                   d_free_cnt_tng_disp_y, d_free_cnt_tng_disp_z,
                   d_free_cnt_tng_disp_x0, d_free_cnt_tng_disp_y0,
                   d_free_cnt_tng_disp_z0):

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
        tot_ctcs = d_total_no_free_cnt[d_idx]
        p = d_idx * d_limit[0]
        q = p + tot_ctcs

        for i in range(p, q):
            d_free_cnt_tng_disp_x0[i] = d_free_cnt_tng_disp_x[i]
            d_free_cnt_tng_disp_y0[i] = d_free_cnt_tng_disp_y[i]
            d_free_cnt_tng_disp_z0[i] = d_free_cnt_tng_disp_z[i]

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_x0, d_y0, d_z0, d_u0, d_v0, d_w0, d_wx0, d_wy0, d_wz0, d_torx,
               d_tory, d_torz, d_wx, d_wy, d_wz, d_m_inverse, d_I_inverse, dt):
        dtb2 = dt / 2.

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_fy[d_idx] * d_m_inverse[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_fz[d_idx] * d_m_inverse[d_idx]

        d_wx[d_idx] = d_wx0[d_idx] + dtb2 * d_torx[d_idx] * d_I_inverse[d_idx]
        d_wy[d_idx] = d_wy0[d_idx] + dtb2 * d_tory[d_idx] * d_I_inverse[d_idx]
        d_wz[d_idx] = d_wz0[d_idx] + dtb2 * d_torz[d_idx] * d_I_inverse[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_x0, d_y0, d_z0, d_u0, d_v0, d_w0, d_wx0, d_wy0, d_wz0, d_torx,
               d_tory, d_torz, d_wx, d_wy, d_wz, d_m_inverse, d_I_inverse, dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_fy[d_idx] * d_m_inverse[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_fz[d_idx] * d_m_inverse[d_idx]

        d_wx[d_idx] = d_wx0[d_idx] + dt * d_torx[d_idx] * d_I_inverse[d_idx]
        d_wy[d_idx] = d_wy0[d_idx] + dt * d_tory[d_idx] * d_I_inverse[d_idx]
        d_wz[d_idx] = d_wz0[d_idx] + dt * d_torz[d_idx] * d_I_inverse[d_idx]
