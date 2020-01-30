"""
This is a straight implementation of DEM from Cundall 1979 paper.
This is only applicable to 2d problems. This doesn't have rolling friction
physics. Details follow in particle array and corresponding equations.
"""
from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from compyle.api import declare
from math import sqrt, log
import numpy as np
from pysph.sph.scheme import Scheme
from pysph.sph.equation import Group, MultiStageEquations
from pysph.dem.discontinuous_dem.dem_nonlinear import EPECIntegratorMultiStage


def get_particle_array_cundall_dem_from_paper(constants=None, **props):
    """Return a particle array for a dem particles
    """
    dem_props = [
        'wz', 'wz0', 'fx', 'fy', 'fz', 'torz', 'rad_s', 'm_inverse',
        'I_inverse', 'u0', 'v0', 'x0', 'y0'
    ]

    dem_id = props.pop('dem_id', None)

    pa = get_particle_array(additional_props=dem_props, **props)

    pa.add_property('dem_id', type='int', data=dem_id)
    # create the array to save the tangential interaction particles
    # index and other variables
    free_cnt_limit = 6

    pa.add_constant('free_cnt_limit', free_cnt_limit)
    pa.add_property('free_cnt_idx', stride=free_cnt_limit, type="int")
    pa.free_cnt_idx[:] = -1
    pa.add_property('free_cnt_idx_dem_id', stride=free_cnt_limit, type="int")
    pa.free_cnt_idx_dem_id[:] = -1

    # forces due to intermediate contacts
    pa.add_property('free_cnt_fn', stride=free_cnt_limit)
    pa.add_property('free_cnt_fn0', stride=free_cnt_limit)
    pa.add_property('free_cnt_ft', stride=free_cnt_limit)
    pa.add_property('free_cnt_ft0', stride=free_cnt_limit)

    pa.add_property('free_cnt_fn_x', stride=free_cnt_limit)
    pa.add_property('free_cnt_fn_y', stride=free_cnt_limit)
    pa.add_property('free_cnt_fn_z', stride=free_cnt_limit)
    pa.add_property('free_cnt_ft_x', stride=free_cnt_limit)
    pa.add_property('free_cnt_ft_y', stride=free_cnt_limit)
    pa.add_property('free_cnt_ft_z', stride=free_cnt_limit)

    # increment of forces due to intermediate contacts
    pa.add_property('free_cnt_delta_fn', stride=free_cnt_limit)
    pa.add_property('free_cnt_delta_fn_x', stride=free_cnt_limit)
    pa.add_property('free_cnt_delta_fn_y', stride=free_cnt_limit)
    pa.add_property('free_cnt_delta_fn_z', stride=free_cnt_limit)

    pa.add_property('free_cnt_delta_ft', stride=free_cnt_limit)
    pa.add_property('free_cnt_delta_ft_x', stride=free_cnt_limit)
    pa.add_property('free_cnt_delta_ft_y', stride=free_cnt_limit)
    pa.add_property('free_cnt_delta_ft_z', stride=free_cnt_limit)

    pa.add_property('total_no_free_cnt', type="int")
    pa.total_no_free_cnt[:] = 0

    pa.set_output_arrays([
        'x', 'y', 'u', 'v', 'm', 'pid', 'tag', 'gid', 'fx', 'fy', 'fz', 'torz',
        'I_inverse', 'wz', 'm_inverse', 'rad_s', 'dem_id', 'free_cnt_idx',
        'free_cnt_idx_dem_id', 'total_no_free_cnt', 'free_cnt_fn',
        'free_cnt_fn0', 'free_cnt_fn_x', 'free_cnt_fn_y', 'free_cnt_fn_z',
        'free_cnt_ft', 'free_cnt_ft0', 'free_cnt_ft_x', 'free_cnt_ft_y',
        'free_cnt_ft_z', 'free_cnt_delta_fn_x', 'free_cnt_delta_fn_y',
        'free_cnt_delta_fn_z', 'free_cnt_delta_ft_x', 'free_cnt_delta_ft_y',
        'free_cnt_delta_ft_z'
    ])

    return pa


class BodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0):
        self.gx = gx
        self.gy = gy
        super(BodyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_fx, d_fy, d_torz):
        d_fx[d_idx] = d_m[d_idx] * self.gx
        d_fy[d_idx] = d_m[d_idx] * self.gy
        d_torz[d_idx] = 0.


class CundallFromPaperForceParticleParticleStage1(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(CundallFromPaperForceParticleParticleStage1,
              self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_x, d_y, d_u, d_v, d_fx, d_fy, d_wz,
             d_free_cnt_idx, d_free_cnt_idx_dem_id, d_total_no_free_cnt,
             d_dem_id, d_free_cnt_limit, d_torz, RIJ, d_rad_s, d_free_cnt_fn,
             d_free_cnt_fn_x, d_free_cnt_fn_y, d_free_cnt_ft, d_free_cnt_ft_x,
             d_free_cnt_ft_y, d_free_cnt_delta_fn_x, d_free_cnt_delta_fn_y,
             d_free_cnt_delta_ft_x, d_free_cnt_delta_ft_y, s_idx, s_m, s_rad_s,
             s_dem_id, s_wz, s_x, s_y, s_u, s_v, dt):
        p, q1, tot_cnts, j, found_at, found = declare('int', 6)
        overlap = -1.
        dtb2 = dt / 2.

        # check the particles are not on top of each other.
        if RIJ > 0:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        # ---------- force computation starts ------------
        # if particles are overlapping
        if overlap > 0:
            rinv = 1.0 / RIJ

            # normal vector passes from d_idx to s_idx
            nx = (s_x[s_idx] - d_x[d_idx]) * rinv
            ny = (s_y[s_idx] - d_y[d_idx]) * rinv

            # the tangential direction is taken to be a vector
            # rotated 90 degrees clock wise of normal vector.
            tx = ny
            ty = -nx

            # the relative velocity of particle s_idx with respect to d_idx
            vij_x = d_u[d_idx] - s_u[s_idx] - (
                d_wz[d_idx] * d_rad_s[d_idx] +
                s_wz[s_idx] * s_rad_s[s_idx]) * tx
            vij_y = d_v[d_idx] - s_v[s_idx] - (
                d_wz[d_idx] * d_rad_s[d_idx] +
                s_wz[s_idx] * s_rad_s[s_idx]) * ty

            # scalar components of relative velocity in normal and
            # tangential directions
            vn = vij_x * nx + vij_y * ny
            vt = vij_x * tx + vij_y * ty

            delta_dn = vn * dtb2
            delta_dt = vt * dtb2

            # # vector components of relative normal velocity
            # vn_x = vn * nx
            # vn_y = vn * ny

            # # vector components of relative tangential velocity
            # vt_x = vij_x - vn_x
            # vt_y = vij_y - vn_y

            # ------------- force computation -----------------------
            # total number of contacts of particle i in destination
            tot_cnts = d_total_no_free_cnt[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_free_cnt_limit[0]
            # ending index is q -1
            q1 = p + tot_cnts

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

            # ------------- Normal force computation ----------------
            # from the relative normal velocity we know the relative
            # displacement and from which the net increment of the
            # normal force is computed as
            delta_fn = self.kn * delta_dn

            # similarly the scalar magnitude of tangential force
            delta_ft = self.kt * delta_dt

            # assign it to the particle for post_processing
            d_free_cnt_delta_fn_x[found_at] = delta_fn * -nx
            d_free_cnt_delta_fn_y[found_at] = delta_fn * -ny

            d_free_cnt_delta_ft_x[found_at] = delta_ft * -tx
            d_free_cnt_delta_ft_y[found_at] = delta_ft * -ty

            # before adding the increment to the tracking force variable
            # add the normal force due to the contact s_idx to the particle
            # d_idx first
            d_fx[d_idx] += (d_free_cnt_fn[found_at] * -nx +
                            d_free_cnt_ft[found_at] * -tx)
            d_fy[d_idx] += (d_free_cnt_fn[found_at] * -ny +
                            d_free_cnt_ft[found_at] * -ty)

            # increment the scalar normal force and tangential force
            d_free_cnt_fn[found_at] += delta_fn
            d_free_cnt_ft[found_at] += delta_ft

            # check for Coulomb friction
            fs_max = self.mu * abs(d_free_cnt_fn[found_at])
            # get the sign of the tangential force
            if d_free_cnt_ft[found_at] > 0.:
                sign = 1.
            else:
                sign = -1.

            if abs(d_free_cnt_ft[found_at]) > fs_max:
                d_free_cnt_ft[found_at] = fs_max * sign

            # set the direction and magnitude of the contact bond normal force
            d_free_cnt_fn_x[found_at] = d_free_cnt_fn[found_at] * -nx
            d_free_cnt_fn_y[found_at] = d_free_cnt_fn[found_at] * -ny
            d_free_cnt_ft_x[found_at] = d_free_cnt_ft[found_at] * -tx
            d_free_cnt_ft_y[found_at] = d_free_cnt_ft[found_at] * -ty


class CundallFromPaperForceParticleParticleStage2(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(CundallFromPaperForceParticleParticleStage2,
              self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_x, d_y, d_u, d_v, d_fx, d_fy, d_wz,
             d_free_cnt_idx, d_free_cnt_idx_dem_id, d_total_no_free_cnt,
             d_dem_id, d_free_cnt_limit, d_torz, RIJ, d_rad_s, d_free_cnt_fn,
             d_free_cnt_fn_x, d_free_cnt_fn_y, d_free_cnt_ft, d_free_cnt_ft_x,
             d_free_cnt_ft_y, d_free_cnt_delta_fn_x, d_free_cnt_delta_fn_y,
             d_free_cnt_delta_ft_x, d_free_cnt_delta_ft_y, s_idx, s_m, s_rad_s,
             s_dem_id, s_wz, s_x, s_y, s_u, s_v, dt, d_free_cnt_fn0,
             d_free_cnt_ft0):
        p, q1, tot_cnts, j, found_at, found = declare('int', 6)
        overlap = -1.

        # check the particles are not on top of each other.
        if RIJ > 0:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        # ---------- force computation starts ------------
        # if particles are overlapping
        if overlap > 0:
            rinv = 1.0 / RIJ

            # normal vector passes from d_idx to s_idx
            nx = (s_x[s_idx] - d_x[d_idx]) * rinv
            ny = (s_y[s_idx] - d_y[d_idx]) * rinv

            # the tangential direction is taken to be a vector
            # rotated 90 degrees clock wise of normal vector.
            tx = ny
            ty = -nx

            # the relative velocity of particle s_idx with respect to d_idx
            vij_x = d_u[d_idx] - s_u[s_idx] - (
                d_wz[d_idx] * d_rad_s[d_idx] +
                s_wz[s_idx] * s_rad_s[s_idx]) * tx
            vij_y = d_v[d_idx] - s_v[s_idx] - (
                d_wz[d_idx] * d_rad_s[d_idx] +
                s_wz[s_idx] * s_rad_s[s_idx]) * ty

            # scalar components of relative velocity in normal and
            # tangential directions
            vn = vij_x * nx + vij_y * ny
            vt = vij_x * tx + vij_y * ty

            delta_dn = vn * dt
            delta_dt = vt * dt

            # # vector components of relative normal velocity
            # vn_x = vn * nx
            # vn_y = vn * ny

            # # vector components of relative tangential velocity
            # vt_x = vij_x - vn_x
            # vt_y = vij_y - vn_y

            # ------------- force computation -----------------------
            # total number of contacts of particle i in destination
            tot_cnts = d_total_no_free_cnt[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_free_cnt_limit[0]
            # ending index is q -1
            q1 = p + tot_cnts

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

            # ------------- Normal force computation ----------------
            # from the relative normal velocity we know the relative
            # displacement and from which the net increment of the
            # normal force is computed as
            delta_fn = self.kn * delta_dn

            # similarly the scalar magnitude of tangential force
            delta_ft = self.kt * delta_dt

            # assign it to the particle for post_processing
            d_free_cnt_delta_fn_x[found_at] = delta_fn * -nx
            d_free_cnt_delta_fn_y[found_at] = delta_fn * -ny

            d_free_cnt_delta_ft_x[found_at] = delta_ft * -tx
            d_free_cnt_delta_ft_y[found_at] = delta_ft * -ty

            # before adding the increment to the tracking force variable
            # add the normal force due to the contact s_idx to the particle
            # d_idx first
            d_fx[d_idx] += (d_free_cnt_fn[found_at] * -nx +
                            d_free_cnt_ft[found_at] * -tx)
            d_fy[d_idx] += (d_free_cnt_fn[found_at] * -ny +
                            d_free_cnt_ft[found_at] * -ty)

            # increment the scalar normal force and tangential force
            d_free_cnt_fn[found_at] = d_free_cnt_fn0[found_at] + delta_fn
            d_free_cnt_ft[found_at] = d_free_cnt_ft0[found_at] + delta_ft

            # check for Coulomb friction
            fs_max = self.mu * abs(d_free_cnt_fn[found_at])
            # get the sign of the tangential force
            if d_free_cnt_ft[found_at] > 0.:
                sign = 1.
            else:
                sign = -1.

            if abs(d_free_cnt_ft[found_at]) > fs_max:
                d_free_cnt_ft[found_at] = fs_max * sign

            # set the direction and magnitude of the contact bond normal force
            d_free_cnt_fn_x[found_at] = d_free_cnt_fn[found_at] * -nx
            d_free_cnt_fn_y[found_at] = d_free_cnt_fn[found_at] * -ny
            d_free_cnt_ft_x[found_at] = d_free_cnt_ft[found_at] * -tx
            d_free_cnt_ft_y[found_at] = d_free_cnt_ft[found_at] * -ty


class CundallFromPaperForceParticleInfiniteWallStage1(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(CundallFromPaperForceParticleInfiniteWallStage1,
              self).__init__(dest, sources)

    def initialize_pair(self, d_idx, d_m, d_x, d_y, d_u, d_v, d_fx, d_fy, d_wz,
                        d_free_cnt_idx, d_free_cnt_idx_dem_id,
                        d_total_no_free_cnt, d_dem_id, d_free_cnt_limit,
                        d_torz, RIJ, d_rad_s, d_free_cnt_fn, d_free_cnt_fn_x,
                        d_free_cnt_fn_y, d_free_cnt_ft, d_free_cnt_ft_x,
                        d_free_cnt_ft_y, d_free_cnt_delta_fn_x,
                        d_free_cnt_delta_fn_y, d_free_cnt_delta_ft_x,
                        d_free_cnt_delta_ft_y, s_idx, s_m, s_dem_id, s_x, s_y,
                        s_u, s_v, dt):
        p, q1, tot_cnts, j, found_at, found = declare('int', 6)
        overlap = -1.
        dtb2 = dt / 2.

        # check the particles are not on top of each other.
        if RIJ > 0:
            overlap = d_rad_s[d_idx] + -RIJ

        # ---------- force computation starts ------------
        # if particles are overlapping
        if overlap > 0:
            rinv = 1.0 / RIJ

            # normal vector passes from d_idx to s_idx
            nx = (s_x[s_idx] - d_x[d_idx]) * rinv
            ny = (s_y[s_idx] - d_y[d_idx]) * rinv

            # the tangential direction is taken to be a vector
            # rotated 90 degrees clock wise of normal vector.
            tx = ny
            ty = -nx

            # the relative velocity of particle s_idx with respect to d_idx
            vij_x = d_u[d_idx] - s_u[s_idx] - (d_wz[d_idx] *
                                               d_rad_s[d_idx]) * tx
            vij_y = d_v[d_idx] - s_v[s_idx] - (d_wz[d_idx] *
                                               d_rad_s[d_idx]) * ty

            # scalar components of relative velocity in normal and
            # tangential directions
            vn = vij_x * nx + vij_y * ny
            vt = vij_x * tx + vij_y * ty

            delta_dn = vn * dtb2
            delta_dt = vt * dtb2

            # # vector components of relative normal velocity
            # vn_x = vn * nx
            # vn_y = vn * ny

            # # vector components of relative tangential velocity
            # vt_x = vij_x - vn_x
            # vt_y = vij_y - vn_y

            # ------------- force computation -----------------------
            # total number of contacts of particle i in destination
            tot_cnts = d_total_no_free_cnt[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_free_cnt_limit[0]
            # ending index is q -1
            q1 = p + tot_cnts

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

            # ------------- Normal force computation ----------------
            # from the relative normal velocity we know the relative
            # displacement and from which the net increment of the
            # normal force is computed as
            delta_fn = self.kn * delta_dn

            # similarly the scalar magnitude of tangential force
            delta_ft = self.kt * delta_dt

            # assign it to the particle for post_processing
            d_free_cnt_delta_fn_x[found_at] = delta_fn * -nx
            d_free_cnt_delta_fn_y[found_at] = delta_fn * -ny

            d_free_cnt_delta_ft_x[found_at] = delta_ft * -tx
            d_free_cnt_delta_ft_y[found_at] = delta_ft * -ty

            # before adding the increment to the tracking force variable
            # add the normal force due to the contact s_idx to the particle
            # d_idx first
            d_fx[d_idx] += (d_free_cnt_fn[found_at] * -nx +
                            d_free_cnt_ft[found_at] * -tx)
            d_fy[d_idx] += (d_free_cnt_fn[found_at] * -ny +
                            d_free_cnt_ft[found_at] * -ty)

            # increment the scalar normal force and tangential force
            d_free_cnt_fn[found_at] += delta_fn
            d_free_cnt_ft[found_at] += delta_ft

            # check for Coulomb friction
            fs_max = self.mu * abs(d_free_cnt_fn[found_at])
            # get the sign of the tangential force
            if d_free_cnt_ft[found_at] > 0.:
                sign = 1.
            else:
                sign = -1.

            if abs(d_free_cnt_ft[found_at]) > fs_max:
                d_free_cnt_ft[found_at] = fs_max * sign

            # set the direction and magnitude of the contact bond normal force
            d_free_cnt_fn_x[found_at] = d_free_cnt_fn[found_at] * -nx
            d_free_cnt_fn_y[found_at] = d_free_cnt_fn[found_at] * -ny
            d_free_cnt_ft_x[found_at] = d_free_cnt_ft[found_at] * -tx
            d_free_cnt_ft_y[found_at] = d_free_cnt_ft[found_at] * -ty


class CundallFromPaperForceParticleInfiniteWallStage2(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(CundallFromPaperForceParticleInfiniteWallStage2,
              self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_x, d_y, d_u, d_v, d_fx, d_fy, d_wz,
             d_free_cnt_idx, d_free_cnt_idx_dem_id, d_total_no_free_cnt,
             d_dem_id, d_free_cnt_limit, d_torz, RIJ, d_rad_s, d_free_cnt_fn,
             d_free_cnt_fn_x, d_free_cnt_fn_y, d_free_cnt_ft, d_free_cnt_ft_x,
             d_free_cnt_ft_y, d_free_cnt_delta_fn_x, d_free_cnt_delta_fn_y,
             d_free_cnt_delta_ft_x, d_free_cnt_delta_ft_y, s_idx, s_m, s_rad_s,
             s_dem_id, s_wz, s_x, s_y, s_u, s_v, dt, d_free_cnt_fn0,
             d_free_cnt_ft0):
        p, q1, tot_cnts, j, found_at, found = declare('int', 6)
        overlap = -1.

        # check the particles are not on top of each other.
        if RIJ > 0:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        # ---------- force computation starts ------------
        # if particles are overlapping
        if overlap > 0:
            rinv = 1.0 / RIJ

            # normal vector passes from d_idx to s_idx
            nx = (s_x[s_idx] - d_x[d_idx]) * rinv
            ny = (s_y[s_idx] - d_y[d_idx]) * rinv

            # the tangential direction is taken to be a vector
            # rotated 90 degrees clock wise of normal vector.
            tx = ny
            ty = -nx

            # the relative velocity of particle s_idx with respect to d_idx
            vij_x = d_u[d_idx] - s_u[s_idx] - (
                d_wz[d_idx] * d_rad_s[d_idx] +
                s_wz[s_idx] * s_rad_s[s_idx]) * tx
            vij_y = d_v[d_idx] - s_v[s_idx] - (
                d_wz[d_idx] * d_rad_s[d_idx] +
                s_wz[s_idx] * s_rad_s[s_idx]) * ty

            # scalar components of relative velocity in normal and
            # tangential directions
            vn = vij_x * nx + vij_y * ny
            vt = vij_x * tx + vij_y * ty

            delta_dn = vn * dt
            delta_dt = vt * dt

            # # vector components of relative normal velocity
            # vn_x = vn * nx
            # vn_y = vn * ny

            # # vector components of relative tangential velocity
            # vt_x = vij_x - vn_x
            # vt_y = vij_y - vn_y

            # ------------- force computation -----------------------
            # total number of contacts of particle i in destination
            tot_cnts = d_total_no_free_cnt[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_free_cnt_limit[0]
            # ending index is q -1
            q1 = p + tot_cnts

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

            # ------------- Normal force computation ----------------
            # from the relative normal velocity we know the relative
            # displacement and from which the net increment of the
            # normal force is computed as
            delta_fn = self.kn * delta_dn

            # similarly the scalar magnitude of tangential force
            delta_ft = self.kt * delta_dt

            # assign it to the particle for post_processing
            d_free_cnt_delta_fn_x[found_at] = delta_fn * -nx
            d_free_cnt_delta_fn_y[found_at] = delta_fn * -ny

            d_free_cnt_delta_ft_x[found_at] = delta_ft * -tx
            d_free_cnt_delta_ft_y[found_at] = delta_ft * -ty

            # before adding the increment to the tracking force variable
            # add the normal force due to the contact s_idx to the particle
            # d_idx first
            d_fx[d_idx] += (d_free_cnt_fn[found_at] * -nx +
                            d_free_cnt_ft[found_at] * -tx)
            d_fy[d_idx] += (d_free_cnt_fn[found_at] * -ny +
                            d_free_cnt_ft[found_at] * -ty)

            # increment the scalar normal force and tangential force
            d_free_cnt_fn[found_at] = d_free_cnt_fn0[found_at] + delta_fn
            d_free_cnt_ft[found_at] = d_free_cnt_ft0[found_at] + delta_ft

            # check for Coulomb friction
            fs_max = self.mu * abs(d_free_cnt_fn[found_at])
            # get the sign of the tangential force
            if d_free_cnt_ft[found_at] > 0.:
                sign = 1.
            else:
                sign = -1.

            if abs(d_free_cnt_ft[found_at]) > fs_max:
                d_free_cnt_ft[found_at] = fs_max * sign

            # set the direction and magnitude of the contact bond normal force
            d_free_cnt_fn_x[found_at] = d_free_cnt_fn[found_at] * -nx
            d_free_cnt_fn_y[found_at] = d_free_cnt_fn[found_at] * -ny
            d_free_cnt_ft_x[found_at] = d_free_cnt_ft[found_at] * -tx
            d_free_cnt_ft_y[found_at] = d_free_cnt_ft[found_at] * -ty


class UpdateFreeContactsCundallFromPaperPaticleParticle(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_rad_s, d_total_no_free_cnt,
                        d_free_cnt_idx, d_free_cnt_limit,
                        d_free_cnt_idx_dem_id, d_free_cnt_fn, d_free_cnt_ft,
                        d_free_cnt_fn0, d_free_cnt_ft0, d_free_cnt_delta_fn_x,
                        d_free_cnt_delta_fn_y, d_free_cnt_delta_ft_x,
                        d_free_cnt_delta_ft_y, s_x, s_y, s_rad_s, s_dem_id):
        p = declare('int')
        count = declare('int')
        k = declare('int')
        xij = declare('matrix(3)')
        last_idx_tmp = declare('int')
        sidx = declare('int')
        dem_id = declare('int')
        rij = 0.0

        idx_total_cnts = declare('int')
        idx_total_cnts = d_total_no_free_cnt[d_idx]

        # particle idx contacts has range of indices
        # and the first index would be
        p = d_idx * d_free_cnt_limit[0]
        last_idx_tmp = p + idx_total_cnts - 1
        k = p
        count = 0

        # loop over all the contacts of particle d_idx
        while count < idx_total_cnts:
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
                    rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

                    overlap = d_rad_s[d_idx] + s_rad_s[sidx] - rij

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_free_cnt_idx[k] = -1
                            d_free_cnt_idx_dem_id[k] = -1
                            d_free_cnt_fn[k] = 0.
                            d_free_cnt_ft[k] = 0.

                            # make increments to zero
                            d_free_cnt_delta_fn_x[k] = 0.
                            d_free_cnt_delta_fn_y[k] = 0.
                            d_free_cnt_delta_ft_x[k] = 0.
                            d_free_cnt_delta_ft_y[k] = 0.
                        else:
                            # swap the current tracking index with the final
                            # contact index
                            d_free_cnt_idx[k] = d_free_cnt_idx[last_idx_tmp]
                            d_free_cnt_idx[last_idx_tmp] = -1

                            # swap free contact normal and tangential forces
                            d_free_cnt_fn[k] = d_free_cnt_fn[last_idx_tmp]
                            d_free_cnt_ft[k] = d_free_cnt_ft[last_idx_tmp]

                            d_free_cnt_fn[last_idx_tmp] = 0.
                            d_free_cnt_ft[last_idx_tmp] = 0.

                            # swap free contact normal and tangential increment
                            # forces
                            d_free_cnt_delta_fn_x[k] = d_free_cnt_delta_fn_x[
                                last_idx_tmp]
                            d_free_cnt_delta_fn_y[k] = d_free_cnt_delta_fn_y[
                                last_idx_tmp]
                            d_free_cnt_delta_ft_x[k] = d_free_cnt_delta_ft_x[
                                last_idx_tmp]
                            d_free_cnt_delta_ft_y[k] = d_free_cnt_delta_ft_y[
                                last_idx_tmp]

                            d_free_cnt_delta_fn_x[last_idx_tmp] = 0.
                            d_free_cnt_delta_fn_y[last_idx_tmp] = 0.
                            d_free_cnt_delta_ft_x[last_idx_tmp] = 0.
                            d_free_cnt_delta_ft_y[last_idx_tmp] = 0.

                            # swap tangential idx dem id
                            d_free_cnt_idx_dem_id[k] = d_free_cnt_idx_dem_id[
                                last_idx_tmp]
                            d_free_cnt_idx_dem_id[last_idx_tmp] = -1

                            # make tangential0 displacements zero
                            d_free_cnt_fn0[last_idx_tmp] = 0.
                            d_free_cnt_ft0[last_idx_tmp] = 0.

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


class UpdateFreeContactsCundallFromPaperPaticleInfiniteWall(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_rad_s, d_total_no_free_cnt,
                        d_free_cnt_idx, d_free_cnt_limit,
                        d_free_cnt_idx_dem_id, d_free_cnt_fn, d_free_cnt_ft,
                        d_free_cnt_fn0, d_free_cnt_ft0, d_free_cnt_delta_fn_x,
                        d_free_cnt_delta_fn_y, d_free_cnt_delta_ft_x,
                        d_free_cnt_delta_ft_y, s_x, s_y, s_rad_s, s_dem_id):
        p = declare('int')
        count = declare('int')
        k = declare('int')
        xij = declare('matrix(3)')
        last_idx_tmp = declare('int')
        sidx = declare('int')
        dem_id = declare('int')
        rij = 0.0

        idx_total_cnts = declare('int')
        idx_total_cnts = d_total_no_free_cnt[d_idx]

        # particle idx contacts has range of indices
        # and the first index would be
        p = d_idx * d_free_cnt_limit[0]
        last_idx_tmp = p + idx_total_cnts - 1
        k = p
        count = 0

        # loop over all the contacts of particle d_idx
        while count < idx_total_cnts:
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
                    rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

                    overlap = d_rad_s[d_idx] + s_rad_s[sidx] - rij

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_free_cnt_idx[k] = -1
                            d_free_cnt_idx_dem_id[k] = -1
                            d_free_cnt_fn[k] = 0.
                            d_free_cnt_ft[k] = 0.

                            # make increments to zero
                            d_free_cnt_delta_fn_x[k] = 0.
                            d_free_cnt_delta_fn_y[k] = 0.
                            d_free_cnt_delta_ft_x[k] = 0.
                            d_free_cnt_delta_ft_y[k] = 0.
                        else:
                            # swap the current tracking index with the final
                            # contact index
                            d_free_cnt_idx[k] = d_free_cnt_idx[last_idx_tmp]
                            d_free_cnt_idx[last_idx_tmp] = -1

                            # swap free contact normal and tangential forces
                            d_free_cnt_fn[k] = d_free_cnt_fn[last_idx_tmp]
                            d_free_cnt_ft[k] = d_free_cnt_ft[last_idx_tmp]

                            d_free_cnt_fn[last_idx_tmp] = 0.
                            d_free_cnt_ft[last_idx_tmp] = 0.

                            # swap free contact normal and tangential increment
                            # forces
                            d_free_cnt_delta_fn_x[k] = d_free_cnt_delta_fn_x[
                                last_idx_tmp]
                            d_free_cnt_delta_fn_y[k] = d_free_cnt_delta_fn_y[
                                last_idx_tmp]
                            d_free_cnt_delta_ft_x[k] = d_free_cnt_delta_ft_x[
                                last_idx_tmp]
                            d_free_cnt_delta_ft_y[k] = d_free_cnt_delta_ft_y[
                                last_idx_tmp]

                            d_free_cnt_delta_fn_x[last_idx_tmp] = 0.
                            d_free_cnt_delta_fn_y[last_idx_tmp] = 0.
                            d_free_cnt_delta_ft_x[last_idx_tmp] = 0.
                            d_free_cnt_delta_ft_y[last_idx_tmp] = 0.

                            # swap tangential idx dem id
                            d_free_cnt_idx_dem_id[k] = d_free_cnt_idx_dem_id[
                                last_idx_tmp]
                            d_free_cnt_idx_dem_id[last_idx_tmp] = -1

                            # make tangential0 displacements zero
                            d_free_cnt_fn0[last_idx_tmp] = 0.
                            d_free_cnt_ft0[last_idx_tmp] = 0.

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


class RK2StepDEMCundallFromPaper(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_x0, d_y0, d_u, d_v, d_u0, d_v0,
                   d_wz, d_wz0, d_total_no_free_cnt, d_free_cnt_limit,
                   d_free_cnt_fn, d_free_cnt_fn0, d_free_cnt_ft,
                   d_free_cnt_ft0):

        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]

        d_wz0[d_idx] = d_wz[d_idx]

        # -----------------------------------------------
        # save the initial tangential contact information
        # -----------------------------------------------
        i = declare('int')
        p = declare('int')
        q = declare('int')
        tot_cnts = declare('int')
        tot_cnts = d_total_no_free_cnt[d_idx]
        p = d_idx * d_free_cnt_limit[0]
        q = p + tot_cnts

        for i in range(p, q):
            d_free_cnt_fn0[i] = d_free_cnt_fn[i]
            d_free_cnt_ft0[i] = d_free_cnt_ft[i]

    def stage1(self, d_idx, d_x, d_y, d_u, d_v, d_w, d_fx, d_fy, d_x0, d_y0,
               d_u0, d_v0, d_wz, d_wz0, d_torz, d_m_inverse, d_I_inverse, dt):
        dtb2 = dt / 2.

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_fy[d_idx] * d_m_inverse[d_idx]

        d_wz[d_idx] = d_wz0[d_idx] + (dtb2 * d_torz[d_idx] *
                                      d_I_inverse[d_idx])

    def stage2(self, d_idx, d_x, d_y, d_u, d_v, d_fx, d_fy, d_x0, d_y0, d_u0,
               d_v0, d_wz, d_wz0, d_torz, d_m_inverse, d_I_inverse, dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_fy[d_idx] * d_m_inverse[d_idx]

        d_wz[d_idx] = d_wz0[d_idx] + (dt * d_torz[d_idx] * d_I_inverse[d_idx])
