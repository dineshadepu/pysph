from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from compyle.api import declare
from math import sqrt, log
import numpy as np
from pysph.sph.scheme import Scheme
from pysph.sph.equation import Group, MultiStageEquations
from pysph.dem.discontinuous_dem.dem_nonlinear import EPECIntegratorMultiStage

# imports used to analyse the contact information
from pysph.solver.output import load
from pysph.solver.utils import get_files, iter_output
from pysph.base.kernels import CubicSpline


def get_particle_array_dem_2d_linear_cundall_scalar_formulation(
        constants=None, **props):
    """Return a particle array for a dem particles
    """
    dim = props.pop('dim', None)

    dem_props = [
        'wz', 'wz0', 'fx', 'fy', 'fz', 'torz', 'rad_s', 'm_inverse',
        'I_inverse', 'u0', 'v0', 'x0', 'y0'
    ]

    dem_id = props.pop('dem_id', None)

    pa = get_particle_array(additional_props=dem_props, constants=constants,
                            **props)

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

    pa.add_property('total_no_free_cnt', type="int")
    pa.total_no_free_cnt[:] = 0

    pa.set_output_arrays([
        'x', 'y', 'u', 'v', 'm', 'pid', 'tag', 'gid', 'fx', 'fy', 'fz', 'torz',
        'I_inverse', 'wz', 'm_inverse', 'rad_s', 'dem_id', 'free_cnt_idx',
        'free_cnt_idx_dem_id', 'total_no_free_cnt', 'free_cnt_fn',
        'free_cnt_fn0', 'free_cnt_ft', 'free_cnt_ft0'
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


class Cundall2dForceParticleParticleStage1(Equation):
    def __init__(self, dest, sources, mu=0.5, en=0.8):
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        # tmp = log(en)
        # self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall2dForceParticleParticleStage1,
              self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_x, d_y, d_u, d_v, d_fx, d_fy, d_torz, d_wz,
             d_kn, d_kt, d_free_cnt_idx, d_free_cnt_idx_dem_id,
             d_total_no_free_cnt, d_dem_id, d_free_cnt_limit, RIJ, d_rad_s,
             d_free_cnt_fn, d_free_cnt_ft, s_idx, s_m, s_rad_s, s_dem_id, s_wz,
             s_kn, s_kt, s_x, s_y, s_u, s_v, dt):
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

            # find the normal and tangential stiffness
            kn = (d_kn[d_idx] * s_kn[s_idx]) / (d_kn[d_idx] + s_kn[s_idx])
            kt = (d_kt[d_idx] * s_kt[s_idx]) / (d_kt[d_idx] + s_kt[s_idx])

            # ------------- Normal force computation ----------------
            # from the relative normal velocity we know the relative
            # displacement and from which the net increment of the
            # normal force is computed as
            delta_fn = kn * delta_dn

            # similarly the scalar magnitude of tangential force
            delta_ft = kt * delta_dt

            # assign it to the particle for post_processing
            # d_free_cnt_delta_fn_x[found_at] = delta_fn * -nx
            # d_free_cnt_delta_fn_y[found_at] = delta_fn * -ny

            # d_free_cnt_delta_ft_x[found_at] = delta_ft * -tx
            # d_free_cnt_delta_ft_y[found_at] = delta_ft * -ty

            # before adding the increment to the tracking force variable
            # add the normal force due to the contact s_idx to the particle
            # d_idx first
            d_fx[d_idx] += (d_free_cnt_fn[found_at] * -nx +
                            d_free_cnt_ft[found_at] * -tx)
            d_fy[d_idx] += (d_free_cnt_fn[found_at] * -ny +
                            d_free_cnt_ft[found_at] * -ty)
            d_torz[d_idx] += d_free_cnt_ft[found_at] * d_rad_s[d_idx]

            # increment the scalar normal force and tangential force
            d_free_cnt_fn[found_at] += delta_fn
            if d_free_cnt_fn[found_at] < 0.:
                d_free_cnt_fn[found_at] = 0.

            d_free_cnt_ft[found_at] += delta_ft

            # check for Coulomb friction
            fs_max = self.mu * d_free_cnt_fn[found_at]
            # get the sign of the tangential force
            if d_free_cnt_ft[found_at] > 0.:
                sign = 1.
            else:
                sign = -1.

            if abs(d_free_cnt_ft[found_at]) > fs_max:
                d_free_cnt_ft[found_at] = fs_max * sign


class Cundall2dForceParticleParticleStage2(Equation):
    def __init__(self, dest, sources, mu=0.5, en=0.8):
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        # tmp = log(en)
        # self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall2dForceParticleParticleStage2,
              self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_x, d_y, d_u, d_v, d_fx, d_fy, d_torz, d_wz,
             d_kn, d_kt, d_free_cnt_idx, d_free_cnt_idx_dem_id,
             d_total_no_free_cnt, d_dem_id, d_free_cnt_limit, RIJ, d_rad_s,
             d_free_cnt_fn, d_free_cnt_ft, s_idx, s_m, s_rad_s, s_dem_id, s_wz,
             s_kn, s_kt, s_x, s_y, s_u, s_v, dt, d_free_cnt_fn0,
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

            # Already assigned contacts are dealt in stage2. This is for
            # algorithm simplicity. so we don't add new particles
            if found == 1:
                # find the normal and tangential stiffness
                kn = (d_kn[d_idx] * s_kn[s_idx]) / (d_kn[d_idx] + s_kn[s_idx])
                kt = (d_kt[d_idx] * s_kt[s_idx]) / (d_kt[d_idx] + s_kt[s_idx])

                # ------------- Normal force computation ----------------
                # from the relative normal velocity we know the relative
                # displacement and from which the net increment of the
                # normal force is computed as
                delta_fn = kn * delta_dn

                # similarly the scalar magnitude of tangential force
                delta_ft = kt * delta_dt

                # before adding the increment to the tracking force variable
                # add the normal force due to the contact s_idx to the particle
                # d_idx first
                d_fx[d_idx] += (d_free_cnt_fn[found_at] * -nx +
                                d_free_cnt_ft[found_at] * -tx)
                d_fy[d_idx] += (d_free_cnt_fn[found_at] * -ny +
                                d_free_cnt_ft[found_at] * -ty)
                d_torz[d_idx] += d_free_cnt_ft[found_at] * d_rad_s[d_idx]

                # increment the scalar normal force
                d_free_cnt_fn[found_at] = d_free_cnt_fn0[found_at] + delta_fn
                if d_free_cnt_fn[found_at] < 0.:
                    d_free_cnt_fn[found_at] = 0.
                d_free_cnt_ft[found_at] = d_free_cnt_ft0[found_at] + delta_ft

                # check for Coulomb friction
                fs_max = self.mu * d_free_cnt_fn[found_at]
                # get the sign of the tangential force
                if d_free_cnt_ft[found_at] > 0.:
                    sign = 1.
                else:
                    sign = -1.

                if abs(d_free_cnt_ft[found_at]) > fs_max:
                    d_free_cnt_ft[found_at] = fs_max * sign


class UpdateFreeContactsWithParticles(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_rad_s, d_total_no_free_cnt,
                        d_free_cnt_idx, d_free_cnt_limit,
                        d_free_cnt_idx_dem_id, d_free_cnt_fn, d_free_cnt_ft,
                        d_free_cnt_fn0, d_free_cnt_ft0, s_x, s_y, s_rad_s,
                        s_dem_id):
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

                            # make forces0 zero
                            d_free_cnt_fn0[k] = 0.
                            d_free_cnt_ft0[k] = 0.

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

                            # swap tangential idx dem id
                            d_free_cnt_idx_dem_id[k] = d_free_cnt_idx_dem_id[
                                last_idx_tmp]
                            d_free_cnt_idx_dem_id[last_idx_tmp] = -1

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


class Cundall2dForceParticleInfinityWallStage1(Equation):
    def __init__(self, dest, sources, mu=0.5, en=0.8):
        # self.kn = kn
        # self.kt = 2. / 7. * kn
        # self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        # tmp = log(en)
        # self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall2dForceParticleInfinityWallStage1,
              self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_x, d_y, d_u, d_v, d_fx, d_fy, d_wz, d_kn,
             d_kt, d_free_cnt_idx, d_free_cnt_idx_dem_id, d_total_no_free_cnt,
             d_dem_id, d_free_cnt_limit, d_torz, RIJ, d_rad_s, d_free_cnt_fn,
             d_free_cnt_fn_x, d_free_cnt_fn_y, d_free_cnt_ft, d_free_cnt_ft_x,
             d_free_cnt_ft_y, d_free_cnt_delta_fn_x, d_free_cnt_delta_fn_y,
             d_free_cnt_delta_ft_x, d_free_cnt_delta_ft_y, s_x, s_y, s_z, s_nx,
             s_ny, s_nz, s_dem_id, s_np, dt):
        p, q1, tot_cnts, j, found_at, found = declare('int', 6)
        xij = declare('matrix(3)')
        overlap = -1.
        dtb2 = dt / 2.

        n = s_np[0]
        for i in range(n):
            # Force calculation starts
            overlap = -1.
            xij[0] = d_x[d_idx] - s_x[i]
            xij[1] = d_y[d_idx] - s_y[i]
            overlap = d_rad_s[d_idx] - (xij[0] * s_nx[i] + xij[1] * s_ny[i])

            # ---------- force computation starts ------------
            # if particles are overlapping
            if overlap > 0:
                # normal vector passes from d_idx to s_idx
                nx = -s_nx[i]
                ny = -s_ny[i]

                # the tangential direction is taken to be a vector
                # rotated 90 degrees clock wise of normal vector.
                tx = ny
                ty = -nx

                # ##############################################
                # TODO
                # THIS HAS TO BE FIXED
                # TODO
                # the relative velocity of particle s_idx with respect to d_idx
                vij_x = d_u[d_idx] - (d_wz[d_idx] * d_rad_s[d_idx]) * tx
                vij_y = d_v[d_idx] - (d_wz[d_idx] * d_rad_s[d_idx]) * ty
                # ##############################################

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

                # ------------- Normal force computation ----------------
                # from the relative normal velocity we know the relative
                # displacement and from which the net increment of the
                # normal force is computed as
                delta_fn = kn * delta_dn

                # similarly the scalar magnitude of tangential force
                delta_ft = kt * delta_dt

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

                # set the direction and magnitude of the contact bond normal
                # force
                d_free_cnt_fn_x[found_at] = d_free_cnt_fn[found_at] * -nx
                d_free_cnt_fn_y[found_at] = d_free_cnt_fn[found_at] * -ny
                d_free_cnt_ft_x[found_at] = d_free_cnt_ft[found_at] * -tx
                d_free_cnt_ft_y[found_at] = d_free_cnt_ft[found_at] * -ty


class Cundall2dForceParticleInfinityWallStage2(Equation):
    def __init__(self, dest, sources, mu=0.5, en=0.8):
        # self.kn = kn
        # self.kt = 2. / 7. * kn
        # self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        # tmp = log(en)
        # self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall2dForceParticleInfinityWallStage2,
              self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_x, d_y, d_u, d_v, d_fx, d_fy, d_wz, d_kn,
             d_kt, d_free_cnt_idx, d_free_cnt_idx_dem_id, d_total_no_free_cnt,
             d_dem_id, d_free_cnt_limit, d_torz, RIJ, d_rad_s, d_free_cnt_fn,
             d_free_cnt_fn0, d_free_cnt_ft0, d_free_cnt_fn_x, d_free_cnt_fn_y,
             d_free_cnt_ft, d_free_cnt_ft_x, d_free_cnt_ft_y,
             d_free_cnt_delta_fn_x, d_free_cnt_delta_fn_y,
             d_free_cnt_delta_ft_x, d_free_cnt_delta_ft_y, s_x, s_y, s_z, s_nx,
             s_ny, s_nz, s_dem_id, s_np, dt):
        p, q1, tot_cnts, j, found_at, found = declare('int', 6)
        xij = declare('matrix(3)')
        overlap = -1.
        dtb2 = dt / 2.

        n = s_np[0]
        for i in range(n):
            # Force calculation starts
            overlap = -1.
            xij[0] = d_x[d_idx] - s_x[i]
            xij[1] = d_y[d_idx] - s_y[i]
            overlap = d_rad_s[d_idx] - (xij[0] * s_nx[i] + xij[1] * s_ny[i])

            # ---------- force computation starts ------------
            # if particles are overlapping
            if overlap > 0:
                # normal vector passes from d_idx to s_idx
                nx = -s_nx[i]
                ny = -s_ny[i]

                # the tangential direction is taken to be a vector
                # rotated 90 degrees clock wise of normal vector.
                tx = ny
                ty = -nx

                # ##############################################
                # TODO
                # THIS HAS TO BE FIXED
                # TODO
                # the relative velocity of particle s_idx with respect to d_idx
                vij_x = d_u[d_idx] - (d_wz[d_idx] * d_rad_s[d_idx]) * tx
                vij_y = d_v[d_idx] - (d_wz[d_idx] * d_rad_s[d_idx]) * ty
                # ##############################################

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
                    if i == d_free_cnt_idx[j]:
                        if s_dem_id[i] == d_free_cnt_idx_dem_id[j]:
                            found_at = j
                            found = 1
                            break

                # Already assigned contacts are dealt in stage2. This is for
                # algorithm simplicity. so we don't add new particles
                if found == 1:
                    # ------------- Normal force computation ----------------
                    # from the relative normal velocity we know the relative
                    # displacement and from which the net increment of the
                    # normal force is computed as
                    delta_fn = kn * delta_dn

                    # similarly the scalar magnitude of tangential force
                    delta_ft = kt * delta_dt

                    # assign it to the particle for post_processing
                    d_free_cnt_delta_fn_x[found_at] = delta_fn * -nx
                    d_free_cnt_delta_fn_y[found_at] = delta_fn * -ny

                    # before adding the increment to the tracking force variable
                    # add the normal force due to the contact s_idx to the particle
                    # d_idx first
                    d_fx[d_idx] += d_free_cnt_fn[found_at] * -nx
                    d_fy[d_idx] += d_free_cnt_fn[found_at] * -ny

                    # increment the scalar normal force
                    d_free_cnt_fn[
                        found_at] = d_free_cnt_fn0[found_at] + delta_fn
                    d_free_cnt_ft[
                        found_at] = d_free_cnt_ft0[found_at] + delta_ft

                    # check for Coulomb friction
                    fs_max = self.mu * abs(d_free_cnt_fn[found_at])
                    # get the sign of the tangential force
                    if d_free_cnt_ft[found_at] > 0.:
                        sign = 1.
                    else:
                        sign = -1.

                    if abs(d_free_cnt_ft[found_at]) > fs_max:
                        d_free_cnt_ft[found_at] = fs_max * sign

                    # set the direction and magnitude of the contact bond normal
                    # force
                    d_free_cnt_fn_x[found_at] = d_free_cnt_fn[found_at] * -nx
                    d_free_cnt_fn_y[found_at] = d_free_cnt_fn[found_at] * -ny
                    d_free_cnt_ft_x[found_at] = d_free_cnt_ft[found_at] * -tx
                    d_free_cnt_ft_y[found_at] = d_free_cnt_ft[found_at] * -ty


class UpdateFreeContactsWithInfinityWall(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_rad_s, d_total_no_free_cnt,
                        d_free_cnt_idx, d_free_cnt_limit,
                        d_free_cnt_idx_dem_id, d_free_cnt_fn_x, d_free_cnt_fn,
                        d_free_cnt_ft, d_free_cnt_fn0, d_free_cnt_ft0,
                        d_free_cnt_fn_y, d_free_cnt_ft_x, d_free_cnt_ft_y,
                        d_free_cnt_fn_x0, d_free_cnt_fn_y0, d_free_cnt_ft_x0,
                        d_free_cnt_ft_y0, d_free_cnt_delta_fn_x,
                        d_free_cnt_delta_fn_y, d_free_cnt_delta_ft_x,
                        d_free_cnt_delta_ft_y, s_x, s_y, s_z, s_nx, s_ny, s_nz,
                        s_dem_id):
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
                    overlap = d_rad_s[d_idx] - (xij[0] * s_nx[sidx] +
                                                xij[1] * s_ny[sidx])

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_free_cnt_idx[k] = -1
                            d_free_cnt_idx_dem_id[k] = -1
                            d_free_cnt_fn[k] = 0.
                            d_free_cnt_fn_x[k] = 0.
                            d_free_cnt_fn_y[k] = 0.

                            d_free_cnt_ft[k] = 0.
                            d_free_cnt_ft_x[k] = 0.
                            d_free_cnt_ft_y[k] = 0.

                            # make forces0 zero
                            d_free_cnt_fn0[k] = 0.
                            d_free_cnt_ft0[k] = 0.
                            d_free_cnt_fn_x0[k] = 0.
                            d_free_cnt_fn_y0[k] = 0.
                            d_free_cnt_ft_x0[k] = 0.
                            d_free_cnt_ft_y0[k] = 0.

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
                            d_free_cnt_fn_x[k] = d_free_cnt_fn_x[last_idx_tmp]
                            d_free_cnt_fn_y[k] = d_free_cnt_fn_y[last_idx_tmp]

                            d_free_cnt_ft[k] = d_free_cnt_ft[last_idx_tmp]
                            d_free_cnt_ft_x[k] = d_free_cnt_ft_x[last_idx_tmp]
                            d_free_cnt_ft_y[k] = d_free_cnt_ft_y[last_idx_tmp]

                            d_free_cnt_fn[last_idx_tmp] = 0.
                            d_free_cnt_ft[last_idx_tmp] = 0.
                            d_free_cnt_fn_x[last_idx_tmp] = 0.
                            d_free_cnt_fn_y[last_idx_tmp] = 0.
                            d_free_cnt_ft_x[last_idx_tmp] = 0.
                            d_free_cnt_ft_y[last_idx_tmp] = 0.

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
                            d_free_cnt_fn_x0[last_idx_tmp] = 0.
                            d_free_cnt_fn_y0[last_idx_tmp] = 0.
                            d_free_cnt_ft_x0[last_idx_tmp] = 0.
                            d_free_cnt_ft_y0[last_idx_tmp] = 0.

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


class Cundall2dForceParticleFiniteWallStage1(Equation):
    def __init__(self, dest, sources, mu=0.5, en=0.8):
        # self.kn = kn
        # self.kt = kt
        # self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        # tmp = log(en)
        # self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall2dForceParticleFiniteWallStage1,
              self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_x, d_y, d_u, d_v, d_fx, d_fy, d_wz, d_kn,
             d_kt, d_free_cnt_idx, d_free_cnt_idx_dem_id, d_total_no_free_cnt,
             d_dem_id, d_free_cnt_limit, d_torz, RIJ, d_rad_s, d_free_cnt_fn,
             d_free_cnt_ft, s_x, s_y, s_u, s_v, s_wz, s_kn, s_kt, s_nx, s_ny,
             s_A_x, s_A_y, s_B_x, s_B_y, s_dem_id, s_np, dt):
        i, n = declare('int', 2)
        p, q1, tot_cnts, j, found_at, found, = declare('int', 6)
        xij = declare('matrix(2)')
        overlap = -1.
        dtb2 = dt / 2.

        n = s_np[0]
        for i in range(n):
            # check if the particle is in contact with the wall, this needs two
            # checks, one is if the particle is in the limits of the wall,
            # and if it is interacting
            overlap = -1.
            xij[0] = d_x[d_idx] - s_x[i]
            xij[1] = d_y[d_idx] - s_y[i]
            x_dot_n = xij[0] * s_nx[i] + xij[1] * s_ny[i]
            overlap = d_rad_s[d_idx] - x_dot_n

            # ---------- force computation starts ------------
            # if particles are overlapping
            if overlap > 0:
                # normal vector passes from d_idx to s_idx
                nx = -s_nx[i]
                ny = -s_ny[i]

                # the tangential direction is taken to be a vector
                # rotated 90 degrees clock wise of normal vector.
                tx = ny
                ty = -nx

                # find the velocity of the wall at the contact point of
                # sphere and the wall
                # first find the contact point
                cp_x = xij[0] - x_dot_n * nx
                cp_y = xij[1] - x_dot_n * ny

                # using the contact point find the velocity
                cp_u = s_u[i] - s_wz[i] * cp_y
                cp_v = s_v[i] + s_wz[i] * cp_x

                # ##############################################
                # TODO
                # THIS HAS TO BE FIXED
                # TODO: Compute the velocity of the wall near the impact
                # the relative velocity of particle s_idx with respect to d_idx
                vij_x = d_u[d_idx] - (d_wz[d_idx] * d_rad_s[d_idx]) * tx - cp_u
                vij_y = d_v[d_idx] - (d_wz[d_idx] * d_rad_s[d_idx]) * ty - cp_v
                # ##############################################

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

                # find the normal and tangential stiffness
                kn = (d_kn[d_idx] * s_kn[i]) / (d_kn[d_idx] + s_kn[i])
                kt = (d_kt[d_idx] * s_kt[i]) / (d_kt[d_idx] + s_kt[i])

                # ------------- Normal force computation ----------------
                # from the relative normal velocity we know the relative
                # displacement and from which the net increment of the
                # normal force is computed as
                delta_fn = kn * delta_dn

                # similarly the scalar magnitude of tangential force
                delta_ft = kt * delta_dt

                # before adding the increment to the tracking force variable
                # add the normal force due to the contact s_idx to the particle
                # d_idx first
                d_fx[d_idx] += (d_free_cnt_fn[found_at] * -nx +
                                d_free_cnt_ft[found_at] * -tx)
                d_fy[d_idx] += (d_free_cnt_fn[found_at] * -ny +
                                d_free_cnt_ft[found_at] * -ty)
                d_torz[d_idx] += d_free_cnt_ft[found_at] * d_rad_s[d_idx]

                # increment the scalar normal force and tangential force
                d_free_cnt_fn[found_at] += delta_fn
                if d_free_cnt_fn[found_at] < 0.:
                    d_free_cnt_fn[found_at] = 0.

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


class Cundall2dForceParticleFiniteWallStage2(Equation):
    def __init__(self, dest, sources, mu=0.5, en=0.8):
        # self.kn = kn
        # self.kt = kt
        # self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        # tmp = log(en)
        # self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall2dForceParticleFiniteWallStage2,
              self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_x, d_y, d_u, d_v, d_fx, d_fy, d_wz, d_kn,
             d_kt, d_free_cnt_idx, d_free_cnt_idx_dem_id, d_total_no_free_cnt,
             d_dem_id, d_free_cnt_limit, d_torz, RIJ, d_rad_s, d_free_cnt_fn,
             d_free_cnt_ft, d_free_cnt_fn0, d_free_cnt_ft0, s_x, s_y, s_u, s_v,
             s_wz, s_kn, s_kt, s_nx, s_ny, s_A_x, s_A_y, s_B_x, s_B_y,
             s_dem_id, s_np, dt):
        i, n = declare('int', 2)
        p, q1, tot_cnts, j, found_at, found = declare('int', 6)
        xij = declare('matrix(2)')
        overlap = -1.
        dtb2 = dt / 2.

        n = s_np[0]
        for i in range(n):
            # Force calculation starts
            overlap = -1.
            xij[0] = d_x[d_idx] - s_x[i]
            xij[1] = d_y[d_idx] - s_y[i]
            x_dot_n = xij[0] * s_nx[i] + xij[1] * s_ny[i]
            overlap = d_rad_s[d_idx] - x_dot_n

            # ---------- force computation starts ------------
            # if particles are overlapping
            if overlap > 0:
                # normal vector passes from d_idx to s_idx
                nx = -s_nx[i]
                ny = -s_ny[i]

                # the tangential direction is taken to be a vector
                # rotated 90 degrees clock wise of normal vector.
                tx = ny
                ty = -nx

                # find the velocity of the wall at the contact point of
                # sphere and the wall
                # first find the contact point
                cp_x = xij[0] - x_dot_n * nx
                cp_y = xij[1] - x_dot_n * ny

                # using the contact point find the velocity
                cp_u = s_u[i] - s_wz[i] * cp_y
                cp_v = s_v[i] + s_wz[i] * cp_x

                # ##############################################
                # TODO
                # THIS HAS TO BE FIXED
                # TODO: Compute the velocity of the wall near the impact
                # the relative velocity of particle s_idx with respect to d_idx
                vij_x = d_u[d_idx] - (d_wz[d_idx] * d_rad_s[d_idx]) * tx - cp_u
                vij_y = d_v[d_idx] - (d_wz[d_idx] * d_rad_s[d_idx]) * ty - cp_v
                # ##############################################

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
                    if i == d_free_cnt_idx[j]:
                        if s_dem_id[i] == d_free_cnt_idx_dem_id[j]:
                            found_at = j
                            found = 1
                            break

                # Already assigned contacts are dealt in stage2. This is for
                # algorithm simplicity. so we don't add new particles
                if found == 1:
                    # find the normal and tangential stiffness
                    kn = (d_kn[d_idx] * s_kn[i]) / (d_kn[d_idx] + s_kn[i])
                    kt = (d_kt[d_idx] * s_kt[i]) / (d_kt[d_idx] + s_kt[i])

                    # ------------- Normal force computation ----------------
                    # from the relative normal velocity we know the relative
                    # displacement and from which the net increment of the
                    # normal force is computed as
                    delta_fn = kn * delta_dn

                    # similarly the scalar magnitude of tangential force
                    delta_ft = kt * delta_dt

                    # before adding the increment to the tracking force variable
                    # add the normal force due to the contact s_idx to the particle
                    # d_idx first
                    d_fx[d_idx] += d_free_cnt_fn[found_at] * -nx
                    d_fy[d_idx] += d_free_cnt_fn[found_at] * -ny
                    d_torz[d_idx] += d_free_cnt_ft[found_at] * d_rad_s[d_idx]

                    # increment the scalar normal force
                    # increment the scalar normal force
                    d_free_cnt_fn[found_at] = (d_free_cnt_fn0[found_at] +
                                               delta_fn)
                    if d_free_cnt_fn[found_at] < 0.:
                        d_free_cnt_fn[found_at] = 0.
                    d_free_cnt_ft[found_at] = (d_free_cnt_ft0[found_at] +
                                               delta_ft)

                    # check for Coulomb friction
                    fs_max = self.mu * abs(d_free_cnt_fn[found_at])
                    # get the sign of the tangential force
                    if d_free_cnt_ft[found_at] > 0.:
                        sign = 1.
                    else:
                        sign = -1.

                    if abs(d_free_cnt_ft[found_at]) > fs_max:
                        d_free_cnt_ft[found_at] = fs_max * sign


class UpdateFreeContactsWithFiniteWall(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_rad_s, d_total_no_free_cnt,
                        d_free_cnt_idx, d_free_cnt_limit,
                        d_free_cnt_idx_dem_id, d_free_cnt_fn, d_free_cnt_ft,
                        d_free_cnt_fn0, d_free_cnt_ft0, s_x, s_y, s_z, s_nx,
                        s_ny, s_dem_id):
        p = declare('int')
        count = declare('int')
        k = declare('int')
        xij = declare('matrix(2)')
        last_idx_tmp = declare('int')
        sidx = declare('int')
        dem_id = declare('int')

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
                    overlap = d_rad_s[d_idx] - (xij[0] * s_nx[sidx] +
                                                xij[1] * s_ny[sidx])

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_free_cnt_idx[k] = -1
                            d_free_cnt_idx_dem_id[k] = -1

                            d_free_cnt_fn[k] = 0.
                            d_free_cnt_ft[k] = 0.

                            # make forces0 zero
                            d_free_cnt_fn0[k] = 0.
                            d_free_cnt_ft0[k] = 0.

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

                            # swap tangential idx dem id
                            d_free_cnt_idx_dem_id[k] = d_free_cnt_idx_dem_id[
                                last_idx_tmp]
                            d_free_cnt_idx_dem_id[last_idx_tmp] = -1

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
                   d_free_cnt_fn, d_free_cnt_ft, d_free_cnt_fn0,
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


def get_particle_array_finite_wall(constants=None, **props):
    """Return a particle array for a finite wall
    """
    dem_props = [
        'x0', 'y0', 'u0', 'v0', 'wz', 'A_x', 'A_y', 'A_u', 'A_v', 'B_x', 'B_y',
        'B_u', 'B_v', 'A_x0', 'A_y0', 'B_x0', 'B_y0', 'wall_length', 'nx',
        'ny', 'nz'
    ]

    dem_id = props.pop('dem_id', None)

    pa = get_particle_array(additional_props=dem_props, constants=constants,
                            **props)

    pa.add_property('dem_id', type='int', data=dem_id)

    pa.set_output_arrays([
        'x', 'y', 'u', 'v', 'pid', 'tag', 'gid', 'wz', 'dem_id', 'nx', 'ny',
        'nz', 'wall_length'
    ])

    return pa


class RK2StepFiniteWall(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_x0, d_y0, d_u, d_v, d_u0, d_v0,
                   d_A_x, d_A_y, d_B_x, d_B_y, d_A_x0, d_A_y0, d_B_x0, d_B_y0):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_A_x0[d_idx] = d_A_x[d_idx]
        d_A_y0[d_idx] = d_A_y[d_idx]
        d_B_x0[d_idx] = d_B_x[d_idx]
        d_B_y0[d_idx] = d_B_y[d_idx]

    def stage1(self, d_idx, d_x, d_y, d_u, d_v, d_x0, d_y0, d_wz, d_nx, d_ny,
               d_A_x, d_A_y, d_B_x, d_B_y, d_A_u, d_A_v, d_B_u, d_B_v, d_A_x0,
               d_A_y0, d_B_x0, d_B_y0, d_wall_length, dt):
        dtb2 = dt / 2.

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]

        # find the velocity of point A due to the linear and angular velocity
        # of point p
        # vector passing from point p to A
        dx_A = d_A_x[d_idx] - d_x[d_idx]
        dy_A = d_A_y[d_idx] - d_y[d_idx]
        d_A_u[d_idx] = d_u[d_idx] - d_wz[d_idx] * dy_A
        d_A_v[d_idx] = d_v[d_idx] + d_wz[d_idx] * dx_A

        # find the velocity of point B due to the linear and angular velocity
        # of point p
        # vector passing from point p to B
        dx_B = d_B_x[d_idx] - d_x[d_idx]
        dy_B = d_B_y[d_idx] - d_y[d_idx]
        d_B_u[d_idx] = d_u[d_idx] - d_wz[d_idx] * dy_B
        d_B_v[d_idx] = d_v[d_idx] + d_wz[d_idx] * dx_B

        # using the velocity of the points A and B update the positions
        d_A_x[d_idx] = d_A_x0[d_idx] + dtb2 * d_A_u[d_idx]
        d_A_y[d_idx] = d_A_y0[d_idx] + dtb2 * d_A_v[d_idx]
        d_B_x[d_idx] = d_B_x0[d_idx] + dtb2 * d_B_u[d_idx]
        d_B_y[d_idx] = d_B_y0[d_idx] + dtb2 * d_B_v[d_idx]

        # now update the normal of the finite wall using the positions of
        # A and B
        dx_ab = d_A_x[d_idx] - d_B_x[d_idx]
        dy_ab = d_A_y[d_idx] - d_B_y[d_idx]

        d_wall_length[d_idx] = (dx_ab**2. + dy_ab**2.)**0.5
        d_nx[d_idx] = -dy_ab / d_wall_length[d_idx]
        d_ny[d_idx] = dx_ab / d_wall_length[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_u, d_v, d_x0, d_y0, d_wz, d_nx, d_ny,
               d_A_x, d_A_y, d_B_x, d_B_y, d_A_u, d_A_v, d_B_u, d_B_v, d_A_x0,
               d_A_y0, d_B_x0, d_B_y0, d_wall_length, dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]

        # find the velocity of point A due to the linear and angular velocity
        # of point p
        # vector passing from point p to A
        dx_A = d_A_x[d_idx] - d_x[d_idx]
        dy_A = d_A_y[d_idx] - d_y[d_idx]
        d_A_u[d_idx] = d_u[d_idx] - d_wz[d_idx] * dy_A
        d_A_v[d_idx] = d_v[d_idx] + d_wz[d_idx] * dx_A

        # find the velocity of point B due to the linear and angular velocity
        # of point p
        # vector passing from point p to B
        dx_B = d_B_x[d_idx] - d_x[d_idx]
        dy_B = d_B_y[d_idx] - d_y[d_idx]
        d_B_u[d_idx] = d_u[d_idx] - d_wz[d_idx] * dy_B
        d_B_v[d_idx] = d_v[d_idx] + d_wz[d_idx] * dx_B

        # using the velocity of the points A and B update the positions
        d_A_x[d_idx] = d_A_x0[d_idx] + dt * d_A_u[d_idx]
        d_A_y[d_idx] = d_A_y0[d_idx] + dt * d_A_v[d_idx]
        d_B_x[d_idx] = d_B_x0[d_idx] + dt * d_B_u[d_idx]
        d_B_y[d_idx] = d_B_y0[d_idx] + dt * d_B_v[d_idx]

        # now update the normal of the finite wall using the positions of
        # A and B
        dx_ab = d_A_x[d_idx] - d_B_x[d_idx]
        dy_ab = d_A_y[d_idx] - d_B_y[d_idx]

        d_wall_length[d_idx] = (dx_ab**2. + dy_ab**2.)**0.5
        d_nx[d_idx] = -dy_ab / d_wall_length[d_idx]
        d_ny[d_idx] = dx_ab / d_wall_length[d_idx]


def get_particle_array_from_file(file_name, name):
    data = load(file_name)
    return data['arrays'][name]


def analyse_tng_cnt_info_cundall_2d_dem_scalar_formulation(pa, idx):
    limit = pa.free_cnt_limit[0]
    print("ids of partticle {idx} in contact".format(idx=idx))
    print(pa.free_cnt_idx[limit * idx:limit * idx + limit])
    print('free cnt fn')
    print(pa.free_cnt_fn[limit * idx:limit * idx + limit])
    print('free cnt ft')
    print(pa.free_cnt_ft[limit * idx:limit * idx + limit])
    print('total tng contacts')
    print(pa.total_no_free_cnt[idx])
    print('contact idx dem number')
    print(pa.free_cnt_idx_dem_id[limit * idx:limit * idx + limit])
    print('force on ' + str(idx) + ' is ')
    print(pa.fx[idx])
    print(pa.fy[idx])
    print(pa.fz[idx])
    return pa


def analyse_tng_cnt_info_from_file(file_name, name, idx):
    pa = get_particle_array_from_file(file_name, name)
    spheres = analyse_tng_cnt_info_cundall_2d_dem_scalar_formulation(pa, idx)
    return spheres
