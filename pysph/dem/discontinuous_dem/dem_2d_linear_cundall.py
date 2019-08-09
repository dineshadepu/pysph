from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from compyle.api import declare
from math import sqrt, log
import numpy as np
from pysph.sph.scheme import Scheme
from pysph.sph.equation import Group, MultiStageEquations
from pysph.dem.discontinuous_dem.dem_nonlinear import EPECIntegratorMultiStage


def get_particle_array_dem_2d_linear_cundall(constants=None, **props):
    """Return a particle array for a dem particles
    """
    dim = props.pop('dim', None)

    dem_props = [
        'theta_dot', 'theta_dot0', 'fx', 'fy', 'fz', 'torz', 'rad_s',
        'm_inverse', 'I_inverse', 'u0', 'v0', 'x0', 'y0'
    ]

    dem_id = props.pop('dem_id', None)

    pa = get_particle_array(additional_props=dem_props, **props)

    pa.add_property('dem_id', type='int', data=dem_id)
    # create the array to save the tangential interaction particles
    # index and other variables
    limit = 6

    pa.add_constant('limit', limit)
    pa.add_property('tng_idx', stride=limit, type="int")
    pa.tng_idx[:] = -1
    pa.add_property('tng_idx_dem_id', stride=limit, type="int")
    pa.tng_idx_dem_id[:] = -1
    pa.add_property('tng_frc', stride=limit)
    pa.add_property('tng_frc0', stride=limit)
    pa.tng_frc[:] = 0.
    pa.tng_frc0[:] = 0.
    pa.add_property('total_tng_contacts', type="int")
    pa.total_tng_contacts[:] = 0

    pa.set_output_arrays([
        'x', 'y', 'u', 'v', 'm', 'pid', 'tag', 'gid', 'fx', 'fy', 'fz', 'torz',
        'I_inverse', 'theta_dot', 'm_inverse', 'rad_s', 'dem_id', 'tng_idx',
        'tng_idx_dem_id', 'tng_frc', 'tng_frc0', 'total_tng_contacts'
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


class Cundall2dForceParticleParticleEuler(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall2dForceParticleParticleEuler, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_fx, d_fy, d_theta_dot, d_tng_idx,
             d_tng_idx_dem_id, d_tng_frc, d_tng_frc0, d_total_tng_contacts,
             d_dem_id, d_limit, d_torz, VIJ, XIJ, RIJ, d_rad_s, s_idx, s_m,
             s_rad_s, s_dem_id, s_theta_dot, dt):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
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
            nx = -XIJ[0] * rinv
            ny = -XIJ[1] * rinv

            # tangential direction (rotate normal vector 90 degrees
            # clockwise)
            tx = ny
            ty = -nx

            # ---- Relative velocity computation (Eq 11) ----
            # follow Cundall equation (11)
            tmp = (d_theta_dot[d_idx] * d_rad_s[d_idx] +
                   s_theta_dot[s_idx] * s_rad_s[s_idx])

            # scalar components of relative velocity in normal and
            # tangential directions
            vn = VIJ[0] * nx + VIJ[1] * ny
            vt = VIJ[0] * tx + VIJ[1] * ty - tmp

            # damping force is taken from
            # "On the Determination of the Damping Coefficient
            # of Non-linear Spring-dashpot System to Model
            # Hertz Contact for Simulation by Discrete Element
            # Method" paper.
            # compute the damping constants
            m_eff = d_m[d_idx] * s_m[s_idx] / (d_m[d_idx] + s_m[s_idx])
            eta_n = self.alpha * sqrt(m_eff)

            # normal force
            kn_overlap = self.kn * overlap
            fn_x = -kn_overlap * nx - eta_n * vn * nx
            fn_y = -kn_overlap * ny - eta_n * vn * ny

            # ------------- tangential force computation ----------------
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
            # if the particle is not been tracked then assign an index in
            # tracking history.
            if found == 0:
                found_at = q1
                d_tng_idx[found_at] = s_idx
                d_total_tng_contacts[d_idx] += 1
                d_tng_idx_dem_id[found_at] = s_dem_id[s_idx]

            # compute the damping constants
            eta_t = 0.5 * eta_n

            # find the tangential force from the tangential displacement
            # and tangential velocity (eq 2.11 Thesis Ye)
            ft = d_tng_frc[found_at]

            # we have to compare with static friction, so
            # this mu has to be static friction coefficient
            fn_magn = (fn_x * fn_x + fn_y * fn_y)**(0.5)
            ft_max = self.mu * fn_magn

            # if the tangential force magnitude is zero, then do nothing,
            # else do following
            if ft >= ft_max:
                ft = ft_max
                d_tng_frc[found_at] = ft_max

            d_tng_frc[found_at] += self.kt * vt * dt

            d_fx[d_idx] += fn_x - ft * tx
            d_fy[d_idx] += fn_y - ft * ty

            # torque
            d_torz[d_idx] += ft * d_rad_s[d_idx]


class Cundall2dForceParticleParticleStage1(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall2dForceParticleParticleStage1, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_fx, d_fy, d_theta_dot, d_tng_idx,
             d_tng_idx_dem_id, d_tng_frc, d_tng_frc0, d_total_tng_contacts,
             d_dem_id, d_limit, d_torz, VIJ, XIJ, RIJ, d_rad_s, s_idx, s_m,
             s_rad_s, s_dem_id, s_theta_dot, dt):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
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
            nx = -XIJ[0] * rinv
            ny = -XIJ[1] * rinv

            # tangential direction (rotate normal vector 90 degrees
            # clockwise)
            tx = ny
            ty = -nx

            # ---- Relative velocity computation (Eq 11) ----
            # follow Cundall equation (11)
            tmp = (d_theta_dot[d_idx] * d_rad_s[d_idx] +
                   s_theta_dot[s_idx] * s_rad_s[s_idx])
            vij_x = VIJ[0] - tmp * tx
            vij_y = VIJ[1] - tmp * ty

            # scalar components of relative velocity in normal and
            # tangential directions
            vn = VIJ[0] * nx + VIJ[1] * ny
            vt = VIJ[0] * tx + VIJ[1] * ty - tmp

            # damping force is taken from
            # "On the Determination of the Damping Coefficient
            # of Non-linear Spring-dashpot System to Model
            # Hertz Contact for Simulation by Discrete Element
            # Method" paper.
            # compute the damping constants
            m_eff = d_m[d_idx] * s_m[s_idx] / (d_m[d_idx] + s_m[s_idx])
            eta_n = self.alpha * sqrt(m_eff)

            # normal force
            kn_overlap = self.kn * overlap
            fn_x = -kn_overlap * nx - eta_n * vn * nx
            fn_y = -kn_overlap * ny - eta_n * vn * ny

            # ------------- tangential force computation ----------------
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
            # if the particle is not been tracked then assign an index in
            # tracking history.
            if found == 0:
                found_at = q1
                d_tng_idx[found_at] = s_idx
                d_total_tng_contacts[d_idx] += 1
                d_tng_idx_dem_id[found_at] = s_dem_id[s_idx]

            # compute the damping constants
            eta_t = 0.5 * eta_n

            # find the tangential force from the tangential displacement
            # and tangential velocity (eq 2.11 Thesis Ye)
            ft = d_tng_frc[found_at]

            # we have to compare with static friction, so
            # this mu has to be static friction coefficient
            fn_magn = (fn_x * fn_x + fn_y * fn_y)**(0.5)
            ft_max = self.mu * fn_magn

            # if the tangential force magnitude is zero, then do nothing,
            # else do following
            if ft >= ft_max:
                ft = ft_max
                d_tng_frc[found_at] = ft_max
                d_tng_frc0[found_at] = ft_max

            dtb2 = dt / 2.
            d_tng_frc[found_at] += self.kt * vt * dtb2

            d_fx[d_idx] += fn_x - ft * tx
            d_fy[d_idx] += fn_y - ft * ty

            # torque
            d_torz[d_idx] += ft * d_rad_s[d_idx]


class Cundall2dForceParticleParticleStage2(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall2dForceParticleParticleStage2, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_fx, d_fy, d_theta_dot, d_tng_idx,
             d_tng_idx_dem_id, d_tng_frc, d_tng_frc0, d_total_tng_contacts,
             d_dem_id, d_limit, d_torz, VIJ, XIJ, RIJ, d_rad_s, s_idx, s_m,
             s_rad_s, s_dem_id, s_theta_dot, dt):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
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
            nx = -XIJ[0] * rinv
            ny = -XIJ[1] * rinv

            # tangential direction (rotate normal vector 90 degrees
            # clockwise)
            tx = ny
            ty = -nx

            # ---- Relative velocity computation (Eq 11) ----
            # follow Cundall equation (11)
            tmp = (d_theta_dot[d_idx] * d_rad_s[d_idx] +
                   s_theta_dot[s_idx] * s_rad_s[s_idx])
            vij_x = VIJ[0] - tmp * tx
            vij_y = VIJ[1] - tmp * ty

            # scalar components of relative velocity in normal and
            # tangential directions
            vn = VIJ[0] * nx + VIJ[1] * ny
            vt = VIJ[0] * tx + VIJ[1] * ty - tmp

            # damping force is taken from
            # "On the Determination of the Damping Coefficient
            # of Non-linear Spring-dashpot System to Model
            # Hertz Contact for Simulation by Discrete Element
            # Method" paper.
            # compute the damping constants
            m_eff = d_m[d_idx] * s_m[s_idx] / (d_m[d_idx] + s_m[s_idx])
            eta_n = self.alpha * sqrt(m_eff)

            # normal force
            kn_overlap = self.kn * overlap
            fn_x = -kn_overlap * nx - eta_n * vn * nx
            fn_y = -kn_overlap * ny - eta_n * vn * ny

            # ------------- tangential force computation ----------------
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
            # if the particle is not been tracked then assign an index in
            # tracking history.
            ft = 0.
            if found == 1:
                eta_t = 0.5 * eta_n

                # find the tangential force from the tangential displacement
                # and tangential velocity (eq 2.11 Thesis Ye)
                ft = d_tng_frc[found_at]

                # don't check for Coulomb limit as we are dealing with
                # RK2 integrator

                d_tng_frc[found_at] = d_tng_frc0[found_at] + self.kt * vt * dt

            d_fx[d_idx] += fn_x - ft * tx
            d_fy[d_idx] += fn_y - ft * ty

            # torque
            d_torz[d_idx] += ft * d_rad_s[d_idx]


class UpdateTangentialContactsCundall2dPaticleParticle(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_rad_s, d_total_tng_contacts,
                        d_tng_idx, d_limit, d_tng_frc, d_tng_idx_dem_id,
                        d_tng_frc0, s_x, s_y, s_rad_s, s_dem_id):
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
                    rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

                    overlap = d_rad_s[d_idx] + s_rad_s[sidx] - rij

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_tng_idx[k] = -1
                            d_tng_idx_dem_id[k] = -1
                            d_tng_frc[k] = 0.
                            # make tangential0 displacements zero
                            d_tng_frc0[k] = 0.
                        else:
                            # swap the current tracking index with the final
                            # contact index
                            d_tng_idx[k] = d_tng_idx[last_idx_tmp]
                            d_tng_idx[last_idx_tmp] = -1

                            # swap tangential x displacement
                            d_tng_frc[k] = d_tng_frc[last_idx_tmp]
                            d_tng_frc[last_idx_tmp] = 0.

                            # swap tangential idx dem id
                            d_tng_idx_dem_id[k] = d_tng_idx_dem_id[
                                last_idx_tmp]
                            d_tng_idx_dem_id[last_idx_tmp] = -1

                            # make tangential0 displacements zero
                            d_tng_frc0[last_idx_tmp] = 0.

                            # decrease the last_idx_tmp, since we swapped it to
                            # -1
                            last_idx_tmp -= 1

                        # decrement the total contacts of the particle
                        d_total_tng_contacts[d_idx] -= 1
                    else:
                        k = k + 1
                else:
                    k = k + 1
                count += 1


class UpdateTangentialContactsWallNoRotation(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_rad_s, d_total_tng_contacts,
                        d_tng_idx, d_limit, d_tng_frc, d_tng_idx_dem_id,
                        d_tng_frc0, s_x, s_y, s_nx, s_ny, s_dem_id):
        p = declare('int')
        count = declare('int')
        k = declare('int')
        xij = declare('matrix(3)')
        last_idx_tmp = declare('int')
        sidx = declare('int')
        dem_id = declare('int')

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
                    overlap = d_rad_s[d_idx] - (
                        xij[0] * s_nx[sidx] + xij[1] * s_ny[sidx])

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_tng_idx[k] = -1
                            d_tng_idx_dem_id[k] = -1
                            d_tng_frc[k] = 0.
                            d_tng_frc0[k] = 0.
                        else:
                            # swap the current tracking index with the final
                            # contact index
                            d_tng_idx[k] = d_tng_idx[last_idx_tmp]
                            d_tng_idx[last_idx_tmp] = -1

                            # swap tangential x displacement
                            d_tng_frc[k] = d_tng_frc[last_idx_tmp]
                            d_tng_frc[last_idx_tmp] = 0.

                            # swap tangential idx dem id
                            d_tng_idx_dem_id[k] = d_tng_idx_dem_id[
                                last_idx_tmp]
                            d_tng_idx_dem_id[last_idx_tmp] = -1

                            # make tangential0 displacements zero
                            d_tng_frc0[last_idx_tmp] = 0.

                            # decrease the last_idx_tmp, since we swapped it to
                            # -1
                            last_idx_tmp -= 1

                        # decrement the total contacts of the particle
                        d_total_tng_contacts[d_idx] -= 1
                    else:
                        k = k + 1
                else:
                    k = k + 1
                count += 1


class RK2StepDEM2dCundall(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_u, d_v, d_u0,
                   d_v0, d_theta_dot, d_theta_dot0, d_total_tng_contacts,
                   d_limit, d_tng_frc, d_tng_frc0):

        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]

        d_theta_dot0[d_idx] = d_theta_dot[d_idx]

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
            d_tng_frc0[i] = d_tng_frc[i]

    def stage1(self, d_idx, d_x, d_y, d_u, d_v, d_w, d_fx, d_fy, d_x0, d_y0,
               d_u0, d_v0, d_theta_dot, d_theta_dot0, d_torz, d_m_inverse,
               d_I_inverse, dt):
        dtb2 = dt / 2.

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_fy[d_idx] * d_m_inverse[d_idx]

        d_theta_dot[d_idx] = d_theta_dot0[d_idx] + (
            dtb2 * d_torz[d_idx] * d_I_inverse[d_idx])

    def stage2(self, d_idx, d_x, d_y, d_u, d_v, d_w, d_fx, d_fy, d_x0, d_y0,
               d_u0, d_v0, d_theta_dot, d_theta_dot0, d_torz, d_m_inverse,
               d_I_inverse, dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_fy[d_idx] * d_m_inverse[d_idx]

        d_theta_dot[d_idx] = d_theta_dot0[d_idx] + (
            dt * d_torz[d_idx] * d_I_inverse[d_idx])


class EulerStepDEM2dCundall(IntegratorStep):
    def stage1(self, d_idx, d_x, d_y, d_u, d_v, d_w, d_fx, d_fy, d_theta_dot,
               d_torz, d_m_inverse, d_I_inverse, dt):
        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]

        d_u[d_idx] = d_u[d_idx] + dt * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v[d_idx] + dt * d_fy[d_idx] * d_m_inverse[d_idx]

        d_theta_dot[d_idx] = d_theta_dot[d_idx] + (
            dt * d_torz[d_idx] * d_I_inverse[d_idx])


class Dem2dCundallScheme(Scheme):
    def __init__(self, bodies, solids, integrator, dim, kn, mu=0.5, en=1.0,
                 gx=0.0, gy=0.0, debug=False):
        self.bodies = bodies
        self.solids = solids
        self.dim = dim
        self.integrator = integrator
        self.kn = kn
        self.mu = mu
        self.en = en
        self.gx = gx
        self.gy = gy
        self.debug = debug

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import CubicSpline
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        if self.integrator == "rk2":
            for body in self.bodies:
                if body not in steppers:
                    steppers[body] = RK2StepDEM2dCundall()

            cls = integrator_cls if integrator_cls is not None else EPECIntegratorMultiStage
        elif self.integrator == "euler":
            for body in self.bodies:
                if body not in steppers:
                    steppers[body] = EulerStepDEM2dCundall()

            cls = integrator_cls if integrator_cls is not None else EulerIntegratorMultiStage

        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def get_rk2_equations(self):
        # stage 1
        stage1 = []
        g1 = []
        if self.solids is not None:
            all = self.bodies + self.solids
        else:
            all = self.bodies

        for name in self.bodies:
            g1.append(
                BodyForce(dest=name, sources=None, gx=self.gx, gy=self.gy))

        for name in self.bodies:
            g1.append(
                Cundall2dForceParticleParticleStage1(dest=name, sources=all, kn=self.kn,
                                     mu=self.mu, en=self.en))
        stage1.append(Group(equations=g1, real=False))

        # stage 2
        stage2 = []
        g1 = []
        if self.solids is not None:
            all = self.bodies + self.solids
        else:
            all = self.bodies

        for name in self.bodies:
            g1.append(
                BodyForce(dest=name, sources=None, gx=self.gx, gy=self.gy))

        for name in self.bodies:
            g1.append(
                Cundall2dForceParticleParticleStage2(dest=name, sources=all, kn=self.kn,
                                     mu=self.mu, en=self.en))
        stage2.append(Group(equations=g1, real=False))
        return MultiStageEquations([stage1, stage2])

    def get_euler_equations(self):
        stage1 = []
        g1 = []
        if self.solids is not None:
            all = self.bodies + self.solids
        else:
            all = self.bodies

        for name in self.bodies:
            g1.append(
                BodyForce(dest=name, sources=None, gx=self.gx, gy=self.gy))

        for name in self.bodies:
            g1.append(
                Cundall2dForceParticleParticleEuler(dest=name, sources=all, kn=self.kn,
                                    mu=self.mu, en=self.en))
        stage1.append(Group(equations=g1, real=False))

        return stage1

    def get_equations(self):
        if self.integrator == "rk2":
            return self.get_rk2_equations()
        elif self.integrator == "euler":
            return self.get_euler_equations()
