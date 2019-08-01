from pysph.base.reduce_array import parallel_reduce_array
from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
import numpy as np
import numpy
from math import sqrt, asin, sin, cos, pi, log
from pysph.sph.rigid_body_setup import (setup_quaternion_rigid_body)
from compyle.api import declare


def get_particle_array_rigid_body_cundall(constants=None, **props):
    """Return a particle array for a rigid body motion.

    For multiple bodies, add a body_id property starting at index 0 with each
    index denoting the body to which the particle corresponds to.

    Parameters
    ----------
    constants : dict
        Dictionary of constants

    Other Parameters
    ----------------
    props : dict
        Additional keywords passed are set as the property arrays.

    See Also
    --------
    get_particle_array

    """
    extra_props = [
        'au', 'av', 'aw', 'V', 'fx', 'fy', 'fz',
        'x0', 'y0', 'z0',
        'rad_s', 'nx', 'ny', 'nz'
    ]

    body_id = props.pop('body_id', None)
    nb = 1 if body_id is None else numpy.max(body_id) + 1

    dem_id = props.pop('dem_id', None)

    consts = {
        'total_mass': numpy.zeros(nb, dtype=float),
        'num_body': numpy.asarray(nb, dtype=int),
        'cm': numpy.zeros(3 * nb, dtype=float),

        # The mi are also used to temporarily reduce mass (1), center of
        # mass (3) and the inertia components (6), total force (3), total
        # torque (3).
        'mi': numpy.zeros(16 * nb, dtype=float),
        'force': numpy.zeros(3 * nb, dtype=float),
        'torque': numpy.zeros(3 * nb, dtype=float),
        # velocity, acceleration of CM.
        'vc': numpy.zeros(3 * nb, dtype=float),
        'ac': numpy.zeros(3 * nb, dtype=float),
        'vc0': numpy.zeros(3 * nb, dtype=float),
        # angular velocity, acceleration of body.
        'omega': numpy.zeros(3 * nb, dtype=float),
        'omega0': numpy.zeros(3 * nb, dtype=float),
        'omega_dot': numpy.zeros(3 * nb, dtype=float)
    }
    if constants:
        consts.update(constants)
    pa = get_particle_array(constants=consts, additional_props=extra_props,
                            **props)
    pa.add_property('body_id', type='int', data=body_id)
    pa.add_property('dem_id', type='int', data=dem_id)

    # create the array to save the tangential interaction particles
    # index and other variables
    limit = 6
    setup_rigid_body_cundall_particle_array(pa, limit)

    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au', 'av',
        'aw', 'tag', 'gid', 'V', 'fx', 'fy', 'fz', 'body_id'
    ])

    return pa


def setup_rigid_body_cundall_particle_array(pa, limit):
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


class RigidBodyCollision2DCundallEuler(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(RigidBodyCollision2DCundallEuler, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_fx, d_fy, d_tng_idx,
             d_tng_idx_dem_id, d_tng_frc, d_tng_frc0, d_total_tng_contacts,
             d_dem_id, d_limit, VIJ, XIJ, RIJ, d_rad_s, s_idx, s_m,
             s_rad_s, s_dem_id, dt):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)

        if d_dem_id[d_idx] != s_dem_id[s_idx]:
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

                # scalar components of relative velocity in normal and
                # tangential directions
                vn = VIJ[0] * nx + VIJ[1] * ny
                vt = VIJ[0] * tx + VIJ[1] * ty

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

                d_tng_frc[found_at] += self.kt * vt * dt

                d_fx[d_idx] += fn_x - ft * tx
                d_fy[d_idx] += fn_y - ft * ty


class RigidBodyCollision2DCundallStage1(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(RigidBodyCollision2DCundallStage1, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_fx, d_fy, d_tng_idx,
             d_tng_idx_dem_id, d_tng_frc, d_tng_frc0, d_total_tng_contacts,
             d_dem_id, d_limit, VIJ, XIJ, RIJ, d_rad_s, s_idx, s_m,
             s_rad_s, s_dem_id, dt):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)

        if d_dem_id[d_idx] != s_dem_id[s_idx]:
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

                # scalar components of relative velocity in normal and
                # tangential directions
                vn = VIJ[0] * nx + VIJ[1] * ny
                vt = VIJ[0] * tx + VIJ[1] * ty

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


class RigidBodyCollision2DCundallStage2(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(RigidBodyCollision2DCundallStage2, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_fx, d_fy, d_tng_idx,
             d_tng_idx_dem_id, d_tng_frc, d_tng_frc0, d_total_tng_contacts,
             d_dem_id, d_limit, VIJ, XIJ, RIJ, d_rad_s, s_idx, s_m,
             s_rad_s, s_dem_id, dt):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)

        if d_dem_id[d_idx] != s_dem_id[s_idx]:
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
                # scalar components of relative velocity in normal and
                # tangential directions
                vn = VIJ[0] * nx + VIJ[1] * ny
                vt = VIJ[0] * tx + VIJ[1] * ty

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


class UpdateTangentialContactsCundall2d(Equation):
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


def get_particle_array_rigid_body_quaternion(constants=None, **props):
    extra_props = [
        'fx', 'fy', 'fz', 'dx0', 'dy0', 'dz0', 'nx0', 'ny0', 'nz0', 'nx', 'ny',
        'nz', 'x0', 'y0', 'z0', 'u0', 'v0', 'w0'
    ]

    body_id = props.pop('body_id', None)
    nb = 1 if body_id is None else numpy.max(body_id) + 1

    dem_id = props.pop('dem_id', None)

    consts = {
        'total_mass': 0.,
        'cm': numpy.zeros(3, dtype=float),
        'cm0': numpy.zeros(3, dtype=float),
        'q': numpy.array([1., 0., 0., 0.]),
        'q0': numpy.array([1., 0., 0., 0.]),
        'qdot': numpy.zeros(4, dtype=float),
        'R': [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        # moment of inertia inverse in body frame
        'mib': numpy.zeros(9, dtype=float),
        # moment of inertia inverse in global frame
        'mig': numpy.zeros(9, dtype=float),
        # total force at the center of mass
        'force': numpy.zeros(3, dtype=float),
        # torque about the center of mass
        'torque': numpy.zeros(3, dtype=float),
        # velocity, acceleration of CM.
        'vc': numpy.zeros(3, dtype=float),
        'vc0': numpy.zeros(3, dtype=float),
        # angular velocity in global frame
        'omega': numpy.zeros(3, dtype=float),
        'omega0': numpy.zeros(3, dtype=float),
        'nb': nb
    }

    if constants:
        consts.update(constants)

    pa = get_particle_array(constants=consts, additional_props=extra_props,
                            **props)
    pa.add_property('body_id', type='int', data=body_id)
    pa.add_property('dem_id', type='int', data=dem_id)

    setup_quaternion_rigid_body(pa)

    pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'fz', 'm'])
    return pa


def normalize_q_orientation(q):
    norm_q = sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    q[:] = q[:] / norm_q


def quaternion_multiplication(p, q, res):
    """Parameters
    ----------
    p   : [float]
          An array of length four
    q   : [float]
          An array of length four
    res : [float]
          An array of length four

    Here `p` is a quaternion. i.e., p = [p.w, p.x, p.y, p.z]. And q is an
    another quaternion.

    This function is used to compute the rate of change of orientation
    when orientation is represented in terms of a quaternion. When the
    angular velocity is represented in terms of global frame
    \frac{dq}{dt} = \frac{1}{2} omega q

    http://www.ams.stonybrook.edu/~coutsias/papers/rrr.pdf
    see equation 8

    """
    res[0] = (p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3])
    res[1] = (p[0] * q[1] + q[0] * p[1] + p[2] * q[3] - p[3] * q[2])
    res[2] = (p[0] * q[2] + q[0] * p[2] + p[3] * q[1] - p[1] * q[3])
    res[3] = (p[0] * q[3] + q[0] * p[3] + p[1] * q[2] - p[2] * q[1])


def scale_quaternion(q, scale):
    q[0] = q[0] * scale
    q[1] = q[1] * scale
    q[2] = q[2] * scale
    q[3] = q[3] * scale


def quaternion_to_matrix(q, matrix):
    matrix[0] = 1. - 2. * (q[2]**2. + q[3]**2.)
    matrix[1] = 2. * (q[1] * q[2] + q[0] * q[3])
    matrix[2] = 2. * (q[1] * q[3] - q[0] * q[2])

    matrix[3] = 2. * (q[1] * q[2] - q[0] * q[3])
    matrix[4] = 1. - 2. * (q[1]**2. + q[3]**2.)
    matrix[5] = 2. * (q[2] * q[3] + q[0] * q[1])

    matrix[6] = 2. * (q[1] * q[3] + q[0] * q[2])
    matrix[7] = 2. * (q[2] * q[3] - q[0] * q[1])
    matrix[8] = 1. - 2. * (q[1]**2. + q[2]**2.)


class RK2StepRigidBodyQuaternionsPotyondy(IntegratorStep):
    def py_initialize(self, dst, t, dt):
        for j in range(3):
            # save the center of mass and center of mass velocity
            dst.cm0[j] = dst.cm[j]
            dst.vc0[j] = dst.vc[j]

            # save the current angular momentum
            dst.omega0[j] = dst.omega[j]

        # save the current orientation
        for j in range(4):
            dst.q0[j] = dst.q[j]

    def initialize(self):
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

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for j in range(3):
            # using velocity at t, move position
            # to t + dt/2.
            dst.cm[j] = dst.cm[j] + dtb2 * dst.vc[j]
            dst.vc[j] = dst.vc[j] + dtb2 * dst.force[j] / dst.total_mass[0]

        # change in quaternion
        delta_quat = np.array([0., 0., 0., 0.])
        # angular velocity magnitude
        omega_magn = sqrt(dst.omega[0]**2 + dst.omega[1]**2 + dst.omega[2]**2)
        axis_rot = np.array([0., 0., 0.])
        if omega_magn > 1e-12:
            axis_rot = dst.omega / omega_magn
        delta_quat[0] = cos(omega_magn * dtb2 * 0.5)
        delta_quat[1] = axis_rot[0] * sin(omega_magn * dtb2 * 0.5)
        delta_quat[2] = axis_rot[1] * sin(omega_magn * dtb2 * 0.5)
        delta_quat[3] = axis_rot[2] * sin(omega_magn * dtb2 * 0.5)

        res = np.array([0., 0., 0., 0.])
        quaternion_multiplication(dst.q, delta_quat, res)
        dst.q = res

        # normalize the orientation
        normalize_q_orientation(dst.q)

        # update the moment of inertia
        quaternion_to_matrix(dst.q, dst.R)
        R = dst.R.reshape(3, 3)
        R = R.T
        dst.R[:] = R.ravel()
        R_t = R.T
        tmp = np.matmul(R, dst.mib.reshape(3, 3))
        dst.mig[:] = (np.matmul(tmp, R_t)).ravel()
        # move angular velocity to t + dt/2.
        # omega_dot is
        tmp = dst.torque - np.cross(
            dst.omega, np.matmul(dst.mig.reshape(3, 3), dst.omega))
        omega_dot = np.matmul(dst.mig.reshape(3, 3), tmp)
        dst.omega = dst.omega0 + omega_dot * dtb2

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_cm, d_vc, d_R, d_omega):
        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[0] * d_dx0[d_idx] + d_R[1] * d_dy0[d_idx] +
              d_R[2] * d_dz0[d_idx])
        dy = (d_R[3] * d_dx0[d_idx] + d_R[4] * d_dy0[d_idx] +
              d_R[5] * d_dz0[d_idx])
        dz = (d_R[6] * d_dx0[d_idx] + d_R[7] * d_dy0[d_idx] +
              d_R[8] * d_dz0[d_idx])

        d_x[d_idx] = d_cm[0] + dx
        d_y[d_idx] = d_cm[1] + dy
        d_z[d_idx] = d_cm[2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[1] * dz - d_omega[2] * dy
        dv = d_omega[2] * dx - d_omega[0] * dz
        dw = d_omega[0] * dy - d_omega[1] * dx

        d_u[d_idx] = d_vc[0] + du
        d_v[d_idx] = d_vc[1] + dv
        d_w[d_idx] = d_vc[2] + dw

    def py_stage2(self, dst, t, dt):
        for j in range(3):
            dst.cm[j] = dst.cm0[j] + dt * dst.vc[j]
            dst.vc[j] = dst.vc0[j] + dt * dst.force[j] / dst.total_mass[0]

        # delta quaternion (change in quaternion)
        delta_quat = np.array([0., 0., 0., 0.])
        # angular velocity magnitude
        omega_magn = sqrt(dst.omega[0]**2 + dst.omega[1]**2 + dst.omega[2]**2)
        axis_rot = np.array([0., 0., 0.])
        if omega_magn > 1e-12:
            axis_rot = dst.omega / omega_magn
        delta_quat[0] = cos(omega_magn * dt * 0.5)
        delta_quat[1] = axis_rot[0] * sin(omega_magn * dt * 0.5)
        delta_quat[2] = axis_rot[1] * sin(omega_magn * dt * 0.5)
        delta_quat[3] = axis_rot[2] * sin(omega_magn * dt * 0.5)

        res = np.array([0., 0., 0., 0.])
        quaternion_multiplication(dst.q0, delta_quat, res)
        dst.q = res

        # normalize the orientation
        normalize_q_orientation(dst.q)

        # update the moment of inertia
        quaternion_to_matrix(dst.q, dst.R)
        R = dst.R.reshape(3, 3)
        R = R.T
        dst.R[:] = R.ravel()
        R_t = R.T
        tmp = np.matmul(R, dst.mib.reshape(3, 3))
        dst.mig[:] = (np.matmul(tmp, R_t)).ravel()
        # move angular velocity to t + dt/2.
        # omega_dot is
        tmp = dst.torque - np.cross(
            dst.omega, np.matmul(dst.mig.reshape(3, 3), dst.omega))
        omega_dot = np.matmul(dst.mig.reshape(3, 3), tmp)
        dst.omega = dst.omega0 + omega_dot * dt

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_cm, d_vc, d_R, d_omega):
        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[0] * d_dx0[d_idx] + d_R[1] * d_dy0[d_idx] +
              d_R[2] * d_dz0[d_idx])
        dy = (d_R[3] * d_dx0[d_idx] + d_R[4] * d_dy0[d_idx] +
              d_R[5] * d_dz0[d_idx])
        dz = (d_R[6] * d_dx0[d_idx] + d_R[7] * d_dy0[d_idx] +
              d_R[8] * d_dz0[d_idx])

        d_x[d_idx] = d_cm[0] + dx
        d_y[d_idx] = d_cm[1] + dy
        d_z[d_idx] = d_cm[2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[1] * dz - d_omega[2] * dy
        dv = d_omega[2] * dx - d_omega[0] * dz
        dw = d_omega[0] * dy - d_omega[1] * dx

        d_u[d_idx] = d_vc[0] + du
        d_v[d_idx] = d_vc[1] + dv
        d_w[d_idx] = d_vc[2] + dw


class RigidBodyQuaternionPotyondyScheme(Scheme):
    def __init__(self, bodies, solids, dim, rho0, kn, mu=0.5, en=1.0, gx=0.0,
                 gy=0.0, gz=0.0, debug=False):
        self.bodies = bodies
        self.solids = solids
        self.dim = dim
        self.rho0 = rho0
        self.kn = kn
        self.mu = mu
        self.en = en
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.debug = debug

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import CubicSpline
        from pysph.sph.integrator import EPECIntegrator
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        for body in self.bodies:
            if body not in steppers:
                steppers[body] = RK2StepRigidBodyQuaternions()

        cls = integrator_cls if integrator_cls is not None else EPECIntegrator
        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def get_equations(self):
        from pysph.sph.equation import Group
        equations = []
        g1 = []
        if self.solids is not None:
            all = self.bodies + self.solids
        else:
            all = self.bodies

        for name in self.bodies:
            g1.append(
                BodyForce(dest=name, sources=None, gx=self.gx, gy=self.gy,
                          gz=self.gz))
        equations.append(Group(equations=g1, real=False))

        g2 = []
        for name in self.bodies:
            g2.append(
                RigidBodyCollision(dest=name, sources=all, kn=self.kn,
                                   mu=self.mu, en=self.en))
        equations.append(Group(equations=g2, real=False))

        g3 = []
        for name in self.bodies:
            g3.append(SumUpExternalForces(dest=name, sources=None))
        equations.append(Group(equations=g3, real=False))

        return equations
