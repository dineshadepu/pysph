from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation
from pysph.cpy.api import declare
from pysph.sph.integrator_step import IntegratorStep
from math import (sqrt, asin, sin, cos)


def get_particle_array_dem(dem_id, total_dem_entities, dim, **props):
    dem_props = [
        'wx', 'wy', 'wz', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'R',
        'm_inverse', 'I_inverse'
    ]

    pa = get_particle_array(additional_props=dem_props, **props)

    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'wx', 'wy', 'wz', 'm', 'p', 'pid', 'tag',
        'gid', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'I_inverse'
    ])
    # create the array to save the tangential interaction particles
    # index and other variables
    if dim == 2:
        limit = 6
    else:
        limit = 30

    pa.add_constant('limit', limit)
    pa.add_constant('tng_idx', [-1] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('tng_disp_x',
                    [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('tng_disp_y',
                    [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('tng_disp_z',
                    [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('tng_disp_x0',
                    [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('tng_disp_y0',
                    [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('tng_disp_z0',
                    [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('n_x', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('n_y', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('n_z', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('n_x0', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('n_y0', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('n_z0', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('total_tng_contacts', [0] * total_dem_entities * len(pa.x))
    pa.add_constant('dem_id', dem_id)
    pa.add_constant('total_dem_entities', total_dem_entities)
    return pa


class DEMLinearForceRK2Stage1(Equation):
    """Contact force law between two discrete element particles which are
    not bonded. The following code follows the notation from the paper:

    "Multi-level modeling of dense gas-solid two-phase flows" by Mao Ye.

    """

    def __init__(self, dest, sources, kn=1e3, eta_n=1., kt=1e3, eta_t=1.,
                 mu=0.3):
        super(DEMLinearForceRK2Stage1, self).__init__(dest, sources)
        self.kn = kn
        self.kt_1 = 1 / kt
        self.eta_n = eta_n
        self.kt = kt
        self.eta_t = eta_t
        self.mu = mu

    def loop(self, d_idx, d_m, d_wx, d_wy, d_wz, d_fx, d_fy, d_fz, d_torx,
             d_tory, d_torz, d_tng_disp_x, d_tng_disp_y, d_tng_disp_z,
             d_tng_disp_x0, d_tng_disp_y0, d_tng_disp_z0, d_tng_idx,
             d_total_tng_contacts, d_dem_id, d_limit, d_total_dem_entities,
             VIJ, XIJ, RIJ, d_R, d_nx, d_ny, d_nz, d_nx0, d_ny0, d_nz0, s_idx,
             s_R, s_wx, s_wy, s_wz, s_dem_id, dt):
        overlap = -1.
        # check if we are not dealing with the same particle
        if RIJ > 0:
            overlap = d_R[d_idx] + s_R[s_idx] - RIJ

        # d_idx has a range of tracking indices with source dem_id.
        # starting index is p
        p = declare('int')
        # ending index is q -1
        q = declare('int')
        # total number of contacts of particle i in destination
        # with source entity
        tot_ctcs = declare('int')
        tot_ctcs = (d_total_tng_contacts[d_idx * d_total_dem_entities[0]
                                         + s_dem_id[0]])
        p = (d_idx * d_total_dem_entities[0] * d_limit[0] +
             s_dem_id[0] * d_limit[0])
        q = p + tot_ctcs

        i = declare('int')
        found_at = declare('int')
        found = declare('int')

        # check if the particle is in the tracking list
        # if so, then save the location at found_at
        found = 0
        for i in range(p, q):
            if s_idx == d_tng_idx[i]:
                found_at = i
                found = 1
                break

        # ---------- force computation starts ------------
        if overlap > 0.:
            dtb2 = dt / 2.
            # normal vector passing from d_idx to s_idx, i.e., i to j
            rinv = 1.0 / RIJ
            nx = -XIJ[0] * rinv
            ny = -XIJ[1] * rinv
            nz = -XIJ[2] * rinv

            # relative velocity of particle d_idx w.r.t particle s_idx at
            # contact point. The velocity different provided by PySPH is
            # only between translational velocities, but we need to consider
            # rotational velocities also.

            # Distance till contact point
            a_i = d_R[d_idx] - overlap / 2.
            a_j = s_R[s_idx] - overlap / 2.
            vr_x = VIJ[0] + (a_i *
                             (nz * d_wy[d_idx] - ny * d_wz[d_idx]) + a_j *
                             (nz * s_wy[s_idx] - ny * s_wz[s_idx]))
            vr_y = VIJ[1] + (a_i *
                             (nx * d_wz[d_idx] - nz * d_wx[d_idx]) + a_j *
                             (nx * s_wz[s_idx] - nz * s_wx[s_idx]))
            vr_z = VIJ[2] + (a_i *
                             (ny * d_wx[d_idx] - nx * d_wy[d_idx]) + a_j *
                             (ny * s_wx[s_idx] - nx * s_wy[s_idx]))

            # normal velocity magnitude
            vr_dot_nij = vr_x * nx + vr_y * ny + vr_z * nz
            vn_x = vr_dot_nij * nx
            vn_y = vr_dot_nij * ny
            vn_z = vr_dot_nij * nz

            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y
            vt_z = vr_z - vn_z

            # tangential velocity magnitude
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**(1. / 2.)

            # normal force
            fn_x = -self.kn * overlap * nx - self.eta_n * vn_x
            fn_y = -self.kn * overlap * ny - self.eta_n * vn_y
            fn_z = -self.kn * overlap * nz - self.eta_n * vn_z

            # ------------- tangential force computation ----------------
            # if the particle is not been tracked then assign an index in
            # tracking history.
            if found == 0:
                found_at = q
                d_tng_idx[found_at] = s_idx
                d_total_tng_contacts[d_idx * d_total_dem_entities[0]
                                     + s_dem_id[0]] += 1

                # setup the contact such that it was present it time
                # of initialize
                d_tng_disp_x0[found_at] = 0.
                d_tng_disp_y0[found_at] = 0.
                d_tng_disp_z0[found_at] = 0.
                d_tng_disp_x[found_at] = vt_x * dtb2
                d_tng_disp_y[found_at] = vt_y * dtb2
                d_tng_disp_z[found_at] = vt_z * dtb2
                d_nx0[found_at] = nx
                d_ny0[found_at] = ny
                d_nz0[found_at] = nz
                d_nx[found_at] = nx
                d_ny[found_at] = ny
                d_nz[found_at] = nz
                ft0_x = 0.
                ft0_y = 0.
                ft0_z = 0.
            else:
                # rotate the spring for current plane
                # find the rotation matrix using the normal (n_0) at previous
                # time (t-dt/2)
                # and normal (n) at current time (time t)
                # the vector about which we need to rotate is computed
                # by (n x n_0), lets call it h (h_x, h_y, h_z)
                h_x = ny * d_nz0[found_at] - nz * d_ny0[found_at]
                h_y = nz * d_nx0[found_at] - nx * d_nz0[found_at]
                h_z = nx * d_ny0[found_at] - ny * d_nx0[found_at]
                h_magn = sqrt(h_x * h_x + h_y * h_y + h_z * h_z)

                # make it a unit vector by dividing by magnitude
                h_x = h_x / h_magn
                h_y = h_y / h_magn
                h_z = h_z / h_magn

                # compute the angle of rotation
                phi = asin(h_magn)
                # get sin cos of phi
                c = cos(phi)
                s = sin(phi)
                one_minus_c = 1. - c

                # compute the matrix
                h_00 = one_minus_c * h_x * h_x + c
                h_01 = one_minus_c * h_x * h_y - s * h_z
                h_02 = one_minus_c * h_x * h_z + s * h_y

                h_10 = one_minus_c * h_x * h_y + s * h_z
                h_11 = one_minus_c * h_y * h_y + c
                h_12 = one_minus_c * h_y * h_z - s * h_x

                h_20 = one_minus_c * h_x * h_z - s * h_y
                h_21 = one_minus_c * h_y * h_z + s * h_x
                h_22 = one_minus_c * h_z * h_z + c

                # rotated spring components are
                # now rotate it and save the tang disp
                d_tng_disp_x[found_at] = (h_00 * d_tng_disp_x0[found_at] +
                                          h_01 * d_tng_disp_y0[found_at] +
                                          h_02 * d_tng_disp_z0[found_at])
                d_tng_disp_y[found_at] = (h_10 * d_tng_disp_x0[found_at] +
                                          h_11 * d_tng_disp_y0[found_at] +
                                          h_12 * d_tng_disp_z0[found_at])
                d_tng_disp_z[found_at] = (h_20 * d_tng_disp_x0[found_at] +
                                          h_21 * d_tng_disp_y0[found_at] +
                                          h_22 * d_tng_disp_z0[found_at])
                # find the tangential force from the tangential displacement
                #  and tangential velocity
                ft0_x = -self.kt * d_tng_disp_x[found_at] - self.eta_t * vt_x
                ft0_y = -self.kt * d_tng_disp_y[found_at] - self.eta_t * vt_y
                ft0_z = -self.kt * d_tng_disp_z[found_at] - self.eta_t * vt_z

                # increment the tangential displacement
                d_tng_disp_x[found_at] += vt_x * dtb2
                d_tng_disp_y[found_at] += vt_y * dtb2
                d_tng_disp_z[found_at] += vt_z * dtb2
                # save the current normal at time t0 to d_n
                d_nx[found_at] = nx
                d_ny[found_at] = ny
                d_nz[found_at] = nz

            # (*) check against Coulomb criterion
            # Tangential force magnitude due to displacement
            ft0_magn = (ft0_x * ft0_x + ft0_y * ft0_y + ft0_z * ft0_z)**(0.5)
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
                    tx = ft0_x / ft0_magn
                    ty = ft0_y / ft0_magn
                    tz = ft0_z / ft0_magn
                    d_tng_disp_x[found_at] = -self.kt_1 * (
                        fn_mu * tx + self.eta_t * vt_x)
                    d_tng_disp_y[found_at] = -self.kt_1 * (
                        fn_mu * ty + self.eta_t * vt_y)
                    d_tng_disp_z[found_at] = -self.kt_1 * (
                        fn_mu * tz + self.eta_t * vt_z)

                    # set the tangential force to static friction
                    # from Coulomb criterion
                    ft0_x = fn_mu * tx
                    ft0_y = fn_mu * ty
                    ft0_z = fn_mu * tz

            d_fx[d_idx] += fn_x + ft0_x
            d_fy[d_idx] += fn_y + ft0_y
            d_fz[d_idx] += fn_z + ft0_z

            d_torx[d_idx] += (ft0_y * nz - ft0_z * ny) * a_i
            d_tory[d_idx] += (ft0_z * nx - ft0_x * nz) * a_i
            d_torz[d_idx] += (ft0_x * ny - ft0_y * nx) * a_i


class DEMLinearForceRK2Stage2(Equation):
    """Contact force law between two discrete element particles which are
    not bonded. The following code follows the notation from the paper:

    "Multi-level modeling of dense gas-solid two-phase flows" by Mao Ye.

    """

    def __init__(self, dest, sources, kn=1e3, eta_n=1., kt=1e3, eta_t=1.,
                 mu=0.3):
        super(DEMLinearForceRK2Stage1, self).__init__(dest, sources)
        self.kn = kn
        self.kt_1 = 1 / kt
        self.eta_n = eta_n
        self.kt = kt
        self.eta_t = eta_t
        self.mu = mu

    def loop(self, d_idx, d_m, d_wx, d_wy, d_wz, d_fx, d_fy, d_fz, d_torx,
             d_tory, d_torz, d_tng_disp_x, d_tng_disp_y, d_tng_disp_z,
             d_tng_disp_x0, d_tng_disp_y0, d_tng_disp_z0, d_tng_idx,
             d_total_tng_contacts, d_dem_id, d_limit, d_total_dem_entities,
             VIJ, XIJ, RIJ, d_R, d_nx, d_ny, d_nz, d_nx0, d_ny0, d_nz0, s_idx,
             s_R, s_wx, s_wy, s_wz, s_dem_id, dt):
        overlap = -1.
        # check if we are not dealing with the same particle
        if RIJ > 0:
            overlap = d_R[d_idx] + s_R[s_idx] - RIJ

        # d_idx has a range of tracking indices with source dem_id.
        # starting index is p
        p = declare('int')
        # ending index is q -1
        q = declare('int')
        # total number of contacts of particle i in destination
        # with source entity
        tot_ctcs = declare('int')
        tot_ctcs = (d_total_tng_contacts[d_idx * d_total_dem_entities[0]
                                         + s_dem_id[0]])
        p = (d_idx * d_total_dem_entities[0] * d_limit[0] +
             s_dem_id[0] * d_limit[0])
        q = p + tot_ctcs

        i = declare('int')
        found_at = declare('int')
        found = declare('int')

        # check if the particle is in the tracking list
        # if so, then save the location at found_at
        found = 0
        for i in range(p, q):
            if s_idx == d_tng_idx[i]:
                found_at = i
                found = 1
                break

        # ---------- force computation starts ------------
        if overlap > 0.:
            # normal vector passing from d_idx to s_idx, i.e., i to j
            # this is at time t + dt / 2
            rinv = 1.0 / RIJ
            nx = -XIJ[0] * rinv
            ny = -XIJ[1] * rinv
            nz = -XIJ[2] * rinv

            # relative velocity of particle d_idx w.r.t particle s_idx at
            # contact point. The velocity different provided by PySPH is
            # only between translational velocities, but we need to consider
            # rotational velocities also.

            # Distance till contact point
            a_i = d_R[d_idx] - overlap / 2.
            a_j = s_R[s_idx] - overlap / 2.
            vr_x = VIJ[0] + (a_i *
                             (nz * d_wy[d_idx] - ny * d_wz[d_idx]) + a_j *
                             (nz * s_wy[s_idx] - ny * s_wz[s_idx]))
            vr_y = VIJ[1] + (a_i *
                             (nx * d_wz[d_idx] - nz * d_wx[d_idx]) + a_j *
                             (nx * s_wz[s_idx] - nz * s_wx[s_idx]))
            vr_z = VIJ[2] + (a_i *
                             (ny * d_wx[d_idx] - nx * d_wy[d_idx]) + a_j *
                             (ny * s_wx[s_idx] - nx * s_wy[s_idx]))

            # normal velocity magnitude
            vr_dot_nij = vr_x * nx + vr_y * ny + vr_z * nz
            vn_x = vr_dot_nij * nx
            vn_y = vr_dot_nij * ny
            vn_z = vr_dot_nij * nz

            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y
            vt_z = vr_z - vn_z

            # tangential velocity magnitude
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**(1. / 2.)

            # normal force
            fn_x = -self.kn * overlap * nx - self.eta_n * vn_x
            fn_y = -self.kn * overlap * ny - self.eta_n * vn_y
            fn_z = -self.kn * overlap * nz - self.eta_n * vn_z

            # ------------- tangential force computation ----------------
            # Compute tangential force only if the particle is already been
            # tracked
            if found == 1:
                # rotate the spring for current plane find the rotation matrix
                # using the normal (nx, ny, nz) at previous time (t) and normal
                # (n) at current time (time t+dt/2) the vector about which we
                # need to rotate is computed by (n_current cross n), lets call
                # it h (h_x, h_y, h_z)
                h_x = ny * d_nz[found_at] - nz * d_ny[found_at]
                h_y = nz * d_nx[found_at] - nx * d_nz[found_at]
                h_z = nx * d_ny[found_at] - ny * d_nx[found_at]
                h_magn = sqrt(h_x * h_x + h_y * h_y + h_z * h_z)

                # make it a unit vector by dividing by magnitude
                h_x = h_x / h_magn
                h_y = h_y / h_magn
                h_z = h_z / h_magn

                # compute the angle of rotation
                phi = asin(h_magn)
                # get sin cos of phi
                c = cos(phi)
                s = sin(phi)
                one_minus_c = 1. - c

                # compute the matrix
                h_00 = one_minus_c * h_x * h_x + c
                h_01 = one_minus_c * h_x * h_y - s * h_z
                h_02 = one_minus_c * h_x * h_z + s * h_y

                h_10 = one_minus_c * h_x * h_y + s * h_z
                h_11 = one_minus_c * h_y * h_y + c
                h_12 = one_minus_c * h_y * h_z - s * h_x

                h_20 = one_minus_c * h_x * h_z - s * h_y
                h_21 = one_minus_c * h_y * h_z + s * h_x
                h_22 = one_minus_c * h_z * h_z + c

                # let unrotated tang spring disp be
                tng_disp_x_star = d_tng_disp_x[found_at]
                tng_disp_y_star = d_tng_disp_y[found_at]
                tng_disp_z_star = d_tng_disp_z[found_at]

                # rotated spring components are
                # now rotate it and save the tang disp
                d_tng_disp_x[found_at] = (
                    h_00 * tng_disp_x_star + h_01 * tng_disp_y_star +
                    h_02 * tng_disp_z_star)
                d_tng_disp_y[found_at] = (
                    h_10 * tng_disp_x_star + h_11 * tng_disp_y_star +
                    h_12 * tng_disp_z_star)
                d_tng_disp_z[found_at] = (
                    h_20 * tng_disp_x_star + h_21 * tng_disp_y_star +
                    h_22 * tng_disp_z_star)
                # find the tangential force from the tangential displacement
                # and tangential velocity
                ft0_x = -self.kt * d_tng_disp_x[found_at] - self.eta_t * vt_x
                ft0_y = -self.kt * d_tng_disp_y[found_at] - self.eta_t * vt_y
                ft0_z = -self.kt * d_tng_disp_z[found_at] - self.eta_t * vt_z

                # increment the tangential displacement

                # this is done by first rotating the tangential displacement at
                # time t0 and adding the increment

                # rotate the spring for current plane find the rotation matrix
                # using the normal (nx0, ny0, nz0) at previous time (t0-dt/2)
                # and normal (n) at current time (time t+dt/2) the vector about
                # which we need to rotate is computed by (n cross n0),
                # lets call it h (h_x, h_y, h_z)
                h_x = ny * d_nz0[found_at] - nz * d_ny0[found_at]
                h_y = nz * d_nx0[found_at] - nx * d_nz0[found_at]
                h_z = nx * d_ny0[found_at] - ny * d_nx0[found_at]
                h_magn = sqrt(h_x * h_x + h_y * h_y + h_z * h_z)

                # make it a unit vector by dividing by magnitude
                h_x = h_x / h_magn
                h_y = h_y / h_magn
                h_z = h_z / h_magn

                # compute the angle of rotation
                phi = asin(h_magn)
                # get sin cos of phi
                c = cos(phi)
                s = sin(phi)
                one_minus_c = 1. - c

                # compute the matrix
                h_00 = one_minus_c * h_x * h_x + c
                h_01 = one_minus_c * h_x * h_y - s * h_z
                h_02 = one_minus_c * h_x * h_z + s * h_y

                h_10 = one_minus_c * h_x * h_y + s * h_z
                h_11 = one_minus_c * h_y * h_y + c
                h_12 = one_minus_c * h_y * h_z - s * h_x

                h_20 = one_minus_c * h_x * h_z - s * h_y
                h_21 = one_minus_c * h_y * h_z + s * h_x
                h_22 = one_minus_c * h_z * h_z + c

                # let unrotated tang spring disp be
                tng_disp_x_star = d_tng_disp_x0[found_at]
                tng_disp_y_star = d_tng_disp_y0[found_at]
                tng_disp_z_star = d_tng_disp_z0[found_at]

                # rotated spring components are
                # now rotate it and save the tang disp
                d_tng_disp_x0[found_at] = (
                    h_00 * tng_disp_x_star + h_01 * tng_disp_y_star +
                    h_02 * tng_disp_z_star)
                d_tng_disp_y0[found_at] = (
                    h_10 * tng_disp_x_star + h_11 * tng_disp_y_star +
                    h_12 * tng_disp_z_star)
                d_tng_disp_z0[found_at] = (
                    h_20 * tng_disp_x_star + h_21 * tng_disp_y_star +
                    h_22 * tng_disp_z_star)

                # increment the tangential displacement to time t0 + dt
                d_tng_disp_x[found_at] = d_tng_disp_x0[found_at] + vt_x * dt
                d_tng_disp_y[found_at] = d_tng_disp_y0[found_at] + vt_y * dt
                d_tng_disp_z[found_at] = d_tng_disp_z0[found_at] + vt_z * dt

                # save the current normal at time t0 + dt to d_n
                d_nx[found_at] = nx
                d_ny[found_at] = ny
                d_nz[found_at] = nz

            # (*) check against Coulomb criterion
            # Tangential force magnitude due to displacement
            ft0_magn = (ft0_x * ft0_x + ft0_y * ft0_y + ft0_z * ft0_z)**(0.5)
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
                    tx = ft0_x / ft0_magn
                    ty = ft0_y / ft0_magn
                    tz = ft0_z / ft0_magn
                    d_tng_disp_x[found_at] = -self.kt_1 * (
                        fn_mu * tx + self.eta_t * vt_x)
                    d_tng_disp_y[found_at] = -self.kt_1 * (
                        fn_mu * ty + self.eta_t * vt_y)
                    d_tng_disp_z[found_at] = -self.kt_1 * (
                        fn_mu * tz + self.eta_t * vt_z)

                    # set the tangential force to static friction
                    # from Coulomb criterion
                    ft0_x = fn_mu * tx
                    ft0_y = fn_mu * ty
                    ft0_z = fn_mu * tz

            d_fx[d_idx] += fn_x + ft0_x
            d_fy[d_idx] += fn_y + ft0_y
            d_fz[d_idx] += fn_z + ft0_z

            d_torx[d_idx] += (ft0_y * nz - ft0_z * ny) * a_i
            d_tory[d_idx] += (ft0_z * nx - ft0_x * nz) * a_i
            d_torz[d_idx] += (ft0_x * ny - ft0_y * nx) * a_i


class RK2DEMStep(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_x0, d_y0, d_z0,
                   d_u0, d_v0, d_w0, d_total_dem_entities,
                   d_total_tng_contacts, d_limit, d_tng_disp_x, d_tng_disp_y,
                   d_tng_disp_z, d_tng_disp_x0, d_tng_disp_y0, d_tng_disp_z0,
                   d_nx, d_ny, d_nz, d_nx0, d_ny0, d_nz0):
        i = declare('int')
        j = declare('int')
        p = declare('int')
        q = declare('int')

        idx_total_ctcs_with_entity_i = declare('int')
        # loop over all entites
        for i in range(0, d_total_dem_entities[0]):
            idx_total_ctcs_with_entity_i = (
                d_total_tng_contacts[d_total_dem_entities[0] * d_idx + i])
            # particle idx contacts with entity i has range of indices
            # and the first index would be
            p = d_idx * d_total_dem_entities[0] * d_limit[0] + i * d_limit[0]
            q = p + idx_total_ctcs_with_entity_i

            # loop over all the contacts of particle d_idx with entity i
            for j in range(p, q):
                d_tng_disp_x0[j] = d_tng_disp_x[j]
                d_tng_disp_y0[j] = d_tng_disp_y[j]
                d_tng_disp_z0[j] = d_tng_disp_z[j]
                d_nx0[j] = d_nx[j]
                d_ny0[j] = d_ny[j]
                d_nz0[j] = d_nz[j]
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_u0, d_v0, d_w0, d_x0, d_y0, d_z0, d_wx0, d_wy0, d_wz0, d_torx,
               d_tory, d_torz, d_wx, d_wy, d_wz, d_m_inverse, d_I_inverse,
               d_total_dem_entities, d_total_tng_contacts, d_limit,
               d_tng_disp_x, d_tng_disp_y, d_tng_disp_z, d_tng_disp_x0,
               d_tng_disp_y0, d_tng_disp_z0, d_vtx, d_vty, d_vtz, dt):
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

    def stage2(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v, d_w,
               d_u0, d_v0, d_w0, d_wx0, d_wy0, d_wz0, d_fx, d_fy, d_fz, d_torx,
               d_tory, d_torz, d_wx, d_wy, d_wz, d_m_inverse, d_I_inverse,
               d_total_dem_entities, d_total_tng_contacts, d_limit,
               d_tng_disp_x, d_tng_disp_y, d_tng_disp_z, d_tng_disp_x0,
               d_tng_disp_y0, d_tng_disp_z0, d_vtx, d_vty, d_vtz, dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_fy[d_idx] * d_m_inverse[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_fz[d_idx] * d_m_inverse[d_idx]

        d_wx[d_idx] = d_wx0[d_idx] + dt * d_torx[d_idx] * d_I_inverse[d_idx]
        d_wy[d_idx] = d_wy0[d_idx] + dt * d_tory[d_idx] * d_I_inverse[d_idx]
        d_wz[d_idx] = d_wz0[d_idx] + dt * d_torz[d_idx] * d_I_inverse[d_idx]
