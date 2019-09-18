from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation
from pysph.cpy.api import declare
from pysph.sph.integrator_step import IntegratorStep
from math import sqrt


def get_particle_array_dem(dem_id, total_dem_entities, dim, **props):
    dem_props = [
        'wx', 'wy', 'wz', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'R',
        'm_inverse', 'I_inverse',
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
    pa.add_constant('tng_x', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('tng_y', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('tng_z', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('tng_x0', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('tng_y0', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('tng_z0', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('vtx', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('vty', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('vtz', [0.] * total_dem_entities * limit * len(pa.x))
    pa.add_constant('total_tng_contacts', [0] * total_dem_entities * len(pa.x))
    pa.add_constant('dem_id', dem_id)
    pa.add_constant('total_dem_entities', total_dem_entities)
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

        d_torx[d_idx] = 0
        d_tory[d_idx] = 0
        d_torz[d_idx] = 0


class LinearSpringForceParticleParticle(Equation):
    """Contact force law between two discrete element particles which are
    not bonded. The following code follows the notation from the paper:

    "Multi-level modeling of dense gas-solid two-phase flows" by Mao Ye.

    """

    def __init__(self, dest, sources, kn=1e3, eta_n=1., kt=1e3, eta_t=1.,
                 mu=0.3):
        super(LinearSpringForceParticleParticle, self).__init__(dest, sources)
        self.kn = kn
        self.kt_1 = 1 / kt
        self.eta_n = eta_n
        self.kt = kt
        self.eta_t = eta_t
        self.mu = mu

    def loop(self, d_idx, d_m, d_wx, d_wy, d_wz, d_fx, d_fy, d_fz, d_torx,
             d_tory, d_torz, d_tng_x, d_tng_y, d_tng_z, d_tng_x0, d_tng_y0,
             d_tng_z0, d_tng_idx, d_total_tng_contacts, d_dem_id, d_limit,
             d_vtx, d_vty, d_vtz, d_total_dem_entities, VIJ, XIJ, RIJ,
             d_R, s_idx, s_R, s_wx, s_wy, s_wz, s_dem_id, dt):
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
        tot_ctcs = (d_total_tng_contacts[d_idx * d_total_dem_entities[0] +
                                         s_dem_id[0]])
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
        # if particles are not overlapping
        if overlap <= 0:
            if found == 1:
                # make its tangential displacement to be zero
                d_tng_x0[found_at] = 0.
                d_tng_y0[found_at] = 0.
                d_tng_z0[found_at] = 0.

                d_tng_x[found_at] = 0.
                d_tng_y[found_at] = 0.
                d_tng_z[found_at] = 0.

                d_vtx[found_at] = 0.
                d_vty[found_at] = 0.
                d_vtz[found_at] = 0.

        # if particles are in contact
        else:
            # normal vector passing from s_idx to d_idx, i.e., j to i
            rinv = 1.0 / RIJ
            nx = XIJ[0] * rinv
            ny = XIJ[1] * rinv
            nz = XIJ[2] * rinv

            # ---- Realtive velocity computation (Eq 10)----
            # relative velocity of particle d_idx w.r.t particle s_idx at
            # contact point. The velocity different provided by PySPH is
            # only between translational velocities, but we need to consider
            # rotational velocities also.

            # Distance till contact point
            a_i = d_R[d_idx] - overlap / 2.
            a_j = s_R[s_idx] - overlap / 2.
            vr_x = VIJ[0] + (a_i * (ny * d_wz[d_idx] - nz * d_wy[d_idx]) +
                             a_j * (ny * s_wz[s_idx] - nz * s_wy[s_idx]))
            vr_y = VIJ[1] + (a_i * (nz * d_wx[d_idx] - nx * d_wz[d_idx]) +
                             a_j * (nz * s_wx[s_idx] - nx * s_wz[s_idx]))
            vr_z = VIJ[2] + (a_i * (nx * d_wy[d_idx] - ny * d_wx[d_idx]) +
                             a_j * (nx * s_wy[s_idx] - ny * s_wx[s_idx]))

            # normal velocity magnitude
            vr_dot_nij = vr_x * nx + vr_y * ny + vr_z * nz
            vn_x = vr_dot_nij * nx
            vn_y = vr_dot_nij * ny
            vn_z = vr_dot_nij * nz

            # tngential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y
            vt_z = vr_z - vn_z

            # tangential velocity magnitude
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**(1. / 2.)

            # normal force
            fn_x = self.kn * overlap * nx - self.eta_n * vn_x
            fn_y = self.kn * overlap * ny - self.eta_n * vn_y
            fn_z = self.kn * overlap * nz - self.eta_n * vn_z

            # ------------- tangential force computation ----------------
            # if the particle is not been tracked then assign an index in
            # tracking history.
            if found == 0:
                found_at = q
                d_tng_idx[found_at] = s_idx
                d_total_tng_contacts[d_idx * d_total_dem_entities[0] +
                                     s_dem_id[0]] += 1

                # compute and set the tangential acceleration for the current
                # time step
                d_vtx[found_at] = vt_x
                d_vty[found_at] = vt_y
                d_vtz[found_at] = vt_z
            else:
                # rotate the spring for current plane
                tng_dot_nij = (d_tng_x[found_at] * nx + d_tng_y[found_at] * ny
                               + d_tng_z[found_at] * nz)
                d_tng_x[found_at] -= tng_dot_nij * nx
                d_tng_y[found_at] -= tng_dot_nij * ny
                d_tng_z[found_at] -= tng_dot_nij * nz

                # compute and set the tangential acceleration for the current
                # time step
                d_vtx[found_at] = vt_x
                d_vty[found_at] = vt_y
                d_vtz[found_at] = vt_z

            # find the tangential force from the tangential displacement
            #  and tangential velocity (eq 18 Luding)
            ft0_x = -self.kt * d_tng_x[found_at] - self.eta_t * vt_x
            ft0_y = -self.kt * d_tng_y[found_at] - self.eta_t * vt_y
            ft0_z = -self.kt * d_tng_z[found_at] - self.eta_t * vt_z

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
                    d_tng_x[found_at] = -self.kt_1 * (
                        fn_mu * tx + self.eta_t * vt_x)
                    d_tng_y[found_at] = -self.kt_1 * (
                        fn_mu * ty + self.eta_t * vt_y)
                    d_tng_z[found_at] = -self.kt_1 * (
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


class UpdateTangentialContacts(Equation):
    def loop_all(self, d_idx, d_x, d_y, d_z, d_R, d_total_dem_entities,
                 d_total_tng_contacts, d_tng_idx, d_limit, d_tng_x, d_tng_y,
                 d_tng_z, s_x, s_y, s_z, s_R):
        i = declare('int')
        p = declare('int')
        count = declare('int')
        k = declare('int')
        xij = declare('matrix(3)')
        last_idx_tmp = declare('int')
        sidx = declare('int')
        rij = 0.0

        idx_total_ctcs_with_entity_i = declare('int')
        # loop over all entites
        for i in range(0, d_total_dem_entities[0]):
            idx_total_ctcs_with_entity_i = (
                d_total_tng_contacts[d_total_dem_entities[0] * d_idx + i])
            # particle idx contacts with entity i has range of indices
            # and the first index would be
            p = d_idx * d_total_dem_entities[0] * d_limit[0] + i * d_limit[0]
            last_idx_tmp = p + idx_total_ctcs_with_entity_i - 1
            k = p
            count = 0

            # loop over all the contacts of particle d_idx with entity i
            while count < idx_total_ctcs_with_entity_i:
                # The index of the particle with which
                # d_idx in contact is
                sidx = d_tng_idx[k]

                if sidx == -1:
                    break
                else:
                    xij[0] = d_x[d_idx] - s_x[sidx]
                    xij[1] = d_y[d_idx] - s_y[sidx]
                    xij[2] = d_z[d_idx] - s_z[sidx]
                    rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] +
                               xij[2] * xij[2])

                    overlap = d_R[d_idx] + s_R[sidx] - rij

                    if overlap < 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_tng_idx[k] = -1
                            d_tng_x[k] = 0.
                            d_tng_y[k] = 0.
                            d_tng_z[k] = 0.
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

                            # decrease the last_idx_tmp, since we swapped it to
                            # -1
                            last_idx_tmp -= 1
                    else:
                        k = k + 1

                    count += 1


class MakeForcesZero(Equation):
    def loop(self, d_idx, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz):
        d_fx[d_idx] = 0
        d_fy[d_idx] = 0
        d_fz[d_idx] = 0

        d_torx[d_idx] = 0
        d_tory[d_idx] = 0
        d_torz[d_idx] = 0


class EulerDEMStep(IntegratorStep):
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_torx, d_tory, d_torz, d_wx, d_wy, d_wz, d_m_inverse,
               d_I_inverse, d_total_dem_entities, d_total_tng_contacts,
               d_limit, d_tng_x, d_tng_y, d_tng_z, d_tng_x0, d_tng_y0,
               d_tng_z0, d_vtx, d_vty, d_vtz, dt):
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
                d_tng_x[j] = d_tng_x[j] + d_vtx[j] * dt
                d_tng_y[j] = d_tng_y[j] + d_vty[j] * dt
                d_tng_z[j] = d_tng_z[j] + d_vtz[j] * dt

        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u[d_idx] + dt * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v[d_idx] + dt * d_fy[d_idx] * d_m_inverse[d_idx]
        d_w[d_idx] = d_w[d_idx] + dt * d_fz[d_idx] * d_m_inverse[d_idx]

        d_wx[d_idx] = d_wx[d_idx] + dt * d_torx[d_idx] * d_I_inverse[d_idx]
        d_wy[d_idx] = d_wy[d_idx] + dt * d_tory[d_idx] * d_I_inverse[d_idx]
        d_wz[d_idx] = d_wz[d_idx] + dt * d_torz[d_idx] * d_I_inverse[d_idx]


class RK2DEMStep(IntegratorStep):
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_torx, d_tory, d_torz, d_wx, d_wy, d_wz, d_m_inverse,
               d_I_inverse, d_total_dem_entities, d_total_tng_contacts,
               d_limit, d_tng_x, d_tng_y, d_tng_z, d_tng_x0, d_tng_y0,
               d_tng_z0, d_vtx, d_vty, d_vtz, dt):
        dtb2 = dt / 2.

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
                d_tng_x[j] = d_tng_x0[j] + d_vtx[j] * dtb2
                d_tng_y[j] = d_tng_y0[j] + d_vty[j] * dtb2
                d_tng_z[j] = d_tng_z0[j] + d_vtz[j] * dtb2

        d_x[d_idx] = d_x[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dtb2 * d_w[d_idx]

        d_u[d_idx] = d_u[d_idx] + dtb2 * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v[d_idx] + dtb2 * d_fy[d_idx] * d_m_inverse[d_idx]
        d_w[d_idx] = d_w[d_idx] + dtb2 * d_fz[d_idx] * d_m_inverse[d_idx]

        d_wx[d_idx] = d_wx[d_idx] + dtb2 * d_torx[d_idx] * d_I_inverse[d_idx]
        d_wy[d_idx] = d_wy[d_idx] + dtb2 * d_tory[d_idx] * d_I_inverse[d_idx]
        d_wz[d_idx] = d_wz[d_idx] + dtb2 * d_torz[d_idx] * d_I_inverse[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_torx, d_tory, d_torz, d_wx, d_wy, d_wz, d_m_inverse,
               d_I_inverse, d_total_dem_entities, d_total_tng_contacts,
               d_limit, d_tng_x, d_tng_y, d_tng_z, d_tng_x0, d_tng_y0,
               d_tng_z0, d_vtx, d_vty, d_vtz, dt):
        i = declare('int')
        j = declare('int')
        p = declare('int')
        q = declare('int')
        idx_total_ctcs_with_entity_i = declare('int')

        # loop over all the entites
        for i in range(0, d_total_dem_entities[0]):
            idx_total_ctcs_with_entity_i = (
                d_total_tng_contacts[d_total_dem_entities[0] * d_idx + i])
            # particle idx contacts with entity i has range of indices
            # and the first index would be
            p = d_idx * d_total_dem_entities[0] * d_limit[0] + i * d_limit[0]
            # and the last index would be
            q = p + idx_total_ctcs_with_entity_i

            # loop over all the contacts of particle d_idx with entity i
            for j in range(p, q):
                d_tng_x[j] = d_tng_x0[j] + d_vtx[j] * dt
                d_tng_y[j] = d_tng_y0[j] + d_vty[j] * dt
                d_tng_z[j] = d_tng_z0[j] + d_vtz[j] * dt

        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u[d_idx] + dt * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v[d_idx] + dt * d_fy[d_idx] * d_m_inverse[d_idx]
        d_w[d_idx] = d_w[d_idx] + dt * d_fz[d_idx] * d_m_inverse[d_idx]

        d_wx[d_idx] = d_wx[d_idx] + dt * d_torx[d_idx] * d_I_inverse[d_idx]
        d_wy[d_idx] = d_wy[d_idx] + dt * d_tory[d_idx] * d_I_inverse[d_idx]
        d_wz[d_idx] = d_wz[d_idx] + dt * d_torz[d_idx] * d_I_inverse[d_idx]
