from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from compyle.api import declare
from math import sqrt, log
import numpy as np
from pysph.sph.scheme import Scheme
from pysph.sph.equation import Group, MultiStageEquations
from pysph.dem.discontinuous_dem.dem_nonlinear import (
    EPECIntegratorMultiStage, EulerIntegratorMultiStage)


def get_particle_array_dem_3d_linear_cundall(constants=None, **props):
    """Return a particle array for a dem particles
    """
    dem_props = [
        'wx0', 'wy0', 'wz0', 'wx', 'wy', 'wz', 'fx', 'fy', 'fz', 'torx',
        'tory', 'torz', 'rad_s', 'm_inverse', 'I_inverse', 'u0', 'v0', 'w0',
        'x0', 'y0', 'z0'
    ]

    dem_id = props.pop('dem_id', None)

    pa = get_particle_array(additional_props=dem_props, **props)

    pa.add_property('dem_id', type='int', data=dem_id)
    # create the array to save the tangential interaction particles
    # index and other variables
    limit = 30

    pa.add_constant('limit', limit)
    pa.add_property('tng_idx', stride=limit, type="int")
    pa.tng_idx[:] = -1
    pa.add_property('tng_idx_dem_id', stride=limit, type="int")
    pa.tng_idx_dem_id[:] = -1
    pa.add_property('tng_fx', stride=limit)
    pa.add_property('tng_fy', stride=limit)
    pa.add_property('tng_fz', stride=limit)
    pa.add_property('tng_fx0', stride=limit)
    pa.add_property('tng_fy0', stride=limit)
    pa.add_property('tng_fz0', stride=limit)
    pa.tng_fx[:] = 0.
    pa.tng_fy[:] = 0.
    pa.tng_fz[:] = 0.
    pa.tng_fx0[:] = 0.
    pa.tng_fy0[:] = 0.
    pa.tng_fz0[:] = 0.
    pa.add_property('total_tng_contacts', type="int")
    pa.total_tng_contacts[:] = 0

    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'm', 'pid', 'tag', 'gid', 'fx', 'fy',
        'fz', 'torx', 'tory', 'torz', 'I_inverse', 'm_inverse', 'rad_s',
        'dem_id', 'tng_idx', 'tng_idx_dem_id', 'total_tng_contacts', 'tng_fx',
        'tng_fy', 'tng_fz'
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


class Cundall3dForceEuler(Equation):
    """
    This is an extension of 2d dem Cundall force formulation to 3d
    formulation. This is detailed in

    DISCRETE ELEMENT METHOD FOR 3D
    SIMULATIONS OF MECHANICAL
    SYSTEMS OF NON-SPHERICAL
    GRANULAR MATERIALS

    by

    JIAN CHEN

    This is a thesis
    """

    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall3dForceEuler, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_wx, d_wy, d_wz, d_fx, d_fy, d_fz, d_tng_idx,
             d_tng_idx_dem_id, d_tng_fx, d_tng_fy, d_tng_fz, d_tng_fx0,
             d_tng_fy0, d_tng_fz0, d_total_tng_contacts, d_dem_id, d_limit,
             d_torx, d_tory, d_torz, VIJ, XIJ, RIJ, d_rad_s, s_idx, s_m, s_wx,
             s_wy, s_wz, s_rad_s, s_dem_id, dt):
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
            nz = -XIJ[2] * rinv

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
            vr_dot_nij = vr_x * nx + vr_y * ny + vr_z * nz
            vn_x = vr_dot_nij * nx
            vn_y = vr_dot_nij * ny
            vn_z = vr_dot_nij * nz

            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y
            vt_z = vr_z - vn_z
            # magnitude of the tangential velocity
            vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

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
            fn_x = -kn_overlap * nx - eta_n * vn_x
            fn_y = -kn_overlap * ny - eta_n * vn_y
            fn_z = -kn_overlap * nz - eta_n * vn_z

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
            ft_x = 0.
            ft_y = 0.
            ft_z = 0.

            if found == 0:
                found_at = q1
                d_tng_idx[found_at] = s_idx
                d_total_tng_contacts[d_idx] += 1
                d_tng_idx_dem_id[found_at] = s_dem_id[s_idx]

            # implies we are tracking the particle
            else:
                # -----------------------#
                # rotate the tangential force to the current plane
                # -----------------------#
                ft_magn = (d_tng_fx[found_at]**2. + d_tng_fy[found_at]**2. +
                           d_tng_fz[found_at]**2.)**0.5
                ft_dot_nij = (d_tng_fx[found_at] * nx + d_tng_fy[found_at] * ny
                              + d_tng_fz[found_at] * nz)
                # tangential force projected onto the current normal of the
                # contact place
                ft_px = d_tng_fx[found_at] - ft_dot_nij * nx
                ft_py = d_tng_fy[found_at] - ft_dot_nij * ny
                ft_pz = d_tng_fz[found_at] - ft_dot_nij * nz

                ftp_magn = (ft_px**2. + ft_py**2. + ft_pz**2.)**0.5
                if ftp_magn > 0:
                    one_by_ftp_magn = 1. / ftp_magn

                    tx = ft_px * one_by_ftp_magn
                    ty = ft_px * one_by_ftp_magn
                    tz = ft_px * one_by_ftp_magn
                else:
                    tx = -vt_x / vt_magn
                    ty = -vt_y / vt_magn
                    tz = -vt_z / vt_magn

                # rescale the projection by the magnitude of the
                # previous tangential force, which gives the tangential
                # force on the current plane
                ft_x = ft_magn * tx
                ft_y = ft_magn * ty
                ft_z = ft_magn * tz

                # (*) check against Coulomb criterion
                # Tangential force magnitude due to displacement
                ftr_magn = (ft_x * ft_x + ft_y * ft_y + ft_z * ft_z)**(0.5)
                fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

                # we have to compare with static friction, so
                # this mu has to be static friction coefficient
                fn_mu = self.mu * fn_magn

                if ftr_magn >= fn_mu:
                    # rescale the tangential displacement
                    d_tng_fx[found_at] = fn_mu * tx
                    d_tng_fy[found_at] = fn_mu * ty
                    d_tng_fz[found_at] = fn_mu * tz

                    # and also adjust the spring elongation
                    # at time t, which is used at stage 2 integrator
                    d_tng_fx0[found_at] = d_tng_fx[found_at]
                    d_tng_fy0[found_at] = d_tng_fy[found_at]
                    d_tng_fz0[found_at] = d_tng_fz[found_at]

                    # set the tangential force to static friction
                    # from Coulomb criterion
                    ft_x = fn_mu * tx
                    ft_y = fn_mu * ty
                    ft_z = fn_mu * tz

            d_tng_fx[found_at] -= self.kt * vt_x * dt
            d_tng_fy[found_at] -= self.kt * vt_y * dt
            d_tng_fz[found_at] -= self.kt * vt_z * dt

            d_fx[d_idx] += fn_x + ft_x
            d_fy[d_idx] += fn_y + ft_y
            d_fz[d_idx] += fn_z + ft_z

            # torque = n cross F
            d_torx[d_idx] += (ny * ft_z - nz * ft_y) * a_i
            d_tory[d_idx] += (nz * ft_x - nx * ft_z) * a_i
            d_torz[d_idx] += (nx * ft_y - ny * ft_x) * a_i


class Cundall3dForceStage1(Equation):
    """
    This is an extension of 2d dem Cundall force formulation to 3d
    formulation. This is detailed in

    DISCRETE ELEMENT METHOD FOR 3D
    SIMULATIONS OF MECHANICAL
    SYSTEMS OF NON-SPHERICAL
    GRANULAR MATERIALS

    by

    JIAN CHEN

    This is a thesis
    """

    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall3dForceStage1, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_wx, d_wy, d_wz, d_fx, d_fy, d_fz, d_tng_idx,
             d_tng_idx_dem_id, d_tng_fx, d_tng_fy, d_tng_fz, d_tng_fx0,
             d_tng_fy0, d_tng_fz0, d_total_tng_contacts, d_dem_id, d_limit,
             d_torx, d_tory, d_torz, VIJ, XIJ, RIJ, d_rad_s, s_idx, s_m, s_wx,
             s_wy, s_wz, s_rad_s, s_dem_id, dt):
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
            nz = -XIJ[2] * rinv

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
            vr_dot_nij = vr_x * nx + vr_y * ny + vr_z * nz
            vn_x = vr_dot_nij * nx
            vn_y = vr_dot_nij * ny
            vn_z = vr_dot_nij * nz

            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y
            vt_z = vr_z - vn_z
            # magnitude of the tangential velocity
            vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

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
            fn_x = -kn_overlap * nx - eta_n * vn_x
            fn_y = -kn_overlap * ny - eta_n * vn_y
            fn_z = -kn_overlap * nz - eta_n * vn_z

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
            ft_x = 0.
            ft_y = 0.
            ft_z = 0.

            if found == 0:
                found_at = q1
                d_tng_idx[found_at] = s_idx
                d_total_tng_contacts[d_idx] += 1
                d_tng_idx_dem_id[found_at] = s_dem_id[s_idx]

            # implies we are tracking the particle
            else:
                # -----------------------#
                # rotate the tangential force to the current plane
                # -----------------------#
                ft_magn = (d_tng_fx[found_at]**2. + d_tng_fy[found_at]**2. +
                           d_tng_fz[found_at]**2.)**0.5
                ft_dot_nij = (d_tng_fx[found_at] * nx + d_tng_fy[found_at] * ny
                              + d_tng_fz[found_at] * nz)
                # tangential force projected onto the current normal of the
                # contact place
                ft_px = d_tng_fx[found_at] - ft_dot_nij * nx
                ft_py = d_tng_fy[found_at] - ft_dot_nij * ny
                ft_pz = d_tng_fz[found_at] - ft_dot_nij * nz

                ftp_magn = (ft_px**2. + ft_py**2. + ft_pz**2.)**0.5
                if ftp_magn > 0:
                    one_by_ftp_magn = 1. / ftp_magn

                    tx = ft_px * one_by_ftp_magn
                    ty = ft_px * one_by_ftp_magn
                    tz = ft_px * one_by_ftp_magn
                else:
                    tx = -vt_x / vt_magn
                    ty = -vt_y / vt_magn
                    tz = -vt_z / vt_magn

                # rescale the projection by the magnitude of the
                # previous tangential force, which gives the tangential
                # force on the current plane
                ft_x = ft_magn * tx
                ft_y = ft_magn * ty
                ft_z = ft_magn * tz

                # (*) check against Coulomb criterion
                # Tangential force magnitude due to displacement
                ftr_magn = (ft_x * ft_x + ft_y * ft_y + ft_z * ft_z)**(0.5)
                fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

                # we have to compare with static friction, so
                # this mu has to be static friction coefficient
                fn_mu = self.mu * fn_magn

                if ftr_magn >= fn_mu:
                    # rescale the tangential displacement
                    d_tng_fx[found_at] = fn_mu * tx
                    d_tng_fy[found_at] = fn_mu * ty
                    d_tng_fz[found_at] = fn_mu * tz

                    # and also adjust the spring elongation
                    # at time t, which is used at stage 2 integrator
                    d_tng_fx0[found_at] = d_tng_fx[found_at]
                    d_tng_fy0[found_at] = d_tng_fy[found_at]
                    d_tng_fz0[found_at] = d_tng_fz[found_at]

                    # set the tangential force to static friction
                    # from Coulomb criterion
                    ft_x = fn_mu * tx
                    ft_y = fn_mu * ty
                    ft_z = fn_mu * tz

            # increment the tangential force to next time step
            dtb2 = dt / 2.
            d_tng_fx[found_at] -= self.kt * vt_x * dtb2
            d_tng_fy[found_at] -= self.kt * vt_y * dtb2
            d_tng_fz[found_at] -= self.kt * vt_z * dtb2

            d_fx[d_idx] += fn_x + ft_x
            d_fy[d_idx] += fn_y + ft_y
            d_fz[d_idx] += fn_z + ft_z

            # torque = n cross F
            d_torx[d_idx] += (ny * ft_z - nz * ft_y) * a_i
            d_tory[d_idx] += (nz * ft_x - nx * ft_z) * a_i
            d_torz[d_idx] += (nx * ft_y - ny * ft_x) * a_i


class Cundall3dForceStage2(Equation):
    """
    This is an extension of 2d dem Cundall force formulation to 3d
    formulation. This is detailed in

    DISCRETE ELEMENT METHOD FOR 3D
    SIMULATIONS OF MECHANICAL
    SYSTEMS OF NON-SPHERICAL
    GRANULAR MATERIALS

    by

    JIAN CHEN

    This is a thesis
    """

    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall3dForceStage2, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_wx, d_wy, d_wz, d_fx, d_fy, d_fz, d_tng_idx,
             d_tng_idx_dem_id, d_tng_fx, d_tng_fy, d_tng_fz, d_tng_fx0,
             d_tng_fy0, d_tng_fz0, d_total_tng_contacts, d_dem_id, d_limit,
             d_torx, d_tory, d_torz, VIJ, XIJ, RIJ, d_rad_s, s_idx, s_m, s_wx,
             s_wy, s_wz, s_rad_s, s_dem_id, dt):
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
            nz = -XIJ[2] * rinv

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
            vr_dot_nij = vr_x * nx + vr_y * ny + vr_z * nz
            vn_x = vr_dot_nij * nx
            vn_y = vr_dot_nij * ny
            vn_z = vr_dot_nij * nz

            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y
            vt_z = vr_z - vn_z
            # magnitude of the tangential velocity
            vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

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
            fn_x = -kn_overlap * nx - eta_n * vn_x
            fn_y = -kn_overlap * ny - eta_n * vn_y
            fn_z = -kn_overlap * nz - eta_n * vn_z

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
            ft_x = 0.
            ft_y = 0.
            ft_z = 0.

            if found == 1:
                # implies we are tracking the particle
                # -----------------------#
                # rotate the tangential force at time (t+dt/2.) to the current
                # plane
                # -----------------------#
                ft_magn = (d_tng_fx[found_at]**2. + d_tng_fy[found_at]**2. +
                           d_tng_fz[found_at]**2.)**0.5
                ft_dot_nij = (d_tng_fx[found_at] * nx + d_tng_fy[found_at] * ny
                              + d_tng_fz[found_at] * nz)
                # tangential force projected onto the current normal of the
                # contact place
                ft_px = d_tng_fx[found_at] - ft_dot_nij * nx
                ft_py = d_tng_fy[found_at] - ft_dot_nij * ny
                ft_pz = d_tng_fz[found_at] - ft_dot_nij * nz

                ftp_magn = (ft_px**2. + ft_py**2. + ft_pz**2.)**0.5
                if ftp_magn > 0:
                    one_by_ftp_magn = 1. / ftp_magn

                    tx = ft_px * one_by_ftp_magn
                    ty = ft_px * one_by_ftp_magn
                    tz = ft_px * one_by_ftp_magn
                else:
                    tx = -vt_x / vt_magn
                    ty = -vt_y / vt_magn
                    tz = -vt_z / vt_magn

                # rescale the projection by the magnitude of the
                # previous tangential force, which gives the tangential
                # force on the current plane
                ft_x = ft_magn * tx
                ft_y = ft_magn * ty
                ft_z = ft_magn * tz

                # (*) check against Coulomb criterion
                # Tangential force magnitude due to displacement
                ftr_magn = (ft_x * ft_x + ft_y * ft_y + ft_z * ft_z)**(0.5)
                fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

                # we have to compare with static friction, so
                # this mu has to be static friction coefficient
                fn_mu = self.mu * fn_magn

                if ftr_magn >= fn_mu:
                    # rescale the tangential displacement
                    d_tng_fx[found_at] = fn_mu * tx
                    d_tng_fy[found_at] = fn_mu * ty
                    d_tng_fz[found_at] = fn_mu * tz

                    # and also adjust the spring elongation
                    # at time t, which is used at stage 2 integrator
                    d_tng_fx0[found_at] = d_tng_fx[found_at]
                    d_tng_fy0[found_at] = d_tng_fy[found_at]
                    d_tng_fz0[found_at] = d_tng_fz[found_at]

                    # set the tangential force to static friction
                    # from Coulomb criterion
                    ft_x = fn_mu * tx
                    ft_y = fn_mu * ty
                    ft_z = fn_mu * tz

                # -----------------------#
                # rotate the tangential force at time (t) to the current
                # plane
                # -----------------------#
                ft0_magn = (d_tng_fx0[found_at]**2. + d_tng_fy0[found_at]**2. +
                            d_tng_fz0[found_at]**2.)**0.5
                ft0_dot_nij = (
                    d_tng_fx0[found_at] * nx + d_tng_fy0[found_at] * ny +
                    d_tng_fz0[found_at] * nz)
                # tangential force projected onto the current normal of the
                # contact place
                ft0_px = d_tng_fx0[found_at] - ft0_dot_nij * nx
                ft0_py = d_tng_fy0[found_at] - ft0_dot_nij * ny
                ft0_pz = d_tng_fz0[found_at] - ft0_dot_nij * nz

                ft0_p_magn = (ft0_px**2. + ft0_py**2. + ft0_pz**2.)**0.5
                if ftp_magn > 0:
                    one_by_ft0_p_magn = 1. / ft0_p_magn

                    tx = ft_px * one_by_ft0_p_magn
                    ty = ft_px * one_by_ft0_p_magn
                    tz = ft_px * one_by_ft0_p_magn
                else:
                    tx = -vt_x / vt_magn
                    ty = -vt_y / vt_magn
                    tz = -vt_z / vt_magn

                # rescale the projection by the magnitude of the
                # previous tangential force, which gives the tangential
                # force on the current plane
                d_tng_fx0[found_at] = ft0_magn * tx
                d_tng_fy0[found_at] = ft0_magn * ty
                d_tng_fz0[found_at] = ft0_magn * tz

            # increment the tangential force to next time step
            d_tng_fx[found_at] = d_tng_fx0[found_at] - self.kt * vt_x * dt
            d_tng_fy[found_at] = d_tng_fy0[found_at] - self.kt * vt_y * dt
            d_tng_fz[found_at] = d_tng_fz0[found_at] - self.kt * vt_z * dt

            d_fx[d_idx] += fn_x + ft_x
            d_fy[d_idx] += fn_y + ft_y
            d_fz[d_idx] += fn_z + ft_z

            # torque = n cross F
            d_torx[d_idx] += (ny * ft_z - nz * ft_y) * a_i
            d_tory[d_idx] += (nz * ft_x - nx * ft_z) * a_i
            d_torz[d_idx] += (nx * ft_y - ny * ft_x) * a_i


class UpdateTangentialContactsCundall3d(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_z, d_rad_s,
                        d_total_tng_contacts, d_tng_idx, d_limit, d_tng_fx,
                        d_tng_fy, d_tng_fz, d_tng_fx0, d_tng_fy0, d_tng_fz0,
                        d_tng_idx_dem_id, s_x, s_y, s_z, s_rad_s, s_dem_id):
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

                    overlap = d_rad_s[d_idx] + s_rad_s[sidx] - rij

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_tng_idx[k] = -1
                            d_tng_idx_dem_id[k] = -1
                            d_tng_fx[k] = 0.
                            d_tng_fy[k] = 0.
                            d_tng_fz[k] = 0.
                            # make tangential0 displacements zero
                            d_tng_fx0[k] = 0.
                            d_tng_fy0[k] = 0.
                            d_tng_fz0[k] = 0.
                        else:
                            # swap the current tracking index with the final
                            # contact index
                            d_tng_idx[k] = d_tng_idx[last_idx_tmp]
                            d_tng_idx[last_idx_tmp] = -1

                            # swap tangential x displacement
                            d_tng_fx[k] = d_tng_fx[last_idx_tmp]
                            d_tng_fx[last_idx_tmp] = 0.

                            # swap tangential y displacement
                            d_tng_fy[k] = d_tng_fy[last_idx_tmp]
                            d_tng_fy[last_idx_tmp] = 0.

                            # swap tangential z displacement
                            d_tng_fz[k] = d_tng_fz[last_idx_tmp]
                            d_tng_fz[last_idx_tmp] = 0.

                            # swap tangential idx dem id
                            d_tng_idx_dem_id[k] = d_tng_idx_dem_id[
                                last_idx_tmp]
                            d_tng_idx_dem_id[last_idx_tmp] = -1

                            # make tangential0 displacements zero
                            d_tng_fx0[last_idx_tmp] = 0.
                            d_tng_fy0[last_idx_tmp] = 0.
                            d_tng_fz0[last_idx_tmp] = 0.

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


class RK2StepDEM3dCundall(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v, d_w,
                   d_u0, d_v0, d_w0, d_tng_fx, d_tng_fy, d_tng_fz, d_tng_fx0,
                   d_tng_fy0, d_tng_fz0, d_total_tng_contacts, d_limit, d_wx,
                   d_wy, d_wz, d_wx0, d_wy0, d_wz0):

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
            d_tng_fx0[i] = d_tng_fx[i]
            d_tng_fy0[i] = d_tng_fy[i]
            d_tng_fz0[i] = d_tng_fz[i]

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_wx, d_wy, d_wz, d_torx, d_tory, d_torz, d_m_inverse,
               d_I_inverse, dt):
        dtb2 = dt / 2.

        d_x[d_idx] = d_x[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dtb2 * d_w[d_idx]

        d_u[d_idx] = d_u[d_idx] + dtb2 * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v[d_idx] + dtb2 * d_fy[d_idx] * d_m_inverse[d_idx]
        d_w[d_idx] = d_w[d_idx] + dtb2 * d_fz[d_idx] * d_m_inverse[d_idx]

        d_wx[d_idx] = d_wx[d_idx] + (dtb2 * d_torx[d_idx] * d_I_inverse[d_idx])
        d_wy[d_idx] = d_wy[d_idx] + (dtb2 * d_tory[d_idx] * d_I_inverse[d_idx])
        d_wz[d_idx] = d_wz[d_idx] + (dtb2 * d_torz[d_idx] * d_I_inverse[d_idx])

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_x0, d_y0, d_z0, d_u0, d_v0, d_w0, d_wx, d_wy, d_wz, d_wx0,
               d_wy0, d_wz0, d_torx, d_tory, d_torz, d_m_inverse, d_I_inverse,
               dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_fy[d_idx] * d_m_inverse[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_fz[d_idx] * d_m_inverse[d_idx]

        d_wx[d_idx] = d_wx0[d_idx] + (dt * d_torx[d_idx] * d_I_inverse[d_idx])
        d_wy[d_idx] = d_wy0[d_idx] + (dt * d_tory[d_idx] * d_I_inverse[d_idx])
        d_wz[d_idx] = d_wz0[d_idx] + (dt * d_torz[d_idx] * d_I_inverse[d_idx])


class EulerStepDEM3dCundall(IntegratorStep):
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_wx, d_wy, d_wz, d_torx, d_tory, d_torz, d_m_inverse,
               d_I_inverse, dt):
        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u[d_idx] + dt * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v[d_idx] + dt * d_fy[d_idx] * d_m_inverse[d_idx]
        d_w[d_idx] = d_v[d_idx] + dt * d_fz[d_idx] * d_m_inverse[d_idx]

        d_wx[d_idx] = d_wx[d_idx] + (dt * d_torx[d_idx] * d_I_inverse[d_idx])
        d_wy[d_idx] = d_wy[d_idx] + (dt * d_tory[d_idx] * d_I_inverse[d_idx])
        d_wz[d_idx] = d_wz[d_idx] + (dt * d_torz[d_idx] * d_I_inverse[d_idx])


class Dem3dCundallScheme(Scheme):
    def __init__(self, bodies, solids, integrator, dim, kn, mu=0.5, en=1.0, gx=0.0,
                 gy=0.0, gz=0.0, debug=False):
        self.bodies = bodies
        self.solids = solids
        self.dim = dim
        self.integrator = integrator
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
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        if self.integrator == "rk2":
            for body in self.bodies:
                if body not in steppers:
                    steppers[body] = RK2StepDEM3dCundall()

            cls = integrator_cls if integrator_cls is not None else EPECIntegratorMultiStage
        elif self.integrator == "euler":
            for body in self.bodies:
                if body not in steppers:
                    steppers[body] = EulerStepDEM3dCundall()

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
                BodyForce(dest=name, sources=None, gx=self.gx, gy=self.gy,
                          gz=self.gz))

        for name in self.bodies:
            g1.append(
                Cundall3dForceStage1(dest=name, sources=all, kn=self.kn,
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
                BodyForce(dest=name, sources=None, gx=self.gx, gy=self.gy,
                          gz=self.gz))

        for name in self.bodies:
            g1.append(
                Cundall3dForceStage2(dest=name, sources=all, kn=self.kn,
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
                BodyForce(dest=name, sources=None, gx=self.gx, gy=self.gy,
                          gz=self.gz))

        for name in self.bodies:
            g1.append(
                Cundall3dForceEuler(dest=name, sources=all, kn=self.kn,
                                    mu=self.mu, en=self.en))
        stage1.append(Group(equations=g1, real=False))

        return stage1

    def get_equations(self):
        if self.integrator == "rk2":
            return self.get_rk2_equations()
        elif self.integrator == "euler":
            return self.get_euler_equations()
