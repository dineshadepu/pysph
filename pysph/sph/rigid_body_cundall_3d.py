from pysph.base.reduce_array import parallel_reduce_array
from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
import numpy as np
import numpy
from math import sqrt, asin, sin, cos, pi, log
from pysph.sph.rigid_body_setup import (setup_quaternion_rigid_body)
from pysph.sph.rigid_body_cundall_2d import (
    RK2StepRigidBodyQuaternionsDEMCundall2d)
from compyle.api import declare


##########################################
# Rigid body simulation using quaternion #
##########################################
def get_particle_array_rigid_body_cundall_dem_3d(constants=None, **props):
    extra_props = [
        'fx', 'fy', 'fz', 'dx0', 'dy0', 'dz0', 'nx0', 'ny0', 'nz0', 'nx', 'ny',
        'nz', 'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'd_au', 'd_av', 'd_aw'
    ]

    body_id = props.pop('body_id', None)
    nb = 1 if body_id is None else numpy.max(body_id) + 1

    dem_id = props.pop('dem_id', None)

    consts = {
        'total_mass': numpy.zeros(nb, dtype=float),
        'cm': numpy.zeros(3 * nb, dtype=float),
        'cm0': numpy.zeros(3 * nb, dtype=float),
        'q': [1., 0., 0., 0.] * nb,
        'q0': [1., 0., 0., 0.] * nb,
        'qdot': numpy.zeros(4 * nb, dtype=float),
        'R': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
        # moment of inertia inverse in body frame
        'mib': numpy.zeros(9 * nb, dtype=float),
        # moment of inertia inverse in global frame
        'mig': numpy.zeros(9 * nb, dtype=float),
        # total force at the center of mass
        'force': numpy.zeros(3 * nb, dtype=float),
        # torque about the center of mass
        'torque': numpy.zeros(3 * nb, dtype=float),
        # linear acceleration
        'lin_acc': numpy.zeros(3 * nb, dtype=float),
        # angular acceleration
        'ang_acc': numpy.zeros(3 * nb, dtype=float),
        # velocity, acceleration of CM.
        'vc': numpy.zeros(3 * nb, dtype=float),
        'vc0': numpy.zeros(3 * nb, dtype=float),
        # angular velocity in global frame
        'omega': numpy.zeros(3 * nb, dtype=float),
        'omega0': numpy.zeros(3 * nb, dtype=float),
        'nb': nb
    }

    if constants:
        consts.update(constants)

    pa = get_particle_array(constants=consts, additional_props=extra_props,
                            **props)
    pa.add_property('body_id', type='int', data=body_id)
    pa.add_property('dem_id', type='int', data=dem_id)

    setup_quaternion_rigid_body(pa)

    # create the array to save the tangential interaction particles
    # index and other variables
    limit = 30
    setup_rigid_body_cundall_particle_array(pa, limit)

    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au', 'av',
        'aw', 'tag', 'gid', 'fx', 'fy', 'fz', 'body_id'
    ])

    return pa


def setup_rigid_body_cundall_particle_array(pa, limit):
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


class RigidBodyCollision3DCundallParticleParticleEuler(Equation):
    def __init__(self, dest, sources, kn=1e7, alpha_n=0.3, nu=0.3, mu=0.5):
        self.kn = kn
        self.ks = kn / (2 * (1 + nu))
        self.alpha_n = alpha_n
        self.cn_fac = alpha_n * 2 * kn**0.5
        self.cs_fac = self.cn_fac / (2. * (1. + nu))
        self.nu = nu
        self.mu = mu
        super(RigidBodyCollision3DCundallParticleParticleEuler, self).__init__(
            dest, sources)

    def loop(self, d_idx, d_m, d_fx, d_fy, d_fz, d_tng_idx, d_tng_idx_dem_id,
             d_tng_fx, d_tng_fy, d_tng_fz, d_total_tng_contacts, d_tng_fx0,
             d_tng_fy0, d_tng_fz0, d_dem_id, d_limit, d_total_mass, d_body_id,
             VIJ, XIJ, RIJ, d_rad_s, s_idx, s_m, s_rad_s, s_dem_id, dt):
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
                nz = -XIJ[2] * rinv

                # ---- Relative velocity computation (Eq 2.9) ----
                # relative velocity of particle d_idx w.r.t particle s_idx at
                # contact point. The velocity difference provided by PySPH is
                # between translational velocities. In rigid body simulation
                # individual particles will not have rotation
                vr_x = VIJ[0]
                vr_y = VIJ[1]
                vr_z = VIJ[2]

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

                # taken from "Simulation of solid–fluid mixture flow using
                # moving particle methods"
                c_n = self.cn_fac * sqrt(d_total_mass[d_body_id[d_idx]])

                # normal force
                kn_overlap = self.kn * overlap
                fn_x = -kn_overlap * nx - c_n * vn_x
                fn_y = -kn_overlap * ny - c_n * vn_y
                fn_z = -kn_overlap * nz - c_n * vn_z

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
                    ft_magn = (d_tng_fx[found_at]**2. + d_tng_fy[found_at]**2.
                               + d_tng_fz[found_at]**2.)**0.5
                    ft_dot_nij = (
                        d_tng_fx[found_at] * nx + d_tng_fy[found_at] * ny +
                        d_tng_fz[found_at] * nz)
                    # tangential force projected onto the current normal of the
                    # contact place
                    ft_px = d_tng_fx[found_at] - ft_dot_nij * nx
                    ft_py = d_tng_fy[found_at] - ft_dot_nij * ny
                    ft_pz = d_tng_fz[found_at] - ft_dot_nij * nz

                    ftp_magn = (ft_px**2. + ft_py**2. + ft_pz**2.)**0.5
                    if ftp_magn > 0:
                        one_by_ftp_magn = 1. / ftp_magn

                        tx = ft_px * one_by_ftp_magn
                        ty = ft_py * one_by_ftp_magn
                        tz = ft_pz * one_by_ftp_magn
                    else:
                        if vt_magn > 0.:
                            tx = -vt_x / vt_magn
                            ty = -vt_y / vt_magn
                            tz = -vt_z / vt_magn
                        else:
                            tx = 0.
                            ty = 0.
                            tz = 0.

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

                        # set the tangential force to static friction
                        # from Coulomb criterion
                        ft_x = fn_mu * tx
                        ft_y = fn_mu * ty
                        ft_z = fn_mu * tz

                # compute the damping constants
                c_s = self.cs_fac * sqrt(d_total_mass[d_body_id[d_idx]])

                # increment the tangential force to next time step
                d_tng_fx[found_at] -= self.ks * vt_x * dt + c_s * vt_x
                d_tng_fy[found_at] -= self.ks * vt_y * dt + c_s * vt_y
                d_tng_fz[found_at] -= self.ks * vt_z * dt + c_s * vt_z

                d_fx[d_idx] += fn_x + ft_x
                d_fy[d_idx] += fn_y + ft_y
                d_fz[d_idx] += fn_z + ft_z


class RigidBodyCollision3DCundallParticleParticleStage1(Equation):
    def __init__(self, dest, sources, kn=1e7, alpha_n=0.3, nu=0.3, mu=0.5):
        self.kn = kn
        self.ks = kn / (2 * (1 + nu))
        self.alpha_n = alpha_n
        self.cn_fac = alpha_n * 2 * kn**0.5
        self.cs_fac = self.cn_fac / (2. * (1. + nu))
        self.nu = nu
        self.mu = mu
        super(RigidBodyCollision3DCundallParticleParticleStage1,
              self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_fx, d_fy, d_fz, d_tng_idx, d_tng_idx_dem_id,
             d_tng_fx, d_tng_fy, d_tng_fz, d_total_tng_contacts, d_tng_fx0,
             d_tng_fy0, d_tng_fz0, d_dem_id, d_limit, VIJ, XIJ, RIJ, d_rad_s,
             d_total_mass, d_body_id, s_idx, s_m, s_rad_s, s_dem_id, dt):
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
                nz = -XIJ[2] * rinv

                # ---- Relative velocity computation (Eq 2.9) ----
                # relative velocity of particle d_idx w.r.t particle s_idx at
                # contact point. The velocity difference provided by PySPH is
                # between translational velocities. In rigid body simulation
                # individual particles will not have rotation
                vr_x = VIJ[0]
                vr_y = VIJ[1]
                vr_z = VIJ[2]

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

                # taken from "Simulation of solid–fluid mixture flow using
                # moving particle methods"
                c_n = self.cn_fac * sqrt(d_total_mass[d_body_id[d_idx]])

                # normal force
                kn_overlap = self.kn * overlap
                fn_x = -kn_overlap * nx - c_n * vn_x
                fn_y = -kn_overlap * ny - c_n * vn_y
                fn_z = -kn_overlap * nz - c_n * vn_z

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
                    ft_magn = (d_tng_fx[found_at]**2. + d_tng_fy[found_at]**2.
                               + d_tng_fz[found_at]**2.)**0.5
                    ft_dot_nij = (
                        d_tng_fx[found_at] * nx + d_tng_fy[found_at] * ny +
                        d_tng_fz[found_at] * nz)
                    # tangential force projected onto the current normal of the
                    # contact place
                    ft_px = d_tng_fx[found_at] - ft_dot_nij * nx
                    ft_py = d_tng_fy[found_at] - ft_dot_nij * ny
                    ft_pz = d_tng_fz[found_at] - ft_dot_nij * nz

                    ftp_magn = (ft_px**2. + ft_py**2. + ft_pz**2.)**0.5
                    if ftp_magn > 0:
                        one_by_ftp_magn = 1. / ftp_magn

                        tx = ft_px * one_by_ftp_magn
                        ty = ft_py * one_by_ftp_magn
                        tz = ft_pz * one_by_ftp_magn
                    else:
                        if vt_magn > 0.:
                            tx = -vt_x / vt_magn
                            ty = -vt_y / vt_magn
                            tz = -vt_z / vt_magn
                        else:
                            tx = 0.
                            ty = 0.
                            tz = 0.

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

                # compute the damping constants
                c_s = self.cs_fac * sqrt(d_total_mass[d_body_id[d_idx]])

                # increment the tangential force to next time step
                dtb2 = dt / 2.
                d_tng_fx[found_at] -= self.ks * vt_x * dtb2 + c_s * vt_x
                d_tng_fy[found_at] -= self.ks * vt_y * dtb2 + c_s * vt_y
                d_tng_fz[found_at] -= self.ks * vt_z * dtb2 + c_s * vt_z

                d_fx[d_idx] += fn_x + ft_x
                d_fy[d_idx] += fn_y + ft_y
                d_fz[d_idx] += fn_z + ft_z


class RigidBodyCollision3DCundallParticleParticleStage2(Equation):
    def __init__(self, dest, sources, kn=1e7, alpha_n=0.3, nu=0.3, mu=0.5):
        self.kn = kn
        self.ks = kn / (2 * (1 + nu))
        self.alpha_n = alpha_n
        self.cn_fac = alpha_n * 2 * kn**0.5
        self.cs_fac = self.cn_fac / (2. * (1. + nu))
        self.nu = nu
        self.mu = mu
        super(RigidBodyCollision3DCundallParticleParticleStage2,
              self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_fx, d_fy, d_fz, d_tng_idx, d_tng_idx_dem_id,
             d_tng_fx, d_tng_fy, d_tng_fz, d_total_tng_contacts, d_tng_fx0,
             d_tng_fy0, d_tng_fz0, d_dem_id, d_limit, VIJ, XIJ, RIJ, d_rad_s,
             d_total_mass, d_body_id, s_idx, s_m, s_rad_s, s_dem_id, dt):
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
                nz = -XIJ[2] * rinv

                # ---- Relative velocity computation (Eq 2.9) ----
                # relative velocity of particle d_idx w.r.t particle s_idx at
                # contact point. The velocity difference provided by PySPH is
                # between translational velocities. In rigid body simulation
                # individual particles will not have rotation
                vr_x = VIJ[0]
                vr_y = VIJ[1]
                vr_z = VIJ[2]

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

                # taken from "Simulation of solid–fluid mixture flow using
                # moving particle methods"
                c_n = self.cn_fac * sqrt(d_total_mass[d_body_id[d_idx]])

                # normal force
                kn_overlap = self.kn * overlap
                fn_x = -kn_overlap * nx - c_n * vn_x
                fn_y = -kn_overlap * ny - c_n * vn_y
                fn_z = -kn_overlap * nz - c_n * vn_z

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
                    # rotate the tangential force at time (t+dt/2.) to the
                    # current plane
                    # -----------------------#
                    ft_magn = (d_tng_fx[found_at]**2. + d_tng_fy[found_at]**2.
                               + d_tng_fz[found_at]**2.)**0.5
                    ft_dot_nij = (
                        d_tng_fx[found_at] * nx + d_tng_fy[found_at] * ny +
                        d_tng_fz[found_at] * nz)
                    # tangential force projected onto the current normal of the
                    # contact place
                    ft_px = d_tng_fx[found_at] - ft_dot_nij * nx
                    ft_py = d_tng_fy[found_at] - ft_dot_nij * ny
                    ft_pz = d_tng_fz[found_at] - ft_dot_nij * nz

                    ftp_magn = (ft_px**2. + ft_py**2. + ft_pz**2.)**0.5
                    if ftp_magn > 0:
                        one_by_ftp_magn = 1. / ftp_magn

                        tx = ft_px * one_by_ftp_magn
                        ty = ft_py * one_by_ftp_magn
                        tz = ft_pz * one_by_ftp_magn
                    else:
                        if vt_magn > 0.:
                            tx = -vt_x / vt_magn
                            ty = -vt_y / vt_magn
                            tz = -vt_z / vt_magn
                        else:
                            tx = 0.
                            ty = 0.
                            tz = 0.

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
                        # set the tangential force to static friction
                        # from Coulomb criterion
                        ft_x = fn_mu * tx
                        ft_y = fn_mu * ty
                        ft_z = fn_mu * tz

                    # -----------------------#
                    # rotate the tangential force at time (t) to the current
                    # plane
                    # -----------------------#
                    ft0_magn = (d_tng_fx0[found_at]**2. + d_tng_fy0[found_at]**
                                2. + d_tng_fz0[found_at]**2.)**0.5
                    ft0_dot_nij = (
                        d_tng_fx0[found_at] * nx + d_tng_fy0[found_at] * ny +
                        d_tng_fz0[found_at] * nz)
                    # tangential force projected onto the current normal of the
                    # contact place
                    ft0_px = d_tng_fx0[found_at] - ft0_dot_nij * nx
                    ft0_py = d_tng_fy0[found_at] - ft0_dot_nij * ny
                    ft0_pz = d_tng_fz0[found_at] - ft0_dot_nij * nz

                    ft0_p_magn = (ft0_px**2. + ft0_py**2. + ft0_pz**2.)**0.5
                    if ft0_p_magn > 0:
                        one_by_ft0_p_magn = 1. / ft0_p_magn

                        tx = ft0_px * one_by_ft0_p_magn
                        ty = ft0_py * one_by_ft0_p_magn
                        tz = ft0_pz * one_by_ft0_p_magn
                    else:
                        if vt_magn > 0.:
                            tx = -vt_x / vt_magn
                            ty = -vt_y / vt_magn
                            tz = -vt_z / vt_magn
                        else:
                            tx = 0.
                            ty = 0.
                            tz = 0.

                    # rescale the projection by the magnitude of the
                    # previous tangential force, which gives the tangential
                    # force on the current plane
                    d_tng_fx0[found_at] = ft0_magn * tx
                    d_tng_fy0[found_at] = ft0_magn * ty
                    d_tng_fz0[found_at] = ft0_magn * tz

                    # compute the damping constants
                    c_s = self.cs_fac * sqrt(d_total_mass[d_body_id[d_idx]])

                    # increment the tangential force to next time step
                    d_tng_fx[found_at] = d_tng_fx0[
                        found_at] - self.ks * vt_x * dt + c_s * vt_x
                    d_tng_fy[found_at] = d_tng_fy0[
                        found_at] - self.ks * vt_y * dt + c_s * vt_y
                    d_tng_fz[found_at] = d_tng_fz0[
                        found_at] - self.ks * vt_z * dt + c_s * vt_z

                d_fx[d_idx] += fn_x + ft_x
                d_fy[d_idx] += fn_y + ft_y
                d_fz[d_idx] += fn_z + ft_z


class UpdateTangentialContactsCundall3dPaticleParticle(Equation):
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


class RigidBodyCollision3DCundallParticleWallEuler(Equation):
    def __init__(self, dest, sources, kn=1e7, alpha_n=0.3, nu=0.3, mu=0.5):
        self.kn = kn
        self.ks = kn / (2 * (1 + nu))
        self.alpha_n = alpha_n
        self.cn_fac = alpha_n * 2 * kn**0.5
        self.cs_fac = self.cn_fac / (2. * (1. + nu))
        self.nu = nu
        self.mu = mu
        super(RigidBodyCollision3DCundallParticleWallEuler, self).__init__(
            dest, sources)

    def initialize_pair(self, d_idx, d_m, d_fx, d_fy, d_fz, d_tng_idx,
                        d_tng_idx_dem_id, d_tng_fx, d_tng_fy, d_tng_fz,
                        d_total_tng_contacts, d_tng_fx0, d_tng_fy0, d_tng_fz0,
                        d_dem_id, d_limit, d_total_mass, d_body_id, VIJ, XIJ,
                        RIJ, d_rad_s, s_idx, s_m, s_rad_s, s_dem_id, dt):
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
                nz = -XIJ[2] * rinv

                # ---- Relative velocity computation (Eq 2.9) ----
                # relative velocity of particle d_idx w.r.t particle s_idx at
                # contact point. The velocity difference provided by PySPH is
                # between translational velocities. In rigid body simulation
                # individual particles will not have rotation
                vr_x = VIJ[0]
                vr_y = VIJ[1]
                vr_z = VIJ[2]

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

                # taken from "Simulation of solid–fluid mixture flow using
                # moving particle methods"
                c_n = self.cn_fac * sqrt(d_total_mass[d_body_id[d_idx]])

                # normal force
                kn_overlap = self.kn * overlap
                fn_x = -kn_overlap * nx - c_n * vn_x
                fn_y = -kn_overlap * ny - c_n * vn_y
                fn_z = -kn_overlap * nz - c_n * vn_z

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
                    ft_magn = (d_tng_fx[found_at]**2. + d_tng_fy[found_at]**2.
                               + d_tng_fz[found_at]**2.)**0.5
                    ft_dot_nij = (
                        d_tng_fx[found_at] * nx + d_tng_fy[found_at] * ny +
                        d_tng_fz[found_at] * nz)
                    # tangential force projected onto the current normal of the
                    # contact place
                    ft_px = d_tng_fx[found_at] - ft_dot_nij * nx
                    ft_py = d_tng_fy[found_at] - ft_dot_nij * ny
                    ft_pz = d_tng_fz[found_at] - ft_dot_nij * nz

                    ftp_magn = (ft_px**2. + ft_py**2. + ft_pz**2.)**0.5
                    if ftp_magn > 0:
                        one_by_ftp_magn = 1. / ftp_magn

                        tx = ft_px * one_by_ftp_magn
                        ty = ft_py * one_by_ftp_magn
                        tz = ft_pz * one_by_ftp_magn
                    else:
                        if vt_magn > 0.:
                            tx = -vt_x / vt_magn
                            ty = -vt_y / vt_magn
                            tz = -vt_z / vt_magn
                        else:
                            tx = 0.
                            ty = 0.
                            tz = 0.

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

                        # set the tangential force to static friction
                        # from Coulomb criterion
                        ft_x = fn_mu * tx
                        ft_y = fn_mu * ty
                        ft_z = fn_mu * tz

                # compute the damping constants
                c_s = self.cs_fac * sqrt(d_total_mass[d_body_id[d_idx]])

                # increment the tangential force to next time step
                d_tng_fx[found_at] -= self.ks * vt_x * dt + c_s * vt_x
                d_tng_fy[found_at] -= self.ks * vt_y * dt + c_s * vt_y
                d_tng_fz[found_at] -= self.ks * vt_z * dt + c_s * vt_z

                d_fx[d_idx] += fn_x + ft_x
                d_fy[d_idx] += fn_y + ft_y
                d_fz[d_idx] += fn_z + ft_z


class RigidBodyCollision3DCundallParticleWallStage1(Equation):
    def __init__(self, dest, sources, kn=1e7, alpha_n=0.3, nu=0.3, mu=0.5):
        self.kn = kn
        self.ks = kn / (2 * (1 + nu))
        self.alpha_n = alpha_n
        self.cn_fac = alpha_n * 2 * kn**0.5
        self.cs_fac = self.cn_fac / (2. * (1. + nu))
        self.nu = nu
        self.mu = mu
        super(RigidBodyCollision3DCundallParticleWallStage1, self).__init__(
            dest, sources)

    def initialize_pair(self, d_idx, d_m, d_x, d_y, d_z, d_u, d_v, d_w,
                        d_fx, d_fy, d_fz, d_tng_idx, d_total_mass,
                        d_body_id, d_tng_idx_dem_id, d_tng_fx, d_tng_fy,
                        d_tng_fz, d_tng_fx0, d_tng_fy0, d_tng_fz0,
                        d_total_tng_contacts, d_dem_id, d_limit,
                        d_rad_s, s_x, s_y, s_z, s_nx, s_ny,
                        s_nz, s_dem_id, s_np, dt):
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
            overlap = d_rad_s[d_idx] - (
                xij[0] * s_nx[i] + xij[1] * s_ny[i] + xij[2] * s_nz[i])

            if overlap > 0:
                # basic variables: normal vector
                # normal vector passing from particle to the wall
                nx = -s_nx[i]
                ny = -s_ny[i]
                nz = -s_nz[i]

                # ---- Relative velocity computation (Eq 2.9) ----
                # relative velocity of particle d_idx w.r.t particle i at
                # contact point.
                vij[0] = d_u[d_idx]
                vij[1] = d_v[d_idx]
                vij[2] = d_w[d_idx]
                vr_x = vij[0]
                vr_y = vij[1]
                vr_z = vij[2]

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

                # taken from "Simulation of solid–fluid mixture flow using
                # moving particle methods"
                c_n = self.cn_fac * sqrt(d_total_mass[d_body_id[d_idx]])

                # normal force
                kn_overlap = self.kn * overlap
                fn_x = -kn_overlap * nx - c_n * vn_x
                fn_y = -kn_overlap * ny - c_n * vn_y
                fn_z = -kn_overlap * nz - c_n * vn_z

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
                    if i == d_tng_idx[j]:
                        if s_dem_id[i] == d_tng_idx_dem_id[j]:
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
                    d_tng_idx[found_at] = i
                    d_total_tng_contacts[d_idx] += 1
                    d_tng_idx_dem_id[found_at] = s_dem_id[i]

                # implies we are tracking the particle
                else:
                    # -----------------------#
                    # rotate the tangential force to the current plane
                    # -----------------------#
                    ft_magn = (d_tng_fx[found_at]**2. + d_tng_fy[found_at]**2.
                               + d_tng_fz[found_at]**2.)**0.5
                    ft_dot_nij = (
                        d_tng_fx[found_at] * nx + d_tng_fy[found_at] * ny +
                        d_tng_fz[found_at] * nz)
                    # tangential force projected onto the current normal of the
                    # contact place
                    ft_px = d_tng_fx[found_at] - ft_dot_nij * nx
                    ft_py = d_tng_fy[found_at] - ft_dot_nij * ny
                    ft_pz = d_tng_fz[found_at] - ft_dot_nij * nz

                    ftp_magn = (ft_px**2. + ft_py**2. + ft_pz**2.)**0.5
                    if ftp_magn > 0:
                        one_by_ftp_magn = 1. / ftp_magn

                        tx = ft_px * one_by_ftp_magn
                        ty = ft_py * one_by_ftp_magn
                        tz = ft_pz * one_by_ftp_magn
                    else:
                        if vt_magn > 0.:
                            tx = -vt_x / vt_magn
                            ty = -vt_y / vt_magn
                            tz = -vt_z / vt_magn
                        else:
                            tx = 0.
                            ty = 0.
                            tz = 0.

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

                # compute the damping constants
                c_s = self.cs_fac * sqrt(d_total_mass[d_body_id[d_idx]])

                # increment the tangential force to next time step
                dtb2 = dt / 2.
                d_tng_fx[found_at] -= self.ks * vt_x * dtb2 + c_s * vt_x
                d_tng_fy[found_at] -= self.ks * vt_y * dtb2 + c_s * vt_y
                d_tng_fz[found_at] -= self.ks * vt_z * dtb2 + c_s * vt_z

                d_fx[d_idx] += fn_x + ft_x
                d_fy[d_idx] += fn_y + ft_y
                d_fz[d_idx] += fn_z + ft_z


class RigidBodyCollision3DCundallParticleWallStage2(Equation):
    def __init__(self, dest, sources, kn=1e7, alpha_n=0.3, nu=0.3, mu=0.5):
        self.kn = kn
        self.ks = kn / (2 * (1 + nu))
        self.alpha_n = alpha_n
        self.cn_fac = alpha_n * 2 * kn**0.5
        self.cs_fac = self.cn_fac / (2. * (1. + nu))
        self.nu = nu
        self.mu = mu
        super(RigidBodyCollision3DCundallParticleWallStage2, self).__init__(
            dest, sources)

    def initialize_pair(self, d_idx, d_m, d_x, d_y, d_z, d_u, d_v, d_w,
                        d_fx, d_fy, d_fz, d_tng_idx, d_total_mass,
                        d_body_id, d_tng_idx_dem_id, d_tng_fx, d_tng_fy,
                        d_tng_fz, d_tng_fx0, d_tng_fy0, d_tng_fz0,
                        d_total_tng_contacts, d_dem_id, d_limit,
                        d_rad_s, s_x, s_y, s_z, s_nx, s_ny,
                        s_nz, s_dem_id, s_np, dt):
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
            overlap = d_rad_s[d_idx] - (
                xij[0] * s_nx[i] + xij[1] * s_ny[i] + xij[2] * s_nz[i])

            if overlap > 0:
                # basic variables: normal vector
                # normal vector passing from particle to the wall
                nx = -s_nx[i]
                ny = -s_ny[i]
                nz = -s_nz[i]

                # ---- Relative velocity computation (Eq 2.9) ----
                # relative velocity of particle d_idx w.r.t particle i at
                # contact point.
                vij[0] = d_u[d_idx]
                vij[1] = d_v[d_idx]
                vij[2] = d_w[d_idx]
                vr_x = vij[0]
                vr_y = vij[1]
                vr_z = vij[2]

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

                # taken from "Simulation of solid–fluid mixture flow using
                # moving particle methods"
                c_n = self.cn_fac * sqrt(d_total_mass[d_body_id[d_idx]])

                # normal force
                kn_overlap = self.kn * overlap
                fn_x = -kn_overlap * nx - c_n * vn_x
                fn_y = -kn_overlap * ny - c_n * vn_y
                fn_z = -kn_overlap * nz - c_n * vn_z

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
                    if i == d_tng_idx[j]:
                        if s_dem_id[i] == d_tng_idx_dem_id[j]:
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
                    # rotate the tangential force to the current plane
                    # -----------------------#
                    ft_magn = (d_tng_fx[found_at]**2. + d_tng_fy[found_at]**2.
                               + d_tng_fz[found_at]**2.)**0.5
                    ft_dot_nij = (
                        d_tng_fx[found_at] * nx + d_tng_fy[found_at] * ny +
                        d_tng_fz[found_at] * nz)
                    # tangential force projected onto the current normal of the
                    # contact place
                    ft_px = d_tng_fx[found_at] - ft_dot_nij * nx
                    ft_py = d_tng_fy[found_at] - ft_dot_nij * ny
                    ft_pz = d_tng_fz[found_at] - ft_dot_nij * nz

                    ftp_magn = (ft_px**2. + ft_py**2. + ft_pz**2.)**0.5
                    if ftp_magn > 0:
                        one_by_ftp_magn = 1. / ftp_magn

                        tx = ft_px * one_by_ftp_magn
                        ty = ft_py * one_by_ftp_magn
                        tz = ft_pz * one_by_ftp_magn
                    else:
                        if vt_magn > 0.:
                            tx = -vt_x / vt_magn
                            ty = -vt_y / vt_magn
                            tz = -vt_z / vt_magn
                        else:
                            tx = 0.
                            ty = 0.
                            tz = 0.

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
                        # set the tangential force to static friction
                        # from Coulomb criterion
                        ft_x = fn_mu * tx
                        ft_y = fn_mu * ty
                        ft_z = fn_mu * tz

                    # -----------------------#
                    # rotate the tangential force at time (t) to the current
                    # plane
                    # -----------------------#
                    ft0_magn = (d_tng_fx0[found_at]**2. + d_tng_fy0[found_at]**
                                2. + d_tng_fz0[found_at]**2.)**0.5
                    ft0_dot_nij = (
                        d_tng_fx0[found_at] * nx + d_tng_fy0[found_at] * ny +
                        d_tng_fz0[found_at] * nz)
                    # tangential force projected onto the current normal of the
                    # contact place
                    ft0_px = d_tng_fx0[found_at] - ft0_dot_nij * nx
                    ft0_py = d_tng_fy0[found_at] - ft0_dot_nij * ny
                    ft0_pz = d_tng_fz0[found_at] - ft0_dot_nij * nz

                    ft0_p_magn = (ft0_px**2. + ft0_py**2. + ft0_pz**2.)**0.5
                    if ft0_p_magn > 0:
                        one_by_ft0_p_magn = 1. / ft0_p_magn

                        tx = ft0_px * one_by_ft0_p_magn
                        ty = ft0_py * one_by_ft0_p_magn
                        tz = ft0_pz * one_by_ft0_p_magn
                    else:
                        if vt_magn > 0.:
                            tx = -vt_x / vt_magn
                            ty = -vt_y / vt_magn
                            tz = -vt_z / vt_magn
                        else:
                            tx = 0.
                            ty = 0.
                            tz = 0.

                    # rescale the projection by the magnitude of the
                    # previous tangential force, which gives the tangential
                    # force on the current plane
                    d_tng_fx0[found_at] = ft0_magn * tx
                    d_tng_fy0[found_at] = ft0_magn * ty
                    d_tng_fz0[found_at] = ft0_magn * tz

                    # compute the damping constants
                    c_s = self.cs_fac * sqrt(d_total_mass[d_body_id[d_idx]])

                    # increment the tangential force to next time step
                    d_tng_fx[found_at] = (
                        d_tng_fx0[found_at] - self.ks * vt_x * dt + c_s * vt_x)
                    d_tng_fy[found_at] = (
                        d_tng_fy0[found_at] - self.ks * vt_y * dt + c_s * vt_y)
                    d_tng_fz[found_at] = (
                        d_tng_fz0[found_at] - self.ks * vt_z * dt + c_s * vt_z)

                d_fx[d_idx] += fn_x + ft_x
                d_fy[d_idx] += fn_y + ft_y
                d_fz[d_idx] += fn_z + ft_z


class UpdateTangentialContactsCundall3dPaticleWall(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_z, d_rad_s,
                        d_total_tng_contacts, d_tng_idx, d_limit, d_tng_fx,
                        d_tng_fy, d_tng_fz, d_tng_fx0, d_tng_fy0, d_tng_fz0,
                        d_tng_idx_dem_id, s_x, s_y, s_z, s_nx, s_ny, s_nz,
                        s_dem_id):
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
                    xij[2] = d_z[d_idx] - s_z[sidx]
                    overlap = d_rad_s[d_idx] - (
                        xij[0] * s_nx[sidx] + xij[1] * s_ny[sidx] +
                        xij[2] * s_nz[sidx])

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


class RK2StepRigidBodyQuaternionsDEMCundall3d(
        RK2StepRigidBodyQuaternionsDEMCundall2d):
    def initialize(self, d_idx, d_tng_fx, d_tng_fy, d_tng_fz, d_tng_fx0,
                   d_tng_fy0, d_tng_fz0, d_total_tng_contacts, d_limit):
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
