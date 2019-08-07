from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from compyle.api import declare
from math import sqrt, asin, sin, cos, log
import numpy as np
from pysph.sph.scheme import Scheme
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EulerIntegrator
from pysph.base.kernels import CubicSpline


def get_particle_array_bonded_dem_onate(constants=None, **props):
    """Return a particle array for a dem particles
    """
    dim = props.pop('dim', None)
    clrnc = props.pop('clrnc', None)

    dem_props = [
        'theta_x', 'theta_y', 'theta_z', 'wx', 'wy', 'wz', 'fx', 'fy', 'fz',
        'torx', 'tory', 'torz', 'rad_s', 'm_inverse', 'I_inverse', 'u0', 'v0',
        'w0', 'wx0', 'wy0', 'wz0', 'x0', 'y0', 'z0'
    ]

    dem_id = props.pop('dem_id', None)

    pa = get_particle_array(additional_props=dem_props, **props)

    pa.add_property('dem_id', type='int', data=dem_id)
    # create the array to save the tangential interaction particles
    # index and other variables
    if dim == 3:
        bc_limit = 30
    elif dim == 2 or dim is None:
        bc_limit = 6

    pa.add_constant('bc_limit', bc_limit)
    pa.add_property('bc_idx', stride=bc_limit, type='int')
    pa.bc_idx[:] = -1

    pa.add_property('bc_total_contacts', type='int')
    pa.add_property('bc_rest_len', stride=bc_limit)
    # pa.add_property('bc_ft_x', stride=bc_limit)
    # pa.add_property('bc_ft_y', stride=bc_limit)
    # pa.add_property('bc_ft_z', stride=bc_limit)
    # pa.add_property('bc_ft0_x', stride=bc_limit)
    # pa.add_property('bc_ft0_y', stride=bc_limit)
    # pa.add_property('bc_ft0_z', stride=bc_limit)

    setup_bc_contacts(dim, pa, clrnc)

    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'wx', 'wy', 'wz', 'm', 'pid', 'tag',
        'gid', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'rad_s', 'dem_id'
    ])

    return pa


def make_accel_eval(equations, pa_arrays, dim):
    from pysph.tools.sph_evaluator import SPHEvaluator
    kernel = CubicSpline(dim=dim)
    seval = SPHEvaluator(arrays=pa_arrays, equations=equations, dim=dim,
                         kernel=kernel)
    return seval


def setup_bc_contacts(dim, pa, beta):
    eqs1 = [
        Group(equations=[
            SetupContactsBC(dest=pa.name, sources=[pa.name], beta=beta),
        ])
    ]
    arrays = [pa]
    a_eval = make_accel_eval(eqs1, arrays, dim)
    a_eval.evaluate()


class SetupContactsBC(Equation):
    def __init__(self, dest, sources, beta):
        self.beta = beta
        super(SetupContactsBC, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_bc_total_contacts, d_bc_idx, d_bc_limit,
             d_rad_s, d_bc_rest_len, s_rad_s, RIJ):
        p = declare('int')
        p = d_bc_limit[0] * d_idx + d_bc_total_contacts[d_idx]
        fac = RIJ / (d_rad_s[d_idx] + s_rad_s[s_idx])
        if RIJ > 1e-12:
            if 1. - self.beta < fac < 1 + self.beta:
                d_bc_idx[p] = s_idx
                d_bc_total_contacts[d_idx] += 1
                d_bc_rest_len[p] = RIJ


class OnateBDEMForceStage1(Equation):
    def __init__(self, dest, sources, kn, cn):
        self.kn = kn
        self.kt = kn / 2.
        self.cn = cn
        super(OnateBDEMForceStage1, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_z, d_bc_limit,
                   d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_fz, d_torx, d_tory,
                   d_torz, d_u, d_v, d_w, d_wx, d_wy, d_wz, d_rad_s, dt):
        p, q1, tot_ctcs, i, sidx = declare('int', 5)
        xij = declare('matrix(3)')
        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        for i in range(p, q1):
            sidx = d_bc_idx[i]

            xij[0] = d_x[d_idx] - d_x[sidx]
            xij[1] = d_y[d_idx] - d_y[sidx]
            xij[2] = d_z[d_idx] - d_z[sidx]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

            # normal vector from i to j
            rinv = 1. / rij
            nji_x = -xij[0] * rinv
            nji_y = -xij[1] * rinv
            nji_z = -xij[2] * rinv

            # ======= WRITE TEST ============
            # find a directions of two other shear directions
            # global z cross normal nji
            s1_x = -nji_y
            s1_y = nji_x
            s1_z = 0.
            if s1_x == 0 and s1_y == 0. and s1_z == 0.:
                s1_x = 0
                s1_y = 1
                s1_z = 0

            # now find the second tangential vector to the plane
            # normal nji cross s1
            s2_x = nji_y * s1_z - nji_z * s1_y
            s2_y = nji_z * s1_x - nji_x * s1_z
            s2_z = nji_x * s1_y - nji_y * s1_x

            xji_dot_nji = -(xij[0] * nji_x + xij[1] * nji_y + xij[2] * nji_z)
            overlap = xji_dot_nji - d_bc_rest_len[i]

            # relative velocity of particle (only translational)
            vr_x = d_u[sidx] - d_u[d_idx]
            vr_y = d_v[sidx] - d_v[d_idx]
            vr_z = d_w[sidx] - d_w[d_idx]
            # velocity in normal direction
            vn = (vr_x * nji_x + vr_y * nji_y + vr_z * nji_z)

            # normal force magnitude
            cn = self.cn
            fn = self.kn * overlap + cn * vn

            # ---------- force computation starts ------------
            d_fx[d_idx] += fn * nji_x
            d_fy[d_idx] += fn * nji_y
            d_fz[d_idx] += fn * nji_z

            # ---------- tangential force ------------
            # find displacements of current time step
            # u is displacement
            dtb2 = dt / 2.
            a_i = d_rad_s[d_idx]
            a_j = d_rad_s[sidx]
            wijx = (a_i * d_wx[d_idx] + a_j * d_wx[sidx]) * dtb2
            wijy = (a_i * d_wy[d_idx] + a_j * d_wy[sidx]) * dtb2
            wijz = (a_i * d_wz[d_idx] + a_j * d_wz[sidx]) * dtb2

            # wij \cross nji
            wcn_x_dt = wijy * nji_z - wijz * nji_y
            wcn_y_dt = wijz * nji_x - wijx * nji_z
            wcn_z_dt = wijx * nji_y - wijy * nji_x

            # net displacement vector
            u_x = (d_u[d_idx] - d_u[sidx]) * dt + wcn_x_dt
            u_y = (d_v[d_idx] - d_v[sidx]) * dt + wcn_y_dt
            u_z = (d_w[d_idx] - d_w[sidx]) * dt + wcn_z_dt

            # net displacement vector in normal direction magn
            u_n = (u_x * nji_x + u_y * nji_y + u_z * nji_z)
            # shear displacement vector in 3d space
            u_sx = u_x - u_n * nji_x
            u_sy = u_y - u_n * nji_y
            u_sz = u_z - u_n * nji_z

            # shear displacement in the contact plane in s1 and s2
            # directions
            u_s1 = (u_sx * s1_x + u_sy * s1_y + u_sz * s1_z)
            u_s2 = (u_sx * s2_x + u_sy * s2_y + u_sz * s2_z)

            # total tangential force
            f_s1 = self.kt * u_s1
            f_s2 = self.kt * u_s2

            ft_x = f_s1 * s1_x + f_s2 * s2_x
            ft_y = f_s1 * s1_y + f_s2 * s2_y
            ft_z = f_s1 * s1_z + f_s2 * s2_z

            # find the torque
            # the vector till contact point will be
            # nji * (r_i + (rij - r_i - r_j)/2.)
            cnst = d_rad_s[d_idx] + (rij - d_rad_s[d_idx] - d_rad_s[sidx]) / 2.
            rc_i_x = cnst * nji_x
            rc_i_y = cnst * nji_y
            rc_i_z = cnst * nji_z

            tor_x = rc_i_x * ft_z - rc_i_z * ft_y
            tor_y = rc_i_z * ft_x - rc_i_x * ft_z
            tor_z = rc_i_x * ft_y - rc_i_y * ft_x

            # add toque and tangential force to global vector
            d_fx[d_idx] += ft_x
            d_fy[d_idx] += ft_y
            d_fz[d_idx] += ft_z

            d_torx[d_idx] += tor_x
            d_tory[d_idx] += tor_y
            d_torz[d_idx] += tor_z


class OnateBDEMForceStage2(Equation):
    def __init__(self, dest, sources, kn, cn):
        self.kn = kn
        self.kt = kn / 2.
        self.cn = cn
        super(OnateBDEMForceStage2, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_z, d_bc_limit,
                   d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_fz, d_torx, d_tory,
                   d_torz, d_u, d_v, d_w, d_wx, d_wy, d_wz, d_rad_s, dt):
        p, q1, tot_ctcs, i, sidx = declare('int', 5)
        xij = declare('matrix(3)')
        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        for i in range(p, q1):
            sidx = d_bc_idx[i]
            overlap = -1.
            rij = 0.0

            xij[0] = d_x[d_idx] - d_x[sidx]
            xij[1] = d_y[d_idx] - d_y[sidx]
            xij[2] = d_z[d_idx] - d_z[sidx]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

            # normal vector from i to j
            rinv = 1. / rij
            nji_x = -xij[0] * rinv
            nji_y = -xij[1] * rinv
            nji_z = -xij[2] * rinv

            # ======= WRITE TEST ============
            # find a directions of two other shear directions
            # global z cross normal nji
            s1_x = -nji_y
            s1_y = nji_x
            s1_z = 0.
            if s1_x == 0 and s1_y == 0. and s1_z == 0.:
                s1_x = 0
                s1_y = 1
                s1_z = 0

            # now find the second tangential vector to the plane
            # normal nji cross s1
            s2_x = nji_y * s1_z - nji_z * s1_y
            s2_y = nji_z * s1_x - nji_x * s1_z
            s2_z = nji_x * s1_y - nji_y * s1_x

            # check the particles are not on top of each other.
            xji_dot_nji = -(xij[0] * nji_x + xij[1] * nji_y + xij[2] * nji_z)
            overlap = xji_dot_nji - d_bc_rest_len[i]

            # relative velocity of particle (only translational)
            vr_x = d_u[sidx] - d_u[d_idx]
            vr_y = d_v[sidx] - d_v[d_idx]
            vr_z = d_z[sidx] - d_z[d_idx]
            # velocity in normal direction
            vn = (vr_x * nji_x + vr_y * nji_y + vr_z * nji_z)

            # normal force magnitude
            cn = self.cn
            fn = self.kn * overlap + cn * vn
            # print("overlap")
            # print(overlap)
            # print(fn)

            # ---------- force computation starts ------------
            d_fx[d_idx] += fn * nji_x
            d_fy[d_idx] += fn * nji_y
            d_fz[d_idx] += fn * nji_z

            # ---------- tangential force ------------
            # find displacements of current time step
            # u is displacement
            a_i = d_rad_s[d_idx]
            a_j = d_rad_s[sidx]
            wijx = (a_i * d_wx[d_idx] + a_j * d_wx[sidx]) * dt
            wijy = (a_i * d_wy[d_idx] + a_j * d_wy[sidx]) * dt
            wijz = (a_i * d_wz[d_idx] + a_j * d_wz[sidx]) * dt

            # wij \cross nji
            wcn_x_dt = wijy * nji_z - wijz * nji_y
            wcn_y_dt = wijz * nji_x - wijx * nji_z
            wcn_z_dt = wijx * nji_y - wijy * nji_x

            # net displacement vector
            u_x = (d_u[d_idx] - d_u[sidx]) * dt + wcn_x_dt
            u_y = (d_v[d_idx] - d_v[sidx]) * dt + wcn_y_dt
            u_z = (d_w[d_idx] - d_w[sidx]) * dt + wcn_z_dt

            # net displacement vector in normal direction magn
            u_n = (u_x * nji_x + u_y * nji_y + u_z * nji_z)
            # shear displacement vector in 3d space
            u_sx = u_x - u_n * nji_x
            u_sy = u_y - u_n * nji_y
            u_sz = u_z - u_n * nji_z

            # shear displacement in the contact plane in s1 and s2
            # directions
            u_s1 = (u_sx * s1_x + u_sy * s1_y + u_sz * s1_z)
            u_s2 = (u_sx * s2_x + u_sy * s2_y + u_sz * s2_z)

            # total tangential force
            f_s1 = self.kt * u_s1
            f_s2 = self.kt * u_s2

            ft_x = f_s1 * s1_x + f_s2 * s2_x
            ft_y = f_s1 * s1_y + f_s2 * s2_y
            ft_z = f_s1 * s1_z + f_s2 * s2_z

            # find the torque
            # the vector till contact point will be
            # nji * (r_i + (rij - r_i - r_j)/2.)
            cnst = d_rad_s[d_idx] + (rij - d_rad_s[d_idx] - d_rad_s[sidx]) / 2.
            rc_i_x = cnst * nji_x
            rc_i_y = cnst * nji_y
            rc_i_z = cnst * nji_z

            tor_x = rc_i_x * ft_z - rc_i_z * ft_y
            tor_y = rc_i_z * ft_x - rc_i_x * ft_z
            tor_z = rc_i_x * ft_y - rc_i_y * ft_x

            # add toque and tangential force to global vector
            d_fx[d_idx] += ft_x
            d_fy[d_idx] += ft_y
            d_fz[d_idx] += ft_z

            d_torx[d_idx] += tor_x
            d_tory[d_idx] += tor_y
            d_torz[d_idx] += tor_z


class RK2StepOnate(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v, d_w,
                   d_u0, d_v0, d_w0, d_wx, d_wy, d_wz, d_wx0, d_wy0, d_wz0,
                   d_bc_total_contacts, d_bc_limit):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_wx0[d_idx] = d_wx[d_idx]
        d_wy0[d_idx] = d_wy[d_idx]
        d_wz0[d_idx] = d_wz[d_idx]

        # # -----------------------------------------------
        # # save the initial tangential contact information
        # # -----------------------------------------------
        # i = declare('int')
        # p = declare('int')
        # q = declare('int')
        # tot_ctcs = declare('int')
        # tot_ctcs = d_bc_total_contacts[d_idx]
        # p = d_idx * d_bc_limit[0]
        # q = p + tot_ctcs

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


class OnateBDEMForceEuler(Equation):
    def __init__(self, dest, sources, kn, cn):
        self.kn = kn
        self.kt = kn / 2.
        self.cn = cn
        super(OnateBDEMForceEuler, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_z, d_bc_limit,
                   d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_fz, d_torx, d_tory,
                   d_torz, d_u, d_v, d_w, d_wx, d_wy, d_wz, d_rad_s, dt):
        p, q1, tot_ctcs, i, sidx = declare('int', 5)
        xij = declare('matrix(3)')
        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        for i in range(p, q1):
            sidx = d_bc_idx[i]
            overlap = -1.
            rij = 0.0

            xij[0] = d_x[d_idx] - d_x[sidx]
            xij[1] = d_y[d_idx] - d_y[sidx]
            xij[2] = d_z[d_idx] - d_z[sidx]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

            # normal vector from i to j
            rinv = 1. / rij
            nji_x = -xij[0] * rinv
            nji_y = -xij[1] * rinv
            nji_z = -xij[2] * rinv

            # ======= WRITE TEST ============
            # find a directions of two other shear directions
            # global z cross normal nji
            s1_x = -nji_y
            s1_y = nji_x
            s1_z = 0.
            if s1_x == 0 and s1_y == 0. and s1_z == 0.:
                s1_x = 0
                s1_y = 1
                s1_z = 0

            # now find the second tangential vector to the plane
            # normal nji cross s1
            s2_x = nji_y * s1_z - nji_z * s1_y
            s2_y = nji_z * s1_x - nji_x * s1_z
            s2_z = nji_x * s1_y - nji_y * s1_x

            # check the particles are not on top of each other.
            xji_dot_nji = -(xij[0] * nji_x + xij[1] * nji_y + xij[2] * nji_z)
            overlap = xji_dot_nji - d_bc_rest_len[i]

            # relative velocity of particle (only translational)
            vr_x = d_u[sidx] - d_u[d_idx]
            vr_y = d_v[sidx] - d_v[d_idx]
            vr_z = d_z[sidx] - d_z[d_idx]
            # velocity in normal direction
            vn = (vr_x * nji_x + vr_y * nji_y + vr_z * nji_z)

            # normal force magnitude
            cn = self.cn
            fn = self.kn * overlap + cn * vn

            # ---------- force computation starts ------------
            d_fx[d_idx] += fn * nji_x
            d_fy[d_idx] += fn * nji_y
            d_fz[d_idx] += fn * nji_z

            # ---------- tangential force ------------
            # find displacements of current time step
            # u is displacement
            a_i = d_rad_s[d_idx]
            a_j = d_rad_s[sidx]
            wijx = (a_i * d_wx[d_idx] + a_j * d_wx[sidx]) * dt
            wijy = (a_i * d_wy[d_idx] + a_j * d_wy[sidx]) * dt
            wijz = (a_i * d_wz[d_idx] + a_j * d_wz[sidx]) * dt

            # wij \cross nji
            wcn_x_dt = wijy * nji_z - wijz * nji_y
            wcn_y_dt = wijz * nji_x - wijx * nji_z
            wcn_z_dt = wijx * nji_y - wijy * nji_x

            # net displacement vector
            u_x = (d_u[d_idx] - d_u[sidx]) * dt + wcn_x_dt
            u_y = (d_v[d_idx] - d_v[sidx]) * dt + wcn_y_dt
            u_z = (d_w[d_idx] - d_w[sidx]) * dt + wcn_z_dt

            # net displacement vector in normal direction magn
            u_n = (u_x * nji_x + u_y * nji_y + u_z * nji_z)
            # shear displacement vector in 3d space
            u_sx = u_x - u_n * nji_x
            u_sy = u_y - u_n * nji_y
            u_sz = u_z - u_n * nji_z

            # shear displacement in the contact plane in s1 and s2
            # directions
            u_s1 = (u_sx * s1_x + u_sy * s1_y + u_sz * s1_z)
            u_s2 = (u_sx * s2_x + u_sy * s2_y + u_sz * s2_z)

            # total tangential force
            f_s1 = self.kt * u_s1
            f_s2 = self.kt * u_s2

            ft_x = f_s1 * s1_x + f_s2 * s2_x
            ft_y = f_s1 * s1_y + f_s2 * s2_y
            ft_z = f_s1 * s1_z + f_s2 * s2_z

            # find the torque
            # the vector till contact point will be
            # nji * (r_i + (rij - r_i - r_j)/2.)
            cnst = d_rad_s[d_idx] + (rij - d_rad_s[d_idx] - d_rad_s[sidx]) / 2.
            rc_i_x = cnst * nji_x
            rc_i_y = cnst * nji_y
            rc_i_z = cnst * nji_z

            tor_x = rc_i_x * ft_z - rc_i_z * ft_y
            tor_y = rc_i_z * ft_x - rc_i_x * ft_z
            tor_z = rc_i_x * ft_y - rc_i_y * ft_x

            # add toque and tangential force to global vector
            d_fx[d_idx] += ft_x
            d_fy[d_idx] += ft_y
            d_fz[d_idx] += ft_z

            d_torx[d_idx] += tor_x
            d_tory[d_idx] += tor_y
            d_torz[d_idx] += tor_z


class EulerStepOnate(IntegratorStep):
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_torx, d_tory, d_torz, d_wx, d_wy, d_wz, d_m_inverse,
               d_I_inverse, dt):
        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u[d_idx] + dt * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v[d_idx] + dt * d_fy[d_idx] * d_m_inverse[d_idx]
        d_w[d_idx] = d_w[d_idx] + dt * d_fz[d_idx] * d_m_inverse[d_idx]

        d_wx[d_idx] = d_wx[d_idx] + dt * d_torx[d_idx] * d_I_inverse[d_idx]
        d_wy[d_idx] = d_wy[d_idx] + dt * d_tory[d_idx] * d_I_inverse[d_idx]
        d_wz[d_idx] = d_wz[d_idx] + dt * d_torz[d_idx] * d_I_inverse[d_idx]


class GlobalDampingForce(Equation):
    def __init__(self, dest, sources, alpha_t=0.1, alpha_r=0.1):
        self.alpha_t = alpha_t
        self.alpha_r = alpha_r
        super(GlobalDampingForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz, d_u,
                   d_v, d_w, d_wx, d_wy, d_wz):
        f_magn = (d_fx[d_idx]**2. + d_fy[d_idx]**2. + d_fz[d_idx]**2.)**0.5
        vel_magn = (d_u[d_idx]**2. + d_v[d_idx]**2. + d_w[d_idx]**2.)**0.5
        if vel_magn > 1e-12:
            d_fx[d_idx] -= self.alpha_t * f_magn * d_u[d_idx] / vel_magn
            d_fy[d_idx] -= self.alpha_t * f_magn * d_v[d_idx] / vel_magn
            d_fz[d_idx] -= self.alpha_t * f_magn * d_w[d_idx] / vel_magn

        tor_magn = (
            d_torx[d_idx]**2. + d_tory[d_idx]**2. + d_torz[d_idx]**2.)**0.5
        omega_magn = (d_wx[d_idx]**2. + d_wy[d_idx]**2. + d_wz[d_idx]**2.)**0.5
        if omega_magn > 1e-12:
            d_torx[d_idx] -= (
                self.alpha_r * tor_magn * d_wx[d_idx] / omega_magn)
            d_tory[d_idx] -= (
                self.alpha_r * tor_magn * d_wy[d_idx] / omega_magn)
            d_torz[d_idx] -= (
                self.alpha_r * tor_magn * d_wz[d_idx] / omega_magn)
