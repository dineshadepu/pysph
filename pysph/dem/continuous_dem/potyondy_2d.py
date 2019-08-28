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


def get_particle_array_bonded_dem_potyondy(constants=None, **props):
    """Return a particle array for a dem particles
    """
    dim = props.pop('dim', None)

    dem_props = [
        'wx', 'wy', 'wz', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'rad_s',
        'm_inverse', 'I_inverse', 'u0', 'v0', 'w0', 'wx0', 'wy0', 'wz0', 'x0',
        'y0', 'z0'
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
    pa.add_property('bc_ft_x', stride=bc_limit)
    pa.add_property('bc_ft_y', stride=bc_limit)
    pa.add_property('bc_ft_z', stride=bc_limit)
    pa.add_property('bc_ft0_x', stride=bc_limit)
    pa.add_property('bc_ft0_y', stride=bc_limit)
    pa.add_property('bc_ft0_z', stride=bc_limit)

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


class PotyondyIPForceStage1(Equation):
    def __init__(self, dest, sources, kn):
        self.kn = kn
        self.kt = kn / 2.
        super(PotyondyIPForceStage1, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_bc_ft_x,
                   d_bc_ft_y, d_bc_ft_z, d_bc_ft0_x, d_bc_ft0_y, d_bc_ft0_z,
                   d_bc_limit, d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_u, d_v,
                   dt):
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
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

            # normal vector from i to j
            nji_x = xij[0] / rij
            nji_y = xij[1] / rij

            rinv = 1. / rij
            # print("didx")
            # print(d_idx)
            # print("sidx")
            # print(sidx)
            # print("hi")
            # print(d_bc_rest_len[i])
            # check the particles are not on top of each other.
            if rij > 0:
                overlap = rij - d_bc_rest_len[i]

            # print("rij")
            # print(rij)
            # print("overlap")
            # print(overlap)

            # relative velocity
            vr_x = d_u[d_idx] - d_u[sidx]
            vr_y = d_v[d_idx] - d_v[sidx]

            # normal velocity magnitude
            vr_dot_nij = vr_x * nji_x + vr_y * nji_y
            vn_x = vr_dot_nij * nji_x
            vn_y = vr_dot_nij * nji_y

            # ---------- force computation starts ------------
            d_fx[d_idx] -= self.kn * nji_x * overlap - vn_x * 10.
            d_fy[d_idx] -= self.kn * nji_y * overlap - vn_y * 10.

            # tangential force
            ft_x = d_bc_ft_x[i]
            ft_y = d_bc_ft_y[i]

            # TODO
            # check the coulomb criterion

            # Add the tangential force
            d_fx[d_idx] += ft_x
            d_fy[d_idx] += ft_y

            # --------- Tangential force -----------
            # -----increment the tangential force---
            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y

            # magnitude of the tangential velocity
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

            # get the incremental force
            dtb2 = dt / 2.
            d_bc_ft_x[i] += - self.kt * vt_x * dtb2 - vt_x * 10.
            d_bc_ft_y[i] += - self.kt * vt_y * dtb2 - vt_y * 10.


class PotyondyIPForceStage2(Equation):
    def __init__(self, dest, sources, kn):
        self.kn = kn
        self.kt = kn / 2.
        super(PotyondyIPForceStage2, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_bc_ft_x,
                   d_bc_ft_y, d_bc_ft_z, d_bc_ft0_x, d_bc_ft0_y, d_bc_ft0_z,
                   d_bc_limit, d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_u, d_v, dt):
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
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

            # normal vector from i to j
            nji_x = xij[0] / rij
            nji_y = xij[1] / rij

            rinv = 1. / rij
            # print("didx")
            # print(d_idx)
            # print("sidx")
            # print(sidx)
            # print("hi")
            # print(d_bc_rest_len[i])
            # check the particles are not on top of each other.
            if rij > 0:
                overlap = rij - d_bc_rest_len[i]

            # print("rij")
            # print(rij)
            # print("overlap")
            # print(overlap)

            # relative velocity
            vr_x = d_u[d_idx] - d_u[sidx]
            vr_y = d_v[d_idx] - d_v[sidx]

            # normal velocity magnitude
            vr_dot_nij = vr_x * nji_x + vr_y * nji_y
            vn_x = vr_dot_nij * nji_x
            vn_y = vr_dot_nij * nji_y

            # ---------- force computation starts ------------
            d_fx[d_idx] -= self.kn * nji_x * overlap - vn_x * 10.
            d_fy[d_idx] -= self.kn * nji_y * overlap - vn_y * 10.

            # tangential force
            ft_x = d_bc_ft_x[i]
            ft_y = d_bc_ft_y[i]

            # TODO
            # check the coulomb criterion

            # Add the tangential force
            d_fx[d_idx] += ft_x
            d_fy[d_idx] += ft_y

            # --------- Tangential force -----------
            # -----increment the tangential force---
            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y

            # magnitude of the tangential velocity
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

            # get the incremental force
            d_bc_ft_x[i] = d_bc_ft0_x[i] - self.kt * vt_x * dt - vt_x * 10.
            d_bc_ft_y[i] = d_bc_ft0_y[i] - self.kt * vt_y * dt - vt_y * 10.


class RK2StepPotyondy(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v, d_w,
                   d_u0, d_v0, d_w0, d_wx, d_wy, d_wz, d_wx0, d_wy0, d_wz0,
                   d_bc_total_contacts, d_bc_limit, d_bc_ft_x, d_bc_ft_y,
                   d_bc_ft_z, d_bc_ft0_x, d_bc_ft0_y, d_bc_ft0_z):

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
        tot_ctcs = d_bc_total_contacts[d_idx]
        p = d_idx * d_bc_limit[0]
        q = p + tot_ctcs

        for i in range(p, q):
            d_bc_ft0_x[i] = d_bc_ft_x[i]
            d_bc_ft0_y[i] = d_bc_ft_y[i]
            d_bc_ft0_z[i] = d_bc_ft_z[i]

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
