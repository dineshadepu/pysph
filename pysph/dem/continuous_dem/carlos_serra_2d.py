"""
This is an implementation of BPM only for 2-d problems
mainly implemented from "
code implementation of particle based
discrete element method for concrete
viscoelastic modeling"

by Carlos Serra
"""
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


def get_particle_array_bonded_dem_potyondy_2d(constants=None, **props):
    """Return a particle array for a dem particles
    """
    dim = props.pop('dim', None)

    dem_props = [
        'wz', 'fx', 'fy', 'torz', 'rad_s', 'm_inverse', 'I_inverse', 'u0',
        'v0', 'x0', 'y0', 'wz0'
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
    pa.add_property('bc_fn', stride=bc_limit)
    pa.add_property('bc_fn0', stride=bc_limit)
    pa.add_property('bc_ft_x', stride=bc_limit)
    pa.add_property('bc_ft_y', stride=bc_limit)
    pa.add_property('bc_ft0_x', stride=bc_limit)
    pa.add_property('bc_ft0_y', stride=bc_limit)

    pa.set_output_arrays([
        'x', 'y', 'u', 'v', 'wz', 'm', 'pid', 'tag', 'gid', 'fx', 'fy', 'torz',
        'rad_s', 'dem_id'
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


class PotyondyIPForceEuler(Equation):
    def __init__(self, dest, sources, kn):
        self.kn = kn
        self.kt = kn / 2.
        super(PotyondyIPForceEuler, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_bc_fn,
                   d_bc_ft_x, d_bc_ft_y, d_bc_fn0, d_bc_ft0_x, d_bc_ft0_y,
                   d_bc_limit, d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_torz,
                   d_u, d_v, d_rad_s, d_wz, dt):
        p, q1, tot_ctcs, i, sidx = declare('int', 5)
        xij = declare('matrix(2)')
        vij = declare('matrix(3)')
        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        # we want to find the force acting on particle d_idx due to
        # contact with particle sidx
        for i in range(p, q1):
            sidx = d_bc_idx[i]

            xij[0] = d_x[sidx] - d_x[d_idx]
            xij[1] = d_y[sidx] - d_y[d_idx]
            vij[0] = d_u[d_idx] - d_u[sidx]
            vij[1] = d_v[d_idx] - d_v[sidx]

            # distance between particle d_idx and sidx
            # equation (1)
            d = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

            # find the overlap
            # equation (2)
            # this can be positive or negative
            overlap = (d_rad_s[d_idx] + d_rad_s[sidx]) - d

            # normal vector from d_idx to sidx
            # equation (3)
            dinv = 1. / d
            nx = xij[0] * dinv
            ny = xij[1] * dinv

            # find the contact point from d_idx
            # equation (4)
            xc = d_x[d_idx] + (d_rad_s[d_idx] - overlap / 2.) * nx
            yc = d_y[d_idx] + (d_rad_s[d_idx] - overlap / 2.) * ny

            # velocity of particle d_idx at contact point
            # equation (6)
            u_didx_c = d_u[d_idx] - d_wz[d_idx] * (xc - d_x[d_idx])
            v_didx_c = d_v[d_idx] + d_wz[d_idx] * (yc - d_y[d_idx])

            # velocity of particle sidx at contact point
            # equation (6)
            u_sidx_c = d_u[sidx] - d_wz[d_idx] * (xc - d_x[sidx])
            v_sidx_c = d_v[sidx] + d_wz[d_idx] * (yc - d_y[sidx])

            # find the velocity of the contact point, which involves both
            # the particles velocities at the contact point
            # V_c = V_B_c - V_A_c
            # this equation is a little different from what I am used
            # to. Need to see what happens when I flip it, i.e.,
            # V_c = V_A_c - V_B_c
            # equation (5)
            uc = u_sidx_c - u_didx_c
            vc = v_sidx_c - v_didx_c
            # find the relative velocity in tangential direction
            # and normal direction
            vc_dot_nij = uc * nx + vc * ny
            # relative normal velocity
            vn_x = vc_dot_nij * nx
            vn_y = vc_dot_nij * ny

            # relative tangential velocity
            vt_x = uc - vn_x
            vt_y = vc - vn_y
            # magnitude of the tangential velocity
            vt_magn = (vt_x * vt_x + vt_y * vt_y)**0.5

            # the displacement increment of the contact point.
            # equation 8
            dx_c = uc * dt / 2.
            dy_c = vc * dt / 2.

            # this can be decomposed into a normal and tangential part
            # equation (9)
            disp_n = dx_c * nx + dy_c * ny
            # equation (10)
            disp_t_x = dx_c - disp_n * nx
            disp_t_y = dy_c - disp_n * ny

            # the force increment from the displacements can be computed as
            df_n = -self.kn * disp_n
            df_t_x = -self.ks * disp_t_x
            df_t_y = -self.ks * disp_t_y

            # ------------- Rotate the spring -----------------------
            # rotate the spring
            ft_magn = (d_bc_ft_x[i]**2. + d_bc_ft_y[i]**2.)
            ft_dot_nij = (d_bc_ft_x[i] * nx + d_bc_ft_y[i] * ny)

            # tangential force projected onto the current normal of the
            # contact place
            ft_px = d_bc_ft_x[i] - ft_dot_nij * nx
            ft_py = d_bc_ft_y[i] - ft_dot_nij * ny

            ftp_magn = (ft_px**2. + ft_py**2.)**0.5
            if ftp_magn > 0:
                one_by_ftp_magn = 1. / ftp_magn

                tx = ft_px * one_by_ftp_magn
                ty = ft_py * one_by_ftp_magn
            else:
                if vt_magn > 0.:
                    tx = -vt_x / vt_magn
                    ty = -vt_y / vt_magn
                else:
                    tx = 0.
                    ty = 0.

            d_bc_ft_x[i] = ft_magn * tx
            d_bc_ft_y[i] = ft_magn * ty

            # ------------- Rotate the spring -----------------------
            # add the force to the particles before the increment
            d_fx[d_idx] += d_bc_ft_x[i] + d_bc_fn[i] * nx
            d_fy[d_idx] += d_bc_ft_y[i] + d_bc_fn[i] * ny

            # compute the moment due to the tangential force
            d_torz[d_idx] += d_bc_ft_x[i] + d_bc_fn[i] * nx

            # Add the increments to the bond forces, for
            # next time step
            d_bc_fn[i] += df_n
            d_bc_ft_x[i] += df_t_x
            d_bc_ft_y[i] += df_t_y


class PotyondyIPForceStage1(Equation):
    def __init__(self, dest, sources, kn):
        self.kn = kn
        self.kt = kn / 2.
        super(PotyondyIPForceStage1, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_bc_fn,
                   d_bc_ft_x, d_bc_ft_y, d_bc_fn0, d_bc_ft0_x, d_bc_ft0_y,
                   d_bc_limit, d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_torz,
                   d_u, d_v, d_rad_s, d_wz, dt):
        p, q1, tot_ctcs, i, sidx = declare('int', 5)
        xij = declare('matrix(2)')
        vij = declare('matrix(3)')
        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        # we want to find the force acting on particle d_idx due to
        # contact with particle sidx
        for i in range(p, q1):
            sidx = d_bc_idx[i]

            xij[0] = d_x[sidx] - d_x[d_idx]
            xij[1] = d_y[sidx] - d_y[d_idx]
            vij[0] = d_u[d_idx] - d_u[sidx]
            vij[1] = d_v[d_idx] - d_v[sidx]

            # distance between particle d_idx and sidx
            # equation (1)
            d = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

            # find the overlap
            # equation (2)
            # this can be positive or negative
            overlap = (d_rad_s[d_idx] + d_rad_s[sidx]) - d

            # normal vector from d_idx to sidx
            # equation (3)
            dinv = 1. / d
            nx = xij[0] * dinv
            ny = xij[1] * dinv

            # find the contact point from d_idx
            # equation (4)
            xc = d_x[d_idx] + (d_rad_s[d_idx] - overlap / 2.) * nx
            yc = d_y[d_idx] + (d_rad_s[d_idx] - overlap / 2.) * ny

            # velocity of particle d_idx at contact point
            # equation (6)
            u_didx_c = d_u[d_idx] - d_wz[d_idx] * (xc - d_x[d_idx])
            v_didx_c = d_v[d_idx] + d_wz[d_idx] * (yc - d_y[d_idx])

            # velocity of particle sidx at contact point
            # equation (6)
            u_sidx_c = d_u[sidx] - d_wz[d_idx] * (xc - d_x[sidx])
            v_sidx_c = d_v[sidx] + d_wz[d_idx] * (yc - d_y[sidx])

            # find the velocity of the contact point, which involves both
            # the particles velocities at the contact point
            # V_c = V_B_c - V_A_c
            # this equation is a little different from what I am used
            # to. Need to see what happens when I flip it, i.e.,
            # V_c = V_A_c - V_B_c
            # equation (5)
            uc = u_sidx_c - u_didx_c
            vc = v_sidx_c - v_didx_c
            # find the relative velocity in tangential direction
            # and normal direction
            vc_dot_nij = uc * nx + vc * ny
            # relative normal velocity
            vn_x = vc_dot_nij * nx
            vn_y = vc_dot_nij * ny

            # relative tangential velocity
            vt_x = uc - vn_x
            vt_y = vc - vn_y
            # magnitude of the tangential velocity
            vt_magn = (vt_x * vt_x + vt_y * vt_y)**0.5

            # the displacement increment of the contact point.
            # equation 8
            dx_c = uc * dt / 2.
            dy_c = vc * dt / 2.

            # this can be decomposed into a normal and tangential part
            # equation (9)
            disp_n = dx_c * nx + dy_c * ny
            # equation (10)
            disp_t_x = dx_c - disp_n * nx
            disp_t_y = dy_c - disp_n * ny

            # the force increment from the displacements can be computed as
            df_n = -self.kn * disp_n
            df_t_x = -self.ks * disp_t_x
            df_t_y = -self.ks * disp_t_y

            # ------------- Rotate the spring -----------------------
            # rotate the spring
            ft_magn = (d_bc_ft_x[i]**2. + d_bc_ft_y[i]**2.)
            ft_dot_nij = (d_bc_ft_x[i] * nx + d_bc_ft_y[i] * ny)

            # tangential force projected onto the current normal of the
            # contact place
            ft_px = d_bc_ft_x[i] - ft_dot_nij * nx
            ft_py = d_bc_ft_y[i] - ft_dot_nij * ny

            ftp_magn = (ft_px**2. + ft_py**2.)**0.5
            if ftp_magn > 0:
                one_by_ftp_magn = 1. / ftp_magn

                tx = ft_px * one_by_ftp_magn
                ty = ft_py * one_by_ftp_magn
            else:
                if vt_magn > 0.:
                    tx = -vt_x / vt_magn
                    ty = -vt_y / vt_magn
                else:
                    tx = 0.
                    ty = 0.

            d_bc_ft_x[i] = ft_magn * tx
            d_bc_ft_y[i] = ft_magn * ty

            # ------------- Rotate the spring -----------------------
            # add the force to the particles before the increment
            d_fx[d_idx] += d_bc_ft_x[i] + d_bc_fn[i] * nx
            d_fy[d_idx] += d_bc_ft_y[i] + d_bc_fn[i] * ny

            # compute the moment due to the tangential force
            d_torz[d_idx] += d_bc_ft_x[i] + d_bc_fn[i] * nx

            # Add the increments to the bond forces, for
            # next time step
            d_bc_fn[i] += df_n
            d_bc_ft_x[i] += df_t_x
            d_bc_ft_y[i] += df_t_y


class PotyondyIPForceStage2(Equation):
    def __init__(self, dest, sources, kn):
        self.kn = kn
        self.kt = kn / 2.
        super(PotyondyIPForceStage2, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_bc_fn,
                   d_bc_ft_x, d_bc_ft_y, d_bc_fn0, d_bc_ft0_x, d_bc_ft0_y,
                   d_bc_limit, d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_torz,
                   d_u, d_v, d_rad_s, d_wz, dt):
        p, q1, tot_ctcs, i, sidx = declare('int', 5)
        xij = declare('matrix(2)')
        vij = declare('matrix(3)')
        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        # we want to find the force acting on particle d_idx due to
        # contact with particle sidx
        for i in range(p, q1):
            sidx = d_bc_idx[i]

            xij[0] = d_x[sidx] - d_x[d_idx]
            xij[1] = d_y[sidx] - d_y[d_idx]
            vij[0] = d_u[d_idx] - d_u[sidx]
            vij[1] = d_v[d_idx] - d_v[sidx]

            # distance between particle d_idx and sidx
            # equation (1)
            d = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

            # find the overlap
            # equation (2)
            # this can be positive or negative
            overlap = (d_rad_s[d_idx] + d_rad_s[sidx]) - d

            # normal vector from d_idx to sidx
            # equation (3)
            dinv = 1. / d
            nx = xij[0] * dinv
            ny = xij[1] * dinv

            # find the contact point from d_idx
            # equation (4)
            xc = d_x[d_idx] + (d_rad_s[d_idx] - overlap / 2.) * nx
            yc = d_y[d_idx] + (d_rad_s[d_idx] - overlap / 2.) * ny

            # velocity of particle d_idx at contact point
            # equation (6)
            u_didx_c = d_u[d_idx] - d_wz[d_idx] * (xc - d_x[d_idx])
            v_didx_c = d_v[d_idx] + d_wz[d_idx] * (yc - d_y[d_idx])

            # velocity of particle sidx at contact point
            # equation (6)
            u_sidx_c = d_u[sidx] - d_wz[d_idx] * (xc - d_x[sidx])
            v_sidx_c = d_v[sidx] + d_wz[d_idx] * (yc - d_y[sidx])

            # find the velocity of the contact point, which involves both
            # the particles velocities at the contact point
            # V_c = V_B_c - V_A_c
            # this equation is a little different from what I am used
            # to. Need to see what happens when I flip it, i.e.,
            # V_c = V_A_c - V_B_c
            # equation (5)
            uc = u_sidx_c - u_didx_c
            vc = v_sidx_c - v_didx_c
            # find the relative velocity in tangential direction
            # and normal direction
            vc_dot_nij = uc * nx + vc * ny
            # relative normal velocity
            vn_x = vc_dot_nij * nx
            vn_y = vc_dot_nij * ny

            # relative tangential velocity
            vt_x = uc - vn_x
            vt_y = vc - vn_y
            # magnitude of the tangential velocity
            vt_magn = (vt_x * vt_x + vt_y * vt_y)**0.5

            # the displacement increment of the contact point.
            # equation 8
            dx_c = uc * dt / 2.
            dy_c = vc * dt / 2.

            # this can be decomposed into a normal and tangential part
            # equation (9)
            disp_n = dx_c * nx + dy_c * ny
            # equation (10)
            disp_t_x = dx_c - disp_n * nx
            disp_t_y = dy_c - disp_n * ny

            # the force increment from the displacements can be computed as
            df_n = -self.kn * disp_n
            df_t_x = -self.ks * disp_t_x
            df_t_y = -self.ks * disp_t_y

            # ------------- Rotate the spring -----------------------
            # rotate the spring
            ft_magn = (d_bc_ft_x[i]**2. + d_bc_ft_y[i]**2.)
            ft_dot_nij = (d_bc_ft_x[i] * nx + d_bc_ft_y[i] * ny)

            # tangential force projected onto the current normal of the
            # contact place
            ft_px = d_bc_ft_x[i] - ft_dot_nij * nx
            ft_py = d_bc_ft_y[i] - ft_dot_nij * ny

            ftp_magn = (ft_px**2. + ft_py**2.)**0.5
            if ftp_magn > 0:
                one_by_ftp_magn = 1. / ftp_magn

                tx = ft_px * one_by_ftp_magn
                ty = ft_py * one_by_ftp_magn
            else:
                if vt_magn > 0.:
                    tx = -vt_x / vt_magn
                    ty = -vt_y / vt_magn
                else:
                    tx = 0.
                    ty = 0.

            d_bc_ft_x[i] = ft_magn * tx
            d_bc_ft_y[i] = ft_magn * ty

            # ------------- Rotate the spring -----------------------
            # add the force to the particles before the increment
            d_fx[d_idx] += d_bc_ft_x[i] + d_bc_fn[i] * nx
            d_fy[d_idx] += d_bc_ft_y[i] + d_bc_fn[i] * ny

            # compute the moment due to the tangential force
            d_torz[d_idx] += d_bc_ft_x[i] + d_bc_fn[i] * nx

            # Add the increments to the bond forces, for
            # next time step
            d_bc_fn[i] += df_n
            d_bc_ft_x[i] += df_t_x
            d_bc_ft_y[i] += df_t_y


class EulerStepCarlos(IntegratorStep):
    def stage1(self, d_idx, d_x, d_y, d_u, d_v, d_fx, d_fy, d_x0, d_y0, d_u0,
               d_v0, d_wz0, d_torz, d_wz, d_m_inverse, d_I_inverse, dt):
        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]

        d_u[d_idx] = d_u[d_idx] + dt * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v[d_idx] + dt * d_fy[d_idx] * d_m_inverse[d_idx]

        d_wz[d_idx] = d_wz[d_idx] + dt * d_torz[d_idx] * d_I_inverse[d_idx]


class RK2StepCarlos(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_x0, d_y0, d_u, d_v, d_u0, d_v0,
                   d_wz, d_wz0, d_bc_total_contacts, d_bc_limit, d_bc_fn,
                   d_bc_ft_x, d_bc_ft_y, d_bc_fn0, d_bc_ft0_x, d_bc_ft0_y):

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
        tot_ctcs = declare('int')
        tot_ctcs = d_bc_total_contacts[d_idx]
        p = d_idx * d_bc_limit[0]
        q = p + tot_ctcs

        for i in range(p, q):
            d_bc_fn0[i] = d_bc_fn[i]
            d_bc_ft0_x[i] = d_bc_ft_x[i]
            d_bc_ft0_y[i] = d_bc_ft_y[i]

    def stage1(self, d_idx, d_x, d_y, d_u, d_v, d_fx, d_fy, d_x0, d_y0, d_u0,
               d_v0, d_wz0, d_torz, d_wz, d_m_inverse, d_I_inverse, dt):
        dtb2 = dt / 2.

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_fy[d_idx] * d_m_inverse[d_idx]

        d_wz[d_idx] = d_wz0[d_idx] + dtb2 * d_torz[d_idx] * d_I_inverse[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_u, d_v, d_fx, d_fy, d_x0, d_y0, d_u0,
               d_v0, d_wz0, d_torz, d_wz, d_m_inverse, d_I_inverse, dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_fy[d_idx] * d_m_inverse[d_idx]

        d_wz[d_idx] = d_wz0[d_idx] + dt * d_torz[d_idx] * d_I_inverse[d_idx]
