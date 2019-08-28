"""
This is a scheme of modeling continuous material using particles from

A new algorithm to model the dynamics of 3-D bonded rigid
bodies with rotations

"""
from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from compyle.api import declare
from math import sqrt, asin, sin, acos, log, cos
import numpy as np
from pysph.sph.scheme import Scheme
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EulerIntegrator
from pysph.base.kernels import CubicSpline
from pysph.sph.quaternion import (
    q_inverse_vec_q, dot_product, cross_product, magnitude,
    quaternion_multiplication, quaternion_inverse,
    rotate_vector_to_current_frame_with_quaternion)


def get_particle_array_bonded_dem_wang(constants=None, **props):
    """Return a particle array for a dem particles
    """
    dim = props.pop('dim', None)

    dem_props = [
        'wx', 'wy', 'wz', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'rad_s',
        'm_inverse', 'I_inverse', 'u0', 'v0', 'w0', 'wx0', 'wy0', 'wz0', 'x0',
        'y0', 'z0', 'q0', 'q1', 'q2', 'q3', 'q0_0', 'q1_0', 'q2_0', 'q3_0',
        'qinv0', 'qinv1', 'qinv2', 'qinv3', 'd_fx_b', 'd_fy_b', 'd_fz_b',
        'torx_b', 'tory_b', 'torz_b'
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
    pa.add_property('bc_x_t0', stride=bc_limit)
    pa.add_property('bc_y_t0', stride=bc_limit)
    pa.add_property('bc_z_t0', stride=bc_limit)
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
             d_rad_s, d_bc_x_t0, d_bc_y_t0, d_bc_z_t0, d_x, d_y, d_z, s_x, s_y,
             s_z, s_rad_s, RIJ):
        p = declare('int')
        p = d_bc_limit[0] * d_idx + d_bc_total_contacts[d_idx]
        fac = RIJ / (d_rad_s[d_idx] + s_rad_s[s_idx])
        if RIJ > 1e-12:
            if 1. - self.beta < fac < 1 + self.beta:
                d_bc_idx[p] = s_idx
                d_bc_total_contacts[d_idx] += 1
                # save the initial position of the contact
                # with respect to the particle d_idx
                # t0 is time 0
                d_bc_x_t0[p] = s_x[s_idx] - d_x[d_idx]
                d_bc_y_t0[p] = s_y[s_idx] - d_y[d_idx]
                d_bc_z_t0[p] = s_z[s_idx] - d_z[d_idx]


class WangIPForce(Equation):
    def __init__(self, dest, sources, kn):
        self.kn = kn
        self.kt = kn / 2.
        super(WangIPForce, self).__init__(dest, sources)

    def _get_helpers_(self):
        return [
            q_inverse_vec_q, dot_product, cross_product,
            quaternion_multiplication, quaternion_inverse,
            rotate_vector_to_current_frame_with_quaternion
        ]

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_z, d_bc_limit,
                   d_bc_idx, d_fx_b, d_fy_b, d_fz_b, d_torx_b, d_tory_b,
                   d_torz_b, d_u, d_v, d_q0, d_q1, d_q2, d_q3, d_qinv0,
                   d_qinv1, d_qinv2, d_qinv3, dt):
        p, q1, tot_ctcs, i, sidx = declare('int', 5)
        rf = declare('matrix(3)')
        rc = declare('matrix(3)')
        r0 = declare('matrix(3)')
        rc_cross_r0 = declare('matrix(3)')
        rc_cross_rc_cross_r0 = declare('matrix(3)')
        r0_cross_rc = declare('matrix(3)')
        s_vec = declare('matrix(3)')
        t_vec = declare('matrix(3)')
        q_didx = declare('matrix(4)')
        qinv_didx = declare('matrix(4)')
        q_sidx = declare('matrix(4)')
        g_not = declare('matrix(4)')
        g = declare('matrix(4)')
        h = declare('matrix(4)')
        h_inv = declare('matrix(4)')
        tmp_quat = declare('matrix(4)')

        # all the force declarations
        # normal force
        fr = declare('matrix(3)')
        # shear force's translational component
        fs_t = declare('matrix(3)')

        # torque due to shear force's translational component
        tors_t = declare('matrix(3)')

        # torque due to bending, represented in X2'Y2'Z2' frame
        tor_b_dash = declare('matrix(3)')

        # torque due to twisting, represented in X2'Y2'Z2' frame
        tor_t_dash = declare('matrix(3)')

        # shear force's rotational component represented in X2'Y2'Z2' frame
        fs_r_dash = declare('matrix(3)')

        # torque due to shear force's rotational component in X2'Y2'Z2' frame
        tors_r_dash = declare('matrix(3)')

        # torque due to bending, represented in X2Y2Z2 frame
        tor_b = declare('matrix(3)')

        # torque due to twisting, represented in X2Y2Z2 frame
        tor_t = declare('matrix(3)')

        # shear force's rotational component represented in X2Y2Z2 frame
        fs_r = declare('matrix(3)')

        # torque due to shear force's translational component
        tors_r = declare('matrix(3)')

        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        # save the quaternion of d_idx for easy computation
        q_didx[0] = d_q0[d_idx]
        q_didx[1] = d_q1[d_idx]
        q_didx[2] = d_q2[d_idx]
        q_didx[3] = d_q3[d_idx]

        qinv_didx[0] = d_qinv0[d_idx]
        qinv_didx[1] = d_qinv1[d_idx]
        qinv_didx[2] = d_qinv2[d_idx]
        qinv_didx[3] = d_qinv3[d_idx]

        for i in range(p, q1):
            sidx = d_bc_idx[i]

            # save the quaternion of sidx for easy computation
            q_sidx[0] = d_q0[sidx]
            q_sidx[1] = d_q1[sidx]
            q_sidx[2] = d_q2[sidx]
            q_sidx[3] = d_q3[sidx]

            # the relative position of particle sidx with respect
            # to d_idx in global frame as well as d_idx body frame
            # at time t=0. (This vector is same in both the frames
            # as at time 0 both frames are same (1., 0., 0., 0.) )
            # QUESTION: Should this vector be converted to the current
            # orientation of the body frame?
            r0[0] = d_bc_x_t0[i]
            r0[1] = d_bc_y_t0[i]
            r0[2] = d_bc_z_t0[i]

            # the relative position of particle sidx with respect
            # to d_idx in global frame
            rf[0] = d_x[sidx] - d_x[d_idx]
            rf[1] = d_y[sidx] - d_y[d_idx]
            rf[2] = d_z[sidx] - d_z[d_idx]

            # the relative position of particle sidx with respect
            # to d_idx in body frame of particle d_idx

            # transform the global vector rf to local frame
            # of particle d_idx by quaternion conversion
            # after the function call we have `rc` to be the
            # relative position of particle sidx with respect to
            # particle d_idx in body frame of d_idx
            # equation (11)
            q_inverse_vec_q(rf, q_didx, qinv_didx, rc)

            # ----------- computation of normal force -----------
            # normal force acting on particle d_idx
            # equation (12)
            rc_magn = magnitude(rc)
            r0_magn = magnitude(r0)

            # change in length between particles d_idx and sidx
            delta_r = rc_magn - r0_magn
            # now compute the normal force from the change in length and the
            # current direction of the connection

            # There are a few points to note here, that the force computed is
            # in body frame of d_idx. One needs to keep in mind that while
            # using this force move the particle forward, he/she has to
            # convert this force and use it.
            tmp = self.kr * delta_r
            fr[0] = tmp * rc[0] / rc_magn
            fr[1] = tmp * rc[1] / rc_magn
            fr[2] = tmp * rc[2] / rc_magn
            # ----------- computation of normal force ends -----------

            # -----------------------------------------------------
            # ----------- computation of shearing force -----------
            # -----------------------------------------------------

            # shearing force has two contributions, one is due to translational
            # and rotational. First we will look at translational shear force
            # then we will compute the rotational part.
            # equation (13)

            # ----------- computation of shear force due to translation -----
            # translational shear force acting on particle d_idx
            # equation (14)
            # compute angle gamma, between initial vector passing from
            # d_idx to sidx (r0) and with vector passing from d_idx to sidx
            # (rc) at current time.
            gamma = acos(dot_product(r0, rc) / (r0_magn * rc_magn))

            # the direction of the translational shear force is s_vec
            cross_product(rc, r0, rc_cross_r0)
            cross_product(rc, rc_cross_r0, rc_cross_rc_cross_r0)
            rc_c_rc_c_r0_magn = magnitude(rc_cross_rc_cross_r0)

            s_vec[0] = rc_cross_rc_cross_r0[0] / rc_c_rc_c_r0_magn
            s_vec[1] = rc_cross_rc_cross_r0[1] / rc_c_rc_c_r0_magn
            s_vec[2] = rc_cross_rc_cross_r0[2] / rc_c_rc_c_r0_magn

            # using gamma, vector and shear stiffness, the force results as
            tmp = self.ks * r0_magn * gamma
            fs_t[0] = tmp * s_vec[0]
            fs_t[1] = tmp * s_vec[1]
            fs_t[2] = tmp * s_vec[2]
            fs_t_magn = magnitude(fs_t)

            # since this force acts on the geometric center of the two body
            # system of d_idx and sidx. It generates a moment or torqe on
            # particle d_idx

            # the direction of the torque by the translational shear force is
            # t_vec
            cross_product(r0, rc, r0_cross_rc)
            r0_cross_rc_magn = magnitude(r0_cross_rc)

            t_vec[0] = r0_cross_rc[0] / r0_cross_rc_magn
            t_vec[1] = r0_cross_rc[1] / r0_cross_rc_magn
            t_vec[2] = r0_cross_rc[2] / r0_cross_rc_magn

            tmp = 0.5 * rc_magn * fs_t_magn
            tors_t[0] = tmp * t_vec[0]
            tors_t[1] = tmp * t_vec[1]
            tors_t[2] = tmp * t_vec[2]

            # ------ computation of shear force due to translation ends -----

            # ------- computation of shear force due to rotation ----------
            # the relative rotation from body frame of d_idx to the body frame
            # of sidx (X2Y2Z2 to X1Y1Z1) g^0 = q_inv * p (say g_not)
            # where q is quaternion of d_idx particle and p for sidx particle

            # lets first compute g_not
            quaternion_multiplication(qinv_didx, q_sidx, g_not)

            # now by using a quaternion (h), which specifies the rotation from
            # X2Y2Z2 to X2'Y2'Z2', we can compute the quaternion (g) which
            # specifies the rotation from X2'Y2'Z2' to X1'Y1'Z1', and given by
            # g = h_inv g_not h
            # first compute h
            tmp = 0.5**0.5
            tmp_1 = 1. / (rc[0]**2. + rc[1]**2.)**0.5
            rc_magn_1 = 1. / rc_magn
            h[0] = tmp * (rc_magn_1 + rc[2] * rc_magn_1)**0.5
            h[1] = -tmp * (rc_magn_1 - rc[2] * rc_magn_1)**0.5 * rc[1] * tmp_1
            h[2] = tmp * (rc_magn_1 - rc[2] * rc_magn_1)**0.5 * rc[0] * tmp_1
            h[3] = 0.

            quaternion_inverse(h, h_inv)

            # inorder to compute g we first multiply h_inv with g_not
            quaternion_multiplication(h_inv, g_not, tmp_quat)

            # then multiply tmp_quat with h resulting in g
            quaternion_multiplication(tmp_quat, h, g)

            # now compute the relative angles using the rotation quaternion g
            tmp_2 = (g[0]**2. + g[3]**2.)**0.5

            # equation (24)
            psi = 2. * acos(g[0] / tmp_2)

            # equation (25)
            theta = acos(g[0]**2. - g[1]**2. - g[2]**2. + g[3]**2.)

            # and finally phi (equation 26)
            tmp_3 = (g[1] * g[3] + g[0] * g[2])**0.5
            tmp_4 = ((g[0]**2. + g[3]**2.) * (g[1]**2. + g[2]**2.))**0.5
            phi = acos(tmp_3 / tmp_4)

            # Using equation (16) we compute the torque due to bending,
            # twisting and shear force (These are expressed in X2'Y2'Z2')
            # coordinate frame
            tmp = self.kb * theta
            tor_b_dash[0] = tmp * -sin(phi)
            tor_b_dash[1] = tmp * cos(phi)
            tor_b_dash[2] = 0.

            tor_t_dash[0] = 0.
            tor_t_dash[1] = 0.
            tor_t_dash[2] = self.kt * psi

            tmp = -self.ks * r0_magn * 0.5 * theta
            fs_r_dash[0] = tmp * cos(phi)
            fs_r_dash[1] = tmp * sin(phi)
            fs_r_dash[2] = 0.

            tmp = -tmp * 0.5
            tors_r_dash[0] = tmp * sin(phi)
            tors_r_dash[1] = tmp * -cos(phi)
            tors_r_dash[2] = 0.

            # shift the forces and torques into X2Y2Z2 frame

            # shift the bending torque
            rotate_vector_to_current_frame_with_quaternion(
                h, h_inv, tor_b_dash, tor_b)

            # shift the twisting torque
            rotate_vector_to_current_frame_with_quaternion(
                h, h_inv, tor_t_dash, tor_t)

            # shift the shear force due to relative rotational
            rotate_vector_to_current_frame_with_quaternion(
                h, h_inv, fs_r_dash, fs_r)

            # shift the torque generated due to shear force due to relative
            # rotational
            rotate_vector_to_current_frame_with_quaternion(
                h, h_inv, tors_r_dash, tors_r)

            # ------- computation of shear force due to rotation ends ---------

            # -----------------------------------------------------
            # ----------- computation of shearing force -----------
            # -----------------------------------------------------

            # now add all the forces and torque due the bond to global force
            # vector which is in body frame of d_idx. In another equation this
            # has to be converted to global frame and used to propagate the
            # system
            d_fx_b[d_idx] += fr[0] + fs_t[0] + fs_r[0]
            d_fy_b[d_idx] += fr[1] + fs_t[1] + fs_r[1]
            d_fz_b[d_idx] += fr[2] + fs_t[2] + fs_r[2]

            d_torx_b[d_idx] += tors_t[0] + tors_r[0] + tor_b[0] + tor_t[0]
            d_tory_b[d_idx] += tors_t[1] + tors_r[1] + tor_b[1] + tor_t[1]
            d_torz_b[d_idx] += tors_t[2] + tors_r[2] + tor_b[2] + tor_t[2]


class RK2StepWang(IntegratorStep):
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
