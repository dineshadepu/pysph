from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from compyle.api import declare
from math import sqrt, asin, sin, cos, log, acos
import numpy as np
from pysph.sph.scheme import Scheme
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EulerIntegrator
from pysph.base.kernels import CubicSpline


def get_particle_array_bonded_dem_wang(constants=None, **props):
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
    # initial bond alignment
    pa.add_property('bc_x1_x', stride=bc_limit)
    pa.add_property('bc_x1_y', stride=bc_limit)
    pa.add_property('bc_x1_z', stride=bc_limit)
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
            SetupContactsCohesiveBeamBC(dest=pa.name, sources=[pa.name],
                                        beta=beta),
        ])
    ]
    arrays = [pa]
    a_eval = make_accel_eval(eqs1, arrays, dim)
    a_eval.evaluate()


class SetupContactsCohesiveBeamBC(Equation):
    def __init__(self, dest, sources, beta):
        self.beta = beta
        super(SetupContactsCohesiveBeamBC, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_bc_total_contacts, d_bc_idx, d_bc_limit,
             d_rad_s, d_bc_rest_len, d_bc_x1_x, d_bc_x1_y, d_bc_x1_z, s_rad_s,
             RIJ, XIJ):
        p = declare('int')
        p = d_bc_limit[0] * d_idx + d_bc_total_contacts[d_idx]
        fac = RIJ / (d_rad_s[d_idx] + s_rad_s[s_idx])
        if RIJ > 1e-12:
            if 1. - self.beta < fac < 1 + self.beta:
                d_bc_idx[p] = s_idx
                d_bc_total_contacts[d_idx] += 1
                d_bc_rest_len[p] = RIJ
                d_bc_x1_x[p] = -XIJ[0] / RIJ
                d_bc_x1_y[p] = -XIJ[1] / RIJ
                d_bc_x1_z[p] = -XIJ[2] / RIJ


def cross_product(a=[1.0, 0.0], b=[1.0, 0.0], result=[0.0, 0.0]):
    """
    Computes a cross b
    """
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]


def dot(a=[1.0, 0.0], b=[1.0, 0.0]):
    """
    Computes a dot b
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class AndreIPForceStage1(Equation):
    def __init__(self, dest, sources, kn):
        self.kn = kn
        self.kt = kn / 2.
        super(AndreIPForceStage1, self).__init__(dest, sources)

    def _get_helpers_(self):
        return [cross_product, dot]

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_z, d_bc_ft_x,
                   d_bc_ft_y, d_bc_ft_z, d_bc_ft0_x, d_bc_ft0_y, d_bc_ft0_z,
                   d_bc_limit, d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_u, d_v,
                   d_bc_x1_x, d_bc_x1_y, d_bc_x1_z, d_bc_x2_x, d_bc_x2_y,
                   d_bc_x2_z, dt):
        p, q1, tot_ctcs, i, sidx = declare('int', 5)
        # local coordinate frame of particle 1 or d_idx
        x1 = declare('matrix(3)')
        x2 = declare('matrix(3)')
        x = declare('matrix(3)')
        y = declare('matrix(3)')
        z = declare('matrix(3)')
        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        for i in range(p, q1):
            sidx = d_bc_idx[i]
            rij = 0.0

            dx = d_x[d_idx] - d_x[sidx]
            dy = d_y[d_idx] - d_y[sidx]
            dz = d_z[d_idx] - d_z[sidx]
            rij = sqrt(x[0] * x[0] + x[1] * x[1])
            # the unit vector passing from particle didx to sidx
            # at time t = 0 (we call it x1)
            x1[0] = d_bc_x1_x[sidx]
            x1[1] = d_bc_x1_y[sidx]
            x1[2] = d_bc_x1_z[sidx]

            x2[0] = d_bc_x1_x[sidx]
            x2[1] = d_bc_x1_y[sidx]
            x2[2] = d_bc_x1_z[sidx]

            # normal vector from i to j
            x[0] = dx / rij
            x[1] = dy / rij
            x[2] = dz / rij

            # determine the other local frame axis (y and z)
            cross_product(x, x1, y)
            cross_product(x, y, z)

            # find the angle made by local axis with the initial
            # median line
            theta1_x = acos(dot(x, x1))
            theta1_y = acos(dot(y, x1))
            theta1_z = acos(dot(z, x1))

            theta2_x = acos(dot(x, x2))
            theta2_y = acos(dot(y, x2))
            theta2_z = acos(dot(z, x2))

            delta_l = 0.
            if rij > 0:
                delta_l = rij - d_bc_rest_len[i]

            # ---------- force computation starts ------------
            # here is kn is needed to be fixed with the beam constants
            fx = self.kn * delta_l * x[0] - self.kn * (
                (theta1_z + theta2_z) * y[0] + (theta1_y + theta2_y) * z[0])
            fy = self.kn * delta_l * x[1] - self.kn * (
                (theta1_z + theta2_z) * y[1] + (theta1_y + theta2_y) * z[1])
            fz = self.kn * delta_l * x[2] - self.kn * (
                (theta1_z + theta2_z) * y[2] + (theta1_y + theta2_y) * z[2])
            # find the moment
            mx = self.kn * (theta2_x - theta1_x) * x[0] - self.kn * (
                (theta2_y + 2. * theta1_y) * y[0] -
                (theta2_z + 2. * theta1_z) * z[0])
            my = self.kn * (theta2_x - theta1_x) * x[1] - self.kn * (
                (theta2_y + 2. * theta1_y) * y[1] -
                (theta2_z + 2. * theta1_z) * z[1])
            mz = self.kn * (theta2_x - theta1_x) * x[2] - self.kn * (
                (theta2_y + 2. * theta1_y) * y[2] -
                (theta2_z + 2. * theta1_z) * z[2])


class AndreIPForceStage2(Equation):
    def __init__(self, dest, sources, kn):
        self.kn = kn
        self.kt = kn / 2.
        super(AndreIPForceStage2, self).__init__(dest, sources)

    def _get_helpers_(self):
        return [cross_product, dot]

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_z, d_bc_ft_x,
                   d_bc_ft_y, d_bc_ft_z, d_bc_ft0_x, d_bc_ft0_y, d_bc_ft0_z,
                   d_bc_limit, d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_u, d_v,
                   d_bc_x1_x, d_bc_x1_y, d_bc_x1_z, d_bc_x2_x, d_bc_x2_y,
                   d_bc_x2_z, dt):
        p, q1, tot_ctcs, i, sidx = declare('int', 5)
        # local coordinate frame of particle 1 or d_idx
        x1 = declare('matrix(3)')
        x2 = declare('matrix(3)')
        x = declare('matrix(3)')
        y = declare('matrix(3)')
        z = declare('matrix(3)')
        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        for i in range(p, q1):
            sidx = d_bc_idx[i]
            rij = 0.0

            dx = d_x[d_idx] - d_x[sidx]
            dy = d_y[d_idx] - d_y[sidx]
            dz = d_z[d_idx] - d_z[sidx]
            rij = sqrt(x[0] * x[0] + x[1] * x[1])
            # the unit vector passing from particle didx to sidx
            # at time t = 0 (we call it x1)
            x1[0] = d_bc_x1_x[sidx]
            x1[1] = d_bc_x1_y[sidx]
            x1[2] = d_bc_x1_z[sidx]

            x2[0] = d_bc_x1_x[sidx]
            x2[1] = d_bc_x1_y[sidx]
            x2[2] = d_bc_x1_z[sidx]

            # normal vector from i to j
            x[0] = dx / rij
            x[1] = dy / rij
            x[2] = dz / rij

            # determine the other local frame axis (y and z)
            cross_product(x, x1, y)
            cross_product(x, y, z)

            # find the angle made by local axis with the initial
            # median line
            theta1_x = acos(dot(x, x1))
            theta1_y = acos(dot(y, x1))
            theta1_z = acos(dot(z, x1))

            theta2_x = acos(dot(x, x2))
            theta2_y = acos(dot(y, x2))
            theta2_z = acos(dot(z, x2))

            delta_l = 0.
            if rij > 0:
                delta_l = rij - d_bc_rest_len[i]

            # ---------- force computation starts ------------
            # here is kn is needed to be fixed with the beam constants
            fx = self.kn * delta_l * x[0] - self.kn * (
                (theta1_z + theta2_z) * y[0] + (theta1_y + theta2_y) * z[0])
            fy = self.kn * delta_l * x[1] - self.kn * (
                (theta1_z + theta2_z) * y[1] + (theta1_y + theta2_y) * z[1])
            fz = self.kn * delta_l * x[2] - self.kn * (
                (theta1_z + theta2_z) * y[2] + (theta1_y + theta2_y) * z[2])
            # find the moment
            mx = self.kn * (theta2_x - theta1_x) * x[0] - self.kn * (
                (theta2_y + 2. * theta1_y) * y[0] -
                (theta2_z + 2. * theta1_z) * z[0])
            my = self.kn * (theta2_x - theta1_x) * x[1] - self.kn * (
                (theta2_y + 2. * theta1_y) * y[1] -
                (theta2_z + 2. * theta1_z) * z[1])
            mz = self.kn * (theta2_x - theta1_x) * x[2] - self.kn * (
                (theta2_y + 2. * theta1_y) * y[2] -
                (theta2_z + 2. * theta1_z) * z[2])


class RK2StepAndre(IntegratorStep):
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
