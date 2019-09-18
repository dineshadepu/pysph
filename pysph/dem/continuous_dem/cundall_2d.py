from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from compyle.api import declare
from math import sqrt, log
import numpy as np
from pysph.sph.scheme import Scheme
from pysph.sph.equation import Group, MultiStageEquations
from pysph.base.kernels import CubicSpline
from pysph.dem.discontinuous_dem.dem_nonlinear import EPECIntegratorMultiStage


def get_particle_array_bonded_dem_cundall_2d(constants=None, **props):
    """Return a particle array for a dem particles
    """
    dim = props.pop('dim', None)

    dem_props = [
        'wz', 'wz0', 'fx', 'fy', 'fz', 'torz', 'rad_s',
        'm_inverse', 'I_inverse', 'u0', 'v0', 'x0', 'y0'
    ]

    dem_id = props.pop('dem_id', None)

    pa = get_particle_array(additional_props=dem_props, **props)

    pa.add_property('dem_id', type='int', data=dem_id)
    # create the array to save the tangential interaction particles
    # index and other variables
    bc_limit = 6

    pa.add_constant('bc_limit', bc_limit)
    pa.add_property('bc_rest_len', stride=bc_limit)

    pa.add_property('bc_idx', stride=bc_limit, type="int")
    pa.bc_idx[:] = -1

    pa.add_property('bc_tng_frc', stride=bc_limit)
    pa.add_property('bc_tng_frc0', stride=bc_limit)
    pa.bc_tng_frc[:] = 0.
    pa.bc_tng_frc0[:] = 0.

    pa.add_property('bc_m_t', stride=bc_limit)
    pa.add_property('bc_m_t0', stride=bc_limit)
    pa.bc_m_t[:] = 0.
    pa.bc_m_t0[:] = 0.

    pa.add_property('bc_total_contacts', type="int")
    pa.bc_total_contacts[:] = 0

    pa.set_output_arrays([
        'x', 'y', 'u', 'v', 'm', 'pid', 'tag', 'gid', 'fx', 'fy', 'fz', 'torz',
        'I_inverse', 'wz', 'm_inverse', 'rad_s', 'dem_id', 'bc_idx',
        'bc_tng_frc', 'bc_tng_frc0', 'bc_total_contacts',
        'bc_m_t', 'bc_m_t0'
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


class BodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0):
        self.gx = gx
        self.gy = gy
        super(BodyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_fx, d_fy, d_torz):
        d_fx[d_idx] = d_m[d_idx] * self.gx
        d_fy[d_idx] = d_m[d_idx] * self.gy
        d_torz[d_idx] = 0.


class Cundall2dIPForceEuler(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall2dIPForceEuler, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_bc_total_contacts, d_x, d_y,
                   d_bc_tng_frc, d_bc_tng_frc0,
                   d_bc_limit, d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_torz,
                   d_u, d_v, d_rad_s, d_wz, dt,
                   d_bc_m_t):
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
            d = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

            # equation 2.8
            # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
            rinv = 1.0 / d

            # normal vector is passing from d_idx to sidx
            nx = xij[0] * rinv
            ny = xij[1] * rinv

            # tangential direction (rotate normal vector 90 degrees
            # clockwise)
            tx = ny
            ty = -nx

            # ---- Relative velocity computation (Eq 11) ----
            # follow Cundall equation (11)
            tmp = (d_wz[d_idx] * d_rad_s[d_idx] +
                   d_wz[sidx] * d_rad_s[sidx])

            # scalar components of relative velocity in normal and
            # tangential directions
            vn = vij[0] * nx + vij[1] * ny
            vt = vij[0] * tx + vij[1] * ty - tmp

            # damping force is taken from
            # "On the Determination of the Damping Coefficient
            # of Non-linear Spring-dashpot System to Model
            # Hertz Contact for Simulation by Discrete Element
            # Method" paper.
            # compute the damping constants
            m_eff = d_m[d_idx] * d_m[sidx] / (d_m[d_idx] + d_m[sidx])
            eta_n = self.alpha * sqrt(m_eff)

            # compute normal force

            # positive if tension and negative if compression
            overlap = d - d_bc_rest_len[i]
            kn_overlap = self.kn * overlap
            fn_x = kn_overlap * nx - eta_n * vn * nx
            fn_y = kn_overlap * ny - eta_n * vn * ny

            # ------------- tangential force computation ----------------
            # total number of contacts of particle i in destination
            tot_ctcs = d_bc_total_contacts[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_bc_limit[0]
            # ending index is q -1
            q1 = p + tot_ctcs

            # compute the damping constants
            eta_t = 0.5 * eta_n

            # find the tangential force from the tangential displacement
            # and tangential velocity (eq 2.11 Thesis Ye)
            ft = d_bc_tng_frc[i]

            # we have to compare with static friction, so
            # this mu has to be static friction coefficient
            fn_magn = (fn_x * fn_x + fn_y * fn_y)**(0.5)
            ft_max = self.mu * fn_magn

            # Coulomb slip
            if ft >= ft_max:
                ft = ft_max
                d_bc_tng_frc[i] = ft_max

            d_bc_tng_frc[i] += self.kt * vt * dt

            d_fx[d_idx] += fn_x - ft * tx
            d_fy[d_idx] += fn_y - ft * ty

            # moment due to the incremental rotation of the contact
            m_t = d_bc_m_t[i]
            # torque
            d_torz[d_idx] += ft * d_rad_s[d_idx] + m_t

            # increment the moment to the next time step
            # increment in the relative rotation (theta) is
            dtheta = (d_wz[d_idx] - d_wz[sidx]) * dt
            d_bc_m_t[i] -= dtheta * self.kn


class Cundall2dIPForceStage1(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall2dIPForceStage1, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_bc_total_contacts, d_x, d_y,
                   d_bc_tng_frc, d_bc_tng_frc0,
                   d_bc_limit, d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_torz,
                   d_u, d_v, d_rad_s, d_wz, dt,
                   d_bc_m_t):
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

        dtb2 = dt / 2.

        # we want to find the force acting on particle d_idx due to
        # contact with particle sidx
        for i in range(p, q1):
            sidx = d_bc_idx[i]

            xij[0] = d_x[sidx] - d_x[d_idx]
            xij[1] = d_y[sidx] - d_y[d_idx]
            vij[0] = d_u[d_idx] - d_u[sidx]
            vij[1] = d_v[d_idx] - d_v[sidx]

            # distance between particle d_idx and sidx
            d = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

            # equation 2.8
            # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
            rinv = 1.0 / d

            # normal vector is passing from d_idx to sidx
            nx = xij[0] * rinv
            ny = xij[1] * rinv

            # tangential direction (rotate normal vector 90 degrees
            # clockwise)
            tx = ny
            ty = -nx

            # ---- Relative velocity computation (Eq 11) ----
            # follow Cundall equation (11)
            tmp = (d_wz[d_idx] * d_rad_s[d_idx] +
                   d_wz[sidx] * d_rad_s[sidx])

            # scalar components of relative velocity in normal and
            # tangential directions
            vn = vij[0] * nx + vij[1] * ny
            vt = vij[0] * tx + vij[1] * ty - tmp

            # damping force is taken from
            # "On the Determination of the Damping Coefficient
            # of Non-linear Spring-dashpot System to Model
            # Hertz Contact for Simulation by Discrete Element
            # Method" paper.
            # compute the damping constants
            m_eff = d_m[d_idx] * d_m[sidx] / (d_m[d_idx] + d_m[sidx])
            eta_n = self.alpha * sqrt(m_eff)

            # compute normal force

            # positive if tension and negative if compression
            overlap = d - d_bc_rest_len[i]
            kn_overlap = self.kn * overlap
            fn_x = kn_overlap * nx - eta_n * vn * nx
            fn_y = kn_overlap * ny - eta_n * vn * ny

            # ------------- tangential force computation ----------------
            # total number of contacts of particle i in destination
            tot_ctcs = d_bc_total_contacts[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_bc_limit[0]
            # ending index is q -1
            q1 = p + tot_ctcs

            # compute the damping constants
            eta_t = 0.5 * eta_n

            # find the tangential force from the tangential displacement
            # and tangential velocity (eq 2.11 Thesis Ye)
            ft = d_bc_tng_frc[i]

            # we have to compare with static friction, so
            # this mu has to be static friction coefficient
            fn_magn = (fn_x * fn_x + fn_y * fn_y)**(0.5)
            ft_max = self.mu * fn_magn

            # Coulomb slip
            if ft >= ft_max:
                ft = ft_max
                d_bc_tng_frc[i] = ft_max

            d_bc_tng_frc[i] += self.kt * vt * dtb2

            d_fx[d_idx] += fn_x + ft * tx
            d_fy[d_idx] += fn_y + ft * ty

            # moment due to the incremental rotation of the contact
            m_t = d_bc_m_t[i]
            # torque
            d_torz[d_idx] += ft * d_rad_s[d_idx] + m_t

            # increment the moment to the next time step
            # increment in the relative rotation (theta) is
            dtheta = (d_wz[d_idx] - d_wz[sidx]) * dtb2
            d_bc_m_t[i] += dtheta * self.kn


class Cundall2dIPForceStage2(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(Cundall2dIPForceStage2, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_bc_total_contacts, d_x, d_y,
                   d_bc_tng_frc, d_bc_tng_frc0,
                   d_bc_limit, d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_torz,
                   d_u, d_v, d_rad_s, d_wz, dt,
                   d_bc_m_t, d_bc_m_t0):
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
            d = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

            # equation 2.8
            # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
            rinv = 1.0 / d

            # normal vector is passing from d_idx to sidx
            nx = xij[0] * rinv
            ny = xij[1] * rinv

            # tangential direction (rotate normal vector 90 degrees
            # clockwise)
            tx = ny
            ty = -nx

            # ---- Relative velocity computation (Eq 11) ----
            # follow Cundall equation (11)
            tmp = (d_wz[d_idx] * d_rad_s[d_idx] +
                   d_wz[sidx] * d_rad_s[sidx])

            # scalar components of relative velocity in normal and
            # tangential directions
            vn = vij[0] * nx + vij[1] * ny
            vt = vij[0] * tx + vij[1] * ty - tmp

            # damping force is taken from
            # "On the Determination of the Damping Coefficient
            # of Non-linear Spring-dashpot System to Model
            # Hertz Contact for Simulation by Discrete Element
            # Method" paper.
            # compute the damping constants
            m_eff = d_m[d_idx] * d_m[sidx] / (d_m[d_idx] + d_m[sidx])
            eta_n = self.alpha * sqrt(m_eff)

            # compute normal force

            # positive if tension and negative if compression
            overlap = d - d_bc_rest_len[i]
            kn_overlap = self.kn * overlap
            fn_x = kn_overlap * nx - eta_n * vn * nx
            fn_y = kn_overlap * ny - eta_n * vn * ny

            # ------------- tangential force computation ----------------
            # total number of contacts of particle i in destination
            tot_ctcs = d_bc_total_contacts[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_bc_limit[0]
            # ending index is q -1
            q1 = p + tot_ctcs

            # compute the damping constants
            eta_t = 0.5 * eta_n

            # find the tangential force from the tangential displacement
            # and tangential velocity (eq 2.11 Thesis Ye)
            ft = d_bc_tng_frc[i]

            # we have to compare with static friction, so
            # this mu has to be static friction coefficient
            fn_magn = (fn_x * fn_x + fn_y * fn_y)**(0.5)
            ft_max = self.mu * fn_magn

            # Coulomb slip
            if ft >= ft_max:
                ft = ft_max
                d_bc_tng_frc[i] = ft_max

            d_bc_tng_frc[i] = d_bc_tng_frc0[i] + self.kt * vt * dt

            d_fx[d_idx] += fn_x + ft * tx
            d_fy[d_idx] += fn_y + ft * ty

            # moment due to the incremental rotation of the contact
            m_t = d_bc_m_t[i]
            # torque
            d_torz[d_idx] += ft * d_rad_s[d_idx] + m_t

            # increment the moment to the next time step
            # increment in the relative rotation (theta) is
            dtheta = (d_wz[d_idx] - d_wz[sidx]) * dt
            d_bc_m_t[i] = d_bc_m_t0[i] + dtheta * self.kn


class RK2StepCundall2d(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_x0, d_y0, d_u, d_v, d_u0, d_v0,
                   d_wz, d_wz0, d_bc_total_contacts, d_bc_limit,
                   d_bc_tng_frc, d_bc_tng_frc0, d_bc_m_t, d_bc_m_t0):

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
            d_bc_tng_frc[i] = d_bc_tng_frc[i]
            d_bc_m_t0[i] = d_bc_m_t[i]

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
