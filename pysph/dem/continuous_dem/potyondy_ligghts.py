from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from compyle.api import declare
from math import sqrt, asin, sin, cos, log, pi
import numpy as np
from pysph.sph.scheme import Scheme
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EulerIntegrator
from pysph.base.kernels import CubicSpline


def get_particle_array_bonded_dem_potyondy_ligghts(constants=None, **props):
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
    pa.add_property('bc_ft_x', stride=bc_limit)
    pa.add_property('bc_ft_y', stride=bc_limit)
    pa.add_property('bc_ft_z', stride=bc_limit)
    pa.add_property('bc_ft0_x', stride=bc_limit)
    pa.add_property('bc_ft0_y', stride=bc_limit)
    pa.add_property('bc_ft0_z', stride=bc_limit)

    pa.add_property('bc_fn_x', stride=bc_limit)
    pa.add_property('bc_fn_y', stride=bc_limit)
    pa.add_property('bc_fn_z', stride=bc_limit)
    pa.add_property('bc_fn0_x', stride=bc_limit)
    pa.add_property('bc_fn0_y', stride=bc_limit)
    pa.add_property('bc_fn0_z', stride=bc_limit)

    pa.add_property('bc_tort_x', stride=bc_limit)
    pa.add_property('bc_tort_y', stride=bc_limit)
    pa.add_property('bc_tort_z', stride=bc_limit)
    pa.add_property('bc_tort0_x', stride=bc_limit)
    pa.add_property('bc_tort0_y', stride=bc_limit)
    pa.add_property('bc_tort0_z', stride=bc_limit)

    pa.add_property('bc_torn_x', stride=bc_limit)
    pa.add_property('bc_torn_y', stride=bc_limit)
    pa.add_property('bc_torn_z', stride=bc_limit)
    pa.add_property('bc_torn0_x', stride=bc_limit)
    pa.add_property('bc_torn0_y', stride=bc_limit)
    pa.add_property('bc_torn0_z', stride=bc_limit)

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


def setup_bc_contacts_from_limit(dim, pa, limit):
    eqs1 = [
        Group(equations=[
            SetupContactsBCfromLimit(dest=pa.name, sources=[pa.name],
                                     limit=limit),
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
             d_rad_s, s_rad_s, RIJ):
        p = declare('int')
        p = d_bc_limit[0] * d_idx + d_bc_total_contacts[d_idx]
        fac = RIJ / (d_rad_s[d_idx] + s_rad_s[s_idx])

        if RIJ > 1e-12:
            # if d_idx == 402:
            #     print("------------------")
            #     print("------------------")
            #     print("------------------")
            #     print("s_idx is")
            #     print(s_idx)
            #     print("fac is ")
            #     print(fac)
            #     print("1 - self.beta")
            #     print(1. - self.beta)
            #     print("1 + self.beta")
            #     print(1. + self.beta)
            if 1. - self.beta < fac < 1 + self.beta:
                d_bc_idx[p] = s_idx
                d_bc_total_contacts[d_idx] += 1


class SetupContactsBCfromLimit(Equation):
    def __init__(self, dest, sources, limit):
        self.limit = limit
        super(SetupContactsBCfromLimit, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_bc_total_contacts, d_bc_idx, d_bc_limit,
             d_rad_s, s_rad_s, RIJ):
        p = declare('int')
        p = d_bc_limit[0] * d_idx + d_bc_total_contacts[d_idx]

        if RIJ > 1e-12:
            # if d_idx == 1607:
            #     print("------------------")
            #     print("------------------")
            #     print("------------------")
            #     print("s_idx is")
            #     print(s_idx)
            #     print("RIJ is")
            #     print(RIJ)
            #     print("and limit is ")
            #     print(self.limit)
            if RIJ < self.limit:
                # if d_idx == 1607:
                #     print("yes in contact")
                d_bc_idx[p] = s_idx
                d_bc_total_contacts[d_idx] += 1


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


class PotyondyLigghtsIPForceStage1(Equation):
    def __init__(self, dest, sources, kn, dim, lmbda=1):
        self.kn = kn
        self.kt = kn / 2.
        self.lmbda = lmbda
        self.dim = dim
        super(PotyondyLigghtsIPForceStage1, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_z, d_bc_limit,
                   d_bc_idx, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz, d_u,
                   d_v, d_w, d_wx, d_wy, d_wz, d_rad_s, d_bc_ft_x, d_bc_ft_y,
                   d_bc_ft_z, d_bc_ft0_x, d_bc_ft0_y, d_bc_ft0_z, d_bc_fn_x,
                   d_bc_fn_y, d_bc_fn_z, d_bc_fn0_x, d_bc_fn0_y, d_bc_fn0_z,
                   d_bc_torn_x, d_bc_torn_y, d_bc_torn_z, d_bc_tort_x,
                   d_bc_tort_y, d_bc_tort_z, dt):
        dtb2 = dt * 0.5
        p, q1, tot_ctcs, i, sidx = declare('int', 5)
        xij = declare('matrix(3)')
        dtforce = declare('matrix(3)')
        dttorque = declare('matrix(3)')
        dnforce = declare('matrix(3)')
        dntorque = declare('matrix(3)')
        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        for i in range(p, q1):
            sidx = d_bc_idx[i]

            rbmin = min(d_rad_s[d_idx], d_rad_s[sidx])
            A = pi * rbmin * rbmin
            J = A * 0.5 * rbmin * rbmin

            xij[0] = d_x[d_idx] - d_x[sidx]
            xij[1] = d_y[d_idx] - d_y[sidx]
            xij[2] = d_z[d_idx] - d_z[sidx]
            rsq = xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2]
            rsqinv = 1. / rsq
            r = sqrt(rsq)
            rinv = 1. / r

            # relative translational velocity
            vr1 = d_u[d_idx] - d_u[sidx]
            vr2 = d_v[d_idx] - d_v[sidx]
            vr3 = d_w[d_idx] - d_w[sidx]

            # normal component of translational relative velocity
            vnnr = vr1 * xij[0] + vr2 * xij[1] + vr3 * xij[2]
            vn1 = xij[0] * vnnr * rsqinv
            vn2 = xij[1] * vnnr * rsqinv
            vn3 = xij[2] * vnnr * rsqinv

            # tangential component of translational relative velocity
            vt1 = vr1 - vn1
            vt2 = vr2 - vn2
            vt3 = vr3 - vn3

            # relative rotational velocity for shear
            wr1 = (d_rad_s[d_idx] * d_wx[d_idx] +
                   d_rad_s[sidx] * d_wx[sidx]) * rinv
            wr2 = (d_rad_s[d_idx] * d_wy[d_idx] +
                   d_rad_s[sidx] * d_wy[sidx]) * rinv
            wr3 = (d_rad_s[d_idx] * d_wz[d_idx] +
                   d_rad_s[sidx] * d_wz[sidx]) * rinv

            # relative velocities for shear

            vtr1 = vt1 - (xij[2] * wr2 - xij[1] * wr3)
            vtr2 = vt2 - (xij[0] * wr3 - xij[2] * wr1)
            vtr3 = vt3 - (xij[1] * wr1 - xij[0] * wr2)

            # relative rotational velocity for tortion and bending
            wr1 = (d_rad_s[d_idx] * d_wx[d_idx] -
                   d_rad_s[sidx] * d_wx[sidx]) * rinv
            wr2 = (d_rad_s[d_idx] * d_wy[d_idx] -
                   d_rad_s[sidx] * d_wy[sidx]) * rinv
            wr3 = (d_rad_s[d_idx] * d_wz[d_idx] -
                   d_rad_s[sidx] * d_wz[sidx]) * rinv

            # normal component
            wnnr = wr1 * xij[0] + wr2 * xij[1] + wr3 * xij[2]
            wn1 = xij[0] * wnnr * rsqinv
            wn2 = xij[1] * wnnr * rsqinv
            wn3 = xij[2] * wnnr * rsqinv

            # tangential component
            wt1 = wr1 - wn1
            wt2 = wr2 - wn2
            wt3 = wr3 - wn3

            # calc change in normal forces
            # what is this Sn type: # XXX: FIXME
            dnforce[0] = -vn1 * self.kn * A * dtb2
            dnforce[1] = -vn2 * self.kn * A * dtb2
            dnforce[2] = -vn3 * self.kn * A * dtb2

            # calc change in shear forces
            dtforce[0] = -vtr1 * self.kt * A * dtb2
            dtforce[1] = -vtr2 * self.kt * A * dtb2
            dtforce[2] = -vtr3 * self.kt * A * dtb2

            # calc change in normal torque
            dntorque[0] = -wn1 * self.kt * J * dtb2
            dntorque[1] = -wn2 * self.kt * J * dtb2
            dntorque[2] = -wn3 * self.kt * J * dtb2

            # calc change in tang torque
            dttorque[0] = -wt1 * self.kn * J * 0.5 * dtb2
            dttorque[1] = -wt2 * self.kn * J * 0.5 * dtb2
            dttorque[2] = -wt3 * self.kn * J * 0.5 * dtb2

            # --------------------------------------
            # rotate forces
            # --------------------------------------

            # rotate normal force
            rot = (d_bc_fn_x[i] * xij[0] + d_bc_fn_y[i] * xij[1] +
                   d_bc_fn_z[i] * xij[2])
            rot *= rsqinv
            d_bc_fn_x[i] = rot * xij[0]
            d_bc_fn_y[i] = rot * xij[1]
            d_bc_fn_z[i] = rot * xij[2]

            # rotate tangential force
            rot = (d_bc_ft_x[i] * xij[0] + d_bc_ft_y[i] * xij[1] +
                   d_bc_ft_z[i] * xij[2])
            rot *= rsqinv
            d_bc_ft_x[i] -= rot * xij[0]
            d_bc_ft_y[i] -= rot * xij[1]
            d_bc_ft_z[i] -= rot * xij[2]

            # rotate normal torque
            rot = (d_bc_torn_x[i] * xij[0] + d_bc_torn_y[i] * xij[1] +
                   d_bc_torn_z[i] * xij[2])
            rot *= rsqinv
            d_bc_torn_x[i] = rot * xij[0]
            d_bc_torn_y[i] = rot * xij[1]
            d_bc_torn_z[i] = rot * xij[2]

            # rotate tangential torque
            rot = (d_bc_tort_x[i] * xij[0] + d_bc_tort_y[i] * xij[1] +
                   d_bc_tort_z[i] * xij[2])
            rot *= rsqinv
            d_bc_tort_x[i] -= rot * xij[0]
            d_bc_tort_y[i] -= rot * xij[1]
            d_bc_tort_z[i] -= rot * xij[2]

            # ------------------------------------------------
            # increment normal and tangential force and torque
            # ------------------------------------------------
            dissipate = 1
            d_bc_fn_x[i] = dissipate * d_bc_fn_x[i] + dnforce[0]
            d_bc_fn_y[i] = dissipate * d_bc_fn_y[i] + dnforce[1]
            d_bc_fn_z[i] = dissipate * d_bc_fn_z[i] + dnforce[2]

            d_bc_ft_x[i] = dissipate * d_bc_ft_x[i] + dtforce[0]
            d_bc_ft_y[i] = dissipate * d_bc_ft_y[i] + dtforce[1]
            d_bc_ft_z[i] = dissipate * d_bc_ft_z[i] + dtforce[2]

            d_bc_torn_x[i] = dissipate * d_bc_torn_x[i] + dntorque[0]
            d_bc_torn_y[i] = dissipate * d_bc_torn_y[i] + dntorque[1]
            d_bc_torn_z[i] = dissipate * d_bc_torn_z[i] + dntorque[2]

            d_bc_tort_x[i] = dissipate * d_bc_tort_x[i] + dttorque[0]
            d_bc_tort_y[i] = dissipate * d_bc_tort_y[i] + dttorque[1]
            d_bc_tort_z[i] = dissipate * d_bc_tort_z[i] + dttorque[2]

            # torque due to the bonded tangential force
            tor1 = -rinv * (xij[1] * d_bc_ft_z[i] - xij[2] * d_bc_ft_y[i])
            tor2 = -rinv * (xij[2] * d_bc_ft_x[i] - xij[0] * d_bc_ft_z[i])
            tor3 = -rinv * (xij[0] * d_bc_ft_y[i] - xij[1] * d_bc_ft_x[i])

            # add torques and forces to the global force of particle d_idx
            d_fx[d_idx] += (d_bc_fn_x[i] + d_bc_ft_x[i])
            d_fy[d_idx] += (d_bc_fn_y[i] + d_bc_ft_y[i])
            d_fz[d_idx] += (d_bc_fn_z[i] + d_bc_ft_z[i])

            d_torx[d_idx] += d_rad_s[d_idx] * tor1 + (d_bc_torn_x[i] +
                                                      d_bc_tort_x[i])
            d_tory[d_idx] += d_rad_s[d_idx] * tor2 + (d_bc_torn_y[i] +
                                                      d_bc_tort_y[i])
            d_torz[d_idx] += d_rad_s[d_idx] * tor3 + (d_bc_torn_z[i] +
                                                      d_bc_tort_z[i])


class PotyondyLigghtsIPForceStage2(Equation):
    def __init__(self, dest, sources, kn, dim, lmbda=1):
        self.kn = kn
        self.kt = kn / 2.
        self.lmbda = lmbda
        self.dim = dim
        super(PotyondyLigghtsIPForceStage2, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_z, d_bc_limit,
                   d_bc_idx, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz, d_u,
                   d_v, d_w, d_wx, d_wy, d_wz, d_rad_s, d_bc_ft_x, d_bc_ft_y,
                   d_bc_ft_z, d_bc_ft0_x, d_bc_ft0_y, d_bc_ft0_z, d_bc_fn_x,
                   d_bc_fn_y, d_bc_fn_z, d_bc_fn0_x, d_bc_fn0_y, d_bc_fn0_z,
                   d_bc_torn_x, d_bc_torn_y, d_bc_torn_z, d_bc_tort_x,
                   d_bc_tort_y, d_bc_tort_z, d_bc_torn0_x, d_bc_torn0_y,
                   d_bc_torn0_z, d_bc_tort0_x, d_bc_tort0_y, d_bc_tort0_z, dt):
        p, q1, tot_ctcs, i, sidx = declare('int', 5)
        xij = declare('matrix(3)')
        dtforce = declare('matrix(3)')
        dttorque = declare('matrix(3)')
        dnforce = declare('matrix(3)')
        dntorque = declare('matrix(3)')
        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        for i in range(p, q1):
            sidx = d_bc_idx[i]

            rbmin = min(d_rad_s[d_idx], d_rad_s[sidx])
            A = pi * rbmin * rbmin
            J = A * 0.5 * rbmin * rbmin

            xij[0] = d_x[d_idx] - d_x[sidx]
            xij[1] = d_y[d_idx] - d_y[sidx]
            xij[2] = d_z[d_idx] - d_z[sidx]
            rsq = xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2]
            rsqinv = 1. / rsq
            r = sqrt(rsq)
            rinv = 1. / r

            # relative translational velocity
            vr1 = d_u[d_idx] - d_u[sidx]
            vr2 = d_v[d_idx] - d_v[sidx]
            vr3 = d_w[d_idx] - d_w[sidx]

            # normal component of translational relative velocity
            vnnr = vr1 * xij[0] + vr2 * xij[1] + vr3 * xij[2]
            vn1 = xij[0] * vnnr * rsqinv
            vn2 = xij[1] * vnnr * rsqinv
            vn3 = xij[2] * vnnr * rsqinv

            # tangential component of translational relative velocity
            vt1 = vr1 - vn1
            vt2 = vr2 - vn2
            vt3 = vr3 - vn3

            # relative rotational velocity for shear
            wr1 = (d_rad_s[d_idx] * d_wx[d_idx] +
                   d_rad_s[sidx] * d_wx[sidx]) * rinv
            wr2 = (d_rad_s[d_idx] * d_wy[d_idx] +
                   d_rad_s[sidx] * d_wy[sidx]) * rinv
            wr3 = (d_rad_s[d_idx] * d_wz[d_idx] +
                   d_rad_s[sidx] * d_wz[sidx]) * rinv

            # relative velocities for shear

            vtr1 = vt1 - (xij[2] * wr2 - xij[1] * wr3)
            vtr2 = vt2 - (xij[0] * wr3 - xij[2] * wr1)
            vtr3 = vt3 - (xij[1] * wr1 - xij[0] * wr2)

            # relative rotational velocity for tortion and bending
            wr1 = (d_rad_s[d_idx] * d_wx[d_idx] -
                   d_rad_s[sidx] * d_wx[sidx]) * rinv
            wr2 = (d_rad_s[d_idx] * d_wy[d_idx] -
                   d_rad_s[sidx] * d_wy[sidx]) * rinv
            wr3 = (d_rad_s[d_idx] * d_wz[d_idx] -
                   d_rad_s[sidx] * d_wz[sidx]) * rinv

            # normal component
            wnnr = wr1 * xij[0] + wr2 * xij[1] + wr3 * xij[2]
            wn1 = xij[0] * wnnr * rsqinv
            wn2 = xij[1] * wnnr * rsqinv
            wn3 = xij[2] * wnnr * rsqinv

            # tangential component
            wt1 = wr1 - wn1
            wt2 = wr2 - wn2
            wt3 = wr3 - wn3

            # calc change in normal forces
            # what is this Sn type: # XXX: FIXME
            dnforce[0] = -vn1 * self.kn * A * dt
            dnforce[1] = -vn2 * self.kn * A * dt
            dnforce[2] = -vn3 * self.kn * A * dt

            # calc change in shear forces
            dtforce[0] = -vtr1 * self.kt * A * dt
            dtforce[1] = -vtr2 * self.kt * A * dt
            dtforce[2] = -vtr3 * self.kt * A * dt

            # calc change in normal torque
            dntorque[0] = -wn1 * self.kt * J * dt
            dntorque[1] = -wn2 * self.kt * J * dt
            dntorque[2] = -wn3 * self.kt * J * dt

            # calc change in tang torque
            dttorque[0] = -wt1 * self.kn * J * 0.5 * dt
            dttorque[1] = -wt2 * self.kn * J * 0.5 * dt
            dttorque[2] = -wt3 * self.kn * J * 0.5 * dt

            # --------------------------------------
            # rotate forces
            # --------------------------------------

            # rotate normal force at beginning of step
            rot = (d_bc_fn0_x[i] * xij[0] + d_bc_fn0_y[i] * xij[1] +
                   d_bc_fn0_z[i] * xij[2])
            rot *= rsqinv
            d_bc_fn0_x[i] = rot * xij[0]
            d_bc_fn0_y[i] = rot * xij[1]
            d_bc_fn0_z[i] = rot * xij[2]

            # rotate tangential force
            rot = (d_bc_ft0_x[i] * xij[0] + d_bc_ft0_y[i] * xij[1] +
                   d_bc_ft0_z[i] * xij[2])
            rot *= rsqinv
            d_bc_ft0_x[i] -= rot * xij[0]
            d_bc_ft0_y[i] -= rot * xij[1]
            d_bc_ft0_z[i] -= rot * xij[2]

            # rotate normal torque
            rot = (d_bc_torn0_x[i] * xij[0] + d_bc_torn0_y[i] * xij[1] +
                   d_bc_torn0_z[i] * xij[2])
            rot *= rsqinv
            d_bc_torn0_x[i] = rot * xij[0]
            d_bc_torn0_y[i] = rot * xij[1]
            d_bc_torn0_z[i] = rot * xij[2]

            # rotate tangential torque
            rot = (d_bc_tort0_x[i] * xij[0] + d_bc_tort0_y[i] * xij[1] +
                   d_bc_tort0_z[i] * xij[2])
            rot *= rsqinv
            d_bc_tort0_x[i] -= rot * xij[0]
            d_bc_tort0_y[i] -= rot * xij[1]
            d_bc_tort0_z[i] -= rot * xij[2]

            # ------------------------------------------------
            # increment normal and tangential force and torque
            # ------------------------------------------------
            dissipate = 1
            d_bc_fn_x[i] = dissipate * d_bc_fn0_x[i] + dnforce[0]
            d_bc_fn_y[i] = dissipate * d_bc_fn0_y[i] + dnforce[1]
            d_bc_fn_z[i] = dissipate * d_bc_fn0_z[i] + dnforce[2]

            d_bc_ft_x[i] = dissipate * d_bc_ft0_x[i] + dtforce[0]
            d_bc_ft_y[i] = dissipate * d_bc_ft0_y[i] + dtforce[1]
            d_bc_ft_z[i] = dissipate * d_bc_ft0_z[i] + dtforce[2]

            d_bc_torn_x[i] = dissipate * d_bc_torn0_x[i] + dntorque[0]
            d_bc_torn_y[i] = dissipate * d_bc_torn0_y[i] + dntorque[1]
            d_bc_torn_z[i] = dissipate * d_bc_torn0_z[i] + dntorque[2]

            d_bc_tort_x[i] = dissipate * d_bc_tort0_x[i] + dttorque[0]
            d_bc_tort_y[i] = dissipate * d_bc_tort0_y[i] + dttorque[1]
            d_bc_tort_z[i] = dissipate * d_bc_tort0_z[i] + dttorque[2]

            # torque due to the bonded tangential force
            tor1 = -rinv * (xij[1] * d_bc_ft_z[i] - xij[2] * d_bc_ft_y[i])
            tor2 = -rinv * (xij[2] * d_bc_ft_x[i] - xij[0] * d_bc_ft_z[i])
            tor3 = -rinv * (xij[0] * d_bc_ft_y[i] - xij[1] * d_bc_ft_x[i])

            # add torques and forces to the global force of particle d_idx
            d_fx[d_idx] += (d_bc_fn_x[i] + d_bc_ft_x[i])
            d_fy[d_idx] += (d_bc_fn_y[i] + d_bc_ft_y[i])
            d_fz[d_idx] += (d_bc_fn_z[i] + d_bc_ft_z[i])

            d_torx[d_idx] += d_rad_s[d_idx] * tor1 + (d_bc_torn_x[i] +
                                                      d_bc_tort_x[i])
            d_tory[d_idx] += d_rad_s[d_idx] * tor2 + (d_bc_torn_y[i] +
                                                      d_bc_tort_y[i])
            d_torz[d_idx] += d_rad_s[d_idx] * tor3 + (d_bc_torn_z[i] +
                                                      d_bc_tort_z[i])


class DampingForce(Equation):
    def __init__(self, dest, sources, alpha=0.7):
        self.alpha = alpha
        super(DampingForce, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_u, d_v, d_w, d_fx, d_fy, d_fz):
        # magnitude of velocity
        vmag = (d_u[d_idx]**2. + d_v[d_idx]**2. + d_w[d_idx]**2.)**0.5
        fmag = (d_fx[d_idx]**2. + d_fy[d_idx]**2. + d_fz[d_idx]**2.)**0.5

        if vmag > 0.:
            nx = d_u[d_idx] / vmag
            ny = d_v[d_idx] / vmag
            nz = d_w[d_idx] / vmag
        else:
            nx = 0.
            ny = 0.
            nz = 0.

        d_fx[d_idx] += -self.alpha * fmag * nx
        d_fy[d_idx] += -self.alpha * fmag * ny
        d_fz[d_idx] += -self.alpha * fmag * nz


class RK2StepPotyondy3d(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v, d_w,
                   d_u0, d_v0, d_w0, d_wx, d_wy, d_wz, d_wx0, d_wy0, d_wz0,
                   d_bc_total_contacts, d_bc_limit, d_bc_ft_x, d_bc_ft_y,
                   d_bc_ft_z, d_bc_ft0_x, d_bc_ft0_y, d_bc_ft0_z, d_bc_fn_x,
                   d_bc_fn_y, d_bc_fn_z, d_bc_fn0_x, d_bc_fn0_y, d_bc_fn0_z,
                   d_bc_tort_x, d_bc_tort_y, d_bc_tort_z, d_bc_tort0_x,
                   d_bc_tort0_y, d_bc_tort0_z, d_bc_torn_x, d_bc_torn_y,
                   d_bc_torn_z, d_bc_torn0_x, d_bc_torn0_y, d_bc_torn0_z):

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
            d_bc_fn0_x[i] = d_bc_fn_x[i]
            d_bc_fn0_y[i] = d_bc_fn_y[i]
            d_bc_fn0_z[i] = d_bc_fn_z[i]

            d_bc_tort0_x[i] = d_bc_tort_x[i]
            d_bc_tort0_y[i] = d_bc_tort_y[i]
            d_bc_tort0_z[i] = d_bc_tort_z[i]
            d_bc_torn0_x[i] = d_bc_torn_x[i]
            d_bc_torn0_y[i] = d_bc_torn_y[i]
            d_bc_torn0_z[i] = d_bc_torn_z[i]

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
