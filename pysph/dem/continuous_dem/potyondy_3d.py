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


def get_particle_array_bonded_dem_potyondy_3d(constants=None, **props):
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
    pa.add_property('bc_fs_x', stride=bc_limit)
    pa.add_property('bc_fs_y', stride=bc_limit)
    pa.add_property('bc_fs_z', stride=bc_limit)
    pa.add_property('bc_fs0_x', stride=bc_limit)
    pa.add_property('bc_fs0_y', stride=bc_limit)
    pa.add_property('bc_fs0_z', stride=bc_limit)

    pa.add_property('bc_fn_x', stride=bc_limit)
    pa.add_property('bc_fn_y', stride=bc_limit)
    pa.add_property('bc_fn_z', stride=bc_limit)
    pa.add_property('bc_fn0_x', stride=bc_limit)
    pa.add_property('bc_fn0_y', stride=bc_limit)
    pa.add_property('bc_fn0_z', stride=bc_limit)

    pa.add_property('bc_ms_x', stride=bc_limit)
    pa.add_property('bc_ms_y', stride=bc_limit)
    pa.add_property('bc_ms_z', stride=bc_limit)
    pa.add_property('bc_ms0_x', stride=bc_limit)
    pa.add_property('bc_ms0_y', stride=bc_limit)
    pa.add_property('bc_ms0_z', stride=bc_limit)

    pa.add_property('bc_mn_x', stride=bc_limit)
    pa.add_property('bc_mn_y', stride=bc_limit)
    pa.add_property('bc_mn_z', stride=bc_limit)
    pa.add_property('bc_mn0_x', stride=bc_limit)
    pa.add_property('bc_mn0_y', stride=bc_limit)
    pa.add_property('bc_mn0_z', stride=bc_limit)

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


class Potyondy3dIPForceStage1(Equation):
    def __init__(self, dest, sources, kn, dim, lmbda=1):
        self.kn = kn
        self.ks = kn / 2.
        self.lmbda = lmbda
        self.dim = dim
        super(Potyondy3dIPForceStage1, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_z, d_bc_limit,
                   d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_fz, d_torx, d_tory,
                   d_torz, d_u, d_v, d_w, d_wx, d_wy, d_wz, d_rad_s, d_bc_fs_x,
                   d_bc_fs_y, d_bc_fs_z, d_bc_fs0_x, d_bc_fs0_y, d_bc_fs0_z,
                   d_bc_fn_x, d_bc_fn_y, d_bc_fn_z, d_bc_fn0_x, d_bc_fn0_y,
                   d_bc_fn0_z, d_bc_mn_x, d_bc_mn_y, d_bc_mn_z, d_bc_ms_x,
                   d_bc_ms_y, d_bc_ms_z, dt):
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
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])

            # normal vector from sidx to d_idx
            nji_x = xij[0] / rij
            nji_y = xij[1] / rij
            nji_z = xij[2] / rij

            # overlap positive for tension, negative for compression
            overlap = d_rad_s[d_idx] + d_rad_s[sidx] - rij

            # find the contact point of contacting bodies
            tmp = d_rad_s[d_idx] + overlap / 2.

            rc_x = d_x[d_idx] - nji_x * tmp
            rc_y = d_y[d_idx] - nji_y * tmp
            rc_z = d_z[d_idx] - nji_z * tmp

            # velocity of particle d_idx at contact point (call it vb_c)
            vb_cx = d_u[d_idx] + (d_wy[d_idx] * (rc_z - d_z[d_idx]) -
                                  d_wz[d_idx] * (rc_y - d_y[d_idx]))
            vb_cy = d_v[d_idx] + (d_wz[d_idx] * (rc_x - d_x[d_idx]) -
                                  d_wx[d_idx] * (rc_z - d_z[d_idx]))
            vb_cz = d_w[d_idx] + (d_wx[d_idx] * (rc_y - d_y[d_idx]) -
                                  d_wy[d_idx] * (rc_x - d_x[d_idx]))

            # velocity of particle d_idx at contact point (call it va_c)
            va_cx = d_u[sidx] + (d_wy[sidx] * (rc_z - d_z[sidx]) - d_wz[sidx] *
                                 (rc_y - d_y[sidx]))
            va_cy = d_v[sidx] + (d_wz[sidx] * (rc_x - d_x[sidx]) - d_wx[sidx] *
                                 (rc_z - d_z[sidx]))
            va_cz = d_w[sidx] + (d_wx[sidx] * (rc_y - d_y[sidx]) - d_wy[sidx] *
                                 (rc_x - d_x[sidx]))

            # relative velocity due to both linear and angular velocities
            vr_x = vb_cx - va_cx
            vr_y = vb_cy - va_cy
            vr_z = vb_cz - va_cz

            vr_dot_n = vr_x * nji_x + vr_y * nji_y + vr_z * nji_z

            # relative velocity in normal direction
            vr_nx = vr_dot_n * nji_x
            vr_ny = vr_dot_n * nji_y
            vr_nz = vr_dot_n * nji_z

            # relative velocity in tangential direction
            vr_tx = vr_x - vr_nx
            vr_ty = vr_y - vr_ny
            vr_tz = vr_z - vr_nz

            # relative shear displacement increment vector
            dtb2 = dt / 2.
            du_sx = vr_tx * dtb2
            du_sy = vr_ty * dtb2
            du_sz = vr_tz * dtb2

            # compute the geometric properties of the bond
            r_eff = self.lmbda * min(d_rad_s[d_idx], d_rad_s[sidx])
            if self.dim == 2:
                A = 2. * r_eff
                I = 2. / 3. * r_eff**3.
                J = 0.
            else:
                A = pi * r_eff**2.
                I = 0.25 * pi * r_eff**4.
                J = 0.5 * pi * r_eff**4.

            # THIS IS EXPERIMENTAL
            # relative normal displacement increment vector
            du_nx = vr_nx * dtb2
            du_ny = vr_ny * dtb2
            du_nz = vr_nz * dtb2

            # add the increment in the normal force due to relative
            # displacement in the normal direction.
            d_bc_fn_x[i] = d_bc_fn_x[i] - self.kn * A * du_nx
            d_bc_fn_y[i] = d_bc_fn_y[i] - self.kn * A * du_ny
            d_bc_fn_z[i] = d_bc_fn_z[i] - self.kn * A * du_nz

            # rotate the shear forces to the current plane
            # shear force in normal direction
            bc_fs_dot_n = (d_bc_fs_x[i] * nji_x + d_bc_fs_y[i] * nji_y +
                           d_bc_fs_z[i] * nji_z)

            # the rotated old shear force
            fs_rot_x = d_bc_fs_x[i] - bc_fs_dot_n * nji_x
            fs_rot_y = d_bc_fs_y[i] - bc_fs_dot_n * nji_y
            fs_rot_z = d_bc_fs_z[i] - bc_fs_dot_n * nji_z

            # now add the contribution of shear displacement to the
            # shear contact forces after rotation

            d_bc_fs_x[i] = fs_rot_x - self.ks * A * du_sx
            d_bc_fs_y[i] = fs_rot_y - self.ks * A * du_sy
            d_bc_fs_z[i] = fs_rot_z - self.ks * A * du_sz

            # now add this force to the global force of particle d_idx
            d_fx[d_idx] += d_bc_fs_x[i] + d_bc_fn_x[i]
            d_fy[d_idx] += d_bc_fs_y[i] + d_bc_fn_y[i]
            d_fz[d_idx] += d_bc_fs_z[i] + d_bc_fn_z[i]

            # add the moment due to the shear force
            d_torx[d_idx] += ((rc_y - d_y[d_idx]) * d_bc_fs_z[i] -
                              (rc_z - d_z[d_idx]) * d_bc_fs_y[i])
            d_tory[d_idx] += ((rc_z - d_z[d_idx]) * d_bc_fs_x[i] -
                              (rc_x - d_x[d_idx]) * d_bc_fs_z[i])
            d_torz[d_idx] += ((rc_x - d_x[d_idx]) * d_bc_fs_y[i] -
                              (rc_y - d_y[d_idx]) * d_bc_fs_x[i])

            # find the incremental moments
            # relative rotation displacement increment vector
            dtb2 = dt / 2.
            dtheta_x = (d_wx[d_idx] - d_wx[sidx]) * dtb2
            dtheta_y = (d_wy[d_idx] - d_wy[sidx]) * dtb2
            dtheta_z = (d_wz[d_idx] - d_wz[sidx]) * dtb2

            # resolve the rotation increment into normal direction
            dtheta_dot_n = (
                dtheta_x * nji_x + dtheta_y * nji_y + dtheta_z * nji_z)
            dtheta_nx = dtheta_dot_n * nji_x
            dtheta_ny = dtheta_dot_n * nji_y
            dtheta_nz = dtheta_dot_n * nji_z

            # resolve the rotation increment into tangential direction
            dtheta_sx = dtheta_x - dtheta_nx
            dtheta_sy = dtheta_y - dtheta_ny
            dtheta_sz = dtheta_z - dtheta_nz

            # using the net increment of the rotation, increment the contact
            # moment
            d_bc_mn_x[i] = d_bc_mn_x[i] - self.kn * J * dtheta_nx
            d_bc_mn_y[i] = d_bc_mn_y[i] - self.kn * J * dtheta_ny
            d_bc_mn_z[i] = d_bc_mn_z[i] - self.kn * J * dtheta_nz

            d_bc_ms_x[i] = d_bc_ms_x[i] - self.ks * I * dtheta_sx
            d_bc_ms_y[i] = d_bc_ms_y[i] - self.ks * I * dtheta_sy
            d_bc_ms_z[i] = d_bc_ms_z[i] - self.ks * I * dtheta_sz

            d_torx[d_idx] += d_bc_mn_x[i] + d_bc_ms_x[i]
            d_tory[d_idx] += d_bc_mn_y[i] + d_bc_ms_y[i]
            d_torz[d_idx] += d_bc_mn_z[i] + d_bc_ms_z[i]


class Potyondy3dIPForceStage2(Equation):
    def __init__(self, dest, sources, kn, dim, lmbda=1.):
        self.kn = kn
        self.ks = kn / 2.
        self.lmbda = lmbda
        self.dim = dim
        super(Potyondy3dIPForceStage2, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_contacts, d_x, d_y, d_z, d_bc_limit,
                   d_bc_idx, d_bc_rest_len, d_fx, d_fy, d_fz, d_torx, d_tory,
                   d_torz, d_u, d_v, d_w, d_wx, d_wy, d_wz, d_rad_s, d_bc_fs_x,
                   d_bc_fs_y, d_bc_fs_z, d_bc_fs0_x, d_bc_fs0_y, d_bc_fs0_z,
                   d_bc_fn_x, d_bc_fn_y, d_bc_fn_z, d_bc_fn0_x, d_bc_fn0_y,
                   d_bc_fn0_z, d_bc_mn_x, d_bc_mn_y, d_bc_mn_z, d_bc_ms_x,
                   d_bc_ms_y, d_bc_ms_z, d_bc_mn0_x, d_bc_mn0_y, d_bc_mn0_z,
                   d_bc_ms0_x, d_bc_ms0_y, d_bc_ms0_z, dt):
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
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])

            # normal vector from sidx to d_idx
            nji_x = xij[0] / rij
            nji_y = xij[1] / rij
            nji_z = xij[2] / rij

            # overlap positive for tension, negative for compression
            overlap = d_rad_s[d_idx] + d_rad_s[sidx] - rij

            # find the contact point of contacting bodies
            tmp = d_rad_s[d_idx] + overlap / 2.

            rc_x = d_x[d_idx] - nji_x * tmp
            rc_y = d_y[d_idx] - nji_y * tmp
            rc_z = d_z[d_idx] - nji_z * tmp

            # velocity of particle d_idx at contact point (call it vb_c)
            vb_cx = d_u[d_idx] + (d_wy[d_idx] * (rc_z - d_z[d_idx]) -
                                  d_wz[d_idx] * (rc_y - d_y[d_idx]))
            vb_cy = d_v[d_idx] + (d_wz[d_idx] * (rc_x - d_x[d_idx]) -
                                  d_wx[d_idx] * (rc_z - d_z[d_idx]))
            vb_cz = d_w[d_idx] + (d_wx[d_idx] * (rc_y - d_y[d_idx]) -
                                  d_wy[d_idx] * (rc_x - d_x[d_idx]))

            # velocity of particle d_idx at contact point (call it va_c)
            va_cx = d_u[sidx] + (d_wy[sidx] * (rc_z - d_z[sidx]) - d_wz[sidx] *
                                 (rc_y - d_y[sidx]))
            va_cy = d_v[sidx] + (d_wz[sidx] * (rc_x - d_x[sidx]) - d_wx[sidx] *
                                 (rc_z - d_z[sidx]))
            va_cz = d_w[sidx] + (d_wx[sidx] * (rc_y - d_y[sidx]) - d_wy[sidx] *
                                 (rc_x - d_x[sidx]))

            # relative velocity due to both linear and angular velocities
            vr_x = vb_cx - va_cx
            vr_y = vb_cy - va_cy
            vr_z = vb_cz - va_cz

            vr_dot_n = vr_x * nji_x + vr_y * nji_y + vr_z * nji_z

            # relative velocity in normal direction
            vr_nx = vr_dot_n * nji_x
            vr_ny = vr_dot_n * nji_y
            vr_nz = vr_dot_n * nji_z

            # relative velocity in tangential direction
            vr_tx = vr_x - vr_nx
            vr_ty = vr_y - vr_ny
            vr_tz = vr_z - vr_nz

            # compute the geometric properties of the bond
            r_eff = self.lmbda * min(d_rad_s[d_idx], d_rad_s[sidx])
            if self.dim == 2:
                A = 2. * r_eff
                I = 2. / 3. * r_eff**3.
                J = 0.
            else:
                A = pi * r_eff**2.
                I = 0.25 * pi * r_eff**4.
                J = 0.5 * pi * r_eff**4.

            # THIS IS EXPERIMENTAL
            # relative normal displacement increment vector
            du_nx = vr_nx * dt
            du_ny = vr_ny * dt
            du_nz = vr_nz * dt

            # add the increment in the normal force due to relative
            # displacement in the normal direction.
            d_bc_fn_x[i] = d_bc_fn0_x[i] - self.kn * A * du_nx
            d_bc_fn_y[i] = d_bc_fn0_y[i] - self.kn * A * du_ny
            d_bc_fn_z[i] = d_bc_fn0_z[i] - self.kn * A * du_nz

            # rotate the shear forces to the current plane
            # shear force in normal direction
            bc_fs_dot_n = (d_bc_fs0_x[i] * nji_x + d_bc_fs0_y[i] * nji_y +
                           d_bc_fs0_z[i] * nji_z)

            # the rotated old shear force
            fs_rot_x = d_bc_fs0_x[i] - bc_fs_dot_n * nji_x
            fs_rot_y = d_bc_fs0_y[i] - bc_fs_dot_n * nji_y
            fs_rot_z = d_bc_fs0_z[i] - bc_fs_dot_n * nji_z

            # relative shear displacement increment vector
            du_sx = vr_tx * dt
            du_sy = vr_ty * dt
            du_sz = vr_tz * dt

            # now add the contribution of shear displacement to the
            # shear contact forces after rotation

            d_bc_fs_x[i] = fs_rot_x - self.ks * A * du_sx
            d_bc_fs_y[i] = fs_rot_y - self.ks * A * du_sy
            d_bc_fs_z[i] = fs_rot_z - self.ks * A * du_sz

            # now add this force to the global force of particle d_idx
            d_fx[d_idx] += d_bc_fs_x[i] + d_bc_fn_x[i]
            d_fy[d_idx] += d_bc_fs_y[i] + d_bc_fn_y[i]
            d_fz[d_idx] += d_bc_fs_z[i] + d_bc_fn_z[i]

            # add the moment due to the shear force
            d_torx[d_idx] += ((rc_y - d_y[d_idx]) * d_bc_fs_z[i] -
                              (rc_z - d_z[d_idx]) * d_bc_fs_y[i])
            d_tory[d_idx] += ((rc_z - d_z[d_idx]) * d_bc_fs_x[i] -
                              (rc_x - d_x[d_idx]) * d_bc_fs_z[i])
            d_torz[d_idx] += ((rc_x - d_x[d_idx]) * d_bc_fs_y[i] -
                              (rc_y - d_y[d_idx]) * d_bc_fs_x[i])

            # find the incremental moments
            # relative rotation displacement increment vector
            dtheta_x = (d_wx[d_idx] - d_wx[sidx]) * dt
            dtheta_y = (d_wy[d_idx] - d_wy[sidx]) * dt
            dtheta_z = (d_wz[d_idx] - d_wz[sidx]) * dt

            # resolve the rotation increment into normal direction
            dtheta_dot_n = (
                dtheta_x * nji_x + dtheta_y * nji_y + dtheta_z * nji_z)
            dtheta_nx = dtheta_dot_n * nji_x
            dtheta_ny = dtheta_dot_n * nji_y
            dtheta_nz = dtheta_dot_n * nji_z

            # resolve the rotation increment into tangential direction
            dtheta_sx = dtheta_x - dtheta_nx
            dtheta_sy = dtheta_y - dtheta_ny
            dtheta_sz = dtheta_z - dtheta_nz

            # using the net increment of the rotation, increment the contact
            # moment
            d_bc_mn_x[i] = d_bc_mn0_x[i] - self.kn * J * dtheta_nx
            d_bc_mn_y[i] = d_bc_mn0_y[i] - self.kn * J * dtheta_ny
            d_bc_mn_z[i] = d_bc_mn0_z[i] - self.kn * J * dtheta_nz

            d_bc_ms_x[i] = d_bc_ms0_x[i] - self.ks * I * dtheta_sx
            d_bc_ms_y[i] = d_bc_ms0_y[i] - self.ks * I * dtheta_sy
            d_bc_ms_z[i] = d_bc_ms0_z[i] - self.ks * I * dtheta_sz

            d_torx[d_idx] += d_bc_mn_x[i] + d_bc_ms_x[i]
            d_tory[d_idx] += d_bc_mn_y[i] + d_bc_ms_y[i]
            d_torz[d_idx] += d_bc_mn_z[i] + d_bc_ms_z[i]


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
                   d_bc_total_contacts, d_bc_limit, d_bc_fs_x, d_bc_fs_y,
                   d_bc_fs_z, d_bc_fs0_x, d_bc_fs0_y, d_bc_fs0_z, d_bc_fn_x,
                   d_bc_fn_y, d_bc_fn_z, d_bc_fn0_x, d_bc_fn0_y, d_bc_fn0_z,
                   d_bc_ms_x, d_bc_ms_y, d_bc_ms_z, d_bc_ms0_x, d_bc_ms0_y,
                   d_bc_ms0_z, d_bc_mn_x, d_bc_mn_y, d_bc_mn_z, d_bc_mn0_x,
                   d_bc_mn0_y, d_bc_mn0_z):

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
            d_bc_fs0_x[i] = d_bc_fs_x[i]
            d_bc_fs0_y[i] = d_bc_fs_y[i]
            d_bc_fs0_z[i] = d_bc_fs_z[i]
            d_bc_fn0_x[i] = d_bc_fn_x[i]
            d_bc_fn0_y[i] = d_bc_fn_y[i]
            d_bc_fn0_z[i] = d_bc_fn_z[i]

            d_bc_ms0_x[i] = d_bc_ms_x[i]
            d_bc_ms0_y[i] = d_bc_ms_y[i]
            d_bc_ms0_z[i] = d_bc_ms_z[i]
            d_bc_mn0_x[i] = d_bc_mn_x[i]
            d_bc_mn0_y[i] = d_bc_mn_y[i]
            d_bc_mn0_z[i] = d_bc_mn_z[i]

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
