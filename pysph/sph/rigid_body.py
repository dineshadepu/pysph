# -*- coding: utf-8 -*-
"""Rigid body related equations.
"""
from pysph.base.reduce_array import parallel_reduce_array
from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
import numpy as np
import numpy
from math import sqrt, asin, sin, cos, pi, log
from pysph.sph.rigid_body_setup import (setup_quaternion_rigid_body,
                                        setup_rotation_matrix_rigid_body)


def skew(vec):
    import sympy as S
    x, y, z = vec[0], vec[1], vec[2]
    return S.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def get_alpha_dot():
    """Use sympy to perform most of the math and use the resulting formulae
    to calculate:

            inv(I) (\tau - w x (I w))
    """
    import sympy as S
    ixx, iyy, izz, ixy, ixz, iyz = S.symbols("ixx, iyy, izz, ixy, ixz, iyz")
    tx, ty, tz = S.symbols("tx, ty, tz")
    wx, wy, wz = S.symbols('wx, wy, wz')
    tau = S.Matrix([tx, ty, tz])
    I = S.Matrix([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    w = S.Matrix([wx, wy, wz])
    Iinv = I.inv()
    Iinv.simplify()
    # inv(I) (\tau - w x (Iw))
    res = Iinv*(tau - w.cross(I*w))
    res.simplify()
    # Now do some awesome sympy magic.
    syms, result = S.cse(res, symbols=S.numbered_symbols('tmp'))
    for lhs, rhs in syms:
        print("%s = %s" % (lhs, rhs))
    for i in range(3):
        print("omega_dot[%d] =" % i, result[0][i])


def get_torque():
    """Use sympy to perform some simple math.
        R x F
        C_m x F
        w x r
    """
    import sympy as S
    x, y, z, fx, fy, fz = S.symbols("x, y, z, fx, fy, fz")
    R = S.Matrix([x, y, z])
    F = S.Matrix([fx, fy, fz])
    print("Torque:", R.cross(F))
    cx, cy, cz = S.symbols('cx, cy, cz')
    d = S.Matrix([cx, cy, cz])
    print("c_m x f = ", d.cross(F))
    wx, wy, wz = S.symbols('wx, wy, wz')
    rx, ry, rz = S.symbols('rx, ry, rz')
    w = S.Matrix([wx, wy, wz])
    r = S.Matrix([rx, ry, rz])
    print("w x r = %s" % w.cross(r))


# This is defined to silence editor warnings for the use of declare.
def declare(*args): pass


class RigidBodyMoments(Equation):
    def reduce(self, dst, t, dt):
        # FIXME: this will be slow in opencl
        nbody = declare('int')
        i = declare('int')
        base_mi = declare('int')
        base = declare('int')
        nbody = dst.num_body[0]
        if dst.gpu:
            dst.gpu.pull('omega', 'x', 'y', 'z', 'fx', 'fy', 'fz')

        d_mi = declare('object')
        m = declare('object')
        x = declare('object')
        y = declare('object')
        z = declare('object')
        fx = declare('object')
        fy = declare('object')
        fz = declare('object')
        d_mi = dst.mi
        cond = declare('object')
        for i in range(nbody):
            cond = dst.body_id == i
            base = i*16
            m = dst.m[cond]
            x = dst.x[cond]
            y = dst.y[cond]
            z = dst.z[cond]
            # Find the total_mass, center of mass and second moments.
            d_mi[base + 0] = numpy.sum(m)
            d_mi[base + 1] = numpy.sum(m*x)
            d_mi[base + 2] = numpy.sum(m*y)
            d_mi[base + 3] = numpy.sum(m*z)
            # Only do the lower triangle of values moments of inertia.
            d_mi[base + 4] = numpy.sum(m*(y*y + z*z))
            d_mi[base + 5] = numpy.sum(m*(x*x + z*z))
            d_mi[base + 6] = numpy.sum(m*(x*x + y*y))

            d_mi[base + 7] = -numpy.sum(m*x*y)
            d_mi[base + 8] = -numpy.sum(m*x*z)
            d_mi[base + 9] = -numpy.sum(m*y*z)

            # the total force and torque
            fx = dst.fx[cond]
            fy = dst.fy[cond]
            fz = dst.fz[cond]
            d_mi[base + 10] = numpy.sum(fx)
            d_mi[base + 11] = numpy.sum(fy)
            d_mi[base + 12] = numpy.sum(fz)

            # Calculate the torque and reduce it.
            d_mi[base + 13] = numpy.sum(y*fz - z*fy)
            d_mi[base + 14] = numpy.sum(z*fx - x*fz)
            d_mi[base + 15] = numpy.sum(x*fy - y*fx)

        # Reduce the temporary mi values in parallel across processors.
        d_mi[:] = parallel_reduce_array(dst.mi)

        # Set the reduced values.
        for i in range(nbody):
            base_mi = i*16
            base = i*3
            m = d_mi[base_mi + 0]
            dst.total_mass[i] = m
            cx = d_mi[base_mi + 1]/m
            cy = d_mi[base_mi + 2]/m
            cz = d_mi[base_mi + 3]/m
            dst.cm[base + 0] = cx
            dst.cm[base + 1] = cy
            dst.cm[base + 2] = cz

            # The actual moment of inertia about center of mass from parallel
            # axes theorem.
            ixx = d_mi[base_mi + 4] - (cy*cy + cz*cz)*m
            iyy = d_mi[base_mi + 5] - (cx*cx + cz*cz)*m
            izz = d_mi[base_mi + 6] - (cx*cx + cy*cy)*m
            ixy = d_mi[base_mi + 7] + cx*cy*m
            ixz = d_mi[base_mi + 8] + cx*cz*m
            iyz = d_mi[base_mi + 9] + cy*cz*m

            d_mi[base_mi + 0] = ixx
            d_mi[base_mi + 1] = ixy
            d_mi[base_mi + 2] = ixz
            d_mi[base_mi + 3] = ixy
            d_mi[base_mi + 4] = iyy
            d_mi[base_mi + 5] = iyz
            d_mi[base_mi + 6] = ixz
            d_mi[base_mi + 7] = iyz
            d_mi[base_mi + 8] = izz

            fx = d_mi[base_mi + 10]
            fy = d_mi[base_mi + 11]
            fz = d_mi[base_mi + 12]
            dst.force[base + 0] = fx
            dst.force[base + 1] = fy
            dst.force[base + 2] = fz

            # Acceleration of CM.
            dst.ac[base + 0] = fx/m
            dst.ac[base + 1] = fy/m
            dst.ac[base + 2] = fz/m

            # Find torque about the Center of Mass and not origin.
            tx = d_mi[base_mi + 13]
            ty = d_mi[base_mi + 14]
            tz = d_mi[base_mi + 15]
            tx -= cy*fz - cz*fy
            ty -= -cx*fz + cz*fx
            tz -= cx*fy - cy*fx
            dst.torque[base + 0] = tx
            dst.torque[base + 1] = ty
            dst.torque[base + 2] = tz

            wx = dst.omega[base + 0]
            wy = dst.omega[base + 1]
            wz = dst.omega[base + 2]
            # Find omega_dot from: omega_dot = inv(I) (\tau - w x (Iw))
            # This was done using the sympy code above.
            tmp0 = iyz**2
            tmp1 = ixy**2
            tmp2 = ixz**2
            tmp3 = ixx*iyy
            tmp4 = ixy*ixz
            tmp5 = 1./(ixx*tmp0 + iyy*tmp2 - 2*iyz*tmp4 + izz*tmp1 - izz*tmp3)
            tmp6 = ixy*izz - ixz*iyz
            tmp7 = ixz*wx + iyz*wy + izz*wz
            tmp8 = ixx*wx + ixy*wy + ixz*wz
            tmp9 = tmp7*wx - tmp8*wz + ty
            tmp10 = ixy*iyz - ixz*iyy
            tmp11 = ixy*wx + iyy*wy + iyz*wz
            tmp12 = -tmp11*wx + tmp8*wy + tz
            tmp13 = tmp11*wz - tmp7*wy + tx
            tmp14 = ixx*iyz - tmp4
            dst.omega_dot[base + 0] = tmp5*(-tmp10*tmp12 -
                                            tmp13*(iyy*izz - tmp0) + tmp6*tmp9)
            dst.omega_dot[base + 1] = tmp5*(tmp12*tmp14 +
                                            tmp13*tmp6 - tmp9*(ixx*izz - tmp2))
            dst.omega_dot[base + 2] = tmp5*(-tmp10*tmp13 -
                                            tmp12*(-tmp1 + tmp3) + tmp14*tmp9)
        if dst.gpu:
            dst.gpu.push(
                'total_mass', 'mi', 'cm', 'force', 'ac', 'torque',
                'omega_dot'
            )


class RigidBodyMotion(Equation):
    def initialize(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w,
                   d_cm, d_vc, d_ac, d_omega, d_body_id):
        base = declare('int')
        base = d_body_id[d_idx]*3
        wx = d_omega[base + 0]
        wy = d_omega[base + 1]
        wz = d_omega[base + 2]
        rx = d_x[d_idx] - d_cm[base + 0]
        ry = d_y[d_idx] - d_cm[base + 1]
        rz = d_z[d_idx] - d_cm[base + 2]

        d_u[d_idx] = d_vc[base + 0] + wy*rz - wz*ry
        d_v[d_idx] = d_vc[base + 1] + wz*rx - wx*rz
        d_w[d_idx] = d_vc[base + 2] + wx*ry - wy*rx


class ResetForces(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_fz[d_idx] = 0.


class BodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(BodyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_fx, d_fy, d_fz):
        d_fx[d_idx] = d_m[d_idx]*self.gx
        d_fy[d_idx] = d_m[d_idx]*self.gy
        d_fz[d_idx] = d_m[d_idx]*self.gz


class SummationDensityBoundary(Equation):
    r"""Equation to find the density of the
    fluid particle due to any boundary or a rigid body

    :math:`\rho_a = \sum_b {\rho}_fluid V_b W_{ab}`

    """
    def __init__(self, dest, sources, fluid_rho=1000.0):
        self.fluid_rho = fluid_rho
        super(SummationDensityBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, s_idx, s_m, s_V, WIJ):
        d_rho[d_idx] += self.fluid_rho * s_V[s_idx] * WIJ


class NumberDensity(Equation):
    def initialize(self, d_idx, d_V):
        d_V[d_idx] = 0.0

    def loop(self, d_idx, d_V, WIJ):
        d_V[d_idx] += WIJ


class SummationDensityRigidBody(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(SummationDensityRigidBody, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, s_idx, s_V, WIJ):
        d_rho[d_idx] += self.rho0/s_V[s_idx]*WIJ


class ViscosityRigidBody(Equation):

    """The viscous acceleration on the fluid/solid due to a boundary.
    Implemented from Akinci et al. http://dx.doi.org/10.1145/2185520.2185558

    Use this with the fluid as a destination and body as source.
    """

    def __init__(self, dest, sources, rho0, nu):
        self.nu = nu
        self.rho0 = rho0
        super(ViscosityRigidBody, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_au, d_av, d_aw, d_rho,
             s_idx, s_V, s_fx, s_fy, s_fz,
             EPS, VIJ, XIJ, R2IJ, DWIJ):
        phi_b = self.rho0/(s_V[s_idx]*d_rho[d_idx])
        vijdotxij = min(VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2], 0.0)

        fac = self.nu*phi_b*vijdotxij/(R2IJ + EPS)
        ax = fac*DWIJ[0]
        ay = fac*DWIJ[1]
        az = fac*DWIJ[2]
        d_au[d_idx] += ax
        d_av[d_idx] += ay
        d_aw[d_idx] += az
        s_fx[s_idx] += -d_m[d_idx]*ax
        s_fy[s_idx] += -d_m[d_idx]*ay
        s_fz[s_idx] += -d_m[d_idx]*az


class PressureRigidBody(Equation):

    """The pressure acceleration on the fluid/solid due to a boundary.
    Implemented from Akinci et al. http://dx.doi.org/10.1145/2185520.2185558

    Use this with the fluid as a destination and body as source.
    """

    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(PressureRigidBody, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_rho, d_au, d_av, d_aw,  d_p,
             s_idx, s_V, s_fx, s_fy, s_fz, DWIJ):
        rho1 = 1.0/d_rho[d_idx]
        fac = -d_p[d_idx]*rho1*rho1*self.rho0/s_V[s_idx]
        ax = fac*DWIJ[0]
        ay = fac*DWIJ[1]
        az = fac*DWIJ[2]
        d_au[d_idx] += ax
        d_av[d_idx] += ay
        d_aw[d_idx] += az
        s_fx[s_idx] += -d_m[d_idx]*ax
        s_fy[s_idx] += -d_m[d_idx]*ay
        s_fz[s_idx] += -d_m[d_idx]*az


class AkinciRigidFluidCoupling(Equation):
    """Force between a solid sphere and a SPH fluid particle.  This is
    implemented using Akinci's[1] force and additional force from solid
    bodies pressure which is implemented by Liu[2]

    [1]'Versatile Rigid-Fluid Coupling for Incompressible SPH'

    URL: https://graphics.ethz.ch/~sobarbar/papers/Sol12/Sol12.pdf

    [2]A 3D Simulation of a Moving Solid in Viscous Free-Surface Flows by
    Coupling SPH and DEM

    https://doi.org/10.1155/2017/3174904


    Note: Here forces for both the phases are added at once.
          Please make sure that this force is applied only once
          for both the particle properties.

    """
    def __init__(self, dest, sources, fluid_rho=1000):
        super(AkinciRigidFluidCoupling, self).__init__(dest, sources)
        self.fluid_rho = fluid_rho

    def loop(self, d_idx, d_m, d_rho, d_au, d_av, d_aw,  d_p,
             s_idx, s_V, s_fx, s_fy, s_fz, DWIJ, s_m, s_p, s_rho):

        psi = s_V[s_idx] * self.fluid_rho

        _t1 = 2 * d_p[d_idx] / (d_rho[d_idx]**2)

        d_au[d_idx] += -psi * _t1 * DWIJ[0]
        d_av[d_idx] += -psi * _t1 * DWIJ[1]
        d_aw[d_idx] += -psi * _t1 * DWIJ[2]

        s_fx[s_idx] += d_m[d_idx] * psi * _t1 * DWIJ[0]
        s_fy[s_idx] += d_m[d_idx] * psi * _t1 * DWIJ[1]
        s_fz[s_idx] += d_m[d_idx] * psi * _t1 * DWIJ[2]


class LiuFluidForce(Equation):
    """Force between a solid sphere and a SPH fluid particle.  This is
    implemented using Akinci's[1] force and additional force from solid
    bodies pressure which is implemented by Liu[2]

    [1]'Versatile Rigid-Fluid Coupling for Incompressible SPH'

    URL: https://graphics.ethz.ch/~sobarbar/papers/Sol12/Sol12.pdf

    [2]A 3D Simulation of a Moving Solid in Viscous Free-Surface Flows by
    Coupling SPH and DEM

    https://doi.org/10.1155/2017/3174904


    Note: Here forces for both the phases are added at once.
          Please make sure that this force is applied only once
          for both the particle properties.

    """
    def __init__(self, dest, sources):
        super(LiuFluidForce, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_rho, d_au, d_av, d_aw,  d_p,
             s_idx, s_V, s_fx, s_fy, s_fz, DWIJ, s_m, s_p, s_rho):
        _t1 = s_p[s_idx] / (s_rho[s_idx]**2) + d_p[d_idx] / (d_rho[d_idx]**2)

        d_au[d_idx] += -s_m[s_idx] * _t1 * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * _t1 * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * _t1 * DWIJ[2]

        s_fx[s_idx] += d_m[d_idx] * s_m[s_idx] * _t1 * DWIJ[0]
        s_fy[s_idx] += d_m[d_idx] * s_m[s_idx] * _t1 * DWIJ[1]
        s_fz[s_idx] += d_m[d_idx] * s_m[s_idx] * _t1 * DWIJ[2]


class RigidBodyForceGPUGems(Equation):
    """This is inspired from
    http://http.developer.nvidia.com/GPUGems3/gpugems3_ch29.html
    and
    BK Mishra's article on DEM
    http://dx.doi.org/10.1016/S0301-7516(03)00032-2
    A review of computer simulation of tumbling mills by the discrete element
    method: Part I - contact mechanics
    """
    def __init__(self, dest, sources, k=1.0, d=1.0, eta=1.0, kt=1.0):
        """Note that d is a factor multiplied with the "h" of the particle.
        """
        self.k = k
        self.d = d
        self.eta = eta
        self.kt = kt
        super(RigidBodyForceGPUGems, self).__init__(dest, sources)

    def loop(self, d_idx, d_fx, d_fy, d_fz, d_h, d_total_mass, XIJ,
             RIJ, R2IJ, VIJ):
        vijdotrij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]
        if RIJ > 1e-9:
            vijdotrij_r2ij = vijdotrij/R2IJ
            nij_x = XIJ[0]/RIJ
            nij_y = XIJ[1]/RIJ
            nij_z = XIJ[2]/RIJ
        else:
            vijdotrij_r2ij = 0.0
            nij_x = 0.0
            nij_y = 0.0
            nij_z = 0.0
        vijt_x = VIJ[0] - vijdotrij_r2ij*XIJ[0]
        vijt_y = VIJ[1] - vijdotrij_r2ij*XIJ[1]
        vijt_z = VIJ[2] - vijdotrij_r2ij*XIJ[2]

        d = self.d*d_h[d_idx]
        fac = self.k*d_total_mass[0]/d*max(d - RIJ, 0.0)

        d_fx[d_idx] += fac*nij_x - self.eta*VIJ[0] - self.kt*vijt_x
        d_fy[d_idx] += fac*nij_y - self.eta*VIJ[1] - self.kt*vijt_y
        d_fz[d_idx] += fac*nij_z - self.eta*VIJ[2] - self.kt*vijt_z


class RigidBodyCollision(Equation):
    """Force between two spheres is implemented using DEM contact force law.
    Refer https://doi.org/10.1016/j.powtec.2011.09.019 for more
    information.
    Open-source MFIX-DEM software for gas–solids flows:
    Part I—Verification studies .
    """
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        """Initialise the required coefficients for force calculation.
        Keyword arguments:
        kn -- Normal spring stiffness (default 1e3)
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)
        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.
        """
        self.kn = kn
        self.kt = 2. / 7. * kn
        m_eff = np.pi * 0.5**2 * 1e-6 * 2120
        self.gamma_n = -(2 * np.sqrt(kn * m_eff) * np.log(en)) / (
            np.sqrt(np.pi**2 + np.log(en)**2))
        self.gamma_t = 0.5 * self.gamma_n
        self.mu = mu
        super(RigidBodyCollision, self).__init__(dest, sources)

    def loop(self, d_idx, d_fx, d_fy, d_fz, d_h, d_total_mass, d_rad_s,
             d_tang_disp_x, d_tang_disp_y, d_tang_disp_z, d_tang_velocity_x,
             d_tang_velocity_y, d_tang_velocity_z, s_idx, s_rad_s, XIJ, RIJ,
             R2IJ, VIJ):
        overlap = 0
        if RIJ > 1e-9:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        if overlap > 0:
            # normal vector passing from particle i to j
            nij_x = -XIJ[0] / RIJ
            nij_y = -XIJ[1] / RIJ
            nij_z = -XIJ[2] / RIJ

            # overlap speed: a scalar
            vijdotnij = VIJ[0] * nij_x + VIJ[1] * nij_y + VIJ[2] * nij_z

            # normal velocity
            vijn_x = vijdotnij * nij_x
            vijn_y = vijdotnij * nij_y
            vijn_z = vijdotnij * nij_z

            # normal force with conservative and dissipation part
            fn_x = -self.kn * overlap * nij_x - self.gamma_n * vijn_x
            fn_y = -self.kn * overlap * nij_y - self.gamma_n * vijn_y
            fn_z = -self.kn * overlap * nij_z - self.gamma_n * vijn_z

            # ----------------------Tangential force---------------------- #

            # tangential velocity
            d_tang_velocity_x[d_idx] = VIJ[0] - vijn_x
            d_tang_velocity_y[d_idx] = VIJ[1] - vijn_y
            d_tang_velocity_z[d_idx] = VIJ[2] - vijn_z

            dtvx = d_tang_velocity_x[d_idx]
            dtvy = d_tang_velocity_y[d_idx]
            dtvz = d_tang_velocity_z[d_idx]
            _tang = sqrt(dtvx*dtvx + dtvy*dtvy + dtvz*dtvz)

            # tangential unit vector
            tij_x = 0
            tij_y = 0
            tij_z = 0
            if _tang > 0:
                tij_x = d_tang_velocity_x[d_idx] / _tang
                tij_y = d_tang_velocity_y[d_idx] / _tang
                tij_z = d_tang_velocity_z[d_idx] / _tang

            # damping force or dissipation
            ft_x_d = -self.gamma_t * d_tang_velocity_x[d_idx]
            ft_y_d = -self.gamma_t * d_tang_velocity_y[d_idx]
            ft_z_d = -self.gamma_t * d_tang_velocity_z[d_idx]

            # tangential spring force
            ft_x_s = -self.kt * d_tang_disp_x[d_idx]
            ft_y_s = -self.kt * d_tang_disp_y[d_idx]
            ft_z_s = -self.kt * d_tang_disp_z[d_idx]

            ft_x = ft_x_d + ft_x_s
            ft_y = ft_y_d + ft_y_s
            ft_z = ft_z_d + ft_z_s

            # coulomb law
            ftij = sqrt((ft_x**2) + (ft_y**2) + (ft_z**2))
            fnij = sqrt((fn_x**2) + (fn_y**2) + (fn_z**2))

            _fnij = self.mu * fnij

            if _fnij < ftij:
                ft_x = -_fnij * tij_x
                ft_y = -_fnij * tij_y
                ft_z = -_fnij * tij_z

            d_fx[d_idx] += fn_x + ft_x
            d_fy[d_idx] += fn_y + ft_y
            d_fz[d_idx] += fn_z + ft_z
        else:
            d_tang_velocity_x[d_idx] = 0
            d_tang_velocity_y[d_idx] = 0
            d_tang_velocity_z[d_idx] = 0

            d_tang_disp_x[d_idx] = 0
            d_tang_disp_y[d_idx] = 0
            d_tang_disp_z[d_idx] = 0


class RigidBodyCollisionStage1(Equation):
    """Force between two spheres is implemented using DEM contact force law.

    The force is modelled mostly from reference [1]. Some ideas are taken from
    reference [2] and added it to the implementation of reference [1] to
    make it complete.

    [1] Multi-level modelling of dense gas-solid two phase flows. PhD Thesis,
    Ye.
    [2] Introduction to discrete element methods. Stefan Luding
    https://pdfs.semanticscholar.org/ec8b/26e2d20b7d8a09d15f0e10660f0d02b4458d.pdf

    """

    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        """the required coefficients for force calculation.


        Keyword arguments:
        kn -- Normal spring stiffness (default 1e3)
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)

        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.

        """
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        super(RigidBodyCollisionStage1, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_fx, d_fy, d_fz,
             d_tng_x, d_tng_y, d_tng_z, d_tng_x0, d_tng_y0,
             d_tng_z0, d_tng_idx, d_tng_idx_dem_id, d_total_tng_contacts,
             d_dem_id, d_limit, d_vtx, d_vty, d_vtz, d_tng_nx, d_tng_ny,
             d_tng_nz, d_tng_nx0, d_tng_ny0, d_tng_nz0,
             VIJ, XIJ, RIJ, d_rad_s, s_idx, s_m, s_rad_s, s_dem_id,
             dt):
        p, q1, tot_ctcs, i, found_at, found = declare('int', 6)

        if d_dem_id[d_idx] != s_dem_id[s_idx]:
            overlap = -1.
            # check the particles are not on top of each other.
            if RIJ > 0:
                overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

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
            for i in range(p, q1):
                if s_idx == d_tng_idx[i]:
                    if s_dem_id[s_idx] == d_tng_idx_dem_id[i]:
                        found_at = i
                        found = 1
                        break

            # ---------- force computation starts ------------
            # if particles are overlapping
            if overlap > 0:
                # equation 2.8
                # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
                rinv = 1.0 / RIJ
                # in PySPH: XIJ[0] = d_x[d_idx] - s_x[s_idx]
                nxc = -XIJ[0] * rinv
                nyc = -XIJ[1] * rinv
                nzc = -XIJ[2] * rinv

                # ---- Relative velocity computation (Eq 2.9) ----
                # relative velocity of particle d_idx w.r.t particle s_idx at
                # contact point. The velocity difference provided by PySPH is
                # only between translational velocities. We don't need to
                # consider rotational velocities

                vr_x = VIJ[0]
                vr_y = VIJ[1]
                vr_z = VIJ[2]

                # normal velocity magnitude
                vr_dot_nij = vr_x * nxc + vr_y * nyc + vr_z * nzc
                vn_x = vr_dot_nij * nxc
                vn_y = vr_dot_nij * nyc
                vn_z = vr_dot_nij * nzc

                # tangential velocity
                vt_x = vr_x - vn_x
                vt_y = vr_y - vn_y
                vt_z = vr_z - vn_z
                # magnitude of the tangential velocity
                # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

                # compute the damping constants
                m_eff = d_m[d_idx] * s_m[s_idx] / (d_m[d_idx] + s_m[s_idx])
                log_en = log(self.en)
                eta_n = -2. * log_en * sqrt(m_eff * self.kn) / sqrt(
                    pi**2. + log_en**2.)

                # normal force
                kn_overlap = self.kn * overlap
                fn_x = -kn_overlap * nxc - eta_n * vn_x
                fn_y = -kn_overlap * nyc - eta_n * vn_y
                fn_z = -kn_overlap * nzc - eta_n * vn_z

                # ------------- tangential force computation ----------------
                # if the particle is not been tracked then assign an index in
                # tracking history.
                if found == 0:
                    found_at = q1
                    d_tng_idx[found_at] = s_idx
                    d_total_tng_contacts[d_idx] += 1
                    d_tng_idx_dem_id[found_at] = s_dem_id[s_idx]
                    d_tng_nx[found_at] = nxc
                    d_tng_ny[found_at] = nyc
                    d_tng_nz[found_at] = nzc
                    d_tng_nx0[found_at] = nxc
                    d_tng_ny0[found_at] = nyc
                    d_tng_nz0[found_at] = nzc

                # compute and set the tangential acceleration for the
                # current time step
                d_vtx[found_at] = vt_x
                d_vty[found_at] = vt_y
                d_vtz[found_at] = vt_z

                # compute the damping constants
                m_eff_t = 2. / 7. * m_eff
                log_et = log(self.et)
                eta_t = -2. * log_en * sqrt(m_eff_t * self.kn) / sqrt(
                    pi**2. + log_et**2.)

                # find the tangential force from the tangential displacement
                # and tangential velocity (eq 2.11 Thesis Ye)
                ft0_x = -self.kt * d_tng_x[found_at] - eta_t * vt_x
                ft0_y = -self.kt * d_tng_y[found_at] - eta_t * vt_y
                ft0_z = -self.kt * d_tng_z[found_at] - eta_t * vt_z

                # (*) check against Coulomb criterion
                # Tangential force magnitude due to displacement
                ft0_magn = (ft0_x * ft0_x + ft0_y * ft0_y + ft0_z * ft0_z)**(
                    0.5)
                fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

                # we have to compare with static friction, so
                # this mu has to be static friction coefficient
                fn_mu = self.mu * fn_magn

                # if the tangential force magnitude is zero, then do nothing,
                # else do following
                if ft0_magn != 0.:
                    # compare tangential force with the static friction
                    if ft0_magn >= fn_mu:
                        # rescale the tangential displacement
                        # find the unit direction in tangential velocity
                        # TODO: ELIMINATE THE SINGULARITY CASE
                        tx = ft0_x / ft0_magn
                        ty = ft0_y / ft0_magn
                        tz = ft0_z / ft0_magn
                        # this taken from Luding paper [2], eq (21)
                        d_tng_x[found_at] = -self.kt_1 * (
                            fn_mu * tx + eta_t * vt_x)
                        d_tng_y[found_at] = -self.kt_1 * (
                            fn_mu * ty + eta_t * vt_y)
                        d_tng_z[found_at] = -self.kt_1 * (
                            fn_mu * tz + eta_t * vt_z)

                        # and also adjust the spring elongation
                        # at time t, which is used at stage 2 integrator
                        d_tng_x0[found_at] = d_tng_x[found_at]
                        d_tng_y0[found_at] = d_tng_y[found_at]
                        d_tng_z0[found_at] = d_tng_z[found_at]

                        # set the tangential force to static friction
                        # from Coulomb criterion
                        ft0_x = fn_mu * tx
                        ft0_y = fn_mu * ty
                        ft0_z = fn_mu * tz

                d_fx[d_idx] += fn_x + ft0_x
                d_fy[d_idx] += fn_y + ft0_y
                d_fz[d_idx] += fn_z + ft0_z


class RigidBodyCollisionStage2(Equation):
    """Force between two spheres is implemented using DEM contact force law.

    The force is modelled mostly from reference [1]. Some ideas are taken from
    reference [2] and added it to the implementation of reference [1] to
    make it complete.

    [1] Multi-level modelling of dense gas-solid two phase flows. PhD Thesis,
    Ye.
    [2] Introduction to discrete element methods. Stefan Luding
    https://pdfs.semanticscholar.org/ec8b/26e2d20b7d8a09d15f0e10660f0d02b4458d.pdf

    """

    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        """the required coefficients for force calculation.


        Keyword arguments:
        kn -- Normal spring stiffness (default 1e3)
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)

        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.

        """
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        super(RigidBodyCollisionStage2, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_fx, d_fy, d_fz,
             d_tng_x, d_tng_y, d_tng_z, d_tng_x0, d_tng_y0,
             d_tng_z0, d_tng_idx, d_tng_idx_dem_id, d_total_tng_contacts,
             d_dem_id, d_limit, d_vtx, d_vty, d_vtz, d_tng_nx, d_tng_ny,
             d_tng_nz, d_tng_nx0, d_tng_ny0, d_tng_nz0,
             VIJ, XIJ, RIJ, d_rad_s, s_idx, s_rad_s, s_m,
             s_dem_id, dt):
        p, q1, tot_ctcs, i, found_at, found = declare('int', 6)
        if d_dem_id[d_idx] != s_dem_id[d_idx]:
            overlap = -1.
            # check the particles are not on top of each other.
            if RIJ > 0:
                overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

            # total number of contacts of particle i in destination
            tot_ctcs = d_total_tng_contacts[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_limit[0]
            # ending index is q1 -1
            q1 = p + tot_ctcs

            # check if the particle is in the tracking list
            # if so, then save the location at found_at
            found = 0
            for i in range(p, q1):
                if s_idx == d_tng_idx[i]:
                    if s_dem_id[s_idx] == d_tng_idx_dem_id[i]:
                        found_at = i
                        found = 1
                        break

            # ---------- force computation starts ------------
            # if particles are not overlapping
            if overlap <= 0:
                if found == 1:
                    # make its tangential displacement to be zero
                    d_tng_x[found_at] = 0.
                    d_tng_y[found_at] = 0.
                    d_tng_z[found_at] = 0.

                    d_tng_nx[found_at] = 0.
                    d_tng_ny[found_at] = 0.
                    d_tng_nz[found_at] = 0.

                    d_vtx[found_at] = 0.
                    d_vty[found_at] = 0.
                    d_vtz[found_at] = 0.

            # if particles are in contact
            else:
                # equation 2.8
                # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
                rinv = 1.0 / RIJ
                # in PySPH: XIJ[0] = d_x[d_idx] - s_x[s_idx]
                nxc = -XIJ[0] * rinv
                nyc = -XIJ[1] * rinv
                nzc = -XIJ[2] * rinv

                # ---- Relative velocity computation (Eq 2.9) ----
                # relative velocity of particle d_idx w.r.t particle s_idx at
                # contact point. The velocity difference provided by PySPH is
                # only between translational velocities. We don' need angular
                # velocity since particles are rigid and won't rotate.
                vr_x = VIJ[0]
                vr_y = VIJ[1]
                vr_z = VIJ[2]

                # normal velocity magnitude
                vr_dot_nij = vr_x * nxc + vr_y * nyc + vr_z * nzc
                vn_x = vr_dot_nij * nxc
                vn_y = vr_dot_nij * nyc
                vn_z = vr_dot_nij * nzc

                # tangential velocity
                vt_x = vr_x - vn_x
                vt_y = vr_y - vn_y
                vt_z = vr_z - vn_z

                # compute the damping constants
                m_eff = d_m[d_idx] * s_m[s_idx] / (d_m[d_idx] + s_m[s_idx])
                log_en = log(self.en)
                eta_n = -2. * log_en * sqrt(m_eff * self.kn) / sqrt(
                    pi**2. + log_en**2.)

                # normal force
                kn_overlap = self.kn * overlap
                fn_x = -kn_overlap * nxc - eta_n * vn_x
                fn_y = -kn_overlap * nyc - eta_n * vn_y
                fn_z = -kn_overlap * nzc - eta_n * vn_z

                # ------------- tangential force computation ----------------
                # do not add new particles to the contact list at step
                # t + dt / 2. But normal force will be computed as above.

                # Tangential force is computed if the particle is been tracked
                # already
                if found == 1:
                    # current normal to the plane is nx, ny, nz
                    # the tangential spring is oriented normal to
                    # nxp, nyp, nzp (p is previous)
                    nxp = d_tng_nx[found_at]
                    nyp = d_tng_ny[found_at]
                    nzp = d_tng_nz[found_at]
                    # in order to compute the tangential force
                    # rotate the spring for current plane
                    # -------------------------
                    # rotation of the spring
                    # -------------------------
                    # rotation matrix
                    # n_current  \cross n_previous
                    tmpx = nyc * nzp - nzc * nyp
                    tmpy = nzc * nxp - nxc * nzp
                    tmpz = nxc * nyp - nyc * nxp
                    tmp_magn = (tmpx**2. + tmpy**2. + tmpz**2.)**0.5
                    # normalized rotation vector
                    hx = tmpx / tmp_magn
                    hy = tmpy / tmp_magn
                    hz = tmpz / tmp_magn

                    phi = asin(tmp_magn)
                    c = cos(phi)
                    s = sin(phi)
                    q = 1. - c

                    # matrix corresponding to the rotation vector
                    H0 = q * hx**2. + c
                    H1 = q * hx * hy - s * hz
                    H2 = q * hx * hz + s * hy

                    H3 = q * hy * hx + s * hz
                    H4 = q * hy**2. + c
                    H5 = q * hy * hz - s * hx

                    H6 = q * hz * hx - s * hy
                    H7 = q * hz * hy + s * hx
                    H8 = q * hz**2. + c

                    # save the tangential displacement temporarily
                    # will be used while rotation
                    tmpx = d_tng_x[found_at]
                    tmpy = d_tng_y[found_at]
                    tmpz = d_tng_z[found_at]

                    d_tng_x[found_at] = H0 * tmpx + H1 * tmpy + H2 * tmpz
                    d_tng_y[found_at] = H3 * tmpx + H4 * tmpy + H5 * tmpz
                    d_tng_z[found_at] = H6 * tmpx + H7 * tmpy + H8 * tmpz

                    # save the current normal of the spring
                    d_tng_nx[found_at] = nxc
                    d_tng_ny[found_at] = nyc
                    d_tng_nz[found_at] = nzc

                    # --------------------------------------
                    # similarly rotate the tangential spring at time t
                    # --------------------------------------
                    # current normal to the plane is nx, ny, nz
                    # the tangential spring is oriented normal to
                    # nxp, nyp, nzp
                    nxp = d_tng_nx0[found_at]
                    nyp = d_tng_ny0[found_at]
                    nzp = d_tng_nz0[found_at]
                    # in order to compute the tangential force
                    # rotate the spring for current plane
                    # -------------------------
                    # rotation of the spring
                    # -------------------------
                    # rotation matrix
                    # n_current  \cross n_previous
                    tmpx = nyc * nzp - nzc * nyp
                    tmpy = nzc * nxp - nxc * nzp
                    tmpz = nxc * nyp - nyc * nxp
                    tmp_magn = (tmpx**2. + tmpy**2. + tmpz**2.)**0.5
                    # normalized rotation vector
                    hx = tmpx / tmp_magn
                    hy = tmpy / tmp_magn
                    hz = tmpz / tmp_magn

                    phi = asin(tmp_magn)
                    c = cos(phi)
                    s = sin(phi)
                    q = 1. - c

                    # matrix corresponding to the rotation vector
                    H0 = q * hx**2. + c
                    H1 = q * hx * hy - s * hz
                    H2 = q * hx * hz + s * hy

                    H3 = q * hy * hx + s * hz
                    H4 = q * hy**2. + c
                    H5 = q * hy * hz - s * hx

                    H6 = q * hz * hx - s * hy
                    H7 = q * hz * hy + s * hx
                    H8 = q * hz**2. + c

                    # save the tangential displacement temporarily
                    # will be used while rotation
                    tmpx = d_tng_x0[found_at]
                    tmpy = d_tng_y0[found_at]
                    tmpz = d_tng_z0[found_at]

                    d_tng_x0[found_at] = H0 * tmpx + H1 * tmpy + H2 * tmpz
                    d_tng_y0[found_at] = H3 * tmpx + H4 * tmpy + H5 * tmpz
                    d_tng_z0[found_at] = H6 * tmpx + H7 * tmpy + H8 * tmpz

                    # save the current normal of the spring
                    d_tng_nx0[found_at] = nxc
                    d_tng_ny0[found_at] = nyc
                    d_tng_nz0[found_at] = nzc

                    # compute and set the tangential acceleration for the
                    # current time step
                    d_vtx[found_at] = vt_x
                    d_vty[found_at] = vt_y
                    d_vtz[found_at] = vt_z

                # compute the damping constants
                m_eff_t = 2. / 7. * m_eff
                log_et = log(self.et)
                eta_t = -2. * log_en * sqrt(m_eff_t * self.kn) / sqrt(
                    pi**2. + log_et**2.)

                # find the tangential force from the tangential displacement
                # and tangential velocity (eq 2.11 Thesis Ye)
                ft0_x = -self.kt * d_tng_x[found_at] - eta_t * vt_x
                ft0_y = -self.kt * d_tng_y[found_at] - eta_t * vt_y
                ft0_z = -self.kt * d_tng_z[found_at] - eta_t * vt_z

                # (*) check against Coulomb criterion
                # Tangential force magnitude due to displacement
                ft0_magn = (ft0_x * ft0_x + ft0_y * ft0_y + ft0_z * ft0_z)**(
                    0.5)
                fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

                # we have to compare with static friction, so
                # this mu has to be static friction coefficient
                fn_mu = self.mu * fn_magn

                # if the tangential force magnitude is zero, then do nothing,
                # else do following
                if ft0_magn != 0.:
                    # compare tangential force with the static friction
                    if ft0_magn >= fn_mu:
                        # rescale the tangential displacement
                        tx = ft0_x / ft0_magn
                        ty = ft0_y / ft0_magn
                        tz = ft0_z / ft0_magn
                        d_tng_x[found_at] = -self.kt_1 * (
                            fn_mu * tx + eta_t * vt_x)
                        d_tng_y[found_at] = -self.kt_1 * (
                            fn_mu * ty + eta_t * vt_y)
                        d_tng_z[found_at] = -self.kt_1 * (
                            fn_mu * tz + eta_t * vt_z)

                        # set the tangential force to static friction
                        # from Coulomb criterion
                        ft0_x = fn_mu * tx
                        ft0_y = fn_mu * ty
                        ft0_z = fn_mu * tz

                d_fx[d_idx] += fn_x + ft0_x
                d_fy[d_idx] += fn_y + ft0_y
                d_fz[d_idx] += fn_z + ft0_z


class UpdateTangentialContacts(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_z, d_rad_s,
                        d_total_tng_contacts, d_tng_idx, d_limit, d_tng_x,
                        d_tng_y, d_tng_z, d_tng_nx, d_tng_ny, d_tng_nz,
                        d_vtx, d_vty, d_vtz, d_tng_idx_dem_id, s_x,
                        s_y, s_z, s_rad_s, s_dem_id):
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
                    rinv = 1. / rij

                    overlap = d_rad_s[d_idx] + s_rad_s[sidx] - rij

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_tng_idx[k] = -1
                            d_tng_idx_dem_id[k] = -1
                            d_tng_x[k] = 0.
                            d_tng_y[k] = 0.
                            d_tng_z[k] = 0.
                            d_tng_nx[k] = 0.
                            d_tng_ny[k] = 0.
                            d_tng_nz[k] = 0.
                            d_vtx[k] = 0.
                            d_vty[k] = 0.
                            d_vtz[k] = 0.
                        else:
                            # swap the current tracking index with the final
                            # contact index
                            d_tng_idx[k] = d_tng_idx[last_idx_tmp]
                            d_tng_idx[last_idx_tmp] = -1

                            # swap tangential x displacement
                            d_tng_x[k] = d_tng_x[last_idx_tmp]
                            d_tng_x[last_idx_tmp] = 0.

                            # swap tangential y displacement
                            d_tng_y[k] = d_tng_x[last_idx_tmp]
                            d_tng_y[last_idx_tmp] = 0.

                            # swap tangential z displacement
                            d_tng_z[k] = d_tng_z[last_idx_tmp]
                            d_tng_z[last_idx_tmp] = 0.

                            # swap tangential nx orientation
                            d_tng_nx[k] = d_tng_nx[last_idx_tmp]
                            d_tng_nx[last_idx_tmp] = 0.

                            # swap tangential ny orientation
                            d_tng_ny[k] = d_tng_nx[last_idx_tmp]
                            d_tng_ny[last_idx_tmp] = 0.

                            # swap tangential nz orientation
                            d_tng_nz[k] = d_tng_nz[last_idx_tmp]
                            d_tng_nz[last_idx_tmp] = 0.

                            # swap tangential nx orientation
                            d_vtx[k] = d_vtx[last_idx_tmp]
                            d_vtx[last_idx_tmp] = 0.

                            # swap tangential ny orientation
                            d_vty[k] = d_vtx[last_idx_tmp]
                            d_vty[last_idx_tmp] = 0.

                            # swap tangential nz orientation
                            d_vtz[k] = d_vtz[last_idx_tmp]
                            d_vtz[last_idx_tmp] = 0.

                            # swap tangential idx dem id
                            d_tng_idx_dem_id[k] = d_tng_idx_dem_id[last_idx_tmp]
                            d_tng_idx_dem_id[last_idx_tmp] = -1

                            # decrease the last_idx_tmp, since we swapped it to
                            # -1
                            last_idx_tmp -= 1

                        # decrement the total contacts of the particle
                        d_total_tng_contacts[d_idx] -= 1
                    else:
                        # ----------------------------------------------------
                        # this implies that the particles are still in contact
                        # now rotate the tangential spring about the new plane
                        # ----------------------------------------------------
                        # current normal to the plane is nx, ny, nz
                        # the tangential spring is oriented normal to
                        # nxp, nyp, nzp
                        nxp = d_tng_nx[k]
                        nyp = d_tng_ny[k]
                        nzp = d_tng_nz[k]
                        # and current normal vector between the particles is
                        nxc = -xij[0] * rinv
                        nyc = -xij[1] * rinv
                        nzc = -xij[2] * rinv

                        # in order to compute the tangential force
                        # rotate the spring for current plane
                        # -------------------------
                        # rotation of the spring
                        # -------------------------
                        # rotation matrix
                        # n_current  \cross n_previous
                        tmpx = nyc * nzp - nzc * nyp
                        tmpy = nzc * nxp - nxc * nzp
                        tmpz = nxc * nyp - nyc * nxp
                        tmp_magn = (tmpx**2. + tmpy**2. + tmpz**2.)**0.5
                        # normalized rotation vector
                        hx = tmpx / tmp_magn
                        hy = tmpy / tmp_magn
                        hz = tmpz / tmp_magn

                        phi = asin(tmp_magn)
                        c = cos(phi)
                        s = sin(phi)
                        q = 1. - c

                        # matrix corresponding to the rotation vector
                        H0 = q * hx**2. + c
                        H1 = q * hx * hy - s * hz
                        H2 = q * hx * hz + s * hy

                        H3 = q * hy * hx + s * hz
                        H4 = q * hy**2. + c
                        H5 = q * hy * hz - s * hx

                        H6 = q * hz * hx - s * hy
                        H7 = q * hz * hy + s * hx
                        H8 = q * hz**2. + c

                        # save the tangential displacement temporarily
                        # will be used while rotation
                        tmpx = d_tng_x[k]
                        tmpy = d_tng_y[k]
                        tmpz = d_tng_z[k]

                        d_tng_x[k] = H0 * tmpx + H1 * tmpy + H2 * tmpz
                        d_tng_y[k] = H3 * tmpx + H4 * tmpy + H5 * tmpz
                        d_tng_z[k] = H6 * tmpx + H7 * tmpy + H8 * tmpz

                        # save the current normal of the spring
                        d_tng_nx[k] = nxc
                        d_tng_ny[k] = nyc
                        d_tng_nz[k] = nzc

                        k = k + 1
                else:
                    k = k + 1
                count += 1


class RigidBodyWallCollision(Equation):
    """Force between sphere and a wall is implemented using
    DEM contact force law.

    Refer https://doi.org/10.1016/j.powtec.2011.09.019 for more
    information.

    Open-source MFIX-DEM software for gas–solids flows:
    Part I—Verification studies .

    """
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        """Initialise the required coefficients for force calculation.


        Keyword arguments:
        kn -- Normal spring stiffness (default 1e3)
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)

        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.

        """
        self.kn = kn
        self.kt = 2. / 7. * kn
        m_eff = np.pi * 0.5**2 * 1e-6 * 2120
        self.gamma_n = -(2 * np.sqrt(kn * m_eff) * np.log(en)) / (
            np.sqrt(np.pi**2 + np.log(en)**2))
        print(self.gamma_n)
        self.gamma_t = 0.5 * self.gamma_n
        self.mu = mu
        super(RigidBodyWallCollision, self).__init__(dest, sources)

    def loop(self, d_idx, d_fx, d_fy, d_fz, d_h, d_total_mass, d_rad_s,
             d_tang_disp_x, d_tang_disp_y, d_tang_disp_z, d_tang_velocity_x,
             d_tang_velocity_y, d_tang_velocity_z, s_idx, XIJ, RIJ,
             R2IJ, VIJ, s_nx, s_ny, s_nz):
        # check overlap amount
        overlap = d_rad_s[d_idx] - (XIJ[0] * s_nx[s_idx] + XIJ[1] *
                                    s_ny[s_idx] + XIJ[2] * s_nz[s_idx])

        if overlap > 0:
            # basic variables: normal vector
            nij_x = -s_nx[s_idx]
            nij_y = -s_ny[s_idx]
            nij_z = -s_nz[s_idx]

            # overlap speed: a scalar
            vijdotnij = VIJ[0] * nij_x + VIJ[1] * nij_y + VIJ[2] * nij_z

            # normal velocity
            vijn_x = vijdotnij * nij_x
            vijn_y = vijdotnij * nij_y
            vijn_z = vijdotnij * nij_z

            # normal force with conservative and dissipation part
            fn_x = -self.kn * overlap * nij_x - self.gamma_n * vijn_x
            fn_y = -self.kn * overlap * nij_y - self.gamma_n * vijn_y
            fn_z = -self.kn * overlap * nij_z - self.gamma_n * vijn_z

            # ----------------------Tangential force---------------------- #

            # tangential velocity
            d_tang_velocity_x[d_idx] = VIJ[0] - vijn_x
            d_tang_velocity_y[d_idx] = VIJ[1] - vijn_y
            d_tang_velocity_z[d_idx] = VIJ[2] - vijn_z

            _tang = (
                (d_tang_velocity_x[d_idx]**2) + (d_tang_velocity_y[d_idx]**2) +
                (d_tang_velocity_z[d_idx]**2))**(1. / 2.)

            # tangential unit vector
            tij_x = 0
            tij_y = 0
            tij_z = 0
            if _tang > 0:
                tij_x = d_tang_velocity_x[d_idx] / _tang
                tij_y = d_tang_velocity_y[d_idx] / _tang
                tij_z = d_tang_velocity_z[d_idx] / _tang

            # damping force or dissipation
            ft_x_d = -self.gamma_t * d_tang_velocity_x[d_idx]
            ft_y_d = -self.gamma_t * d_tang_velocity_y[d_idx]
            ft_z_d = -self.gamma_t * d_tang_velocity_z[d_idx]

            # tangential spring force
            ft_x_s = -self.kt * d_tang_disp_x[d_idx]
            ft_y_s = -self.kt * d_tang_disp_y[d_idx]
            ft_z_s = -self.kt * d_tang_disp_z[d_idx]

            ft_x = ft_x_d + ft_x_s
            ft_y = ft_y_d + ft_y_s
            ft_z = ft_z_d + ft_z_s

            # coulomb law
            ftij = ((ft_x**2) + (ft_y**2) + (ft_z**2))**(1. / 2.)
            fnij = ((fn_x**2) + (fn_y**2) + (fn_z**2))**(1. / 2.)

            _fnij = self.mu * fnij

            if _fnij < ftij:
                ft_x = -_fnij * tij_x
                ft_y = -_fnij * tij_y
                ft_z = -_fnij * tij_z

            d_fx[d_idx] += fn_x + ft_x
            d_fy[d_idx] += fn_y + ft_y
            d_fz[d_idx] += fn_z + ft_z
        else:
            d_tang_velocity_x[d_idx] = 0
            d_tang_velocity_y[d_idx] = 0
            d_tang_velocity_z[d_idx] = 0

            d_tang_disp_x[d_idx] = 0
            d_tang_disp_y[d_idx] = 0
            d_tang_disp_z[d_idx] = 0


class EulerStepRigidBody(IntegratorStep):
    """Fast but inaccurate integrator. Use this for testing"""
    def initialize(self):
        pass

    def stage1(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z,
               d_omega, d_omega_dot, d_vc, d_ac, d_num_body,
               dt=0.0):
        _i = declare('int')
        _j = declare('int')
        base = declare('int')
        if d_idx == 0:
            for _i in range(d_num_body[0]):
                base = 3*_i
                for _j in range(3):
                    d_vc[base + _j] += d_ac[base + _j]*dt
                    d_omega[base + _j] += d_omega_dot[base + _j]*dt

        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]


class RK2StepRigidBody(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0,
                   d_omega, d_omega0, d_vc, d_vc0, d_num_body):
        _i = declare('int')
        _j = declare('int')
        base = declare('int')
        if d_idx == 0:
            for _i in range(d_num_body[0]):
                base = 3*_i
                for _j in range(3):
                    d_vc0[base + _j] = d_vc[base + _j]
                    d_omega0[base + _j] = d_omega[base + _j]

        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

    def stage1(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, d_x0, d_y0, d_z0,
               d_omega, d_omega_dot, d_vc, d_ac, d_omega0, d_vc0, d_num_body,
               dt=0.0):
        dtb2 = 0.5*dt
        _i = declare('int')
        j = declare('int')
        base = declare('int')
        if d_idx == 0:
            for _i in range(d_num_body[0]):
                base = 3*_i
                for j in range(3):
                    d_vc[base + j] = d_vc0[base + j] + d_ac[base + j]*dtb2
                    d_omega[base + j] = (d_omega0[base + j] +
                                         d_omega_dot[base + j]*dtb2)

        d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2*d_w[d_idx]

    def stage2(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, d_x0, d_y0, d_z0,
               d_omega, d_omega_dot, d_vc, d_ac, d_omega0, d_vc0, d_num_body,
               dt=0.0):
        _i = declare('int')
        j = declare('int')
        base = declare('int')
        if d_idx == 0:
            for _i in range(d_num_body[0]):
                base = 3*_i
                for j in range(3):
                    d_vc[base + j] = d_vc0[base + j] + d_ac[base + j]*dt
                    d_omega[base + j] = (d_omega0[base + j] +
                                         d_omega_dot[base + j]*dt)

        d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt*d_w[d_idx]


def get_particle_array_rigid_body_dem(constants=None, **props):
    """Return a particle array for a rigid body motion.

    For multiple bodies, add a body_id property starting at index 0 with each
    index denoting the body to which the particle corresponds to.

    Parameters
    ----------
    constants : dict
        Dictionary of constants

    Other Parameters
    ----------------
    props : dict
        Additional keywords passed are set as the property arrays.

    See Also
    --------
    get_particle_array

    """
    extra_props = [
        'au', 'av', 'aw', 'V', 'fx', 'fy', 'fz',
        'x0', 'y0', 'z0',
        'rad_s', 'nx', 'ny', 'nz'
    ]

    body_id = props.pop('body_id', None)
    nb = 1 if body_id is None else numpy.max(body_id) + 1

    dem_id = props.pop('dem_id', None)

    consts = {
        'total_mass': numpy.zeros(nb, dtype=float),
        'num_body': numpy.asarray(nb, dtype=int),
        'cm': numpy.zeros(3 * nb, dtype=float),

        # The mi are also used to temporarily reduce mass (1), center of
        # mass (3) and the inertia components (6), total force (3), total
        # torque (3).
        'mi': numpy.zeros(16 * nb, dtype=float),
        'force': numpy.zeros(3 * nb, dtype=float),
        'torque': numpy.zeros(3 * nb, dtype=float),
        # velocity, acceleration of CM.
        'vc': numpy.zeros(3 * nb, dtype=float),
        'ac': numpy.zeros(3 * nb, dtype=float),
        'vc0': numpy.zeros(3 * nb, dtype=float),
        # angular velocity, acceleration of body.
        'omega': numpy.zeros(3 * nb, dtype=float),
        'omega0': numpy.zeros(3 * nb, dtype=float),
        'omega_dot': numpy.zeros(3 * nb, dtype=float)
    }
    if constants:
        consts.update(constants)
    pa = get_particle_array(constants=consts, additional_props=extra_props,
                            **props)
    pa.add_property('body_id', type='int', data=body_id)
    pa.add_property('dem_id', type='int', data=dem_id)

    # create the array to save the tangential interaction particles
    # index and other variables
    limit = 30

    pa.add_constant('limit', limit)
    pa.add_constant('tng_idx', [-1] * limit * len(pa.x))
    pa.add_constant('tng_idx_dem_id', [-1] * limit * len(pa.x))
    pa.add_constant('tng_x', [0.] * limit * len(pa.x))
    pa.add_constant('tng_y', [0.] * limit * len(pa.x))
    pa.add_constant('tng_z', [0.] * limit * len(pa.x))
    pa.add_constant('tng_x0', [0.] * limit * len(pa.x))
    pa.add_constant('tng_y0', [0.] * limit * len(pa.x))
    pa.add_constant('tng_z0', [0.] * limit * len(pa.x))
    pa.add_constant('tng_nx', [0.] * limit * len(pa.x))
    pa.add_constant('tng_ny', [0.] * limit * len(pa.x))
    pa.add_constant('tng_nz', [0.] * limit * len(pa.x))
    pa.add_constant('tng_nx0', [0.] * limit * len(pa.x))
    pa.add_constant('tng_ny0', [0.] * limit * len(pa.x))
    pa.add_constant('tng_nz0', [0.] * limit * len(pa.x))
    pa.add_constant('vtx', [0.] * limit * len(pa.x))
    pa.add_constant('vty', [0.] * limit * len(pa.x))
    pa.add_constant('vtz', [0.] * limit * len(pa.x))
    pa.add_constant('total_tng_contacts', [0] * len(pa.x))

    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au', 'av',
        'aw', 'tag', 'gid', 'V', 'fx', 'fy', 'fz', 'body_id'
    ])

    return pa


class RK2StepRigidBodyDEM(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_omega,
                   d_omega0, d_vc, d_vc0, d_num_body, d_total_tng_contacts,
                   d_limit, d_tng_x, d_tng_y, d_tng_z, d_tng_x0, d_tng_y0,
                   d_tng_z0, d_tng_nx, d_tng_ny, d_tng_nz, d_tng_nx0,
                   d_tng_ny0, d_tng_nz0):
        _i = declare('int')
        _j = declare('int')
        base = declare('int')
        if d_idx == 0:
            for _i in range(d_num_body[0]):
                base = 3*_i
                for _j in range(3):
                    d_vc0[base + _j] = d_vc[base + _j]
                    d_omega0[base + _j] = d_omega[base + _j]

        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

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
            d_tng_x0[i] = d_tng_x[i]
            d_tng_y0[i] = d_tng_y[i]
            d_tng_z0[i] = d_tng_z[i]
            d_tng_nx0[i] = d_tng_nx[i]
            d_tng_ny0[i] = d_tng_ny[i]
            d_tng_nz0[i] = d_tng_nz[i]

    def stage1(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, d_x0, d_y0, d_z0,
               d_omega, d_omega_dot, d_vc, d_ac, d_omega0, d_vc0, d_num_body,
               d_total_tng_contacts, d_limit, d_tng_x, d_tng_y, d_tng_z,
               d_vtx, d_vty, d_vtz, dt=0.0):
        dtb2 = 0.5*dt
        _i = declare('int')
        j = declare('int')
        base = declare('int')
        if d_idx == 0:
            for _i in range(d_num_body[0]):
                base = 3*_i
                for j in range(3):
                    d_vc[base + j] = d_vc0[base + j] + d_ac[base + j]*dtb2
                    d_omega[base + j] = (d_omega0[base + j] +
                                         d_omega_dot[base + j]*dtb2)

        d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2*d_w[d_idx]

        # --------------------------------------
        # increment the tangential displacement
        # --------------------------------------
        i = declare('int')
        p = declare('int')
        q = declare('int')
        tot_ctcs = declare('int')
        tot_ctcs = d_total_tng_contacts[d_idx]
        p = d_idx * d_limit[0]
        q = p + tot_ctcs

        for i in range(p, q):
            d_tng_x[i] += d_vtx[i] * dtb2
            d_tng_y[i] += d_vty[i] * dtb2
            d_tng_z[i] += d_vtz[i] * dtb2

    def stage2(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, d_x0, d_y0, d_z0,
               d_omega, d_omega_dot, d_vc, d_ac, d_omega0, d_vc0, d_num_body,
               d_total_tng_contacts, d_limit, d_tng_x, d_tng_y, d_tng_z,
               d_tng_x0, d_tng_y0, d_tng_z0, d_vtx, d_vty, d_vtz, dt=0.0):
        _i = declare('int')
        j = declare('int')
        base = declare('int')
        if d_idx == 0:
            for _i in range(d_num_body[0]):
                base = 3*_i
                for j in range(3):
                    d_vc[base + j] = d_vc0[base + j] + d_ac[base + j]*dt
                    d_omega[base + j] = (d_omega0[base + j] +
                                         d_omega_dot[base + j]*dt)

        d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt*d_w[d_idx]

        # --------------------------------------
        # increment the tangential displacement
        # --------------------------------------
        i = declare('int')
        p = declare('int')
        q = declare('int')
        tot_ctcs = declare('int')
        tot_ctcs = d_total_tng_contacts[d_idx]
        p = d_idx * d_limit[0]
        q = p + tot_ctcs

        for i in range(p, q):
            d_tng_x[i] = d_tng_x0[i] + d_vtx[i] * dt
            d_tng_y[i] = d_tng_y0[i] + d_vty[i] * dt
            d_tng_z[i] = d_tng_z0[i] + d_vtz[i] * dt


class RigidBodySimpleScheme(Scheme):
    def __init__(self, bodies, solids, dim, rho0, kn,
                 mu=0.5, en=1.0,
                 gx=0.0, gy=0.0, gz=0.0,
                 debug=False):
        self.bodies = bodies
        self.solids = solids
        self.dim = dim
        self.rho0 = rho0
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
        from pysph.sph.integrator import EPECIntegrator
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        for body in self.bodies:
            if body not in steppers:
                steppers[body] = RK2StepRigidBody()

        cls = integrator_cls if integrator_cls is not None else EPECIntegrator
        integrator = cls(**steppers)

        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        from pysph.sph.equation import Group
        equations = []
        g1 = []
        if self.solids is not None:
            all = self.bodies + self.solids
        else:
            all = self.bodies

        for name in self.bodies:
            g1.append(BodyForce(
                dest=name, sources=None, gx=self.gx, gy=self.gy, gz=self.gz
            ))
        equations.append(Group(equations=g1, real=False))

        g2 = []
        for name in self.bodies:
            g2.append(RigidBodyCollision(
                dest=name, sources=all, kn=self.kn, mu=self.mu, en=self.en
            ))
        equations.append(Group(equations=g2, real=False))

        g3 = []
        for name in self.bodies:
            g3.append(RigidBodyMoments(
                dest=name, sources=None))
        equations.append(Group(equations=g3, real=False))

        g4 = []
        for name in self.bodies:
            g4.append(RigidBodyMotion(
                dest=name, sources=None))
        equations.append(Group(equations=g4, real=False))

        return equations


#################################################
# Rigid body simulation using rotation matrices #
#################################################
def get_particle_array_rigid_body_rotation_matrix(constants=None, **props):
    extra_props = [
        'fx', 'fy', 'fz', 'dx0', 'dy0', 'dz0', 'nx0', 'ny0', 'nz0', 'nx', 'ny',
        'nz'
    ]

    body_id = props.pop('body_id', None)
    nb = 1 if body_id is None else numpy.max(body_id) + 1

    dem_id = props.pop('dem_id', None)

    consts = {
        'total_mass': numpy.zeros(nb, dtype=float),
        'num_body': numpy.asarray(nb, dtype=int),
        'cm': numpy.zeros(3*nb, dtype=float),
        'cm0': numpy.zeros(3*nb, dtype=float),
        'R': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
        'R0': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
        # moment of inertia inverse in body frame
        'mib': numpy.zeros(9*nb, dtype=float),
        # moment of inertia inverse in global frame
        'mig': numpy.zeros(9*nb, dtype=float),
        # total force at the center of mass
        'force': numpy.zeros(3*nb, dtype=float),
        # torque about the center of mass
        'torque': numpy.zeros(3*nb, dtype=float),
        # velocity, acceleration of CM.
        'vc': numpy.zeros(3*nb, dtype=float),
        'vc0': numpy.zeros(3*nb, dtype=float),
        # angular momentum
        'L': numpy.zeros(3*nb, dtype=float),
        'L0': numpy.zeros(3*nb, dtype=float),
        # angular velocity in global frame
        'omega': numpy.zeros(3*nb, dtype=float),
        'omega0': numpy.zeros(3*nb, dtype=float),
    }
    if constants:
        consts.update(constants)

    pa = get_particle_array(constants=consts, additional_props=extra_props,
                            **props)
    pa.add_property('body_id', type='int', data=body_id)
    pa.add_property('dem_id', type='int', data=dem_id)

    setup_rotation_matrix_rigid_body(pa)

    pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'fz', 'm'])
    return pa


def normalize_R_orientation(orien):
    a1 = np.array([orien[0], orien[3], orien[6]])
    a2 = np.array([orien[1], orien[4], orien[7]])
    a3 = np.array([orien[2], orien[5], orien[8]])
    # norm of col0
    na1 = np.linalg.norm(a1)

    b1 = a1 / na1

    b2 = a2 - np.dot(b1, a2) * b1
    nb2 = np.linalg.norm(b2)
    b2 = b2 / nb2

    b3 = a3 - np.dot(b1, a3) * b1 - np.dot(b2, a3) * b2
    nb3 = np.linalg.norm(b3)
    b3 = b3 / nb3

    orien[0] = b1[0]
    orien[3] = b1[1]
    orien[6] = b1[2]
    orien[1] = b2[0]
    orien[4] = b2[1]
    orien[7] = b2[2]
    orien[2] = b3[0]
    orien[5] = b3[1]
    orien[8] = b3[2]


class RK2StepRigidBodyRotationMatrices(IntegratorStep):
    def py_initialize(self, dst, t, dt):
        for j in range(3):
            # save the center of mass and center of mass velocity
            dst.cm0[j] = dst.cm[j]
            dst.vc0[j] = dst.vc[j]

            # save the current angular momentum
            # dst.L0[j] = dst.L[j]
            dst.omega0[j] = dst.omega[j]

        # save the current orientation
        for j in range(9):
            dst.R0[j] = dst.R[j]

    def initialize(self):
        pass

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for j in range(3):
            # using velocity at t, move position
            # to t + dt/2.
            dst.cm[j] = dst.cm[j] + dtb2 * dst.vc[j]
            dst.vc[j] = dst.vc[j] + dtb2 * dst.force[j] / dst.total_mass[0]
        # angular velocity in terms of matrix
        omega_mat = np.array([[0, -dst.omega[2], dst.omega[1]],
                              [dst.omega[2], 0, -dst.omega[0]],
                              [-dst.omega[1], dst.omega[0], 0]])

        # Currently the orientation is at time t
        R = dst.R.reshape(3, 3)

        # Rate of change of orientation is
        r_dot = np.matmul(omega_mat, R)
        r_dot = r_dot.ravel()

        # update the orientation to next time step
        dst.R[:] = dst.R0[:] + r_dot[:] * dtb2

        # normalize the orientation using Gram Schmidt process
        normalize_R_orientation(dst.R)

        # update the moment of inertia
        R = dst.R.reshape(3, 3)
        R_t = R.transpose()
        tmp = np.matmul(R, dst.mib.reshape(3, 3))
        dst.mig[:] = (np.matmul(tmp, R_t)).ravel()
        # move angular velocity to t + dt/2.
        # omega_dot is
        tmp = dst.torque - np.cross(
            dst.omega, np.matmul(dst.mig.reshape(3, 3), dst.omega))
        omega_dot = np.matmul(dst.mig.reshape(3, 3), tmp)
        dst.omega = dst.omega0 + omega_dot * dtb2

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_cm, d_vc, d_R, d_omega):
        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[0] * d_dx0[d_idx] + d_R[1] * d_dy0[d_idx] +
              d_R[2] * d_dz0[d_idx])
        dy = (d_R[3] * d_dx0[d_idx] + d_R[4] * d_dy0[d_idx] +
              d_R[5] * d_dz0[d_idx])
        dz = (d_R[6] * d_dx0[d_idx] + d_R[7] * d_dy0[d_idx] +
              d_R[8] * d_dz0[d_idx])

        d_x[d_idx] = d_cm[0] + dx
        d_y[d_idx] = d_cm[1] + dy
        d_z[d_idx] = d_cm[2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[1] * dz - d_omega[2] * dy
        dv = d_omega[2] * dx - d_omega[0] * dz
        dw = d_omega[0] * dy - d_omega[1] * dx

        d_u[d_idx] = d_vc[0] + du
        d_v[d_idx] = d_vc[1] + dv
        d_w[d_idx] = d_vc[2] + dw

    def py_stage2(self, dst, t, dt):
        for j in range(3):
            # using velocity at t + dt/2., move position
            # to t + dt
            dst.cm[j] = dst.cm0[j] + dt * dst.vc[j]
            # move linear velocity to t + dt
            dst.vc[j] = dst.vc0[j] + (dt * dst.force[j] / dst.total_mass[0])

        # angular velocity in terms of matrix
        omega_mat = np.array([[0, -dst.omega[2], dst.omega[1]],
                              [dst.omega[2], 0, -dst.omega[0]],
                              [-dst.omega[1], dst.omega[0], 0]])

        # Currently the orientation is at time t + dt/2.
        R = dst.R.reshape(3, 3)

        # Rate of change of orientation is
        r_dot = np.matmul(omega_mat, R)
        r_dot = r_dot.ravel()

        # update the orientation to next time step
        dst.R[:] = dst.R0[:] + r_dot[:] * dt

        # normalize the orientation using Gram Schmidt process
        normalize_R_orientation(dst.R)

        # update the moment of inertia
        R = dst.R.reshape(3, 3)
        R_t = R.transpose()
        tmp = np.matmul(R, dst.mib.reshape(3, 3))
        dst.mig[:] = (np.matmul(tmp, R_t)).ravel()

        # move angular velocity to t + dt
        tmp = dst.torque - np.cross(
            dst.omega, np.matmul(dst.mig.reshape(3, 3), dst.omega))
        omega_dot = np.matmul(dst.mig.reshape(3, 3), tmp)
        dst.omega = dst.omega0 + omega_dot * dt

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_cm, d_vc, d_R, d_omega):
        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[0] * d_dx0[d_idx] + d_R[1] * d_dy0[d_idx] +
              d_R[2] * d_dz0[d_idx])
        dy = (d_R[3] * d_dx0[d_idx] + d_R[4] * d_dy0[d_idx] +
              d_R[5] * d_dz0[d_idx])
        dz = (d_R[6] * d_dx0[d_idx] + d_R[7] * d_dy0[d_idx] +
              d_R[8] * d_dz0[d_idx])

        d_x[d_idx] = d_cm[0] + dx
        d_y[d_idx] = d_cm[1] + dy
        d_z[d_idx] = d_cm[2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[1] * dz - d_omega[2] * dy
        dv = d_omega[2] * dx - d_omega[0] * dz
        dw = d_omega[0] * dy - d_omega[1] * dx

        d_u[d_idx] = d_vc[0] + du
        d_v[d_idx] = d_vc[1] + dv
        d_w[d_idx] = d_vc[2] + dw


class RigidBodyRotationMatricesScheme(Scheme):
    def __init__(self, bodies, solids, dim, rho0, kn, mu=0.5, en=1.0, gx=0.0,
                 gy=0.0, gz=0.0, debug=False):
        self.bodies = bodies
        self.solids = solids
        self.dim = dim
        self.rho0 = rho0
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
        from pysph.sph.integrator import EPECIntegrator
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        for body in self.bodies:
            if body not in steppers:
                steppers[body] = RK2StepRigidBodyRotationMatrices()

        cls = integrator_cls if integrator_cls is not None else EPECIntegrator
        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def get_equations(self):
        from pysph.sph.equation import Group
        equations = []
        g1 = []
        if self.solids is not None:
            all = self.bodies + self.solids
        else:
            all = self.bodies

        for name in self.bodies:
            g1.append(
                BodyForce(dest=name, sources=None, gx=self.gx, gy=self.gy,
                          gz=self.gz))
        equations.append(Group(equations=g1, real=False))

        g2 = []
        for name in self.bodies:
            g2.append(
                RigidBodyCollision(dest=name, sources=all, kn=self.kn,
                                   mu=self.mu, en=self.en))
        equations.append(Group(equations=g2, real=False))

        g3 = []
        for name in self.bodies:
            g3.append(SumUpExternalForces(dest=name, sources=None))
        equations.append(Group(equations=g3, real=False))

        return equations


##########################################
# Rigid body simulation using quaternion #
##########################################
class SumUpExternalForces(Equation):
    def reduce(self, dst, t, dt):
        frc = declare('object')
        trq = declare('object')
        fx = declare('object')
        fy = declare('object')
        fz = declare('object')
        x = declare('object')
        y = declare('object')
        z = declare('object')
        cm = declare('object')
        j = declare('int')

        frc = dst.force
        trq = dst.torque
        fx = dst.fx
        fy = dst.fy
        fz = dst.fz
        x = dst.x
        y = dst.y
        z = dst.z
        cm = dst.cm

        frc[:] = 0
        trq[:] = 0

        for j in range(len(x)):
            frc[0] += fx[j]
            frc[1] += fy[j]
            frc[2] += fz[j]

            # torque due to force on particle i
            # (r_i - com) \cross f_i
            dx = x[j] - cm[0]
            dy = y[j] - cm[1]
            dz = z[j] - cm[2]

            # torque due to force on particle i
            # dri \cross fi
            trq[0] += (dy * fz[j] - dz * fy[j])
            trq[1] += (dz * fx[j] - dx * fz[j])
            trq[2] += (dx * fy[j] - dy * fx[j])


def get_particle_array_rigid_body_quaternion(constants=None, **props):
    extra_props = [
        'fx', 'fy', 'fz', 'dx0', 'dy0', 'dz0', 'nx0', 'ny0', 'nz0', 'nx', 'ny',
        'nz', 'x0', 'y0', 'z0', 'u0', 'v0', 'w0'
    ]

    consts = {
        'total_mass': 0.,
        'cm': numpy.zeros(3, dtype=float),
        'cm0': numpy.zeros(3, dtype=float),
        'q': numpy.array([1., 0., 0., 0.]),
        'q0': numpy.array([1., 0., 0., 0.]),
        'qdot': numpy.zeros(4, dtype=float),
        'R': [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        # moment of inertia inverse in body frame
        'mib': numpy.zeros(9, dtype=float),
        # moment of inertia inverse in global frame
        'mig': numpy.zeros(9, dtype=float),
        # total force at the center of mass
        'force': numpy.zeros(3, dtype=float),
        # torque about the center of mass
        'torque': numpy.zeros(3, dtype=float),
        # velocity, acceleration of CM.
        'vc': numpy.zeros(3, dtype=float),
        'vc0': numpy.zeros(3, dtype=float),
        # angular velocity in global frame
        'omega': numpy.zeros(3, dtype=float),
        'omega0': numpy.zeros(3, dtype=float),
    }

    if constants:
        consts.update(constants)

    pa = get_particle_array(constants=consts, additional_props=extra_props,
                            **props)
    setup_quaternion_rigid_body(pa)

    pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'fz', 'm'])
    return pa


def normalize_q_orientation(q):
    norm_q = sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    q[:] = q[:] / norm_q


def quaternion_multiplication(p, q, res):
    """Parameters
    ----------
    p   : [float]
          An array of length four
    q   : [float]
          An array of length four
    res : [float]
          An array of length four

    Here `p` is a quaternion. i.e., p = [p.w, p.x, p.y, p.z]. And q is an
    another quaternion.

    This function is used to compute the rate of change of orientation
    when orientation is represented in terms of a quaternion. When the
    angular velocity is represented in terms of global frame
    \frac{dq}{dt} = \frac{1}{2} omega q

    http://www.ams.stonybrook.edu/~coutsias/papers/rrr.pdf
    see equation 8

    """
    res[0] = (p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3])
    res[1] = (p[0] * q[1] + q[0] * p[1] + p[2] * q[3] - p[3] * q[2])
    res[2] = (p[0] * q[2] + q[0] * p[2] + p[3] * q[1] - p[1] * q[3])
    res[3] = (p[0] * q[3] + q[0] * p[3] + p[1] * q[2] - p[2] * q[1])


def scale_quaternion(q, scale):
    q[0] = q[0] * scale
    q[1] = q[1] * scale
    q[2] = q[2] * scale
    q[3] = q[3] * scale


def quaternion_to_matrix(q, matrix):
    matrix[0] = 1. - 2. * (q[2]**2. + q[3]**2.)
    matrix[1] = 2. * (q[1] * q[2] + q[0] * q[3])
    matrix[2] = 2. * (q[1] * q[3] - q[0] * q[2])

    matrix[3] = 2. * (q[1] * q[2] - q[0] * q[3])
    matrix[4] = 1. - 2. * (q[1]**2. + q[3]**2.)
    matrix[5] = 2. * (q[2] * q[3] + q[0] * q[1])

    matrix[6] = 2. * (q[1] * q[3] + q[0] * q[2])
    matrix[7] = 2. * (q[2] * q[3] - q[0] * q[1])
    matrix[8] = 1. - 2. * (q[1]**2. + q[2]**2.)


class RK2StepRigidBodyQuaternions(IntegratorStep):
    def py_initialize(self, dst, t, dt):
        for j in range(3):
            # save the center of mass and center of mass velocity
            dst.cm0[j] = dst.cm[j]
            dst.vc0[j] = dst.vc[j]

            # save the current angular momentum
            dst.omega0[j] = dst.omega[j]

        # save the current orientation
        for j in range(4):
            dst.q0[j] = dst.q[j]

    def initialize(self):
        pass

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for j in range(3):
            # using velocity at t, move position
            # to t + dt/2.
            dst.cm[j] = dst.cm[j] + dtb2 * dst.vc[j]
            dst.vc[j] = dst.vc[j] + dtb2 * dst.force[j] / dst.total_mass[0]

        # change in quaternion
        delta_quat = np.array([0., 0., 0., 0.])
        # angular velocity magnitude
        omega_magn = sqrt(dst.omega[0]**2 + dst.omega[1]**2 + dst.omega[2]**2)
        axis_rot = np.array([0., 0., 0.])
        if omega_magn > 1e-12:
            axis_rot = dst.omega / omega_magn
        delta_quat[0] = cos(omega_magn * dtb2 * 0.5)
        delta_quat[1] = axis_rot[0] * sin(omega_magn * dtb2 * 0.5)
        delta_quat[2] = axis_rot[1] * sin(omega_magn * dtb2 * 0.5)
        delta_quat[3] = axis_rot[2] * sin(omega_magn * dtb2 * 0.5)

        res = np.array([0., 0., 0., 0.])
        quaternion_multiplication(dst.q, delta_quat, res)
        dst.q = res

        # normalize the orientation
        normalize_q_orientation(dst.q)

        # update the moment of inertia
        quaternion_to_matrix(dst.q, dst.R)
        R = dst.R.reshape(3, 3)
        R_t = R.T
        tmp = np.matmul(R, dst.mib.reshape(3, 3))
        dst.mig[:] = (np.matmul(tmp, R_t)).ravel()
        # move angular velocity to t + dt/2.
        # omega_dot is
        tmp = dst.torque - np.cross(
            dst.omega, np.matmul(dst.mig.reshape(3, 3), dst.omega))
        omega_dot = np.matmul(dst.mig.reshape(3, 3), tmp)
        dst.omega = dst.omega0 + omega_dot * dtb2

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_cm, d_vc, d_R, d_omega):
        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[0] * d_dx0[d_idx] + d_R[1] * d_dy0[d_idx] +
              d_R[2] * d_dz0[d_idx])
        dy = (d_R[3] * d_dx0[d_idx] + d_R[4] * d_dy0[d_idx] +
              d_R[5] * d_dz0[d_idx])
        dz = (d_R[6] * d_dx0[d_idx] + d_R[7] * d_dy0[d_idx] +
              d_R[8] * d_dz0[d_idx])

        d_x[d_idx] = d_cm[0] + dx
        d_y[d_idx] = d_cm[1] + dy
        d_z[d_idx] = d_cm[2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[1] * dz - d_omega[2] * dy
        dv = d_omega[2] * dx - d_omega[0] * dz
        dw = d_omega[0] * dy - d_omega[1] * dx

        d_u[d_idx] = d_vc[0] + du
        d_v[d_idx] = d_vc[1] + dv
        d_w[d_idx] = d_vc[2] + dw

    def py_stage2(self, dst, t, dt):
        for j in range(3):
            dst.cm[j] = dst.cm0[j] + dt * dst.vc[j]
            dst.vc[j] = dst.vc0[j] + dt * dst.force[j] / dst.total_mass[0]

        # delta quaternion (change in quaternion)
        delta_quat = np.array([0., 0., 0., 0.])
        # angular velocity magnitude
        omega_magn = sqrt(dst.omega[0]**2 + dst.omega[1]**2 + dst.omega[2]**2)
        axis_rot = np.array([0., 0., 0.])
        if omega_magn > 1e-12:
            axis_rot = dst.omega / omega_magn
        delta_quat[0] = cos(omega_magn * dt * 0.5)
        delta_quat[1] = axis_rot[0] * sin(omega_magn * dt * 0.5)
        delta_quat[2] = axis_rot[1] * sin(omega_magn * dt * 0.5)
        delta_quat[3] = axis_rot[2] * sin(omega_magn * dt * 0.5)

        res = np.array([0., 0., 0., 0.])
        quaternion_multiplication(dst.q0, delta_quat, res)
        dst.q = res

        # normalize the orientation
        normalize_q_orientation(dst.q)

        # update the moment of inertia
        quaternion_to_matrix(dst.q, dst.R)
        R = dst.R.reshape(3, 3)
        R_t = R.T
        tmp = np.matmul(R, dst.mib.reshape(3, 3))
        dst.mig[:] = (np.matmul(tmp, R_t)).ravel()
        # move angular velocity to t + dt/2.
        # omega_dot is
        tmp = dst.torque - np.cross(
            dst.omega, np.matmul(dst.mig.reshape(3, 3), dst.omega))
        omega_dot = np.matmul(dst.mig.reshape(3, 3), tmp)
        dst.omega = dst.omega0 + omega_dot * dt

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_cm, d_vc, d_R, d_omega):
        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[0] * d_dx0[d_idx] + d_R[1] * d_dy0[d_idx] +
              d_R[2] * d_dz0[d_idx])
        dy = (d_R[3] * d_dx0[d_idx] + d_R[4] * d_dy0[d_idx] +
              d_R[5] * d_dz0[d_idx])
        dz = (d_R[6] * d_dx0[d_idx] + d_R[7] * d_dy0[d_idx] +
              d_R[8] * d_dz0[d_idx])

        d_x[d_idx] = d_cm[0] + dx
        d_y[d_idx] = d_cm[1] + dy
        d_z[d_idx] = d_cm[2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[1] * dz - d_omega[2] * dy
        dv = d_omega[2] * dx - d_omega[0] * dz
        dw = d_omega[0] * dy - d_omega[1] * dx

        d_u[d_idx] = d_vc[0] + du
        d_v[d_idx] = d_vc[1] + dv
        d_w[d_idx] = d_vc[2] + dw


class RigidBodyQuaternionScheme(Scheme):
    def __init__(self, bodies, solids, dim, rho0, kn, mu=0.5, en=1.0, gx=0.0,
                 gy=0.0, gz=0.0, debug=False):
        self.bodies = bodies
        self.solids = solids
        self.dim = dim
        self.rho0 = rho0
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
        from pysph.sph.integrator import EPECIntegrator
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        for body in self.bodies:
            if body not in steppers:
                steppers[body] = RK2StepRigidBodyQuaternions()

        cls = integrator_cls if integrator_cls is not None else EPECIntegrator
        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def get_equations(self):
        from pysph.sph.equation import Group
        equations = []
        g1 = []
        if self.solids is not None:
            all = self.bodies + self.solids
        else:
            all = self.bodies

        for name in self.bodies:
            g1.append(
                BodyForce(dest=name, sources=None, gx=self.gx, gy=self.gy,
                          gz=self.gz))
        equations.append(Group(equations=g1, real=False))

        g2 = []
        for name in self.bodies:
            g2.append(
                RigidBodyCollision(dest=name, sources=all, kn=self.kn,
                                   mu=self.mu, en=self.en))
        equations.append(Group(equations=g2, real=False))

        g3 = []
        for name in self.bodies:
            g3.append(SumUpExternalForces(dest=name, sources=None))
        equations.append(Group(equations=g3, real=False))

        return equations
