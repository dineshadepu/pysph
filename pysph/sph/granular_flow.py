"""1. A new SPH-based approach to simulation of granularflows using viscous
damping and stress regularisation


2. Lagrangian meshfree particles method (SPH) for large deformationand failure
flows of geomaterial using elastic-plastic soil constitutive model. Ha H Bui
(2008) in International journal for Numerical and analytical methods in
Geomechanics.

3. Numerical Simulations for Large Deformation of GranularMaterials Using
Smoothed Particle HydrodynamicsMethod by Wei Chen

"""
from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep


def setup_granular_flow_bui_2008(pa):
    """This is taken from


    Lagrangian meshfree particles method (SPH) for large deformationand failure
    flows of geomaterial using elastic-plastic soilconstitutive model

    """
    # add stress tensor with a strided limit of 9
    limit = 9

    pa.add_property('sigma', stride=limit)
    pa.sigma[:] = 0.

    # rate of the stress tensor (Jaumann stress rate)
    pa.add_property('sigma_dot', stride=limit)
    pa.sigma_dot[:] = 0.

    pa.add_property('deviatoric_stress', stride=limit)
    pa.devia_stress[:] = 0.

    # spin rate tensor
    pa.add_property('omega_dot', stride=limit)
    pa.omega_dot[:] = 0.

    # total strain rate tensor
    # [eps_xx, eps_xy, eps_xz, eps_yx, eps_yy, eps_yz, eps_zx, eps_zy, eps_zz]
    pa.add_property('epsilon_dot', stride=limit)
    pa.epsilon_dot[:] = 0.

    pa.add_property('epsilon_dot_ii')
    pa.epsilon_dot_ii[:] = 0.

    # deviatoric strain rate tensor
    # [e_xx, e_xy, e_xz, e_yx, e_yy, e_yz, e_zx, e_zy, e_zz]
    pa.add_property('e_dot', stride=limit)
    pa.e_dot[:] = 0.

    # first invariant of the tensor
    pa.add_property('I1', 0.)
    # second invariant of the tensor
    pa.add_property('J2', 0.)

    # alpha_phi and k_c are Drucker-Prager's constants (eq 18). These will be
    # computed from Couloumb's material constants c (cohesion) and phi (internal
    # friction)
    pa.add_constant('alpha_phi', 0.)
    pa.add_constant('k_c', 0.)

    # bulk modulus and shear modulus
    pa.add_constant('K', 0.)
    pa.add_constant('G', 0.)
    pa.add_constant('nu', 0.)


def set_drucker_prager_constants(pa, cohesion, internal_friction):
    pass


class TotalStrainAndSpinRate(Equation):
    def initialize(self, d_idx, d_epsilon_dot, d_omega_dot):
        didx9 = declare('int')
        i = declare('int')
        didx9 = 9 * d_idx

        for i in range(didx9, didx9 + 9):
            d_epsilon_dot[i] = 0.0
            d_omega_dot[i] = 0.0

    def loop(self, d_idx, d_epsilon_dot, d_u, d_v, d_w, s_idx, s_m, s_rho, s_u,
             s_v, s_w, d_G, DWIJ):
        tmp = s_m[s_idx] / s_rho[s_idx]

        # the following three rates are for epsilon xx, xy, xz
        d_epsilon_dot[9 * d_idx +
                      0] += (tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[0] + tmp *
                             (s_u[s_idx] - d_u[d_idx]) * DWIJ[0])
        d_epsilon_dot[9 * d_idx +
                      1] += (tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[1] + tmp *
                             (s_v[s_idx] - d_v[d_idx]) * DWIJ[0])
        d_epsilon_dot[9 * d_idx +
                      2] += (tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[2] + tmp *
                             (s_w[s_idx] - d_w[d_idx]) * DWIJ[0])

        # the following three rates are for epsilon yx, yy, yz
        d_epsilon_dot[9 * d_idx +
                      3] += (tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[0] + tmp *
                             (s_u[s_idx] - d_u[d_idx]) * DWIJ[1])
        d_epsilon_dot[9 * d_idx +
                      4] += (tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[1] + tmp *
                             (s_v[s_idx] - d_v[d_idx]) * DWIJ[1])
        d_epsilon_dot[9 * d_idx +
                      5] += (tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[2] + tmp *
                             (s_w[s_idx] - d_w[d_idx]) * DWIJ[1])

        # the following three rates are for epsilon zx, zy, zz
        d_epsilon_dot[9 * d_idx +
                      6] += (tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[0] + tmp *
                             (s_u[s_idx] - d_u[d_idx]) * DWIJ[2])
        d_epsilon_dot[9 * d_idx +
                      7] += (tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[1] + tmp *
                             (s_v[s_idx] - d_v[d_idx]) * DWIJ[2])
        d_epsilon_dot[9 * d_idx +
                      8] += (tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[2] + tmp *
                             (s_w[s_idx] - d_w[d_idx]) * DWIJ[2])

        # the following three rates are for omega xx, xy, xz
        d_omega_dot[9 * d_idx +
                      0] += (tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[0] - tmp *
                             (s_u[s_idx] - d_u[d_idx]) * DWIJ[0])
        d_omega_dot[9 * d_idx +
                      1] += (tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[1] - tmp *
                             (s_v[s_idx] - d_v[d_idx]) * DWIJ[0])
        d_omega_dot[9 * d_idx +
                      2] += (tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[2] - tmp *
                             (s_w[s_idx] - d_w[d_idx]) * DWIJ[0])

        # the following three rates are for omega yx, yy, yz
        d_omega_dot[9 * d_idx +
                      3] += (tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[0] - tmp *
                             (s_u[s_idx] - d_u[d_idx]) * DWIJ[1])
        d_omega_dot[9 * d_idx +
                      4] += (tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[1] - tmp *
                             (s_v[s_idx] - d_v[d_idx]) * DWIJ[1])
        d_omega_dot[9 * d_idx +
                      5] += (tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[2] - tmp *
                             (s_w[s_idx] - d_w[d_idx]) * DWIJ[1])

        # the following three rates are for omega zx, zy, zz
        d_omega_dot[9 * d_idx +
                      6] += (tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[0] - tmp *
                             (s_u[s_idx] - d_u[d_idx]) * DWIJ[2])
        d_omega_dot[9 * d_idx +
                      7] += (tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[1] - tmp *
                             (s_v[s_idx] - d_v[d_idx]) * DWIJ[2])
        d_omega_dot[9 * d_idx +
                      8] += (tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[2] - tmp *
                             (s_w[s_idx] - d_w[d_idx]) * DWIJ[2])

    def post_looop(self, d_idx, d_epsilon_dot, d_epsilon_dot_ii, d_omega_dot, d_e_dot):
        didx9 = declare('int')
        i = declare('int')
        didx9 = 9 * d_idx


        for i in range(didx9, didx9 + 9):
            d_epsilon_dot[i] = d_epsilon_dot[i] / 2.
            d_omega_dot[i] = d_omega_dot[i] / 2.

        d_epsilon_dot_ii[d_idx] = d_epsilon_dot[didx9] + d_epsilon_dot[didx9 + 4] + d_epsilon_dot[didx9 + 8]

        # for i in range(didx9, didx9 + 9):
        #     # Equation 13 describes what deviatoric strain is
        #     d_e_dot[i] = d_epsilon_dot[i] - 1. / 3. * epsilon_dot_ii

        d_e_dot[didx9 + 0] = d_epsilon_dot[didx9 + 0] - 1. / 3. * d_epsilon_dot_ii[d_idx]
        d_e_dot[didx9 + 1] = d_epsilon_dot[didx9 + 1]
        d_e_dot[didx9 + 2] = d_epsilon_dot[didx9 + 2]
        d_e_dot[didx9 + 3] = d_epsilon_dot[didx9 + 3]
        d_e_dot[didx9 + 4] = d_epsilon_dot[didx9 + 4] - 1. / 3. * d_epsilon_dot_ii[d_idx]
        d_e_dot[didx9 + 5] = d_epsilon_dot[didx9 + 5]
        d_e_dot[didx9 + 6] = d_epsilon_dot[didx9 + 6]
        d_e_dot[didx9 + 7] = d_epsilon_dot[didx9 + 7]
        d_e_dot[didx9 + 8] = d_epsilon_dot[didx9 + 8] - 1. / 3. * d_epsilon_dot_ii[d_idx]



class StressRateBiu2008(Equation):
    r"""Equation 27 of reference (2) for

    Equation 11 in reference (3) for partial look up

    Stress-strain relationship of the Drucker-Prager elastic-perfectly plastic
    soil model with associatedflow rule.

    """
    def initialize(self, d_idx, d_omega_dot, d, d_as00, d_as01, d_as02, d_as11,
                   d_as12, d_as22):
        didx9 = declare('int')
        i = declare('int')
        didx9 = 9 * d_idx

        # -------------------------------------------
        # compute the placticity multiplier
        # this lamda is taken from equation 27 of reference 2
        # -------------------------------------------
        numerator_tmp1 = 3. * d_alpha_phi[0] * d_K[0] * d_epsilon_dot_ii[d_idx]

        numerator_tmp2a = 0.
        for i in range(didx9, didx9+9):
            numerator_tmp2a += d_deviatoric_stress[i] * d_epsilon_dot[i]

        numerator_tmp2 = (d_G[0] / d_J2[d_idx]**0.5) * numerator_tmp2a

        num_tmp = numerator_tmp1 + numerator_tmp_2


        denom_tmp = 9. * d_alpha_phi[d_idx]**2. * d_K[0] + d_G[0]

        d_lambda[d_idx] = num_tmp / denom_tmp
        # -------------------------------------------
        # compute the placticity multiplier ENDS
        # -------------------------------------------

        d_sigma_dot[didx9 + 0] = 2. * d_G[0] * d_epsilon_dot[didx9 + 0] + d_K[0] * d_epsilon_dot_ii[d_idx]
        d_sigma_dot[didx9 + 1] = 2. * d_G[0] * d_epsilon_dot[didx9 + 1]
        d_sigma_dot[didx9 + 2] = 2. * d_G[0] * d_epsilon_dot[didx9 + 2]
        d_sigma_dot[didx9 + 3] = 2. * d_G[0] * d_epsilon_dot[didx9 + 3]
        d_sigma_dot[didx9 + 4] = 2. * d_G[0] * d_epsilon_dot[didx9 + 4] + d_K[0] * d_epsilon_dot_ii[d_idx]
        d_sigma_dot[didx9 + 5] = 2. * d_G[0] * d_epsilon_dot[didx9 + 5]
        d_sigma_dot[didx9 + 6] = 2. * d_G[0] * d_epsilon_dot[didx9 + 6]
        d_sigma_dot[didx9 + 7] = 2. * d_G[0] * d_epsilon_dot[didx9 + 7]
        d_sigma_dot[didx9 + 8] = 2. * d_G[0] * d_epsilon_dot[didx9 + 8] + d_K[0] * d_epsilon_dot_ii[d_idx]


class GranularFlowRK2(IntegratorStep):
    def initialize(self, d_x, d_idx):
        d_x[d_idx] = d_x[d_idx]

    def stage1(self, d_x, d_idx):
        d_x[d_idx] = d_x[d_idx]

    def stage2(self, d_x, d_idx):
        d_x[d_idx] = d_x[d_idx]
