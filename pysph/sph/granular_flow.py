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
from compyle.api import declare
from compyle.low_level import address


def setup_granular_flow_bui_2008(pa):
    """This is taken from


    Lagrangian meshfree particles method (SPH) for large deformationand failure
    flows of geomaterial using elastic-plastic soilconstitutive model

    """
    # add stress tensor with a strided limit of 9
    limit = 9

    pa.add_property('sigma', stride=limit)
    pa.sigma[:] = 0.

    pa.add_property('sigma0', stride=limit)
    pa.sigma0[:] = 0.

    pa.add_property('sigma_ii')
    pa.sigma_ii[:] = 0.

    # rate of the stress tensor (Jaumann stress rate)
    pa.add_property('sigma_dot', stride=limit)
    pa.sigma_dot[:] = 0.

    pa.add_property('deviatoric_stress', stride=limit)
    pa.deviatoric_stress[:] = 0.

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
    pa.add_property('I1')
    # second invariant of the tensor
    pa.add_property('J2')

    # plasticity constant lambda
    pa.add_property('lambda_dot')

    # alpha_phi and k_c are Drucker-Prager's constants (eq 18). These will be
    # computed from Couloumb's material constants c (cohesion) and phi (internal
    # friction)
    pa.add_constant('alpha_phi', 0.)
    pa.add_constant('k_c', 0.)

    # bulk modulus and shear modulus
    pa.add_constant('K', 0.)
    pa.add_constant('G', 0.)
    pa.add_constant('nu', 0.)


def setup_granular_flow_boundary_bui_2008(pa):
    """This is taken from


    Lagrangian meshfree particles method (SPH) for large deformationand failure
    flows of geomaterial using elastic-plastic soilconstitutive model

    """
    # add stress tensor with a strided limit of 9
    limit = 9

    pa.add_property('sigma', stride=limit)
    pa.sigma[:] = 0.

    # pa.add_property('sigma_ii')
    # pa.sigma_ii[:] = 0.

    # # rate of the stress tensor (Jaumann stress rate)
    # pa.add_property('sigma_dot', stride=limit)
    # pa.sigma_dot[:] = 0.

    # pa.add_property('deviatoric_stress', stride=limit)
    # pa.deviatoric_stress[:] = 0.

    # # spin rate tensor
    # pa.add_property('omega_dot', stride=limit)
    # pa.omega_dot[:] = 0.

    # # total strain rate tensor
    # # [eps_xx, eps_xy, eps_xz, eps_yx, eps_yy, eps_yz, eps_zx, eps_zy, eps_zz]
    # pa.add_property('epsilon_dot', stride=limit)
    # pa.epsilon_dot[:] = 0.

    # pa.add_property('epsilon_dot_ii')
    # pa.epsilon_dot_ii[:] = 0.

    # # deviatoric strain rate tensor
    # # [e_xx, e_xy, e_xz, e_yx, e_yy, e_yz, e_zx, e_zy, e_zz]
    # pa.add_property('e_dot', stride=limit)
    # pa.e_dot[:] = 0.

    # # first invariant of the tensor
    # pa.add_property('I1')
    # # second invariant of the tensor
    # pa.add_property('J2')

    # # plasticity constant lambda
    # pa.add_property('lambda_dot')

    # # alpha_phi and k_c are Drucker-Prager's constants (eq 18). These will be
    # # computed from Couloumb's material constants c (cohesion) and phi (internal
    # # friction)
    # pa.add_constant('alpha_phi', 0.)
    # pa.add_constant('k_c', 0.)

    # # bulk modulus and shear modulus
    # pa.add_constant('K', 0.)
    # pa.add_constant('G', 0.)
    # pa.add_constant('nu', 0.)


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

    def loop(self, d_idx, d_epsilon_dot, d_omega_dot, d_u, d_v, d_w, s_idx, s_m, s_rho, s_u,
             s_v, s_w, d_G, DWIJ):
        tmp = s_m[s_idx] / s_rho[s_idx]

        # the following three rates are for epsilon xx, xy, xz
        d_epsilon_dot[9 * d_idx + 0] += (
            tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[0] +
            tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[0])
        d_epsilon_dot[9 * d_idx + 1] += (
            tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[1] +
            tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[0])
        d_epsilon_dot[9 * d_idx + 2] += (
            tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[2] +
            tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[0])

        # the following three rates are for epsilon yx, yy, yz
        d_epsilon_dot[9 * d_idx + 3] += (
            tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[0] +
            tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[1])
        d_epsilon_dot[9 * d_idx + 4] += (
            tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[1] +
            tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[1])
        d_epsilon_dot[9 * d_idx + 5] += (
            tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[2] +
            tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[1])

        # the following three rates are for epsilon zx, zy, zz
        d_epsilon_dot[9 * d_idx + 6] += (
            tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[0] +
            tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[2])
        d_epsilon_dot[9 * d_idx + 7] += (
            tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[1] +
            tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[2])
        d_epsilon_dot[9 * d_idx + 8] += (
            tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[2] +
            tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[2])

        # the following three rates are for omega xx, xy, xz
        d_omega_dot[9 * d_idx + 0] += (
            tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[0] -
            tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[0])
        d_omega_dot[9 * d_idx + 1] += (
            tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[1] -
            tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[0])
        d_omega_dot[9 * d_idx + 2] += (
            tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[2] -
            tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[0])

        # the following three rates are for omega yx, yy, yz
        d_omega_dot[9 * d_idx + 3] += (
            tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[0] -
            tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[1])
        d_omega_dot[9 * d_idx + 4] += (
            tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[1] -
            tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[1])
        d_omega_dot[9 * d_idx + 5] += (
            tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[2] -
            tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[1])

        # the following three rates are for omega zx, zy, zz
        d_omega_dot[9 * d_idx + 6] += (
            tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[0] -
            tmp * (s_u[s_idx] - d_u[d_idx]) * DWIJ[2])
        d_omega_dot[9 * d_idx + 7] += (
            tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[1] -
            tmp * (s_v[s_idx] - d_v[d_idx]) * DWIJ[2])
        d_omega_dot[9 * d_idx + 8] += (
            tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[2] -
            tmp * (s_w[s_idx] - d_w[d_idx]) * DWIJ[2])

    def post_looop(self, d_idx, d_epsilon_dot, d_epsilon_dot_ii, d_omega_dot,
                   d_e_dot):
        didx9 = declare('int')
        i = declare('int')
        didx9 = 9 * d_idx

        for i in range(didx9, didx9 + 9):
            d_epsilon_dot[i] = d_epsilon_dot[i] / 2.
            d_omega_dot[i] = d_omega_dot[i] / 2.

        d_epsilon_dot_ii[d_idx] = d_epsilon_dot[didx9] + d_epsilon_dot[
            didx9 + 4] + d_epsilon_dot[didx9 + 8]

        # for i in range(didx9, didx9 + 9):
        #     # Equation 13 describes what deviatoric strain is
        #     d_e_dot[i] = d_epsilon_dot[i] - 1. / 3. * epsilon_dot_ii

        d_e_dot[didx9 + 0] = (
            d_epsilon_dot[didx9 + 0] - 1. / 3. * d_epsilon_dot_ii[d_idx])
        d_e_dot[didx9 + 1] = d_epsilon_dot[didx9 + 1]
        d_e_dot[didx9 + 2] = d_epsilon_dot[didx9 + 2]
        d_e_dot[didx9 + 3] = d_epsilon_dot[didx9 + 3]
        d_e_dot[didx9 + 4] = (
            d_epsilon_dot[didx9 + 4] - 1. / 3. * d_epsilon_dot_ii[d_idx])
        d_e_dot[didx9 + 5] = d_epsilon_dot[didx9 + 5]
        d_e_dot[didx9 + 6] = d_epsilon_dot[didx9 + 6]
        d_e_dot[didx9 + 7] = d_epsilon_dot[didx9 + 7]
        d_e_dot[didx9 + 8] = (
            d_epsilon_dot[didx9 + 8] - 1. / 3. * d_epsilon_dot_ii[d_idx])


class StressRateBiu2008(Equation):
    r"""Equation 27 of reference (2) for

    Equation 11 in reference (3) for partial look up

    Stress-strain relationship of the Drucker-Prager elastic-perfectly plastic
    soil model with associatedflow rule.

    """

    def initialize(self, d_idx, d_alpha_phi, d_K, d_epsilon_dot_ii,
                   d_deviatoric_stress, d_epsilon_dot, d_G, d_J2, d_lambda_dot,
                   d_sigma_ii, d_sigma, d_sigma_dot):
        didx9 = declare('int')
        i = declare('int')
        didx9 = 9 * d_idx

        # -------------------------------------------
        # compute the deviatoric stress from the stress tensor
        # -------------------------------------------
        # find the mean stress
        d_sigma_ii[d_idx] = (
            d_sigma[didx9] + d_sigma[didx9 + 4] + d_sigma[didx9 + 8])

        # deviatoric stress
        d_deviatoric_stress[didx9 + 0] = (
            d_sigma[didx9 + 0] - 1. / 3. * d_sigma_ii[d_idx])
        d_deviatoric_stress[didx9 + 1] = d_sigma[didx9 + 1]
        d_deviatoric_stress[didx9 + 2] = d_sigma[didx9 + 2]
        d_deviatoric_stress[didx9 + 3] = d_sigma[didx9 + 3]
        d_deviatoric_stress[didx9 + 4] = (
            d_sigma[didx9 + 4] - 1. / 3. * d_sigma_ii[d_idx])
        d_deviatoric_stress[didx9 + 5] = d_sigma[didx9 + 5]
        d_deviatoric_stress[didx9 + 6] = d_sigma[didx9 + 6]
        d_deviatoric_stress[didx9 + 7] = d_sigma[didx9 + 7]
        d_deviatoric_stress[didx9 + 8] = (
            d_sigma[didx9 + 8] - 1. / 3. * d_sigma_ii[d_idx])
        # -------------------------------------------
        # compute the deviatoric stress from the stress tensor
        # -------------------------------------------

        # -------------------------------------------
        # compute the placticity multiplier
        # this lamda is taken from equation 27 of reference 2
        # -------------------------------------------
        numerator_tmp1 = 3. * d_alpha_phi[0] * d_K[0] * d_epsilon_dot_ii[d_idx]

        numerator_tmp2a = 0.
        for i in range(didx9, didx9 + 9):
            numerator_tmp2a += d_deviatoric_stress[i] * d_epsilon_dot[i]

        numerator_tmp2 = (d_G[0] / d_J2[d_idx]**0.5) * numerator_tmp2a

        num_tmp = numerator_tmp1 + numerator_tmp2

        denom_tmp = 9. * d_alpha_phi[d_idx]**2. * d_K[0] + d_G[0]

        d_lambda_dot[d_idx] = num_tmp / denom_tmp
        # -------------------------------------------
        # compute the placticity multiplier ENDS
        # -------------------------------------------

        # -------------------------------------------
        # finally compute the stress rate
        # -------------------------------------------
        tmp1 = 2. * d_G[0]
        tmp2 = d_lambda_dot[d_idx] * d_G[0] / d_J2[d_idx]**0.5
        for i in range(didx9, didx9 + 9):
            d_sigma_dot[i] = (
                tmp1 * d_epsilon_dot[i] - tmp2 * d_deviatoric_stress[i])

        # add terms which are only influence the diagonal
        tmp3 = d_K[0] * d_epsilon_dot_ii[d_idx] - d_lambda_dot[
            d_idx] * 3. * d_alpha_phi[0] * d_K[0]
        d_sigma_dot[didx9 + 0] += tmp3
        d_sigma_dot[didx9 + 4] += tmp3
        d_sigma_dot[didx9 + 8] += tmp3
        # -------------------------------------------
        # finally compute the stress rate ENDS
        # -------------------------------------------


class MomentumEquationWithStress(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_sigma, d_rho, d_au, d_av, d_aw, s_idx, s_m,
             s_sigma, s_rho, DWIJ):
        didx9 = declare('int')
        didx9 = 9 * d_idx

        sigma = declare('double*')
        sigma = address(d_sigma[didx9])

        # compute accelerations
        mb = s_m[s_idx]
        rhoa21 = 1. / d_rho[d_idx]**2.
        rhob21 = 1. / s_rho[s_idx]**2.

        d_au[d_idx] += (
            mb * (d_sigma[didx9 + 0] * rhoa21 + d_sigma[didx9 + 0] * rhob21) *
            DWIJ[0] + mb * (d_sigma[didx9 + 1] * rhoa21 +
                            d_sigma[didx9 + 1] * rhob21) * DWIJ[1] +
            mb * (d_sigma[didx9 + 2] * rhoa21 + d_sigma[didx9 + 2] * rhob21) *
            DWIJ[2])

        d_av[d_idx] += (
            mb * (d_sigma[didx9 + 3] * rhoa21 + d_sigma[didx9 + 3] * rhob21) *
            DWIJ[0] + mb * (d_sigma[didx9 + 4] * rhoa21 +
                            d_sigma[didx9 + 4] * rhob21) * DWIJ[1] +
            mb * (d_sigma[didx9 + 5] * rhoa21 + d_sigma[didx9 + 5] * rhob21) *
            DWIJ[2])

        d_aw[d_idx] += (
            mb * (d_sigma[didx9 + 6] * rhoa21 + d_sigma[didx9 + 6] * rhob21) *
            DWIJ[0] + mb * (d_sigma[didx9 + 7] * rhoa21 +
                            d_sigma[didx9 + 7] * rhob21) * DWIJ[1] +
            mb * (d_sigma[didx9 + 8] * rhoa21 + d_sigma[didx9 + 8] * rhob21) *
            DWIJ[2])


class GranularFlowRK2(IntegratorStep):
    def initialize(self, d_x, d_y, d_z, d_u, d_v, d_w, d_rho, d_x0, d_y0, d_z0,
                   d_u0, d_v0, d_w0, d_rho0, d_sigma, d_sigma0, d_idx):
        i = declare('int')
        didx9 = declare('int')
        didx9 = 9 * d_idx

        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

        for i in range(didx9, didx9 + 9):
            d_sigma0[i] = d_sigma[i]

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_rho, d_au, d_av,
               d_aw, d_arho, d_sigma, d_sigma0, d_sigma_dot, dt):
        i = declare('int')
        didx9 = declare('int')
        didx9 = 9 * d_idx

        dtb2 = dt * 0.5

        d_x[d_idx] = d_x[d_idx] + d_u[d_idx] * dtb2
        d_y[d_idx] = d_y[d_idx] + d_v[d_idx] * dtb2
        d_z[d_idx] = d_z[d_idx] + d_w[d_idx] * dtb2

        d_u[d_idx] = d_u[d_idx] + d_au[d_idx] * dtb2
        d_v[d_idx] = d_v[d_idx] + d_av[d_idx] * dtb2
        d_w[d_idx] = d_w[d_idx] + d_aw[d_idx] * dtb2

        d_rho[d_idx] = d_rho[d_idx] + d_arho[d_idx] * dtb2

        for i in range(didx9, didx9 + 9):
            d_sigma[i] = d_sigma[i] + d_sigma_dot[i] * dtb2

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_rho, d_x0, d_y0,
               d_z0, d_u0, d_v0, d_w0, d_rho0, d_au, d_av, d_aw, d_arho,
               d_sigma, d_sigma0, d_sigma_dot, dt):
        i = declare('int')
        didx9 = declare('int')
        didx9 = 9 * d_idx

        d_x[d_idx] = d_x0[d_idx] + d_u[d_idx] * dt
        d_y[d_idx] = d_y0[d_idx] + d_v[d_idx] * dt
        d_z[d_idx] = d_z0[d_idx] + d_w[d_idx] * dt

        d_u[d_idx] = d_u0[d_idx] + d_au[d_idx] * dt
        d_v[d_idx] = d_v0[d_idx] + d_av[d_idx] * dt
        d_w[d_idx] = d_w0[d_idx] + d_aw[d_idx] * dt

        d_rho[d_idx] = d_rho0[d_idx] + d_arho[d_idx] * dt

        for i in range(didx9, didx9 + 9):
            d_sigma[i] = d_sigma0[i] + d_sigma_dot[i] * dt
