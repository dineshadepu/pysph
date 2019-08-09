from pysph.sph.equation import Equation


class ContinuityEquation(Equation):
    r"""Density rate:

    :math:`\frac{d\rho_a}{dt} = \sum_b m_b \boldsymbol{v}_{ab}\cdot
    \nabla_a W_{ab}`

    """

    def __init__(self, dest, sources, c0, delta=0.2):
        r"""
        Parameters
        ----------
        rho0 : float
            Reference density of the fluid (:math:`\rho_0`)
        c0 : float
            Maximum speed of sound expected in the system (:math:`c0`)
        p0 : float
            Reference pressure in the system (:math:`p0`)
        """

        self.c0 = c0
        self.delta = delta
        super(ContinuityEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_arho, s_idx, s_m, s_rho, s_V, DWIJ, VIJ,
             XIJ, R2IJ, HIJ):
        tmp1 = d_rho[d_idx] - s_rho[s_idx]
        tmp2 = R2IJ + (0.01 * HIJ)**2.
        tmp = tmp1 / tmp2
        psi_ijdotdwij = tmp * (
            XIJ[0] * DWIJ[0] + XIJ[1] * DWIJ[1] + XIJ[2] * DWIJ[2])

        vijdotdwij = DWIJ[0] * VIJ[0] + DWIJ[1] * VIJ[1] + DWIJ[2] * VIJ[2]

        d_arho[d_idx] += d_rho[d_idx] * vijdotdwij * s_V[s_idx]

        # diffusive term
        tmp3 = self.delta * self.c0 * HIJ
        d_arho[d_idx] += tmp3 * psi_ijdotdwij * s_V[s_idx]


class MomentumEquation(Equation):
    def __init__(self, dest, sources, c0, rho0, dim, nu=0.0):
        self.c0 = c0
        self.alpha_fac = 2. * (dim + 2) * nu
        self.rho0 = rho0
        self.dim = dim
        super(MomentumEquation, self).__init__(dest, sources)

    def initialize(self, d_au, d_av, d_aw, d_idx):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_au, d_av, d_aw, s_V, d_p, s_p, DWIJ, s_idx,
             d_m, R2IJ, XIJ, EPS, VIJ, d_nu, s_nu, d_rho, s_rho, HIJ):
        p_ij = (s_rho[s_idx] * d_p[d_idx] + d_rho[d_idx] * s_p[s_idx]) / (
            d_rho[d_idx] * s_rho[s_idx])
        uijdotxij = XIJ[0] * VIJ[0] + XIJ[1] * VIJ[1] + XIJ[2] * VIJ[2]
        tmp2 = R2IJ + (0.01 * HIJ)**2.

        # artificial viscosity coefficient
        pi_ij = uijdotxij / tmp2
        mi_1 = 1. / d_m[d_idx]
        rhoi_1 = 1. / d_m[d_idx]
        tmp1 = self.alpha_fac * self.rho0 * rhoi_1 * pi_ij

        factor = (d_V[d_idx]**2. + s_V[s_idx]**2.) * p_ij
        d_au[d_idx] += (-mi_1 * factor * DWIJ[0]) + (
            tmp1 * DWIJ[0] * s_V[s_idx])
        d_av[d_idx] += (-mi_1 * factor * DWIJ[1]) + (
            tmp1 * DWIJ[1] * s_V[s_idx])
        d_aw[d_idx] += (-mi_1 * factor * DWIJ[2]) + (
            tmp1 * DWIJ[2] * s_V[s_idx])


class StateEquation(Equation):
    def __init__(self, dest, sources, c0, rho0):
        self.c0 = c0
        self.c0_2 = c0**2.
        self.rho0 = rho0
        super(StateEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_rho):
        d_p[d_idx] = self.c0_2 * (d_rho[d_idx] - self.rho0)
