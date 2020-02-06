"""
Corrected TVF equations
###################
"""

from pysph.sph.equation import Equation
from compyle.api import declare


def add_ctvf_properties(pa):
    # for free surface reference density
    pa.add_property('rho_tmp')

    # gradient of density
    pa.add_property('grad_rho_x')
    pa.add_property('grad_rho_y')
    pa.add_property('grad_rho_z')

    # minimum rho of a neighbour
    pa.add_property('min_neigh_rho')

    # accelerations
    pa.add_property('auhat')
    pa.add_property('avhat')
    pa.add_property('awhat')


class SummationDensityTmp(Equation):
    def initialize(self, d_idx, d_rho_tmp):
        d_rho_tmp[d_idx] = 0.0

    def loop(self, d_idx, d_rho_tmp, s_idx, s_m, WIJ):
        d_rho_tmp[d_idx] += s_m[s_idx] * WIJ


class GradientRhoTmp(Equation):
    def initialize(self, d_idx, d_grad_rho_x, d_grad_rho_y, d_grad_rho_z):
        d_grad_rho_x[d_idx] = 0.0
        d_grad_rho_y[d_idx] = 0.0
        d_grad_rho_z[d_idx] = 0.0

    def loop(self, d_idx, d_rho_tmp, d_grad_rho_x, d_grad_rho_y, d_grad_rho_z,
             s_idx, s_rho_tmp, s_m, DWIJ):
        rhoi21 = 1.0 / (d_rho_tmp[d_idx] * d_rho_tmp[d_idx])
        rhoj21 = 1.0 / (s_rho_tmp[s_idx] * s_rho_tmp[s_idx])

        tmp_1 = d_rho_tmp[d_idx] * rhoi21 + s_rho_tmp[s_idx] * rhoj21

        tmp_2 = d_rho_tmp[d_idx] * s_m[s_idx] * tmp_1

        d_grad_rho_x[d_idx] += tmp_2 * DWIJ[0]
        d_grad_rho_y[d_idx] += tmp_2 * DWIJ[1]
        d_grad_rho_z[d_idx] += tmp_2 * DWIJ[2]


class MinNeighbourRho(Equation):
    def initialize(self, d_idx, d_min_neigh_rho, d_rho_tmp):
        d_min_neigh_rho[d_idx] = d_rho_tmp[d_idx]

    def loop(self, d_idx, d_min_neigh_rho, d_rho_tmp, s_idx, s_rho_tmp, s_m,
             DWIJ):
        if s_rho_tmp[s_idx] < d_min_neigh_rho[d_idx]:
            d_min_neigh_rho[d_idx] = s_rho_tmp[s_idx]


class MomentumEquationPressureGradientCTVF(Equation):
    def __init__(self, dest, sources, pb, rho, gx=0.0, gy=0.0, gz=0.0):
        self.pb = pb
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.rho = rho
        super(MomentumEquationPressureGradientCTVF, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_auhat, d_avhat, d_awhat,
                   d_p):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_rho, s_rho, d_idx, d_rho_tmp, s_idx, d_p, s_p, s_m, d_au, d_av,
             d_aw, DWIJ, d_p0, d_auhat, d_avhat, d_awhat, XIJ, RIJ, SPH_KERNEL,
             HIJ):
        dwijhat = declare('matrix(3)')

        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p[d_idx]/rhoi2 + s_p[s_idx]/rhoj2

        tmp = -s_m[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]

        # if d_rho_tmp[d_idx] > 0.95 * self.rho:

        tmp = -self.pb * s_m[s_idx]/rhoi2

        SPH_KERNEL.gradient(XIJ, RIJ, HIJ, dwijhat)

        d_auhat[d_idx] += tmp * dwijhat[0]
        d_avhat[d_idx] += tmp * dwijhat[1]
        d_awhat[d_idx] += tmp * dwijhat[2]


class PsuedoForceOnFreeSurface(Equation):
    def __init__(self, dest, sources, dx, m0, pb, rho):
        self.dx = dx
        self.m0 = m0
        self.pb = pb
        self.rho = rho
        super(PsuedoForceOnFreeSurface, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_x, d_y, d_z, d_rho, d_rho_tmp,
                   d_grad_rho_x, d_grad_rho_y, d_grad_rho_z, d_min_neigh_rho,
                   d_auhat, d_avhat, d_awhat, d_h, SPH_KERNEL):
        i = declare('int')
        j = declare('int')
        xij = declare('matrix(3)')
        DWIJ = declare('matrix(3)')
        dwijhat = declare('matrix(3)')

        # do this only if it is comes under a free surface
        if d_rho_tmp[d_idx] < 0.95 * self.rho:
            for i in range(-3, 3):
                # dummy x
                x_d = d_x[d_idx] + i * self.dx
                for j in range(-3, 3):
                    # dummy y
                    y_d = d_y[d_idx] + j * self.dx

                    # find the density of the point using Taylor's approximation
                    dx = d_x[d_idx] - x_d
                    dy = d_y[d_idx] - y_d

                    gdotdr = -(d_grad_rho_x[d_idx] * dx +
                               d_grad_rho_y[d_idx] * dy)

                    # dummy density with Taylor expansion
                    rho_d = d_rho_tmp[d_idx] + gdotdr

                    # if the density is less than the minimum then compute the force
                    # if rho_d < d_min_neigh_rho[d_idx]:
                    if rho_d < d_rho_tmp[d_idx]:
                        rij = (dx**2. + dy**2.)**0.5
                        xij[0] = dx
                        xij[1] = dy
                        xij[2] = 0.

                        SPH_KERNEL.gradient(xij, rij, d_h[d_idx], DWIJ)

                        rhoi2 = d_rho[d_idx]**2.

                        tmp = -self.pb * self.m0 / rhoi2

                        SPH_KERNEL.gradient(xij, rij, d_h[d_idx], dwijhat)

                        d_auhat[d_idx] += tmp * dwijhat[0]
                        d_avhat[d_idx] += tmp * dwijhat[1]
                        d_awhat[d_idx] += tmp * dwijhat[2]
