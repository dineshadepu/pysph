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

    # for normals
    pa.add_property('normal', stride=3)
    pa.add_property('normal_tmp', stride=3)
    pa.add_property('normal_norm')

    # check for boundary particle
    pa.add_property('is_boundary', type='int')
    pa.add_property('is_boundary_second', type='int')


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
        super(MomentumEquationPressureGradientCTVF,
              self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_auhat, d_avhat, d_awhat,
                   d_p):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_rho, s_rho, d_idx, d_rho_tmp, s_idx, d_p, s_p, s_m, d_au,
             d_av, d_aw, DWIJ, d_p0, d_auhat, d_avhat, d_awhat, XIJ, RIJ,
             SPH_KERNEL, d_normal, d_normal_norm, HIJ):
        dwijhat = declare('matrix(3)')
        idx3 = declare('int')
        idx3 = 3 * d_idx

        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p[d_idx] / rhoi2 + s_p[s_idx] / rhoj2

        tmp = -s_m[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]

        # compute the magnitude of the normal

        normal_norm = (d_normal[idx3]**2. + d_normal[idx3 + 1]**2. +
                       d_normal[idx3 + 2]**2.)

        d_normal_norm[d_idx] = normal_norm

        if normal_norm < 1e-12:
            tmp = -self.pb * s_m[s_idx] / rhoi2

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


class IdentifyBoundaryParticle1(Equation):
    def __init__(self, dest, sources, fac):
        self.fac = fac
        super(IdentifyBoundaryParticle1, self).__init__(dest, sources)

    def initialize(self, d_idx, d_is_boundary):
        # set all of them to be boundary
        d_is_boundary[d_idx] = 0

    def loop_all(self, d_idx, d_x, d_y, d_z, d_rho, d_h, d_is_boundary,
                 d_normal, d_normal_norm, s_m, s_x, s_y, s_z, s_h, SPH_KERNEL,
                 NBRS, N_NBRS):
        i = declare('int')
        idx3 = declare('int')
        s_idx = declare('int')
        xij = declare('matrix(3)')
        idx3 = 3 * d_idx

        normal_norm = (d_normal[idx3]**2. + d_normal[idx3 + 1]**2. +
                       d_normal[idx3 + 2]**2.)

        d_normal_norm[d_idx] = normal_norm

        if d_normal_norm[d_idx] > 1e-12:
            d_is_boundary[d_idx] = 1

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            if s_idx != d_idx:
                xij[0] = d_x[d_idx] - s_x[s_idx]
                xij[1] = d_y[d_idx] - s_y[s_idx]
                xij[2] = d_z[d_idx] - s_z[s_idx]
                # rij = xij[0]**2. + xij[1]**2. + xij[2]**2.

                if d_is_boundary[d_idx] == 1:
                    xijdotnormal = -(d_normal[idx3] * xij[0] +
                                     d_normal[idx3 + 1] * xij[1] +
                                     d_normal[idx3 + 2] * xij[2])

                    if xijdotnormal > self.fac:
                        d_is_boundary[d_idx] = 0
                        break


class IdentifyBoundaryParticle2(Equation):
    def __init__(self, dest, sources, fac):
        self.fac = fac
        super(IdentifyBoundaryParticle2, self).__init__(dest, sources)

    def initialize(self, d_idx, d_is_boundary, d_is_boundary_second):
        # set all of them to be boundary
        d_is_boundary_second[d_idx] = d_is_boundary[d_idx]

    def loop_all(self, d_idx, d_x, d_y, d_z, d_rho, d_h, d_is_boundary, d_is_boundary_second,
                 d_normal, d_normal_norm, s_m, s_x, s_y, s_z, s_h, s_is_boundary, SPH_KERNEL,
                 NBRS, N_NBRS):
        i = declare('int')
        idx3 = declare('int')
        s_idx = declare('int')
        xij = declare('matrix(3)')
        idx3 = 3 * d_idx

        normal_norm = (d_normal[idx3]**2. + d_normal[idx3 + 1]**2. +
                       d_normal[idx3 + 2]**2.)

        d_normal_norm[d_idx] = normal_norm

        if d_normal_norm[d_idx] > 1e-12:
            # if it is not a boundary
            if d_is_boundary[d_idx] == 0:
                for i in range(N_NBRS):
                    s_idx = NBRS[i]
                    if s_idx != d_idx:
                        xij[0] = d_x[d_idx] - s_x[s_idx]
                        xij[1] = d_y[d_idx] - s_y[s_idx]
                        xij[2] = d_z[d_idx] - s_z[s_idx]
                        xijdotnormal = -(d_normal[idx3] * xij[0] +
                                         d_normal[idx3 + 1] * xij[1] +
                                         d_normal[idx3 + 2] * xij[2])

                        if s_is_boundary[s_idx] == 1 and xijdotnormal < self.fac:
                            d_is_boundary_second[d_idx] = 1
                            break

    def post_loop(self, d_idx, d_is_boundary, d_is_boundary_second):
        # set all of them to be boundary
        d_is_boundary[d_idx] = d_is_boundary_second[d_idx]


# normal[::3], normal[1::3], normal[2::3]
# grad_rho_x, grad_rho_y, grad_rho_z
# auhat, avhat, awhat
