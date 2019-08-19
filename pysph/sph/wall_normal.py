from math import sqrt
from compyle.api import declare

from pysph.sph.equation import Equation


class ComputeNormals(Equation):
    """Compute normals using a simple approach

    .. math::

       -\frac{m_j}{\rho_j} DW_{ij}

    First compute the normals, then average them and finally normalize them.

    """

    def initialize(self, d_idx, d_normal_tmp_x, d_normal_tmp_y, d_normal_tmp_z,
                   d_normal_x, d_normal_y, d_normal_z):
        d_normal_x[d_idx] = 0.0
        d_normal_y[d_idx] = 0.0
        d_normal_z[d_idx] = 0.0
        d_normal_tmp_x[d_idx] = 0.0
        d_normal_tmp_y[d_idx] = 0.0
        d_normal_tmp_z[d_idx] = 0.0

    def loop(self, d_idx, d_normal_tmp_x, d_normal_tmp_y, d_normal_tmp_z,
             s_idx, s_m, s_rho, DWIJ):
        fac = -s_m[s_idx] / s_rho[s_idx]

        d_normal_tmp_x[d_idx] += fac * DWIJ[0]
        d_normal_tmp_y[d_idx] += fac * DWIJ[1]
        d_normal_tmp_z[d_idx] += fac * DWIJ[2]

    def post_loop(self, d_idx, d_normal_tmp_x, d_normal_tmp_y, d_normal_tmp_z,
                  d_h):
        mag = sqrt(d_normal_tmp_x[d_idx]**2 + d_normal_tmp_y[d_idx]**2 +
                   d_normal_tmp_z[d_idx]**2)
        if mag > 0.25 / d_h[d_idx]:
            d_normal_tmp_x[d_idx] /= mag
            d_normal_tmp_y[d_idx] /= mag
            d_normal_tmp_z[d_idx] /= mag
        else:
            d_normal_tmp_x[d_idx] = 0.
            d_normal_tmp_y[d_idx] = 0.
            d_normal_tmp_z[d_idx] = 0.


class SmoothNormals(Equation):
    def loop(self, d_idx, d_normal_x, d_normal_y, d_normal_z, s_normal_tmp_x,
             s_normal_tmp_y, s_normal_tmp_z, s_idx, s_m, s_rho, WIJ):
        fac = s_m[s_idx] / s_rho[s_idx] * WIJ
        d_normal_x[d_idx] += fac * s_normal_tmp_x[s_idx]
        d_normal_y[d_idx] += fac * s_normal_tmp_y[s_idx]
        d_normal_z[d_idx] += fac * s_normal_tmp_z[s_idx]

    def post_loop(self, d_idx, d_normal_x, d_normal_y, d_normal_z, d_h):
        mag = sqrt(d_normal_x[d_idx]**2 + d_normal_y[d_idx]**2 +
                   d_normal_z[d_idx]**2)
        if mag > 1e-3:
            d_normal_x[d_idx] /= mag
            d_normal_y[d_idx] /= mag
            d_normal_z[d_idx] /= mag
        else:
            d_normal_x[d_idx] = 0.0
            d_normal_y[d_idx] = 0.0
            d_normal_z[d_idx] = 0.0
