from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from pysph.base.kernels import CubicSpline
import numpy as np
import numpy
from pysph.sph.scheme import Scheme

from compyle.api import declare
from math import sqrt, asin, sin, cos, pi, log
import sys


def get_particle_array_continuous_dem_spring_model(constants=None, **props):
    """Return a particle array for a continuous dem particles, where the
    equations govern the structure belong to spring model [1][2].

    [1] A coupled SPH-DEM model for fluid-structure interaction problems with
    free-surface flow and structural failure
    [2] Code Implementation Of Particle Based Discrete Element Method For
    Concrete Viscoelastic Modeling

    """
    dim = props.pop('dim', None)
    clrnc = props.pop('clrnc', None)
    if clrnc is None:
        print("Some clearance value is needed to create bonds")
        sys.exit()

    dem_props = [
        'wx', 'wy', 'wz', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'rad_s',
        'm_inverse', 'I_inverse'
    ]

    dem_id = props.pop('dem_id', None)

    pa = get_particle_array(additional_props=dem_props, **props)

    if pa.rad_s[0] == 0.:
        print("Radius of the particle has to be greater than zero")
        sys.exit()

    pa.add_property('dem_id', type='int', data=dem_id)
    # create the array to save the tangential interaction particles
    # index and other variables
    if dim == 3:
        bc_limit = 30
    elif dim == 2 or dim is None:
        bc_limit = 6

    # Bonded contacts (bc) will be tracked, these will be created at once and
    # will never leave out of contact unless there is a bond breakage.

    pa.add_constant('bc_limit', bc_limit)
    pa.add_constant('bc_idx', [-1] * bc_limit * len(pa.x))
    pa.add_constant('bc_total_contacts', [0] * len(pa.x))
    pa.add_constant('bc_rest_len', [0.] * bc_limit * len(pa.x))
    pa.add_constant('bc_ft_x', [0.] * bc_limit * len(pa.x))
    pa.add_constant('bc_ft_y', [0.] * bc_limit * len(pa.x))
    pa.add_constant('bc_ft_x0', [0.] * bc_limit * len(pa.x))
    pa.add_constant('bc_ft_y0', [0.] * bc_limit * len(pa.x))

    # setup the contacts
    from pysph.tools.sph_evaluator import SPHEvaluator
    equations = [
        SetupBondedContacts(dest=pa.name, sources=[pa.name], clrnc=clrnc)
    ]
    sph_eval = SPHEvaluator(
        arrays=[pa], equations=equations, dim=dim,
        kernel=CubicSpline(dim=dim),
    )
    sph_eval.evaluate()

    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'wx', 'wy', 'wz', 'm', 'p', 'pid', 'tag',
        'gid', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'I_inverse'
    ])

    return pa


class SetupBondedContacts(Equation):
    def __init__(self, dest, sources, clrnc):
        self.clrnc = clrnc
        super(SetupBondedContacts, self).__init__(dest, sources)

    def loop(self, d_idx, d_bc_idx, d_bc_total_contacts, s_idx, RIJ,
             d_rad_s, d_bc_limit, d_bc_rest_len, s_rad_s):
        tot_ctcs, strt_idx, tmp_idx = declare('int', 3)
        if RIJ != 0:
            tmp = RIJ / (d_rad_s[d_idx] + s_rad_s[s_idx])
            tmp1 = 1. - self.clrnc
            tmp2 = 1. + self.clrnc
            if tmp1 <= tmp <= tmp2:
                strt_idx = d_idx * d_bc_limit[0]
                tot_ctcs = d_bc_total_contacts[d_idx]
                tmp_idx = strt_idx + tot_ctcs
                d_bc_idx[tmp_idx] = s_idx
                d_bc_rest_len[tmp_idx] = RIJ
                d_bc_total_contacts[d_idx] += 1


class BondedDEMPotyndyForce(Equation):
    def __init__(self, dest, sources, kn):
        self.kn = kn
        super(BondedDEMPotyndyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_tng_contacts, d_x, d_y, d_bc_ft_x,
                   d_bc_ft_y, d_bc_ft_z, d_bc_limit, d_bc_rest_len, d_fx,
                   d_fy, d_u, d_v):
        p, q1, tot_ctcs, i, found_at, found = declare('int', 6)
        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_tng_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        for i in range(p, q1):
            overlap = -1.
            xij = declare('matrix(3)')
            rij = 0.0

            xij[0] = d_x[d_idx] - d_x[i]
            xij[1] = d_y[d_idx] - d_y[i]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

            # normal vector from i to j
            nji_x = xij[0] / rij
            nji_y = xij[1] / rij

            rinv = 1. / rij
            # check the particles are not on top of each other.
            if rij > 0:
                overlap = rij - d_bc_rest_len[i]
            # ---------- force computation starts ------------
            d_fx[d_idx] += self.kn * nji_x * overlap
            d_fy[d_idx] += self.kn * nji_y * overlap

            ft_x = d_bc_ft_x[i]
            ft_y = d_bc_ft_y[i]

            # TODO
            # check the coulomb criterion

            # Add the tangential force
            d_fx[d_idx] += ft_x
            d_fy[d_idx] += ft_y

            # --------- Tangential force -----------
            # -----increment the tangential force---
            # relative velocity
            vr_x = d_u[d_idx] - d_u[i]
            vr_y = d_v[d_idx] - d_v[i]

            # normal velocity magnitude
            vr_dot_nij = vr_x * nji_x + vr_y * nji_y
            vn_x = vr_dot_nij * nji_x
            vn_y = vr_dot_nij * nji_y

            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y

            # magnitude of the tangential velocity
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5


class BondedDEMPotyndyScheme(Scheme):
    def __init__(self, beam, solids, dim, clearence=0.1):
        self.beam = beam
        self.dim = dim
        self.solver = None
        self.clearence = clearence

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import (
            ContinuityEquation, MonaghanArtificialViscosity, XSPHCorrection,
            VelocityGradient2D)
        from pysph.sph.solid_mech.basic import (
            IsothermalEOS, MomentumEquationWithStress,
            HookesDeviatoricStressRate, MonaghanArtificialStress)

        equations = []
        g1 = []
        all = self.solids + self.elastic_solids
        for elastic_solid in self.elastic_solids:
            g1.append(
                # p
                IsothermalEOS(elastic_solid, sources=None))
            g1.append(
                # vi,j : requires properties v00, v01, v10, v11
                VelocityGradient2D(dest=elastic_solid, sources=all))
            g1.append(
                # rij : requires properties r00, r01, r02, r11, r12, r22,
                #                           s00, s01, s02, s11, s12, s22
                MonaghanArtificialStress(dest=elastic_solid, sources=None,
                                         eps=self.artificial_stress_eps))

        equations.append(Group(equations=g1))

        g2 = []
        for elastic_solid in self.elastic_solids:
            g2.append(ContinuityEquation(dest=elastic_solid, sources=all), )
            g2.append(
                # au, av
                MomentumEquationWithStress(dest=elastic_solid, sources=all), )
            g2.append(
                # au, av
                MonaghanArtificialViscosity(dest=elastic_solid, sources=all,
                                            alpha=self.alpha,
                                            beta=self.beta), )
            g2.append(
                # a_s00, a_s01, a_s11
                HookesDeviatoricStressRate(dest=elastic_solid, sources=None), )
            g2.append(
                # ax, ay, az
                XSPHCorrection(dest=elastic_solid, sources=[elastic_solid],
                               eps=self.xsph_eps), )
        equations.append(Group(g2))

        return equations

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import CubicSpline
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import EPECIntegrator
        from pysph.sph.integrator_step import SolidMechStep

        cls = integrator_cls if integrator_cls is not None else EPECIntegrator
        step_cls = SolidMechStep
        for name in self.elastic_solids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)
