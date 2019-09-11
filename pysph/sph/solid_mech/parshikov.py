from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import Integrator
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.scheme import Scheme
from compyle.api import declare
from pysph.sph.wc.linalg import mat_vec_mult, mat_mult
import numpy as np
import numpy


def get_bulk_mod(G, nu):
    ''' Get the bulk modulus from shear modulus and Poisson ratio '''
    return 2. * G * (1 + nu) / (3 * (1 - 2 * nu))


def get_speed_of_sound(E, nu, rho0):
    return np.sqrt(E / (3 * (1. - 2 * nu) * rho0))


def get_particle_array_gtvf(constants=None, **props):
    gtvf_props = [
        'uhat', 'vhat', 'what', 'rho0', 'rho_div', 'p0', 'auhat', 'avhat',
        'awhat', 'arho', 'arho0', 'rho_tmp', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0'
    ]

    # Check E and nu values in the constants
    if 'E' in constants:
        pass
    else:
        raise ValueError('could not find %c in %s' % ('E', "Constants"))

    if 'nu' in constants:
        pass
    else:
        raise ValueError('could not find %c in %s' % ('nu', "Constants"))

    consts = {
        'rho_ref': [1000.],  # initial density of the particles
        'E': [1e6],  # Young's modulus
        'nu': [0.3],  # Poisson ration
        'b_mod': [0.],  # bulk modulus
        'p_max': [0.],  # maximum pressure of all the particles
        'p_ref': [0.]  # reference pressure for gtvf
    }
    if constants:
        consts.update(constants)
    pa = get_particle_array(constants=consts, additional_props=gtvf_props,
                            **props)

    pa.add_property('strain_tensor', stride=9)
    pa.add_property('rotation_tensor', stride=9)
    # del outer product with velocity
    pa.add_property('gradvel', stride=9)

    # deviatoric stress rate
    pa.add_property('ds', stride=9)
    pa.add_property('ds0', stride=9)
    # rate of change of deviatoric stress rate
    pa.add_property('ads', stride=9)

    # compute shear modulus
    G = pa.E[0] / (2. * (1. + pa.nu[0]))
    pa.add_constant('G', G)
    # compute speed of sound
    pa.b_mod[0] = get_bulk_mod(G, pa.nu[0])
    pa.p_ref[0] = pa.b_mod[0]
    # speed of sound
    c0 = get_speed_of_sound(pa.E[0], pa.nu[0], pa.rho_ref[0])
    pa.add_constant('c0', c0)

    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'h', 'm', 'au', 'av', 'aw',
        'pid', 'gid', 'tag'
    ])
    return pa


def add_gtvf_solid_properties(pa):
    properties = [
        'uhat', 'vhat', 'what', 'rho0', 'rho_div', 'p0', 'auhat', 'avhat',
        'awhat', 'arho', 'arho0', 'cs', 'rho_tmp', 'rho_div'
    ]

    for i in properties:
        pa.add_property(i)

    stride_properties = [
        'strain_tensor', 'rotation_tensor', 'gradvel', 'ds', 'ads', 'ds0'
    ]

    for i in stride_properties:
        pa.add_property(i, stride=9)


def remove_gray_solid_properties(pa):
    properties = [
        'cs', 'e', 'v00', 'v01', 'v02', 'v10', 'v11', 'v12', 'v20', 'v21',
        'v22', 'r00', 'r01', 'r02', 'r11', 'r12', 'r22', 's00', 's01', 's02',
        's11', 's12', 's22', 'as00', 'as01', 'as02', 'as11', 'as12', 'as22',
        's000', 's010', 's020', 's110', 's120', 's220', 'ae', 'e0'
    ]

    for i in properties:
        pa.remove_property(i)


class CorrectDensity(Equation):
    def initialize(self, d_idx, d_rho, d_rho_tmp, d_rho_div):
        d_rho_tmp[d_idx] = d_rho[d_idx]
        d_rho[d_idx] = 0.0
        # denominator of the corrected density
        d_rho_div[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_rho_div, s_m, WIJ, s_rho_tmp):
        d_rho[d_idx] += s_m[s_idx] * WIJ
        d_rho_div[d_idx] += s_m[s_idx] * WIJ / s_rho_tmp[s_idx]

    def post_loop(self, d_idx, d_rho, d_rho0, d_rho_div):
        d_rho[d_idx] = d_rho[d_idx] / min(1, d_rho_div[d_idx])


class DensityEvolution(Equation):
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_uhat, d_vhat, d_what,
             s_uhat, s_vhat, s_what, d_arho, DWIJ):
        uhatij = d_uhat[d_idx] - s_uhat[s_idx]
        vhatij = d_vhat[d_idx] - s_vhat[s_idx]
        whatij = d_what[d_idx] - s_what[s_idx]

        udotdij = DWIJ[0] * uhatij + DWIJ[1] * vhatij + DWIJ[2] * whatij
        fac = d_rho[d_idx] * s_m[s_idx] / s_rho[s_idx]
        d_arho[d_idx] += fac * udotdij


class StateEquationGTVF(Equation):
    def __init__(self, dest, sources):
        super(StateEquationGTVF, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_rho, d_b_mod, d_rho_ref):
        d_p[d_idx] = d_b_mod[0] * ((d_rho[d_idx] / d_rho_ref[0]) - 1.)
        # if d_p[d_idx] < 0.:
        #     d_p[d_idx] = 0.

    def reduce(self, dst, t, dt):
        p_max = numpy.max(numpy.abs(dst.p))
        dst.p_max[0] = p_max
        dst.p_ref[0] = max(p_max, dst.b_mod[0])
        # dst.p_ref[0] = dst.b_mod[0]


class MomentumEquationSolidGTVF(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationSolidGTVF, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_auhat, d_avhat, d_awhat,
                   d_p0, d_p, d_p_max):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

        d_p0[d_idx] = min(10 * abs(d_p[d_idx]), d_p_max[0])

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, d_ds, s_ds, s_p, s_m, d_au,
             d_av, d_aw, DWIJ, d_p0, d_auhat, d_avhat, d_awhat, XIJ, RIJ,
             SPH_KERNEL, HIJ):
        rhoi21 = 1. / (d_rho[d_idx] * d_rho[d_idx])
        rhoj21 = 1. / (s_rho[s_idx] * s_rho[s_idx])

        pij = d_p[d_idx] * rhoi21 + s_p[s_idx] * rhoj21
        d_idx_9, s_idx_9 = declare("int", 2)
        d_idx_9 = 9 * d_idx
        s_idx_9 = 9 * s_idx

        # d_ds deviatoric stress of particle with id d_idx. It is a stride with
        # 9 elements. For a particle with id 0 ( simple case for understanding)
        # will have its 9 properties as follows
        tmp00 = pij - d_ds[d_idx_9] * rhoi21 - s_ds[s_idx_9] * rhoj21
        tmp01 = -d_ds[d_idx_9 + 1] * rhoi21 - s_ds[s_idx_9 + 1] * rhoj21
        tmp02 = -d_ds[d_idx_9 + 2] * rhoi21 - s_ds[s_idx_9 + 2] * rhoj21

        d_au[d_idx] += -s_m[s_idx] * (
            tmp00 * DWIJ[0] + tmp01 * DWIJ[1] + tmp02 * DWIJ[2])

        tmp10 = -d_ds[d_idx_9 + 3] * rhoi21 - s_ds[s_idx_9 + 3] * rhoj21
        tmp11 = pij - d_ds[d_idx_9 + 4] * rhoi21 - s_ds[s_idx_9 + 4] * rhoj21
        tmp12 = -d_ds[d_idx_9 + 5] * rhoi21 - s_ds[s_idx_9 + 5] * rhoj21

        d_av[d_idx] += -s_m[s_idx] * (
            tmp10 * DWIJ[0] + tmp11 * DWIJ[1] + tmp12 * DWIJ[2])

        tmp20 = -d_ds[d_idx_9 + 6] * rhoi21 - s_ds[s_idx_9 + 6] * rhoj21
        tmp21 = -d_ds[d_idx_9 + 7] * rhoi21 - s_ds[s_idx_9 + 7] * rhoj21
        tmp22 = pij - d_ds[d_idx_9 + 8] * rhoi21 - s_ds[s_idx_9 + 8] * rhoj21

        d_aw[d_idx] += -s_m[s_idx] * (
            tmp20 * DWIJ[0] + tmp21 * DWIJ[1] + tmp22 * DWIJ[2])

        tmp = -d_p0[d_idx] * s_m[s_idx] * rhoi21

        dwijhat = declare('matrix(3)')
        SPH_KERNEL.gradient(XIJ, RIJ, 0.5 * HIJ, dwijhat)

        d_auhat[d_idx] += tmp * dwijhat[0]
        d_avhat[d_idx] += tmp * dwijhat[1]
        d_awhat[d_idx] += tmp * dwijhat[2]


class VelocityGradientHat(Equation):
    def __init__(self, dest, sources):
        super(VelocityGradientHat, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradvel):
        idx = declare("int")
        idx = 9 * d_idx

        i = declare('int')
        for i in range(9):
            d_gradvel[idx + i] = 0.

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_uhat, d_vhat, d_what, d_p,
             d_ads, d_gradvel, s_p, s_m, s_uhat, s_vhat, s_what, d_au, d_av,
             d_aw, DWIJ, d_p0, XIJ, RIJ, SPH_KERNEL, HIJ):
        vol_j = s_m[s_idx] / s_rho[s_idx]
        i, j, p, idx = declare("int", 4)
        idx = 9 * d_idx
        vij = declare('matrix(9)', 1)
        vij[0] = s_uhat[s_idx] - d_uhat[d_idx]
        vij[1] = s_vhat[s_idx] - d_vhat[d_idx]
        vij[2] = s_what[s_idx] - d_what[d_idx]

        # compute the outer product
        for i in range(3):
            for j in range(3):
                p = 3 * i + j
                d_gradvel[idx + p] += vol_j * vij[i] * DWIJ[j]


class DeviatoricStressRate(Equation):
    def __init__(self, dest, sources):
        super(DeviatoricStressRate, self).__init__(dest, sources)

    def _get_helpers_(self):
        return [mat_vec_mult, mat_mult]

    def initialize(self, d_idx, d_ads):
        idx = declare("int")
        idx = 9 * d_idx

        i = declare('int')
        for i in range(9):
            d_ads[idx + i] = 0.

    def post_loop(self, d_idx, d_ds, d_ads, d_strain_tensor, d_rotation_tensor,
                  d_gradvel, d_G):
        # get the epsilon or strain tensor of the matrix
        i, j, idx, p, p_t, ind = declare("int", 5)
        idx = 9 * d_idx

        # the idea of these two loops is to compute the strain and rotation
        # tensor from the gradient of velocity since strain tensor is a tensor
        # with 9 elements and defined the following way

        # strain_tensor = 0.5 * (gradvel + gradvel^(transpose))

        # and in component form

        # strain_tensor[i] = 0.5 * (gradvel[i] + gradvel^(transpose)[i])
        # since we will use single gradvel and don't have gradvel^(transpose)
        # for a given index we will compute the transpose index
        # by using the two for loops
        # trace of the strain tensor
        epsilon_trace = 0.
        for i in range(3):
            for j in range(3):
                # local index
                p = i * 3 + j
                # local index transpose
                p_t = i + j * 3
                d_strain_tensor[idx + p] = 0.5 * (
                    d_gradvel[idx + p] + d_gradvel[idx + p_t])
                d_rotation_tensor[idx + p] = 0.5 * (
                    d_gradvel[idx + p] - d_gradvel[idx + p_t])
                # if both the indices are same then compute the
                # trace
                if i == j:
                    epsilon_trace += d_strain_tensor[idx + p]

        # strain tensor, rotation tensor, rotation tensor transpose, deviatoric
        # stress
        eps, omega, omegaT, ds = declare('matrix(9)', 5)

        for i in range(3):
            for j in range(3):
                p = 3 * i + j
                p_t = i + 3 * j
                omega[p] = d_rotation_tensor[idx + p]
                omegaT[p] = d_rotation_tensor[idx + p_t]
                ds[p] = d_ds[idx + p]
                eps[p] = d_strain_tensor[idx + p]

        # Matrix multiplication between deviatoric stress and rotation
        # tensor transpose
        dsomegaT, omegads = declare('matrix(9)', 2)

        mat_mult(ds, omegaT, 3, dsomegaT)
        mat_mult(omega, ds, 3, omegads)

        G = d_G[0]

        for i in range(3):
            for j in range(3):
                ind = 3 * i + j
                d_ads[idx +
                      ind] += 2 * G * eps[ind] + dsomegaT[ind] + omegads[ind]
                if i == j:
                    d_ads[idx + ind] += -2 * G * epsilon_trace / 3.0


class GTVFSolidRK2Step(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v, d_w,
                   d_rho, d_ds, d_ds0, d_ads, d_rho0, d_u0, d_v0, d_w0, dt):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

        i, idx = declare('int', 2)
        idx = d_idx * 9
        for i in range(9):
            d_ds0[idx + i] = d_ds[idx + i]

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_rho, d_au, d_av,
               d_aw, d_uhat, d_vhat, d_what, d_auhat, d_avhat, d_awhat, d_arho,
               d_ds, d_ds0, d_ads, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]

        d_x[d_idx] += dtb2 * d_uhat[d_idx]
        d_y[d_idx] += dtb2 * d_vhat[d_idx]
        d_z[d_idx] += dtb2 * d_what[d_idx]

        d_rho[d_idx] += dtb2 * d_arho[d_idx]

        i, idx = declare('int', 2)
        idx = d_idx * 9
        for i in range(9):
            d_ds[idx + i] = d_ds0[idx + i] + dtb2 * d_ads[idx + i]

    def stage2(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v, d_w,
               d_u0, d_v0, d_w0, d_rho, d_rho0, d_au, d_av, d_aw, d_uhat,
               d_vhat, d_what, d_auhat, d_avhat, d_awhat, d_arho, d_ds, d_ads,
               d_ds0, dt):
        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dt * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dt * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dt * d_awhat[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_uhat[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_vhat[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_what[d_idx]

        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]

        i, idx = declare('int', 2)
        idx = d_idx * 9
        for i in range(9):
            d_ds[idx + i] = d_ds0[idx + i] + dt * d_ads[idx + i]


class GTVFEPECIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.initialize()

        self.compute_accelerations()

        # Predict
        self.stage1()

        # Call any post-stage functions.
        self.do_post_stage(0.5 * dt, 1)

        self.compute_accelerations()

        # Correct
        self.stage2()

        # Call any post-stage functions.
        self.do_post_stage(dt, 2)

    def initial_acceleration(self, t, dt):
        pass
