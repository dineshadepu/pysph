from pysph.sph.integrator import Integrator
import numpy as np
from math import asin, cos, sin
from pysph.solver.output import load
from pysph.solver.utils import get_files, iter_output
from pysph.base.kernels import CubicSpline


class EPECIntegratorMultiStage(Integrator):
    def one_timestep(self, t, dt):
        self.initialize()

        self.compute_accelerations(0, update_nnps=True)

        # Predict
        self.stage1()
        self.update_domain()

        # Call any post-stage functions.
        self.do_post_stage(0.5*dt, 1)

        self.compute_accelerations(1, update_nnps=True)

        # Correct
        self.stage2()
        self.update_domain()

        # Call any post-stage functions.
        self.do_post_stage(dt, 2)

    def initial_acceleration(self, t, dt):
        pass


def check_rotation():
    n = np.array([1., 0., 0.])
    v = np.array([0., 2., 0.])
    t_tdt = np.array([0., 0., 0.]) + v * 1e-3
    ntdt = np.array([0., 1., 0.])
    nxp = n[0]
    nyp = n[1]
    nzp = n[2]
    # and current normal vector between the particles is
    nxc = ntdt[0]
    nyc = ntdt[1]
    nzc = ntdt[2]
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
    tmpx = t_tdt[0]
    tmpy = t_tdt[1]
    tmpz = t_tdt[2]
    t_tdt_dash = np.array([0., 0., 0.])
    t_tdt_dash[0] = H0 * tmpx + H1 * tmpy + H2 * tmpz
    t_tdt_dash[1] = H3 * tmpx + H4 * tmpy + H5 * tmpz
    t_tdt_dash[2] = H6 * tmpx + H7 * tmpy + H8 * tmpz
    np.dot(t_tdt_dash, ntdt)
    np.dot(t_tdt, ntdt)


def get_particle_array_from_file(file_name, name):
    data = load(file_name)
    return data['arrays'][name]


def analyse_bonded_cnt_info(pa, idx):
    limit = pa.bc_limit[0]
    print("ids of partticle {idx} in contact".format(idx=idx))
    print(pa.bc_idx[limit * idx:limit * idx + limit])

    print("bc_fn_x, bc_fn_y, bc_fn_z")
    print(pa.bc_fn_x[limit * idx:limit * idx + limit])
    print(pa.bc_fn_y[limit * idx:limit * idx + limit])
    print(pa.bc_fn_z[limit * idx:limit * idx + limit])

    print("bc_fs_x, bc_fs_y, bc_fs_z")
    print(pa.bc_fs_x[limit * idx:limit * idx + limit])
    print(pa.bc_fs_y[limit * idx:limit * idx + limit])
    print(pa.bc_fs_z[limit * idx:limit * idx + limit])

    print("bc_mn_x, bc_mn_y, bc_mn_z")
    print(pa.bc_mn_x[limit * idx:limit * idx + limit])
    print(pa.bc_mn_y[limit * idx:limit * idx + limit])
    print(pa.bc_mn_z[limit * idx:limit * idx + limit])
    print("bc_ms_x, bc_ms_y, bc_ms_z")
    print(pa.bc_ms_x[limit * idx:limit * idx + limit])
    print(pa.bc_ms_y[limit * idx:limit * idx + limit])
    print(pa.bc_ms_z[limit * idx:limit * idx + limit])

    print('total bc contacts')
    print(pa.bc_total_contacts[idx])
    print('force on ' + str(idx) + ' is ')
    print(pa.fx[idx])
    print(pa.fy[idx])
    print(pa.fz[idx])
    return pa


def analyse_tng_cnt_info_from_file(file_name, name, idx):
    data = load(file_name)
    spheres = data['arrays'][name]
    spheres = analyse_bonded_cnt_info(spheres, idx)
    return spheres


def are_in_contact(pa_d, pa_s, d_idx, s_idx):
    xd = pa_d.x
    yd = pa_d.y
    zd = pa_d.z

    xs = pa_s.x
    ys = pa_s.y
    zs = pa_s.z

    rd = pa_d.rad_s
    rs = pa_s.rad_s

    rij = ((xd[d_idx] - xs[s_idx])**2. + (yd[d_idx] - ys[s_idx])**2. +
           (zd[d_idx] - zs[s_idx])**2.)**0.5
    overlap = rd[d_idx] + rs[s_idx] - rij
    print(overlap)

    if overlap > 0.:
        return True
    else:
        return False


def excute_sph_equation(pa_arrays, eq, dim=2):
    from pysph.tools.sph_evaluator import SPHEvaluator
    kernel = CubicSpline(dim=dim)
    seval = SPHEvaluator(arrays=pa_arrays, equations=eq, dim=dim,
                         kernel=kernel)
    seval.evaluate(0, 1e-3)


def print_bonded_output(output_folder, name, idx, ti, tf):
    """
    Usage
    print_bonded_output('ball_slipping_surface_output', 'sand', 2.5, 2.5001)
    print_bonded_output('ball_slipping_surface_output', 'sand', 2.002, 2.004)
    """
    files = get_files(output_folder)
    for sd, arrays in iter_output(files):
        sand = arrays[name]
        t = sd['t']
        if t > ti and t < tf:
            print("time is ", t)
            print(sand.bc_x[idx], sand.bc_y[idx], sand.bc_z[idx])


def print_variable_name(output_folder, particle_arr_name, variable_name, idx,
                        ti, tf):
    """
    Usage
    """
    files = get_files(output_folder)
    for sd, arrays in iter_output(files):
        pa = arrays[particle_arr_name]
        t = sd['t']
        if t > ti and t < tf:
            print("time is ", t)
            print(pa.x[idx])


def plot_a_vs_time(output_folder, particle_arr_name, a, idx, ti, tf):
    """
    Usage
    """
    import matplotlib.pyplot as plt
    files = get_files(output_folder)
    t_arr = []
    a_arr = []
    for sd, arrays in iter_output(files):
        pa = arrays[particle_arr_name]
        t_arr.append(sd['t'])
        a_arr.append(getattr(pa, a)[idx])

    plt.plot(t_arr, a_arr)
    plt.show()
