import numpy as np
from pysph.solver.utils import iter_output, get_files
from pysph.examples.solid_mech.oscillating_plate import get_files_at_given_times_from_log


def post_process(case):
    # load data from file
    # data = np.loadtxt('amplitude_gray.csv', delimiter=',')
    # t_d = data[:, 0]
    # ampl_d = data[:, 1]

    # iters = range(0, 100000, 1000)
    # logfile = "oscillating_plate_nx_10_output/oscillating_plate.log"
    # files = get_files_at_given_times_from_log(files, iters, logfile)

    files = get_files(case + "_rbrms_output", case)
    t, energy_rbrms = [], []
    ang_mom_x_rbrms, ang_mom_y_rbrms, ang_mom_z_rbrms = [], [], []
    lin_mom_x_rbrms, lin_mom_y_rbrms, lin_mom_z_rbrms = [], [], []

    for sd, array in iter_output(files[1:], 'body'):
        # get the inverse of the moment of inertia inverse
        I = np.linalg.inv(array.mig.reshape(3, 3))
        _t = sd['t']
        t.append(_t)
        # linear momentum
        linmom = array.total_mass[0] * array.vc
        # angular momentum
        angmom = np.matmul(I, array.omega)

        ang_mom_x_rbrms.append(angmom[0])
        ang_mom_y_rbrms.append(angmom[1])
        ang_mom_z_rbrms.append(angmom[2])

        lin_mom_x_rbrms.append(linmom[0])
        lin_mom_y_rbrms.append(linmom[1])
        lin_mom_z_rbrms.append(linmom[2])

        pot_energy = abs(array.total_mass[0] * 9.81 * array.cm[2])
        total_energy = 0.5 * np.sum(array.u[:]**2. + array.v[:]**2. +
                                    array.w[:]**2.) + pot_energy
        energy_rbrms.append(total_energy)

    # -----------------------------
    # -----------------------------
    # -----------------------------
    # with quaternion
    files = get_files(case + "_rbqs_output", case)
    t, energy_rbqs = [], []
    ang_mom_x_rbqs, ang_mom_y_rbqs, ang_mom_z_rbqs = [], [], []
    lin_mom_x_rbqs, lin_mom_y_rbqs, lin_mom_z_rbqs = [], [], []

    for sd, array in iter_output(files[1:], 'body'):
        # get the inverse of the moment of inertia inverse
        I = np.linalg.inv(array.mig.reshape(3, 3))
        _t = sd['t']
        t.append(_t)
        # linear momentum
        linmom = array.total_mass[0] * array.vc
        # angular momentum
        angmom = np.matmul(I, array.omega)

        ang_mom_x_rbqs.append(angmom[0])
        ang_mom_y_rbqs.append(angmom[1])
        ang_mom_z_rbqs.append(angmom[2])

        lin_mom_x_rbqs.append(linmom[0])
        lin_mom_y_rbqs.append(linmom[1])
        lin_mom_z_rbqs.append(linmom[2])

        pot_energy = abs(array.total_mass[0] * 9.81 * array.cm[2])
        total_energy = 0.5 * np.sum(array.u[:]**2. + array.v[:]**2. +
                                    array.w[:]**2.) + pot_energy
        energy_rbqs.append(total_energy)

    # different for different directory.

    # plotting
    # import matplotlib
    # matplotlib.use('Agg')

    from matplotlib import pyplot as plt
    plt.plot(t, ang_mom_x_rbrms, label="angular momentum x with matrices")
    plt.plot(t, ang_mom_y_rbrms, 'rp--', markersize=3,
             label="angular momentum y with matrices")
    plt.plot(t, ang_mom_z_rbrms, label="angular momentum z with matrices")

    plt.plot(t, ang_mom_x_rbqs, label="angular momentum x with quaternion")
    plt.plot(t, ang_mom_y_rbqs, 'rp--', markersize=3,
             label="angular momentum y with quaternion")
    plt.plot(t, ang_mom_z_rbqs, label="angular momentum z with quaternion")

    plt.xlabel("t")
    plt.ylabel("Angular momentum")
    plt.legend()
    plt.savefig("case_2_ang_mom", dpi=300)

    plt.clf()
    plt.plot(t, lin_mom_x_rbrms, label="linear momentum x with matrices")
    plt.plot(t, lin_mom_y_rbrms, 'rp--', markersize=3,
             label="linear momentum y with matrices")
    plt.plot(t, lin_mom_z_rbrms, label="linear momentum z with matrices")

    plt.plot(t, lin_mom_x_rbqs, label="linear momentum x with quaternion")
    plt.plot(t, lin_mom_y_rbqs, 'rp--', markersize=3,
             label="linear momentum y with quaternion")
    plt.plot(t, lin_mom_z_rbqs, label="linear momentum z with quaternion")

    plt.xlabel("t")
    plt.ylabel("linear momentum")
    plt.legend()
    plt.savefig("case_2_lin_mom", dpi=300)

    plt.clf()
    plt.plot(t, energy_rbrms, label="Total energy with matrices")
    plt.plot(t, energy_rbqs, label="Total energy with quaternion")
    plt.xlabel("t")
    plt.ylabel("Total energy")
    plt.legend()
    plt.savefig("case_2_energy", dpi=300)


if __name__ == '__main__':
    post_process("case_2")
