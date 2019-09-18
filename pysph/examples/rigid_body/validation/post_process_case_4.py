import numpy as np
from pysph.solver.utils import iter_output, get_files
from pysph.examples.solid_mech.oscillating_plate import get_files_at_given_times_from_log


def comupte_ang_mom(body):
    ang_mom_x = 0.
    ang_mom_y = 0.
    ang_mom_z = 0.
    for i in range(len(body.x)):
        # r cross v
        rcv = body.m[i] * np.cross(np.array([body.x[i], body.y[i], body.z[i]]),
                                   np.array([body.u[i], body.v[i], body.w[i]]))
        ang_mom_x += rcv[0]
        ang_mom_y += rcv[1]
        ang_mom_z += rcv[2]

    return np.array([ang_mom_x, ang_mom_y, ang_mom_z])


def post_process(case):
    # load data from file
    # data = np.loadtxt('amplitude_gray.csv', delimiter=',')
    # t_d = data[:, 0]
    # ampl_d = data[:, 1]
    interval = 1000

    files = get_files(case + "_rbrms_output", case)
    iters = range(0, 100000, interval)
    logfile = "case_4_rbrms_output/case_4.log"
    files = get_files_at_given_times_from_log(files, iters, logfile)

    t, energy_rbrms = [], []
    ang_mom_x_rbrms, ang_mom_y_rbrms, ang_mom_z_rbrms = [], [], []
    lin_mom_x_rbrms, lin_mom_y_rbrms, lin_mom_z_rbrms = [], [], []

    for sd, body1, body2 in iter_output(files[1:], 'body1', 'body2'):
        # get the inverse of the moment of inertia inverse
        I1 = np.linalg.inv(body1.mig.reshape(3, 3))
        I2 = np.linalg.inv(body2.mig.reshape(3, 3))

        _t = sd['t']
        t.append(_t)
        # linear momentum
        linmom = body1.total_mass[0] * body1.vc + body2.total_mass[0] * body2.vc
        # linmom = body1.total_mass[0] * body1.vc
        # angular momentum
        ang_mom_b1 = comupte_ang_mom(body1)
        ang_mom_b2 = comupte_ang_mom(body2)
        angmom = ang_mom_b1 + ang_mom_b2

        ang_mom_x_rbrms.append(angmom[0])
        ang_mom_y_rbrms.append(angmom[1])
        ang_mom_z_rbrms.append(angmom[2])

        lin_mom_x_rbrms.append(linmom[0])
        lin_mom_y_rbrms.append(linmom[1])
        lin_mom_z_rbrms.append(linmom[2])

        total_energy = 0.5 * np.sum(
            body1.m[:] * body1.u[:]**2. + body1.m[:] * body1.v[:]**2. +
            body1.m[:] * body1.w[:]**2.) + 0.5 * np.sum(
                body2.m[:] * body2.u[:]**2. + body2.m[:] * body2.v[:]**2. +
                body2.m[:] * body2.w[:]**2.)

        energy_rbrms.append(total_energy)

    # -----------------------------
    # -----------------------------
    # -----------------------------
    # with quaternion
    files = get_files(case + "_rbqs_output", case)
    iters = range(0, 100000, interval)
    logfile = "case_4_rbqs_output/case_4.log"
    files = get_files_at_given_times_from_log(files, iters, logfile)

    t, energy_rbqs = [], []
    ang_mom_x_rbqs, ang_mom_y_rbqs, ang_mom_z_rbqs = [], [], []
    lin_mom_x_rbqs, lin_mom_y_rbqs, lin_mom_z_rbqs = [], [], []

    for sd, body1, body2 in iter_output(files[1:], 'body1', 'body2'):
        # get the inverse of the moment of inertia inverse
        I1 = np.linalg.inv(body1.mig.reshape(3, 3))
        I2 = np.linalg.inv(body2.mig.reshape(3, 3))

        _t = sd['t']
        print("time")
        print(_t)
        t.append(_t)
        # linear momentum
        linmom = body1.total_mass[0] * body1.vc + body2.total_mass[0] * body2.vc
        # linmom = body1.total_mass[0] * body1.vc
        # angular momentum
        ang_mom_b1 = comupte_ang_mom(body1)
        ang_mom_b2 = comupte_ang_mom(body2)
        angmom = ang_mom_b1 + ang_mom_b2

        ang_mom_x_rbqs.append(angmom[0])
        ang_mom_y_rbqs.append(angmom[1])
        ang_mom_z_rbqs.append(angmom[2])

        lin_mom_x_rbqs.append(linmom[0])
        lin_mom_y_rbqs.append(linmom[1])
        lin_mom_z_rbqs.append(linmom[2])

        total_energy = 0.5 * np.sum(
            body1.m[:] * body1.u[:]**2. + body1.m[:] * body1.v[:]**2. +
            body1.m[:] * body1.w[:]**2.) + 0.5 * np.sum(
                body2.m[:] * body2.u[:]**2. + body2.m[:] * body2.v[:]**2. +
                body2.m[:] * body2.w[:]**2.)
        energy_rbqs.append(total_energy)

    # different for different directory.

    # plotting
    # import matplotlib
    # matplotlib.use('Agg')

    from matplotlib import pyplot as plt
    plt.plot(t, ang_mom_x_rbrms, label="angular momentum x with matrices")
    plt.plot(t, ang_mom_y_rbrms, markersize=3,
             label="angular momentum y with matrices")
    plt.plot(t, ang_mom_z_rbrms, label="angular momentum z with matrices")

    plt.plot(t, ang_mom_x_rbqs, 'o',
             label="angular momentum x with quaternion")
    plt.plot(t, ang_mom_y_rbqs, '*', markersize=3,
             label="angular momentum y with quaternion")
    plt.plot(t, ang_mom_z_rbqs, '>--',
             label="angular momentum z with quaternion")

    plt.xlabel("t")
    plt.ylabel("Angular momentum")
    plt.legend()
    plt.savefig("case_4_ang_mom", dpi=300)
    plt.show()

    plt.clf()
    plt.plot(t, lin_mom_x_rbrms, label="linear momentum x with matrices")
    plt.plot(t, lin_mom_y_rbrms, markersize=3,
             label="linear momentum y with matrices")
    plt.plot(t, lin_mom_z_rbrms, label="linear momentum z with matrices")

    plt.plot(t, lin_mom_x_rbqs, 'o--',
             label="linear momentum x with quaternion")
    plt.plot(t, lin_mom_y_rbqs, '*--', markersize=3,
             label="linear momentum y with quaternion")
    plt.plot(t, lin_mom_z_rbqs, '>--',
             label="linear momentum z with quaternion")

    plt.xlabel("t")
    plt.ylabel("linear momentum")
    plt.legend()
    plt.savefig("case_4_lin_mom", dpi=300)
    plt.show()

    plt.clf()
    plt.plot(t, energy_rbrms, label="Total energy with matrices")
    plt.plot(t, energy_rbqs, '--', label="Total energy with quaternion")
    plt.xlabel("t")
    plt.ylabel("Total energy")
    plt.legend()

    plt.savefig("case_4_energy", dpi=300)
    plt.show()


if __name__ == '__main__':
    post_process("case_4")
