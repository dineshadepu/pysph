import numpy as np
from pysph.solver.utils import iter_output, get_files
from oscillating_plate import get_files_at_given_times_from_log


def post_process():
    # load data from file
    # data = np.loadtxt('amplitude_gray.csv', delimiter=',')
    # t_d = data[:, 0]
    # ampl_d = data[:, 1]

    files = get_files("oscillating_plate_nx_10_output", "oscillating_plate")
    t, amplitude = [], []
    iters = range(0, 100000, 1000)
    logfile = "oscillating_plate_nx_10_output/oscillating_plate.log"
    to_plot = get_files_at_given_times_from_log(files, iters, logfile)

    for sd, array in iter_output(to_plot, 'plate'):
        _t = sd['t']
        t.append(_t)
        amplitude.append(array.y[array.amplitude_idx[0]])

    # different for different directory.
    files = get_files("oscillating_plate_nx_30_output", "oscillating_plate")
    t_nx_30, amplitude_nx_30 = [], []
    iters = range(0, 100000, 1000)
    logfile = "oscillating_plate_nx_30_output/oscillating_plate.log"
    to_plot = get_files_at_given_times_from_log(files, iters, logfile)

    for sd, array in iter_output(to_plot, 'plate'):
        _t = sd['t']
        t_nx_30.append(_t)
        amplitude_nx_30.append(array.y[array.amplitude_idx[0]])

    import matplotlib
    matplotlib.use('Agg')

    from matplotlib import pyplot as plt
    # plt.clf()
    plt.plot(t, amplitude, label="nx=10")
    plt.plot(t_nx_30, amplitude_nx_30, label="nx=30")
    plt.xlabel('t')
    plt.ylabel('Amplitude')
    plt.legend()
    fig = "amplitude.png"
    plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    post_process()
