from pysph.solver.utils import get_files, iter_output
import sys


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


args = sys.argv
plot_a_vs_time(args[1], args[2], args[3], int(args[4]), args[5], args[6])
