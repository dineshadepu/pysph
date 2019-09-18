import numpy as np


def post_process():
    from pysph.solver.utils import iter_output, get_files

    files = get_files("./2d/with_cundall_particle_particle_output")
    t = []
    system_x = []
    system_y = []
    for sd, array in iter_output(files, 'sand'):
        _t = sd['t']
        print(_t)
        t.append(_t)
        cm_x = np.sum(array.m * array.x) / np.sum(array.m)
        cm_y = np.sum(array.m * array.y) / np.sum(array.m)

        system_x.append(cm_x)
        system_y.append(cm_y)

    import matplotlib.pyplot as plt

    plt.figure(0)
    plt.scatter(t, system_x, label='x com vs t for par-par 2d')
    plt.legend()
    plt.figure(1)
    plt.scatter(t, system_y, label='y com vs t for par-par 2d')
    plt.legend()

    files = get_files("./2d/with_cundall_particle_wall_output")
    t = []
    system_x = []
    system_y = []
    for sd, array in iter_output(files, 'sand'):
        _t = sd['t']
        print(_t)
        t.append(_t)
        cm_x = np.sum(array.m * array.x) / np.sum(array.m)
        cm_y = np.sum(array.m * array.y) / np.sum(array.m)

        system_x.append(cm_x)
        system_y.append(cm_y)

    import matplotlib.pyplot as plt

    plt.figure(0)
    plt.scatter(t, system_x, label='x com vs t for par-wall 2d')
    plt.legend()
    plt.figure(1)
    plt.scatter(t, system_y, label='y com vs t for par-wall 2d')
    plt.legend()

    files = get_files("./3d/with_cundall_particle_particle_output")
    t = []
    system_x = []
    system_y = []
    for sd, array in iter_output(files, 'sand'):
        _t = sd['t']
        print(_t)
        t.append(_t)
        cm_x = np.sum(array.m * array.x) / np.sum(array.m)
        cm_y = np.sum(array.m * array.y) / np.sum(array.m)

        system_x.append(cm_x)
        system_y.append(cm_y)

    import matplotlib.pyplot as plt

    plt.figure(0)
    plt.scatter(t, system_x, label='x com vs t for par-par 3d')
    plt.legend()
    plt.figure(1)
    plt.scatter(t, system_y, label='y com vs t for par-wall 3d')
    plt.legend()

    files = get_files("./3d/with_cundall_particle_wall_output")
    t = []
    system_x = []
    system_y = []
    for sd, array in iter_output(files, 'sand'):
        _t = sd['t']
        print(_t)
        t.append(_t)
        cm_x = np.sum(array.m * array.x) / np.sum(array.m)
        cm_y = np.sum(array.m * array.y) / np.sum(array.m)

        system_x.append(cm_x)
        system_y.append(cm_y)

    import matplotlib.pyplot as plt

    plt.figure(0)
    plt.scatter(t, system_x, label='x com vs t for par-wall 3d')
    plt.legend()
    plt.figure(1)
    plt.scatter(t, system_y, label='y com vs t for par-wall 3d')
    plt.legend()

    plt.figure(0)
    plt.savefig("granular_collapse_compare_xcom")
    plt.figure(1)
    plt.savefig("granular_collapse_compare_ycom")


post_process()
