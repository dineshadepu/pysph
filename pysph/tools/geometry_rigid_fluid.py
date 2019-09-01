import numpy as np


def get_2d_block(block_spacing=0.1, block_length=2., block_height=3):
    dx = block_spacing
    x, y = np.mgrid[0.:block_length + dx / 2.:dx, 0.:block_height + dx / 2.:dx]
    x = x.ravel()
    y = y.ravel()
    return x, y


def get_2d_block_hexagonal(block_spacing=0.05, block_length=1., block_height=0.3):
    dx = block_spacing
    # create the first layer of particles
    x = np.array([])
    y = np.array([])

    y_tmp = 0.
    odd = False
    while y_tmp <= block_height + block_spacing / 2.:
        if odd is False:
            x_vec = np.arange(0., block_length+dx/2., dx)
            y_vec = np.ones_like(x_vec) * y_tmp
            odd = True
        else:
            x_vec = np.arange(dx/2., block_length+dx + dx/2., dx)
            y_vec = np.ones_like(x_vec) * y_tmp
            odd = False

        x = np.concatenate((x, x_vec))
        y = np.concatenate((y, y_vec))
        y_tmp = y_tmp + dx
        print(y_tmp)

    return x, y


def get_2d_hydrostatic_tank(ht_length=2., ht_height=3, fluid_height=2,
                            spacing=0.1, layers=2):
    """
    Creates hydrostatic tank in 2 dimensions. We need tank height, length,
    where height is in y direction and length is in x direction.

    Since in hydrostatic the fluid fills the whole tank, we need fluid height
    only and not its length.

    The layers of the tank are always outside the given dimension. The fluid
    particles starts at origin.
    """
    # ------------- create the tank ------------------
    #
    # x-dir left extreme point
    xl = -layers * spacing
    # x-dir right extreme point
    xr = ht_length + layers * spacing + spacing / 2.

    # y-dir bottom extreme point
    yb = -layers * spacing
    # y-dir top extreme point
    yt = ht_height + layers * spacing + spacing / 2.

    # create a grid with the given limits
    x, y = np.mgrid[xl:xr:spacing, yb:yt:spacing]

    # now filter the particles which are inside the tank
    interior = ((x > -spacing / 2.) &
                (x < ht_length + spacing / 2.)) & (y > -spacing / 2.)

    x_t = x[~interior]
    y_t = y[~interior]

    # ------------- create the fluid ------------------
    #
    # x-dir left extreme point
    xl = 0.
    # x-dir right extreme point
    xr = ht_length + spacing / 2.

    # y-dir bottom extreme point
    yb = 0.
    # y-dir top extreme point
    yt = fluid_height + spacing / 2.

    # create a grid with the given limits
    x_f, y_f = np.mgrid[xl:xr:spacing, yb:yt:spacing]

    return x_t, y_t, x_f, y_f


def get_2d_hydrostatic_tank_with_sponge_layer(
        ht_length=2., ht_height=3, fluid_height=2, spacing=0.1, thickness=0.3):
    # ------------- create the tank ------------------
    #
    # x-dir left extreme point
    xl = -thickness
    # x-dir right extreme point
    xr = ht_length + thickness + spacing / 2.

    # y-dir bottom extreme point
    yb = -thickness
    # y-dir top extreme point
    yt = ht_height + thickness + spacing / 2.

    # create a grid with the given limits
    x, y = np.mgrid[xl:xr:spacing, yb:yt:spacing]

    # now filter the particles which are inside the tank
    interior = ((x > -spacing / 2.) &
                (x < ht_length + spacing / 2.)) & (y > -spacing / 2.)

    x_t = x[~interior]
    y_t = y[~interior]

    # ------------- create the fluid ------------------
    #
    # x-dir left extreme point
    xl = 0.
    # x-dir right extreme point
    xr = ht_length + spacing / 2.

    # y-dir bottom extreme point
    yb = 0.
    # y-dir top extreme point
    yt = fluid_height + spacing / 2.

    # create a grid with the given limits
    x_f, y_f = np.mgrid[xl:xr:spacing, yb:yt:spacing]

    return x_t, y_t, x_f, y_f


def get_2d_dam_break(ht_length=2., ht_height=3, fluid_height=2,
                     fluid_length=1., spacing=0.1, layers=2):
    # ------------- create the tank ------------------
    #
    # x-dir left extreme point
    xl = -layers * spacing
    # x-dir right extreme point
    xr = ht_length + layers * spacing + spacing / 2.

    # y-dir bottom extreme point
    yb = -layers * spacing
    # y-dir top extreme point
    yt = ht_height + layers * spacing + spacing / 2.

    # create a grid with the given limits
    x, y = np.mgrid[xl:xr:spacing, yb:yt:spacing]

    # now filter the particles which are inside the tank
    interior = ((x > -spacing / 2.) &
                (x < ht_length + spacing / 2.)) & (y > -spacing / 2.)

    x_t = x[~interior]
    y_t = y[~interior]

    # ------------- create the fluid ------------------
    #
    # x-dir left extreme point
    xl = 0.
    # x-dir right extreme point
    xr = fluid_length + spacing / 2.

    # y-dir bottom extreme point
    yb = 0.
    # y-dir top extreme point
    yt = fluid_height + spacing / 2.

    # create a grid with the given limits
    x_f, y_f = np.mgrid[xl:xr:spacing, yb:yt:spacing]

    return x_t, y_t, x_f, y_f


def test_2d_hydrostatic_tank():
    # test_1
    import matplotlib.pyplot as plt
    xt, yt, xf, yf = get_2d_hydrostatic_tank()
    plt.scatter(xt, yt)
    plt.scatter(xf, yf)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

    xt, yt, xf, yf = get_2d_hydrostatic_tank(
        ht_length=2., ht_height=3, fluid_height=2, spacing=0.3, layers=3)
    plt.scatter(xt, yt)
    plt.scatter(xf, yf)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

    xt, yt, xf, yf = get_2d_hydrostatic_tank(
        ht_length=1., ht_height=1.5, fluid_height=1., spacing=0.03, layers=1)
    plt.scatter(xt, yt)
    plt.scatter(xf, yf)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

    xt, yt, xf, yf = get_2d_hydrostatic_tank(
        ht_length=1., ht_height=1.5, fluid_height=1., spacing=0.02, layers=1)
    plt.scatter(xt, yt)
    plt.scatter(xf, yf)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()


def test_2d_hydrostatic_tank_with_sponge_layer():
    # test_1
    import matplotlib.pyplot as plt
    xt, yt, xf, yf = get_2d_hydrostatic_tank_with_sponge_layer()
    plt.scatter(xt, yt)
    plt.scatter(xf, yf)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

    xt, yt, xf, yf = get_2d_hydrostatic_tank_with_sponge_layer(
        ht_length=2., ht_height=3, fluid_height=2, spacing=0.3, thickness=0.5)
    plt.scatter(xt, yt)
    plt.scatter(xf, yf)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

    xt, yt, xf, yf = get_2d_hydrostatic_tank_with_sponge_layer(
        ht_length=1., ht_height=1.5, fluid_height=1., spacing=0.03,
        thickness=0.1)
    plt.scatter(xt, yt)
    plt.scatter(xf, yf)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

    xt, yt, xf, yf = get_2d_hydrostatic_tank_with_sponge_layer(
        ht_length=1., ht_height=1.5, fluid_height=1., spacing=0.02,
        thickness=0.5)
    plt.scatter(xt, yt)
    plt.scatter(xf, yf)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()


# test_2d_hydrostatic_tank()

# test_2d_hydrostatic_tank_with_sponge_layer()

def test_hexagonal_block():
    x, y = get_2d_block_hexagonal()
    import matplotlib.pyplot as plt
    plt.scatter(x, y)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
