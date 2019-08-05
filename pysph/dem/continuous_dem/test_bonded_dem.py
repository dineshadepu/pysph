from pysph.dem.continuous_dem.potyondy import get_particle_array_bonded_dem_potyondy, setup_bc_contacts


def test_potyondy_particle_array():
    pa = get_particle_array_bonded_dem_potyondy(x=[1., 2., 3.], rad_s=0.5,
                                                h=2. * 1.)
    setup_bc_contacts(2, pa, 0.2)
    print(pa.bc_idx)


test_potyondy_particle_array()
