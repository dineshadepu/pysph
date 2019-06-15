import numpy as np


def set_total_mass(pa):
    # left limit of body i
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        pa.total_mass[i] = np.sum(pa.m[fltr])
        assert pa.total_mass[i] > 0., "Total mass has to be greater than zero"


def set_center_of_mass(pa):
    # loop over all the bodies
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        pa.cm[3 * i] = np.sum(pa.m[fltr] * pa.x[fltr]) / pa.total_mass[i]
        pa.cm[3 * i + 1] = np.sum(pa.m[fltr] * pa.y[fltr]) / pa.total_mass[i]
        pa.cm[3 * i + 2] = np.sum(pa.m[fltr] * pa.z[fltr]) / pa.total_mass[i]


def set_mi_in_body_frame(pa):
    """Compute the moment of inertia at the beginning of the simulation.
    And set moment of inertia inverse for further computations.
    This method assumes the center of mass is already computed."""
    # no of bodies
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.cm[3 * i:3 * i + 3]

        I = np.zeros(9)
        for j in fltr:
            # Ixx
            I[0] += pa.m[j] * (
                (pa.y[j] - cm_i[1])**2. + (pa.z[j] - cm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.z[j] - cm_i[2])**2.)

            # Izz
            I[8] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.y[j] - cm_i[1])**2.)

            # Ixy
            I[1] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.y[j] - cm_i[1])

            # Ixz
            I[2] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.z[j] - cm_i[2])

            # Iyz
            I[5] -= pa.m[j] * (pa.y[j] - cm_i[1]) * (pa.z[j] - cm_i[2])

        I[3] = I[1]
        I[6] = I[2]
        I[7] = I[5]
        I_inv = np.linalg.inv(I.reshape(3, 3))
        I_inv = I_inv.ravel()
        pa.mib[9 * i:9 * i + 9] = I_inv[:]


def set_mi_in_body_frame_optimized(pa):
    """Compute the moment of inertia at the beginning of the simulation.
    And set moment of inertia inverse for further computations.
    This method assumes the center of mass is already computed."""
    # no of bodies
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.cm[3 * i:3 * i + 3]

        I = np.zeros(9)
        for j in fltr:
            # Ixx
            I[0] += pa.m[j] * (
                (pa.y[j] - cm_i[1])**2. + (pa.z[j] - cm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.z[j] - cm_i[2])**2.)

            # Izz
            I[8] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.y[j] - cm_i[1])**2.)

            # Ixy
            I[1] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.y[j] - cm_i[1])

            # Ixz
            I[2] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.z[j] - cm_i[2])

            # Iyz
            I[5] -= pa.m[j] * (pa.y[j] - cm_i[1]) * (pa.z[j] - cm_i[2])

        I[3] = I[1]
        I[6] = I[2]
        I[7] = I[5]
        # find the eigen vectors and eigen values of the moi
        vals, vecs = np.linalg.eigh(I.reshape(3, 3))
        # and set the eigen values as the principal moi values
        pa.mbp[3 * i:3 * i + 3] = vals
        inv = 1. / vals
        pa.mib[9 * i:9 * i + 9] = np.diag(inv).ravel()
        # and similarly set the corresponding body frame
        # which will be the eigen vectors
        pa.R[9 * i:9 * i + 9] = vecs.ravel()


def get_mi_in_global_frame(pa):
    """Given particle array at an instant compute the global moment of inertia
    from the global positions."""
    no = pa.body_id + 1
    # loop over all the bodies
    for i in range(no):
        fltr = np.where(pa.body_id == i)
        cm_i = pa.cm[3 * i:3 * i + 3]

        I = np.zeros(9)
        for j in fltr:
            # Ixx
            I[0] += pa.m[j] * (
                (pa.y[j] - cm_i[1])**2. + (pa.z[j] - cm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.z[j] - cm_i[2])**2.)

            # Izz
            I[8] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.y[j] - cm_i[1])**2.)

            # Ixy
            I[1] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.y[j] - cm_i[1])

            # Ixz
            I[2] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.z[j] - cm_i[2])

            # Iyz
            I[5] -= pa.m[j] * (pa.y[j] - cm_i[1]) * (pa.z[j] - cm_i[2])

        I[3] = I[1]
        I[6] = I[2]
        I[7] = I[5]
    return I


def set_principal_mi_in_body_frame(pa):
    """Compute the moment of inertia at the beginning of the simulation.
    And set moment of inertia inverse for further computations.
    This method assumes the center of mass is already computed."""
    cm_i = pa.cm

    I = np.zeros(9)
    for j in range(len(pa.x)):
        # Ixx
        I[0] += pa.m[j] * ((pa.y[j] - cm_i[1])**2. + (pa.z[j] - cm_i[2])**2.)

        # Iyy
        I[4] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. + (pa.z[j] - cm_i[2])**2.)

        # Izz
        I[8] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. + (pa.y[j] - cm_i[1])**2.)

        # Ixy
        I[1] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.y[j] - cm_i[1])

        # Ixz
        I[2] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.z[j] - cm_i[2])

        # Iyz
        I[5] -= pa.m[j] * (pa.y[j] - cm_i[1]) * (pa.z[j] - cm_i[2])

    I[3] = I[1]
    I[6] = I[2]
    I[7] = I[5]
    I_inv = np.linalg.inv(I.reshape(3, 3))
    I_inv = I_inv.ravel()
    pa.mib[0:9] = I_inv[:]


def set_body_frame_position_vectors(pa):
    """Save the position vectors w.r.t body frame"""
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.cm[3 * i:3 * i + 3]
        for j in fltr:
            pa.dx0[j] = pa.x[j] - cm_i[0]
            pa.dy0[j] = pa.y[j] - cm_i[1]
            pa.dz0[j] = pa.z[j] - cm_i[2]


def set_body_frame_position_vectors_optimized(pa):
    """Save the position vectors w.r.t body frame"""
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.cm[3 * i:3 * i + 3]
        R_i = pa.R[9 * i:9 * i + 9]
        for j in fltr:
            dx = pa.x[j] - cm_i[0]
            dy = pa.y[j] - cm_i[1]
            dz = pa.z[j] - cm_i[2]

            pa.dx0[j] = (R_i[0] * dx + R_i[3] * dy + R_i[6] * dz)
            pa.dy0[j] = (R_i[1] * dx + R_i[4] * dy + R_i[7] * dz)
            pa.dz0[j] = (R_i[2] * dx + R_i[5] * dy + R_i[8] * dz)


def set_angular_momentum(pa):
    Ig = get_mi_in_global_frame(pa)

    # L = I omega
    pa.L = np.matmul(Ig, pa.omega)


def setup_rotation_matrix_rigid_body(pa):
    """Setup total mass, center of mass, moment of inertia and
    angular momentum of a rigid body defined using rotation matrices."""
    set_total_mass(pa)
    set_center_of_mass(pa)
    set_mi_in_body_frame(pa)
    set_body_frame_position_vectors(pa)


def setup_quaternion_rigid_body(pa):
    """Setup total mass, center of mass, moment of inertia and
    angular momentum of a rigid body defined using quaternion."""
    set_total_mass(pa)
    set_center_of_mass(pa)
    set_mi_in_body_frame(pa)
    set_body_frame_position_vectors(pa)


def setup_rotation_matrix_rigid_body_optimized(pa):
    """Setup total mass, center of mass, moment of inertia and
    angular momentum of a rigid body defined using rotation matrices."""
    set_total_mass(pa)
    set_center_of_mass(pa)
    set_mi_in_body_frame_optimized(pa)
    set_body_frame_position_vectors_optimized(pa)


def setup_quaternion_rigid_body_optimized(pa):
    """Setup total mass, center of mass, moment of inertia and
    angular momentum of a rigid body defined using quaternion."""
    set_total_mass(pa)
    set_center_of_mass(pa)
    set_mi_in_body_frame_optimized(pa)
    set_body_frame_position_vectors_optimized(pa)
