import numpy as np
import numpy
from scipy.spatial.transform import Rotation as Rot


def add_properties(pa, prop_list):
    for prop in prop_list:
        pa.add_property(name=prop)


def normalize_q_orientation(q):
    norm_q = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    q[:] = q[:] / norm_q


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
            I[0] += pa.m[j] * ((pa.y[j] - cm_i[1])**2. +
                               (pa.z[j] - cm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. +
                               (pa.z[j] - cm_i[2])**2.)

            # Izz
            I[8] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. +
                               (pa.y[j] - cm_i[1])**2.)

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


def set_mi_in_body_frame_rot_mat_optimized(pa):
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
            I[0] += pa.m[j] * ((pa.y[j] - cm_i[1])**2. +
                               (pa.z[j] - cm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. +
                               (pa.z[j] - cm_i[2])**2.)

            # Izz
            I[8] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. +
                               (pa.y[j] - cm_i[1])**2.)

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
        vals, R = np.linalg.eigh(I.reshape(3, 3))
        # find the determinant of R
        determinant = np.linalg.det(R)
        if determinant == -1.:
            R[:, 0] = -R[:, 0]

        # recompute the moment of inertia about the new coordinate frame
        # if flipping of one of the axis due the determinant value
        R = R.ravel()

        if determinant == -1.:
            I = np.zeros(9)
            for j in fltr:
                dx = pa.x[j] - cm_i[0]
                dy = pa.y[j] - cm_i[1]
                dz = pa.z[j] - cm_i[2]

                dx0 = (R[0] * dx + R[3] * dy + R[6] * dz)
                dy0 = (R[1] * dx + R[4] * dy + R[7] * dz)
                dz0 = (R[2] * dx + R[5] * dy + R[8] * dz)

                # Ixx
                I[0] += pa.m[j] * ((dy0)**2. + (dz0)**2.)

                # Iyy
                I[4] += pa.m[j] * ((dx0)**2. + (dz0)**2.)

                # Izz
                I[8] += pa.m[j] * ((dx0)**2. + (dy0)**2.)

                # Ixy
                I[1] -= pa.m[j] * (dx0) * (dy0)

                # Ixz
                I[2] -= pa.m[j] * (dx0) * (dz0)

                # Iyz
                I[5] -= pa.m[j] * (dy0) * (dz0)

            I[3] = I[1]
            I[6] = I[2]
            I[7] = I[5]

            # set the inverse inertia values
            vals = np.array([I[0], I[4], I[8]])

        pa.mibp[3 * i:3 * i + 3] = 1. / vals
        pa.R[9 * i:9 * i + 9] = R


def rotation_mat_to_quat(R, q):
    """This code is taken from
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    q[0] = np.sqrt(R[0] + R[4] + R[8]) / 2
    q[1] = (R[7] - R[5]) / (4. * q[0])
    q[2] = (R[2] - R[6]) / (4. * q[0])
    q[3] = (R[3] - R[1]) / (4. * q[0])


def set_mi_in_body_frame_quaternion_optimized(pa):
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
            I[0] += pa.m[j] * ((pa.y[j] - cm_i[1])**2. +
                               (pa.z[j] - cm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. +
                               (pa.z[j] - cm_i[2])**2.)

            # Izz
            I[8] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. +
                               (pa.y[j] - cm_i[1])**2.)

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
        vals, R = np.linalg.eigh(I.reshape(3, 3))
        # find the determinant of R
        determinant = np.linalg.det(R)
        if determinant == -1.:
            R[:, 0] = -R[:, 0]

        # recompute the moment of inertia about the new coordinate frame
        # if flipping of one of the axis due the determinant value
        R = R.ravel()

        if determinant == -1.:
            I = np.zeros(9)
            for j in fltr:
                dx = pa.x[j] - cm_i[0]
                dy = pa.y[j] - cm_i[1]
                dz = pa.z[j] - cm_i[2]

                dx0 = (R[0] * dx + R[3] * dy + R[6] * dz)
                dy0 = (R[1] * dx + R[4] * dy + R[7] * dz)
                dz0 = (R[2] * dx + R[5] * dy + R[8] * dz)

                # Ixx
                I[0] += pa.m[j] * ((dy0)**2. + (dz0)**2.)

                # Iyy
                I[4] += pa.m[j] * ((dx0)**2. + (dz0)**2.)

                # Izz
                I[8] += pa.m[j] * ((dx0)**2. + (dy0)**2.)

                # Ixy
                I[1] -= pa.m[j] * (dx0) * (dy0)

                # Ixz
                I[2] -= pa.m[j] * (dx0) * (dz0)

                # Iyz
                I[5] -= pa.m[j] * (dy0) * (dz0)

            I[3] = I[1]
            I[6] = I[2]
            I[7] = I[5]

            # set the inverse inertia values
            vals = np.array([I[0], I[4], I[8]])

        pa.mibp[3 * i:3 * i + 3] = 1. / vals

        # get the quaternion from the rotation matrix
        r = Rot.from_dcm(R.reshape(3, 3))
        q_tmp = r.as_quat()
        q = np.zeros(4)
        q[0] = q_tmp[3]
        q[1] = q_tmp[0]
        q[2] = q_tmp[1]
        q[3] = q_tmp[2]

        normalize_q_orientation(q)
        pa.q[4 * i:4 * i + 4] = q

        # also set the rotation matrix
        pa.R[9 * i:9 * i + 9] = R


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
    set_mi_in_body_frame_rot_mat_optimized(pa)
    set_body_frame_position_vectors_optimized(pa)


def setup_quaternion_rigid_body_optimized(pa):
    """Setup total mass, center of mass, moment of inertia and
    angular momentum of a rigid body defined using rotation matrices."""
    set_total_mass(pa)
    set_center_of_mass(pa)
    set_mi_in_body_frame_quaternion_optimized(pa)
    set_body_frame_position_vectors_optimized(pa)


def add_basic_properties_for_rigid_body(pa, body_id):
    """

    Args:
    -------

    pa: particle array
    body_id: an array describing the particle constituent body

    We have a total of 4 or more schemes in PySPH to simulate rigid body
    dynamics. There are a few properties which are common to all the schemes.
    This function used to add such basic properties. This function is invoked to
    add all such properties before adding any scheme specific properties.

    As an example, any rigid body will have center of mass, total mass and
    velocity of center of mass, similar properties are added to the particle
    array.

    Note
    -------

    This will only add the properties, but doesn't compute them.
    The computation is done differently for different schemes.

    Example
    -------

    Basic usage::
        >>> pa = get_particle_array(x=np.linspace(0, 1, 10))

        >>> pa.properties()
    {'tag': <cyarray.carray.IntArray at 0x7ff27603ab40>,
    'pid': <cyarray.carray.IntArray at 0x7ff27603a9a0>,
    'gid': <cyarray.carray.UIntArray at 0x7ff27603a528>,
    'x': <cyarray.carray.DoubleArray at 0x7ff28f49bc88>,
    'aw': <cyarray.carray.DoubleArray at 0x7ff28d7ce828>,
    'au': <cyarray.carray.DoubleArray at 0x7ff28d7ce978>,
    'u': <cyarray.carray.DoubleArray at 0x7ff28d7ce898>,
    'rho': <cyarray.carray.DoubleArray at 0x7ff28d7ce908>,
    'h': <cyarray.carray.DoubleArray at 0x7ff28d7ce7b8>,
    'av': <cyarray.carray.DoubleArray at 0x7ff28d7ce748>,
    'w': <cyarray.carray.DoubleArray at 0x7ff28d7ce6d8>,
    'p': <cyarray.carray.DoubleArray at 0x7ff28d7ce668>,
    'y': <cyarray.carray.DoubleArray at 0x7ff28d7ce5f8>,
    'm': <cyarray.carray.DoubleArray at 0x7ff28d7ce588>,
    'v': <cyarray.carray.DoubleArray at 0x7ff28d7ce518>,
    'z': <cyarray.carray.DoubleArray at 0x7ff28d7ce4a8>}

        >>> pa.constants()
    {}

        >>> add_basic_properties_for_rigid_body(pa, body_id=None):

        >>> pa.properties()
    {'tag': <cyarray.carray.IntArray at 0x7ff27603ab40>,
    'pid': <cyarray.carray.IntArray at 0x7ff27603a9a0>,
    'gid': <cyarray.carray.UIntArray at 0x7ff27603a528>,
    'x': <cyarray.carray.DoubleArray at 0x7ff28f49bc88>,
    'aw': <cyarray.carray.DoubleArray at 0x7ff28d7ce828>,
    'au': <cyarray.carray.DoubleArray at 0x7ff28d7ce978>,
    'u': <cyarray.carray.DoubleArray at 0x7ff28d7ce898>,
    'rho': <cyarray.carray.DoubleArray at 0x7ff28d7ce908>,
    'h': <cyarray.carray.DoubleArray at 0x7ff28d7ce7b8>,
    'av': <cyarray.carray.DoubleArray at 0x7ff28d7ce748>,
    'w': <cyarray.carray.DoubleArray at 0x7ff28d7ce6d8>,
    'p': <cyarray.carray.DoubleArray at 0x7ff28d7ce668>,
    'y': <cyarray.carray.DoubleArray at 0x7ff28d7ce5f8>,
    'm': <cyarray.carray.DoubleArray at 0x7ff28d7ce588>,
    'v': <cyarray.carray.DoubleArray at 0x7ff28d7ce518>,
    'z': <cyarray.carray.DoubleArray at 0x7ff28d7ce4a8>,
    'body_id': <cyarray.carray.IntArray at 0x7ff27603ad48>,
    'fx': <cyarray.carray.DoubleArray at 0x7ff28d7cea58>,
    'fy': <cyarray.carray.DoubleArray at 0x7ff28d7ceb38>,
    'fz': <cyarray.carray.DoubleArray at 0x7ff28d7ceac8>,
    'x0': <cyarray.carray.DoubleArray at 0x7ff28d7cec88>,
    'y0': <cyarray.carray.DoubleArray at 0x7ff28d7ce278>,
    'z0': <cyarray.carray.DoubleArray at 0x7ff28d7cec18>,
    'u0': <cyarray.carray.DoubleArray at 0x7ff28d7ce438>,
    'v0': <cyarray.carray.DoubleArray at 0x7ff28d7ce3c8>,
    'w0': <cyarray.carray.DoubleArray at 0x7ff28d7ce2e8>}

        >>> pa.constants()
    {'nb': <cyarray.carray.LongArray at 0x7ff28d7cef28>,
    'total_mass': <cyarray.carray.DoubleArray at 0x7ff28d7ce358>,
    'cm': <cyarray.carray.DoubleArray at 0x7ff28d7cecf8>,
    'cm0': <cyarray.carray.DoubleArray at 0x7ff28d7ce208>,
    'R': <cyarray.carray.DoubleArray at 0x7ff28d7ce0b8>,
    'mig': <cyarray.carray.DoubleArray at 0x7ff28d7ce048>,
    'force': <cyarray.carray.DoubleArray at 0x7ff28d7ce198>,
    'torque': <cyarray.carray.DoubleArray at 0x7ff28d7ce128>,
    'vc': <cyarray.carray.DoubleArray at 0x7ff28f7205f8>,
    'vc0': <cyarray.carray.DoubleArray at 0x7ff275fd1588>,
    'omega': <cyarray.carray.DoubleArray at 0x7ff275fd1668>,
    'omega0': <cyarray.carray.DoubleArray at 0x7ff275fd14a8>}

    The address of the array will differ on your system.

    """
    if body_id is None:
        nb = 1
        body_id = 0
    else:
        nb = np.max(body_id) + 1

    pa.add_constant('nb', data=nb)
    pa.add_property('body_id', type='int', data=body_id)

    # these properties are used in sumupexternalforces equation
    sumupexternal_props = [
        'fx', 'fy', 'fz', 'dx0', 'dy0', 'dz0', 'x0', 'y0', 'z0', 'u0', 'v0',
        'w0'
    ]

    add_properties(pa, sumupexternal_props)

    # some constants which are common to all schemes
    consts = {
        'total_mass': numpy.zeros(nb, dtype=float),
        'cm': numpy.zeros(3 * nb, dtype=float),
        'cm0': numpy.zeros(3 * nb, dtype=float),
        'R': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
        # moment of inertia inverse in global frame
        'mig': numpy.zeros(9 * nb, dtype=float),
        # total force at the center of mass
        'force': numpy.zeros(3 * nb, dtype=float),
        # torque about the center of mass
        'torque': numpy.zeros(3 * nb, dtype=float),
        # velocity, acceleration of CM.
        'vc': numpy.zeros(3 * nb, dtype=float),
        'vc0': numpy.zeros(3 * nb, dtype=float),
        # angular velocity in global frame
        'omega': numpy.zeros(3 * nb, dtype=float),
        'omega0': numpy.zeros(3 * nb, dtype=float),
    }
    for constant in consts:
        pa.add_constant(constant, consts[constant])


def setup_3d_rigid_body_for_dcm_formulation(pa, body_id=None):
    # This will add all the basic properties common to all rigid body schemes
    add_basic_properties_for_rigid_body(pa, body_id)

    # some constants needed by specific 3d quaternion scheme
    nb = pa.nb[0]
    consts = {
        'q': [1., 0., 0., 0.] * nb,
        'q0': [1., 0., 0., 0.] * nb,
        'qdot': numpy.zeros(4 * nb, dtype=float),
        # moment of inertia inverse in body frame
        'mib': numpy.zeros(9 * nb, dtype=float),
    }
    for constant in consts:
        pa.add_constant(constant, consts[constant])

    # Now compute the properties of the rigid body for 3d quaternion scheme
    setup_quaternion_rigid_body(pa)


def setup_3d_rigid_body_for_quaternion_formulation(pa, body_id=None):
    # This will add all the basic properties common to all rigid body schemes
    add_basic_properties_for_rigid_body(pa, body_id)

    # some constants needed by specific 3d quaternion scheme
    nb = pa.nb[0]
    consts = {
        'q': [1., 0., 0., 0.] * nb,
        'q0': [1., 0., 0., 0.] * nb,
        'qdot': numpy.zeros(4 * nb, dtype=float),
        # moment of inertia inverse in body frame
        'mib': numpy.zeros(9 * nb, dtype=float),
    }
    for constant in consts:
        pa.add_constant(constant, consts[constant])

    # Now compute the properties of the rigid body for 3d quaternion scheme
    setup_quaternion_rigid_body(pa)
