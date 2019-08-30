import unittest
from scipy.spatial.transform import Rotation as Rot
from compyle.api import declare
import numpy as np
from numpy import sin, cos



def q_inverse_vec_q(rf=[1.0, 0.0], q=[1.0, 0.0], qinv=[1.0, 0.0],
                    rc=[0.0, 0.0]):
    """Multiply a square matrix with a vector.

    Parameters
    ----------

    a: list
    b: list
    n: int : number of rows/columns
    result: list
    """
    i, j = declare('int', 2)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += a[n * i + j] * b[j]
        result[i] = s


def cross_product(a, b, res):
    res[0] = a[1] * b[2] - a[2] * b[1]
    res[1] = a[2] * b[0] - a[0] * b[2]
    res[2] = a[0] * b[1] - a[1] * b[0]


def dot_product(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def magnitude(a):
    return (a[0] * a[0] + a[1] * a[1] + a[2] * a[2])**0.5


def quaternion_multiplication(p, q, res):
    """Parameters
    ----------
    p   : [float]
          An array of length four
    q   : [float]
          An array of length four
    res : [float]
          An array of length four
    Here `p` is a quaternion. i.e., p = [p.w, p.x, p.y, p.z]. And q is an
    another quaternion.
    This function is used to compute the rate of change of orientation
    when orientation is represented in terms of a quaternion. When the
    angular velocity is represented in terms of global frame
    \frac{dq}{dt} = \frac{1}{2} omega q
    http://www.ams.stonybrook.edu/~coutsias/papers/rrr.pdf
    see equation 8
    """
    res[0] = (p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3])
    res[1] = (p[0] * q[1] + q[0] * p[1] + p[2] * q[3] - p[3] * q[2])
    res[2] = (p[0] * q[2] + q[0] * p[2] + p[3] * q[1] - p[1] * q[3])
    res[3] = (p[0] * q[3] + q[0] * p[3] + p[1] * q[2] - p[2] * q[1])


def quaternion_norm(q):
    return (q[0]**2. + q[1]**2. + q[2]**2. + q[3]**2.)**0.5


def quaternion_inverse(p, p_inv):
    """
    Section 5.5 of Kuviper.
    inverse is complex conjugate.
    """
    p_inv[0] = p[0]
    p_inv[1] = -p[1]
    p_inv[2] = -p[2]
    p_inv[3] = -p[3]


def quaternion_to_matrix(q, matrix):
    matrix[0] = 1. - 2. * (q[2]**2. + q[3]**2.)
    matrix[1] = 2. * (q[1] * q[2] - q[0] * q[3])
    matrix[2] = 2. * (q[1] * q[3] + q[0] * q[2])

    matrix[3] = 2. * (q[1] * q[2] + q[0] * q[3])
    matrix[4] = 1. - 2. * (q[1]**2. + q[3]**2.)
    matrix[5] = 2. * (q[2] * q[3] - q[0] * q[1])

    matrix[6] = 2. * (q[1] * q[3] - q[0] * q[2])
    matrix[7] = 2. * (q[2] * q[3] + q[0] * q[1])
    matrix[8] = 1. - 2. * (q[1]**2. + q[2]**2.)


def matrix_to_quaternion(matrix, q):
    """
    This is taken from
    A bonded particle model for concrete
    """
    # this is applicable only when t is positive
    t = matrix[0] + matrix[4] + matrix[8]
    if t < 0.:
        print("fails")

    tmp = (t + 1)**0.5
    q[0] = 0.5 * tmp
    q[1] = 0.5 * (matrix[7] - matrix[5]) / tmp
    q[2] = 0.5 * (matrix[2] - matrix[6]) / tmp
    q[3] = 0.5 * (matrix[3] - matrix[1]) / tmp


def change_from_scipy_quat_to_pysph_quat_repr(q_scipy, q_pysph):
    q_pysph[0] = q_scipy[3]
    q_pysph[1] = q_scipy[0]
    q_pysph[2] = q_scipy[1]
    q_pysph[3] = q_scipy[2]


def rotate_local_vec_to_global_vec_with_dcm(matrix=[0., 0.], v=[0., 0.],
                                            result=[0., 0.]):
    result[0] = matrix[0] * v[0] + matrix[1] * v[1] + matrix[2] * v[2]
    result[1] = matrix[3] * v[0] + matrix[4] * v[1] + matrix[5] * v[2]
    result[2] = matrix[6] * v[0] + matrix[7] * v[1] + matrix[8] * v[2]


def rotate_local_vec_to_global_vec_with_quat(q=[0., 0.], q_inv=[0., 0.],
                                             vec=[0., 0.], res=[0., 0.]):
    """Generally a quaternion represents the frame of the body. Suppose if the body
    frame of body is represented by quaternion q, then it means, when we
    transform that quaternion to a matrix, we get the rotation matrix with its
    body frame axis to be its three columns. So when we want to transform the
    vector in the current frame (i.e, body frame to the global frame , or what
    ever frame this current body frame is represented with), then we do

    v = q v' q-1

    where v' is represented in the frame of q

    [1] This function represents Equation 17 of Bonded rigid bodies by Wang Y

    The algorithm is taken from [2], equation 13

    [2] Effect of the integration scheme on the rotation of nonspherical
    particles with the discrete element method
    """
    # get the cross product
    # v = q_0^2 v + q_0 t + vec{q} \cross t
    # t = 2 (\vec{q} \cross v)
    t = declare('matrix(3)')
    qct = declare('matrix(3)')
    t[0] = 2. * (q[2] * vec[2] - q[3] * vec[1])
    t[1] = 2. * (q[3] * vec[0] - q[1] * vec[2])
    t[2] = 2. * (q[1] * vec[1] - q[2] * vec[0])

    # \vec{q} \cross t
    qct[0] = (q[2] * t[2] - q[3] * t[1])
    qct[1] = (q[3] * t[0] - q[1] * t[2])
    qct[2] = (q[1] * t[1] - q[2] * t[0])

    res[0] = q[0]**2. * vec[0] + q[0] * t[0] + qct[0]
    res[1] = q[0]**2. * vec[1] + q[0] * t[1] + qct[1]
    res[2] = q[0]**2. * vec[2] + q[0] * t[2] + qct[2]


class TestMatrixQuaternionConversions(unittest.TestCase):
    def test_rotation_about_z_axis(self):
        # crate rotation matrix
        theta = np.pi / 3
        ct = cos(theta)
        st = sin(theta)
        dcm = np.array([ct, -st, 0., st, ct, 0., 0., 0., 1.])

        # get the corresponding quaternion
        q = np.zeros(4)
        q_inv = np.zeros(4)
        matrix_to_quaternion(dcm, q)
        quaternion_inverse(q, q_inv)

        r = Rot.from_dcm(dcm.reshape(3, 3))
        q_tmp = r.as_quat()
        q_scipy = np.zeros(4)
        change_from_scipy_quat_to_pysph_quat_repr(q_tmp, q_scipy)

        print(q)
        print(q_scipy)


class TestVectorTransformation(unittest.TestCase):
    def test_vec_rotation_about_z(self):
        # crate rotation matrix
        theta = np.pi / 3
        ct = cos(theta)
        st = sin(theta)
        dcm = np.array([ct, -st, 0., st, ct, 0., 0., 0., 1.])
        # get the corresponding quaternion
        q = np.zeros(4)
        q_inv = np.zeros(4)
        matrix_to_quaternion(dcm, q)
        quaternion_inverse(q, q_inv)

        result_dcm = np.zeros(3)
        result_quat = np.zeros(3)

        # take a vector on the x axis of local frame
        v = np.array([1., 0., 0.])
        rotate_local_vec_to_global_vec_with_dcm(dcm, v, result_dcm)
        np.testing.assert_almost_equal(result_dcm, np.array([ct, st, 0.]))

        # take a vector on the x axis of local frame
        rotate_local_vec_to_global_vec_with_quat(q, q_inv, v, result_quat)
        np.testing.assert_almost_equal(result_quat, np.array([ct, st, 0.]))

        print(result_dcm)
        print(result_quat)
