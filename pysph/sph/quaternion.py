from compyle.api import declare


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
    p_inv[1] = - p[1]
    p_inv[2] = - p[2]
    p_inv[3] = - p[3]


def rotate_vector_to_current_frame_with_quaternion(q, q_inv, vec, res):
    """
    """
    # not implemented
    p = 3.
