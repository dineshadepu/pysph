import numpy as np
import unittest
from pysph.base.kernels import CubicSpline
from pysph.base.nnps import LinkedListNNPS
from pysph.sph.sph_compiler import SPHCompiler
from pysph.sph.equation import Equation, Group
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.rigid_body import (
    get_particle_array_rigid_body_quaternion, SumUpExternalForces,
    get_particle_array_rigid_body_rotation_matrix, RK2StepRigidBodyQuaternions,
    RK2StepRigidBodyRotationMatrices)
from pysph.sph.rigid_body import (get_particle_array_rigid_body_dem,
                                  UpdateTangentialContacts)


class TestUpdateTangentialContacts(unittest.TestCase):
    def setUp(self):
        self.dim = 2

    def _make_accel_eval(self, equations, *args):
        kernel = CubicSpline(dim=self.dim)
        seval = SPHEvaluator(
            arrays=list(args), equations=equations, dim=self.dim,
            kernel=kernel)
        return seval

    def test_tangential_contacts(self):
        # create a body with a single dem id
        x = np.array([0., 1.0, 2., -0.8])
        y = np.array([0., 0., 0., 0.])
        rad_s = np.array([0.7, 0.7, 0.7, 0.7])
        body1 = get_particle_array_rigid_body_dem(
            x=x, y=y, rad_s=rad_s, m=1, h=1.2, dem_id=0, name="body1")

        body1.tng_idx[0] = 1
        body1.tng_idx[1] = 2
        body1.tng_idx[2] = 3
        body1.tng_idx_dem_id[0:3] = 0
        body1.total_tng_contacts[0] = 3

        # Given
        eqs1 = [
            Group(equations=[
                UpdateTangentialContacts(dest='body1', sources=["body1"]),
            ]),
        ]
        a_eval = self._make_accel_eval(eqs1, body1)

        # When
        a_eval.evaluate(0.0, 0.1)

        np.testing.assert_equal(body1.total_tng_contacts[0], 2)

        # create one more body
        x = np.array([0.1, 1.1, 2.1, -0.7])
        y = np.array([0., 0., 0., 0.])
        body2 = get_particle_array_rigid_body_dem(
            x=x, y=y, rad_s=rad_s, m=1, h=1.2, dem_id=1, name="body2")
        # add tangential contacts to the body1 of body 2
        body1.tng_idx[0] = 1
        body1.tng_idx_dem_id[0] = 0
        body1.tng_idx[1] = 1
        body1.tng_idx_dem_id[1] = 1
        body1.tng_idx[2] = 2
        body1.tng_idx_dem_id[2] = 0
        body1.tng_idx[3] = 2
        body1.tng_idx_dem_id[3] = 1
        body1.tng_idx[4] = 3
        body1.tng_idx_dem_id[4] = 0
        body1.tng_idx[5] = 3
        body1.tng_idx_dem_id[5] = 1

        body1.total_tng_contacts[0] = 6

        # Given
        eqs1 = [
            Group(equations=[
                UpdateTangentialContacts(dest='body1', sources=["body1"]),
            ]),
        ]
        a_eval = self._make_accel_eval(eqs1, body1)

        # When
        a_eval.evaluate(0.0, 0.1)

        np.testing.assert_equal(body1.total_tng_contacts[0], 5)

        # Given
        eqs1 = [
            Group(equations=[
                UpdateTangentialContacts(dest='body1', sources=["body2"]),
            ]),
        ]
        a_eval = self._make_accel_eval(eqs1, body1, body2)

        # When
        a_eval.evaluate(0.0, 0.1)

        np.testing.assert_equal(body1.total_tng_contacts[0], 4)

        # Given
        body1.tng_idx[0] = 1
        body1.tng_idx_dem_id[0] = 0
        body1.tng_idx[1] = 1
        body1.tng_idx_dem_id[1] = 1
        body1.tng_idx[2] = 2
        body1.tng_idx_dem_id[2] = 0
        body1.tng_idx[3] = 2
        body1.tng_idx_dem_id[3] = 1
        body1.tng_idx[4] = 3
        body1.tng_idx_dem_id[4] = 0
        body1.tng_idx[5] = 3
        body1.tng_idx_dem_id[5] = 1
        body1.total_tng_contacts[0] = 6
        eqs1 = [
            Group(equations=[
                UpdateTangentialContacts(dest='body1',
                                         sources=["body2", "body1"]),
            ]),
        ]
        a_eval = self._make_accel_eval(eqs1, body1, body2)

        # When
        a_eval.evaluate(0.0, 0.1)
        np.testing.assert_equal(body1.total_tng_contacts[0], 4)


class RigidBodyStep(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        dx = 0.1
        rho = 1000
        m = rho * dx * dx
        x, y = np.mgrid[0:1:dx, 0:1:dx]
        x = x.ravel()
        y = y.ravel()

        # apply some forces on each bodies
        fx = np.random.rand(len(x)) * 100
        fy = np.random.rand(len(x)) * 100

        # create a body using rotation matrices
        self.body_R = get_particle_array_rigid_body_rotation_matrix(
            x=x, y=y, m=m, fx=fx, fy=fy, h=1.2 * dx, name='matrix')

        # create a body using quaternion
        self.body_q = get_particle_array_rigid_body_quaternion(
            x=x, y=y, m=m, fx=fx, fy=fy, h=1.2 * dx, name='quat')

    def _make_accel_eval(self, equations, pa):
        kernel = CubicSpline(dim=self.dim)
        seval = SPHEvaluator(arrays=[pa], equations=equations, dim=self.dim,
                             kernel=kernel)
        return seval

    def test_forces_and_torques_to_be_same(self):
        # Given
        eqs1 = [
            Group(equations=[
                SumUpExternalForces(dest='matrix', sources=None),
            ]),
        ]
        a_eval1 = self._make_accel_eval(eqs1, self.body_R)

        eqs2 = [
            Group(equations=[
                SumUpExternalForces(dest='quat', sources=None),
            ]),
        ]
        a_eval2 = self._make_accel_eval(eqs2, self.body_q)

        # When
        a_eval1.evaluate(0.0, 0.1)
        a_eval2.evaluate(0.0, 0.1)
        print("inside test forces")

        # Then
        np.testing.assert_array_almost_equal(self.body_R.torque,
                                             self.body_q.torque)

        np.testing.assert_array_almost_equal(self.body_R.force,
                                             self.body_q.force)

    def _setup_integrator(self, equations, pa_arrays, integrator):
        kernel = CubicSpline(dim=1)
        a_eval = AccelerationEval(particle_arrays=pa_arrays,
                                  equations=equations, kernel=kernel)
        comp = SPHCompiler(a_eval, integrator=integrator)
        comp.compile()
        nnps = LinkedListNNPS(dim=kernel.dim, particles=pa_arrays)
        a_eval.set_nnps(nnps)
        integrator.set_nnps(nnps)

    def test_angular_momentum(self):
        # Given
        eqs1 = [
            Group(equations=[
                SumUpExternalForces(dest='matrix', sources=None),
            ]),
        ]
        a_eval1 = self._make_accel_eval(eqs1, self.body_R)

        eqs2 = [
            Group(equations=[
                SumUpExternalForces(dest='quat', sources=None),
            ]),
        ]
        a_eval2 = self._make_accel_eval(eqs2, self.body_q)

        # When
        a_eval1.evaluate(0.0, 0.1)
        a_eval2.evaluate(0.0, 0.1)
        print("inside test forces")

        integrator = EPECIntegrator(matrix=RK2StepRigidBodyRotationMatrices())
        equations = [
            Group(equations=[
                SumUpExternalForces(dest='quat', sources=None),
            ]),
        ]
        self._setup_integrator(equations=equations, integrator=integrator)
        tf = np.pi
        dt = 0.02 * tf


class TestTorqueOnRigidBody(unittest.TestCase):
    def setUp(self):
        # create a rigid body with four particles each placed
        # at the corners of a square
        self.dim = 2
        # and the mass be unit
        m = 1.
        x = np.array([0., 1., 1., 0.])
        y = np.array([0., 0., 1., 1.])

        # create the center of mass
        self.body = get_particle_array_rigid_body_quaternion(
            x=x, y=y, m=m, name='body')

    def _make_accel_eval(self, equations, pa):
        kernel = CubicSpline(dim=self.dim)
        seval = SPHEvaluator(arrays=[pa], equations=equations, dim=self.dim,
                             kernel=kernel)
        return seval

    def test_center_of_mass(self):
        com_expected = np.array([0.5, 0.5, 0.])
        np.testing.assert_array_almost_equal(self.body.cm, com_expected)

    def test_torque(self):
        # apply force on one of the body particle. Here say particle with index
        # 2 in the x direction.
        # body looks like following
        #          o 3                       o  2           #
        #                                                   #
        #                                                   #
        #                                                   #
        #                                                   #
        #                                                   #
        #                                                   #
        #                                                   #
        #                                                   #
        #                                                   #
        #          o 0                       o  1           #

        body = self.body
        body.fx[2] = 1.
        # vector from center to particle 2
        xij = np.array([0.5, 0.5, 0.])
        # torque due to force at idx 2 is
        torque_expected = np.cross(xij, [1., 0., 0.])

        # Given
        eqs1 = [
            Group(equations=[
                SumUpExternalForces(dest='body', sources=None),
            ]),
        ]
        a_eval = self._make_accel_eval(eqs1, self.body)

        # When
        a_eval.evaluate(0.0, 0.1)

        # then
        np.testing.assert_array_almost_equal(self.body.torque, torque_expected)

        # similarly if the force is applied at index 0
        # reset force
        body.fx[:] = 0.
        body.fx[0] = 1.
        # when
        a_eval.evaluate(0.0, 0.1)

        # the torque will be of same magnitude with different direction
        # then
        np.testing.assert_array_almost_equal(self.body.torque,
                                             -torque_expected)

        # if the force is applied in the direction of the vector passing
        # from particle to com then the torque will be zero
        # reset force
        body.fx[:] = 0.
        body.fx[0] = 0.5
        body.fy[0] = 0.5
        # when
        a_eval.evaluate(0.0, 0.1)

        # the torque will be of same magnitude with different direction
        # then
        torque_expected = np.array([0., 0., 0.])
        force_expected = np.array([0.5, 0.5, 0.])
        np.testing.assert_array_almost_equal(self.body.torque, torque_expected)
        np.testing.assert_array_almost_equal(self.body.force, force_expected)

        # In the following cases lets apply force on two particles and check
        # total force and torque due to both the particles

        # case 1
        body.fx[:] = 0.
        body.fy[:] = 0.
        # same force in the same direction on both the particles
        body.fx[1] = 0.5
        body.fx[2] = 0.5
        # when
        a_eval.evaluate(0.0, 0.1)
        # then
        torque_expected = np.array([0., 0., 0.])
        force_expected = np.array([1, 0., 0.])
        np.testing.assert_array_almost_equal(self.body.torque, torque_expected)
        np.testing.assert_array_almost_equal(self.body.force, force_expected)

        # case 2
        body.fx[:] = 0.
        body.fy[:] = 0.
        # same force in the different direction on both the particles
        body.fx[0] = 1.
        body.fx[2] = -1.
        # when
        a_eval.evaluate(0.0, 0.1)
        # then
        # vector from center to particle 2
        xij = np.array([0.5, 0.5, 0.])
        # torque due to force at idx 2 is
        torque_expected_1 = np.cross(xij, [-1., 0., 0.])
        # vector from center to particle 0
        xij = np.array([-0.5, -0.5, 0.])
        # torque due to force at idx 0 is
        torque_expected_2 = np.cross(xij, [1., 0., 0.])
        torque_expected = torque_expected_1 + torque_expected_2
        force_expected = np.array([0, 0., 0.])
        np.testing.assert_array_almost_equal(self.body.torque, torque_expected)
        np.testing.assert_array_almost_equal(self.body.force, force_expected)


class TestOrientationOfRigidBody(unittest.TestCase):
    def setUp(self):
        # create a rigid body with four particles each placed
        # at the corners of a square
        self.dim = 2
        # and the mass be unit
        m = 1.
        x = np.array([0., 1., 1., 0.])
        y = np.array([0., 0., 1., 1.])

        self.body_q = get_particle_array_rigid_body_quaternion(
            x=x, y=y, m=m, name='body_q')

        self.body_r = get_particle_array_rigid_body_rotation_matrix(
            x=x, y=y, m=m, name='body_r')

    def _setup_integrator(self, equations, integrator):
        kernel = CubicSpline(dim=self.dim)
        arrays = [self.body_q, self.body_r]
        a_eval = AccelerationEval(particle_arrays=arrays, equations=equations,
                                  kernel=kernel)
        comp = SPHCompiler(a_eval, integrator=integrator)
        comp.compile()
        nnps = LinkedListNNPS(dim=kernel.dim, particles=arrays)
        a_eval.set_nnps(nnps)
        integrator.set_nnps(nnps)

    def _integrate(self, integrator, dt, tf):
        """The post_step_callback is called after each step and is passed the
        current time.
        """
        t = 0.0
        while t < tf:
            integrator.step(t, dt)
            t += dt

    def _make_accel_eval(self, equations, pa):
        kernel = CubicSpline(dim=self.dim)
        seval = SPHEvaluator(arrays=[pa], equations=equations, dim=self.dim,
                             kernel=kernel)
        return seval

    def get_equation(self, name):
        eqs = [
            Group(equations=[
                SumUpExternalForces(dest=name, sources=None),
            ])
        ]
        return eqs

    def compare_quaternion_rotation_matrix(self):
        # first apply some force on the bodies
        b_r = self.body_r
        b_q = self.body_q

        print("orientation of rotation matrix")
        print(b_r.R.reshape(3, 3))
        # positive x force on particle index 0
        b_r.fx[0] = 1.
        b_q.fx[0] = 1.

        # Update body to the next step
        integrator = EPECIntegrator(body_r=RK2StepRigidBodyRotationMatrices(),
                                    body_q=RK2StepRigidBodyQuaternions())
        equations = [
            SumUpExternalForces(dest="body_r", sources=None),
            SumUpExternalForces(dest="body_q", sources=None),
        ]
        self._setup_integrator(equations=equations, integrator=integrator)
        dt = 1
        tf = 1. * dt
        self._integrate(integrator, dt, tf)

        # now check the orientation
        print("orientation of rotation matrix")
        print(b_r.R.reshape(3, 3))
        print(b_r.torque)
        print(b_q.R.reshape(3, 3))
        print(b_q.torque)
