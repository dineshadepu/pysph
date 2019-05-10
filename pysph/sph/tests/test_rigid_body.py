import numpy as np
import unittest
from pysph.sph.rigid_body import (get_particle_array_rigid_body_dem,
                                  UpdateTangentialContacts)
from pysph.base.kernels import CubicSpline
from pysph.base.nnps import LinkedListNNPS
from pysph.sph.sph_compiler import SPHCompiler
from pysph.sph.equation import Equation, Group
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.acceleration_eval import AccelerationEval


class TestUpdateTangentialContacts(unittest.TestCase):
    def setUp(self):
        self.dim = 2

    def _make_accel_eval(self, equations, *args):
        kernel = CubicSpline(dim=self.dim)
        seval = SPHEvaluator(arrays=list(args), equations=equations, dim=self.dim,
                             kernel=kernel)
        return seval

    def test_tangential_contacts(self):
        # create a body with a single dem id
        x = np.array([0., 1.0, 2., -0.8])
        y = np.array([0., 0., 0., 0.])
        rad_s = np.array([0.7, 0.7, 0.7, 0.7])
        body1 = get_particle_array_rigid_body_dem(x=x, y=y, rad_s=rad_s, m=1,
                                                 h=1.2, dem_id=0, name="body1")

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
        body2 = get_particle_array_rigid_body_dem(x=x, y=y, rad_s=rad_s, m=1,
                                                  h=1.2, dem_id=1, name="body2")
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
        eqs1 = [
            Group(equations=[
                UpdateTangentialContacts(dest='body1', sources=["body2", "body1"]),
            ]),
        ]
        a_eval = self._make_accel_eval(eqs1, body1, body2)

        # When
        a_eval.evaluate(0.0, 0.1)
        print(body1.tng_idx)

        np.testing.assert_equal(body1.total_tng_contacts[0], 4)
