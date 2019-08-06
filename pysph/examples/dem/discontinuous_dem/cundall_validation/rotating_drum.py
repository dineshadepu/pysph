from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser
from pysph.dem.discontinuous_dem.dem_2d_linear_cundall import (
    Dem2dCundallScheme, get_particle_array_dem_2d_linear_cundall,
    UpdateTangentialContactsCundall2d, RK2StepDEM2dCundall,
    Cundall2dForceStage1, Cundall2dForceStage2, BodyForce,
)
from pysph.dem.discontinuous_dem.dem_nonlinear import EPECIntegratorMultiStage
from pysph.sph.equation import Group, MultiStageEquations
from pysph.tools.geometry import get_2d_hollow_circle
from pysph.sph.rigid_body import (RigidBodyMotion,
                                  RK2StepRigidBody, get_particle_array_rigid_body_quaternion,
                                  RK2StepRigidBodyQuaternions)


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class RotatingDrum(Application):
    def __init__(self):
        super(RotatingDrum, self).__init__()

    def initialize(self):
        self.dx = 0.05
        self.dt = 1e-4
        self.tf = 5
        self.dim = 2
        self.en = 0.1
        self.kn = 1e5
        self.rotate_time = 1.
        # friction coefficient
        self.mu = 0.5
        self.gy = -9.81
        self.seval = None
        self.pfreq = 100

    def create_particles(self):
        dx = self.dx

        # create 2d drum with particles
        xd, yd = get_2d_hollow_circle(dx, 1.)

        rho = 2699.
        m = rho * dx**2.
        drum = get_particle_array_rigid_body_quaternion(
            x=xd, y=yd, m=m, rho=rho, name="drum", rad_s=self.dx / 2)
        drum.add_property('dem_id', type='int')
        drum.add_property('theta_dot')
        drum.dem_id[:] = 2
        add_properties(drum, 'wx', 'wy', 'wz')

        # create bunch of particle
        limit = 0.55
        xp, yp = np.mgrid[-limit:limit:dx, -limit:limit:dx]

        rad_s = np.ones_like(xp) * self.dx / 2.
        rho = 2500
        m = rho * rad_s**2.
        inertia = m * 2. * rad_s**2. / 10.
        m_inverse = 1. / m
        I_inverse = 1. / inertia
        sand = get_particle_array_dem_2d_linear_cundall(
            x=xp, y=yp, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=rad_s, dem_id=1, h=1.2 * self.dx / 2., name="sand")

        return [sand, drum]

    def create_equations(self):
        eq1 = [
            Group(
                equations=[
                    BodyForce(dest='sand', sources=None, gx=0.0, gy=-9.81),
                    Cundall2dForceStage1(dest='sand', sources=['drum', 'sand'],
                                         kn=self.kn, mu=0.5, en=self.en)
                ])
        ]
        eq2 = [
            Group(
                equations=[
                    BodyForce(dest='sand', sources=None, gx=0.0, gy=-9.81),
                    Cundall2dForceStage2(dest='sand', sources=['drum', 'sand'],
                                         kn=self.kn, mu=0.5, en=self.en)
                ])
        ]

        return MultiStageEquations([eq1, eq2])

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = EPECIntegratorMultiStage(sand=RK2StepDEM2dCundall(),
                                              drum=RK2StepRigidBodyQuaternions())

        dt = self.dt
        tf = self.tf
        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
                        dt=dt, tf=tf)
        solver.set_disable_output(True)
        return solver

    def _make_accel_eval(self, equations, pa_arrays):
        from pysph.tools.sph_evaluator import SPHEvaluator
        if self.seval is None:
            kernel = CubicSpline(dim=self.dim)
            seval = SPHEvaluator(arrays=pa_arrays, equations=equations,
                                 dim=self.dim, kernel=kernel)
            self.seval = seval
            return self.seval
        else:
            return self.seval

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        T = self.rotate_time
        if t > T and t < T + 0.5:
            for pa in self.particles:
                if pa.name == 'drum':
                    pa.omega[2] += 5. * dt

        eqs1 = [
            Group(equations=[
                UpdateTangentialContactsCundall2d(dest='sand',
                                                  sources=['sand', 'drum']),
            ])
        ]
        arrays = self.particles
        a_eval = self._make_accel_eval(eqs1, arrays)

        # When
        a_eval.evaluate(t, dt)


if __name__ == '__main__':
    app = RotatingDrum()
    app.run()
