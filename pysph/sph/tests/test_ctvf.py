import os
# numpy
import numpy as np
from numpy import pi, sin, cos

from matplotlib import pyplot as plt

from pysph.base.kernels import QuinticSpline

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application

from pysph.sph.equation import Group
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.utils import (get_particle_array_wcsph)
from pysph.sph.ctvf import (SummationDensityTmp, GradientRhoTmp,
                            PsuedoForceOnFreeSurface, add_ctvf_properties,
                            MinNeighbourRho)

from pysph.tools.geometry import get_2d_block
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.solver.output import dump

from compyle.api import get_config


def create_domain(L):
    # domain for periodicity
    domain = DomainManager(xmin=0, xmax=L, ymin=0, ymax=L, periodic_in_x=True,
                           periodic_in_y=True)
    return domain


def create_circle(diameter=1, spacing=0.05, center=None):
    dx = spacing
    x = [0.0]
    y = [0.0]
    r = spacing
    nt = 0
    while r < diameter / 2:
        nnew = int(np.pi * r**2 / dx**2 + 0.5)
        tomake = nnew - nt
        theta = np.linspace(0., 2. * np.pi, tomake + 1)
        for t in theta[:-1]:
            x.append(r * np.cos(t))
            y.append(r * np.sin(t))
        nt = nnew
        r = r + dx
    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


def create_sin(dx, L):
    dx = dx
    x, y = get_2d_block(dx, L, L, center=[L / 2., L / 2.])

    sin_indices = []
    for i in range(0, len(x)):
        if y[i] < L / 2. + np.sin(np.pi * x[i]):
            sin_indices.append(i)

    return x[sin_indices], y[sin_indices]


# def create_equations(solid):
#     solid_name = solid.name
#     # create all the particles
#     eqn = [
#         Group(
#             equations=[
#                 VelocityGradient(dest=solid_name, sources=[solid_name]),
#             ], ),
#         Group(
#             equations=[
#                 DeviatoricStressRate(dest=solid_name, sources=[solid_name]),
#             ], ),
#     ]

#     return eqn


def evaluate(parrays, equations, kernel, dim):
    # domain_manager = create_domain(L)
    # return SPHEvaluator(arrays=parrays, equations=equations, dim=dim,
    #                     kernel=kernel, domain_manager=domain_manager)
    return SPHEvaluator(arrays=parrays, equations=equations, dim=dim,
                        kernel=kernel)


def test_summation_density_tmp_and_gradrho_circle_case_1():
    # ------------------
    # flags of compyle
    # ------------------

    get_config().use_openmp = True

    # ------------------
    # flags of compyle
    # ------------------

    # ----------------
    # create particles
    # ----------------
    dim = 2
    dx = 0.03
    rad = 0.5
    # center = [0.0, 0.0]
    x, y = create_circle(2. * rad, dx)
    rho = 1000.
    m = rho * dx**2.
    h = 1.2 * dx

    fluid = get_particle_array_wcsph(x=x, y=y, m=m, h=h, rho=rho, name="fluid")
    add_ctvf_properties(fluid)
    # ----------------
    # create particles ends
    # ----------------

    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS
    # ----------------
    filename = "sum_grad_case_1_output"
    os.makedirs(filename, exist_ok=True)
    dump(os.path.join(filename, 'sum_grad_case_1_0'), [fluid],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=False)
    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS ENDS
    # ----------------

    # ---------------------
    # kernel
    # ---------------------
    kernel = QuinticSpline(dim=dim)
    # ---------------------
    # kernel ends
    # ---------------------

    # ---------------------
    # Create equations
    # ---------------------
    fluid_name = fluid.name
    # create all the particles
    eqns = [
        Group(
            equations=[
                SummationDensityTmp(dest=fluid_name, sources=[fluid_name]),
            ], ),
        Group(
            equations=[
                GradientRhoTmp(dest=fluid_name, sources=[fluid_name]),
            ], ),
    ]
    sph_eval = evaluate([fluid], eqns, kernel, dim)
    sph_eval.evaluate()
    # ---------------------
    # Create equations ends
    # ---------------------

    # ----------------
    # DUMP THE AFTER BEFORE RUNNING THE EQUATIONS
    # ----------------
    # cond = fluid.x**2. + fluid.y**2. < 0.46**2.
    # fluid.tag[cond] = 2
    # fluid.align_particles()

    dump(os.path.join(filename, 'sum_grad_case_1_1'), [fluid],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=True)
    # ----------------
    # DUMP THE DATA AFTER RUNNING THE EQUATIONS ENDS
    # ----------------

    # fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # ax = ax.ravel()
    # plt.jet()

    # im0 = ax[0].scatter(fluid.x, fluid.y, c=fluid.u)
    # # ax[0].scatter(s.x, s.y, c=s.p)
    # # ax[0].scatter(b.x, b.y, c=b.p)
    # ax[0].set_title('u velocity')
    # # plt.axis('equal')
    # fig.colorbar(im0, ax=ax[0])

    # im1 = ax[1].scatter(fluid.x, fluid.y, c=fluid.v)
    # fig.colorbar(im1, ax=ax[1])
    # ax[1].set_title('v velocity')
    # plt.savefig(os.path.join(filename, 'velocity.png'))
    # plt.show()

    # fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # ax = ax.ravel()
    # plt.jet()

    # im0 = ax[0].scatter(fluid.x, fluid.y, c=fluid.arho)
    # # ax[0].scatter(s.x, s.y, c=s.p)
    # # ax[0].scatter(b.x, b.y, c=b.p)
    # ax[0].set_title('epsilon xx')
    # # plt.axis('equal')
    # fig.colorbar(im0, ax=ax[0])

    # im1 = ax[1].scatter(fluid.x, fluid.y, c=fluid.arho)
    # fig.colorbar(im1, ax=ax[1])
    # ax[1].set_title('epsilon yy')
    # plt.savefig(os.path.join(filename, 'epsilon_xx.png'))
    # plt.show()


def test_summation_density_tmp_and_gradrho_sin_case_2():
    # ------------------
    # flags of compyle
    # ------------------

    get_config().use_openmp = True

    # ------------------
    # flags of compyle
    # ------------------

    # ----------------
    # create particles
    # ----------------
    dim = 2
    dx = 0.1
    x, y = create_sin(dx, 2. * np.pi)
    rho = 1000.
    m = rho * dx**2.
    h = 1.2 * dx

    fluid = get_particle_array_wcsph(x=x, y=y, m=m, h=h, rho=rho, name="fluid")
    add_ctvf_properties(fluid)
    # ----------------
    # create particles ends
    # ----------------

    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS
    # ----------------
    filename = "sum_grad_case_2_output"
    os.makedirs(filename, exist_ok=True)
    dump(os.path.join(filename, 'sum_grad_case_2_0'), [fluid],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=False)
    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS ENDS
    # ----------------

    # ---------------------
    # kernel
    # ---------------------
    kernel = QuinticSpline(dim=dim)
    # ---------------------
    # kernel ends
    # ---------------------

    # ---------------------
    # Create equations
    # ---------------------
    fluid_name = fluid.name
    # create all the particles
    eqns = [
        Group(
            equations=[
                SummationDensityTmp(dest=fluid_name, sources=[fluid_name]),
            ], ),
        Group(
            equations=[
                GradientRhoTmp(dest=fluid_name, sources=[fluid_name]),
            ], ),
    ]
    sph_eval = evaluate([fluid], eqns, kernel, dim)
    sph_eval.evaluate()
    # ---------------------
    # Create equations ends
    # ---------------------

    # ----------------
    # DUMP THE AFTER BEFORE RUNNING THE EQUATIONS
    # ----------------
    # cond = fluid.x**2. + fluid.y**2. < 0.46**2.
    # fluid.tag[cond] = 2
    # fluid.align_particles()

    dump(os.path.join(filename, 'sum_grad_case_2_1'), [fluid],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=True)
    # ----------------
    # DUMP THE DATA AFTER RUNNING THE EQUATIONS ENDS
    # ----------------

    # fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # ax = ax.ravel()
    # plt.jet()

    # im0 = ax[0].scatter(fluid.x, fluid.y, c=fluid.u)
    # # ax[0].scatter(s.x, s.y, c=s.p)
    # # ax[0].scatter(b.x, b.y, c=b.p)
    # ax[0].set_title('u velocity')
    # # plt.axis('equal')
    # fig.colorbar(im0, ax=ax[0])

    # im1 = ax[1].scatter(fluid.x, fluid.y, c=fluid.v)
    # fig.colorbar(im1, ax=ax[1])
    # ax[1].set_title('v velocity')
    # plt.savefig(os.path.join(filename, 'velocity.png'))
    # plt.show()

    # fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # ax = ax.ravel()
    # plt.jet()

    # im0 = ax[0].scatter(fluid.x, fluid.y, c=fluid.arho)
    # # ax[0].scatter(s.x, s.y, c=s.p)
    # # ax[0].scatter(b.x, b.y, c=b.p)
    # ax[0].set_title('epsilon xx')
    # # plt.axis('equal')
    # fig.colorbar(im0, ax=ax[0])

    # im1 = ax[1].scatter(fluid.x, fluid.y, c=fluid.arho)
    # fig.colorbar(im1, ax=ax[1])
    # ax[1].set_title('epsilon yy')
    # plt.savefig(os.path.join(filename, 'epsilon_xx.png'))
    # plt.show()


def test_summation_density_tmp_and_gradrho_with_boundary_dam_break_case_1():
    # ------------------
    # flags of compyle
    # ------------------

    get_config().use_openmp = True

    # ------------------
    # flags of compyle
    # ------------------

    # ----------------
    # create particles
    # ----------------
    dim = 2
    dx = 0.1
    tank_length = 4.
    tank_height = 4.
    # center = [0.0, 0.0]
    # x, y = create_circle(2. * rad, dx)
    rho = 1000.
    m = rho * dx**2.
    h = 1.2 * dx

    fluid_length = 2.
    fluid_height = 2.

    xt, yt = get_2d_tank(dx, length=tank_length, height=tank_height,
                         base_center=[2, 0], num_layers=2)

    xf, yf = get_2d_block(dx=dx, length=fluid_length, height=fluid_height,
                          center=[0.5, 1])

    xf += 6. * dx
    yf += 1. * dx

    fluid = get_particle_array_wcsph(x=xf, y=yf, m=m, h=h, rho=rho,
                                     name="fluid")
    tank = get_particle_array_wcsph(x=xt, y=yt, m=m, h=h, rho=rho, name="tank")

    add_ctvf_properties(fluid)
    add_ctvf_properties(tank)
    # ----------------
    # create particles ends
    # ----------------

    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS
    # ----------------
    filename = "sum_grad_with_boundary_case_output"
    os.makedirs(filename, exist_ok=True)
    dump(os.path.join(filename,
                      'sum_grad_with_boundary_case_1_0'), [fluid, tank],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=False)
    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS ENDS
    # ----------------

    # ---------------------
    # kernel
    # ---------------------
    kernel = QuinticSpline(dim=dim)
    # ---------------------
    # kernel ends
    # ---------------------

    # ---------------------
    # Create equations
    # ---------------------
    fluid_name = fluid.name
    tank_name = tank.name
    # create all the particles
    eqns = [
        Group(
            equations=[
                SummationDensityTmp(dest=fluid_name,
                                    sources=[fluid_name, tank_name]),
                SummationDensityTmp(dest=tank_name,
                                    sources=[fluid_name, tank_name]),
            ], ),
        Group(
            equations=[
                # experimental
                GradientRhoTmp(dest=fluid_name,
                               sources=[fluid_name, tank_name]),
            ], ),
    ]
    sph_eval = evaluate([fluid, tank], eqns, kernel, dim)
    sph_eval.evaluate()
    # ---------------------
    # Create equations ends
    # ---------------------

    # ----------------
    # DUMP THE AFTER BEFORE RUNNING THE EQUATIONS
    # ----------------
    # cond = fluid.x**2. + fluid.y**2. < 0.46**2.
    # fluid.tag[cond] = 2
    # fluid.align_particles()

    dump(os.path.join(filename,
                      'sum_grad_with_boundary_case_1_1'), [fluid, tank],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=True)
    # ----------------
    # DUMP THE DATA AFTER RUNNING THE EQUATIONS ENDS
    # ----------------

    # fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # ax = ax.ravel()
    # plt.jet()

    # im0 = ax[0].scatter(fluid.x, fluid.y, c=fluid.u)
    # # ax[0].scatter(s.x, s.y, c=s.p)
    # # ax[0].scatter(b.x, b.y, c=b.p)
    # ax[0].set_title('u velocity')
    # # plt.axis('equal')
    # fig.colorbar(im0, ax=ax[0])

    # im1 = ax[1].scatter(fluid.x, fluid.y, c=fluid.v)
    # fig.colorbar(im1, ax=ax[1])
    # ax[1].set_title('v velocity')
    # plt.savefig(os.path.join(filename, 'velocity.png'))
    # plt.show()

    # fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # ax = ax.ravel()
    # plt.jet()

    # im0 = ax[0].scatter(fluid.x, fluid.y, c=fluid.arho)
    # # ax[0].scatter(s.x, s.y, c=s.p)
    # # ax[0].scatter(b.x, b.y, c=b.p)
    # ax[0].set_title('epsilon xx')
    # # plt.axis('equal')
    # fig.colorbar(im0, ax=ax[0])

    # im1 = ax[1].scatter(fluid.x, fluid.y, c=fluid.arho)
    # fig.colorbar(im1, ax=ax[1])
    # ax[1].set_title('epsilon yy')
    # plt.savefig(os.path.join(filename, 'epsilon_xx.png'))
    # plt.show()


def test_summation_density_tmp_and_gradrho_sin_case_2():
    # ------------------
    # flags of compyle
    # ------------------

    get_config().use_openmp = True

    # ------------------
    # flags of compyle
    # ------------------

    # ----------------
    # create particles
    # ----------------
    dim = 2
    dx = 0.1
    x, y = create_sin(dx, 2. * np.pi)
    rho = 1000.
    m = rho * dx**2.
    h = 1.2 * dx

    fluid = get_particle_array_wcsph(x=x, y=y, m=m, h=h, rho=rho, name="fluid")
    add_ctvf_properties(fluid)
    # ----------------
    # create particles ends
    # ----------------

    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS
    # ----------------
    filename = "sum_grad_case_2_output"
    os.makedirs(filename, exist_ok=True)
    dump(os.path.join(filename, 'sum_grad_case_2_0'), [fluid],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=False)
    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS ENDS
    # ----------------

    # ---------------------
    # kernel
    # ---------------------
    kernel = QuinticSpline(dim=dim)
    # ---------------------
    # kernel ends
    # ---------------------

    # ---------------------
    # Create equations
    # ---------------------
    fluid_name = fluid.name
    # create all the particles
    eqns = [
        Group(
            equations=[
                SummationDensityTmp(dest=fluid_name, sources=[fluid_name]),
            ], ),
        Group(
            equations=[
                GradientRhoTmp(dest=fluid_name, sources=[fluid_name]),
            ], ),
    ]
    sph_eval = evaluate([fluid], eqns, kernel, dim)
    sph_eval.evaluate()
    # ---------------------
    # Create equations ends
    # ---------------------

    # ----------------
    # DUMP THE AFTER BEFORE RUNNING THE EQUATIONS
    # ----------------
    # cond = fluid.x**2. + fluid.y**2. < 0.46**2.
    # fluid.tag[cond] = 2
    # fluid.align_particles()

    dump(os.path.join(filename, 'sum_grad_case_2_1'), [fluid],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=True)
    # ----------------
    # DUMP THE DATA AFTER RUNNING THE EQUATIONS ENDS
    # ----------------


def test_min_rho():
    # ------------------
    # flags of compyle
    # ------------------

    get_config().use_openmp = True

    # ------------------
    # flags of compyle
    # ------------------

    # ----------------
    # create particles
    # ----------------
    dim = 2
    dx = 0.1
    x, y = create_sin(dx, 2. * np.pi)
    rho = 1000.
    m = rho * dx**2.
    h = 1.2 * dx

    fluid = get_particle_array_wcsph(x=x, y=y, m=m, h=h, rho=rho, name="fluid")
    add_ctvf_properties(fluid)
    # ----------------
    # create particles ends
    # ----------------

    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS
    # ----------------
    filename = "min_rho_sin_output"
    os.makedirs(filename, exist_ok=True)
    dump(os.path.join(filename, 'min_rho_sin_0'), [fluid],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=False)
    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS ENDS
    # ----------------

    # ---------------------
    # kernel
    # ---------------------
    kernel = QuinticSpline(dim=dim)
    # ---------------------
    # kernel ends
    # ---------------------

    # ---------------------
    # Create equations
    # ---------------------
    fluid_name = fluid.name
    # create all the particles
    eqns = [
        Group(
            equations=[
                SummationDensityTmp(dest=fluid_name, sources=[fluid_name]),
            ], ),
        Group(
            equations=[
                MinNeighbourRho(dest=fluid_name, sources=[fluid_name]),
                GradientRhoTmp(dest=fluid_name, sources=[fluid_name]),
            ], ),
    ]
    sph_eval = evaluate([fluid], eqns, kernel, dim)
    sph_eval.evaluate()
    # ---------------------
    # Create equations ends
    # ---------------------

    # ----------------
    # DUMP THE AFTER BEFORE RUNNING THE EQUATIONS
    # ----------------
    # cond = fluid.x**2. + fluid.y**2. < 0.46**2.
    # fluid.tag[cond] = 2
    # fluid.align_particles()

    dump(os.path.join(filename, 'min_rho_sin_1'), [fluid],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=True)
    # ----------------
    # DUMP THE DATA AFTER RUNNING THE EQUATIONS ENDS
    # ----------------


def test_psuedoforce_sin():
    # ------------------
    # flags of compyle
    # ------------------

    get_config().use_openmp = True

    # ------------------
    # flags of compyle
    # ------------------

    # ----------------
    # create particles
    # ----------------
    dim = 2
    dx = 0.1
    x, y = create_sin(dx, 2. * np.pi)
    rho = 1000.
    m = rho * dx**2.
    h = 1.2 * dx

    fluid = get_particle_array_wcsph(x=x, y=y, m=m, h=h, rho=rho, name="fluid")
    add_ctvf_properties(fluid)
    # ----------------
    # create particles ends
    # ----------------

    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS
    # ----------------
    filename = "psuedo_force_sin_output"
    os.makedirs(filename, exist_ok=True)
    dump(os.path.join(filename, 'psuedo_force_sin_0'), [fluid],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=False)
    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS ENDS
    # ----------------

    # ---------------------
    # kernel
    # ---------------------
    kernel = QuinticSpline(dim=dim)
    # ---------------------
    # kernel ends
    # ---------------------

    # ---------------------
    # Create equations
    # ---------------------
    fluid_name = fluid.name
    # create all the particles
    eqns = [
        Group(
            equations=[
                SummationDensityTmp(dest=fluid_name, sources=[fluid_name]),
            ], ),
        Group(
            equations=[
                MinNeighbourRho(dest=fluid_name, sources=[fluid_name]),
                GradientRhoTmp(dest=fluid_name, sources=[fluid_name]),
            ], ),
        Group(
            equations=[
                PsuedoForceOnFreeSurface(dest=fluid_name, sources=[fluid_name],
                                         dx=dx, m0=fluid.m[0], pb=1000.,
                                         rho=rho),
            ], ),
    ]
    sph_eval = evaluate([fluid], eqns, kernel, dim)
    sph_eval.evaluate()
    # ---------------------
    # Create equations ends
    # ---------------------

    # ----------------
    # DUMP THE AFTER BEFORE RUNNING THE EQUATIONS
    # ----------------
    # cond = fluid.x**2. + fluid.y**2. < 0.46**2.
    # fluid.tag[cond] = 2
    # fluid.align_particles()

    dump(os.path.join(filename, 'psuedo_force_sin_1'), [fluid],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=True)
    # ----------------
    # DUMP THE DATA AFTER RUNNING THE EQUATIONS ENDS
    # ----------------


def test_psuedoforce_sin_boundary():
    # ------------------
    # flags of compyle
    # ------------------

    get_config().use_openmp = True

    # ------------------
    # flags of compyle
    # ------------------

    # ----------------
    # create particles
    # ----------------
    dim = 2
    dx = 0.1
    tank_length = 4.
    tank_height = 4.
    # center = [0.0, 0.0]
    # x, y = create_circle(2. * rad, dx)
    rho = 1000.
    m = rho * dx**2.
    h = 1.2 * dx

    fluid_length = 2.
    fluid_height = 2.

    xt, yt = get_2d_tank(dx, length=tank_length, height=tank_height,
                         base_center=[2, 0], num_layers=2)

    xf, yf = get_2d_block(dx=dx, length=fluid_length, height=fluid_height,
                          center=[0.5, 1])

    xf += 6. * dx
    yf += 1. * dx

    fluid = get_particle_array_wcsph(x=xf, y=yf, m=m, h=h, rho=rho,
                                     name="fluid")
    tank = get_particle_array_wcsph(x=xt, y=yt, m=m, h=h, rho=rho, name="tank")

    add_ctvf_properties(fluid)
    add_ctvf_properties(tank)
    # ----------------
    # create particles ends
    # ----------------

    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS
    # ----------------
    filename = "psuedo_force_sin_boundary_output"
    os.makedirs(filename, exist_ok=True)
    dump(os.path.join(filename, 'psuedo_force_sin_boundary_0'), [fluid, tank],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=False)
    # ----------------
    # DUMP THE DATA BEFORE RUNNING THE EQUATIONS ENDS
    # ----------------

    # ---------------------
    # kernel
    # ---------------------
    kernel = QuinticSpline(dim=dim)
    # ---------------------
    # kernel ends
    # ---------------------

    # ---------------------
    # Create equations
    # ---------------------
    fluid_name = fluid.name
    tank_name = tank.name
    # create all the particles
    eqns = [
        Group(
            equations=[
                SummationDensityTmp(dest=fluid_name,
                                    sources=[fluid_name, tank_name]),
                SummationDensityTmp(dest=tank_name,
                                    sources=[fluid_name, tank_name]),
            ], ),
        Group(
            equations=[
                MinNeighbourRho(dest=fluid_name, sources=[fluid_name]),
                GradientRhoTmp(dest=fluid_name,
                               sources=[fluid_name, tank_name]),
            ], ),
        Group(
            equations=[
                PsuedoForceOnFreeSurface(dest=fluid_name, sources=[fluid_name],
                                         dx=dx, m0=fluid.m[0], pb=1000.,
                                         rho=rho),
            ], ),
    ]
    sph_eval = evaluate([fluid, tank], eqns, kernel, dim)
    sph_eval.evaluate()
    # ---------------------
    # Create equations ends
    # ---------------------

    # ----------------
    # DUMP THE AFTER BEFORE RUNNING THE EQUATIONS
    # ----------------
    # cond = fluid.x**2. + fluid.y**2. < 0.46**2.
    # fluid.tag[cond] = 2
    # fluid.align_particles()

    dump(os.path.join(filename, 'psuedo_force_sin_boundary_1'), [fluid],
         dict(t=0, dt=0.1, count=0), detailed_output=True, only_real=True)
    # ----------------
    # DUMP THE DATA AFTER RUNNING THE EQUATIONS ENDS
    # ----------------


if __name__ == '__main__':
    # test_summation_density_tmp_and_gradrho_circle_case_1()
    # test_summation_density_tmp_and_gradrho_sin_case_2()
    # test_summation_density_tmp_and_gradrho_with_boundary_dam_break_case_1()

    # test_min_rho()

    test_psuedoforce_sin()
    # test_psuedoforce_sin_boundary()

# au, av, aw
# grad_rho_x, grad_rho_y, grad_rho_z
