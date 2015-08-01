"""Evolution of a circular patch of incompressible fluid. (20 seconds)

See J. J. Monaghan "Simulating Free Surface Flows with SPH", JCP, 1994, 100, pp
399 - 406

An initially circular patch of fluid is subjected to a velocity profile that
causes it to deform into an ellipse. Incompressibility causes the initially
circular patch to deform into an ellipse such that the area is conserved. An
analytical solution for the locus of the patch is available (exact_solution)

This is a standard test for the formulations for the incompressible SPH
equations.

"""
from __future__ import print_function

import os
from numpy import array, ones_like, mgrid, sqrt

# PySPH base and carray imports
from pysph.base.utils import get_particle_array_wcsph
from pysph.base.kernels import Gaussian

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator, PECIntegrator, TVDRK3Integrator
from pysph.sph.integrator_step import WCSPHStep, WCSPHTVDRK3Step

# PySPH sph imports
from pysph.sph.equation import Group
from pysph.sph.basic_equations import XSPHCorrection, ContinuityEquation
from pysph.sph.wc.basic import TaitEOS, MomentumEquation, UpdateSmoothingLengthFerrari, \
    ContinuityEquationDeltaSPH, MomentumEquationDeltaSPH

def _derivative(x, t):
    A, a = x
    Anew = A*A*(a**4 -1)/(a**4 + 1)
    anew = -a*A
    return array((Anew, anew))

def _scipy_integrate(y0, tf, dt):
    from scipy.integrate import odeint
    result = odeint(_derivative, y0, [0.0, tf])
    return result[-1]

def _numpy_integrate(y0, tf, dt):
    t = 0.0
    while t <= tf:
        t += dt
        y += dt*_derivative(y, t)
    return y

def exact_solution(tf=0.0075, dt=1e-6):
    """Exact solution for the locus of the circular patch.

    Returns the semi-minor axis, A, pressure, x, y.

    Where x, y are the points corresponding to the ellipse.
    """
    import numpy

    y0 = array([100.0, 1.0])

    try:
        from scipy.integrate import odeint
    except ImportError:
        Anew, anew = _numpy_integrate(y0, tf, dt)
    else:
        Anew, anew = _scipy_integrate(y0, tf, dt)

    dadt = _derivative([Anew, anew], tf)[0]
    po = 0.5*-anew**2 * (dadt - Anew**2)

    theta = numpy.linspace(0,2*numpy.pi, 101)

    return anew, Anew, po, anew*numpy.cos(theta), 1/anew*numpy.sin(theta)


class EllipticalDrop(Application):
    def initialize(self):
        self.co = 1400.0
        self.ro = 1.0
        self.hdx = 1.3
        self.dx = 0.025

    def create_particles(self):
        """Create the circular patch of fluid."""
        dx = self.dx
        hdx = self.hdx
        co = self.co
        ro = self.ro
        name = 'fluid'
        x, y = mgrid[-1.05:1.05+1e-4:dx, -1.05:1.05+1e-4:dx]
        x = x.ravel()
        y = y.ravel()

        m = ones_like(x)*dx*dx
        h = ones_like(x)*hdx*dx
        rho = ones_like(x) * ro

        p = ones_like(x) * 1./7.0 * co**2
        cs = ones_like(x) * co

        u = -100*x
        v = 100*y

        # remove particles outside the circle
        indices = []
        for i in range(len(x)):
            if sqrt(x[i]*x[i] + y[i]*y[i]) - 1 > 1e-10:
                indices.append(i)

        pa = get_particle_array_wcsph(x=x, y=y, m=m, rho=rho, h=h, p=p, u=u, v=v,
                                    cs=cs, name=name)
        pa.remove_particles(indices)

        print("Elliptical drop :: %d particles"%(pa.get_number_of_particles()))

        # add requisite variables needed for this formulation
        for name in ('arho', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'rho0', 'u0',
                    'v0', 'w0', 'x0', 'y0', 'z0'):
            pa.add_property(name)

        # set the output property arrays
        pa.set_output_arrays( ['x', 'y', 'u', 'v', 'rho', 'h', 'p', 'pid', 'tag', 'gid'] )

        return [pa]

    def create_solver(self):
        kernel = Gaussian(dim=2)

        # Create the Integrator. Currently, PySPH supports multi-stage,
        # predictor corrector and a TVD-RK3 integrators.

        #integrator = PECIntegrator(fluid=WCSPHStep())
        #integrator = EPECIntegrator(fluid=WCSPHStep())
        integrator = TVDRK3Integrator( fluid=WCSPHTVDRK3Step() )

        # Construct the solver. n_damp determines the iterations until which smaller
        # time-steps are used when using adaptive time-steps. Use the output_at_times
        # list to specify instants of time at which the output solution is
        # required.
        dt = 5e-6; tf = 0.0076
        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        dt=dt, tf=tf, adaptive_timestep=True,
                        cfl=0.05, n_damp=50,
                        output_at_times=[0.0008, 0.0038])

        # select True if you want to dump out remote particle properties in
        # parallel runs. This can be over-ridden with the --output-remote
        # command line option
        solver.set_output_only_real(True)

        return solver

    def create_equations(self):
        # Define the SPH equations used to solve this problem
        equations = [

            # Equation of state: p = f(rho)
            Group(equations=[
                    TaitEOS(dest='fluid', sources=None, rho0=self.ro, c0=self.co, gamma=7.0),
                    ], real=False),

            # Block for the accelerations. Choose between either the Delta-SPH
            # formulation or the standard Monaghan 1994 formulation
            Group( equations=[

                    # Density rate: drho/dt with dissipative penalization
                    #ContinuityEquationDeltaSPH(dest='fluid',  sources=['fluid',], delta=0.1, c0=co),
                    ContinuityEquation(dest='fluid',  sources=['fluid',]),

                    # Acceleration: du,v/dt
                    #MomentumEquationDeltaSPH(dest='fluid', sources=['fluid'], alpha=0.2, rho0=ro, c0=co),
                    MomentumEquation(dest='fluid', sources=['fluid'], alpha=0.2, beta=0.0),

                    # XSPH velocity correction
                    XSPHCorrection(dest='fluid', sources=['fluid']),

                    ],),

            # Update smoothing lengths at the end.
            Group( equations=[
                    UpdateSmoothingLengthFerrari(
                        dest='fluid', sources=None, dim=2, hdx=self.hdx
                    ),
                    ], real=False),


            ]
        return equations

    def post_process(self, info_file_or_dir):
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            print("Post processing requires matplotlib.")
            return
        if self.rank > 0:
            return
        info = self.read_info(info_file_or_dir)
        files = self.output_files
        last_output = files[-1]
        from pysph.solver.utils import load
        data = load(last_output)
        pa = data['arrays']['fluid']
        tf = data['solver_data']['t']
        a, A, po, xe, ye = exact_solution(tf)
        print("At tf=%s"%tf)
        print("Semi-major axis length (exact, computed) = %s, %s"
                %(1.0/a, max(pa.y)))
        plt.plot(xe, ye)
        plt.scatter(pa.x, pa.y, marker='.')
        plt.ylim(-2, 2)
        plt.xlim(plt.ylim())
        plt.title("Particles at %s secs"%tf)
        plt.xlabel('x'); plt.ylabel('y')
        fig = os.path.join(self.output_dir, "comparison.png")
        plt.savefig(fig, dpi=300)
        print("Figure written to %s."%fig)


if __name__ == '__main__':
    app = EllipticalDrop()
    app.run()
    app.post_process(app.info_filename)