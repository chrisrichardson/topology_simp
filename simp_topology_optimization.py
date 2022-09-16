#!/usr/bin/env python3
# coding: utf-8

# FEniCSx implementation

from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType, meshtags, locate_entities_boundary
from dolfinx.fem import FunctionSpace, VectorFunctionSpace, Function, assemble_scalar, locate_dofs_topological, form
from dolfinx.fem import Constant, dirichletbc, Expression
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from ufl import Measure, dx, TestFunction, TrialFunction
from ufl import sym, grad, dot, inner, div, Identity, action
import numpy as np

# Algorithmic parameters
niternp = 20  # number of non-penalized iterations
niter = 80   # total number of iterations
pmax = 4     # maximum SIMP exponent
exponent_update_frequency = 4  # minimum number of steps between exponent update
tol_mass = 1e-4  # tolerance on mass when finding Lagrange multiplier
thetamin = 0.001  # minimum density modeling void


# Mesh
mesh = create_rectangle(
    MPI.COMM_WORLD, [(-2, 0), (2, 1)], [100, 60], CellType.quadrilateral)


# Problem parameters
thetamoy = 0.4  # target average material density
E = 1.0
nu = 0.3
lamda = E*nu/(1+nu)/(1-2*nu)

mu = E/(2*(1+nu))
f = Constant(mesh, (0.0, -1.0))  # vertical downwards force

# Boundaries
left_boundary = locate_entities_boundary(mesh, dim=1,
                                         marker=lambda x: np.isclose(x[0], -2.0))

indices = locate_entities_boundary(mesh, dim=1,
                                   marker=lambda x: np.logical_and(np.isclose(x[0], 2.0), np.isclose(x[1], 0.5, atol=0.1)))
values = np.ones_like(indices)
facets = meshtags(mesh, mesh.topology.dim - 1, indices, values)
ds = Measure("ds", subdomain_data=facets)

# Function space for density field
V0 = FunctionSpace(mesh, ("DG", 0))
# Function space for displacement
V2 = VectorFunctionSpace(mesh, ("CG", 2))

# Fixed boundary condtions
left_dofs = locate_dofs_topological(V=V2,
                                    entity_dim=1,
                                    entities=left_boundary)
bc = dirichletbc(V=V2,
                 value=np.array((0.0, 0.0), dtype=np.float64),
                 dofs=left_dofs)

p = Constant(mesh, 1.0)  # SIMP penalty exponent
exponent_counter = 0  # exponent update counter
lagrange = Constant(mesh, 1.0)  # Lagrange multiplier for volume constraint

thetaold = Function(V0)
thetaold.vector.set(thetamoy)

coeff = thetaold**p
theta = Function(V0)

volume = assemble_scalar(form(1.*dx(domain=mesh)))
avg_density_0 = assemble_scalar(
    form(thetaold*dx))/volume  # initial average density
avg_density = 0.


def eps(v):
    return sym(grad(v))


def sigma(v):
    return coeff*(lamda*div(v)*Identity(2)+2*mu*eps(v))


def energy_density(u, v):
    return inner(sigma(u), eps(v))


# Inhomogeneous elastic variational problem
u_ = TestFunction(V2)
du = TrialFunction(V2)
a = inner(sigma(u_), eps(du))*dx
L = dot(f, u_)*ds(1)


def update_theta(u):
    theta_exp = Expression((p*coeff*energy_density(u, u)/lagrange)
                           ** (1/(p+1)), V0.element.interpolation_points())
    theta.interpolate(theta_exp)
    thetav = theta.vector[:]
    theta.vector[:] = np.maximum(np.minimum(1, thetav), thetamin)
#    theta.vector.apply("insert")
    avg_density = assemble_scalar(form(theta*dx))/volume
    return avg_density


# We now define a function for finding the correct value of the Lagrange multiplier $\lambda$. First, a rough bracketing of $\lambda$ is obtained, then a dichotomy is performed in the interval `[lagmin,lagmax]` until the correct average density is obtained to a certain tolerance.


def update_lagrange_multiplier(u, avg_density):

    avg_density1 = avg_density

    print(lagrange.value, avg_density1, avg_density_0)
    # Initial bracketing of Lagrange multiplier
    if (avg_density1 < avg_density_0):
        lagmin = float(lagrange.value)
        while (avg_density < avg_density_0):
            lagrange.value = 0.5 * lagrange.value
            avg_density = update_theta(u)
        lagmax = float(lagrange.value)
    elif (avg_density1 > avg_density_0):
        lagmax = float(lagrange.value)
        while (avg_density > avg_density_0):
            print('lval = ', lagrange.value)
            lagrange.value = 2.0 * lagrange.value
            avg_density = update_theta(u)
        lagmin = float(lagrange.value)
    else:
        lagmin = float(lagrange.value)
        lagmax = float(lagrange.value)

    # Dichotomy on Lagrange multiplier
    inddico = 0
    while ((abs(1.-avg_density/avg_density_0)) > tol_mass):
        print(lagmin, lagmax, lagrange.value)
        lagrange.value = 0.5 * (lagmax+lagmin)
        avg_density = update_theta(u)
        inddico += 1
        if (avg_density < avg_density_0):
            lagmin = float(lagrange.value)
        else:
            lagmax = float(lagrange.value)
    print("   Dichotomy iterations:", inddico)


# Finally, the exponent update strategy is implemented:
#
# * first, $p=1$ for the first `niternp` iterations
# * then, $p$ is increased by some amount which depends on the average gray level of the density field computed as $g = \frac{1}{\text{Vol(D)}}\int_D 4(\theta-\theta_{min})(1-\theta)\text{ dx}$, that is $g=0$ is $\theta(x)=\theta_{min}$ or $1$ everywhere and $g=1$ is $\theta=(\theta_{min}+1)/2$ everywhere.
# * Note that $p$ can increase only if at least `exponent_update_frequency` AM iterations have been performed since the last update and only if the compliance evolution falls below a certain threshold.


def update_exponent(exponent_counter):
    exponent_counter += 1
    if (i < niternp):
        p.value = 1.0
    elif (i >= niternp):
        if i == niternp:
            print("\n Starting penalized iterations\n")
        if ((abs(compliance-old_compliance) < 0.01*compliance_history[0]) and
                (exponent_counter > exponent_update_frequency)):
            # average gray level
            fgray = form((theta-thetamin)*(1.0-theta)*dx)
            gray_level = 4.0/volume * assemble_scalar(fgray)
            p.value = min(p.value*(1+0.3**(1.+gray_level/2)), pmax)
            exponent_counter = 0
            print("   Updated SIMP exponent p = ", p.value)
    return exponent_counter


u = Function(V2, name="Displacement")
old_compliance = 1e30
ffile = XDMFFile(MPI.COMM_WORLD, "topology_optimization.xdmf", "w")
ffile.write_mesh(mesh)

compliance_history = []
for i in range(niter):
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={
        "ksp_type": "cg", "pc_type": "hypre"})
    u = problem.solve()

    ffile.write_function(theta, i)

    compliance = assemble_scalar(form(action(L, u)))
    compliance_history.append(compliance)
    print("Iteration {}: compliance =".format(i), compliance)

    avg_density = update_theta(u)

    update_lagrange_multiplier(u, avg_density)

    exponent_counter = update_exponent(exponent_counter)

    # Update theta field and compliance
    thetaold.vector[:] = theta.vector[:]
    old_compliance = compliance