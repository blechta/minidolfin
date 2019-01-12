import ufl
import dijitso
import ffc
import numpy
from matplotlib import pyplot, tri

import timeit
import math
import argparse

from minidolfin.meshing import build_unit_square_mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.dofmap import build_sparsity_pattern
from minidolfin.dofmap import pattern_to_csr
from minidolfin.dofmap import interpolate_vertex_values
from minidolfin.assembling import assemble
from minidolfin.bcs import build_dirichlet_dofs


# Parse command-line arguments
parser = argparse.ArgumentParser(description="minidolfin Helmholtz demo",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--mesh-size", type=int, dest="n", default=25,
                    help="mesh resolution")
parser.add_argument("-c", "--form-compiler", type=str, dest="form_compiler",
                    default="tsfc", choices=["tsfc", "ffc"],
                    help="form compiler")
parser.add_argument("-r", "--representation", type=str, dest="representation",
                    default="uflacs", choices=["uflacs", "tsfc"],
                    help="ffc representation")
parser.add_argument("-f", action="append", dest="form_compiler_parameters",
                    metavar="parameter=value", default=[],
                    help="additional form compiler paramter")
parser.add_argument("-d", "--debug", action='store_true', default=False,
                    help="enable debug output")
args = parser.parse_args()

# Make dijitso talk to us
if args.debug:
    dijitso.set_log_level("DEBUG")
    ffc.logger.setLevel("DEBUG")

# Build form compiler parameters
form_compiler_parameters = {}
form_compiler_parameters["compiler"] = args.form_compiler
form_compiler_parameters["representation"] = args.representation
for p in args.form_compiler_parameters:
    k, v = p.split("=")
    form_compiler_parameters[k] = v

# Plane wave
omega2 = 1.5**2 + 1**2
u_exact = lambda x: math.cos(-1.5*x[0] + x[1])

# UFL form
element = ufl.FiniteElement("P", ufl.triangle, 3)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - omega2*ufl.dot(u, v))*ufl.dx

# Build mesh
mesh = build_unit_square_mesh(args.n, args.n)
tdim = mesh.reference_cell.get_dimension()
print('Number cells: {}'.format(mesh.num_entities(tdim)))

# Build dofmap
dofmap = build_dofmap(element, mesh)
print('Number dofs: {}'.format(dofmap.dim))

# Run and time assembly
t = -timeit.default_timer()
A = assemble(dofmap, a, form_compiler_parameters)
t += timeit.default_timer()
print('Assembly time a: {}'.format(t))

# Prepare solution and rhs vectors and apply boundary conditions
x, b = numpy.zeros(A.shape[1]), numpy.zeros(A.shape[0])
bc_dofs, bc_vals = build_dirichlet_dofs(dofmap, u_exact)

# Set Dirichlet BCs
for i, v in zip(bc_dofs, bc_vals):
    A[i, :] *= 0.0 # Clear row
    A[i, i] = 1.0 # Set diagonal
    b[i] = v # Set RHS

# Solve linear system
import scipy.sparse.linalg
#A.setOption(PETSc.Mat.Option.SPD, True)  # FIXME: Is that true?
t = -timeit.default_timer()
x = scipy.sparse.linalg.spsolve(A, b)
t += timeit.default_timer()
print('Solve linear system time: {}'.format(t))

# Plot solution
vertex_values = interpolate_vertex_values(dofmap, x)
triang = tri.Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1],
                           mesh.get_connectivity(tdim, 0))
pyplot.tripcolor(triang, vertex_values)
pyplot.show()
