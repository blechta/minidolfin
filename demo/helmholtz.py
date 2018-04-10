import ufl
from petsc4py import PETSc
from matplotlib import pyplot, tri

import timeit
import math

from minidolfin.meshing import build_unit_square_mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.dofmap import build_sparsity_pattern
from minidolfin.dofmap import pattern_to_csr
from minidolfin.dofmap import interpolate_vertex_values
from minidolfin.petsc import create_matrix_from_csr
from minidolfin.assembling import assemble
from minidolfin.bcs import build_dirichlet_dofs
from minidolfin.petsc import set_solver_package


# Plane wave
omega2 = 1.5**2 + 1**2
u_exact = lambda x: math.cos(-1.5*x[0] + x[1])

# UFL form
element = ufl.FiniteElement("P", ufl.triangle, 3)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - omega2*ufl.dot(u, v))*ufl.dx

# Build mesh
mesh = build_unit_square_mesh(128, 128)
tdim = mesh.reference_cell.get_dimension()
print('Number cells: {}'.format(mesh.num_entities(tdim)))

# Build dofmap
dofmap = build_dofmap(element, mesh)
print('Number dofs: {}'.format(dofmap.dim))

# Build sparsity pattern and create matrix
pattern = build_sparsity_pattern(dofmap)
i, j = pattern_to_csr(pattern)
A = create_matrix_from_csr((i, j))

# Run and time the assembly
t = -timeit.default_timer()
assemble(A, dofmap, a)
t += timeit.default_timer()
print('Assembly time a: {}'.format(t))

# Prepare solution and rhs vectors and apply boundary conditions
x, b = A.createVecs()
bc_dofs, bc_vals = build_dirichlet_dofs(dofmap, u_exact)
x.setValues(bc_dofs, bc_vals)
A.zeroRowsColumns(bc_dofs, diag=1, x=x, b=b)

# Solver linear system
ksp = PETSc.KSP().create(A.comm)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.pc.setType(PETSc.PC.Type.CHOLESKY)
set_solver_package(ksp.pc, "mumps")
#A.setOption(PETSc.Mat.Option.SPD, True)  # FIXME: Is that true?
ksp.setOperators(A)
ksp.setUp()
t = -timeit.default_timer()
ksp.solve(b, x)
t += timeit.default_timer()
print('Solve linear system time a: {}'.format(t))

# Plot solution
vertex_values = interpolate_vertex_values(dofmap, x)
triang = tri.Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1],
                           mesh.get_connectivity(tdim, 0))
pyplot.tripcolor(triang, vertex_values)
pyplot.show()
