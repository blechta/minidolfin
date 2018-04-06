import ufl

import timeit
import math

from minidolfin.meshing import build_unit_cube_mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.dofmap import build_sparsity_pattern
from minidolfin.dofmap import pattern_to_csr
from minidolfin.petsc import create_matrix_from_csr
from minidolfin.assembling import assemble
from minidolfin.bcs import build_dirichlet_dofs


# Plane wave
omega2 = 1.5**2 + 1**2
u_exact = lambda x: math.cos(-1.5*x[0] + x[1])

# UFL form
element = ufl.FiniteElement("P", ufl.tetrahedron, 2)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)
omega2 = 1e3
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - omega2*ufl.dot(u, v))*ufl.dx

# Build mesh
mesh = build_unit_cube_mesh(1, 1, 1)
tdim = mesh.reference_cell.get_dimension()
print('Number cells: {}'.format(mesh.num_entities(tdim)))

# Build dofmap
dofmap = build_dofmap(element, mesh)
print('Number dofs: {}'.format(dofmap.dim))

# Build sparsity pattern
pattern = build_sparsity_pattern(dofmap)
i, j = pattern_to_csr(pattern)
A = create_matrix_from_csr((i, j))

# Run and time the assembly
t = -timeit.default_timer()
assemble(A, dofmap, a)
t += timeit.default_timer()
print('Assembly time a: {}'.format(t))

#A.view()

bc_dofs, bc_vals = build_dirichlet_dofs(dofmap, u_exact)
