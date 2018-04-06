import ufl

import timeit

from minidolfin.meshing import build_unit_cube_mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.dofmap import build_sparsity_pattern
from minidolfin.dofmap import pattern_to_csr
from minidolfin.petsc import create_matrix_from_csr
from minidolfin.assembling import assemble

import dijitso
dijitso.set_log_level("debug")


# UFL form
element = ufl.FiniteElement("N1E", ufl.tetrahedron, 2)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)
omega2 = 1e3
a = (ufl.inner(ufl.curl(u), ufl.curl(v)) - omega2*ufl.dot(u, v))*ufl.dx

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
