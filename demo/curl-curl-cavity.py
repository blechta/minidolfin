import ufl

import timeit

from minidolfin.meshing import build_unit_cube_mesh
from minidolfin.assembling import build_dofmap, assemble


# UFL form
element = ufl.FiniteElement("N1E", ufl.tetrahedron, 2)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)
omega2 = 1e3
a = (ufl.inner(ufl.curl(u), ufl.curl(v)) - omega2*ufl.dot(u, v))*ufl.dx
L = v[0]*ufl.dx

# Build mesh
mesh = build_unit_cube_mesh(32, 32, 32)
tdim = mesh.reference_cell.get_dimension()
print('Number cells: {}'.format(mesh.num_entities(tdim)))

# Build dofmap
dofmap = build_dofmap(element, mesh)
print('Number dofs: {}'.format(dofmap.dim))

# Run and time the assembly
t = -timeit.default_timer()
assemble(dofmap, a)
t += timeit.default_timer()
print('Assembly time a: {}'.format(t))

t = -timeit.default_timer()
assemble(dofmap, L)
t += timeit.default_timer()
print('Assembly time L: {}'.format(t))
