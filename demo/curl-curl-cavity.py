import ufl

import timeit

from minidolfin.meshing import build_unit_cube_mesh
from minidolfin.assembling import assemble


# UFL form
e = ufl.FiniteElement("N1E", ufl.tetrahedron, 2)
u, v = ufl.TrialFunction(e), ufl.TestFunction(e)
omega2 = 1e3
a = (ufl.inner(ufl.curl(u), ufl.curl(v)) - omega2*ufl.dot(u, v))*ufl.dx
L = v[0]*ufl.dx

# Build mesh
mesh = build_unit_cube_mesh(32, 32, 32)
print('Number cells: {}'.format(mesh.cells.shape[0]))

# Run and time the assembly
t = -timeit.default_timer()
assemble(mesh, a)
t += timeit.default_timer()
print('Assembly time a: {}'.format(t))

t = -timeit.default_timer()
assemble(mesh, L)
t += timeit.default_timer()
print('Assembly time L: {}'.format(t))
