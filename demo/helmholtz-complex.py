# minidolfin
# Copyright (C) 2019 Chris Richardson and Jan Blechta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
import numpy
from matplotlib import pyplot, tri
import scipy.sparse.linalg

import timeit

from minidolfin.meshing import build_unit_square_mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.dofmap import interpolate_vertex_values
from minidolfin.assembling import assemble
from minidolfin.bcs import build_dirichlet_dofs
from minidolfin.bcs import bc_apply


# Plane wave
omega2 = (15**2 + 12**2) - 300j


def u_exact(x):
    return numpy.exp(-15j*x[0] + 12j*x[1])


# UFL form
element = ufl.FiniteElement("P", ufl.triangle, 3)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - omega2*ufl.inner(u, v))*ufl.dx

# Build mesh
mesh = build_unit_square_mesh(80, 80)
tdim = mesh.reference_cell.get_dimension()
print('Number cells: {}'.format(mesh.num_entities(tdim)))

# Build dofmap
dofmap = build_dofmap(element, mesh)
print('Number dofs: {}'.format(dofmap.dim))

# Run and time assembly
t = -timeit.default_timer()
A = assemble(dofmap, a, dtype=numpy.complex128)
t += timeit.default_timer()
print('Assembly time a: {}'.format(t))

# Prepare solution and rhs vectors and apply boundary conditions
x = numpy.zeros(A.shape[1], dtype=A.dtype)
b = numpy.zeros(A.shape[0], dtype=A.dtype)

# Set Dirichlet BCs

t = -timeit.default_timer()
bc_dofs, bc_vals = build_dirichlet_dofs(dofmap, u_exact, dtype=A.dtype)

print(bc_vals.dtype)
bc_apply(bc_dofs, bc_vals, A, b)

t += timeit.default_timer()
print('Apply BCs: {}'.format(t))

# Solve linear system
t = -timeit.default_timer()
x = scipy.sparse.linalg.spsolve(A, b)

r = (A*x - b)
print(r.max(), r.min())

t += timeit.default_timer()
print('Solve linear system time: {}'.format(t))

# Plot solution
vertex_values = interpolate_vertex_values(dofmap, x)
triang = tri.Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1],
                           mesh.get_connectivity(tdim, 0))
pyplot.axis('equal')
pyplot.tripcolor(triang, numpy.real(vertex_values))
pyplot.show()
pyplot.axis('equal')
pyplot.tripcolor(triang, numpy.imag(vertex_values))
pyplot.show()
