# minidolfin
# Copyright (C) 2019 Chris Richardson and Jan Blechta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pyamg
import ufl
import numpy
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import timeit

from minidolfin.meshing import build_unit_cube_mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.assembling import assemble
from minidolfin.bcs import bc_apply


# UFL form
element = ufl.VectorElement("P", ufl.tetrahedron, 1)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)

E = 1.0
nu = 0.25
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def epsilon(v):
    return 0.5*(ufl.grad(v) + ufl.grad(v).T)


def sigma(v):
    return 2.0*mu*epsilon(v) + lmbda*ufl.tr(epsilon(v)) \
        * ufl.Identity(v.geometric_dimension())


a = ufl.inner(sigma(u), epsilon(v))*ufl.dx

# Build mesh
n = 20
mesh = build_unit_cube_mesh(n, n, n)
tdim = mesh.reference_cell.get_dimension()
print('Number cells: {}'.format(mesh.num_entities(tdim)))

# Build dofmap
dofmap = build_dofmap(element, mesh)
print('Number dofs: {}'.format(dofmap.dim))

# Run and time assembly
t = -timeit.default_timer()
A = assemble(dofmap, a, dtype=numpy.float32)
t += timeit.default_timer()
print('Assembly time a: {}'.format(t))

# Null space
B = numpy.zeros((A.shape[0], 6), dtype=A.dtype)
# Translation
B[0::3, 0] = 1
B[1::3, 1] = 1
B[2::3, 2] = 1
# Rotation
B[0::3, 3] = 1
B[1::3, 3] = -1
B[1::3, 4] = 1
B[2::3, 4] = -1
B[2::3, 5] = 1
B[0::3, 5] = -1
B = scipy.linalg.orth(B)

b = numpy.zeros(A.shape[0], dtype=A.dtype)
x = numpy.zeros(A.shape[1], dtype=A.dtype)

# Set some BCs - fix bottom edge, and move top corner
bc_dofs = list(range((n + 1) * (n + 1) * 3)) \
          + [len(b) - 3, len(b) - 2, len(b) - 1]
bc_vals = numpy.zeros_like(bc_dofs, dtype=A.dtype)
bc_vals[-2] = 0.01

bc_apply(bc_dofs, bc_vals, A, b)

t = -timeit.default_timer()
ml = pyamg.smoothed_aggregation_solver(A, B)
print(ml)

res = []
x = ml.solve(b, residuals=res, tol=numpy.finfo(A.dtype).eps, accel='cg')
# x = scipy.sparse.linalg.spsolve(A, b)
t += timeit.default_timer()
print('Solve time: {}'.format(t))

print(len(res), res)
plt.semilogy(res, marker='o', label='SA')
plt.legend()
plt.show()

res = b - A*x
print("residual: ", numpy.linalg.norm(res), res.max(), res.min())

# Plotting...

# Deform mesh by displacement
# x = x.reshape(-1, 3)
# mesh.vertices += x*5

# Get magnitude of displacement
# xmag = numpy.sqrt(x[:, 0]**2 + x[:, 1]**2)

# plot(mesh, xmag)
# plt.show()
