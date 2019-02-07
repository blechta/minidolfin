# minidolfin
# Copyright (C) 2019 Chris Richardson and Jan Blechta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import ufl
import numpy
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import timeit

from minidolfin.meshing import build_unit_square_mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.assembling import assemble
from minidolfin.bcs import bc_apply
from minidolfin.plot import plot


# UFL form
element = ufl.VectorElement("P", ufl.triangle, 1)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)

E = 1.0e9
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
n = 50
mesh = build_unit_square_mesh(n, n)
tdim = mesh.reference_cell.get_dimension()
print('Number cells: {}'.format(mesh.num_entities(tdim)))

# Build dofmap
dofmap = build_dofmap(element, mesh)
print('Number dofs: {}'.format(dofmap.dim))

scalar = 'float'

# Run and time assembly
t = -timeit.default_timer()
A = assemble(dofmap, a, {'scalar_type': scalar})
t += timeit.default_timer()
print('Assembly time a: {}'.format(t))

b = numpy.zeros(A.shape[0])
x = numpy.zeros(A.shape[1])

# Set some BCs - fix bottom edge, and move top corner
bc_dofs = list(range(n * 2 + 2)) + [len(b) - 2, len(b) - 1]
bc_vals = numpy.zeros_like(bc_dofs, dtype=float)
bc_vals[-1] = 0.01

bc_apply(bc_dofs, bc_vals, A, b)

t = -timeit.default_timer()
x = scipy.sparse.linalg.spsolve(A, b)
t += timeit.default_timer()
print('Solve time: {}'.format(t))

# Plotting...

# Deform mesh by displacement
x = x.reshape(-1, 2)
mesh.vertices += x*5

# Get magnitude of displacement
xmag = numpy.sqrt(x[:, 0]**2 + x[:, 1]**2)

plot(mesh, xmag)
plt.show()

print(x)
