# minidolfin
# Copyright (C) 2019 Chris Richardson and Jan Blechta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import matplotlib.pyplot as plt
import pyamg
import ufl
import FIAT
import meshio
import subprocess
import timeit
import numpy

from minidolfin.meshing import Mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.assembling import assemble, c_to_numpy
from minidolfin.bcs import bc_apply

# Call gmsh
subprocess.call(['gmsh', '-3', 'cylinder.geo'])

# Read gmsh format
mesh = meshio.read('cylinder.msh')
cell_data = mesh.cell_data['tetra']['gmsh:physical']
cells = mesh.cells['tetra']
points = mesh.points

# Convert to minidolfin Mesh
cells.sort(axis=1)
fiat_cell = FIAT.reference_element.ufc_cell("tetrahedron")
mesh = Mesh(fiat_cell, points, cells)

element = ufl.VectorElement("P", ufl.tetrahedron, 1)
DG0 = ufl.FiniteElement("DG", ufl.tetrahedron, 0)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)

E = ufl.Coefficient(DG0)
nu = 0.25
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def epsilon(v):
    return 0.5*(ufl.grad(v) + ufl.grad(v).T)


def sigma(v):
    return 2.0*mu*epsilon(v) + lmbda*ufl.tr(epsilon(v)) \
        * ufl.Identity(v.geometric_dimension())

a = ufl.inner(sigma(u), epsilon(v))*ufl.dx

# Build dofmap
dofmap = build_dofmap(element, mesh)
print('Number dofs: {}'.format(dofmap.dim))

scalar = 'double'

# Coefficients are for "E", the Youngs Modulus, defined cell-wise (DG0)
cell_data = [0.1 if idx==2 else 10.0 for idx in cell_data]
E = numpy.array(cell_data, dtype=c_to_numpy(scalar))

t = -timeit.default_timer()
A = assemble(dofmap, a, {'scalar_type': scalar}, coefficients=E)
t += timeit.default_timer()
print('Assembly time a: {}'.format(t))

print(A.max(), A.min())

b = numpy.zeros(A.shape[1], dtype=A.dtype)

dofs = []
vals = []
# Set BCs
for i, x in enumerate(mesh.vertices):
    if (x[0] == -10.0):
        idx = i * 3
        dofs.append(idx)
        vals.append(0.0)

    if (x[0] == 0.0):
        idx = i * 3
        dofs.append(idx)
        vals.append(0.1)

bc_apply(dofs, vals, A, b)

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
# B = scipy.linalg.orth(B)

t = -timeit.default_timer()
ml = pyamg.smoothed_aggregation_solver(A, B, max_coarse=10, strength=('symmetric', {'theta': 0.01}))
print(ml)

res = []
x = ml.solve(b, residuals=res, tol=numpy.finfo(A.dtype).eps*100, accel='gmres')
# x = scipy.sparse.linalg.spsolve(A, b)
t += timeit.default_timer()
print('Solve time: {}'.format(t))

print(res)
plt.semilogy(res, marker='o', label='SA')
plt.legend()
plt.show()
