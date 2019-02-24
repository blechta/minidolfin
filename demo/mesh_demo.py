# minidolfin
# Copyright (C) 2019 Chris Richardson and Jan Blechta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import time
import pyamg
import ufl
import numpy
import matplotlib.pyplot as plt

from minidolfin.meshing import read_mesh, write_meshio
from minidolfin.dofmap import build_dofmap, interpolate_vertex_values
from minidolfin.assembling import symass
from minidolfin.bcs import build_dirichlet_dofs
from minidolfin.plot import plot

mesh = read_mesh('https://raw.githubusercontent.com/chrisrichardson/meshdata/master/data/rectangle_mesh.xdmf') # noqa

element = ufl.FiniteElement("P", ufl.triangle, 1)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)
x = ufl.SpatialCoordinate(ufl.triangle)
a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
L = 50.0*ufl.cos(6.28*x[0])*v*ufl.dx
dofmap = build_dofmap(element, mesh)


def u_bound(x):
    return x[0]


t = time.time()
bc_dofs, bc_vals = build_dirichlet_dofs(dofmap, u_bound, dtype=numpy.float64)
bc_map = {i: v for i, v in zip(bc_dofs, bc_vals)}
elapsed = time.time() - t
print('BC time = ', elapsed)

t = time.time()
A, b = symass(dofmap, a, L, bc_map, dtype=numpy.float64)

elapsed = time.time() - t
print('Ass time = ', elapsed)

# Smoothed aggregation -------------
print("****** Smoothed Aggregation solver ******")
ml = pyamg.smoothed_aggregation_solver(A, coarse_solver='lu')
print(ml)

t = time.time()
res = []
x = ml.solve(b, residuals=res, tol=1e-12, accel='cg')

print(res)
print("residual: ", numpy.linalg.norm(b - A*x))

elapsed = time.time() - t
timescale = numpy.linspace(0, elapsed, num=len(res))
plt.semilogy(timescale, res, marker='o', label='SA')
print('solve time = ', elapsed)

# Classical -------------------
print("****** Ruge Stuben solver ******")
ml = pyamg.ruge_stuben_solver(A, max_coarse=100)
print(ml)

n = 0


def residual(w):
    global n
    res = A*w - b
    write_meshio('file%d.xdmf' % n, mesh, res)
    n += 1
    print(numpy.linalg.norm(res))


t = time.time()
res = []
x = ml.solve(b, residuals=res, tol=1e-12, accel='cg', callback=residual)

print(res)
print("residual: ", numpy.linalg.norm(b - A * x))

elapsed = time.time() - t
timescale = numpy.linspace(0, elapsed, num=len(res))
plt.semilogy(timescale, res, marker='o', label='Ruge-Stuben')

print('solve time = ', elapsed)
# --------------------------------

print(x.min(), x.max())

plt.legend()
plt.show()

vertex_values = interpolate_vertex_values(dofmap, x)
plot(mesh, vertex_values)
plt.show()
