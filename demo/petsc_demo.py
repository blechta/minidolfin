# minidolfin
# Copyright (C) 2019 Chris Richardson and Jan Blechta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

try:
    from petsc4py import PETSc
except ImportError:
    raise("petsc4py needed for this demo")

import time
import ufl
import numpy
from minidolfin.meshing import read_mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.assembling import symass
from minidolfin.bcs import build_dirichlet_dofs

mesh = read_mesh('https://raw.githubusercontent.com/chrisrichardson/meshdata/master/data/rectangle_mesh.xdmf') # noqa

element = ufl.FiniteElement("P", ufl.triangle, 1)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)
f = ufl.Coefficient(element)
a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
L = ufl.cos(1.0)*v*ufl.dx
dofmap = build_dofmap(element, mesh)


def u_bound(x):
    return x[0]


t = time.time()
bc_dofs, bc_vals = build_dirichlet_dofs(dofmap, u_bound)
bc_map = {i: v for i, v in zip(bc_dofs, bc_vals)}
elapsed = time.time() - t
print('BC time = ', elapsed)

t = time.time()
A, b = symass(dofmap, a, L, bc_map, dtype=numpy.float64)


petsc_mat = PETSc.Mat().createAIJ(size=A.shape,
                                  csr=(A.indptr, A.indices, A.data))

print(petsc_mat)
