
import time
import pyamg
import ufl
import numpy
import scipy
from minidolfin.meshing import read_meshio, write_meshio
from minidolfin.dofmap import build_dofmap, interpolate_vertex_values
from minidolfin.assembling import assemble
from minidolfin.bcs import build_dirichlet_dofs, bc_apply

mesh = read_meshio('tet_cube.xdmf')

element = ufl.FiniteElement("P", ufl.tetrahedron, 1)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)
x = ufl.SpatialCoordinate(ufl.tetrahedron)
a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
L = ufl.cos(x[1])*v*ufl.dx
dofmap = build_dofmap(element, mesh)


def u_bound(x):
    return x[0]


t = time.time()
bc_dofs, bc_vals = build_dirichlet_dofs(dofmap, u_bound)
bc_map = {i: v for i, v in zip(bc_dofs, bc_vals)}
elapsed = time.time() - t
print('BC time = ', elapsed)

t = time.time()
# A, b = symass(dofmap, a, L, bc_map, dtype=numpy.float32)
A = assemble(dofmap, a, dtype=numpy.float32)
b = assemble(dofmap, L, dtype=A.dtype)
elapsed = time.time() - t
print('Ass time = ', elapsed)

bc_apply(bc_dofs, bc_vals, A, b)

ml = pyamg.ruge_stuben_solver(A)
# ml = pyamg.smoothed_aggregation_solver(A)
print(ml)

t = time.time()
x = scipy.sparse.linalg.spsolve(A, b)
# x = ml.solve(b, tol=1e-16)
print("residual: ", numpy.linalg.norm(b-A*x))

elapsed = time.time() - t
print('solve time = ', elapsed)

print(x.min(), x.max())

vertex_values = interpolate_vertex_values(dofmap, x)

write_meshio('result.xdmf', mesh, vertex_values)
