
import pyamg
import ufl
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from minidolfin.meshing import get_mesh_from_url
from minidolfin.dofmap import build_dofmap, interpolate_vertex_values
from minidolfin.assembling import assemble, symass
from minidolfin.bcs import build_dirichlet_dofs, bc_apply
from minidolfin.plot import plot

mesh = get_mesh_from_url('https://raw.githubusercontent.com/chrisrichardson/meshdata/master/data/rectangle_mesh.xdmf')

element = ufl.FiniteElement("P", ufl.triangle, 1)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)
f = ufl.Coefficient(element)
a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
L = ufl.cos(f)*v*ufl.dx
dofmap = build_dofmap(element, mesh)

# A = assemble(dofmap, a, None)
# b = assemble(dofmap, L, None)

def u_bound(x):
    return x[0]

bc_dofs, bc_vals = build_dirichlet_dofs(dofmap, u_bound)

A, b = symass(dofmap, a, L, bc_map, None)

bc_apply(bc_dofs, bc_vals, A, b)

# ml = pyamg.ruge_stuben_solver(A)
ml = pyamg.smoothed_aggregation_solver(A)
print(ml)

# x = scipy.sparse.linalg.spsolve(A, b)
x = ml.solve(b, tol=1e-8)
print("residual: ", numpy.linalg.norm(b-A*x))

print(x.min(), x.max())

vertex_values = interpolate_vertex_values(dofmap, x)
plot(mesh, vertex_values)
plt.savefig('a.pdf')
