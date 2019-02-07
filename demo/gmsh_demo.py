
import ufl
import FIAT
import meshio
import subprocess
import timeit

from minidolfin.meshing import Mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.assembling import assemble

# Call gmsh
subprocess.call(['gmsh', '-3', 'cylinder.geo'])

# Read gmsh format
mesh = meshio.read('cylinder.msh')
print(mesh.cell_data['tetra']['gmsh:physical'])
cells = mesh.cells['tetra']
points = mesh.points

# Convert to minidolfin Mesh
cells.sort(axis=1)
fiat_cell = FIAT.reference_element.ufc_cell("tetrahedron")
mesh = Mesh(fiat_cell, points, cells)

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

# Build dofmap
dofmap = build_dofmap(element, mesh)
print('Number dofs: {}'.format(dofmap.dim))

scalar = 'float'

t = -timeit.default_timer()
A = assemble(dofmap, a, {'scalar_type': scalar})
t += timeit.default_timer()
print('Assembly time a: {}'.format(t))
