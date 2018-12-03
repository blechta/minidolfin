import ufl

from minidolfin.meshing import build_unit_square_mesh
from minidolfin.petsc import create_scalar
from minidolfin.assembling import assemble


def test_assemble_scalar():


    n = 6
    mesh = build_unit_square_mesh(n, n)

    F = 1*ufl.dx(domain=mesh)

    scalar = create_scalar()
    assemble(scalar, [], F)

    assert numpy.isclose(scalar, 1)
