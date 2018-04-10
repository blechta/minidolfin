import tsfc
import numpy

import collections


# Data structure representing dofmap
DofMap = collections.namedtuple("DofMap", "cell_dofs dim mesh element")


def build_dofmap(element, mesh):
    fiat_element = tsfc.fiatinterface.create_element(element)

    assert mesh.reference_cell == fiat_element.get_reference_element()
    tdim = mesh.reference_cell.get_dimension()

    # Build cell dofs - mapping of cells to global dofs.
    # cell_dofs(i, j) is global dof number for cell i and local dof
    # index j.
    cell_dofs = numpy.ndarray((mesh.num_entities(tdim),
                               fiat_element.space_dimension()),
                              dtype=numpy.int32)
    offset = 0

    for dim, local_dofs in fiat_element.entity_dofs().items():
        dofs_per_entity = len(local_dofs[0])
        connectivity = mesh.get_connectivity(tdim, dim)

        # FIXME: Dofs on single entity are not consecutive?
        for k in range(dofs_per_entity):
            entity_dofs = [dofs[k] for entity, dofs in sorted(local_dofs.items())]
            cell_dofs[:, entity_dofs] = dofs_per_entity*connectivity + (offset+k)

        offset += dofs_per_entity*mesh.num_entities(dim)

    # Build dofmap structure and store what it depends on
    return DofMap(cell_dofs=cell_dofs, dim=offset, mesh=mesh, element=element)


def build_sparsity_pattern(dofmap):

    # Fetch data
    tdim = dofmap.mesh.reference_cell.get_dimension()
    num_cells = dofmap.mesh.num_entities(tdim)
    cell_dofs = dofmap.cell_dofs

    # Resulting data structure
    pattern = [set() for i in range(dofmap.dim)]

    # Build cell integral pattern
    for c in range(num_cells):
        dofs = cell_dofs[c]
        for dof0 in dofs:
            for dof1 in dofs:
                pattern[dof0].add(dof1)

    return pattern


def pattern_to_csr(pattern, dtype=numpy.int32):

    # Fetch data
    nrows = len(pattern)
    nnz = sum(len(row) for row in pattern)

    # Allocate result
    i = numpy.ndarray(nrows+1, dtype=dtype)
    j = numpy.ndarray(nnz, dtype=dtype)

    # Compute CSR
    offset = 0
    for r, row in enumerate(pattern):
        ncols = len(row)
        i[r] = offset
        j[offset:offset+ncols] = sorted(row)
        offset += ncols
    i[nrows] = offset

    return i, j


def interpolate_vertex_values(dofmap, x):

    # Fetch data from mesh
    tdim = dofmap.mesh.reference_cell.get_dimension()
    num_cells = dofmap.mesh.num_entities(tdim)
    num_vertices = dofmap.mesh.num_entities(0)

    # Fetch data from dofmap
    cell_dofs = dofmap.cell_dofs
    cell_vertex_conn = dofmap.mesh.get_connectivity(tdim, 0)

    # Build local vertex-dof connectivity
    fiat_element = tsfc.fiatinterface.create_element(dofmap.element)
    vertex_dofs = fiat_element.entity_dofs()[0]
    dofs_per_vertex = len(vertex_dofs[0])
    assert dofs_per_vertex == 1, "just have to agree on ordering"
    k = 0
    vertex_dofs = [dofs[k] for entity, dofs in sorted(vertex_dofs.items())]

    # Build vertex values
    vertex_values = numpy.ndarray((num_vertices,), dtype=numpy.double)
    vertex_values[cell_vertex_conn[:]] = x[cell_dofs[:, vertex_dofs]]

    return vertex_values
