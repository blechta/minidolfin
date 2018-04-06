import tsfc
import numpy


def build_dirichlet_dofs(dofmap, value):
    tdim = dofmap.mesh.reference_cell.get_dimension()
    cell_vertex_conn = dofmap.mesh.get_connectivity(tdim, 0)
    cell_facet_conn = dofmap.mesh.get_connectivity(tdim, tdim-1)

    vertices = dofmap.mesh.vertices

    cell_dofs = dofmap.cell_dofs

    num_facets_per_cell = cell_facet_conn.shape[1]

    bc_map = {}

    fiat_element = tsfc.fiatinterface.create_element(dofmap.element)
    mapping, = set(fiat_element.mapping())
    if mapping == "affine":
        # FIXME: This probably does not match FIAT cell!
        f_hat = lambda B, b: lambda xhat: value(B.dot(xhat) + b)
    elif mapping == "covariant piola":
        # FIXME: This probably does not match FIAT cell!
        f_hat = lambda B, b: lambda xhat: value(B.dot(xhat) + b).dot(B)
    else:
        raise NotImplementedError

    facet_dofs = fiat_element.entity_closure_dofs()[tdim-1]

    def interpolation_operator(f):
        dim = fiat_element.space_dimension()
        return numpy.fromiter((phi(f) for phi in fiat_element.get_dual_set().get_nodes()),
                              numpy.double, count=dim)

    for c in range(cell_facet_conn.shape[0]):
        is_boundary = [cell_facet_conn[c][f] for f in range(num_facets_per_cell)]

        if any(is_boundary):

            b = vertices[cell_vertex_conn[c, tdim]]
            B = vertices[cell_vertex_conn[c, 0:tdim]] - b.reshape(b.shape+(1,))

            dof_vals = interpolation_operator(f_hat(B, b))
            dof_indices = cell_dofs[c]

            local_boundary_facets, = numpy.where(is_boundary)
            local_boundary_dofs = numpy.fromiter(set(d for f in range(num_facets_per_cell) for d in facet_dofs[f]),
                                                 dof_indices.dtype)
            for d in local_boundary_dofs:
                bc_map[dof_indices[d]] = dof_vals[d]

    dofs = numpy.fromiter(bc_map.keys(), cell_dofs.dtype, count=len(bc_map))
    vals = numpy.fromiter(bc_map.values(), numpy.double, count=len(bc_map))

    return dofs, vals
