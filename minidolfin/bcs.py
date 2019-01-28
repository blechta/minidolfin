import ffc
import numpy
import scipy.sparse


def build_dirichlet_dofs(dofmap, value):

    # Fetch mesh data
    tdim = dofmap.mesh.reference_cell.get_dimension()
    cell_vertex_conn = dofmap.mesh.get_connectivity(tdim, 0)
    cell_facet_conn = dofmap.mesh.get_connectivity(tdim, tdim-1)
    vertices = dofmap.mesh.vertices
    boundary_facets = dofmap.mesh.boundary_facets
    num_facets_per_cell = cell_facet_conn.shape[1]

    # Fetch dofmap data
    cell_dofs = dofmap.cell_dofs

    # Fetch data from reference element
    fiat_element = ffc.fiatinterface.create_element(dofmap.element)
    facet_dofs = fiat_element.entity_closure_dofs()[tdim-1]
    mapping, = set(fiat_element.mapping())

    # Define appropriate pull-back
    if mapping == "affine":
        def f_hat(B, b):
            def _f_hat(xhat):
                return value(B.dot(xhat) + b)
            return _f_hat
    elif mapping == "covariant piola":
        def f_hat(B, b):
            def _f_hat(xhat):
                return value(B.dot(xhat) + b).dot(B)
            return _f_hat
    elif mapping == "contravariant piola":
        def f_hat(B, b):
            Binv = numpy.linalg.inv(B)

            def _f_hat(xhat):
                return Binv.dot(value(B.dot(xhat) + b))
            return _f_hat
    else:
        raise NotImplementedError

    # Turn dual basis nodes into interpolation operator
    dim = fiat_element.space_dimension()

    def interpolation_operator(f):
        return numpy.fromiter(
            (phi(f) for phi in fiat_element.get_dual_set().get_nodes()),
            numpy.double, count=dim)

    # Temporary
    bc_map = {}

    # Iterate over cells
    for c in range(cell_facet_conn.shape[0]):

        # Check which facets are on boundary
        is_boundary = [cell_facet_conn[c][f] in boundary_facets
                       for f in range(num_facets_per_cell)]

        if any(is_boundary):

            # NB: This is affine transformation resulting from UFC
            #     simplex definition in FIAT
            b = vertices[cell_vertex_conn[c, 0]]
            B = (vertices[cell_vertex_conn[c, 1:]] - b).T

            # Interpolate Dirichlet datum
            dof_vals = interpolation_operator(f_hat(B, b))
            dof_indices = cell_dofs[c]

            # Figure out which facets and dofs are on boundary
            local_boundary_facets, = numpy.where(is_boundary)
            local_boundary_dofs = numpy.fromiter(
                set(d for f in local_boundary_facets for d in facet_dofs[f]),
                dof_indices.dtype)

            # Store dof-value pair into temporary
            for d in local_boundary_dofs:
                bc_map[dof_indices[d]] = dof_vals[d]

    # Convert to arrays
    # FIXME: Is deterministic?
    dofs = numpy.fromiter(bc_map.keys(), numpy.int32, count=len(bc_map))
    vals = numpy.fromiter(bc_map.values(), numpy.double, count=len(bc_map))

    return dofs, vals


def bc_apply(dofs, vals, A, b):
    """ Apply BCs in dofs, vals to CSR matrix A and RHS b. """

    assert isinstance(A, scipy.sparse.csr_matrix)

    # Clear rows and set diagonal
    for i in dofs:
        A.data[A.indptr[i]:A.indptr[i+1]] = 0.0
        A[i, i] = 1.0

    # Set RHS
    for i, v in zip(dofs, vals):
        b[i] = v
