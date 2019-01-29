
import ffc
import ffc.codegeneration.jit
import cffi
import scipy.sparse
import numpy
import numba


def jit_compile_forms(forms, params):

    compiled_forms, module = ffc.codegeneration.jit.compile_forms(
        forms, parameters={'scalar_type': 'double'})

    for f, compiled_f in zip(forms, compiled_forms):
        assert compiled_f.rank == len(f.arguments())

    return compiled_forms


def assemble(dofmap, form, form_compiler_parameters=None):

    # JIT compile UFL form into ctypes function
    module = jit_compile_forms([form], form_compiler_parameters)[0][0]
    assembly_kernel = module.create_default_cell_integral().tabulate_tensor

    # Fetch data
    tdim = dofmap.mesh.reference_cell.get_dimension()
    cells = dofmap.mesh.get_connectivity(tdim, 0)
    vertices = dofmap.mesh.vertices
    cell_dofs = dofmap.cell_dofs

    # Prepare cell tensor temporary
    elements = tuple(arg.ufl_element() for arg in form.arguments())
    fiat_elements = map(ffc.fiatinterface.create_element, elements)
    element_dims = tuple(fe.space_dimension() for fe in fiat_elements)

    ffi = cffi.FFI()

    @numba.jit(nopython=True)
    def _assemble_bilinear(assembly_kernel, cells, vertices, cell_dofs):
        nrows = ncols = cell_dofs.shape[1]
        ncells = cells.shape[0]

        # Loop over cells
        ci = numpy.zeros(nrows*ncols*ncells, dtype=numpy.int32)
        cj = numpy.zeros(nrows*ncols*ncells, dtype=numpy.int32)
        val = numpy.zeros(nrows*ncols*ncells, dtype=numpy.float64)
        n = 0
        coords = numpy.empty((cells.shape[1], vertices.shape[1]))
        for c in range(ncells):

            # Assemble cell tensor
            A = numpy.zeros(element_dims, dtype=numpy.float64)
            w = numpy.array([0], dtype=numpy.float64)
            for i, q in enumerate(cells[c]):
                coords[i, :] = vertices[q]

            assembly_kernel(ffi.from_buffer(A),
                            ffi.from_buffer(w),
                            ffi.from_buffer(coords), 0)

            # Add to global tensor
            rows = cols = cell_dofs[c]
            for i, ig in enumerate(rows):
                for j, jg in enumerate(cols):
                    ci[n] = ig
                    cj[n] = jg
                    val[n] = A[i, j]
                    n += 1

        return ci, cj, val

    @numba.jit(nopython=True)
    def _assemble_linear(assembly_kernel, cells, vertices, cell_dofs, dim):
        ncells = cells.shape[0]

        vec = numpy.zeros(dim, dtype=numpy.float64)
        coords = numpy.empty((cells.shape[1], vertices.shape[1]))

        # Loop over cells
        for c in range(ncells):

            # Assemble cell tensor
            b = numpy.zeros(element_dims[0], dtype=numpy.float64)
            w = numpy.array([0], dtype=numpy.float64)
            for i, q in enumerate(cells[c]):
                coords[i, :] = vertices[q]
            assembly_kernel(ffi.from_buffer(b),
                            ffi.from_buffer(w),
                            ffi.from_buffer(coords), 0)

            # Add to global tensor
            rows = cell_dofs[c]
            for i, ig in enumerate(rows):
                vec[ig] += b[i]

        return vec

    # Call assembly loop
    dim = len(form.arguments())
    if dim == 2:
        ci, cj, val = _assemble_bilinear(assembly_kernel,
                                         cells, vertices, cell_dofs)
        mat = scipy.sparse.coo_matrix((val, (ci, cj)))
        return mat.tocsr()
    elif dim == 1:
        vec = _assemble_linear(assembly_kernel,
                               cells, vertices, cell_dofs, dofmap.dim)
        return vec
    raise RuntimeError("Form is neither linear nor bilinear.")


def symass(dofmap, LHSform, RHSform, bc_map, form_compiler_parameters):
    """ Assemble LHS and RHS together """
    # JIT compile UFL form into ctypes functions
    module = jit_compile_forms([LHSform, RHSform], form_compiler_parameters)
    LHS_kernel = module[0][0].create_default_cell_integral().tabulate_tensor
    RHS_kernel = module[1][0].create_default_cell_integral().tabulate_tensor

    # Fetch data
    tdim = dofmap.mesh.reference_cell.get_dimension()
    cells = dofmap.mesh.get_connectivity(tdim, 0)
    vertices = dofmap.mesh.vertices
    cell_dofs = dofmap.cell_dofs

    # Prepare cell tensor temporary
    elements = tuple(arg.ufl_element() for arg in LHSform.arguments())
    fiat_elements = map(ffc.fiatinterface.create_element, elements)
    element_dims = tuple(fe.space_dimension() for fe in fiat_elements)

    ffi = cffi.FFI()

    @numba.jit(nopython=True)
    def _assemble(LHS_kernel, RHS_kernel, cells, vertices, cell_dofs, dim):
        nrows = ncols = cell_dofs.shape[1]
        ncells = cells.shape[0]

        # Storage for Vector
        vec = numpy.zeros(dim, dtype=numpy.float64)
        # Storage for COO Matrix
        ci = numpy.zeros(nrows*ncols*ncells, dtype=numpy.int32)
        cj = numpy.zeros(nrows*ncols*ncells, dtype=numpy.int32)
        val = numpy.zeros(nrows*ncols*ncells, dtype=numpy.float64)

        n = 0
        # Temporary for cell geometry
        coords = numpy.empty((cells.shape[1], vertices.shape[1]))
        for c in range(ncells):

            # Assemble cell vector and matrix
            b = numpy.zeros(element_dims[0], dtype=numpy.float64)
            A = numpy.zeros(element_dims, dtype=numpy.float64)
            wA = numpy.array([0], dtype=numpy.float64)
            wb = numpy.array([0], dtype=numpy.float64)
            for i, q in enumerate(cells[c]):
                coords[i, :] = vertices[q]

            LHS_kernel(ffi.from_buffer(A),
                       ffi.from_buffer(wA),
                       ffi.from_buffer(coords), 0)

            RHS_kernel(ffi.from_buffer(b),
                       ffi.from_buffer(wb),
                       ffi.from_buffer(coords), 0)

            # Add to global vector and matrix
            rows = cols = cell_dofs[c]
            for i, ig in enumerate(rows):
                vec[ig] += b[i]
                for j, jg in enumerate(cols):
                    ci[n] = ig
                    cj[n] = jg
                    val[n] = A[i, j]
                    n += 1

        return ci, cj, val, vec

    # Call assembly loop
    ci, cj, val, vec = _assemble(LHS_kernel, RHS_kernel,
                                 cells, vertices, cell_dofs, dofmap.dim)
    mat = scipy.sparse.coo_matrix((val, (ci, cj)))
    return mat.tocsr(), vec
