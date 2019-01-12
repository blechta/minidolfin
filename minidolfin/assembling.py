
import ffc
import ffc.codegeneration.jit
import cffi
import scipy.sparse
import numpy
import numba


def jit_compile_form(form, params):

    compiled_forms, module = ffc.codegeneration.jit.compile_forms(
        [form], parameters={'scalar_type': 'double'})

    for f, compiled_f in zip([form], compiled_forms):
        assert compiled_f.rank == len(f.arguments())

    return compiled_forms[0][0]


def assemble(dofmap, form, form_compiler_parameters=None):

    # JIT compile UFL form into ctypes function
    assembly_kernel = jit_compile_form(form, form_compiler_parameters) \
        .create_default_cell_integral().tabulate_tensor

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
    raise RuntimeError("Something went wrong (not linear or bilinear).")
