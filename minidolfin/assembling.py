
import ffc
from ffc.codegeneration import jit as ffc_jit
import cffi
import scipy.sparse
import numpy
import numba
import numba.cffi_support

# Register C complex types
ffi = cffi.FFI()
numba.cffi_support.register_type(ffi.typeof('double _Complex'),
                                 numba.types.complex128)
numba.cffi_support.register_type(ffi.typeof('float _Complex'),
                                 numba.types.complex64)


def c_to_numpy(ctype):
    c2numpy = {'double': numpy.float64,
               'float': numpy.float32,
               'double complex': numpy.complex128,
               'float complex': numpy.complex64,
               'long double': numpy.longdouble}
    return c2numpy.get(ctype)


def numpy_to_c(dtype):
    numpy2c = {numpy.float64: 'double',
               numpy.float32: 'float',
               numpy.complex128: 'double complex',
               numpy.complex64: 'float complex',
               numpy.longdouble: 'long double'}
    return numpy2c[dtype]


def jit_compile_forms(forms, params):

    compiled_forms, module = ffc_jit.compile_forms(
        forms, parameters=params)

    for f, compiled_f in zip(forms, compiled_forms):
        assert compiled_f.rank == len(f.arguments())

    return compiled_forms


def assemble(dofmap, form, dtype=numpy.double,
             form_compiler_parameters={}):

    if isinstance(dtype, numpy.dtype):
        dtype = dtype.type

    # Overwrite with given dtype
    form_compiler_parameters['scalar_type'] = numpy_to_c(dtype)

    # JIT compile UFL form into ctypes function
    module = jit_compile_forms([form], form_compiler_parameters)[0][0]
    dim = len(form.arguments())
    nc = module.num_coefficients
    for i in range(nc):
        print(i, module.original_coefficient_position(i))
    assembly_kernel = module.create_default_cell_integral()
    assembly_kernel = assembly_kernel.tabulate_tensor

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

    @numba.jit(nopython=True, cache=False)
    def _assemble_bilinear(coefficients, val):

        ncells = cells.shape[0]
        local_size = len(val) // ncells
        coeff_size = len(coefficients) // ncells

        # Preallocate output buffer, and temporary for coords

        coords = numpy.empty((cells.shape[1], vertices.shape[1]),
                             dtype=numpy.float64)
        caddr = ffi.from_buffer(coords)
        for i in range(ncells):

            # Get cell coordinates
            for j, q in enumerate(cells[i]):
                coords[j, :] = vertices[q]

            # Assemble cell tensor into buffer
            assembly_kernel(ffi.from_buffer(val[i * local_size:]),
                            ffi.from_buffer(coefficients
                                            [i * coeff_size:]),
                            caddr, 0)

    @numba.jit(nopython=True, cache=False)
    def _assemble_linear(coefficients, vec):

        ncells = cells.shape[0]
        num_coeffs = len(coefficients) // ncells

        # Temporaries
        b = numpy.empty(element_dims[0], dtype=dtype)
        baddr = ffi.from_buffer(b)
        coords = numpy.empty((cells.shape[1], vertices.shape[1]),
                             dtype=numpy.float64)
        caddr = ffi.from_buffer(coords)

        # Loop over cells
        for i in range(ncells):

            b.fill(0)
            # Get cell coordinates
            for j, q in enumerate(cells[i]):
                coords[j, :] = vertices[q]

            # Assemble into temporary
            assembly_kernel(baddr,
                            ffi.from_buffer(coefficients
                                            [i * num_coeffs:]),
                            caddr, 0)

            # Add to global tensor
            for i, ig in enumerate(cell_dofs[i]):
                vec[ig] += b[i]

    if 'coefficients' in form._cache.keys():
        coefficients = form._cache['coefficients']
    else:
        coefficients = numpy.empty(0, dtype=dtype)

    if len(coefficients.shape) == 2:
        coefficients = coefficients.resize(coefficients.shape[0]
                                           * coefficients.shape[1])

    if dim == 2:
        # Form global insertion indices for all cells
        cij = (numpy.repeat(cell_dofs, cell_dofs.shape[1]),
               numpy.tile(cell_dofs, cell_dofs.shape[1]).flatten())

        # Fill local tensors
        nrows = ncols = cell_dofs.shape[1]
        val = numpy.zeros(nrows * ncols * cells.shape[0], dtype=dtype)
        _assemble_bilinear(coefficients, val)

        mat = scipy.sparse.coo_matrix((val, cij))
        mat.eliminate_zeros()
        return mat.tocsr()
    elif dim == 1:
        vec = numpy.zeros(dofmap.dim, dtype=dtype)
        _assemble_linear(coefficients, vec)
        return vec

    raise RuntimeError("Form is neither linear nor bilinear.")


def symass(dofmap, LHSform, RHSform, bc_map, dtype=numpy.float64,
           form_compiler_parameters={}):
    """ Assemble LHS and RHS together """

    form_compiler_parameters['scalar_type'] = numpy_to_c(dtype)

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
    def _assemble(kernels, cells, vertices, cell_dofs, bcs, bcvals):
        dim = bcs.shape[0]
        LHS_kernel, RHS_kernel = kernels
        nrows = ncols = cell_dofs.shape[1]
        nval = nrows*ncols
        ncells = cells.shape[0]

        # Storage for Vector.
        vec = numpy.zeros(dim, dtype=dtype)

        # Storage for COO Matrix
        val = numpy.zeros(nval*ncells, dtype=dtype)

        n = 0
        # Temporary for cell geometry
        coords = numpy.empty((cells.shape[1], vertices.shape[1]),
                             dtype=numpy.float64)
        for c in range(ncells):

            # Assemble cell vector and matrix
            b = numpy.zeros(element_dims[0], dtype=dtype)
            A = numpy.zeros(element_dims, dtype=dtype)
            wA = numpy.array([0], dtype=dtype)
            wb = numpy.array([0], dtype=dtype)
            for i, q in enumerate(cells[c]):
                coords[i, :] = vertices[q]

            LHS_kernel(ffi.from_buffer(A),
                       ffi.from_buffer(wA),
                       ffi.from_buffer(coords), 0)

            RHS_kernel(ffi.from_buffer(b),
                       ffi.from_buffer(wb),
                       ffi.from_buffer(coords), 0)

            rows = cell_dofs[c]

            # Set Dirichlet BCs symmetrically
            for i, iglobal in enumerate(rows):
                if bcs[iglobal]:
                    A[i, :] = 0.0
                    b[:] -= A[:, i]*bcvals[iglobal]
                    A[:, i] = 0.0
                    A[i, i] = 1.0
                    b[i] = bcvals[iglobal]

            # Add to global vector and matrix
            for i, iglobal in enumerate(rows):
                vec[iglobal] += b[i]
                for j in range(ncols):
                    val[n] = A[i, j]
                    n += 1

        return val, vec

    # Mark any dofs which have BCs. FIXME: better way?
    bcs = numpy.zeros(dofmap.dim, dtype=bool)
    bcvals = numpy.zeros(dofmap.dim, dtype=dtype)
    for dof, val in bc_map.items():
        bcs[dof] = True
        bcvals[dof] = val

    # Form global insertion indices for all cells
    cij = (numpy.repeat(cell_dofs, cell_dofs.shape[1]),
           numpy.tile(cell_dofs, cell_dofs.shape[1]).flatten())

    # Call assembly loop
    val, vec = _assemble((LHS_kernel, RHS_kernel),
                         cells, vertices,
                         cell_dofs, bcs, bcvals)

    mat = scipy.sparse.coo_matrix((val, cij))
    mat.eliminate_zeros()
    return mat.tocsr(), vec
