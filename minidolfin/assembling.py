import tsfc
import dijitso
from coffee.plan import ASTKernel
import numpy
import numba
from petsc4py import PETSc

import ctypes
import hashlib


def compile_form(a):

    # Define generation function executed on cache miss
    def generate(form, name, signature, params):
        kernel, = tsfc.compile_form(form, parameters={'mode': 'spectral'})

        k = ASTKernel(kernel.ast)
        k.plan_cpu(dict(optlevel='Ov'))

        code = kernel.ast.gencode()

        code = code.replace('static inline', '')
        code = "#include <math.h>\n\n" + code

        return None, code, ()

    # Compute unique name
    name = "mfc" + a.signature()
    name = hashlib.sha1(name.encode()).hexdigest()

    # Set dijitso into C mode
    params = {
         'build': {
             'cxx': 'cc',
             'cxxflags': ('-Wall', '-shared', '-fPIC', '-std=c11'),
         },
         'cache': {'src_postfix': '.c'},
    }

    # Do JIT compilation
    module, name = dijitso.jit(a, name, params, generate=generate)

    # Grab assembly kernel from ctypes module and set its arguments
    func = getattr(module, 'form_cell_integral_otherwise')
    func.argtypes = (ctypes.c_void_p, ctypes.c_void_p)

    return func


# Get C MatSetValues function from PETSc because can't call
# petsc4py.PETSc.Mat.setValues() with numba.jit(nopython=True)
petsc = ctypes.CDLL('libpetsc.so')
MatSetValues = petsc.MatSetValues
MatSetValues.argtypes = 7*(ctypes.c_void_p,)
ADD_VALUES = PETSc.InsertMode.ADD_VALUES
del petsc


def assemble(petsc_tensor, dofmap, form):
    assert len(form.arguments()) == 2, "Now only bilinear forms"

    # JIT compile UFL form into ctypes function
    assembly_kernel = compile_form(form)

    # Fetch data
    tdim = dofmap.mesh.reference_cell.get_dimension()
    cells = dofmap.mesh.get_connectivity(tdim, 0)
    vertices = dofmap.mesh.vertices
    cell_dofs = dofmap.cell_dofs
    mat = petsc_tensor.handle

    # Prepare cell tensor temporary
    elements = tuple(arg.ufl_element() for arg in form.arguments())
    fiat_elements = map(tsfc.fiatinterface.create_element, elements)
    element_dims = tuple(fe.space_dimension() for fe in fiat_elements)
    _A = numpy.ndarray(element_dims)

    # Prepary coordinates temporary
    num_vertices_per_cell = cells.shape[1]
    gdim = vertices.shape[1]
    _coords = numpy.ndarray((num_vertices_per_cell, gdim), dtype=numpy.double)

    @numba.jit(nopython=True)
    def _assemble(assembly_kernel, cells, vertices, cell_dofs, mat, _coords, _A):
        coords_ptr = _coords.ctypes.data
        A_ptr = _A.ctypes.data
        nrows = ncols = cell_dofs.shape[1]

        # Loop over cells
        for i in range(cells.shape[0]):

            # Update temporaries
            _coords[:] = vertices[cells[i]]
            _A[:] = 0

            # Assemble cell tensor
            assembly_kernel(A_ptr, coords_ptr)

            # Add to global tensor
            rows = cols = cell_dofs[i].ctypes.data
            ierr = MatSetValues(mat, nrows, rows, ncols, cols, A_ptr, ADD_VALUES)
            assert ierr == 0

    # Call jitted hot loop
    _assemble(assembly_kernel, cells, vertices, cell_dofs, mat, _coords, _A)

    petsc_tensor.assemble()
