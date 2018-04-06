import tsfc
import dijitso
from coffee.plan import ASTKernel
import numpy
import numba
from petsc4py import PETSc

import ctypes
import hashlib


def compile_form(a):
    def generate(form, name, signature, params):
        kernel, = tsfc.compile_form(form, parameters={'mode': 'spectral'})

        k = ASTKernel(kernel.ast)
        k.plan_cpu(dict(optlevel='Ov'))

        code = kernel.ast.gencode()

        code = code.replace('static inline', '')
        code = "#include <math.h>\n\n" + code

        return None, code, ()

    name = "mfc" + a.signature()
    name = hashlib.sha1(name.encode()).hexdigest()

    params = {
         'build': {
             'cxx': 'cc',
             'cxxflags': ('-Wall', '-shared', '-fPIC', '-std=c11'),
         },
         'cache': {'src_postfix': '.c'},
    }
    module, name = dijitso.jit(a, name, params, generate=generate)

    func = getattr(module, 'form_cell_integral_otherwise')
    func.argtypes = (ctypes.c_void_p, ctypes.c_void_p)

    return func


petsc = ctypes.CDLL('libpetsc.so')
MatSetValues = petsc.MatSetValues
MatSetValues.argtypes = 7*(ctypes.c_void_p,)
ADD_VALUES = PETSc.InsertMode.ADD_VALUES
del petsc


def assemble(petsc_tensor, dofmap, form):
    assembly_kernel = compile_form(form)

    elements = tuple(arg.ufl_element() for arg in form.arguments())
    fiat_elements = map(tsfc.fiatinterface.create_element, elements)

    element_dims = tuple(fe.space_dimension() for fe in fiat_elements)
    A = numpy.ndarray(element_dims)

    tdim = dofmap.mesh.reference_cell.get_dimension()
    cells = dofmap.mesh.get_connectivity(tdim, 0)
    vertices = dofmap.mesh.vertices
    cell_dofs = dofmap.cell_dofs

    num_vertices_per_cell = cells.shape[1]
    coords_addr = numpy.ndarray(num_vertices_per_cell, dtype=numpy.uintp)

    sizeof_double = ctypes.sizeof(ctypes.c_double)
    gdim = vertices.shape[1]
    coord_offset = gdim*sizeof_double

    mat = petsc_tensor.handle

    @numba.jit(nopython=True)
    def _assemble(cells, vertices, assembly_kernel, coords_addr, A, coord_offset, cell_dofs, mat):
        vertices_addr = vertices.ctypes.data
        coords_ptr = coords_addr.ctypes.data
        A_ptr = A.ctypes.data
        nrows = ncols = cells.shape[0]

        for i in range(cells.shape[0]):
            coords_addr[:] = vertices_addr + coord_offset*cells[i]
            A[:] = 0
            assembly_kernel(A_ptr, coords_ptr)

            rows = cols = cell_dofs[i].ctypes.data
            ierr = MatSetValues(mat, nrows, rows, ncols, cols, A_ptr, ADD_VALUES)
            assert ierr == 0

    _assemble(cells, vertices, assembly_kernel, coords_addr, A, coord_offset, cell_dofs, mat)

    petsc_tensor.assemble()
