import tsfc
import dijitso
from coffee.plan import ASTKernel
import numpy
import numba
from petsc4py import PETSc
from cffi import FFI

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

    ffi = FFI()
    ffi.cdef("""\
typedef enum {NOT_SET_VALUES, INSERT_VALUES, ADD_VALUES, MAX_VALUES, INSERT_ALL_VALUES, ADD_ALL_VALUES, INSERT_BC_VALUES, ADD_BC_VALUES} InsertMode;
int MatSetValues(void* mat,int m,const int idxm[],int n,const int idxn[],const double v[],int addv);
    """)
    petsc = ffi.dlopen("petsc")
    msv = petsc.MatSetValues
    #addv = PETSc.InsertMode.ADD_VALUES
    addv = petsc.ADD_VALUES
    petsc_tensor_handle = petsc_tensor.handle

    petsc_tensor_ = ffi.cast('char*', petsc_tensor_handle)
    A_ = ffi.cast('char*', A.ctypes.data)
    nrows = ncols = cells.shape[0]


    #ti = numpy.ctypeslib.ndpointer(dtype=PETSc.IntType, ndim=1, flags='C_CONTIGUOUS')
    #tv = numpy.ctypeslib.ndpointer(dtype=PETSc.ScalarType, ndim=1, flags='C_CONTIGUOUS')
    #ti = numpy.ctypeslib.ndpointer()
    #tv = numpy.ctypeslib.ndpointer()
    #setv = ctypes.CFUNCTYPE(None, ti, ti, tv)(setv)

    #@numba.jit(nopython=True)
    @numba.jit
    def _assemble(cells, vertices, assembly_kernel, coords_addr, A, coord_offset, cell_dofs):
        vertices_addr = vertices.ctypes.data
        coords_ptr = coords_addr.ctypes.data
        A_ptr = A.ctypes.data

        for i in range(cells.shape[0]):
            coords_addr[:] = vertices_addr + coord_offset*cells[i]
            A[:] = 0
            assembly_kernel(A_ptr, coords_ptr)

            rows = cols = ffi.cast('void*', cell_dofs[i].ctypes.data)
            ierr = msv(petsc_tensor_, nrows, rows, ncols, cols, A_, addv)
            #msv(petsc_tensor_handle, nrows, cell_dofs[i].ctypes.data, ncols, cell_dofs[i].ctypes.data, A.ctypes.data, addv)

    _assemble(cells, vertices, assembly_kernel, coords_addr, A, coord_offset, cell_dofs)

    petsc_tensor.assemble()
