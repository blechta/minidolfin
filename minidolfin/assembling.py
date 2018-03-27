import ufl
import tsfc
import dijitso
from coffee.plan import ASTKernel
import numpy
import numba

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


def assemble(mesh, form):
    assembly_kernel = compile_form(form)

    elements = tuple(arg.ufl_element() for arg in  form.arguments())
    fiat_elements = map(tsfc.fiatinterface.create_element, elements)

    element_dims = tuple(fe.space_dimension() for fe in fiat_elements)
    A = numpy.ndarray(element_dims)

    num_vertices_per_cell = mesh.cells.shape[1]
    coords_addr = numpy.ndarray(num_vertices_per_cell, dtype=numpy.uintp)

    sizeof_double = ctypes.sizeof(ctypes.c_double)
    gdim = mesh.vertices.shape[1]
    coord_offset = gdim*sizeof_double

    @numba.jit(nopython=True)
    def _assemble(mesh, assembly_kernel, coords_addr, A):
        vertices_addr = mesh.vertices.ctypes.data
        coords_ptr = coords_addr.ctypes.data
        A_ptr = A.ctypes.data
        cells = mesh.cells

        for i in range(cells.shape[0]):
            coords_addr[:] = vertices_addr + coord_offset*cells[i]
            A[:] = 0
            assembly_kernel(A_ptr, coords_ptr)

    _assemble(mesh, assembly_kernel, coords_addr, A)
