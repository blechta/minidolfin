import ufl
import tsfc
import dijitso
from coffee.plan import ASTKernel
import numpy
import numba

import ctypes
import hashlib
import collections


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
                              dtype=numpy.uint32)
    offset = 0

    for dim, local_dofs in fiat_element.entity_dofs().items():
        dofs_per_entity = len(local_dofs[0])
        connectivity = mesh.get_connectivity(tdim, dim)

        for k in range(dofs_per_entity):
            entity_dofs = [dofs[k] for entity, dofs in sorted(local_dofs.items())]
            cell_dofs[:, entity_dofs] = dofs_per_entity*connectivity + (offset+k)

        offset += dofs_per_entity*mesh.num_entities(dim)

    # Build dofmap structure and store what it depends on
    return DofMap(cell_dofs=cell_dofs, dim=offset, mesh=mesh, element=element)


def assemble(dofmap, form):
    assembly_kernel = compile_form(form)

    elements = tuple(arg.ufl_element() for arg in  form.arguments())
    fiat_elements = map(tsfc.fiatinterface.create_element, elements)

    element_dims = tuple(fe.space_dimension() for fe in fiat_elements)
    A = numpy.ndarray(element_dims)

    tdim = dofmap.mesh.reference_cell.get_dimension()
    cells = dofmap.mesh.get_connectivity(tdim, 0)
    vertices = dofmap.mesh.vertices

    num_vertices_per_cell = cells.shape[1]
    coords_addr = numpy.ndarray(num_vertices_per_cell, dtype=numpy.uintp)

    sizeof_double = ctypes.sizeof(ctypes.c_double)
    gdim = vertices.shape[1]
    coord_offset = gdim*sizeof_double

    @numba.jit(nopython=True)
    def _assemble(cells, vertices, assembly_kernel, coords_addr, A, coord_offset):
        vertices_addr = vertices.ctypes.data
        coords_ptr = coords_addr.ctypes.data
        A_ptr = A.ctypes.data

        for i in range(cells.shape[0]):
            coords_addr[:] = vertices_addr + coord_offset*cells[i]
            A[:] = 0
            assembly_kernel(A_ptr, coords_ptr)

    _assemble(cells, vertices, assembly_kernel, coords_addr, A, coord_offset)
