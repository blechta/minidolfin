import ffc
from ufl.utils.sorting import canonicalize_metadata
import ffc.codegeneration.jit
import cffi

import scipy.sparse
import numpy
import numba

import sys
import os
import ctypes
import ctypes.util
import hashlib

def jit_compile_form(form, params):

    compiled_forms, module = ffc.codegeneration.jit.compile_forms(
        [form], parameters={'scalar_type': 'double'})

    for f, compiled_f in zip([form], compiled_forms):
        assert compiled_f.rank == len(f.arguments())

    return compiled_forms[0][0]


def assemble(dofmap, form, form_compiler_parameters=None):
    assert len(form.arguments()) == 2, "Now only bilinear forms"

    # JIT compile UFL form into ctypes function
    assembly_kernel = jit_compile_form(form, form_compiler_parameters).create_default_cell_integral().tabulate_tensor

    # Fetch data
    tdim = dofmap.mesh.reference_cell.get_dimension()
    cells = dofmap.mesh.get_connectivity(tdim, 0)
    vertices = dofmap.mesh.vertices
    cell_dofs = dofmap.cell_dofs

    # Prepare cell tensor temporary
    elements = tuple(arg.ufl_element() for arg in form.arguments())
    fiat_elements = map(ffc.fiatinterface.create_element, elements)
    element_dims = tuple(fe.space_dimension() for fe in fiat_elements)

    # Prepare coordinates temporary
    num_vertices_per_cell = cells.shape[1]
    gdim = vertices.shape[1]

    def _assemble(assembly_kernel, cells, vertices, cell_dofs):
        nrows = ncols = cell_dofs.shape[1]

        # Loop over cells
        ci = []
        cj = []
        val = []
        for c in range(cells.shape[0]):

            # Assemble cell tensor
            A = numpy.zeros(element_dims, dtype=numpy.float64)
            w = numpy.array([], dtype=numpy.float64)
            ffi = cffi.FFI()
            coords = numpy.array(vertices[cells[c]], dtype=numpy.float64)
            assembly_kernel(ffi.cast('double *', A.ctypes.data),
                            ffi.cast('double *', w.ctypes.data),
                            ffi.cast('double *', coords.ctypes.data), 0)

            # Add to global tensor
            rows = cols = cell_dofs[c]
            for i, ig in enumerate(rows):
                for j, jg in enumerate(cols):
                    ci += [ig]
                    cj += [jg]
                    val += [A[i, j]]
        return ci, cj, val

    # Call assembly loop
    ci, cj, val = _assemble(assembly_kernel, cells, vertices, cell_dofs)

    print(len(ci), len(cj), len(val))

    mat = scipy.sparse.coo_matrix((val, (ci, cj)))

    return mat.tocsr()
