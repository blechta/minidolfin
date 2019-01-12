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
    ffi = cffi.FFI()

    @numba.jit(nopython=True)
    def _assemble(assembly_kernel, cells, vertices, cell_dofs):
        nrows = ncols = cell_dofs.shape[1]
        ncells = cells.shape[0]

        # Loop over cells
        ci = numpy.zeros(nrows*ncols*ncells, dtype=numpy.int32)
        cj = numpy.zeros(nrows*ncols*ncells, dtype=numpy.int32)
        val = numpy.zeros(nrows*ncols*ncells, dtype=numpy.float64)
        n = 0
        for c in range(ncells):

            # Assemble cell tensor
            A = numpy.zeros(element_dims, dtype=numpy.float64)
            w = numpy.array([0], dtype=numpy.float64)
            coords = numpy.empty((cells.shape[1], vertices.shape[1]))
            for i, q in enumerate(cells[c]):
                coords[i, :] = vertices[q]
#            coords = numpy.array(vertices[cells[c]], dtype=numpy.float64)
            assembly_kernel(ffi.from_buffer(A), #cast('double *', A.ctypes.data),
                            ffi.from_buffer(w), # cast('double *', w.ctypes.data),
                            ffi.from_buffer(coords), 0) # cast('double *', coords.ctypes.data), 0)

            # Add to global tensor
            rows = cols = cell_dofs[c]
            for i, ig in enumerate(rows):
                for j, jg in enumerate(cols):
                    ci[n] = ig
                    cj[n] = jg
                    val[n] = A[i, j]
                    n += 1

        return ci, cj, val

    # Call assembly loop
    ci, cj, val = _assemble(assembly_kernel, cells, vertices, cell_dofs)

    mat = scipy.sparse.coo_matrix((val, (ci, cj)))

    return mat.tocsr()
