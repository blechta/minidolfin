import tsfc
import dijitso
import ffc.compiler
from coffee.plan import ASTKernel
from ufl.utils.sorting import canonicalize_metadata

import numpy
import numba
from petsc4py import PETSc

import sys
import os
import ctypes
import ctypes.util
import hashlib


def tsfc_compile_wrapper(form, parameters=None):
    """Compile form with TSFC and return source code"""

    parameters_ = {'mode': 'spectral'}
    parameters_.update(parameters or {})

    kernel, = tsfc.compile_form(form, parameters=parameters_)

    k = ASTKernel(kernel.ast)
    k.plan_cpu(dict(optlevel='Ov'))

    code = kernel.ast.gencode()

    code = code.replace('static inline', '')
    code = "#include <math.h>\n\n" + code

    return code


def ffc_compile_wrapper(form, parameters=None):
    """Compile form with FFC and return source code"""

    parameters_ = ffc.parameters.default_parameters()
    parameters_.update(parameters or {})

    # Call FFC
    code_h, code_c = ffc.compiler.compile_form(form, parameters=parameters_)

    prefix = "form"
    form_index = 0

    # Extract tabulate_tensor definition
    function_name = "tabulate_tensor_{}_cell_integral_{}_otherwise".format(prefix, form_index)
    index_start = code_c.index("void {}(".format(function_name))
    index_end = code_c.index("ufc_cell_integral* create_{}_cell_integral_{}_otherwise(void)".format(prefix, form_index),
                             index_start)
    tabulate_tensor_code = code_c[index_start:index_end].strip()

    # Extract tabulate_tensor body
    body_start = tabulate_tensor_code.index("{")
    tabulate_tensor_body = tabulate_tensor_code[body_start:].strip()

    tabulate_tensor_signature = "void form_cell_integral_otherwise (double* restrict A, const double *restrict coordinate_dofs)"

    return "\n".join([
        "#include <math.h>",
        "#include <stdalign.h>",
        "#include <string.h>",
        ""
        tabulate_tensor_signature,
        tabulate_tensor_body,
    ])


def jit_compile_form(a, parameters=None):
    """JIT-compile form and return ctypes function pointer"""

    # Prevent modification of user parameters
    parameters = parameters.copy() if parameters is not None else {}

    # Use tsfc as default form compiler
    compiler = parameters.pop("compiler", "tsfc")
    compile_form = {"ffc": ffc_compile_wrapper,
                    "tsfc": tsfc_compile_wrapper
                   }[compiler]

    # Define generation function executed on cache miss
    def generate(form, name, signature, jit_params):
        code = compile_form(form, parameters=parameters)
        return None, code, ()

    # Compute unique name
    hash_data = ("minidolfin", compiler,
                 canonicalize_metadata(parameters),
                 a.signature())
    hash = hashlib.sha512(str(hash_data).encode("utf-8")).hexdigest()
    name = "minidolfin_{}_{}".format(compiler, hash)

    # Set dijitso into C mode
    jit_params = {
         'build': {
             'cxx': 'cc',
             'cxxflags': ('-Wall', '-shared', '-fPIC', '-std=c11'),
         },
         'cache': {'src_postfix': '.c'},
    }

    # Do JIT compilation
    module, name = dijitso.jit(a, name, jit_params, generate=generate)

    # Grab assembly kernel from ctypes module and set its arguments
    func = getattr(module, 'form_cell_integral_otherwise')
    func.argtypes = (ctypes.c_void_p, ctypes.c_void_p)

    return func


# Get C MatSetValues function from PETSc because can't call
# petsc4py.PETSc.Mat.setValues() with numba.jit(nopython=True)
libpetsc_path = ctypes.util.find_library('petsc')
if libpetsc_path is None:
    # NB: This is a hack for ctypes 3.5 which does not look into LD_LIBRARY_PATH
    petsc_dir = os.environ.get('PETSC_DIR', None)
    if petsc_dir is None:
        raise RuntimeError("Didn't find libpetsc. Try exporting PETSC_DIR.")
    so = {'linux': '.so', 'darwin': '.dylib'}[sys.platform]
    libpetsc_path = os.path.join(petsc_dir, 'lib', 'libpetsc' + so)
petsc = ctypes.CDLL(libpetsc_path)
MatSetValues = petsc.MatSetValues
MatSetValues.argtypes = 7*(ctypes.c_void_p,)
ADD_VALUES = PETSc.InsertMode.ADD_VALUES
del petsc


def assemble(petsc_tensor, dofmap, form, form_compiler_parameters=None):
    assert len(form.arguments()) == 2, "Now only bilinear forms"

    # JIT compile UFL form into ctypes function
    assembly_kernel = jit_compile_form(form, form_compiler_parameters)

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

    # Prepare coordinates temporary
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
