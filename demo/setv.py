from cffi import FFI
ffi = FFI()
ffi.cdef("""\
typedef enum {NOT_SET_VALUES, INSERT_VALUES, ADD_VALUES, MAX_VALUES, INSERT_ALL_VALUES, ADD_ALL_VALUES, INSERT_BC_VALUES, ADD_BC_VALUES} InsertMode;
int MatSetValues(void* mat,int m,const int idxm[],int n,const int idxn[],const double v[],int addv);
""")
petsc = ffi.dlopen("petsc")

from petsc4py import PETSc
A = PETSc.Mat().createAIJ((1, 1), 1)
A.setPreallocationNNZ(1)
A.setUp()

import numpy
A_handle = A.handle
i = j = numpy.array((0,), dtype=numpy.int32)
v = numpy.array((42,), dtype=numpy.double)
msv = petsc.MatSetValues
addv = petsc.ADD_VALUES

import numba

#@numba.jit(nopython=True)
@numba.jit
def foo(A_handle):
    A_ = ffi.cast('void*', A_handle)
    i_ = ffi.cast('void*', i.ctypes.data)
    j_ = ffi.cast('void*', j.ctypes.data)
    v_ = ffi.cast('void*', v.ctypes.data)
    ierr = msv(A_, 1, i_, 1, j_, v_, addv)

foo(A_handle)
A.assemble()
A.view()
