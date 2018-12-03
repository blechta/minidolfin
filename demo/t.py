import numpy as np
import numba

a = np.ndarray((1,), dtype=np.uintp)
arr = np.ndarray((1,), dtype=np.uintp)

@numba.jit(nopython=True)
def foo(arr, a):
    print(a.ctypes.data)
    arr[0] = a.ctypes.data
    print(arr[0], arr)

foo(arr, a)
print(a.ctypes.data, arr)
