import dijitso
import numba
import numpy
import ctypes
import hashlib


def cjit(code, func_name, func_argtypes):
    """JIT code and return ctypes function with given name and arguments"""
    obj = None
    name = "dijit" + hashlib.sha1(code.encode()).hexdigest()
    params = {
         'build': {
             'cxx': 'cc',
             'cxxflags': ('-Wall', '-shared', '-fPIC', '-std=c11'),
         },
         'cache': {'src_postfix': '.c'},
    }

    def generate(obj, name, signature, params):
        return None, code, ()

    module, name = dijitso.jit(obj, name, params, generate=generate)
    func = getattr(module, func_name)
    func.argtypes = func_argtypes
    return func


# JIT the following code
# NB: we have to specify func names and argtypes (that's deficiency of
#     numba and ctypes interaction)
code = """\
void sq(double* a)
{
  *a = (*a)*(*a);
}
"""
func_name = "sq"
func_argtypes = (ctypes.c_void_p,)
sq = cjit(code, func_name, func_argtypes)


# Example 1: execute function from ctypes
z = ctypes.c_double(42)
sq(ctypes.addressof(z))
print(z.value)


# Examples 2: execute function from Numba
sizeof_double = ctypes.sizeof(ctypes.c_double)


@numba.jit(nopython=True)
def vecsq(a):
    for i in range(a.size):
        sq(a.ctypes.data + i*sizeof_double)


v = numpy.arange(1024, dtype=numpy.double)
vecsq(v)
print(v)
