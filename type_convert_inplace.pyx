'''
| Filename    : type_convert_inplace.pyx
| Description : Perform inplace Type conversin of numpy ndarrays
| Author      :
| Created     :
| Last-Updated: Mon Jan  2 03:05:05 2017 (-0500)
|           By: System User
|     Update #: 15

CC=g++ python -c "import pyximport; pyximport.install(); import type_convert_inplace; import numpy as np; a = np.random.randint(10, size=(10,), dtype='int32'); b = list(a); type_convert_inplace.convert(a); print a, b"
'''
cimport scipy.linalg.cython_blas as blas
cimport cython
from cython.operator cimport dereference as deref
import numpy as np
cimport numpy as np

cdef float* convert_impl2(void* a, unsigned long int n):
    cdef unsigned long int i
    for i in range(n):
        (<float*>(a+4*i))[0] = (<float>(deref(<int*>(a+4*i))))
    return <float*>(a)

cdef convert_impl1(np.ndarray a):
    convert_impl2(a.data, a.size)
    a.dtype = 'float32'
    return

def convert(a):
    convert_impl1(a)
