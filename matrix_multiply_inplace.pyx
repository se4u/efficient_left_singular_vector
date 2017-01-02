'''
| Filename    : matrix_multiply_inplace.pyx
| Description : Perform inplace matrix multiplication
| Author      : Pushpendre Rastogi
| Created     : Sun Jan  1 22:43:06 2017 (-0500)
| Last-Updated: Mon Jan  2 01:15:07 2017 (-0500)
|           By: System User
|     Update #: 11
'''
cimport scipy.linalg.cython_blas as blas
cimport cython
import numpy as np
cimport numpy as np
# cdef void dgemm(char *transa, char *transb, int *m, int *n, int *k, d *alpha, d *a, int *lda, d *b, int *ldb, d *beta, d *c, int *ldc)

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.overflowcheck(False)
cdef np.ndarray[float,ndim=2] matrix_multiply_impl1(
    np.ndarray[float,ndim=2] a,
    np.ndarray[float,ndim=2] b):
    cdef:
        unsigned int i = 0
        char trans = 't'
        int m=b.shape[0], n=b.shape[1], incx=1, incy=1
        int lda=m
        float alpha=1, x, beta=0
        np.ndarray[float,ndim=1] y = np.zeros((m,), dtype='float32', order='C')
    for i in range(a.shape[0]):
        # (char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy)
        blas.sgemv(&trans, &m, &n, &alpha, &b[0, 0], &lda, &a[i,0], &incx, &beta, &y[0], &incy)
        a[i,:] = y
    return a

# @cython.initializedcheck(False)
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.overflowcheck(False)
# cdef np.ndarray[float,ndim=2] matrix_multiply_impl2(
#     np.ndarray[float,ndim=2] a,
#     np.ndarray[float,ndim=2] b):
#     ''' In order to compute ab, where a is really tall and thing,
#     internally I compute (b'a')' in chunks.
#     based on the knowledge that b is laid out in 'F' format.
#     '''
#     cdef:
#         unsigned int i = 0, offset
#         char transa='t', transb='t'
#         int m=1000, n=b.shape[1], k=b.shape[0]
#         int lda=k, ldb=k, ldc=m
#         int tmp_size = a.shape[0]%m
#         float alpha=1, beta=0
#         # a, b, c
#         # np.ndarray[float,ndim=2] y = np.zeros((m,), dtype='float32', order='C')
#         # np.ndarray[float,ndim=2] y_end = np.zeros((m,), dtype='float32', order='C')
#     for i in range(a.shape[0]/m):
#         offset = i * m
#         # (char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc)
#         blas.sgemm(&transa, &transb, &m, &n, &k, &alpha, )
#         a[i,:] = y
#     offset = (i+1) * m
#     print offset
#     return a


def matmul(a, b, method=1):
    assert method == 1
    a=np.ascontiguousarray(a)
    b=np.asfortranarray(b)
    return matrix_multiply_impl1(a,b)
