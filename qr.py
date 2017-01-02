# for e in {1..4}; do
#   qsub -b y -V -j y -r yes -l h_rt=1:00:00 -l hostname='r7n01*' -cwd ./qr.sh ${e}0;
# done
import os
assert os.environ.get('MKL_NUM_THREADS')=='8'
import sys
import numpy
import numpy.linalg
import numpy.random
from rasengan import tictoc
import scipy.linalg
import pdb
def print_config(numpy=0, hostname=0, ps=0):
    if numpy:
        numpy.show_config()
    pid = os.getpid()
    print 'pid', pid
    import subprocess
    cmd = ("ps uf %d;"%pid if ps else "")
    cmd += " grep '[TV][hm][rHRS][eSWw]' /proc/%d/status; "%pid
    if hostname:
        cmd += "echo hostname `hostname`"
    print subprocess.check_output(
        [cmd],
        stderr=subprocess.STDOUT,
        shell=True)
    return


def scale_by_diagonal(a, bs):
    for i in xrange(a.shape[1]-1, -1, -1):
        if bs[i] > 1e-6:
            scipy.linalg.blas.sscal(1/numpy.sqrt(bs[i]), a, n=a.shape[0], offx=i*a.shape[0])
        else:
            break
    a = a[:, i:]
    return [a, i]


def matrix_multiply(a, bu, inplace=True):
    if inplace:
        from matrix_multiply_inplace import matmul
        return matmul(a, bu)
    else:
        return scipy.linalg.blas.sgemm(1, a, bu)


def make_rand(shape, shape2=None):
    a = numpy.random.randint(-2**31, high=2**31, size=shape, dtype='int32')
    from type_convert_inplace import convert
    convert(a)
    a /= 2**31
    return a


def svd_1(a, debug=True, inplace=True):
    assert a.flags.c_contiguous
    if debug: print_config()
    # NOTE: scipy.linalg.blas.ssyrk(1, a, trans=1, lower=1)
    # Causes an unnecessary copy, because a is c_contiguous.
    b = scipy.linalg.blas.ssyrk(1, a.T, trans=0, lower=1)
    if debug: print_config()
    [bs, bu] = scipy.linalg.eigh(
        b,
        turbo=True,
        overwrite_a=True,
        check_finite=True)
    [bu, i] = scale_by_diagonal(bu, bs)
    if debug:
        print 'i', i
        print_config()
    c = matrix_multiply(a, bu, inplace=inplace)
    del bu
    del bs
    if debug: print_config()
    return [c, i]

def svd_2(a, debug=True, inplace=True):
    #     [q,r] = numpy.linalg.qr(a)
    #     del a
    #     print_config()
    #     [ru, rs, rv] = numpy.linalg.svd(r)
    #     print_config()
    #     numpy.dot(q, ru, out=q)
    pass
