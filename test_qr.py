import numpy
from qr import *

for inplace in [True, False]:
    shape = (15000, 300)
    a = make_rand(shape)
    au = numpy.linalg.svd(a, full_matrices=False)[0]
    c, i = svd_1(a, debug=False, inplace=inplace)
    assert numpy.linalg.norm(
        numpy.abs(numpy.dot(numpy.fliplr(au).T, c)) - numpy.eye(shape[1])) < 1e-2
print "Passed"
