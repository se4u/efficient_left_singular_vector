from qr import *
shape = (int(1e7), int(sys.argv[1]))
# When ncol = 10 then Size = .25 GB
# When ncol = 1200 then Size = 30 GB
# Total size = 30 + 30 + 15MB
a = make_rand(shape)
for inplace in [True]: # False
    print_config(numpy=0, hostname=0)
    with tictoc('Doing SVD_1 inplace=%s'%str(inplace)):
        [c, i] = svd_1(a, debug=True, inplace=inplace)
        del c
