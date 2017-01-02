# CC=g++ CXX=g++ python setup.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
name = 'matrix_multiply_inplace'
setup(ext_modules=cythonize(
                     [Extension(name, sources=[name+'.pyx'])],
                     language='c++'),
      cmdclass=dict(build_ext=build_ext))
