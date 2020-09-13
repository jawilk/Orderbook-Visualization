from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("book_opt.pyx"),
    include_dirs=[numpy.get_include()],
)
