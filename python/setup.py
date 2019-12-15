#!/usr/bin/evn python3

import setuptools
from distutils.core import setup, Extension

MOD = 'mbsgd'
VERSION = '2.0'
LICENSE = 'apache-2.0'
NUMPY_INCLUDE_DIR = './ve_python3.6/lib/python3.6/site-packages/numpy/core/include'

setup(
    name = MOD,
    version = VERSION,
    license = LICENSE,
    author = 'Sid Chen',
    author_email = 'cyzchina@gmail.com',
    install_requires = [
        'pandas',
        'numpy>=1.14.0',
    ],

    ext_modules = [
        Extension(
            MOD,
            sources = ['./src/mbsgd.c', '../src/train.c', '../src/lr.c', '../src/random.c', '../src/task.c'],
            include_dirs = ['./include', '../include', NUMPY_INCLUDE_DIR],
            libraries = ['m'],
            define_macros = [('_GNU_SOURCE', None), ('_PYTHON_MBSGD', None)],
        )
    ],
)

