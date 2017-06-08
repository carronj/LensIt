#!/usr/bin/env python
# to compile fortran code for spherical harmonic transforms inplace, run
# python setup.py build_ext --inplace --fcompiler=gnu95

import glob
import numpy as np
import distutils

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('',parent_package,top_path)
    config.add_extension('fsht', ['shts.f95'],
                             libraries=['gomp'], f2py_options=[],
                             extra_compile_args=['-fopenmp'], extra_link_args=[],)

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(name='lensing',
          configuration=configuration)
