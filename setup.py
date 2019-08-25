import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    config.add_extension('lensit.bicubic.bicubic', ['lensit/bicubic/bicubic.f90'])
    return config

setup(
    name='lensit',
    packages=['lensit', 'lensit.qcinv', 'lensit.ffs_covs', 'lensit.ffs_deflect',
              'lensit.pbs', 'lensit.misc', 'lensit.pseudocls', 'lensit.ffs_iterators'],
    data_files=[('lensit/data/cls', ['lensit/data/cls/fiducial_flatsky_lensedCls.dat',
                                         'lensit/data/cls/fiducial_flatsky_lenspotentialCls.dat',
                                         'lensit/data/cls/fiducial_params_tensor.ini',
                                          'lensit/data/cls/fiducial_tensCls.dat'])],
    url='https://github.com/carronj/lensit',
    author='Julien Carron',
    author_email='to.jcarron@gmail.com',
    description='CMB lensing flat-sky quadratic and iterative estimation tools',
    install_requires=['numpy', 'pyfftw', 'scipy'], #skipped mpi4py for travis
    long_description=long_description,
    configuration=configuration)

