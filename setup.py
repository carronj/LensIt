import setuptools
from setuptools import setup, find_packages

# For Fortran extension support with modern Python/NumPy
try:
    from numpy.distutils.core import setup as numpy_setup
    from numpy.distutils.misc_util import Configuration
    USE_NUMPY_DISTUTILS = True
except ImportError:
    # NumPy 2.0+ removed numpy.distutils, use meson-python or setuptools
    USE_NUMPY_DISTUTILS = False

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    config.add_extension('lensit.bicubic.bicubic', ['lensit/bicubic/bicubic.f90'])
    return config


# Common setup arguments
setup_args = dict(
    name='lensit',
    version='1.0.0',
    packages=find_packages(include=['lensit', 'lensit.*']),
    package_data={
        'lensit': ['data/cls/*.dat', 'data/cls/*.ini'],
    },
    include_package_data=True,
    url='https://github.com/carronj/lensit',
    author='Julien Carron',
    author_email='to.jcarron@gmail.com',
    description='CMB lensing flat-sky quadratic and iterative estimation tools',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy',
        'pyfftw',
        'scipy',
    ],
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)

if USE_NUMPY_DISTUTILS:
    # Use numpy.distutils for Fortran extension (NumPy < 2.0)
    setup_args['configuration'] = configuration
    numpy_setup(**setup_args)
else:
    # For NumPy 2.0+, you'll need to use meson-python for Fortran extensions
    # or pre-compile the extension. For now, install without the Fortran extension.
    import warnings
    warnings.warn(
        "numpy.distutils is not available (NumPy 2.0+). "
        "The Fortran bicubic extension will not be built. "
        "Consider using meson-python for building Fortran extensions, "
        "or install with NumPy < 2.0 for full functionality."
    )
    setup(**setup_args)

