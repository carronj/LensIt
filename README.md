# Lensit

[![PyPI version](https://badge.fury.io/py/lensit.svg)](https://badge.fury.io/py/lensit)[![alt text](https://readthedocs.org/projects/lensit/badge/?version=latest)](https://lensit.readthedocs.io/en/latest)[![Build Status](https://travis-ci.com/carronj/lensit.svg?branch=master)](https://travis-ci.com/carronj/lensit)

This is a set of python tools dedicated to CMB lensing and CMB delensing, by Julien Carron.

**Main features are:**  
 - Maximum a posterior estimation of CMB lensing deflection maps from temperature and/or polarization maps.  
 (See https://arxiv.org/abs/1704.08230 by J.Carron and A. Lewis)  
 - Wiener filtering of masked CMB data and allowing for inhomogenous noise, including lensing deflections, using a multigrid preconditioner.  
 (Described in the same reference)
 - Fast and accurate simulation libraries for lensed CMB skies, and standard quadratic estimator lensing reconstruction tools.  
 (See https://arxiv.org/abs/1611.01446 by J. Peloton et al.)
 - CMB internal delensing tools, including internal delensing biases calculation for temperature and/or polarization maps.  
 (See https://arxiv.org/abs/1701.01712 by J. Carron, A. Lewis and A. Challinor)
 
Several parts were directly adapted from or inspired by qcinv (https://github.com/dhanson/qcinv) and quicklens (https://github.com/dhanson/quicklens) by Duncan Hanson, many thanks to him.

Many parts use the flat-sky approximation, with likely extension to curved-sky in a near future.  
To use the GPU implementation of some of the routines, you will need pyCUDA. (https://mathema.tician.de/software/pycuda)

An ipython notebook 'demo_basics.ipynb' covers the simple aspects of building simulation librairies. 

(**New** Sept. 2018)  The notebook 'demo_lensit.ipynb' shows an example of iterative lensing map reconstruction for a configuration roughly in line with CMB Stage IV specifications.

(**New** May 2019)  The notebook 'demo_curvlens.ipynb' shows how to generate a lensed CMB temperature map.


Other example and tests scripts might follow, or you may just write to me.

![alt text](https://erc.europa.eu/sites/default/files/content/erc_banner-vertical.jpg)