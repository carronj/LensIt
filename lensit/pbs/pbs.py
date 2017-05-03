"""
jcarron wrapper to mpi4py to adapt pypar based DH code with minimal changes
"""

import os
import sys

if './' not in sys.path: sys.path.append('./')

verbose = True
if all(os.environ.has_key(k) for k in ['SLURM_SUBMIT_DIR']):
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        barrier = MPI.COMM_WORLD.Barrier
        finalize = MPI.Finalize
        if verbose:
            print 'pbs.py : setup OK, rank %s in %s' % (rank, size)
    except:
        if verbose: sys.stderr.write('pbs.py: unable to setup mpi4py\n')
elif not os.environ.has_key('NERSC_HOST'):
    if verbose: print 'pbs.py : This looks like invocation on the laptop'
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    barrier = MPI.COMM_WORLD.Barrier
    finalize = MPI.Finalize
    if verbose: print 'pbs.py : setup OK, rank %s in %s' % (rank, size)

else:
    # job on login node
    rank = 0
    size = 1
    # workdir = jobdir = os.getcwd()
    barrier = lambda: -1
    finalize = lambda: -1
    if verbose: print 'pbs.py : This looks like invocation on login nodes'
