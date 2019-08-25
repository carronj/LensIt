"""mpi4py wrapper


"""
from __future__ import print_function
import os
import sys

verbose = False
if 'SLURM_SUBMIT_DIR' in os.environ.keys():
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        barrier = MPI.COMM_WORLD.Barrier
        finalize = MPI.Finalize
        if verbose:
            print('pbs.py : setup OK, rank %s in %s' % (rank, size))
    except:
        if verbose: sys.stderr.write('pbs.py: unable to setup mpi4py\n')
elif 'NERSC_HOST' not in os.environ.keys():
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        barrier = MPI.COMM_WORLD.Barrier
        finalize = MPI.Finalize
    except:
        rank = 0
        size = 1
        barrier = lambda: -1
        finalize = lambda: -1
        if verbose: print('pbs.py : This looks like invocation on login nodes')

else:
    rank = 0
    size = 1
    barrier = lambda: -1
    finalize = lambda: -1
    if verbose: print('pbs.py : This looks like invocation on login nodes')
