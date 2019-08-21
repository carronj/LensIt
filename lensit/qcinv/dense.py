from __future__ import print_function

import numpy as np
import os
from lensit.qcinv.utils import ffs_converter
import pickle as pk


class pre_op_dense:
    def __init__(self, cov, fwd_op, TEBlen, cache_fname=None):
        self.cov = cov
        lmax = self.cov.lib_skyalm.ellmax
        self.converter = ffs_converter(cov.lib_skyalm)
        self.fwd_op = fwd_op
        self.TEBlen = TEBlen

        if cache_fname is not None and os.path.exists(cache_fname):
            assert cache_fname[-3:] == '.pk'
            [cache_lmax, cache_hashdict] = pk.load(open(cache_fname, 'r'))
            self.minv = np.load(cache_fname[:-3] + '.npy')

            if (lmax != cache_lmax) or (self.hashdict() != cache_hashdict):
                print("WARNING: PRE_OP_DENSE CACHE: hashcheck failed. recomputing.")
                os.remove(cache_fname)
                self.compute_minv(cache_fname=cache_fname)
        else:
            self.compute_minv(cache_fname=cache_fname)

    def _rlms2datalms(self, rlms):
        return self.converter.rlms2datalms(self.TEBlen, rlms)

    def _datalms2rlms(self, alms):
        return self.converter.datalms2rlms(self.TEBlen, alms)

    def compute_minv(self, cache_fname=None):
        if cache_fname is not None: assert (not os.path.exists(cache_fname))
        # ! the rlm in the current scheme still contain redundant frequencies. kx = 0
        rlms = self._datalms2rlms(np.zeros((self.TEBlen, self.cov.lib_skyalm.alm_size), dtype=complex))
        nrlm = rlms.size
        tmat = np.zeros((nrlm, nrlm), dtype=float)

        print("computing dense preconditioner:")
        print("     lmin,lmax  = (%s, %s)" % (self.cov.lib_skyalm.ellmin, self.cov.lib_skyalm.ellmax))
        print("     dense matrix shape = ", tmat.shape)
        for i in np.arange(0, nrlm):
            if np.mod(i, int(0.1 * nrlm)) == 0: print ("   filling M: %4.1f" % (100. * i / nrlm)), "%"
            rlms[i] = 1.0
            tmat[:, i] = self._datalms2rlms(self.fwd_op(self._rlms2datalms(rlms)))
            rlms[i] = 0.0

        print("   inverting M...")
        if not self.converter.has_ell0:
            # The matrix is not symmetric if the zero mode is present !! # FIXME what ??
            eigv, eigw = np.linalg.eigh(tmat)
            if not np.all(eigv > 0.):
                print(" ! --- negative eigenvalues in dense covariance --- ")
            eigv_inv = np.zeros_like(eigv)
            eigv_inv[np.where(eigv > 0.)] = 1. / eigv[np.where(eigv > 0.)]

            self.minv = np.dot(np.dot(eigw, np.diag(eigv_inv)), np.transpose(eigw))
        else:
            eigv, eigw = np.linalg.eigh(tmat)
            if not np.all(eigv > 0.):
                print(" ! --- negative eigenvalues in dense covariance --- ")
            eigv_inv = np.zeros_like(eigv)
            eigv_inv[np.where(eigv > 0.)] = 1. / eigv[np.where(eigv > 0.)]

            self.minv = np.dot(np.dot(eigw, np.diag(eigv_inv)), np.transpose(eigw))

            #self.minv = np.linalg.inv(tmat)
        if cache_fname is not None:
            assert cache_fname[-3:] == '.pk'
            pk.dump([self.cov.lib_skyalm.ellmax, self.hashdict()], open(cache_fname, 'w'))
            np.save(cache_fname[:-3] + '.npy', self.minv)

    def hashdict(self):
        return {'lmax': self.cov.lib_skyalm.ellmax,
                'cov': self.cov.hashdict()}

    def __call__(self, alms):
        return self._rlms2datalms(np.dot(self.minv, self._datalms2rlms(alms)))

    def _testcond(self, alms):
        alms_new = self.fwd_op(self(alms))
        print(" test dense cond :: allclose ", np.allclose(alms_new, alms))
        print(" std dev :", np.std(alms_new - alms))
        return alms_new
