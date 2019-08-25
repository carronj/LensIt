from __future__ import print_function

import glob
import os
import shutil
import time

import numpy as np

from lensit.pbs import pbs
from lensit.ffs_deflect import ffs_deflect
from lensit.ffs_qlms import qlms as ql
from lensit.ffs_covs import ffs_specmat, ffs_cov
from lensit.misc.misc_utils import PartialDerivativePeriodic as PDP, cl_inverse
from lensit.ffs_iterators import bfgs
from lensit.qcinv import multigrid, chain_samples
from lensit.sims import ffs_phas

_types = ['T', 'QU', 'TQU']


def prt_time(dt, label=''):
    dh = np.floor(dt / 3600.)
    dm = np.floor(np.mod(dt, 3600.) / 60.)
    ds = np.floor(np.mod(dt, 60))
    print("\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " + label)
    return


class ffs_iterator(object):
    r"""Flat-sky iterator template class

        Args:
            lib_dir: many things will be written there
            typ: 'T', 'QU' or 'TQU' for estimation on temperature data, polarization data or jointly
            filt: inverse-variance filtering instance (e.g. *lensit.qcinv.ffs_ninv_filt* )
            dat_maps: data maps or path to maps.
            lib_qlm: lib_alm (*lensit.ffs_covs.ell_mat.ffs_alm*) instance describing the lensing estimate Fourier arrays
            Plm0: Starting point for the iterative search. alm array consistent with *lib_qlm*
            H0: initial isotropic likelihood curvature approximation (roughly, inverse lensing noise bias :math:`N^{(0)}_L`)
            cpp_prior: fiducial lensing power spectrum, used for the prior part of the posterior density.
            chain_descr: multigrid conjugate gradient inversion chain description


    """
    def __init__(self, lib_dir, typ, filt, dat_maps, lib_qlm, Plm0, H0, cpp_prior,
                 use_Pool_lens=0, use_Pool_inverse=0, chain_descr=None, opfilt=None, soltn0=None, cache_magn=False,
                 no_deglensing=False, NR_method=100, tidy=10, verbose=True, maxcgiter=150, PBSSIZE=None, PBSRANK=None,
                 **kwargs):

        assert typ in _types
        assert chain_descr is not None
        assert opfilt is not None
        assert filt.lib_skyalm.lsides == lib_qlm.lsides


        self.PBSSIZE = pbs.size if PBSSIZE is None else PBSSIZE
        self.PBSRANK = pbs.rank if PBSRANK is None else PBSRANK
        assert self.PBSRANK < self.PBSSIZE, (self.PBSRANK, self.PBSSIZE)
        self.barrier = (lambda: 0) if self.PBSSIZE == 1 else pbs.barrier

        self.type = typ
        self.lib_dir = lib_dir
        self.dat_maps = dat_maps

        self.chain_descr = chain_descr
        self.opfilt = opfilt
        self.cl_pp = cpp_prior
        self.lib_qlm = lib_qlm

        self.cache_magn = cache_magn

        self.lsides = filt.lib_skyalm.lsides
        self.lmax_qlm = self.lib_qlm.ellmax
        self.NR_method = NR_method

        self.tidy = tidy
        self.maxiter = maxcgiter
        self.verbose = verbose

        self.nodeglensing = no_deglensing
        if self.verbose:
            print(" I see t", filt.Nlev_uKamin('t'))
            print(" I see q", filt.Nlev_uKamin('q'))
            print(" I see u", filt.Nlev_uKamin('u'))

            # Defining a trial newton step length :

        def newton_step_length(it, norm_incr):  # FIXME
            # Just trying if half the step is better for S4 QU
            if filt.Nlev_uKamin('t') > 2.1: return 1.0
            if filt.Nlev_uKamin('t') <= 2.1 and norm_incr >= 0.5:
                return 0.5
            return 0.5

        self.newton_step_length = newton_step_length
        self.soltn0 = soltn0

        f_id = ffs_deflect.ffs_id_displacement(filt.lib_skyalm.shape, filt.lib_skyalm.lsides)
        if not hasattr(filt, 'f') or not hasattr(filt, 'fi'):
            self.cov = filt.turn2wlfilt(f_id, f_id)
        else:
            filt.set_ffi(f_id, f_id)
            self.cov = filt
        if self.PBSRANK == 0:
            if not os.path.exists(self.lib_dir): os.makedirs(self.lib_dir)
        self.barrier()

        #FIXME
        #self.soltn_cond = np.all([np.all(self.filt.get_mask(_t) == 1.) for _t in self.type])
        self.soltn_cond = False
        print('ffs iterator : This is %s trying to setup %s' % (self.PBSRANK, lib_dir))
        # Lensed covariance matrix library :
        # We will redefine the displacement at each iteration step
        self.use_Pool = use_Pool_lens
        self.use_Pool_inverse = use_Pool_inverse

        if self.PBSRANK == 0:  # FIXME : hash and hashcheck
            if not os.path.exists(self.lib_dir):
                os.makedirs(self.lib_dir)
            if not os.path.exists(self.lib_dir + '/MAPlms'):
                os.makedirs(self.lib_dir + '/MAPlms')
            if not os.path.exists(self.lib_dir + '/cghistories'):
                os.makedirs(self.lib_dir + '/cghistories')

        # pre_calculation of qlm_norms with rank 0:
        if self.PBSRANK == 0 and \
                (not os.path.exists(self.lib_dir + '/qlm_%s_H0.dat' % ('P'))
                 or not os.path.exists(self.lib_dir + '/%shi_plm_it%03d.npy' % ('P', 0))):
            print('++ ffs_%s_iterator: Caching qlm_norms and N0s' % typ + self.lib_dir)

            # Caching qlm norm that we will use as zeroth order curvature : (with lensed weights)
            # Prior curvature :
            # Gaussian priors
            prior_pp = cl_inverse(self.cl_pp[0:self.lmax_qlm + 1])
            prior_pp[0] *= 0.5

            curv_pp = H0 + prior_pp  # isotropic estimate of the posterior curvature at the starting point
            self.cache_cl(self.lib_dir + '/qlm_%s_H0.dat' % ('P'), cl_inverse(curv_pp))
            print("     cached %s" % self.lib_dir + '/qlm_%s_H0.dat' % 'P')
            fname_P = self.lib_dir + '/%shi_plm_it%03d.npy' % ('P', 0)
            self.cache_qlm(fname_P, self.load_qlm(Plm0))
        self.barrier()

        if not os.path.exists(self.lib_dir + '/Hessian') and self.PBSRANK == 0:
            os.makedirs(self.lib_dir + '/Hessian')
            # We store here the rank 2 updates to the Hessian according to the BFGS iterations.

        if not os.path.exists(self.lib_dir + '/history_increment.txt') and self.PBSRANK == 0:
            with open(self.lib_dir + '/history_increment.txt', 'w') as file:
                file.write('# Iteration step \n' +
                           '# Exec. time in sec.\n' +
                           '# Increment norm (normalized to starting point displacement norm) \n' +
                           '# Total gradient norm  (all grad. norms normalized to initial total gradient norm)\n' +
                           '# Quad. gradient norm\n' +
                           '# Det. gradient norm\n' +
                           '# Pri. gradient norm\n' +
                           '# Newton step length\n')
                file.close()

        if self.PBSRANK == 0: print('++ ffs_%s masked iterator : setup OK' % type)
        self.barrier()

    def get_mask(self):
        ret = np.ones(self.cov.lib_datalm.shape, dtype=float)
        ret[np.where(self.cov.ninv_rad <= 0.)] *= 0
        return ret

    def get_datmaps(self):
        return np.load(self.dat_maps) if isinstance(self.dat_maps, str) else self.dat_maps

    def cache_qlm(self, fname, alm, pbs_rank=None):
        if pbs_rank is not None and self.PBSRANK != pbs_rank:
            return
        else:
            assert self.load_qlm(alm).ndim == 1 and self.load_qlm(alm).size == self.lib_qlm.alm_size
            print('rank %s caching ' % self.PBSRANK + fname)
            self.lib_qlm.write_alm(fname, self.load_qlm(alm))
            return

    def load_qlm(self, fname):
        return self.lib_qlm.read_alm(fname) if isinstance(fname, str) else fname

    def cache_rlm(self, fname, rlm):
        assert rlm.ndim == 1 and rlm.size == 2 * self.lib_qlm.alm_size, (rlm.ndim, rlm.size)
        print('rank %s caching ' % self.PBSRANK, fname)
        np.save(fname, rlm)

    def load_rlm(self, fname):
        rlm = np.load(fname)
        assert rlm.ndim == 1 and rlm.size == 2 * self.lib_qlm.alm_size, (rlm.ndim, rlm.size)
        return rlm

    @staticmethod
    def cache_cl(fname, cl):
        assert cl.ndim == 1
        np.savetxt(fname, cl)

    @staticmethod
    def load_cl(fname):
        assert os.path.exists(fname), fname
        return np.loadtxt(fname)

    def get_H0(self, key):
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname = os.path.join(self.lib_dir, 'qlm_%s_H0.dat' % key.upper())
        assert os.path.exists(fname), fname
        return self.load_cl(fname)

    def is_previous_iter_done(self, it, key):
        if it == 0: return True
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fn = os.path.join(self.lib_dir, '%s_plm_it%03d.npy' % ({'p': 'Phi', 'o': 'Om'}[key.lower()], it - 1))
        return os.path.exists(fn)


    def how_many_iter_done(self, key):
        """ Returns the number of points already calculated. 0th is the qest.

        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fn = os.path.join(self.lib_dir, '%s_plm_it*.npy' % {'p': 'Phi', 'o': 'Om'}[key.lower()])
        return len( glob.glob(fn))

    def get_Plm(self, it, key):
        """Loads solution at iteration *it*

        """
        if it < 0:
            return np.zeros(self.lib_qlm.alm_size, dtype=complex)
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fn = os.path.join(self.lib_dir,'%s_plm_it%03d.npy' % ({'p': 'Phi', 'o': 'Om'}[key.lower()], it))
        assert os.path.exists(fn), fn
        return self.load_qlm(fn)

    def get_Phimap(self, it, key):
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        return self.lib_qlm.alm2map(self.get_Plm(it, key))

    def _getfnames_f(self, key, it):
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_dx = os.path.join(self.lib_dir, 'f_%s_it%03d_dx.npy' % (key.lower(), it))
        fname_dy = os.path.join(self.lib_dir, 'f_%s_it%03d_dy.npy' % (key.lower(), it))
        return fname_dx, fname_dy

    def _getfnames_finv(self, key, it):
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_dx = os.path.join(self.lib_dir,  'finv_%s_it%03d_dx.npy' % (key.lower(), it))
        fname_dy = os.path.join(self.lib_dir,  'finv_%s_it%03d_dy.npy' % (key.lower(), it))
        return fname_dx, fname_dy

    def _calc_ffinv(self, it, key):
        """Calculates displacement at iter and its inverse. Only mpi rank 0 can do this.

        """
        assert self.PBSRANK == 0, 'NO MPI METHOD'
        if it < 0: return
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_dx, fname_dy = self._getfnames_f(key, it)

        if not os.path.exists(fname_dx) or not os.path.exists(fname_dy):
            # FIXME : does this from plm
            assert self.is_previous_iter_done(it, key)
            Phi_est_WF = self.get_Phimap(it, key)
            assert self.cov.lib_skyalm.shape == Phi_est_WF.shape
            assert self.cov.lib_skyalm.shape == self.lib_qlm.shape
            assert self.cov.lib_skyalm.lsides == self.lib_qlm.lsides
            rmin = np.array(self.cov.lib_skyalm.lsides) / np.array(self.cov.lib_skyalm.shape)
            print('rank %s caching displacement comp. for it. %s for key %s' % (self.PBSRANK, it, key))
            if key.lower() == 'p':
                dx = PDP(Phi_est_WF, axis=1, h=rmin[1])
                dy = PDP(Phi_est_WF, axis=0, h=rmin[0])
            else:
                dx = -PDP(Phi_est_WF, axis=0, h=rmin[0])
                dy = PDP(Phi_est_WF, axis=1, h=rmin[1])
            if self.PBSRANK == 0:
                np.save(fname_dx, dx)
                np.save(fname_dy, dy)
            del dx, dy
        lib_dir = os.path.join(self.lib_dir, 'f_%04d_libdir' % it)
        if not os.path.exists(lib_dir): os.makedirs(lib_dir)
        fname_invdx, fname_invdy = self._getfnames_finv(key, it)
        if not os.path.exists(fname_invdx) or not os.path.exists(fname_invdy):
            f = self._load_f(it, key)
            print('rank %s inverting displacement it. %s for key %s' % (self.PBSRANK, it, key))
            f_inv = f.get_inverse(use_Pool=self.use_Pool_inverse)
            np.save(fname_invdx, f_inv.get_dx())
            np.save(fname_invdy, f_inv.get_dy())
        lib_dir = os.path.join(self.lib_dir, 'finv_%04d_libdir' % it)
        if not os.path.exists(lib_dir): os.makedirs(lib_dir)
        assert os.path.exists(fname_invdx), fname_invdx
        assert os.path.exists(fname_invdy), fname_invdy
        return

    def _load_f(self, it, key):
        """Loads current displacement solution at iteration iter

        """
        fname_dx, fname_dy = self._getfnames_f(key, it)
        lib_dir = os.path.join(self.lib_dir,  'f_%04d_libdir' % it)
        assert os.path.exists(fname_dx), fname_dx
        assert os.path.exists(fname_dx), fname_dy
        assert os.path.exists(lib_dir), lib_dir
        return ffs_deflect.ffs_displacement(fname_dx, fname_dy, self.lsides,
                                            verbose=(self.PBSRANK == 0), lib_dir=lib_dir, cache_magn=self.cache_magn)

    def _load_finv(self, it, key):
        """Loads current inverse displacement solution at iteration iter.

        """
        fname_invdx, fname_invdy = self._getfnames_finv(key, it)
        lib_dir = os.path.join(self.lib_dir, 'finv_%04d_libdir' % it)
        assert os.path.exists(fname_invdx), fname_invdx
        assert os.path.exists(fname_invdx), fname_invdy
        assert os.path.exists(lib_dir), lib_dir
        return ffs_deflect.ffs_displacement(fname_invdx, fname_invdy, self.lsides,
                                            verbose=(self.PBSRANK == 0), lib_dir=lib_dir, cache_magn=self.cache_magn)

    def load_soltn(self, it, key):
        assert key.lower() in ['p', 'o']
        for i in np.arange(it, -1, -1):
            fname = os.path.join(self.lib_dir, 'MAPlms/Mlik_%s_it%s.npy' % (key.lower(), i))
            if os.path.exists(fname):
                print("rank %s loading " % pbs.rank + fname)
                return np.load(fname)
        if self.soltn0 is not None: return np.load(self.soltn0)[:self.opfilt.TEBlen(self.type)]
        return np.zeros((self.opfilt.TEBlen(self.type), self.cov.lib_skyalm.alm_size), dtype=complex)

    def _cache_tebwf(self, TEBMAP, it, key):
        assert key.lower() in ['p', 'o']
        fname = os.path.join(self.lib_dir,  'MAPlms/Mlik_%s_it%s.npy' % (key.lower(), it))
        print("rank %s caching " % pbs.rank + fname)
        np.save(fname, TEBMAP)

    def get_gradPpri(self, it, key, cache_only=False):
        """Builds prior gradient at iteration *it*

        """
        assert self.PBSRANK == 0, 'NO MPI method!'
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        assert it > 0, it
        fname = os.path.join(self.lib_dir, 'qlm_grad%spri_it%03d.npy' % (key.upper(), it - 1))
        if os.path.exists(fname):
            return None if cache_only else self.load_qlm(fname)
        assert self.is_previous_iter_done(it, key)
        grad = self.lib_qlm.almxfl(self.get_Plm(it - 1, key),
                                   cl_inverse(self.cl_pp if key.lower() == 'p' else self.cl_oo))
        self.cache_qlm(fname, grad, pbs_rank=0)
        return None if cache_only else self.load_qlm(fname)

    def _mlik2rest_tqumlik(self, TQUMlik, it, key):
        """Produces B^t Ni (data - B D Mlik) in TQU space, that is fed into the qlm estimator.

        """
        f_id = ffs_deflect.ffs_id_displacement(self.cov.lib_skyalm.shape, self.cov.lib_skyalm.lsides)
        self.cov.set_ffi(self._load_f(it - 1, key), self._load_finv(it - 1, key))
        temp = ffs_specmat.TQU2TEBlms(self.type, self.cov.lib_skyalm, TQUMlik)
        maps = self.get_datmaps() - self.cov.apply_Rs(self.type, temp)
        self.cov.apply_maps(self.type, maps, inplace=True)
        self.cov.set_ffi(f_id, f_id)
        temp = self.cov.apply_Rts(self.type, maps)
        return ffs_specmat.TEB2TQUlms(self.type, self.cov.lib_skyalm, temp)

    def calc_gradplikpdet(self, it, key):
        """Calculates the likelihood gradient (quadratic and mean-field parts)

        """
        assert 0, 'subclass this'

    def load_graddet(self, it, key):
        """Loads mean-field gradient at iteration *it*

            Gradient must have already been calculated

        """
        fname_detterm = os.path.join(self.lib_dir, 'qlm_grad%sdet_it%03d.npy' % (key.upper(), it))
        assert os.path.exists(fname_detterm), fname_detterm
        return self.load_qlm(fname_detterm)

    def load_gradpri(self, it, key):
        """Loads prior gradient at iteration *it*

            Gradient must have already been calculated

        """
        fname_prior = os.path.join(self.lib_dir, 'qlm_grad%spri_it%03d.npy' % (key.upper(), it))
        assert os.path.exists(fname_prior), fname_prior
        return self.load_qlm(fname_prior)

    def load_gradquad(self, it, key):
        """Loads likelihood quadratic piece gradient at iteration *it*

            Gradient must have already been calculated

        """
        fname_likterm = os.path.join(self.lib_dir, 'qlm_grad%slik_it%03d.npy' % (key.upper(), it))
        assert os.path.exists(fname_likterm), fname_likterm
        return self.load_qlm(fname_likterm)

    def load_total_grad(self, it, key):
        """Load the total gradient at iteration *it*.

            All gradients must have already been calculated.

         """
        return self.load_gradpri(it, key) + self.load_gradquad(it, key) + self.load_graddet(it, key)

    def _calc_norm(self, qlm):
        return np.sqrt(np.sum(self.lib_qlm.alm2rlm(qlm) ** 2))

    def _apply_curv(self, k, key, alphak, plm):
        """Apply curvature matrix making use of information incuding sk and yk.

        Applies v B_{k + 1}v = v B_k v +   (y^t v)** 2/(y^t s) - (s^t B v) ** 2 / (s^t B s))
        (B_k+1 = B  + yy^t / (y^ts) - B s s^t B / (s^t Bk s))   (all k on the RHS))

            For quasi Newton, s_k = x_k1 - x_k = - alpha_k Hk grad_k with alpha_k newton step-length.

        --> s^t B s at k is alpha_k^2 g_k H g_k
            B s = -alpha g_k
        """
        H = self.get_Hessian(max(k + 1,0), key) # get_Hessian(k) loads sk and yk from 0 to k - 1
        assert H.L > k, 'not implemented'
        assert len(alphak) >= (k + 1),(k + 1,len(alphak))
        dot_op = lambda plm1,plm2,:np.sum(self.lib_qlm.alm2cl(plm1,alm2=plm2) * self.lib_qlm.get_Nell()[:self.lib_qlm.ellmax + 1])
        if k <= -1:
            return dot_op(plm,self.lib_qlm.rlm2alm(H.applyB0k(self.lib_qlm.alm2rlm(plm),0)))
        ret = self._apply_curv(k - 1, key, alphak, plm)
        Hgk = H.get_mHkgk(self.lib_qlm.alm2rlm(self.load_total_grad(k, key)), k)
        st_Bs = alphak[k] ** 2 * dot_op(self.load_total_grad(k, key),self.lib_qlm.rlm2alm(Hgk))
        yt_s = dot_op(self.lib_qlm.rlm2alm(H.s(k)),self.lib_qlm.rlm2alm(H.y(k)))
        yt_v = dot_op(self.lib_qlm.rlm2alm(H.y(k)),plm)
        st_Bv = - alphak[k] *dot_op(self.load_total_grad(k, key),plm)
        return ret + yt_v ** 2 / yt_s - st_Bv ** 2 / st_Bs

    def get_lndetcurv_update(self, k, key, alphak):
        #Builds update to the BFGS log-determinant

        H = self.get_Hessian(k, key)
        Hgk = H.get_mHkgk(self.lib_qlm.alm2rlm(self.load_total_grad(k, key)), k)
        denom = np.sum(self.lib_qlm.alm2rlm(self.load_total_grad(k, key)) * Hgk)
        num = np.sum(self.lib_qlm.alm2rlm(self.load_total_grad(k + 1, key)) * Hgk)
        assert 1. - num / denom / alphak > 0.
        return np.log(1. - num / denom / alphak)

    def get_Gaussnoisesample(self, it, key,plm_noisephas, real_space=False, verbose=False):
        """Produce a Gaussian random field from the approximate BFGS covariance

            Args:
                it: iteration index
                key: 'p' or 'o' for lensing gradient or curl iteration
                plm_noisepha: unit spectra random phases of the right shape
                real_space: produces random field in real space if set, otherwise alm array


        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        assert plm_noisephas.shape == (self.lib_qlm.alm_size,),(plm_noisephas.shape,self.lib_qlm.alm_size)

        alm_0 = self.lib_qlm.almxfl(plm_noisephas, np.sqrt(self.get_H0(key)))
        ret = self.get_Hessian(max(it,0), key).sample_Gaussian(it, self.lib_qlm.alm2rlm(alm_0))
        return self.lib_qlm.alm2map(self.lib_qlm.rlm2alm(ret)) if real_space else self.lib_qlm.rlm2alm(ret)


    def get_Hessian(self, it, key):
        """Build the L-BFGS Hessian at iteration *it*


        """
        # Zeroth order inverse Hessian :
        apply_H0k = lambda rlm, k: \
            self.lib_qlm.alm2rlm(self.lib_qlm.almxfl(self.lib_qlm.rlm2alm(rlm), self.get_H0(key)))
        apply_B0k = lambda rlm, k: \
            self.lib_qlm.alm2rlm(self.lib_qlm.almxfl(self.lib_qlm.rlm2alm(rlm), cl_inverse(self.get_H0(key))))
        BFGS_H = bfgs.BFGS_Hessian(os.path.join(self.lib_dir,  'Hessian'), apply_H0k, {}, {}, L=self.NR_method,
                                             verbose=self.verbose,apply_B0k=apply_B0k)
        # Adding the required y and s vectors :
        for k in range(np.max([0, it - BFGS_H.L]), it):
            BFGS_H.add_ys(os.path.join(self.lib_dir,  'Hessian', 'rlm_yn_%s_%s.npy' % (k, key)),
                          os.path.join(self.lib_dir,  'Hessian', 'rlm_sn_%s_%s.npy' % (k, key)), k)
        return BFGS_H

    def build_incr(self, it, key, gradn):
        """Search direction

            BGFS method with 'self.NR method' BFGS updates to the Hessian.
            Initial Hessian are built from N0s.
            It must be rank 0 here.

            Args:
                it: current iteration level. Will produce the increment to phi_{k-1}, from gradient est. g_{k-1}
                      phi_{k_1} + output = phi_k
                key: 'p' or 'o'
                gradn: current estimate of the gradient (alm array)

            Returns:
                 increment for next iteration (alm array)

        """
        assert self.PBSRANK == 0, 'single MPI process method !'
        assert it > 0, it
        k = it - 2
        yk_fname = os.path.join(self.lib_dir, 'Hessian', 'rlm_yn_%s_%s.npy' % (k, key))
        if k >= 0 and not os.path.exists(yk_fname):  # Caching Hessian BFGS yk update :
            yk = self.lib_qlm.alm2rlm(gradn - self.load_total_grad(k, key))
            self.cache_rlm(yk_fname, yk)
        k = it - 1
        BFGS = self.get_Hessian(k, key)  # Constructing L-BFGS Hessian
        # get descent direction sk = - H_k gk : (rlm array). Will be cached directly
        sk_fname = os.path.join(self.lib_dir, 'Hessian', 'rlm_sn_%s_%s.npy' % (k, key))
        step = 0.
        if not os.path.exists(sk_fname):
            print("rank %s calculating descent direction" % self.PBSRANK)
            t0 = time.time()
            incr = BFGS.get_mHkgk(self.lib_qlm.alm2rlm(gradn), k)
            norm_inc = self._calc_norm(self.lib_qlm.rlm2alm(incr)) / self._calc_norm(self.get_Plm(0, key))
            step = self.newton_step_length(it, norm_inc)
            self.cache_rlm(sk_fname,incr * step)
            prt_time(time.time() - t0, label=' Exec. time for descent direction calculation')
        assert os.path.exists(sk_fname), sk_fname
        return self.lib_qlm.rlm2alm(self.load_rlm(sk_fname)),step

    def iterate(self, it, key, cache_only=False):
        """Performs an iteration

            This builds the gradients at iteration *it*, and the  potential estimate, and saves the *it* + 1 estimate.

        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        plm_fname = os.path.join(self.lib_dir, '%s_plm_it%03d.npy' % ({'p': 'Phi', 'o': 'Om'}[key.lower()], it))
        if os.path.exists(plm_fname): return None if cache_only else self.load_qlm(plm_fname)

        assert self.is_previous_iter_done(it, key), 'previous iteration not done'
        # Calculation in // of lik and det term :
        ti = time.time()
        if self.PBSRANK == 0:  # Single processes routines :
            self._calc_ffinv(it - 1, key)
            self.get_gradPpri(it, key, cache_only=True)
        self.barrier()
        # Calculation of the likelihood term, involving the det term over MCs :
        irrelevant = self.calc_gradplikpdet(it, key)
        self.barrier()  # Everything should be on disk now.
        if self.PBSRANK == 0:
            incr,steplength = self.build_incr(it, key, self.load_total_grad(it - 1, key))
            self.cache_qlm(plm_fname, self.get_Plm(it - 1, key) + incr, pbs_rank=0)

            # Saves some info about increment norm and exec. time :
            norm_inc = self._calc_norm(incr) / self._calc_norm(self.get_Plm(0, key))
            norms = [self._calc_norm(self.load_gradquad(it - 1, key))]
            norms.append(self._calc_norm(self.load_graddet(it - 1, key)))
            norms.append(self._calc_norm(self.load_gradpri(it - 1, key)))
            norm_grad = self._calc_norm(self.load_total_grad(it - 1, key))
            norm_grad_0 = self._calc_norm(self.load_total_grad(0, key))
            for i in [0, 1, 2]: norms[i] = norms[i] / norm_grad_0

            with open(os.path.join(self.lib_dir, 'history_increment.txt'), 'a') as file:
                file.write('%03d %.1f %.6f %.6f %.6f %.6f %.6f %.12f \n'
                           % (it, time.time() - ti, norm_inc, norm_grad / norm_grad_0, norms[0], norms[1], norms[2],
                              steplength))
                file.close()

            if self.tidy > 2:  # Erasing dx,dy and det magn (12GB for full sky at 0.74 amin per iteration)
                f1, f2 = self._getfnames_f(key, it - 1)
                f3, f4 = self._getfnames_finv(key, it - 1)
                for _f in [f1, f2, f3, f4]:
                    if os.path.exists(_f):
                        os.remove(_f)
                        if self.verbose: print("     removed :", _f)
                if os.path.exists(os.path.join(self.lib_dir, 'f_%04d_libdir' % (it - 1))):
                    shutil.rmtree(os.path.join(self.lib_dir, 'f_%04d_libdir' % (it - 1)))
                    if self.verbose: print("Removed :", os.path.join(self.lib_dir, 'f_%04d_libdir' % (it - 1)))
                if os.path.exists(os.path.join(self.lib_dir, 'finv_%04d_libdir' % (it - 1))):
                    shutil.rmtree(os.path.join(self.lib_dir, 'finv_%04d_libdir' % (it - 1)))
                    if self.verbose: print("Removed :", os.path.join(self.lib_dir, 'finv_%04d_libdir' % (it - 1)))

        self.barrier()
        return None if cache_only else self.load_qlm(plm_fname)


class ffs_iterator_cstMF(ffs_iterator):
    r"""Iterator instance, that uses fixed, input mean-field at each step.

        Args:
            lib_dir: many things will be written there
            typ: 'T', 'QU' or 'TQU' for estimation on temperature data, polarization data or jointly
            filt: inverse-variance filtering instance (e.g. *lensit.qcinv.ffs_ninv_filt* )
            dat_maps: data maps or path to maps.
            lib_qlm: lib_alm (*lensit.ffs_covs.ell_mat.ffs_alm*) instance describing the lensing estimate Fourier arrays
            Plm0: Starting point for the iterative search. alm array consistent with *lib_qlm*
            H0: initial isotropic likelihood curvature approximation (roughly, inverse lensing noise bias :math:`N^{(0)}_L`)
            MF_qlms: mean-field alm array (also desribed by lib_qlm)
            cpp_prior: fiducial lensing power spectrum, used for the prior part of the posterior density.


    """

    def __init__(self, lib_dir, typ, filt, dat_maps, lib_qlm, Plm0, H0, MF_qlms, cpp_prior, **kwargs):
        super(ffs_iterator_cstMF, self).__init__(lib_dir, typ, filt, dat_maps, lib_qlm, Plm0, H0, cpp_prior,
                                                 PBSSIZE=1, PBSRANK=0,  # so that all proc. act independently
                                                 **kwargs)
        self.MF_qlms = MF_qlms

    def calc_gradplikpdet(self, it, key):
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_likterm = os.path.join(self.lib_dir, 'qlm_grad%slik_it%03d.npy' % (key.upper(), it - 1))
        fname_detterm = os.path.join(self.lib_dir, 'qlm_grad%sdet_it%03d.npy' % (key.upper(), it - 1))
        assert it > 0, it
        if os.path.exists(fname_likterm) and os.path.exists(fname_detterm):
            return 0

        assert self.is_previous_iter_done(it, key)

        # Identical MF here
        self.cache_qlm(fname_detterm, self.load_qlm(self.MF_qlms))
        self.cov.set_ffi(self._load_f(it - 1, key), self._load_finv(it - 1, key))
        mchain = multigrid.multigrid_chain(self.opfilt, self.type, self.chain_descr, self.cov,
                                                    no_deglensing=self.nodeglensing)
        # FIXME : The solution input is not working properly sometimes. We give it up for now.
        # FIXME  don't manage to find the right d0 to input for a given sol ?!!
        soltn = self.load_soltn(it, key).copy() * self.soltn_cond
        self.opfilt._type = self.type
        mchain.solve(soltn, self.get_datmaps(), finiop='MLIK')
        self._cache_tebwf(soltn, it - 1, key)
        # soltn = self.opfilt.MLIK2BINV(soltn,self.cov,self.get_datmaps())
        # grad = - ql.get_qlms(self.type, self.cov.lib_skyalm, soltn, self.cov.cls, self.lib_qlm,
        #                     use_Pool=self.use_Pool, f=self.cov.f)[{'p': 0, 'o': 1}[key.lower()]]
        TQUMlik = self.opfilt.soltn2TQUMlik(soltn, self.cov)
        ResTQUMlik = self._mlik2rest_tqumlik(TQUMlik, it, key)
        grad = - ql.get_qlms_wl(self.type, self.cov.lib_skyalm, TQUMlik, ResTQUMlik, self.lib_qlm,
                                use_Pool=self.use_Pool, f=self._load_f(it - 1, key))[{'p': 0, 'o': 1}[key.lower()]]

        self.cache_qlm(fname_likterm, grad, pbs_rank=self.PBSRANK)
        # It does not help to cache both grad_O and grad_P as they do not follow the trajectory in plm space.
        return 0


class ffs_iterator_pertMF(ffs_iterator):
    """Iterator instance, that uses the deflection-perturbative prediction for the mean-field at each step.

        Args:
            lib_dir: many things will be written there
            typ: 'T', 'QU' or 'TQU' for estimation on temperature data, polarization data or jointly
            filt: inverse-variance filtering instance (e.g. *lensit.qcinv.ffs_ninv_filt* )
            dat_maps: data maps or path to maps.
            lib_qlm: lib_alm (*lensit.ffs_covs.ell_mat.ffs_alm*) instance describing the lensing estimate Fourier arrays
            Plm0: Starting point for the iterative search. alm array consistent with *lib_qlm*
            H0: initial isotropic likelihood curvature approximation (roughly, inverse lensing noise bias :math:`N^{(0)}_L`)
            cpp_prior: fiducial lensing power spectrum, used for the prior part of the posterior density.


    """

    def __init__(self, lib_dir, typ, filt, dat_maps, lib_qlm, Plm0, H0, cpp_prior,
                 init_rank=pbs.rank, init_barrier=pbs.barrier, **kwargs):
        super(ffs_iterator_pertMF, self).__init__(lib_dir, typ, filt, dat_maps, lib_qlm, Plm0, H0, cpp_prior,
                                                  PBSSIZE=1, PBSRANK=0,  # so that all proc. act independently
                                                  **kwargs)
        lmax_sky_ivf = filt.lib_skyalm.ellmax
        cls_noise = {'t': (filt.Nlev_uKamin('t') / 60. / 180. * np.pi) ** 2 * np.ones(lmax_sky_ivf + 1),
                     'q': (filt.Nlev_uKamin('q') / 60. / 180. * np.pi) ** 2 * np.ones(lmax_sky_ivf + 1),
                     'u': (filt.Nlev_uKamin('u') / 60. / 180. * np.pi) ** 2 * np.ones(lmax_sky_ivf + 1)}

        self.isocov = ffs_cov.ffs_diagcov_alm(os.path.join(lib_dir, 'isocov'),
                                              filt.lib_skyalm, filt.cls, filt.cls, filt.cl_transf, cls_noise,
                                              lib_skyalm=filt.lib_skyalm, init_rank=init_rank,
                                              init_barrier=init_barrier)

    def get_mfresp(self, key):
        return self.isocov.get_mfresplms(self.type, self.lib_qlm, use_cls_len=False)[{'p': 0, 'o': 1}[key.lower()]]

    def calc_gradplikpdet(self, it, key):
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_likterm = os.path.join(self.lib_dir, 'qlm_grad%slik_it%03d.npy' % (key.upper(), it - 1))
        fname_detterm = os.path.join(self.lib_dir, 'qlm_grad%sdet_it%03d.npy' % (key.upper(), it - 1))
        assert it > 0, it
        if os.path.exists(fname_likterm) and os.path.exists(fname_detterm):
            return 0

        assert self.is_previous_iter_done(it, key)
        # Identical MF here
        self.cache_qlm(fname_detterm, self.load_qlm(self.get_mfresp(key.lower()) * self.get_Plm(it - 1, key.lower())))
        self.cov.set_ffi(self._load_f(it - 1, key), self._load_finv(it - 1, key))
        mchain = multigrid.multigrid_chain(self.opfilt, self.type, self.chain_descr, self.cov,
                                           no_deglensing=self.nodeglensing)
        # FIXME : The solution input is not working properly sometimes. We give it up for now.
        # FIXME  don't manage to find the right d0 to input for a given sol ?!!
        soltn = self.load_soltn(it, key).copy() * self.soltn_cond
        self.opfilt._type = self.type
        mchain.solve(soltn, self.get_datmaps(), finiop='MLIK')
        self._cache_tebwf(soltn, it - 1, key)
        TQUMlik = self.opfilt.soltn2TQUMlik(soltn, self.cov)
        ResTQUMlik = self._mlik2rest_tqumlik(TQUMlik, it, key)
        grad = - ql.get_qlms_wl(self.type, self.cov.lib_skyalm, TQUMlik, ResTQUMlik, self.lib_qlm,
                                use_Pool=self.use_Pool, f=self._load_f(it - 1, key))[{'p': 0, 'o': 1}[key.lower()]]

        self.cache_qlm(fname_likterm, grad, pbs_rank=self.PBSRANK)
        # It does not help to cache both grad_O and grad_P as they do not follow the trajectory in plm space.
        return 0


class ffs_iterator_simMF(ffs_iterator):
    r"""Iterator instance, that estimate the mean-field at each steps from Monte-Carlos.

        Args:
            lib_dir: many things will be written there
            typ: 'T', 'QU' or 'TQU' for estimation on temperature data, polarization data or jointly
            MFkey: mean-field estimator key
            nsims: number of sims to use at each step
            filt: inverse-variance filtering instance (e.g. *lensit.qcinv.ffs_ninv_filt* )
            dat_maps: data maps or path to maps.
            lib_qlm: lib_alm (*lensit.ffs_covs.ell_mat.ffs_alm*) instance describing the lensing estimate Fourier arrays
            Plm0: Starting point for the iterative search. alm array consistent with *lib_qlm*
            H0: initial isotropic likelihood curvature approximation (roughly, inverse lensing noise bias :math:`N^{(0)}_L`)
            cpp_prior: fiducial lensing power spectrum, used for the prior part of the posterior density.


    """
    def __init__(self, lib_dir, typ, MFkey, nsims, filt, dat_maps, lib_qlm, Plm0, H0, cpp_prior, **kwargs):
        super(ffs_iterator_simMF, self).__init__(lib_dir, typ, filt, dat_maps, lib_qlm, Plm0, H0, cpp_prior,
                                                 **kwargs)
        print('++ ffs_%s simMF iterator (PBSSIZE %s pbs.size %s) : setup OK' % (self.type, self.PBSSIZE, pbs.size))
        self.MFkey = MFkey
        self.nsims = nsims
        self.same_seeds = kwargs.pop('same_seeds', False)
        self.subtract_phi0 = kwargs.pop('subtract_phi0', True)
        self.barrier()

    def build_pha(self, it):
        """Builds sims for the mean-field evaluation at iter *it*

        """
        if self.nsims == 0: return None
        phas_pix = ffs_phas.pix_lib_phas(
            os.path.join(self.lib_dir,  '%s_sky_noise_iter%s' % (self.type, it * (not self.same_seeds))),
            len(self.type), self.cov.lib_datalm.shape, nsims_max=self.nsims)
        phas_cmb = None  # dont need it so far
        if self.PBSRANK == 0:
            for lib, lab in zip([phas_pix, phas_cmb], ['phas pix', 'phas_cmb']):
                if not lib is None and not lib.is_full():
                    print("++ run iterator regenerating %s phases mf_sims rank %s..." % (lab, self.PBSRANK))
                    for idx in np.arange(self.nsims): lib.get_sim(idx, phas_only=True)
        self.barrier()
        return phas_pix, phas_cmb

    def calc_gradplikpdet(self, it, key, callback='default_callback'):
        """Caches the det term for iter via MC sims, together with the data one, with MPI maximal //isation.

        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_detterm = os.path.join(self.lib_dir, 'qlm_grad%sdet_it%03d.npy' % (key.upper(), it - 1))
        fname_likterm = os.path.join(self.lib_dir, 'qlm_grad%slik_it%03d.npy' % (key.upper(), it - 1))
        if os.path.exists(fname_detterm) and os.path.exists(fname_likterm):
            return 0
        assert self.is_previous_iter_done(it, key)

        pix_pha, cmb_pha = self.build_pha(it)
        if self.PBSRANK == 0 and not os.path.exists(os.path.join(self.lib_dir,  'mf_it%03d' % (it - 1))):
            os.makedirs(os.path.join(self.lib_dir,  'mf_it%03d' % (it - 1)))
        self.barrier()

        # Caching gradients for the mc_sims_mf sims , plus the dat map.
        # The gradient of the det term is the data averaged lik term, with the opposite sign.

        jobs = []
        try:
            self.load_qlm(fname_likterm)
        except:
            jobs.append(-1)  # data map
        for idx in range(self.nsims):  # sims
            if not os.path.exists(os.path.join(self.lib_dir, 'mf_it%03d/g%s_%04d.npy' % (it - 1, key.lower(), idx))):
                jobs.append(idx)
            else:
                try:  # just checking if file is OK.
                    self.load_qlm(os.path.join(self.lib_dir, 'mf_it%03d/g%s_%04d.npy' % (it - 1, key.lower(), idx)))
                except:
                    jobs.append(idx)
        self.opfilt._type = self.type
        # By setting the chain outside the main loop we avoid potential MPI barriers
        # in degrading the lib_alm libraries:
        mchain = multigrid.multigrid_chain(self.opfilt, self.type, self.chain_descr, self.cov,
                                                    no_deglensing=self.nodeglensing)
        for i in range(self.PBSRANK, len(jobs), self.PBSSIZE):
            idx = jobs[i]
            print("rank %s, doing mc det. gradients idx %s, job %s in %s at iter level %s:" \
                  % (self.PBSRANK, idx, i, len(jobs), it))
            ti = time.time()

            if idx >= 0:  # sim
                grad_fname = os.path.join(self.lib_dir, 'mf_it%03d/g%s_%04d.npy' % (it - 1, key.lower(), idx))
                self.cov.set_ffi(self._load_f(it - 1, key), self._load_finv(it - 1, key))
                MFest = ql.MFestimator(self.cov, self.opfilt, mchain, self.lib_qlm,
                                       pix_pha=pix_pha, cmb_pha=cmb_pha, use_Pool=self.use_Pool)
                grad = MFest.get_MFqlms(self.type, self.MFkey, idx)[{'p': 0, 'o': 1}[key.lower()]]
                if self.subtract_phi0:
                    isofilt = self.cov.turn2isofilt()
                    chain_descr_iso = chain_samples.get_isomgchain(
                        self.cov.lib_skyalm.ellmax, self.cov.lib_datalm.shape, iter_max=self.maxiter)
                    mchain_iso = multigrid.multigrid_chain(
                        self.opfilt, self.type, chain_descr_iso, isofilt, no_deglensing=self.nodeglensing)
                    MFest = ql.MFestimator(isofilt, self.opfilt, mchain_iso, self.lib_qlm,
                                           pix_pha=pix_pha, cmb_pha=cmb_pha, use_Pool=self.use_Pool)
                    grad -= MFest.get_MFqlms(self.type, self.MFkey, idx)[{'p': 0, 'o': 1}[key.lower()]]
                self.cache_qlm(grad_fname, grad, pbs_rank=self.PBSRANK)
            else:
                # This is the data.
                # FIXME : The solution input is not working properly sometimes. We give it up for now.
                # FIXME  don't manage to find the right d0 to input for a given sol ?!!
                self.cov.set_ffi(self._load_f(it - 1, key), self._load_finv(it - 1, key))
                soltn = self.load_soltn(it, key).copy() * self.soltn_cond
                mchain.solve(soltn, self.get_datmaps(), finiop='MLIK')
                self._cache_tebwf(soltn, it - 1, key)
                TQUMlik = self.opfilt.soltn2TQUMlik(soltn, self.cov)
                ResTQUMlik = self._mlik2rest_tqumlik(TQUMlik, it, key)
                grad = - ql.get_qlms_wl(self.type, self.cov.lib_skyalm, TQUMlik, ResTQUMlik, self.lib_qlm,
                                        use_Pool=self.use_Pool, f=self._load_f(it - 1, key))[
                    {'p': 0, 'o': 1}[key.lower()]]
                self.cache_qlm(fname_likterm, grad, pbs_rank=self.PBSRANK)

            print("%s it. %s sim %s, rank %s cg status  " % (key.lower(), it, idx, self.PBSRANK))
            # It does not help to cache both grad_O and grad_P as they do not follow the trajectory in plm space.
            # Saves some info about current iteration :
            if idx == -1:  # Saves some info about iteration times etc.
                with open(os.path.join(self.lib_dir, 'cghistories','history_dat.txt'), 'a') as file:
                    file.write('%04d %.3f \n' % (it, time.time() - ti))
                    file.close()
            else:
                with open(os.path.join(self.lib_dir, 'cghistories', 'history_sim%04d.txt' % idx), 'a') as file:
                    file.write('%04d %.3f \n' % (it, time.time() - ti))
                    file.close()
        self.barrier()
        if self.PBSRANK == 0:
            # Collecting terms and caching det term.
            # We also cache arrays formed from independent sims for tests.
            print("rank 0, collecting mc det. %s gradients :" % key.lower())
            det_term = np.zeros(self.lib_qlm.alm_size, dtype=complex)
            for i in range(self.nsims):
                fname = os.path.join(self.lib_dir, 'mf_it%03d'%(it -1),'g%s_%04d.npy'%(key.lower(), i))
                det_term = (det_term * i + self.load_qlm(fname)) / (i + 1.)
            self.cache_qlm(fname_detterm, det_term, pbs_rank=0)
            det_term *= 0.
            fname_detterm1 = fname_detterm.replace('.npy', 'MF1.npy')
            assert 'MF1' in fname_detterm1
            for i in np.arange(self.nsims)[0::2]:
                fname = os.path.join(self.lib_dir, 'mf_it%03d'%(it - 1),'g%s_%04d.npy'%(key.lower(), i))
                det_term = (det_term * i + self.load_qlm(fname)) / (i + 1.)
            self.cache_qlm(fname_detterm1, det_term, pbs_rank=0)
            det_term *= 0.
            fname_detterm2 = fname_detterm.replace('.npy', 'MF2.npy')
            assert 'MF2' in fname_detterm2
            for i in np.arange(self.nsims)[1::2]:
                fname = os.path.join(self.lib_dir, 'mf_it%03d'%(it - 1),'g%s_%04d.npy'%(key.lower(), i))
                det_term = (det_term * i + self.load_qlm(fname)) / (i + 1.)
            self.cache_qlm(fname_detterm2, det_term, pbs_rank=0)

            # Erase some temp files if requested to do so :
            if self.tidy > 1:
                # We erase as well the gradient determinant term that were stored on disk :
                files_to_remove = \
                    [os.path.join(self.lib_dir, 'mf_it%03d'%(it -1), 'g%s_%04d.npy'%(key.lower(), i)) for i in range(self.nsims)]
                print('rank %s removing %s maps in ' % (
                    self.PBSRANK, len(files_to_remove)), os.path.join(self.lib_dir, 'mf_it%03d'%(it - 1)))
                for file in files_to_remove: os.remove(file)
        self.barrier()
