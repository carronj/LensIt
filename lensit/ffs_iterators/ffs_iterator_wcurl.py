from __future__ import print_function
from __future__ import annotations
import glob
import os
import shutil
import time

import numpy as np

from lensit.ffs_covs import ell_mat
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
class harmonicbump:
    def __init__(self, xa=400, xb=1500, a=0.5, b=0.1, scale=50):
        """Harmonic bumpy step that were useful for s06b and s08b

        """
        self.scale = scale
        self.bump_params = (xa, xb, a, b)

    def steplen(self, lmax_qlm):
        xa, xb, a, b = self.bump_params
        return self.bp(np.arange(lmax_qlm + 1),xa, a, xb, b, scale=self.scale)


    def build_incr(self, incrlm, lib_qlm):
        s = lib_qlm.alm_size
        assert incrlm.size % s == 0
        ncomp = incrlm.size // s
        assert ncomp == 2
        lib_qlm.almxfl(incrlm, [self.steplen(lib_qlm.ellmax) for i in range(ncomp)], inplace=True)
        return incrlm

    @staticmethod
    def bp(x, xa, a, xb, b, scale=50):
            """Bump function with f(xa) = a and f(xb) =  b with transition at midpoint over scale scale

            """
            x0 = (xa + xb) * 0.5
            r = lambda x_: np.arctan(np.sign(b - a) * (x_ - x0) / scale) + np.sign(b - a) * np.pi * 0.5
            return a + r(x) * (b - a) / r(xb)

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
    def __init__(self, lib_dir, typ, filt, dat_maps, lib_qlm:ell_mat.ffs_alm, POlm0:list[np.ndarray], H0s:list[np.ndarray], cl_priors,
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
        assert len(cl_priors) == 2
        self.cl_pp = cl_priors[0]
        self.cl_oo = cl_priors[1]
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

        #def newton_step_length(it, norm_incr):  # FIXME
        #    # Just trying if half the step is better for S4 QU
        #    # return 0.1
        #    if filt.Nlev_uKamin('t') > 2.1: return 1.0
        #    if filt.Nlev_uKamin('t') <= 2.1 and norm_incr >= 0.5:
        #        return 0.5
        #    return 0.5

        def steplength(incr):  # FIXME
            return harmonicbump(a=0.5, b=0.49).build_incr(incr, self.lib_qlm)

        self.newton_step_length = steplength
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
            prior_oo = cl_inverse(self.cl_oo[0:self.lmax_qlm + 1])
            prior_oo[0] *= 0.5
            assert len(H0s) == 2
            curv_pp = H0s[0] + prior_pp  # isotropic estimate of the posterior curvature at the starting point
            curv_oo = H0s[1] + prior_oo
            #FIXME: there is really no need to cache this...

            self.cache_cl(self.lib_dir + '/qlm_%s_H0.dat' %'P', cl_inverse(curv_pp))
            self.cache_cl(self.lib_dir + '/qlm_%s_H0.dat' %'O', cl_inverse(curv_oo))

            print("     cached %s" % self.lib_dir + '/qlm_%s_H0.dat' % 'P')
            print("     cached %s" % self.lib_dir + '/qlm_%s_H0.dat' % 'O')

            fname_P = self.lib_dir + '/polm_it%03d.npy'%0
            self.cache_qlm(fname_P, self.load_qlm(POlm0).flatten(order='C'))
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
            assert self.load_qlm(alm).ndim == 1 and self.load_qlm(alm).size == 2 * self.lib_qlm.alm_size
            print('rank %s caching ' % self.PBSRANK + fname)
            #self.lib_qlm.write_alm(fname, self.load_qlm(alm))
            np.save(fname, alm)
            return

    @staticmethod
    def load_qlm(fname):
        # return self.lib_qlm.read_alm(fname) if isinstance(fname, str) else fname
        return np.load(fname) if isinstance(fname, str) else fname


    def cache_rlm(self, fname, rlm):
        assert rlm.ndim == 1 and rlm.size%(2 * self.lib_qlm.alm_size) == 0, (rlm.ndim, rlm.size)
        assert rlm.size // self.lib_qlm.alm_size == 4
        print('rank %s caching ' % self.PBSRANK, fname)
        np.save(fname, rlm)

    def load_rlm(self, fname):
        rlm = np.load(fname)
        assert rlm.size // self.lib_qlm.alm_size == 4
        assert rlm.ndim == 1 and rlm.size%(2 * self.lib_qlm.alm_size) == 0, (rlm.ndim, rlm.size)
        return rlm

    @staticmethod
    def cache_cl(fname, cl):
        assert cl.ndim == 1
        np.savetxt(fname, cl)

    @staticmethod
    def load_cl(fname):
        assert os.path.exists(fname), fname
        return np.loadtxt(fname)

    def get_H0(self):
        fname_P = os.path.join(self.lib_dir, 'qlm_P_H0.dat')
        fname_O = os.path.join(self.lib_dir, 'qlm_O_H0.dat')
        assert os.path.exists(fname_P), fname_P
        assert os.path.exists(fname_O), fname_O
        return [self.load_cl(fname_P), self.load_cl(fname_O)]

    def is_previous_iter_done(self, it):
        if it == 0: return True
        fn = os.path.join(self.lib_dir, '%slm_it%03d.npy' % ('po', it - 1))
        return os.path.exists(fn)


    def how_many_iter_done(self):
        """ Returns the number of points already calculated. 0th is the qest.

        """
        fn = os.path.join(self.lib_dir, '%slm_it*.npy' %'po')
        return len( glob.glob(fn))

    def get_POlm(self, it):
        """Loads solution at iteration *it*

        """
        if it < 0:
            return np.zeros(2 * self.lib_qlm.alm_size, dtype=complex)
        fn = os.path.join(self.lib_dir,'polm_it%03d.npy' %it)
        assert os.path.exists(fn), fn
        ret =  self.load_qlm(fn)
        assert ret.size == 2 * self.lib_qlm.alm_size
        return ret

    def get_Phimap(self, it):
        s = self.lib_qlm.alm_size
        arr = self.get_POlm(it)
        assert arr.size == 2 * self.lib_qlm.alm_size
        return self.lib_qlm.alm2map(arr[0*s: 1*s])

    def get_Ommap(self, it):
        s = self.lib_qlm.alm_size
        arr = self.get_POlm(it)
        assert arr.size == 2 * self.lib_qlm.alm_size
        return self.lib_qlm.alm2map(arr[1*s: 2*s])

    def _getfnames_f(self, it):
        fname_dx = os.path.join(self.lib_dir, 'f_po_it%03d_dx.npy' %it)
        fname_dy = os.path.join(self.lib_dir, 'f_po_it%03d_dy.npy' %it)
        return fname_dx, fname_dy

    def _getfnames_finv(self, it):
        fname_dx = os.path.join(self.lib_dir,  'finv_po_it%03d_dx.npy'%it)
        fname_dy = os.path.join(self.lib_dir,  'finv_po_it%03d_dy.npy'%it)
        return fname_dx, fname_dy

    def _calc_ffinv(self, it):
        """Calculates displacement at iter and its inverse. Only mpi rank 0 can do this.

        """
        assert self.PBSRANK == 0, 'NO MPI METHOD'
        if it < 0: return
        fname_dx, fname_dy = self._getfnames_f(it)

        if not os.path.exists(fname_dx) or not os.path.exists(fname_dy):
            # FIXME : does this from plm
            assert self.is_previous_iter_done(it)
            Phi_est_WF = self.get_Phimap(it)
            Om_est_WF = self.get_Ommap(it)

            assert self.cov.lib_skyalm.shape == Phi_est_WF.shape
            assert self.cov.lib_skyalm.shape == self.lib_qlm.shape
            assert self.cov.lib_skyalm.lsides == self.lib_qlm.lsides
            rmin = np.array(self.cov.lib_skyalm.lsides) / np.array(self.cov.lib_skyalm.shape)
            print('rank %s caching displacement comp. for it. %s' % (self.PBSRANK, it))
            dx = PDP(Phi_est_WF, axis=1, h=rmin[1])
            dy = PDP(Phi_est_WF, axis=0, h=rmin[0])
            dx += -PDP(Om_est_WF, axis=0, h=rmin[0])
            dy += +PDP(Om_est_WF, axis=1, h=rmin[1])
            if self.PBSRANK == 0:
                np.save(fname_dx, dx)
                np.save(fname_dy, dy)
            del dx, dy
        lib_dir = os.path.join(self.lib_dir, 'f_%04d_libdir' % it)
        if not os.path.exists(lib_dir): os.makedirs(lib_dir)
        fname_invdx, fname_invdy = self._getfnames_finv(it)
        if not os.path.exists(fname_invdx) or not os.path.exists(fname_invdy):
            f = self._load_f(it)
            print('rank %s inverting displacement it. %s' % (self.PBSRANK, it))
            f_inv = f.get_inverse(use_Pool=self.use_Pool_inverse)
            np.save(fname_invdx, f_inv.get_dx())
            np.save(fname_invdy, f_inv.get_dy())
        lib_dir = os.path.join(self.lib_dir, 'finv_%04d_libdir' % it)
        if not os.path.exists(lib_dir): os.makedirs(lib_dir)
        assert os.path.exists(fname_invdx), fname_invdx
        assert os.path.exists(fname_invdy), fname_invdy
        return

    def _load_f(self, it):
        """Loads current displacement solution at iteration iter

        """
        fname_dx, fname_dy = self._getfnames_f(it)
        lib_dir = os.path.join(self.lib_dir,  'f_%04d_libdir' % it)
        assert os.path.exists(fname_dx), fname_dx
        assert os.path.exists(fname_dx), fname_dy
        assert os.path.exists(lib_dir), lib_dir
        return ffs_deflect.ffs_displacement(fname_dx, fname_dy, self.lsides,
                                            verbose=(self.PBSRANK == 0), lib_dir=lib_dir, cache_magn=self.cache_magn)

    def _load_finv(self, it):
        """Loads current inverse displacement solution at iteration iter.

        """
        fname_invdx, fname_invdy = self._getfnames_finv(it)
        lib_dir = os.path.join(self.lib_dir, 'finv_%04d_libdir' % it)
        assert os.path.exists(fname_invdx), fname_invdx
        assert os.path.exists(fname_invdx), fname_invdy
        assert os.path.exists(lib_dir), lib_dir
        return ffs_deflect.ffs_displacement(fname_invdx, fname_invdy, self.lsides,
                                            verbose=(self.PBSRANK == 0), lib_dir=lib_dir, cache_magn=self.cache_magn)

    def load_soltn(self, it):
        for i in np.arange(it, -1, -1):
            fname = os.path.join(self.lib_dir, 'MAPlms/Mlik_%s_it%s.npy' % ('po', i))
            if os.path.exists(fname):
                print("rank %s loading " % pbs.rank + fname)
                return np.load(fname)
        if self.soltn0 is not None: return np.load(self.soltn0)[:self.opfilt.TEBlen(self.type)]
        return np.zeros((self.opfilt.TEBlen(self.type), self.cov.lib_skyalm.alm_size), dtype=complex)

    def _cache_tebwf(self, TEBMAP, it):
        fname = os.path.join(self.lib_dir,  'MAPlms/Mlik_%s_it%s.npy' % ('po', it))
        print("rank %s caching " % pbs.rank + fname)
        np.save(fname, TEBMAP)

    def get_gradPpri(self, it, cache_only=False):
        """Builds prior gradient at iteration *it*

        """
        assert self.PBSRANK == 0, 'NO MPI method!'
        assert it > 0, it
        fname = os.path.join(self.lib_dir, 'qlm_grad%spri_it%03d.npy' % ('po', it - 1))
        if os.path.exists(fname):
            return None if cache_only else self.load_qlm(fname)
        assert self.is_previous_iter_done(it)
        POlm = self.get_POlm(it - 1)
        self.lib_qlm.almxfl(POlm, [cl_inverse(self.cl_pp), cl_inverse(self.cl_oo)], inplace=True)
        self.cache_qlm(fname, POlm, pbs_rank=0) # first p, then o.
        return None if cache_only else self.load_qlm(fname)

    def _mlik2rest_tqumlik(self, TQUMlik, it):
        """Produces B^t Ni (data - B D Mlik) in TQU space, that is fed into the qlm estimator.

        """
        f_id = ffs_deflect.ffs_id_displacement(self.cov.lib_skyalm.shape, self.cov.lib_skyalm.lsides)
        self.cov.set_ffi(self._load_f(it - 1), self._load_finv(it - 1))
        temp = ffs_specmat.TQU2TEBlms(self.type, self.cov.lib_skyalm, TQUMlik)
        maps = self.get_datmaps() - self.cov.apply_Rs(self.type, temp)
        self.cov.apply_maps(self.type, maps, inplace=True)
        self.cov.set_ffi(f_id, f_id)
        temp = self.cov.apply_Rts(self.type, maps)
        return ffs_specmat.TEB2TQUlms(self.type, self.cov.lib_skyalm, temp)

    def calc_gradplikpdet(self, it):
        """Calculates the likelihood gradient (quadratic and mean-field parts)

        """
        assert 0, 'subclass this'

    def load_graddet(self, it):
        """Loads mean-field gradient at iteration *it*

            Gradient must have already been calculated

        """
        fname_detterm = os.path.join(self.lib_dir, 'qlm_grad%sdet_it%03d.npy' % ('po', it))
        assert os.path.exists(fname_detterm), fname_detterm
        return self.load_qlm(fname_detterm)

    def load_gradpri(self, it):
        """Loads prior gradient at iteration *it*

            Gradient must have already been calculated

        """
        fname_prior = os.path.join(self.lib_dir, 'qlm_grad%spri_it%03d.npy' % ('po', it))
        assert os.path.exists(fname_prior), fname_prior
        return self.load_qlm(fname_prior)

    def load_gradquad(self, it):
        """Loads likelihood quadratic piece gradient at iteration *it*

            Gradient must have already been calculated

        """
        fname_likterm = os.path.join(self.lib_dir, 'qlm_grad%slik_it%03d.npy' % ('po', it))
        assert os.path.exists(fname_likterm), fname_likterm
        return self.load_qlm(fname_likterm)

    def load_total_grad(self, it):
        """Load the total gradient at iteration *it*.

            All gradients must have already been calculated.

         """
        return self.load_gradpri(it) + self.load_gradquad(it) + self.load_graddet(it)

    def _calc_norm(self, qlm):
        return np.sqrt(np.sum(self.lib_qlm.alm2rlm(qlm) ** 2))



    def get_Hessian(self, it):
        """Build the L-BFGS Hessian at iteration *it*


        """
        # Zeroth order inverse Hessian :
        H0p, H0o = self.get_H0()
        apply_H0k = lambda rlm, k: \
            self.lib_qlm.alm2rlm(self.lib_qlm.almxfl(self.lib_qlm.rlm2alm(rlm), [H0p, H0o]))
        apply_B0k = lambda rlm, k: \
            self.lib_qlm.alm2rlm(self.lib_qlm.almxfl(self.lib_qlm.rlm2alm(rlm), [cl_inverse(H0p), cl_inverse(H0o)]))
        BFGS_H = bfgs.BFGS_Hessian(os.path.join(self.lib_dir,  'Hessian'), apply_H0k, {}, {}, L=self.NR_method,
                                             verbose=self.verbose,apply_B0k=apply_B0k)
        # Adding the required y and s vectors :
        for k in range(np.max([0, it - BFGS_H.L]), it):
            BFGS_H.add_ys(os.path.join(self.lib_dir,  'Hessian', 'rlm_yn_%s_%s.npy' % (k, 'po')),
                          os.path.join(self.lib_dir,  'Hessian', 'rlm_sn_%s_%s.npy' % (k, 'po')), k)
        return BFGS_H

    def build_incr(self, it, gradn):
        """Search direction

            BGFS method with 'self.NR method' BFGS updates to the Hessian.
            Initial Hessian are built from N0s.
            It must be rank 0 here.

            Args:
                it: current iteration level. Will produce the increment to phi_{k-1}, from gradient est. g_{k-1}
                      phi_{k_1} + output = phi_k
                gradn: current estimate of the gradient (alm array)

            Returns:
                 increment for next iteration (alm array)

        """
        assert self.PBSRANK == 0, 'single MPI process method !'
        assert it > 0, it
        assert gradn.size == 2 * self.lib_qlm.alm_size, (gradn.size / (2 * self.lib_qlm.alm_size))
        k = it - 2
        yk_fname = os.path.join(self.lib_dir, 'Hessian', 'rlm_yn_%s_%s.npy' % (k, 'po'))
        if k >= 0 and not os.path.exists(yk_fname):  # Caching Hessian BFGS yk update :
            yk = self.lib_qlm.alm2rlm(gradn - self.load_total_grad(k))
            self.cache_rlm(yk_fname, yk)
        k = it - 1
        BFGS = self.get_Hessian(k)  # Constructing L-BFGS Hessian
        # get descent direction sk = - H_k gk : (rlm array). Will be cached directly
        sk_fname = os.path.join(self.lib_dir, 'Hessian', 'rlm_sn_%s_%s.npy' % (k, 'po'))
        step = 0.
        if not os.path.exists(sk_fname):
            print("rank %s calculating descent direction" % self.PBSRANK)
            t0 = time.time()
            incr = BFGS.get_mHkgk(self.lib_qlm.alm2rlm(gradn), k)
            #norm_inc = self._calc_norm(self.lib_qlm.rlm2alm(incr)) / self._calc_norm(self.get_POlm(0))
            step = self.newton_step_length(self.lib_qlm.rlm2alm(incr))
            assert step.size == 2 * self.lib_qlm.alm_size, (step.size / (2 * self.lib_qlm.alm_size))

            self.cache_rlm(sk_fname,self.lib_qlm.alm2rlm(step))
            prt_time(time.time() - t0, label=' Exec. time for descent direction calculation')
        assert os.path.exists(sk_fname), sk_fname
        return self.lib_qlm.rlm2alm(self.load_rlm(sk_fname)),step

    def iterate(self, it, cache_only=False):
        """Performs an iteration

            This builds the gradients at iteration *it*, and the  potential estimate, and saves the *it* + 1 estimate.

        """
        plm_fname = os.path.join(self.lib_dir, 'polm_it%03d.npy' %it)
        if os.path.exists(plm_fname): return None if cache_only else self.load_qlm(plm_fname)

        assert self.is_previous_iter_done(it), 'previous iteration not done'
        # Calculation in // of lik and det term :
        ti = time.time()
        if self.PBSRANK == 0:  # Single processes routines :
            self._calc_ffinv(it - 1)
            self.get_gradPpri(it, cache_only=True)
        self.barrier()
        # Calculation of the likelihood term, involving the det term over MCs :
        irrelevant = self.calc_gradplikpdet(it)
        self.barrier()  # Everything should be on disk now.
        if self.PBSRANK == 0:
            incr, _ = self.build_incr(it, self.load_total_grad(it - 1))
            assert self.get_POlm(it-1).size == 2 * self.lib_qlm.alm_size
            assert incr.size == 2 * self.lib_qlm.alm_size, incr.size / (2 * self.lib_qlm.alm_size)

            self.cache_qlm(plm_fname, self.get_POlm(it - 1) + incr, pbs_rank=0)

            # Saves some info about increment norm and exec. time :
            norm_inc = self._calc_norm(incr) / self._calc_norm(self.get_POlm(0))
            norms = [self._calc_norm(self.load_gradquad(it - 1))]
            norms.append(self._calc_norm(self.load_graddet(it - 1)))
            norms.append(self._calc_norm(self.load_gradpri(it - 1)))
            norm_grad = self._calc_norm(self.load_total_grad(it - 1))
            norm_grad_0 = self._calc_norm(self.load_total_grad(0))
            for i in [0, 1, 2]: norms[i] = norms[i] / norm_grad_0

            with open(os.path.join(self.lib_dir, 'history_increment.txt'), 'a') as file:
                file.write('%03d %.1f %.6f %.6f %.6f %.6f %.6f \n'
                           % (it, time.time() - ti, norm_inc, norm_grad / norm_grad_0, norms[0], norms[1], norms[2]))
                file.close()

            if self.tidy > 2:  # Erasing dx,dy and det magn (12GB for full sky at 0.74 amin per iteration)
                f1, f2 = self._getfnames_f(it - 1)
                f3, f4 = self._getfnames_finv(it - 1)
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

    def __init__(self, lib_dir, typ, filt, dat_maps, lib_qlm, POlm0, H0s, MF_qlms, cpp_priors, **kwargs):
        super(ffs_iterator_cstMF, self).__init__(lib_dir, typ, filt, dat_maps, lib_qlm, POlm0, H0s, cpp_priors,
                                                 PBSSIZE=1, PBSRANK=0,  # so that all proc. act independently
                                                 **kwargs)
        assert MF_qlms.size == 2 * self.lib_qlm.alm_size and MF_qlms.ndim == 2
        self.MF_qlms = np.concatenate([MF_qlms[0], MF_qlms[1]])

    def calc_gradplikpdet(self, it):
        fname_likterm = os.path.join(self.lib_dir, 'qlm_grad%slik_it%03d.npy' % ('po', it - 1))
        #FIXME: no need to cache the MF at all iterations, it is constant...:
        fname_detterm = os.path.join(self.lib_dir, 'qlm_grad%sdet_it%03d.npy' % ('po', it - 1))
        assert it > 0, it
        if os.path.exists(fname_likterm) and os.path.exists(fname_detterm):
            return 0
        assert self.is_previous_iter_done(it)

        # Identical MF here
        self.cache_qlm(fname_detterm, self.load_qlm(self.MF_qlms))
        self.cov.set_ffi(self._load_f(it - 1), self._load_finv(it - 1))
        mchain = multigrid.multigrid_chain(self.opfilt, self.type, self.chain_descr, self.cov,
                                                    no_deglensing=self.nodeglensing)
        # FIXME : The solution input is not working properly sometimes. We give it up for now.
        # FIXME  don't manage to find the right d0 to input for a given sol ?!!
        soltn = self.load_soltn(it).copy() * self.soltn_cond
        self.opfilt._type = self.type
        mchain.solve(soltn, self.get_datmaps(), finiop='MLIK')
        self._cache_tebwf(soltn, it - 1)
        # soltn = self.opfilt.MLIK2BINV(soltn,self.cov,self.get_datmaps())
        # grad = - ql.get_qlms(self.type, self.cov.lib_skyalm, soltn, self.cov.cls, self.lib_qlm,
        #                     use_Pool=self.use_Pool, f=self.cov.f)[{'p': 0, 'o': 1}[key.lower()]]
        TQUMlik = self.opfilt.soltn2TQUMlik(soltn, self.cov)
        ResTQUMlik = self._mlik2rest_tqumlik(TQUMlik, it)
        grad = - ql.get_qlms_wl(self.type, self.cov.lib_skyalm, TQUMlik, ResTQUMlik, self.lib_qlm,
                                use_Pool=self.use_Pool, f=self._load_f(it - 1))
        grad = np.concatenate([grad[0], grad[1]])
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

    def __init__(self, lib_dir, typ, filt, dat_maps, lib_qlm, Plm0s, H0s, cpp_priors,
                 init_rank=pbs.rank, init_barrier=pbs.barrier, **kwargs):
        super(ffs_iterator_pertMF, self).__init__(lib_dir, typ, filt, dat_maps, lib_qlm, Plm0s, H0s, cpp_priors,
                                                  PBSSIZE=1, PBSRANK=0,  # so that all proc. act independently
                                                  **kwargs)
        #lmax_sky_ivf = filt.lib_skyalm.ellmax
        #iso_libdat = filt.lib_skyalm
        #cls_noise = {'t': (filt.Nlev_uKamin('t') / 60. / 180. * np.pi) ** 2 * np.ones(lmax_sky_ivf + 1),
        #             'q': (filt.Nlev_uKamin('q') / 60. / 180. * np.pi) ** 2 * np.ones(lmax_sky_ivf + 1),
        #             'u': (filt.Nlev_uKamin('u') / 60. / 180. * np.pi) ** 2 * np.ones(lmax_sky_ivf + 1)}
        lmax_ivf = filt.lib_datalm.ellmax
        iso_libdat = filt.lib_datalm
        cls_noise = {'t': (filt.Nlev_uKamin('t') / 60. / 180. * np.pi) ** 2 * np.ones(lmax_ivf + 1),
                     'q': (filt.Nlev_uKamin('q') / 60. / 180. * np.pi) ** 2 * np.ones(lmax_ivf + 1),
                     'u': (filt.Nlev_uKamin('u') / 60. / 180. * np.pi) ** 2 * np.ones(lmax_ivf + 1)}
        self.isocov = ffs_cov.ffs_diagcov_alm(os.path.join(lib_dir, 'isocov'),
                                              iso_libdat, filt.cls, filt.cls, filt.cl_transf, cls_noise,
                                              lib_skyalm=filt.lib_skyalm, init_rank=init_rank,
                                              init_barrier=init_barrier)

    def get_mfresp(self):
        return self.isocov.get_mfresplms(self.type, self.lib_qlm, use_cls_len=False).flatten(order='C')

    def calc_gradplikpdet(self, it):
        fname_likterm = os.path.join(self.lib_dir, 'qlm_grad%slik_it%03d.npy' % ('po', it - 1))
        fname_detterm = os.path.join(self.lib_dir, 'qlm_grad%sdet_it%03d.npy' % ('po', it - 1))
        assert it > 0, it
        if os.path.exists(fname_likterm) and os.path.exists(fname_detterm):
            return 0

        assert self.is_previous_iter_done(it)
        # Identical MF here
        self.cache_qlm(fname_detterm, self.load_qlm(self.get_mfresp() * self.get_POlm(it - 1)))
        self.cov.set_ffi(self._load_f(it - 1), self._load_finv(it - 1))
        mchain = multigrid.multigrid_chain(self.opfilt, self.type, self.chain_descr, self.cov,
                                           no_deglensing=self.nodeglensing)
        # FIXME : The solution input is not working properly sometimes. We give it up for now.
        # FIXME  don't manage to find the right d0 to input for a given sol ?!!
        soltn = self.load_soltn(it).copy() * self.soltn_cond
        self.opfilt._type = self.type
        mchain.solve(soltn, self.get_datmaps(), finiop='MLIK')
        self._cache_tebwf(soltn, it - 1)
        TQUMlik = self.opfilt.soltn2TQUMlik(soltn, self.cov)
        ResTQUMlik = self._mlik2rest_tqumlik(TQUMlik, it)
        grad = - ql.get_qlms_wl(self.type, self.cov.lib_skyalm, TQUMlik, ResTQUMlik, self.lib_qlm,
                                use_Pool=self.use_Pool, f=self._load_f(it - 1))

        self.cache_qlm(fname_likterm, grad, pbs_rank=self.PBSRANK)
        # It does not help to cache both grad_O and grad_P as they do not follow the trajectory in plm space.
        return 0