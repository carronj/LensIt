import glob
import os
import shutil
import time

import numpy as np

import lensit as fs
from lensit import pbs
from lensit.ffs_deflect import ffs_deflect
from lensit.ffs_qlms import qlms as ql
from lensit.misc.misc_utils import PartialDerivativePeriodic as PDP

_types = ['T', 'QU', 'TQU']


def cl_inverse(cl):
    clinv = np.zeros_like(cl)
    clinv[np.where(cl != 0.)] = 1. / cl[np.where(cl != 0.)]
    return clinv


def prt_time(dt, label=''):
    dh = np.floor(dt / 3600.)
    dm = np.floor(np.mod(dt, 3600.) / 60.)
    ds = np.floor(np.mod(dt, 60))
    print "\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " + label
    return


class ffs_iterator(object):
    def __init__(self, lib_dir, type, cov, dat_maps, lib_qlm, Plm0, H0, cpp_prior,
                 use_Pool_lens=0, use_Pool_inverse=0, chain_descr=None, opfilt=None, soltn0=None,
                 no_deglensing=False, NR_method=100, tidy=10, verbose=True, maxcgiter=150, PBSSIZE=None, PBSRANK=None,
                 **kwargs):
        """
        Normalisation of gradients etc are now complex-like, not real and imag.

        qlm_norm is the normalization of the qlms.

        H0 the starting Hessian estimate. (cl array, ~ 1 / N0)
        """
        assert type in _types

        self.PBSSIZE = pbs.size if PBSSIZE is None else PBSSIZE
        self.PBSRANK = pbs.rank if PBSRANK is None else PBSRANK
        assert self.PBSRANK < self.PBSSIZE, (self.PBSRANK, self.PBSSIZE)
        self.barrier = (lambda: 0) if self.PBSSIZE == 1 else pbs.barrier

        self.type = type
        self.lib_dir = lib_dir
        self.dat_maps = dat_maps

        self.chain_descr = chain_descr
        self.opfilt = opfilt
        assert self.chain_descr is not None
        assert opfilt is not None
        # lib_noise = getattr(par, 'lib_noise_%s' % type)
        # lib_cmb_unl = getattr(par, 'lib_cmb_unl_%s' % type)

        self.cl_pp = cpp_prior
        self.lib_qlm = lib_qlm

        self.lsides = cov.lib_skyalm.lsides
        assert cov.lib_skyalm.lsides == lib_qlm.lsides
        self.lmax_qlm = self.lib_qlm.ellmax
        self.NR_method = NR_method

        self.tidy = tidy
        self.maxiter = maxcgiter
        self.verbose = verbose

        self.nodeglensing = no_deglensing
        if self.verbose:
            print " I see t", cov.Nlev_uKamin('t')
            print " I see q", cov.Nlev_uKamin('q')
            print " I see u", cov.Nlev_uKamin('u')

            # Defining a trial newton step length :

        def newton_step_length(iter, norm_incr):  # FIXME
            # Just trying if half the step is better for S4 QU
            if cov.Nlev_uKamin('t') > 2.1: return 1.
            if cov.Nlev_uKamin('t') <= 2.1 :
                return 0.5
            return 0.5

        self.newton_step_length = newton_step_length
        self.soltn0 = soltn0
        # Default tolerance function(iter,key)
        # FIXME Put tolerance and maxiter in chain descrt
        # def tol_func(iter, key, **kwargs):
        #    return 1e-3

        # self.tol_func = tol_func
        f_id = fs.ffs_deflect.ffs_deflect.ffs_id_displacement(cov.lib_skyalm.shape, cov.lib_skyalm.lsides)
        if not hasattr(cov, 'f') or not hasattr(cov, 'fi'):
            self.cov = cov.turn2wlfilt(f_id, f_id)
        else:
            cov.set_ffi(f_id, f_id)
            self.cov = cov
        if self.PBSRANK == 0:
            if not os.path.exists(self.lib_dir): os.makedirs(self.lib_dir)
        pbs.barrier()

        print 'ffs iterator : This is %s trying to setup %s' % (self.PBSRANK, lib_dir)
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
            print '++ ffs_%s_iterator: Caching qlm_norms and N0s' % type, self.lib_dir

            # Caching qlm norm that we will use as zeroth order curvature : (with lensed weights)
            # Prior curvature :
            # Gaussian priors
            prior_pp = cl_inverse(self.cl_pp[0:self.lmax_qlm + 1])
            prior_pp[0] *= 0.5

            curv_pp = H0 + prior_pp  # isotropic estimate of the posterior curvature at the starting point
            self.cache_cl(self.lib_dir + '/qlm_%s_H0.dat' % ('P'), cl_inverse(curv_pp))
            print "     cached %s" % self.lib_dir + '/qlm_%s_H0.dat' % ('P')
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

        if self.PBSRANK == 0: print '++ ffs_%s masked iterator : setup OK' % type
        self.barrier()

    def get_mask(self):
        ret = np.ones(self.cov.lib_datalm.shape, dtype=float)
        ret[np.where(self.cov.ninv_rad <= 0.)] *= 0
        return ret

    def get_datmaps(self):
        return np.load(self.dat_maps) if isinstance(self.dat_maps, str) else self.dat_maps

    def cache_qlm(self, fname, alm, pbs_rank=None):
        """
        Method that caches the various qlm arrays. Used for likelihood gradients and potential estimates.
        pbs_rank set to some integer makes sure only pbs.rank is effectively caching the array.
        :param fname:
        :param alm:
        :param pbs_rank:
        :return:
        """
        if pbs_rank is not None and self.PBSRANK != pbs_rank:
            return
        else:
            assert self.load_qlm(alm).ndim == 1 and self.load_qlm(alm).size == self.lib_qlm.alm_size
            print 'rank %s caching ' % self.PBSRANK, fname
            self.lib_qlm.write_alm(fname, self.load_qlm(alm))
            return

    def load_qlm(self, fname):
        return (self.lib_qlm.read_alm(fname) if isinstance(fname, str) else fname)

    def cache_rlm(self, fname, rlm):
        """
        Caches real alm vectors (used for updates of the Hessian matrix)
        :param fname:
        :param rlm:
        :return:
        """
        assert rlm.ndim == 1 and rlm.size == 2 * self.lib_qlm.alm_size, (rlm.ndim, rlm.size)
        print 'rank %s caching ' % self.PBSRANK, fname
        np.save(fname, rlm)

    def load_rlm(self, fname):
        rlm = np.load(fname)
        assert rlm.ndim == 1 and rlm.size == 2 * self.lib_qlm.alm_size, (rlm.ndim, rlm.size)
        return rlm

    def cache_cl(self, fname, cl):
        assert cl.ndim == 1
        np.savetxt(fname, cl)

    def load_cl(self, fname):
        assert os.path.exists(fname), fname
        return np.loadtxt(fname)

    def get_H0(self, key):
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname = self.lib_dir + '/qlm_%s_H0.dat' % key.upper()
        assert os.path.exists(fname), fname
        return self.load_cl(fname)

    def is_previous_iter_done(self, iter, key):
        if iter == 0: return True
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        return os.path.exists(
            self.lib_dir + '/%s_plm_it%03d.npy' % ({'p': 'Phi', 'o': 'Om'}[key.lower()], iter - 1))

    def get_Gausssample(self, iter, key, real_space=False, verbose=False):
        """
        Produce a Gaussian random field from the approximate covariance (H, from Broyden) and mean at iteration k
        :param iter:
        :param key:
        :return:
        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        assert iter < self.how_many_iter_done(key), iter
        # FIXME : redundant freqs.
        rlm_0 = self.lib_qlm.almxfl(self.lib_qlm.rlm2alm(np.random.standard_normal(2 * self.lib_qlm.alm_size)),
                                    np.sqrt(self.get_H0(key)))

        ret = self.get_Hessian(iter, key).sample_Gaussian(iter, self.lib_qlm.alm2rlm(rlm_0))
        ret = self.lib_qlm.rlm2alm(ret)
        if real_space:
            return self.lib_qlm.alm2map(ret + self.get_Plm(iter, key))
        else:
            return ret + self.get_Plm(iter, key)

    def how_many_iter_done(self, key):
        """
        Returns the number of points already calculated. Zeroth is the qest.
        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        files = glob.glob(self.lib_dir + '/%s_plm_it*.npy' % {'p': 'Phi', 'o': 'Om'}[key.lower()])
        return len(files)

    def get_Plm(self, iter, key):
        if iter < 0:
            return np.zeros(self.lib_qlm.alm_size, dtype=complex)
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname = self.lib_dir + '/%s_plm_it%03d.npy' % ({'p': 'Phi', 'o': 'Om'}[key.lower()], iter)
        assert os.path.exists(fname), fname
        return self.load_qlm(fname)

    def get_Phimap(self, iter, key):
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        return self.lib_qlm.alm2map(self.get_Plm(iter, key))

    def getfnames_f(self, key, iter):
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_dx = self.lib_dir + '/f_%s_it%03d_dx.npy' % (key.lower(), iter)
        fname_dy = self.lib_dir + '/f_%s_it%03d_dy.npy' % (key.lower(), iter)
        return fname_dx, fname_dy

    def getfnames_finv(self, key, iter):
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_dx = self.lib_dir + '/finv_%s_it%03d_dx.npy' % (key.lower(), iter)
        fname_dy = self.lib_dir + '/finv_%s_it%03d_dy.npy' % (key.lower(), iter)
        return fname_dx, fname_dy

    def calc_ffinv(self, iter, key):
        """
        Calculate displacement at iter and its inverse. Only pbs rank 0 can do this.
        :param iter:
        :param key:
        :return:
        """
        assert self.PBSRANK == 0, 'SINGLE MPI METHOD'
        if iter < 0: return
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_dx, fname_dy = self.getfnames_f(key, iter)

        if not os.path.exists(fname_dx) or not os.path.exists(fname_dy):
            # FIXME : does this from plm
            assert self.is_previous_iter_done(iter, key)
            Phi_est_WF = self.get_Phimap(iter, key)
            assert self.cov.lib_skyalm.shape == Phi_est_WF.shape
            assert self.cov.lib_skyalm.shape == self.lib_qlm.shape
            assert self.cov.lib_skyalm.lsides == self.lib_qlm.lsides
            rmin = np.array(self.cov.lib_skyalm.lsides) / np.array(self.cov.lib_skyalm.shape)
            print 'rank %s caching displacement comp. for it. %s for key %s' % (self.PBSRANK, iter, key)
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
        lib_dir = self.lib_dir + '/f_%04d_libdir' % iter
        if not os.path.exists(lib_dir): os.makedirs(lib_dir)
        fname_invdx, fname_invdy = self.getfnames_finv(key, iter)
        if not os.path.exists(fname_invdx) or not os.path.exists(fname_invdy):
            f = self.load_f(iter, key)
            print 'rank %s inverting displacement it. %s for key %s' % (self.PBSRANK, iter, key)
            f_inv = f.get_inverse(use_Pool=self.use_Pool_inverse)
            np.save(fname_invdx, f_inv.get_dx())
            np.save(fname_invdy, f_inv.get_dy())
        lib_dir = self.lib_dir + '/finv_%04d_libdir' % iter
        if not os.path.exists(lib_dir): os.makedirs(lib_dir)
        assert os.path.exists(fname_invdx), fname_invdx
        assert os.path.exists(fname_invdy), fname_invdy
        return

    def load_f(self, iter, key):
        """
        Loads current displacement solution at iteration iter
        """
        fname_dx, fname_dy = self.getfnames_f(key, iter)
        lib_dir = self.lib_dir + '/f_%04d_libdir' % iter
        assert os.path.exists(fname_dx), fname_dx
        assert os.path.exists(fname_dx), fname_dy
        assert os.path.exists(lib_dir), lib_dir
        return ffs_deflect.ffs_displacement(fname_dx, fname_dy, self.lsides,
                                            verbose=(self.PBSRANK == 0), lib_dir=lib_dir, cache_magn=True)

    def load_finv(self, iter, key):
        """
        Loads current inverse displacement solution at iteration iter.
        """
        fname_invdx, fname_invdy = self.getfnames_finv(key, iter)
        lib_dir = self.lib_dir + '/finv_%04d_libdir' % iter
        assert os.path.exists(fname_invdx), fname_invdx
        assert os.path.exists(fname_invdx), fname_invdy
        assert os.path.exists(lib_dir), lib_dir
        return ffs_deflect.ffs_displacement(fname_invdx, fname_invdy, self.lsides,
                                            verbose=(self.PBSRANK == 0), lib_dir=lib_dir, cache_magn=True)

    def load_soltn(self, iter, key):
        """
        Load starting point for the conjugate gradient inversion, by looking for file on disk from the previous
        iteration point.
        """
        assert key.lower() in ['p', 'o']
        for i in np.arange(iter, -1, -1):
            fname = self.lib_dir + '/MAPlms/Mlik_%s_it%s.npy' % (key.lower(), i)
            if os.path.exists(fname):
                print "rank %s loading " % pbs.rank, fname
                return np.load(fname)
        if self.soltn0 is not None: return np.load(self.soltn0)[:self.opfilt.TEBlen(self.type)]
        return np.zeros((self.opfilt.TEBlen(self.type), self.cov.lib_skyalm.alm_size), dtype=complex)

    def cache_TEBmap(self, TEBMAP, iter, key):
        assert key.lower() in ['p', 'o']
        fname = self.lib_dir + '/MAPlms/Mlik_%s_it%s.npy' % (key.lower(), iter)
        print "rank %s caching " % pbs.rank, fname
        np.save(fname, TEBMAP)

    def get_gradPpri(self, iter, key, cache_only=False):
        """
        Calculates and returns the gradient from Gaussian prior with cl_pp (or cl_OO) at iteration 'iter'.
        ! Does not consider purely real frequencies.
        :param iter:
        :param key: 'p' or 'o'
        :param cache_only:
        :return:
        """
        assert self.PBSRANK == 0, 'single MPI method!'
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        assert iter > 0, iter
        fname = self.lib_dir + '/qlm_grad%spri_it%03d.npy' % (key.upper(), iter - 1)
        if os.path.exists(fname):
            return None if cache_only else self.load_qlm(fname)
        assert self.is_previous_iter_done(iter, key)
        grad = self.lib_qlm.almxfl(self.get_Plm(iter - 1, key),
                                   cl_inverse(self.cl_pp if key.lower() == 'p' else self.cl_oo))
        self.cache_qlm(fname, grad, pbs_rank=0)
        return None if cache_only else self.load_qlm(fname)

    def Mlik2ResTQUMlik(self, TQUMlik, iter, key):
        """
        Produces B^t Ni (data - B D Mlik) in TQU space,
        that is fed into the qlm estimator.
        """
        Ret = np.empty_like(TQUMlik)
        f_id = fs.ffs_deflect.ffs_deflect.ffs_id_displacement(self.cov.lib_skyalm.shape, self.cov.lib_skyalm.lsides)

        for i, f in enumerate(self.type):
            self.cov.set_ffi(self.load_f(iter - 1, key), self.load_finv(iter - 1, key))
            _map = self.get_datmaps()[i] - self.cov.apply_R(f, TQUMlik[i])
            self.cov.apply_map(f, _map, inplace=True)
            self.cov.set_ffi(f_id, f_id)
            Ret[i] = self.cov.apply_Rt(f, _map)
        return Ret

    def calc_gradPlikPdet(self, iter, key):
        """
        Caches the det term for iter via MC sims, together with the data one, for maximal //isation.
        """
        assert 0, 'subclass this'

    def load_graddet(self, k, key):
        fname_detterm = self.lib_dir + '/qlm_grad%sdet_it%03d.npy' % (key.upper(), k)
        assert os.path.exists(fname_detterm), fname_detterm
        return self.load_qlm(fname_detterm)

    def load_gradpri(self, k, key):
        fname_prior = self.lib_dir + '/qlm_grad%spri_it%03d.npy' % (key.upper(), k)
        assert os.path.exists(fname_prior), fname_prior
        return self.load_qlm(fname_prior)

    def load_gradquad(self, k, key):
        fname_likterm = self.lib_dir + '/qlm_grad%slik_it%03d.npy' % (key.upper(), k)
        assert os.path.exists(fname_likterm), fname_likterm
        return self.load_qlm(fname_likterm)

    def load_total_grad(self, k, key):
        """
        Load the total gradient at iteration iter.
        All maps must be previously cached on disk.
        """
        return self.load_gradpri(k, key) + self.load_gradquad(k, key) + self.load_graddet(k, key)

    def calc_norm(self,qlm):
        return np.sqrt(np.sum(self.lib_qlm.alm2rlm(qlm) ** 2))

    def get_Hessian(self, k, key):
        """
        We need the inverse Hessian that will produce phi_iter. If iter == 1 this is simply the first guess.
        """
        # Zeroth order inverse Hessian :
        apply_H0k = lambda rlm, k: \
            self.lib_qlm.alm2rlm(self.lib_qlm.almxfl(self.lib_qlm.rlm2alm(rlm), self.get_H0(key)))
        apply_B0k = lambda rlm,k: \
            self.lib_qlm.alm2rlm(self.lib_qlm.almxfl(self.lib_qlm.rlm2alm(rlm), cl_inverse(self.get_H0(key))))
        BFGS_H = fs.ffs_iterators.bfgs.BFGS_Hessian(self.lib_dir + '/Hessian', apply_H0k, {}, {}, L=self.NR_method,
                                             verbose=self.verbose,apply_B0k=apply_B0k)
        # Adding the required y and s vectors :
        for _k in xrange(np.max([0, k - BFGS_H.L]), k):
            BFGS_H.add_ys(self.lib_dir + '/Hessian/rlm_yn_%s_%s.npy' % (_k, key),
                          self.lib_dir + '/Hessian/rlm_sn_%s_%s.npy' % (_k, key), _k)
        return BFGS_H

    def build_incr(self, iter, key, gradn):
        """
        Search direction :    BGFS method with 'self.NR method' BFGS updates to the Hessian.
        Initial Hessian are built from N0s.
        It must be rank 0 here.
        :param iter: current iteration level. Will produce the increment to phi_{k-1}, from gradient est. g_{k-1}
                      phi_{k_1} + output = phi_k
        :param key: 'p' or 'o'
        :param gradn: current estimate of the gradient (alm array)
        :return: increment for next iteration (alm array)
        s_k = x_k+1 - x_k = - H_k g_k
        y_k = g_k+1 - g_k
        """
        assert self.PBSRANK == 0, 'single MPI process method !'
        assert iter > 0, iter
        k = iter - 2
        yk_fname = self.lib_dir + '/Hessian/rlm_yn_%s_%s.npy' % (k, key)
        if k >= 0 and not os.path.exists(yk_fname):  # Caching Hessian BFGS yk update :
            yk = self.lib_qlm.alm2rlm(gradn - self.load_total_grad(k, key))
            self.cache_rlm(yk_fname, yk)
        k = iter - 1
        BFGS = self.get_Hessian(k, key)  # Constructing L-BFGS Hessian
        # get descent direction sk = - H_k gk : (rlm array). Will be cached directly
        sk_fname = self.lib_dir + '/Hessian/rlm_sn_%s_%s.npy' % (k, key)
        step = 0.
        if not os.path.exists(sk_fname):
            print "rank %s calculating descent direction" % self.PBSRANK
            t0 = time.time()
            incr = BFGS.get_mHkgk(self.lib_qlm.alm2rlm(gradn), k)
            norm_inc = self.calc_norm(self.lib_qlm.rlm2alm(incr)) / self.calc_norm(self.get_Plm(0, key))
            step = self.newton_step_length(iter, norm_inc)
            self.cache_rlm(sk_fname,incr * step)
            prt_time(time.time() - t0, label=' Exec. time for descent direction calculation')
        assert os.path.exists(sk_fname), sk_fname
        return self.lib_qlm.rlm2alm(self.load_rlm(sk_fname)),step

    def iterate(self, iter, key, cache_only=False, callback='default_callback'):
        """
        Performs an iteration, by collecting the gradients at level iter, and the lower level potential,
        saving then the iter + 1 potential map.
        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        plm_fname = self.lib_dir + '/%s_plm_it%03d.npy' % ({'p': 'Phi', 'o': 'Om'}[key.lower()], iter)
        if os.path.exists(plm_fname): return None if cache_only else self.load_qlm(plm_fname)

        assert self.is_previous_iter_done(iter, key), 'previous iteration not done'
        # Calculation in // of lik and det term :
        ti = time.time()
        if self.PBSRANK == 0:  # Single processes routines :
            self.calc_ffinv(iter - 1, key)
            self.get_gradPpri(iter, key, cache_only=True)
        pbs.barrier()
        # Calculation of the likelihood term, involving the det term over MCs :
        irrelevant = self.calc_gradPlikPdet(iter, key)
        pbs.barrier()  # Everything should be on disk now.
        if self.PBSRANK == 0:
            incr,steplength = self.build_incr(iter, key, self.load_total_grad(iter - 1, key))
            self.cache_qlm(plm_fname, self.get_Plm(iter - 1, key) + incr, pbs_rank=0)

            # Saves some info about increment norm and exec. time :
            norm_inc = self.calc_norm(incr) / self.calc_norm(self.get_Plm(0, key))
            norms = [self.calc_norm(self.load_gradquad(iter - 1, key))]
            norms.append(self.calc_norm(self.load_graddet(iter - 1, key)))
            norms.append(self.calc_norm(self.load_gradpri(iter - 1, key)))
            norm_grad = self.calc_norm(self.load_total_grad(iter - 1, key))
            norm_grad_0 = self.calc_norm(self.load_total_grad(0, key))
            for i in [0, 1, 2]: norms[i] = norms[i] / norm_grad_0

            with open(self.lib_dir + '/history_increment.txt', 'a') as file:
                file.write('%03d %.1f %.6f %.6f %.6f %.6f %.6f %.12f \n'
                           % (iter, time.time() - ti, norm_inc, norm_grad / norm_grad_0, norms[0], norms[1], norms[2],
                              steplength))
                file.close()

            if self.tidy > 2:  # Erasing dx,dy and det magn (12GB for full sky at 0.74 amin per iteration)
                f1, f2 = self.getfnames_f(key, iter - 1)
                f3, f4 = self.getfnames_finv(key, iter - 1)
                for _f in [f1, f2, f3, f4]:
                    if os.path.exists(_f):
                        os.remove(_f)
                        if self.verbose: print "     removed :", _f
                if os.path.exists(self.lib_dir + '/f_%04d_libdir' % (iter - 1)):
                    shutil.rmtree(self.lib_dir + '/f_%04d_libdir' % (iter - 1))
                    if self.verbose: print "Removed :", self.lib_dir + '/f_%04d_libdir' % (iter - 1)
                if os.path.exists(self.lib_dir + '/finv_%04d_libdir' % (iter - 1)):
                    shutil.rmtree(self.lib_dir + '/finv_%04d_libdir' % (iter - 1))
                    if self.verbose: print "Removed :", self.lib_dir + '/finv_%04d_libdir' % (iter - 1)

        pbs.barrier()
        return None if cache_only else self.load_qlm(plm_fname)


class ffs_iterator_cstMF(ffs_iterator):
    """
    Identical mean field as each step
    """

    def __init__(self, lib_dir, _type, cov, dat_maps, lib_qlm, Plm0, H0, MF_qlms, cpp_prior, **kwargs):
        super(ffs_iterator_cstMF, self).__init__(lib_dir, _type, cov, dat_maps, lib_qlm, Plm0, H0, cpp_prior,
                                                 PBSSIZE=1, PBSRANK=0,  # so that all proc. act independently
                                                 **kwargs)
        self.MF_qlms = MF_qlms

    def calc_gradPlikPdet(self, iter, key):
        """
        Caches the det term for iter via MC sims, together with the data one, for maximal //isation.
        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_likterm = self.lib_dir + '/qlm_grad%slik_it%03d.npy' % (key.upper(), iter - 1)
        fname_detterm = self.lib_dir + '/qlm_grad%sdet_it%03d.npy' % (key.upper(), iter - 1)
        assert iter > 0, iter
        if os.path.exists(fname_likterm) and os.path.exists(fname_detterm):
            return 0

        assert self.is_previous_iter_done(iter, key)

        # Identical MF here
        self.cache_qlm(fname_detterm, self.load_qlm(self.MF_qlms))
        self.cov.set_ffi(self.load_f(iter - 1, key), self.load_finv(iter - 1, key))
        mchain = fs.qcinv.multigrid.multigrid_chain(self.opfilt, self.type, self.chain_descr, self.cov,
                                                    no_deglensing=self.nodeglensing)
        # FIXME : The solution input is not working properly sometimes. We give it up for now.
        # FIXME  don't manage to find the right d0 to input for a given sol ?!!
        soltn = self.load_soltn(iter, key).copy() * 0.
        self.opfilt._type = self.type
        mchain.solve(soltn, self.get_datmaps(), finiop='MLIK')
        self.cache_TEBmap(soltn, iter - 1, key)
        # soltn = self.opfilt.MLIK2BINV(soltn,self.cov,self.get_datmaps())
        # grad = - ql.get_qlms(self.type, self.cov.lib_skyalm, soltn, self.cov.cls, self.lib_qlm,
        #                     use_Pool=self.use_Pool, f=self.cov.f)[{'p': 0, 'o': 1}[key.lower()]]
        TQUMlik = self.opfilt.soltn2TQUMlik(soltn, self.cov)
        ResTQUMlik = self.Mlik2ResTQUMlik(TQUMlik, iter, key)
        grad = - ql.get_qlms_wl(self.type, self.cov.lib_skyalm, TQUMlik, ResTQUMlik, self.lib_qlm,
                                use_Pool=self.use_Pool, f=self.load_f(iter - 1, key))[{'p': 0, 'o': 1}[key.lower()]]

        self.cache_qlm(fname_likterm, grad, pbs_rank=self.PBSRANK)
        # It does not help to cache both grad_O and grad_P as they do not follow the trajectory in plm space.
        return 0


class ffs_iterator_pertMF(ffs_iterator):
    """
    Mean field from theory, perturbatively
    """

    def __init__(self, lib_dir, _type, cov, dat_maps, lib_qlm, Plm0, H0, cpp_prior, **kwargs):
        super(ffs_iterator_pertMF, self).__init__(lib_dir, _type, cov, dat_maps, lib_qlm, Plm0, H0, cpp_prior,
                                                  PBSSIZE=1, PBSRANK=0,  # so that all proc. act independently
                                                  **kwargs)
        lmax_sky_ivf = cov.lib_skyalm.ellmax
        cls_noise = {'t': (cov.Nlev_uKamin('t') / 60. / 180. * np.pi) ** 2 * np.ones(lmax_sky_ivf + 1),
                     'q': (cov.Nlev_uKamin('q') / 60. / 180. * np.pi) ** 2 * np.ones(lmax_sky_ivf + 1),
                     'u': (cov.Nlev_uKamin('u') / 60. / 180. * np.pi) ** 2 * np.ones(lmax_sky_ivf + 1)}

        self.isocov = fs.ffs_covs.ffs_cov.ffs_diagcov_alm(lib_dir + '/isocov',
                                                          cov.lib_skyalm, cov.cls, cov.cls, cov.cl_transf, cls_noise,
                                                          lib_skyalm=cov.lib_skyalm)
        # FIXME : could simply cache the response...

    def get_MFresp(self, key):
        return self.isocov.get_MFresplms(self.type, self.lib_qlm, use_cls_len=False)[{'p': 0, 'o': 1}[key.lower()]]

    def calc_gradPlikPdet(self, iter, key):
        """
        Caches the det term for iter via MC sims, together with the data one, for maximal //isation.
        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_likterm = self.lib_dir + '/qlm_grad%slik_it%03d.npy' % (key.upper(), iter - 1)
        fname_detterm = self.lib_dir + '/qlm_grad%sdet_it%03d.npy' % (key.upper(), iter - 1)
        assert iter > 0, iter
        if os.path.exists(fname_likterm) and os.path.exists(fname_detterm):
            return 0

        assert self.is_previous_iter_done(iter, key)

        # Identical MF here
        self.cache_qlm(fname_detterm, self.load_qlm(self.get_MFresp(key.lower()) * self.get_Plm(iter - 1, key.lower())))
        self.cov.set_ffi(self.load_f(iter - 1, key), self.load_finv(iter - 1, key))
        mchain = fs.qcinv.multigrid.multigrid_chain(self.opfilt, self.type, self.chain_descr, self.cov,
                                                    no_deglensing=self.nodeglensing)
        # FIXME : The solution input is not working properly sometimes. We give it up for now.
        # FIXME  don't manage to find the right d0 to input for a given sol ?!!
        soltn = self.load_soltn(iter, key).copy() * 0.
        self.opfilt._type = self.type
        mchain.solve(soltn, self.get_datmaps(), finiop='MLIK')
        self.cache_TEBmap(soltn, iter - 1, key)
        # soltn = self.opfilt.MLIK2BINV(soltn,self.cov,self.get_datmaps())
        # grad = - ql.get_qlms(self.type, self.cov.lib_skyalm, soltn, self.cov.cls, self.lib_qlm,
        #                     use_Pool=self.use_Pool, f=self.cov.f)[{'p': 0, 'o': 1}[key.lower()]]
        TQUMlik = self.opfilt.soltn2TQUMlik(soltn, self.cov)
        ResTQUMlik = self.Mlik2ResTQUMlik(TQUMlik, iter, key)
        grad = - ql.get_qlms_wl(self.type, self.cov.lib_skyalm, TQUMlik, ResTQUMlik, self.lib_qlm,
                                use_Pool=self.use_Pool, f=self.load_f(iter - 1, key))[{'p': 0, 'o': 1}[key.lower()]]

        self.cache_qlm(fname_likterm, grad, pbs_rank=self.PBSRANK)
        # It does not help to cache both grad_O and grad_P as they do not follow the trajectory in plm space.
        return 0


class ffs_iterator_simMF(ffs_iterator):
    """
    Mean field calculated with simulation according to input MFkey.
    # FIXME : this requires the pbs.size to be a multiple of the maps to do
    otherwise the MPI barriers will not work (in 'degrade' libraries for
    the W.F.)
    """

    def __init__(self, lib_dir, _type, MFkey, nsims, cov, dat_maps, lib_qlm, Plm0, H0, cpp_prior, **kwargs):
        # lib_dir, type, cov, dat_maps, lib_qlm, Plm0, MF_qlms, H0, cpp_prior,
        # use_Pool_lens = 0, use_Pool_inverse = 0, chain_descr = None, opfilt = None, soltn0 = None,
        # no_deglensing = False, NR_method = 100, tidy = 10, verbose = True, maxcgiter = 150, PBSSIZE = None, PBSRANK = None
        super(ffs_iterator_simMF, self).__init__(lib_dir, _type, cov, dat_maps, lib_qlm, Plm0, H0, cpp_prior,
                                                 **kwargs)
        print '++ ffs_%s simMF iterator (PBSSIZE %s pbs.size %s) : setup OK' % (self.type, self.PBSSIZE, pbs.size)
        self.MFkey = MFkey
        self.nsims = nsims
        self.same_seeds = kwargs.pop('same_seeds', False)
        self.subtract_phi0 = kwargs.pop('subtract_phi0', True)
        self.barrier()

    def build_pha(self, iter):
        """
        Sets up sim libraries for the MF evaluation
        :param iter:
        :return:
        """
        if self.nsims == 0: return None
        phas_pix = fs.sims.ffs_phas.pix_lib_phas(
            self.lib_dir + '/%s_sky_noise_iter%s' % (self.type, iter * (not self.same_seeds)),
            len(self.type), self.cov.lib_datalm.shape, nsims_max=self.nsims)
        phas_cmb = None  # dont need it so far
        if self.PBSRANK == 0:
            for lib, lab in zip([phas_pix, phas_cmb], ['phas pix', 'phas_cmb']):
                if not lib is None and not lib.is_full():
                    print "++ run iterator regenerating %s phases mf_sims rank %s..." % (lab, self.PBSRANK)
                    for idx in np.arange(self.nsims): lib.get_sim(idx, phas_only=True)
        pbs.barrier()
        return phas_pix, phas_cmb

    def calc_gradPlikPdet(self, iter, key, callback='default_callback'):
        """
        Caches the det term for iter via MC sims, together with the data one, for maximal //isation.
        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_detterm = self.lib_dir + '/qlm_grad%sdet_it%03d.npy' % (key.upper(), iter - 1)
        fname_likterm = self.lib_dir + '/qlm_grad%slik_it%03d.npy' % (key.upper(), iter - 1)
        if os.path.exists(fname_detterm) and os.path.exists(fname_likterm):
            return 0
        assert self.is_previous_iter_done(iter, key)

        pix_pha, cmb_pha = self.build_pha(iter)
        if self.PBSRANK == 0 and not os.path.exists(self.lib_dir + '/mf_it%03d' % (iter - 1)):
            os.makedirs(self.lib_dir + '/mf_it%03d' % (iter - 1))
        pbs.barrier()

        # Caching gradients for the mc_sims_mf sims , plus the dat map.
        # The gradient of the det term is the data averaged lik term, with the opposite sign.

        jobs = []
        try:
            self.load_qlm(fname_likterm)
        except:
            jobs.append(-1)  # data map
        for idx in range(self.nsims):  # sims
            if not os.path.exists(self.lib_dir + '/mf_it%03d/g%s_%04d.npy' % (iter - 1, key.lower(), idx)):
                jobs.append(idx)
            else:
                try:  # just checking if file is OK.
                    self.load_qlm(self.lib_dir + '/mf_it%03d/g%s_%04d.npy' % (iter - 1, key.lower(), idx))
                except:
                    jobs.append(idx)
        self.opfilt._type = self.type
        # By setting the chain outside the main loop we avoid potential MPI barriers
        # in degrading the lib_alm libraries:
        mchain = fs.qcinv.multigrid.multigrid_chain(self.opfilt, self.type, self.chain_descr, self.cov,
                                                    no_deglensing=self.nodeglensing)
        for i in range(self.PBSRANK, len(jobs), self.PBSSIZE):
            idx = jobs[i]
            print "rank %s, doing mc det. gradients idx %s, job %s in %s at iter level %s:" \
                  % (self.PBSRANK, idx, i, len(jobs), iter)
            ti = time.time()

            if idx >= 0:  # sim
                grad_fname = self.lib_dir + '/mf_it%03d/g%s_%04d.npy' % (iter - 1, key.lower(), idx)
                self.cov.set_ffi(self.load_f(iter - 1, key), self.load_finv(iter - 1, key))
                MFest = ql.MFestimator(self.cov, self.opfilt, mchain, self.lib_qlm,
                                       pix_pha=pix_pha, cmb_pha=cmb_pha, use_Pool=self.use_Pool)
                grad = MFest.get_MFqlms(self.type, self.MFkey, idx)[{'p': 0, 'o': 1}[key.lower()]]
                if self.subtract_phi0:
                    isofilt = self.cov.turn2isofilt()
                    chain_descr_iso = fs.qcinv.chain_samples.get_isomgchain(
                        self.cov.lib_skyalm.ellmax, self.cov.lib_datalm.shape, iter_max=self.maxiter)
                    mchain_iso = fs.qcinv.multigrid.multigrid_chain(
                        self.opfilt, self.type, chain_descr_iso, isofilt, no_deglensing=self.nodeglensing)
                    MFest = ql.MFestimator(isofilt, self.opfilt, mchain_iso, self.lib_qlm,
                                           pix_pha=pix_pha, cmb_pha=cmb_pha, use_Pool=self.use_Pool)
                    grad -= MFest.get_MFqlms(self.type, self.MFkey, idx)[{'p': 0, 'o': 1}[key.lower()]]
                self.cache_qlm(grad_fname, grad, pbs_rank=self.PBSRANK)
            else:
                # This is the data.
                # FIXME : The solution input is not working properly sometimes. We give it up for now.
                # FIXME  don't manage to find the right d0 to input for a given sol ?!!
                self.cov.set_ffi(self.load_f(iter - 1, key), self.load_finv(iter - 1, key))
                soltn = self.load_soltn(iter, key).copy() * 0.
                mchain.solve(soltn, self.get_datmaps(), finiop='MLIK')
                self.cache_TEBmap(soltn, iter - 1, key)
                TQUMlik = self.opfilt.soltn2TQUMlik(soltn, self.cov)
                ResTQUMlik = self.Mlik2ResTQUMlik(TQUMlik, iter, key)
                grad = - ql.get_qlms_wl(self.type, self.cov.lib_skyalm, TQUMlik, ResTQUMlik, self.lib_qlm,
                                        use_Pool=self.use_Pool, f=self.load_f(iter - 1, key))[
                    {'p': 0, 'o': 1}[key.lower()]]
                self.cache_qlm(fname_likterm, grad, pbs_rank=self.PBSRANK)

            print "%s it. %s sim %s, rank %s cg status  " % (key.lower(), iter, idx, self.PBSRANK)
            # It does not help to cache both grad_O and grad_P as they do not follow the trajectory in plm space.
            # Saves some info about current iteration :
            if idx == -1:  # Saves some info about iteration times etc.
                with open(self.lib_dir + '/cghistories/history_dat.txt', 'a') as file:
                    file.write('%04d %.3f \n' % (iter, time.time() - ti))
                    file.close()
            else:
                with open(self.lib_dir + '/cghistories/history_sim%04d.txt' % idx, 'a') as file:
                    file.write('%04d %.3f \n' % (iter, time.time() - ti))
                    file.close()
        pbs.barrier()
        if self.PBSRANK == 0:
            # Collecting terms and caching det term.
            # We also cache arrays formed from independent sims for tests.
            print "rank 0, collecting mc det. %s gradients :" % key.lower()
            det_term = np.zeros(self.lib_qlm.alm_size, dtype=complex)
            for i in range(self.nsims):
                fname = self.lib_dir + '/mf_it%03d/g%s_%04d.npy' % (iter - 1, key.lower(), i)
                det_term = (det_term * i + self.load_qlm(fname)) / (i + 1.)
            self.cache_qlm(fname_detterm, det_term, pbs_rank=0)
            det_term *= 0.
            fname_detterm1 = fname_detterm.replace('.npy', 'MF1.npy')
            assert 'MF1' in fname_detterm1
            for i in np.arange(self.nsims)[0::2]:
                fname = self.lib_dir + '/mf_it%03d/g%s_%04d.npy' % (iter - 1, key.lower(), i)
                det_term = (det_term * i + self.load_qlm(fname)) / (i + 1.)
            self.cache_qlm(fname_detterm1, det_term, pbs_rank=0)
            det_term *= 0.
            fname_detterm2 = fname_detterm.replace('.npy', 'MF2.npy')
            assert 'MF2' in fname_detterm2
            for i in np.arange(self.nsims)[1::2]:
                fname = self.lib_dir + '/mf_it%03d/g%s_%04d.npy' % (iter - 1, key.lower(), i)
                det_term = (det_term * i + self.load_qlm(fname)) / (i + 1.)
            self.cache_qlm(fname_detterm2, det_term, pbs_rank=0)

            # Erase some temp files if requested to do so :
            if self.tidy > 1:
                # We erase as well the gradient determinant term that were stored on disk :
                files_to_remove = \
                    [self.lib_dir + '/mf_it%03d/g%s_%04d.npy' % (iter - 1, key.lower(), i) for i in range(self.nsims)]
                print 'rank %s removing %s maps in ' % (
                    self.PBSRANK, len(files_to_remove)), self.lib_dir + '/mf_it%03d/' % (iter - 1)
                for file in files_to_remove: os.remove(file)
        pbs.barrier()
