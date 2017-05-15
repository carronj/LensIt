import glob
import imp
import os
import shutil
import time

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import lensit as fs
from lensit import pbs
from lensit.ffs_covs import ell_mat
from lensit.ffs_deflect import ffs_deflect, ffs_pool
from lensit.misc.misc_utils import PartialDerivativePeriodic as PDP, check_attributes, enumerate_progress

_types = ['T', 'QU', 'TQU']


def cl_inverse(cl):
    clinv = np.zeros_like(cl)
    clinv[np.where(cl != 0.)] = 1. / cl[np.where(cl != 0.)]
    return clinv


class _iterator(object):
    def __init__(self, lib_dir, parfile, type,
                 pyFFTW_nthreads=4, use_Pool_lens=0, use_Pool_inverse=0, NR_method=100, tidy=10,
                 weights='len', Hess_weights='unl', maxcgiter=150, get_dat_map=None, get_input_pmap=None,
                 cond='3', PBSSIZE=None, PBSRANK=None):
        # FIXME : Switch normalization to complex norm not real comp. norm.
        """
        
        :param lib_dir:
        :param par_file:
        :param dat_map:
        :param tlm_filt_func: filter function for the input maps. tlm_filt_func(ell) must be boolean and True only for modes
        used in the analysis.
        :param qlm_filt_func:
        :param mc_sims_mf: Array of sim indices to use for the calculation of the determinant term.
        Defaults to the full set of sim.
        :param use_Pool: Set this to use hyper threading for lensing operations and inverse displacement operations.
        :param cache_ulms: Array of iteration idcs for which ulm arrays (sol. to cg. inversion) will be stored.
        :param NR_method : Does NR_method BFGS updates to the Hessian matrix, starting from the Fisher matrix with
        lensed weights.
        :param weights : quadratic estimator weights
        :param Hess_weights : weights for the zeroth order Hessian calculation.
        :param tidy :   if > 0 temp maps to lens that were stored on disk will be erased
                        if > 1 gradient terms stored on disk for the MC det terms will be erased as well
        :param maxcgiter : maximal number of iterations in the cg inversions.        :return:
        """

        assert os.path.exists(parfile)
        assert type in _types
        assert weights in ['unl', 'len'], weights

        self.PBSSIZE = pbs.size if PBSSIZE is None else PBSSIZE
        self.PBSRANK = pbs.rank if PBSRANK is None else PBSRANK
        assert self.PBSRANK < self.PBSSIZE, (self.PBSRANK, self.PBSSIZE)
        self.barrier = (lambda: 0) if self.PBSSIZE == 1 else pbs.barrier

        fs.ffs_covs.ffs_cov._timed = (pbs.rank == pbs.size - 1)

        self.type = type
        self.lib_dir = lib_dir
        self.timer = fs.misc.misc_utils.timer(True, prefix=__name__)

        par = imp.load_source('parfile', parfile)
        check_attributes(par, ['isocov', 'lencov'])

        self.isocov = par.isocov
        self.lencov = par.lencov

        self.get_datmap = get_dat_map or getattr(par, 'get_dat_%s_map' % type)
        self.get_inputpmap = get_input_pmap or getattr(par, 'get_input_pmap')

        self.cl_pp = self.lencov.cls_unl['pp'][:]
        self.cl_OO = self.cl_pp * 0.1 if not hasattr(self.lencov.cls_unl, 'OO')  else self.lencov.cls_unl['OO']
        assert np.all(self.cl_pp >= 0.) and np.all(self.cl_OO >= 0.)

        self.lib_qlm = ell_mat.ffs_alm_pyFFTW(self.lencov.lib_skyalm.ell_mat,
                                              filt_func=lambda ell: (ell <= np.max(np.where(self.cl_pp > 0))) & (
                                                  ell > 0), num_threads=pyFFTW_nthreads)
        # self.lib_qlm = ell_mat.ffs_alm(self.lencov.lib_skyalm.ell_mat,
        #            filt_func= lambda ell: (ell <= np.max(np.where(self.cl_pp > 0))) & (ell > 0))

        self.lmax_qlm = self.lib_qlm.ellmax
        self.alms_shape = (len(type), self.isocov.lib_datalm.alm_size)
        self.qlms_shape = (len(type), self.lib_qlm.alm_size)

        self.NR_method = NR_method
        self.tidy = tidy
        self.maxiter = maxcgiter
        self.cg_cond = str(cond)  # conditioner for cg inversion

        # Defining a trial newton step length :
        if not hasattr(par, 'newton_step_length'):
            def newton_step_length(iter):
                # Just trying if half the step is better for S4 QU
                if ('S4' in parfile or 'ideal' in parfile) and 'QU' in self.type and iter >= 1:
                    return 0.5
                elif par.sN_uKamin <= 1.5 and iter >= 1 and par.sN_uKamin > 0.4:
                    return 0.5
                elif iter >= 1 and par.sN_uKamin <= 0.4:
                    return 0.3
                else:
                    return 1.
        else:
            newton_step_length = par.newton_step_length
        self.newton_step_length = newton_step_length

        # Default tolerance function(iter,key)
        if not hasattr(par, 'tol_func'):
            def tol_func(iter, key, **kwargs):
                def calc_norm(qlm):
                    return np.sqrt(np.sum(self.lib_qlm.alm2rlm(qlm) ** 2))

                norm_grad = calc_norm(self.load_total_grad(iter - 1, key))
                norm_grad_0 = calc_norm(self.load_total_grad(0, key))
                tol = np.max([norm_grad / norm_grad_0 * 0.001, 1e-3])
                return np.min([tol, 0.1])
        else:
            tol_func = par.tol_func

        self.tol_func = tol_func

        print '++ rank (PBSRANK %s pbs.rank %s) trying to setup ffs_%s iterator (PBSSIZE %s pbs.size %s) ' \
              % (self.PBSRANK, pbs.rank, self.type, self.PBSSIZE, pbs.size)
        # Lensed covariance matrix library :

        self.use_Pool = use_Pool_lens
        self.use_Pool_inverse = use_Pool_inverse
        self.Hessweights = Hess_weights
        self.qestweights = weights
        # Array of indices stating for which iteration we will store the ulm arrays :
        # (caches only if the convergence was not immediate).
        # Does not really help to cache ulms if anyways the seeds at each iter are different.

        # dat map. A path or a memmap array is expected here.
        ffs_pool.verbose = False

        if self.PBSRANK == 0:
            if not os.path.exists(self.lib_dir): os.makedirs(self.lib_dir)
            if not os.path.exists(self.lib_dir + '/cghistories'): os.makedirs(self.lib_dir + '/cghistories')
            if not os.path.exists(self.lib_dir + '/Hessian'): os.makedirs(self.lib_dir + '/Hessian')
            if not os.path.exists(self.lib_dir + '/cls'): os.makedirs(self.lib_dir + '/cls')
            if not os.path.exists(self.lib_dir + '/cls/residuals'): os.makedirs(self.lib_dir + '/cls/residuals')
            if not os.path.exists(self.lib_dir + '/cls/cross2input'): os.makedirs(self.lib_dir + '/cls/cross2input')
            if not os.path.exists(self.lib_dir + '/figs'): os.makedirs(self.lib_dir + '/figs')
            if not os.path.exists(self.lib_dir + '/input_plm.npy'):
                np.save(self.lib_dir + '/input_plm.npy', self.lib_qlm.map2alm(self.get_inputpmap()))
                print "Cached ", self.lib_dir + '/input_plm.npy'
            if not os.path.exists(self.lib_dir + '/cls/input_plmCls.dat'):
                _cl = self.lib_qlm.alm2cl(np.load(self.lib_dir + '/input_plm.npy'))
                np.savetxt(self.lib_dir + '/cls/input_plmCls.dat', _cl);
                del _cl
                print "Cached ", self.lib_dir + '/cls/input_plmCls.dat'
            if not os.path.exists(self.lib_dir + '/dat_alms.npy'):
                dat_alm = np.empty(self.alms_shape, dtype=complex)
                # FIXME : There must be some pbs barrier in there that I can't trace :
                _datmap = self.get_datmap()
                for _i in range(len(type)): dat_alm[_i] = self.isocov.lib_datalm.map2alm(_datmap[_i])
                self.cache_alms(self.lib_dir + '/dat_alms.npy', dat_alm)
                del dat_alm, _datmap
            if not os.path.exists(self.lib_dir + '/cls/input_datCls.dat'):
                if self.type == 'TQU':
                    T, E, B = self.isocov.lib_datalm.TQUlms2TEBalms(self.load_datalms())
                    cls = np.zeros((4, self.isocov.lib_datalm.ellmax + 1))
                    cls[0] = self.isocov.lib_datalm.alm2cl(T)
                    cls[1] = self.isocov.lib_datalm.alm2cl(T, alm2=E)
                    cls[2] = self.isocov.lib_datalm.alm2cl(E)
                    cls[3] = self.isocov.lib_datalm.alm2cl(B)
                    header = 'dat cl TT TE EE BB'
                    fmt = ['%.8e'] * 4
                elif self.type == 'QU':
                    E, B = self.isocov.lib_datalm.QUlms2EBalms(self.load_datalms())
                    cls = np.zeros((2, self.isocov.lib_datalm.ellmax + 1))
                    cls[0] = self.isocov.lib_datalm.alm2cl(E)
                    cls[1] = self.isocov.lib_datalm.alm2cl(B)
                    header = 'dat cls EE BB'
                    fmt = ['%.8e'] * 2
                elif self.type == 'T':
                    cls = np.zeros((1, self.isocov.lib_datalm.ellmax + 1))
                    cls[0] = self.isocov.lib_datalm.alm2cl(self.load_datalms()[0])
                    header = 'dat cl TT'
                    fmt = ['%.8e'] * 1
                else:
                    cls = 0
                    fmt = 0
                    header = 0
                    assert 0
                np.savetxt(self.lib_dir + '/cls/input_datCls.dat', cls.transpose(), fmt=fmt, header=header)
                del cls
                print 'Cached  :', self.lib_dir + '/cls/input_datCls.dat'
        self.barrier()

        if self.PBSRANK == 0 and \
                (not os.path.exists(self.lib_dir + '/qlm_%s_norm.dat' % 'P')
                 or not os.path.exists(self.lib_dir + '/qlm_%s_norm.dat' % 'O')):
            print '++ ffs_%s_iterator: Caching qlm_norms and N0s' % type, self.lib_dir

            # Caching qlm norm that we will use as zeroth order curvature : (with lensed weights)
            # Prior curvature :
            # Gaussian priors
            # FIXME :
            ells = self.lib_qlm.ell_mat.get_unique_ells()
            ells = ells[self.lib_qlm.filt_func(ells)]
            prior_pp = np.zeros(self.lmax_qlm + 1)
            prior_pp[ells] = 2 * cl_inverse(self.cl_pp[0:self.lmax_qlm + 1])[ells]
            prior_pp[0] *= 0.5
            prior_OO = np.zeros(self.lmax_qlm + 1)
            prior_OO[ells] = 2 * cl_inverse(self.cl_OO[0:self.lmax_qlm + 1])[ells]
            prior_OO[0] *= 0.5

            self.cache_cl(self.lib_dir + '/CurvPrior_pp.dat',
                          prior_pp * self.lib_qlm.filt_func(np.arange(len(prior_pp))))
            self.cache_cl(self.lib_dir + '/CurvPrior_OO.dat',
                          prior_OO * self.lib_qlm.filt_func(np.arange(len(prior_pp))))
            print "     cached", self.lib_dir + '/CurvPrior_pp.dat'
            print "     cached", self.lib_dir + '/CurvPrior_OO.dat'

            N0s = self.isocov.get_N0cls(type, self.lib_qlm, use_cls_len=False)
            self.cache_cl(self.lib_dir + '/N0_unl_%s.dat' % 'P', N0s[0])
            self.cache_cl(self.lib_dir + '/N0_unl_%s.dat' % 'O', N0s[1])
            print "     cached %s" % self.lib_dir + '/N0_unl_%s.dat' % 'P'
            print "            %s" % self.lib_dir + '/N0_unl_%s.dat' % 'O'

            N0s = self.isocov.get_N0cls(type, self.lib_qlm, use_cls_len=True)
            self.cache_cl(self.lib_dir + '/N0_len_%s.dat' % 'P', N0s[0])
            self.cache_cl(self.lib_dir + '/N0_len_%s.dat' % 'O', N0s[1])
            print "     cached %s" % self.lib_dir + '/N0_len_%s.dat' % 'P'
            print "            %s" % self.lib_dir + '/N0_len_%s.dat' % 'O'

            # N0 = 2 / cuvpp -> Curvpp = 2 / N0
            # Curv Gaussian prior = 2 / clinv[ftl]

            for _w, lab in zip(['unl', 'len', Hess_weights], ['unl', 'len', '']):
                curv_pp = 2. * cl_inverse(self.load_N0('P', _w)) + self.load_cl(self.lib_dir + '/CurvPrior_pp.dat')
                curv_OO = 2. * cl_inverse(self.load_N0('O', _w)) + self.load_cl(self.lib_dir + '/CurvPrior_OO.dat')
                self.cache_cl(self.lib_dir + '/qlm_%s_%snorm.dat' % ('P', lab),
                              cl_inverse(curv_pp * self.lib_qlm.filt_func(np.arange(len(curv_pp)))))
                self.cache_cl(self.lib_dir + '/qlm_%s_%snorm.dat' % ('O', lab),
                              cl_inverse(curv_OO * self.lib_qlm.filt_func(np.arange(len(curv_OO)))))
                print "     cached %s" % self.lib_dir + '/qlm_%s_%snorm.dat' % ('P', lab)
                print "            %s" % self.lib_dir + '/qlm_%s_%snorm.dat' % ('O', lab)

        # pre_calculation of quadratic estimators with rank pbs.size:
        if self.PBSRANK == 0 and \
                (not os.path.exists(self.lib_dir + '/qlm_grad%slik_it%03d.npy' % ('P', 0))
                 or not os.path.exists(self.lib_dir + '/qlm_grad%slik_it%03d.npy' % ('O', 0))):
            print '++ ffs_%s_iterator: Caching quadratic estimators ' % type, self.lib_dir
            iblms, _cgiter = self.isocov.get_iblms(type, self.load_datalms(), use_cls_len=(weights == 'len'))
            gp, gO = -self.isocov.get_qlms(type, iblms, self.lib_qlm, use_cls_len=(weights == 'len'))
            # norm is 1 / Fpp, N0 = 2 / Fpp -> norm = 0.5 * N0
            # Extra weights if Hess_weights is not the same than weights
            extra_norm_p = self.load_cl(self.lib_dir + '/qlm_%s_%snorm.dat' % ('P', weights)) * \
                           cl_inverse(self.load_cl(self.lib_dir + '/qlm_%s_%snorm.dat' % ('P', Hess_weights)))
            extra_norm_o = self.load_cl(self.lib_dir + '/qlm_%s_%snorm.dat' % ('O', weights)) * \
                           cl_inverse(self.load_cl(self.lib_dir + '/qlm_%s_%snorm.dat' % ('O', Hess_weights)))

            fname_P = self.lib_dir + '/qlm_grad%slik_it%03d.npy' % ('P', 0)
            self.cache_qlm(fname_P, gp * extra_norm_p[self.lib_qlm.reduced_ellmat()])
            fname_O = self.lib_dir + '/qlm_grad%slik_it%03d.npy' % ('O', 0)
            self.cache_qlm(fname_O, gO * extra_norm_o[self.lib_qlm.reduced_ellmat()])
            del gp, gO, iblms
        self.barrier()

        if self.PBSRANK == 0 and not os.path.exists(self.lib_dir + '/figs/check_figs.pdf'):
            import pylab as pl
            pl.figure()
            figname = self.lib_dir + '/figs/check_figs.pdf'
            pp = PdfPages(figname)
            w = lambda ell: ell ** 2 * (ell + 1.) ** 2 / 2. / np.pi * 1e7
            for _k in ['P', 'O']:  # plotting N0s
                for _w in ['unl', 'len']:
                    N0 = self.load_N0(_k, _w)
                    ell = np.where(N0 > 0)[0]
                    pl.loglog(ell, N0[ell] * w(ell), label=r'%s %s' % (_k, _w))
            pl.loglog(ell, self.cl_pp[ell] * w(ell), label=r'$C_L^{\phi}$', color='black')
            pl.ylim(1e-3, 1e2)
            pl.legend(frameon=False)
            pl.title('N0s')
            pp.savefig()
            pl.clf()
            w = lambda ell: ell ** 2 * (ell + 1.) ** 2 / 2. / np.pi * 1e7
            for _k in ['P', 'O']:
                for _w in ['unl', 'len']:  # plotting qlm norms (incl. prior)
                    norm = self.load_cl(self.lib_dir + '/qlm_%s_%snorm.dat' % ('P', _w))
                    ell = np.where(norm > 0)[0]
                    pl.loglog(ell, norm[ell] * w(ell), label=r'%s %s' % (_k, _w))
            pl.loglog(ell, self.cl_pp[ell] * w(ell), label=r'$C_L^{\phi}$', color='black')

            pl.legend(frameon=False)
            pl.title('qlm norms')
            pp.savefig()
            pl.clf()
            # plot pi and curl qest spectrum
            w = lambda ell: ell ** 2 * (ell + 1.) ** 2 / 2. / np.pi * 1e7
            for _w in ['P', 'O']:
                cl = self.lib_qlm.alm2cl(self.load_qest_nofilt(_w))
                ell = np.where(cl > 0)[0]
                pl.loglog(ell, cl[ell] * w(ell), label=_w)
                cl -= self.load_N0(_w, weights)
                pl.loglog(ell, cl[ell] * w(ell), label=_w + ' (-N0)')
            pl.loglog(ell, self.cl_pp[ell] * w(ell), label=r'$C_L^{\phi}$', color='black')
            pl.ylim(1e-4, 1e1)

            pl.legend(frameon=False)
            pl.title('qests')
            pp.savefig()
            pl.clf()
            # data :
            # building cls :
            dat_cls = np.loadtxt(self.lib_dir + '/cls/input_datCls.dat').transpose()
            w = lambda ell: ell * (ell + 1) / 2. / np.pi
            if type == 'QU':
                clee, clbb = dat_cls
                ell = np.where(clee > 0)[0]
                clee_th = self.isocov.cls_len['ee'][ell] * self.isocov.cl_transf[ell] ** 2 \
                          + 0.5 * (self.lencov.cls_noise['q'][ell] + self.lencov.cls_noise['u'][ell])
                clbb_th = self.isocov.cls_len['bb'][ell] * self.isocov.cl_transf[ell] ** 2 \
                          + 0.5 * (self.lencov.cls_noise['q'][ell] + self.lencov.cls_noise['u'][ell])
                pl.loglog(ell, clee[ell] * w(ell), label='EE', color='red')
                pl.loglog(ell, clbb[ell] * w(ell), label='BB', color='blue')
                pl.loglog(ell, clee_th * w(ell), color='red')
                pl.loglog(ell, clbb_th * w(ell), color='blue')
            elif type == 'TQU':
                cltt, clte, clee, clbb = dat_cls
                ell = np.where(cltt > 0)[0]
                clee_th = self.isocov.cls_len['ee'][ell] * self.isocov.cl_transf[ell] ** 2 \
                          + 0.5 * (self.lencov.cls_noise['q'][ell] + self.lencov.cls_noise['u'][ell])
                clbb_th = self.isocov.cls_len['bb'][ell] * self.isocov.cl_transf[ell] ** 2 \
                          + 0.5 * (self.lencov.cls_noise['q'][ell] + self.lencov.cls_noise['u'][ell])
                cltt_th = self.isocov.cls_len['tt'][ell] * self.isocov.cl_transf[ell] ** 2 \
                          + self.lencov.cls_noise['t'][ell]
                clte_th = self.isocov.cls_len['te'][ell] * self.isocov.cl_transf[ell] ** 2

                pl.loglog(ell, cltt[ell] * w(ell), label='TT', color='green')
                pl.loglog(ell, np.abs(clte)[ell] * w(ell), label='TE', color='purple')
                pl.loglog(ell, clee[ell] * w(ell), label='EE', color='red')
                pl.loglog(ell, clbb[ell] * w(ell), label='BB', color='blue')
                pl.loglog(ell, cltt_th * w(ell), color='green')
                pl.loglog(ell, np.abs(clte_th) * w(ell), color='purple')
                pl.loglog(ell, clee_th * w(ell), color='red')
                pl.loglog(ell, clbb_th * w(ell), color='blue')
            elif type == 'T':
                cltt = dat_cls
                ell = np.where(cltt > 0)[0]

                cltt_th = self.isocov.cls_len['tt'][ell] * self.isocov.cl_transf[ell] ** 2 \
                          + self.lencov.cls_noise['t'][ell]
                pl.loglog(ell, cltt[ell] * w(ell), label='TT', color='green')
                pl.loglog(ell, cltt_th * w(ell), color='green')

            pl.legend(frameon=False)
            pp.savefig()
            pl.close()
            pp.close()
            print "     saved ", figname

    def plot_residualCls(self, key):
        Ndone = self.how_many_iter_done(key)
        figname = self.lib_dir + '/figs/residual_upto%s.pdf' % Ndone
        if self.PBSRANK == 0 and Ndone > 0 and not os.path.exists(figname):
            import pylab as pl
            if not os.path.exists(self.lib_dir + '/cls/residuals'): os.makedirs(self.lib_dir + '/cls/residuals')
            N0len = self.load_N0('p', 'len')
            N0unl = self.load_N0('p', 'unl')
            ell = np.where((N0len > 0.) & (np.arange(len(N0len)) < 2 * self.isocov.lib_datalm.ellmax))[0]
            w = lambda ell: ell ** 2 * (ell + 1.) ** 2 * 1e7 / 2. / np.pi
            pl.figure()
            for _i, idx in enumerate_progress(range(Ndone), label='plotting residual Cls'):
                fname = self.lib_dir + '/cls/residuals/ResCls_%s_it%03d.dat' % (key, idx)
                if not os.path.exists(fname):
                    np.savetxt(fname, self.lib_qlm.alm2cl(self.load_inputplm() - self.get_qlm(idx, 'p')))
                pl.loglog(ell, np.loadtxt(fname)[ell] * w(ell), label='it %s' % idx)
            _clWF = self.cl_pp[ell] * cl_inverse(self.cl_pp[ell] + N0len[ell])
            pl.loglog(ell, (self.cl_pp[ell] * (1. - _clWF)) * w(ell),
                      label=r'N0 (len and unl)', color='black', linestyle='--')
            _clWF = self.cl_pp[ell] * cl_inverse(self.cl_pp[ell] + N0unl[ell])
            pl.loglog(ell, (self.cl_pp[ell] * (1. - _clWF)) * w(ell), color='black', linestyle='--')
            pl.loglog(ell, self.cl_pp[ell] * w(ell), color='black')
            pl.text(3., 2., 'Iterations %s to %s ' % (0, Ndone))
            pl.legend(frameon=False)
            pl.ylim(1e-3, 1e1)
            pl.xlim(ell[0], ell[-1])
            pl.xlabel('$L$')
            pl.ylabel(r'$L^2(L + 1)^2 C_L^{\phi\phi}$ $[\times 10^7]$')
            pl.title(r'$C_L^{\hat \phi - \phi_{\rm{in}}}$')
            pl.savefig(figname)
            print "Saved :", figname

    def plot_maps(self, key, cmap='jet'):
        Ndone = self.how_many_iter_done(key)
        figname = self.lib_dir + '/figs/mapsupto%s.pdf' % Ndone
        if not os.path.exists(figname) and self.PBSRANK == 0:
            import pylab as pl
            pp = PdfPages(figname)
            ftl = np.arange(self.lib_qlm.ellmax + 1)
            ftl[:40] *= 0
            ipnmap = self.lib_qlm.alm2map(self.lib_qlm.almxfl(self.load_inputplm(), ftl))
            vmax = np.max(np.abs(ipnmap))

            def plot_map(_iter):
                pl.clf()
                qlm = self.get_qlm(_iter, key) if _iter >= 0 else self.load_inputplm()
                pl.imshow(self.lib_qlm.alm2map(self.lib_qlm.almxfl(qlm, ftl)), cmap=cmap, vmax=vmax, vmin=-vmax)
                pp.savefig()

            plot_map(-1)
            for _i, idx in enumerate_progress(range(Ndone), label='plotting maps'):
                plot_map(idx)
            pp.close()
            print "Saved ", figname

    def plot_gradientsCls(self, key):
        Ndone = self.how_many_iter_done(key)
        figname = self.lib_dir + '/figs/gradients_upto%s.pdf' % Ndone
        if self.PBSRANK == 0 and Ndone > 0 and not os.path.exists(figname):
            import pylab as pl
            pp = PdfPages(figname)
            if not os.path.exists(self.lib_dir + '/cls/gradients'): os.makedirs(self.lib_dir + '/cls/gradients')
            w = lambda ell: ell ** 2 * (ell + 1.) ** 2 * 1e7 / 2. / np.pi
            norm = self.load_N0(key, 'len')
            pl.figure()
            for _i, idx in enumerate_progress(range(Ndone), label='plotting gradients Cls'):
                fname = self.lib_dir + '/cls/gradients/GDCls_%s_it%03d.dat' % (key, idx)
                if not os.path.exists(fname): np.savetxt(fname, self.lib_qlm.alm2cl(self.load_gradquad(idx, key)))
                ell = np.where((norm > 0.) & (np.arange(len(norm)) <= self.isocov.lib_datalm.ellmax))[0]
                pl.loglog(ell, np.loadtxt(fname)[ell] * w(ell) * norm[ell] ** 2, label=r'$g^{\rm{QD}}$')
                fname = self.lib_dir + '/cls/gradients/PriCls_%s_it%03d.dat' % (key, idx)
                if not os.path.exists(fname): np.savetxt(fname, self.lib_qlm.alm2cl(self.load_gradpri(idx, key)))
                pl.loglog(ell, np.loadtxt(fname)[ell] * w(ell) * norm[ell] ** 2, label=r'$g^{\rm{Pri}}$')
                fname = self.lib_dir + '/cls/gradients/MFCls_%s_it%03d.dat' % (key, idx)
                if not os.path.exists(fname): np.savetxt(fname, self.lib_qlm.alm2cl(self.load_graddet(idx, key)))
                pl.loglog(ell, np.loadtxt(fname)[ell] * w(ell) * norm[ell] ** 2, label=r'$g^{\rm{MF}}$')
                fname = self.lib_dir + '/cls/gradients/gTOTCls_%s_it%03d.dat' % (key, idx)
                if not os.path.exists(fname): np.savetxt(fname, self.lib_qlm.alm2cl(self.load_total_grad(idx, key)))
                pl.loglog(ell, np.loadtxt(fname)[ell] * w(ell) * norm[ell] ** 2, label=r'$g^{\rm{tot}}$')

                pl.xlabel('$L$')
                pl.ylabel(r'$L^2(L + 1)^2 C_L^{\phi\phi}$ $[\times 10^7]$')
                pl.title('Normed gradients it %s' % idx)
                pl.loglog(ell, self.cl_pp[ell] * w(ell), color='black')
                pl.legend(frameon=False)
                pl.xlim(ell[0], ell[-1])
                # pl.ylim(1e-6,1e1)
                pp.savefig()
                pl.clf()
            pl.close()
            pp.close()
            print "Saved :", figname

    def plot_plmxinputCls(self, key):
        Ndone = self.how_many_iter_done(key)
        if self.PBSRANK == 0 and Ndone > 0:
            import pylab as pl
            figname = self.lib_dir + '/figs/plmxinput_upto%s.pdf' % Ndone
            if not os.path.exists(self.lib_dir + '/cls/cross2input'): os.makedirs(self.lib_dir + '/cls/cross2input')
            w = lambda ell: ell ** 2 * (ell + 1.) ** 2 * 1e7 / 2. / np.pi
            pl.figure()
            norm = self.get_norm('p')
            ell = np.where((norm > 0.) & (np.arange(len(norm)) < 2 * self.isocov.lib_datalm.ellmax))[0]
            for _i, idx in enumerate_progress(range(Ndone), label='plotting plm x input Cls'):
                fname = self.lib_dir + '/cls/cross2input/plmxinputCls_%s_it%03d.dat' % (key, idx)
                if not os.path.exists(fname):
                    np.savetxt(fname, self.lib_qlm.alm2cl(self.get_qlm(idx, key), alm2=self.load_inputplm()))
                pl.loglog(ell, np.loadtxt(fname)[ell] * w(ell), label=r'it %s' % idx)
            pl.xlabel('$L$')
            pl.ylabel(r'$L^2(L + 1)^2 C_L^{\phi\phi}$ $[\times 10^7]$')
            pl.title('Cross-spectra to input')
            pl.loglog(ell, self.cl_pp[ell] * w(ell), color='black')
            pl.legend(frameon=False)
            pl.xlim(ell[0], ell[-1])
            pl.savefig(figname)
            pl.close()
            print "Saved :", figname

    def plot_plmautoCls(self, key):
        Ndone = self.how_many_iter_done(key)
        if self.PBSRANK == 0 and Ndone > 0:
            import pylab as pl
            figname = self.lib_dir + '/figs/plmauto_upto%s.pdf' % Ndone
            if not os.path.exists(self.lib_dir + '/cls/auto'): os.makedirs(self.lib_dir + '/cls/auto')
            w = lambda ell: ell ** 2 * (ell + 1.) ** 2 * 1e7 / 2. / np.pi
            pl.figure()
            N0 = self.load_N0(key.lower(), self.qestweights)
            clpp = self.cl_pp[:len(N0)] if key.lower() == 'p' else self.cl_OO[:len(N0)]
            ell = np.where((N0 > 0.) & (np.arange(len(N0)) < self.isocov.lib_datalm.ellmax))[0]
            for _i, idx in enumerate_progress(range(Ndone), label='plotting auto Cls'):
                fname = self.lib_dir + '/cls/auto/plmautoCls_%s_it%03d.dat' % (key, idx)
                if not os.path.exists(fname):
                    np.savetxt(fname, self.lib_qlm.alm2cl(self.get_qlm(idx, key)))
                pl.semilogx(ell, np.loadtxt(fname)[ell] * cl_inverse(N0 + clpp)[ell] - 1, label=r'it %s' % idx)
            pl.xlabel('$L$')
            pl.ylabel(r'$C_L^{\hat \phi\hat\phi} / (N_0 + C^{\phi\phi}) -1.$')
            pl.title('Auto spectra')
            pl.axhline(0., linestyle='--', color='black')
            pl.legend(frameon=False)
            pl.xlim(ell[0], ell[-1])
            pl.savefig(figname)
            pl.close()
            print "Saved :", figname

    def hashdict(self):
        # FIXME :
        return {}

    def load_datalms(self):
        return self.load_alms(self.lib_dir + '/dat_alms.npy')

    def load_inputplm(self):
        return np.load(self.lib_dir + '/input_plm.npy', mmap_mode='r')

    def load_qest_nofilt(self, key):
        norm = cl_inverse(self.load_cl(self.lib_dir + '/qlm_%s_%snorm.dat' % (key, self.qestweights))) * \
               self.load_cl(self.lib_dir + '/qlm_%s_%snorm.dat' % (key, self.Hessweights))
        norm *= self.load_N0(key, self.qestweights) * 0.5
        return self.lib_qlm.almxfl(self.load_qlm(self.lib_dir + '/qlm_grad%slik_it%03d.npy' % (key, 0)), norm)

    def load_N0(self, key, weights):
        assert key.lower() in ['p', 'o'], key
        return self.load_cl(self.lib_dir + '/N0_%s_%s.dat' % (weights, key.upper()))

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
            assert alm.ndim == 1 and alm.size == self.lib_qlm.alm_size, (alm.ndim, alm.size)
            print 'rank %s caching ' % pbs.rank, fname
            self.lib_qlm.write_alm(fname, alm)
            return

    def load_qlm(self, fname):
        return self.lib_qlm.read_alm(fname)

    def cache_rlm(self, fname, rlm):
        """
        Caches real alm vectors (used for updates of the Hessian matrix)
        :param fname:
        :param rlm:
        :return:
        """
        assert rlm.ndim == 1 and rlm.size == 2 * self.lib_qlm.alm_size, (rlm.ndim, rlm.size)
        print 'rank %s caching ' % pbs.rank, fname
        np.save(fname, rlm)

    def load_rlm(self, fname):
        rlm = np.load(fname)
        assert rlm.ndim == 1 and rlm.size == 2 * self.lib_qlm.alm_size, (rlm.ndim, rlm.size)
        return rlm

    def cache_iblms(self, fname, iblms, pbs_rank=None):
        """
        caches tlm arrays. Used for the cg solutions ulm maps.
        :param fname:
        :param alm:
        :param pbs_rank:
        :return:
        """
        if pbs_rank is not None and self.PBSRANK != pbs_rank:
            return
        else:
            print 'rank %s caching ' % pbs.rank, fname
            assert iblms.shape == self.lencov._skyalms_shape(self.type), (iblms.shape, self.alms_shape)
            np.save(fname, iblms)
            return

    def load_iblms(self, fname):
        assert os.path.exists(fname)
        ret = np.load(fname)
        assert ret.shape == self.lencov._skyalms_shape(self.type), ret.shape
        return ret

    def cache_alms(self, fname, alms, pbs_rank=None):
        """
        caches tlm arrays. Used for the cg solutions ulm maps.
        :param fname:
        :param alm:
        :param pbs_rank:
        :return:
        """
        if pbs_rank is not None and self.PBSRANK != pbs_rank:
            return
        else:
            print 'rank %s caching ' % pbs.rank, fname
            assert alms.shape == self.alms_shape, (alms.shape, self.alms_shape)
            np.save(fname, alms)
            return

    def load_alms(self, fname):
        assert os.path.exists(fname)
        ret = np.load(fname)
        assert ret.shape == self.alms_shape, ret.shape
        return ret

    def cache_cl(self, fname, cl):
        assert cl.ndim == 1
        np.savetxt(fname, cl)

    def load_cl(self, fname):
        assert os.path.exists(fname), fname
        return np.loadtxt(fname)

    def get_norm(self, key):
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname = self.lib_dir + '/qlm_%s_norm.dat' % key.upper()
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
        rlm_0 = self.lib_qlm.almxfl(self.lib_qlm.rlm2alm(np.random.standard_normal(2 * self.lib_qlm.alm_size)),
                                    np.sqrt(self.get_norm(key)))

        ret = self.get_Hessian(iter, key, verbose=verbose).sample_Gaussian(iter, self.lib_qlm.alm2rlm(rlm_0))
        ret = self.lib_qlm.rlm2alm(ret)
        if real_space:
            return self.lib_qlm.alm2map(ret + self.get_qlm(iter, key))
        else:
            return ret + self.get_qlm(iter, key)

    def how_many_iter_done(self, key):
        """
        Returns the number of points already calculated. Zeroth is the qest.
        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        files = glob.glob(self.lib_dir + '/%s_plm_it*.npy' % {'p': 'Phi', 'o': 'Om'}[key.lower()])
        return len(files)

    def get_qlm(self, iter, key):
        if iter < 0:
            return np.zeros(self.lib_qlm.alm_size, dtype=complex)
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname = self.lib_dir + '/%s_plm_it%03d.npy' % ({'p': 'Phi', 'o': 'Om'}[key.lower()], iter)
        assert os.path.exists(fname), fname
        return self.load_qlm(fname)

    def get_Phimap(self, iter, key):
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        # qlm = self.get_qlm(iter, key)
        # ret = self.lib_qlm.alm2map(qlm)
        # print "pbs %s qlm and phiap : "%pbs.rank
        # print qlm[0:30],ret[0:30,0]
        return self.lib_qlm.alm2map(self.get_qlm(iter, key))

    def load_ulm0(self, iter, key, MF_idx=-1):
        """
        Load starting point for the conjugate gradient inversion, by looking for file on disk from the previous
        iteration point.
        :param iter: iteration index.
        :return: MF_idx : index of the mc_sim if this is for the determinant term.
        """
        assert key.lower() in ['p', 'o']
        if iter <= 0:
            return None
        if MF_idx < 0:  # It is about the dat map
            for i in np.arange(iter, -1, -1):
                fname = self.lib_dir + '/ulms/ulm_%s_it%s.npy' % (key.lower(), i)
                if os.path.exists(fname):
                    print "rank %s loading " % pbs.rank, fname
                    return self.load_alms(fname)
            return None
        else:
            assert 0
            return None

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
            assert self.is_previous_iter_done(iter, key)
            Phi_est_WF = self.get_Phimap(iter, key)
            rmin = self.lencov.f.lsides / np.array(self.lencov.f.shape)
            print 'rank %s caching displacement comp. for it. %s for key %s' % (pbs.rank, iter, key)
            # print Phi_est_WF[0:100]
            if key.lower() == 'p':
                dx = PDP(Phi_est_WF, axis=1, h=rmin[1])
                dy = PDP(Phi_est_WF, axis=0, h=rmin[0])
            else:
                dx = -PDP(Phi_est_WF, axis=0, h=rmin[0])
                dy = PDP(Phi_est_WF, axis=1, h=rmin[1])
            np.save(fname_dx, dx)
            np.save(fname_dy, dy)
            del dx, dy
        lib_dirf = self.lib_dir + '/f_%04d_libdir' % iter
        if not os.path.exists(lib_dirf): os.makedirs(lib_dirf)
        lib_dirfi = self.lib_dir + '/finv_%04d_libdir' % iter
        if not os.path.exists(lib_dirfi): os.makedirs(lib_dirfi)

        fname_invdx, fname_invdy = self.getfnames_finv(key, iter)
        if not os.path.exists(fname_invdx) or not os.path.exists(fname_invdy):
            f = self.load_f(iter, key)
            print 'rank %s inverting displacement it. %s for key %s' % (pbs.rank, iter, key)
            f_inv = f.get_inverse(use_Pool=self.use_Pool_inverse)
            np.save(fname_invdx, f_inv.get_dx())
            np.save(fname_invdy, f_inv.get_dy())
        assert os.path.exists(fname_invdx), fname_invdx
        assert os.path.exists(fname_invdy), fname_invdy

    def load_f(self, iter, key):
        """
        Loads current displacement solution at iteration iter
        """
        fname_dx, fname_dy = self.getfnames_f(key, iter)
        assert os.path.exists(fname_dx), fname_dx
        assert os.path.exists(fname_dx), fname_dy
        lib_dir = self.lib_dir + '/f_%04d_libdir' % iter
        assert os.path.exists(lib_dir)
        return ffs_deflect.ffs_displacement(fname_dx, fname_dy, self.lencov.f.lsides, verbose=(self.PBSRANK == 0),
                                            lib_dir=lib_dir, cache_magn=True)

    def load_finv(self, iter, key):
        """
        Loads current inverse displacement solution at iteration iter.
        """
        fname_invdx, fname_invdy = self.getfnames_finv(key, iter)
        assert os.path.exists(fname_invdx), fname_invdx
        assert os.path.exists(fname_invdx), fname_invdy
        lib_dir = self.lib_dir + '/finv_%04d_libdir' % iter
        assert os.path.exists(lib_dir)
        return ffs_deflect.ffs_displacement(fname_invdx, fname_invdy, self.lencov.f_inv.lsides,
                                            verbose=(self.PBSRANK == 0),
                                            lib_dir=lib_dir, cache_magn=True)

    def get_gradPlik_0(self, key, cache_only=False):
        """
        Gradient at iteration 0, i.e. unnormalised quadratic estimator.
        Cache both potential and curl if not present on disk.
        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname = self.lib_dir + '/qlm_grad%slik_it%03d.npy' % (key.upper(), 0)
        assert os.path.exists(fname), fname
        return 0 if cache_only else self.load_qlm(fname)

    def get_gradPpri(self, iter, key, cache_only=False, pbsrank=0):
        """
        Calculates and returns the gradient from Gaussian prior with cl_pp (or cl_OO) at iteration 'iter'.
        ! Does not consider purely real frequencies.
        :param iter:
        :param key: 'p' or 'o'
        :param cache_only:
        :return:
        """
        assert self.PBSRANK == pbsrank, 'single MPI method!'
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname = self.lib_dir + '/qlm_grad%spri_it%03d.npy' % (key.upper(), iter)
        if os.path.exists(fname):
            return None if cache_only else self.load_qlm(fname)
        if iter == 0:
            return None if cache_only else np.zeros(self.lib_qlm.alm_size, dtype=complex)
        assert self.is_previous_iter_done(iter, key)

        cl = self.cl_pp[0:self.lmax_qlm + 1] if key.lower() == 'p' else self.cl_OO[0:self.lmax_qlm + 1]
        clinv = np.zeros(self.lmax_qlm + 1)
        clinv[np.where(cl > 0.)] = 1. / cl[np.where(cl > 0.)]

        grad = self.lib_qlm.almxfl(self.get_qlm(iter - 1, key), 2 * clinv)
        self.cache_qlm(fname, grad, pbs_rank=pbsrank)
        return None if cache_only else self.load_qlm(fname)

    def load_graddet(self, iter, key):
        if iter <= 0:
            return np.zeros(self.lib_qlm.alm_size, dtype=complex)
        fname_detterm = self.lib_dir + '/qlm_grad%sdet_it%03d.npy' % (key.upper(), iter)
        assert os.path.exists(fname_detterm), fname_detterm
        return self.load_qlm(fname_detterm)

    def load_gradpri(self, iter, key):
        if iter <= 0:
            return np.zeros(self.lib_qlm.alm_size, dtype=complex)
        fname_prior = self.lib_dir + '/qlm_grad%spri_it%03d.npy' % (key.upper(), iter)
        assert os.path.exists(fname_prior), fname_prior
        return self.load_qlm(fname_prior)

    def load_gradquad(self, iter, key):
        if iter < 0:
            return np.zeros(self.lib_qlm.alm_size, dtype=complex)
        fname_likterm = self.lib_dir + '/qlm_grad%slik_it%03d.npy' % (key.upper(), iter)
        assert os.path.exists(fname_likterm), fname_likterm
        return self.load_qlm(fname_likterm)

    def load_total_grad(self, iter, key):
        """
        Load the total gradient at iteration iter.
        All maps must be previously cached on disk.
        :param iter:
        :param key:'o' and 'p'
        :return:
        """
        ret = self.load_gradpri(iter, key) + self.load_gradquad(iter, key) + self.load_graddet(iter, key)
        return ret

    def load_mf(self, mc_sims, iter, key, incr=False):
        """
        Collects the determinant grad term from a subset of the MFsims
        :param mc_sims:
        :param iter:
        :param key:
        :return:
        """
        assert self.PBSRANK == 0, self.PBSRANK
        if iter <= 0: return np.zeros(self.lib_qlm.alm_size, dtype=complex)
        tot = 0
        det_term = np.zeros(self.lib_qlm.alm_size, dtype=complex)
        for i, idx in enumerate_progress(mc_sims, label='collecting graddets '):
            grad_fname = self.lib_dir + '/mf_it%03d/g%s_%04d.npy' % (iter, key.lower(), idx)
            if os.path.exists(grad_fname):
                det_term = (det_term * tot + self.load_qlm(grad_fname)) / (tot + 1.)
                tot += 1
            else:
                print 'rank %s couldnt find ' % pbs.rank, grad_fname
        if incr:
            Hessian = self.get_Hessian(iter, key)
            det_term = self.lib_qlm.rlm2alm(Hessian.get_mHkgk(self.lib_qlm.alm2rlm(det_term), iter))
        return det_term

    def get_Hessian(self, iter, key, verbose=True):
        # Zeroth order inverse Hessian :
        apply_H0k = lambda rlm, k: \
            self.lib_qlm.alm2rlm(self.lib_qlm.almxfl(self.lib_qlm.rlm2alm(rlm), self.get_norm(key)))
        BFGS_H = fs.ffs_iterators.bfgs.BFGS_Hessian(self.lib_dir + '/Hessian', apply_H0k, {}, {}, L=self.NR_method,
                                                    verbose=verbose)
        # Adding the required y and s vectors :
        for k in xrange(np.max([0, iter - self.NR_method]), iter):
            BFGS_H.add_ys(self.lib_dir + '/Hessian/rlm_yn_%s_%s.npy' % (k, key),
                          self.lib_dir + '/Hessian/rlm_sn_%s_%s.npy' % (k, key), k)
        return BFGS_H

    def build_incr(self, iter, key, gradn):
        """
        Search direction :    BGFS method with 'self.NR method' BFGS updates to the Hessian. Initial Hessian is the
        Fisher matrix.
        It must be rank 0 here.
        :param iter: current iteration level.
        :param key: 'p' or 'o'
        :param gradn: current estimate of the gradient (alm array)
        :return: increment for next iteration (alm array)
        s_k = x_k+1 - x_k = - H_k g_k
        y_k = g_k+1 - g_k
        """
        assert self.PBSRANK == 0, 'single MPI process method !'

        yk_fname = self.lib_dir + '/Hessian/rlm_yn_%s_%s.npy' % (iter - 1, key)
        if iter > 0 and self.NR_method > 0 and not os.path.exists(yk_fname):  # Caching Hessian BFGS yk update :
            yk = self.lib_qlm.alm2rlm(gradn - self.load_total_grad(iter - 1, key))
            self.cache_rlm(yk_fname, yk)
        BFGS = self.get_Hessian(iter, key)  # Constructing L-BFGS Hessian
        # get descent direction sk = - H_k gk : (rlm array). Will be cached directly
        sk_fname = self.lib_dir + '/Hessian/rlm_sn_%s_%s.npy' % (iter, key)
        if not os.path.exists(sk_fname):
            print "rank %s calculating descent direction" % pbs.rank
            BFGS.get_mHkgk(self.lib_qlm.alm2rlm(gradn), iter, output_fname=sk_fname)
            self.timer.checkpoint(' Descent direction calculation done')
        assert os.path.exists(sk_fname), sk_fname
        return self.lib_qlm.rlm2alm(self.load_rlm(sk_fname))

    def iterate(self, iter, key, cache_only=False, callback='default_callback'):
        """
        Performs an iteration, by collecting the gradients at level iter, and the lower level potential,
        saving then the iter + 1 potential map.
        Uses chord method (same Hessian at each iteration point).
        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        plm_fname = self.lib_dir + '/%s_plm_it%03d.npy' % ({'p': 'Phi', 'o': 'Om'}[key.lower()], iter)
        if os.path.exists(plm_fname): return None if cache_only else self.load_qlm(plm_fname)

        assert self.is_previous_iter_done(iter, key), 'previous iteration %s not done' % iter
        # Calculation in // of lik and det term :
        ti = time.time()
        # Single processes routines :
        if self.PBSRANK == 0: self.calc_ffinv(iter - 1, key)
        if self.PBSRANK == self.PBSSIZE - 1: self.get_gradPpri(iter, key, cache_only=True, pbsrank=self.PBSSIZE - 1)

        self.barrier()
        if iter > 0: self.lencov.set_ffinv(self.load_f(iter - 1, key), self.load_finv(iter - 1, key))
        # Calculation of the likelihood term, involving the det term over MCs :
        self.calc_gradPlikPdet(iter, key)
        self.barrier()  # Everything should be on disk now.
        if self.PBSRANK == 0:
            increment = self.build_incr(iter, key, self.load_total_grad(iter, key))
            self.cache_qlm(plm_fname, self.get_qlm(iter - 1, key) + self.newton_step_length(iter) * increment,
                           pbs_rank=0)

            # Saves some info about increment norm and exec. time :
            calc_norm = lambda qlm: np.sqrt(np.sum(self.lib_qlm.alm2rlm(qlm) ** 2))

            norm_inc = calc_norm(increment) / calc_norm(self.get_qlm(0, key))
            norms = [calc_norm(self.load_gradquad(iter, key))]
            norms.append(calc_norm(self.load_graddet(iter, key)))
            norms.append(calc_norm(self.load_gradpri(iter, key)))
            norm_grad = calc_norm(self.load_total_grad(iter, key))
            norm_grad_0 = calc_norm(self.load_total_grad(0, key))
            for i in [0, 1, 2]: norms[i] = norms[i] / norm_grad_0

            with open(self.lib_dir + '/history_increment.txt', 'a') as file:
                # iter, time in sec, increment norm, grad_norm / norm_0, QD norm / norm_0,Pri norm / norm_0,det norm / norm_0
                file.write('%04d %s %s %s %s %s %s %s \n' % (
                    iter, time.time() - ti, norm_inc, norm_grad / norm_grad_0, norms[0], norms[1], norms[2],
                    self.newton_step_length(iter)))
                file.close()

            if self.tidy > 2:  # Erasing dx,dy and det magn (12GB for full sky per iteration)
                f1, f2 = self.getfnames_f(key, iter - 1)
                f3, f4 = self.getfnames_finv(key, iter - 1)
                for _f in [f1, f2, f3, f4]:
                    if os.path.exists(_f):
                        os.remove(_f)
                        print "     removed :", _f
                if os.path.exists(self.lib_dir + '/f_%04d_libdir' % (iter - 1)):
                    shutil.rmtree(self.lib_dir + '/f_%04d_libdir' % (iter - 1))
                    print "Removed :", self.lib_dir + '/f_%04d_libdir' % (iter - 1)
                if os.path.exists(self.lib_dir + '/finv_%04d_libdir' % (iter - 1)):
                    shutil.rmtree(self.lib_dir + '/finv_%04d_libdir' % (iter - 1))
                    print "Removed :", self.lib_dir + '/finv_%04d_libdir' % (iter - 1)

        self.barrier()
        return None if cache_only else self.load_qlm(plm_fname)

    def calc_gradPlikPdet(self, iter, key):
        assert 0, "subclass this"


class iterator_pertMF(_iterator):
    """
    Mean field calculated with leading order perturbative approach
    """

    def __init__(self, *args, **kwargs):
        super(iterator_pertMF, self).__init__(*args, **kwargs)
        if self.PBSRANK == 0:
            if not os.path.exists(self.lib_dir + '/ulms'):
                os.makedirs(self.lib_dir + '/ulms')
            print '++ ffs_%s pertMF iterator (PBSSIZE %s pbs.size %s) : setup OK' % (self.type, self.PBSSIZE, pbs.size)

    def get_MFresplm(self, key):
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname = self.lib_dir + '/MFresplm_%s.npy' % key
        if not os.path.exists(fname):
            np.save(fname, self.isocov.get_MFresplms(self.type, self.lib_qlm, use_cls_len=False)[{'p': 0, 'x': 1}[key]])
        return np.load(fname)

    def calc_gradPlikPdet(self, iter, key):
        """
        Caches the det term for iter via MC sims, together with the data one, for maximal //isation.
        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_detterm = self.lib_dir + '/qlm_grad%sdet_it%03d.npy' % (key.upper(), iter)
        fname_likterm = self.lib_dir + '/qlm_grad%slik_it%03d.npy' % (key.upper(), iter)

        if os.path.exists(fname_detterm) and os.path.exists(fname_likterm):
            return 0
        if iter == 0:  # No det term
            return self.get_gradPlik_0(key, cache_only=True)

        assert self.is_previous_iter_done(iter, key)
        if not os.path.exists(fname_likterm):
            ti = time.time()
            tol = self.tol_func(iter, key)
            if self.PBSRANK == 0: print 'Setting up cd_solve, tol ', tol
            sim = self.load_datalms()
            ulm_fname = self.lib_dir + '/ulms/ulm_%s_it%s.npy' % (key.lower(), iter)
            grad_fname = fname_likterm

            d0 = np.sum(self.lencov.lib_datalm.alms2rlms(sim) ** 2)
            ulms, cgit_done = self.lencov.cd_solve(self.type, sim,
                                                   ulm0=self.load_ulm0(iter, key), use_Pool=self.use_Pool,
                                                   maxiter=self.maxiter, tol=tol, d0=d0, cond=self.cg_cond)

            if cgit_done > 1:
                self.cache_alms(ulm_fname, ulms)
            for _i in range(len(self.type)):
                ulms[_i] = self.lencov.lib_datalm.almxfl(ulms[_i], self.lencov.cl_transf)
            iblms = np.array([self.lencov.lib_skyalm.udgrade(self.lencov.lib_datalm, _u) for _u in ulms])
            del ulms
            grad = - self.lencov.get_qlms(self.type, iblms, self.lib_qlm, use_Pool=self.use_Pool)[
                {'p': 0, 'o': 1}[key.lower()]]
            del iblms

            print "%s it. %s sim %s, rank %s cg status, Ncgit %s  " % (key.lower(), iter, -1, pbs.rank, cgit_done)
            self.cache_qlm(grad_fname, grad, pbs_rank=self.PBSRANK)
            # It does not help to cache both grad_O and grad_P as they do not follow the trajectory in plm space.
            # Saves some info about current iteration :
            with open(self.lib_dir + '/cghistories/history_dat.txt', 'a') as file:
                file.write('%04d %04d %s %s\n' % (iter, cgit_done, time.time() - ti, tol))
                file.close()

        if self.PBSRANK == 0 and not os.path.exists(fname_detterm):
            # Collecting terms and caching det term :
            print "rank %s, building %s MF term:" % (pbs.rank, key.lower())
            det_term = self.get_qlm(iter - 1, key)
            # FIXME : factor of two because MFresplms returns complex normalized gradient
            det_term *= 2 * self.get_MFresplm(key)
            self.cache_qlm(fname_detterm, det_term, pbs_rank=0)
            # Erase some temp files if requested to do so :
        self.barrier()
        return


class iterator_simMF(_iterator):
    """
    Mean field calculated with simulations.
    Isotropic case. (zero mean field at zero)
    """

    def __init__(self, lib_dir, parfile, type, MFkey, nsims, **kwargs):
        super(iterator_simMF, self).__init__(lib_dir, parfile, type, **kwargs)
        if not os.path.exists(self.lib_dir + '/ulms') and self.PBSRANK == 0:
            os.makedirs(self.lib_dir + '/ulms')
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
        if self.nsims == 0: return None, None
        phas_dat = fs.sims.ffs_phas.ffs_lib_phas(
            self.lib_dir + '/%s_phas_dat_iter%s' % (self.type, iter * (not self.same_seeds)),
            len(self.type), self.lencov.lib_datalm, nsims_max=self.nsims)

        phas_sky = None if self.MFkey != 0 else \
            fs.sims.ffs_phas.ffs_lib_phas(
                self.lib_dir + '/%s_sky_noise_iter%s' % (self.type, iter * (not self.same_seeds)),
                len(self.type), self.lencov.lib_skyalm, nsims_max=self.nsims)
        if self.PBSRANK == 0:
            for lib, lab in zip([phas_dat, phas_sky], ['phas dat', 'phas sky']):
                if not lib is None and not lib.is_full():
                    print "++ run iterator regenerating %s phases mf_sims rank %s..." % (lab, self.PBSRANK)
                    for idx in np.arange(self.nsims):
                        lib.get_sim(idx, phas_only=True)
        self.barrier()
        return phas_dat, phas_sky

    def calc_gradPlikPdet(self, iter, key, callback='default_callback'):
        """
        Caches the det term for iter via MC sims, together with the data one, for maximal //isation.
        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fname_detterm = self.lib_dir + '/qlm_grad%sdet_it%03d.npy' % (key.upper(), iter)
        fname_likterm = self.lib_dir + '/qlm_grad%slik_it%03d.npy' % (key.upper(), iter)

        if os.path.exists(fname_detterm) and os.path.exists(fname_likterm):
            return 0
        if iter == 0:  # No det term
            return self.get_gradPlik_0(key, cache_only=True)

        assert self.is_previous_iter_done(iter, key)

        phas_dat, phas_sky = self.build_pha(iter)
        if self.PBSRANK == 0 and not os.path.exists(self.lib_dir + '/mf_it%03d' % iter):
            os.makedirs(self.lib_dir + '/mf_it%03d' % iter)
        self.barrier()

        # Caching gradients for the mc_sims_mf sims , plus the dat map.
        # The gradient of the det term is the data averaged lik term, with the opposite sign.

        # build job list, by checking if relevant maps already on disks or not :
        jobs = []
        try:
            self.load_qlm(fname_likterm)
        except:
            jobs.append(-1)  # data map
        for idx in range(self.nsims):  # sims
            if not os.path.exists(self.lib_dir + '/mf_it%03d/g%s_%04d.npy' % (iter, key.lower(), idx)):
                jobs.append(idx)
            else:
                try:  # just checking if file is OK.
                    self.load_qlm(self.lib_dir + '/mf_it%03d/g%s_%04d.npy' % (iter, key.lower(), idx))
                except:
                    jobs.append(idx)
        for i in range(self.PBSRANK, len(jobs), self.PBSSIZE):

            self.lencov.Nit = 0
            self.lencov.t0 = time.time()

            idx = jobs[i]
            print "rank %s, doing mc det. gradients idx %s, job %s in %s at iter level %s:" \
                  % (self.PBSRANK, idx, i, len(jobs), iter)
            ti = time.time()
            tol = self.tol_func(iter, key)
            if self.PBSRANK == 0: print 'Setting up cd_solve, tol ', tol

            if idx >= 0:  # sim
                grad_fname = self.lib_dir + '/mf_it%03d/g%s_%04d.npy' % (iter, key.lower(), idx)
                sgn = -1
                # _type, MFkey, xlms_sky, xlms_dat, lib_qlm, use_Pool = 0\
                pha_sky = None if phas_sky is None else phas_sky.get_sim(idx)
                grad, cgit_done = self.lencov.evalMF(self.type, self.MFkey, pha_sky, phas_dat.get_sim(idx),
                                                     self.lib_qlm,
                                                     tol=tol, maxiter=self.maxiter, cond=self.cg_cond)
                # Sign of this is pot-like, not gradient-like
                grad = -grad[{'p': 0, 'o': 1}[key.lower()]]
                if self.subtract_phi0:
                    _f = fs.ffs_deflect.ffs_deflect.ffs_id_displacement(self.lencov.sky_shape,
                                                                                self.lencov.lsides)
                    _fi = fs.ffs_deflect.ffs_deflect.ffs_id_displacement(self.lencov.sky_shape,
                                                                                 self.lencov.lsides)
                    self.lencov.set_ffinv(_f, _fi)
                    grad0, _cgit_done = self.lencov.evalMF(self.type, self.MFkey, None, phas_dat.get_sim(idx),
                                                           self.lib_qlm, tol=tol, maxiter=self.maxiter, cond='0unl')
                    grad += grad0[{'p': 0, 'o': 1}[key.lower()]]
                    del grad0

            else:  # data
                sim = self.load_datalms()
                ulm_fname = self.lib_dir + '/ulms/ulm_%s_it%s.npy' % (key.lower(), iter)
                grad_fname = fname_likterm
                sgn = 1

                d0 = np.sum(self.isocov.lib_datalm.alms2rlms(sim) ** 2)
                ulms, cgit_done = self.lencov.cd_solve(self.type, sim,
                                                       ulm0=self.load_ulm0(iter, key), use_Pool=self.use_Pool,
                                                       maxiter=self.maxiter, tol=tol, d0=d0, cond=self.cg_cond)

                if cgit_done > 1: self.cache_alms(ulm_fname, ulms)
                for _i in range(len(self.type)):
                    ulms[_i] = self.lencov.lib_datalm.almxfl(ulms[_i], self.lencov.cl_transf)
                iblms = np.array([self.lencov.lib_skyalm.udgrade(self.lencov.lib_datalm, _u) for _u in ulms])
                del ulms
                grad = - self.lencov.get_qlms(self.type, iblms, self.lib_qlm, use_Pool=self.use_Pool)[
                    {'p': 0, 'o': 1}[key.lower()]]
                del iblms

            print "%s it. %s sim %s, rank %s cg status, Ncgit %s  " % (key.lower(), iter, idx, self.PBSRANK, cgit_done)
            self.cache_qlm(grad_fname, sgn * grad, pbs_rank=self.PBSRANK)
            # It does not help to cache both grad_O and grad_P as they do not follow the trajectory in plm space.
            # Saves some info about current iteration :
            if idx == -1:  # Saves some info about iteration times etc.
                with open(self.lib_dir + '/cghistories/history_dat.txt', 'a') as file:
                    file.write('%04d %04d %s %s\n' % (iter, cgit_done, time.time() - ti, tol))
                    file.close()
            else:
                with open(self.lib_dir + '/cghistories/history_sim%04d.txt' % idx, 'a') as file:
                    file.write('%04d %04d %s %s\n' % (iter, cgit_done, time.time() - ti, tol))
                    file.close()

                    # file.write('input\n')

        self.barrier()
        if self.PBSRANK == 0:
            # Collecting terms and caching det term :
            print "rank 0, collecting mc det. %s gradients :" % key.lower()
            det_term = np.zeros(self.lib_qlm.alm_size, dtype=complex)
            for i in range(self.nsims):
                fname = self.lib_dir + '/mf_it%03d/g%s_%04d.npy' % (iter, key.lower(), i)
                det_term = (det_term * i + self.load_qlm(fname)) / (i + 1.)
            self.cache_qlm(fname_detterm, det_term, pbs_rank=0)
            # Erase some temp files if requested to do so :
            if self.tidy > 0:
                # We erase the map to lens that were stored on disk :
                files_to_remove = glob.glob(self.lencov.lib_dir + '/temp_map_to_lens_rank*.npy')
                for file in files_to_remove: os.remove(file)
                print 'rank %s removed %s maps in ' % (self.PBSRANK, len(files_to_remove)), self.lencov.lib_dir
            if self.tidy > 1:
                # We erase as well the gradient determinant term that were stored on disk :
                files_to_remove = \
                    [self.lib_dir + '/mf_it%03d/g%s_%04d.npy' % (iter, key.lower(), i) for i in range(self.nsims)]
                print 'rank %s removing %s maps in ' % (
                    self.PBSRANK, len(files_to_remove)), self.lib_dir + '/mf_it%03d/' % (iter)
                for file in files_to_remove: os.remove(file)
        self.barrier()
