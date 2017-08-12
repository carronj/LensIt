import datetime
import hashlib
import os
import pickle as pk
import shutil

import numpy as np

import lensit as fs
import lensit.pbs
import lensit.qcinv
from lensit.ffs_covs import ffs_specmat as SM
from lensit.ffs_covs.ffs_specmat import get_unlPmat_ij, get_Pmat, get_datPmat_ij, \
    TQUPmats2TEBcls, get_rootunlPmat_ij, get_unlrotPmat_ij
from lensit.misc.misc_utils import timer, cls_hash
from lensit.sims.sims_generic import hash_check

_timed = True
_types = ['T', 'QU', 'TQU']
_MFkeys = [0, 14]
_runtimebarriers = False  # Can avoid problems with different MPI processes requiring different number of iterations
_runtimerankzero = True


def xylms_to_phiOmegalm(lib_alm, Fxx, Fyy, Fxy, Fyx=None):
    # dx = phi_x + Om_y
    # dy = phi_y - Om_x
    # -> d dphi = ikx d/dx + iky d/dy
    # -> d dOm = iky d/dy - ikx d/dy

    lx = lib_alm.get_kx
    ly = lib_alm.get_ky
    if Fyx is None:
        Fpp = Fxx * lx() ** 2 + Fyy * ly() ** 2 + 2. * Fxy * lx() * ly()
        FOO = Fxx * ly() ** 2 + Fyy * lx() ** 2 - 2. * Fxy * lx() * ly()
        # FIXME:
        FpO = lx() * ly() * (Fxx - Fyy) + Fxy * (ly() ** 2 - lx() ** 2)
    else:
        Fpp = Fxx * lx() ** 2 + Fyy * ly() ** 2 + (Fxy + Fyx) * lx() * ly()
        FOO = Fxx * ly() ** 2 + Fyy * lx() ** 2 - (Fxy + Fyx) * lx() * ly()
        FpO = lx() * ly() * (Fxx - Fyy) + Fxy * (ly() ** 2 - lx() ** 2)
        print 'Fxy Fyx equal, allclose', np.all(Fxy == Fyx), np.allclose(Fxy, Fyx)

    # FIXME: is the sign of the following line correct ? (anyway result should be close to zero)
    return np.array([Fpp, FOO, FpO])


def cl_inverse(cl):
    clinv = np.zeros_like(cl)
    clinv[np.where(cl != 0.)] = 1. / cl[np.where(cl != 0.)]
    return clinv


def extend_cl(_cl, ell_max, fill_val=0.):
    cl = np.zeros(ell_max + 1)
    cl[0:min([len(_cl), ell_max + 1])] = _cl[0:min([len(_cl), ell_max + 1])]
    if min([len(_cl), ell_max + 1]) < len(cl):
        cl[min([len(_cl), ell_max + 1]): len(cl)] = fill_val
    return cl


class ffs_diagcov_alm(object):
    def __init__(self, lib_dir, lib_datalm, cls_unl, cls_len, cl_transf, cls_noise,
                 lib_skyalm=None, init_rank=lensit.pbs.rank, init_barrier=lensit.pbs.barrier):

        self.lib_datalm = lib_datalm
        self.lib_skyalm = lib_datalm.clone() if lib_skyalm is None else lib_skyalm
        self.cls_unl = cls_unl
        self.cls_len = cls_len
        self.cl_transf = cl_transf
        self.cls_noise = cls_noise

        for cl in self.cls_noise.values(): assert len(cl) > self.lib_datalm.ellmax, (len(cl), self.lib_datalm.ellmax)
        for cl in self.cls_unl.values(): assert len(cl) > self.lib_skyalm.ellmax, (len(cl), self.lib_skyalm.ellmax)
        for cl in self.cls_len.values(): assert len(cl) > self.lib_skyalm.ellmax, (len(cl), self.lib_skyalm.ellmax)
        assert len(cl_transf) > self.lib_datalm.ellmax, (len(cl_transf), self.lib_datalm.ellmax)

        self.dat_shape = self.lib_datalm.ell_mat.shape
        self.lsides = self.lib_datalm.ell_mat.lsides

        self.lib_dir = lib_dir

        if not os.path.exists(lib_dir) and init_rank == 0:
            os.makedirs(lib_dir)
        init_barrier()
        if not os.path.exists(lib_dir + '/cov_hash.pk') and init_rank == 0:
            pk.dump(self.hashdict(), open(lib_dir + '/cov_hash.pk', 'w'))
        init_barrier()
        hash_check(pk.load(open(lib_dir + '/cov_hash.pk', 'r')), self.hashdict())

        self.barrier = lensit.pbs.barrier if _runtimebarriers else lambda: -1
        self.pbsrank = 0 if _runtimerankzero else lensit.pbs.rank

    def _deg(self, skyalm):
        assert skyalm.shape == (self.lib_skyalm.alm_size,), (skyalm.shape, self.lib_skyalm.alm_size)
        if self.lib_skyalm.iseq(self.lib_datalm, allow_shape=True): return skyalm
        return self.lib_datalm.udgrade(self.lib_skyalm, skyalm)

    def _upg(self, datalm):
        assert datalm.shape == (self.lib_datalm.alm_size,), (datalm.shape, self.lib_datalm.alm_size)
        if self.lib_datalm.iseq(self.lib_skyalm, allow_shape=True): return datalm
        return self.lib_skyalm.udgrade(self.lib_datalm, datalm)

    def _2smap(self, alm):
        return self.lib_skyalm.alm2map(alm)

    def _2dmap(self, alm):
        return self.lib_datalm.alm2map(alm)

    def _datalms_shape(self, _type):
        assert _type in _types, (_type, _types)
        return (len(_type), self.lib_datalm.alm_size)

    def _skyalms_shape(self, _type):
        assert _type in _types, (_type, _types)
        return (len(_type), self.lib_skyalm.alm_size)

    def _datmaps_shape(self, _type):
        assert _type in _types, (_type, _types)
        return (len(_type), self.dat_shape[0], self.dat_shape[1])

    def hashdict(self):
        hash = {'lib_alm': self.lib_datalm.hashdict(), 'lib_skyalm': self.lib_skyalm.hashdict()}
        for key, cl in self.cls_noise.iteritems():
            hash['cls_noise ' + key] = hashlib.sha1(cl).hexdigest()
        for key, cl in self.cls_unl.iteritems():
            hash['cls_unl ' + key] = hashlib.sha1(cl).hexdigest()
        for key, cl in self.cls_len.iteritems():
            hash['cls_len ' + key] = hashlib.sha1(cl).hexdigest()
        hash['cl_transf'] = hashlib.sha1(self.cl_transf.copy()).hexdigest()
        return hash

    def _get_Nell(self, field):
        return self.cls_noise[field.lower()][0]

    def get_SN(self, _type):
        """
        Estimate of the number of independent modes with S / N > 1
        Returns sum_ell Tr (bP)_ell Cov^-1_ell (bP)_ell Cov^-1_ell
        """
        assert 0
        # FIXME

    def degrade(self, LD_shape, ellmin=None, ellmax=None, lib_dir=None, libtodegrade='sky', **kwargs):
        assert 0, 'FIXME'
        if lib_dir is None: lib_dir = self.lib_dir + '/%sdegraded%sx%s_%s' % (LD_shape[0], LD_shape[1], ellmin, ellmax)
        if libtodegrade == 'sky':
            lib_datalmLD = self.lib_datalm.degrade(LD_shape)
            lib_skyalmLD = self.lib_skyalm.degrade(LD_shape, ellmax=ellmax, ellmin=ellmin)
        else:
            lib_dir += 'dat'
            lib_datalmLD = self.lib_datalm.degrade(LD_shape, ellmax=ellmax, ellmin=ellmin)
            lib_skyalmLD = self.lib_skyalm.degrade(LD_shape)

        return ffs_diagcov_alm(lib_dir, lib_datalmLD, self.cls_unl, self.cls_len, self.cl_transf, self.cls_noise,
                               lib_skyalm=lib_skyalmLD)

    def get_Pmatinv(self, _type, i, j, use_cls_len=True):
        if i < j: return self.get_Pmatinv(_type, j, i, use_cls_len=use_cls_len)
        _str = {True: 'len', False: 'unl'}[use_cls_len]
        fname = self.lib_dir + '/%s_Pmatinv_%s_%s%s.npy' % (_type, _str, i, j)
        if not os.path.exists(fname) and self.pbsrank == 0:
            cls_cmb = self.cls_len if use_cls_len else self.cls_unl
            Pmatinv = get_Pmat(_type, self.lib_datalm, cls_cmb,
                               cl_transf=self.cl_transf, cls_noise=self.cls_noise, inverse=True)
            for _j in range(len(_type)):
                for _i in range(_j, len(_type)):
                    np.save(self.lib_dir + '/%s_Pmatinv_%s_%s%s.npy' % (_type, _str, _i, _j), Pmatinv[:, _i, _j])
                    print "     get_Pmatinv:: cached", self.lib_dir + '/%s_Pmatinv_%s_%s%s.npy' % (_type, _str, _i, _j)
        self.barrier()
        return np.load(fname)

    def get_rootPmatsky(self, _type, i, j, use_cls_len=False):
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl
        return get_rootunlPmat_ij(_type, self.lib_skyalm, cls_cmb, i, j)

    def get_delensinguncorrbias(self, lib_qlm, clpp_rec, wNoise=True, wCMB=True, recache=False, use_cls_len=True):
        """
        Calculate delensing bias given a reconstructed potential map with spectum clpp_rec (includes N0, etc),
        (** clpp_rec should not contain clpp if what you really want is the bias ** Put clpp + N0 if what you want is the
         perturbative delensing)
        assuming the noise is independent from the data maps, and the modes given by lib_qlm are reconstructed.
        Returns a (3,3,ellmax) array with bias C_ell^{ij}, i,j in T,E,B.
        The sign of the output is such that it is a pos. contrib. to C^len - C^del
        """
        # assert len(clpp_rec) > lib_qlm.ellmax,(len(clpp_rec),lib_qlm.ellmax)
        if len(clpp_rec) <= lib_qlm.ellmax: clpp_rec = extend_cl(clpp_rec, lib_qlm.ellmax)
        fname = self.lib_dir + '/TEBdelensUncorrBias_wN%s_w%sCMB%s_%s_%s.dat' \
                               % (wNoise, {True: 'len', False: 'unl'}[use_cls_len],
                                  wCMB, hashlib.sha1(clpp_rec[lib_qlm.ellmin:lib_qlm.ellmax + 1]).hexdigest(),
                                  lib_qlm.filt_hash())
        if (not os.path.exists(fname) or recache) and self.pbsrank == 0:
            def ik_q(a):
                assert a in [0, 1], a
                return lib_qlm.get_ikx() if a == 1 else lib_qlm.get_iky()

            def ik_d(a):
                assert a in [0, 1], a
                return self.lib_datalm.get_ikx() if a == 1 else self.lib_datalm.get_iky()

            retalms = np.zeros((3, 3, self.lib_datalm.alm_size), dtype=complex)
            cls_noise = {}
            for _k, _cl in self.cls_noise.iteritems():
                cls_noise[_k] = _cl / self.cl_transf[:len(_cl)] ** 2 * (wNoise)
            for _i in range(3):
                for _j in range(_i, 3):
                    if wCMB or (_i == _j):
                        _map = np.zeros(self.dat_shape, dtype=float)
                        if wCMB:
                            Pmat = get_datPmat_ij('TQU', self.lib_datalm, self.cls_len, np.ones_like(self.cl_transf),
                                                  cls_noise, _i, _j)
                        elif wNoise:
                            assert _i == _j, (_i, _j)
                            Pmat = cls_noise[{0: 't', 1: 'q', 2: 'u'}[_i]][self.lib_datalm.reduced_ellmat()]
                        else:
                            Pmat = 0
                        for a in [0, 1]:
                            for b in [0, 1][a:]:
                                facab = (2. - (a == b))
                                _phiab = lib_qlm.alm2map(clpp_rec[lib_qlm.reduced_ellmat()] * ik_q(a) * ik_q(b) * facab)
                                _map += (_phiab - _phiab[0, 0]) \
                                        * self.lib_datalm.alm2map(Pmat * ik_d(a) * ik_d(b))
                    retalms[_i, _j, :] = (self.lib_datalm.map2alm(_map))
            retalms = TQUPmats2TEBcls(self.lib_datalm, retalms) * (- 1. / np.sqrt(np.prod(self.lsides)))
            save_arr = np.zeros((6, self.lib_datalm.ellmax + 1), dtype=float)
            k = 0
            for _i in range(3):
                for _j in range(_i, 3):
                    save_arr[k, :] = retalms[_i, _j]
                    k += 1
            header = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n' + __file__
            header += "\n Delensing Uncorr. Bias. qlm ell-range (%s - %s)" % (lib_qlm.ellmin, lib_qlm.ellmax)
            header += "\n Performed with (%s %s) on fsky = %s" % (
                self.lib_datalm.shape[0], self.lib_datalm.shape[1], np.round(self.lib_datalm.fsky(), 2))
            header += "\n Positive Contr. to C(len) - C(del)."
            if wCMB: header += "\n incl. CMB "
            if wNoise: header += "\n incl. Noise "
            header += "\n TT TE TB EE EB BB"
            np.savetxt(fname, save_arr.transpose(), fmt=['%.8e'] * 6, header=header)
            print "Cached ", fname
        self.barrier()
        cls = np.loadtxt(fname).transpose()
        ret = np.zeros((3, 3, self.lib_datalm.ellmax + 1), dtype=float)
        ret[0, 0] = cls[0]
        ret[0, 1] = cls[1]
        ret[1, 0] = cls[1]
        ret[0, 2] = cls[2]
        ret[2, 0] = cls[2]
        ret[1, 1] = cls[3]
        ret[1, 2] = cls[4]
        ret[2, 1] = cls[4]
        ret[2, 2] = cls[5]
        return ret

    def get_RDdelensinguncorrbias(self, lib_qlm, clpp_rec, clsobs_deconv, clsobs_deconv2=None, recache=False):
        """
        Calculate delensing bias given a reconstructed potential map with spectum clpp_rec (includes N0, etc),
        (** clpp_rec should not contain clpp if what you really want is the bias ** Put clpp + N0 if what you want is the
         perturbative delensing)
        assuming the noise is independent from the data maps, and the modes given by lib_qlm are reconstructed.
        Returns a (3,3,ellmax) array with bias C_ell^{ij}, i,j in T,E,B.
        The sign of the output is such that it is a pos. contrib. to C^len - C^del

        putting cls_obs being cls_len + noise / transf ** 2 should give the same thing as get_delensinguncorrbias.
        cls_obs should be dict. witn tt te ee bb
        """
        if len(clpp_rec) <= lib_qlm.ellmax: clpp_rec = extend_cl(clpp_rec, lib_qlm.ellmax)
        fname = None
        if (not False or recache) and self.pbsrank == 0:
            def ik_q(a):
                assert a in [0, 1], a
                return lib_qlm.get_ikx() if a == 1 else lib_qlm.get_iky()

            def ik_d(a):
                assert a in [0, 1], a
                return self.lib_datalm.get_ikx() if a == 1 else self.lib_datalm.get_iky()

            retalms = np.zeros((3, 3, self.lib_datalm.alm_size), dtype=complex)
            for _i in range(3):
                for _j in range(_i, 3):
                    _map = np.zeros(self.dat_shape, dtype=float)
                    Pmat1 = get_unlPmat_ij('TQU', self.lib_datalm, clsobs_deconv, _i, _j)
                    Pmat2 = Pmat1 if clsobs_deconv2 is None else  get_unlPmat_ij('TQU', self.lib_datalm, clsobs_deconv2,
                                                                                 _i, _j)
                    for a in [0, 1]:
                        for b in [0, 1][a:]:
                            facab = (2. - (a == b))
                            _phiab = lib_qlm.alm2map(clpp_rec[lib_qlm.reduced_ellmat()] * ik_q(a) * ik_q(b) * facab)
                            if clsobs_deconv2 is None:
                                _map += (_phiab - _phiab[0, 0]) * self.lib_datalm.alm2map(Pmat1 * ik_d(a) * ik_d(b))
                            else:
                                _map += (_phiab) * self.lib_datalm.alm2map(Pmat1 * ik_d(a) * ik_d(b))
                                _map -= (_phiab[0, 0]) * self.lib_datalm.alm2map(Pmat2 * ik_d(a) * ik_d(b))

                    retalms[_i, _j, :] = (self.lib_datalm.map2alm(_map))
            return TQUPmats2TEBcls(self.lib_datalm, retalms) * (- 1. / np.sqrt(np.prod(self.lsides)))

    def get_delensingcorrbias(self, _type, lib_qlm, ALWFcl, CMBonly=False):
        """
        Let the unnormalized qest be written as d_a = A^{im} X^i B^{a,l m} X^l
        with X in T,Q,U beam ** deconvolved ** maps.
        -> A^{im} = b^2 Cov^{-1}_{m i}
           B^{a, lm} = [ik_a b^2 C_len  Cov^{-1}]_{m l}
        WFcl is the filter applied to the qest prior delensing the maps. (e.g. Cpp / (Cpp + N0) for WF filtering)

        See Eq. A19 in 1701.01712.

        The sign of the output is such that is a positive contr. to C^len - C^unl.
        
        For template subtraction this is slightly different, since the template is built with (E = E_dat^WF, B = 0) 
        template. In this case, the leg, with a {,b} on the spectra should be the cross-spectra
        between (T = T^dat,E^dat,B^dat) and (0,E^WF,0). The other leg is identical, since it pairs a data map and
        the quadratic estiamtor legs built with the data.
        can get dat with RDdelensingcorrbias below with custom cl 2

        !NB : norm is N0/2-like, -> AL ~ N0/2
        """

        assert _type in _types, (_type, _types)

        t = timer(_timed)
        t.checkpoint("delensing bias : Just started")
        retalms = np.zeros((3, 3, self.lib_datalm.alm_size), dtype=complex)
        cls_noise = {}

        for _k, _cl in self.cls_noise.iteritems():
            cls_noise[_k] = _cl / self.cl_transf[:len(_cl)] ** 2 * (not CMBonly)

        def get_datcl(l, j):  # beam deconvolved Cls of the TQU data maps
            # The first index is one in th qest estimator of type '_type' and
            # the second any in TQU.
            assert (l in range(len(_type))) and (j in range(3)), (l, j)
            if _type == 'QU':
                l_idx = l + 1
            elif _type == 'T':
                l_idx = 0
            else:
                assert _type == 'TQU'
                l_idx = l
            return get_datPmat_ij(
                'TQU', self.lib_datalm, self.cls_len, np.ones_like(self.cl_transf), cls_noise, l_idx, j)

        def _get_Balm(a, l, m):  # [ik_a b^2 C_len  Cov^{-1}]_{m l}
            # Here both indices should refer to the qest.
            assert a in [0, 1], a
            assert (l in range(len(_type))) and (m in range(len(_type))), (l, m)
            ik = self.lib_datalm.get_ikx if a == 1 else self.lib_datalm.get_iky
            ret = get_unlPmat_ij(_type, self.lib_datalm, self.cls_len, m, 0) \
                  * self.get_Pmatinv(_type, 0, l, use_cls_len=True)
            for _i in range(1, len(_type)):
                ret += get_unlPmat_ij(_type, self.lib_datalm, self.cls_len, m, _i) \
                       * self.get_Pmatinv(_type, _i, l, use_cls_len=True)
            return self.lib_datalm.almxfl(ret, self.cl_transf ** 2) * ik()

        def _get_Akm(l, m):  # [b^2  Cov^{-1}]_{m k}
            # Here both indices should refer to the qest.
            assert (l in range(len(_type))) and (m in range(len(_type))), (l, m)
            return self.lib_datalm.almxfl(self.get_Pmatinv(_type, m, l, use_cls_len=True), self.cl_transf ** 2)

        def get_BCamj(a, m, j):  # sum_l B^{a, l m} \hat C^{lj}
            # The first index m is one in th qest estimator of type '_type' and
            # the second any in TQU.
            assert a in [0, 1], a
            assert m in range(len(_type)), (m, _type)
            assert j in range(3), j
            ret = _get_Balm(a, 0, m) * get_datcl(0, j)
            for _i in range(1, len(_type)):
                ret += _get_Balm(a, _i, m) * get_datcl(_i, j)
            return ret

        def get_ACmj(m, j):  # sum_k A^{k m} \hat C^{kj}
            # The first index m is one in th qest estimator of type '_type' and
            # the second any in TQU.
            assert m in range(len(_type)), (m, _type)
            assert j in range(3), j
            ret = _get_Akm(0, m) * get_datcl(0, j)
            for _i in range(1, len(_type)):
                ret += _get_Akm(_i, m) * get_datcl(_i, j)
            return ret

        def ik(a, libalm=self.lib_datalm):
            assert a in [0, 1], a
            return libalm.get_ikx() if a == 1 else libalm.get_iky()

        _map = lambda _alm: self.lib_datalm.alm2map(_alm)
        for a in [0, 1]:
            for b in [0, 1]:
                t.checkpoint("    Doing axes %s %s" % (a, b))
                Hab = lib_qlm.alm2map(ALWFcl[lib_qlm.reduced_ellmat()] * ik(a, libalm=lib_qlm) * ik(b, libalm=lib_qlm))
                for _i in range(3):  # Building TQU biases, before rotation to Gradient / Curl
                    for _j in range(0, 3):
                        t.checkpoint(
                            "          Doing %s" % ({0: 'T', 1: 'Q', 2: 'U'}[_i] + {0: 'T', 1: 'Q', 2: 'U'}[_j]))
                        # Need a sum over a and m
                        for m in range(len(_type)):  # Hab(z) * [ (AC_m_dai)(-z) (BC_bmj) + (AC_mj)(-z) (BC_bmdai)]
                            Pmat = (get_ACmj(m, _i) * ik(a)).conjugate()
                            retalms[_i, _j, :] += self.lib_datalm.map2alm(Hab * _map(Pmat)) * get_BCamj(b, m, _j)
                            Pmat = (get_BCamj(b, m, _i) * ik(a)).conjugate()
                            retalms[_i, _j, :] += self.lib_datalm.map2alm(Hab * _map(Pmat)) * get_ACmj(m, _j)
        norm = 1. / np.sqrt(np.prod(self.lsides))  # ?
        # Sure that i - j x-y is the same ?
        for _i in range(3):  # Building TQU biases, before rotation to Gradient / Curl
            for _j in range(_i, 3):
                if _i == 0 and _j == 0:
                    print "Testing conjecture that in the MV case this is symmetric :"
                print _type + ' :', np.allclose(retalms[_j, _i, :].real, retalms[_i, _j, :].real)
                retalms[_i, _j, :] += retalms[_j, _i, :].conjugate()

        return TQUPmats2TEBcls(self.lib_datalm, retalms) * norm

    def get_RDdelensingcorrbias(self, _type, lib_qlm, ALWFcl, clsobs_deconv, clsobs_deconv2=None, cls_weights=None):
        """
        Let the unnormalized qest be written as d_a = A^{im} X^i B^{a,l m} X^l
        with X in T,Q,U beam ** deconvolved ** maps.
        -> A^{im} = b^2 Cov^{-1}_{m i}
           B^{a, lm} = [ik_a b^2 C_len  Cov^{-1}]_{m l}
        and Bias is Hab(z) * [ (AC_m_dai)(-z) (BC_bmj) + (AC_mj)(-z) (BC_bmdai)]

        WFcl is the filter applied to the qest prior delensing the maps. (e.g. Cpp / (Cpp + N0) for WF filtering)
        putting cls_obs being cls_len  + noise /bl2 should give the same thing as get_delensingcorrbias.
        The filtering Cls is always set to Cl len
        The sign of the output is such that is a positive contr. to C^len - C^unl

        If two clsobs are put in, AC is calculate with the first one and BC with the second one.
        Useful for dcl-leading order deviations of the bias.
        
                
        For template subtraction this is slightly different, since the template is built with (E = E_dat^WF, B = 0) 
        template. In this case, the leg, with a {,b} on the spectra should be the cross-spectra
        between (T = T^dat,E^dat,B^dat) and (0,E^WF,0). The other leg is identical, since it pairs a data map and
        the quadratic estiamtor legs built with the data.
        """

        assert _type in _types, (_type, _types)

        t = timer(_timed, suffix=' (delensing RD corr. Bias)')
        retalms = np.zeros((3, 3, self.lib_datalm.alm_size), dtype=complex)
        _cls_weights = cls_weights or self.cls_len
        _clsobs_deconv2 = clsobs_deconv2 or clsobs_deconv
        if cls_weights is not None: t.checkpoint("Using custom cls weights")

        def get_datcl(l, j, id):  # beam deconvolved Cls of the TQU data maps
            # The first index is one in th qest estimator of type '_type' and
            # the second any in TQU.
            assert (l in range(len(_type))) and (j in range(3)), (l, j)
            assert id == 1 or id == 2, id
            if _type == 'QU':
                l_idx = l + 1
            elif _type == 'T':
                l_idx = 0
            else:
                assert _type == 'TQU'
                l_idx = l
            _cmb_cls = clsobs_deconv if id == 1 else _clsobs_deconv2
            return get_unlPmat_ij('TQU', self.lib_datalm, _cmb_cls, l_idx, j)

        def _get_Balm(a, l, m):  # [ik_a b^2 C_len  Cov^{-1}]_{m l}
            # Here both indices should refer to the qest.
            assert a in [0, 1], a
            assert (l in range(len(_type))) and (m in range(len(_type))), (l, m)
            ik = self.lib_datalm.get_ikx if a == 1 else self.lib_datalm.get_iky
            ret = get_unlPmat_ij(_type, self.lib_datalm, _cls_weights, m, 0) \
                  * self.get_Pmatinv(_type, 0, l, use_cls_len=True)
            for _i in range(1, len(_type)):
                ret += get_unlPmat_ij(_type, self.lib_datalm, _cls_weights, m, _i) \
                       * self.get_Pmatinv(_type, _i, l, use_cls_len=True)
            return self.lib_datalm.almxfl(ret, self.cl_transf ** 2) * ik()

        def _get_Akm(l, m):  # [b^2  Cov^{-1}]_{m k}
            # Here both indices should refer to the qest.
            assert (l in range(len(_type))) and (m in range(len(_type))), (l, m)
            return self.lib_datalm.almxfl(self.get_Pmatinv(_type, m, l, use_cls_len=True), self.cl_transf ** 2)

        def get_BCamj(a, m, j):  # sum_l B^{a, l m} \hat C^{lj}
            # The first index m is one in th qest estimator of type '_type' and
            # the second any in TQU.
            assert a in [0, 1], a
            assert m in range(len(_type)), (m, _type)
            assert j in range(3), j
            ret = _get_Balm(a, 0, m) * get_datcl(0, j, 2)
            for _i in range(1, len(_type)):
                ret += _get_Balm(a, _i, m) * get_datcl(_i, j, 2)
            return ret

        def get_ACmj(m, j):  # sum_k A^{k m} \hat C^{kj}
            # The first index m is one in th qest estimator of type '_type' and
            # the second any in TQU.
            assert m in range(len(_type)), (m, _type)
            assert j in range(3), j
            ret = _get_Akm(0, m) * get_datcl(0, j, 1)
            for _i in range(1, len(_type)):
                ret += _get_Akm(_i, m) * get_datcl(_i, j, 1)
            return ret

        def ik(a, libalm=self.lib_datalm):
            assert a in [0, 1], a
            return libalm.get_ikx() if a == 1 else libalm.get_iky()

        _map = lambda _alm: self.lib_datalm.alm2map(_alm)
        for a in [0, 1]:
            for b in [0, 1]:
                t.checkpoint("  Doing axes %s %s" % (a, b))
                Hab = lib_qlm.alm2map(ALWFcl[lib_qlm.reduced_ellmat()] * ik(a, libalm=lib_qlm) * ik(b, libalm=lib_qlm))
                for _i in range(3):  # Building TQU biases, before rotation to Gradient / Curl
                    for _j in range(0, 3):
                        t.checkpoint("    Doing %s" % ({0: 'T', 1: 'Q', 2: 'U'}[_i] + {0: 'T', 1: 'Q', 2: 'U'}[_j]))
                        # Need a sum over a and m
                        for m in range(len(_type)):  # Hab(z) * [ (AC_m_dai)(-z) (BC_bmj) + (AC_mj)(-z) (BC_bmdai)]
                            Pmat = (get_ACmj(m, _i) * ik(a)).conjugate()
                            retalms[_i, _j, :] += self.lib_datalm.map2alm(Hab * _map(Pmat)) * get_BCamj(b, m, _j)
                            Pmat = (get_BCamj(b, m, _i) * ik(a)).conjugate()
                            retalms[_i, _j, :] += self.lib_datalm.map2alm(Hab * _map(Pmat)) * get_ACmj(m, _j)
        norm = 1. / np.sqrt(np.prod(self.lsides))  # ?
        # Sure that i - j x-y is the same ?
        for _i in range(3):  # Building TQU biases, before rotation to Gradient / Curl
            for _j in range(_i, 3):
                if _i == 0 and _j == 0:
                    print "Testing conjecture that in the MV case this is symmetric :"
                print _type + ' :', np.allclose(retalms[_j, _i, :].real, retalms[_i, _j, :].real)
                retalms[_i, _j, :] += retalms[_j, _i, :].conjugate()

        return TQUPmats2TEBcls(self.lib_datalm, retalms) * norm

    def _apply_beams(self, _type, alms):
        assert alms.shape == self._skyalms_shape(_type), (alms.shape, self._skyalms_shape(_type))
        ret = np.empty_like(alms)
        for _i in range(len(_type)): ret[_i] = self.lib_skyalm.almxfl(alms[_i], self.cl_transf)
        return ret

    def apply(self, _type, alms, **kwargs):
        assert alms.shape == self._datalms_shape(_type), (alms.shape, self._datalms_shape(_type))
        ret = np.zeros_like(alms)
        for j in range(len(_type)):
            for i in range(len(_type)):
                ret[i] += \
                    get_datPmat_ij(_type, self.lib_datalm, self.cls_unl, self.cl_transf, self.cls_noise, i, j) * alms[j]
        return ret

    def apply_noise(self, _type, alms, inverse=False):
        """
        Apply noise matrix or its inverse to uqlms.
        """
        assert _type in _types, _type
        assert alms.shape == self._datalms_shape(_type), (alms.shape, self._datalms_shape(_type))

        ret = np.zeros_like(alms)
        for _i in range(len(_type)):
            _cl = self.cls_noise[_type[_i].lower()] if not inverse else 1. / self.cls_noise[_type[_i].lower()]
            ret[_i] = self.lib_datalm.almxfl(alms[_i], _cl)
        return ret

    def get_MLlms(self, _type, datmaps, use_cls_len=True, use_Pool=0, **kwargs):
        """
        Returns maximum likelihood sky CMB modes. (P B^t F^t Cov^-1 = (P^-1 + B^F^t N^{-1} F B)^{-1} F B N^{-1} d)
        Outputs are sky-shaped TEB alms
        """
        ilms = self.apply_conddiagcl(_type, np.array([self.lib_datalm.map2alm(_m) for _m in datmaps]),
                                     use_Pool=use_Pool, use_cls_len=use_cls_len)
        cmb_cls = self.cls_len if use_cls_len else self.cls_unl
        for i in range(len(_type)): ilms[i] = self.lib_datalm.almxfl(ilms[i], self.cl_transf)
        ilms = SM.apply_TEBmat(_type, self.lib_datalm, cmb_cls, SM.TQU2TEBlms(_type, self.lib_datalm, ilms))
        return np.array([self.lib_skyalm.udgrade(self.lib_datalm, _alm) for _alm in ilms])

    def get_iblms(self, _type, datalms, use_cls_len=True, use_Pool=0, **kwargs):
        """
         Returns P^{-1} maximum likelihood sky CMB modes.
         (inputs to quadratc estimator routines)
        """
        if datalms.shape == ((len(_type), self.dat_shape[0], self.dat_shape[1])):
            _datalms = np.array([self.lib_datalm.map2alm(_m) for _m in datalms])
            return self.get_iblms(_type, _datalms, use_cls_len=use_cls_len, use_Pool=use_Pool, **kwargs)
        assert datalms.shape == self._datalms_shape(_type), (datalms.shape, self._datalms_shape(_type))
        ilms = self.apply_conddiagcl(_type, datalms, use_Pool=use_Pool, use_cls_len=use_cls_len)
        ret = np.empty(self._skyalms_shape(_type), dtype=complex)
        for _i in range(len(_type)):
            ret[_i] = self.lib_skyalm.udgrade(self.lib_datalm, self.lib_datalm.almxfl(ilms[_i], self.cl_transf))
        return ret, 0


    def cd_solve(self, _type, alms, cond='3', maxiter=50, ulm0=None,
                 use_Pool=0, tol=1e-5, tr_cd=lensit.qcinv.cd_solve.tr_cg, cd_roundoff=25, d0=None):
        """

        Solves for (F B D xi_unl D^t B^t F^t + N)^-1 dot input alms

        :param cond: conditioner type.
        :param maxiter: maximal number of cg iterations
        :param ulm0: Start guess
        :param use_Pool: See displacements.py
        :param tol: tolerance on the norm of the residual (! w.r.t. the initial guess, or d0)
        :param tr_cd: see cd_solve.py for meaning of this.
        :param cd_roundoff: Number of cg steps before recalculation of the residual.
        :param d0: The criterion for stopping the cg search is np.sum(residual ** 2) * d0 < tol
        """
        assert alms.shape == self._datalms_shape(_type), (alms.shape, self._datalms_shape(_type))
        if ulm0 is not None:
            assert ulm0.shape == self._datalms_shape(_type), (ulm0.shape, self._datalms_shape(_type))

        class dot_op():
            def __init__(self):
                pass

            def __call__(self, alms1, alms2, **kwargs):
                return np.sum(alms1.real * alms2.real + alms1.imag * alms2.imag)

        cond_func = getattr(self, 'apply_cond%s' % cond)

        # ! fwd_op and pre_ops must not change their arguments !
        def fwd_op(_alms):
            return self.apply(_type, _alms, use_Pool=use_Pool)

        def pre_ops(_alms):
            return cond_func(_type, _alms, use_Pool=use_Pool)

        dot_op = dot_op()

        if d0 is None:
            d0 = dot_op(alms, alms)
        if ulm0 is None: ulm0 = np.zeros_like(alms)
        criterion = lensit.qcinv.cd_monitors.monitor_basic(dot_op, iter_max=maxiter, eps_min=tol, d0=d0)
        print "++ ffs_cov cd_solve: starting, cond %s " % cond

        iter = lensit.qcinv.cd_solve.cd_solve(ulm0, alms, fwd_op, [pre_ops], dot_op, criterion, tr_cd,
                                              roundoff=cd_roundoff)
        return ulm0, iter

    def apply_cond3(self, _type, alms, use_Pool=0):
        return self.apply_conddiagcl(_type, alms, use_Pool=use_Pool)

    def apply_cond0(self, _type, alms, use_Pool=0, use_cls_len=True):
        return self.apply_conddiagcl(_type, alms, use_Pool=use_Pool, use_cls_len=use_cls_len)

    def apply_cond0unl(self, _type, alms, **kwargs):
        return self.apply_conddiagcl(_type, alms, use_cls_len=False)

    def apply_cond0len(self, _type, alms, **kwargs):
        return self.apply_conddiagcl(_type, alms, use_cls_len=True)

    def apply_conddiagcl(self, _type, alms, use_cls_len=True, use_Pool=0):
        assert alms.shape == self._datalms_shape(_type), (alms.shape, self._datalms_shape(_type))
        ret = np.zeros_like(alms)
        for i in range(len(_type)):
            for j in range(len(_type)):
                ret[j] += self.get_Pmatinv(_type, j, i, use_cls_len=use_cls_len) * alms[i]
        return ret

    def apply_condpseudiagcl(self, _type, alms, use_Pool=0):
        return self.apply_conddiagcl(_type, alms, use_Pool=use_Pool)

    def get_qlms(self, _type, iblms, lib_qlm, use_cls_len=True, **kwargs):
        """
        Unormalized quadratic estimates (potential and curl).
        Input are max. likelihood sky_alms, given by B^t F^t Cov^{-1} = P^{-1} (...)
        The qlms are (B F^t Cov^1 dat)(z)(d_a xi B F^t Cov^1 dat).
        Only lib_skyalm enters this.

        The sign is correct for the potential estimate ! (not for lik. gradient)
        Corresponds to \hat d_a =  (B F^t Cov^1 dat)(z)(d_a xi B F^t Cov^1 dat)(z).
        The rotation to phi is then \hat phi = - div \hat d
        (Follows from d/dphi = sum_a d / da dda /dphi and dda/dphi = ddelta^D / dam integrating by parts brings a minus
        sign)
        """
        assert iblms.shape == self._skyalms_shape(_type), (iblms.shape, self._skyalms_shape(_type))
        assert lib_qlm.lsides == self.lsides, (self.lsides, lib_qlm.lsides)

        t = timer(_timed)

        weights_cls = self.cls_len if use_cls_len else self.cls_unl
        clms = np.zeros((len(_type), self.lib_skyalm.alm_size), dtype=complex)
        for _i in range(len(_type)):
            for _j in range(len(_type)):
                clms[_i] += get_unlPmat_ij(_type, self.lib_skyalm, weights_cls, _i, _j) * iblms[_j]

        t.checkpoint("  get_qlms::mult with %s Pmat" % ({True: 'len', False: 'unl'}[use_cls_len]))

        _map = lambda alm: self.lib_skyalm.alm2map(alm)
        _2qlm = lambda _m: lib_qlm.udgrade(self.lib_skyalm, self.lib_skyalm.map2alm(_m))

        retdx = _2qlm(_map(iblms[0]) * _map(clms[0] * self.lib_skyalm.get_ikx()))
        retdy = _2qlm(_map(iblms[0]) * _map(clms[0] * self.lib_skyalm.get_iky()))
        for _i in range(1, len(_type)):
            retdx += _2qlm(_map(iblms[_i]) * _map(clms[_i] * self.lib_skyalm.get_ikx()))
            retdy += _2qlm(_map(iblms[_i]) * _map(clms[_i] * self.lib_skyalm.get_iky()))

        t.checkpoint("  get_qlms::cartesian gradients")

        dphi = -retdx * lib_qlm.get_ikx() - retdy * lib_qlm.get_iky()
        dOm = retdx * lib_qlm.get_iky() - retdy * lib_qlm.get_ikx()

        t.checkpoint("  get_qlms::rotation to phi Omega")

        return np.array([2 * dphi, 2 * dOm])  # Factor 2 since gradient w.r.t. real and imag. parts.

    def get_qlm_resprlm(self, _type, lib_qlm,
                        use_cls_len=True, cls_obs=None, cls_obs2=None, cls_filt=None, cls_weights=None):
        """
        Full-fledged alm response matrix. qlm * Rpp_lm is the normalized qest.
        If you want to take into account subtle large scales ell-binning issues in the normalization.
        :param _type:
        :param lib_qlm:
        :param use_cls_len:
        :return:
        """
        assert _type in _types, (_type, _types)
        t = timer(_timed)
        Fpp, FOO, FpO = self.get_qlm_curvature(_type, lib_qlm,
                                               use_cls_len=use_cls_len, cls_obs=cls_obs, cls_filt=cls_filt,
                                               cls_weights=cls_weights, cls_obs2=cls_obs2)

        t.checkpoint("  get_qlm_resplm:: get curvature matrices")

        del FpO
        Rpp = np.zeros_like(Fpp)
        ROO = np.zeros_like(FOO)
        Rpp[np.where(Fpp > 0.)] = 1. / Fpp[np.where(Fpp > 0.)]
        ROO[np.where(FOO > 0.)] = 1. / FOO[np.where(FOO > 0.)]
        t.checkpoint("  get_qlm_resplm:: inversion")

        return Rpp, ROO

    def get_N0cls(self, _type, lib_qlm,
                  use_cls_len=True, cls_obs=None, cls_obs2=None, cls_weights=None, cls_filt=None):
        """
        N0 as cl array. (Norm can be seen as N0 = 2 / F -> norm = N0 / 2)
        """
        assert _type in _types, (_type, _types)
        if cls_obs is None and cls_obs2 is None and cls_weights is None and cls_filt is None:  # default behavior is cached
            fname = self.lib_dir + '/%s_N0cls_%sCls.dat' % (_type, {True: 'len', False: 'unl'}[use_cls_len])
            if not os.path.exists(fname):
                if self.pbsrank == 0:
                    lib_full = lensit.ffs_covs.ell_mat.ffs_alm_pyFFTW(self.lib_datalm.ell_mat,
                                                                      filt_func=lambda ell: ell > 0)
                    Rpp, ROO = self.get_qlm_resprlm(_type, lib_full, use_cls_len=use_cls_len)
                    header = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n' + __file__
                    np.savetxt(fname,
                               np.array(
                                   (2 * lib_full.alm2cl(np.sqrt(Rpp)), 2 * lib_full.alm2cl(np.sqrt(ROO)))).transpose(),
                               fmt=['%.8e'] * 2, header=header)
                self.barrier()
            cl = np.loadtxt(fname).transpose()[:, 0:lib_qlm.ellmax + 1]
            cl[0] *= lib_qlm.filt_func(np.arange(len(cl[0])))
            cl[1] *= lib_qlm.filt_func(np.arange(len(cl[1])))
            return cl
        else:
            Rpp, ROO = self.get_qlm_resprlm(_type, lib_qlm,
                                            use_cls_len=use_cls_len, cls_obs=cls_obs, cls_weights=cls_weights,
                                            cls_filt=cls_filt, cls_obs2=cls_obs2)
            return (2 * lib_qlm.alm2cl(np.sqrt(Rpp)), 2 * lib_qlm.alm2cl(np.sqrt(ROO)))

    def iterateN0cls(self, _type, lib_qlm, Nitmax, Nit=0):
        """
        Iterates delensing and N0 calculation to estimate the noise levels of the iterative estimator.
        This uses perturbative approach in Wiener filtered displacement, consistent with box shape and
        mode structure.
        See iterateN0cls_camb for alternative approach.
        """
        N0 = self.get_N0cls(_type, lib_qlm, use_cls_len=True)[0][:lib_qlm.ellmax + 1]
        if Nit == Nitmax: return N0
        cpp = np.zeros(lib_qlm.ellmax + 1)
        cpp[:min(len(cpp), len(self.cls_unl['pp']))] = (self.cls_unl['pp'][:min(len(cpp), len(self.cls_unl['pp']))])
        clWF = cpp * cl_inverse(cpp + N0[:lib_qlm.ellmax + 1])
        Bl = self.get_delensinguncorrbias(lib_qlm, cpp * (1. - clWF), wNoise=False,
                                          use_cls_len=False)  # TEB matrix output
        cls_delen = {}
        for _k, _cl in self.cls_len.iteritems():
            # Pertubative calc for lensed cl
            cls_delen[_k] = self.cls_unl[_k][:self.lib_datalm.ellmax + 1] \
                            - Bl[{'t': 0, 'e': 1, 'b': 2}[_k[0]], {'t': 0, 'e': 1, 'b': 2}[_k[1]],
                              :self.lib_datalm.ellmax + 1]
        cls_unl = {}
        for _k, _cl in self.cls_unl.iteritems():
            cls_unl[_k] = _cl.copy()
        # cls_unl['pp'][0:min(len(cpp), len(cls_unl['pp']))] = (cpp * (1. - clWF))[0:min(len(cpp), len(cls_unl['pp']))]
        new_libdir = self.lib_dir + '/%s_N0iter/N0iter%04d' % (_type, Nit + 1) if Nit == 0 else \
            self.lib_dir.replace('/N0iter%04d' % (Nit), '/N0iter%04d' % (Nit + 1))

        try:
            new_cov = lensit.ffs_covs.ffs_cov.ffs_diagcov_alm(new_libdir, self.lib_datalm,
                                                              cls_unl, cls_delen, self.cl_transf, self.cls_noise,
                                                              lib_skyalm=self.lib_skyalm)
        except:
            print "hash check failed, removing ", new_libdir
            shutil.rmtree(new_libdir)
            new_cov = lensit.ffs_covs.ffs_cov.ffs_diagcov_alm(new_libdir, self.lib_datalm, cls_unl, cls_delen,
                                                              self.cl_transf, self.cls_noise,
                                                              lib_skyalm=self.lib_skyalm)

        return new_cov.iterateN0cls(_type, lib_qlm, Nitmax, Nit=Nit + 1)

    def iterateN0cls_camb(self, _type, lib_qlm, Nitmax, cambfile, Nit=0):
        """
        Iterates delensing and N0 calculation to estimate the noise levels of the iterative estimator.
        This version is not fully-self-consistent as it calls camb using lensing at all ell
        to predict the result in the box where many ells can be missing.
        See iterateN0cls for alternative approach.
        """
        assert os.path.exists(cambfile), cambfile
        N0 = self.get_N0cls(_type, lib_qlm, use_cls_len=True)[0][:lib_qlm.ellmax + 1]
        if Nit == Nitmax: return N0

        def build_cppweight():
            ret = np.ones(lib_qlm.ellmax + 1, dtype=float)
            _path = self.lib_dir + '/iterN0_cpp_weights.dat'
            ret *= np.loadtxt(_path).transpose()[1]
            return ret

        # cppWeight = build_cppweight()
        cpp = np.zeros(lib_qlm.ellmax + 1)
        cpp[:min(len(cpp), len(self.cls_unl['pp']))] = (
            self.cls_unl['pp'][:min(len(cpp), len(self.cls_unl['pp']))]).copy()
        # cpp *= cppWeight
        # FIXME * (N0 > 0.) ? The modes not entering should or should not be counted for in the lensing ? Better use the self-consistent pert. approach if this is an issue.
        clWF = cpp * cl_inverse(cpp + N0[:lib_qlm.ellmax + 1])
        # build delensed Cls :
        params = fs.misc.jc_camb.read_params(cambfile)
        params['output_root'] += '/iter_%04d' % (Nit + 1)
        params['lensing_method'] = 4  # custom modified rescaling
        if not os.path.exists(params['output_root']): os.makedirs(params['output_root'])
        # np.savetxt('/Users/jcarron/camb/cpp_weights.txt', np.array([np.arange(len(clWF)), cppWeight * (1. - clWF)]).transpose(),fmt=['%i', '%10.5f'])
        np.savetxt('/Users/jcarron/camb/cpp_weights.txt',
                   np.array([np.arange(len(clWF)), (1. - clWF)]).transpose(), fmt=['%i', '%10.5f'])
        fs.misc.jc_camb.run_camb_fromparams(params)
        new_libdir = self.lib_dir + '/%s_N0cambiter_%s/N0iter%04d' % (
            _type, lib_qlm.filt_hash(), Nit + 1) if Nit == 0 else \
            self.lib_dir.replace('/N0iter%04d' % (Nit), '/N0iter%04d' % (Nit + 1))
        cls_len = {}
        for _k, _cl in fs.misc.jc_camb.spectra_fromcambfile(params['output_root'] + '_lensedCls.dat').iteritems():
            cls_len[_k] = _cl[:self.lib_skyalm.ellmax + 1]
        cls_unl = {}
        for _k, _cl in fs.misc.jc_camb.spectra_fromcambfile(
                        params['output_root'] + '_lenspotentialCls.dat').iteritems():
            cls_unl[_k] = _cl[:self.lib_skyalm.ellmax + 1]
        try:
            new_cov = lensit.ffs_covs.ffs_cov.ffs_diagcov_alm(new_libdir, self.lib_datalm,
                                                              self.cls_unl, cls_len, self.cl_transf, self.cls_noise,
                                                              lib_skyalm=self.lib_skyalm)
        except:
            print "hash check failed, removing ", new_libdir
            shutil.rmtree(new_libdir)
            new_cov = lensit.ffs_covs.ffs_cov.ffs_diagcov_alm(new_libdir, self.lib_datalm, self.cls_unl, cls_len,
                                                              self.cl_transf, self.cls_noise,
                                                              lib_skyalm=self.lib_skyalm)
        np.savetxt(new_libdir + '/iterN0_cpp_weights.dat', np.array([np.arange(len(clWF)), (1. - clWF)]).transpose(),
                   fmt=['%i', '%10.5f'])
        return new_cov.iterateN0cls_camb(_type, lib_qlm, Nitmax, cambfile, Nit=Nit + 1)

    def get_N0Pk_minimal(self, _type, lib_qlm, use_cls_len=True, cls_obs=None):
        """
        Same as N0cls but binning only in exactly identical frequencies.
        """
        assert _type in _types, (_type, _types)
        Rpp, ROO = self.get_qlm_resprlm(_type, lib_qlm, use_cls_len=use_cls_len, cls_obs=cls_obs)
        return (lib_qlm.alm2Pk_minimal(np.sqrt(2 * Rpp)), lib_qlm.alm2Pk_minimal(np.sqrt(2 * ROO)))

    def get_qlm_curvature(self, _type, lib_qlm,
                          use_cls_len=True, cls_weights=None, cls_filt=None, cls_obs=None, cls_obs2=None):
        """
        The F matrix for the displacement components phi and Omega (potential and Curl)

        Tries to be memory friendlier.
        Numerical cost : roughly 6 + ( (len(type) * (len(type) + 1))/2 * 10 + 2 * len(type) ** 2) ffts
        -> T : 18 ffts
          QU : 44 ffts
          TQU: 84 ffts

        Since threaded pyFFTW works for me at about 110 Mpix / sec for full sky 16384 ** 2 points (2.4 sec per fft),
        we expect roughly
        45 sec. for T  (16384 ** 2)
        2 min for QU (16384 ** 2)
        3-4 min. for TQU (16384 ** 2)
        Much much faster for any lower res, pyFFTW reaching 200-500 Mpix / sec, and > 4 times lower number of pixels.
        IPython test :  T ex. time in sec :114.453059196
                        QU ex. time in sec :480.987432957
                        TQU ex. time in sec :1131.20375109
                        boh.

        To get the response (and not N0) : put cls_obs to None, cls_filt to your filtering and cls_weights to the true
        len Cls.

        -(db xi B Cov^{-1} B^t )_{ab}(z) (daxi B Cov^{-1} B^t)^{ba}(z)
         - (B Cov^-1 B^t)(z) (daxi B Cov^-1 B^t dbxi)(z)

        To include curl polarisation rotation :
        d0 xi -> d0 xi + d1 R xi
        d1 xi -> d1 xi - d0 R xi

        """
        assert _type in _types, (_type, _types)
        t = timer(_timed, prefix=__name__, suffix=' curvpOlm')

        _cls_weights = cls_weights or (self.cls_len if use_cls_len else self.cls_unl)
        _cls_filt = cls_filt or (self.cls_len if use_cls_len else self.cls_unl)
        _cls_obs = cls_obs or (self.cls_len if use_cls_len else self.cls_unl)
        _cls_obs2 = cls_obs2 or (self.cls_len if use_cls_len else self.cls_unl)

        if not cls_weights is None: t.checkpoint('Using custom Cls weights')
        if not cls_filt is None: t.checkpoint('Using custom Cls filt')
        if not cls_obs is None: t.checkpoint('Using custom Cls obs')
        if not cls_obs2 is None: t.checkpoint('Using custom Cls obs2')

        ## Build the inverse covariance matrix part
        ## For a standard N0 computation, this will just be cov^{-1}.
        ## For RDN0 compuation, it will be cov^{-1} cov_obs cov^{-1}
        ## Notice that in the case of RDN0 computation, only one of the two inverse covariance matrix is replaced.
        if cls_obs is None:
            assert cls_obs2 is None
            _lib_qlm = lensit.ffs_covs.ell_mat.ffs_alm_pyFFTW(self.lib_datalm.ell_mat,
                                                              filt_func=lambda ell: (ell > 0) & (
                                                              ell <= 2 * self.lib_datalm.ellmax))
            if cls_obs is None and cls_weights is None and cls_filt is None and cls_obs2 is None:  # default is cached
                fname = self.lib_dir + '/%s_resplm_%sCls.npy' % (_type, {True: 'len', False: 'unl'}[use_cls_len])
                if os.path.exists(fname):
                    return np.array([lib_qlm.udgrade(_lib_qlm, _a) for _a in np.load(fname)])
            Pinv_obs1 = get_Pmat(_type, self.lib_datalm, _cls_filt, cls_noise=self.cls_noise, cl_transf=self.cl_transf,
                                 inverse=True)
            Pinv_obs2 = Pinv_obs1
        else:

            # FIXME : this will fail if lib_qlm does not have the right shape
            _lib_qlm = lib_qlm
            Covi = get_Pmat(_type, self.lib_datalm, _cls_filt, cls_noise=self.cls_noise, cl_transf=self.cl_transf,
                            inverse=True)
            Pinv_obs1 = np.array([np.dot(a, b) for a, b in
                                  zip(get_Pmat(_type, self.lib_datalm, _cls_obs, cls_noise=None, cl_transf=None),
                                      Covi)])
            Pinv_obs1 = np.array([np.dot(a, b) for a, b in zip(Covi, Pinv_obs1)])
            if cls_obs2 is None:
                Pinv_obs2 = Pinv_obs1
            else:
                Pinv_obs2 = np.array([np.dot(a, b) for a, b in
                                      zip(get_Pmat(_type, self.lib_datalm, _cls_obs2, cls_noise=None, cl_transf=None),
                                          Covi)])
                Pinv_obs2 = np.array([np.dot(a, b) for a, b in zip(Covi, Pinv_obs2)])
            del Covi

        # B xi B^t Cov^{-1} (or Cov^-1 Cov_obs Cov^-1 for semi-analytical N0)
        def get_BPBCovi(i, j, id):
            assert id in [1, 2]
            _Pinv_obs = Pinv_obs1 if id == 1 else Pinv_obs2
            ret = get_unlPmat_ij(_type, self.lib_datalm, _cls_weights, i, 0) * _Pinv_obs[:, 0, j]
            for _k in range(1, len(_type)):
                ret += get_unlPmat_ij(_type, self.lib_datalm, _cls_weights, i, _k) * _Pinv_obs[:, _k, j]
            return self.lib_datalm.almxfl(ret, self.cl_transf ** 2)

        def get_BPBCoviP(i, j, id):
            assert id in [1, 2]
            ret = get_BPBCovi(i, 0, id) * get_unlPmat_ij(_type, self.lib_datalm, _cls_weights, 0, j)
            for _k in range(1, len(_type)):
                ret += get_BPBCovi(i, _k, id) * get_unlPmat_ij(_type, self.lib_datalm, _cls_weights, _k, j)
            return ret

        def get_BPBCovi_rot(i, j, id):
            assert id in [1, 2]
            _Pinv_obs = Pinv_obs1 if id == 1 else Pinv_obs2
            ret = get_unlrotPmat_ij(_type, self.lib_datalm, _cls_weights, i, 0) * _Pinv_obs[:, 0, j]
            for _k in range(1, len(_type)):
                ret += get_unlrotPmat_ij(_type, self.lib_datalm, _cls_weights, i, _k) * _Pinv_obs[:, _k, j]
            return self.lib_datalm.almxfl(ret, self.cl_transf ** 2)

        ikx = self.lib_datalm.get_ikx
        iky = self.lib_datalm.get_iky
        t.checkpoint("  inverse %s Pmats" % ({True: 'len', False: 'unl'}[use_cls_len]))
        F = np.zeros(self.lib_datalm.ell_mat.shape, dtype=float)
        # 2.1 GB in memory for full sky 16384 ** 2 points. Note however that we can without any loss of accuracy
        # calculate this using a twice as sparse grid, for reasonable input parameters.

        # Calculation of (db xi B Cov^{-1} B^t )_{ab}(z) (daxi B Cov^{-1} B^t)^{ba}(z)
        for i in range(len(_type)):
            for j in range(i, len(_type)):
                # ! BPBCovi Matrix not symmetric for TQU or non identical noises. But xx or yy element ok.
                F += (2 - (i == j)) * self.lib_datalm.alm2map(ikx() * get_BPBCovi(i, j, 1)) \
                     * self.lib_datalm.alm2map(ikx() * get_BPBCovi(j, i, 2))
        Fxx = _lib_qlm.map2alm(F)
        F *= 0
        t.checkpoint("  Fxx , part 1")

        for i in range(len(_type)):
            for j in range(i, len(_type)):
                # ! BPBCovi Matrix not symmetric for TQU or non identical noises. But xx or yy element ok.
                F += (2 - (i == j)) * self.lib_datalm.alm2map(iky() * get_BPBCovi(i, j, 1)) \
                     * self.lib_datalm.alm2map(iky() * get_BPBCovi(j, i, 2))
        Fyy = _lib_qlm.map2alm(F)
        F *= 0
        t.checkpoint("  Fyy , part 1")

        for i in range(len(_type)):
            for j in range(len(_type)):
                # ! BPBCovi Matrix not symmetric for TQU or non identical noises.
                F += self.lib_datalm.alm2map(ikx() * get_BPBCovi(i, j, 1)) \
                     * self.lib_datalm.alm2map(iky() * get_BPBCovi(j, i, 2))
        Fxy = _lib_qlm.map2alm(F)
        F *= 0
        t.checkpoint("  Fxy , part 1")

        # Adding to that (B Cov^-1 B^t)(z) (daxi B Cov^-1 B^t dbxi)(z)
        # Construct Pmat:
        #  Cl * bl ** 2 * cov^{-1} cov_obs cov^{-1} * Cl if semianalytic N0
        #  Cl * bl ** 2 * cov^{-1} * Cl if N0
        #  Now both spectral matrices are symmetric.
        tmap = lambda i, j: self.lib_datalm.alm2map(
            self.lib_datalm.almxfl(Pinv_obs1[:, i, j], (2 - (j == i)) * self.cl_transf ** 2))

        for i in range(len(_type)):
            for j in range(i, len(_type)):
                F += tmap(i, j) * self.lib_datalm.alm2map(ikx() ** 2 * get_BPBCoviP(i, j, 2))
        Fxx += _lib_qlm.map2alm(F)
        F *= 0
        t.checkpoint("  Fxx , part 2")

        for i in range(len(_type)):
            for j in range(i, len(_type)):
                F += tmap(i, j) * self.lib_datalm.alm2map(iky() ** 2 * get_BPBCoviP(i, j, 2))
        Fyy += _lib_qlm.map2alm(F)
        F *= 0
        t.checkpoint("  Fyy , part 2")

        for i in range(len(_type)):
            for j in range(i, len(_type)):
                F += tmap(i, j) * self.lib_datalm.alm2map(iky() * ikx() * get_BPBCoviP(i, j, 2))
        Fxy += _lib_qlm.map2alm(F)
        t.checkpoint("  Fxy , part 2")

        facunits = -2. / np.sqrt(np.prod(self.lsides))
        ret = xylms_to_phiOmegalm(_lib_qlm, Fxx.real * facunits, Fyy.real * facunits, Fxy.real * facunits)
        if cls_obs is None and cls_weights is None and cls_filt is None:
            fname = self.lib_dir + '/%s_resplm_%sCls.npy' % (_type, {True: 'len', False: 'unl'}[use_cls_len])
            np.save(fname, ret)
            return np.array([lib_qlm.udgrade(_lib_qlm, _a) for _a in np.load(fname, mmap_mode='r')])
        return np.array([lib_qlm.udgrade(_lib_qlm, _a) for _a in ret])

    def get_MFrespcls(self, _type, lib_qlm, use_cls_len=True):
        return [lib_qlm.bin_realpart_inell(_r) for _r in self.get_MFresplms(_type, lib_qlm, use_cls_len=use_cls_len)]

    def get_MFresplms(self, _type, lib_qlm, use_cls_len=False, recache=False):
        # (xiK) (xiK) + K (xi K xi - xi) - (l == 0 term)
        """
        Get linear response mean field matrix : In harmonic space, the mean field
        is given by phi_MF = <\hat phi >_phi = RMF_L phi_{lm}

        *** This is the N0-like unnormalised mean field ! ***

        Let lnp_phi = -S_phi  - ln Z.
        The deflection is the negative gradient of the negative lnp hence -dS dphi
        --> The mean field is <-dS/phi> = <dlnZ/dphi> (<dlnp> always vanishes)

        The mean field response is thus

          dphi_MF / dphi = d^2 ln Z / dphi^2 = 0.5 d ln det Cov / dphi^2

        A short calc. gives

        - 1/2 Tr Cov^{-1} dCov/dpa(x) Cov^{-1} dCov/dpb(y)    ( 1 / N0, Fisher curvature)
        + Dirac(x - y) Tr (B^t Cov^{-1} B d^2xi/dadab)(x-y) # this just take out the l == 0 term
        - Tr (B^t Cov^{-1} B)(x -y) (d^2xi/dadab)(x - y)

        The expected normalised MF spectrum is then Rl ** 2 * (qlm_norm) ** 2 * C_hatphi_hatphi.

        The normalisation of the output is for the qlm_norm ~ N0
        ! The xi_ab and Dirac matrices depends on lib_sky but the other on lib_dat
        # FIXME : adapt this for custom cls_filt and cls_weights etc
         
        # (xia K)_ij(xib K)_ji  +(K) (xi K xi - xi)
        #=====
        NB sign:
        This returns the second variation of 1/2 ln det Cov
        #=====
        
        """
        assert self.lib_skyalm.shape == self.lib_datalm.shape
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl
        _lib_qlm = lensit.ffs_covs.ell_mat.ffs_alm_pyFFTW(self.lib_skyalm.ell_mat,
                                                          filt_func=lambda ell: (ell <= 2 * self.lib_skyalm.ellmax))
        fname = self.lib_dir + '/%s_MFresplm_%s.npy' % (_type, {True: 'len', False: 'unl'}[use_cls_len])
        if not os.path.exists(fname) or recache:
            def get_K(i, j):
                # B Covi B
                return self.lib_datalm.almxfl(self.get_Pmatinv(_type, i, j, use_cls_len=use_cls_len),
                                              self.cl_transf ** 2)

            def get_xiK(i, j):
                # xi K
                ret = get_unlPmat_ij(_type, self.lib_skyalm, cls_cmb, i, 0) * self._upg(get_K(0, j))
                for k in range(1, len(_type)):
                    ret += get_unlPmat_ij(_type, self.lib_skyalm, cls_cmb, i, k) * self._upg(get_K(k, j))
                return ret

            _2map = lambda alm: self.lib_skyalm.alm2map(alm)
            _dat2map = lambda alm: self.lib_datalm.alm2map(alm)

            _2qlm = lambda _map: _lib_qlm.map2alm(_map)
            ik = lambda ax: self.lib_skyalm.get_ikx() if ax == 1 else self.lib_skyalm.get_iky()

            Fxx = np.zeros(_lib_qlm.alm_size, dtype=complex)
            Fyy = np.zeros(_lib_qlm.alm_size, dtype=complex)
            Fxy = np.zeros(_lib_qlm.alm_size, dtype=complex)

            for i in range(len(_type)):  # (xia K)_ij(xib K)_ji
                for j in range(len(_type)):
                    xiK1 = get_xiK(i, j)
                    xiK2 = get_xiK(j, i)
                    Fxx += _2qlm(_2map(xiK1 * ik(1)) * (_2map(xiK2 * ik(1))))
                    Fyy += _2qlm(_2map(xiK1 * ik(0)) * (_2map(xiK2 * ik(0))))
                    Fxy += _2qlm(_2map(xiK1 * ik(1)) * (_2map(xiK2 * ik(0))))

            # adding to that (K) (xi K xi - xi) =
            def get_xiKxi_xi(i, j):
                ret = get_xiK(i, 0) * get_unlPmat_ij(_type, self.lib_skyalm, cls_cmb, 0, j)
                for k in range(1, len(_type)):
                    ret += get_xiK(i, k) * get_unlPmat_ij(_type, self.lib_skyalm, cls_cmb, k, j)
                ret -= get_unlPmat_ij(_type, self.lib_skyalm, cls_cmb, i, j)
                return ret

            for i in range(len(_type)):  # (K) (xi K xi - xi)
                for j in range(len(_type)):
                    A = _dat2map(get_K(i, j))
                    B = get_xiKxi_xi(j, i)
                    Fxx += _2qlm(A * _2map(B * ik(1) * ik(1)))
                    Fyy += _2qlm(A * _2map(B * ik(0) * ik(0)))
                    Fxy += _2qlm(A * _2map(B * ik(1) * ik(0)))

            assert _lib_qlm.reduced_ellmat()[0] == 0
            Fxx -= Fxx[0]
            Fyy -= Fyy[0]
            Fxy -= Fxy[0]

            facunits = 1. / np.sqrt(np.prod(self.lsides))
            ret = xylms_to_phiOmegalm(_lib_qlm, Fxx.real * facunits, Fyy.real * facunits, Fxy.real * facunits)
            np.save(fname, ret)
            print "Cached ", fname
        return np.array([lib_qlm.udgrade(_lib_qlm, _a) for _a in np.load(fname)])

    def get_dMFrespcls(self, _type, cmb_dcls, lib_qlm, use_cls_len=False):
        return [lib_qlm.bin_realpart_inell(_r) for _r in
                self.get_dMFresplms(_type, cmb_dcls, lib_qlm, use_cls_len=use_cls_len)]

    def get_dMFresplms(self, _type, cmb_dcls, lib_qlm, use_cls_len=False, recache=False):
        """
        derivative of MF resp (xiK) (xiK) + K (xi K xi - xi) - (l == 0 term) w.r.t. some parameters.
        Uses dxi and dK = -K dxi K, where dxi is built from the CMB spectra variations
        
        Dumbest possible implementation.
        Finite difference test check OK for 'r', inclusive sign and O(h**2) behavior, but strangely the cls derivative
        in both cases look slightly randomly scattered.
        """
        assert self.lib_skyalm.shape == self.lib_datalm.shape
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl
        _lib_qlm = lensit.ffs_covs.ell_mat.ffs_alm_pyFFTW(self.lib_skyalm.ell_mat,
                                                          filt_func=lambda ell: (ell <= 2 * self.lib_skyalm.ellmax))
        # FIXME dclhash !
        print "!!!! dMFresplms::cmb_dcls hash is missing here !! ?"
        fname = self.lib_dir + '/%s_dMFresplm_%s.npy' % (_type, {True: 'len', False: 'unl'}[use_cls_len])
        if not os.path.exists(fname) or recache:

            def mu(a, b, i, j):
                ret = a(i, 0) * b(0, j)
                for _k in range(1, len(_type)):
                    ret += a(i, _k) * b(_k, j)
                return ret

            def dxi(i, j):
                return get_unlPmat_ij(_type, self.lib_skyalm, cmb_dcls, i, j)

            def xi(i, j):
                return get_unlPmat_ij(_type, self.lib_skyalm, cls_cmb, i, j)

            def K(i, j):
                return self._upg(
                    self.lib_datalm.almxfl(self.get_Pmatinv(_type, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2))

            xiK = lambda i, j: mu(xi, K, i, j)
            Kxi = lambda i, j: mu(K, xi, i, j)
            dxiK = lambda i, j: mu(dxi, K, i, j)
            dK = lambda i, j: (-mu(K, dxiK, i, j))
            Kdxi = lambda i, j: mu(K, dxi, i, j)
            xidK = lambda i, j: mu(xi, dK, i, j)
            dKxi = lambda i, j: mu(dK, xi, i, j)
            xiKdxi = lambda i, j: mu(xi, Kdxi, i, j)
            dxiKxi = lambda i, j: mu(dxi, Kxi, i, j)
            xiKxi_xi = lambda i, j: (mu(xiK, xi, i, j) - xi(i, j))
            xidKxi = lambda i, j: mu(xi, dKxi, i, j)
            d_xiK = lambda i, j: (dxiK(i, j) + xidK(i, j))
            d_xiKxi_xi = lambda i, j: (xidKxi(i, j) + dxiKxi(i, j) + xiKdxi(i, j) - dxi(i, j))

            _2map = lambda alm: self.lib_skyalm.alm2map(alm)
            _2qlm = lambda _map: _lib_qlm.map2alm(_map)

            ik = lambda ax: self.lib_skyalm.get_ikx() if ax == 1 else self.lib_skyalm.get_iky()
            Fxx = np.zeros(_lib_qlm.alm_size, dtype=complex)
            Fyy = np.zeros(_lib_qlm.alm_size, dtype=complex)
            Fxy = np.zeros(_lib_qlm.alm_size, dtype=complex)
            for i in range(len(_type)):  # d[(xia K)_ij(xib K)_ji]
                for j in range(len(_type)):
                    xiK1 = d_xiK(i, j)
                    xiK2 = xiK(j, i)
                    Fxx += _2qlm(_2map(xiK1 * ik(1)) * (_2map(xiK2 * ik(1))))
                    Fyy += _2qlm(_2map(xiK1 * ik(0)) * (_2map(xiK2 * ik(0))))
                    Fxy += _2qlm(_2map(xiK1 * ik(1)) * (_2map(xiK2 * ik(0))))
                    xiK1 = xiK(i, j)
                    xiK2 = d_xiK(j, i)
                    Fxx += _2qlm(_2map(xiK1 * ik(1)) * (_2map(xiK2 * ik(1))))
                    Fyy += _2qlm(_2map(xiK1 * ik(0)) * (_2map(xiK2 * ik(0))))
                    Fxy += _2qlm(_2map(xiK1 * ik(1)) * (_2map(xiK2 * ik(0))))

            # adding to that (K) (xi K xi - xi) =
            for i in range(len(_type)):  # d [(K) (xi K xi - xi)]
                for j in range(len(_type)):
                    A = _2map(dK(i, j))
                    B = xiKxi_xi(j, i)
                    Fxx += _2qlm(A * _2map(B * ik(1) * ik(1)))
                    Fyy += _2qlm(A * _2map(B * ik(0) * ik(0)))
                    Fxy += _2qlm(A * _2map(B * ik(1) * ik(0)))
                    A = _2map(K(i, j))
                    B = d_xiKxi_xi(j, i)
                    Fxx += _2qlm(A * _2map(B * ik(1) * ik(1)))
                    Fyy += _2qlm(A * _2map(B * ik(0) * ik(0)))
                    Fxy += _2qlm(A * _2map(B * ik(1) * ik(0)))

            assert _lib_qlm.reduced_ellmat()[0] == 0
            Fxx -= Fxx[0]
            Fyy -= Fyy[0]
            Fxy -= Fxy[0]

            facunits = 1. / np.sqrt(np.prod(self.lsides))
            ret = xylms_to_phiOmegalm(_lib_qlm, Fxx.real * facunits, Fyy.real * facunits, Fxy.real * facunits)
            np.save(fname, ret)
            print "Cached ", fname
        return np.array([lib_qlm.udgrade(_lib_qlm, _a) for _a in np.load(fname)])

    def get_lndetcurv(self, _type, lib_qlm, get_A=None, use_cls_len=False, recache=False):
        """
        This returns 
        1/2 Tr A \frac{\delta^2}{\delta dx_a \delta dx_b} Cov at phi == 0.
        This is - xi_ab(r)(B^t A B) - (ell = 0) term  (for A (x-y) symmetric. Not necessarily symmetric w.r.t. TQU indices)
        if A is not set, A is cov^-1 and the output reduces to second term of the 2nd variation of 1/2 ln det Cov
        = -1/2 Tr dCov Covi dCov Covi + 1/2 Tr Covi d^2Cov 
        
        N0-like normalisation.
        -(xi,ab)(K)
        """
        _lib_qlm = lib_qlm
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl
        if get_A is None:
            get_A = self.get_Pmatinv

        def get_K(i, j):
            # B^t A B
            return self._upg(self.lib_datalm.almxfl(get_A(_type, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2))

        _2qlm = lambda _map: _lib_qlm.map2alm(_map).real
        _2map = lambda alm: self.lib_skyalm.alm2map(alm)
        ik = lambda ax: self.lib_skyalm.get_ikx() if ax == 1 else self.lib_skyalm.get_iky()

        Fxx = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fyy = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fxy = np.zeros(_lib_qlm.alm_size, dtype=float)
        for i in range(len(_type)):  # (K) (xi_ab)
            for j in range(i, len(_type)):  # This is i-j symmetric
                fac = 2 - (i == j)
                K = _2map(get_K(i, j))
                xi = get_unlPmat_ij(_type, self.lib_skyalm, cls_cmb, j, i)
                Fxx -= fac * _2qlm(K * _2map(xi * ik(1) * ik(1)))
                Fyy -= fac * _2qlm(K * _2map(xi * ik(0) * ik(0)))
                Fxy -= fac * _2qlm(K * _2map(xi * ik(1) * ik(0)))

        assert _lib_qlm.reduced_ellmat()[0] == 0
        Fxx -= Fxx[0]
        Fyy -= Fyy[0]
        Fxy -= Fxy[0]
        facunits = 1. / np.sqrt(np.prod(self.lsides))
        return xylms_to_phiOmegalm(_lib_qlm, Fxx * facunits, Fyy * facunits, Fxy * facunits)

    def get_dlndetcurv(self, _type, cmb_dcls, lib_qlm, K=None, dK=None, use_cls_len=False, recache=False):
        """
        This returns 
        1/2 Tr A \frac{\delta^2}{\delta dx_a \delta dx_b} Cov at phi == 0.
        This is - xi_ab(r)(B^t A B) - (ell = 0) term  (for A (x-y) symmetric. Not necessarily symmetric w.r.t. TQU indices)
        if A is not set, A is cov^-1 and the output reduces to second term of the 2nd variation of 1/2 ln det Cov
        = -1/2 Tr dCov Covi dCov Covi + 1/2 Tr Covi d^2Cov 

        N0-like normalisation.
        -(xi,ab)(K) --> -(dxi,ab)(K) - (xi,ab)(dK)
        
        Finite difference test ok.
        """
        _lib_qlm = lib_qlm
        if K is None: assert dK is None
        if K is not None: assert dK is not None

        cls_cmb = self.cls_len if use_cls_len else self.cls_unl

        def mu(a, b, i, j):
            ret = a(i, 0) * b(0, j)
            for _k in range(1, len(_type)):
                ret += a(i, _k) * b(_k, j)
            return ret

        def dxi(i, j):
            return get_unlPmat_ij(_type, self.lib_skyalm, cmb_dcls, i, j)

        def xi(i, j):
            return get_unlPmat_ij(_type, self.lib_skyalm, cls_cmb, i, j)

        if K is None:
            K = lambda i, j: (self._upg(
                self.lib_datalm.almxfl(self.get_Pmatinv(_type, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2)))
        if dK is None:
            dxiK = lambda i, j: mu(dxi, K, i, j)
            dK = lambda i, j: (-mu(K, dxiK, i, j))

        _2qlm = lambda _map: _lib_qlm.map2alm(_map).real
        _2map = lambda alm: self.lib_skyalm.alm2map(alm)
        ik = lambda ax: self.lib_skyalm.get_ikx() if ax == 1 else self.lib_skyalm.get_iky()

        Fxx = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fyy = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fxy = np.zeros(_lib_qlm.alm_size, dtype=float)
        for i in range(len(_type)):  # -(dK) (xi_ab) - (K)(dxi_ab)
            for j in range(len(_type)):
                A = _2map(dK(i, j))
                B = xi(j, i)
                Fxx -= _2qlm(A * _2map(B * ik(1) * ik(1)))
                Fyy -= _2qlm(A * _2map(B * ik(0) * ik(0)))
                Fxy -= _2qlm(A * _2map(B * ik(1) * ik(0)))
                A = _2map(K(i, j))
                B = dxi(j, i)
                Fxx -= _2qlm(A * _2map(B * ik(1) * ik(1)))
                Fyy -= _2qlm(A * _2map(B * ik(0) * ik(0)))
                Fxy -= _2qlm(A * _2map(B * ik(1) * ik(0)))

        assert _lib_qlm.reduced_ellmat()[0] == 0
        Fxx -= Fxx[0]
        Fyy -= Fyy[0]
        Fxy -= Fxy[0]
        facunits = 1. / np.sqrt(np.prod(self.lsides))
        return xylms_to_phiOmegalm(_lib_qlm, Fxx, Fyy, Fxy) * facunits

    def get_fishertrace(self, _type, lib_qlm, get_A1=None, get_A2=None, use_cls_len=True, recache=False):
        """
        This returns 
        1/2 Tr A1 dCov A2 dCov at phi == 0.
        if A or B not set, they reduce to Covi, and the result is the F info at phi = 0.
        N0-like normalisation.
        #-1/2 (xi,a K1)(xi,b K2)  -1/2(xi,b K1)(xi_a K2)
        #-1/2 (K1)(xi,a K2 xib)  -1/2(xi,a K1 xib)(K2)
        K1 = Bt A1 B,  K2 = Bt A2 B, with A1,A2 defaulting to Cov^{-1} 
        """
        # FIXME : The rel . agreement with 1/N0 is only 1e-6 not double prec., not sure what is going on.

        t = timer(_timed, prefix=__name__, suffix=' curvpOlm')
        _lib_qlm = lib_qlm
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl
        if get_A1 is None:
            get_A1 = self.get_Pmatinv
        if get_A2 is None:
            get_A2 = self.get_Pmatinv

        def get_K1(i, j):
            return self.lib_datalm.almxfl(get_A1(_type, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2)

        def get_K2(i, j):
            return self.lib_datalm.almxfl(get_A2(_type, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2)

        def get_xiK(mat, i, j):
            assert mat in [1, 2], mat
            K = get_K1 if mat == 1 else get_K2
            ret = get_unlPmat_ij(_type, self.lib_datalm, cls_cmb, i, 0) * K(0, j)
            for _k in range(1, len(_type)):
                ret += get_unlPmat_ij(_type, self.lib_datalm, cls_cmb, i, _k) * K(_k, j)
            return ret

        def get_xiKxi(mat, i, j):
            assert mat in [1, 2], mat
            ret = get_xiK(mat, i, 0) * get_unlPmat_ij(_type, self.lib_datalm, cls_cmb, 0, j)
            for _k in range(1, len(_type)):
                ret += get_xiK(mat, i, _k) * get_unlPmat_ij(_type, self.lib_datalm, cls_cmb, _k, j)
            return ret

        _2qlm = lambda _map: _lib_qlm.map2alm(_map).real
        _2map = lambda alm: self.lib_datalm.alm2map(alm)
        k = lambda ax: self.lib_datalm.get_ikx() if ax == 1 else self.lib_datalm.get_iky()

        Fxx = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fyy = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fxy = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fyx = np.zeros(_lib_qlm.alm_size, dtype=float)
        # -1/2 (xia B^t A1 B)(xib B^t A2 B)  -1/2(xi_b B^t A1 B)(xi_a B^t A2 B)
        # -1/2 (B^t A1 B)(xia B^t A2 B xib)  -1/2(xia B^t A1 B xib)(B^t A2 B)
        for i in range(len(_type)):  # (K) (xi_ab)
            for j in range(len(_type)):
                _K1 = _2map(get_K1(i, j))
                xiK2xi = get_xiKxi(2, j, i)  # (B^t A1 B)(xia B^t A2 B xib)
                Fxx += _2qlm(_K1 * _2map(xiK2xi * k(1) * k(1)))
                Fyy += _2qlm(_K1 * _2map(xiK2xi * k(0) * k(0)))
                Fxy += _2qlm(_K1 * _2map(xiK2xi * k(1) * k(0)))
                Fyx += _2qlm(_K1 * _2map(xiK2xi * k(0) * k(1)))

                del _K1, xiK2xi
                _K2 = _2map(get_K2(i, j))
                xiK1xi = get_xiKxi(1, j, i)  # (xia B^t A B xib)(B^t A2 B)
                Fxx += _2qlm(_K2 * _2map(xiK1xi * k(1) * k(1)))
                Fyy += _2qlm(_K2 * _2map(xiK1xi * k(0) * k(0)))
                Fxy += _2qlm(_K2 * _2map(xiK1xi * k(1) * k(0)))
                Fyx += _2qlm(_K2 * _2map(xiK1xi * k(0) * k(1)))

                del _K2, xiK1xi
                xiK1 = get_xiK(1, i, j)
                xiK2 = get_xiK(2, j, i)  # (xia B^t A1 B)(xib B^t A2 B)
                Fxx += _2qlm(_2map(xiK1 * k(1)) * _2map(xiK2 * k(1)))
                Fyy += _2qlm(_2map(xiK1 * k(0)) * _2map(xiK2 * k(0)))
                Fxy += _2qlm(_2map(xiK1 * k(1)) * _2map(xiK2 * k(0)))
                Fyx += _2qlm(_2map(xiK1 * k(0)) * _2map(xiK2 * k(1)))
                #                K1xi = get_xiK(1, i, j)
                #                K2xi = get_xiK(2, j, i)#(xi_a B^t A2 B)(xi_b B^t A1 B)
                Fxx += _2qlm(_2map(xiK2 * k(1)) * _2map(xiK1 * k(1)))
                Fyy += _2qlm(_2map(xiK2 * k(0)) * _2map(xiK1 * k(0)))
                Fxy += _2qlm(_2map(xiK2 * k(1)) * _2map(xiK1 * k(0)))
                Fyx += _2qlm(_2map(xiK2 * k(0)) * _2map(xiK1 * k(1)))
                # del K1xi,K2xi

                t.checkpoint('%s %s done' % (i, j))
        facunits = -0.5 * (1. / np.sqrt(np.prod(self.lsides)))  # N0-like norm
        return xylms_to_phiOmegalm(_lib_qlm, Fxx * facunits, Fyy * facunits, Fxy * facunits, Fyx=Fyx * facunits)

    def get_dfishertrace(self, _type, cmb_dcls, lib_qlm, K1=None, K2=None, dK1=None, dK2=None,
                         use_cls_len=True, recache=False):
        """
        Variation of fisher trace
           #-1/2 (xi,a K1)(xi,b K2)  -1/2(xi,b K1)(xi_a K2)
           #-1/2 (K1)(xi,a K2 xib)   -1/2(xi,a K1 xib)(K2)
        with respect to dcls_cmb
        Finite difference test OK
        """
        if K1 is not None: assert dK1 is not None
        if K2 is not None: assert dK2 is not None
        t = timer(_timed, prefix=__name__, suffix=' curvpOlm')
        _lib_qlm = lib_qlm
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl

        def mu(a, b, i, j):
            ret = a(i, 0) * b(0, j)
            for _k in range(1, len(_type)):
                ret += a(i, _k) * b(_k, j)
            return ret

        def dxi(i, j):
            return get_unlPmat_ij(_type, self.lib_skyalm, cmb_dcls, i, j)

        def xi(i, j):
            return get_unlPmat_ij(_type, self.lib_skyalm, cls_cmb, i, j)

        if K1 is None:
            assert dK1 is None
            K1 = lambda i, j: self._upg(
                self.lib_datalm.almxfl(self.get_Pmatinv(_type, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2))
        if K2 is None:
            assert dK2 is None
            K2 = lambda i, j: self._upg(
                self.lib_datalm.almxfl(self.get_Pmatinv(_type, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2))

        xiK1 = lambda i, j: mu(xi, K1, i, j)
        xiK2 = lambda i, j: mu(xi, K2, i, j)
        dxiK1 = lambda i, j: mu(dxi, K1, i, j)
        dxiK2 = lambda i, j: mu(dxi, K2, i, j)
        if dK1 is None: dK1 = lambda i, j: (-mu(K1, dxiK1, i, j))
        if dK2 is None: dK2 = lambda i, j: (-mu(K2, dxiK2, i, j))
        xiK1xi = lambda i, j: mu(xiK1, xi, i, j)
        xiK2xi = lambda i, j: mu(xiK2, xi, i, j)
        xidK1 = lambda i, j: mu(xi, dK1, i, j)
        xidK2 = lambda i, j: mu(xi, dK2, i, j)
        d_xiK1 = lambda i, j: (dxiK1(i, j) + xidK1(i, j))
        d_xiK2 = lambda i, j: (dxiK2(i, j) + xidK2(i, j))
        d_xiK1xi = lambda i, j: (mu(d_xiK1, xi, i, j) + mu(xiK1, dxi, i, j))
        d_xiK2xi = lambda i, j: (mu(d_xiK2, xi, i, j) + mu(xiK2, dxi, i, j))

        _2qlm = lambda _map: _lib_qlm.map2alm(_map).real
        _2map = lambda alm: self.lib_skyalm.alm2map(alm)
        ik = lambda ax: self.lib_skyalm.get_ikx() if ax == 1 else self.lib_skyalm.get_iky()

        Fxx = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fyy = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fxy = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fyx = np.zeros(_lib_qlm.alm_size, dtype=float)
        # -1/2 (xi,a K1)(xi,b K2)  -1/2(xi,b K1)(xi_a K2)
        # -1/2 (K1)(xi,a K2 xib)   -1/2(xi,a K1 xib)(K2)
        for i in range(len(_type)):
            for j in range(len(_type)):
                # -1/2 d[ (xi,a K1)(xi,b K2)]
                A = d_xiK1(i, j)
                B = xiK2(j, i)
                Fxx += _2qlm(_2map(A * ik(1)) * _2map(B * ik(1)))
                Fyy += _2qlm(_2map(A * ik(0)) * _2map(B * ik(0)))
                Fxy += _2qlm(_2map(A * ik(1)) * _2map(B * ik(0)))
                Fyx += _2qlm(_2map(A * ik(0)) * _2map(B * ik(1)))
                A = xiK1(i, j)
                B = d_xiK2(j, i)
                Fxx += _2qlm(_2map(A * ik(1)) * _2map(B * ik(1)))
                Fyy += _2qlm(_2map(A * ik(0)) * _2map(B * ik(0)))
                Fxy += _2qlm(_2map(A * ik(1)) * _2map(B * ik(0)))
                Fyx += _2qlm(_2map(A * ik(0)) * _2map(B * ik(1)))
                # -1/2 d[(xi_a K2)(xi,b K1)]
                A = d_xiK2(i, j)
                B = xiK1(j, i)
                Fxx += _2qlm(_2map(A * ik(1)) * _2map(B * ik(1)))
                Fyy += _2qlm(_2map(A * ik(0)) * _2map(B * ik(0)))
                Fxy += _2qlm(_2map(A * ik(1)) * _2map(B * ik(0)))
                Fyx += _2qlm(_2map(A * ik(0)) * _2map(B * ik(1)))
                A = xiK2(i, j)
                B = d_xiK1(j, i)
                Fxx += _2qlm(_2map(A * ik(1)) * _2map(B * ik(1)))
                Fyy += _2qlm(_2map(A * ik(0)) * _2map(B * ik(0)))
                Fxy += _2qlm(_2map(A * ik(1)) * _2map(B * ik(0)))
                Fyx += _2qlm(_2map(A * ik(0)) * _2map(B * ik(1)))
                # -1/2 (K1)(xi,a K2 xib)
                A = _2map(dK1(i, j))
                B = xiK2xi(j, i)
                Fxx += _2qlm(A * _2map(B * ik(1) * ik(1)))
                Fyy += _2qlm(A * _2map(B * ik(0) * ik(0)))
                Fxy += _2qlm(A * _2map(B * ik(1) * ik(0)))
                Fyx += _2qlm(A * _2map(B * ik(0) * ik(1)))
                A = _2map(K1(i, j))
                B = d_xiK2xi(j, i)
                Fxx += _2qlm(A * _2map(B * ik(1) * ik(1)))
                Fyy += _2qlm(A * _2map(B * ik(0) * ik(0)))
                Fxy += _2qlm(A * _2map(B * ik(1) * ik(0)))
                Fyx += _2qlm(A * _2map(B * ik(0) * ik(1)))
                # -1/2(xi,a K1 xib)(K2)
                A = _2map(dK2(i, j))
                B = xiK1xi(j, i)
                Fxx += _2qlm(A * _2map(B * ik(1) * ik(1)))
                Fyy += _2qlm(A * _2map(B * ik(0) * ik(0)))
                Fxy += _2qlm(A * _2map(B * ik(1) * ik(0)))
                Fyx += _2qlm(A * _2map(B * ik(0) * ik(1)))
                A = _2map(K2(i, j))
                B = d_xiK1xi(j, i)
                Fxx += _2qlm(A * _2map(B * ik(1) * ik(1)))
                Fyy += _2qlm(A * _2map(B * ik(0) * ik(0)))
                Fxy += _2qlm(A * _2map(B * ik(1) * ik(0)))
                Fyx += _2qlm(A * _2map(B * ik(0) * ik(1)))
                t.checkpoint('%s %s done' % (i, j))
        facunits = -0.5 * (1. / np.sqrt(np.prod(self.lsides)))  # N0-like norm
        return xylms_to_phiOmegalm(_lib_qlm, Fxx * facunits, Fyy * facunits, Fxy * facunits, Fyx=Fyx * facunits)

    def get_plmlikcurvcls(self, _type, datcmb_cls, lib_qlm, use_cls_len=True, recache=False, dat_only=False):
        """
        returns second variation (curvature) of <1/2Xdat Covi Xdat + 1/2 ln det Cov>_dat,
        where dat Cls does not match the fiducial Cls of this instance.
        If they do match, the result should be 1/N_0.
        
        datcls includes only cmb cls
        
        This can be written as (see rPDF notes)
        1/2 Tr (2Covi Covdat - 1) Covi dCov Covi dCov  (Fisher trace)
        -1/2 Tr (Covi Covdat -1) Covi d2Cov  (lndet curv)
        N0-like normalisation 
        """
        # FIXME : The rel . agreement with 1/N0 is only 1e-6 not double prec., not sure what is going on.
        fname = self.lib_dir + '/%splmlikcurv_cls%s_cldat' % (_type, {True: 'len', False: 'unl'}[use_cls_len]) \
                + cls_hash(datcmb_cls, lmax=self.lib_datalm.ellmax) + '.dat'
        if not os.path.exists(fname) or recache:
            def get_dcov(_type, i, j, use_cls_len=True):
                # Covi (datCov - Cov) = Covi datCov - 1
                ret = self.get_Pmatinv(_type, i, 0, use_cls_len=use_cls_len) \
                      * get_datPmat_ij(_type, self.lib_datalm, datcmb_cls, self.cl_transf, self.cls_noise, 0, j)
                for _k in range(1, len(_type)):
                    ret += self.get_Pmatinv(_type, i, _k, use_cls_len=use_cls_len) \
                           * get_datPmat_ij(_type, self.lib_datalm, datcmb_cls, self.cl_transf, self.cls_noise, _k, j)
                if i == j and not dat_only:
                    ret -= 1.
                return ret

            def get_dcovd(_type, i, j, use_cls_len=True):
                # Covi(datCov - Cov)Covi
                ret = get_dcov(_type, i, 0, use_cls_len=use_cls_len) \
                      * self.get_Pmatinv(_type, 0, j, use_cls_len=use_cls_len)
                for _k in range(1, len(_type)):
                    ret += get_dcov(_type, i, _k, use_cls_len=use_cls_len) \
                           * self.get_Pmatinv(_type, _k, j, use_cls_len=use_cls_len)
                return ret

            def get_d2cov(_type, i, j, use_cls_len=True):
                # Cov^-1 (2 datCov - Cov)  = 2 Covi datCov - 1
                ret = self.get_Pmatinv(_type, i, 0, use_cls_len=use_cls_len) \
                      * get_datPmat_ij(_type, self.lib_datalm, datcmb_cls, self.cl_transf, self.cls_noise, 0, j)
                for _k in range(1, len(_type)):
                    ret += self.get_Pmatinv(_type, i, _k, use_cls_len=use_cls_len) \
                           * get_datPmat_ij(_type, self.lib_datalm, datcmb_cls, self.cl_transf, self.cls_noise, _k, j)
                if i == j and not dat_only:
                    return 2 * ret - 1.
                return 2 * ret

            def get_idCi(_type, i, j, use_cls_len=True):
                # Cov^-1 (2 datCov - Cov) Cov^-1
                ret = get_d2cov(_type, i, 0, use_cls_len=use_cls_len) * self.get_Pmatinv(_type, 0, j,
                                                                                         use_cls_len=use_cls_len)
                for _k in range(1, len(_type)):
                    ret += get_d2cov(_type, i, _k, use_cls_len=use_cls_len) * self.get_Pmatinv(_type, _k, j,
                                                                                               use_cls_len=use_cls_len)
                return ret

            # First term :
            _lib_qlm = fs.ffs_covs.ell_mat.ffs_alm_pyFFTW(lib_qlm.ell_mat, filt_func=lambda ell: ell >= 0)
            curv = -self.get_lndetcurv(_type, _lib_qlm, get_A=get_dcovd, use_cls_len=use_cls_len)
            Fish = self.get_fishertrace(_type, _lib_qlm, get_A1=get_idCi, use_cls_len=use_cls_len)
            ret = np.array([_lib_qlm.bin_realpart_inell(_r) for _r in (curv + Fish)[:2]])
            np.savetxt(fname, ret.transpose(),
                       header='second variation (curvature) of <1/2Xdat Covi Xdat + 1/2 ln det Cov>_dat')
            print "Cached ", fname
        print "loading ", fname
        ret = np.array([(_r * lib_qlm.cond)[:lib_qlm.ellmax + 1] for _r in np.loadtxt(fname).transpose()])
        return ret

    def get_plmRDlikcurvcls(self, _type, datcls_obs, lib_qlm, use_cls_len=True, use_cls_len_D=None, recache=False,
                            dat_only=False):
        """
        returns second variation (curvature) of <1/2Xdat Covi Xdat + 1/2 ln det Cov>_dat,
        where dat Cls does not match the fiducial Cls of this instance.
        If they do match, the result should be 1/N_0.

        datcls includes transfer fct and noise (bl**2 Cl + noise)

        This can be written as (see rPDF notes)
        1/2 Tr (2Covi Covdat - 1) Covi dCov Covi dCov  (Fisher trace)
        -1/2 Tr (Covi Covdat -1) Covi d2Cov  (lndet curv)
        N0-like normalisation 
        The data-independent part must be the var. of 1/2 ln det Cov i.e. the MF response.
        """
        # FIXME : The rel . agreement with 1/N0 is only 1e-6 not double prec., not sure what is going on.
        fname = self.lib_dir + '/%splmRDlikcurv_cls%s_cldat' % (_type, {True: 'len', False: 'unl'}[use_cls_len]) \
                + cls_hash(datcls_obs, lmax=self.lib_datalm.ellmax) + '.dat'
        if dat_only:
            fname = fname.replace(self.lib_dir + '/', self.lib_dir + '/datonly')
            assert 'datonly' in fname
        if use_cls_len_D is not None and use_cls_len_D != use_cls_len:
            fname = fname.replace('.dat', '_clsD%s.dat' % {True: 'len', False: 'unl'}[use_cls_len_D])
        else:
            use_cls_len_D = use_cls_len
        if not os.path.exists(fname) or recache:
            def get_dcov(_type, i, j, use_cls_len=use_cls_len):
                # Covi (datCov - Cov) = Covi datCov - 1
                ret = self.get_Pmatinv(_type, i, 0, use_cls_len=use_cls_len) \
                      * get_unlPmat_ij(_type, self.lib_datalm, datcls_obs, 0, j)
                for _k in range(1, len(_type)):
                    ret += self.get_Pmatinv(_type, i, _k, use_cls_len=use_cls_len) \
                           * get_unlPmat_ij(_type, self.lib_datalm, datcls_obs, _k, j)
                if i == j and not dat_only:
                    ret -= 1.
                return ret

            def get_dcovd(_type, i, j, use_cls_len=use_cls_len):
                # Covi(datCov - Cov)Covi
                ret = get_dcov(_type, i, 0, use_cls_len=use_cls_len) \
                      * self.get_Pmatinv(_type, 0, j, use_cls_len=use_cls_len)
                for _k in range(1, len(_type)):
                    ret += get_dcov(_type, i, _k, use_cls_len=use_cls_len) \
                           * self.get_Pmatinv(_type, _k, j, use_cls_len=use_cls_len)
                return ret

            def get_d2cov(_type, i, j, use_cls_len=use_cls_len):
                # Cov^-1 (2 datCov - Cov)  = 2 Covi datCov - 1
                ret = self.get_Pmatinv(_type, i, 0, use_cls_len=use_cls_len) \
                      * get_unlPmat_ij(_type, self.lib_datalm, datcls_obs, 0, j)
                for _k in range(1, len(_type)):
                    ret += self.get_Pmatinv(_type, i, _k, use_cls_len=use_cls_len) \
                           * get_unlPmat_ij(_type, self.lib_datalm, datcls_obs, _k, j)
                if i == j and not dat_only:
                    return 2 * ret - 1.
                return 2 * ret

            def get_idCi(_type, i, j, use_cls_len=use_cls_len):
                # Cov^-1 (2 datCov - Cov) Cov^-1
                ret = get_d2cov(_type, i, 0, use_cls_len=use_cls_len) * self.get_Pmatinv(_type, 0, j,
                                                                                         use_cls_len=use_cls_len)
                for _k in range(1, len(_type)):
                    ret += get_d2cov(_type, i, _k, use_cls_len=use_cls_len) * self.get_Pmatinv(_type, _k, j,
                                                                                               use_cls_len=use_cls_len)
                return ret

            # First term :
            _lib_qlm = fs.ffs_covs.ell_mat.ffs_alm_pyFFTW(lib_qlm.ell_mat, filt_func=lambda ell: ell >= 0)
            curv = -self.get_lndetcurv(_type, _lib_qlm, get_A=get_dcovd, use_cls_len=use_cls_len_D)
            Fish = self.get_fishertrace(_type, _lib_qlm, get_A1=get_idCi, use_cls_len=use_cls_len)
            ret = np.array([_lib_qlm.bin_realpart_inell(_r) for _r in (curv + Fish)[:2]])
            np.savetxt(fname, ret.transpose(),
                       header='second variation (curvature) of <1/2Xdat Covi Xdat + 1/2 ln det Cov>_dat')
            # retc = _lib_qlm.bin_realpart_inell(curv[0])
            # retF = _lib_qlm.bin_realpart_inell(Fish[0])
            # np.savetxt(fname, np.array([retc, retF]).transpose(),
            #           header='second variation (curvature) of <1/2Xdat Covi Xdat + 1/2 ln det Cov>_dat')
            # print "FIXE THIS get_plm"
            # ret = withD * np.array([_lib_qlm.bin_realpart_inell(_r) for _r in curv[:2]])
            # if withF: ret += np.array([_lib_qlm.bin_realpart_inell(_r) for _r in Fish[:2]])
            # return ret
        print "loading ", fname
        ret = np.array([(_r * lib_qlm.cond)[:lib_qlm.ellmax + 1] for _r in np.loadtxt(fname).transpose()])
        return ret

    def get_dplmRDlikcurvcls(self, _type, cmb_dcls, datcls_obs, lib_qlm, use_cls_len=True, recache=False,
                             dat_only=False):
        """
        derivative of plmRDlikcurvcls (data held fixed)
        Finite difference test OK (+ much faster)
        """
        # FIXME : this is like really, really, really inefficient.
        fname = self.lib_dir + '/%sdplmRDlikcurv_cls%s_cldat' % (_type, {True: 'len', False: 'unl'}[use_cls_len]) \
                + cls_hash(datcls_obs, lmax=self.lib_datalm.ellmax) + cls_hash(cmb_dcls) + '.dat'
        if dat_only:
            fname = fname.replace(self.lib_dir + '/', self.lib_dir + '/datonly')
            assert 'datonly' in fname
        if not os.path.exists(fname) or recache:
            # K going into trace should be 2 K datcls K - K
            # K going into lndet should be K datcls K
            # dK is -K dxi K
            def mu(a, b, i, j):
                ret = a(i, 0) * b(0, j)
                for _k in range(1, len(_type)):
                    ret += a(i, _k) * b(_k, j)
                return ret

            dxi = lambda i, j: get_unlPmat_ij(_type, self.lib_skyalm, cmb_dcls, i, j)
            K = lambda i, j: self._upg(
                self.lib_datalm.almxfl(self.get_Pmatinv(_type, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2))
            dxiK = lambda i, j: mu(dxi, K, i, j)
            datKi = lambda i, j: self._upg(
                self.lib_datalm.almxfl(get_unlPmat_ij(_type, self.lib_datalm, datcls_obs, i, j),
                                       cl_inverse(self.cl_transf ** 2)))
            KdatKi = lambda i, j: mu(K, datKi, i, j)
            datKiK = lambda i, j: mu(datKi, K, i, j)

            dK = lambda i, j: (-mu(K, dxiK, i, j))
            KdatKiK = lambda i, j: mu(KdatKi, K, i, j)
            if not dat_only:
                Kdet = lambda i, j: (KdatKiK(i, j) - K(i, j))
                dKdet = lambda i, j: (mu(dK, datKiK, i, j) + mu(KdatKi, dK, i, j) - dK(i, j))
                Ktrace = lambda i, j: (2 * KdatKiK(i, j) - K(i, j))
                dKtrace = lambda i, j: (2 * (mu(dK, datKiK, i, j) + mu(KdatKi, dK, i, j)) - dK(i, j))
            else:
                Kdet = lambda i, j: KdatKiK(i, j)
                dKdet = lambda i, j: (mu(dK, datKiK, i, j) + mu(KdatKi, dK, i, j))
                Ktrace = lambda i, j: 2 * Kdet(i, j)
                dKtrace = lambda i, j: 2 * dKdet(i, j)

            # First term :
            _lib_qlm = fs.ffs_covs.ell_mat.ffs_alm_pyFFTW(lib_qlm.ell_mat, filt_func=lambda ell: ell >= 0)
            curv = -self.get_dlndetcurv(_type, cmb_dcls, _lib_qlm, K=Kdet, dK=dKdet, use_cls_len=use_cls_len)
            Fish = self.get_dfishertrace(_type, cmb_dcls, _lib_qlm, K1=Ktrace, dK1=dKtrace, use_cls_len=use_cls_len)
            ret = np.array([_lib_qlm.bin_realpart_inell(_r) for _r in (curv + Fish)[:2]])
            np.savetxt(fname, ret.transpose(),
                       header='second variation (curvature) of <1/2Xdat Covi Xdat + 1/2 ln det Cov>_dat')
            print "cached ", fname
        print "loading ", fname
        return np.array([(_r * lib_qlm.cond)[:lib_qlm.ellmax + 1] for _r in np.loadtxt(fname).transpose()])
        # return np.loadtxt(fname).transpose()

class ffs_lencov_alm(ffs_diagcov_alm):
    """
    Class for flat sky displaced real_space cov.matrix from full-sky Cls, and deflection field.
    Must be strictly positive definite.
    """

    def __init__(self, lib_dir, lib_datalm, lib_skyalm, cls_unl, cls_len, cl_transf, cls_noise, f, f_inv,
                 init_rank=lensit.pbs.rank, init_barrier=lensit.pbs.barrier):
        """
        f and finv are displacement field classes. Number of points on each side 2**HD_res,2**HD_res.
        f and f_inv must have a f.lens_map routine that does the lensing of map 2**HD_res by 2**HD_res.
        f_inv must also have a f.det_M routine which returns the determinant of the magnification matrix
        at all points of the map.
        """

        assert lib_datalm.ell_mat.lsides == lib_skyalm.ell_mat.lsides, \
            (lib_datalm.ell_mat.lsides, lib_skyalm.ell_mat.lsides)

        super(ffs_lencov_alm, self).__init__(lib_dir, lib_datalm, cls_unl, cls_len, cl_transf, cls_noise,
                                             lib_skyalm=lib_skyalm, init_barrier=init_barrier, init_rank=init_rank)

        self.lmax_dat = self.lib_datalm.ellmax
        self.lmax_sky = self.lib_skyalm.ellmax

        self.sky_shape = self.lib_skyalm.ell_mat.shape
        # assert self.lmax_dat <= self.lmax_sky, (self.lmax_dat, self.lmax_sky)

        for cl in self.cls_unl.values():   assert len(cl) > lib_skyalm.ellmax

        assert f.shape == self.sky_shape and f_inv.shape == self.sky_shape, (f.shape, f_inv.shape, self.sky_shape)
        assert f.lsides == self.lsides and f_inv.lsides == self.lsides, (f.lsides, f_inv.lsides, self.lsides)
        self.f_inv = f_inv  # inverse displacement
        self.f = f  # displacement

    def hashdict(self):
        hash = {'lib_alm': self.lib_datalm.hashdict(), 'lib_skyalm': self.lib_skyalm.hashdict()}
        for key, cl in self.cls_noise.iteritems():
            hash['cls_noise ' + key] = hashlib.sha1(cl).hexdigest()
        for key, cl in self.cls_unl.iteritems():
            hash['cls_unl ' + key] = hashlib.sha1(cl).hexdigest()
        for key, cl in self.cls_len.iteritems():
            hash['cls_len ' + key] = hashlib.sha1(cl).hexdigest()
        hash['cl_transf'] = hashlib.sha1(self.cl_transf.copy()).hexdigest()
        return hash

    def set_ffinv(self, f, finv):
        assert f.shape == self.sky_shape and f.lsides == self.lsides, (f.shape, f.lsides)
        assert finv.shape == self.sky_shape and finv.lsides == self.lsides, (finv.shape, finv.lsides)
        self.f = f
        self.f_inv = finv

    def apply(self, _type, alms, use_Pool=0):
        assert alms.shape == self._datalms_shape(_type), (alms.shape, self._datalms_shape(_type))
        ret = self._apply_signal(_type, alms, use_Pool=use_Pool)
        ret += self.apply_noise(_type, alms)
        return ret

    def _apply_signal(self, _type, alms, use_Pool=0):
        assert alms.shape == self._datalms_shape(_type), (alms.shape, self._datalms_shape(_type))
        ret = np.empty_like(alms)

        if use_Pool <= -100:
            from lensit.gpu import apply_GPU
            ablms = np.array([self.lib_datalm.almxfl(_a, self.cl_transf) for _a in alms])
            apply_GPU.apply_FDxiDtFt_GPU_inplace(_type, self.lib_datalm, self.lib_skyalm, ablms,
                                                 self.f, self.f_inv, self.cls_unl)
            for i in range(len(_type)):
                ret[i] = self.lib_datalm.almxfl(ablms[i], self.cl_transf)
            return ret
        # could do with less
        t = timer(_timed, prefix=__name__, suffix='apply_signal')
        t.checkpoint("just started")

        tempalms = np.empty(self._skyalms_shape(_type), dtype=complex)
        for _i in range(len(_type)):  # Lens with inverse and mult with determinant magnification.
            tempalms[_i] = self.f_inv.lens_alm(self.lib_skyalm,
                                               self._upg(self.lib_datalm.almxfl(alms[_i], self.cl_transf)),
                                               lib_alm_out=self.lib_skyalm, mult_magn=True, use_Pool=use_Pool)
        # NB : 7 new full sky alms for TQU in this routine - > 4 GB total for full sky lmax_sky =  6000.
        t.checkpoint("backward lens + det magn")

        skyalms = np.zeros_like(tempalms)
        for j in range(len(_type)):
            for i in range(len(_type)):
                skyalms[i] += get_unlPmat_ij(_type, self.lib_skyalm, self.cls_unl, i, j) * tempalms[j]
        del tempalms
        t.checkpoint("mult with Punl mat ")

        for i in range(len(_type)):  # Lens with forward displacement
            ret[i] = self._deg(self.f.lens_alm(self.lib_skyalm, skyalms[i], use_Pool=use_Pool))
        t.checkpoint("Forward lensing mat ")

        for i in range(len(_type)):
            ret[i] = self.lib_datalm.almxfl(ret[i], self.cl_transf)
        t.checkpoint("Beams")
        return ret

    def apply_cond3(self, _type, alms, use_Pool=0):
        """
        (DBxiB ^ tD ^ t + N) ^ -1 \sim D ^ -t(BxiBt + N) ^ -1 D ^ -1
        :param alms:
        :return:
        """
        assert alms.shape == self._datalms_shape(_type), (alms.shape, self._datalms_shape(_type))
        t = timer(_timed, prefix=__name__, suffix='apply_cond3')
        t.checkpoint("just started")

        if use_Pool <= -100:
            # Try entire evaluation on GPU :
            # FIXME !! lib_sky vs lib_dat
            from lensit.gpu.apply_cond3_GPU import apply_cond3_GPU_inplace as c3GPU
            ret = alms.copy()
            c3GPU(_type, self.lib_datalm, ret, self.f, self.f_inv, self.cls_unl, self.cl_transf, self.cls_noise)
            return ret
        temp = np.empty_like(alms)  # Cond. must not change their arguments
        for i in range(len(_type)):  # D^{-1}
            temp[i] = self._deg(self.f_inv.lens_alm(self.lib_skyalm, self._upg(alms[i]), use_Pool=use_Pool))

        t.checkpoint("Lensing with inverse")

        ret = np.zeros_like(alms)  # (B xi B^t + N)^{-1}
        for i in range(len(_type)):
            for j in range(len(_type)):
                ret[i] += self.get_Pmatinv(_type, i, j) * temp[j]
        del temp
        t.checkpoint("Mult. w. inv Pmat")

        for i in range(len(_type)):  # D^{-t}
            ret[i] = self._deg(self.f.lens_alm(self.lib_skyalm, self._upg(ret[i]), use_Pool=use_Pool, mult_magn=True))

        t.checkpoint("Lens w. forward and det magn.")

        return ret

    def get_iblms(self, _type, datalms, use_cls_len=False, use_Pool=0, **kwargs):
        """
         Returns B^t F^t Cov^{-1} alms
         (inputs to quadratc estimator routines)
         Ouput of lib_skalm shape
         All **kwargs to cd_solve
        """
        assert use_cls_len == False, 'not implemented'
        if _type == 'QU':
            return self.get_iblms_new(_type, datalms, use_cls_len=use_cls_len, use_Pool=use_cls_len, **kwargs)
        if datalms.shape == ((len(_type), self.dat_shape[0], self.dat_shape[1])):
            _datalms = np.array([self.lib_datalm.map2alm(_m) for _m in datalms])
            return self.get_iblms(_type, _datalms, use_cls_len=use_cls_len, use_Pool=use_Pool, **kwargs)

        assert datalms.shape == self._datalms_shape(_type), (datalms.shape, self._datalms_shape(_type))
        ilms, iter = self.cd_solve(_type, datalms, use_Pool=use_Pool, **kwargs)
        ret = np.empty(self._skyalms_shape(_type), dtype=complex)
        for _i in range(len(_type)):
            ret[_i] = self._upg(self.lib_datalm.almxfl(ilms[_i], self.cl_transf))
        return ret, iter

    def get_iblms_new(self, _type, datalms, use_cls_len=False, use_Pool=0, **kwargs):
        """
         Returns B^t F^t Cov^{-1} alms
          = B^t N^-1(datmaps - B D X^WF)
         (inputs to quadratc estimator routines)
         Ouput of lib_skalm shape
         All **kwargs to cd_solve
         output TQU sky-shaped alms
        """
        # FIXME : some weird things happening with very low noise T ?
        # if datmaps.shape ==  self._datalms_shape(_type):
        #    _datmaps = np.array([self.lib_datalm.alm2map(_m) for _m in datmaps])
        #    return self.get_iblms(_type, _datmaps, use_cls_len=use_cls_len, use_Pool=use_Pool, **kwargs)
        # assert datmaps.shape == (len(_type), self.dat_shape[0], self.dat_shape[1]),(datmaps.shape,self.dat_shape)
        assert use_cls_len == False, 'not implemented'
        MLik = SM.TEB2TQUlms(_type, self.lib_skyalm,
                             self.get_MLlms_new(_type, datalms, use_cls_len=use_cls_len, use_Pool=use_Pool, **kwargs))
        ret = np.zeros(self._skyalms_shape(_type), dtype=complex)
        for i in range(len(_type)):
            temp = datalms[i] - self.lib_datalm.almxfl(self.f.lens_alm(self.lib_skyalm, MLik[i],
                                                                       lib_alm_out=self.lib_datalm, use_Pool=use_Pool),
                                                       self.cl_transf)
            self.lib_datalm.almxfl(temp, self.cl_transf[:self.lib_datalm.ellmax + 1]
                                   * cl_inverse(self.cls_noise[_type[i].lower()][:self.lib_datalm.ellmax + 1]),
                                   inplace=True)
            ret[i] = self._upg(temp)
        return ret, -1  # No iterations info implemented

    def iblm2MLlms(self, _type, iblms, use_Pool=0, use_cls_len=False):
        # C_unl D^t B^t F^t Cov^{-1}
        cmb_cls = self.cls_len if use_cls_len else self.cls_unl
        ret = np.empty(self._skyalms_shape(_type), dtype=complex)
        for _i in range(len(_type)):
            ret[_i] = self.f_inv.lens_alm(self.lib_skyalm, iblms[_i], use_Pool=use_Pool, mult_magn=True)
        return SM.apply_TEBmat(_type, self.lib_skyalm, cmb_cls, SM.TQU2TEBlms(_type, self.lib_skyalm, ret))

    def get_MLlms(self, _type, datmaps, use_Pool=0, use_cls_len=False, **kwargs):
        """
        Returns maximum likelihood sky CMB modes. (P D^t B^t F^t Cov^-1 = (P^-1 + B^F^t N^{-1} F B)^{-1} F B N^{-1} d)
        Outpu TEB shape sky alms
        """
        iblms, iter = self.get_iblms(_type, datmaps, use_cls_len=use_cls_len)
        return self.iblm2MLlms(_type, iblms, use_Pool=use_Pool, use_cls_len=use_cls_len)

    def get_MLlms_new(self, _type, datalms, use_Pool=0, use_cls_len=False, **kwargs):
        assert np.all(self.cls_noise['t'] == self.cls_noise['t'][0]), 'adapt ninv filt ideal for coloured cls(easy)'
        assert np.all(self.cls_noise['q'] == self.cls_noise['q'][0]), 'adapt ninv filt ideal for coloured cls(easy'
        assert np.all(self.cls_noise['u'] == self.cls_noise['q'][0]), 'adapt ninv filt ideal for coloured cls(easy'
        # FIXME could use opfilt_cinvBB
        nlev_t = np.sqrt(self.cls_noise['t'][0] * (180. * 60 / np.pi) ** 2)
        nlev_p = np.sqrt(self.cls_noise['q'][0] * (180. * 60 / np.pi) ** 2)

        cmb_cls = self.cls_len if use_cls_len else self.cls_unl
        filt = lensit.qcinv.ffs_ninv_filt_ideal.ffs_ninv_filt_wl(self.lib_datalm, self.lib_skyalm,
                                                                 cmb_cls, self.cl_transf, nlev_t, nlev_p, self.f,
                                                                 self.f_inv, lens_pool=use_Pool)
        opfilt = lensit.qcinv.opfilt_cinv
        opfilt._type = _type
        chain = lensit.qcinv.chain_samples.get_isomgchain(self.lib_skyalm.ellmax, self.lib_datalm.shape, **kwargs)
        mchain = fs.qcinv.multigrid.multigrid_chain(opfilt, _type, chain, filt)
        soltn = np.zeros((opfilt.TEBlen(_type), self.lib_skyalm.alm_size), dtype=complex)
        mchain.solve(soltn, datalms, finiop='MLIK')
        return soltn

    def get_qlms(self, _type, iblms, lib_qlm, use_Pool=0, use_cls_len=False, **kwargs):
        """
        Likelihood gradient (from the quadratic part).
        (B^t F^t ulm)^a(z) (D dxi_unl/da D^t B^t F^t ulm)_a(z)
        ilms : inverse filtered maps (B^t F^t Cov^-1 dat_alms)
        Only lib_skyalms enter this.
        Sign is correct for pot. estimate, not gradient
        """
        assert iblms.shape == self._skyalms_shape(_type), (iblms.shape, self._skyalms_shape(_type))
        assert lib_qlm.ell_mat.lsides == self.lsides, (self.lsides, lib_qlm.ell_mat.lsides)
        cls = self.cls_len if use_cls_len else self.cls_unl
        almsky1 = np.empty((len(_type), self.lib_skyalm.alm_size), dtype=complex)

        Bu = lambda idx: self.lib_skyalm.alm2map(iblms[idx])
        _2qlm = lambda _m: lib_qlm.udgrade(self.lib_skyalm, self.lib_skyalm.map2alm(_m))

        def DxiDt(alms, axis):
            assert axis in [0, 1]
            kfunc = self.lib_skyalm.get_ikx if axis == 1 else self.lib_skyalm.get_iky
            return self.f.alm2lenmap(self.lib_skyalm, alms * kfunc(), use_Pool=use_Pool)

        t = timer(_timed)
        t.checkpoint("  get_likgrad::just started ")

        for _j in range(len(_type)):  # apply Dt and back to harmonic space :
            almsky1[_j] = self.f_inv.lens_alm(self.lib_skyalm, iblms[_j],
                                              mult_magn=True, use_Pool=use_Pool)

        t.checkpoint("  get_likgrad::Forward lensing maps, (%s map(s)) " % len(_type))
        almsky2 = np.zeros((len(_type), self.lib_skyalm.alm_size), dtype=complex)
        for _i in range(len(_type)):
            for _j in range(len(_type)):
                almsky2[_i] += get_unlPmat_ij(_type, self.lib_skyalm, cls, _i, _j) * almsky1[_j]

        del almsky1
        t.checkpoint("  get_likgrad::Mult. w. unlPmat, %s field(s)" % len(_type))

        retdx = _2qlm(Bu(0) * DxiDt(almsky2[0], 1))
        retdy = _2qlm(Bu(0) * DxiDt(almsky2[0], 0))
        for _i in range(1, len(_type)):
            retdx += _2qlm(Bu(_i) * DxiDt(almsky2[_i], 1))
            retdy += _2qlm(Bu(_i) * DxiDt(almsky2[_i], 0))

        t.checkpoint("  get_likgrad::Cartesian Grad. (%s map(s) lensed, %s fft(s)) " % (2 * len(_type), 2 * len(_type)))

        dphi = retdx * lib_qlm.get_ikx() + retdy * lib_qlm.get_iky()
        dOm = - retdx * lib_qlm.get_iky() + retdy * lib_qlm.get_ikx()
        return np.array([-2 * dphi, -2 * dOm])  # Factor 2 since gradient w.r.t. real and imag. parts.

    def degrade(self, LD_shape, no_lensing=False, ellmax=None, ellmin=None, libtodegrade='sky', lib_dir=None):
        """
        Degrades covariance matrix to some lower resolution.
        """
        # assert 0, 'FIXME'

        if lib_dir is None: lib_dir = self.lib_dir + '/%sdegraded%sx%s_%s_%s' % (
            {True: 'unl', False: 'len'}[no_lensing], LD_shape[0], LD_shape[1], ellmin, ellmax)

        if libtodegrade == 'sky':
            lib_datalmLD = self.lib_datalm.degrade(LD_shape)
            lib_skyalmLD = self.lib_skyalm.degrade(LD_shape, ellmax=ellmax, ellmin=ellmin)
        else:
            lib_dir += 'dat'
            lib_datalmLD = self.lib_datalm.degrade(LD_shape, ellmax=ellmax, ellmin=ellmin)
            lib_skyalmLD = self.lib_skyalm.degrade(LD_shape)

        if no_lensing:
            return ffs_diagcov_alm(lib_dir, lib_datalmLD, self.cls_unl, self.cls_len, self.cl_transf, self.cls_noise,
                                   lib_skyalm=lib_skyalmLD)

        fLD = self.f.degrade(LD_shape, no_lensing)
        finvLD = self.f_inv.degrade(LD_shape, no_lensing)
        return ffs_lencov_alm(lib_dir, lib_datalmLD, lib_skyalmLD,
                              self.cls_unl, self.cls_len, self.cl_transf, self.cls_noise, fLD, finvLD)

    def predMFpOlms(self, _type, lib_qlm, use_cls_len=False):
        """
        Perturbative prediction for the mean field <hat phi>_f,
        which is Rlm plm
        """
        Rpp, ROO, RpO = self.get_MFrespcls(_type, lib_qlm, use_cls_len=use_cls_len)
        del RpO
        plm, Olm = self.f.get_pOlm(lib_qlm)
        plm *= Rpp
        Olm *= ROO
        return np.array([plm, Olm])

    def evalMF(self, _type, MFkey, xlms_sky, xlms_dat, lib_qlm, use_Pool=0, **kwargs):
        """
        All kwargs to cd_solve
        Sign of the output plm-like, not gradient-like
        :param _type:
        :param MFkey:
        :param xlms_sky: unit spectra random phases with sky_alm shape (Not always necessary)
        :param xlms_dat: unit spectra random phases with dat_alm shape
        :param lib_qlm:
        :return:
        """
        assert lib_qlm.ell_mat.lsides == self.lsides, (self.lsides, lib_qlm.ell_mat.lsides)
        assert _type in _types, (_type, _types)
        timer = fs.misc.misc_utils.timer(_timed)
        ikx = lambda: self.lib_skyalm.get_ikx()
        iky = lambda: self.lib_skyalm.get_iky()
        timer.checkpoint('Just started eval MF %s %s' % (_type, MFkey))
        if MFkey == 14:
            # W1 =  B^t F^t l^{1/2},  W2 = D daxi  D^t B^t F^t Cov_f^{-1}l^{-1/2}
            assert np.all([(_x.size == self.lib_datalm.alm_size) for _x in xlms_dat])

            ell_w = 1. / np.sqrt(np.arange(1, self.lib_datalm.ellmax + 2) - 0.5)
            for _i in range(len(_type)):
                xlms_dat[_i] = self.lib_datalm.almxfl(xlms_dat[_i], ell_w)

            def Bx(i):
                _cl = self.cl_transf[:self.lib_datalm.ellmax + 1] / ell_w ** 2
                _alm = self.lib_datalm.almxfl(xlms_dat[i], _cl)
                _alm = self.lib_skyalm.udgrade(self.lib_datalm, _alm)
                return self.lib_skyalm.alm2map(_alm)

            ilms, iter = self.cd_solve(_type, xlms_dat, use_Pool=use_Pool, **kwargs)

            timer.checkpoint('   Done with cd solving')

            for _i in range(len(_type)):
                ilms[_i] = self.lib_datalm.almxfl(ilms[_i], self.cl_transf)
            skyalms = np.empty((len(_type), self.lib_skyalm.alm_size), dtype=complex)
            for _i in range(len(_type)):
                skyalms[_i] = self.lib_skyalm.udgrade(self.lib_datalm, ilms[_i])
                skyalms[_i] = self.f_inv.lens_alm(self.lib_skyalm, skyalms[_i], mult_magn=True, use_Pool=use_Pool)
            del ilms
            timer.checkpoint('   Done with first lensing')

            def _2lenmap(_alm):
                return self.f.alm2lenmap(self.lib_skyalm, _alm, use_Pool=use_Pool)

            dx = np.zeros(lib_qlm.alm_size, dtype=complex)
            dy = np.zeros(lib_qlm.alm_size, dtype=complex)

            for _i in range(len(_type)):
                tempalms = get_unlPmat_ij(_type, self.lib_skyalm, self.cls_unl, _i, 0) * skyalms[0]
                for _j in range(1, len(_type)):
                    tempalms += get_unlPmat_ij(_type, self.lib_skyalm, self.cls_unl, _i, _j) * skyalms[_j]
                dx += lib_qlm.map2alm(Bx(_i) * _2lenmap(tempalms * ikx()))
                dy += lib_qlm.map2alm(Bx(_i) * _2lenmap(tempalms * iky()))
            timer.checkpoint('   Done with second lensing. Done.')
            del skyalms, tempalms

        elif MFkey == 0:
            # Std qest. We build the sim and use the std methods
            assert np.all([(_x.size == self.lib_skyalm.alm_size) for _x in xlms_sky])
            assert np.all([(_x.size == self.lib_datalm.alm_size) for _x in xlms_dat])

            sim = np.empty((len(_type), self.lib_datalm.alm_size), dtype=complex)
            for _i in range(len(_type)):
                skysim = self.get_rootPmatsky(_type, _i, 0) * xlms_sky[0]
                for _j in range(1, len(_type)):
                    skysim += self.get_rootPmatsky(_type, _i, _j) * xlms_sky[_j]
                skysim = self.f.lens_alm(self.lib_skyalm, skysim, use_Pool=use_Pool)
                sim[_i] = self.lib_datalm.udgrade(self.lib_skyalm, skysim)
                sim[_i] = self.lib_datalm.almxfl(sim[_i], self.cl_transf)
            if _type == 'QU':
                sim[0] += self.lib_datalm.almxfl(xlms_dat[0], np.sqrt(self.cls_noise['q']))
                sim[1] += self.lib_datalm.almxfl(xlms_dat[1], np.sqrt(self.cls_noise['u']))
            elif _type == 'TQU':
                sim[0] += self.lib_datalm.almxfl(xlms_dat[0], np.sqrt(self.cls_noise['t']))
                sim[1] += self.lib_datalm.almxfl(xlms_dat[1], np.sqrt(self.cls_noise['q']))
                sim[2] += self.lib_datalm.almxfl(xlms_dat[2], np.sqrt(self.cls_noise['u']))
            elif _type == 'T':
                sim[0] += self.lib_datalm.almxfl(xlms_dat[0], np.sqrt(self.cls_noise['t']))
            else:
                assert 0
            timer.checkpoint('   Done with building sim')

            sim, iter = self.cd_solve(_type, sim, use_Pool=use_Pool, **kwargs)
            for _i in range(len(_type)):  # xlms is now iblms
                sim[_i] = self.lib_datalm.almxfl(sim[_i], self.cl_transf)
            timer.checkpoint('   Done with ivf')
            return self.get_qlms(_type, np.array([self.lib_skyalm.udgrade(self.lib_datalm, _s) for _s in sim]),
                                 lib_qlm=lib_qlm, use_Pool=use_Pool)

        else:
            dx = 0
            dy = 0
            iter = 0
            assert 0, 'MFkey %s not implemented' % MFkey

        dphi = dx * lib_qlm.get_ikx() + dy * lib_qlm.get_iky()
        dOm = -dx * lib_qlm.get_iky() + dy * lib_qlm.get_ikx()
        return np.array([-2 * dphi, -2 * dOm]), iter  # Factor 2 since gradient w.r.t. real and imag. parts.
