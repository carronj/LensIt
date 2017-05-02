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
from lensit.misc.lens_utils import timer
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
    if Fyx is not None: assert 0, "FIXME"
    # Fyx = Fxy
    Fpp = Fxx * lx() ** 2 + Fyy * ly() ** 2 + 2. * Fxy * lx() * ly()
    FOO = Fxx * ly() ** 2 + Fyy * lx() ** 2 - 2. * Fxy * lx() * ly()
    # FIXME: is the sign of the following line correct ? (anyway result should be close to zero)
    FpO = lx() * ly() * (Fxx - Fyy) + Fxy * (ly() ** 2 - lx() ** 2)
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
    def __init__(self, lib_dir, lib_datalm, cls_unl, cls_len, cl_transf, cls_noise, lib_skyalm=None):

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

        if not os.path.exists(lib_dir) and lensit.pbs.rank == 0:
            os.makedirs(lib_dir)
        lensit.pbs.barrier()
        if not os.path.exists(lib_dir + '/cov_hash.pk') and lensit.pbs.rank == 0:
            pk.dump(self.hashdict(), open(lib_dir + '/cov_hash.pk', 'w'))
        lensit.pbs.barrier()
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

    def get_RDdelensinguncorrbias(self, lib_qlm, clpp_rec, clsobs_deconv, recache=False):
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
                    Pmat = get_unlPmat_ij('TQU', self.lib_datalm, clsobs_deconv, _i, _j)
                    for a in [0, 1]:
                        for b in [0, 1][a:]:
                            facab = (2. - (a == b))
                            _phiab = lib_qlm.alm2map(clpp_rec[lib_qlm.reduced_ellmat()] * ik_q(a) * ik_q(b) * facab)
                            _map += (_phiab - _phiab[0, 0]) \
                                    * self.lib_datalm.alm2map(Pmat * ik_d(a) * ik_d(b))
                    retalms[_i, _j, :] = (self.lib_datalm.map2alm(_map))
            return TQUPmats2TEBcls(self.lib_datalm, retalms) * (- 1. / np.sqrt(np.prod(self.lsides)))

    def get_delensingcorrbias(self, _type, lib_qlm, ALWFcl, CMBonly=False):
        """
        Let the unnormalized qest be written as d_a = A^{im} X^i B^{a,l m} X^l
        with X in T,Q,U beam ** deconvolved ** maps.
        -> A^{im} = b^2 Cov^{-1}_{m i}
           B^{a, lm} = [ik_a b^2 C_len  Cov^{-1}]_{m l}
        WFcl is the filter applied to the qest prior delensing the maps. (e.g. Cpp / (Cpp + N0) for WF filtering)

        The sign of the output is such that is a positive contr. to C^len - C^unl
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
        ret = np.zeros(self._skyalms_shape(_type), dtype=complex)
        for _i in range(len(_type)):
            ret[_i] = self.lib_skyalm.udgrade(self.lib_datalm, self.lib_datalm.almxfl(ilms[_i], self.cl_transf))
        return ret, 0

    def get_Reslms(self, _type, datalms, use_cls_len=True, use_Pool=0, **kwargs):
        TQUiblms, iter = self.get_iblms(_type, datalms, use_cls_len=use_cls_len, use_Pool=use_Pool, **kwargs)
        return SM.TQU2TEBlms(_type, self.lib_skyalm, TQUiblms)

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

        if ulm0 is None: ulm0 = np.zeros_like(alms)

        criterion = lensit.qcinv.cd_monitors.monitor_basic(dot_op, iter_max=maxiter, eps_min=tol, d0=d0)
        print "++ cd_solve: starting, cond %s " % cond

        iter = lensit.qcinv.cd_solve.cd_solve(ulm0, alms, fwd_op, [pre_ops], dot_op, criterion,
                                              tr_cd, roundoff=cd_roundoff)
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

        (db xi B Cov^{-1} B^t )_{ab}(z) (daxi B Cov^{-1} B^t)^{ba}(z)
        +  (B Cov^-1 B^t)(z) (daxi B Cov^-1 B^t dbxi)(z)

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
        # (xiK) (xiK) + K (xi K xi - xi) - l ==0
        """
        Get linear response mean field matrix : In harmonic space, the mean field
        is given by phi_MF = <\hat phi >_phi = RMF_L phi_{lm}

        *** This is the N0/2-like unnormalised mean field ! ***

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
        return np.array([lib_qlm.udgrade(_lib_qlm, _a) for _a in np.load(fname)])

    def get_MFresplms_old(self, _type, lib_qlm, use_cls_len=True, recache=True):
        """
        Get linear response mean field matrix : In harmonic space, the mean field
        is given by phi_MF = <\hat phi >_phi = RMF_L phi_{lm}

        *** This is the N0/2-like unnormalised mean field ! ***

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
        """
        # FIXME :  strong cancellation between N0 and the other terms
        fname = self.lib_dir + '/%s_MFresplm_%sCls.npy' % (_type, {True: 'len', False: 'unl'}[use_cls_len])
        _lib_qlm = lensit.ffs_covs.ell_mat.ffs_alm_pyFFTW(self.lib_skyalm.ell_mat,
                                                          filt_func=lambda ell: (ell <= 2 * self.lib_skyalm.ellmax))
        if os.path.exists(fname) and not recache:
            return 0.5 * (np.array([lib_qlm.udgrade(_lib_qlm, _a) for _a in np.load(fname)]) \
                          - self.get_qlm_curvature(_type, lib_qlm, use_cls_len=use_cls_len).real)

        cls_cmb = self.cls_len if use_cls_len else self.cls_unl
        _2qlm = lambda _map: _lib_qlm.map2alm(_map)

        def get_bCib(i, j, fac=1.):  # B^t Cov^{-1} B
            return self.lib_datalm.almxfl(self.get_Pmatinv(_type, i, j, use_cls_len=use_cls_len),
                                          fac * self.cl_transf ** 2)

        def get_ikaikbP(lib_alm, i, j, a, b):  # P ika ikb
            assert a in [0, 1] and b in [0, 1], (a, b)
            ka = lib_alm.get_kx if a == 1 else lib_alm.get_ky
            kb = lib_alm.get_kx if b == 1 else lib_alm.get_ky
            return -get_unlPmat_ij(_type, lib_alm, cls_cmb, i, j) * ka() * kb()

        Fxx = np.zeros(_lib_qlm.alm_size, dtype=complex)
        Fyy = np.zeros(_lib_qlm.alm_size, dtype=complex)
        Fxy = np.zeros(_lib_qlm.alm_size, dtype=complex)

        for _i in range(len(_type)):  # Tr (B^t Cov^{-1} B)(z -z') (xi_ab)(z -z')
            for _j in range(len(_type)):
                _BciB = self.lib_skyalm.alm2map(self._upg(get_bCib(_i, _j, fac=1.)))  # 2 - (_i == _j))))
                Fxx += _2qlm(_BciB * self.lib_skyalm.alm2map(get_ikaikbP(self.lib_skyalm, _i, _j, 1, 1)))
                Fyy += _2qlm(_BciB * self.lib_skyalm.alm2map(get_ikaikbP(self.lib_skyalm, _i, _j, 0, 0)))
                Fxy += _2qlm(_BciB * self.lib_skyalm.alm2map(get_ikaikbP(self.lib_skyalm, _i, _j, 1, 0)))
        del _BciB

        # def get_cst():  # delta^D(z -z') Tr (B^t Cov^{-1} B xi_ab)_(z - z')
        # retxx = 0.
        # retyy = 0.
        # retyx = 0.
        # Pmatxx = np.zeros(self.lib_datalm.alm_size, dtype=complex)
        # Pmatyy = np.zeros(self.lib_datalm.alm_size, dtype=complex)
        # Pmatxy = np.zeros(self.lib_datalm.alm_size, dtype=complex)

        # for _i in range(len(_type)):
        # for _j in range(len(_type)):
        # fac = 2 - (_i == _j)
        # fac = 1.
        # Pmatxx += get_bCib(_i, _j, fac=fac) * get_ikaikbP(self.lib_datalm, _j, _i, 1, 1)
        # Pmatyy += get_bCib(_i, _j, fac=fac) * get_ikaikbP(self.lib_datalm, _j, _i, 0, 0)
        # Pmatxy += get_bCib(_i, _j, fac=fac) * get_ikaikbP(self.lib_datalm, _j, _i, 0, 1)


        # fac = (2 - (_i == _j)) * (2 / np.sqrt(np.prod(self.lsides)))
        # retyy += np.sum(get_bCib(_i, _j,fac = fac) * get_ikaikbP(_j, _i, 0, 0))
        # retxx += np.sum(get_bCib(_i, _j,fac = fac) * get_ikaikbP(_j, _i, 1, 1))
        # retyx += fac * np.sum(get_bCib(_i, _j) * get_ikaikbP(_j, _i, 0, 1))
        # lib_full = fs.ell_mat.ffs_alm_pyFFTW(self.lib_skyalm.ell_mat,filt_func=lambda ell : ell >= 0)
        # Dirac = lib_full.alm2map(np.ones(lib_full.alm_size, dtype=complex))
        # retxx = _2qlm(Dirac * self.lib_skyalm.alm2map(self._upg(Pmatxx)))
        # retyy = _2qlm(Dirac * self.lib_skyalm.alm2map(self._upg(Pmatyy)))
        # retxy = _2qlm(Dirac * self.lib_skyalm.alm2map(self._upg(Pmatxy)))

        # return (retyy, retxx, retxy)
        # Factor of two because of redundancy. The pure real freq should not matter here, and the double counting redudancy of kx ==0, ky <>0 neither

        # retyy, retxx, retxy = get_cst()
        # print retxx[0], retyy[0],retxy[0]
        # print Fxx[0],Fyy[0],Fxy[0]
        # FIXME : this should be the k == 0 term. OK !
        # Fxx -= retxx[0]
        # Fyy -= retyy[0]
        # Fxy -= retxy[0]
        assert _lib_qlm.reduced_ellmat()[0] == 0
        Fxx -= Fxx[0]
        Fyy -= Fyy[0]
        Fxy -= Fxy[0]

        facunits = -2. / np.sqrt(np.prod(self.lsides))

        ret = xylms_to_phiOmegalm(_lib_qlm, Fxx.real * facunits, Fyy.real * facunits, Fxy.real * facunits)
        np.save(fname, ret)
        del ret
        return 0.5 * (np.array([lib_qlm.udgrade(_lib_qlm, _a) for _a in np.load(fname)]) \
                      - self.get_qlm_curvature(_type, lib_qlm, use_cls_len=use_cls_len).real)


class ffs_lencov_alm(ffs_diagcov_alm):
    """
    Class for flat sky displaced real_space cov.matrix from full-sky Cls, and deflection field.
    Must be strictly positive definite.
    """

    def __init__(self, lib_dir, lib_datalm, lib_skyalm, cls_unl, cls_len, cl_transf, cls_noise, f, f_inv):
        """
        f and finv are displacement field classes. Number of points on each side 2**HD_res,2**HD_res.
        f and f_inv must have a f.lens_map routine that does the lensing of map 2**HD_res by 2**HD_res.
        f_inv must also have a f.det_M routine which returns the determinant of the magnification matrix
        at all points of the map.
        """

        assert lib_datalm.ell_mat.lsides == lib_skyalm.ell_mat.lsides, \
            (lib_datalm.ell_mat.lsides, lib_skyalm.ell_mat.lsides)

        super(ffs_lencov_alm, self).__init__(lib_dir, lib_datalm, cls_unl, cls_len, cl_transf, cls_noise,
                                             lib_skyalm=lib_skyalm)

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
            import mllens_GPU.apply_GPU as apply_GPU
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
            from mllens_GPU.apply_cond3_GPU import apply_cond3_GPU_inplace as c3GPU
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
        # FIXME : inputs
        if datalms.shape == ((len(_type), self.dat_shape[0], self.dat_shape[1])):
            _datalms = np.array([self.lib_datalm.map2alm(_m) for _m in datalms])
            return self.get_iblms(_type, _datalms, use_cls_len=use_cls_len, use_Pool=use_Pool, **kwargs)

        assert datalms.shape == self._datalms_shape(_type), (datalms.shape, self._datalms_shape(_type))
        ilms, iter = self.cd_solve(_type, datalms, use_Pool=use_Pool, **kwargs)
        ret = np.zeros(self._skyalms_shape(_type), dtype=complex)
        for _i in range(len(_type)):
            ret[_i] = self.lib_skyalm.udgrade(self.lib_datalm, self.lib_datalm.almxfl(ilms[_i], self.cl_transf))
        return ret, iter

    def get_MLlms(self, _type, datmaps, use_Pool=0, **kwargs):
        """
        Returns maximum likelihood sky CMB modes. (P D^t B^t F^t Cov^-1 = (P^-1 + B^F^t N^{-1} F B)^{-1} F B N^{-1} d)
        """
        # FIXME : does not need this for isotropic cov.
        ilms, iter = self.cd_solve(_type, np.array([self.lib_datalm.map2alm(_m) for _m in datmaps]),
                                   use_Pool=use_Pool, **kwargs)
        ret = np.zeros(self._skyalms_shape(_type), dtype=complex)
        assert 0, "FIX THIS :"
        for _i in range(len(_type)):
            bilm = self.lib_datalm.almxfl(ilms[_i], self.cl_transf)
            bilm = self.f_inv.lens_alm(self.lib_skyalm, self._upg(bilm), lib_alm_out=self.lib_skyalm, use_Pool=use_Pool,
                                       mult_magn=True)
            for _j in range(len(_type)):
                ret[_j] += get_unlPmat_ij(_type, self.lib_skyalm, self.cls_unl, _j, _i) * bilm
        return ret

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
        assert 0, 'FIXME'

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
        timer = fs.misc.lens_utils.timer(_timed)
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
