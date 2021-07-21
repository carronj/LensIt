from __future__ import print_function

import datetime
import os
import pickle as pk
import shutil

import numpy as np

from lensit.pbs import pbs
from lensit.qcinv import opfilt_cinv, chain_samples, cd_solve, cd_monitors, multigrid, ffs_ninv_filt_ideal
from lensit.ffs_covs import ell_mat
from lensit.ffs_covs import ffs_specmat as SM
from lensit.ffs_covs import ffs_specmat as pmat
from lensit.misc.misc_utils import timer, cls_hash, npy_hash, cl_inverse, extend_cl
from lensit.sims.sims_generic import hash_check

typs = ['T', 'QU', 'TQU']

_timed = True
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
        print('Fxy Fyx equal, allclose', np.all(Fxy == Fyx), np.allclose(Fxy, Fyx))

    # FIXME: is the sign of the following line correct ? (anyway result should be close to zero)
    return np.array([Fpp, FOO, FpO])


class ffs_diagcov_alm(object):
    """Library for flat-sky calculations of various lensing biases, responses, etc. in a idealized, isotropic case

        Args:
            lib_dir: many things will be saved there
            lib_datalm: lib_alm instance (see *lensit.ffs_covs.ell_mat* containing mode filtering and flat-sky patch info
            cls_unl(dict): unlensed CMB cls
            cls_len(dict): lensed CMB cls
            cl_transf: instrument transfer function
            cls_noise(dict): 't', 'q' and 'u' noise arrays
            lib_skyalm(optional): lib_alm instance describing the sky mode. Irrelevant with some exceptions. Defaults to lib_datalm


    """
    def __init__(self, lib_dir, lib_datalm, cls_unl, cls_len, cl_transf, cls_noise,
                 lib_skyalm=None, init_rank=pbs.rank, init_barrier=pbs.barrier, alpha_cpp=1.0):

        self.lib_datalm = lib_datalm
        self.lib_skyalm = lib_datalm.clone() if lib_skyalm is None else lib_skyalm
        self.cls_unl = cls_unl
        self.cls_len = cls_len
        self.cl_transf = cl_transf
        self.cls_noise = cls_noise
        self.alpha_cpp = alpha_cpp
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
        fn = os.path.join(lib_dir, 'cov_hash.pk')
        if not os.path.exists(fn) and init_rank == 0:
            pk.dump(self.hashdict(), open(fn, 'wb'), protocol=2)
        init_barrier()
        # print(lib_dir)
        # hash_check(pk.load(open(fn, 'rb')), self.hashdict())
        hash_check(pk.load(open(fn, 'rb')), self.hashdict(), keychain=[self.lib_dir])

        self.barrier = pbs.barrier if _runtimebarriers else lambda: -1
        self.pbsrank = 0 if _runtimerankzero else pbs.rank

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

    def _datalms_shape(self, typ):
        assert typ in typs, (typ, typs)
        return len(typ), self.lib_datalm.alm_size

    def _skyalms_shape(self, typ):
        assert typ in typs, (typ, typs)
        return len(typ), self.lib_skyalm.alm_size

    def _datmaps_shape(self, typ):
        assert typ in typs, (typ, typs)
        return len(typ), self.dat_shape[0], self.dat_shape[1]

    def hashdict(self):
        h = {'lib_alm': self.lib_datalm.hashdict(), 'lib_skyalm': self.lib_skyalm.hashdict()}
        for key in self.cls_noise.keys():
            h['cls_noise ' + key] = npy_hash(self.cls_noise[key])
        for key in self.cls_unl.keys():
            h['cls_unl ' + key] = npy_hash(self.cls_unl[key])
        for key in self.cls_len.keys():
            h['cls_len ' + key] =  npy_hash(self.cls_len[key])
        h['cl_transf'] =  npy_hash(self.cl_transf)
        return h

    def _get_Nell(self, field):
        return self.cls_noise[field.lower()][0]


    def degrade(self, LD_shape, ellmin=None, ellmax=None, lib_dir=None, libtodegrade='sky', **kwargs):
        assert 0, 'FIXME'

    def _get_pmati(self, typ, i, j, use_cls_len=True):
        r"""Inverse spectral matrix

        """
        if i < j: return self._get_pmati(typ, j, i, use_cls_len=use_cls_len)
        _str = {True: 'len', False: 'unl'}[use_cls_len]
        fname = os.path.join(self.lib_dir, '%s_Pmatinv_%s_%s%s.npy' % (typ, _str, i, j))
        if not os.path.exists(fname) and self.pbsrank == 0:
            cls_cmb = self.cls_len if use_cls_len else self.cls_unl
            Pmatinv = pmat.get_Pmat(typ, self.lib_datalm, cls_cmb,
                               cl_transf=self.cl_transf, cls_noise=self.cls_noise, inverse=True)
            for _j in range(len(typ)):
                for _i in range(_j, len(typ)):
                    np.save(os.path.join(self.lib_dir, '%s_Pmatinv_%s_%s%s.npy' % (typ, _str, _i, _j)), Pmatinv[:, _i, _j])
                    print("     _get_pmati:: cached", os.path.join(self.lib_dir, '%s_Pmatinv_%s_%s%s.npy' % (typ, _str, _i, _j)))
        self.barrier()
        return np.load(fname)

    def _get_rootpmatsky(self, typ, i, j, use_cls_len=False):
        """Root of sky rfft spectra matrix

        """
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl
        return pmat.get_rootunlPmat_ij(typ, self.lib_skyalm, cls_cmb, i, j)

    def get_delensinguncorrbias(self, lib_qlm, clpp_rec, wNoise=True, wCMB=True, recache=False, use_cls_len=True):
        r"""Calculate delensing bias given a reconstructed potential map spectrum

            Crudely, the delensing bias is defined as :math:`C_\ell^{\rm lensed} - C_\ell^{\rm delensed}` on Gaussian CMB maps (with lensed power spectrum).
            More precisely this method calculates the 4-point disconnected contribution (see https://arxiv.org/abs/1701.01712)
            if no statistical dependence of the lensing tracer reconstruction noise to the CMB maps

            Args:
                lib_qlm: *ffs_alm* instance describing the lensing alm arrays
                clpp_rec: lensing tracer power.
                          For the true bias calculation *clpp_rec* should be reconstruction noise power only.
                          For perturbative lensing calculation, include the signal power in *clpp_rec*
                wNoise: include noise spectra in total observed CMB spectra (defaults to True)
                wCMB: include signal CMB spectra in total observed CMB spectra (defaults to True)
                use_cls_len: use lensed CMB cls if set, unlensed if not (defaults to True)


            Returns:
                (3, 3, lmax +1) array with bias :math:`C_\ell^{ij}, \textrm{ with }i,j \in (T,E,B)`

        """
        # assert len(clpp_rec) > lib_qlm.ellmax,(len(clpp_rec),lib_qlm.ellmax)
        if len(clpp_rec) <= lib_qlm.ellmax: clpp_rec = extend_cl(clpp_rec, lib_qlm.ellmax)
        fname = os.path.join(self.lib_dir, 'TEBdelensUncorrBias_wN%s_w%sCMB%s_%s_%s.dat' \
                               % (wNoise, {True: 'len', False: 'unl'}[use_cls_len],
                                  wCMB,  npy_hash(clpp_rec[lib_qlm.ellmin:lib_qlm.ellmax + 1]),
                                  lib_qlm.filt_hash()))
        # print(fname)
        if (not os.path.exists(fname) or recache) and self.pbsrank == 0:
            def ik_q(a):
                assert a in [0, 1], a
                return lib_qlm.get_ikx() if a == 1 else lib_qlm.get_iky()

            def ik_d(a):
                assert a in [0, 1], a
                return self.lib_datalm.get_ikx() if a == 1 else self.lib_datalm.get_iky()

            retalms = np.zeros((3, 3, self.lib_datalm.alm_size), dtype=complex)
            cls_noise = {}
            for key in self.cls_noise.keys():
                cls_noise[key] = self.cls_noise[key] / self.cl_transf[:len(self.cls_noise[key])] ** 2 * wNoise
            cmb_cls = self.cls_len if use_cls_len else self.cls_unl
            for _i in range(3):
                for _j in range(_i, 3):
                    if wCMB or (_i == _j):
                        _map = np.zeros(self.dat_shape, dtype=float)
                        if wCMB:
                            Pmat = pmat.get_datPmat_ij('TQU', self.lib_datalm, cmb_cls, np.ones_like(self.cl_transf),
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
            retalms = pmat.TQUPmats2TEBcls(self.lib_datalm, retalms) * (- 1. / np.sqrt(np.prod(self.lsides)))
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
            print("Cached " + fname)
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
        #putting cls_obs being cls_len + noise / transf ** 2 should give the same thing as get_delensinguncorrbias.
        if len(clpp_rec) <= lib_qlm.ellmax: clpp_rec = extend_cl(clpp_rec, lib_qlm.ellmax)
        fname = os.path.join(self.lib_dir, 'TEBdelensUncorrBias_RD_%s_%s.dat' \
                               % (npy_hash(clpp_rec[lib_qlm.ellmin:lib_qlm.ellmax + 1]),
                                  lib_qlm.filt_hash()))
        
        if (not os.path.exists(fname) or recache) and self.pbsrank == 0:
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
                    Pmat1 = pmat.get_unlPmat_ij('TQU', self.lib_datalm, clsobs_deconv, _i, _j)
                    Pmat2 = Pmat1 if clsobs_deconv2 is None else  pmat.get_unlPmat_ij('TQU', self.lib_datalm, clsobs_deconv2,
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
                
                
            retalms = pmat.TQUPmats2TEBcls(self.lib_datalm, retalms) * (- 1. / np.sqrt(np.prod(self.lsides)))
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
            header += "\n TT TE TB EE EB BB"
            np.savetxt(fname, save_arr.transpose(), fmt=['%.8e'] * 6, header=header)
            print("Cached " + fname)
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
            # return pmat.TQUPmats2TEBcls(self.lib_datalm, retalms) * (- 1. / np.sqrt(np.prod(self.lsides)))

    def get_delensingcorrbias(self, typ, lib_qlm, alwfcl, CMBonly=False):
        r"""Calculate delensing bias given a reconstructed potential map spectrum

            Crudely, the delensing bias is defined as :math:`C_\ell^{\rm lensed} - C_\ell^{\rm delensed}` on Gaussian CMB maps (with lensed power spectrum).
            More precisely this method calculates the 4-point disconnected contribution (see https://arxiv.org/abs/1701.01712)
            from the statistical dependence of the lensing tracer reconstruction noise to the CMB maps

            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                lib_qlm: *ffs_alm* instance describing the lensing alm arrays
                alwfcl: normalization of the Wiener-filter quadratic estimate (i.e. Wiener-filter times inverse response)
                CMBonly: do not include noise spectra if set (defaults to False)

            Returns:
                (3, 3, lmax +1) array with bias :math:`C_\ell^{ij}, \textrm{ with }i,j \in (T,E,B)`

        """
        assert typ in typs, (typ, typs)

        t = timer(_timed)
        t.checkpoint("delensing bias : Just started")
        retalms = np.zeros((3, 3, self.lib_datalm.alm_size), dtype=complex)
        cls_noise = {}

        for key in self.cls_noise.keys():
            cls_noise[key] = self.cls_noise[key] / self.cl_transf[:len(self.cls_noise[key])] ** 2 * (not CMBonly)

        def get_datcl(l, j):  # beam deconvolved Cls of the TQU data maps
            # The first index is one in th qest estimator of type 'typ' and
            # the second any in TQU.
            assert (l in range(len(typ))) and (j in range(3)), (l, j)
            if typ == 'QU':
                l_idx = l + 1
            elif typ == 'T':
                l_idx = 0
            else:
                assert typ == 'TQU'
                l_idx = l
            return pmat.get_datPmat_ij(
                'TQU', self.lib_datalm, self.cls_len, np.ones_like(self.cl_transf), cls_noise, l_idx, j)

        def _get_Balm(a, l, m):  # [ik_a b^2 C_len  Cov^{-1}]_{m l}
            # Here both indices should refer to the qest.
            assert a in [0, 1], a
            assert (l in range(len(typ))) and (m in range(len(typ))), (l, m)
            ik = self.lib_datalm.get_ikx if a == 1 else self.lib_datalm.get_iky
            ret = pmat.get_unlPmat_ij(typ, self.lib_datalm, self.cls_len, m, 0) \
                  * self._get_pmati(typ, 0, l, use_cls_len=True)
            for _i in range(1, len(typ)):
                ret += pmat.get_unlPmat_ij(typ, self.lib_datalm, self.cls_len, m, _i) \
                       * self._get_pmati(typ, _i, l, use_cls_len=True)
            return self.lib_datalm.almxfl(ret, self.cl_transf ** 2) * ik()

        def _get_Akm(l, m):  # [b^2  Cov^{-1}]_{m k}
            # Here both indices should refer to the qest.
            assert (l in range(len(typ))) and (m in range(len(typ))), (l, m)
            return self.lib_datalm.almxfl(self._get_pmati(typ, m, l, use_cls_len=True), self.cl_transf ** 2)

        def get_BCamj(a, m, j):  # sum_l B^{a, l m} \hat C^{lj}
            # The first index m is one in th qest estimator of type 'typ' and
            # the second any in TQU.
            assert a in [0, 1], a
            assert m in range(len(typ)), (m, typ)
            assert j in range(3), j
            ret = _get_Balm(a, 0, m) * get_datcl(0, j)
            for _i in range(1, len(typ)):
                ret += _get_Balm(a, _i, m) * get_datcl(_i, j)
            return ret

        def get_ACmj(m, j):  # sum_k A^{k m} \hat C^{kj}
            # The first index m is one in th qest estimator of type 'typ' and
            # the second any in TQU.
            assert m in range(len(typ)), (m, typ)
            assert j in range(3), j
            ret = _get_Akm(0, m) * get_datcl(0, j)
            for _i in range(1, len(typ)):
                ret += _get_Akm(_i, m) * get_datcl(_i, j)
            return ret

        def ik(a, libalm=self.lib_datalm):
            assert a in [0, 1], a
            return libalm.get_ikx() if a == 1 else libalm.get_iky()

        _map = lambda _alm: self.lib_datalm.alm2map(_alm)
        for a in [0, 1]:
            for b in [0, 1]:
                t.checkpoint("    Doing axes %s %s" % (a, b))
                Hab = lib_qlm.alm2map(alwfcl[lib_qlm.reduced_ellmat()] * ik(a, libalm=lib_qlm) * ik(b, libalm=lib_qlm))
                for _i in range(3):  # Building TQU biases, before rotation to Gradient / Curl
                    for _j in range(0, 3):
                        t.checkpoint(
                            "          Doing %s" % ({0: 'T', 1: 'Q', 2: 'U'}[_i] + {0: 'T', 1: 'Q', 2: 'U'}[_j]))
                        # Need a sum over a and m
                        for m in range(len(typ)):  # Hab(z) * [ (AC_m_dai)(-z) (BC_bmj) + (AC_mj)(-z) (BC_bmdai)]
                            Pmat = (get_ACmj(m, _i) * ik(a)).conjugate()
                            retalms[_i, _j, :] += self.lib_datalm.map2alm(Hab * _map(Pmat)) * get_BCamj(b, m, _j)
                            Pmat = (get_BCamj(b, m, _i) * ik(a)).conjugate()
                            retalms[_i, _j, :] += self.lib_datalm.map2alm(Hab * _map(Pmat)) * get_ACmj(m, _j)
        norm = 1. / np.sqrt(np.prod(self.lsides))
        # Sure that i - j x-y is the same ?
        for i in range(3):  # Building TQU biases, before rotation to Gradient / Curl
            for j in range(i, 3):
                print(typ + ' :', np.allclose(retalms[j, i, :].real, retalms[i, j, :].real))
                retalms[i, j, :] += retalms[j, i, :].conjugate()

        return pmat.TQUPmats2TEBcls(self.lib_datalm, retalms) * norm

    def get_RDdelensingcorrbias(self, typ, lib_qlm, alwfcl, clsobs_deconv, clsobs_deconv2=None, cls_weights=None):
        r"""Calculate delensing bias given a reconstructed potential map spectrum

            Same as *get_delensingcorrbias* using empirical input CMB spectra

            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                lib_qlm: *ffs_alm* instance describing the lensing alm arrays
                alwfcl: normalization of the Wiener-filter quadratic estimate (i.e. Wiener-filter times inverse response)
                clsobs_deconv(dict): emprical beam-deconvolved CMB data spectra.

            Returns:
                (3, 3, lmax +1) array with bias :math:`C_\ell^{ij}, \textrm{ with }i,j \in (T,E,B)`

        """

        assert typ in typs, (typ, typs)

        t = timer(_timed, suffix=' (delensing RD corr. Bias)')
        retalms = np.zeros((3, 3, self.lib_datalm.alm_size), dtype=complex)
        _cls_weights = cls_weights or self.cls_len
        _clsobs_deconv2 = clsobs_deconv2 or clsobs_deconv
        if cls_weights is not None: t.checkpoint("Using custom cls weights")

        def get_datcl(l, j, id):  # beam deconvolved Cls of the TQU data maps
            # The first index is one in th qest estimator of type 'typ' and
            # the second any in TQU.
            assert (l in range(len(typ))) and (j in range(3)), (l, j)
            assert id == 1 or id == 2, id
            if typ == 'QU':
                l_idx = l + 1
            elif typ == 'T':
                l_idx = 0
            else:
                assert typ == 'TQU'
                l_idx = l
            _cmb_cls = clsobs_deconv if id == 1 else _clsobs_deconv2
            return pmat.get_unlPmat_ij('TQU', self.lib_datalm, _cmb_cls, l_idx, j)

        def _get_Balm(a, l, m):  # [ik_a b^2 C_len  Cov^{-1}]_{m l}
            # Here both indices should refer to the qest.
            assert a in [0, 1], a
            assert (l in range(len(typ))) and (m in range(len(typ))), (l, m)
            ik = self.lib_datalm.get_ikx if a == 1 else self.lib_datalm.get_iky
            ret = pmat.get_unlPmat_ij(typ, self.lib_datalm, _cls_weights, m, 0) \
                  * self._get_pmati(typ, 0, l, use_cls_len=True)
            for _i in range(1, len(typ)):
                ret += pmat.get_unlPmat_ij(typ, self.lib_datalm, _cls_weights, m, _i) \
                       * self._get_pmati(typ, _i, l, use_cls_len=True)
            return self.lib_datalm.almxfl(ret, self.cl_transf ** 2) * ik()

        def _get_Akm(l, m):  # [b^2  Cov^{-1}]_{m k}
            # Here both indices should refer to the qest.
            assert (l in range(len(typ))) and (m in range(len(typ))), (l, m)
            return self.lib_datalm.almxfl(self._get_pmati(typ, m, l, use_cls_len=True), self.cl_transf ** 2)

        def get_BCamj(a, m, j):  # sum_l B^{a, l m} \hat C^{lj}
            # The first index m is one in th qest estimator of type 'typ' and
            # the second any in TQU.
            assert a in [0, 1], a
            assert m in range(len(typ)), (m, typ)
            assert j in range(3), j
            ret = _get_Balm(a, 0, m) * get_datcl(0, j, 2)
            for _i in range(1, len(typ)):
                ret += _get_Balm(a, _i, m) * get_datcl(_i, j, 2)
            return ret

        def get_ACmj(m, j):  # sum_k A^{k m} \hat C^{kj}
            # The first index m is one in th qest estimator of type 'typ' and
            # the second any in TQU.
            assert m in range(len(typ)), (m, typ)
            assert j in range(3), j
            ret = _get_Akm(0, m) * get_datcl(0, j, 1)
            for _i in range(1, len(typ)):
                ret += _get_Akm(_i, m) * get_datcl(_i, j, 1)
            return ret

        def ik(a, libalm=self.lib_datalm):
            assert a in [0, 1], a
            return libalm.get_ikx() if a == 1 else libalm.get_iky()

        _map = lambda _alm: self.lib_datalm.alm2map(_alm)
        for a in [0, 1]:
            for b in [0, 1]:
                t.checkpoint("  Doing axes %s %s" % (a, b))
                Hab = lib_qlm.alm2map(alwfcl[lib_qlm.reduced_ellmat()] * ik(a, libalm=lib_qlm) * ik(b, libalm=lib_qlm))
                for _i in range(3):  # Building TQU biases, before rotation to Gradient / Curl
                    for _j in range(0, 3):
                        t.checkpoint("    Doing %s" % ({0: 'T', 1: 'Q', 2: 'U'}[_i] + {0: 'T', 1: 'Q', 2: 'U'}[_j]))
                        # Need a sum over a and m
                        for m in range(len(typ)):  # Hab(z) * [ (AC_m_dai)(-z) (BC_bmj) + (AC_mj)(-z) (BC_bmdai)]
                            Pmat = (get_ACmj(m, _i) * ik(a)).conjugate()
                            retalms[_i, _j, :] += self.lib_datalm.map2alm(Hab * _map(Pmat)) * get_BCamj(b, m, _j)
                            Pmat = (get_BCamj(b, m, _i) * ik(a)).conjugate()
                            retalms[_i, _j, :] += self.lib_datalm.map2alm(Hab * _map(Pmat)) * get_ACmj(m, _j)
        norm = 1. / np.sqrt(np.prod(self.lsides))  # ?
        # Sure that i - j x-y is the same ?
        for _i in range(3):  # Building TQU biases, before rotation to Gradient / Curl
            for _j in range(_i, 3):
                if _i == 0 and _j == 0:
                    print("Testing conjecture that in the MV case this is symmetric :")
                print( typ + ' :', np.allclose(retalms[_j, _i, :].real, retalms[_i, _j, :].real))
                retalms[_i, _j, :] += retalms[_j, _i, :].conjugate()

        return pmat.TQUPmats2TEBcls(self.lib_datalm, retalms) * norm

    def _apply_beams(self, typ, alms):
        assert alms.shape == self._skyalms_shape(typ), (alms.shape, self._skyalms_shape(typ))
        ret = np.empty_like(alms)
        for _i in range(len(typ)): ret[_i] = self.lib_skyalm.almxfl(alms[_i], self.cl_transf)
        return ret

    def apply(self, typ, alms, **kwargs):
        assert alms.shape == self._datalms_shape(typ), (alms.shape, self._datalms_shape(typ))
        ret = np.zeros_like(alms)
        for j in range(len(typ)):
            for i in range(len(typ)):
                ret[i] += \
                    pmat.get_datPmat_ij(typ, self.lib_datalm, self.cls_unl, self.cl_transf, self.cls_noise, i, j) * alms[j]
        return ret

    def apply_noise(self, typ, alms, inverse=False):
        #Apply noise matrix or its inverse to inputs uqlms.
        assert typ in typs, typ
        assert alms.shape == self._datalms_shape(typ), (alms.shape, self._datalms_shape(typ))

        ret = np.zeros_like(alms)
        for _i in range(len(typ)):
            _cl = self.cls_noise[typ[_i].lower()] if not inverse else 1. / self.cls_noise[typ[_i].lower()]
            ret[_i] = self.lib_datalm.almxfl(alms[_i], _cl)
        return ret

    def get_mllms(self, typ, datmaps, use_cls_len=True, use_Pool=0, **kwargs):
        r"""Returns maximum likelihood sky CMB modes.

            This instance uses isotropic filtering

            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                datmaps: data real-space maps array
                use_cls_len: use lensed cls in filtering (defaults to True)

            Returns:
                T,E,B alm array


        """
        ilms = self.apply_conddiagcl(typ, np.array([self.lib_datalm.map2alm(m) for m in datmaps]),
                                     use_Pool=use_Pool, use_cls_len=use_cls_len)
        cmb_cls = self.cls_len if use_cls_len else self.cls_unl
        for i in range(len(typ)): ilms[i] = self.lib_datalm.almxfl(ilms[i], self.cl_transf)
        ilms = SM.apply_TEBmat(typ, self.lib_datalm, cmb_cls, SM.TQU2TEBlms(typ, self.lib_datalm, ilms))
        return np.array([self.lib_skyalm.udgrade(self.lib_datalm, _alm) for _alm in ilms])

    def get_iblms(self, typ, datalms, use_cls_len=True, use_Pool=0, **kwargs):
        r"""Inverse-variance filters input CMB maps

            Produces :math:`B^t \rm{Cov}^{-1} X^{\rm dat}` (inputs to quadratic estimator routines)
            This instance applies isotropic filtering, with no deflection field

            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                datalms: data alms array
                use_cls_len: use lensed cls in filtering (defaults to True)

            Returns:
                inverse-variance filtered alms (T and/or Q, Ulms)

        """
        if datalms.shape == ((len(typ), self.dat_shape[0], self.dat_shape[1])):
            _datalms = np.array([self.lib_datalm.map2alm(_m) for _m in datalms])
            return self.get_iblms(typ, _datalms, use_cls_len=use_cls_len, use_Pool=use_Pool, **kwargs)
        assert datalms.shape == self._datalms_shape(typ), (datalms.shape, self._datalms_shape(typ))
        ilms = self.apply_conddiagcl(typ, datalms, use_Pool=use_Pool, use_cls_len=use_cls_len)
        ret = np.empty(self._skyalms_shape(typ), dtype=complex)
        for _i in range(len(typ)):
            ret[_i] = self.lib_skyalm.udgrade(self.lib_datalm, self.lib_datalm.almxfl(ilms[_i], self.cl_transf))
        return ret, 0


    def cd_solve(self, typ, alms, cond='3', maxiter=50, ulm0=None,
                 use_Pool=0, tol=1e-5, tr_cd=cd_solve.tr_cg, cd_roundoff=25, d0=None):
        #Solves for (F B D xi_unl D^t B^t F^t + N)^-1 dot input alms
        assert alms.shape == self._datalms_shape(typ), (alms.shape, self._datalms_shape(typ))
        if ulm0 is not None:
            assert ulm0.shape == self._datalms_shape(typ), (ulm0.shape, self._datalms_shape(typ))

        class dot_op():
            def __init__(self):
                pass

            def __call__(self, alms1, alms2, **kwargs):
                return np.sum(alms1.real * alms2.real + alms1.imag * alms2.imag)

        cond_func = getattr(self, 'apply_cond%s' % cond)

        # ! fwd_op and pre_ops must not change their arguments !
        def fwd_op(_alms):
            return self.apply(typ, _alms, use_Pool=use_Pool)

        def pre_ops(_alms):
            return cond_func(typ, _alms, use_Pool=use_Pool)

        dot_op = dot_op()

        if d0 is None:
            d0 = dot_op(alms, alms)
        if ulm0 is None: ulm0 = np.zeros_like(alms)
        criterion = cd_monitors.monitor_basic(dot_op, iter_max=maxiter, eps_min=tol, d0=d0)
        print("++ ffs_cov cd_solve: starting, cond %s " % cond)

        it = cd_solve.cd_solve(ulm0, alms, fwd_op, [pre_ops], dot_op, criterion, tr_cd,
                                              roundoff=cd_roundoff)
        return ulm0, it

    def _apply_cond3(self, typ, alms, use_Pool=0):
        return self.apply_conddiagcl(typ, alms, use_Pool=use_Pool)

    def apply_cond0(self, typ, alms, use_Pool=0, use_cls_len=True):
        return self.apply_conddiagcl(typ, alms, use_Pool=use_Pool, use_cls_len=use_cls_len)

    def apply_cond0unl(self, typ, alms, **kwargs):
        return self.apply_conddiagcl(typ, alms, use_cls_len=False)

    def apply_cond0len(self, typ, alms, **kwargs):
        return self.apply_conddiagcl(typ, alms, use_cls_len=True)

    def apply_conddiagcl(self, typ, alms, use_cls_len=True, use_Pool=0):
        assert alms.shape == self._datalms_shape(typ), (alms.shape, self._datalms_shape(typ))
        ret = np.zeros_like(alms)
        for i in range(len(typ)):
            for j in range(len(typ)):
                ret[j] += self._get_pmati(typ, j, i, use_cls_len=use_cls_len) * alms[i]
        return ret

    def apply_condpseudiagcl(self, typ, alms, use_Pool=0):
        return self.apply_conddiagcl(typ, alms, use_Pool=use_Pool)

    def get_qlms(self, typ, iblms, lib_qlm, use_cls_len=True, **kwargs):
        r"""Unormalized quadratic estimates (potential and curl).

        Note:
            the output differs by a factor of two from the standard QE. This is because this was written initially
            as gradient function of the CMB likelihood w.r.t. real and imaginary parts. So to use this as QE's,
            the normalization is 1/2 the standard normalization the inverse response. The *lensit.ffs_qlms.qlms.py* module contains methods
            of the QE's with standard normalizations which you may want to use instead.

        Args:
            typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
            iblms: inverse-variance filtered CMB alm arrays
            lib_qlm: *ffs_alm* instance describing the lensing alm arrays
            use_cls_len: use lensed or unlensed cls in QE weights (numerator), defaults to lensed cls


        Note Louis: 
            cblms: Wiener filtered map ?
            get_ikx and get_iky: 1j * kx or 1j * ky 
        """
        assert iblms.shape == self._skyalms_shape(typ), (iblms.shape, self._skyalms_shape(typ))
        assert lib_qlm.lsides == self.lsides, (self.lsides, lib_qlm.lsides)

        t = timer(_timed)

        weights_cls = self.cls_len if use_cls_len else self.cls_unl
        clms = np.zeros((len(typ), self.lib_skyalm.alm_size), dtype=complex)
        for _i in range(len(typ)):
            for _j in range(len(typ)):
                clms[_i] += pmat.get_unlPmat_ij(typ, self.lib_skyalm, weights_cls, _i, _j) * iblms[_j]

        t.checkpoint("  get_qlms::mult with %s Pmat" % ({True: 'len', False: 'unl'}[use_cls_len]))

        _map = lambda alm: self.lib_skyalm.alm2map(alm)
        _2qlm = lambda _m: lib_qlm.udgrade(self.lib_skyalm, self.lib_skyalm.map2alm(_m))

        # retdx = g_a^QD(n) = IVF * (grad WF) 
        retdx = _2qlm(_map(iblms[0]) * _map(clms[0] * self.lib_skyalm.get_ikx()))
        retdy = _2qlm(_map(iblms[0]) * _map(clms[0] * self.lib_skyalm.get_iky()))
        for _i in range(1, len(typ)):
            retdx += _2qlm(_map(iblms[_i]) * _map(clms[_i] * self.lib_skyalm.get_ikx()))
            retdy += _2qlm(_map(iblms[_i]) * _map(clms[_i] * self.lib_skyalm.get_iky()))

        t.checkpoint("  get_qlms::cartesian gradients")
        # dphi = -1j L \cdot g_L
        dphi = -retdx * lib_qlm.get_ikx() - retdy * lib_qlm.get_iky()
        dOm = retdx * lib_qlm.get_iky() - retdy * lib_qlm.get_ikx()

        t.checkpoint("  get_qlms::rotation to phi Omega")

        return np.array([2 * dphi, 2 * dOm])  # Factor 2 since gradient w.r.t. real and imag. parts.

    def  _get_qlm_resprlm(self, typ, lib_qlm,
                          use_cls_len=True, cls_obs=None, cls_obs2=None, cls_filt=None, cls_weights=None, verbose=False):
        assert typ in typs, (typ, typs)
        t = timer(_timed)
        Fpp, FOO, FpO = self._get_qlm_curvature(typ, lib_qlm,
                                                use_cls_len=use_cls_len, cls_obs=cls_obs, cls_filt=cls_filt, cls_weights=cls_weights, cls_obs2=cls_obs2)
        if verbose:
            t.checkpoint("  get_qlm_resplm:: get curvature matrices")

        del FpO
        Rpp = np.zeros_like(Fpp)
        ROO = np.zeros_like(FOO)
        Rpp[np.where(Fpp > 0.)] = 1. / Fpp[np.where(Fpp > 0.)]
        ROO[np.where(FOO > 0.)] = 1. / FOO[np.where(FOO > 0.)]
        if verbose:
            t.checkpoint("  get_qlm_resplm:: inversion")

        return Rpp, ROO

    def get_N0cls(self, typ, lib_qlm,
                  use_cls_len=True, cls_obs=None, cls_obs2=None, cls_weights=None, cls_filt=None):
        r"""Lensing Gaussian bias :math:`N^{(0)}_L`

            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                lib_qlm: *ffs_alm* instance decribing the lensing alm arrays
                cls_obs(dict, optional): empirical observed data spectra, for a realization-dependent estimate
                cls_weights(dict, optional): CMB cls entering the QE weights (numerator)
                cls_filt(dict, optional): CMB cls entering the inverse-variance filtering step (denominator of QE weights)
                use_cls_len(optional): Uses lensed or unlensed CMB cls (when not superseeded byt the other keywords)

            Returns:
                Gradient and curl lensing mode noise :math:`N^{(0)}_L`


        """
        assert typ in typs, (typ, typs)
        if cls_obs is None and cls_obs2 is None and cls_weights is None and cls_filt is None:  # default behavior is cached
            fname = self.lib_dir + '/%s_N0cls_%sCls.dat' % (typ, {True: 'len', False: 'unl'}[use_cls_len])
            if not os.path.exists(fname):
                if self.pbsrank == 0:
                    lib_full = ell_mat.ffs_alm_pyFFTW(self.lib_datalm.ell_mat, filt_func=lambda ell: ell > 0)
                    Rpp, ROO = self._get_qlm_resprlm(typ, lib_full, use_cls_len=use_cls_len)
                    header = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n' + __file__
                    np.savetxt(fname, np.array((2 * lib_full.alm2cl(np.sqrt(Rpp)), 2 * lib_full.alm2cl(np.sqrt(ROO)))).transpose(),fmt=['%.8e'] * 2, header=header)
                self.barrier()
            cl = np.loadtxt(fname).transpose()[:, 0:lib_qlm.ellmax + 1]
            cl[0] *= lib_qlm.filt_func(np.arange(len(cl[0])))
            cl[1] *= lib_qlm.filt_func(np.arange(len(cl[1])))
            return cl
        else:
            fname = self.lib_dir + '/%s_RDN0cls_%sCls_%s_%s.dat' % (typ, {True: 'len', False: 'unl'}[use_cls_len], 
                npy_hash(cls_obs['tt'][lib_qlm.ellmin:lib_qlm.ellmax + 1]), lib_qlm.filt_hash())
            if not os.path.exists(fname):
                if self.pbsrank == 0:
                    Rpp, ROO = self._get_qlm_resprlm(typ, lib_qlm,
                                                    use_cls_len=use_cls_len, cls_obs=cls_obs, cls_weights=cls_weights, cls_filt=cls_filt, cls_obs2=cls_obs2)
                    header = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n' + __file__
                
                    np.savetxt(fname, np.array((2 * lib_qlm.alm2cl(np.sqrt(Rpp)), 2 * lib_qlm.alm2cl(np.sqrt(ROO)))).transpose(), fmt=['%.8e'] * 2, header=header)
                self.barrier()
            cl = np.loadtxt(fname).transpose()[:, 0:lib_qlm.ellmax + 1]
            cl[0] *= lib_qlm.filt_func(np.arange(len(cl[0])))
            cl[1] *= lib_qlm.filt_func(np.arange(len(cl[1])))
            return cl

    def iterateN0cls(self, typ, lib_qlm, itmax, return_delcls=False, _it=0, _cpp=None):
        """Iterative flat-sky :math:`N^{(0)}_L` calculation to estimate the noise levels of the iterative estimator.
            This uses perturbative approach in Wiener-filtered displacement, consistent with box shape and mode structure.
            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                lib_qlm: *ffs_alm* instance describing the lensing alm arrays
                itmax: Number of iterations to performs
                return_delcls: optionally return partially delensed cmb cls as well
        """
        N0 = self.get_N0cls(typ, lib_qlm, use_cls_len=True)[0][:lib_qlm.ellmax + 1]
        if _it == itmax: return N0 if not return_delcls else (N0, self.cls_len)
        if _cpp is None:
            _cpp = np.copy(self.cls_unl['pp'])
        cpp = np.zeros(lib_qlm.ellmax + 1)
        cpp[:min(len(cpp), len(_cpp))] = (_cpp[:min(len(cpp), len(_cpp))])
        clWF = cpp * cl_inverse(cpp + N0[:lib_qlm.ellmax + 1])
        Bl = self.get_delensinguncorrbias(lib_qlm, cpp * (1. - clWF), wNoise=False, use_cls_len=False)  # TEB matrix output
        # Bl = self.get_delensinguncorrbias(lib_qlm, cpp * (1. - clWF), wNoise=False, use_cls_len=True)  # TEB matrix output
        cls_delen = {}
        for key in self.cls_len.keys():
            cls_delen[key] = self.cls_unl[key].copy()
            _Bl = Bl[{'t': 0, 'e': 1, 'b': 2}[key[0]], {'t': 0, 'e': 1, 'b': 2}[key[1]]]
            cls_delen[key][:min(len(cls_delen[key]),len(_Bl))] -= _Bl[:min(len(cls_delen[key]),len(_Bl))]
        cls_unl = {}
        for key in self.cls_unl.keys():
            cls_unl[key] = self.cls_unl[key].copy()
        # cls_unl['pp'][0:min(len(cpp), len(cls_unl['pp']))] = (cpp * (1. - clWF))[0:min(len(cpp), len(cls_unl['pp']))]
        new_libdir = os.path.join(self.lib_dir, '%s_N0iter' % typ, 'N0iter%04d' % (_it + 1)) if _it == 0 else \
            self.lib_dir.replace('N0iter%04d' % _it, 'N0iter%04d' % (_it + 1))

        try:
            new_cov = ffs_diagcov_alm(new_libdir, self.lib_datalm, cls_unl, cls_delen, self.cl_transf, self.cls_noise,
                                                              lib_skyalm=self.lib_skyalm)
        except:
            print("hash check failed, removing " + new_libdir)
            shutil.rmtree(new_libdir)
            new_cov = ffs_diagcov_alm(new_libdir, self.lib_datalm, cls_unl, cls_delen, self.cl_transf, self.cls_noise,
                                                              lib_skyalm=self.lib_skyalm)

        return new_cov.iterateN0cls(typ, lib_qlm, itmax, _it=_it + 1, return_delcls=return_delcls, _cpp=_cpp)



    # def iterateN0cls(self, typ, lib_qlm, itmax, return_delcls=False, _it=0, cls_obs=None):
    #     """Iterative flat-sky :math:`N^{(0)}_L` calculation to estimate the noise levels of the iterative estimator.

    #         This uses perturbative approach in Wiener-filtered displacement, consistent with box shape and mode structure.

    #         Args:
    #             typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
    #             lib_qlm: *ffs_alm* instance describing the lensing alm arrays
    #             itmax: Number of iterations to performs
    #             return_delcls: optionally return partially delensed cmb cls as well


    #     """
    #     N0 = self.get_N0cls(typ, lib_qlm, use_cls_len=True)[0][:lib_qlm.ellmax + 1]  # get fiducail N0
    #     # if cls_obs is not None:
    #     #     RDN0 = self.get_N0cls(typ, lib_qlm, use_cls_len=True, cls_obs = cls_obs)[0][:lib_qlm.ellmax + 1] # get RDN0 
        
    #     if _it == itmax:  
    #         if cls_obs is None:
    #             return N0 if not return_delcls else (N0, self.cls_len)
    #         else:
    #             RDN0 = N0**2 * cl_inverse(self.get_N0cls(typ, lib_qlm, use_cls_len=True, cls_obs = cls_obs)[0][:lib_qlm.ellmax + 1]) # get RDN0 
    #             return RDN0 if not return_delcls else (RDN0, self.cls_len)

    #     cpp = np.zeros(lib_qlm.ellmax + 1)
    #     cpp[:min(len(cpp), len(self.cls_unl['pp']))] = (self.cls_unl['pp'][:min(len(cpp), len(self.cls_unl['pp']))])
        
    #     # Always use N0 fiducial for the WF
    #     clWF = cpp * cl_inverse(cpp + N0[:lib_qlm.ellmax + 1])   
    #     if cls_obs is None:
    #         Bl = self.get_delensinguncorrbias(lib_qlm, cpp * (1. - clWF), wNoise=False, use_cls_len=False)  # TEB matrix output
    #     else :
    #         Bl = self.get_RDdelensinguncorrbias(lib_qlm, cpp * (1. - clWF), cls_obs, clsobs_deconv2=cls_obs, recache=False)
        
    #     cls_delen = {}
    #     for key in self.cls_len.keys():
    #         _Bl = Bl[{'t': 0, 'e': 1, 'b': 2}[key[0]], {'t': 0, 'e': 1, 'b': 2}[key[1]]]
    #         if cls_obs is None:
    #             cls_delen[key] = self.cls_unl[key].copy()
    #             cls_delen[key][:min(len(cls_delen[key]),len(_Bl))] -= _Bl[:min(len(cls_delen[key]),len(_Bl))]
    #         else:               
    #             # cls_delen[key] = cls_obs[key].copy() + self.cls_unl[key].copy() - self.cls_len[key].copy()
    #             cls_delen[key] = self.cls_unl[key].copy()
    #             cls_delen[key][:min(len(cls_delen[key]),len(_Bl))] -= _Bl[:min(len(cls_delen[key]),len(_Bl))]
        
    #     cls_unl = {}
    #     for key in self.cls_unl.keys():
    #         cls_unl[key] = self.cls_unl[key].copy()
    #     # cls_unl['pp'][0:min(len(cpp), len(cls_unl['pp']))] = (cpp * (1. - clWF))[0:min(len(cpp), len(cls_unl['pp']))]

    #     if cls_obs is None:        
    #         new_libdir = os.path.join(self.lib_dir, '%s_N0iter' % typ, 'N0iter%04d' % (_it + 1)) if _it == 0 else \
    #             self.lib_dir.replace('N0iter%04d' % _it, 'N0iter%04d' % (_it + 1))
    #     else:
    #         new_libdir = os.path.join(self.lib_dir, '%s_RDN0iter_%s' % (typ, npy_hash(cls_obs['tt'][lib_qlm.ellmin:lib_qlm.ellmax + 1])), 'N0iter%04d' % (_it + 1)) if _it == 0 else \
    #         self.lib_dir.replace('N0iter%04d' % _it, 'N0iter%04d' % (_it + 1))

    #     try:
    #         new_cov = ffs_diagcov_alm(new_libdir, self.lib_datalm, cls_unl, cls_delen, self.cl_transf, self.cls_noise,
    #                                                           lib_skyalm=self.lib_skyalm)
    #     except:
    #         print("hash check failed, removing " + new_libdir)
    #         shutil.rmtree(new_libdir)
    #         new_cov = ffs_diagcov_alm(new_libdir, self.lib_datalm, cls_unl, cls_delen, self.cl_transf, self.cls_noise,
    #                                                           lib_skyalm=self.lib_skyalm)
    #     if cls_obs is None:
    #         return new_cov.iterateN0cls(typ, lib_qlm, itmax, _it=_it + 1, return_delcls=return_delcls, cls_obs=None)
    #     else:
    #         _cls_obs = {}
    #         for key in self.cls_len.keys():
    #             _cls_obs[key] = cls_obs[key].copy() - self.cls_len[key].copy() + self.cls_unl[key].copy()
    #             # _cls_obs[key] = cls_obs[key].copy() 
    #             _cls_obs[key][:min(len(cls_delen[key]),len(_Bl))] -= _Bl[:min(len(cls_delen[key]),len(_Bl))]
    #         return new_cov.iterateN0cls(typ, lib_qlm, itmax, _it=_it + 1, return_delcls=return_delcls, cls_obs=_cls_obs)

    def get_N0Pk_minimal(self, typ, lib_qlm, use_cls_len=True, cls_obs=None):
        # Same as N0cls but binning only in exactly identical frequencies.
        assert typ in typs, (typ, typs)
        Rpp, ROO = self._get_qlm_resprlm(typ, lib_qlm, use_cls_len=use_cls_len, cls_obs=cls_obs)
        return lib_qlm.alm2Pk_minimal(np.sqrt(2 * Rpp)), lib_qlm.alm2Pk_minimal(np.sqrt(2 * ROO))

    def get_response(self, typ, lib_qlm, cls_weights=None, cls_filt=None, cls_cmb=None, use_cls_len=True, verbose=False):
        r"""Lensing quadratic estimator gradient and curl response functions.

            Args:
                cls_filt: CMB spectra used in the filtering procedure ( i.e. those entering :math:`\rm{Cov}^{-1}`).
                          Defaults to *self.cls_len* if set else *self.cls_unl*
                cls_weights(dict): CMB spectra used in the QE weights (those entering the numerator in the usual Okamoto & Hu formulae)
                          Defaults to *self.cls_len* if set else *self.cls_unl*
                cls_cmb(dict): CMB spectra of the sky entering the response contractions (in principle lensed cls or grad-lensed cls)

        """
        assert typ in typs, (typ, typs)
        t = timer(_timed, prefix=__name__, suffix=' curvpOlm')

        _cls_weights = cls_weights or (self.cls_len if use_cls_len else self.cls_unl)
        _cls_filt = cls_filt or (self.cls_len if use_cls_len else self.cls_unl)
        _cls_cmb = cls_cmb or (self.cls_len if use_cls_len else self.cls_unl)
        if not cls_weights is None: t.checkpoint('Using custom Cls weights')
        if not cls_filt is None: t.checkpoint('Using custom Cls filt')
        if not cls_cmb is None: t.checkpoint('Using custom Cls cmb')

        Pinv_obs1 = pmat.get_Pmat(typ, self.lib_datalm, _cls_filt,
                             cls_noise=self.cls_noise, cl_transf=self.cl_transf, inverse=True)
        Pinv_obs2 = Pinv_obs1

        # xi K
        def get_xiK(i, j, id, cls):
            assert id in [1, 2]
            _Pinv_obs = Pinv_obs1 if id == 1 else Pinv_obs2
            ret = pmat.get_unlPmat_ij(typ, self.lib_datalm, cls, i, 0) * _Pinv_obs[:, 0, j]
            for _k in range(1, len(typ)):
                ret += pmat.get_unlPmat_ij(typ, self.lib_datalm, cls, i, _k) * _Pinv_obs[:, _k, j]
            return self.lib_datalm.almxfl(ret, self.cl_transf ** 2)
        # xi^w K xi^cmb
        def get_xiwKxicmb(i, j, id):
            assert id in [1, 2]
            ret = get_xiK(i, 0, id, _cls_weights) * pmat.get_unlPmat_ij(typ, self.lib_datalm, _cls_cmb, 0, j)
            for _k in range(1, len(typ)):
                ret += get_xiK(i, _k, id, _cls_weights) * pmat.get_unlPmat_ij(typ, self.lib_datalm, _cls_cmb, _k, j)
            return ret

        ikx = self.lib_datalm.get_ikx
        iky = self.lib_datalm.get_iky
        if verbose:
            t.checkpoint("  inverse %s Pmats" % ({True: 'len', False: 'unl'}[use_cls_len]))
        F = np.zeros(self.lib_datalm.ell_mat.shape, dtype=float)

        # Calculation of (xi^cmb,b K) (xi^w,a K)
        for i in range(len(typ)):
            for j in range(0, len(typ)):
                # ! Matrix not symmetric for TQU or non identical noises. But xx or yy element ok.#(2 - (i == j)) *
                F +=   self.lib_datalm.alm2map(ikx() * get_xiK(i, j, 1, _cls_cmb)) \
                     * self.lib_datalm.alm2map(ikx() * get_xiK(j, i, 2, _cls_weights))

        Fxx = lib_qlm.map2alm(F)
        F *= 0
        if verbose:
            t.checkpoint("  Fxx , part 1")

        for i in range(len(typ)):
            for j in range(0, len(typ)):
                # ! Matrix not symmetric for TQU or non identical noises. But xx or yy element ok.#(2 - (i == j)) *
                F +=  self.lib_datalm.alm2map(iky() * get_xiK(i, j, 1, _cls_cmb)) \
                     * self.lib_datalm.alm2map(iky() * get_xiK(j, i, 2, _cls_weights))
        Fyy = lib_qlm.map2alm(F)
        F *= 0
        if verbose:
            t.checkpoint("  Fyy , part 1")

        for i in range(len(typ)):
            for j in range(len(typ)):
                # ! BPBCovi Matrix not symmetric for TQU or non identical noises.
                F += self.lib_datalm.alm2map(ikx() * get_xiK(i, j, 1, _cls_cmb)) \
                     * self.lib_datalm.alm2map(iky() * get_xiK(j, i, 2, _cls_weights))
        Fxy = lib_qlm.map2alm(F)
        F *= 0
        if verbose:
            t.checkpoint("  Fxy , part 1")

        # Adding to that (K)(z) (xi^w,a K xi^cmb,b)(z)
        tmap = lambda i, j: self.lib_datalm.alm2map(
            self.lib_datalm.almxfl(Pinv_obs1[:, i, j], self.cl_transf ** 2))

        for i in range(len(typ)):
            for j in range(0, len(typ)):
                F += tmap(i, j) * self.lib_datalm.alm2map(ikx() ** 2 * get_xiwKxicmb(i, j, 2))
        Fxx += lib_qlm.map2alm(F)
        F *= 0
        if verbose:
            t.checkpoint("  Fxx , part 2")

        for i in range(len(typ)):
            for j in range(0, len(typ)):
                F += tmap(i, j) * self.lib_datalm.alm2map(iky() ** 2 * get_xiwKxicmb(i, j, 2))
        Fyy += lib_qlm.map2alm(F)
        F *= 0
        if verbose:
            t.checkpoint("  Fyy , part 2")

        for i in range(len(typ)):
            for j in range(0, len(typ)):
                F += tmap(i, j) * self.lib_datalm.alm2map(iky() * ikx() * get_xiwKxicmb(i, j, 2))
        Fxy += lib_qlm.map2alm(F)
        if verbose:
            t.checkpoint("  Fxy , part 2")

        facunits = -1. / np.sqrt(np.prod(self.lsides))
        return np.array([lib_qlm.bin_realpart_inell(r) for r in xylms_to_phiOmegalm(lib_qlm, Fxx.real * facunits, Fyy.real * facunits, Fxy.real * facunits)])

    def _get_qlm_curvature(self, typ, lib_qlm,
                           cls_weights=None, cls_filt=None, cls_obs=None, cls_obs2=None, use_cls_len=True, verbose=False):
        """Fisher matrix for the displacement components phi and Omega (gradient and curl potentials)


        """
        assert typ in typs, (typ, typs)
        t = timer(_timed, prefix=__name__, suffix=' curvpOlm')

        _cls_weights = cls_weights or (self.cls_len if use_cls_len else self.cls_unl)
        _cls_filt = cls_filt or (self.cls_len if use_cls_len else self.cls_unl)
        _cls_obs = cls_obs or (self.cls_len if use_cls_len else self.cls_unl)
        _cls_obs2 = cls_obs2 or (self.cls_len if use_cls_len else self.cls_unl)


        if not cls_weights is None: t.checkpoint('Using custom Cls weights')
        if not cls_filt is None: t.checkpoint('Using custom Cls filt')
        if not cls_obs is None: t.checkpoint('Using custom Cls obs')
        if not cls_obs2 is None: t.checkpoint('Using custom Cls obs2')

        # Build the inverse covariance matrix part
        # For a standard N0 computation, this will just be cov^{-1}.
        # For RDN0 compuation, it will be cov^{-1} cov_obs cov^{-1}
        # In the case of RDN0 computation, only one of the two inverse covariance matrix is replaced.
        if cls_obs is None:
            assert cls_obs2 is None
            _lib_qlm = ell_mat.ffs_alm_pyFFTW(self.lib_datalm.ell_mat,
                                                              filt_func=lambda ell: (ell > 0) & (
                                                              ell <= 2 * self.lib_datalm.ellmax))
            if cls_obs is None and cls_weights is None and cls_filt is None and cls_obs2 is None:  # default is cached
                fname = os.path.join(self.lib_dir, '%s_resplm_%sCls.npy' % (typ, {True: 'len', False: 'unl'}[use_cls_len]))
                if os.path.exists(fname):
                    return np.array([lib_qlm.udgrade(_lib_qlm, _a) for _a in np.load(fname)])
            Pinv_obs1 = pmat.get_Pmat(typ, self.lib_datalm, _cls_filt,
                                      cls_noise=self.cls_noise, cl_transf=self.cl_transf, inverse=True)
            Pinv_obs2 = Pinv_obs1
        else:

            # FIXME : this will fail if lib_qlm does not have the right shape
            _lib_qlm = lib_qlm
            Covi = pmat.get_Pmat(typ, self.lib_datalm, _cls_filt, cls_noise=self.cls_noise, cl_transf=self.cl_transf,
                            inverse=True)
            Pinv_obs1 = np.array([np.dot(a, b) for a, b in
                                  zip(pmat.get_Pmat(typ, self.lib_datalm, _cls_obs, cls_noise=None, cl_transf=None),
                                      Covi)])
            Pinv_obs1 = np.array([np.dot(a, b) for a, b in zip(Covi, Pinv_obs1)])
            if cls_obs2 is None:
                Pinv_obs2 = Pinv_obs1
            else:
                Pinv_obs2 = np.array([np.dot(a, b) for a, b in
                                      zip(pmat.get_Pmat(typ, self.lib_datalm, _cls_obs2, cls_noise=None, cl_transf=None),
                                          Covi)])
                Pinv_obs2 = np.array([np.dot(a, b) for a, b in zip(Covi, Pinv_obs2)])
            del Covi

        # B xi B^t Cov^{-1} (or Cov^-1 Cov_obs Cov^-1 for semi-analytical N0)
        def get_BPBCovi(i, j, id):
            assert id in [1, 2]
            _Pinv_obs = Pinv_obs1 if id == 1 else Pinv_obs2
            ret = pmat.get_unlPmat_ij(typ, self.lib_datalm, _cls_weights, i, 0) * _Pinv_obs[:, 0, j]
            for _k in range(1, len(typ)):
                ret += pmat.get_unlPmat_ij(typ, self.lib_datalm, _cls_weights, i, _k) * _Pinv_obs[:, _k, j]
            return self.lib_datalm.almxfl(ret, self.cl_transf ** 2)

        def get_BPBCoviP(i, j, id):
            assert id in [1, 2]
            ret = get_BPBCovi(i, 0, id) * pmat.get_unlPmat_ij(typ, self.lib_datalm, _cls_weights, 0, j)
            for _k in range(1, len(typ)):
                ret += get_BPBCovi(i, _k, id) * pmat.get_unlPmat_ij(typ, self.lib_datalm, _cls_weights, _k, j)
            return ret

        def get_BPBCovi_rot(i, j, id):
            assert id in [1, 2]
            _Pinv_obs = Pinv_obs1 if id == 1 else Pinv_obs2
            ret = pmat.get_unlrotPmat_ij(typ, self.lib_datalm, _cls_weights, i, 0) * _Pinv_obs[:, 0, j]
            for _k in range(1, len(typ)):
                ret += pmat.get_unlrotPmat_ij(typ, self.lib_datalm, _cls_weights, i, _k) * _Pinv_obs[:, _k, j]
            return self.lib_datalm.almxfl(ret, self.cl_transf ** 2)

        ikx = self.lib_datalm.get_ikx
        iky = self.lib_datalm.get_iky
        if verbose:
            t.checkpoint("  inverse %s Pmats" % ({True: 'len', False: 'unl'}[use_cls_len]))
        F = np.zeros(self.lib_datalm.ell_mat.shape, dtype=float)
        # 2.1 GB in memory for full sky 16384 ** 2 points. Note however that we can without any loss of accuracy
        # calculate this using a twice as sparse grid, for reasonable input parameters.

        # Calculation of (db xi B Cov^{-1} B^t )_{ab}(z) (daxi B Cov^{-1} B^t)^{ba}(z)
        for i in range(len(typ)):
            for j in range(i, len(typ)):
                # ! BPBCovi Matrix not symmetric for TQU or non identical noises. But xx or yy element ok.
                F += (2 - (i == j)) * self.lib_datalm.alm2map(ikx() * get_BPBCovi(i, j, 1)) \
                     * self.lib_datalm.alm2map(ikx() * get_BPBCovi(j, i, 2))
        Fxx = _lib_qlm.map2alm(F)
        F *= 0
        if verbose:
            t.checkpoint("  Fxx , part 1")

        for i in range(len(typ)):
            for j in range(i, len(typ)):
                # ! BPBCovi Matrix not symmetric for TQU or non identical noises. But xx or yy element ok.
                F += (2 - (i == j)) * self.lib_datalm.alm2map(iky() * get_BPBCovi(i, j, 1)) \
                     * self.lib_datalm.alm2map(iky() * get_BPBCovi(j, i, 2))
        Fyy = _lib_qlm.map2alm(F)
        F *= 0
        if verbose:
            t.checkpoint("  Fyy , part 1")

        for i in range(len(typ)):
            for j in range(len(typ)):
                # ! BPBCovi Matrix not symmetric for TQU or non identical noises.
                F += self.lib_datalm.alm2map(ikx() * get_BPBCovi(i, j, 1)) \
                     * self.lib_datalm.alm2map(iky() * get_BPBCovi(j, i, 2))
        Fxy = _lib_qlm.map2alm(F)
        F *= 0
        if verbose:
            t.checkpoint("  Fxy , part 1")

        # Adding to that (B Cov^-1 B^t)(z) (daxi B Cov^-1 B^t dbxi)(z)
        # Construct Pmat:
        #  Cl * bl ** 2 * cov^{-1} cov_obs cov^{-1} * Cl if semianalytic N0
        #  Cl * bl ** 2 * cov^{-1} * Cl if N0
        #  Now both spectral matrices are symmetric.
        tmap = lambda i, j: self.lib_datalm.alm2map(
            self.lib_datalm.almxfl(Pinv_obs1[:, i, j], (2 - (j == i)) * self.cl_transf ** 2))

        for i in range(len(typ)):
            for j in range(i, len(typ)):
                F += tmap(i, j) * self.lib_datalm.alm2map(ikx() ** 2 * get_BPBCoviP(i, j, 2))
        Fxx += _lib_qlm.map2alm(F)
        F *= 0
        if verbose:
            t.checkpoint("  Fxx , part 2")

        for i in range(len(typ)):
            for j in range(i, len(typ)):
                F += tmap(i, j) * self.lib_datalm.alm2map(iky() ** 2 * get_BPBCoviP(i, j, 2))
        Fyy += _lib_qlm.map2alm(F)
        F *= 0
        if verbose:
            t.checkpoint("  Fyy , part 2")

        for i in range(len(typ)):
            for j in range(i, len(typ)):
                F += tmap(i, j) * self.lib_datalm.alm2map(iky() * ikx() * get_BPBCoviP(i, j, 2))
        Fxy += _lib_qlm.map2alm(F)
        if verbose:
            t.checkpoint("  Fxy , part 2")

        facunits = -2. / np.sqrt(np.prod(self.lsides))
        ret = xylms_to_phiOmegalm(_lib_qlm, Fxx.real * facunits, Fyy.real * facunits, Fxy.real * facunits)
        if cls_obs is None and cls_weights is None and cls_filt is None:
            fname = os.path.join(self.lib_dir, '%s_resplm_%sCls.npy' % (typ, {True: 'len', False: 'unl'}[use_cls_len]))
            np.save(fname, ret)
            return np.array([lib_qlm.udgrade(_lib_qlm, _a) for _a in np.load(fname, mmap_mode='r')])
        return np.array([lib_qlm.udgrade(_lib_qlm, _a) for _a in ret])

    def get_mfrespcls(self, typ, lib_qlm, use_cls_len=True):
        r"""Linear mean-field response :math:`R_L` of the unnormalized quadratic estimators (gradient and curl components)

            This deflection-induced mean-field linear response is

            :math:`\frac{\delta g^{\rm MF}(x)}{\delta \alpha(y)} = \frac 12 \frac{\delta \ln \rm{Cov}}{\delta \alpha(x) \delta \alpha(y)} = R(x- y)`

            See https://arxiv.org/abs/1704.08230

            The expected normalised MF spectrum is then :math:`\frac{R_L^2}{\mathcal R_L^2} C_L^{\hat g\hat g}`

            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                lib_qlm: *ffs_alm* instance describing the lensing alm arrays
                use_cls_len: use lensed CMB cls if true, unlensed if not (default)


        """
        return [lib_qlm.bin_realpart_inell(_r) for _r in self.get_mfresplms(typ, lib_qlm, use_cls_len=use_cls_len)]

    def get_mfresplms(self, typ, lib_qlm, use_cls_len=False, recache=False):
        assert self.lib_skyalm.shape == self.lib_datalm.shape
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl
        _lib_qlm = ell_mat.ffs_alm_pyFFTW(self.lib_skyalm.ell_mat,
                                                          filt_func=lambda ell: (ell <= 2 * self.lib_skyalm.ellmax))
        fname = os.path.join(self.lib_dir, '%s_MFresplm_%s.npy' % (typ, {True: 'len', False: 'unl'}[use_cls_len]))
        if not os.path.exists(fname) or recache:
            def get_K(i, j):
                # B Covi B
                return self.lib_datalm.almxfl(self._get_pmati(typ, i, j, use_cls_len=use_cls_len),
                                              self.cl_transf ** 2)

            def get_xiK(i, j):
                # xi K
                ret = pmat.get_unlPmat_ij(typ, self.lib_skyalm, cls_cmb, i, 0) * self._upg(get_K(0, j))
                for k in range(1, len(typ)):
                    ret += pmat.get_unlPmat_ij(typ, self.lib_skyalm, cls_cmb, i, k) * self._upg(get_K(k, j))
                return ret

            _2map = lambda alm: self.lib_skyalm.alm2map(alm)
            _dat2map = lambda alm: self.lib_datalm.alm2map(alm)

            _2qlm = lambda _map: _lib_qlm.map2alm(_map)
            ik = lambda ax: self.lib_skyalm.get_ikx() if ax == 1 else self.lib_skyalm.get_iky()

            Fxx = np.zeros(_lib_qlm.alm_size, dtype=complex)
            Fyy = np.zeros(_lib_qlm.alm_size, dtype=complex)
            Fxy = np.zeros(_lib_qlm.alm_size, dtype=complex)

            for i in range(len(typ)):  # (xia K)_ij(xib K)_ji
                for j in range(len(typ)):
                    xiK1 = get_xiK(i, j)
                    xiK2 = get_xiK(j, i)
                    Fxx += _2qlm(_2map(xiK1 * ik(1)) * (_2map(xiK2 * ik(1))))
                    Fyy += _2qlm(_2map(xiK1 * ik(0)) * (_2map(xiK2 * ik(0))))
                    Fxy += _2qlm(_2map(xiK1 * ik(1)) * (_2map(xiK2 * ik(0))))

            # adding to that (K) (xi K xi - xi) =
            def get_xiKxi_xi(i, j):
                ret = get_xiK(i, 0) * pmat.get_unlPmat_ij(typ, self.lib_skyalm, cls_cmb, 0, j)
                for k in range(1, len(typ)):
                    ret += get_xiK(i, k) * pmat.get_unlPmat_ij(typ, self.lib_skyalm, cls_cmb, k, j)
                ret -= pmat.get_unlPmat_ij(typ, self.lib_skyalm, cls_cmb, i, j)
                return ret

            for i in range(len(typ)):  # (K) (xi K xi - xi)
                for j in range(len(typ)):
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
            print("Cached " + fname)
        return np.array([lib_qlm.udgrade(_lib_qlm, _a) for _a in np.load(fname)])

    def get_dmfrespcls(self, typ, cmb_dcls, lib_qlm, use_cls_len=False):
        r"""Variation of the linear mean-field response :math:`R_L` of the unnormalized quadratic estimators

            Variation of *get_mfrespcls* given input CMB spectra variation

            This deflection-induced mean-field linear response is

            :math:`\frac{\delta g^{\rm MF}(x)}{\delta \alpha(y)} = \frac 12 \frac{\delta \ln \rm{Cov}}{\delta \alpha(x) \delta \alpha(y)} = R(x- y)`

            See https://arxiv.org/abs/1704.08230

            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                cmb_dcls(dict): CMB spectra variations
                lib_qlm: *ffs_alm* instance describing the lensing alm arrays
                use_cls_len: use lensed CMB cls if true, unlensed if not (default)


        """
        return [lib_qlm.bin_realpart_inell(_r) for _r in
                self.get_dmfresplms(typ, cmb_dcls, lib_qlm, use_cls_len=use_cls_len)]

    def get_dmfresplms(self, typ, cmb_dcls, lib_qlm, use_cls_len=False, recache=False):
        assert self.lib_skyalm.shape == self.lib_datalm.shape
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl
        _lib_qlm = ell_mat.ffs_alm_pyFFTW(self.lib_skyalm.ell_mat, filt_func=lambda ell: (ell <= 2 * self.lib_skyalm.ellmax))
        # FIXME dclhash !
        print("!!!! dMFresplms::cmb_dcls hash is missing here !! ?")
        fname = os.path.join(self.lib_dir, '%s_dMFresplm_%s.npy' % (typ, {True: 'len', False: 'unl'}[use_cls_len]))
        if not os.path.exists(fname) or recache:

            def mu(a, b, i, j):
                ret = a(i, 0) * b(0, j)
                for _k in range(1, len(typ)):
                    ret += a(i, _k) * b(_k, j)
                return ret

            def dxi(i, j):
                return pmat.get_unlPmat_ij(typ, self.lib_skyalm, cmb_dcls, i, j)

            def xi(i, j):
                return pmat.get_unlPmat_ij(typ, self.lib_skyalm, cls_cmb, i, j)

            def K(i, j):
                return self._upg(
                    self.lib_datalm.almxfl(self._get_pmati(typ, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2))

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
            for i in range(len(typ)):  # d[(xia K)_ij(xib K)_ji]
                for j in range(len(typ)):
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
            for i in range(len(typ)):  # d [(K) (xi K xi - xi)]
                for j in range(len(typ)):
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
            print("Cached " + fname)
        return np.array([lib_qlm.udgrade(_lib_qlm, _a) for _a in np.load(fname)])

    def get_lndetcurv(self, typ, lib_qlm, get_A=None, use_cls_len=False):
        r"""(Part of) covariance log-det second variation

        Harmonic transform of :math:`\frac 1 2 \rm{Tr} A \frac{\delta^2 \rm{Cov}}{\delta \alpha(x) \delta \alpha(y)}` for vanishing deflection

        if *get_A* argument is not set, output becomes :math:`\frac 1 2 \rm{Tr} \rm{Cov}^{-1} \delta^2 \rm{Cov}`

        Args:
            typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
            lib_qlm: *ffs_alm* instance describing the lensing alm arrays
            get_A(optional, callable): 1st trace operator. Defaults to :math:`\rm{Cov}^{-1}`
            use_cls_len: use lensed or unlensed cls in QE weights (numerator), defaults to lensed cls

        Returns:
            gradient and curl Fourier components


        """
        _lib_qlm = lib_qlm
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl
        if get_A is None:
            get_A = self._get_pmati

        def get_K(i, j):
            # B^t A B
            return self._upg(self.lib_datalm.almxfl(get_A(typ, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2))

        _2qlm = lambda _map: _lib_qlm.map2alm(_map).real
        _2map = lambda alm: self.lib_skyalm.alm2map(alm)
        ik = lambda ax: self.lib_skyalm.get_ikx() if ax == 1 else self.lib_skyalm.get_iky()

        Fxx = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fyy = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fxy = np.zeros(_lib_qlm.alm_size, dtype=float)
        for i in range(len(typ)):  # (K) (xi_ab)
            for j in range(i, len(typ)):  # This is i-j symmetric
                fac = 2 - (i == j)
                K = _2map(get_K(i, j))
                xi = pmat.get_unlPmat_ij(typ, self.lib_skyalm, cls_cmb, j, i)
                Fxx -= fac * _2qlm(K * _2map(xi * ik(1) * ik(1)))
                Fyy -= fac * _2qlm(K * _2map(xi * ik(0) * ik(0)))
                Fxy -= fac * _2qlm(K * _2map(xi * ik(1) * ik(0)))

        assert _lib_qlm.reduced_ellmat()[0] == 0
        Fxx -= Fxx[0]
        Fyy -= Fyy[0]
        Fxy -= Fxy[0]
        facunits = 1. / np.sqrt(np.prod(self.lsides))
        return xylms_to_phiOmegalm(_lib_qlm, Fxx * facunits, Fyy * facunits, Fxy * facunits)

    def get_dlndetcurv(self, typ, cmb_dcls, lib_qlm, K=None, dK=None, use_cls_len=False):
        r"""Variation of *get_lndetcurv* with respect to CMB spectra variation

        *get_lndetcurv* is the harmonic transform of :math:`\frac 1 2 \rm{Tr} A \frac{\delta^2 \rm{Cov}}{\delta \alpha(x) \delta \alpha(y)}` for vanishing deflection


        Args:
            typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
            cmb_dcls(dict): CMB spectra variation
            lib_qlm: *ffs_alm* instance describing the lensing alm arrays

            use_cls_len: use lensed or unlensed cls in QE weights (numerator), defaults to lensed cls

        Returns:
            gradient and curl Fourier components

        """
        #Finite differences test ok.
        _lib_qlm = lib_qlm
        if K is None: assert dK is None
        if K is not None: assert dK is not None

        cls_cmb = self.cls_len if use_cls_len else self.cls_unl

        def mu(a, b, i, j):
            ret = a(i, 0) * b(0, j)
            for _k in range(1, len(typ)):
                ret += a(i, _k) * b(_k, j)
            return ret

        def dxi(i, j):
            return pmat.get_unlPmat_ij(typ, self.lib_skyalm, cmb_dcls, i, j)

        def xi(i, j):
            return pmat.get_unlPmat_ij(typ, self.lib_skyalm, cls_cmb, i, j)

        if K is None:
            K = lambda i, j: (self._upg(
                self.lib_datalm.almxfl(self._get_pmati(typ, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2)))
        if dK is None:
            dxiK = lambda i, j: mu(dxi, K, i, j)
            dK = lambda i, j: (-mu(K, dxiK, i, j))

        _2qlm = lambda _map: _lib_qlm.map2alm(_map).real
        _2map = lambda alm: self.lib_skyalm.alm2map(alm)
        ik = lambda ax: self.lib_skyalm.get_ikx() if ax == 1 else self.lib_skyalm.get_iky()

        Fxx = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fyy = np.zeros(_lib_qlm.alm_size, dtype=float)
        Fxy = np.zeros(_lib_qlm.alm_size, dtype=float)
        for i in range(len(typ)):  # -(dK) (xi_ab) - (K)(dxi_ab)
            for j in range(len(typ)):
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

    def get_fishertrace(self, typ, lib_qlm, get_A1=None, get_A2=None, use_cls_len=True):
        r"""Fisher-matrix like trace

            :math:`\frac 1 2 \textrm{Tr}\: A_1 \:\frac{\delta\rm{Cov}}{\delta\phi} \:A_2  \:\frac{\delta\rm{Cov}}{\delta\phi}`

            if operators :math:`A_1` or :math:`A_2` not set, they reduce to :math:`\rm{Cov}^{-1}`, and the result is the Fisher info for vanishing deflection

            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                lib_qlm: *ffs_alm* instance describing the lensing alm arrays
                get_A1(optional, callable): 1st trace operator
                get_A2(optional, callable): 2nd trace operator
                use_cls_len: use lensed or unlensed cls in QE weights (numerator), defaults to lensed cls


        """

        t = timer(_timed, prefix=__name__, suffix=' curvpOlm')
        _lib_qlm = lib_qlm
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl
        if get_A1 is None:
            get_A1 = self._get_pmati
        if get_A2 is None:
            get_A2 = self._get_pmati

        def get_K1(i, j):
            return self.lib_datalm.almxfl(get_A1(typ, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2)

        def get_K2(i, j):
            return self.lib_datalm.almxfl(get_A2(typ, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2)

        def get_xiK(mat, i, j):
            assert mat in [1, 2], mat
            K = get_K1 if mat == 1 else get_K2
            ret = pmat.get_unlPmat_ij(typ, self.lib_datalm, cls_cmb, i, 0) * K(0, j)
            for _k in range(1, len(typ)):
                ret += pmat.get_unlPmat_ij(typ, self.lib_datalm, cls_cmb, i, _k) * K(_k, j)
            return ret

        def get_xiKxi(mat, i, j):
            assert mat in [1, 2], mat
            ret = get_xiK(mat, i, 0) * pmat.get_unlPmat_ij(typ, self.lib_datalm, cls_cmb, 0, j)
            for _k in range(1, len(typ)):
                ret += get_xiK(mat, i, _k) * pmat.get_unlPmat_ij(typ, self.lib_datalm, cls_cmb, _k, j)
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
        for i in range(len(typ)):  # (K) (xi_ab)
            for j in range(len(typ)):
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

    def get_dfishertrace(self, typ, cmb_dcls, lib_qlm, K1=None, K2=None, dK1=None, dK2=None,
                         use_cls_len=True, recache=False):
        r"""Variation of Fisher-trace *get_fishertrace* with respect to input CMB variation

            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                 *cmb_dcls* (dict): CMB spectra variation
                lib_qlm: *ffs_alm* instance describing the lensing alm arrays
                use_cls_len: use lensed or unlensed cls in QE weights (numerator), defaults to lensed cls

        """
        if K1 is not None: assert dK1 is not None
        if K2 is not None: assert dK2 is not None
        t = timer(_timed, prefix=__name__, suffix=' curvpOlm')
        _lib_qlm = lib_qlm
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl

        def mu(a, b, i, j):
            ret = a(i, 0) * b(0, j)
            for _k in range(1, len(typ)):
                ret += a(i, _k) * b(_k, j)
            return ret

        def dxi(i, j):
            return pmat.get_unlPmat_ij(typ, self.lib_skyalm, cmb_dcls, i, j)

        def xi(i, j):
            return pmat.get_unlPmat_ij(typ, self.lib_skyalm, cls_cmb, i, j)

        if K1 is None:
            assert dK1 is None
            K1 = lambda i, j: self._upg(
                self.lib_datalm.almxfl(self._get_pmati(typ, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2))
        if K2 is None:
            assert dK2 is None
            K2 = lambda i, j: self._upg(
                self.lib_datalm.almxfl(self._get_pmati(typ, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2))

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
        for i in range(len(typ)):
            for j in range(len(typ)):
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

    def get_plmlikcurvcls(self, typ, datcmb_cls, lib_qlm, use_cls_len=True, recache=False, dat_only=False):
        r"""Returns realization-dependent second variation (curvature) of lensing deflection likelihood

            Second variation of

             :math:`\frac 12 X^{\rm dat} \rm{Cov_\alpha}^{-1} X^{\rm dat} + \frac 1 2 \ln \det \rm{Cov_\alpha}`

             with respect to deflection for custom data CMB spectra. See https://arxiv.org/abs/1808.10349

            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                datcmb_cls(dict): data CMB spectra  (if they do match lensed cls attribute, the result should be inverse lensing Gaussian noise bias)
                lib_qlm: *ffs_alm* instance describing the lensing alm's arrays.
                use_cls_len: use lensed cls in filtering (defaults to True)


        """
        fname = os.path.join(self.lib_dir, '%splmlikcurv_cls%s_cldat' % (typ, {True: 'len', False: 'unl'}[use_cls_len]) \
                + cls_hash(datcmb_cls, lmax=self.lib_datalm.ellmax) +  '.dat')
        if not os.path.exists(fname) or recache:
            def get_dcov(typ, i, j, use_cls_len=True):
                # Covi (datCov - Cov) = Covi datCov - 1
                ret = self._get_pmati(typ, i, 0, use_cls_len=use_cls_len) \
                      * pmat.get_datPmat_ij(typ, self.lib_datalm, datcmb_cls, self.cl_transf, self.cls_noise, 0, j)
                for _k in range(1, len(typ)):
                    ret += self._get_pmati(typ, i, _k, use_cls_len=use_cls_len) \
                           * pmat.get_datPmat_ij(typ, self.lib_datalm, datcmb_cls, self.cl_transf, self.cls_noise, _k, j)
                if i == j and not dat_only:
                    ret -= 1.
                return ret

            def get_dcovd(typ, i, j, use_cls_len=True):
                # Covi(datCov - Cov)Covi
                ret = get_dcov(typ, i, 0, use_cls_len=use_cls_len) \
                      * self._get_pmati(typ, 0, j, use_cls_len=use_cls_len)
                for _k in range(1, len(typ)):
                    ret += get_dcov(typ, i, _k, use_cls_len=use_cls_len) \
                           * self._get_pmati(typ, _k, j, use_cls_len=use_cls_len)
                return ret

            def get_d2cov(typ, i, j, use_cls_len=True):
                # Cov^-1 (2 datCov - Cov)  = 2 Covi datCov - 1
                ret = self._get_pmati(typ, i, 0, use_cls_len=use_cls_len) \
                      * pmat.get_datPmat_ij(typ, self.lib_datalm, datcmb_cls, self.cl_transf, self.cls_noise, 0, j)
                for _k in range(1, len(typ)):
                    ret += self._get_pmati(typ, i, _k, use_cls_len=use_cls_len) \
                           * pmat.get_datPmat_ij(typ, self.lib_datalm, datcmb_cls, self.cl_transf, self.cls_noise, _k, j)
                if i == j and not dat_only:
                    return 2 * ret - 1.
                return 2 * ret

            def get_idCi(typ, i, j, use_cls_len=True):
                # Cov^-1 (2 datCov - Cov) Cov^-1
                ret = get_d2cov(typ, i, 0, use_cls_len=use_cls_len) * self._get_pmati(typ, 0, j,
                                                                                      use_cls_len=use_cls_len)
                for _k in range(1, len(typ)):
                    ret += get_d2cov(typ, i, _k, use_cls_len=use_cls_len) * self._get_pmati(typ, _k, j,
                                                                                            use_cls_len=use_cls_len)
                return ret

            # First term :
            _lib_qlm = ell_mat.ffs_alm_pyFFTW(lib_qlm.ell_mat, filt_func=lambda ell: ell >= 0)
            curv = -self.get_lndetcurv(typ, _lib_qlm, get_A=get_dcovd, use_cls_len=use_cls_len)
            Fish = self.get_fishertrace(typ, _lib_qlm, get_A1=get_idCi, use_cls_len=use_cls_len)
            ret = np.array([_lib_qlm.bin_realpart_inell(_r) for _r in (curv + Fish)[:2]])
            np.savetxt(fname, ret.transpose(),
                       header='second variation (curvature) of <1/2Xdat Covi Xdat + 1/2 ln det Cov>_dat')
            print("Cached " + fname)
        print("loading " + fname)
        cond = lib_qlm.ell_mat.get_Nell() > 0
        ret = np.array([(_r * cond)[:lib_qlm.ellmax + 1] for _r in np.loadtxt(fname).transpose()])
        return ret

    def get_plmRDlikcurvcls(self, typ, datcls_obs, lib_qlm, use_cls_len=True, use_cls_len_D=None, recache=False,
                            dat_only=False):
        r"""Returns realization-dependent second variation (curvature) of lensing deflection likelihood

            Second variation of

             :math:`\frac 12 X^{\rm dat} \rm{Cov_\alpha}^{-1} X^{\rm dat} + \frac 1 2 \ln \det \rm{Cov_\alpha}`

             with respect to deflection for custom data spectra. See https://arxiv.org/abs/1808.10349

            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                datcls_obs(dict): data (beam-convovled) CMB and noise spectra
                lib_qlm: *ffs_alm* instance describing the lensing alm's arrays
                use_cls_len: use lensed cls in filtering (defaults to True)


        """
        # FIXME : The rel . agreement with 1/N0 is only 1e-6 not double prec., not sure what is going on.
        fname = os.path.join(self.lib_dir, '%splmRDlikcurv_cls%s_cldat' % (typ, {True: 'len', False: 'unl'}[use_cls_len]) \
                + cls_hash(datcls_obs, lmax=self.lib_datalm.ellmax) + '.dat')
        if dat_only:
            fname = fname.replace('.dat', 'datonly.dat')
            assert 'datonly' in fname
        if use_cls_len_D is not None and use_cls_len_D != use_cls_len:
            fname = fname.replace('.dat', '_clsD%s.dat' % {True: 'len', False: 'unl'}[use_cls_len_D])
        else:
            use_cls_len_D = use_cls_len
        if not os.path.exists(fname) or recache:
            def get_dcov(typ, i, j, use_cls_len=use_cls_len):
                # Covi (datCov - Cov) = Covi datCov - 1
                ret = self._get_pmati(typ, i, 0, use_cls_len=use_cls_len) \
                      * pmat.get_unlPmat_ij(typ, self.lib_datalm, datcls_obs, 0, j)
                for _k in range(1, len(typ)):
                    ret += self._get_pmati(typ, i, _k, use_cls_len=use_cls_len) \
                           * pmat.get_unlPmat_ij(typ, self.lib_datalm, datcls_obs, _k, j)
                if i == j and not dat_only:
                    ret -= 1.
                return ret

            def get_dcovd(typ, i, j, use_cls_len=use_cls_len):
                # Covi(datCov - Cov)Covi
                ret = get_dcov(typ, i, 0, use_cls_len=use_cls_len) \
                      * self._get_pmati(typ, 0, j, use_cls_len=use_cls_len)
                for _k in range(1, len(typ)):
                    ret += get_dcov(typ, i, _k, use_cls_len=use_cls_len) \
                           * self._get_pmati(typ, _k, j, use_cls_len=use_cls_len)
                return ret

            def get_d2cov(typ, i, j, use_cls_len=use_cls_len):
                # Cov^-1 (2 datCov - Cov)  = 2 Covi datCov - 1
                ret = self._get_pmati(typ, i, 0, use_cls_len=use_cls_len) \
                      * pmat.get_unlPmat_ij(typ, self.lib_datalm, datcls_obs, 0, j)
                for _k in range(1, len(typ)):
                    ret += self._get_pmati(typ, i, _k, use_cls_len=use_cls_len) \
                           * pmat.get_unlPmat_ij(typ, self.lib_datalm, datcls_obs, _k, j)
                if i == j and not dat_only:
                    return 2 * ret - 1.
                return 2 * ret

            def get_idCi(typ, i, j, use_cls_len=use_cls_len):
                # Cov^-1 (2 datCov - Cov) Cov^-1
                ret = get_d2cov(typ, i, 0, use_cls_len=use_cls_len) * self._get_pmati(typ, 0, j,
                                                                                      use_cls_len=use_cls_len)
                for _k in range(1, len(typ)):
                    ret += get_d2cov(typ, i, _k, use_cls_len=use_cls_len) * self._get_pmati(typ, _k, j,
                                                                                            use_cls_len=use_cls_len)
                return ret

            # First term :
            _lib_qlm = ell_mat.ffs_alm_pyFFTW(lib_qlm.ell_mat, filt_func=lambda ell: ell >= 0)
            curv = -self.get_lndetcurv(typ, _lib_qlm, get_A=get_dcovd, use_cls_len=use_cls_len_D)
            Fish = self.get_fishertrace(typ, _lib_qlm, get_A1=get_idCi, use_cls_len=use_cls_len)
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
        print("loading ", fname)
        cond = lib_qlm.ell_mat.get_Nell() > 0
        ret = np.array([(_r * cond)[:lib_qlm.ellmax + 1] for _r in np.loadtxt(fname).transpose()])
        return ret

    def get_dplmRDlikcurvcls(self, typ, cmb_dcls, datcls_obs, lib_qlm, use_cls_len=True, recache=False,
                             dat_only=False):
        """
        derivative of plmRDlikcurvcls (data held fixed)
        Finite difference test OK (+ much faster)
        """
        # FIXME : this is like really, really, really inefficient.
        fname = os.path.join(self.lib_dir, '%sdplmRDlikcurv_cls%s_cldat' % (typ, {True: 'len', False: 'unl'}[use_cls_len]) \
                + cls_hash(datcls_obs, lmax=self.lib_datalm.ellmax) + cls_hash(cmb_dcls) + '.dat')
        if dat_only:
            fname = fname.replace('.dat', 'datonly.dat')
            assert 'datonly' in fname
        if not os.path.exists(fname) or recache:
            # K going into trace should be 2 K datcls K - K
            # K going into lndet should be K datcls K
            # dK is -K dxi K
            def mu(a, b, i, j):
                ret = a(i, 0) * b(0, j)
                for _k in range(1, len(typ)):
                    ret += a(i, _k) * b(_k, j)
                return ret

            dxi = lambda i, j: pmat.get_unlPmat_ij(typ, self.lib_skyalm, cmb_dcls, i, j)
            K = lambda i, j: self._upg(
                self.lib_datalm.almxfl(self._get_pmati(typ, i, j, use_cls_len=use_cls_len), self.cl_transf ** 2))
            dxiK = lambda i, j: mu(dxi, K, i, j)
            datKi = lambda i, j: self._upg(
                self.lib_datalm.almxfl(pmat.get_unlPmat_ij(typ, self.lib_datalm, datcls_obs, i, j),
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
            _lib_qlm = ell_mat.ffs_alm_pyFFTW(lib_qlm.ell_mat, filt_func=lambda ell: ell >= 0)
            curv = -self.get_dlndetcurv(typ, cmb_dcls, _lib_qlm, K=Kdet, dK=dKdet, use_cls_len=use_cls_len)
            Fish = self.get_dfishertrace(typ, cmb_dcls, _lib_qlm, K1=Ktrace, dK1=dKtrace, use_cls_len=use_cls_len)
            ret = np.array([_lib_qlm.bin_realpart_inell(_r) for _r in (curv + Fish)[:2]])
            np.savetxt(fname, ret.transpose(),
                       header='second variation (curvature) of <1/2Xdat Covi Xdat + 1/2 ln det Cov>_dat')
            print("cached ", fname)
        print("loading ", fname)
        cond = lib_qlm.ell_mat.get_Nell() > 0
        return np.array([(_r * cond)[:lib_qlm.ellmax + 1] for _r in np.loadtxt(fname).transpose()])


class ffs_lencov_alm(ffs_diagcov_alm):
    """Extended *ffs_diagcov_alm* sub-class, adding a deflection field and its inverse

        Args:
            lib_dir: many things will be saved there
            lib_datalm: *ffs_alm* instance containing mode filtering and flat-sky patch info
            lib_skyalm: *ffs_alm* instance describing the sky modes.
            cls_unl(dict): unlensed CMB cls
            cls_len(dict): lensed CMB cls
            cl_transf: instrument transfer function
            cls_noise(dict): 't', 'q' and 'u' noise arrays
            f: deflection field, forward operation (e.g. *ffs_displacement* instance)
            fi: deflection field, backward operation (ideally the inverse f-deflection, *f.get_inverse()*)


    """
    def __init__(self, lib_dir, lib_datalm, lib_skyalm, cls_unl, cls_len, cl_transf, cls_noise, f, fi,
                 init_rank=pbs.rank, init_barrier=pbs.barrier):
        assert lib_datalm.ell_mat.lsides == lib_skyalm.ell_mat.lsides, (lib_datalm.ell_mat.lsides, lib_skyalm.ell_mat.lsides)

        super(ffs_lencov_alm, self).__init__(lib_dir, lib_datalm, cls_unl, cls_len, cl_transf, cls_noise,
                                             lib_skyalm=lib_skyalm, init_barrier=init_barrier, init_rank=init_rank)

        self.lmax_dat = self.lib_datalm.ellmax
        self.lmax_sky = self.lib_skyalm.ellmax

        self.sky_shape = self.lib_skyalm.ell_mat.shape
        # assert self.lmax_dat <= self.lmax_sky, (self.lmax_dat, self.lmax_sky)

        for cl in self.cls_unl.values():   assert len(cl) > lib_skyalm.ellmax

        assert f.shape == self.sky_shape and fi.shape == self.sky_shape, (f.shape, fi.shape, self.sky_shape)
        assert f.lsides == self.lsides and fi.lsides == self.lsides, (f.lsides, fi.lsides, self.lsides)
        self.fi = fi  # inverse displacement
        self.f = f  # displacement

    def hashdict(self):
        h = {'lib_alm': self.lib_datalm.hashdict(), 'lib_skyalm': self.lib_skyalm.hashdict()}
        for key in self.cls_noise.keys():
            h['cls_noise ' + key] =  npy_hash(self.cls_noise[key])
        for key in self.cls_unl.keys():
            h['cls_unl ' + key] =  npy_hash(self.cls_unl[key])
        for key in self.cls_len.keys():
            h['cls_len ' + key] =  npy_hash(self.cls_len[key])
        h['cl_transf'] =  npy_hash(self.cl_transf)
        return h

    def set_ffinv(self, f, fi):
        """Replace deflection-field attributes and its inverse

            Args:
                f: new deflection-field instance
                fi: new inverse deflection-field instance

        """
        assert f.shape == self.sky_shape and f.lsides == self.lsides, (f.shape, f.lsides)
        assert fi.shape == self.sky_shape and fi.lsides == self.lsides, (fi.shape, fi.lsides)
        assert hasattr(self, 'f') and  hasattr(self, 'fi')
        setattr(self, 'f', f)
        setattr(self, 'fi', fi)

    def apply(self, typ, alms, use_Pool=0):
        assert alms.shape == self._datalms_shape(typ), (alms.shape, self._datalms_shape(typ))
        ret = self._apply_signal(typ, alms, use_Pool=use_Pool)
        ret += self.apply_noise(typ, alms)
        return ret

    def _apply_signal(self, typ, alms, use_Pool=0):
        assert alms.shape == self._datalms_shape(typ), (alms.shape, self._datalms_shape(typ))
        ret = np.empty_like(alms)

        if use_Pool <= -100:
            from lensit.gpu import apply_GPU
            ablms = np.array([self.lib_datalm.almxfl(_a, self.cl_transf) for _a in alms])
            apply_GPU.apply_FDxiDtFt_GPU_inplace(typ, self.lib_datalm, self.lib_skyalm, ablms,
                                                 self.f, self.fi, self.cls_unl)
            for i in range(len(typ)):
                ret[i] = self.lib_datalm.almxfl(ablms[i], self.cl_transf)
            return ret
        # could do with less
        t = timer(_timed, prefix=__name__, suffix='apply_signal')
        t.checkpoint("just started")

        tempalms = np.empty(self._skyalms_shape(typ), dtype=complex)
        for _i in range(len(typ)):  # Lens with inverse and mult with determinant magnification.
            tempalms[_i] = self.fi.lens_alm(self.lib_skyalm,
                                            self._upg(self.lib_datalm.almxfl(alms[_i], self.cl_transf)),
                                            lib_alm_out=self.lib_skyalm, mult_magn=True, use_Pool=use_Pool)
        # NB : 7 new full sky alms for TQU in this routine - > 4 GB total for full sky lmax_sky =  6000.
        t.checkpoint("backward lens + det magn")

        skyalms = np.zeros_like(tempalms)
        for j in range(len(typ)):
            for i in range(len(typ)):
                skyalms[i] += pmat.get_unlPmat_ij(typ, self.lib_skyalm, self.cls_unl, i, j) * tempalms[j]
        del tempalms
        t.checkpoint("mult with Punl mat ")

        for i in range(len(typ)):  # Lens with forward displacement
            ret[i] = self._deg(self.f.lens_alm(self.lib_skyalm, skyalms[i], use_Pool=use_Pool))
        t.checkpoint("Forward lensing mat ")

        for i in range(len(typ)):
            ret[i] = self.lib_datalm.almxfl(ret[i], self.cl_transf)
        t.checkpoint("Beams")
        return ret

    def _apply_cond3(self, typ, alms, use_Pool=0):
        #(DBxiB ^ tD ^ t + N) ^ -1 \sim D ^ -t(BxiBt + N) ^ -1 D ^ -1
        assert alms.shape == self._datalms_shape(typ), (alms.shape, self._datalms_shape(typ))
        t = timer(_timed, prefix=__name__, suffix='_apply_cond3')
        t.checkpoint("just started")

        if use_Pool <= -100:
            # Try entire evaluation on GPU :
            # FIXME !! lib_sky vs lib_dat
            from lensit.gpu.apply_cond3_GPU import apply_cond3_GPU_inplace as c3GPU
            ret = alms.copy()
            c3GPU(typ, self.lib_datalm, ret, self.f, self.fi, self.cls_unl, self.cl_transf, self.cls_noise)
            return ret
        temp = np.empty_like(alms)  # Cond. must not change their arguments
        for i in range(len(typ)):  # D^{-1}
            temp[i] = self._deg(self.fi.lens_alm(self.lib_skyalm, self._upg(alms[i]), use_Pool=use_Pool))

        t.checkpoint("Lensing with inverse")

        ret = np.zeros_like(alms)  # (B xi B^t + N)^{-1}
        for i in range(len(typ)):
            for j in range(len(typ)):
                ret[i] += self._get_pmati(typ, i, j) * temp[j]
        del temp
        t.checkpoint("Mult. w. inv Pmat")

        for i in range(len(typ)):  # D^{-t}
            ret[i] = self._deg(self.f.lens_alm(self.lib_skyalm, self._upg(ret[i]), use_Pool=use_Pool, mult_magn=True))

        t.checkpoint("Lens w. forward and det magn.")

        return ret

    def get_iblms(self, typ, datalms, use_cls_len=False, use_Pool=0, **kwargs):
        r"""Inverse-variance filters input CMB maps

            Produces :math:`B^t \rm{Cov_\alpha}^{-1}X^{\rm dat}` (inputs to quadratic estimator routines)

            The covariance matrix here includes the lensing deflection field :math:`\alpha` as given by the *self.f* and *self.fi* attributes

        """
        assert use_cls_len == False, 'not implemented'
        if typ == 'QU':
            return self._get_iblms_v2(typ, datalms, use_cls_len=use_cls_len, use_Pool=use_cls_len, **kwargs)
        if datalms.shape == (len(typ), self.dat_shape[0], self.dat_shape[1]):
            _datalms = np.array([self.lib_datalm.map2alm(_m) for _m in datalms])
            return self.get_iblms(typ, _datalms, use_cls_len=use_cls_len, use_Pool=use_Pool, **kwargs)

        assert datalms.shape == self._datalms_shape(typ), (datalms.shape, self._datalms_shape(typ))
        ilms, it = self.cd_solve(typ, datalms, use_Pool=use_Pool, **kwargs)
        ret = np.empty(self._skyalms_shape(typ), dtype=complex)
        for _i in range(len(typ)):
            ret[_i] = self._upg(self.lib_datalm.almxfl(ilms[_i], self.cl_transf))
        return ret, it

    def _get_iblms_v2(self, typ, datalms, use_cls_len=False, use_Pool=0, **kwargs):
        # some weird things happening with very low noise T ?
        assert use_cls_len == False, 'not implemented'
        MLTQUlms = SM.TEB2TQUlms(typ, self.lib_skyalm,self._get_mllms(typ, datalms,
                                                                      use_cls_len=use_cls_len, use_Pool=use_Pool, **kwargs))
        ret = self._mltqulms2ibtqulms(typ, MLTQUlms, datalms, use_Pool=use_Pool)
        return ret, -1  # No iterations info implemented

    def _mltqulms2ibtqulms(self, typ, MLTQUlms, datalms, use_Pool = 0):
        """ Output TQU skyalm shaped """
        ret = np.zeros(self._skyalms_shape(typ), dtype=complex)
        for i in range(len(typ)):
            temp = datalms[i] - self.lib_datalm.almxfl(self.f.lens_alm(self.lib_skyalm, MLTQUlms[i],
                                lib_alm_out=self.lib_datalm, use_Pool=use_Pool),self.cl_transf)
            self.lib_datalm.almxfl(temp, self.cl_transf[:self.lib_datalm.ellmax + 1]
                                   * cl_inverse(self.cls_noise[typ[i].lower()][:self.lib_datalm.ellmax + 1]),
                                   inplace=True)
            ret[i] = self._upg(temp)
        return ret


    def get_mllms(self, typ, datmaps, use_Pool=0, use_cls_len=False, **kwargs):
        r"""Returns maximum likelihood sky CMB modes.

            This instance uses anisotropic filtering, with lensing deflections as defined by
            *self.f* and *self.fiv*

            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                datmaps: data real-space maps array
                use_cls_len: use lensed cls in filtering (defaults to True)

            Returns:
                T,E,B alm array


        """
        return self._get_mllms(typ, np.array([self.lib_datalm.map2alm(m) for m in datmaps]),
                               use_cls_len=use_cls_len, use_Pool=use_Pool)

    def _get_mllms(self, typ, datalms, use_Pool=0, use_cls_len=False, **kwargs):
        assert np.all(self.cls_noise['t'] == self.cls_noise['t'][0]), 'adapt ninv filt ideal for coloured cls(easy)'
        assert np.all(self.cls_noise['q'] == self.cls_noise['q'][0]), 'adapt ninv filt ideal for coloured cls(easy'
        assert np.all(self.cls_noise['u'] == self.cls_noise['q'][0]), 'adapt ninv filt ideal for coloured cls(easy'
        # FIXME could use opfilt_cinvBB
        nlev_t = np.sqrt(self.cls_noise['t'][0] * (180. * 60 / np.pi) ** 2)
        nlev_p = np.sqrt(self.cls_noise['q'][0] * (180. * 60 / np.pi) ** 2)

        cmb_cls = self.cls_len if use_cls_len else self.cls_unl
        filt = ffs_ninv_filt_ideal.ffs_ninv_filt_wl(self.lib_datalm, self.lib_skyalm,
                                                    cmb_cls, self.cl_transf, nlev_t, nlev_p, self.f,
                                                    self.fi, lens_pool=use_Pool)
        opfilt = opfilt_cinv
        opfilt._type = typ
        chain = chain_samples.get_isomgchain(self.lib_skyalm.ellmax, self.lib_datalm.shape, **kwargs)
        mchain =multigrid.multigrid_chain(opfilt, typ, chain, filt)
        soltn = np.zeros((opfilt.TEBlen(typ), self.lib_skyalm.alm_size), dtype=complex)
        mchain.solve(soltn, datalms, finiop='MLIK')
        return soltn

    def get_qlms(self, typ, iblms, lib_qlm, use_Pool=0, use_cls_len=False, **kwargs):
        r"""Unormalized quadratic estimates (potential and curl), including current lensing deflection estimate

        Note:
            the output differs by a factor of two from the standard QE. This is because this was written initially
            as gradient function of the CMB likelihood w.r.t. real and imaginary parts. So to use this as QE's,
            the normalization is 1/2 the standard normalization the inverse response. The *lensit.ffs_qlms.qlms.py* module contains methods
            of the QE's with standard normalizations which you may want to use instead.

        Args:
            typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
            iblms: inverse-variance filtered CMB alm arrays
            lib_qlm: *ffs_alm* instance describing the lensing alm arrays
            use_cls_len: use lensed or unlensed cls in QE weights (numerator), defaults to lensed cls


        """
        assert iblms.shape == self._skyalms_shape(typ), (iblms.shape, self._skyalms_shape(typ))
        assert lib_qlm.ell_mat.lsides == self.lsides, (self.lsides, lib_qlm.ell_mat.lsides)
        cls = self.cls_len if use_cls_len else self.cls_unl
        almsky1 = np.empty((len(typ), self.lib_skyalm.alm_size), dtype=complex)

        Bu = lambda idx: self.lib_skyalm.alm2map(iblms[idx])
        _2qlm = lambda _m: lib_qlm.udgrade(self.lib_skyalm, self.lib_skyalm.map2alm(_m))

        def DxiDt(alms, axis):
            assert axis in [0, 1]
            kfunc = self.lib_skyalm.get_ikx if axis == 1 else self.lib_skyalm.get_iky
            return self.f.alm2lenmap(self.lib_skyalm, alms * kfunc(), use_Pool=use_Pool)

        t = timer(_timed)
        t.checkpoint("  get_likgrad::just started ")

        for _j in range(len(typ)):  # apply Dt and back to harmonic space :
            almsky1[_j] = self.fi.lens_alm(self.lib_skyalm, iblms[_j],
                                           mult_magn=True, use_Pool=use_Pool)

        t.checkpoint("  get_likgrad::Forward lensing maps, (%s map(s)) " % len(typ))
        almsky2 = np.zeros((len(typ), self.lib_skyalm.alm_size), dtype=complex)
        for _i in range(len(typ)):
            for _j in range(len(typ)):
                almsky2[_i] += pmat.get_unlPmat_ij(typ, self.lib_skyalm, cls, _i, _j) * almsky1[_j]

        del almsky1
        t.checkpoint("  get_likgrad::Mult. w. unlPmat, %s field(s)" % len(typ))

        retdx = _2qlm(Bu(0) * DxiDt(almsky2[0], 1))
        retdy = _2qlm(Bu(0) * DxiDt(almsky2[0], 0))
        for _i in range(1, len(typ)):
            retdx += _2qlm(Bu(_i) * DxiDt(almsky2[_i], 1))
            retdy += _2qlm(Bu(_i) * DxiDt(almsky2[_i], 0))

        t.checkpoint("  get_likgrad::Cartesian Grad. (%s map(s) lensed, %s fft(s)) " % (2 * len(typ), 2 * len(typ)))

        dphi = retdx * lib_qlm.get_ikx() + retdy * lib_qlm.get_iky()
        dOm = - retdx * lib_qlm.get_iky() + retdy * lib_qlm.get_ikx()
        return np.array([-2 * dphi, -2 * dOm])  # Factor 2 since gradient w.r.t. real and imag. parts.

    def degrade(self, LD_shape, no_lensing=False, ellmax=None, ellmin=None, libtodegrade='sky', lib_dir=None):
        """Degrades covariance matrix to some lower resolution.

        """

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
        finvLD = self.fi.degrade(LD_shape, no_lensing)
        return ffs_lencov_alm(lib_dir, lib_datalmLD, lib_skyalmLD,
                              self.cls_unl, self.cls_len, self.cl_transf, self.cls_noise, fLD, finvLD)

    def predMFpOlms(self, typ, lib_qlm, use_cls_len=False):
        """
        Perturbative prediction for the mean field <hat phi>_f,
        which is Rlm plm
        """
        Rpp, ROO, RpO = self.get_mfrespcls(typ, lib_qlm, use_cls_len=use_cls_len)
        del RpO
        plm, Olm = self.f.get_pOlm(lib_qlm)
        plm *= Rpp
        Olm *= ROO
        return np.array([plm, Olm])

    def eval_mf(self, typ, mfkey, xlms_sky, xlms_dat, lib_qlm, use_Pool=0, **kwargs):
        """Delfection-induced mean-field estimation from input random phases

            Args:
                typ: 'T', 'QU', 'TQU' for temperature-only, polarization-only or joint analysis
                mfkey: mean-field estimator key
                xlms_sky: i.i.d. sky modes random phases (if relevant)
                xlms_dat: i.i.d. data modes random phases (if relevant)
                lib_qlm: *ffs_alm* instance describing the lensing alm arrays

            Returns:
                mean-field estimate (gradient and curl component)


        """
        assert lib_qlm.ell_mat.lsides == self.lsides, (self.lsides, lib_qlm.ell_mat.lsides)
        assert typ in typs, (typ, typs)
        times = timer(_timed)
        ikx = lambda: self.lib_skyalm.get_ikx()
        iky = lambda: self.lib_skyalm.get_iky()
        times.checkpoint('Just started eval MF %s %s' % (typ, mfkey))
        if mfkey == 14:
            # W1 =  B^t F^t l^{1/2},  W2 = D daxi  D^t B^t F^t Cov_f^{-1}l^{-1/2}
            assert np.all([(_x.size == self.lib_datalm.alm_size) for _x in xlms_dat])

            ell_w = 1. / np.sqrt(np.arange(1, self.lib_datalm.ellmax + 2) - 0.5)
            for _i in range(len(typ)):
                xlms_dat[_i] = self.lib_datalm.almxfl(xlms_dat[_i], ell_w)

            def Bx(i):
                _cl = self.cl_transf[:self.lib_datalm.ellmax + 1] / ell_w ** 2
                _alm = self.lib_datalm.almxfl(xlms_dat[i], _cl)
                _alm = self.lib_skyalm.udgrade(self.lib_datalm, _alm)
                return self.lib_skyalm.alm2map(_alm)

            ilms, it = self.cd_solve(typ, xlms_dat, use_Pool=use_Pool, **kwargs)

            times.checkpoint('   Done with cd solving')

            for _i in range(len(typ)):
                ilms[_i] = self.lib_datalm.almxfl(ilms[_i], self.cl_transf)
            skyalms = np.empty((len(typ), self.lib_skyalm.alm_size), dtype=complex)
            for _i in range(len(typ)):
                skyalms[_i] = self.lib_skyalm.udgrade(self.lib_datalm, ilms[_i])
                skyalms[_i] = self.fi.lens_alm(self.lib_skyalm, skyalms[_i], mult_magn=True, use_Pool=use_Pool)
            del ilms
            times.checkpoint('   Done with first lensing')

            def _2lenmap(_alm):
                return self.f.alm2lenmap(self.lib_skyalm, _alm, use_Pool=use_Pool)

            dx = np.zeros(lib_qlm.alm_size, dtype=complex)
            dy = np.zeros(lib_qlm.alm_size, dtype=complex)

            for _i in range(len(typ)):
                tempalms = pmat.get_unlPmat_ij(typ, self.lib_skyalm, self.cls_unl, _i, 0) * skyalms[0]
                for _j in range(1, len(typ)):
                    tempalms += pmat.get_unlPmat_ij(typ, self.lib_skyalm, self.cls_unl, _i, _j) * skyalms[_j]
                dx += lib_qlm.map2alm(Bx(_i) * _2lenmap(tempalms * ikx()))
                dy += lib_qlm.map2alm(Bx(_i) * _2lenmap(tempalms * iky()))
            times.checkpoint('   Done with second lensing. Done.')
            del skyalms, tempalms

        elif mfkey == 0:
            # Std qest. We build the sim and use the std methods
            assert np.all([(_x.size == self.lib_skyalm.alm_size) for _x in xlms_sky])
            assert np.all([(_x.size == self.lib_datalm.alm_size) for _x in xlms_dat])

            sim = np.empty((len(typ), self.lib_datalm.alm_size), dtype=complex)
            for _i in range(len(typ)):
                skysim = self._get_rootpmatsky(typ, _i, 0) * xlms_sky[0]
                for _j in range(1, len(typ)):
                    skysim += self._get_rootpmatsky(typ, _i, _j) * xlms_sky[_j]
                skysim = self.f.lens_alm(self.lib_skyalm, skysim, use_Pool=use_Pool)
                sim[_i] = self.lib_datalm.udgrade(self.lib_skyalm, skysim)
                sim[_i] = self.lib_datalm.almxfl(sim[_i], self.cl_transf)
            if typ == 'QU':
                sim[0] += self.lib_datalm.almxfl(xlms_dat[0], np.sqrt(self.cls_noise['q']))
                sim[1] += self.lib_datalm.almxfl(xlms_dat[1], np.sqrt(self.cls_noise['u']))
            elif typ == 'TQU':
                sim[0] += self.lib_datalm.almxfl(xlms_dat[0], np.sqrt(self.cls_noise['t']))
                sim[1] += self.lib_datalm.almxfl(xlms_dat[1], np.sqrt(self.cls_noise['q']))
                sim[2] += self.lib_datalm.almxfl(xlms_dat[2], np.sqrt(self.cls_noise['u']))
            elif typ == 'T':
                sim[0] += self.lib_datalm.almxfl(xlms_dat[0], np.sqrt(self.cls_noise['t']))
            else:
                assert 0
            times.checkpoint('   Done with building sim')

            sim, it = self.cd_solve(typ, sim, use_Pool=use_Pool, **kwargs)
            for _i in range(len(typ)):  # xlms is now iblms
                sim[_i] = self.lib_datalm.almxfl(sim[_i], self.cl_transf)
            times.checkpoint('   Done with ivf')
            return self.get_qlms(typ, np.array([self.lib_skyalm.udgrade(self.lib_datalm, _s) for _s in sim]),
                                 lib_qlm=lib_qlm, use_Pool=use_Pool)

        else:
            dx = 0
            dy = 0
            it = 0
            assert 0, 'mfkey %s not implemented' % mfkey

        dphi = dx * lib_qlm.get_ikx() + dy * lib_qlm.get_iky()
        dOm = -dx * lib_qlm.get_iky() + dy * lib_qlm.get_ikx()
        return np.array([-2 * dphi, -2 * dOm]), it  # Factor 2 since gradient w.r.t. real and imag. parts.
