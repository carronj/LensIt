"""
Given quadratic estimator q =  W^{1a}_ij x^j W^{2a}_ik x^k (x : unit spectrum random phases,a in (0,1))
and q' with weights W <-> V
this has MC noise
<q q'> =  W^{1a} V^{*,1b} W^{2a} V^{*,2b} +  W^{1a} V^{*,2b} W^{2a} V^{*,1b}
"""
from __future__ import print_function

import hashlib
import os
import pickle as pk

import numpy as np

from lensit.pbs import pbs
from lensit.ffs_covs import ffs_cov as COV, ffs_specmat as SM, ell_mat
from lensit.misc.misc_utils import timer
from lensit.sims.sims_generic import hash_check

_timed = True

types = ['T', 'QU', 'TQU']


class MFMCnoise_lib:
    def __init__(self, lib_dir, lib_alm, cls_len, cl_transf, cls_noise):
        self.lib_alm = lib_alm
        self.cls_len = cls_len
        self.cls_noise = cls_noise
        self.cl_transf = np.zeros(self.lib_alm.ellmax + 1, dtype=float)
        self.cl_transf[:min(len(self.cl_transf), len(cl_transf))] = cl_transf[:min(len(self.cl_transf), len(cl_transf))]
        self.lib_dir = lib_dir
        if not os.path.exists(lib_dir) and pbs.rank == 0:
            os.makedirs(lib_dir)
        pbs.barrier()
        if not os.path.exists(lib_dir + '/Pmats') and pbs.rank == 0:
            os.makedirs(lib_dir + '/Pmats')
        pbs.barrier()

        if not os.path.exists(lib_dir + '/MFMC_hash.pk') and pbs.rank == 0:
            pk.dump(self.hashdict(), open(lib_dir + '/MFMC_hash.pk', 'w'))
        pbs.barrier()
        hash_check(self.hashdict(), pk.load(open(lib_dir + '/MFMC_hash.pk', 'r')))

    def hashdict(self):
        hash = {'transf': hashlib.sha1(self.cl_transf).hexdigest(),
                'lib_alm': self.lib_alm.hashdict()}
        for _k, _cl in self.cls_noise.iteritems():
            hash[_k] = hashlib.sha1(_cl).hexdigest()
        for _k, _cl in self.cls_len.iteritems():
            hash[_k] = hashlib.sha1(_cl).hexdigest()
        return hash

    def _get_datcls(self, fa, fb):
        """
        Anisotropic data spectrum (*not* beam deconvolved)
        :param fa:
        :param fb:
        :return:
        """
        assert fa in ['t', 'q', 'u'] and fb in ['t', 'q', 'u'], (fa, fb)
        i = {'t': 0, 'q': 1, 'u': 2}[fa]
        j = {'t': 0, 'q': 1, 'u': 2}[fb]
        return SM.get_datPmat_ij('TQU', self.lib_alm, self.cls_len, self.cl_transf, self.cls_noise, i, j)

    def _get_rootPmat(self, typ, i, j):
        assert i < len(typ) and j < len(typ), (i, j, typ)
        if i > j: return self._get_rootPmat(typ, j, i)
        fname = self.lib_dir + '/Pmats/%s_rootPmat_%s%s.npy' % (typ, i, j)
        if not os.path.exists(fname):
            rPmat = SM.get_Pmat(typ, self.lib_alm, self.cls_len,
                                cl_transf=self.cl_transf, cls_noise=self.cls_noise, square_root=True)
            for _i in range(len(typ)):
                for _j in range(_i, len(typ)):
                    _fname = self.lib_dir + '/Pmats/%s_rootPmat_%s%s.npy' % (typ, _i, _j)
                    np.save(_fname, rPmat[:, _i, _j])
                    print("Cached :", _fname)
        return np.load(fname)

    def _get_rootPmatinv(self, typ, i, j):
        assert i < len(typ) and j < len(typ), (i, j, typ)
        if i > j: return self._get_rootPmatinv(typ, j, i)
        fname = self.lib_dir + '/Pmats/%s_rootPmatinv_%s%s.npy' % (typ, i, j)
        if not os.path.exists(fname):
            rPmat = SM.get_Pmat(typ, self.lib_alm, self.cls_len,
                                cl_transf=self.cl_transf, cls_noise=self.cls_noise, square_root=True, inverse=True)
            for _i in range(len(typ)):
                for _j in range(_i, len(typ)):
                    _fname = self.lib_dir + '/Pmats/%s_rootPmatinv_%s%s.npy' % (typ, _i, _j)
                    np.save(_fname, rPmat[:, _i, _j])
                    print("Cached :", _fname)
        return np.load(fname)

    def _get_Pmatinv(self, typ, i, j):
        assert i < len(typ) and j < len(typ), (i, j, typ)
        if i > j: return self._get_Pmatinv(typ, j, i)
        fname = self.lib_dir + '/Pmats/%s_Pmatinv_%s%s.npy' % (typ, i, j)
        if not os.path.exists(fname):
            rPmat = SM.get_Pmat(typ, self.lib_alm, self.cls_len,
                                cl_transf=self.cl_transf, cls_noise=self.cls_noise, inverse=True)
            for _i in range(len(typ)):
                for _j in range(_i, len(typ)):
                    _fname = self.lib_dir + '/Pmats/%s_Pmatinv_%s%s.npy' % (typ, _i, _j)
                    np.save(_fname, rPmat[:, _i, _j])
                    print("Cached :", _fname)
        return np.load(fname)

    def _get_rootH(self, typ, i, j):
        """
        Return square root R of matrix H =  0.5(Cov^-1 Cl + (Cov^-1 Cl)^t)
        i.e. R R^t = H. (no beam)
        :param type:
        :return:
        """
        assert i < len(typ) and j < len(typ), (i, j, typ)
        if i > j: return self._get_rootH(typ, j, i)
        fname = self.lib_dir + '/%s_rootH_%s%s.npy' % (typ, i, j)
        if not os.path.exists(fname):
            Cinv = SM.get_Pmat(typ, self.lib_alm, self.cls_len,
                               cl_transf=self.cl_transf, cls_noise=self.cls_noise, inverse=True)
            R = np.zeros_like(Cinv)
            for _i in range(len(typ)):
                for _j in range(len(typ)):
                    for _k in range(0, len(typ)):
                        R[:, _i, _j] += Cinv[:, i, _k] * SM.get_unlPmat_ij(typ, self.lib_alm, self.cls_len, _k, j)
            del Cinv
            R = 0.5 * (R + np.swapaxes(R, 1, 2))  # symmetrisation without changing diagonal
            for _ell in range(self.lib_alm.ellmin, self.lib_alm.ellmax + 1):
                w, v = np.linalg.eigh(R[_ell, :, :])
                assert np.all(w >= 0.), w
                R_ell = np.dot(v, np.dot(np.diag(np.sqrt(w)), v.T))
                assert np.allclose(np.dot(R_ell, R_ell.transpose()), R[_ell, :, :])
                R[_ell, :, :] = np.copy(R_ell)
            for _i in range(len(typ)):
                for _j in range(_i, len(typ)):
                    _fname = self.lib_dir + '/%s_rootH_%s%s.npy' % (typ, _i, _j)
                    np.save(_fname, R[:, _i, _j])
                    print("Cached :", _fname)
        return np.load(fname)

    def _buildPmats(self, MFkey):
        W1 = []  # Weights for the first map in the pair (first in that list for y-axis, second for x-axis)
        W2 = []  # Weights for the second map in the pair (first in that list for y-axis, second for x-axis)
        # These two are list of functions.
        if MFkey == 0:
            # Standard qest thing, with N0 MC noise.
            # B^t F^t Cov^{-1} Cov 1/2 x,    B^t F^t ika Cl Cov^{-1} Cov 1/2 x
            ijsymmetry = False  # Cl Cov^{-1} is not symmetric if the noise is not identical in all fields

            def get_W1(typ, i, j):
                return self._get_rootPmatinv(typ, i, j) * self.cl_transf[self.lib_alm.reduced_ellmat()]

            def get_W2(typ, i, j):
                ret = SM.get_unlPmat_ij(typ, self.lib_alm, self.cls_len, i, 0) * self._get_rootPmatinv(typ, 0, j)
                for _k in range(1, len(typ)):
                    ret += SM.get_unlPmat_ij(typ, self.lib_alm, self.cls_len, i, _k) * self._get_rootPmatinv(typ,
                                                                                                               _k, j)
                return ret * self.cl_transf[self.lib_alm.reduced_ellmat()]

            W1.append(get_W1)
            W1.append(get_W1)
            W2.append(lambda t, i, j: get_W2(t, i, j) * self.lib_alm.get_iky())
            W2.append(lambda t, i, j: get_W2(t, i, j) * self.lib_alm.get_ikx())

        elif MFkey == 14:
            # W1 =  B^t l^{1/2},  W2 = ik_a C_l b_l Cov^{-1} l^{-1/2}
            ijsymmetry = False

            def get_W1(typ, i, j):
                _cl = np.sqrt(np.arange(1, self.lib_alm.ellmax + 2) - 0.5) \
                      * self.cl_transf[0:self.lib_alm.ellmax + 1] * (i == j)
                return _cl[self.lib_alm.reduced_ellmat()]

            def get_W2(typ, i, j):
                ret = SM.get_unlPmat_ij(typ, self.lib_alm, self.cls_len, i, 0) * self._get_Pmatinv(typ, 0, j)
                for _k in range(1, len(typ)):
                    ret += SM.get_unlPmat_ij(typ, self.lib_alm, self.cls_len, i, _k) * self._get_Pmatinv(typ, _k, j)
                _cl = self.cl_transf[0:self.lib_alm.ellmax + 1] / np.sqrt(np.arange(1, self.lib_alm.ellmax + 2) - 0.5)
                return ret * _cl[self.lib_alm.reduced_ellmat()]

            W1.append(get_W1)
            W1.append(get_W1)
            W2.append(lambda t, i, j: get_W2(t, i, j) * self.lib_alm.get_iky())
            W2.append(lambda t, i, j: get_W2(t, i, j) * self.lib_alm.get_ikx())

        elif MFkey == 12:
            # W1 =  B^t,  W2 = ik_a C_l b_l Cov^{-1}
            ijsymmetry = False

            def get_W1(typ, i, j):
                _cl = self.cl_transf[0:self.lib_alm.ellmax + 1] * (i == j)
                return _cl[self.lib_alm.reduced_ellmat()]

            def get_W2(typ, i, j):
                ret = SM.get_unlPmat_ij(typ, self.lib_alm, self.cls_len, i, 0) * self._get_Pmatinv(typ, 0, j)
                for _k in range(1, len(typ)):
                    ret += SM.get_unlPmat_ij(typ, self.lib_alm, self.cls_len, i, _k) * self._get_Pmatinv(typ, _k, j)
                _cl = self.cl_transf[0:self.lib_alm.ellmax + 1]
                return ret * _cl[self.lib_alm.reduced_ellmat()]

            W1.append(get_W1)
            W1.append(get_W1)
            W2.append(lambda t, i, j: get_W2(t, i, j) * self.lib_alm.get_iky())
            W2.append(lambda t, i, j: get_W2(t, i, j) * self.lib_alm.get_ikx())

        elif MFkey == 2:
            # W1 =  1,  W2 = ik_a C_l b_l Cov^{-1} b_l
            ijsymmetry = False

            def get_W1(typ, i, j):
                _cl = np.ones(self.lib_alm.ellmax + 1) * (i == j)
                return _cl[self.lib_alm.reduced_ellmat()]

            def get_W2(typ, i, j):
                ret = SM.get_unlPmat_ij(typ, self.lib_alm, self.cls_len, i, 0) * self._get_Pmatinv(typ, 0, j)
                for _k in range(1, len(typ)):
                    ret += SM.get_unlPmat_ij(typ, self.lib_alm, self.cls_len, i, _k) * self._get_Pmatinv(typ, _k, j)
                _cl = self.cl_transf[0:self.lib_alm.ellmax + 1] ** 2
                return ret * _cl[self.lib_alm.reduced_ellmat()]

            W1.append(get_W1)
            W1.append(get_W1)
            W2.append(lambda t, i, j: get_W2(t, i, j) * self.lib_alm.get_iky())
            W2.append(lambda t, i, j: get_W2(t, i, j) * self.lib_alm.get_ikx())

        else:
            ijsymmetry = False
            assert 0
        return W1, W2, ijsymmetry

    def eval(self, typ, MFkey, xlms, lib_qlm):
        """
        :param typ:
        :param MFkey:
        :param xlms: unit spectra random phases
        :param lib_qlm: output ffs_alm library
        :return:
        """
        assert typ in types, (typ, types)
        timer = timer(_timed)
        W1, W2, ijsymmetry = self._buildPmats(MFkey)
        W1a, W1b = W1
        W2a, W2b = W2
        timer.checkpoint('Starting MF qest eval. %s MFkey %s' % (typ, MFkey))
        dx = np.zeros(lib_qlm.alm_size, dtype=complex)
        dy = np.zeros(lib_qlm.alm_size, dtype=complex)
        _2map = lambda _alm: self.lib_alm.alm2map(_alm)
        _2alm = lambda _map: lib_qlm.map2alm(_map)
        for i in range(len(typ)):
            d1 = np.zeros(self.lib_alm.alm_size, dtype=complex)
            d2 = np.zeros(self.lib_alm.alm_size, dtype=complex)
            for j in range(len(typ)):
                d1 += W1a(typ, i, j) * xlms[j]
                d2 += W2a(typ, i, j) * xlms[j]
            dy += _2alm(_2map(d1) * _2map(d2))
            d1 = np.zeros(self.lib_alm.alm_size, dtype=complex)
            d2 = np.zeros(self.lib_alm.alm_size, dtype=complex)
            for j in range(len(typ)):
                d1 += W1b(typ, i, j) * xlms[j]
                d2 += W2b(typ, i, j) * xlms[j]
            dx += _2alm(_2map(d1) * _2map(d2))
        timer.checkpoint('Done.')

        fac = -2  # same convention as get_qlms, factor of 2 because of grad. w.r.t. real and imag. parts.
        dx *= fac
        dy *= fac
        return dx * lib_qlm.get_ikx() + dy * lib_qlm.get_iky(), - dx * lib_qlm.get_iky() + dy * lib_qlm.get_ikx()

    def evalMCnoise(self, typ, MFkey1, recache=False, MCnoise_floor=False):
        """
        N0-unnormalized qest estimators.
        From (Wa1^{ij} x_j Wa2^{ik} x_k) *(Vb1^{iq} x_q Vb2^{ij} x_q)^*
        = (Wa1 Vb1^dag)_{ij}(z)(Wa2 Vb2^dag)_{ij}(z) + (Wa1 Vb2^dag)_{ij}(z)(Wa2 Vb1^dag)_{ij}(z)
        :param typ: 'T', 'QU' or 'TQU'
        :param MFkey1:
        :param MFkey2:
        :return:
        """
        assert typ in types, (typ, types)
        MFkey2 = MFkey1
        fname = self.lib_dir + '/%s_%04d_%04d.dat' % (typ, MFkey1, MFkey2)
        if not os.path.exists(fname) or recache or MCnoise_floor:
            times = timer(_timed)
            lib_qlm = ell_mat.ffs_alm_pyFFTW(self.lib_alm.ell_mat, filt_func=lambda ell: ell >= 0)

            assert MFkey1 == MFkey2, 'Fix the rotation to phi-Omega space at the end'
            W1, W2, ijsymmetry = self._buildPmats(MFkey1)
            V1, V2 = W1, W2 if MFkey2 == MFkey1 else self._buildPmats(MFkey2)
            W1a, W1b = W1
            W2a, W2b = W2
            V1a, V1b = V1
            V2a, V2b = V2
            t = typ
            Covxx = np.zeros(lib_qlm.alm_size, dtype=complex)
            Covyy = np.zeros(lib_qlm.alm_size, dtype=complex)
            Covxy = np.zeros(lib_qlm.alm_size, dtype=complex)
            if MFkey2 != MFkey1:
                Covyx = np.zeros(lib_qlm.alm_size, dtype=complex)
            else:
                Covyx = Covxy

            _2map = lambda _alm: self.lib_alm.alm2map(_alm)
            _2alm = lambda _map: lib_qlm.map2alm(_map)

            def symm_fac(i, j):
                # If the weights are symmetric, i != j contributions count twice.
                return 1 + ijsymmetry * (i != j)

            times.checkpoint('Starting MCnoise calc.')
            for i in range(len(t)):  # Build Fourier space covariances :
                for j in range(i * ijsymmetry, len(t)):
                    sfac = symm_fac(i, j)
                    # First pairing :
                    if not MCnoise_floor:
                        Pmata = np.zeros(self.lib_alm.alm_size, dtype=complex)
                        Pmatb = np.zeros(self.lib_alm.alm_size, dtype=complex)
                        for k in range(len(t)):
                            Pmata += W1a(t, i, k) * V1a(t, j, k).conj()
                            Pmatb += W2a(t, i, k) * V2a(t, j, k).conj()
                        Covyy += _2alm(_2map(Pmata * sfac) * _2map(Pmatb))

                        Pmata = np.zeros(self.lib_alm.alm_size, dtype=complex)
                        Pmatb = np.zeros(self.lib_alm.alm_size, dtype=complex)
                        for k in range(len(t)):
                            Pmata += W1b(t, i, k) * V1b(t, j, k).conj()
                            Pmatb += W2b(t, i, k) * V2b(t, j, k).conj()
                        Covxx += _2alm(_2map(Pmata * sfac) * _2map(Pmatb))

                        Pmata = np.zeros(self.lib_alm.alm_size, dtype=complex)
                        Pmatb = np.zeros(self.lib_alm.alm_size, dtype=complex)
                        for k in range(len(t)):
                            Pmata += W1b(t, i, k) * V1a(t, j, k).conj()
                            Pmatb += W2b(t, i, k) * V2a(t, j, k).conj()
                        Covxy += _2alm(_2map(Pmata * sfac) * _2map(Pmatb))
                        # second pairing :
                    Pmata = np.zeros(self.lib_alm.alm_size, dtype=complex)
                    Pmatb = np.zeros(self.lib_alm.alm_size, dtype=complex)
                    for k in range(len(t)):
                        Pmata += W1a(t, i, k) * V2a(t, j, k).conj()
                        Pmatb += W2a(t, i, k) * V1a(t, j, k).conj()
                    Covyy += _2alm(_2map(Pmata * sfac) * _2map(Pmatb))

                    Pmata = np.zeros(self.lib_alm.alm_size, dtype=complex)
                    Pmatb = np.zeros(self.lib_alm.alm_size, dtype=complex)
                    for k in range(len(t)):
                        Pmata += W1b(t, i, k) * V2b(t, j, k).conj()
                        Pmatb += W2b(t, i, k) * V1b(t, j, k).conj()
                    Covxx += _2alm(_2map(Pmata * sfac) * _2map(Pmatb))

                    Pmata = np.zeros(self.lib_alm.alm_size, dtype=complex)
                    Pmatb = np.zeros(self.lib_alm.alm_size, dtype=complex)
                    for k in range(len(t)):
                        Pmata += W1b(t, i, k) * V2a(t, j, k).conj()
                        Pmatb += W2b(t, i, k) * V1a(t, j, k).conj()
                    Covxy += _2alm(_2map(Pmata * sfac) * _2map(Pmatb))

                    if MFkey2 != MFkey1:
                        Pmata = np.zeros(self.lib_alm.alm_size, dtype=complex)
                        Pmatb = np.zeros(self.lib_alm.alm_size, dtype=complex)
                        for k in range(len(t)):
                            Pmata += W1a(t, i, k) * V2b(t, j, k).conj()
                            Pmatb += W2a(t, i, k) * V1b(t, j, k).conj()
                        Covyx += _2alm(_2map(Pmata * sfac) * _2map(Pmatb))
                        Pmata = np.zeros(self.lib_alm.alm_size, dtype=complex)
                        Pmatb = np.zeros(self.lib_alm.alm_size, dtype=complex)
                        for k in range(len(t)):
                            Pmata += W1a(t, i, k) * V1b(t, j, k).conj()
                            Pmatb += W2a(t, i, k) * V2b(t, j, k).conj()
                        Covyx += _2alm(_2map(Pmata * sfac) * _2map(Pmatb))
                    times.checkpoint('%s %s Done' % (i, j))

            print('xy and yx allclose :', np.allclose(Covxy, Covxy))
            facunits = 1. / np.sqrt(np.prod(self.lib_alm.ell_mat.lsides))
            Covxx, Covyy, Covxy = COV.xylms_to_phiOmegalm(lib_qlm, Covxx * facunits, Covyy * facunits, Covxy * facunits)
            MCpp = lib_qlm.bin_realpart_inell(Covxx)[0:2 * self.lib_alm.ellmax + 1]
            MCOO = lib_qlm.bin_realpart_inell(Covyy)[0:2 * self.lib_alm.ellmax + 1]
            MCpO = lib_qlm.bin_realpart_inell(Covxy)[0:2 * self.lib_alm.ellmax + 1]
            sl = np.where((MCpp != 0) & (MCOO != 0))
            MCpO[sl] /= np.sqrt(np.abs(MCpp[sl] * MCOO[sl]))
            fsky = np.round(np.prod(self.lib_alm.ell_mat.lsides) / 4. / np.pi, 2)
            header = '%s MF MC noise for MF key %s' \
                     '\n 1st column Potential \n 2nd column Curl \n 3rd column Pot x Curl / root(Pot * Curl)\n' % (
                         typ, MFkey1)
            header += 'Produced by %s \n' % __file__
            header += 'ell-range (%s - %s) (possible gaps in-between), fsky = %s' % (
                self.lib_alm.ellmin, self.lib_alm.ellmax, fsky)
            np.savetxt(fname, np.array([MCpp, MCOO, MCpO]).transpose(), fmt=['%.8e'] * 3, header=header)
        return np.loadtxt(fname).transpose()
