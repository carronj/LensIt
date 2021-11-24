from __future__ import print_function
import os
import hashlib
import numpy as np, healpy as hp

from lensit.misc.misc_utils import binned, enumerate_progress
from lensit.ffs_covs import ell_mat, ffs_specmat


def apodize(lib_datalm, mask, sigma_fwhm_armin=12., lmax=None, method='hybrid', mult_factor=3, min_factor=0.1):
    """Flat sky apodizer directly adapted from AL curved sky libaml.apodize

    """
    if sigma_fwhm_armin <= 0.: return mask
    lmax = lmax or lib_datalm.ell_mat.ellmax
    libalm = ell_mat.ffs_alm_pyFFTW(lib_datalm.ell_mat, filt_func=lambda ell: ell <= lmax)
    print('fsky_unapodized = %.5f' % (np.sum(mask ** 2) / mask.size))
    bl = hp.gauss_beam(sigma_fwhm_armin / 60. / 180. * np.pi, lmax=lmax)
    apomask = libalm.alm2map(libalm.almxfl(libalm.map2alm(mask), bl))
    print('Min/max mask smoothed mask', np.min(apomask), np.max(apomask))
    print('fsky = %.5f' % (np.sum(apomask ** 2) / apomask.size))
    if method == 'gaussian': return apomask
    if method != 'hybrid': raise ValueError('Unknown apodization method')
    apomask = 1 - np.minimum(1., np.maximum(0., mult_factor * (1 - apomask) - min_factor))
    bl = hp.gauss_beam(sigma_fwhm_armin * 0.5 / 60. / 180. * np.pi, lmax=lmax)
    apomask = libalm.alm2map(libalm.almxfl(libalm.map2alm(apomask), bl))
    print('Min/max mask re-smoothed mask', np.min(apomask), np.max(apomask))
    print('fsky = %.5f' % (np.sum(apomask ** 2) / apomask.size))
    return apomask


class MSC_T:
    """Masked sky coupling pseudoCl deconvolution library.

    For temperature only. See MSC_P for polarization only.

    Set pedges and/or weights to bin the pCl part of the matrix.
    Set tedges and/or weights to bin Cl part of the matrix.

    If the coupling matrix is not square the deconvolution if performed via np.lstsq.

    MSC_T.map2cls(tmap) outputs estimated clTT up to MSC_T.lmax
    """

    def __init__(self, lib_datalm, apomask, lmax, cache_dir,
                 pedges=None, wp=None, tedges=None, wt=None):
        self.apomask = apomask
        self.lib_alm = ell_mat.ffs_alm_pyFFTW(lib_datalm.ell_mat, filt_func=lambda ell: ell <= lmax)
        lmax = self.lib_alm.ellmax  # !this might actually differ very slightly from lmax for small patches
        fname = cache_dir + '/lmax%s' % lmax + hashlib.sha1(apomask).hexdigest()
        shape_p = len(pedges) - 1 if pedges is not None else lmax + 1
        shape_t = len(tedges) - 1 if tedges is not None else lmax + 1

        if pedges is not None: fname + '_p' + hashlib.sha1(pedges).hexdigest()
        if wp is not None: fname + '_wp' + hashlib.sha1(wp(np.arange(lmax + 1))).hexdigest()
        if tedges is not None: fname + '_t' + hashlib.sha1(tedges).hexdigest()
        if wt is not None: fname + '_wt' + hashlib.sha1(wt(np.arange(lmax + 1))).hexdigest()
        fname += '.npy'
        if not os.path.exists(fname):
            if not os.path.exists(cache_dir): os.makedirs(cache_dir)
            M = get_MSCdense('T', self.lib_alm, apomask, lmax, full=pedges is not None or tedges is not None)
            if tedges is not None:
                print("Binning coupling matrix Cl side")
                ii, = np.where(self.lib_alm.get_Nell()[:lmax + 1] > 0)
                if wt is None: wt = lambda ell: np.ones(len(ell), dtype=float)
                wti = lambda ell: 1. / wt(ell)
                bu = tedges[1:] - 1
                bu[-1] += 1
                M = np.array([binned(m, ii, tedges[:-1], bu, w=wti, meanorsum='sum') for m in M])
            if pedges is not None:
                print("Binning coupling matrix pCl side")
                if wp is None: wp = lambda ell: np.ones(len(ell), dtype=float)
                ii, = np.where(self.lib_alm.get_Nell()[:lmax + 1] > 0)
                bu = pedges[1:] - 1
                bu[-1] += 1
                _M = np.zeros((len(bu), M.shape[1]))
                for i in range(M.shape[1]):
                    _M[:, i] = binned(M[:, i], ii, pedges[:-1], bu, w=wp)
                M = _M
            np.save(fname, self._invM(M) if M.shape[0] == M.shape[1] else M)
            print("Cached ", fname)

        self.M = None if shape_p == shape_t else np.load(fname)
        self.Mi = np.load(fname) if shape_p == shape_t  else None

        self.lmax = self.lib_alm.ellmax
        self.pedges = pedges
        self.tedges = tedges
        self.wt = wt
        self.wp = wp

    def _invM(self, M):
        return np.linalg.inv(M)

    def map2cl(self, tmap):
        return self._pcl2cl(self.lib_alm.alm2cl(self.lib_alm.map2alm(tmap * self.apomask)))

    def _pcl2cl(self, pcl):
        assert len(pcl) == self.lmax + 1
        if self.pedges is None and self.tedges is None:
            ret = np.zeros(self.lmax + 1, dtype=float)
            ii, = np.where(self.lib_alm.get_Nell()[:self.lmax + 1] > 0)
            ret[ii] = np.dot(self.Mi, pcl[ii])
            return ret
        if self.pedges is not None:
            ii, = np.where(self.lib_alm.get_Nell()[:self.lmax + 1] > 0)
            bu = self.pedges[1:] - 1
            bu[-1] += 1
            pcl = binned(pcl, ii, self.pedges[:-1], bu, w=self.wp)
        if self.Mi is not None:
            wtCl = np.dot(self.Mi, pcl)
        else:
            assert self.M is not None
            wtCl, res, rank, s = np.linalg.lstsq(self.M, pcl)
        if self.tedges is not None:
            bu = self.tedges[1:] - 1
            bu[-1] += 1
            bc = 0.5 * (bu + self.tedges[:-1])
            Cl = wtCl / self.wt(bc)
        else:
            Cl = wtCl
        return Cl

    def _map2pcl(self, tmap):
        return


class MSC_P:
    """Masked sky coupling pseudoCl deconvolution library.

    For polarization only. See MSC_T for temperature only.

    Set pedges and/or weights to bin the pCl part of the matrix (EE BB and EB separately).
    Set tedges and/or weights to bin Cl part of the matrix (EE BB and EB separately).

    If the coupling matrix is not square the deconvolution if performed via np.lstsq.

    MSC_P.map2cls(qumap) outputs estimated clEE,clBB and clEB up to MSC_P.lmax


    """

    def __init__(self, lib_datalm, apomask, lmax, cache_dir,
                 pedgess=(None, None, None), tedgess=(None, None, None), wts=(None, None, None),
                 wps=(None, None, None)):
        # Edges and weights : EE BB EB
        self.apomask = apomask
        self.lib_alm = ell_mat.ffs_alm_pyFFTW(lib_datalm.ell_mat, filt_func=lambda ell: ell <= lmax)
        self.wbins = np.any([p is not None for p in pedgess]) or np.any([p is not None for p in tedgess])
        lmax = self.lib_alm.ellmax  # !this might actually differ very slightly from lmax for small patches
        shape_p = np.array([len(p) - 1 if p is not None else lmax + 1 for p in pedgess])
        shape_t = np.array([len(t) - 1 if t is not None else lmax + 1 for t in tedgess])

        fname = cache_dir + '/lmax%s' % lmax + hashlib.sha1(apomask).hexdigest()
        for iA, (pedges, wp) in enumerate(zip(pedgess, wps)):
            if pedges is not None: fname + '_p%s' % iA + hashlib.sha1(pedges).hexdigest()
            if wp is not None: fname + '_wp%s' % iA + hashlib.sha1(wp(np.arange(lmax + 1))).hexdigest()
        for iA, (tedges, wt) in enumerate(zip(tedgess, wts)):
            if tedges is not None: fname + '_t%s' % iA + hashlib.sha1(tedges).hexdigest()
            if wt is not None: fname + '_wt%s' % iA + hashlib.sha1(wt(np.arange(lmax + 1))).hexdigest()
        fname += '.npy'
        if not os.path.exists(fname):
            if not os.path.exists(cache_dir): os.makedirs(cache_dir)
            M = get_MSCdense('QU', self.lib_alm, apomask, lmax, full=self.wbins)
            if self.wbins:
                assert M.shape == (3 * (lmax + 1), 3 * (lmax + 1)), M.shape
                _M = np.zeros((M.shape[0], np.sum(shape_t)), dtype=float)
                ii, = np.where(self.lib_alm.get_Nell()[:lmax + 1] > 0)
                print(shape_p, shape_t)
                for iA, (lab, tedges, wt) in enumerate(zip(['EE', 'BB', 'EB'], tedgess, wts)):
                    slice_t = slice(np.sum(shape_t[:iA]), np.sum(shape_t[:iA + 1]))
                    if tedges is not None:
                        print("Binning coupling matrix Cl side, %s" % lab)
                        if wt is None: wt = lambda ell: np.ones(len(ell), dtype=float)
                        wti = lambda ell: 1. / wt(ell)
                        bu = tedges[1:] - 1
                        bu[-1] += 1
                        for l in np.concatenate([ii, ii + 1 * (lmax + 1), ii + 2 * (lmax + 1)]):
                            _M[l, slice_t] = binned(M[l, iA * (lmax + 1):(iA + 1) * (lmax + 1)], ii, tedges[:-1], bu,
                                                    w=wti, meanorsum='sum')
                    else:
                        _M[:, slice_t] = M[:, slice_t]
                M = _M.copy()
                _M = np.zeros((np.sum(shape_t), M.shape[1]), dtype=float)
                for iA, (lab, pedges, wp) in enumerate(zip(['EE', 'BB', 'EB'], pedgess, wps)):
                    slice_p = slice(np.sum(shape_p[:iA]), np.sum(shape_p[:iA + 1]))
                    if pedges is not None:
                        print("Binning coupling matrix pCl side, %s" % lab)
                        if wp is None: wp = lambda ell: np.ones(len(ell), dtype=float)
                        bu = pedges[1:] - 1
                        bu[-1] += 1
                        for ib in range(M.shape[1]):
                            _M[slice_p, ib] = binned(M[iA * (lmax + 1):(iA + 1) * (lmax + 1), ib], ii, pedges[:-1], bu, w=wp)
                    else:
                        _M[slice_p, :] = M[slice_p, :]
                M = _M.copy()
            np.save(fname, self._invM(M) if M.shape[0] == M.shape[1] else M)
            print("Cached ", fname)

        self.M = None if np.sum(shape_t) == np.sum(shape_p) else np.load(fname)
        self.Mi = np.load(fname) if np.sum(shape_t) == np.sum(shape_p) else None
        print("Loaded ", fname)
        self.lmax = self.lib_alm.ellmax
        self.pedgess = pedgess
        self.tedgess = tedgess
        self.wts = wts
        self.wps = wps

    def map2cls(self, qumap):
        assert len(qumap) == 2, qumap.shape
        EB = self.lib_alm.QUlms2EBalms(np.array([self.lib_alm.map2alm(m * self.apomask) for m in qumap]))
        pcls = [self.lib_alm.alm2cl(EB[0])]
        pcls.append(self.lib_alm.alm2cl(EB[1]))
        pcls.append(self.lib_alm.alm2cl(EB[0], alm2=EB[1]))
        return self._pcls2cls(np.concatenate(pcls))

    def _invM(self, M):
        return np.linalg.inv(M)

    def _pcls2cls(self, pcls):
        assert len(pcls) == 3 * (self.lmax + 1)  # EE BB EB
        ii, = np.where(self.lib_alm.get_Nell()[:self.lmax + 1] > 0)
        if not self.wbins:
            assert not self.Mi is None
            ret = np.zeros(3 * (self.lmax + 1), dtype=float)
            jj = np.concatenate([ii, ii + 1 * (self.lmax + 1), ii + 2 * (self.lmax + 1)])
            ret[jj] = np.dot(self.Mi, pcls[jj])
            Cl = np.array([ret[:self.lmax + 1], ret[self.lmax + 1:2 * (self.lmax + 1)],
                           ret[2 * (self.lmax + 1):3 * (self.lmax + 1)]])
            return Cl
        # First bin the pcl, then rescale the output
        bpcl = []
        for iA, (lab, pedges, wp) in enumerate(zip(['EE', 'BB', 'EB'], self.pedgess, self.wps)):
            if pedges is not None:
                if wp is None: wp = lambda ell: np.ones(len(ell), dtype=float)
                bu = pedges[1:] - 1
                bu[-1] += 1
                bpcl.append(binned(pcls[iA * (self.lmax + 1): (iA + 1) * (self.lmax + 1)], ii, pedges[:-1], bu, w=wp))
            else:
                bpcl.append(pcls[iA * (self.lmax + 1): (iA + 1) * (self.lmax + 1)])
        bpcl = np.concatenate(bpcl)
        if self.Mi is not None:
            wtCl = np.dot(self.Mi, bpcl)
        else:
            assert self.M is not None
            wtCl, res, rank, s = np.linalg.lstsq(self.M, bpcl)
        Cl = []
        shape_t = np.array([len(t) - 1 if t is not None else self.lmax + 1 for t in self.tedgess])
        for iA, (lab, tedges, wt) in enumerate(zip(['EE', 'BB', 'EB'], self.tedgess, self.wts)):
            slice_t = slice(np.sum(shape_t[:iA]), np.sum(shape_t[:iA + 1]))
            if tedges is not None:
                if wt is None: wt = lambda ell: np.ones(len(ell), dtype=float)
                bu = tedges[1:] - 1
                bu[-1] += 1
                bc = 0.5 * (bu + tedges[:-1])
                Cl.append(wtCl[slice_t] / wt(bc))
            else:
                Cl.append(wtCl[slice_t])
        return np.array(Cl)


def apply_MSC(_type, cl, lib_datalm, mask):
    """Fast application of the cut-sky coupling matrix to cl vector, using 2D FFT methods.

    For polarization cl input is clEE,clBB,clEB.
    This uses on the flat sky:
        M_ll'^{AB,A'B'} = (bin in m, sum in m') |W_l-l'|^2 R_l AX R_l^*BY R_l XA' R_l^*YB'
    evaluating this in real space with 2D FFTs. (R is the rotation between X =T, Q,U and A in T,E,B.)


    """
    assert _type in ['T', 'QU', 'TQU']
    libalm = ell_mat.ffs_alm_pyFFTW(lib_datalm.ell_mat, filt_func=lambda ell: ell >= 0)
    fac = 1. / np.sqrt(np.prod(libalm.lsides))
    Bx = libalm.alm2map(np.abs(libalm.map2alm(mask)) ** 2)
    ii, = np.where(libalm.get_Nell()[:min(len(cl), libalm.ellmax + 1)] > 0)
    if _type == 'T':
        Xlm = np.ones(libalm.alm_size, dtype=complex)
        _cl = np.zeros(libalm.ellmax + 1, dtype=float)
        _cl[ii] = cl[ii]
        libalm.almxfl(Xlm, _cl, inplace=True)
        retlm = libalm.map2alm(Bx * libalm.alm2map(Xlm))
        return libalm.bin_realpart_inell(retlm) * fac
    elif _type == 'QU':
        assert len(cl == 3), len(cl)  # EE BB EB
        XYlms = np.ones((2, libalm.alm_size), dtype=complex)
        clmat = np.zeros((2, 2, libalm.ellmax + 1), dtype=float)
        clmat[0, 0, :min(libalm.ellmax + 1, len(cl[0]))] = cl[0, :min(libalm.ellmax + 1, len(cl[0]))]
        clmat[1, 1, :min(libalm.ellmax + 1, len(cl[1]))] = cl[1, :min(libalm.ellmax + 1, len(cl[1]))]
        clmat[0, 1, :min(libalm.ellmax + 1, len(cl[2]))] = cl[2, :min(libalm.ellmax + 1, len(cl[2]))]
        clmat[1, 0] = clmat[0, 1]
        c, s = libalm.get_cossin_2iphi()
        XYlms[0, 0] = libalm.map2alm(Bx * libalm.alm2map(_EBcls2QUPmatij(libalm, clmat, 0, 0, c=c, s=s)))
        XYlms[1, 1] = libalm.map2alm(Bx * libalm.alm2map(_EBcls2QUPmatij(libalm, clmat, 1, 1, c=c, s=s)))
        XYlms[0, 1] = libalm.map2alm(Bx * libalm.alm2map(_EBcls2QUPmatij(libalm, clmat, 0, 1, c=c, s=s)))
        XYlms[1, 0] = XYlms[0, 1]
        retmat = ffs_specmat.QUPmats2EBcls(libalm, XYlms)
        return np.array([retmat[0, 0], retmat[1, 1], retmat[0, 1]]) * fac
    else:
        assert 0, '%s not implemented' % _type


def get_MSCdense(_type, lib_datalm, mask, lmax, full=False):
    """
    Returns the dense cut-sky coupling matrix M.
    This assumes that the X_lms (true) are cut at lmax. Apart from that the calculation is exact to FFT prec.
    This uses on the flat sky:
        M_ll'^{AB,A'B'} = (bin in m, sum in m') |W_l-l'|^2 R_l AX R_l^*BY R_l XA' R_l^*YB'
    evaluating this in real space with 2D FFTs. (R is the rotation between X =T, Q,U and A in T,E,B.)
    """
    assert _type in ['T', 'QU', 'TQU'], _type
    libalm = ell_mat.ffs_alm_pyFFTW(lib_datalm.ell_mat, filt_func=lambda ell: ell <= 2 * lmax)
    ii, = np.where(libalm.get_Nell()[:lmax + 1] > 0)
    Bx = libalm.alm2map(np.abs(libalm.map2alm(mask)) ** 2)
    fac = 1. / np.sqrt(np.prod(libalm.lsides))
    if _type == 'T':
        M = np.zeros((lmax + 1, lmax + 1), dtype=float) if full else np.zeros((len(ii), len(ii)), dtype=float)
        cl = np.zeros(libalm.ellmax + 1, dtype=float)
        for i, idx in enumerate_progress(ii, label='filling %s MSC matrix up to %s' % (_type, lmax)):
            cl[idx] = 1.
            Alm = np.ones(libalm.alm_size, dtype=complex)
            libalm.almxfl(Alm, cl, inplace=True)
            M[:, i if not full else idx] = libalm.bin_realpart_inell(libalm.map2alm(Bx * libalm.alm2map(Alm)))[
                ii if not full else slice(0, lmax + 1)]
            cl[idx] = 0.

    elif _type == 'QU':  # EE BB EB
        clmat = np.zeros((2, 2, libalm.ellmax + 1), dtype=float)
        XYlms = np.zeros((2, 2, libalm.alm_size), dtype=complex)
        M = np.zeros((3 * (lmax + 1), 3 * (lmax + 1)), dtype=float) if full else np.zeros((len(ii) * 3, len(ii) * 3),
                                                                                          dtype=float)
        c, s = libalm.get_cossin_2iphi()
        for i, idx in enumerate_progress(ii, label='filling %s MSC matrix up to %s' % (_type, lmax)):
            for iA, ab in enumerate([(0, 0), (1, 1), (0, 1)]):
                a, b = ab
                clmat[a, b, idx] = 1.  # EE, BB or EB part
                clmat[b, a, idx] = 1.  # EE, BB or EB part
                XYlms[0, 0] = libalm.map2alm(Bx * libalm.alm2map(_EBcls2QUPmatij(libalm, clmat, 0, 0, c=c, s=s)))
                XYlms[1, 1] = libalm.map2alm(Bx * libalm.alm2map(_EBcls2QUPmatij(libalm, clmat, 1, 1, c=c, s=s)))
                XYlms[0, 1] = libalm.map2alm(Bx * libalm.alm2map(_EBcls2QUPmatij(libalm, clmat, 0, 1, c=c, s=s)))
                XYlms[1, 0] = XYlms[0, 1]
                retmat = _QUPmats2EBcls(libalm, XYlms, c=c, s=s)
                M[0 * M.shape[0] // 3:1 * M.shape[0] // 3, iA * M.shape[1] // 3 + (i if not full else idx)] = retmat[
                    0, 0, ii if not full else slice(0, lmax + 1)]
                M[1 * M.shape[0] // 3:2 * M.shape[0] // 3, iA * M.shape[1] // 3 + (i if not full else idx)] = retmat[
                    1, 1, ii if not full else slice(0, lmax + 1)]
                M[2 * M.shape[0] // 3:3 * M.shape[0] // 3, iA * M.shape[1] // 3 + (i if not full else idx)] = retmat[
                    0, 1, ii if not full else slice(0, lmax + 1)]
                clmat[a, b, idx] = 0.
                clmat[b, a, idx] = 0.
    else:
        assert 0, '%s not implemented' % _type
    return M * fac


def _EBcls2QUPmatij(lib_alm, TEBcls, i, j, c=None, s=None):
    """
    Turns E,B spectra into Q,U spectral matrices according to
        E = cos Q + sin U
        B = -sin Q + cos U
    """
    assert TEBcls.ndim == 3 and TEBcls.shape[0] == 2 and TEBcls.shape[1] == 2, (TEBcls.shape)
    fl = lambda alm, i, j: lib_alm.almxfl(alm, TEBcls[i, j])
    if c is None or s is None: c, s = lib_alm.get_cossin_2iphi()
    if i == 0 and j == 0:
        return fl(c ** 2, 0, 0) + fl(s ** 2, 1, 1) - fl(c * s, 0, 1) - fl(c * s, 1, 0)
    if i == 1 and j == 1:
        return fl(s ** 2, 0, 0) + fl(c ** 2, 1, 1) + fl(c * s, 0, 1) + fl(c * s, 1, 0)
    if i == 0 and j == 1:
        return fl(c * s, 0, 0) - fl(c * s, 1, 1) + fl(c ** 2, 0, 1) - fl(s ** 2, 1, 0)
    if i == 1 and j == 0:
        return fl(c * s, 0, 0) - fl(c * s, 1, 1) - fl(s ** 2, 0, 1) + fl(c ** 2, 1, 0)
    assert 0, (i, j)


def _QUPmats2EBcls(lib_alm, QUpmat, c=None, s=None):
    """
    Turns Q,U spectral matrices into E,B cls, according to
        E = cos Q + sin U
        B = -sin Q + cos U
    """
    assert QUpmat.shape == (2, 2, lib_alm.alm_size), ((2, 2, lib_alm.alm_size), QUpmat.shape)
    ret = np.zeros((2, 2, lib_alm.ellmax + 1), dtype=float)
    bin2cl = lambda _alm: lib_alm.bin_realpart_inell(_alm)[0:lib_alm.ellmax + 1]
    if c is None or s is None: c, s = lib_alm.get_cossin_2iphi()
    ret[0, 0, :] = bin2cl(QUpmat[0, 0, :] * c ** 2 + QUpmat[1, 1, :] * s ** 2 + 2 * c * s * QUpmat[0, 1, :])
    ret[1, 1, :] = bin2cl(QUpmat[1, 1, :] * c ** 2 + QUpmat[0, 0, :] * s ** 2 - 2 * c * s * QUpmat[0, 1, :])
    ret[0, 1, :] = bin2cl((QUpmat[1, 1, :] - QUpmat[0, 0, :]) * c * s + QUpmat[0, 1, :] * (c ** 2 - s ** 2))
    ret[1, 0, :] = ret[0, 1, :]
    return ret
