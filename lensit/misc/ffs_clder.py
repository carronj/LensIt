from __future__ import print_function

import numpy as np

from lensit.misc.misc_utils import enumerate_progress

def get_dcllendclunl_pert(_type, ell_mat,lmaxlen,lmaxunl,clpp):
    """
    e.g. for the BB-EE part:
    MBB = M[0*M.shape[0]/3:1*M.shape[0]/3,M.shape[1]/3:2 * M.shape[1]/3]
    np.dot(MBB,cl_len['ee'][elllen]) is approximation to lensed BB
    (works vs CAMB outputs to better than a percent apparently.)
    Does not contain the zeroth order part.

    Based on: dC_llen^{A'B'} / dC^{AB}_lunl
       = R^{A'X}_llen R^{B'Y}_llenR ^{AX}_lunl R^{BY}_lunl * Sig^{ab}(len -unl) * (i ellunl_a i ellunl_b)
       with a bin over m', sum over m
    :param _type: 'T', 'QU' or 'TQU'
    :param ell_mat: ell_mat instance, containing info on patch size and resolution and flat-sky ell structure.
    :param lmaxlen:  calculates the matrix up to lensed Cls lmax lmaxlen
    :param lmaxunl:  calculates the matrix up to unlensed Cls lmax lmaxlen
    :param clpp: lensing potential spectrum cl
    :returns dC_llen / dCl_unl
    """
    if lmaxlen > lmaxunl:
        print(" this routine best for lmaxlen <= lmaxunl")
    assert _type in ['T', 'QU', 'TQU'], _type
    #FIXME: lmax ?
    libalm = ell_mat.ffs_alm_pyFFTW(ell_mat, filt_func=lambda ell: ell >= 0)
    _clpp = np.zeros(libalm.ellmax + 1)
    _clpp[:min(len(clpp),len(_clpp))]= clpp[:min(len(clpp),len(_clpp))]

    sigxx = libalm.alm2map(libalm.almxfl(libalm.get_ikx() ** 2, _clpp))
    sigyy = libalm.alm2map(libalm.almxfl(libalm.get_iky() ** 2, _clpp))
    sigxy = libalm.alm2map(libalm.almxfl(libalm.get_ikx() * libalm.get_iky(), _clpp))
    sigxx -= sigxx[0,0]
    sigyy -= sigyy[0,0]
    sigxy -= sigxy[0,0]

    fac = 1. / np.sqrt(np.prod(libalm.lsides))
    Nell = libalm.get_Nell()[:max(lmaxlen,lmaxunl) + 1]
    elllen, = Nell[:lmaxlen + 1].nonzero()
    ellunl, = Nell[:lmaxunl + 1].nonzero()
    ikx2 = libalm.get_ikx() ** 2
    iky2 = libalm.get_iky() ** 2
    i2kxy = 2 * libalm.get_ikx() * libalm.get_iky()

    if _type == 'T': # returns dC_llen^TT / dC_lunl^TT (tranposed)
        M = np.zeros((len(elllen), len(ellunl)), dtype=float)
        cl = np.zeros(libalm.ellmax + 1, dtype=float)
        for i, l in enumerate_progress(elllen, label='filling %s der. matrix up to %s' % (_type, lmaxlen)):
            cl[l] = 1. / Nell[l]
            ones = libalm.alm2map(libalm.almxfl(np.ones(libalm.alm_size,dtype = complex),cl))
            alm = ikx2 * libalm.map2alm(sigxx * ones) \
                + iky2 * libalm.map2alm(sigyy * ones) \
                + i2kxy * libalm.map2alm(sigxy * ones)
            M[i,:] = libalm.bin_realpart_inell(alm)[ellunl] * Nell[ellunl]
            cl[l] = 0.

    elif _type == 'QU':  # # returns dC_llen^(EE,BB,EB) / dC_lunl^(EE,BB,EB) (tranposed)
        # At least EE-BB part seems to work ok, although the output BB is not as smooth as it maybe should be.
        clmat = np.zeros((2, 2, libalm.ellmax + 1), dtype=float)
        XYlms = np.zeros((2, 2, libalm.alm_size), dtype=complex)
        M = np.zeros((3 * len(elllen), 3 * len(ellunl)), dtype=float)
        c, s = libalm.get_cossin_2iphi()
        for i, l in enumerate_progress(elllen, label='filling %s der. matrix up to %s' % (_type, lmaxlen)):
            for iA, ab in enumerate([(0, 0), (1, 1), (0, 1)]):
                a, b = ab
                clmat[a, b, l] =  1. / Nell[l]  # EE, BB or EB part
                clmat[b, a, l] =  1. / Nell[l]
                for iX,jX in [(0,0),(1,1),(0,1)]:
                    QUmap =  libalm.alm2map(_EBcls2QUPmatij(libalm, clmat, iX, jX, c=c, s=s))
                    XYlms[iX, jX] =  libalm.map2alm(sigxx * QUmap) * ikx2
                    XYlms[iX, jX] += libalm.map2alm(sigyy * QUmap) * iky2
                    XYlms[iX, jX] += libalm.map2alm(sigxy * QUmap) * i2kxy
                XYlms[1,0] = XYlms[0,1]
                retmat = _QUPmats2EBcls(libalm, XYlms, c=c, s=s)
                M[iA * M.shape[0] // 3 + i,0 * M.shape[1] // 3:1 * M.shape[1] // 3] = retmat[0, 0, ellunl] * Nell[ellunl]
                M[iA * M.shape[0] // 3 + i,1 * M.shape[1] // 3:2 * M.shape[1] // 3] = retmat[1, 1, ellunl] * Nell[ellunl]
                M[iA * M.shape[0] // 3 + i,2 * M.shape[1] // 3:3 * M.shape[1] // 3] = retmat[0, 1, ellunl] * Nell[ellunl]
                clmat[a, b, l] = 0.
                clmat[b, a, l] = 0.
    else:
        assert 0, '%s not implemented' % _type
    return M * fac

def get_dcllendclphi_pert(_type, ell_mat, lmaxlen, lmaxunl, cl_unl, BBonly = False):
    """
    It seems to work OK even if again thigs are not smooth, presumably because of non-smooth Nell beahvior.
    """
    #FIXME: disgusting piece of code.
    if lmaxlen > lmaxunl:
        print(" this routine best for lmaxlen <= lmaxunl")
    assert _type in ['T', 'QU', 'TQU'], _type
    #FIXME: lmax ?
    libalm = ell_mat.ffs_alm_pyFFTW(ell_mat, filt_func=lambda ell: ell >= 0)

    fac = 1. / np.sqrt(np.prod(libalm.lsides))
    Nell = libalm.get_Nell()[:max(lmaxlen,lmaxunl) + 1]
    elllen, = Nell[:lmaxlen + 1].nonzero()
    ellunl, = Nell[:lmaxunl + 1].nonzero()
    ikx = libalm.get_ikx
    iky = libalm.get_iky
    ikx2 = ikx() ** 2
    iky2 = iky() ** 2
    i2kxy = 2. * ikx() * iky()

    if _type == 'T': # returns dC_llen^TT / dC_lunl^TT (tranposed)
        _cltt = np.zeros(libalm.ellmax + 1)
        _cltt[:min(len(cl_unl['tt']), len(_cltt))] = cl_unl['tt'][:min(len(cl_unl['tt']), len(_cltt))]
        assert 0,'this is garbage'
        M = np.zeros((len(elllen), len(ellunl)), dtype=float)
        cl = np.zeros(libalm.ellmax + 1, dtype=float)
        Cxx = libalm.alm2map(libalm.almxfl(ikx2, _cltt))
        Cyy = libalm.alm2map(libalm.almxfl(iky2, _cltt))
        Cxy = libalm.alm2map(libalm.almxfl(iky() * ikx(), _cltt))
        for i, l in li.misc.misc_utils.enumerate_progress(elllen, label='filling %s der. matrix up to %s' % (_type, lmaxlen)):
            cl[l] = 1. / Nell[l]
            ones = libalm.alm2map(libalm.almxfl(np.ones(libalm.alm_size,dtype = complex),cl))
            alm = np.zeros(libalm.alm_size,dtype = complex)
            for kfac,_C in zip([ikx2,iky2,i2kxy],[Cxx,Cyy,Cxy]):
                qlm = libalm.map2alm(_C * ones)
                alm += kfac * (qlm -qlm[0])
            M[i,:] = libalm.bin_realpart_inell(alm)[ellunl] * Nell[ellunl]
            cl[l] = 0.

    elif _type == 'QU':  # # returns dC_llen^(EE,BB,EB) / dC_lunl^pp (tranposed)
        # At least EE-BB part seems to work ok, although the output BB is not as smooth as it maybe should be.
        M = np.zeros((3 * len(elllen) if not BBonly else len(elllen), len(ellunl)), dtype=float)
        c, s = libalm.get_cossin_2iphi()
        TEBcls = np.zeros((2,2,libalm.ellmax + 1))
        TEBcls[0, 0, :min(len(cl_unl['ee']), libalm.ellmax + 1)] = cl_unl['ee'][:min(len(cl_unl['ee']),libalm.ellmax + 1)]
        TEBcls[1, 1, :min(len(cl_unl['bb']), libalm.ellmax + 1)] = cl_unl['bb'][:min(len(cl_unl['bb']), libalm.ellmax + 1)]
        if 'eb' in cl_unl.keys():
            TEBcls[0, 1, :min(len(cl_unl['ee']), libalm.ellmax + 1)] = cl_unl['bb'][min(len(cl_unl['bb']), libalm.ellmax + 1)]
            TEBcls[1, 0] = TEBcls[0,1]

        Cxx = np.zeros((2,2,libalm.shape[0],libalm.shape[1]))
        Cxy = np.zeros((2,2,libalm.shape[0],libalm.shape[1]))
        Cyy = np.zeros((2,2,libalm.shape[0],libalm.shape[1]))
        for iX,jX in [(0, 0), (1, 1), (0, 1)]:
            Cxx[iX, jX] = libalm.alm2map(ikx2 * _EBcls2QUPmatij(libalm,TEBcls,iX,jX))
            Cyy[iX, jX] = libalm.alm2map(iky2 * _EBcls2QUPmatij(libalm, TEBcls, iX, jX))
            Cxy[iX, jX] = libalm.alm2map(ikx() * iky() * _EBcls2QUPmatij(libalm, TEBcls, iX, jX))
        clmat = np.zeros((2, 2, libalm.ellmax + 1), dtype=float)
        for i, l in enumerate_progress(elllen, label='filling %s der. matrix up to %s' % (_type, lmaxlen)):
            for iA, (a,b) in enumerate([(0, 0), (1, 1), (0, 1)] if not BBonly else [(1,1)]):
                clmat[a, b, l] =  1. / Nell[l]  # EE, BB or EB part
                clmat[b, a, l] =  1. / Nell[l]
                plms = np.zeros(libalm.alm_size, dtype=complex)
                for iX,jX in [(0,0),(1,1),(0,1)]:
                    ones = libalm.alm2map(_EBcls2QUPmatij(libalm, clmat, iX, jX, c=c, s=s))

                    QUlm =  libalm.map2alm(Cxx[iX,jX] * ones)
                    plms +=  ikx2 * (QUlm - QUlm[0])

                    QUlm =  libalm.map2alm(Cyy[iX,jX] * ones)
                    plms +=  iky2 * (QUlm - QUlm[0])

                    QUlm =  libalm.map2alm(Cxy[iX,jX] * ones)
                    plms += i2kxy * (QUlm - QUlm[0])
                M[iA * M.shape[0] // 3 + i if not BBonly else i ,:] = libalm.bin_realpart_inell(plms)[ellunl] * Nell[ellunl]
                clmat[a, b, l] = 0.
                clmat[b, a, l] = 0.
    else:
        assert 0, '%s not implemented' % _type
    return M * fac

def build_BBcov_pert(ellmat, lmaxBB, cl_unl, clBBobs, lmax_unl=6000):
    """Approximation to the BB covariance matrix.

    """
    Nell = ellmat.get_Nell()[:max(lmax_unl,lmaxBB) + 1]
    elllen, = Nell[:lmaxBB + 1].nonzero()
    ellunl, = Nell[:lmax_unl + 1].nonzero()

    # Gaussian part:
    covG = np.diag(2. * clBBobs[elllen] ** 2 / Nell[elllen])
    # sum_lunl dB/dphi_l covphi dB/dphi_l

    dBdp = get_dcllendclphi_pert('QU',ellmat,lmaxBB,lmax_unl,cl_unl,BBonly=True)
    covp = np.zeros( (len(elllen),len(elllen)),dtype = float)
    covpG = 2. * cl_unl['pp'][ellunl] ** 2 / Nell[ellunl]
    for i,l1 in enumerate_progress(elllen,label = 'filling dBdphi part of Cov matrix'):
        for j,l2 in enumerate(elllen):
            covp[i,j] = np.sum(covpG * dBdp[i] * dBdp[j])
    del dBdp
    # sum_lunl dB/dE_l covE dB/dE_l
    dBdE = get_dcllendclunl_pert('QU',ellmat,lmaxBB,lmax_unl,cl_unl['pp'])
    dBdE = dBdE[dBdE.shape[0]/3:2 * dBdE.shape[0]/3,0:dBdE.shape[1]/3]
    covE = np.zeros( (len(elllen),len(elllen)),dtype = float)
    covEG = 2. * cl_unl['ee'][ellunl] ** 2 / Nell[ellunl]
    for i,l1 in enumerate_progress(elllen,label = 'filling dBdE part of Cov matrix'):
        for j,l2 in enumerate(elllen):
            covE[i,j] = np.sum(covEG * dBdE[i] * dBdE[j])
    del dBdE

    return covG,covE,covp


def _EBcls2QUPmatij(lib_alm, TEBcls, i, j, c=None, s=None):
    """Turns E,B spectra into Q,U spectral matrices according to

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
    """Turns Q,U spectral matrices into E,B cls, according to

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
