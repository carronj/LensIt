"""
Contains some pure convenience functions for quick startup.
"""
from __future__ import print_function

import numpy as np
import healpy as hp
import os

from lensit.ffs_covs import ffs_cov, ell_mat
from lensit.sims import ffs_phas, ffs_maps, ffs_cmbs
from lensit.pbs import pbs
from lensit.misc.misc_utils import enumerate_progress, camb_clfile

LENSITDIR = os.environ.get('LENSIT', './')
CLSPATH = os.path.join(LENSITDIR, 'inputs', 'cls')

ellmax_sky = 6000


def get_config(exp):
    sN_uKaminP = None
    if exp == 'Planck':
        sN_uKamin = 35.
        Beam_FWHM_amin = 7.
        ellmin = 10
        ellmax = 2048
    elif exp == 'Planck_65':
        sN_uKamin = 35.
        Beam_FWHM_amin = 6.5
        ellmin = 100
        ellmax = 2048
    elif exp == 'S4':
        sN_uKamin = 1.5
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    elif exp == 'S5':
        sN_uKamin = 1.5 / 4.
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    elif exp == 'S6':
        sN_uKamin = 1.5 / 4. / 4.
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    elif exp == 'SO':
        sN_uKamin = 3.
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    elif exp == 'SOb1':
        sN_uKamin = 3.
        Beam_FWHM_amin = 1.
        ellmin = 10
        ellmax = 3000
    elif exp == 'SOmark':
        sN_uKamin = 10.
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    elif exp == 'PB85':
        sN_uKamin = 8.5 /np.sqrt(2.)
        Beam_FWHM_amin = 3.5
        ellmin = 10
        ellmax = 3000
    elif exp == 'PB5':
        sN_uKamin = 5. / np.sqrt(2.)
        Beam_FWHM_amin = 3.5
        ellmin = 10
        ellmax = 3000
    elif exp == 'fcy_mark':
        sN_uKamin = 5.
        Beam_FWHM_amin = 1.4
        ellmin=10
        ellmax=3000
    else:
        sN_uKamin = 0
        Beam_FWHM_amin = 0
        ellmin = 0
        ellmax = 0
        assert 0, '%s not implemented' % exp
    sN_uKaminP = sN_uKaminP or np.sqrt(2.) * sN_uKamin
    return sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax


def get_fidcls(ellmax_sky=ellmax_sky):
    cls_unl = {}
    cls_unlr = camb_clfile(os.path.join(CLSPATH, 'fiducial_flatsky_lenspotentialCls.dat'))
    for key in cls_unlr.keys():
        cls_unl[key] = cls_unlr[key][0:ellmax_sky + 1]
        if key == 'pp': cls_unl[key] = cls_unlr[key][:]  # might need this one to higher lmax
    cls_len = {}
    cls_lenr = camb_clfile(os.path.join(CLSPATH, 'fiducial_flatsky_lensedCls.dat'))
    for key in cls_lenr.keys():
        cls_len[key] = cls_lenr[key][0:ellmax_sky + 1]
    return cls_unl, cls_len


def get_fidtenscls(ellmax_sky=ellmax_sky):
    cls = {}
    cls_tens = camb_clfile(os.path.join(CLSPATH, 'fiducial_tensCls.dat'))
    for key in cls_tens.keys():
        cls[key] = cls_tens[key][0:ellmax_sky + 1]
    return cls

def get_ellmat(LD_res, HD_res=14):
    """
    Standardized ellmat instances.
    Returns ellmat with 2 ** LD_res squared points with
    lcell = 0.745 * (2 ** (HD_res - LD_res)) and lsides lcell * 2 ** LD_res.
    Set HD_res to 14 for full sky ell_mat.
    :param LD_res:
    :param HD_res:
    :return:
    """
    assert HD_res <= 14 and LD_res <= 14, (LD_res, HD_res)
    lcell_rad = (np.sqrt(4. * np.pi) / 2 ** 14) * (2 ** (HD_res - LD_res))
    shape = (2 ** LD_res, 2 ** LD_res)
    lsides = (lcell_rad * 2 ** LD_res, lcell_rad * 2 ** LD_res)
    lib_dir = os.path.join(LENSITDIR, 'temp', 'ellmats', 'ellmat_%s_%s' % (HD_res, LD_res))
    return ell_mat.ell_mat(lib_dir, shape, lsides)


def get_lencmbs_lib(res=14, cache_sims=True, nsims=120, num_threads=4):
    """
    Default simulation library of 120 lensed CMB sims.
    Lensing is always performed at lcell 0.745 amin or so, and lensed CMB are generated on a square with sides lcell 2 ** res
    Will build all phases at the very first call if not already present.
    """
    HD_ellmat = get_ellmat(res, HD_res=res)
    ellmax_sky = 6000
    fsky = int(np.round(np.prod(HD_ellmat.lsides) / 4. / np.pi * 1000.))
    lib_skyalm = ell_mat.ffs_alm_pyFFTW(HD_ellmat, num_threads=num_threads,
                                                 filt_func=lambda ell: ell <= ellmax_sky)
    skypha_libdir = os.path.join(LENSITDIR, 'temp', '%s_sims'%nsims, 'fsky%04d'%fsky, 'len_alms', 'skypha')
    skypha = ffs_phas.ffs_lib_phas(skypha_libdir, 4, lib_skyalm, nsims_max=nsims)
    if not skypha.is_full() and pbs.rank == 0:
        for i, idx in enumerate_progress(np.arange(nsims, dtype=int), label='Generating CMB phases'):
            skypha.get_sim(int(idx))
    pbs.barrier()
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky)
    sims_libdir = os.path.join(LENSITDIR, 'temp', '%s_sims'%nsims,'fsky%04d'%fsky, 'len_alms')
    return ffs_cmbs.sims_cmb_len(sims_libdir, lib_skyalm, cls_unl, lib_pha=skypha, cache_lens=cache_sims)


def get_maps_lib(exp, LDres, HDres=14, cache_lenalms=True, cache_maps=False, nsims=120, num_threads=4):
    """
    Default simulation library of 120 full flat sky sims for exp 'exp' at resolution LDres.
    Different exp at same resolution share the same random phases both in CMB and noise
        Will build all phases at the very first call if not already present.
    :param exp: 'Planck', 'S4' ... See get_config
    :param LDres: 14 : cell length is 0.745 amin, 13 : 1.49 etc.
    :return: sim library instance
    """
    sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    len_cmbs = get_lencmbs_lib(res=HDres, cache_sims=cache_lenalms, nsims=nsims)
    lmax_sky = len_cmbs.lib_skyalm.ellmax
    cl_transf = hp.gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=lmax_sky)
    lib_datalm = ffs_covs.ell_mat.ffs_alm_pyFFTW(get_ellmat(LDres, HDres), filt_func=lambda ell: ell <= lmax_sky,
                                                 num_threads=num_threads)
    fsky = int(np.round(np.prod(len_cmbs.lib_skyalm.ell_mat.lsides) / 4. / np.pi * 1000.))
    vcell_amin2 = np.prod(lib_datalm.ell_mat.lsides) / np.prod(lib_datalm.ell_mat.shape) * (180 * 60. / np.pi) ** 2
    nTpix = sN_uKamin / np.sqrt(vcell_amin2)
    nPpix = sN_uKaminP / np.sqrt(vcell_amin2)

    pixpha_libdir = os.path.join(LENSITDIR, 'temp', '%s_sims'%nsims, 'fsky%04d'%fsky, 'res%s'%LDres, 'pixpha')
    pixpha = ffs_phas.pix_lib_phas(pixpha_libdir, 3, lib_datalm.ell_mat.shape, nsims_max=nsims)

    if not pixpha.is_full() and pbs.rank == 0:
        for _i, idx in enumerate_progress(np.arange(nsims), label='Generating Noise phases'):
            pixpha.get_sim(idx)
    pbs.barrier()
    sims_libdir = os.path.join(LENSITDIR, 'temp', '%s_sims'%nsims,'fsky%04d'%fsky, 'res%s'%LDres,'%s'%exp, 'maps')
    return ffs_maps.lib_noisemap(sims_libdir, lib_datalm, len_cmbs, cl_transf, nTpix, nPpix, nPpix,
                                      pix_pha=pixpha, cache_sims=cache_maps)


def get_isocov(exp, LD_res, HD_res=14, pyFFTWthreads=4):
    """
    Set HD_res to 14 for full sky sampled at res LD.
    """
    sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky)

    cls_noise = {'t': (sN_uKamin * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1),
                 'q':(sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1),
                 'u':(sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1)}  # simple flat noise Cls
    cl_transf = hp.gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=ellmax_sky)
    lib_alm = ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                        filt_func=lambda ell: (ell >= ellmin) & (ell <= ellmax), num_threads=pyFFTWthreads)
    lib_skyalm = ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                        filt_func=lambda ell: (ell <= ellmax_sky), num_threads=pyFFTWthreads)

    lib_dir = os.path.join(LENSITDIR, 'temp', 'Covs', '%s'%exp, 'LD%sHD%s'%(exp, LD_res, HD_res))
    return ffs_cov.ffs_diagcov_alm(lib_dir, lib_alm, cls_unl, cls_len, cl_transf, cls_noise, lib_skyalm=lib_skyalm)
