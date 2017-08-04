"""
Contains some pure convenience functions for quick startup.
"""
import numpy as np
import healpy as hp
import os,sys
try:
    import gpu
except:
    print "NB : import of GPU module unsuccessful"

import ffs_covs
import ffs_iterators
import ffs_deflect
import curvedskylensing
import ffs_qlms
import misc
import pbs
import qcinv
import sims
import shts
import pseudocls

LENSITDIR = os.environ.get('LENSIT', './')

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
    for key, cl in misc.jc_camb.spectra_fromcambfile(
                    LENSITDIR + '/inputs/cls/fiducial_flatsky_lenspotentialCls.dat').iteritems():
        cls_unl[key] = cl[0:ellmax_sky + 1]
        if key == 'pp': cls_unl[key] = cl[:]  # might need this one
    cls_len = {}
    for key, cl in misc.jc_camb.spectra_fromcambfile(
                    LENSITDIR + '/inputs/cls/fiducial_flatsky_lensedCls.dat').iteritems():
        cls_len[key] = cl[0:ellmax_sky + 1]
    return cls_unl, cls_len

def get_partiallylenfidcls(w,ellmax_sky=ellmax_sky):
    # Produces spectra lensed with w_L * cpp_L
    params = misc.jc_camb.read_params(LENSITDIR + '/inputs/cls/fiducial_flatsky_params.ini')
    params['lensing_method'] = 4
    #FIXME : this would anyway not work in MPI mode beacause lensing method 4 does not.
    params['output_root'] = os.path.abspath(LENSITDIR + '/temp/camb_rank%s' % pbs.rank)
    ell = np.arange(len(w),dtype = int)
    np.savetxt(misc.jc_camb.PathToCamb + '/cpp_weights.txt', np.array([ell, w]).transpose(), fmt=['%i', '%10.5f'])
    misc.jc_camb.run_camb_fromparams(params)
    cllen = misc.jc_camb.spectra_fromcambfile(params['output_root'] + '_' + params['lensed_output_file'])
    ret = {}
    for key, cl in cllen.iteritems():
        ret[key] = cl[0:ellmax_sky + 1]
    return ret


def get_fidtenscls(ellmax_sky=ellmax_sky):
    cls = {}
    for key, cl in misc.jc_camb.spectra_fromcambfile(LENSITDIR + '/inputs/cls/fiducial_tensCls.dat').iteritems():
        cls[key] = cl[0:ellmax_sky + 1]
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
    return ffs_covs.ell_mat.ell_mat(LENSITDIR + '/temp/ellmats/ellmat_%s_%s' % (HD_res, LD_res), shape, lsides)


def get_lencmbs_lib(res=14, cache_sims=True, nsims=120, num_threads=4):
    """
    Default simulation library of 120 lensed CMB sims.
    Lensing is always performed at lcell 0.745 amin or so, and lensed CMB are generated on a square with sides lcell 2 ** res
    Will build all phases at the very first call if not already present.
    """
    HD_ellmat = get_ellmat(res, HD_res=res)
    ellmax_sky = 6000
    fsky = int(np.round(np.prod(HD_ellmat.lsides) / 4. / np.pi * 1000.))
    lib_skyalm = ffs_covs.ell_mat.ffs_alm_pyFFTW(HD_ellmat, num_threads=num_threads,
                                                 filt_func=lambda ell: ell <= ellmax_sky)
    skypha = sims.ffs_phas.ffs_lib_phas(LENSITDIR + '/temp/%s_sims/fsky%04d/len_alms/skypha' % (nsims, fsky), 4,
                                        lib_skyalm,
                                        nsims_max=nsims)
    if not skypha.is_full() and pbs.rank == 0:
        for _i, idx in misc.misc_utils.enumerate_progress(np.arange(nsims), label='Generating CMB phases'):
            skypha.get_sim(idx)
    pbs.barrier()
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky)
    return sims.ffs_cmbs.sims_cmb_len(LENSITDIR + '/temp/%s_sims/fsky%04d/len_alms' % (nsims, fsky), lib_skyalm,
                                      cls_unl,
                                      lib_pha=skypha, cache_lens=cache_sims)


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

    pixpha = sims.ffs_phas.pix_lib_phas(LENSITDIR + '/temp/%s_sims/fsky%04d/res%s/pixpha' % (nsims, fsky, LDres), 3,
                                        lib_datalm.ell_mat.shape, nsims_max=nsims)
    if not pixpha.is_full() and pbs.rank == 0:
        for _i, idx in misc.misc_utils.enumerate_progress(np.arange(nsims), label='Generating Noise phases'):
            pixpha.get_sim(idx)
    pbs.barrier()
    lib_dir = LENSITDIR + '/temp/%s_sims/fsky%04d/res%s/%s/maps' % (nsims, fsky, LDres, exp)
    return sims.ffs_maps.lib_noisemap(lib_dir, lib_datalm, len_cmbs, cl_transf, nTpix, nPpix, nPpix,
                                      pix_pha=pixpha, cache_sims=cache_maps)


def get_isocov(exp, LD_res, HD_res=14, pyFFTWthreads=4):
    """
    Set HD_res to 14 for full sky sampled at res LD.
    """
    sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky)

    cls_noise = {}
    cls_noise['t'] = (sN_uKamin * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1)  # simple flat noise Cls
    cls_noise['q'] = (sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1)  # simple flat noise Cls
    cls_noise['u'] = (sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1)  # simple flat noise Cls
    cl_transf = hp.gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=ellmax_sky)
    lib_alm = ffs_covs.ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                                              filt_func=lambda ell: (ell >= ellmin) & (ell <= ellmax),
                                              num_threads=pyFFTWthreads)
    lib_skyalm = ffs_covs.ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                                                 filt_func=lambda ell: (ell <= ellmax_sky), num_threads=pyFFTWthreads)

    lib_dir = LENSITDIR + '/temp/Covs/%s/LD%sHD%s' % (exp, LD_res, HD_res)
    return ffs_covs.ffs_cov.ffs_diagcov_alm(lib_dir, lib_alm, cls_unl, cls_len, cl_transf, cls_noise,
                                            lib_skyalm=lib_skyalm)
