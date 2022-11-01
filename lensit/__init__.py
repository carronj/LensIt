"""This lensit package contains some convenience functions in its __init__.py for quick startup.

"""
from __future__ import print_function

import numpy as np
import os

from lensit.ffs_covs import ffs_cov, ell_mat
from lensit.sims import ffs_phas, ffs_maps, ffs_cmbs
from lensit.pbs import pbs
from lensit.misc.misc_utils import enumerate_progress, camb_clfile, gauss_beam, cl_inverse, npy_hash


def _get_lensitdir():
    assert 'LENSIT' in os.environ.keys(), 'Set LENSIT env. variable to somewhere safe to write'
    LENSITDIR = os.environ.get('LENSIT')
    CLSPATH = os.path.join(os.path.dirname(__file__), 'data', 'cls')
    return LENSITDIR, CLSPATH


def get_fidcls(ellmax_sky=6000, alpha_cpp=1., new_cls=False, cls_grad=False):
    r"""Returns *lensit* fiducial CMB spectra (Planck 2015 cosmology)

    Args:
        ellmax_sky: optionally reduces outputs spectra :math:`\ell_{\rm max}`
        alpha_cpp: Multiplicative factor for Cl_pp, in order to test response 
                of the estimator to a linear change of the input Cl_pp
    Returns:
        unlensed and lensed CMB spectra (dicts)

    """
    cls_unl = {}
    cls_unlr = camb_clfile(os.path.join(_get_lensitdir()[1], 'fiducial_flatsky_lenspotentialCls.dat'))
    for key in cls_unlr.keys():
        cls_unl[key] = cls_unlr[key][0:ellmax_sky + 1]
        if key == 'pp': cls_unl[key] = cls_unlr[key][:] * alpha_cpp  # might need this one to higher lmax
    cls_len = {}
    #print(new_cls)
    if alpha_cpp == 1.:
        if new_cls:
            print('using new cls')
            from scipy.interpolate import UnivariateSpline as spl
            #cls_lenr = camb_clfile(os.path.join(_get_lensitdir()[1], 'inports', 'lensit1011_lensedCls.dat'))
            cls_len_d = np.loadtxt(os.path.join(_get_lensitdir()[1], 'inports', 'lensit1011_lensedCls.dat')).T
            cls_lenr = {}
            ell = cls_len_d[0]
            # cls_lenr['tt'] = np.interp(np.arange(8000), ell, cls_len_d[1])
            # cls_lenr['ee'] = np.interp(np.arange(8000), ell, cls_len_d[2])
            # cls_lenr['bb'] = np.interp(np.arange(8000), ell, cls_len_d[3])
            # cls_lenr['te'] = np.interp(np.arange(8000), ell, cls_len_d[4])
            ls = np.arange(8000)
            cls_lenr['tt'] = spl(ell, cls_len_d[1], k=2, s=0, ext='zeros')(ls)
            cls_lenr['ee'] = spl(ell, cls_len_d[2], k=2, s=0, ext='zeros')(ls)
            cls_lenr['bb'] = spl(ell, cls_len_d[3], k=2, s=0, ext='zeros')(ls)
            cls_lenr['te'] = spl(ell, cls_len_d[4], k=2, s=0, ext='zeros')(ls)
            # for k in cls_lenr.keys():
            #     cls_lenr[k] = np.append(cls_lenr[k], np.zeros(8000))
        else:
            cls_lenr = camb_clfile(os.path.join(_get_lensitdir()[1], 'fiducial_flatsky_lensedCls.dat'))
    else:
        from camb.correlations import lensed_cls
        cls_unlr, cldd = cls2dls(cls_unl)
        cls_lenr = dls2cls(lensed_cls(cls_unlr, cldd))
    for key in cls_lenr.keys():
        cls_len[key] = cls_lenr[key][0:ellmax_sky + 1]
    
    if cls_grad:
        cls_grad_d = np.loadtxt(os.path.join(_get_lensitdir()[1], 'inports', 'lensit1011_gradlensedCls.dat')).T
        cls_grad = {}
        ell = cls_grad_d[0]
        ls = np.arange(ellmax_sky+1)
        cls_grad['tt'] = spl(ell, cls_grad_d[1], k=2, s=0, ext='zeros')(ls)
        cls_grad['ee'] = spl(ell, cls_grad_d[2], k=2, s=0, ext='zeros')(ls)
        cls_grad['bb'] = spl(ell, cls_grad_d[3], k=2, s=0, ext='zeros')(ls)
        cls_grad['te'] = spl(ell, cls_grad_d[4], k=2, s=0, ext='zeros')(ls)

        return cls_unl, cls_len, cls_grad

    return cls_unl, cls_len



def cls2dls(cls):
    """Turns cls dict. into camb cl array format"""
    keys = ['tt', 'ee', 'bb', 'te']
    lmax = np.max([len(cl) for cl in cls.values()]) - 1
    dls = np.zeros((lmax + 1, 4), dtype=float)
    refac = np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float) / (2. * np.pi)
    for i, k in enumerate(keys):
        cl = cls.get(k, np.zeros(lmax + 1, dtype=float))
        sli = slice(0, min(len(cl), lmax + 1))
        dls[sli, i] = cl[sli] * refac[sli]
    cldd = np.copy(cls.get('pp', None))
    if cldd is not None:
        cldd *= np.arange(len(cldd)) ** 2 * np.arange(1, len(cldd) + 1, dtype=float) ** 2 /  (2. * np.pi)
    return dls, cldd

def dls2cls(dls):
    """Inverse operation to cls2dls"""
    assert dls.shape[1] == 4
    lmax = dls.shape[0] - 1
    cls = {}
    refac = 2. * np.pi * cl_inverse( np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float))
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls[k] = dls[:, i] * refac
    return cls

def get_fidtenscls(ellmax_sky=6000):
    cls = {}
    cls_tens = camb_clfile(os.path.join(_get_lensitdir()[1], 'fiducial_tensCls.dat'))
    for key in cls_tens.keys():
        cls[key] = cls_tens[key][0:ellmax_sky + 1]
    return cls

def get_ellmat(LD_res, HD_res):
    r"""Default ellmat instances.


    Returns:
        *ell_mat* instance describing a flat-sky square patch of physical size :math:`\sim 0.74 *2^{\rm HDres}` arcmin,
        sampled with :math:`2^{\rm LDres}` points on a side.

    The patch area is :math:`4\pi` if *HD_res* = 14

    """
    assert HD_res <= 14 and LD_res <= 14, (LD_res, HD_res)
    lcell_rad = (np.sqrt(4. * np.pi) / 2 ** 14) * (2 ** (HD_res - LD_res))
    shape = (2 ** LD_res, 2 ** LD_res)
    lsides = (lcell_rad * 2 ** LD_res, lcell_rad * 2 ** LD_res)
    lib_dir = os.path.join(_get_lensitdir()[0], 'temp', 'ellmats', 'ellmat_%s_%s' % (HD_res, LD_res))
    return ell_mat.ell_mat(lib_dir, shape, lsides)


def get_lencmbs_lib(res=14, cache_sims=True, nsims=120, num_threads=int(os.environ.get('OMP_NUM_THREADS', 1)), alpha_cpp=1.):
    r"""Default lensed CMB simulation library

    Lensing is always performed at resolution of :math:`0.75` arcmin

    Args:
        res: lensed CMBs are generated on a square box with of physical size  :math:`\sim 0.74 \cdot 2^{\rm res}` arcmin
        cache_sims: saves the lensed CMBs when produced for the first time
        nsims: number of simulations in the library
        num_threads: number of threads used by the pyFFTW fft-engine.
        alpha_cpp: Multiplicative factor for input Cl_pp, in order to test response 
                of the estimator to a linear change of the input Cl_pp
    Note:
        All simulations random phases will be generated at the very first call if not performed previously; this might take some time

    """
    HD_ellmat = get_ellmat(res, HD_res=res)
    # Degrade the lensing resolution by a factor of 2
    #HD_ellmat = get_ellmat(res-1, HD_res=res)
    ellmax_sky = 6000
    fsky = int(np.round(np.prod(HD_ellmat.lsides) / 4. / np.pi * 1000.))
    lib_skyalm = ell_mat.ffs_alm_pyFFTW(HD_ellmat, num_threads=num_threads,
                                                 filt_func=lambda ell: ell <= ellmax_sky)
    skypha_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'len_alms', 'skypha')
    skypha = ffs_phas.ffs_lib_phas(skypha_libdir, 4, lib_skyalm, nsims_max=nsims)
    if not skypha.is_full() and pbs.rank == 0:
        for i, idx in enumerate_progress(np.arange(nsims, dtype=int), label='Generating CMB phases'):
            skypha.get_sim(int(idx))
    pbs.barrier()
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky, alpha_cpp=alpha_cpp)
    print("    [lensit.init.get_lencmbs_lib:] Input Cls_unl['pp'][1] = {}".format(cls_unl['pp'][1]))

    # sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'len_alms')
    sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d_alphacpp_%s' % (fsky, alpha_cpp) , 'len_alms')
    return ffs_cmbs.sims_cmb_len(sims_libdir, lib_skyalm, cls_unl, lib_pha=skypha, cache_lens=cache_sims, alpha_cpp=alpha_cpp)


def get_maps_lib(exp, LDres, HDres=14, cache_lenalms=True, cache_maps=False,
                 nsims=120, num_threads=int(os.environ.get('OMP_NUM_THREADS', 1)), alpha_cpp=1.):
    r"""Default CMB data maps simulation library

    Args:
        exp: experimental configuration (see *get_config*)
        LDres: the data is generated on a square patch with :math:` 2^{\rm LDres}` pixels on a side
        HDres: The physical size of the path is :math:`\sim 0.74 \cdot 2^{\rm HDres}` arcmin
        cache_lenalms: saves the lensed CMBs when produced for the first time (defaults to True)
        cache_maps: saves the data maps when produced for the first time (defaults to False)
        nsims: number of simulations in the library
        num_threads: number of threads used by the pyFFTW fft-engine.
        alpha_cpp: Multiplicative factor for input Cl_pp, in order to test response 
                of the estimator to a linear change of the input Cl_pp
    Note:
        All simulations random phases (CMB sky and noise) will be generated at the very first call if not performed previously; this might take some time

    """
    sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    print('    [lensit.init.get_maps_lib:] beam FWHM {} amin, noise T {} muK.amin'.format(Beam_FWHM_amin, sN_uKamin))
    # print('get_maps_lib alpha_cpp = {}'.format(alpha_cpp))
    len_cmbs = get_lencmbs_lib(res=HDres, cache_sims=cache_lenalms, nsims=nsims, alpha_cpp=alpha_cpp)
    lmax_sky = len_cmbs.lib_skyalm.ellmax
    cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=lmax_sky)
    lib_datalm = ell_mat.ffs_alm_pyFFTW(get_ellmat(LDres, HDres), filt_func=lambda ell: ell <= lmax_sky,
                                                 num_threads=num_threads)
    fsky = int(np.round(np.prod(len_cmbs.lib_skyalm.ell_mat.lsides) / 4. / np.pi * 1000.))
    vcell_amin2 = np.prod(lib_datalm.ell_mat.lsides) / np.prod(lib_datalm.ell_mat.shape) * (180 * 60. / np.pi) ** 2
    nTpix = sN_uKamin / np.sqrt(vcell_amin2)
    nPpix = sN_uKaminP / np.sqrt(vcell_amin2)
    # pixpha_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'LD%sHD%s' % (LDres, HDres), 'pixpha')
    pixpha_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'res%s' % LDres, 'pixpha')
    pixpha = ffs_phas.pix_lib_phas(pixpha_libdir, 3, lib_datalm.ell_mat.shape, nsims_max=nsims)

    if not pixpha.is_full() and pbs.rank == 0:
        for _i, idx in enumerate_progress(np.arange(nsims), label='Generating Noise phases'):
            pixpha.get_sim(idx)
    pbs.barrier()
    # sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims'%nsims,'fsky%04d'%fsky, 'res%s'%LDres,'%s'%exp, 'maps')
    sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d_alphacpp_%s' % (fsky, alpha_cpp) , 'res%s'%LDres,'%s'%exp, 'maps')
    #sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d_alphacpp_%s' % (fsky, alpha_cpp) , 'LD%sHD%s' % (LDres, HDres), 'maps')
    print('    [lensit.init.get_maps_lib:] sims_libdir: ' + sims_libdir)
    return ffs_maps.lib_noisemap(sims_libdir, lib_datalm, len_cmbs, cl_transf, nTpix, nPpix, nPpix,
                                      pix_pha=pixpha, cache_sims=cache_maps, nsims=nsims)


def get_lencmbs_lib_fixed_phi(res=14, cache_sims=True, nsims=120, num_threads=int(os.environ.get('OMP_NUM_THREADS', 1)), alpha_cpp=1., ellmax_sky = 6000, phimap=None, pbsrank=pbs.rank, pbsbarrier=pbs.barrier):
    r"""Default lensed CMB simulation library

    # TODO: can probably change this to lens at lower resolution adn speed up the computations
    Lensing is always performed at resolution of :math:`0.75` arcmin

    Args:
        res: lensed CMBs are generated on a square box with of physical size  :math:`\sim 0.74 \cdot 2^{\rm res}` arcmin
        cache_sims: saves the lensed CMBs when produced for the first time
        nsims: number of simulations in the library
        num_threads: number of threads used by the pyFFTW fft-engine.
        alpha_cpp: Multiplicative factor for input Cl_pp, in order to test response 
                of the estimator to a linear change of the input Cl_pp
    Note:
        All simulations random phases will be generated at the very first call if not performed previously; this might take some time

    """
    HD_ellmat = get_ellmat(res, HD_res=res)
    fsky = int(np.round(np.prod(HD_ellmat.lsides) / 4. / np.pi * 1000.))
    lib_skyalm = ell_mat.ffs_alm_pyFFTW(HD_ellmat, num_threads=num_threads,
                                                 filt_func=lambda ell: ell <= ellmax_sky)
    #TODO check that the lib dir is ok when alpha_cpp =! 1, also in the other functions get_lencmbs_lib and get_maps_lib
    # skypha_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims_fixed_phi' % nsims, 'fsky%04d' % fsky, 'len_alms', 'skypha')
    skypha_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d_alphacpp_%s' % (fsky, alpha_cpp), 'input_plmmap_hash%s' % npy_hash(phimap), 'len_alms', 'skypha')
    # print(skypha_libdir)
    skypha = ffs_phas.ffs_lib_phas(skypha_libdir, 4, lib_skyalm, nsims_max=nsims, pbsrank=pbsrank, pbsbarrier=pbsbarrier)
    if not skypha.is_full() and pbsrank == 0:
        for i, idx in enumerate_progress(np.arange(nsims, dtype=int), label='Generating CMB phases'):
            skypha.get_sim(int(idx))
    pbsbarrier()
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky, alpha_cpp=alpha_cpp)
    # print("Input Cls_unl['pp'][1] = {}".format(cls_unl['pp'][1]))

    # sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'len_alms')
    sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims_fixed_phi' % nsims, 'fsky%04d_alphacpp_%s' % (fsky, alpha_cpp), 'input_plmmap_hash%s' % npy_hash(phimap), 'len_alms')
    return ffs_cmbs.sim_cmb_len_fixed_phi(sims_libdir, lib_skyalm, cls_unl, lib_pha=skypha, cache_lens=cache_sims, alpha_cpp=alpha_cpp, phimap=phimap, pbsrank=pbsrank, pbsbarrier=pbsbarrier)


def get_maps_lib_fixed_phi(exp, LDres=10, HDres=11, cache_lenalms=True, cache_maps=False,
                 nsims=120, num_threads=int(os.environ.get('OMP_NUM_THREADS', 1)), alpha_cpp=1., phimap=None, pbsrank=pbs.rank, pbsbarrier=pbs.barrier):
    r"""Default CMB data maps simulation library

    Args:
        exp: experimental configuration (see *get_config*)
        LDres: the data is generated on a square patch with :math:` 2^{\rm LDres}` pixels on a side
        HDres: The physical size of the path is :math:`\sim 0.74 \cdot 2^{\rm HDres}` arcmin
        cache_lenalms: saves the lensed CMBs when produced for the first time (defaults to True)
        cache_maps: saves the data maps when produced for the first time (defaults to False)
        nsims: number of simulations in the library
        num_threads: number of threads used by the pyFFTW fft-engine.
        alpha_cpp: Multiplicative factor for input Cl_pp, in order to test response 
                of the estimator to a linear change of the input Cl_pp
    Note:
        All simulations random phases (CMB sky and noise) will be generated at the very first call if not performed previously; this might take some time

    """

    sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    # print('    [li.__init__.get_maps_lib_fixed_phi:] alpha_cpp = {}'.format(alpha_cpp))
    len_cmbs = get_lencmbs_lib_fixed_phi(res=HDres, cache_sims=cache_lenalms, nsims=nsims, alpha_cpp=alpha_cpp, phimap=phimap, pbsrank=pbsrank, pbsbarrier=pbsbarrier)
    lmax_sky = len_cmbs.lib_skyalm.ellmax
    cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=lmax_sky)
    lib_datalm = ell_mat.ffs_alm_pyFFTW(get_ellmat(LDres, HDres), filt_func=lambda ell: ell <= lmax_sky,
                                                num_threads=num_threads)

    fsky = int(np.round(np.prod(len_cmbs.lib_skyalm.ell_mat.lsides) / 4. / np.pi * 1000.))
    vcell_amin2 = np.prod(lib_datalm.ell_mat.lsides) / np.prod(lib_datalm.ell_mat.shape) * (180 * 60. / np.pi) ** 2
    nTpix = sN_uKamin / np.sqrt(vcell_amin2)
    nPpix = sN_uKaminP / np.sqrt(vcell_amin2)

    # pixpha_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims_fixed_phi' % nsims, 'fsky%04d' % fsky, 'res%s' % LDres, 'pixpha')
    pixpha_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'res%s' % LDres, 'pixpha')
    pixpha = ffs_phas.pix_lib_phas(pixpha_libdir, 3, lib_datalm.ell_mat.shape, nsims_max=nsims)

    if not pixpha.is_full() and pbsrank == 0:
        for _i, idx in enumerate_progress(np.arange(nsims), label='Generating Noise phases'):
            pixpha.get_sim(idx)
    pbsbarrier()
    # sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims'%nsims,'fsky%04d'%fsky, 'res%s'%LDres,'%s'%exp, 'maps')
    sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims_fixed_phi' % nsims, 'fsky%04d_alphacpp_%s' % (fsky, alpha_cpp) , 'input_plmmap_hash%s' % npy_hash(phimap), 'res%s'%LDres,'%s'%exp, 'maps')
    # No need to define the libdir as a function of LDHD, becasue it's conatined in the resolution and the skyfraction, already in the path
    #sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d_alphacpp_%s' % (fsky, alpha_cpp) , 'input_plmmap_hash%s' % npy_hash(phimap), 'res%s'%LDres,'%s'%exp, 'maps')
    print('    [li.__init__.get_maps_lib_fixed_phi:] ' + sims_libdir)
    return ffs_maps.lib_noisemap(sims_libdir, lib_datalm, len_cmbs, cl_transf, nTpix, nPpix, nPpix,
                                      pix_pha=pixpha, cache_sims=cache_maps, nsims=nsims, pbsrank=pbsrank, pbsbarrier=pbsbarrier)



def get_isocov(exp, LD_res, HD_res=14, pyFFTWthreads=int(os.environ.get('OMP_NUM_THREADS', 1)), alpha_cpp=1., ellmax_sky=6000, new_cls=False):
    r"""Default *ffs_cov.ffs_diagcov_alm* instances.


    Returns:
        *ffs_cov.ffs_diagcov_alm* instance on a flat-sky square patch of physical size :math:`\sim 0.74 \cdot 2^{\rm HDres}` arcmin,
        sampled with :math:`2^{\rm LDres}` points on a side.


    """
    sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky, alpha_cpp=alpha_cpp, new_cls=new_cls)

    cls_noise = {'t': (sN_uKamin * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1),
                 'q':(sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1),
                 'u':(sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1)}  # simple flat noise Cls
    cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=ellmax_sky)
    lib_alm = ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                        filt_func=lambda ell: (ell >= ellmin) & (ell <= ellmax), num_threads=pyFFTWthreads)
    lib_skyalm = ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                        filt_func=lambda ell: (ell <= ellmax_sky), num_threads=pyFFTWthreads)

    # lib_dir = os.path.join(_get_lensitdir()[0], 'temp', 'Covs', '%s' % exp, 'alpha_cpp_%s' % (alpha_cpp), 'LD%sHD%sellmax_sky%s' % (LD_res, HD_res, ellmax_sky))
    lib_dir = os.path.join(_get_lensitdir()[0], 'temp', 'Covs', '%s' % exp, 'alpha_cpp_%s' % (alpha_cpp), 'LD%sHD%s' % (LD_res, HD_res))
    if new_cls:
        lib_dir = os.path.join(_get_lensitdir()[0], 'temp', 'Covs', '%s' % exp, 'alpha_cpp_%s_new_cls' % (alpha_cpp), 'LD%sHD%s' % (LD_res, HD_res))
    return ffs_cov.ffs_diagcov_alm(lib_dir, lib_alm, cls_unl, cls_len, cl_transf, cls_noise, lib_skyalm=lib_skyalm, alpha_cpp=alpha_cpp)



def get_config(exp):
    """Returns noise levels, beam size and multipole cuts for some configurations

    """
    sN_uKaminP = None
    if exp == 'Planck':
        sN_uKamin = 35.
        Beam_FWHM_amin = 7.
        ellmin = 10
        ellmax = 2048
    elif exp == 'Planck_euclid':
        sN_uKamin = 23.
        sN_uKaminP = 42
        Beam_FWHM_amin = 7.
        ellmin = 2
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
    elif exp == 'S4_opti':
        sN_uKamin = 1.
        Beam_FWHM_amin = 1.
        ellmin = 10
        ellmax = 3000
    elif exp == 'S4_opti_0.98':
        sN_uKamin = 0.98
        Beam_FWHM_amin = 1.
        ellmin = 10
        ellmax = 3000
    elif exp == 'S4_opti_1.02':
        sN_uKamin = 1.02
        Beam_FWHM_amin = 1.
        ellmin = 10
        ellmax = 3000
    elif exp == 'S4_opti_1.5':
        sN_uKamin = 1.5
        Beam_FWHM_amin = 1.
        ellmin = 10
        ellmax = 3000
    elif exp == 'S4_opti_2':
        sN_uKamin = 2.
        Beam_FWHM_amin = 1.
        ellmin = 10
        ellmax = 3000
    elif exp == 'S4_euclid':
        sN_uKamin = 1.
        Beam_FWHM_amin = 1.
        ellmin = 2
        ellmax = 3000
    elif exp == 'S4_biased':
        sN_uKamin = 4
        Beam_FWHM_amin = 3.
        ellmin = 2
        ellmax = 3000
    elif exp == 'S4_2':
        sN_uKamin = 2.
        Beam_FWHM_amin = 3.
        ellmin = 2
        ellmax = 3000
    elif exp == 'S4_1.52':
        sN_uKamin = 1.52
        Beam_FWHM_amin = 3.
        ellmin = 2
        ellmax = 3000
    elif exp == 'S4_1.49':
        sN_uKamin = 1.49
        Beam_FWHM_amin = 3.
        ellmin = 2
        ellmax = 3000
    elif exp == 'S4_opti_6000':
        sN_uKamin = 1.
        Beam_FWHM_amin = 1.
        ellmin = 10
        ellmax = 6000
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
    elif exp == 'SO_opti0':
        sN_uKamin = 5
        Beam_FWHM_amin = 1.4
        ellmin = 10
        ellmax = 3000
    elif exp == 'SO_opti':
        sN_uKamin = 11.
        Beam_FWHM_amin = 4.
        ellmin = 10
        ellmax = 3000
    elif exp == 'SO':
        sN_uKamin = 3.
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    elif exp == 'SO_euclid3':
        sN_uKamin = 10.
        Beam_FWHM_amin = 4.
        ellmin = 2
        ellmax = 3000
    elif exp == 'SO_euclid4':
        sN_uKamin = 12.
        Beam_FWHM_amin = 4.
        ellmin = 2
        ellmax = 3000
    elif exp == 'SO_euclid2':
        sN_uKamin = 8.
        Beam_FWHM_amin = 4.5
        ellmin = 2
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
    elif exp == 'fcy_mark':
        sN_uKamin = 5.
        Beam_FWHM_amin = 1.4
        ellmin=10
        ellmax=3000
    elif exp == '5muKamin_1amin':
        sN_uKamin = 5.
        Beam_FWHM_amin = 1.
        ellmin=10
        ellmax=3000
    elif exp == 'Planck45':
        sN_uKamin = 45.
        Beam_FWHM_amin = 5
        ellmin = 10
        ellmax = 2048
    elif exp == 'Planck45_lmax3000':
        sN_uKamin = 45.
        Beam_FWHM_amin = 5
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
