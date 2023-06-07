"""This lensit package contains some convenience functions in its __init__.py for quick startup.

"""
from __future__ import print_function

import numpy as np
import os

from lensit.ffs_covs import ffs_cov, ell_mat
from lensit.sims import ffs_phas, ffs_maps, ffs_cmbs
from lensit.pbs import pbs
from lensit.misc.misc_utils import enumerate_progress, camb_clfile, gauss_beam, cl_inverse, npy_hash

LMAX_SKY = 5120

def _get_lensitdir():
    assert 'LENSIT' in os.environ.keys(), 'Set LENSIT env. variable to somewhere safe to write'
    LENSITDIR = os.environ.get('LENSIT')
    CLSPATH = os.path.join(os.path.dirname(__file__), 'data', 'cls')
    return LENSITDIR, CLSPATH


def get_fidcls(ellmax_sky=LMAX_SKY, wrotationCls=False, cls_grad=False):
    r"""Returns *lensit* fiducial CMB spectra (Planck 2015 cosmology)

    Args:
        ellmax_sky: optionally reduces outputs spectra :math:`\ell_{\rm max}`
        cls_grad: optionaly returns the grad-lensed spectra

    Returns:
        unlensed and lensed CMB spectra (dicts)

    """
    cls_unl = {}
    cls_unlr = camb_clfile(os.path.join(_get_lensitdir()[1], 'fiducial_flatsky_lenspotentialCls.dat'))
    for key in cls_unlr.keys():
        cls_unl[key] = cls_unlr[key][0:ellmax_sky + 1]
        if key == 'pp': cls_unl[key] = cls_unlr[key][:]  # might need this one to higher lmax
    if wrotationCls:
        cls_unl['oo'] = np.loadtxt(os.path.join(_get_lensitdir()[1], 'fiducial_fieldrotationCls.dat'))
    cls_len = {}
    #print(new_cls)
    cls_lenr = camb_clfile(os.path.join(_get_lensitdir()[1], 'fiducial_flatsky_lensedCls.dat'))
    for key in cls_lenr.keys():
        cls_len[key] = cls_lenr[key][0:ellmax_sky + 1]
    
    if cls_grad:
        cls_grad_d = np.loadtxt(os.path.join(_get_lensitdir()[1], 'fiducial_gradlensed_cls.dat')).T
        cls_grad = {'tt':cls_grad_d[0][:ellmax_sky+1], 'ee': cls_grad_d[:ellmax_sky+1], 'bb':cls_grad_d[:ellmax_sky+1], 'te':cls_grad_d[:ellmax_sky+1]}
        return cls_unl, cls_len, cls_grad

    return cls_unl, cls_len


def get_fidtenscls(ellmax_sky=LMAX_SKY):
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


def get_lencmbs_lib(res=14, wrotation=False, cache_sims=True,
                    nsims=120, num_threads=int(os.environ.get('OMP_NUM_THREADS', 1)), ellmax_sky=LMAX_SKY):
    r"""Default lensed CMB simulation library

    Lensing is always performed at resolution of :math:`0.75` arcmin

    Args:
        res: lensed CMBs are generated on a square box with of physical size  :math:`\sim 0.74 \cdot 2^{\rm res}` arcmin
        wrotation: includes lensing curl mode in deflection fields
        cache_sims: saves the lensed CMBs when produced for the first time
        nsims: number of simulations in the library, if None will generate simulations when requested
        num_threads: number of threads used by the pyFFTW fft-engine.
    Note:
        All simulations random phases will be generated at the very first call if not performed previously; this might take some time

    """
    HD_ellmat = get_ellmat(res, HD_res=res)
    fsky = int(np.round(np.prod(HD_ellmat.lsides) / 4. / np.pi * 1000.))
    lib_skyalm = ell_mat.ffs_alm_pyFFTW(HD_ellmat, num_threads=num_threads,
                                                 filt_func=lambda ell: ell <= ellmax_sky)
    skypha_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'len_alms', 'skypha' + wrotation * '_wcurl')
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky, wrotationCls=wrotation)
    nfield = len(ffs_cmbs.get_fields(cls_unl))
    if nsims is not None:
        skypha = ffs_phas.ffs_lib_phas(skypha_libdir, nfield, lib_skyalm, nsims_max=nsims)
        if not skypha.is_full() and pbs.rank == 0:
            for i, idx in enumerate_progress(np.arange(nsims, dtype=int), label='Generating CMB phases'):
                print(i)
                skypha.get_sim(int(idx))
    else:
        skypha = None

    pbs.barrier()

    sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'len_alms' + wrotation * '_wcurl')
    return ffs_cmbs.sims_cmb_len(sims_libdir, lib_skyalm, cls_unl, lib_pha=skypha, cache_lens=cache_sims)



def get_maps_lib(exp, LDres,  HDres=14, wrotation=False, cache_lenalms=True, cache_maps=False,
                 nsims=120, num_threads=int(os.environ.get('OMP_NUM_THREADS', 1)), ellmax_sky=LMAX_SKY):
    r"""Default CMB data maps simulation library

    Args:
        exp: experimental configuration (see *get_config*)
        LDres: the data is generated on a square patch with :math:` 2^{\rm LDres}` pixels on a side
        HDres: The physical size of the path is :math:`\sim 0.74 \cdot 2^{\rm HDres}` arcmin
        wrotation: includes lensing curl mode in deflection fields
        cache_lenalms: saves the lensed CMBs when produced for the first time (defaults to True)
        cache_maps: saves the data maps when produced for the first time (defaults to False)
        nsims: number of simulations in the library, if None will generate simulations when requested
        num_threads: number of threads used by the pyFFTW fft-engine.
    Note:
        All simulations random phases (CMB sky and noise) will be generated at the very first call if not performed previously; this might take some time

    """
    sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    print('    [lensit.init.get_maps_lib:] beam FWHM {} amin, noise T {} muK.amin'.format(Beam_FWHM_amin, sN_uKamin))
    len_cmbs = get_lencmbs_lib(res=HDres, cache_sims=cache_lenalms, nsims=nsims, wrotation=wrotation, ellmax_sky=ellmax_sky)
    lmax_sky = len_cmbs.lib_skyalm.ellmax
    cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=lmax_sky)
    lib_datalm = ell_mat.ffs_alm_pyFFTW(get_ellmat(LDres, HDres), filt_func=lambda ell: ell <= lmax_sky,
                                                 num_threads=num_threads)
    fsky = int(np.round(np.prod(len_cmbs.lib_skyalm.ell_mat.lsides) / 4. / np.pi * 1000.))
    vcell_amin2 = np.prod(lib_datalm.ell_mat.lsides) / np.prod(lib_datalm.ell_mat.shape) * (180 * 60. / np.pi) ** 2
    nTpix = sN_uKamin / np.sqrt(vcell_amin2)
    nPpix = sN_uKaminP / np.sqrt(vcell_amin2)
    # pixpha_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'LD%sHD%s' % (LDres, HDres), 'pixpha')
    if nsims is not None:
        pixpha_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'res%s' % LDres, 'pixpha')
        pixpha = ffs_phas.pix_lib_phas(pixpha_libdir, 3, lib_datalm.ell_mat.shape, nsims_max=nsims)
        if not pixpha.is_full() and pbs.rank == 0:
            for _i, idx in enumerate_progress(np.arange(nsims), label='Generating Noise phases'):
                pixpha.get_sim(idx)
        sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims'%nsims,'fsky%04d'%fsky, 'res%s'%LDres,'%s'%exp, 'maps' + wrotation * '_wcurl')
    else:
        pixpha = None
        sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', 'all_sims','fsky%04d'%fsky, 'res%s'%LDres,'%s'%exp, 'maps' + wrotation * '_wcurl')
    pbs.barrier()

    print('    [lensit.init.get_maps_lib:] sims_libdir: ' + sims_libdir)
    return ffs_maps.lib_noisemap(sims_libdir, lib_datalm, len_cmbs, cl_transf, nTpix, nPpix, nPpix,
                                      pix_pha=pixpha, cache_sims=cache_maps, nsims=nsims)


def get_lencmbs_lib_fixed_phi(res=14, cache_sims=True, nsims=120, num_threads=int(os.environ.get('OMP_NUM_THREADS', 1)), 
                              ellmax_sky = 6000, phimap=None, pbsrank=pbs.rank, pbsbarrier=pbs.barrier):
    r"""Default lensed CMB simulation library

    # TODO: can probably change this to lens at lower resolution adn speed up the computations
    Lensing is always performed at resolution of :math:`0.75` arcmin

    Args:
        res: lensed CMBs are generated on a square box with of physical size  :math:`\sim 0.74 \cdot 2^{\rm res}` arcmin
        cache_sims: saves the lensed CMBs when produced for the first time
        nsims: number of simulations in the library
        num_threads: number of threads used by the pyFFTW fft-engine.

    Note:
        All simulations random phases will be generated at the very first call if not performed previously; this might take some time

    """
    HD_ellmat = get_ellmat(res, HD_res=res)
    fsky = int(np.round(np.prod(HD_ellmat.lsides) / 4. / np.pi * 1000.))
    lib_skyalm = ell_mat.ffs_alm_pyFFTW(HD_ellmat, num_threads=num_threads,
                                                 filt_func=lambda ell: ell <= ellmax_sky)
    skypha_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'input_plmmap_hash%s' % npy_hash(phimap), 'len_alms', 'skypha')
    skypha = ffs_phas.ffs_lib_phas(skypha_libdir, 4, lib_skyalm, nsims_max=nsims, pbsrank=pbsrank, pbsbarrier=pbsbarrier)
    if not skypha.is_full(): #and pbsrank == 0:
        for i, idx in enumerate_progress(np.arange(nsims, dtype=int), label='Generating CMB phases'):
            skypha.get_sim(int(idx))
    pbsbarrier()
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky)

    sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims_fixed_phi' % nsims, 'fsky%04d' % fsky, 'input_plmmap_hash%s' % npy_hash(phimap), 'len_alms')
    return ffs_cmbs.sim_cmb_len_fixed_phi(sims_libdir, lib_skyalm, cls_unl, lib_pha=skypha, cache_lens=cache_sims, phimap=phimap, pbsrank=pbsrank, pbsbarrier=pbsbarrier)


def get_maps_lib_fixed_phi(exp, LDres=10, HDres=11, cache_lenalms=True, cache_maps=False,
                 nsims=120, num_threads=int(os.environ.get('OMP_NUM_THREADS', 1)), phimap=None, pbsrank=pbs.rank, pbsbarrier=pbs.barrier):
    r"""Default CMB data maps simulation library

    Args:
        exp: experimental configuration (see *get_config*)
        LDres: the data is generated on a square patch with :math:` 2^{\rm LDres}` pixels on a side
        HDres: The physical size of the path is :math:`\sim 0.74 \cdot 2^{\rm HDres}` arcmin
        cache_lenalms: saves the lensed CMBs when produced for the first time (defaults to True)
        cache_maps: saves the data maps when produced for the first time (defaults to False)
        nsims: number of simulations in the library
        num_threads: number of threads used by the pyFFTW fft-engine.
    Note:
        All simulations random phases (CMB sky and noise) will be generated at the very first call if not performed previously; this might take some time

    """

    sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    len_cmbs = get_lencmbs_lib_fixed_phi(res=HDres, cache_sims=cache_lenalms, nsims=nsims, phimap=phimap, pbsrank=pbsrank, pbsbarrier=pbsbarrier)
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

    if not pixpha.is_full():# and pbsrank == 0:
        for _i, idx in enumerate_progress(np.arange(nsims), label='Generating Noise phases'):
            pixpha.get_sim(idx)
    pbsbarrier()
    sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims_fixed_phi' % nsims, 'fsky%04d' % fsky , 'input_plmmap_hash%s' % npy_hash(phimap), 'res%s'%LDres,'%s'%exp, 'maps')
    print('    [li.__init__.get_maps_lib_fixed_phi:] ' + sims_libdir)
    return ffs_maps.lib_noisemap(sims_libdir, lib_datalm, len_cmbs, cl_transf, nTpix, nPpix, nPpix,
                                      pix_pha=pixpha, cache_sims=cache_maps, nsims=nsims, pbsrank=pbsrank, pbsbarrier=pbsbarrier)



def get_isocov(exp, LD_res, HD_res=14, pyFFTWthreads=int(os.environ.get('OMP_NUM_THREADS', 1)),  ellmax_sky=LMAX_SKY, use_cls_grad = False):
    r"""Default *ffs_cov.ffs_diagcov_alm* instances.


    Returns:
        *ffs_cov.ffs_diagcov_alm* instance on a flat-sky square patch of physical size :math:`\sim 0.74 \cdot 2^{\rm HDres}` arcmin,
        sampled with :math:`2^{\rm LDres}` points on a side.


    """
    # ellmax_sky = LMAX_SKY
    sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    if use_cls_grad:
        cls_unl, cls_len, cls_grad = get_fidcls(ellmax_sky=ellmax_sky, cls_grad=use_cls_grad)
    else:
        cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky, cls_grad=use_cls_grad)
        cls_grad = None
    cls_noise = {'t': (sN_uKamin * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1),
                 'q':(sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1),
                 'u':(sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1)}  # simple flat noise Cls
    cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=ellmax_sky)
    lib_alm = ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                        filt_func=lambda ell: (ell >= ellmin) & (ell <= ellmax), num_threads=pyFFTWthreads)
    lib_skyalm = ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                        filt_func=lambda ell: (ell <= ellmax_sky), num_threads=pyFFTWthreads)

    lib_dir = os.path.join(_get_lensitdir()[0], 'temp', 'Covs', '%s' % exp, 'LD%sHD%s' % (LD_res, HD_res), 'Lmaxsky%s' % ellmax_sky)
    if use_cls_grad: 
        lib_dir = os.path.join(lib_dir, "Cls_grad_in_QE")
        print(lib_dir)
    return ffs_cov.ffs_diagcov_alm(lib_dir, lib_alm, cls_unl, cls_len, cl_transf, cls_noise, lib_skyalm=lib_skyalm, cls_grad=cls_grad)



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
    elif exp == 'Peloton17':
        sN_uKamin = 1.5
        Beam_FWHM_amin = 3.
        ellmin = 20
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
    elif exp == 'S4_SPDP':
        sN_uKamin = 0.5 / np.sqrt(2.)
        Beam_FWHM_amin = 1.
        ellmin = 10
        ellmax = 4096
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
