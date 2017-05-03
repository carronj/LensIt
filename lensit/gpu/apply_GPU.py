import time

import numpy as np
import pycuda.gpuarray as gpuarray
from skcuda.fft import fft, ifft

from lensit.misc.misc_utils import IsPowerOfTwo, Freq
from lensit.ffs_covs.ffs_specmat import get_Pmat
from . import CUDA_module, setup_texture_nparr, GPU_block, get_rfft_plans, setup_texture_gpuarr


def apply_FDxiDtFt_GPU_inplace(type, lib_alm_dat, lib_alm_sky, alms_unlCMB, f, f_inv, cls_unl, func='bicubic',
                               double_precision_ffts=False):
    """
    Note that the first call might be substantially slower than subsequent calls, as it caches the fft and ifft plans
    for subsequent usage.
    :param type : 'T', 'QU' or 'TQU'
    :param alms_unlCMB: ffs_alms to apply FDxiDtFt to.
    :param func: bicubic or bilinear
    :param cls_unl : unlensed CMB cls dictionary (used in get_P_mat)
    :return: ffs_alms of shape (len(type,lib_alm_dat.alm_size)
    """
    assert func in ['bicubic', 'bilinear'], func
    assert alms_unlCMB.shape == (len(type), lib_alm_dat.alm_size)
    assert lib_alm_dat.ell_mat.shape == lib_alm_sky.ell_mat.shape
    assert lib_alm_dat.ell_mat.lsides == lib_alm_sky.ell_mat.lsides
    # Useful declarations :
    nfields = len(type)
    rshape = lib_alm_sky.ell_mat.rshape
    shape = (rshape[0], 2 * (rshape[1] - 1))
    flat_shape = np.prod(shape)

    GPU_grid = (shape[0] / GPU_block[0], shape[1] / GPU_block[1], 1)

    assert shape[0] % GPU_block[0] == 0

    assert shape[0] == shape[1], shape
    assert IsPowerOfTwo(shape[0]), shape
    assert f.shape == shape, (f.shape, shape)
    assert f_inv.shape == shape, (f_inv.shape, shape)
    assert np.all(np.array(shape) % GPU_block[0] == 0), shape

    if shape[0] > 4096:
        print "--- Exercise caution, array shapes larger than 4096 have never been tested so far ---"

    def get_rfft_unlCMB(idx):
        return lib_alm_dat.alm2rfft(alms_unlCMB[idx])

    # TODO : some get_Pij method
    unlPmat = get_Pmat(type, lib_alm_sky, cls_unl)
    # 2D texture references :
    unl_CMB_tex = CUDA_module.get_texref("unl_CMB")
    dx_tex = CUDA_module.get_texref("tex_dx")
    dy_tex = CUDA_module.get_texref("tex_dy")

    # loading fft plans :
    plan, plan_inv = get_rfft_plans(shape, double_precision=double_precision_ffts)
    # Function references :
    # Spline bicubic prefiltering, bicubic interpolation and multiplication with magnification.
    prefilter = CUDA_module.get_function("cf_outer_w") if not double_precision_ffts else CUDA_module.get_function(
        "cdd_outer_w")
    lens_func = CUDA_module.get_function("%slensKernel_normtex" % func)
    magn_func = CUDA_module.get_function("detmagn_normtex")

    cplx_type = np.complex64 if not double_precision_ffts else np.complex128
    f_type = np.float32 if not double_precision_ffts else np.float64

    # We will store in host memory some maps for convenience
    temp_alms = np.zeros((nfields, lib_alm_sky.alm_size), dtype=cplx_type)

    setup_texture_nparr(dx_tex, f_inv.get_dx_ingridunits())
    setup_texture_nparr(dy_tex, f_inv.get_dy_ingridunits())
    coeffs_gpu = gpuarray.empty(shape, dtype=f_type, order='C')
    # Setting up the texture references to the displacement
    # (This is what  contribute most to the cost actually)
    rfft2_unlCMB_gpu = gpuarray.empty(rshape, dtype=cplx_type)
    wx_gpu = (6. / (2. * np.cos(2. * np.pi * Freq(np.arange(shape[0]), shape[0]) / shape[0]) + 4.) / shape[0])
    wx_gpu = gpuarray.to_gpu(wx_gpu.astype(f_type))

    for _f in xrange(nfields):
        # Multiplying with the spline coefficients and Fourier transforming
        rfft2_unlCMB_gpu.set(get_rfft_unlCMB(_f).astype(cplx_type))
        prefilter(rfft2_unlCMB_gpu, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
        ifft(rfft2_unlCMB_gpu, coeffs_gpu, plan_inv, False)
        # coeffs_gpu now contains the prefiltered map to be now bicubic interpolated
        if f_type != np.float32:  coeffs_gpu = coeffs_gpu.astype(np.float32)
        setup_texture_gpuarr(unl_CMB_tex, coeffs_gpu)
        # Now bicubic interpolation with inverse displacement, and mult. with magnification.
        lens_func(coeffs_gpu, np.int32(shape[0]), block=GPU_block, grid=GPU_grid, texrefs=[unl_CMB_tex, dx_tex, dy_tex])
        magn_func(coeffs_gpu, np.int32(shape[0]), np.int32(flat_shape), block=GPU_block, grid=GPU_grid,
                  texrefs=[dx_tex, dy_tex])
        if f_type != np.float32: coeffs_gpu = coeffs_gpu.astype(f_type)

        fft(coeffs_gpu, rfft2_unlCMB_gpu, plan)

        # To be GPU memory friendly these maps are in the host memory :
        # TODO : should be possible to adapt the code to do everything on the GPU, by using 4 displacement textures.
        temp_alm = lib_alm_sky.rfftmap2alm(rfft2_unlCMB_gpu.get())
        for _g in xrange(nfields): temp_alms[_g] += temp_alm * unlPmat[:, _g, _f]  # CPU operations

    # We now lens and then fft each map, and return.
    # We lens now with the forward displacement :
    setup_texture_nparr(dx_tex, f.get_dx_ingridunits())
    setup_texture_nparr(dy_tex, f.get_dy_ingridunits())
    lenCMB_gpu = gpuarray.empty(shape, dtype=np.float32, order='C')
    for _g in xrange(nfields):

        rfft2_unlCMB_gpu.set(lib_alm_sky.alm2rfft(temp_alms[_g]).astype(cplx_type))
        prefilter(rfft2_unlCMB_gpu, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
        ifft(rfft2_unlCMB_gpu, coeffs_gpu, plan_inv, False)
        # Lensing :
        if f_type != np.float32:  coeffs_gpu = coeffs_gpu.astype(np.float32)
        setup_texture_gpuarr(unl_CMB_tex, coeffs_gpu)
        lens_func(lenCMB_gpu, np.int32(shape[0]), block=GPU_block, grid=GPU_grid, texrefs=[unl_CMB_tex, dx_tex, dy_tex])
        if f_type != np.float32: lenCMB_gpu = lenCMB_gpu.astype(f_type)

        # coeffs_gpu is now D xi D^t. Turn this to Fourier space :
        fft(lenCMB_gpu, rfft2_unlCMB_gpu, plan)
        alms_unlCMB[_g] = lib_alm_dat.rfftmap2alm(rfft2_unlCMB_gpu.get())  # Pulling result from GPU to CPUcd
    return


def apply_FDxiDtFt_GPU_inplace_timed(type, lib_alm_dat, lib_alm_sky, alms_unlCMB, f, f_inv, cls_unl, func='bicubic',
                                     double_precision_ffts=False):
    """
    Note that the first call might be substantially slower than subsequent calls, as it caches the fft and ifft plans
    for subsequent usage.
    :param type : 'T', 'QU' or 'TQU'
    :param alms_unlCMB: ffs_alms to apply FDxiDtFt to.
    :param func: bicubic or bilinear
    :param cls_unl : unlensed CMB cls dictionary (used in get_P_mat)
    :return: ffs_alms of shape (len(type,lib_alm_dat.alm_size)
    """
    if True:
        ti = time.time()
    assert func in ['bicubic', 'bilinear'], func
    assert alms_unlCMB.shape == (len(type), lib_alm_dat.alm_size)
    assert lib_alm_dat.ell_mat.shape == lib_alm_sky.ell_mat.shape
    assert lib_alm_dat.ell_mat.lsides == lib_alm_sky.ell_mat.lsides
    # Useful declarations :
    nfields = len(type)
    rshape = lib_alm_sky.ell_mat.rshape
    shape = (rshape[0], 2 * (rshape[1] - 1))
    flat_shape = np.prod(shape)

    GPU_grid = (shape[0] / GPU_block[0], shape[1] / GPU_block[1], 1)

    assert shape[0] % GPU_block[0] == 0

    assert shape[0] == shape[1], shape
    assert IsPowerOfTwo(shape[0]), shape
    assert f.shape == shape, (f.shape, shape)
    assert f_inv.shape == shape, (f_inv.shape, shape)
    assert np.all(np.array(shape) % GPU_block[0] == 0), shape

    if shape[0] > 4096:
        print "--- Exercise caution, array shapes larger than 4096 have never been tested so far ---"

    def get_rfft_unlCMB(idx):
        return lib_alm_dat.alm2rfft(alms_unlCMB[idx])

    # TODO : some get_Pij method
    if True:
        t0 = time.time()
    unlPmat = get_Pmat(type, lib_alm_sky, cls_unl)
    if True:
        dt = time.time() - t0
        print "     unl Pmat at %s Mpixel / sec, ex. time %s sec." % (np.prod(shape) / 1e6 / dt, dt)
        t0 = time.time()

    # 2D texture references :
    unl_CMB_tex = CUDA_module.get_texref("unl_CMB")
    dx_tex = CUDA_module.get_texref("tex_dx")
    dy_tex = CUDA_module.get_texref("tex_dy")

    # loading fft plans :
    plan, plan_inv = get_rfft_plans(shape, double_precision=double_precision_ffts)
    # Function references :
    # Spline bicubic prefiltering, bicubic interpolation and multiplication with magnification.
    prefilter = CUDA_module.get_function("cf_outer_w") if not double_precision_ffts else CUDA_module.get_function(
        "cdd_outer_w")
    lens_func = CUDA_module.get_function("%slensKernel_normtex" % func)
    magn_func = CUDA_module.get_function("detmagn_normtex")

    cplx_type = np.complex64 if not double_precision_ffts else np.complex128
    f_type = np.float32 if not double_precision_ffts else np.float64

    # We will store in host memory some maps for convenience
    temp_alms = np.zeros((nfields, lib_alm_sky.alm_size), dtype=cplx_type)

    setup_texture_nparr(dx_tex, f_inv.get_dx_ingridunits())
    setup_texture_nparr(dy_tex, f_inv.get_dy_ingridunits())
    coeffs_gpu = gpuarray.empty(shape, dtype=f_type, order='C')
    # Setting up the texture references to the displacement
    # (This is what  contribute most to the cost actually)
    rfft2_unlCMB_gpu = gpuarray.empty(rshape, dtype=cplx_type)
    if True:
        dt = time.time() - t0
        print "  First tex. setup at %s Mpixel / sec, ex. time %s sec." % (np.prod(shape) / 1e6 / dt, dt)
        t0 = time.time()
    wx_gpu = (6. / (2. * np.cos(2. * np.pi * Freq(np.arange(shape[0]), shape[0]) / shape[0]) + 4.) / shape[0])
    wx_gpu = gpuarray.to_gpu(wx_gpu.astype(f_type))

    for _f in xrange(nfields):
        # Multiplying with the spline coefficients and Fourier transforming
        rfft2_unlCMB_gpu.set(get_rfft_unlCMB(_f).astype(cplx_type))
        prefilter(rfft2_unlCMB_gpu, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
        ifft(rfft2_unlCMB_gpu, coeffs_gpu, plan_inv, False)
        # coeffs_gpu now contains the prefiltered map to be now bicubic interpolated
        if f_type != np.float32:  coeffs_gpu = coeffs_gpu.astype(np.float32)
        setup_texture_gpuarr(unl_CMB_tex, coeffs_gpu)
        if True:
            dt = time.time() - t0
            print "     CMB field %s texture setup at %s Mpixel / sec, ex. time %s sec." % (
                _f, np.prod(shape) / 1e6 / dt, dt)
            t0 = time.time()

        # Now bicubic interpolation with inverse displacement, and mult. with magnification.
        lens_func(coeffs_gpu, np.int32(shape[0]), block=GPU_block, grid=GPU_grid, texrefs=[unl_CMB_tex, dx_tex, dy_tex])
        magn_func(coeffs_gpu, np.int32(shape[0]), np.int32(flat_shape), block=GPU_block, grid=GPU_grid,
                  texrefs=[dx_tex, dy_tex])
        if True:
            dt = time.time() - t0
            print "     CMB field %s lensed and magnified at %s Mpixel / sec, ex. time %s sec." % (
                _f, np.prod(shape) / 1e6 / dt, dt)
            t0 = time.time()

        if f_type != np.float32: coeffs_gpu = coeffs_gpu.astype(f_type)

        fft(coeffs_gpu, rfft2_unlCMB_gpu, plan)

        # To be GPU memory friendly these maps are in the host memory :
        # TODO : should be possible to adapt the code to do everything on the GPU, by using 4 displacement textures.
        temp_alm = lib_alm_sky.rfftmap2alm(rfft2_unlCMB_gpu.get())
        for _g in xrange(nfields): temp_alms[_g] += temp_alm * unlPmat[:, _g, _f]

        if True:
            dt = time.time() - t0
            print "     CMB field %s built temp_alms at %s Mpixel / sec, ex. time %s sec." % (
                _f, np.prod(shape) / 1e6 / dt, dt)
            t0 = time.time()

    # We now lens and then fft each map, and return.
    # We lens now with the forward displacement :
    setup_texture_nparr(dx_tex, f.get_dx_ingridunits())
    setup_texture_nparr(dy_tex, f.get_dy_ingridunits())
    if True:
        dt = time.time() - t0
        print "     Setup of forw. displ. textures at %s Mpixel / sec, ex. time %s sec." % (
            np.prod(shape) / 1e6 / dt, dt)
        t0 = time.time()

    lenCMB_gpu = gpuarray.empty(shape, dtype=np.float32, order='C')
    for _g in xrange(nfields):

        rfft2_unlCMB_gpu.set(lib_alm_sky.alm2rfft(temp_alms[_g]).astype(cplx_type))
        if True:
            dt = time.time() - t0
            print "     Pushing temp alm field %s at %s Mpixel / sec, ex. time %s sec." % (
                _g, np.prod(shape) / 1e6 / dt, dt)
            t0 = time.time()

        prefilter(rfft2_unlCMB_gpu, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
        ifft(rfft2_unlCMB_gpu, coeffs_gpu, plan_inv, False)
        # Lensing :
        if f_type != np.float32:  coeffs_gpu = coeffs_gpu.astype(np.float32)
        setup_texture_gpuarr(unl_CMB_tex, coeffs_gpu)
        lens_func(lenCMB_gpu, np.int32(shape[0]), block=GPU_block, grid=GPU_grid, texrefs=[unl_CMB_tex, dx_tex, dy_tex])
        if f_type != np.float32: lenCMB_gpu = lenCMB_gpu.astype(f_type)

        # coeffs_gpu is now D xi D^t. Turn this to Fourier space :
        fft(lenCMB_gpu, rfft2_unlCMB_gpu, plan)
        if True:
            dt = time.time() - t0
            print "     Lensing + rfft of field %s at %s Mpixel / sec, ex. time %s sec." % (
                _g, np.prod(shape) / 1e6 / dt, dt)
            t0 = time.time()

        alms_unlCMB[_g] = lib_alm_dat.rfftmap2alm(rfft2_unlCMB_gpu.get())  # Pulling result from GPU to CPUcd
        if True:
            dt = time.time() - t0
            print "     Pulling back field %s at %s Mpixel / sec, ex. time %s sec." % (
                _g, np.prod(shape) / 1e6 / dt, dt)
            t0 = time.time()

    if True:
        dt = time.time() - ti
        print "GPU TQU did G D xi D^t G^t at %s Mpixel / sec, ex. time %s sec." % (np.prod(shape) / 1e6 / dt, dt)

    return alms_unlCMB


"""
1024 x 1024 :

  Apply Cov,  :
           0.544 Mpixels / sec, total time 1.92757916451 sec
  Apply Cov lens GPU,  :
           1.6151 Mpixels / sec, total time 0.649219989777 sec
     unl Pmat at 48.818906982 Mpixel / sec, ex. time 0.0214788913727 sec.
  First tex. setup at 46.4056229673 Mpixel / sec, ex. time 0.0225958824158 sec.
     CMB field 0 texture setup at 106.987606089 Mpixel / sec, ex. time 0.00980091094971 sec.
     CMB field 0 lensed and magnified at 11277.0423362 Mpixel / sec, ex. time 9.29832458496e-05 sec.
     CMB field 0 built temp_alms at 28.7013183092 Mpixel / sec, ex. time 0.0365340709686 sec.
     CMB field 1 texture setup at 55.1735163788 Mpixel / sec, ex. time 0.0190050601959 sec.
     CMB field 1 lensed and magnified at 10496.5310528 Mpixel / sec, ex. time 9.98973846436e-05 sec.
     CMB field 1 built temp_alms at 39.1713931714 Mpixel / sec, ex. time 0.0267689228058 sec.
     Setup of forw. displ. textures at 50.5046566581 Mpixel / sec, ex. time 0.0207619667053 sec.
     Pushing temp alm field 0 at 166.997513332 Mpixel / sec, ex. time 0.00627899169922 sec.
     Lensing + rfft of field 0 at 94.8508995666 Mpixel / sec, ex. time 0.0110549926758 sec.
     Pulling back field 0 at 81.1777199436 Mpixel / sec, ex. time 0.0129170417786 sec.
     Pushing temp alm field 1 at 165.651469345 Mpixel / sec, ex. time 0.00633001327515 sec.
     Lensing + rfft of field 1 at 127.239881704 Mpixel / sec, ex. time 0.00824093818665 sec.
     Pulling back field 1 at 55.3603357221 Mpixel / sec, ex. time 0.0189409255981 sec.
GPU TQU did G D xi D^t G^t at 4.72926025292 Mpixel / sec, ex. time 0.221720933914 sec.
  Apply Cov full GPU, :
           4.4012 Mpixels / sec, total time 0.23824596405 sec

2048 x 2048
  Apply Cov,  :
           0.5079 Mpixels / sec, total time 8.25870609283 sec
  Apply Cov lens GPU,  :
           1.4734 Mpixels / sec, total time 2.8467040062 sec
     unl Pmat at 54.6930575633 Mpixel / sec, ex. time 0.0766880512238 sec.
  First tex. setup at 56.9831696858 Mpixel / sec, ex. time 0.0736060142517 sec.
     CMB field 0 texture setup at 140.672216447 Mpixel / sec, ex. time 0.0298161506653 sec.
     CMB field 0 lensed and magnified at 42803.3723708 Mpixel / sec, ex. time 9.79900360107e-05 sec.
     CMB field 0 built temp_alms at 53.9821412948 Mpixel / sec, ex. time 0.0776979923248 sec.
     CMB field 1 texture setup at 90.8940822565 Mpixel / sec, ex. time 0.0461449623108 sec.
     CMB field 1 lensed and magnified at 29766.8122579 Mpixel / sec, ex. time 0.000140905380249 sec.
     CMB field 1 built temp_alms at 72.9926852262 Mpixel / sec, ex. time 0.057461977005 sec.
     Setup of forw. displ. textures at 57.5570134417 Mpixel / sec, ex. time 0.0728721618652 sec.
     Pushing temp alm field 0 at 139.211727818 Mpixel / sec, ex. time 0.0301289558411 sec.
     Lensing + rfft of field 0 at 278.820604555 Mpixel / sec, ex. time 0.0150430202484 sec.
     Pulling back field 0 at 102.75149403 Mpixel / sec, ex. time 0.0408198833466 sec.
     Pushing temp alm field 1 at 150.977378044 Mpixel / sec, ex. time 0.0277810096741 sec.
     Lensing + rfft of field 1 at 259.437331983 Mpixel / sec, ex. time 0.0161669254303 sec.
     Pulling back field 1 at 126.900281645 Mpixel / sec, ex. time 0.0330519676208 sec.
GPU TQU did G D xi D^t G^t at 7.00666727382 Mpixel / sec, ex. time 0.598616123199 sec.
  Apply Cov full GPU, :
           6.359 Mpixels / sec, total time 0.659584999084 sec
"""
