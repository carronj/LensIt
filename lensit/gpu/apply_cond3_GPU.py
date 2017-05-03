# Timing information
timed = False

import time

import numpy as np
import pycuda.gpuarray as gpuarray
from skcuda.fft import fft, ifft

from lensit.misc.misc_utils import IsPowerOfTwo, Freq
from lensit.ffs_covs.ffs_specmat import get_Pmat
from . import CUDA_module, setup_texture_nparr, get_rfft_plans, GPU_block, setup_texture_gpuarr


def apply_cond3_GPU_inplace(type, lib_alm_dat, alms_unlCMB, f, f_inv, cls_unl, cl_transf, cls_noise,
                            func='bicubic', double_precision_ffts=False):
    """
    cond3 is F D^-t (B xi B^t + N)D^-1 F^t
    Note that the first call might be substantially slower than subsequent calls, as it caches the fft and ifft plans
    for subsequent usage, if not already in the fft plans (See __init__.py)
    :param type : 'T', 'QU' or 'TQU'
    :param alms_unlCMB: ffs_alms to apply FDxiDtFt to.
    :param func: bicubic or bilinear
    :param cls_unl : unlensed CMB cls dictionary (used in get_P_mat)
    :return: ffs_alms of shape (len(type,lib_alm_dat.alm_size)
    """
    if timed:
        ti = time.time()

    assert func in ['bicubic', 'bilinear'], func
    assert alms_unlCMB.shape == (len(type), lib_alm_dat.alm_size)

    # Useful declarations :
    nfields = len(type)
    rshape = lib_alm_dat.ell_mat.rshape
    shape = (rshape[0], 2 * (rshape[1] - 1))
    flat_shape = np.prod(shape)

    GPU_grid = (shape[0] / GPU_block[0], shape[1] / GPU_block[1], 1)

    assert shape[0] % GPU_block[0] == 0, shape

    assert shape[0] == shape[1], shape
    assert IsPowerOfTwo(shape[0]), shape
    assert f.shape == shape, (f.shape, shape)
    assert f_inv.shape == shape, (f_inv.shape, shape)
    assert f.lsides == lib_alm_dat.ell_mat.lsides, (f.lsides, lib_alm_dat.ell_mat.lsides)
    assert f_inv.lsides == lib_alm_dat.ell_mat.lsides, (f_inv.lsides, lib_alm_dat.ell_mat.lsides)

    assert np.all(np.array(shape) % GPU_block[0] == 0), shape

    if shape[0] > 4096:
        print "--- Exercise caution, array shapes larger than 4096 have never been tested so far ---"

    def get_rfft_unlCMB(idx):
        return lib_alm_dat.alm2rfft(alms_unlCMB[idx])

    unlPmat = get_Pmat(type, lib_alm_dat, cls_unl, cl_transf=cl_transf, cls_noise=cls_noise, inverse=True)

    # 2D texture references :
    unl_CMB_tex = CUDA_module.get_texref("unl_CMB")
    dx_tex = CUDA_module.get_texref("tex_dx")
    dy_tex = CUDA_module.get_texref("tex_dy")

    # loading fft plans :
    plan, plan_inv = get_rfft_plans(shape, double_precision=double_precision_ffts)
    # Function references :
    prefilter = CUDA_module.get_function("cf_outer_w") if not double_precision_ffts else CUDA_module.get_function(
        "cdd_outer_w")
    lens_func = CUDA_module.get_function("%slensKernel_normtex" % func)
    magn_func = CUDA_module.get_function("detmagn_normtex")

    cplx_type = np.complex64 if not double_precision_ffts else np.complex128
    f_type = np.float32 if not double_precision_ffts else np.float64

    # We will store in host memory some maps for convenience
    temp_alm = np.zeros((nfields, lib_alm_dat.alm_size), dtype=cplx_type)

    # Setting up the texture references to the displacement
    # (This is what  contribute most to the cost actually)
    setup_texture_nparr(dx_tex, f_inv.get_dx_ingridunits())
    setup_texture_nparr(dy_tex, f_inv.get_dy_ingridunits())
    # Building spline coefficients (1 / shape[0] comes from ifft convention)
    wx_gpu = (6. / (2. * np.cos(2. * np.pi * Freq(np.arange(shape[0]), shape[0]) / shape[0]) + 4.) / shape[0])
    wx_gpu = gpuarray.to_gpu(wx_gpu.astype(f_type))
    coeffs_gpu = gpuarray.empty(shape, dtype=f_type, order='C')
    for _f in xrange(nfields):
        # Multiplying with the spline coefficients and Fourier transforming
        rfft2_unlCMB_gpu = gpuarray.to_gpu(get_rfft_unlCMB(_f).astype(cplx_type))
        prefilter(rfft2_unlCMB_gpu, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
        ifft(rfft2_unlCMB_gpu, coeffs_gpu, plan_inv, False)
        # coeffs_gpu now contains the prefiltered map to be now bicubic interpolated

        # Now bicubic interpolation with inverse displacement.
        setup_texture_gpuarr(unl_CMB_tex, coeffs_gpu)
        lenCMB_gpu = gpuarray.empty(shape, dtype=np.float32, order='C')
        lens_func(lenCMB_gpu, np.int32(shape[0]), block=GPU_block, grid=GPU_grid, texrefs=[unl_CMB_tex, dx_tex, dy_tex])
        if f_type != np.float32: lenCMB_gpu = lenCMB_gpu.astype(f_type)

        # Back to Fourier space :
        rfft2_unlCMB_gpu = gpuarray.empty(rshape, dtype=cplx_type, order='C')
        fft(lenCMB_gpu, rfft2_unlCMB_gpu, plan)

        # We construct the map P_ij m_j which we will have to lens afterwards.
        # To be GPU memory friendly these maps are in the host memory :
        # for _g in xrange(nfields): ret[_g] += rfft2_unlCMB_gpu.get() * get_unlPmat(_g,_f)
        for _g in xrange(nfields): temp_alm[_g] += lib_alm_dat.rfftmap2alm(rfft2_unlCMB_gpu.get()) * unlPmat[:, _g, _f]

    # We now lens and then fft each map, and return.
    # We lens now with the forward displacement :
    setup_texture_nparr(dx_tex, f.get_dx_ingridunits())
    setup_texture_nparr(dy_tex, f.get_dy_ingridunits())
    for _g in xrange(nfields):
        rfft2_unlCMB_gpu = gpuarray.to_gpu(lib_alm_dat.alm2rfft(temp_alm[_g]).astype(cplx_type))
        prefilter(rfft2_unlCMB_gpu, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
        ifft(rfft2_unlCMB_gpu, coeffs_gpu, plan_inv, False)
        # Lensing by forward displacement, and multiplication by magnification :
        setup_texture_gpuarr(unl_CMB_tex, coeffs_gpu)
        lenCMB_gpu = gpuarray.empty(shape, dtype=np.float32, order='C')
        lens_func(lenCMB_gpu, np.int32(shape[0]), block=GPU_block, grid=GPU_grid, texrefs=[unl_CMB_tex, dx_tex, dy_tex])
        magn_func(lenCMB_gpu, np.int32(shape[0]), np.int32(flat_shape), block=GPU_block, grid=GPU_grid,
                  texrefs=[dx_tex, dy_tex])
        if f_type != np.float32: lenCMB_gpu = lenCMB_gpu.astype(f_type)
        # coeffs_gpu is now D xi D^t. Turn this to Fourier space :
        fft(lenCMB_gpu, rfft2_unlCMB_gpu, plan)
        alms_unlCMB[_g] = lib_alm_dat.rfftmap2alm(
            rfft2_unlCMB_gpu.get().astype(np.complex128))  # Pulling result from GPU to CPUcd
    if timed:
        dt = time.time() - ti
        print "GPU TQU did conditioner 3 at %s Mpixel / sec, ex. time %s sec." % (np.prod(shape) / 1e6 / dt, dt)
    return
