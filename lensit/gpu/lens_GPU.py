# Some parameters :

# Maximal memorywise input max is 2 ** GPU_HDres_max :
GPU_HDres_max = (11, 11)  # Bordeline on the 2048MB memory NVIDIA, but mostly OK.

# Gives some timing information :
timed = False

# If False, the displacements are also assigned to 2D textures.
texture_count = 3  # 0,1, or 3

import pycuda.autoinit
import pycuda.driver as cuda

from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import time
from skcuda.fft import fft, ifft
from . import load_map, get_rfft_plans, CUDA_module, GPU_block
from lensit.misc.misc_utils import Freq, IsPowerOfTwo
import numpy as np


def lens_onGPU(unl_CMB, dx_gu, dy_gu, do_not_prefilter=False):
    """
    Lens the input unl_CMB map on the GPU using the pyCUDA interface.
    dx dy displacement in grid units. (f.get_dx_ingridunits() e.g.)
    Can be path to arrays or arrays or memmap.
    Will probably crash for too large maps, with need to split the job.
    Works for 4096 x 4096 at least on my laptop.

    Cost dominated by texture setup. # FIXME : try get rid of texture
    Note that the first call might be substantially slower than subsequent calls, as it caches the fft and ifft plans
    for subsequent usage.
    :param unl_CMB:
    :param func: bicubic or bilinear
    :param normalized_tex: use a modified version of the GPU bicubic spline to account for periodicity of the map
    :return:
    """
    if timed:
        ti = time.time()
    shape = load_map(unl_CMB).shape
    rshape = (shape[0], shape[1] / 2 + 1)
    assert shape[0] == shape[1], shape
    assert IsPowerOfTwo(shape[0]), shape
    assert load_map(dx_gu).shape == shape, (load_map(dx_gu).shape, unl_CMB.shape)
    assert load_map(dy_gu).shape == shape, (load_map(dy_gu).shape, unl_CMB.shape)

    assert np.all(np.array(shape) % GPU_block[0] == 0), shape
    if shape[0] > 4096:
        print "--- Exercise caution, array shapes larger than 4096 have never been tested so far ---"

    GPU_grid = (shape[0] / GPU_block[0], shape[1] / GPU_block[1], 1)

    # Prefiltering forces the interpolant to pass through the samples and increase accuracy, but dominates the cost.
    coeffs_gpu = gpuarray.to_gpu(unl_CMB.astype(np.float32))  # ,allocator=lambda x : alloc)
    if not do_not_prefilter:
        # The prefilter makes sure the spline is exact at the nodes.
        plan, plan_inv = get_rfft_plans(shape)
        # Uncomments this to put coeffs_gpu on pitched memory to allow later for 2D texture binding :
        # alloc,pitch  = cuda.mem_alloc_pitch(shape[0] * 4,shape[1],4) # 4 bytes per float32
        rfft2_unlCMB_gpu = gpuarray.empty(rshape, np.complex64)
        fft(coeffs_gpu, rfft2_unlCMB_gpu, plan)

        wx = (6. / (2. * np.cos(2. * np.pi * Freq(np.arange(shape[0]), shape[0]) / shape[0]) + 4.) / shape[0])
        wx_gpu = gpuarray.to_gpu(wx.astype(np.float32))
        prefilter = CUDA_module.get_function("cf_outer_w")
        prefilter(rfft2_unlCMB_gpu, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
        ifft(rfft2_unlCMB_gpu, coeffs_gpu, plan_inv, False)
        del wx_gpu, rfft2_unlCMB_gpu

    # Binding arrays to textures and getting lensing func.
    if texture_count == 0:
        lens_func = CUDA_module.get_function("bicubiclensKernel_notex")
        tex_refs = []
        dx_gu = gpuarray.to_gpu(load_map(dx_gu).astype(np.float32))
        dy_gu = gpuarray.to_gpu(load_map(dy_gu).astype(np.float32))
    elif texture_count == 1:
        unl_CMB_tex = CUDA_module.get_texref("unl_CMB")
        tex_refs = [unl_CMB_tex]
        unl_CMB_tex.set_array(cuda.gpuarray_to_array(coeffs_gpu, "C"))
        del coeffs_gpu
        dx_gu = gpuarray.to_gpu(load_map(dx_gu).astype(np.float32))
        dy_gu = gpuarray.to_gpu(load_map(dy_gu).astype(np.float32))
        lens_func = CUDA_module.get_function("bicubiclensKernel_normtex_singletex")
    elif texture_count == 3:
        unl_CMB_tex = CUDA_module.get_texref("unl_CMB")
        dx_tex = CUDA_module.get_texref("tex_dx")
        dy_tex = CUDA_module.get_texref("tex_dy")
        tex_refs = ([unl_CMB_tex, dx_tex, dy_tex])
        unl_CMB_tex.set_array(cuda.gpuarray_to_array(coeffs_gpu, "C"))
        del coeffs_gpu
        cuda.matrix_to_texref(load_map(dx_gu).astype(np.float32), dx_tex, order="C")
        cuda.matrix_to_texref(load_map(dy_gu).astype(np.float32), dy_tex, order="C")
        lens_func = CUDA_module.get_function("bicubiclensKernel_normtex")
    else:
        tex_refs = []
        lens_func = 0
        assert 0
    # Wraping, important for periodic boundary conditions.
    # Note that WRAP has not effect for unnormalized texture coordinates.

    for tex_ref in tex_refs:
        tex_ref.set_address_mode(0, cuda.address_mode.WRAP)
        tex_ref.set_address_mode(1, cuda.address_mode.WRAP)
        tex_ref.set_filter_mode(cuda.filter_mode.POINT)
        tex_ref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)

    if timed: t0 = time.time()

    len_CMB = np.empty(unl_CMB.shape, dtype=np.float32)

    if texture_count == 0:
        lens_func(cuda.Out(len_CMB), coeffs_gpu, dx_gu, dy_gu, np.int32(unl_CMB.shape[0]),
                  block=GPU_block, grid=GPU_grid, texrefs=tex_refs)
    elif texture_count == 1:
        lens_func(cuda.Out(len_CMB), dx_gu, dy_gu, np.int32(unl_CMB.shape[0]),
                  block=GPU_block, grid=GPU_grid, texrefs=tex_refs)
    elif texture_count == 3:
        lens_func(cuda.Out(len_CMB), np.int32(unl_CMB.shape[0]),
                  block=GPU_block, grid=GPU_grid, texrefs=tex_refs)

    if timed:
        dt = time.time() - t0
        t_tot = time.time() - ti
        print "     GPU bicubic spline and transfer at %s Mpixel / sec, time %s sec" % (
            np.prod(unl_CMB.shape) / 1e6 / dt, dt)
        print " Total ex. time at %s Mpixel / sec, ex. time %s sec." % (np.prod(unl_CMB.shape) / 1e6 / t_tot, t_tot)
    return len_CMB.astype(np.float64)


def lens_alm_onGPU(lib_alm, unlalm, dx_gu, dy_gu, do_not_prefilter=False, mult_magn=False, lib_alm_out=None):
    """
    Lens the input unl_CMB map on the GPU using the pyCUDA interface.
    dx dy displacement in grid units. (f.get_dx_ingridunits() e.g.)
    Can be path to arrays or arrays or memmap.
    Will probably crash for too large maps, with need to split the job.
    Works for 4096 x 4096 at least on my laptop.

    Cost dominated by texture setup. # FIXME : try get rid of texture
    Note that the first call might be substantially slower than subsequent calls, as it caches the fft and ifft plans
    for subsequent usage.
    :param unlalm: alms of the unlensed CMB
    :param func: bicubic or bilinear
    :param normalized_tex: use a modified version of the GPU bicubic spline to account for periodicity of the map
    :param mult_magn : multiplies with the magnification prior to taking second harmonic transform.
    :return: alms of the lensed CMB.
    """
    if timed:
        ti = time.time()
    shape = lib_alm.ell_mat.shape
    rshape = (shape[0], shape[1] / 2 + 1)
    assert shape[0] == shape[1], shape
    assert IsPowerOfTwo(shape[0]), shape
    assert load_map(dx_gu).shape == shape, (load_map(dx_gu).shape, lib_alm.ell_mat.shape)
    assert load_map(dy_gu).shape == shape, (load_map(dy_gu).shape, lib_alm.ell_mat.shape)

    assert np.all(np.array(shape) % GPU_block[0] == 0), shape
    if shape[0] > 4096:
        print "--- Exercise caution, array shapes larger than 4096 have never been tested so far ---"
    if lib_alm_out is None: lib_alm_out = lib_alm
    GPU_grid = (shape[0] / GPU_block[0], shape[1] / GPU_block[1], 1)
    if mult_magn: assert texture_count == 3, \
        "lens_alm_onGPU :: multyplying with magn. only implemented with textured dx dy."
    # Prefiltering forces the interpolant to pass through the samples and increase accuracy, but dominates the cost.
    rfft2_unlCMB_gpu = gpuarray.to_gpu(lib_alm.alm2rfft(unlalm / np.prod(shape)).astype(np.complex64))
    coeffs_gpu = gpuarray.empty(lib_alm.ell_mat.shape, dtype=np.float32)
    plan, plan_inv = get_rfft_plans(shape)

    if not do_not_prefilter:
        # The prefilter makes sure the spline is exact at the nodes.
        # Uncomments this to put coeffs_gpu on pitched memory to allow later for 2D texture binding :
        # alloc,pitch  = cuda.mem_alloc_pitch(shape[0] * 4,shape[1],4) # 4 bytes per float32
        wx = (6. / (2. * np.cos(2. * np.pi * Freq(np.arange(shape[0]), shape[0]) / shape[0]) + 4.))
        wx_gpu = gpuarray.to_gpu(wx.astype(np.float32))
        prefilter = CUDA_module.get_function("cf_outer_w")
        prefilter(rfft2_unlCMB_gpu, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
        del wx_gpu

    ifft(rfft2_unlCMB_gpu, coeffs_gpu, plan_inv, False)

    # Binding arrays to textures and getting lensing func.
    if texture_count == 0:
        lens_func = CUDA_module.get_function("bicubiclensKernel_notex")
        tex_refs = []
        dx_gu = gpuarray.to_gpu(load_map(dx_gu).astype(np.float32))
        dy_gu = gpuarray.to_gpu(load_map(dy_gu).astype(np.float32))
        if mult_magn:
            det_func = CUDA_module.get_function("detmagn_notex")
    elif texture_count == 1:
        unl_CMB_tex = CUDA_module.get_texref("unl_CMB")
        tex_refs = [unl_CMB_tex]
        unl_CMB_tex.set_array(cuda.gpuarray_to_array(coeffs_gpu, "C"))
        del coeffs_gpu
        dx_gu = gpuarray.to_gpu(load_map(dx_gu).astype(np.float32))
        dy_gu = gpuarray.to_gpu(load_map(dy_gu).astype(np.float32))
        lens_func = CUDA_module.get_function("bicubiclensKernel_normtex_singletex")
        if mult_magn:
            det_func = CUDA_module.get_function("detmagn_notex")
    elif texture_count == 3:
        unl_CMB_tex = CUDA_module.get_texref("unl_CMB")
        dx_tex = CUDA_module.get_texref("tex_dx")
        dy_tex = CUDA_module.get_texref("tex_dy")
        tex_refs = ([unl_CMB_tex, dx_tex, dy_tex])
        unl_CMB_tex.set_array(cuda.gpuarray_to_array(coeffs_gpu, "C"))
        del coeffs_gpu
        cuda.matrix_to_texref(load_map(dx_gu).astype(np.float32), dx_tex, order="C")
        cuda.matrix_to_texref(load_map(dy_gu).astype(np.float32), dy_tex, order="C")
        lens_func = CUDA_module.get_function("bicubiclensKernel_normtex")
        if mult_magn:
            det_func = CUDA_module.get_function("detmagn_normtex")

    else:
        tex_refs = []
        lens_func = 0
        assert 0
    # Wraping, important for periodic boundary conditions.
    # Note that WRAP has not effect for unnormalized texture coordinates.

    for tex_ref in tex_refs:
        tex_ref.set_address_mode(0, cuda.address_mode.WRAP)
        tex_ref.set_address_mode(1, cuda.address_mode.WRAP)
        tex_ref.set_filter_mode(cuda.filter_mode.POINT)
        tex_ref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)

    if timed: t0 = time.time()

    len_CMB = gpuarray.empty(shape, dtype=np.float32)

    if texture_count == 0:
        lens_func(len_CMB, coeffs_gpu, dx_gu, dy_gu, np.int32(shape[0]),
                  block=GPU_block, grid=GPU_grid, texrefs=tex_refs)
        if mult_magn: det_func(len_CMB, dx_gu, dy_gu, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    elif texture_count == 1:
        lens_func(len_CMB, dx_gu, dy_gu, np.int32(shape[0]),
                  block=GPU_block, grid=GPU_grid, texrefs=tex_refs)
        if mult_magn: det_func(len_CMB, dx_gu, dy_gu, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    elif texture_count == 3:
        lens_func(len_CMB, np.int32(shape[0]),
                  block=GPU_block, grid=GPU_grid, texrefs=tex_refs)
        if mult_magn: det_func(len_CMB, np.int32(shape[0]), block=GPU_block, grid=GPU_grid, texrefs=[dx_tex, dy_tex])

    fft(len_CMB, rfft2_unlCMB_gpu, plan)
    lens_alm = lib_alm_out.rfftmap2alm(rfft2_unlCMB_gpu.get())
    if timed:
        dt = time.time() - t0
        t_tot = time.time() - ti
        print "     GPU bicubic spline and transfer at %s Mpixel / sec, time %s sec" % (
            np.prod(lib_alm.ell_mat.shape) / 1e6 / dt, dt)
        print " Total ex. time at %s Mpixel / sec, ex. time %s sec." % (np.prod(shape) / 1e6 / t_tot, t_tot)
    return lens_alm.astype(np.complex128)


def alm2lenmap_onGPU(lib_alm, unlalm, dx_gu, dy_gu, do_not_prefilter=False):
    """
    Lens the input unl_CMB map on the GPU using the pyCUDA interface.
    dx dy displacement in grid units. (f.get_dx_ingridunits() e.g.)
    Can be path to arrays or arrays or memmap.
    Will probably crash for too large maps, with need to split the job.
    Works for 4096 x 4096 at least on my laptop.

    Cost dominated by texture setup. # FIXME : try get rid of texture
    Note that the first call might be substantially slower than subsequent calls, as it caches the fft and ifft plans
    for subsequent usage.
    :param unl_CMB:
    :param func: bicubic or bilinear
    :param normalized_tex: use a modified version of the GPU bicubic spline to account for periodicity of the map
    :return:
    """
    if timed:
        ti = time.time()
    shape = lib_alm.ell_mat.shape
    rshape = (shape[0], shape[1] / 2 + 1)
    assert shape[0] == shape[1], shape
    assert IsPowerOfTwo(shape[0]), shape
    assert load_map(dx_gu).shape == shape, (load_map(dx_gu).shape, lib_alm.ell_mat.shape)
    assert load_map(dy_gu).shape == shape, (load_map(dy_gu).shape, lib_alm.ell_mat.shape)

    assert np.all(np.array(shape) % GPU_block[0] == 0), shape
    if shape[0] > 4096:
        print "--- Exercise caution, array shapes larger than 4096 have never been tested so far ---"

    GPU_grid = (shape[0] / GPU_block[0], shape[1] / GPU_block[1], 1)

    # Prefiltering forces the interpolant to pass through the samples and increase accuracy, but dominates the cost.
    rfft2_unlCMB_gpu = gpuarray.to_gpu(lib_alm.alm2rfft(unlalm / np.prod(shape)).astype(np.complex64))
    coeffs_gpu = gpuarray.empty(lib_alm.ell_mat.shape, dtype=np.float32)
    plan, plan_inv = get_rfft_plans(shape)

    if not do_not_prefilter:
        # The prefilter makes sure the spline is exact at the nodes.
        # Uncomments this to put coeffs_gpu on pitched memory to allow later for 2D texture binding :
        # alloc,pitch  = cuda.mem_alloc_pitch(shape[0] * 4,shape[1],4) # 4 bytes per float32
        wx = (6. / (2. * np.cos(2. * np.pi * Freq(np.arange(shape[0]), shape[0]) / shape[0]) + 4.))
        wx_gpu = gpuarray.to_gpu(wx.astype(np.float32))
        prefilter = CUDA_module.get_function("cf_outer_w")
        prefilter(rfft2_unlCMB_gpu, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
        del wx_gpu

    ifft(rfft2_unlCMB_gpu, coeffs_gpu, plan_inv, False)

    # Binding arrays to textures and getting lensing func.
    if texture_count == 0:
        lens_func = CUDA_module.get_function("bicubiclensKernel_notex")
        tex_refs = []
        dx_gu = gpuarray.to_gpu(load_map(dx_gu).astype(np.float32))
        dy_gu = gpuarray.to_gpu(load_map(dy_gu).astype(np.float32))
    elif texture_count == 1:
        unl_CMB_tex = CUDA_module.get_texref("unl_CMB")
        tex_refs = [unl_CMB_tex]
        unl_CMB_tex.set_array(cuda.gpuarray_to_array(coeffs_gpu, "C"))
        del coeffs_gpu
        dx_gu = gpuarray.to_gpu(load_map(dx_gu).astype(np.float32))
        dy_gu = gpuarray.to_gpu(load_map(dy_gu).astype(np.float32))
        lens_func = CUDA_module.get_function("bicubiclensKernel_normtex_singletex")
    elif texture_count == 3:
        unl_CMB_tex = CUDA_module.get_texref("unl_CMB")
        dx_tex = CUDA_module.get_texref("tex_dx")
        dy_tex = CUDA_module.get_texref("tex_dy")
        tex_refs = ([unl_CMB_tex, dx_tex, dy_tex])
        unl_CMB_tex.set_array(cuda.gpuarray_to_array(coeffs_gpu, "C"))
        del coeffs_gpu
        cuda.matrix_to_texref(load_map(dx_gu).astype(np.float32), dx_tex, order="C")
        cuda.matrix_to_texref(load_map(dy_gu).astype(np.float32), dy_tex, order="C")
        lens_func = CUDA_module.get_function("bicubiclensKernel_normtex")
    else:
        tex_refs = []
        lens_func = 0
        assert 0
    # Wraping, important for periodic boundary conditions.
    # Note that WRAP has not effect for unnormalized texture coordinates.

    for tex_ref in tex_refs:
        tex_ref.set_address_mode(0, cuda.address_mode.WRAP)
        tex_ref.set_address_mode(1, cuda.address_mode.WRAP)
        tex_ref.set_filter_mode(cuda.filter_mode.POINT)
        tex_ref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)

    if timed: t0 = time.time()

    len_CMB = np.empty(shape, dtype=np.float32)

    if texture_count == 0:
        lens_func(cuda.Out(len_CMB), coeffs_gpu, dx_gu, dy_gu, np.int32(shape[0]),
                  block=GPU_block, grid=GPU_grid, texrefs=tex_refs)
    elif texture_count == 1:
        lens_func(cuda.Out(len_CMB), dx_gu, dy_gu, np.int32(shape[0]),
                  block=GPU_block, grid=GPU_grid, texrefs=tex_refs)
    elif texture_count == 3:
        lens_func(cuda.Out(len_CMB), np.int32(shape[0]),
                  block=GPU_block, grid=GPU_grid, texrefs=tex_refs)

    if timed:
        dt = time.time() - t0
        t_tot = time.time() - ti
        print "     GPU bicubic spline and transfer at %s Mpixel / sec, time %s sec" % (
            np.prod(lib_alm.ell_mat.shape) / 1e6 / dt, dt)
        print " Total ex. time at %s Mpixel / sec, ex. time %s sec." % (np.prod(shape) / 1e6 / t_tot, t_tot)
    return len_CMB.astype(np.float64)
