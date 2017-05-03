# Maximum size of array to be inverted here is 2 ** 11 = 2048
# Things will run out of memory on my laptop if bigger.

GPU_HDres_max = (11, 11)

import time
import numpy as np
from lensit.misc.misc_utils import IsPowerOfTwo
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from skcuda.fft import fft, ifft
from . import GPU_block, get_rfft_plans, CUDA_inv_module, load_map, cuda, setup_texture_gpuarr


def inverse_GPU(dx, dy, rmins, NR):
    """
    Inverse displacement field on the GPU.
    The cost is completely dominated by, first, getting the results from the GPU to the host,
    and to a lesser degree by loading the dx and dy textures. Rest is super-fast.
    :return: inverse displacement in physical units
    """
    dx = load_map(dx)
    dy = load_map(dy)
    assert dx.shape == dy.shape
    assert dx.shape[0] % GPU_block[0] == 0, (dx.shape, GPU_block)
    assert dx.shape[1] % GPU_block[1] == 0, (dx.shape, GPU_block)
    assert dx.shape[0] == dx.shape[1], dx.shape
    # FIXME : THIS DOES NOT APPEAR TO WORK PROPERLY FOR NON POWER OF TWO INPUT MAPS BUT WHY ?"
    assert IsPowerOfTwo(dx.shape[0]), dx.shape

    shape = dx.shape
    GPU_grid = (shape[0] / GPU_block[0], shape[1] / GPU_block[1], 1)
    rshape = (shape[0], shape[1] / 2 + 1)

    if shape[0] > 2048:
        print "--- Exercise caution, array shapes larger than 2048 have never been tested so far ---"
    # if NR < 3:
    #    NR = 3
    #    print "--- inverse GPU : NR parameter changed to 3 ----" # This is just better
    # 2D texture references :
    Minvxx_tex = CUDA_inv_module.get_texref("Minv_xx")
    Minvyy_tex = CUDA_inv_module.get_texref("Minv_yy")
    Minvxy_tex = CUDA_inv_module.get_texref("Minv_xy")
    Minvyx_tex = CUDA_inv_module.get_texref("Minv_yx")
    dx_tex = CUDA_inv_module.get_texref("tex_dx")
    dy_tex = CUDA_inv_module.get_texref("tex_dy")

    # loading fft plans :
    plan, plan_inv = get_rfft_plans(shape)
    # Function references :
    # Spline bicubic prefiltering, bicubic interpolation and multiplication with magnification.
    prefilter = CUDA_inv_module.get_function("cf_outer_w")
    mult_inplace = CUDA_inv_module.get_function('ff_mult_inplace')
    divide_detM = CUDA_inv_module.get_function("divide_detmagn")

    cplx_type = np.complex64
    f_type = np.float32

    rminx = rmins[1].astype(f_type)
    rminy = rmins[0].astype(f_type)
    rminx_inv = (1. / rmins[1]).astype(f_type)
    rminy_inv = (1. / rmins[0]).astype(f_type)

    # The prefiltering done in this way requires square matrix, but we could change that.
    wx_gpu = (6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(shape[0])) + 4.) / shape[0])
    wx_gpu = gpuarray.to_gpu(wx_gpu.astype(f_type))

    gpu_rfft = gpuarray.empty(rshape, dtype=cplx_type, order='C')

    # Setting up dx texture. The dx texture is in grid units
    gpu_map = gpuarray.to_gpu(dx.astype(f_type))
    mult_inplace(gpu_map, rminx_inv, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    fft(gpu_map, gpu_rfft, plan)
    prefilter(gpu_rfft, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
    ifft(gpu_rfft, gpu_map, plan_inv, False)
    setup_texture_gpuarr(dx_tex, gpu_map)

    # Setting up dy texture :
    gpu_map.set(dy.astype(f_type))
    mult_inplace(gpu_map, rminy_inv, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    fft(gpu_map, gpu_rfft, plan)
    prefilter(gpu_rfft, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
    ifft(gpu_rfft, gpu_map, plan_inv, False)
    setup_texture_gpuarr(dy_tex, gpu_map)

    # Setting magnification textures  Mxx :
    func = CUDA_inv_module.get_function("get_m1pMyy")
    func(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    divide_detM(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    fft(gpu_map, gpu_rfft, plan)
    prefilter(gpu_rfft, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
    ifft(gpu_rfft, gpu_map, plan_inv, False)
    setup_texture_gpuarr(Minvxx_tex, gpu_map)

    # Setting magnification textures  Myy :
    func = CUDA_inv_module.get_function("get_m1pMxx")
    func(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    divide_detM(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    fft(gpu_map, gpu_rfft, plan)
    prefilter(gpu_rfft, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
    ifft(gpu_rfft, gpu_map, plan_inv, False)
    setup_texture_gpuarr(Minvyy_tex, gpu_map)

    # Setting magnification textures  Mxy :
    func = CUDA_inv_module.get_function("get_Mxy")
    func(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    divide_detM(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    fft(gpu_map, gpu_rfft, plan)
    prefilter(gpu_rfft, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
    ifft(gpu_rfft, gpu_map, plan_inv, False)
    setup_texture_gpuarr(Minvxy_tex, gpu_map)

    # Setting magnification textures  Myx :
    func = CUDA_inv_module.get_function("get_Myx")
    func(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    divide_detM(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    fft(gpu_map, gpu_rfft, plan)
    prefilter(gpu_rfft, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
    ifft(gpu_rfft, gpu_map, plan_inv, False)
    setup_texture_gpuarr(Minvyx_tex, gpu_map)

    # iterations proper :
    # First iteration is simpler, no need to lens maps :
    iterate_0 = CUDA_inv_module.get_function("displ_0th")
    gpu_map2 = gpuarray.empty(shape, f_type, order='C')  # We use the already declared gpu_map for the dx component.
    iterate_0(gpu_map, gpu_map2, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    if NR > 0:
        iterate = CUDA_inv_module.get_function("iterate")
        for i in xrange(NR):
            iterate(gpu_map, gpu_map2, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    mult_inplace(gpu_map, rminx, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)  # Turning to physical units
    mult_inplace(gpu_map2, rminy, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)

    return gpu_map.get(), gpu_map2.get()  # in physical units


def inverse_GPU_timed(dx, dy, rmins, NR):
    """
    Same with some timing info
    """
    ti = time.time()
    dx = load_map(dx)
    dy = load_map(dy)

    assert dx.shape == dy.shape
    assert dx.shape[0] % GPU_block[0] == 0, dx.shape
    assert dx.shape[1] % GPU_block[1] == 0, dx.shape
    assert np.all(np.array(dx.shape) % GPU_block[0] == 0), dx.shape
    assert dx.shape[0] == dx.shape[1], dx.shape
    # assert IsPowerOfTwo(dx.shape[0]), dx.shape

    shape = dx.shape
    GPU_grid = (shape[0] / GPU_block[0], shape[1] / GPU_block[1], 1)
    rshape = (shape[0], shape[1] / 2 + 1)

    if shape[0] > 4096:
        print "--- Exercise caution, array shapes larger than 2048 have never been tested so far ---"
    # if NR < 3:
    #    NR = 3
    #    print "--- inverse GPU : NR parameter changed to 3 ----" # This is just better
    # 2D texture references :
    Minvxx_tex = CUDA_inv_module.get_texref("Minv_xx")
    Minvyy_tex = CUDA_inv_module.get_texref("Minv_yy")
    Minvxy_tex = CUDA_inv_module.get_texref("Minv_xy")
    Minvyx_tex = CUDA_inv_module.get_texref("Minv_yx")
    dx_tex = CUDA_inv_module.get_texref("tex_dx")
    dy_tex = CUDA_inv_module.get_texref("tex_dy")

    double_precision_ffts = False  # If double precision ffts are wanted, need to check more carefully how gpu_map behaves.
    # loading fft plans :
    plan, plan_inv = get_rfft_plans(shape, double_precision=double_precision_ffts)
    # Function references :
    # Spline bicubic prefiltering, bicubic interpolation and multiplication with magnification.
    if not double_precision_ffts:
        prefilter = CUDA_inv_module.get_function("cf_outer_w")
        mult_inplace = CUDA_inv_module.get_function('ff_mult_inplace')
        mult = CUDA_inv_module.get_function('ff_mult')

    else:
        prefilter = CUDA_inv_module.get_function("cdd_outer_w")
        mult_inplace = CUDA_inv_module.get_function('dd_mult_inplace')
        mult = CUDA_inv_module.get_function('dd_mult')

    divide_detM = CUDA_inv_module.get_function("divide_detmagn")

    cplx_type = np.complex64 if not double_precision_ffts else np.complex128
    f_type = np.float32 if not double_precision_ffts else np.float64
    ret_dx = np.empty(shape, dtype=f_type)
    ret_dy = np.empty(shape, dtype=f_type)

    wx_gpu = (6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(shape[0])) + 4.) / shape[0])
    wx_gpu = gpuarray.to_gpu(wx_gpu.astype(f_type))
    # rmins_gpu = gpuarray.to_gpu(np.array([rmins[0]]).astype(f_type))
    # rmins_inv_gpu = gpuarray.to_gpu(np.array([(1. / rmins[0])]).astype(f_type))
    rminx = rmins[1].astype(f_type)
    rminy = rmins[0].astype(f_type)

    rminx_inv = (1. / rmins[1]).astype(f_type)
    rminy_inv = (1. / rmins[0]).astype(f_type)

    gpu_rfft = gpuarray.empty(rshape, dtype=cplx_type, order='C')

    gpu_map = gpuarray.empty(shape, dtype=f_type, order='C')

    if True:
        print " Setup OK", time.time() - ti
        t0 = time.time()
    # Setting up dx texture. The dx texture is in grid units

    gpu_map.set(dx.astype(f_type))  # )
    mult_inplace(gpu_map, rminx_inv, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    fft(gpu_map, gpu_rfft, plan)
    prefilter(gpu_rfft, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
    ifft(gpu_rfft, gpu_map, plan_inv, False)
    setup_texture_gpuarr(dx_tex, gpu_map)
    # setup_texture_flags(dx_tex)
    # setup_texture(dx_tex,gpu_map.get())
    # setup_pitched_texture(dx_tex, shape, pitch, alloc) # Boh, does not work
    if True:
        print "     dx texture setup", time.time() - t0
        t0 = time.time()
    # Setting up dy texture :
    gpu_map.set(dy.astype(f_type))
    mult_inplace(gpu_map, rminy_inv, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    fft(gpu_map, gpu_rfft, plan)
    prefilter(gpu_rfft, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
    ifft(gpu_rfft, gpu_map, plan_inv, False)
    setup_texture_gpuarr(dy_tex, gpu_map)

    # dy_tex.set_array(cuda.gpuarray_to_array(gpu_map,"C"))
    # setup_texture_flags(dy_tex)
    # setup_texture(dy_tex,gpu_map.get())
    # setup_pitched_texture(dy_tex, shape, pitch, alloc)

    if True:
        print "     dy texture setup", time.time() - t0
        t0 = time.time()
    # Setting magnification textures  Mxx :
    func = CUDA_inv_module.get_function("get_m1pMyy")
    func(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    divide_detM(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    fft(gpu_map, gpu_rfft, plan)
    prefilter(gpu_rfft, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
    ifft(gpu_rfft, gpu_map, plan_inv, False)
    setup_texture_gpuarr(Minvxx_tex, gpu_map)
    # Minvxx_tex.set_array(cuda.gpuarray_to_array(gpu_map,"C"))
    # setup_texture_flags(Minvxx_tex)
    # setup_texture(Minvxx_tex,gpu_map.get())

    if True:
        print "     Mxx texture setup", time.time() - t0
        t0 = time.time()
    # Setting magnification textures  Myy :
    func = CUDA_inv_module.get_function("get_m1pMxx")
    func(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    divide_detM(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    fft(gpu_map, gpu_rfft, plan)
    prefilter(gpu_rfft, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
    ifft(gpu_rfft, gpu_map, plan_inv, False)
    setup_texture_gpuarr(Minvyy_tex, gpu_map)
    # setup_texture(Minvyy_tex,gpu_map.get())

    if True:
        print "     Myy texture setup", time.time() - t0
        t0 = time.time()
    # Setting magnification textures  Mxy :
    func = CUDA_inv_module.get_function("get_Mxy")
    func(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    divide_detM(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    fft(gpu_map, gpu_rfft, plan)
    prefilter(gpu_rfft, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
    ifft(gpu_rfft, gpu_map, plan_inv, False)
    setup_texture_gpuarr(Minvxy_tex, gpu_map)
    # Minvxy_tex.set_array(cuda.gpuarray_to_array(gpu_map,"C"))
    # setup_texture(Minvxy_tex,gpu_map.get())

    if True:
        print "     Mxy texture setup", time.time() - t0
        t0 = time.time()
    # Setting magnification textures  Myx :
    func = CUDA_inv_module.get_function("get_Myx")
    func(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    divide_detM(gpu_map, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    fft(gpu_map, gpu_rfft, plan)
    prefilter(gpu_rfft, wx_gpu, np.int32(rshape[1]), np.int32(rshape[0]), block=GPU_block, grid=GPU_grid)
    ifft(gpu_rfft, gpu_map, plan_inv, False)
    setup_texture_gpuarr(Minvyx_tex, gpu_map)
    # Minvyx_tex.set_array(cuda.gpuarray_to_array(gpu_map,"C"))
    # setup_texture_flags(Minvyx_tex)

    # setup_texture(Minvyx_tex,gpu_map.get())
    if True:
        print "     Myx texture setup", time.time() - t0
        t0 = time.time()
    # del wx_gpu,gpu_rfft,rmins_inv_gpu
    if True:
        print "    some deletes :", time.time() - t0
        t0 = time.time()

    # iterations proper :
    # First iteration is simpler, no need to lens maps :
    iterate_0 = CUDA_inv_module.get_function("displ_0th")
    # ret_dx = gpuarray.empty(shape,f_type,order='C')
    # We use already declared gpu_map as dx
    gpu_map2 = gpuarray.empty(shape, f_type, order='C')
    if True:
        print "     Initializing additional displ for iterations ", time.time() - t0
        t0 = time.time()

    iterate_0(gpu_map, gpu_map2, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    if True:
        print "     First Iteration", time.time() - t0
        t0 = time.time()
    if NR > 0:
        iterate = CUDA_inv_module.get_function("iterate")
        for i in xrange(NR):
            iterate(gpu_map, gpu_map2, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    if True:
        print "     %s addtional it." % NR, time.time() - t0
        t0 = time.time()
        # Looks like this has the same cost :
        # mult(cuda.Out(ret_dx), rmins_gpu,np.int32(shape[0]), block=GPU_block, grid=GPU_grid) # Turning to physical units
    # mult(cuda.Out(ret_dy), rmins_gpu,np.int32(shape[0]), block=GPU_block, grid=GPU_grid)
    mult(cuda.Out(ret_dx), gpu_map, rminx, np.int32(shape[0]), block=GPU_block,
         grid=GPU_grid)  # Turning to physical units
    mult(cuda.Out(ret_dy), gpu_map2, rminy, np.int32(shape[0]), block=GPU_block, grid=GPU_grid)

    # if True:
    #    print "     rescaling to physical units and pulling", time.time() - t0

    # ret_dx = gpu_map.get()
    # ret_dy = gpu_map2.get()
    if True:
        dt = time.time() - t0
        print "    rescaling and  Pplling result from GPU ", dt
        GB = 2 * np.prod(dx.shape) * 4 / 1e9  # GigaBytes transfered
        print "     rate of %s GB / s" % (GB / dt)

    if True:
        dt = time.time() - ti
        print " Total time", dt
        print " GPU inverse total run time %s Mpixels / sec " % (np.round(np.prod(shape) / 1e6 / dt, 4))

    return ret_dx.astype(float), ret_dy.astype(float)  # in physical units


"""
On 1024 x 1024 map :
 Setup OK 0.00304102897644
     dx texture setup 0.0110950469971
     dy texture setup 0.0146839618683
     Mxx texture setup 0.000366926193237
     Myy texture setup 0.000336170196533
     Mxy texture setup 0.000375986099243
     Myx texture setup 0.000324964523315
     Initializing additional displ for iterations  0.000133037567139
     First Iteration 6.5803527832e-05
     3 addtional it. 0.0001220703125
     rescaling to physical units  6.19888305664e-05
     Pulling result from GPU  0.150506973267
     rate of 0.0557356766795 GB / s
 Total time 0.181234121323
 GPU inverse total run time 5.7858 Mpixels / sec
++ CHECK for timed inverse GPU : std rel. dev. dx,  max  rel. dev. dev  0.000156698339281 0.00889743896917

On 2048 x 2048 map :
 Setup OK 0.00144386291504
     dx texture setup 0.0357120037079
     dy texture setup 0.0789849758148
     Mxx texture setup 0.000344038009644
     Myy texture setup 0.00036096572876
     Mxy texture setup 0.000359773635864
     Myx texture setup 0.000334024429321
     Initializing additional displ for iterations  9.51290130615e-05
     First Iteration 4.6968460083e-05
     3 addtional it. 8.51154327393e-05
     rescaling to physical units  4.29153442383e-05
     Pulling result from GPU  0.379819869995
     rate of 0.0883430137566 GB / s
 Total time 0.497756958008
 GPU inverse total run time 8.4264 Mpixels / sec
++ GPU inverse 8.2198 Mpixels / sec
++ GPU inverse 9.2638 Mpixels / sec
"""
