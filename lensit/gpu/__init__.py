import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from skcuda.fft import Plan
import os

CUDA_module = SourceModule(open(os.path.join(os.path.dirname(__file__), "kernels_lensing.c"), 'r').read())
CUDA_inv_module = SourceModule(open(os.path.join(os.path.dirname(__file__), "kernels_inversion.c"), 'r').read())

# Size of the GPU block (Num. of threads per block) :
GPU_block = (32, 32, 1)

# This will store CUDA fft and ifft plans :
fft_plans = {}
fft_inv_plans = {}


def load_map(map):
    if isinstance(map, str):
        return np.load(map, mmap_mode='r')
    else:
        return map


# In fact, the fastest way to setup textures so far seems to be :
#    tex_ref.set_array(cuda.gpuarray_to_array(coeffs_gpu,"C"))
#    and then set the flags.

def setup_texture_nparr(tex_ref, arr):
    cuda.matrix_to_texref(load_map(arr).astype(np.float32), tex_ref, order="C")
    tex_ref.set_address_mode(0, cuda.address_mode.WRAP)
    tex_ref.set_address_mode(1, cuda.address_mode.WRAP)
    tex_ref.set_filter_mode(cuda.filter_mode.POINT)
    tex_ref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)


def setup_texture_gpuarr(tex_ref, arr):
    _arr = arr.astype(np.float32) if arr.dtype != np.float32 else arr
    tex_ref.set_array(cuda.gpuarray_to_array(_arr, "C"))
    tex_ref.set_address_mode(0, cuda.address_mode.WRAP)
    tex_ref.set_address_mode(1, cuda.address_mode.WRAP)
    tex_ref.set_filter_mode(cuda.filter_mode.POINT)
    tex_ref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)


def setup_texture_flags(tex_ref):
    tex_ref.set_address_mode(0, cuda.address_mode.WRAP)
    tex_ref.set_address_mode(1, cuda.address_mode.WRAP)
    tex_ref.set_filter_mode(cuda.filter_mode.POINT)
    tex_ref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)


def setup_pitched_texture(tex_ref, shape, pitch, alloc):
    """
    Bind 2D texture to memory location given by alloc with pitch size pitch and shape shape.
    alloc and pitch might come from  e.g.
    alloc,pitch  = cuda.mem_alloc_pitch(shape[0] * 4,shape[1],4) # 4 bytes per float32

    :param tex_reference: 2D texture reference
    :param shape: shape of the array to be placed there
    :param pitch: pitch parameter for CUDA texture binding
    :param alloc: address
    :return:
    """
    assert (pitch % 8) == 0  # for float types
    descr = cuda.ArrayDescriptor()
    descr.format = cuda.array_format.FLOAT
    descr.height = shape[0]
    descr.width = shape[1]
    descr.num_channels = 1
    tex_ref.set_address_2d(alloc, descr, pitch)
    tex_ref.set_address_mode(0, cuda.address_mode.WRAP)
    tex_ref.set_address_mode(1, cuda.address_mode.WRAP)
    tex_ref.set_filter_mode(cuda.filter_mode.POINT)
    tex_ref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)


def get_rfft_plans(shape, double_precision=False):
    """
    Loads or computes fft plans for ffts performed on the GPU.
    """
    real_type = np.float32 if not double_precision else np.float64
    cplx_type = np.complex64 if not double_precision else np.complex128

    lab = '%s x %s real2complex' % (shape[0], shape[1])
    if double_precision: lab += ' (double)'
    if lab not in fft_plans.keys():
        print "lens_GPU : building and caching fft plan %s" % lab
        fft_plans[lab] = Plan(shape, real_type, cplx_type)
    if lab not in fft_inv_plans.keys():
        print "lens_GPU : building and caching ifft plan %s" % lab
        fft_inv_plans[lab] = Plan(shape, cplx_type, real_type)
    return fft_plans[lab], fft_inv_plans[lab]
