import os
import numpy as np
from scipy import interpolate
from multiprocessing import Pool

from lensit.misc.map_spliter import periodicmap_spliter
from lensit.misc.misc_utils import IsPowerOfTwo, Log2ofPowerof2, PartialDerivativePeriodic, enumerate_progress, Freq
try:
    from scipy import weave
except:
    import weave
verbose = True
bicubicspline_header = r' "%s/lensit/gpu/bicubicspline.h" ' % os.path.abspath(os.curdir)


def setup_Pool():
    if os.getenv("OMP_NUM_THREADS", None) is not None:
        if verbose: print 'pool: I see ', str(int(os.getenv("OMP_NUM_THREADS"))), " OMP_NUM_THREADS"
        pool = Pool(processes=int(os.getenv("OMP_NUM_THREADS")))
    else:
        if verbose: print "pool: I do not see OMP_NUM_THREADS"
        pool = Pool()
    return pool


# args = path_to_map,path_to_dx,path_to_dy,self.buffers[0],self.buffers[1],self.HD_res[0],self.HD_res[1],self.NR_iter,self.k]

def get_lens_Pooled(args, root_Nthreads=16, do_not_prefilter=False):
    """
    Lens the input map according to the displacement fields dx dy.
    map typically would be 8192 * 8192 for 1'4 amin res.
    Does this by splitting the job in chunks of typically (256 * 256) as specified by the LD_res parameters,
    allowing a buffer size to ensure the junctions are properly performed.

    This version uses Pool to split the work into chunks, but avoiding pickling of the full class data.
    """
    return Pool_generic(_lens_chk_N, args, root_Nthreads=root_Nthreads, do_not_prefilter=do_not_prefilter)[0]


def get_lens_Pooled_weave(args, root_Nthreads=16, do_not_prefilter=False):
    """
    Same with much faster weave inline C++ compilation and GPU inspired algorithm.
    """
    return Pool_generic(_lens_chk_N_weave, args, root_Nthreads=root_Nthreads, do_not_prefilter=do_not_prefilter)[0]


def get_inverse_Pooled(args, root_Nthreads=16):
    return Pool_generic(_get_inverse_chk_N, args, root_Nthreads=root_Nthreads)


def Pool_generic(func, arg, root_Nthreads, do_not_prefilter=False):
    assert len(arg) == 11, arg
    path_to_map, path_to_dx, path_to_dy, buff0, buff1, lside0, lside1, HD_res0, HD_res1, NR_iter, kspl = arg

    assert os.path.exists(path_to_dx) and os.path.exists(path_to_dy)
    assert IsPowerOfTwo(root_Nthreads)
    diff0, diff1 = Log2ofPowerof2((root_Nthreads, root_Nthreads))
    HD_shape = (2 ** HD_res0, 2 ** HD_res1)
    LD = (HD_res0 - diff0, HD_res1 - diff1)
    pool = setup_Pool()
    ret_list = pool.map(func, [
        [i, path_to_map, path_to_dx, path_to_dy, buff0, buff1, lside0, lside1, HD_res0, HD_res1, NR_iter, kspl, LD[0],
         LD[1], do_not_prefilter]
        for i in range(root_Nthreads ** 2)])
    pool.close()
    pool.join()
    # Recombines from the lensed_chks :
    spliter_lib = periodicmap_spliter()  # library to split periodic maps.
    ret = []  # one map for lens, two for inverse
    for i in range(len(ret_list[0])):
        map = np.empty(HD_shape)
        if verbose:
            for j, N in enumerate_progress(xrange(root_Nthreads ** 2), label='Pool_generic:patching chks together'):
                sLDs, sHDs = spliter_lib.get_slices_chk_N(N, LD, (HD_res0, HD_res1), (buff0, buff1), inverse=True)
                map[sHDs[0]] = ret_list[N][i][sLDs[0]]
            ret.append(map)
        else:
            for j, N in enumerate(xrange(root_Nthreads ** 2)):
                sLDs, sHDs = spliter_lib.get_slices_chk_N(N, LD, (HD_res0, HD_res1), (buff0, buff1), inverse=True)
                map[sHDs[0]] = ret_list[N][i][sLDs[0]]
            ret.append(map)
    return ret


def _lens_chk_N(args):
    assert len(args) == 15, args

    N, path_to_map, path_to_dx, path_to_dy, buff0, buff1, \
    lside0, lside1, HD_res0, HD_res1, NR_iter, kspl, LD0, LD1, do_not_prefilter = args
    HD_res = (HD_res0, HD_res1)
    LD = (LD0, LD1)
    s = (2 ** LD[0] + 2 * buff0, 2 ** LD[1] + 2 * buff1)  # Chunk shape

    dx = np.load(path_to_dx, mmap_mode='r')
    dy = np.load(path_to_dy, mmap_mode='r')
    map = np.load(path_to_map, mmap_mode='r')

    rmin0 = lside0 / 2 ** HD_res[0]
    rmin1 = lside1 / 2 ** HD_res[1]

    map_chk_N = np.empty(s, dtype=float)
    dx_gu = np.empty(s, dtype=float)  # will dx displ. in grid units of each chunk (typ. (256 * 256) )
    dy_gu = np.empty(s, dtype=float)  # will dy displ. in grid units of each chunk (typ. (256 * 256) )
    sLDs, sHDs = periodicmap_spliter().get_slices_chk_N(N, LD, HD_res, (buff0, buff1))
    for sLD, sHD in zip(sLDs, sHDs):
        # Displacements chunk in grid units, and map chunk to displace.
        dx_gu[sLD] = dx[sHD] / rmin1
        dy_gu[sLD] = dy[sHD] / rmin0
        map_chk_N[sLD] = map[sHD]

    if do_not_prefilter:
        # Undoing the prefiltering prior to apply bicubic interpolation
        map_chk_N = np.fft.rfft2(map_chk_N)
        w0 = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(s[0])) + 4.)
        map_chk_N /= np.outer(w0, w0[0:map_chk_N.shape[1]])
        map_chk_N = np.fft.irfft2(map_chk_N, s)

    idc0, idc1 = np.indices(s)  # Two typically (256 + 2 * buff * 256 + 2* buff) maps
    lx = (idc1 + dx_gu).flatten()  # No need to enforce periodicity here.
    ly = (idc0 + dy_gu).flatten()  # No need to enforce periodicity here.
    return (interpolate.RectBivariateSpline(np.arange(s[0]), np.arange(s[1]),
                                            map_chk_N, kx=kspl, ky=kspl).ev(ly, lx).reshape(s),)


def _lens_chk_N_weave(args):
    assert len(args) == 15, args

    N, path_to_map, path_to_dx, path_to_dy, buff0, buff1, \
    lside0, lside1, HD_res0, HD_res1, NR_iter, kspl, LD0, LD1, do_not_prefilter = args
    HD_res = (HD_res0, HD_res1)
    LD = (LD0, LD1)

    s = (2 ** LD[0] + 2 * buff0, 2 ** LD[1] + 2 * buff1)  # Chunk shape
    assert s[0] == s[1], 'only square matrices here.'  # TODO

    dx = np.load(path_to_dx, mmap_mode='r')
    dy = np.load(path_to_dy, mmap_mode='r')
    map = np.load(path_to_map, mmap_mode='r')

    rmin0 = lside0 / 2 ** HD_res[0]
    rmin1 = lside1 / 2 ** HD_res[1]

    filtmap = np.empty(s, dtype=np.float64)
    dx_gu = np.empty(s, dtype=np.float64)  # will dx displ. in grid units of each chunk (typ. (256 * 256) )
    dy_gu = np.empty(s, dtype=np.float64)  # will dy displ. in grid units of each chunk (typ. (256 * 256) )
    lenmap = np.empty(s, dtype=np.float64)
    sLDs, sHDs = periodicmap_spliter().get_slices_chk_N(N, LD, HD_res, (buff0, buff1))
    for sLD, sHD in zip(sLDs, sHDs):
        # Displacements chunk in grid units, and map chunk to displace.
        dx_gu[sLD] = dx[sHD] / rmin1
        dy_gu[sLD] = dy[sHD] / rmin0
        filtmap[sLD] = map[sHD]

    if not do_not_prefilter:
        # Prefiltering the map prior application of bicubic interpolation
        filtmap = np.fft.rfft2(filtmap)
        w0 = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(s[0])) + 4.)
        filtmap *= np.outer(w0, w0[0:filtmap.shape[1]])
        filtmap = np.fft.irfft2(filtmap, s)

    bicubicspline = r"\
                int i,j;\
               for( j= 0; j < width; j++ )\
                   {\
                   for( i = 0; i < width; i++)\
                       {\
                       lenmap[j * width + i] = bicubiclensKernel(filtmap,i + dx_gu[j * width + i],j + dy_gu[j * width + i],width);\
                       }\
                   }"
    width = int(s[0])
    weave.inline(bicubicspline, ['lenmap', 'filtmap', 'dx_gu', 'dy_gu', 'width'], headers=[bicubicspline_header])
    return (lenmap,)


def _get_inverse_chk_N(args):
    """
    Returns inverse displacement in chunk N
    Uses periodic boundary conditions, which is not applicable to chunks, thus there
    will be boudary effects on the edges (2 or 4 pixels depending on the rule). Make sure the buffer is large enough.
    """
    assert len(args) == 15, args

    N, path_to_map, path_to_dx, path_to_dy, buff0, buff1, \
    lside0, lside1, HD_res0, HD_res1, NR_iter, kspl, LD0, LD1, do_not_prefilter = args

    HD_res = (HD_res0, HD_res1)
    LD = (LD0, LD1)
    s = (2 ** LD[0] + 2 * buff0, 2 ** LD[1] + 2 * buff1)  # Chunk shape

    rmin0 = lside0 / 2 ** HD_res[0]
    rmin1 = lside1 / 2 ** HD_res[1]

    # Get magn. matrix of the chunk:
    extra_buff = np.array((5, 5))  # To avoid surprises with the periodic derivatives
    dx = np.zeros(s + 2 * extra_buff)  # will dx displ. in grid units of each chunk (typ. (256 * 256) )
    dy = np.zeros(s + 2 * extra_buff)  # will dy displ. in grid units of each chunk (typ. (256 * 256) )
    sLDs, sHDs = periodicmap_spliter().get_slices_chk_N(N, LD, HD_res, (buff0 + extra_buff[0], buff1 + extra_buff[1]))
    for sLD, sHD in zip(sLDs, sHDs):
        dx[sLD] = np.load(path_to_dx, mmap_mode='r')[sHD] / rmin1  # Need grid units displacement for the bicubic spline
        dy[sLD] = np.load(path_to_dy, mmap_mode='r')[sHD] / rmin0

    # Jacobian matrix of the chunk :
    sl0 = slice(extra_buff[0], dx.shape[0] - extra_buff[0])
    sl1 = slice(extra_buff[1], dx.shape[1] - extra_buff[1])

    Minv_yy = - (PartialDerivativePeriodic(dx, axis=1)[sl0, sl1] + 1.)
    Minv_xx = - (PartialDerivativePeriodic(dy, axis=0)[sl0, sl1] + 1.)
    Minv_xy = PartialDerivativePeriodic(dy, axis=1)[sl0, sl1]
    Minv_yx = PartialDerivativePeriodic(dx, axis=0)[sl0, sl1]
    det = Minv_yy * Minv_xx - Minv_xy * Minv_yx
    if not np.all(det > 0.): print "ffs_displ::Negative value in det k : something's weird, you'd better check that"
    # Inverse magn. elements. (with a minus sign) We may need to spline these later for further NR iterations :
    Minv_xx /= det
    Minv_yy /= det
    Minv_xy /= det
    Minv_yx /= det
    del det

    dx = dx[sl0, sl1]  # Getting rid of extra buffer
    dy = dy[sl0, sl1]  # Getting rid of extra buffer
    ex = (Minv_xx * dx + Minv_xy * dy)
    ey = (Minv_yx * dx + Minv_yy * dy)

    if NR_iter == 0: return ex * rmin1, ey * rmin0

    # Setting up a bunch of splines to interpolate the increment to the displacement according to Newton-Raphson.
    # Needed are splines of the forward displacement and of the (inverse, as implemented here) magnification matrix.
    # Hopefully the map resolution is enough to spline the magnification matrix.
    s0, s1 = dx.shape
    r0 = s0
    r1 = s1 / 2 + 1  # rfft shape

    w0 = 6. / (2. * np.cos(2. * np.pi * Freq(np.arange(r0), s0) / s0) + 4.)
    w1 = 6. / (2. * np.cos(2. * np.pi * Freq(np.arange(r1), s1) / s1) + 4.)
    # FIXME: switch to pyfftw :
    bic_filter = lambda _map: np.fft.irfft2(np.fft.rfft2(_map) * np.outer(w0, w1))

    dx = bic_filter(dx)
    dy = bic_filter(dy)
    Minv_xy = bic_filter(Minv_xy)
    Minv_yx = bic_filter(Minv_yx)
    Minv_xx = bic_filter(Minv_xx)
    Minv_yy = bic_filter(Minv_yy)

    header = r' "%s/lensit/gpu/bicubicspline.h" ' % os.path.abspath(os.curdir)
    iterate = r"\
        double fx,fy;\
        double ex_len_dx,ey_len_dy,len_Mxx,len_Mxy,len_Myx,len_Myy;\
        int i = 0;\
        for(int y= 0; y < width; y++ )\
           {\
           for(int x = 0; x < width; x++,i++)\
            {\
            fx = x +  ex[i];\
            fy = y +  ey[i];\
            ex_len_dx = ex[i] +  bicubiclensKernel(dx,fx,fy,width);\
            ey_len_dy = ey[i] +  bicubiclensKernel(dy,fx,fy,width);\
            len_Mxx =  bicubiclensKernel(Minv_xx,fx,fy,width);\
            len_Myy =  bicubiclensKernel(Minv_yy,fx,fy,width);\
            len_Mxy =  bicubiclensKernel(Minv_xy,fx,fy,width);\
            len_Myx =  bicubiclensKernel(Minv_yx,fx,fy,width);\
            ex[i] += len_Mxx * ex_len_dx + len_Mxy * ey_len_dy;\
            ey[i] += len_Myx * ex_len_dx + len_Myy * ey_len_dy;\
            }\
        }\
        "
    width = int(s0)
    assert s0 == s1, 'Havent checked how this works with rectangular maps'
    for i in range(0, NR_iter):
        weave.inline(iterate, ['ex', 'ey', 'dx', 'dy', 'Minv_xx', 'Minv_yy', 'Minv_xy', 'Minv_yx', 'width'],
                     headers=[header])
    return ex * rmin1, ey * rmin0


def _get_inverse_chk_N_old(args):
    """
    Returns inverse displacement in chunk N
    Uses periodic boundary conditions, which is not applicable to chunks, thus there
    will be boudary effects on the edges (2 or 4 pixels depending on the rule). Make sure the buffer is large enough.
    """
    assert len(args) == 15, args

    N, path_to_map, path_to_dx, path_to_dy, buff0, buff1, \
    lside0, lside1, HD_res0, HD_res1, NR_iter, kspl, LD0, LD1, do_not_prefilter = args

    HD_res = (HD_res0, HD_res1)
    LD = (LD0, LD1)
    s = (2 ** LD[0] + 2 * buff0, 2 ** LD[1] + 2 * buff1)  # Chunk shape

    rmin0 = lside0 / 2 ** HD_res[0]
    rmin1 = lside1 / 2 ** HD_res[1]

    # Get magn. matrix of the chunk:
    extra_buff = np.array((5, 5))  # To avoid surprises with the periodic derivatives
    dx = np.zeros(s + 2 * extra_buff)  # will dx displ. in grid units of each chunk (typ. (256 * 256) )
    dy = np.zeros(s + 2 * extra_buff)  # will dy displ. in grid units of each chunk (typ. (256 * 256) )
    sLDs, sHDs = periodicmap_spliter().get_slices_chk_N(N, LD, HD_res, (buff0 + extra_buff[0], buff1 + extra_buff[1]))
    for sLD, sHD in zip(sLDs, sHDs):
        dx[sLD] = np.load(path_to_dx, mmap_mode='r')[sHD]
        dy[sLD] = np.load(path_to_dy, mmap_mode='r')[sHD]

    # Jacobian matrix of the chunk :
    sl0 = slice(extra_buff[0], dx.shape[0] - extra_buff[0])
    sl1 = slice(extra_buff[1], dx.shape[1] - extra_buff[1])

    dfxdx_1 = PartialDerivativePeriodic(dx, axis=1, h=rmin1, rule='4pts')[sl0, sl1] + 1.
    dfydy_1 = PartialDerivativePeriodic(dy, axis=0, h=rmin0, rule='4pts')[sl0, sl1] + 1.
    dfydx = PartialDerivativePeriodic(dy, axis=1, h=rmin1, rule='4pts')[sl0, sl1]
    dfxdy = PartialDerivativePeriodic(dx, axis=0, h=rmin0, rule='4pts')[sl0, sl1]
    det = (dfxdx_1) * (dfydy_1) - dfydx * dfxdy

    if not np.all(det > 0.): print "ffs_displ::Negative value in det k : something's weird, you'd better check that"
    # Inverse magn. elements. (with a minus sign) We may need to spline these later for further NR iterations :
    _Minv_xx = - dfydy_1 / det
    _Minv_yy = - dfxdx_1 / det
    _Minv_xy = dfxdy / det
    _Minv_yx = dfydx / det
    del dfxdx_1, dfydx, dfydy_1, dfxdy

    dx = dx[sl0, sl1]  # Getting rid of extra buffer
    dy = dy[sl0, sl1]  # Getting rid of extra buffer
    dxn = (_Minv_xx * dx + _Minv_xy * dy)
    dyn = (_Minv_yx * dx + _Minv_yy * dy)

    if NR_iter == 0: return dxn, dyn

    # Setting up a bunch of splines to interpolate the increment to the displacement according to Newton-Raphson.
    # Needed are splines of the forward displacement and of the (inverse, as implemented here) magnification matrix.
    # Hopefully the map resolution is enough to spline the magnification matrix.

    xcoord = np.arange(s[1]) * rmin1
    ycoord = np.arange(s[0]) * rmin0
    spl_dx = interpolate.RectBivariateSpline(ycoord, xcoord, dx, kx=kspl, ky=kspl)
    spl_dy = interpolate.RectBivariateSpline(ycoord, xcoord, dy, kx=kspl, ky=kspl)
    spl_xx = interpolate.RectBivariateSpline(ycoord, xcoord, _Minv_xx, kx=kspl, ky=kspl)
    spl_yy = interpolate.RectBivariateSpline(ycoord, xcoord, _Minv_yy, kx=kspl, ky=kspl)
    spl_xy = interpolate.RectBivariateSpline(ycoord, xcoord, _Minv_xy, kx=kspl, ky=kspl)
    spl_yx = interpolate.RectBivariateSpline(ycoord, xcoord, _Minv_yx, kx=kspl, ky=kspl)

    idc = np.indices(s)
    y_x = idc[1] * rmin1
    y_y = idc[0] * rmin0
    del idc

    for i in range(NR_iter):
        dxn_1 = dxn
        dyn_1 = dyn
        lx = (y_x + dxn_1).flatten()
        ly = (y_y + dyn_1).flatten()
        res_x = dxn_1 + spl_dx.ev(ly, lx).reshape(s)  # dx residuals
        res_y = dyn_1 + spl_dy.ev(ly, lx).reshape(s)  # dy residuals
        dxn = dxn_1 + spl_xx.ev(ly, lx).reshape(s) * res_x + spl_xy.ev(ly, lx).reshape(s) * res_y
        dyn = dyn_1 + spl_yx.ev(ly, lx).reshape(s) * res_x + spl_yy.ev(ly, lx).reshape(s) * res_y
    return dxn, dyn
