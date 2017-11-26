import hashlib
import os

import numpy as np
from scipy import interpolate
try :
    from scipy import weave
except:
    import weave
import lensit as li
import lensit.misc.map_spliter as map_spliter
from lensit.ffs_deflect import ffs_pool
from lensit.misc import misc_utils as utils
from lensit.misc import rfft2_utils
from lensit.misc.misc_utils import PartialDerivativePeriodic as PDP, Log2ofPowerof2, Freq
from lensit import pbs


def get_GPUbuffers(GPU_res):
    """
    Defines the splitting of a big map to smaller chunks if above GPU memory.
    :param GPU_res: 2 ** GPU_res is the supported size of the map on the GPU.
    """
    assert len(GPU_res) == 2
    # This forces the use of power-of-two-sized input maps
    LD_res = (GPU_res[0] - 1, GPU_res[1] - 1)
    buffers = (2 ** (LD_res[0] - 1), 2 ** (LD_res[1] - 1))
    return LD_res, buffers
    # FIXME : It looks like the lens and inverse routines on GPU do not work properly for non-power of two shapes,
    # FIXME : but cant figure out exactly why (textures)


def FlatIndices(coord, shape):
    """
    Returns the indices in a flattened 'C' convention array of multidimensional indices
    """
    ndim = len(shape)
    idc = coord[ndim - 1, :]
    for j in xrange(1, ndim): idc += np.prod(shape[ndim - j:ndim]) * coord[ndim - 1 - j, :]
    return idc


def displacement_fromplm(lib_plm, plm, **kwargs):
    return ffs_displacement(lib_plm.alm2map(plm * lib_plm.get_ikx()),
                            lib_plm.alm2map(plm * lib_plm.get_iky()),
                            lib_plm.ell_mat.lsides, **kwargs)


def displacement_fromolm(lib_plm, olm, **kwargs):
    return ffs_displacement(lib_plm.alm2map(-olm * lib_plm.get_iky()),
                            lib_plm.alm2map(olm * lib_plm.get_ikx()),
                            lib_plm.ell_mat.lsides, **kwargs)


def displacement_frompolm(lib_plm, plm, olm, **kwargs):
    return ffs_displacement(lib_plm.alm2map(plm * lib_plm.get_ikx() - olm * lib_plm.get_iky()),
                            lib_plm.alm2map(plm * lib_plm.get_iky() + olm * lib_plm.get_ikx()),
                            lib_plm.ell_mat.lsides, **kwargs)


class ffs_displacement(object):
    """
    Full flat sky displacement library. Typically divides things in chunks for the lensing operation
    and finding the inverse mapping.
    """

    def __init__(self, dx, dy, lsides, LD_res=(11, 11), verbose=True, spline_order=3, rule_for_derivative='4pts',
                 NR_iter=3, lib_dir=None, cache_magn=False):
        """
         dx and dy path to .npy array, x and y displacements. (displaced map(x) = map(x + d(x))
         Note that the first index is 'y' and the second 'x'
        """
        if not hasattr(dx, 'shape'): assert os.path.exists(dx), (pbs.rank, dx)
        if not hasattr(dy, 'shape'): assert os.path.exists(dy), (pbs.rank, dy)
        # dx, dy can be either the array of the path to the array.
        assert len(lsides) == 2
        self.dx = dx
        self.dy = dy

        self.verbose = verbose
        self.rule = rule_for_derivative  # rule for derivatives

        # Checking inputs :
        self.shape = self.get_dx().shape
        self.lsides = tuple(lsides)
        self.rmin = (1. * np.array(self.lsides)) / np.array(self.shape)

        HD_res = Log2ofPowerof2(self.shape)
        LD_res = LD_res or HD_res
        self.HD_res = (HD_res[0], HD_res[1])
        self.LD_res = (min(LD_res[0], HD_res[0]), min(LD_res[1], HD_res[1]))
        assert self.get_dx().shape == self.get_dy().shape
        assert len(self.LD_res) == 2 and (np.array(self.LD_res) <= np.array(self.HD_res)).all()

        # Buffer sizes and co :
        # Here buffer size 6 times the maximal displacement in grid units.
        # Might want to think about variable buffer size etc.

        buffer0 = np.int16(np.max([10, (6 * np.max(np.abs(self.get_dy())) / self.rmin[0])]))
        buffer1 = np.int16(np.max([10, (6 * np.max(np.abs(self.get_dx())) / self.rmin[1])]))

        self.buffers = (max(buffer0, buffer1) * (self.LD_res[0] < self.HD_res[0]),
                        max(buffer0, buffer1) * (self.LD_res[1] < self.HD_res[1]))
        self.chk_shape = 2 ** np.array(self.LD_res) + 2 * np.array(self.buffers)  # shape of the chunks
        self.N_chks = np.prod(2 ** (np.array(self.HD_res) - np.array(self.LD_res)))  # Number of chunks on each side.
        if verbose:
            print 'rank %s, ffs_deflect::buffers size, chk_shape' % pbs.rank, (buffer0, buffer1), self.chk_shape

        self.k = spline_order  # order of the spline for displacement interpolation

        self.NR_iter = NR_iter  # Number of NR iterations for inverse displacement.
        self.lib_dir = lib_dir
        self.cache_magn = cache_magn
        if self.lib_dir is not None:
            if not os.path.exists(self.lib_dir):
                try:
                    os.makedirs(self.lib_dir)
                except:
                    print "ffs_displacement:: unable to create lib. dir.", self.lib_dir

    def load_map(self, map):
        if isinstance(map, str):
            return np.load(map)
        else:
            return map

    def is_dxdy_ondisk(self):
        """ Checks whether the displacements are on disk or in memory. Bool."""
        return isinstance(self.dx, str) and isinstance(self.dy, str)

    def has_lib_dir(self):
        """ Just checks whether there is an libdir specified for this instance. Bool. """
        return self.lib_dir is not None

    def mk_args(self, path_to_map, path_to_dx, path_to_dy, NR_iter=None):
        if NR_iter is None: NR_iter = self.NR_iter
        return [path_to_map, path_to_dx, path_to_dy, self.buffers[0], self.buffers[1], self.lsides[0], self.lsides[1],
                self.HD_res[0], self.HD_res[1], NR_iter, self.k]

    def write_npy(self, fname):
        """
        Writes dx dy to disk and sets self.dx, self.dy to path names.
        """
        if self.verbose: print "write_npy:: rank %s caching " % pbs.rank, fname + '_dx.npy'
        try:
            np.save(fname + '_dx.npy', self.get_dx())
        except:
            assert 0, 'could not write %s' % (fname + '_dx.npy')
        if self.verbose: print "write_npy:: rank %s caching " % pbs.rank, fname + '_dy.npy'
        try:
            np.save(fname + '_dy.npy', self.get_dy())
        except:
            assert 0, 'could not write %s' % (fname + '_dy.npy')
        self.dx = fname + '_dx.npy'
        self.dy = fname + '_dy.npy'
        return

    def get_dx(self):
        if isinstance(self.dx, str):
            return np.load(self.dx)
        else:
            return self.dx

    def get_dy(self):
        if isinstance(self.dy, str):
            return np.load(self.dy)
        else:
            return self.dy

    def get_dnorm(self):
        return np.sqrt(self.get_dx() ** 2 + self.get_dy() ** 2)

    def get_dnorm_phi(self):
        """
        Norm of the displacement due to phi
        :return:
        """
        phi = self.get_phi()
        return np.sqrt(PDP(phi, 1, h=self.rmin[1]) ** 2
                       + PDP(phi, 0, h=self.rmin[0]) ** 2)

    def get_dnorm_Omega(self):
        """
        Norm of the displacement due to curl
        :return:
        """
        Omega = self.get_Omega()
        return np.sqrt(PDP(Omega, 1, h=self.rmin[1]) ** 2
                       + PDP(Omega, 0, h=self.rmin[0]) ** 2)

    def get_dx_ingridunits(self):
        return self.get_dx() / self.rmin[1]

    def get_dy_ingridunits(self):
        return self.get_dy() / self.rmin[0]

    def get_dxdy_chk_N(self, N, buffers=None):
        if buffers is None: buffers = self.buffers
        shape = (2 ** self.LD_res[0] + 2 * buffers[0], 2 ** self.LD_res[1] + 2 * buffers[1])
        dx, dy = np.zeros(shape), np.zeros(shape)
        spliter_lib = map_spliter.periodicmap_spliter()  # library to split periodic maps.
        sLDs, sHDs = spliter_lib.get_slices_chk_N(N, self.LD_res, self.HD_res, buffers)
        for sLD, sHD in zip(sLDs, sHDs):
            dx[sLD] = self.get_dx()[sHD]
            dy[sLD] = self.get_dy()[sHD]
        return dx, dy

    def apply(self, map, **kwargs):
        """ For compatibility purposes """
        return self.lens_map(map, **kwargs)

    def _get_e2iomega(self):
        return np.exp(-2j * self.get_omega())

    def lens_map_crude(self, map, crude):
        """
        Crudest lens operation, just rounding to nearest pixel.
        :param map:
        :return:
        """
        if crude == 1:
            # Plain interpolation to nearest pixel
            ly, lx = np.indices(self.shape)

            lx = np.int32(np.round((lx + self.get_dx_ingridunits()).flatten())) % self.shape[1]  # Periodicity
            ly = np.int32(np.round((ly + self.get_dy_ingridunits())).flatten()) % self.shape[0]  # Periodicity
            return self.load_map(map).flatten()[FlatIndices(np.array([ly, lx]), self.shape)].reshape(self.shape)
        elif crude == 2:
            # First order series expansion
            return self.load_map(map) \
                   + PDP(self.load_map(map), axis=0, h=self.rmin[0],
                         rule=self.rule) * self.get_dy() \
                   + PDP(self.load_map(map), axis=1, h=self.rmin[1],
                         rule=self.rule) * self.get_dx()
        else:
            assert 0, crude

    def lens_map(self, map, use_Pool=0, tidy=True, crude=0, do_not_prefilter=False):
        """
        Lens the input map according to the displacement fields dx dy. 'map' typically could be (8192 * 8192) np array,
        or the path to the array on disk.

        Does this by splitting the job in chunks (of typically (256 * 256), as specified by the LD_res parameters)
        allowing a buffer size to ensure the junctions are properly performed.

        Set use_Pool to a power of two to use explicit threading via the multiprocessing module, or, if < 0,
        to perform the operation on the GPU.
        if > 0 'use_Pool' ** 2 is the number of threads. On laptop and Darwin use_Pool = 16 has the best performances.
        It use_Pool is set, then 'map' must be the path to the map to lens or map will be saved to disk.
        """
        # TODO : could evaluate the splines at low res.
        assert self.load_map(map).shape == self.shape, (self.load_map(map).shape, self.shape)
        if crude > 0:
            return self.lens_map_crude(map, crude)
        if use_Pool < 0:
            # use of GPU :
            try:
                from lensit.gpu import lens_GPU
            except:
                assert 0, 'Import of mllens lens_GPU failed !'

            GPU_res = np.array(lens_GPU.GPU_HDres_max)
            if np.all(np.array(self.HD_res) <= GPU_res):
                return lens_GPU.lens_onGPU(map, self.get_dx_ingridunits(), self.get_dy_ingridunits(),
                                           do_not_prefilter=do_not_prefilter)
            LD_res, buffers = get_GPUbuffers(GPU_res)
            assert np.all(np.array(buffers) > (np.array(self.buffers) + 5.)), (buffers, self.buffers)
            Nchunks = 2 ** (np.sum(np.array(self.HD_res) - np.array(LD_res)))
            lensed_map = np.empty(self.shape)  # Output
            dx_N = np.empty((2 ** LD_res[0] + 2 * buffers[0], 2 ** LD_res[1] + 2 * buffers[1]))
            dy_N = np.empty((2 ** LD_res[0] + 2 * buffers[0], 2 ** LD_res[1] + 2 * buffers[1]))
            unl_CMBN = np.empty((2 ** LD_res[0] + 2 * buffers[0], 2 ** LD_res[1] + 2 * buffers[1]))
            if self.verbose:
                print '++ lensing map :' \
                      '   splitting map on GPU , chunk shape %s, buffers %s' % (dx_N.shape, buffers)
            spliter_lib = map_spliter.periodicmap_spliter()  # library to split periodic maps.
            for N in xrange(Nchunks):
                sLDs, sHDs = spliter_lib.get_slices_chk_N(N, LD_res, self.HD_res, buffers)
                for sLD, sHD in zip(sLDs, sHDs):
                    dx_N[sLD] = self.get_dx()[sHD] / self.rmin[1]
                    dy_N[sLD] = self.get_dy()[sHD] / self.rmin[0]
                    unl_CMBN[sLD] = self.load_map(map)[sHD]
                sLDs, sHDs = spliter_lib.get_slices_chk_N(N, LD_res, self.HD_res, buffers, inverse=True)
                lensed_map[sHDs[0]] = lens_GPU.lens_onGPU(unl_CMBN, dx_N, dy_N, do_not_prefilter=do_not_prefilter)[
                    sLDs[0]]
            return lensed_map

        elif use_Pool > 100:
            if not isinstance(map, str):
                assert self.has_lib_dir(), "Specify lib. dir. if you want to use Pool."
                np.save(self.lib_dir + '/temp_maptolens_rank%s.npy' % pbs.rank, map)
            if not self.is_dxdy_ondisk():
                assert self.has_lib_dir(), "Specify lib. dir. if you want to use Pool."
                print "lens_map::writing displacements on disk :"
                self.write_npy(self.lib_dir + '/temp_displ' + str(pbs.rank))  # this turns dx and dy to the paths
            path_to_map = map if isinstance(map, str) else self.lib_dir + '/temp_maptolens_rank%s.npy' % pbs.rank
            ret = ffs_pool.get_lens_Pooled(self.mk_args(path_to_map, self.dx, self.dy),
                                           root_Nthreads=use_Pool % 100, do_not_prefilter=do_not_prefilter)
            if tidy:
                if os.path.exists(self.lib_dir + '/temp_maptolens_rank%s.npy' % pbs.rank):
                    os.remove(self.lib_dir + '/temp_maptolens_rank%s.npy' % pbs.rank)
            return ret

        elif use_Pool == 100 or use_Pool == 101:
            assert self.load_map(map).shape == self.shape, self.load_map(map).shape
            s = self.chk_shape
            idc0, idc1 = np.indices(s)  # Two (256 * 256) maps
            dx_gu = np.empty(s)  # will dx displ. in grid units of each chunk (typ. (256 * 256) )
            dy_gu = np.empty(s)  # will dy displ. in grid units of each chunk (typ. (256 * 256) )
            map_chk = np.empty(s)  # Will be map chunk
            # (typ. (256 * 256) )
            lensed_map = np.empty(self.shape)
            spliter_lib = map_spliter.periodicmap_spliter()  # library to split periodic maps.
            for N in xrange(self.N_chks):
                # doing chunk N
                sLDs, sHDs = spliter_lib.get_slices_chk_N(N, self.LD_res, self.HD_res, self.buffers)
                for sLD, sHD in zip(sLDs, sHDs):
                    # Displacements chunk in grid units, and map chunk to displace.
                    dx_gu[sLD] = self.get_dx()[sHD] / self.rmin[1]
                    dy_gu[sLD] = self.get_dy()[sHD] / self.rmin[0]
                    map_chk[sLD] = self.load_map(map)[sHD]

                if do_not_prefilter:
                    # Undoing the prefiltering prior to apply bicubic interpolation
                    map_chk = np.fft.rfft2(map_chk)
                    w0 = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(s[0])) + 4.)
                    map_chk /= np.outer(w0, w0[0:map_chk.shape[1]])
                    map_chk = np.fft.irfft2(map_chk, s)

                lx = (idc1 + dx_gu).flatten()  # No need to enforce periodicity here.
                ly = (idc0 + dy_gu).flatten()  # No need to enforce periodicity here.
                sLDs, sHDs = spliter_lib.get_slices_chk_N(N, self.LD_res, self.HD_res, self.buffers, inverse=True)
                lensed_map[sHDs[0]] = interpolate.RectBivariateSpline(np.arange(s[0]), np.arange(s[1]),
                                                                      map_chk, kx=self.k, ky=self.k).ev(ly, lx).reshape(
                    self.chk_shape)[sLDs[0]]
            return lensed_map
        elif use_Pool == 0 or use_Pool == 1:
            assert self.shape[0] == self.shape[1], self.shape
            bicubicspline = r"\
                           int i,j;\
                          for( j= 0; j < width; j++ )\
                              {\
                              for( i = 0; i < width; i++)\
                                  {\
                                  lenmap[j * width + i] = bicubiclensKernel(filtmap,i + dx_gu[j * width + i],j + dy_gu[j * width + i],width);\
                                  }\
                              }"
            header = r' "%s/lensit/gpu/bicubicspline.h" ' % li.LENSITDIR
            if do_not_prefilter:
                filtmap = self.load_map(map).astype(np.float64)
            else:
                # TODO : may want to add pyFFTW here as well
                filtmap = np.fft.rfft2(self.load_map(map))
                w0 = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(filtmap.shape[0])) + 4.)
                filtmap *= np.outer(w0, w0[0:filtmap.shape[1]])
                filtmap = np.fft.irfft2(filtmap, self.shape)

            lenmap = np.empty(self.shape, dtype=np.float64)
            dx_gu = self.get_dx_ingridunits().astype(np.float64)
            dy_gu = self.get_dy_ingridunits().astype(np.float64)
            width = int(self.shape[0])
            assert self.shape[0] == self.shape[1]
            weave.inline(bicubicspline, ['lenmap', 'filtmap', 'dx_gu', 'dy_gu', 'width'], headers=[header])
            return lenmap

        elif use_Pool > 1 and use_Pool < 100:
            # TODO : may want to add pyFFTW here as well
            if not isinstance(map, str):
                assert self.has_lib_dir(), "Specify lib. dir. if you want to use Pool."
                np.save(self.lib_dir + '/temp_maptolens_rank%s.npy' % pbs.rank, map)
            if not self.is_dxdy_ondisk():
                assert self.has_lib_dir(), "Specify lib. dir. if you want to use Pool."
                print "lens_map::writing displacements on disk :"
                self.write_npy(self.lib_dir + '/temp_displ' + str(pbs.rank))  # this turns dx and dy to the paths
            path_to_map = map if isinstance(map, str) else self.lib_dir + '/temp_maptolens_rank%s.npy' % pbs.rank
            ret = ffs_pool.get_lens_Pooled_weave(self.mk_args(path_to_map, self.dx, self.dy),
                                                 root_Nthreads=use_Pool, do_not_prefilter=do_not_prefilter)
            if tidy:
                if os.path.exists(self.lib_dir + '/temp_maptolens_rank%s.npy' % pbs.rank):
                    os.remove(self.lib_dir + '/temp_maptolens_rank%s.npy' % pbs.rank)
            return ret

        else:
            assert 0

    def lens_alm(self, lib_alm, alm,
                 lib_alm_out=None, use_Pool=0, no_lensing=False, mult_magn=False):
        """
        Turn alms into the alms of the lensed map. If mult_magn is set, multiplies the lensed map by the determinant
        of the magnification matrix prior to take the harmonic transform.
        """
        if no_lensing and lib_alm_out is None: return alm
        if lib_alm_out is None: lib_alm_out = lib_alm
        if no_lensing: return lib_alm_out.udgrade(lib_alm, alm)
        if use_Pool < 0:  # can we fit the full map on the GPU ?
            from lensit.gpu import lens_GPU
            GPU_res = np.array(lens_GPU.GPU_HDres_max)
            if np.all(np.array(self.HD_res) <= GPU_res):
                return lens_GPU.lens_alm_onGPU(lib_alm, lib_alm.bicubic_prefilter(alm),
                                               self.get_dx_ingridunits(), self.get_dy_ingridunits()
                                               , do_not_prefilter=True, mult_magn=mult_magn, lib_alm_out=lib_alm_out)
        temp_map = self.alm2lenmap(lib_alm, alm, use_Pool=use_Pool)
        if mult_magn:
            self.mult_wmagn(temp_map, inplace=True)
        return lib_alm_out.map2alm(temp_map)

    def mult_wmagn(self, _map, inplace=False):
        if not inplace:
            return self.get_det_magn() * _map
        else:
            _map *= self.get_det_magn()
            return

    def alm2lenmap(self, lib_alm, alm, use_Pool=0, no_lensing=False):
        """
        Turn alm into the lensed map.
        """
        assert alm.shape == (lib_alm.alm_size,), (alm.shape, lib_alm.alm_size)
        assert lib_alm.ell_mat.shape == self.shape, (lib_alm.ell_mat.shape, self.shape)
        if no_lensing:
            return lib_alm.alm2map(alm)
        if use_Pool < 0:  # can we fit the full map on the GPU ? If we can't we send it the lens_map
            from lensit.gpu import lens_GPU
            GPU_res = np.array(lens_GPU.GPU_HDres_max)
            if np.all(np.array(self.HD_res) <= GPU_res):
                return lens_GPU.alm2lenmap_onGPU(lib_alm, lib_alm.bicubic_prefilter(alm),
                                                 self.get_dx_ingridunits(), self.get_dy_ingridunits(),
                                                 do_not_prefilter=True)

        if use_Pool >= 100:  # scipy RectBivariate. Slow.
            unlmap = lib_alm.alm2map(alm)
            return self.lens_map(unlmap, use_Pool=use_Pool, do_not_prefilter=False)
        else:
            return self.lens_map(lib_alm.alm2map(lib_alm.bicubic_prefilter(alm)),
                                 use_Pool=use_Pool, do_not_prefilter=True)

    def get_det_magn(self):
        """
        Returns entire magnification det map.
        """
        # FIXME : bad
        if not self.cache_magn:
            det = (PDP(self.get_dx(), axis=1, h=self.rmin[1], rule=self.rule) + 1.) \
                  * (PDP(self.get_dy(), axis=0, h=self.rmin[0], rule=self.rule) + 1.)
            det -= PDP(self.get_dy(), axis=1, h=self.rmin[1], rule=self.rule) * \
                   PDP(self.get_dx(), axis=0, h=self.rmin[0], rule=self.rule)
            return det
        else:
            assert self.has_lib_dir(), 'Specify lib_dir if you want to cache magn'
            fname = self.lib_dir + '/det_magn_%s_%s_rank%s.npy' % \
                                   (hashlib.sha1(self.get_dx()[0, 0:100]).hexdigest(),
                                    hashlib.sha1(self.get_dy()[0, 0:100]).hexdigest(), pbs.rank)
            if not os.path.exists(fname):  # and pbs.rank == 0:
                det = (PDP(self.get_dx(), axis=1, h=self.rmin[1], rule=self.rule) + 1.) \
                      * (PDP(self.get_dy(), axis=0, h=self.rmin[0], rule=self.rule) + 1.)
                det -= PDP(self.get_dy(), axis=1, h=self.rmin[1], rule=self.rule) * \
                       PDP(self.get_dx(), axis=0, h=self.rmin[0], rule=self.rule)
                print "  ffs_displacement caching ", fname
                np.save(fname, det)
                del det
            # pbs.barrier()
            return np.load(fname)

    def pOlm(self, lib_qlm):
        pass

    def get_kappa(self):
        """
        kappa map. kappa is -1/2 del phi
        :return:
        """
        dfxdx = PDP(self.get_dx(), axis=1, h=self.rmin[1], rule=self.rule)
        dfydy = PDP(self.get_dy(), axis=0, h=self.rmin[0], rule=self.rule)
        return -0.5 * (dfxdx + dfydy)

    def get_omega(self):
        """
        curl kappa map
        :return:
        """
        dfxdy = PDP(self.get_dx(), axis=0, h=self.rmin[0], rule=self.rule)
        dfydx = PDP(self.get_dy(), axis=1, h=self.rmin[1], rule=self.rule)
        return 0.5 * (dfxdy - dfydx)

    def get_phi(self):
        """
        -1/2 Laplac phi = kappa
        :return:
        """
        rfft_phi = np.fft.rfft2(self.get_kappa())
        rs = rfft_phi.shape
        ky = (2. * np.pi) / self.lsides[0] * Freq(np.arange(self.shape[0]), self.shape[0])
        ky[self.shape[0] / 2:] *= -1.
        kx = (2. * np.pi) / self.lsides[1] * Freq(np.arange(rs[1]), self.shape[1])
        rfft_phi = rfft_phi.flatten()
        rfft_phi[1:] /= (np.outer(ky ** 2, np.ones(rs[1])) + np.outer(np.ones(rs[0]), kx ** 2)).flatten()[1:]
        return np.fft.irfft2(2 * rfft_phi.reshape(rs), self.shape)

    def get_Omega(self):
        """
        -1/2 Laplac Omega = omega
        :return:
        """
        rfft_Om = np.fft.rfft2(self.get_omega())
        rs = rfft_Om.shape
        ky = (2. * np.pi) / self.lsides[0] * Freq(np.arange(self.shape[0]), self.shape[0])
        ky[self.shape[0] / 2:] *= -1.
        kx = (2. * np.pi) / self.lsides[1] * Freq(np.arange(rs[1]), self.shape[1])
        rfft_Om = rfft_Om.flatten()
        rfft_Om[1:] /= (np.outer(ky ** 2, np.ones(rs[1])) + np.outer(np.ones(rs[0]), kx ** 2)).flatten()[1:]
        return np.fft.irfft2(2 * rfft_Om.reshape(rs), self.shape)

    def get_det_magn_chk_N(self, N, extra_buff=np.array((5, 5))):
        """
        Returns chunk N of magnification map, adding an extra buffer 'extra_buff' to avoid
        surprises with the periodic derivatives.
        """
        dx, dy = self.get_dxdy_chk_N(N, buffers=self.buffers + extra_buff)  # In physical units
        det = (PDP(dx, axis=1, h=self.rmin[1], rule=self.rule) + 1.) \
              * (PDP(dy, axis=0, h=self.rmin[0], rule=self.rule) + 1.)
        det -= PDP(dy, axis=1, h=self.rmin[1], rule=self.rule) * \
               PDP(dx, axis=0, h=self.rmin[0], rule=self.rule)
        return det[extra_buff[0]:dx.shape[0] - extra_buff[0], extra_buff[1]:dx.shape[1] - extra_buff[1]]

    def get_magn_mat_chk_N(self, N, extra_buff=np.array((5, 5))):
        """
        Returns chk number 'N' of magnification matrix, adding an extra-buffer to avoid surprises with
        periodic derivatives.
        """
        # Setting up dx and dy displacement chunks :
        # We add small extra buffer to get correct derivatives in the chunks
        dx, dy = self.get_dxdy_chk_N(N, buffers=self.buffers + extra_buff)  # In physical units
        # Jacobian matrix :
        sl0 = slice(extra_buff[0], dx.shape[0] - extra_buff[0])
        sl1 = slice(extra_buff[1], dx.shape[1] - extra_buff[1])
        dfxdx = PDP(dx, axis=1, h=self.rmin[1], rule=self.rule)[sl0, sl1]
        dfydx = PDP(dy, axis=1, h=self.rmin[1], rule=self.rule)[sl0, sl1]
        dfxdy = PDP(dx, axis=0, h=self.rmin[0], rule=self.rule)[sl0, sl1]
        dfydy = PDP(dy, axis=0, h=self.rmin[0], rule=self.rule)[sl0, sl1]
        return {'kappa': 0.5 * (dfxdx + dfydy), 'omega': 0.5 * (dfxdy - dfydx), \
                'gamma_U': 0.5 * (dfxdy + dfydx), 'gamma_Q': 0.5 * (dfxdx - dfydy), \
                'det': (dfxdx + 1.) * (dfydy + 1.) - dfydx * dfxdy}

    def get_inverse_crude(self, crude):
        """
        Crude inversions of the displacement field
        :param crude:
        :return:
        """
        assert crude in [1], crude
        if crude == 1:
            return ffs_displacement(- self.get_dx(), -self.get_dy(), self.lsides, lib_dir=self.lib_dir,
                                    LD_res=self.LD_res, verbose=self.verbose, spline_order=self.k, NR_iter=self.NR_iter)
        else:
            assert 0, crude

    def get_inverse(self, NR_iter=None, use_Pool=0, crude=0, HD_res=None):
        """

        :param NR_iter:
        :param use_Pool: if positive, use Python multiprocessing Pool packacge.
                         if 0, serial calculation
                         if negative, send it on the GPU.
        :param crude: Uses some crude scheme
        :param HD_res: augmente the resolution to perform the inversion.
        :return:
        """
        if HD_res is not None:
            # FIXME : this upgrade can be done in GPU for use_Pool < 0
            lib_dir = self.lib_dir if self.lib_dir is None else self.lib_dir + '/temp_fup'
            f_up = ffs_displacement(rfft2_utils.upgrade_map(self.get_dx(), HD_res),
                                    rfft2_utils.upgrade_map(self.get_dy(), HD_res), self.lsides,
                                    LD_res=self.LD_res, verbose=self.verbose, spline_order=self.k,
                                    rule_for_derivative=self.rule,
                                    NR_iter=self.NR_iter, lib_dir=lib_dir)
            f_up_inv = f_up.get_inverse(NR_iter=NR_iter, use_Pool=use_Pool, crude=crude)
            LD_res = Log2ofPowerof2(self.shape)
            return ffs_displacement(rfft2_utils.subsample(f_up_inv.get_dx(), LD_res),
                                    rfft2_utils.subsample(f_up_inv.get_dy(), LD_res), self.lsides,
                                    LD_res=self.LD_res, verbose=self.verbose, spline_order=self.k,
                                    rule_for_derivative=self.rule,
                                    NR_iter=self.NR_iter, lib_dir=self.lib_dir)

        if crude > 0:
            return self.get_inverse_crude(crude)

        if NR_iter is None: NR_iter = self.NR_iter

        if use_Pool > 0:
            if not self.is_dxdy_ondisk():
                assert self.has_lib_dir(), "Specify lib. dir. if you want to use Pool."
                print "lens_map::writing displacements on disk :"
                if not os.path.exists(self.lib_dir): os.makedirs(self.lib_dir)
                self.write_npy(self.lib_dir + '/temp_displ' + str(pbs.rank))  # this turns dx and dy to the paths
            dx_inv, dy_inv = ffs_pool.get_inverse_Pooled(self.mk_args('', self.dx, self.dy, NR_iter=NR_iter),
                                                         root_Nthreads=abs(use_Pool))
            return ffs_displacement(dx_inv, dy_inv, self.lsides, lib_dir=self.lib_dir,
                                    LD_res=self.LD_res, verbose=self.verbose, spline_order=self.k, NR_iter=self.NR_iter)

        elif use_Pool == 0:
            spliter_lib = map_spliter.periodicmap_spliter()  # library to split periodic maps.
            dx_inv, dy_inv = np.empty(self.shape), np.empty(self.shape)
            label = 'ffs_deflect::calculating inverse displ. field'
            for i, N in utils.enumerate_progress(xrange(self.N_chks), label=label):
                # Doing chunk N
                dx_inv_N, dy_inv_N = self.get_inverse_chk_N(N, NR_iter=NR_iter)
                sLDs, sHDs = spliter_lib.get_slices_chk_N(N, self.LD_res, self.HD_res, self.buffers, inverse=True)
                # Pasting it onto the full map
                dx_inv[sHDs[0]] = dx_inv_N[sLDs[0]]
                dy_inv[sHDs[0]] = dy_inv_N[sLDs[0]]
            return ffs_displacement(dx_inv, dy_inv, self.lsides, lib_dir=self.lib_dir,
                                    LD_res=self.LD_res, verbose=self.verbose, spline_order=self.k, NR_iter=self.NR_iter)
        elif use_Pool < 0:
            # GPU calculation.
            from lensit.gpu import inverse_GPU as inverse_GPU
            GPU_res = np.array(inverse_GPU.GPU_HDres_max)
            if np.all(np.array(self.HD_res) <= GPU_res):
                # No need to split maps :
                dx_inv, dy_inv = inverse_GPU.inverse_GPU(self.get_dx(), self.get_dy(), self.rmin, NR_iter)
                return ffs_displacement(dx_inv, dy_inv, self.lsides, lib_dir=self.lib_dir,
                                        LD_res=self.LD_res, verbose=self.verbose, spline_order=self.k,
                                        NR_iter=self.NR_iter)
            else:
                LD_res, buffers = get_GPUbuffers(GPU_res)
                assert np.all(np.array(buffers) > (np.array(self.buffers) + 5.)), (buffers, self.buffers)
                Nchunks = 2 ** (np.sum(np.array(self.HD_res) - np.array(LD_res)))
                dx_N = np.empty((2 ** LD_res[0] + 2 * buffers[0], 2 ** LD_res[1] + 2 * buffers[1]))
                dy_N = np.empty((2 ** LD_res[0] + 2 * buffers[0], 2 ** LD_res[1] + 2 * buffers[1]))
                if self.verbose:
                    print '++ inverse displacement :' \
                          '   splitting inverse on GPU , chunk shape %s, buffers %s' % (dx_N.shape, buffers)
                spliter_lib = map_spliter.periodicmap_spliter()  # library to split periodic maps.
                dx_inv, dy_inv = np.empty(self.shape), np.empty(self.shape)  # Outputs
                for N in xrange(Nchunks):
                    sLDs, sHDs = spliter_lib.get_slices_chk_N(N, LD_res, self.HD_res, buffers)
                    for sLD, sHD in zip(sLDs, sHDs):
                        dx_N[sLD] = self.get_dx()[sHD]
                        dy_N[sLD] = self.get_dy()[sHD]
                    dx_inv_N, dy_inv_N = inverse_GPU.inverse_GPU(dx_N, dy_N, self.rmin, NR_iter)
                    sLDs, sHDs = spliter_lib.get_slices_chk_N(N, LD_res, self.HD_res, buffers, inverse=True)
                    dx_inv[sHDs[0]] = dx_inv_N[sLDs[0]]
                    dy_inv[sHDs[0]] = dy_inv_N[sLDs[0]]

                return ffs_displacement(dx_inv, dy_inv, self.lsides, lib_dir=self.lib_dir,
                                        LD_res=self.LD_res, verbose=self.verbose, spline_order=self.k,
                                        NR_iter=self.NR_iter)
        elif use_Pool == 100:
            assert 0
        else:
            assert 0

    def get_inverse_chk_N(self, N, NR_iter=None):
        """
        Returns inverse displacement in chunk N
        Uses periodic boundary conditions, which is not applicable to chunks, thus there
        will be boudary effects on the edges (2 or 4 pixels depending on the rule). Make sure the buffer is large enough.
        """
        if NR_iter is None: NR_iter = self.NR_iter

        # Inverse magn. elements. (with a minus sign) We may need to spline these later for further NR iterations :
        extra_buff = np.array((5, 5)) * (
            np.array(self.chk_shape) != np.array(self.shape))  # To avoid surprises with the periodic derivatives
        dx = np.zeros(self.chk_shape + 2 * extra_buff)  # will dx displ. in grid units of each chunk (typ. (256 * 256) )
        dy = np.zeros(self.chk_shape + 2 * extra_buff)  # will dy displ. in grid units of each chunk (typ. (256 * 256) )
        sLDs, sHDs = map_spliter.periodicmap_spliter().get_slices_chk_N(N, self.LD_res, self.HD_res,
                                                                        (self.buffers[0] + extra_buff[0],
                                                                         self.buffers[1] + extra_buff[1]))

        rmin0 = self.lsides[0] / self.shape[0]
        rmin1 = self.lsides[1] / self.shape[1]

        for sLD, sHD in zip(sLDs, sHDs):
            dx[sLD] = self.get_dx()[sHD] / rmin1  # Need grid units displacement for the bicubic spline
            dy[sLD] = self.get_dy()[sHD] / rmin0
        # Jacobian matrix of the chunk :
        sl0 = slice(extra_buff[0], dx.shape[0] - extra_buff[0])
        sl1 = slice(extra_buff[1], dx.shape[1] - extra_buff[1])

        Minv_yy = - (PDP(dx, axis=1)[sl0, sl1] + 1.)
        Minv_xx = - (PDP(dy, axis=0)[sl0, sl1] + 1.)
        Minv_xy = PDP(dy, axis=1)[sl0, sl1]
        Minv_yx = PDP(dx, axis=0)[sl0, sl1]
        dx = dx[sl0, sl1]
        dy = dy[sl0, sl1]

        det = Minv_yy * Minv_xx - Minv_xy * Minv_yx
        if not np.all(det > 0.): print "ffs_displ::Negative value in det k : something's weird, you'd better check that"
        # Inverse magn. elements. (with a minus sign) We may need to spline these later for further NR iterations :
        Minv_xx /= det
        Minv_yy /= det
        Minv_xy /= det
        Minv_yx /= det
        del det
        ex = (Minv_xx * dx + Minv_xy * dy)
        ey = (Minv_yx * dx + Minv_yy * dy)

        if NR_iter == 0: return ex * rmin1, ey * rmin0

        # Setting up a bunch of splines to interpolate the increment to the displacement according to Newton-Raphson.
        # Needed are splines of the forward displacement and of the (inverse, as implemented here) magnification matrix.
        # Hopefully the map resolution is enough to spline the magnification matrix.
        s0, s1 = self.chk_shape
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

        header = r' "%s/lensit/gpu/bicubicspline.h" ' % li.LENSITDIR
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

    def get_inverse_chk_N_old(self, N, NR_iter=None):
        """
        Old version with scipy.interpolate
        Returns inverse displacement in chunk N
        Uses periodic boundary conditions, which is not applicable to chunks, thus there
        will be boudary effects on the edges (2 or 4 pixels depending on the rule). Make sure the buffer is large enough.
        """
        if NR_iter is None: NR_iter = self.NR_iter

        s = self.chk_shape
        M_mat = self.get_magn_mat_chk_N(N)
        if not np.all(M_mat['det'] > 0.):
            print "ffs_displ::Negative value in det k : something's weird, you'd better check that"

        # Inverse magn. elements. (with a minus sign) We may need to spline these later for further NR iterations :

        _Minv_xx = -(1. + M_mat['kappa'] - M_mat['gamma_Q']) / M_mat['det']
        _Minv_yy = -(1. + M_mat['kappa'] + M_mat['gamma_Q']) / M_mat['det']
        _Minv_xy = (M_mat['gamma_U'] + M_mat['omega']) / M_mat['det']
        _Minv_yx = (M_mat['gamma_U'] - M_mat['omega']) / M_mat['det']
        del M_mat

        # First NR step. This one is easy :
        dx, dy = self.get_dxdy_chk_N(N)
        dxn = (_Minv_xx * dx + _Minv_xy * dy)
        dyn = (_Minv_yx * dx + _Minv_yy * dy)
        if NR_iter == 0: return [dxn, dyn]

        # Setting up a bunch of splines to interpolate the increment to the displacement according to Newton-Raphson.
        # Needed are splines of the forward displacement and of the (inverse, as implemented here) magnification matrix.
        # Hopefully the map resolution is enough to spline the magnification matrix.

        xcoord = np.arange(s[1]) * self.rmin[1]
        ycoord = np.arange(s[0]) * self.rmin[0]

        spl_dx = interpolate.RectBivariateSpline(ycoord, xcoord, dx, kx=self.k, ky=self.k)
        spl_dy = interpolate.RectBivariateSpline(ycoord, xcoord, dy, kx=self.k, ky=self.k)

        spl_xx = interpolate.RectBivariateSpline(ycoord, xcoord, _Minv_xx, kx=self.k, ky=self.k)
        spl_yy = interpolate.RectBivariateSpline(ycoord, xcoord, _Minv_yy, kx=self.k, ky=self.k)
        spl_xy = interpolate.RectBivariateSpline(ycoord, xcoord, _Minv_xy, kx=self.k, ky=self.k)
        spl_yx = interpolate.RectBivariateSpline(ycoord, xcoord, _Minv_yx, kx=self.k, ky=self.k)

        idc = np.indices(s)
        y_x = idc[1] * self.rmin[1]
        y_y = idc[0] * self.rmin[0]
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
        return [dxn, dyn]

    def degrade(self, LD_shape, no_lensing, **kwargs):
        if no_lensing: return ffs_id_displacement(LD_shape, self.lsides)
        dx = rfft2_utils.degrade(self.get_dx(), LD_shape)
        dy = rfft2_utils.degrade(self.get_dy(), LD_shape)
        return ffs_displacement(dx, dy, self.lsides, **kwargs)

    def get_MF(self, lib_qlm):
        """
        :return: The (N0-like) unnormalised mean field for noisefree deflection estimates
        """
        alm_magn = lib_qlm.map2alm(1. / self.get_det_magn())
        alm_d0 = lib_qlm.map2alm(self.get_dy())
        alm_d1 = lib_qlm.map2alm(self.get_dx())
        fac = 1./ np.sqrt(np.prod(self.lsides))
        ik1 = lambda: lib_qlm.get_ikx()
        ik0 = lambda: lib_qlm.get_iky()
        g0 = (1. + lib_qlm.alm2map(ik1() * alm_d1)) * (lib_qlm.alm2map(ik0() * alm_magn)) \
             - lib_qlm.alm2map(ik0() * alm_d1) * lib_qlm.alm2map(ik0() * alm_magn)
        g1 = (1. + lib_qlm.alm2map(ik0() * alm_d0)) * (lib_qlm.alm2map(ik1() * alm_magn)) \
             - lib_qlm.alm2map(ik1() * alm_d0) * lib_qlm.alm2map(ik1() * alm_magn)

        del alm_magn, alm_d0, alm_d1
        # Rotates to phi Omega :
        # 2 * sqrt(V) / N,  rfft2alm factor. times add. factors fac
        dy_ell, dx_ell = (fac * lib_qlm.map2alm(g) for g in [g0, g1])
        dphi_ell = dx_ell * ik1() + dy_ell * ik0()
        dOm_ell = - dx_ell * ik0() + dy_ell * ik1()
        return np.array([dphi_ell, dOm_ell])


class ffs_id_displacement():
    """ Displacement instance where there is actually no displacement. For sheer convenience """

    def __init__(self, shape, lsides):
        self.shape = shape
        self.lsides = lsides

    def degrade(self, LDshape, *args, **kwargs):
        return ffs_id_displacement(LDshape, self.lsides)

    def get_inverse(self, **kwargs):
        return ffs_id_displacement(self.shape, self.lsides)

    def apply(self, map, **kwargs):
        return self.lens_map(map, **kwargs)

    def get_dx(self):
        return np.zeros(self.shape, dtype=float)

    def get_dy(self):
        return np.zeros(self.shape, dtype=float)

    def get_dx_ingridunits(self):
        return np.zeros(self.shape, dtype=float)

    def get_dy_ingridunits(self):
        return np.zeros(self.shape, dtype=float)

    def lens_map(self, map, **kwargs):
        if isinstance(map, str):
            return np.load(map)
        else:
            return map

    def clone(self):
        return ffs_id_displacement(self.shape, self.lsides)

    def mult_wmagn(self, _map, inplace=False):
        if inplace:
            return
        else:
            return _map

    def lens_alm(self, lib_alm, alm, lib_alm_out=None, **kwargs):
        if lib_alm_out is not None:
            return lib_alm_out.udgrade(lib_alm, alm)
        return alm

    def alm2lenmap(self, lib_alm, alm, **kwargs):
        return lib_alm.alm2map(alm)

    def get_det_magn(self):
        return np.ones(self.shape, dtype=float)

    def rotpol(self, QpiU, **kwargs):
        assert np.iscomplexobj(QpiU) and QpiU.shape == self.shape, (QpiU.shape, np.iscomplexobj(QpiU))
        return QpiU
