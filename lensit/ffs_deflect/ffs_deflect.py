from __future__ import print_function

import hashlib
import os
import numpy as np

try:
    from lensit.bicubic import bicubic
except ImportError:
    print("***could not import bicubic fortran module")
    print("***I wont be able to lens maps or invert a deflection field")
    bicubic = 'could not import bicubic fortran module'

from lensit.misc import map_spliter
from lensit.misc import misc_utils as utils
from lensit.misc import rfft2_utils
from lensit.misc.misc_utils import PartialDerivativePeriodic as PDP, Log2ofPowerof2, Freq, flatindices
from lensit.pbs import pbs


class ffs_displacement(object):
    r"""Flat-sky deflection-field class

        Used to perform lensing on maps and to obtain the inverse deflection for iterative lensing estimation

        Args:
            dx: deflection field, x-component :math:`\alpha_x` (2d-array or path to array on disk)
            dy: deflection field, y-component :math:`\alpha_y` (2d-array or path to array on disk)
            lsides(tuple): physical size in radians of the flat-sky patch.
            LD_res(optional): to perform inversion or lensing large maps are split into chunks sized these powers of two
            verbose(optional): various prints out
            NR_iter(optional): Number of Newton-Raphson iterations in deflection inversion.
                                Default works very well for LCDM-like deflection fields
            cache_magn(optional): optionally caches magnification determinant matrix when needed
            lib_dir(optional): required only if cache_magn is set


    """

    def __init__(self, dx, dy, lsides, LD_res=(11, 11), verbose=False, NR_iter=3, lib_dir=None, cache_magn=False):
        """
         dx and dy arrays or path to .npy arrays, x and y displacements. (displaced map(x) = map(x + d(x))
         Note that the first index is 'y' and the second 'x'
        """
        if not hasattr(dx, 'shape'): assert os.path.exists(dx), (pbs.rank, dx)
        if not hasattr(dy, 'shape'): assert os.path.exists(dy), (pbs.rank, dy)
        # dx, dy can be either the array of the path to the array.
        assert len(lsides) == 2
        self.dx = dx
        self.dy = dy

        self.verbose = verbose
        self.rule = '4pts'  # rule for derivatives

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
        self.N_chks = int(np.prod(2 ** (np.array(self.HD_res) - np.array(self.LD_res))))  # Number of chunks on each side.
        if verbose:
            print('rank %s, ffs_deflect::buffers size, chk_shape' % pbs.rank, (buffer0, buffer1), self.chk_shape)

        self.NR_iter = NR_iter  # Number of NR iterations for inverse displacement.
        self.lib_dir = lib_dir
        self.cache_magn = cache_magn
        if self.lib_dir is not None:
            if not os.path.exists(self.lib_dir):
                try:
                    os.makedirs(self.lib_dir)
                except:
                    print("ffs_displacement:: unable to create lib. dir. " + self.lib_dir)

    def copy(self):
        return ffs_displacement(self.dx.copy(), self.dy.copy(), self.lsides, LD_res=self.LD_res, NR_iter=self.NR_iter, lib_dir=None, cache_magn=False)

    @staticmethod
    def load_map(m):
        if isinstance(m, str):
            return np.load(m)
        else:
            return m

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

    def get_dx_ingridunits(self):
        return self.get_dx() / self.rmin[1]

    def get_dy_ingridunits(self):
        return self.get_dy() / self.rmin[0]

    def lens_map_crude(self, m, crude):
        """Performs crude approximations to lens operation

            Args:
                m: map to lens (2d-array of the right shape)
                crude: approximation method key

            Now supported are *crude* = 1 (nearest pixel rouding) or 2 (first order series expansion in deflection)

        """
        if crude == 1:
            # Plain interpolation to nearest pixel
            ly, lx = np.indices(self.shape)

            lx = np.int32(np.round((lx + self.get_dx_ingridunits()).flatten())) % self.shape[1]  # Periodicity
            ly = np.int32(np.round((ly + self.get_dy_ingridunits())).flatten()) % self.shape[0]
            return self.load_map(m).flatten()[flatindices(np.array([ly, lx]), self.shape)].reshape(self.shape)
        elif crude == 2:
            # First order series expansion
            return self.load_map(m) \
                   + PDP(self.load_map(m), axis=0, h=self.rmin[0], rule=self.rule) * self.get_dy() \
                   + PDP(self.load_map(m), axis=1, h=self.rmin[1], rule=self.rule) * self.get_dx()
        else:
            assert 0, crude

    def lens_map(self, m, use_Pool=0, crude=0, do_not_prefilter=False):
        """Lens the input flat-sky map, using a bicubic spline interpolation algorithm

            The task is split in chunks (of typically (2048 * 2048) or specified by the LD_res parameters)
            with a buffer size to ensure the junctions are properly performed.

            Args:
                 m: real-space map to deflect. numpy array, or the path to an array on disk.
                 use_Pool(optional): set this to < 0 to perform the operation on the GPU
                 crude(optional): uses an alternative crude approximation to lensing if set (check *lens_map_crude*)
                 do_not_prefilter(optional): sidesteps the bicubic interpolation prefiltering step.
                                             Only use this if you know what you are doing

            Returns: deflected real-space map (array)


        """
        assert self.load_map(m).shape == self.shape, (self.load_map(m).shape, self.shape)
        if crude > 0:
            return self.lens_map_crude(m, crude)
        if use_Pool < 0:
            # use of GPU :
            try:
                from lensit.gpu import lens_GPU
            except ImportError:
                assert 0, 'Import of mllens lens_GPU failed !'

            GPU_res = np.array(lens_GPU.GPU_HDres_max)
            if np.all(np.array(self.HD_res) <= GPU_res):
                return lens_GPU.lens_onGPU(m, self.get_dx_ingridunits(), self.get_dy_ingridunits(),
                                           do_not_prefilter=do_not_prefilter)
            LD_res, buffers = lens_GPU.get_GPUbuffers(GPU_res)
            assert np.all(np.array(buffers) > (np.array(self.buffers) + 5.)), (buffers, self.buffers)
            Nchunks = 2 ** (np.sum(np.array(self.HD_res) - np.array(LD_res)))
            lensed_map = np.empty(self.shape)  # Output
            dx_N = np.empty((2 ** LD_res[0] + 2 * buffers[0], 2 ** LD_res[1] + 2 * buffers[1]))
            dy_N = np.empty((2 ** LD_res[0] + 2 * buffers[0], 2 ** LD_res[1] + 2 * buffers[1]))
            unl_CMBN = np.empty((2 ** LD_res[0] + 2 * buffers[0], 2 ** LD_res[1] + 2 * buffers[1]))
            if self.verbose:
                print('++ lensing map :' \
                      '   splitting map on GPU , chunk shape %s, buffers %s' % (dx_N.shape, buffers))
            spliter_lib = map_spliter.periodicmap_spliter()  # library to split periodic maps.
            for N in range(Nchunks):
                sLDs, sHDs = spliter_lib.get_slices_chk_N(N, LD_res, self.HD_res, buffers)
                for sLD, sHD in zip(sLDs, sHDs):
                    dx_N[sLD] = self.get_dx()[sHD] / self.rmin[1]
                    dy_N[sLD] = self.get_dy()[sHD] / self.rmin[0]
                    unl_CMBN[sLD] = self.load_map(m)[sHD]
                sLDs, sHDs = spliter_lib.get_slices_chk_N(N, LD_res, self.HD_res, buffers, inverse=True)
                lensed_map[sHDs[0]] = lens_GPU.lens_onGPU(unl_CMBN, dx_N, dy_N, do_not_prefilter=do_not_prefilter)[sLDs[0]]
            return lensed_map

        elif use_Pool == 0 or use_Pool == 1:
            assert self.shape[0] == self.shape[1], self.shape
            if do_not_prefilter:
                filtmap = self.load_map(m).astype(np.float64)
            else:
                # TODO : may want to add pyFFTW here as well
                filtmap = np.fft.rfft2(self.load_map(m))
                w0 = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(filtmap.shape[0])) + 4.)
                filtmap *= np.outer(w0, w0[0:filtmap.shape[1]])
                filtmap = np.fft.irfft2(filtmap, self.shape)

            i = np.arange(int(np.prod(self.shape)), dtype=int)
            # new coordinates in grid units:
            x_gu = self.get_dx_ingridunits().flatten() + i % self.shape[1]
            y_gu = self.get_dy_ingridunits().flatten() + i // self.shape[1]
            del i
            return bicubic.deflect(filtmap, x_gu , y_gu).reshape(self.shape)

    def lens_alm(self, lib_alm, alm, lib_alm_out=None, mult_magn=False, use_Pool=0):
        """Returns lensed harmonic coefficients from the unlensed input coefficients

            Args:
                lib_alm: *lensit.ffs_covs.ell_mat.ffs_alm* instance adapted to input *alm* array
                alm: input unlensed flat-sky alm array
                lib_alm_out(optional): output *ffs_alm* instance if difference from input
                mult_magn(optional): optionally multiplies the real-space lensed map with the magnification det. if set.
                use_Pool(optional): calculations are performed on the GPU if negative.

            Returns:
                lensed alm array

        """
        if lib_alm_out is None: lib_alm_out = lib_alm
        if use_Pool < 0:  # can we fit the full map on the GPU ?
            from lensit.gpu import lens_GPU
            GPU_res = np.array(lens_GPU.GPU_HDres_max)
            if np.all(np.array(self.HD_res) <= GPU_res):
                return lens_GPU.lens_alm_onGPU(lib_alm, lib_alm.bicubic_prefilter(alm),
                                               self.get_dx_ingridunits(), self.get_dy_ingridunits(),
                                               do_not_prefilter=True, mult_magn=mult_magn, lib_alm_out=lib_alm_out)
        temp_map = self.alm2lenmap(lib_alm, alm, use_Pool=use_Pool)
        if mult_magn:
            self.mult_wmagn(temp_map, inplace=True)
        return lib_alm_out.map2alm(temp_map)

    def mult_wmagn(self, m, inplace=False):
        if not inplace:
            return self.get_det_magn() * m
        else:
            m *= self.get_det_magn()
            return

    def alm2lenmap(self, lib_alm, alm, use_Pool=0, crude=0):
        """Return deflected position-space map from its unlensed input harmonic coeffients.

            Args:
                lib_alm: *lensit.ffs_covs.ell_mat.ffs_alm* instance adapted to input *alm* array
                alm: input unlensed flat-sky alm array

            Returns:
                position space map of shape *lib_alm.shape*

        """
        assert alm.shape == (lib_alm.alm_size,), (alm.shape, lib_alm.alm_size)
        assert lib_alm.ell_mat.shape == self.shape, (lib_alm.ell_mat.shape, self.shape)
        if use_Pool < 0:  # can we fit the full map on the GPU ? If we can't we send it the lens_map
            from lensit.gpu import lens_GPU
            GPU_res = np.array(lens_GPU.GPU_HDres_max)
            if np.all(np.array(self.HD_res) <= GPU_res):
                return lens_GPU.alm2lenmap_onGPU(lib_alm, lib_alm.bicubic_prefilter(alm),
                                                 self.get_dx_ingridunits(), self.get_dy_ingridunits(),
                                                 do_not_prefilter=True)
        else:
            return self.lens_map(lib_alm.alm2map(lib_alm.bicubic_prefilter(alm)),
                                 use_Pool=use_Pool, do_not_prefilter=True, crude=crude)

    def get_det_magn(self):
        r"""Returns magnification determinant map

            :math:`|M| = \det \begin{pmatrix} 1 + \frac{\partial \alpha_x}{\partial x} & \frac{\partial \alpha_x}{\partial y}
            \\ \frac{\partial \alpha_y}{\partial x} & 1 + \frac{\partial \alpha_y}{\partial y} \end{pmatrix}`

        """
        # FIXME : bad
        if not self.cache_magn:
            det = (PDP(self.get_dx(), axis=1, h=self.rmin[1], rule=self.rule) + 1.) \
                  * (PDP(self.get_dy(), axis=0, h=self.rmin[0], rule=self.rule) + 1.)
            det -= PDP(self.get_dy(), axis=1, h=self.rmin[1], rule=self.rule) * \
                   PDP(self.get_dx(), axis=0, h=self.rmin[0], rule=self.rule)
            return det
        else:
            assert self.lib_dir is not None, 'Specify lib_dir if you want to cache magn'
            fname = os.path.join(self.lib_dir, 'det_magn_%s_%s_rank%s.npy' % \
                                   (hashlib.sha1(self.get_dx()[0, 0:100]).hexdigest(),
                                    hashlib.sha1(self.get_dy()[0, 0:100]).hexdigest(), pbs.rank))
            if not os.path.exists(fname):  # and pbs.rank == 0:
                det = (PDP(self.get_dx(), axis=1, h=self.rmin[1], rule=self.rule) + 1.) \
                      * (PDP(self.get_dy(), axis=0, h=self.rmin[0], rule=self.rule) + 1.)
                det -= PDP(self.get_dy(), axis=1, h=self.rmin[1], rule=self.rule) * \
                       PDP(self.get_dx(), axis=0, h=self.rmin[0], rule=self.rule)
                print("  ffs_displacement caching ", fname)
                np.save(fname, det)
                del det
            return np.load(fname)


    def get_kappa(self):
        r"""Convergence map.

            Returns:
                :math:`\kappa = -\frac 12  \left( \frac{\partial \alpha_x}{\partial x} +  \frac{\partial \alpha_y}{\partial y} \right)`

        """
        dfxdx = PDP(self.get_dx(), axis=1, h=self.rmin[1], rule=self.rule)
        dfydy = PDP(self.get_dy(), axis=0, h=self.rmin[0], rule=self.rule)
        return -0.5 * (dfxdx + dfydy)

    def get_omega(self):
        r"""Field rotation map

            Returns:
                :math:`\omega = \frac 12  \left( \frac{\partial \alpha_x}{\partial y} -  \frac{\partial \alpha_y}{\partial x} \right)`

        """
        dfxdy = PDP(self.get_dx(), axis=0, h=self.rmin[0], rule=self.rule)
        dfydx = PDP(self.get_dy(), axis=1, h=self.rmin[1], rule=self.rule)
        return 0.5 * (dfxdy - dfydx)

    def get_phi(self):
        """Lensing (gradient) potential map


        """
        rfft_phi = np.fft.rfft2(self.get_kappa())
        rs = rfft_phi.shape
        ky = (2. * np.pi) / self.lsides[0] * Freq(np.arange(self.shape[0]), self.shape[0])
        ky[int(self.shape[0] / 2):] *= -1.
        kx = (2. * np.pi) / self.lsides[1] * Freq(np.arange(rs[1]), self.shape[1])
        rfft_phi = rfft_phi.flatten()
        rfft_phi[1:] /= (np.outer(ky ** 2, np.ones(rs[1])) + np.outer(np.ones(rs[0]), kx ** 2)).flatten()[1:]
        return np.fft.irfft2(2 * rfft_phi.reshape(rs), self.shape)

    def get_Omega(self):
        """Lensing (curl) potential map


        """
        rfft_Om = np.fft.rfft2(self.get_omega())
        rs = rfft_Om.shape
        ky = (2. * np.pi) / self.lsides[0] * Freq(np.arange(self.shape[0]), self.shape[0])
        ky[self.shape[0] / 2:] *= -1.
        kx = (2. * np.pi) / self.lsides[1] * Freq(np.arange(rs[1]), self.shape[1])
        rfft_Om = rfft_Om.flatten()
        rfft_Om[1:] /= (np.outer(ky ** 2, np.ones(rs[1])) + np.outer(np.ones(rs[0]), kx ** 2)).flatten()[1:]
        return np.fft.irfft2(2 * rfft_Om.reshape(rs), self.shape)

    def get_inverse_crude(self, crude):
        """Crude inversions of the displacement field

            Args:
                crude: method key

            Supported now is only *crude* = 1, for which the inverse is approximated as the negative deflection

        """
        assert crude in [1], crude
        if crude == 1:
            return ffs_displacement(- self.get_dx(), -self.get_dy(), self.lsides, lib_dir=self.lib_dir,
                                    LD_res=self.LD_res, verbose=self.verbose, NR_iter=self.NR_iter)
        else:
            assert 0, crude

    def get_inverse(self, NR_iter=None, use_Pool=0, crude=0, HD_res=None):
        """Builds deflection field inverse

            The inverse deflection is defined by the condition that forward-deflected points remap to the original points
            under the inverse displacement.

            Args:
                NR_iter(optional): Number of Newton-Raphson iterations. Superseeds the instance NR_iter argument if set
                use_Pool(optional): Send the calculation to the GPU if negative
                crude(optional): Uses some faster but crude method for the inverse (check *get_inverse_crude*)
                HD_res(optional): Set this to perform the inversion at higher resolution (tuple of powers of two)

            Returns: inverse deflection field as *ffs_displacement* instance

        """
        if HD_res is not None:
            lib_dir = self.lib_dir if self.lib_dir is None else os.path.join(self.lib_dir, 'temp_fup')
            f_up = ffs_displacement(rfft2_utils.upgrade_map(self.get_dx(), HD_res),
                                    rfft2_utils.upgrade_map(self.get_dy(), HD_res), self.lsides,
                                    LD_res=self.LD_res, verbose=self.verbose,
                                    NR_iter=self.NR_iter, lib_dir=lib_dir)
            f_up_inv = f_up.get_inverse(NR_iter=NR_iter, use_Pool=use_Pool, crude=crude)
            LD_res = Log2ofPowerof2(self.shape)
            return ffs_displacement(rfft2_utils.subsample(f_up_inv.get_dx(), LD_res),
                                    rfft2_utils.subsample(f_up_inv.get_dy(), LD_res), self.lsides,
                                    LD_res=self.LD_res, verbose=self.verbose,
                                    NR_iter=self.NR_iter, lib_dir=self.lib_dir)

        if crude > 0:
            return self.get_inverse_crude(crude)

        if NR_iter is None: NR_iter = self.NR_iter

        if use_Pool == 0:
            spliter_lib = map_spliter.periodicmap_spliter()  # library to split periodic maps.
            dx_inv, dy_inv = np.empty(self.shape), np.empty(self.shape)
            label = 'ffs_deflect::calculating inverse displ. field'
            for i, N in utils.enumerate_progress(range(self.N_chks), label=label):
                # Doing chunk N
                dx_inv_N, dy_inv_N = self._get_inverse_chk(N, NR_iter=NR_iter)
                sLDs, sHDs = spliter_lib.get_slices_chk_N(N, self.LD_res, self.HD_res, self.buffers, inverse=True)
                # Pasting it onto the full map
                dx_inv[sHDs[0]] = dx_inv_N[sLDs[0]]
                dy_inv[sHDs[0]] = dy_inv_N[sLDs[0]]
            return ffs_displacement(dx_inv, dy_inv, self.lsides, lib_dir=self.lib_dir,
                                    LD_res=self.LD_res, verbose=self.verbose, NR_iter=self.NR_iter)
        elif use_Pool < 0:
            # GPU calculation.
            from lensit.gpu_old import lens_GPU
            from lensit.gpu_old import inverse_GPU as inverse_GPU
            GPU_res = np.array(inverse_GPU.GPU_HDres_max)
            if np.all(np.array(self.HD_res) <= GPU_res):
                # No need to split maps :
                dx_inv, dy_inv = inverse_GPU.inverse_GPU(self.get_dx(), self.get_dy(), self.rmin, NR_iter)
                return ffs_displacement(dx_inv, dy_inv, self.lsides, lib_dir=self.lib_dir,
                                        LD_res=self.LD_res, verbose=self.verbose,   NR_iter=self.NR_iter)
            else:
                LD_res, buffers = lens_GPU.get_GPUbuffers(GPU_res)
                assert np.all(np.array(buffers) > (np.array(self.buffers) + 5.)), (buffers, self.buffers)
                Nchunks = 2 ** (np.sum(np.array(self.HD_res) - np.array(LD_res)))
                dx_N = np.empty((2 ** LD_res[0] + 2 * buffers[0], 2 ** LD_res[1] + 2 * buffers[1]))
                dy_N = np.empty((2 ** LD_res[0] + 2 * buffers[0], 2 ** LD_res[1] + 2 * buffers[1]))
                if self.verbose:
                    print('++ inverse displacement :' \
                          '   splitting inverse on GPU , chunk shape %s, buffers %s' % (dx_N.shape, buffers))
                spliter_lib = map_spliter.periodicmap_spliter()  # library to split periodic maps.
                dx_inv, dy_inv = np.empty(self.shape), np.empty(self.shape)  # Outputs
                for N in range(Nchunks):
                    sLDs, sHDs = spliter_lib.get_slices_chk_N(N, LD_res, self.HD_res, buffers)
                    for sLD, sHD in zip(sLDs, sHDs):
                        dx_N[sLD] = self.get_dx()[sHD]
                        dy_N[sLD] = self.get_dy()[sHD]
                    dx_inv_N, dy_inv_N = inverse_GPU.inverse_GPU(dx_N, dy_N, self.rmin, NR_iter)
                    sLDs, sHDs = spliter_lib.get_slices_chk_N(N, LD_res, self.HD_res, buffers, inverse=True)
                    dx_inv[sHDs[0]] = dx_inv_N[sLDs[0]]
                    dy_inv[sHDs[0]] = dy_inv_N[sLDs[0]]

                return ffs_displacement(dx_inv, dy_inv, self.lsides,
                            lib_dir=self.lib_dir, LD_res=self.LD_res, verbose=self.verbose, NR_iter=self.NR_iter)
        elif use_Pool == 100:
            assert 0
        else:
            assert 0

    def _get_inverse_chk(self, N, NR_iter=None):
        """Returns inverse displacement in chunk N

            NB: Uses periodic boundary conditions, which is not applicable to chunks, thus there
            will be boudary effects on the edges (2 or 4 pixels depending on the rule). Make sure the buffer is large enough.

        """
        if NR_iter is None: NR_iter = self.NR_iter

        # Inverse magn. elements. (with a minus sign) We may need to spline these later for further NR iterations :
        extra_buff = np.array((5, 5)) * (np.array(self.chk_shape) != np.array(self.shape))
        # :to avoid surprises with the periodic derivatives
        dx = np.zeros(self.chk_shape + 2 * extra_buff)  # will dx displ. in grid units of each chunk (typ. (256 * 256) )
        dy = np.zeros(self.chk_shape + 2 * extra_buff)  # will dy displ. in grid units of each chunk (typ. (256 * 256) )
        sLDs, sHDs = map_spliter.periodicmap_spliter.get_slices_chk_N(N, self.LD_res, self.HD_res,
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
        #if not np.all(det > 0.): print("ffs_displ::Negative value in det k : something's weird, you'd better check that")
        assert np.all(det > 0.), "ffs_displ::Negative value in det k : something's weird, you'd better check that"
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
        r1 = s1 // 2 + 1  # rfft shape

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
        for i in range(0, NR_iter):
            ex1, ey1 = bicubic.deflect_inverse(ex, ey, dx, dy, Minv_xx, Minv_yy, Minv_xy, Minv_yx)
            if self.verbose: # prints some convergence info.
                max_incr = max(np.max(np.abs(ex1 - ex)) * self.rmin[1], np.max(np.abs(ey1- ey)) * self.rmin[0])
                max_incr *=  180. * 60. / np.pi
                print('NR iter %s: max. increment size in NR deflection inverse %.2e amin'%(i + 1, max_incr))
                rms_x = np.mean(np.abs(ex1- ex)) * self.rmin[1] / np.pi * 180. * 60.
                rms_y = np.mean(np.abs(ey1- ey)) * self.rmin[0] / np.pi * 180. * 60.
                print('           mean x, y rms increment : %.2e, %.2e amin'%(rms_x, rms_y))
            ex = ex1
            ey = ey1
        return ex * rmin1, ey * rmin0

    def degrade(self, LD_shape, no_lensing, **kwargs):
        if no_lensing: return ffs_id_displacement(LD_shape, self.lsides)
        dx = rfft2_utils.degrade(self.get_dx(), LD_shape)
        dy = rfft2_utils.degrade(self.get_dy(), LD_shape)
        return ffs_displacement(dx, dy, self.lsides, **kwargs)

    def get_noisefreemf(self, lib_qlm):
        """Deflection induced mean-field estimate for a noisefree, isotropic experimental setting

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


class ffs_id_displacement:
    """Displacement instance where there is actually no displacement.

        Mostly useful for testing purposes

    """

    def __init__(self, shape, lsides):
        self.shape = shape
        self.lsides = lsides

    def copy(self):
        return ffs_id_displacement(self.shape, self.lsides)

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

    @staticmethod
    def lens_map(m, **kwargs):
        if isinstance(m, str):
            return np.load(m)
        else:
            return m

    def clone(self):
        return ffs_id_displacement(self.shape, self.lsides)

    @staticmethod
    def mult_wmagn(m, inplace=False):
        if inplace:
            return
        else:
            return m

    @staticmethod
    def lens_alm(lib_alm, alm, lib_alm_out=None, **kwargs):
        if lib_alm_out is not None:
            return lib_alm_out.udgrade(lib_alm, alm)
        return alm

    @staticmethod
    def alm2lenmap(lib_alm, alm, **kwargs):
        return lib_alm.alm2map(alm)

    def get_det_magn(self):
        return np.ones(self.shape, dtype=float)

    def rotpol(self, QpiU, **kwargs):
        assert np.iscomplexobj(QpiU) and QpiU.shape == self.shape, (QpiU.shape, np.iscomplexobj(QpiU))
        return QpiU


def displacement_fromplm(lib_plm, plm, **kwargs):
    r"""Returns deflection-field instance from lensing gradient potential

        Args:
             lib_plm: *ffs_alm* or *ffs_alm_pyFFTW* instance describing the plm array
             plm: flat-sky lensing gradient potential alm array

        Returns:
            Deflection field as *ffs_displacement* instance

    """
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