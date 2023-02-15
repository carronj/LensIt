from __future__ import print_function

import numpy as np
import os
import pickle as pk

import pyfftw

from lensit.sims.sims_generic import hash_check
from lensit.misc.misc_utils import npy_hash, Freq
from lensit.misc.rfft2_utils import udgrade_rfft2, supersample
from lensit.pbs import pbs


class ell_mat:
    """Library helping with flat-sky patch discretization and harmonic mode structure.

        This handles Fourier mode structure on the flat sky, at the given resolution and size of
        specified rectangular box.

        Args:
            lib_dir: various things might be cached there
                    (unlens *cache* is set to 0, in which case only a hashdict is put there at instantiation)
            shape(2-tuple): pair of int defining the number of pixels on each side of the box
            lsides(2-tuple): physical size (in radians) of the box sides
            cache(optional): if non-zero, a bunch of matrices might be cached to speed up some calculations.

    """

    def __init__(self, lib_dir, shape, lsides, cache=1):
        assert len(shape) == 2 and len(lsides) == 2
        assert shape[0] % 2 == 0 and shape[1] % 2 == 0
        assert shape[0] < 2 ** 16 and shape[1] < 2 ** 16
        self.shape = tuple(shape)
        self.rshape = (shape[0], shape[1] // 2 + 1)
        self.lsides = tuple(lsides)
        self.lib_dir = lib_dir
        self.mmap_mode = None
        self.cache=cache

        fn_hash = os.path.join(lib_dir, "ellmat_hash.pk")
        if pbs.rank == 0 and self.cache > 0:
            if not os.path.exists(lib_dir): os.makedirs(lib_dir)
            if not os.path.exists(fn_hash):
                pk.dump(self.hash_dict(), open(fn_hash, 'wb'), protocol=2)
        pbs.barrier()

        if self.cache > 0:
            hash_check(pk.load(open(fn_hash, 'rb')), self.hash_dict())

        if pbs.rank == 0 and self.cache > 0 and not os.path.exists(os.path.join(self.lib_dir, 'ellmat.npy')):
                print('ell_mat:caching ells in ' + os.path.join(self.lib_dir, 'ellmat.npy'))
                np.save(os.path.join(self.lib_dir, 'ellmat.npy'), self._build_ellmat())
        pbs.barrier()

        self.ellmax = int(self._get_ellmax())
        self._ell_counts = self._build_ell_counts()
        self._nz_counts = self._ell_counts.nonzero()

    def __eq__(self, other):
        return self.shape == other.shape and self.lsides == self.lsides

    def _build_ellmat(self):
        kmin = 2. * np.pi / np.array(self.lsides)
        ky2 = Freq(np.arange(self.shape[0]), self.shape[0]) ** 2 * kmin[0] ** 2
        kx2 = Freq(np.arange(self.rshape[1]), self.shape[1]) ** 2 * kmin[1] ** 2
        ones = np.ones(np.max(self.shape))
        return self.k2ell(np.sqrt(np.outer(ky2, ones[0:self.rshape[1]]) + np.outer(ones[0:self.rshape[0]], kx2)))

    def hash_dict(self):
        return {'shape': self.shape, 'lsides': self.lsides}

    @staticmethod
    def k2ell(k):
        r"""Mapping of 2d-frequency :math:`k` to multipole :math:`\ell`

            :math:`\ell = \rm{int}\left(|k| - \frac 12 \right)`

        """
        ret = np.uint32(np.round(k - 0.5) + 0.5 * ((k - 0.5) < 0))
        return ret

    def __call__(self, *args, **kwargs):
        return self.get_ellmat(*args, **kwargs)

    def __getitem__(self, item):
        return self.get_ellmat()[item]

    def get_pixwinmat(self):
        r"""Pixel window function rfft array :math:`\sin(k_x l_{\rm cell, x} / 2) \sin (k_y l_{\rm cell, y} / 2 )`


        """
        ky = (np.pi/self.shape[0]) * Freq(np.arange(self.shape[0]), self.shape[0])
        ky[self.shape[0] // 2:] *= -1.
        kx = (np.pi/self.shape[1]) * Freq(np.arange(self.rshape[1]), self.shape[1])
        rety = np.sin(ky)
        rety[1:] /= ky[1:];rety[0] = 1.
        retx = np.sin(kx)
        retx[1:] /= kx[1:];retx[0] = 1.
        return np.outer(rety,retx)

    def get_ellmat(self, ellmax=None):
        r"""Returns the matrix containing the multipole ell assigned to 2d-frequency :math:`k = (k_x,k_y)`

        """
        if self.cache == 0:
            ret = self._build_ellmat()
            return ret if ellmax is None else ret[np.where(ret <= ellmax)]

        if ellmax is None:
            return np.load(os.path.join(self.lib_dir, 'ellmat.npy'), mmap_mode=self.mmap_mode)
        else:
            fname = os.path.join(self.lib_dir, 'ellmat_ellmax%s.npy' % ellmax)
            if os.path.exists(fname): return np.load(fname, mmap_mode=self.mmap_mode)
            if pbs.rank == 0:
                print('ell_mat:caching ells in ' + fname)
                np.save(fname, self.get_ellmat()[np.where(self() <= ellmax)])
            pbs.barrier()
            return np.load(fname, mmap_mode=self.mmap_mode)

    def get_phasemat(self, ellmax=None):
        r"""Returns the rfft array containing the phase :math:`\phi` where 2d-frequencies are :math:`k = |k| e^{i \phi}`

        """
        if self.cache == 0:
            ret = np.arctan2(self.get_ky_mat(), self.get_kx_mat())
            return ret if ellmax is None else ret[np.where(self() <= ellmax)]
        if ellmax is None:
            fname = os.path.join(self.lib_dir, 'phasemat.npy')
            if os.path.exists(fname): return np.load(fname, mmap_mode=self.mmap_mode)
            if not os.path.exists(fname) and pbs.rank == 0:
                print('ell_mat:caching phases in '+ fname)
                np.save(fname, np.arctan2(self.get_ky_mat(), self.get_kx_mat()))
            pbs.barrier()
            return np.load(fname, mmap_mode=self.mmap_mode)
        else:
            fname = os.path.join(self.lib_dir, 'phase_ellmax%s.npy' % ellmax)
            if not os.path.exists(fname) and pbs.rank == 0:
                print('ell_mat:caching phases in '+ fname)
                np.save(fname, np.arctan2(self.get_ky_mat(), self.get_kx_mat())[np.where(self.get_ellmat() <= ellmax)])
            pbs.barrier()
            return np.load(fname, mmap_mode=self.mmap_mode)

    def get_e2iphi_mat(self, cache_only=False):
        r"""Returns the matrix containing the phase :math:`e^{2 i \phi}`

        """
        if self.cache == 0:
            return np.exp(2j * np.arctan2(self.get_ky_mat(), self.get_kx_mat()))

        fname = os.path.join(self.lib_dir, 'e2iphimat.npy')
        if os.path.exists(fname):
            return None if cache_only else np.load(fname, mmap_mode=self.mmap_mode)
        if not os.path.exists(fname) and pbs.rank == 0:
            print('ell_mat:caching e2iphi in ' + fname)
            np.save(fname, np.exp(2j * np.arctan2(self.get_ky_mat(), self.get_kx_mat())))
        pbs.barrier()
        return None if cache_only else np.load(fname, mmap_mode=self.mmap_mode)

    def degrade(self, LDshape, lib_dir=None):
        """Returns an *ell_mat* instance on the same physical flat-sky patch but a degraded sampling resolution

        """
        if np.all(LDshape >= self.shape): return self
        if lib_dir is None:
            lib_dir = os.path.join(self.lib_dir, 'degraded%sx%s' % (LDshape[0], LDshape[1]))
        return ell_mat(lib_dir, LDshape, self.lsides, cache=self.cache)

    def get_cossin_2iphi_mat(self):
        """

            Returns:
                :math:`\cos(2i\phi), \sin(2i\phi)` as rfft arrays

        """
        e2iphi = self.get_e2iphi_mat()
        return e2iphi.real, e2iphi.imag


    def _get_ellmax(self):
        r""" Max. ell present in the grid

        """
        return np.max(self.get_ellmat())

    def _build_ell_counts(self):
        counts = np.bincount(self.get_ellmat()[:, 1:self.rshape[1] - 1].flatten(), minlength=self.ellmax + 1)
        s_counts = np.bincount(self.get_ellmat()[0:self.shape[0] // 2 + 1, [-1, 0]].flatten())
        counts[0:len(s_counts)] += s_counts
        return counts

    def _get_ellcounts(self):
        r"""Number of non-redundant entries in freq map. for each ell, in the rfftmap.

            Corresponds roughly to :math:`\ell + \frac 1/2`.

        """
        return self._ell_counts

    def get_Nell(self):
        """Mode number counts on the flat-sky patch.

            Goes roughly like :math:`2\ell + 1`

        """
        Nell = 2 * self._get_ellcounts()
        for ell in self.get_ellmat()[self.rfft2_reals()]:
            Nell[ell] -= 1
        return Nell

    def get_nonzero_ellcounts(self):
        return self._nz_counts

    def map2cl(self, m, m2=None):
        r"""Returns s pseudo-:math:`C_\ell` estimate from a map.

            Returns a cross-:math:`C_\ell` if m2 is set. Must have compatible shape.

        """
        assert m.shape == self.shape, m.shape
        if m2 is not None:
            assert m2.shape == self.shape, m2.shape
            return self._rfft2cl(np.fft.rfft2(m), rfftmap2=np.fft.rfft2(m2))
        return self._rfft2cl(np.fft.rfft2(m))

    def _rfft2cl(self, rfftmap, rfftmap2=None):
        """Returns a :math:`C_\ell` estimate from a rfftmap.

            (e.g. np.fft.rfft2(sim) where sim is the output of this library)
            Must have the same shape then self.rshape

        """
        assert rfftmap.shape == self.rshape, rfftmap.shape
        if rfftmap2 is not None: assert rfftmap2.shape == self.rshape, rfftmap2.shape
        weights = rfftmap.real ** 2 + rfftmap.imag ** 2 if rfftmap2 is None else (rfftmap * np.conjugate(rfftmap2)).real
        Cl = np.bincount(self.get_ellmat()[:, 1:self.rshape[1] - 1].flatten(),
                         weights=weights[:, 1:self.rshape[1] - 1].flatten(), minlength=self.ellmax + 1)
        Cl += np.bincount(self.get_ellmat()[0:self.shape[0] // 2 + 1, [-1, 0]].flatten(),
                          weights=weights[0:self.shape[0] // 2 + 1, [-1, 0]].flatten(), minlength=self.ellmax + 1)
        Cl[self._nz_counts] *= (np.prod(self.lsides) / np.prod(self.shape) ** 2) / self._ell_counts[self._nz_counts]
        return Cl

    def bin_inell(self, rfftmap):
        """Bin in a rfftmap according to multipole :math:`\ell`

            Input must have the same shape then self.rshape

        """
        assert rfftmap.shape == self.rshape, rfftmap.shape
        weights = rfftmap
        Cl = np.bincount(self.get_ellmat()[:, 1:self.rshape[1] - 1].flatten(),
                         weights=weights[:, 1:self.rshape[1] - 1].flatten(), minlength=self.ellmax + 1)
        Cl += np.bincount(self.get_ellmat()[0:self.shape[0] // 2 + 1, [-1, 0]].flatten(),
                          weights=weights[0:self.shape[0] // 2 + 1, [-1, 0]].flatten(), minlength=self.ellmax + 1)
        Cl[self._nz_counts] /= self._ell_counts[self._nz_counts]
        return Cl

    def get_kx_mat(self):
        kx_min = (2. * np.pi) / self.lsides[1]
        kx = kx_min * Freq(np.arange(self.rshape[1]), self.shape[1])
        return np.outer(np.ones(self.shape[0]), kx)

    def get_ky_mat(self):
        ky_min = (2. * np.pi) / self.lsides[0]
        ky = ky_min * Freq(np.arange(self.shape[0]), self.shape[0])
        ky[self.shape[0] // 2:] *= -1.
        return np.outer(ky, np.ones(self.rshape[1]))

    def get_ikx_mat(self):
        return 1j * self.get_kx_mat()

    def get_iky_mat(self):
        return 1j * self.get_ky_mat()

    def get_unique_ells(self, return_index=False, return_inverse=False, return_counts=False):
        """Returns list of multipoles :math:`\ell` present in the flat-sky patch

        """
        return np.unique(self.get_ellmat(),
                         return_index=return_index, return_inverse=return_inverse, return_counts=return_counts)

    def rfftmap2alm(self, rfftmap, filt_func=lambda ell: True):
        """Turns a rfft map to ffs_alm array.
        
            This performs the filtering defined by the instance together with the normalization
            

        """
        cond = filt_func(np.arange(self.ellmax + 1))
        return rfftmap[cond[self.get_ellmat()]] * np.sqrt(np.prod(self.lsides)) / np.prod(self.shape)

    def alm2rfftmap(self, alm, filt_func=lambda ell: True):
        """Returns a ffs_alm array from a rfftmap.

            Pseudo-inverse to *fftmap2alm*

        """
        cond = filt_func(np.arange(self.ellmax + 1))
        ret = np.zeros(self.rshape, dtype=complex)
        ret[cond[self.get_ellmat()]] = alm * np.prod(self.shape) / np.sqrt(np.prod(self.lsides))
        return ret

    def rfft2_reals(self):
        """Pure reals modes in 2d rfft according to patch specifics

        """
        N0, N1 = self.shape
        fx = [0]
        fy = [0]
        if N1 % 2 == 0:
            fx.append(0)
            fy.append(N1 // 2)
        if N0 % 2 == 0:
            fx.append(N0 // 2)
            fy.append(0)
        if N1 % 2 == 0 and N0 % 2 == 0:
            fx.append(N0 // 2)
            fy.append(N1 // 2)
        return np.array(fx), np.array(fy)


class ffs_alm(object):
    """Library to facilitate operations on flat-sky alms in the ffs scheme.

        Methods includes alm2map and map2alm, among other things.
        The instance contains info on the sky patch size and discretization.

        Args:
            ellmat: *lensit.ffs_covs.ell_mat.ell_mat* instance carrying flat-sky patch definitions
            filt_func(callable, optional): mutlipole filtering. Set this to discard multipoles in map2alm
                                           Defaults to lambda ell : ell > 0
            kxfilt_func(callable, optional): filtering on x-component of the 2d-frequencies

            kyfilt_func(callable, optional): filtering on y-component of the 2d-frequencies


    """

    def __init__(self, ellmat, filt_func=lambda ell: ell > 0, kxfilt_func=None, kyfilt_func=None):

        self.ell_mat = ellmat
        self.shape = self.ell_mat.shape
        self.lsides = self.ell_mat.lsides

        self.filt_func = filt_func
        self.kxfilt_func = kxfilt_func
        self.kyfilt_func = kyfilt_func


        self.alm_size = np.count_nonzero(self._cond())
        # The mapping ell[i] for i in alm array :
        self.reduced_ellmat = lambda: ellmat()[self._cond()]
        self.ellmax = np.max(self.reduced_ellmat()) if self.alm_size > 0 else None
        self.ellmin = np.min(self.reduced_ellmat()) if self.alm_size > 0 else None

        # Some trivial convenience factors :
        self.fac_rfft2alm = np.sqrt(np.prod(ellmat.lsides)) / np.prod(self.ell_mat.shape)
        self.fac_alm2rfft = 1. / self.fac_rfft2alm
        self.__ellcounts = None

    def _cond(self):
        ret =  self.filt_func(self.ell_mat())
        if self.kxfilt_func is not None:
            ret &= self.kxfilt_func(self.ell_mat.get_kx_mat())
        if self.kyfilt_func is not None:
            ret &= self.kyfilt_func(self.ell_mat.get_ky_mat())
        return ret

    def __eq__(self, lib_alm):
        if not np.all(self.ell_mat.lsides == lib_alm.ell_mat.lsides):
            return False
        if not np.all(self.ell_mat.shape == lib_alm.ell_mat.shape):
            return False
        ellmax = max(self.ellmax, lib_alm.ellmax)
        if not np.all(self.filt_func(np.arange(ellmax + 1)) == lib_alm.filt_func(np.arange(ellmax + 1))):
            return False
        kxf = self.kxfilt_func if self.kxfilt_func is not None else lambda kx : np.ones_like(kx, dtype = bool)
        _kxf = lib_alm.kxfilt_func if lib_alm.kxfilt_func is not None else lambda kx: np.ones_like(kx, dtype=bool)
        if not np.all(kxf(np.arange(-ellmax,ellmax + 1)) == _kxf(np.arange(-ellmax,ellmax + 1))):
            return False

        kyf = self.kyfilt_func if self.kyfilt_func is not None else lambda ky : np.ones_like(ky, dtype = bool)
        _kyf = lib_alm.kyfilt_func if lib_alm.kyfilt_func is not None else lambda ky: np.ones_like(ky, dtype=bool)
        if not np.all(kyf(np.arange(-ellmax,ellmax + 1)) == _kyf(np.arange(-ellmax,ellmax + 1))):
            return False

        return True

    def iseq(self, lib_alm, allow_shape=False):
        if not allow_shape: return self == lib_alm
        # We allow differences in resolution provided the filtering is the scheme
        # (ordering should then be the same as well)
        if not np.all(self.ell_mat.lsides == lib_alm.ell_mat.lsides):
            return False
        if not self.alm_size == lib_alm.alm_size:
            return False

        ellmax = max(self.ellmax, lib_alm.ellmax)
        if not np.all(self.filt_func(np.arange(ellmax + 1)) == lib_alm.filt_func(np.arange(ellmax + 1))):
            return False
        kxf = self.kxfilt_func if self.kxfilt_func is not None else lambda kx: np.ones_like(kx, dtype=bool)
        _kxf = lib_alm.kxfilt_func if lib_alm.kxfilt_func is not None else lambda kx: np.ones_like(kx, dtype=bool)
        if not np.all(kxf(np.arange(-ellmax, ellmax + 1)) == _kxf(np.arange(-ellmax, ellmax + 1))):
            return False

        kyf = self.kyfilt_func if self.kyfilt_func is not None else lambda ky: np.ones_like(ky, dtype=bool)
        _kyf = lib_alm.kyfilt_func if lib_alm.kyfilt_func is not None else lambda ky: np.ones_like(ky, dtype=bool)
        if not np.all(kyf(np.arange(-ellmax, ellmax + 1)) == _kyf(np.arange(-ellmax, ellmax + 1))):
            return False

        return True

    def hashdict(self):
        ret = {'ellmat': self.ell_mat.hash_dict(),
                'filt_func': npy_hash(self.filt_func(np.arange(self.ell_mat.ellmax + 1)))}
        if self.kxfilt_func is not None:
            ret[ 'kxfilt_func'] =  npy_hash(self.kxfilt_func(np.arange(-self.ell_mat.ellmax,self.ell_mat.ellmax + 1)))
        if self.kyfilt_func is not None:
            ret[ 'kyfilt_func'] =  npy_hash(self.kyfilt_func(np.arange(-self.ell_mat.ellmax,self.ell_mat.ellmax + 1)))
        return ret

    def degrade(self, LD_shape, ellmax=None, ellmin=None):
        LD_ellmat = self.ell_mat.degrade(LD_shape)
        if ellmax is None: ellmax = self.ellmax
        if ellmin is None: ellmin = self.ellmin
        filt_func = lambda ell: (self.filt_func(ell) & (ell <= ellmax) & (ell >= ellmin))
        return ffs_alm(LD_ellmat, filt_func=filt_func)

    def fsky(self):
        return np.prod(self.ell_mat.lsides) / 4. / np.pi

    def filt_hash(self):
        if self.kyfilt_func is None and self.kxfilt_func is None :
            return  npy_hash(self.filt_func(np.arange(self.ellmax + 1)))
        else:
            assert 0,'implement this'

    def clone(self):
        return ffs_alm(self.ell_mat, filt_func=self.filt_func)

    def nbar(self):
        return np.prod(self.ell_mat.shape) / np.prod(self.ell_mat.lsides)

    def rfftmap2alm(self, rfftmap):
        assert rfftmap.shape == self.ell_mat.rshape, rfftmap.shape
        return self.fac_rfft2alm * rfftmap[self._cond()]

    def almmap2alm(self, almmap):
        assert almmap.shape == self.ell_mat.rshape, almmap.shape
        return almmap[self._cond()]

    def map2rfft(self, _map):
        return np.fft.rfft2(_map)

    def map2alm(self, m, lib_almin=None):
        """Computes alm array of input position-space map

        """
        if lib_almin is None or self.shape == lib_almin.shape:
            return self.rfftmap2alm(self.map2rfft(m))
        else:
            return self.rfftmap2alm(self.map2rfft(supersample(m, lib_almin.ell_mat.shape)))

    def alm2rfft(self, alm):
        assert alm.size == self.alm_size, alm.size
        ret = np.zeros(self.ell_mat.rshape, dtype=complex)
        ret[self._cond()] = alm * self.fac_alm2rfft
        return ret

    def alm2almmap(self, alm):
        assert alm.size == self.alm_size, alm.size
        ret = np.zeros(self.ell_mat.rshape, dtype=complex)
        ret[self._cond()] = alm
        return ret

    def alm2map(self, alm, lib_almout=None):
        """Returns position-space map from alm array


        """
        if lib_almout is None:
            assert alm.size == self.alm_size, alm.size
            return np.fft.irfft2(self.alm2rfft(alm), self.ell_mat.shape)
        else:
            return lib_almout.alm2map(lib_almout.udgrade(self, alm))

    def almxfl(self, alm, fl, inplace=False):
        """Multiply :flat-sky math:`a_{\ell lm}` array with isotropic function :math:`f_\ell`


        """
        if alm.size % self.alm_size == 0: #possibly several components concatenated
            s = self.alm_size
            ncomp = alm.size // s
            if ncomp != 1:
                print("almxfl, %s comp"%ncomp)
            fl2d = np.atleast_2d(np.array(fl))
            assert len(fl2d) == ncomp, (len(fl2d), ncomp)
            ret = alm if inplace else np.copy(alm)
            for ia, a in enumerate(range(ncomp)):
                assert len(fl2d[ia]) > self.ellmax
                ret[a*s:(a + 1)*s] *= fl2d[ia][self.reduced_ellmat()]
            if inplace:
                return
            return ret
        else:
            assert 0, ('dont know what to do', alm.size, self.alm_size)


    def get_Nell(self):
        r"""Mode number counts on the flat-sky patch.

            Analog of :math:`2\ell + 1` on the flat-sky.
            instance mode filtering is included (i.e. set to zero for mods filtered out)

        """
        Nell = 2 * self._get_ell_counts()
        for ell,cond in zip(self.ell_mat()[self.ell_mat.rfft2_reals()],self._cond()[self.ell_mat.rfft2_reals()]):
            Nell[ell] -= cond
        return Nell[:self.ellmax + 1]

    def _get_ell_counts(self):
        if self.__ellcounts is None:
            self.__ellcounts = self._build_ell_counts()
        return self.__ellcounts

    def _build_ell_counts(self):
        r"""Number of non-redundant entries in freq map. for each ell, in the rfftmap.

            Corresponds roughly to :math:`\ell + \frac 1 2`.

        """
        weights = self._cond()
        counts = np.bincount(self.ell_mat()[:, 1:self.ell_mat.rshape[1] - 1].flatten(), minlength=self.ell_mat.ellmax + 1,weights=weights[:, 1:self.ell_mat.rshape[1] - 1].flatten())
        s_counts = np.bincount(self.ell_mat()[0:self.shape[0] // 2 + 1, [-1, 0]].flatten(),weights=weights[0:self.shape[0] // 2 + 1, [-1, 0]].flatten())
        counts[0:len(s_counts)] += s_counts
        return counts

    def alm2Pk_minimal(self, alm):
        """Power spectrum estimation on the grid, with minimal binning (only exact same frequencies).

            Many bins will have only number count 4 or something.

            Returns:
                list with integer array of squared frequencies, integer array number counts, and Pk estimates.
                (only non zero counts frequencies)

        """
        almmap = self.alm2almmap(alm)
        assert len(almmap.shape) == 2
        assert self.ell_mat.lsides[0] == self.ell_mat.lsides[1], self.ell_mat.lsides
        assert 2 * (almmap.shape[1] - 1) == almmap.shape[0], 'Only for square maps'

        N0, N1 = almmap.shape
        almmap = almmap.real.flatten() ** 2 + almmap.imag.flatten() ** 2

        l02 = Freq(np.arange(N0, dtype=int), N0) ** 2
        l12 = Freq(np.arange(N1, dtype=int), 2 * (N1 - 1)) ** 2
        sqd_freq = (np.outer(l02, np.ones(N1, dtype=int)) + np.outer(np.ones(N0, dtype=int), l12)).flatten()
        counts = np.bincount(sqd_freq)

        # The following frequencies have their opposite in the map and should not be double counted.
        for i in sqd_freq.reshape(N0, N1)[N0 / 2 + 1:, [-1, 0]]: counts[i] -= 1
        Pk = np.bincount(sqd_freq, weights=almmap)
        sqd_freq = np.where(counts > 0)[0]  # This is the output array with the squared (unnormalised) frequencies
        kmin = 2. * np.pi / self.ell_mat.lsides[0]
        return np.sqrt(sqd_freq) * kmin, counts[sqd_freq], Pk[sqd_freq] / counts[sqd_freq]

    def bicubic_prefilter(self, alm):
        """Refilter the map for use in bicubic spline prefiltering.

            Mostly useful for lensing of maps.

        """
        assert alm.size == self.alm_size, (alm.size, self.alm_size)
        s0, s1 = self.ell_mat.shape
        r0, r1 = self.ell_mat.rshape

        w0 = 6. / (2. * np.cos(2. * np.pi * Freq(np.arange(r0), s0) / s0) + 4.)
        w1 = 6. / (2. * np.cos(2. * np.pi * Freq(np.arange(r1), s1) / s1) + 4.)
        return alm * self.almmap2alm(np.outer(w0, w1))

    def get_kx(self):
        return self.ell_mat.get_kx_mat()[self._cond()]

    def get_ky(self):
        return self.ell_mat.get_ky_mat()[self._cond()]

    def get_ikx(self):
        return self.ell_mat.get_ikx_mat()[self._cond()]

    def get_iky(self):
        return self.ell_mat.get_iky_mat()[self._cond()]

    def get_cossin_2iphi(self):
        cos, sin = self.ell_mat.get_cossin_2iphi_mat()
        return cos[self._cond()], sin[self._cond()]

    def alm2rlm(self, alm):
        if alm.size == self.alm_size:
            return np.concatenate((alm.real, alm.imag))
        elif alm.size%self.alm_size == 0: # probably just concatenated
            na = alm.size // self.alm_size
            s = self.alm_size
            return np.concatenate([self.alm2rlm(alm[a * s: (a + 1) * s]) for a in range(na)])
        else:
            assert 0, ('dont know what to do', alm.size, self.alm_size)

    def rlm2alm(self, rlm):
        # ! still contains redundant information
        if rlm.size == 2 * self.alm_size:
            return rlm[0:self.alm_size] + 1j * rlm[self.alm_size:]
        assert rlm.size % (2 * self.alm_size) == 0
        na = rlm.size // (2 * self.alm_size)
        s = self.alm_size
        return np.concatenate([self.rlm2alm(rlm[a * 2 * s: (a + 1) * 2 * s]) for a in range(na)])

    def alms2rlms(self, alms):
        assert alms.ndim == 2 and np.all(alm.size == self.alm_size for alm in alms), alms.shape
        rlms = np.zeros((alms.shape[0] * 2 * self.alm_size,))
        for i in range(alms.shape[0]):
            rlms[i * (2 * self.alm_size): (i + 1) * 2 * self.alm_size] = self.alm2rlm(alms[i])
        return rlms

    def rlms2alms(self, rlms):
        assert rlms.ndim == 1 and rlms.size % (2 * self.alm_size) == 0, rlms.shape
        alms = np.zeros((rlms.size / (2 * self.alm_size), self.alm_size), dtype=complex)
        for i in range(alms.shape[0]):
            alms[i, :] = self.rlm2alm(rlms[i * (2 * self.alm_size):(i + 1) * 2 * self.alm_size])
        return alms

    def write_alm(self, fname, alm):
        assert alm.size == self.alm_size
        np.save(fname, alm)

    def read_alm(self, fname):
        alm = np.load(fname)
        assert alm.size == self.alm_size
        return alm

    def map2cl(self, m1, m2=None, ellmax=None):
        """Returns power spectrum of alm array

        """
        return self.alm2cl(self.map2alm(m1),alm2= None if m2 is None else self.map2alm(m2))[:(ellmax or self.ellmax) +1]

    def alm2cl(self, alm, alm2=None, ellmax=None):
        """Returns power spectrum of alm array

        """
        assert alm.size == self.alm_size, (alm.size, self.alm_size)
        if alm2 is not None: assert alm2.size == self.alm_size, (alm2.size, self.alm_size)
        return self.bin_realpart_inell(np.abs(alm) ** 2 if alm2 is None else (alm * np.conjugate(alm2)).real,ellmax=ellmax)

    def bin_realpart_inell(self,alm,ellmax = None):
        """Bin real part of a rfftmap according to multipole :math:`\ell`

            Input must have the same shape then self.rshape

        """
        #FIXME: dont go to full rfft
        assert alm.size == self.alm_size, (alm.size, self.alm_size)
        weights = self.alm2almmap(alm).real
        Cl = np.bincount(self.ell_mat()[:, 1:self.ell_mat.rshape[1] - 1].flatten(),
                         weights=weights[:, 1:self.ell_mat.rshape[1] - 1].flatten(), minlength=self.ell_mat.ellmax + 1)
        Cl += np.bincount(self.ell_mat()[0:self.shape[0] // 2 + 1, [-1, 0]].flatten(),
                          weights=weights[0:self.shape[0] // 2 + 1, [-1, 0]].flatten(), minlength=self.ell_mat.ellmax + 1)
        Cl[np.where(self._get_ell_counts())] /= self._get_ell_counts()[np.where(self._get_ell_counts())]
        return Cl[:(ellmax or self.ellmax) + 1]

    def udgrade(self, lib_alm, alm):
        """Degrades or upgrades a alm vector from this *ffs_alm* instance to another instance.

        """
        # FIXME : high time to devise a better flat sky alm scheme, this one becomes fairly convoluted.
        if self.iseq(lib_alm, allow_shape=True): return alm
        assert alm.size == lib_alm.alm_size, (alm.size, lib_alm.alm_size)
        assert self.ell_mat.lsides == lib_alm.ell_mat.lsides  # Must have same frequencies in the map.
        return self.almmap2alm(udgrade_rfft2(lib_alm.alm2almmap(alm), self.ell_mat.shape))

    def QUlms2EBalms(self, QUlms):
        """ Turns QU alms list into EB alms.

        """
        assert QUlms.shape == (2, self.alm_size), QUlms.shape
        cos, sin = self.get_cossin_2iphi()
        return np.array([cos * QUlms[0] + sin * QUlms[1], -sin * QUlms[0] + cos * QUlms[1]])

    def TQUlms2TEBalms(self, TQUlms):
        """Turns TQU alms list into TEB alms.

        """
        assert TQUlms.shape == (3, self.alm_size), TQUlms.shape
        cos, sin = self.get_cossin_2iphi()
        return np.array([TQUlms[0], cos * TQUlms[1] + sin * TQUlms[2], -sin * TQUlms[1] + cos * TQUlms[2]])

    def EBlms2QUalms(self, EBlms):
        """Turns EB alms list into QU alms.

        """
        assert EBlms.shape == (2, self.alm_size), EBlms.shape
        cos, sin = self.get_cossin_2iphi()
        return np.array([cos * EBlms[0] - sin * EBlms[1], sin * EBlms[0] + cos * EBlms[1]])

    def TEBlms2TQUalms(self, TEBlms):
        """Turns TEB alms list into TQU alms.

        """
        assert TEBlms.shape == (3, self.alm_size), TEBlms.shape
        cos, sin = self.get_cossin_2iphi()
        return np.array([TEBlms[0], cos * TEBlms[1] - sin * TEBlms[2], sin * TEBlms[1] + cos * TEBlms[2]])


class ffs_alm_pyFFTW(ffs_alm):
    r"""Same as ffs_alm but with pyfftw fft-engine threaded fftw library (up to 10 times faster than numpy's).

    """

    def __init__(self, ellmat, filt_func=lambda ell: ell > 0, num_threads=int(os.environ.get('OMP_NUM_THREADS', 1)), flags_init=('FFTW_MEASURE',)):
        super(ffs_alm_pyFFTW, self).__init__(ellmat, filt_func=filt_func)
        self.flags = flags_init
        self.threads = num_threads

    def alm2rfft(self, alm):
        assert alm.size == self.alm_size, alm.size
        ret = pyfftw.zeros_aligned(self.ell_mat.rshape, dtype='complex128')
        ret[self._cond()] = alm * self.fac_alm2rfft
        return ret

    def map2rfft(self, _map):
        inpt = pyfftw.empty_aligned(self.ell_mat.shape, dtype='float64')
        oupt = pyfftw.empty_aligned(self.ell_mat.rshape, dtype='complex128')
        fft = pyfftw.FFTW(inpt, oupt, axes=(0, 1), direction='FFTW_FORWARD', flags=self.flags, threads=self.threads)
        return fft(pyfftw.byte_align(_map, dtype='float64'))

    def alm2map(self, alm, lib_almout=None):
        assert alm.size == self.alm_size, (alm.size, self.alm_size)
        if lib_almout is None:
            oupt = pyfftw.empty_aligned(self.ell_mat.shape, dtype='float64')
            inpt = pyfftw.empty_aligned(self.ell_mat.rshape, dtype='complex128')

            ifft = pyfftw.FFTW(inpt, oupt, axes=(0, 1), direction='FFTW_BACKWARD', flags=self.flags, threads=self.threads)
            return ifft(pyfftw.byte_align(self.alm2rfft(alm), dtype='complex128'))
        else:
            return lib_almout.alm2map(lib_almout.udgrade(self, alm))

    def clone(self):
        return ffs_alm_pyFFTW(self.ell_mat, filt_func=self.filt_func, num_threads=self.threads)

    def degrade(self, LD_shape, ellmax=None, ellmin=None, num_threads=None):
        LD_ellmat = self.ell_mat.degrade(LD_shape)
        if ellmax is None: ellmax = self.ellmax
        if ellmin is None: ellmin = self.ellmin
        filt_func = lambda ell: (self.filt_func(ell) & (ell <= ellmax) & (ell >= ellmin))
        num_threads = self.threads if num_threads is None else num_threads
        return ffs_alm_pyFFTW(LD_ellmat, filt_func=filt_func, num_threads=num_threads)
