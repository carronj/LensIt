import numpy as np
import os
import pickle as pk
import lensit as fs
from lensit.sims import sims_generic
from lensit.sims.sims_generic import hash_check
import hashlib
from lensit import pbs

try:
    import pyfftw
except:
    print "-- NB : import of pyfftw unsucessful -- "


def Freq(i, N):
    """
     Outputs the absolute integers frequencies [0,1,...,N/2,N/2-1,...,1]
     in numpy fft convention as integer i runs from 0 to N-1.
     Inputs can be numpy arrays e.g. i (i1,i2,i3) with N (N1,N2,N3)
                                  or i (i1,i2,...) with N
     Both inputs must be integers.
     All entries of N must be even.
    """
    assert np.all(N % 2 == 0), "This routine only for even numbers of points"
    return i - 2 * (i >= (N / 2)) * (i % (N / 2))


class ell_mat():
    """
    Class to setup and handle Fourier mode structure on the flat sky, at the given resolution and size of
    the rectangular box. Each 2d vector kx,ky is assigned to a multipole ell according to ell = int(|k| - 1/2).
    Caches various matrices to facilitate multiple calls.
    """

    def __init__(self, lib_dir, shape, lsides):
        assert len(shape) == 2 and len(lsides) == 2
        assert shape[0] % 2 == 0 and shape[1] % 2 == 0
        assert shape[0] < 2 ** 16 and shape[1] < 2 ** 16
        self.shape = tuple(shape)
        self.rshape = (shape[0], shape[1] / 2 + 1)
        self.lsides = tuple(lsides)
        self.lib_dir = lib_dir
        # Dumping ell mat in lib_dir. Overwrites if already present.

        if pbs.rank == 0:
            if not os.path.exists(lib_dir): os.makedirs(lib_dir)
            if not os.path.exists(lib_dir + "/ellmat_hash.pk"):
                pk.dump(self.hash_dict(), open(lib_dir + "/ellmat_hash.pk", 'w'))
        pbs.barrier()

        hash_check(pk.load(open(lib_dir + "/ellmat_hash.pk", 'r')), self.hash_dict())

        if pbs.rank == 0 and not os.path.exists(self.lib_dir + '/ellmat.npy'):
                print 'ell_mat:caching ells in ', self.lib_dir + '/ellmat.npy'
                np.save(self.lib_dir + '/ellmat.npy', self._build_ellmat())
        pbs.barrier()
        # FIXME
        self.ellmax = int(self._get_ellmax())
        self._ell_counts = self._build_ell_counts()
        self._nz_counts = self._ell_counts.nonzero()

    def _build_ellmat(self):
        kmin = 2. * np.pi / np.array(self.lsides)
        ky2 = Freq(np.arange(self.shape[0]), self.shape[0]) ** 2 * kmin[0] ** 2
        kx2 = Freq(np.arange(self.rshape[1]), self.shape[1]) ** 2 * kmin[1] ** 2
        ones = np.ones(np.max(self.shape))
        return self.k2ell(np.sqrt(np.outer(ky2, ones[0:self.rshape[1]]) + np.outer(ones[0:self.rshape[0]], kx2)))

    def hash_dict(self):
        return {'shape': self.shape, 'lsides': self.lsides}

    def Nyq(self, axis):
        assert axis in [0, 1], axis
        return np.pi / self.lsides[axis] * self.shape[axis]

    def k2ell(self, k):
        ret = np.uint16(np.round(k - 0.5) + 0.5 * ((k - 0.5) < 0))
        return ret

    def pbssafe_save(self, fname, data_to_save, pbs_rank=None):
        if pbs_rank is not None and pbs.rank != pbs_rank:
            return
        np.save(fname, data_to_save)
        return

    def check_compatible(self, ellmat):
        hash_check(self.hash_dict(), ellmat.hash_dict())


    def __call__(self, *args, **kwargs):
        return self.get_ellmat(*args, **kwargs)

    def __getitem__(self, item):
        return self.get_ellmat()[item]

    def get_ellmat(self, ellmax=None):
        """
        Returns the matrix containing the multipole ell assigned to k = (kx,ky)
        """
        if ellmax is None:
            return np.load(self.lib_dir + '/ellmat.npy', mmap_mode='r')
        else:
            fname = self.lib_dir + '/ellmat_ellmax%s.npy' % ellmax
            if os.path.exists(fname): return np.load(fname, mmap_mode='r')
            if pbs.rank == 0:
                print 'ell_mat:caching ells in ', fname
                np.save(fname, self.get_ellmat()[np.where(self.get_ellmat() <= ellmax)])
            pbs.barrier()
            return np.load(fname, mmap_mode='r')

    def get_phasemat(self, ellmax=None):
        """
        Returns the matrix containing the phase k = 'k' e^i phi
        """
        if ellmax is None:
            fname = self.lib_dir + '/phasemat.npy'
            if os.path.exists(fname): return np.load(fname, mmap_mode='r')
            if not os.path.exists(fname) and pbs.rank == 0:
                print 'ell_mat:caching phases in ', fname
                np.save(fname, np.arctan2(self.get_ky_mat(), self.get_kx_mat()))
            pbs.barrier()
            return np.load(fname, mmap_mode='r')
        else:
            fname = self.lib_dir + '/phase_ellmax%s.npy' % ellmax
            if not os.path.exists(fname) and pbs.rank == 0:
                print 'ell_mat:caching phases in ', fname
                np.save(fname, np.arctan2(self.get_ky_mat(), self.get_kx_mat())[np.where(self.get_ellmat() <= ellmax)])
            pbs.barrier()
            return np.load(fname, mmap_mode='r')

    def get_e2iphi_mat(self, cache_only=False):
        """
        Built such that it should hit the pbs barrier only if the matrix is not already cached.
        """
        fname = self.lib_dir + '/e2iphimat.npy'
        if os.path.exists(fname):
            return None if cache_only else np.load(fname, mmap_mode='r')
        if not os.path.exists(fname) and pbs.rank == 0:
            print 'ell_mat:caching e2iphi in ', fname
            np.save(fname, np.exp(2j * np.arctan2(self.get_ky_mat(), self.get_kx_mat())))
        pbs.barrier()
        return None if cache_only else np.load(fname, mmap_mode='r')

    def degrade(self, LDshape, lib_dir=None):
        if np.all(LDshape >= self.shape): return self
        if lib_dir is None: lib_dir = self.lib_dir + '/degraded%sx%s' % (LDshape[0], LDshape[1])
        return ell_mat(lib_dir, LDshape, self.lsides)

    def get_cossin_2iphi_mat(self):
        e2iphi = self.get_e2iphi_mat()
        return e2iphi.real, e2iphi.imag

    def filt_map(self, map, ellmin, ellmax):
        assert map.shape == self.shape, map.shape
        rfftmap = self.filt_rfftmap_high(self.filt_rfftmap_low(np.fft.rfft2(map), ellmin), ellmax)
        return np.fft.irfft2(rfftmap, map.shape)

    def filt_map_low(self, map, ellmin):
        assert map.shape == self.shape, map.shape
        return np.fft.irfft2(self.filt_rfftmap_low(np.fft.rfft2(map), ellmin), map.shape)

    def filt_map_high(self, map, ellmax):
        assert map.shape == self.shape, map.shape
        return np.fft.irfft2(self.filt_rfftmap_high(np.fft.rfft2(map), ellmax), map.shape)

    def filt_rfftmap(self, rfftmap, ellmin, ellmax):
        assert rfftmap.shape == self.rshape, rfftmap.shape
        return self.filt_rfftmap_high(self.filt_rfftmap_low(rfftmap, ellmin), ellmax)

    def filt_rfftmap_low(self, rfftmap, ellmin):
        assert rfftmap.shape == self.rshape, rfftmap.shape
        return np.where(self.get_ellmat() < ellmin, np.zeros(rfftmap.shape), rfftmap)

    def filt_rfftmap_high(self, rfftmap, ellmax):
        assert rfftmap.shape == self.rshape, rfftmap.shape
        return np.where(self.get_ellmat() > ellmax, np.zeros(rfftmap.shape), rfftmap)

    def _get_ellmax(self):
        """ Max. ell present in the grid """
        return np.max(self.get_ellmat())

    def _build_ell_counts(self):
        """ Number of entries in freq map. for each ell, in the rfftmap. Corresponds roughly to ell + 1/2."""
        counts = np.bincount(self.get_ellmat()[:, 1:self.rshape[1] - 1].flatten(), minlength=self.ellmax + 1)
        s_counts = np.bincount(self.get_ellmat()[0:self.shape[0] / 2 + 1, [-1, 0]].flatten())
        counts[0:len(s_counts)] += s_counts
        return counts

    def get_ellcounts(self):
        # This is in fact close to ell + 1/2 and not 2ell + 1.
        return self._ell_counts

    def get_Nell(self):
        # Analog of 2ell + 1 on the flat sky
        Nell = 2 * self.get_ellcounts()
        for ell in self.get_ellmat()[self.rfft2_reals()]:
            Nell[ell] -= 1
        return Nell

    def get_nonzero_ellcounts(self):
        return self._nz_counts

    def map2cl(self, map, map2=None):
        """
        Returns Cl estimates from a map.
        Included in self for convenience.
        Must have the same shape than self.shape.
        """
        assert map.shape == self.shape, map.shape
        if map2 is not None: assert map2.shape == self.shape, map2.shape
        return self.rfft2cl(np.fft.rfft2(map)) if map2 is None else \
            self.rfft2cl(np.fft.rfft2(map), rfftmap2=np.fft.rfft2(map2))

    def rfft2cl(self, rfftmap, rfftmap2=None):
        """
        Returns Cl estimates from a rfftmap.
        (e.g. np.fft.rfft2(sim) where sim is the output of this library)
        Included in self for convenience.
        Must have the same shape then self.rshape
        """
        assert rfftmap.shape == self.rshape, rfftmap.shape
        if rfftmap2 is not None: assert rfftmap2.shape == self.rshape, rfftmap2.shape
        weights = rfftmap.real ** 2 + rfftmap.imag ** 2 if rfftmap2 is None else (rfftmap * np.conjugate(rfftmap2)).real
        Cl = np.bincount(self.get_ellmat()[:, 1:self.rshape[1] - 1].flatten(),
                         weights=weights[:, 1:self.rshape[1] - 1].flatten(), minlength=self.ellmax + 1)
        Cl += np.bincount(self.get_ellmat()[0:self.shape[0] / 2 + 1, [-1, 0]].flatten(),
                          weights=weights[0:self.shape[0] / 2 + 1, [-1, 0]].flatten(), minlength=self.ellmax + 1)
        Cl[self._nz_counts] *= (np.prod(self.lsides) / np.prod(self.shape) ** 2) / self._ell_counts[self._nz_counts]
        return Cl

    def bin_inell(self, rfftmap):
        """
        Included in self for convenience.
        Must have the same shape then self.rshape
        """
        assert rfftmap.shape == self.rshape, rfftmap.shape
        weights = rfftmap
        Cl = np.bincount(self.get_ellmat()[:, 1:self.rshape[1] - 1].flatten(),
                         weights=weights[:, 1:self.rshape[1] - 1].flatten(), minlength=self.ellmax + 1)
        Cl += np.bincount(self.get_ellmat()[0:self.shape[0] / 2 + 1, [-1, 0]].flatten(),
                          weights=weights[0:self.shape[0] / 2 + 1, [-1, 0]].flatten(), minlength=self.ellmax + 1)
        Cl[self._nz_counts] /= self._ell_counts[self._nz_counts]
        return Cl

    def get_rx_mat(self):
        rx_min = self.lsides[1] / self.shape[1]
        rx = rx_min * Freq(np.arange(self.shape[1]), self.shape[1])
        rx[self.shape[1] / 2:] *= -1.
        return np.outer(np.ones(self.shape[0]), rx)

    def get_kx_mat(self):
        kx_min = (2. * np.pi) / self.lsides[1]
        kx = kx_min * Freq(np.arange(self.rshape[1]), self.shape[1])
        return np.outer(np.ones(self.shape[0]), kx)

    def get_ry_mat(self):
        ry_min = self.lsides[0] / self.shape[0]
        ry = ry_min * Freq(np.arange(self.shape[0]), self.shape[0])
        ry[self.shape[0] / 2:] *= -1.
        return np.outer(ry, np.ones(self.shape[1]))

    def get_ky_mat(self):
        ky_min = (2. * np.pi) / self.lsides[0]
        ky = ky_min * Freq(np.arange(self.shape[0]), self.shape[0])
        ky[self.shape[0] / 2:] *= -1.
        return np.outer(ky, np.ones(self.rshape[1]))

    def get_ikx_mat(self):
        return 1j * self.get_kx_mat()

    def get_iky_mat(self):
        return 1j * self.get_ky_mat()

    def get_unique_ells(self, return_index=False, return_inverse=False, return_counts=False):
        return np.unique(self.get_ellmat(),
                         return_index=return_index, return_inverse=return_inverse, return_counts=return_counts)

    def rfftmap2alm(self, rfftmap, filt_func=lambda ell: True):
        """
        Returns a ffs_alm array from a rfftmap.
        :param rfftmap:
        :return:
        """
        cond = filt_func(np.arange(self.ellmax + 1))
        return rfftmap[cond[self.get_ellmat()]] * np.sqrt(np.prod(self.lsides)) / np.prod(self.shape)

    def alm2rfftmap(self, alm, filt_func=lambda ell: True):
        """
        Returns a ffs_alm array from a rfftmap.
        :param rfftmap:
        :return:
        """
        cond = filt_func(np.arange(self.ellmax + 1))
        ret = np.zeros(self.rshape, dtype=complex)
        ret[cond[self.get_ellmat()]] = alm * np.prod(self.shape) / np.sqrt(np.prod(self.lsides))
        return ret

    def rfft2_reals(self):
        """ Pure reals modes in 2d rfft array. (from real map shape, not rfft array) """
        N0, N1 = self.shape
        fx = [0]
        fy = [0]
        if N1 % 2 == 0: fx.append(0); fy.append(N1 / 2)
        if N0 % 2 == 0: fx.append(N0 / 2); fy.append(0)
        if N1 % 2 == 0 and N0 % 2 == 0: fx.append(N0 / 2); fy.append(N1 / 2)
        return np.array(fx), np.array(fy)

    def QU2EBrfft(self, QUmap):
        """
        Turns QU map list into EB rfft maps.
        """
        assert QUmap[0].shape == self.shape and QUmap[1].shape == self.shape
        Qk = np.fft.rfft2(QUmap[0])
        Uk = np.fft.rfft2(QUmap[1])
        cos, sin = self.get_cossin_2iphi_mat()
        return cos * Qk + sin * Uk, -sin * Qk + cos * Uk


class ffs_alm(object):
    """
    Simple-minded library to facilitate filtering operations on flat-sky alms in the ffs scheme.
    """

    def __init__(self, ellmat, filt_func=lambda ell: ell > 0):
        """
        :param ellmat: ell_mat instance that defines the mode structure in the ffs scheme.
        :param filt: is callable with Boolean return. filt(ell) tells whether or not mode ell is considered
                or filtered away.
        """
        self.ell_mat = ellmat
        self.shape = self.ell_mat.shape
        self.lsides = self.ell_mat.lsides
        self.filt_func = filt_func
        self.cond = filt_func(np.arange(self.ell_mat.ellmax + 1))
        self.alm_size = np.count_nonzero(self.cond[self.ell_mat()])
        # The mapping ell[i] for i in alm array :
        self.reduced_ellmat = lambda: ellmat()[self.cond[ellmat()]]
        self.ellmax = np.max(self.reduced_ellmat())
        self.ellmin = np.min(self.reduced_ellmat())
        # Some trivial convenience factors :
        self.fac_rfft2alm = np.sqrt(np.prod(ellmat.lsides)) / np.prod(self.ell_mat.shape)
        self.fac_alm2rfft = 1. / self.fac_rfft2alm
        # assert self.ellmax < ellmat()[0, -1], (self.ellmax, ellmat()[0, -1])  # Dont want to deal with redundant frequencies

    def __eq__(self, lib_alm):
        if not np.all(self.ell_mat.lsides == lib_alm.ell_mat.lsides):
            return False
        if not np.all(self.ell_mat.shape == lib_alm.ell_mat.shape):
            return False
        ellmax = max(self.ellmax, lib_alm.ellmax)
        if not np.all(self.filt_func(np.arange(ellmax + 1)) == lib_alm.filt_func(np.arange(ellmax + 1))):
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
        return True

    def hashdict(self):
        return {'ellmat': self.ell_mat.hash_dict(),
                'filt_func': hashlib.sha1(self.cond).hexdigest()}

    def degrade(self, LD_shape, ellmax=None, ellmin=None):
        LD_ellmat = self.ell_mat.degrade(LD_shape)
        if ellmax is None: ellmax = self.ellmax
        if ellmin is None: ellmin = self.ellmin
        filt_func = lambda ell: (self.filt_func(ell) & (ell <= ellmax) & (ell >= ellmin))
        return ffs_alm(LD_ellmat, filt_func=filt_func)

    def fsky(self):
        return np.prod(self.ell_mat.lsides) / 4. / np.pi

    def filt_hash(self):
        return hashlib.sha1(self.cond).hexdigest()

    def clone(self):
        return ffs_alm(self.ell_mat, filt_func=self.filt_func)

    def nbar(self):
        return np.prod(self.ell_mat.shape) / np.prod(self.ell_mat.lsides)

    def rfftmap2alm(self, rfftmap):
        assert rfftmap.shape == self.ell_mat.rshape, rfftmap.shape
        return self.fac_rfft2alm * rfftmap[self.cond[self.ell_mat()]]

    def almmap2alm(self, almmap):
        assert almmap.shape == self.ell_mat.rshape, almmap.shape
        return almmap[self.cond[self.ell_mat()]]

    def map2rfft(self, _map):
        return np.fft.rfft2(_map)

    def map2alm(self, _map, lib_almin=None):
        if lib_almin is None or self.shape == lib_almin.shape:
            return self.rfftmap2alm(self.map2rfft(_map))
        else:
            return self.rfftmap2alm(self.map2rfft(fs.misc.rfft2_utils.supersample(_map, lib_almin.ell_mat.shape)))

    def alm2rfft(self, alm):
        assert alm.size == self.alm_size, alm.size
        ret = np.zeros(self.ell_mat.rshape, dtype=complex)
        ret[self.cond[self.ell_mat()]] = alm * self.fac_alm2rfft
        return ret

    def alm2almmap(self, alm):
        assert alm.size == self.alm_size, alm.size
        ret = np.zeros(self.ell_mat.rshape, dtype=complex)
        ret[self.cond[self.ell_mat()]] = alm
        return ret

    def alm2map(self, alm, lib_almout=None):
        if lib_almout is None:
            assert alm.size == self.alm_size, alm.size
            return np.fft.irfft2(self.alm2rfft(alm), self.ell_mat.shape)
        else:
            return lib_almout.alm2map(lib_almout.udgrade(self, alm))

    def almxfl(self, alm, fl, inplace=False):
        assert alm.size == self.alm_size, (alm.size, self.alm_size)
        assert len(fl) > self.ellmax
        if inplace:
            alm *= fl[self.reduced_ellmat()]
            return
        else:
            return alm * fl[self.reduced_ellmat()]

    def get_unique_ells(self, return_index=False, return_inverse=False, return_counts=False):
        return np.unique(self.reduced_ellmat(),
                         return_index=return_index, return_inverse=return_inverse, return_counts=return_counts)

    def get_ellcounts(self):
        return self.ell_mat.get_ellcounts() * self.cond

    def get_Nell(self):
        return self.ell_mat.get_Nell() * self.cond

    def alm2Pk_minimal(self, alm):
        """
        Overkill power spectrum estimation on the grid, with minimal binning (only exact same frequencies).
        Many bins will have only number count 4 or something.
        Outputs a list with integer array of squared frequencies,integer array number counts, and Pk estimates.
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
        """
        Prefilter the map for use in bicubic spline prefiltering. Useful for lensing of maps.
        """
        assert alm.size == self.alm_size, (alm.size, self.alm_size)
        s0, s1 = self.ell_mat.shape
        r0, r1 = self.ell_mat.rshape

        w0 = 6. / (2. * np.cos(2. * np.pi * Freq(np.arange(r0), s0) / s0) + 4.)
        w1 = 6. / (2. * np.cos(2. * np.pi * Freq(np.arange(r1), s1) / s1) + 4.)
        return alm * self.almmap2alm(np.outer(w0, w1))

    def get_kx(self):
        return self.ell_mat.get_kx_mat()[self.cond[self.ell_mat()]]

    def get_ky(self):
        return self.ell_mat.get_ky_mat()[self.cond[self.ell_mat()]]

    def get_ikx(self):
        return self.ell_mat.get_ikx_mat()[self.cond[self.ell_mat()]]

    def get_iky(self):
        return self.ell_mat.get_iky_mat()[self.cond[self.ell_mat()]]

    def get_cossin_2iphi(self):
        cos, sin = self.ell_mat.get_cossin_2iphi_mat()
        return cos[self.cond[self.ell_mat()]], sin[self.cond[self.ell_mat()]]

    def alm2rlm(self, alm):
        assert alm.size == self.alm_size, alm.size
        return np.concatenate((alm.real, alm.imag))

    def rlm2alm(self, rlm):
        # ! still contains redundant information
        assert rlm.size == 2 * self.alm_size, rlm.size
        return rlm[0:self.alm_size] + 1j * rlm[self.alm_size:]

    def alms2rlms(self, alms):
        assert alms.ndim == 2 and np.all(alm.size == self.alm_size for alm in alms), alms.shape
        rlms = np.zeros((alms.shape[0] * 2 * self.alm_size,))
        for i in xrange(alms.shape[0]):
            rlms[i * (2 * self.alm_size): (i + 1) * 2 * self.alm_size] = self.alm2rlm(alms[i])
        return rlms

    def rlms2alms(self, rlms):
        assert rlms.ndim == 1 and rlms.size % (2 * self.alm_size) == 0, rlms.shape
        alms = np.zeros((rlms.size / (2 * self.alm_size), self.alm_size), dtype=complex)
        for i in xrange(alms.shape[0]):
            alms[i, :] = self.rlm2alm(rlms[i * (2 * self.alm_size):(i + 1) * 2 * self.alm_size])
        return alms

    def write_alm(self, fname, alm):
        assert alm.size == self.alm_size
        np.save(fname, alm)

    def read_alm(self, fname):
        alm = np.load(fname)
        assert alm.size == self.alm_size
        return alm

    def map2cl(self, map1, map2=None, ellmax=None):
        ellmax = ellmax or self.ellmax
        return self.ell_mat.map2cl(map1, map2=map2)[:ellmax + 1]

    def alm2cl(self, alm, alm2=None, ellmax=None):
        ellmax = ellmax or self.ellmax
        assert alm.size == self.alm_size, (alm.size, self.alm_size)
        if alm2 is None:
            return self.ell_mat.rfft2cl(self.alm2rfft(alm))[0:ellmax + 1]
        assert alm2.size == self.alm_size, (alm.size, self.alm_size)
        return self.ell_mat.rfft2cl(self.alm2rfft(alm), rfftmap2=self.alm2rfft(alm2))[0:ellmax + 1]

    def bin_realpart_inell(self, alm):
        return self.ell_mat.bin_inell(self.alm2almmap(alm).real)[0:self.ellmax + 1]

    def udgrade(self, lib_alm, alm):
        """
        Degrades or upgrades a alm vector from lib_alm to the self lib_alm
        :param lib_alm:
        :param alm:
        :return:
        """
        # FIXME : high time to devise a better flat sky alm scheme, this one becomes fairly convoluted.
        if self.iseq(lib_alm, allow_shape=True): return alm
        assert alm.size == lib_alm.alm_size, (alm.size, lib_alm.alm_size)
        assert self.ell_mat.lsides == lib_alm.ell_mat.lsides  # Must have same frequencies in the map.
        return self.almmap2alm(fs.misc.rfft2_utils.udgrade_rfft2(lib_alm.alm2almmap(alm), self.ell_mat.shape))

    def QUlms2EBalms(self, QUlms):
        """
        Turns QU alms list into EB alms.
        """
        assert QUlms.shape == (2, self.alm_size), QUlms.shape
        cos, sin = self.get_cossin_2iphi()
        return np.array([cos * QUlms[0] + sin * QUlms[1], -sin * QUlms[0] + cos * QUlms[1]])

    def TQUlms2TEBalms(self, TQUlms):
        """
        Turns TQU alms list into TEB alms.
        """
        assert TQUlms.shape == (3, self.alm_size), TQUlms.shape
        cos, sin = self.get_cossin_2iphi()
        return np.array([TQUlms[0], cos * TQUlms[1] + sin * TQUlms[2], -sin * TQUlms[1] + cos * TQUlms[2]])

    def EBlms2QUalms(self, EBlms):
        """
        Turns EB alms list into QU alms.
        """
        assert EBlms.shape == (2, self.alm_size), EBlms.shape
        cos, sin = self.get_cossin_2iphi()
        return np.array([cos * EBlms[0] - sin * EBlms[1], sin * EBlms[0] + cos * EBlms[1]])

    def TEBlms2TQUalms(self, TEBlms):
        """
        Turns EB alms list into QU alms.
        """
        assert TEBlms.shape == (3, self.alm_size), TEBlms.shape
        cos, sin = self.get_cossin_2iphi()
        return np.array([TEBlms[0], cos * TEBlms[1] - sin * TEBlms[2], sin * TEBlms[1] + cos * TEBlms[2]])

    def extend(self, filt_func):
        return ffs_alm(self.ell_mat, filt_func=filt_func)


class ffs_lib_phas(sims_generic.sim_lib):
    """
    Sim lib for Gaussian phases (Cl == 1 alm vectors for ell range set by lib_alm,
    in the alm scheme as given by lib_alm).
    """

    def __init__(self, lib_dir, nfields, lib_alm, **kwargs):
        self.lib_dir = lib_dir
        self.lib_alm = lib_alm
        self.nfields = nfields
        super(ffs_lib_phas, self).__init__(lib_dir, **kwargs)

    def _build_sim_from_rng(self, rng_state, phas_only=False):
        np.random.set_state(rng_state)
        alms = np.array([(np.random.standard_normal(self.lib_alm.alm_size) +
                          1j * np.random.standard_normal(self.lib_alm.alm_size)) / np.sqrt(2.) for i in
                         xrange(self.nfields)])
        if phas_only: return
        # Reality conditions on the rfft maps
        sla = slice(self.lib_alm.ell_mat.shape[0] / 2 + 1, self.lib_alm.ell_mat.shape[0], 1)
        slb = slice(self.lib_alm.ell_mat.shape[0] / 2 - 1, 0, -1)

        for i in xrange(self.nfields):
            rfft = self.lib_alm.alm2rfft(alms[i])
            rfft[sla, [-1, 0]] = np.conjugate(rfft[slb, [-1, 0]])
            rfft.real[self.lib_alm.ell_mat.rfft2_reals()] *= np.sqrt(2.)
            rfft.imag[self.lib_alm.ell_mat.rfft2_reals()] = 0.
            alms[i] = self.lib_alm.rfftmap2alm(rfft)
        return alms

    def hashdict(self):
        return {'nfields': self.nfields, 'lib_alm': self.lib_alm.hashdict()}


class ffs_alm_pyFFTW(ffs_alm):
    """
    Same as ffs_alm but ffts are done with much faster (10 x !) threaded fftw library.
    """

    def __init__(self, ellmat, filt_func=lambda ell: ell > 0, num_threads=4, flags_init=('FFTW_MEASURE',)):
        super(ffs_alm_pyFFTW, self).__init__(ellmat, filt_func=filt_func)
        # FIXME : This can be tricky in in hybrid MPI-OPENMP
        # Builds FFTW Wisdom :
        wisdom_fname = self.ell_mat.lib_dir + '/FFTW_wisdom_%s_%s.npy' % (num_threads, ''.join(flags_init))
        if not os.path.exists(wisdom_fname):
            print "++ ffs_alm_pyFFTW :: building and caching FFTW wisdom, this might take a little while..."
            if pbs.rank == 0:
                inpt = pyfftw.empty_aligned(self.ell_mat.shape, dtype='float64')
                oupt = pyfftw.empty_aligned(self.ell_mat.rshape, dtype='complex128')
                fft = pyfftw.FFTW(inpt, oupt, axes=(0, 1), direction='FFTW_FORWARD', flags=flags_init,
                                  threads=num_threads)
                ifft = pyfftw.FFTW(oupt, inpt, axes=(0, 1), direction='FFTW_BACKWARD', flags=flags_init,
                                   threads=num_threads)
                wisdom = pyfftw.export_wisdom()
                np.save(wisdom_fname, wisdom)
                del inpt, oupt, fft, ifft
            pbs.barrier()
        pyfftw.import_wisdom(np.load(wisdom_fname))
        # print "++ ffs_alm_pyFFTW :: loaded widsom ", wisdom_fname
        self.flags = ('FFTW_WISDOM_ONLY',)  # This will make the code crash if arrays are not properly aligned.
        # self.flags = ('FFTW_MEASURE',)
        self.threads = num_threads

    def alm2rfft(self, alm):
        assert alm.size == self.alm_size, alm.size
        ret = pyfftw.zeros_aligned(self.ell_mat.rshape, dtype='complex128')
        ret[self.cond[self.ell_mat()]] = alm * self.fac_alm2rfft
        return ret

    def map2rfft(self, _map):
        oupt = pyfftw.empty_aligned(self.ell_mat.rshape, dtype='complex128')
        fft = pyfftw.FFTW(pyfftw.byte_align(_map, dtype='float64'), oupt,
                          axes=(0, 1), direction='FFTW_FORWARD', flags=self.flags, threads=self.threads)
        fft()
        return oupt

    def alm2map(self, alm, lib_almout=None):
        assert alm.size == self.alm_size, (alm.size, self.alm_size)
        if lib_almout is None:
            rfftalm = self.alm2rfft(alm)
            oupt = pyfftw.empty_aligned(self.ell_mat.shape, dtype='float64')
            ifft = pyfftw.FFTW(rfftalm, oupt, axes=(0, 1), direction='FFTW_BACKWARD', flags=self.flags,
                               threads=self.threads)
            ifft()
            return oupt

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

    def extend(self, filt_func):
        return ffs_alm_pyFFTW(self.ell_mat, filt_func=filt_func, num_threads=self.threads)
