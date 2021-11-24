from __future__ import print_function
import numpy as np


class ffs_converter:
    # FIXME : what a mess !!
    # converts ffs_alm arrays to non-redundant rlm etc.
    def __init__(self, lib_alm):

        self.lib_alm = lib_alm
        # precalculate non-redundant entries : (and sorting them according to amplitude)

        self.sorted_idc = np.argsort(self.lib_alm.reduced_ellmat())
        self.reverse_idc = np.argsort(self.sorted_idc)

        self.sorted = lambda arr: arr[self.sorted_idc]
        self.reverse_sorted = lambda arr: arr[self.reverse_idc]

        kxs = self.sorted(self.lib_alm.get_kx())
        kys = self.sorted(self.lib_alm.get_ky())

        kx0 = np.where(kxs == 0)[0]  # second axes at zero contains negative frequencies
        pos = np.where(kys[kx0] >= 0.)[0]
        neg = np.empty_like(pos)

        for i, _p in enumerate(pos):
            _j = np.where(np.abs(kys[kx0] + kys[kx0][_p]) < 1e-5)
            assert len(_j[0]) in [0, 1]
            if len(_j[0]) == 0:  # not found -> self-negative frequencies
                neg[i] = i
            else:
                neg[i] = _j[0]
        self.kx0 = kx0
        self.neg = neg
        self.pos = pos
        self.rlm_cond = ((kys >= 0.) & (kxs == 0)) | (kxs > 0.)
        self.has_ell0 = 0 in self.sorted(self.lib_alm.reduced_ellmat())[self.rlm_cond]
        if self.has_ell0:
            print('zero mode in alms')
            assert np.all((self.sorted(self.lib_alm.reduced_ellmat())[self.rlm_cond][1:]) > 0)

        self._rlm_size = np.count_nonzero(self.rlm_cond)
        self.rlms_size = 2 * np.count_nonzero(self.rlm_cond) - self.has_ell0

        print("rlm size " + str(self._rlm_size))
        print("alm size " + str(self.lib_alm.alm_size))

    def rlms2datalms(self, TEBlen, rlms):
        assert rlms.size == TEBlen * self.rlms_size, (rlms.size, TEBlen * self.rlms_size)
        ret = np.zeros((TEBlen, self.lib_alm.alm_size), dtype=complex)
        slice_size = 2 * self._rlm_size - 1 * self.has_ell0
        for _i in range(TEBlen):
            #FIXME: this is wrong if 0 in ells and more than 1 fields ?!
            _ret = np.zeros(self.lib_alm.alm_size, dtype=complex)
            start_im = _i * slice_size
            end_im = _i * slice_size + self._rlm_size - 1 * self.has_ell0
            start_re = end_im
            end_re = start_re + self._rlm_size
            sl_imag = slice(start_im,end_im)
            sl_real = slice(start_re,end_re)
            imag = np.zeros(self._rlm_size)
            imag[1 * self.has_ell0:] = rlms[sl_imag]
            reals = rlms[sl_real]
            _ret[self.rlm_cond] = 1j * imag + reals
            _ret[self.kx0[self.neg]] = (_ret[self.kx0[self.pos]]).conj()
            ret[_i] = self.reverse_sorted(_ret)
        return ret

    def datalms2rlms(self, TEBlen, alms):
        assert len(alms) == TEBlen, (TEBlen, len(alms))
        rlms = []
        for _alm in alms:
            blm = self.sorted(_alm)[self.rlm_cond]
            rlms.append(blm.imag[self.has_ell0:])
            _reals = blm.real
            rlms.append(_reals)
        return np.concatenate(rlms)
