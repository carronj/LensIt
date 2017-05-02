import numpy as np

from utils import ffs_converter


class template():
    def __init__(self):
        self.nmodes = 0
        assert (0)

    def apply(self, map, coeffs):
        # map -> map*[coeffs combination of templates]
        assert (0)

    def apply_mode(self, map, mode):
        assert (mode < self.nmodes)
        assert (mode >= 0)

        tcoeffs = np.zeros(self.nmodes)
        tcoeffs[mode] = 1.0
        self.apply(map, tcoeffs)

    def accum(self, map, coeffs):
        assert (0)

    def dot(self, map):
        ret = []

        for i in range(0, self.nmodes):
            tmap = np.copy(map)
            self.apply_mode(tmap, i)
            ret.append(np.sum(tmap))

        return ret


class template_map(template):
    def __init__(self, map):
        self.nmodes = 1
        self.map = map

    def apply(self, map, coeffs):
        assert (len(coeffs) == self.nmodes)

        map *= self.map * coeffs[0]

    def accum(self, map, coeffs):
        assert (len(coeffs) == self.nmodes)

        map += self.map * coeffs[0]

    def dot(self, map):
        return [(self.map * map).sum()]


class template_uptolmin(template):
    def __init__(self, ellmat, lmin):
        try:
            from lensit.ffs_covs.ell_mat import ffs_alm_pyFFTW as ffs_alm
        except:
            from lensit.ffs_covs.ell_mat import ffs_alm as ffs_alm
        self.lmin = lmin
        lib_alm = ffs_alm(ellmat, filt_func=lambda ell: (ell <= lmin))
        self.conv = ffs_converter(lib_alm)
        self.nmodes = self.conv.rlms_size
        self.lib_alm = lib_alm

    def _rlm2alm(self, rlm):
        return self.conv.rlms2datalms(1, rlm)[0]

    def _alm2rlm(self, alm):
        return self.conv.datalms2rlms(1, [alm])

    def apply(self, tmap, coeffs):  # V
        assert (len(coeffs) == self.nmodes)
        assert tmap.shape == self.lib_alm.shape
        tmap *= self.lib_alm.alm2map(self._rlm2alm(coeffs))

    def accum(self, tmap, coeffs):
        assert (len(coeffs) == self.nmodes)
        tmap += self.lib_alm.alm2map(self._rlm2alm(coeffs))

    def dot(self, tmap):  # V^t
        assert tmap.shape == self.lib_alm.shape
        return self._alm2rlm(self.lib_alm.map2alm(tmap)) * self.lib_alm.nbar()
