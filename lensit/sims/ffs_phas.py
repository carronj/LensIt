import numpy as np
import os

from lensit.sims import sims_generic


class _lib_ffsphas(sims_generic.sim_lib):
    def __init__(self, lib_dir, lib_alm, **kwargs):
        self.lib_alm = lib_alm
        super(_lib_ffsphas, self).__init__(lib_dir, **kwargs)

    def _build_sim_from_rng(self, rng_state, phas_only=False):
        np.random.set_state(rng_state)
        alm = (np.random.standard_normal(self.lib_alm.alm_size) +
               1j * np.random.standard_normal(self.lib_alm.alm_size)) / np.sqrt(2.)
        if phas_only: return
        # Reality conditions on the rfft maps
        sla = slice(self.lib_alm.ell_mat.shape[0] // 2 + 1, self.lib_alm.ell_mat.shape[0], 1)
        slb = slice(self.lib_alm.ell_mat.shape[0] // 2 - 1, 0, -1)

        rfft = self.lib_alm.alm2rfft(alm)
        rfft[sla, [-1, 0]] = np.conjugate(rfft[slb, [-1, 0]])
        rfft.real[self.lib_alm.ell_mat.rfft2_reals()] *= np.sqrt(2.)
        rfft.imag[self.lib_alm.ell_mat.rfft2_reals()] = 0.
        alm = self.lib_alm.rfftmap2alm(rfft)
        return alm

    def hashdict(self):
        return {'lib_alm': self.lib_alm.hashdict()}


class _pix_lib_phas(sims_generic.sim_lib):
    def __init__(self, lib_dir, shape, **kwargs):
        self.shape = shape
        super(_pix_lib_phas, self).__init__(lib_dir, **kwargs)

    def _build_sim_from_rng(self, rng_state, **kwargs):
        np.random.set_state(rng_state)
        return np.random.standard_normal(self.shape)

    def hashdict(self):
        return {'shape': self.shape}


class ffs_lib_phas:
    def __init__(self, lib_dir, nfields, lib_alm, **kwargs):
        self.lib_alm = lib_alm
        self.nfields = nfields
        self.lib_phas = {}
        for i in range(nfields):
            self.lib_phas[i] = _lib_ffsphas(os.path.join(lib_dir, 'ffs_pha_%04d' % i), lib_alm, **kwargs)

    def is_full(self):
        return np.all([lib.is_full() for lib in self.lib_phas.values()])

    def get_sim(self, idx, idf=None, phas_only=False):
        if idf is not None:
            assert idf < self.nfields, (idf, self.nfields)
            return self.lib_phas[idf].get_sim(idx, phas_only=phas_only)
        return np.array([self.lib_phas[_idf].get_sim(int(idx), phas_only=phas_only) for _idf in range(self.nfields)])

    def hashdict(self):
        return {'nfields': self.nfields, 'lib_alm': self.lib_alm.hashdict()}


class pix_lib_phas:
    def __init__(self, lib_dir, nfields, shape, **kwargs):
        self.nfields = nfields
        self.lib_pix = {}
        self.shape = shape
        for i in range(nfields):
            self.lib_pix[i] = _pix_lib_phas(os.path.join(lib_dir, 'pix_pha_%04d'%i), shape, **kwargs)

    def is_full(self):
        return np.all([lib.is_full() for lib in self.lib_pix.values()])

    def get_sim(self, idx, idf=None, phas_only=False):
        if idf is not None:
            assert idf < self.nfields, (idf, self.nfields)
            return self.lib_pix[idf].get_sim(idx, phas_only=phas_only)
        return np.array([self.lib_pix[_idf].get_sim(int(idx), phas_only=phas_only) for _idf in range(self.nfields)])

    def hashdict(self):
        return {'nfields': self.nfields, 'shape': self.lib_pix[0].shape}
