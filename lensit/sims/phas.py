"""
Curved sky sim phase library for healpix scheme.
"""
import sims_generic
import numpy as np
import healpy as hp

class _lib_phas(sims_generic.sim_lib):
    def __init__(self, lib_dir,lmax, **kwargs):
        self.lmax = lmax
        super(_lib_phas, self).__init__(lib_dir, **kwargs)

    def _build_sim_from_rng(self, rng_state, phas_only=False):
        np.random.set_state(rng_state)
        alm = (np.random.standard_normal(hp.Alm.getsize(self.lmax)) +
               1j * np.random.standard_normal(hp.Alm.getsize(self.lmax))) / np.sqrt(2.)
        if phas_only: return
        m0 = hp.Alm.getidx(self.lmax,np.arange(self.lmax + 1,dtype = int),0)
        alm[m0] = np.sqrt(2.) * alm[m0].real
        return alm

    def hashdict(self):
        return {'lmax':self.lmax}

class lib_phas():
    def __init__(self, lib_dir, nfields,lmax, **kwargs):
        self.lmax = lmax
        self.nfields = nfields
        self.lib_phas = {}
        for _i in range(nfields):
            self.lib_phas[_i] = _lib_phas(lib_dir + '/pha_%04d' % _i, lmax, **kwargs)

    def is_full(self):
        return np.all([lib.is_full() for lib in self.lib_phas.values()])

    def get_sim(self, idx, idf=None, phas_only=False):
        if idf is not None:
            assert idf < self.nfields, (idf, self.nfields)
            return self.lib_phas[idf].get_sim(idx, phas_only=phas_only)
        return np.array([self.lib_phas[_idf].get_sim(idx, phas_only=phas_only) for _idf in range(self.nfields)])


    def hashdict(self):
        return {'nfields': self.nfields, 'lmax':self.lmax}

class lib_phas_lcut():
    """
    Share same phases as above but with smaller lmax.
    """
    def __init__(self,  lib_phas,lmax):
        assert lmax <= lib_phas.lmax,(lmax,lib_phas.lmax)
        self.lmax = lmax
        self.lib_phas = lib_phas
        self.nfields = self.lib_phas.nfields

    def is_full(self):
        return self.lib_phas.is_full()

    def _deg(self,alm):
        assert hp.Alm.getlmax(alm.size) == self.lib_phas.lmax
        alm_lmax = self.lib_phas.lmax
        if self.lmax == alm_lmax:
            return alm
        ret = np.zeros(hp.Alm.getsize(self.lmax), dtype=np.complex)
        for m in xrange(0, self.lmax + 1):
            ret[((m * (2 * self.lmax + 1 - m) / 2) + m):(m * (2 * self.lmax + 1 - m) / 2 + self.lmax + 1)] \
            = alm[(m * (2 * alm_lmax + 1 - m) / 2 + m):(m * (2 * alm_lmax + 1 - m) / 2 + self.lmax + 1)]
        return ret

    def get_sim(self, idx, idf=None, phas_only=False):
        if idf is not None:
            assert idf < self.nfields, (idf, self.nfields)
            return self._deg(self.lib_phas.lib_phas[idf].get_sim(idx, phas_only=phas_only))
        return np.array([self._deg(self.lib_phas.lib_phas[i].get_sim(idx, phas_only=phas_only)) for i in range(self.nfields)])

    def hashdict(self):
        return {'lib_phas': self.lib_phas.hashdict(), 'lmax':self.lmax}
