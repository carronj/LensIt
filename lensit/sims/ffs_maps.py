import numpy as np
import os
from lensit import pbs
from lensit.misc.misc_utils import npy_hash
import pickle as pk
from .sims_generic import hash_check
from . import ffs_phas
from .. import misc


class lib_noisevmap():
    def __init__(self, lib_dir, lib_datalm, lib_lencmb, cl_transf, TQUcovfname, pix_pha=None, cache_sims=True):
        """
        Library for sims with pixel to pixel independent noise with specified noise variance maps.
        :param lib_dir:
        :param lib_datalm:  ffs_alm library for the data maps.
        :param lib_lencmb: library of lensed cmb sims
        :param cl_transf: transfer function cl (identical for T Q U)
        :param TQUcov: (npix,npix,3,3) shaped array, TQU covariance matrix
        :param pix_pha: random for phases for the noise maps
        :param cache_sims: does cache ims on disk if set
        """
        assert np.load(TQUcovfname, mmap_mode='r').shape == (3, 3, lib_datalm.shape[0], lib_datalm.shape[1])
        self.lencmbs = lib_lencmb
        self.lib_datalm = lib_datalm
        self.lib_skyalm = lib_lencmb.lib_skyalm
        self.cl_transf = np.zeros(self.lib_skyalm.ellmax + 1, dtype=float, order='C')
        self.cl_transf[:min(len(self.cl_transf), len(cl_transf))] = cl_transf[:min(len(self.cl_transf), len(cl_transf))]
        self.lib_dir = lib_dir
        self.cache_sims = cache_sims
        self.TQUcovfname = TQUcovfname
        if not os.path.exists(lib_dir) and pbs.rank == 0:
            os.makedirs(lib_dir)
        pbs.barrier()

        if pix_pha is None:
            self.pix_pha = ffs_phas.pix_lib_phas(lib_dir + '/pix_pha', 3, lib_datalm.shape)
        else:
            self.pix_pha = pix_pha
            assert pix_pha.shape == self.lib_datalm.shape, (pix_pha.shape, self.lib_datalm.shape)
            assert pix_pha.nfields == 3, (pix_pha.nfields, 3)
        if (np.array([not os.path.exists(lib_dir + '/rmat%s.npy' % a) for a in
                      ['TT', 'TQ', 'TU', 'QQ', 'QU', 'UU']]).any()) and pbs.rank == 0:
            # FIXME : triangularise this
            TQUcov = np.load(TQUcovfname)
            rmat = np.zeros((3, 3, self.lib_datalm.shape[0], self.lib_datalm.shape[1]), dtype=float)
            for _i, i in misc.misc_utils.enumerate_progress(range(self.lib_datalm.shape[0]), 'building root matrix'):
                for j in range(self.lib_datalm.shape[1]):
                    t, v = np.linalg.eigh(TQUcov[:, :, i, j], UPLO='U')
                    assert np.all(t >= 0.), (i, j, t)  # Matrix not positive semidefinite
                    rmat[:, :, i, j] = np.dot(v, np.dot(np.diag(np.sqrt(t)), v.T))
            for (i, j), lab in zip([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)],
                                   ['TT', 'TQ', 'TU', 'QQ', 'QU', 'UU']):
                _sav = rmat[i, j, :, :]
                np.save(lib_dir + '/rmat%s.npy' % lab, _sav if _sav.any() else np.array([0.]))
        if not os.path.exists(lib_dir + '/sim_hash.pk') and pbs.rank == 0:
            pk.dump(self.hashdict(), open(lib_dir + '/sim_hash.pk', 'w'))
        pbs.barrier()
        hash_check(self.hashdict(), pk.load(open(lib_dir + '/sim_hash.pk', 'r')))

    def hashdict(self):
        hash = {'len_cmb': self.lencmbs.hashdict()}
        hash['transf'] = npy_hash(self.cl_transf)

        def noisehash(_m):
            return npy_hash(np.array([_m]) if _m.size == 1 else np.diag(_m))

        TQU = np.load(self.TQUcovfname, mmap_mode='r')
        hash['NoiseT'] = noisehash(np.diag(TQU[0, 0]))
        hash['NoiseQ'] = noisehash(np.diag(TQU[1, 1]))
        hash['NoiseU'] = noisehash(np.diag(TQU[2, 2]))
        hash['NoiseQU'] = noisehash(np.diag(TQU[1, 2]))
        hash['NoiseTQ'] = noisehash(np.diag(TQU[0, 1]))
        hash['NoiseTU'] = noisehash(np.diag(TQU[0, 2]))
        return hash

    def _get_rmat(self, a, b):
        if a > b: return self._get_rmat(b, a)
        return np.load(self.lib_dir + '/rmat%s%s.npy' % ({0: 'T', 1: 'Q', 2: 'U'}[a], {0: 'T', 1: 'Q', 2: 'U'}[b]))

    def _get_noise_map(self, idx, idf):
        # FIXME : triangularise this
        # 0 T, 1 Q, 2 U
        ret = np.zeros(self.lib_datalm.shape, dtype=float)
        for i in range(0, 3):
            r = self._get_rmat(idf, i)
            if r.any():
                ret += self.pix_pha.get_sim(idx, idf=i) * r
        return ret

    def _build_sim_tmap(self, idx):
        tmap = self.lib_skyalm.almxfl(self.lencmbs.get_sim_tlm(idx), self.cl_transf)
        tmap = self.lib_datalm.alm2map(self.lib_datalm.udgrade(self.lencmbs.lib_skyalm, tmap))
        return tmap + self.get_noise_sim_tmap(idx)

    def _build_sim_qumap(self, idx):
        qmap, umap = self.lencmbs.get_sim_qulm(idx)
        self.lib_skyalm.almxfl(qmap, self.cl_transf, inplace=True)
        self.lib_skyalm.almxfl(umap, self.cl_transf, inplace=True)
        qmap = self.lib_datalm.alm2map(self.lib_datalm.udgrade(self.lencmbs.lib_skyalm, qmap))
        umap = self.lib_datalm.alm2map(self.lib_datalm.udgrade(self.lencmbs.lib_skyalm, umap))
        return np.array([qmap + self.get_noise_sim_qmap(idx), umap + self.get_noise_sim_umap(idx)])

    def get_noise_sim_tmap(self, idx):
        return self._get_noise_map(idx, 0)

    def get_noise_sim_qmap(self, idx):
        return self._get_noise_map(idx, 1)

    def get_noise_sim_umap(self, idx):
        return self._get_noise_map(idx, 2)

    def get_sim_tmap(self, idx):
        if self.cache_sims and not os.path.exists(self.lib_dir + '/sim_tmap_%05d.npy' % idx):
            if pbs.rank == 0:
                np.save(self.lib_dir + '/sim_tmap_%05d.npy' % idx, self._build_sim_tmap(idx))
            pbs.barrier()
        if self.cache_sims:
            return np.load(self.lib_dir + '/sim_tmap_%05d.npy' % idx)
        else:
            return self._build_sim_tmap(idx)

    def get_sim_qumap(self, idx):
        if self.cache_sims and not os.path.exists(self.lib_dir + '/sim_qumap_%05d.npy' % idx):
            if pbs.rank == 0:
                np.save(self.lib_dir + '/sim_qumap_%05d.npy' % idx, np.array(self._build_sim_qumap(idx)))
            pbs.barrier()
        if self.cache_sims:
            return np.load(self.lib_dir + '/sim_qumap_%05d.npy' % idx)
        else:
            return self._build_sim_qumap(idx)

class lib_noisemap():
    def __init__(self, lib_dir, lib_datalm, lib_lencmb, cl_transf, nTpix, nQpix, nUpix, pix_pha=None, cache_sims=True):
        """
        Library for sims with pixel to pixel independent noise with specified noise variance maps.
        :param lib_dir:
        :param lib_datalm:  ffs_alm library for the data maps.
        :param lib_lencmb: library of lensed cmb sims
        :param cl_transf: transfer function cl (identical for T Q U)
        :param nTpix:  pixel noise rms noise in T (either scalar or a map of the right shape, or a path to the map)
        :param nQpix:  pixel noise rms noise in Q
        :param nUpix:  pixel noise rms noise in U
        :param pix_pha: random for phases for the noise maps
        :param cache_sims: does cache ims on disk if set
        """
        self.lencmbs = lib_lencmb
        self.lib_datalm = lib_datalm
        self.lib_skyalm = lib_lencmb.lib_skyalm
        self.cl_transf = np.zeros(self.lib_skyalm.ellmax + 1, dtype=float)
        self.cl_transf[:min(len(self.cl_transf), len(cl_transf))] = cl_transf[:min(len(self.cl_transf), len(cl_transf))]
        self.lib_dir = lib_dir
        self.cache_sims = cache_sims

        if not os.path.exists(lib_dir) and pbs.rank == 0:
            os.makedirs(lib_dir)
        pbs.barrier()

        if pix_pha is None:
            self.pix_pha = ffs_phas.pix_lib_phas(lib_dir + '/pix_pha', 3, lib_datalm.shape)
        else:
            self.pix_pha = pix_pha
            assert pix_pha.shape == self.lib_datalm.shape, (pix_pha.shape, self.lib_datalm.shape)
            assert pix_pha.nfields == 3, (pix_pha.nfields, 3)

        if not isinstance(nTpix, str):
            if not os.path.exists(lib_dir + '/nTpix.npy') and pbs.rank == 0:
                np.save(lib_dir + '/nTpix.npy', nTpix)
            pbs.barrier()
            self.nTpix = lib_dir + '/nTpix.npy'
        else:
            assert os.path.exists(nTpix), nTpix
            self.nTpix = nTpix

        if not isinstance(nQpix, str):
            if not os.path.exists(lib_dir + '/nQpix.npy') and pbs.rank == 0:
                np.save(lib_dir + '/nQpix.npy', nQpix)
            pbs.barrier()
            self.nQpix = lib_dir + '/nQpix.npy'
        else:
            assert os.path.exists(nQpix), nQpix
            self.nQpix = nQpix

        if not isinstance(nUpix, str):
            if not os.path.exists(lib_dir + '/nUpix.npy') and pbs.rank == 0:
                np.save(lib_dir + '/nUpix.npy', nUpix)
            pbs.barrier()
            self.nUpix = lib_dir + '/nUpix.npy'
        else:
            assert os.path.exists(nUpix), nUpix
            self.nUpix = nUpix

        # Check noise maps inputs
        for _noise in [self._loadTnoise, self._loadQnoise, self._loadUnoise]:
            assert _noise().size == 1 or _noise().shape == self.lib_datalm.shape, (_noise().size, self.lib_datalm.shape)

        if not os.path.exists(lib_dir + '/sim_hash.pk') and pbs.rank == 0:
            pk.dump(self.hashdict(), open(lib_dir + '/sim_hash.pk', 'w'))
        pbs.barrier()
        hash_check(self.hashdict(), pk.load(open(lib_dir + '/sim_hash.pk', 'r')))

    def hashdict(self):
        hash = {'len_cmb': self.lencmbs.hashdict()}
        hash['transf'] = npy_hash(self.cl_transf)

        def noisehash(_m):
            return npy_hash(np.array([_m]) if _m.size == 1 else np.diag(_m))

        hash['NoiseT'] = noisehash(self._loadTnoise())
        hash['NoiseQ'] = noisehash(self._loadQnoise())
        hash['NoiseU'] = noisehash(self._loadUnoise())
        return hash

    def _loadTnoise(self):
        return np.load(self.nTpix)

    def _loadQnoise(self):
        return np.load(self.nQpix)

    def _loadUnoise(self):
        return np.load(self.nUpix)

    def _build_sim_tmap(self, idx):
        tmap = self.lib_skyalm.almxfl(self.lencmbs.get_sim_tlm(idx), self.cl_transf)
        tmap = self.lib_datalm.alm2map(self.lib_datalm.udgrade(self.lencmbs.lib_skyalm, tmap))
        return tmap + self.get_noise_sim_tmap(idx)

    def _build_sim_qumap(self, idx):
        qmap, umap = self.lencmbs.get_sim_qulm(idx)
        self.lib_skyalm.almxfl(qmap, self.cl_transf, inplace=True)
        self.lib_skyalm.almxfl(umap, self.cl_transf, inplace=True)
        qmap = self.lib_datalm.alm2map(self.lib_datalm.udgrade(self.lencmbs.lib_skyalm, qmap))
        umap = self.lib_datalm.alm2map(self.lib_datalm.udgrade(self.lencmbs.lib_skyalm, umap))
        return np.array([qmap + self.get_noise_sim_qmap(idx), umap + self.get_noise_sim_umap(idx)])

    def get_noise_sim_tmap(self, idx):
        return self._loadTnoise() * self.pix_pha.get_sim(idx, 0)

    def get_noise_sim_qmap(self, idx):
        return self._loadQnoise() * self.pix_pha.get_sim(idx, 1)

    def get_noise_sim_umap(self, idx):
        return self._loadUnoise() * self.pix_pha.get_sim(idx, 2)

    def get_sim_tmap(self, idx):
        if self.cache_sims and not os.path.exists(self.lib_dir + '/sim_tmap_%05d.npy' % idx):
            if pbs.rank == 0:
                np.save(self.lib_dir + '/sim_tmap_%05d.npy' % idx, self._build_sim_tmap(idx))
            pbs.barrier()
        if self.cache_sims:
            return np.load(self.lib_dir + '/sim_tmap_%05d.npy' % idx, mmap_mode='r')
        else:
            return self._build_sim_tmap(idx)

    def get_sim_qumap(self, idx):
        if self.cache_sims and not os.path.exists(self.lib_dir + '/sim_qumap_%05d.npy' % idx):
            if pbs.rank == 0:
                np.save(self.lib_dir + '/sim_qumap_%05d.npy' % idx, np.array(self._build_sim_qumap(idx)))
            pbs.barrier()
        if self.cache_sims:
            return np.load(self.lib_dir + '/sim_qumap_%05d.npy' % idx, mmap_mode='r')
        else:
            return self._build_sim_qumap(idx)

class lib_noisefree():
    def __init__(self, lib_dir, lib_datalm, lib_lencmb, cl_transf, cache_sims=False):

        self.lencmbs = lib_lencmb
        self.lib_datalm = lib_datalm
        self.lib_skyalm = lib_lencmb.lib_skyalm
        self.cl_transf = np.zeros(self.lib_skyalm.ellmax + 1, dtype=float)
        self.cl_transf[:min(len(self.cl_transf), len(cl_transf))] = cl_transf[:min(len(self.cl_transf), len(cl_transf))]
        self.lib_dir = lib_dir
        self.cache_sims = cache_sims
        if self.cache_sims :
            assert lib_dir is not None
            if not os.path.exists(lib_dir) and pbs.rank == 0:
                os.makedirs(lib_dir)
            pbs.barrier()

            if not os.path.exists(lib_dir + '/sim_hash.pk') and pbs.rank == 0:
                pk.dump(self.hashdict(), open(lib_dir + '/sim_hash.pk', 'w'))
            pbs.barrier()
            hash_check(self.hashdict(), pk.load(open(lib_dir + '/sim_hash.pk', 'r')))

    def hashdict(self):
        hash = {'len_cmb': self.lencmbs.hashdict()}
        hash['transf'] = npy_hash(self.cl_transf)
        return hash

    def _build_sim_tmap(self, idx):
        tmap = self.lib_skyalm.almxfl(self.lencmbs.get_sim_tlm(idx), self.cl_transf)
        tmap = self.lib_datalm.alm2map(self.lib_datalm.udgrade(self.lencmbs.lib_skyalm, tmap))
        return tmap

    def _build_sim_qumap(self, idx):
        qmap, umap = self.lencmbs.get_sim_qulm(idx)
        self.lib_skyalm.almxfl(qmap, self.cl_transf, inplace=True)
        self.lib_skyalm.almxfl(umap, self.cl_transf, inplace=True)
        qmap = self.lib_datalm.alm2map(self.lib_datalm.udgrade(self.lencmbs.lib_skyalm, qmap))
        umap = self.lib_datalm.alm2map(self.lib_datalm.udgrade(self.lencmbs.lib_skyalm, umap))
        return np.array([qmap, umap])

    def get_sim_tmap(self, idx):
        if self.cache_sims and not os.path.exists(self.lib_dir + '/sim_tmap_%05d.npy' % idx):
            if pbs.rank == 0:
                np.save(self.lib_dir + '/sim_tmap_%05d.npy' % idx, self._build_sim_tmap(idx))
            pbs.barrier()
        if self.cache_sims:
            return np.load(self.lib_dir + '/sim_tmap_%05d.npy' % idx, mmap_mode='r')
        else:
            return self._build_sim_tmap(idx)

    def get_sim_qumap(self, idx):
        if self.cache_sims and not os.path.exists(self.lib_dir + '/sim_qumap_%05d.npy' % idx):
            if pbs.rank == 0:
                np.save(self.lib_dir + '/sim_qumap_%05d.npy' % idx, np.array(self._build_sim_qumap(idx)))
            pbs.barrier()
        if self.cache_sims:
            return np.load(self.lib_dir + '/sim_qumap_%05d.npy' % idx, mmap_mode='r')
        else:
            return self._build_sim_qumap(idx)

class library_sum():
    def __init__(self,sims_list,weights = None):
        weights = np.ones(len(sims_list),dtype = float) if weights is None else weights
        assert len(weights) == len(sims_list),(len(weights),len(sims_list))
        shape = sims_list[0].lib_datalm.shape
        assert np.all([sims.lib_datalm.shape == shape for sims in sims_list])
        self.nlib = len(sims_list)
        self.sims_list = sims_list
        self.weights = weights

    def get_sim_tmap(self,idx):
        ret = self.weights[0] * self.sims_list[0].get_sim_tmap(idx)
        for w,sim in zip(self.weights,self.sims_list):
            ret += w * sim.get_sim_tmap(idx)
        return ret

    def get_sim_qumap(self, idx):
        ret = self.weights[0] * self.sims_list[0].get_sim_qumap(idx)
        for w, sim in zip(self.weights[1:], self.sims_list[1:]):
            ret += w * sim.get_sim_qumap(idx)
        return ret
