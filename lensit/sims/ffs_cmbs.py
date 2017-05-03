import hashlib
import os
import pickle as pk

import numpy as np

import lensit as fs
from lensit import pbs

verbose = True

def get_fields(cls):
    fields = ['p', 't', 'e', 'b', 'o']
    for _f in fields:
        if (_f + _f) not in cls.keys(): fields.remove(_f)
    for _k in cls.keys():
        for _f in _k:
            if _f not in fields: fields.append(_f)
    return fields


class sim_cmb_unl():
    def __init__(self, cls_unl, lib_pha):
        lib_alm = lib_pha.lib_alm
        fields = get_fields(cls_unl)
        Nf = len(fields)
        if verbose: print "I see %s fields: " % Nf + " ".join(fields)
        rmat = np.zeros((lib_alm.ellmax + 1, Nf, Nf), dtype=float)
        str = ''
        for _i, _t1 in enumerate(fields):
            for _j, _t2 in enumerate(fields):
                if _j >= _i:
                    if _t1 + _t2 in cls_unl.keys():
                        rmat[:, _i, _j] = cls_unl[_t1 + _t2][:lib_alm.ellmax + 1]
                        rmat[:, _j, _i] = rmat[:, _i, _j]
                    else:
                        str += " " + _t1 + _t2
        if verbose and str != '': print str + ' set to zero'
        for ell in range(lib_alm.ellmin, lib_alm.ellmax + 1):
            t, v = np.linalg.eigh(rmat[ell, :, :])
            assert np.all(t >= 0.), (ell, t, rmat[ell, :, :])  # Matrix not positive semidefinite
            rmat[ell, :, :] = np.dot(v, np.dot(np.diag(np.sqrt(t)), v.T))

        self._cl_hash = {}
        for _k, cl in cls_unl.iteritems():
            self._cl_hash[_k] = hashlib.sha1(cl[lib_alm.ellmin:lib_alm.ellmax + 1]).hexdigest()
        self.rmat = rmat
        self.lib_pha = lib_pha
        self.lib_alm = self.lib_pha.lib_alm
        self.fields = fields

    def hashdict(self):
        ret = {'phas': self.lib_pha.hashdict()}
        for _k, _h in self._cl_hash.iteritems():
            ret[_k] = _h
        return ret

    def _get_sim_alm(self, idx, idf):
        # FIXME : triangularise this
        ret = np.zeros(self.lib_alm.alm_size, dtype=complex)
        for _i in range(len(self.fields)):
            ret += self.lib_alm.almxfl(self.lib_pha.get_sim(idx, idf=_i), self.rmat[:, idf, _i])
        return ret

    def get_sim_plm(self, idx):
        assert 'p' in self.fields, self.fields
        return self._get_sim_alm(idx, self.fields.index('p'))

    def get_sim_olm(self, idx):
        assert 'o' in self.fields, self.fields
        return self._get_sim_alm(idx, self.fields.index('o'))

    def get_sim_tlm(self, idx):
        assert 't' in self.fields, self.fields
        return self._get_sim_alm(idx, self.fields.index('t'))

    def get_sim_elm(self, idx):
        assert 'e' in self.fields, self.fields
        return self._get_sim_alm(idx, self.fields.index('e'))

    def get_sim_blm(self, idx):
        assert 'b' in self.fields, self.fields
        return self._get_sim_alm(idx, self.fields.index('b'))

    def get_sim_alms(self, idx):
        phases = self.lib_pha.get_sim(idx)
        ret = np.zeros_like(phases)
        Nf = len(self.fields)
        for _i in range(Nf):
            for _j in range(Nf):
                ret[_i] += self.lib_alm.almxfl(phases[_j], self.rmat[:, _i, _j])


class sims_cmb_len():
    def __init__(self, lib_dir, lib_skyalm, cls_unl, lib_pha=None, use_Pool=0, cache_lens=False):
        if not os.path.exists(lib_dir) and pbs.rank == 0:
            os.makedirs(lib_dir)
        pbs.barrier()
        self.lib_skyalm = lib_skyalm
        fields = get_fields(cls_unl)
        if lib_pha is None and pbs.rank == 0:
            lib_pha = fs.sims.ffs_phas.ffs_lib_phas(lib_dir + '/phas', len(fields), lib_skyalm)
        else:  # Check that the lib_alms are compatible :
            assert lib_pha.lib_alm == lib_skyalm
        pbs.barrier()

        self.unlcmbs = sim_cmb_unl(cls_unl, lib_pha)
        self.Pool = use_Pool
        self.cache_lens = cache_lens
        if not os.path.exists(lib_dir + '/sim_hash.pk') and pbs.rank == 0:
            pk.dump(self.hashdict(), open(lib_dir + '/sim_hash.pk', 'w'))
        pbs.barrier()
        fs.sims.sims_generic.hash_check(self.hashdict(), pk.load(open(lib_dir + '/sim_hash.pk', 'r')))
        self.lib_dir = lib_dir
        self.fields = fields

    def hashdict(self):
        return {'unl_cmbs': self.unlcmbs.hashdict()}

    def is_full(self):
        return self.unlcmbs.lib_pha.is_full()

    def get_sim_plm(self, idx):
        return self.unlcmbs.get_sim_plm(idx)

    def get_sim_olm(self, idx):
        return self.unlcmbs.get_sim_olm(idx)

    def _get_f(self, idx):
        if 'p' in self.unlcmbs.fields and 'o' in self.unlcmbs.fields:
            plm = self.get_sim_plm(idx)
            olm = self.get_sim_olm(idx)
            return fs.ffs_deflect.ffs_deflect.displacement_frompolm(self.lib_skyalm, plm, olm)
        elif 'p' in self.unlcmbs.fields:
            plm = self.get_sim_plm(idx)
            return fs.ffs_deflect.ffs_deflect.displacement_fromplm(self.lib_skyalm, plm)
        elif 'o' in self.unlcmbs.fields:
            olm = self.get_sim_olm(idx)
            return fs.ffs_deflect.ffs_deflect.displacement_fromolm(self.lib_skyalm, olm)
        else:
            assert 0

    def get_sim_tlm(self, idx):
        fname = self.lib_dir + '/sim_%04d_tlm.npy' % idx
        if not os.path.exists(fname):
            Tlm = self._get_f(idx).lens_alm(self.lib_skyalm, self.unlcmbs.get_sim_tlm(idx), use_Pool=self.Pool)
            if not self.cache_lens: return Tlm
            np.save(fname, Tlm)
        return np.load(fname)

    def get_sim_qulm(self, idx):
        fname = self.lib_dir + '/sim_%04d_qulm.npy' % idx
        if not os.path.exists(fname):
            Qlm, Ulm = self.lib_skyalm.EBlms2QUalms(
                np.array([self.unlcmbs.get_sim_elm(idx), self.unlcmbs.get_sim_blm(idx)]))
            f = self._get_f(idx)
            Qlm = f.lens_alm(self.lib_skyalm, Qlm, use_Pool=self.Pool)
            Ulm = f.lens_alm(self.lib_skyalm, Ulm, use_Pool=self.Pool)
            if not self.cache_lens: return np.array([Qlm, Ulm])
            np.save(fname, np.array([Qlm, Ulm]))
        return np.load(fname)
