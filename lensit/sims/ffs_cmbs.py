from __future__ import print_function

import os
import pickle as pk

import numpy as np

from lensit.pbs import pbs
from lensit.misc.misc_utils import npy_hash
from lensit.sims import ffs_phas, sims_generic
from lensit.ffs_deflect import ffs_deflect

verbose = False

def get_fields(cls):
    fields = ['p', 't', 'e', 'b', 'o']
    ret = ['p', 't', 'e', 'b', 'o']
    for f in fields:
        if not ((f + f) in cls.keys()): ret.remove(f)
    for k in cls.keys():
        for f in k:
            if f not in ret: ret.append(f)
    return ret


class sim_cmb_unl:
    def __init__(self, cls_unl, lib_pha, alpha_cpp=1.):
        self.cls_unl = cls_unl
        self.alpha_cpp = alpha_cpp
        lib_alm = lib_pha.lib_alm
        fields = get_fields(cls_unl)
        Nf = len(fields)
        if verbose: print("I see %s fields: " % Nf + " ".join(fields))
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
        if verbose and str != '': print(str + ' set to zero')
        for ell in range(lib_alm.ellmin, lib_alm.ellmax + 1):
            t, v = np.linalg.eigh(rmat[ell, :, :])
            assert np.all(t >= 0.), (ell, t, rmat[ell, :, :])  # Matrix not positive semidefinite
            rmat[ell, :, :] = np.dot(v, np.dot(np.diag(np.sqrt(t)), v.T))

        self._cl_hash = {}
        for k in cls_unl.keys():
            self._cl_hash[k] = npy_hash(cls_unl[k][lib_alm.ellmin:lib_alm.ellmax + 1])
        self.rmat = rmat
        self.lib_pha = lib_pha
        self.lib_skyalm = self.lib_pha.lib_alm
        self.fields = fields

    def hashdict(self):
        ret = {'phas': self.lib_pha.hashdict()}
        for k in self._cl_hash.keys():
            ret[k] = self._cl_hash[k]
        return ret

    def _get_sim_alm(self, idx, idf):
        # FIXME : triangularise this
        ret = np.zeros(self.lib_skyalm.alm_size, dtype=complex)
        for i in range(len(self.fields)):
            ret += self.lib_skyalm.almxfl(self.lib_pha.get_sim(idx, idf=i), self.rmat[:, idf, i])
        return ret

    def get_sim_alm(self, idx, field):
        assert field in self.fields, self.fields
        return self._get_sim_alm(idx, self.fields.index(field))

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
        for i in range(Nf):
            for j in range(Nf):
                ret[i] += self.lib_skyalm.almxfl(phases[j], self.rmat[:, i, j])

    def get_sim_qulm(self,idx):
        return self.lib_skyalm.EBlms2QUalms(np.array([self.get_sim_elm(idx), self.get_sim_blm(idx)]))

class sims_cmb_len:
    def __init__(self, lib_dir, lib_skyalm, cls_unl, lib_pha=None, use_Pool=0, cache_lens=False, alpha_cpp=1., pbsrank=pbs.rank, pbsbarrier=pbs.barrier):
        if not os.path.exists(lib_dir) and pbsrank == 0:
            os.makedirs(lib_dir)
        pbsbarrier()
        self.lib_skyalm = lib_skyalm
        fields = get_fields(cls_unl)
        if lib_pha is None and pbsrank == 0:
            lib_pha = ffs_phas.ffs_lib_phas(os.path.join(lib_dir, 'phas'), len(fields), lib_skyalm, pbsrank=pbsrank, pbsbarrier=pbsbarrier)
        else:  # Check that the lib_alms are compatible :
            assert lib_pha.lib_alm == lib_skyalm
        pbsbarrier()

        self.unlcmbs = sim_cmb_unl(cls_unl, lib_pha, alpha_cpp=alpha_cpp)
        self.Pool = use_Pool
        self.cache_lens = cache_lens
        fn_hash = os.path.join(lib_dir, 'sim_hash.pk')
        if not os.path.exists(fn_hash) and pbsrank == 0:
            pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
        pbsbarrier()
        sims_generic.hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')))
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
            return ffs_deflect.displacement_frompolm(self.lib_skyalm, plm, olm, verbose=False)
        elif 'p' in self.unlcmbs.fields:
            plm = self.get_sim_plm(idx)
            return ffs_deflect.displacement_fromplm(self.lib_skyalm, plm)
        elif 'o' in self.unlcmbs.fields:
            olm = self.get_sim_olm(idx)
            return ffs_deflect.displacement_fromolm(self.lib_skyalm, olm, verbose=False)
        else:
            assert 0

    def get_sim_alm(self, idx, field):
        if field == 't':
            return self.get_sim_tlm(idx)
        elif field == 'p':
            return self.get_sim_plm(idx)
        elif field == 'o':
            return self.get_sim_olm(idx)
        elif field == 'q':
            return self.get_sim_qulm(idx)[0]
        elif field == 'u':
            return self.get_sim_qulm(idx)[1]
        elif field == 'e':
            return self.lib_skyalm.QUlms2EBalms(self.get_sim_qulm(idx))[0]
        elif field == 'b':
            return self.lib_skyalm.QUlms2EBalms(self.get_sim_qulm(idx))[1]
        else:
            assert 0, (field, self.fields)

    def get_sim_tlm(self, idx):
        fname = os.path.join(self.lib_dir, 'sim_%04d_tlm.npy'%idx)
        if not os.path.exists(fname):
            Tlm = self._get_f(idx).lens_alm(self.lib_skyalm, self.unlcmbs.get_sim_tlm(idx), use_Pool=self.Pool)
            if not self.cache_lens: return Tlm
            np.save(fname, Tlm)
        return np.load(fname)

    def get_sim_qulm(self, idx):
        fname = os.path.join(self.lib_dir, 'sim_%04d_qulm.npy'%idx)
        if not os.path.exists(fname):
            Qlm, Ulm = self.lib_skyalm.EBlms2QUalms(
                np.array([self.unlcmbs.get_sim_elm(idx), self.unlcmbs.get_sim_blm(idx)]))
            f = self._get_f(idx)
            Qlm = f.lens_alm(self.lib_skyalm, Qlm, use_Pool=self.Pool)
            Ulm = f.lens_alm(self.lib_skyalm, Ulm, use_Pool=self.Pool)
            if not self.cache_lens: return np.array([Qlm, Ulm])
            np.save(fname, np.array([Qlm, Ulm]))
        return np.load(fname)


class sim_cmb_unl_fixed_phi(sim_cmb_unl):
    def __init__(self, cls_unl, lib_pha, alpha_cpp=1., phimap=None):
        super(sim_cmb_unl_fixed_phi, self).__init__(cls_unl, lib_pha, alpha_cpp=1.)
        self.phimap = phimap
    
    
    def _get_sim_alm(self, idx, idf):
        # print(self.phimap)
        # We set the phi field to always return the same index (0) or phimap if phimap is not None
        ret = np.zeros(self.lib_skyalm.alm_size, dtype=complex)
        for i in range(len(self.fields)):
            if idf == self.fields.index('p'):
                # print('Get alm of {} for simu {} from simu idx 0'.format(idf, idx))
                if self.phimap is not None:
                    # print('phimap')
                    ret = self.phimap
                else:
                    ret += self.lib_skyalm.almxfl(self.lib_pha.get_sim(0, idf=i), self.rmat[:, idf, i])
            else:
                # print('Get alm of {} for simu {}'.format(idf, idx))
                ret += self.lib_skyalm.almxfl(self.lib_pha.get_sim(idx, idf=i), self.rmat[:, idf, i])

        return ret
    
    
    # def get_sim_plm(self, idx):
    #     assert 'p' in self.fields, self.fields
    #     return self._get_sim_alm(idx, self.fields.index('p'))


class sim_cmb_len_fixed_phi(sims_cmb_len):
    def __init__(self, lib_dir, lib_skyalm, cls_unl, lib_pha=None, use_Pool=0, cache_lens=False, alpha_cpp=1., phimap=None, pbsrank=pbs.rank, pbsbarrier=pbs.barrier):
        super(sim_cmb_len_fixed_phi, self).__init__(lib_dir, lib_skyalm, cls_unl, lib_pha=None, use_Pool=0, cache_lens=False, alpha_cpp=1., pbsrank=pbsrank, pbsbarrier=pbsbarrier)

        # print(phimap)
        self.unlcmbs = sim_cmb_unl_fixed_phi(cls_unl, lib_pha, alpha_cpp=alpha_cpp, phimap=phimap)
        if phimap is not None:
            print("    [sims.ffs_cmbs.sim_cmb_len_fixed_phi:] Created sim lib with given plm")
        else:
            print("    [sims.ffs_cmbs.sim_cmb_len_fixed_phi:] Created sim lib with plm from simu idx 0")

