"""
Unlensed and lensed curved sky CMB simulation libraries

"""
# FIXME : To recover the flat sky usual accuracy,
# FIXME : need smth like nside_lens 4096, at 2048, something happens at 5000

import cPickle as pk
import hashlib
import os

import healpy as hp
import numpy as np

import phas
import sims_generic
from lensit import pbs
from lensit.curvedskylensing import lenscurv as lens

verbose = True


def get_fields(cls):
    print cls.keys()
    fields = ['p', 't', 'e', 'b', 'o']
    ret = ['p', 't', 'e', 'b', 'o']
    for _f in fields:
        if not ((_f + _f) in cls.keys()): ret.remove(_f)
    for _k in cls.keys():
        for _f in _k:
            if _f not in ret: ret.append(_f)
    return ret

class sims_cmb_unl():
    def __init__(self, cls_unl, lib_pha):
        lmax = lib_pha.lmax
        lmin = 0
        fields = get_fields(cls_unl)
        Nf = len(fields)
        if verbose: print "I see %s fields: " % Nf + " ".join(fields)
        rmat = np.zeros((lmax + 1, Nf, Nf), dtype=float)
        str = ''
        for _i, _t1 in enumerate(fields):
            for _j, _t2 in enumerate(fields):
                if _j >= _i:
                    if _t1 + _t2 in cls_unl.keys():
                        rmat[lmin:, _i, _j] = cls_unl[_t1 + _t2][:lmax + 1]
                        rmat[lmin:, _j, _i] = rmat[lmin:, _i, _j]
                    else:
                        str += " " + _t1 + _t2
        if verbose and str != '': print str + ' set to zero'
        for ell in range(lmin,lmax + 1):
            t, v = np.linalg.eigh(rmat[ell, :, :])
            assert np.all(t >= 0.), (ell, t, rmat[ell, :, :])  # Matrix not positive semidefinite
            rmat[ell, :, :] = np.dot(v, np.dot(np.diag(np.sqrt(t)), v.T))

        self._cl_hash = {}
        for _k, cl in cls_unl.iteritems():
            self._cl_hash[_k] = hashlib.sha1(cl[lmin:lmax + 1]).hexdigest()
        self.rmat = rmat
        self.lib_pha = lib_pha
        self.fields = fields

    def hashdict(self):
        ret = {'phas': self.lib_pha.hashdict()}
        for _k, _h in self._cl_hash.iteritems():
            ret[_k] = _h
        return ret

    def _get_sim_alm(self, idx, idf):
        # FIXME : triangularise this
        ret = hp.almxfl(self.lib_pha.get_sim(idx, idf=0), self.rmat[:, idf, 0])
        for _i in range(1,len(self.fields)):
            ret += hp.almxfl(self.lib_pha.get_sim(idx, idf=_i), self.rmat[:, idf, _i])
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
        for _i in range(Nf):
            for _j in range(Nf):
                ret[_i] += hp.almxfl(phases[_j], self.rmat[:, _i, _j])

class sims_cmb_len():
    def __init__(self, lib_dir, lmax, cls_unl,dlmax = 1024,nside_lens = 2048,lib_pha=None):
        #FIXME : add aberration and modulation
        if not os.path.exists(lib_dir) and pbs.rank == 0:
            os.makedirs(lib_dir)
        pbs.barrier()
        self.lmax = lmax
        self.dlmax = dlmax
        fields = get_fields(cls_unl)
        assert 'o' not in fields,'Check lenscurv.py if everything is implemented. Should be easy.'
        if lib_pha is None and pbs.rank == 0:
            lib_pha = phas.lib_phas(lib_dir + '/phas', len(fields), lmax + dlmax)
        else:  # Check that the lib_alms are compatible :
            assert lib_pha.lmax == lmax + dlmax
        pbs.barrier()
        self.nside_lens = nside_lens
        self.unlcmbs = sims_cmb_unl(cls_unl, lib_pha)
        self.lib_dir = lib_dir
        self.fields = get_fields(cls_unl)
        if pbs.rank == 0 and not os.path.exists(lib_dir + '/sim_hash.pk') :
            pk.dump(self.hashdict(), open(lib_dir + '/sim_hash.pk', 'w'))
        pbs.barrier()
        sims_generic.hash_check(self.hashdict(), pk.load(open(lib_dir + '/sim_hash.pk', 'r')))

    def hashdict(self):
        return {'unl_cmbs': self.unlcmbs.hashdict(),'lmax':self.lmax,'nside_lens':self.nside_lens}

    def is_full(self):
        return self.unlcmbs.lib_pha.is_full()

    def get_sim_alm(self, idx, field):
        if field == 't':
            return self.get_sim_tlm(idx)
        elif field == 'e':
            return self.get_sim_elm(idx)
        elif field == 'b':
            return self.get_sim_blm(idx)
        elif field == 'p':
            return self.get_sim_plm(idx)
        elif field == 'o':
            return self.get_sim_olm(idx)
        else :
            assert 0,(field,self.fields)

    def get_sim_cl(self,key,idx):
        assert len(key) == 2,key
        f1,f2 = key
        i = self.fields.index(f1)
        j = self.fields.index(f2)
        if i > j: return self.get_sim_cl(f2 + f1,idx)
        fname = self.lib_dir + '/cl%s_%04d.dat'%(key,idx)
        if not os.path.exists(fname):
            alm1 = self.get_sim_alm(idx,f1)
            alm2 = alm1 if f2 == f1 else self.get_sim_alm(idx, f2)
            np.savetxt(fname,hp.alm2cl(alm1,alms2=alm2))
        return np.loadtxt(fname)

    def get_sim_plm(self, idx):
        return self.unlcmbs.get_sim_plm(idx)

    def get_sim_olm(self, idx):
        return self.unlcmbs.get_sim_olm(idx)

    def _cache_eblm(self, idx):
        elm = self.unlcmbs.get_sim_elm(idx)
        blm = None if 'b' not in self.fields else self.unlcmbs.get_sim_blm(idx)
        dlm = self.get_sim_plm(idx)
        lmaxd = hp.Alm.getlmax(dlm.size)
        hp.almxfl(dlm, np.sqrt(np.arange(lmaxd + 1, dtype=float) * np.arange(1, lmaxd + 2)), inplace=True)
        elm, blm = lens.lens_eblm(elm,dlm,blm = blm, nside=self.nside_lens,lmaxout=self.lmax)
        hp.write_alm(self.lib_dir + '/sim_%04d_elm.fits' % idx, elm)
        del elm
        hp.write_alm(self.lib_dir + '/sim_%04d_blm.fits' % idx, blm)

    def get_sim_tlm(self, idx):
        fname = self.lib_dir + '/sim_%04d_tlm.fits' % idx
        if not os.path.exists(fname):
            tlm= self.unlcmbs.get_sim_tlm(idx)
            dlm = self.get_sim_plm(idx)
            lmaxd = hp.Alm.getlmax(dlm.size)
            hp.almxfl(dlm, np.sqrt(np.arange(lmaxd + 1, dtype=float) * np.arange(1, lmaxd + 2)), inplace=True)
            hp.write_alm(fname, lens.lens_tlm(tlm, dlm, nside=self.nside_lens, lmaxout=self.lmax))
        return hp.read_alm(fname)

    def get_sim_elm(self, idx):
        fname = self.lib_dir + '/sim_%04d_elm.fits' % idx
        if not os.path.exists(fname):
            self._cache_eblm(idx)
        return hp.read_alm(fname)

    def get_sim_blm(self, idx):
        fname = self.lib_dir + '/sim_%04d_blm.fits' % idx
        if not os.path.exists(fname):
            self._cache_eblm(idx)
        return hp.read_alm(fname)

