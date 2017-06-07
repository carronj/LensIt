
import cPickle as pk
import hashlib
import os

import healpy as hp
import sims_generic
from lensit import pbs


class sims_cmb_maps(object):
    def __init__(self,sims_cmb_len,cl_ttransf,cl_ptransf,nside = 2048,lib_dir = None):
        self.sims_cmb_len = sims_cmb_len
        self.cl_ttransf = cl_ttransf
        self.cl_ptransf = cl_ptransf
        self.nside = nside
        if lib_dir is not None:
            if pbs.rank == 0 and not os.path.exists(lib_dir + '/sim_hash.pk'):
                pk.dump(self.hashdict(), open(lib_dir + '/sim_hash.pk', 'w'))
            pbs.barrier()
            sims_generic.hash_check(self.hashdict(), pk.load(open(lib_dir + '/sim_hash.pk', 'r')))

    def hashdict(self):
        return {'sims_cmb_len':self.sims_cmb_len.hashdict(),'nside':self.nside,
                'cl_ttransf':hashlib.sha1(self.cl_ttransf.copy(order='C')).hexdigest(),
                'cl_ptransf': hashlib.sha1(self.cl_ptransf.copy(order='C')).hexdigest()}

    def get_sim_tmap(self,idx):
        tmap = self.sims_cmb_len.get_sim_tlm(idx)
        hp.almxfl(tmap,self.cl_ttransf,inplace=True)
        tmap = hp.alm2map(tmap,self.nside)
        return tmap + self.get_sim_tnoise(idx)

    def get_sim_qumap(self,idx):
        elm = self.sims_cmb_len.get_sim_elm(idx)
        hp.almxfl(elm,self.cl_ptransf,inplace=True)
        blm = self.sims_cmb_len.get_sim_blm(idx)
        hp.almxfl(blm, self.cl_ptransf, inplace=True)
        Q,U = hp.alm2map_spin([elm,blm], self.nside, 2,hp.Alm.getlmax(elm.size))
        del elm,blm
        return [Q + self.get_sim_qnoise(idx),U + self.get_sim_unoise(idx)]

    def get_sim_tnoise(self,idx):
        assert 0,'subclass this'

    def get_sim_qnoise(self, idx):
        assert 0, 'subclass this'

    def get_sim_unoise(self, idx):
        assert 0, 'subclass this'


