from __future__ import print_function

from lensit.clusterlens.profile import  profile
import numpy as np
from lensit import ell_mat, pbs
from lensit.sims import ffs_phas, ffs_cmbs, ffs_maps
from lensit.misc.misc_utils import enumerate_progress, gauss_beam
from lensit.ffs_deflect import ffs_deflect
import lensit as li
from os.path import join as opj
import os
from camb import CAMBdata

import os
import pickle as pk

import numpy as np

from lensit.sims.ffs_cmbs import sim_cmb_unl, sims_cmb_len
from lensit.pbs import pbs
from lensit.misc.misc_utils import npy_hash
from lensit.sims import ffs_phas, sims_generic
from lensit.ffs_deflect import ffs_deflect

def get_cluster_libdir(cambinifile, profilename, key, npix, lpix_amin, ellmax_sky, M200, z, nsims, cmbexp):
    return opj(li._get_lensitdir()[0], 'temp', 'clustermaps', 'camb_%s' % cambinifile, 'cmbexp_%s' % cmbexp, '%s_profile' % profilename,"lensed_by_%s" % key, 
    'npix%s_lpix_%samin_lmaxsky%s' % (npix, lpix_amin, ellmax_sky), 'M200_%.6E_z%s' % (M200, z), '%s_sims' % nsims)



class cluster_maps(object):
    def __init__(self, libdir:str, key:str, npix:int, lpix_amin:float, nsims:int, cosmo:CAMBdata, profparams:dict, profilename='nfw', ellmax_sky = 6000, ellmax_data = 3000, ellmin_data=10, cmb_exp='5muKamin_1amin', cache_maps=False):
        """Library for flat-sky CMB simulations lensed by a galaxy cluster.

        Args:
            lib_dir: various things will be cached there
            key: a keyword among "lss", "cluster" and "lss_plus_cluster", which is gonna decide how the patch has been lensed
            npix: number of pixels on one side of the square box
            lpix_amin: physical size (in arcmin) of the pixels
            nsims: number of CMB maps to simulate
            cosmo: Instantiated camb.results.CAMBdata object 
            profparams: dict containing the parameters defining the profile 
            profilename: string defining the density profile of the cluster (e.g. nfw)
            ellmax_sky: maximum multipole of CMB spectra used to generate the CMB maps 
    """
        assert key in ['lss', 'cluster', 'lss_plus_cluster'], key
        self.libdir = libdir
        self.key = key
        self.cosmo = cosmo
        if profilename == 'nfw':
            self.M200 = profparams['M200c']
            self.z = profparams['z']
        self.npix = npix
        self.lpix_amin = lpix_amin
        shape = (self.npix, self.npix)
        lpix_rad = self.lpix_amin * np.pi / 180 / 60
        lsides = (lpix_rad*self.npix, lpix_rad*self.npix)
        ellmatdir = opj(li._get_lensitdir()[0], 'temp', 'ellmats', 'ellmat_npix%s_lpix_%samin' % (self.npix, self.lpix_amin))
        self.ellmat = ell_mat.ell_mat(ellmatdir, shape, lsides)
        self.cmb_exp = cmb_exp
        
        # assert ellmax_sky < cosmo.Params.max_l, "Ask for a higher ellmax in CAMB, see l_max_scalar in param inifile or camb.CAMBparams.set_for_lmax()"
        self.ellmax_sky = ellmax_sky

        self.num_threads=int(os.environ.get('OMP_NUM_THREADS', 1))
        self.lib_skyalm =  ell_mat.ffs_alm_pyFFTW(self.ellmat, num_threads=self.num_threads, filt_func=lambda ell: ell <= self.ellmax_sky)

        #nfields = 4 
        if self.key == 'cluster':
            nfields = 4
        else:
            nfields = 4

        camb_cls = cosmo.get_unlensed_scalar_cls(CMB_unit='muK', raw_cl=True, lmax=self.ellmax_sky).T
        #camb_cls = cosmo.get_unlensed_scalar_cls(CMB_unit='muK', raw_cl=True, lmax=self.ellmax_sky).T/100

        #camb_cls = cosmo.get_lensed_scalar_cls(CMB_unit='muK', raw_cl=True, lmax=self.ellmax_sky).T
       
        cpp_fid = cosmo.get_lens_potential_cls(lmax=self.ellmax_sky, raw_cl=True).T[0]
        
        #if self.key == 'cluster':
        #    self.cls_unl = {'tt':camb_cls[0], 'ee':camb_cls[1], 'bb':camb_cls[2], 'te':camb_cls[3]}
        #else:
        #    self.cls_unl = {'tt':camb_cls[0], 'ee':camb_cls[1], 'bb':camb_cls[2], 'te':camb_cls[3], 'pp':cpp_fid}
        self.cls_unl = {'tt':camb_cls[0], 'ee':camb_cls[1], 'bb':camb_cls[2], 'te':camb_cls[3], 'pp':cpp_fid}
        
        # Set the high ell to zero except for the lensing potential cpp_fid
        for key, cl in zip(self.cls_unl.keys(), self.cls_unl.values()):
            if key != 'pp':
                cl[6000:] = 0

        
        # Generate the CMB random phases
        skypha_libdir = opj(self.libdir,  'len_alms', 'skypha')
        skypha = ffs_phas.ffs_lib_phas(skypha_libdir, nfields, self.lib_skyalm, nsims_max=nsims)
        if not skypha.is_full() and pbs.rank == 0:
            for i, idx in enumerate_progress(np.arange(nsims, dtype=int), label='Generating CMB phases'):
                skypha.get_sim(int(idx))
        pbs.barrier()

        self.haloprofile = profile(self.cosmo, profilename)

        lencmbs_libdir = opj(self.libdir, 'len_alms')

        # Get the noise and beam of the experiment
        sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = li.get_config(self.cmb_exp)
        #self.len_cmbs = sim_cmb_len_cluster(lencmbs_libdir, self.lib_skyalm, self.cls_unl, self.M200, self.z, self.haloprofile, lib_pha=skypha)
        #self.len_cmbs = sim_cmb_len_lss_plus_cluster(lencmbs_libdir, self.lib_skyalm, self.cls_unl, self.M200, self.z, self.haloprofile, lib_pha=skypha)
        #self.len_cmbs = sims_cmb_len(lencmbs_libdir, self.lib_skyalm, self.cls_unl, lib_pha=skypha)
        #self.len_cmbs = sim_cmb_unl(self.cls_unl, lib_pha=skypha)
        if self.key == 'lss':
            self.len_cmbs = sims_cmb_len(lencmbs_libdir, self.lib_skyalm, self.cls_unl, lib_pha=skypha)
        elif self.key == 'cluster':
            self.len_cmbs = sim_cmb_len_cluster(lencmbs_libdir, self.lib_skyalm, self.cls_unl, self.M200, self.z, self.haloprofile, lib_pha=skypha)
        elif self.key == 'lss_plus_cluster':
            self.len_cmbs = sim_cmb_len_lss_plus_cluster(lencmbs_libdir, self.lib_skyalm, self.cls_unl, self.M200, self.z, self.haloprofile, lib_pha=skypha)
        else:
            assert 0
            
        # Set the lmax of the data maps
        self.ellmax_data = ellmax_data
        self.ellmin_data = ellmin_data
        self.cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=self.ellmax_data)

        # The resolution of the data map could be lower than the resolution of the sky map (=the true map)
        #self.lib_datalm = ell_mat.ffs_alm_pyFFTW(self.ellmat, filt_func=lambda ell: ell <= self.ellmax_data, num_threads=self.num_threads)
        self.lib_datalm = ell_mat.ffs_alm_pyFFTW(self.ellmat, filt_func=lambda ell: np.logical_and(ell <= self.ellmax_data , ell>=self.ellmin_data), num_threads=self.num_threads)
        
        vcell_amin2 = np.prod(self.lib_datalm.ell_mat.lsides) / np.prod(self.lib_datalm.ell_mat.shape) * (180 * 60. / np.pi) ** 2
        nTpix = sN_uKamin / np.sqrt(vcell_amin2)
        nPpix = sN_uKaminP / np.sqrt(vcell_amin2)

        # Generate the noise random phases
        pixpha_libdir = opj(self.libdir, 'pixpha')
        pixpha = ffs_phas.pix_lib_phas(pixpha_libdir, 3, self.lib_skyalm.ell_mat.shape, nsims_max=nsims)
        if not pixpha.is_full() and pbs.rank == 0:
            for _i, idx in enumerate_progress(np.arange(nsims), label='Generating Noise phases'):
                pixpha.get_sim(idx)
        pbs.barrier()

        self.dat_libdir = opj(self.libdir, f"lmaxdat{self.ellmax_data}")

        maps_libdir = opj(self.dat_libdir, 'maps')

        self.maps_lib = ffs_maps.lib_noisemap(maps_libdir, self.lib_datalm, self.len_cmbs, self.cl_transf, nTpix, nPpix, nPpix,
                                        pix_pha=pixpha, cache_sims=cache_maps)
            

    def get_unl_map(self, idx, field='t'):
        """Get the unlensed CMB map
        Args: 
            idx: index of the simulation to return 
            field: 't', 'e' or 'b' for temperature, E and B modes of polarization respectively
        Returns:
            unlensed map: numpy array of shape self.lib_skyalm.shape
        """
        if field in ['t', 'e', 'b']:
            return self.lib_skyalm.alm2map(self.len_cmbs.unlcmbs.get_sim_alm(idx, field))
        elif field in ['q', 'u']:
            i = 0 if field =='q' else 1
            return self.lib_skyalm.alm2map(self.len_cmbs.unlcmbs.get_sim_qulm(idx)[i])

    def get_len_map(self, idx, field='t'):
        """Get the lensed CMB map
        Args: 
            idx: index of the simulation to return 
            field: 't', 'e' or 'b' for temperature, E and B modes of polarization respectively
        Returns:
            lensed map: numpy array of shape self.lib_skyalm.shape
        """
        return self.lib_skyalm.alm2map(self.len_cmbs.get_sim_alm(idx, field))


    def get_obs_map(self, idx, field='t'):
        """Get the observed CMB map, inclusing the noise and the bean of the instrument
        Args: 
            idx: index of the simulation to return 
            field: 't', 'e', 'b', 'q' or 'u' for temperature, E, B, Q and U modes of polarization respectively
        Returns:
            lensed map: numpy array of shape self.lib_datalm.shape
        """
        if field=='t':
            return self.maps_lib.get_sim_tmap(idx)
        elif field=='q':
            return self.maps_lib.get_sim_qumap(idx)[0]
        elif field =='u':
            return self.maps_lib.get_sim_qumap(idx)[1]
        elif field in ['e', 'b']:
            qmap, umap = self.maps_lib.get_sim_qumap(idx)
            qlm = self.lib_datalm.map2alm(qmap)
            ulm = self.lib_datalm.map2alm(umap)
            elm, blm = self.lib_datalm.QUlms2EBalms(np.array([qlm, ulm]))
            if field == 'e':
                return self.lib_datalm.alm2map(elm)
            elif field == 'b':
                return self.lib_datalm.alm2map(blm)            

    def get_noise_map(self, idx, field='t'):
        """Get the noise map
        Args: 
            idx: index of the simulation to return 
            field: 't', 'q' or 'u' for temperature, Q and U modes of polarization respectively
        Returns:
            noise map: numpy array of shape self.lib_datalm.shape
        """
        assert field in ['t', 'q', 'u'], "The noise maps are generated for t, q and u fields, not {}".format(field)
        if field =='t':
            return self.maps_lib.get_noise_sim_tmap(idx)
        elif field =='q':
            return self.maps_lib.get_noise_sim_qmap(idx)
        elif field =='u':
            return self.maps_lib.get_noise_sim_umap(idx)


    def get_kappa_map(self, M200, z, xmax=None):
        """Get the convergence map of the cluster 
        Args: 
            M200: mass (in Msol) of the cluster in a sphere of density 
                200 times the critical density of the universe
            z: redshift of the cluster
        Returns:
            convergence map: numpy array of shape self.lib_skyalm.shape
        """
        return self.haloprofile.kappa_map(M200, z, self.lib_skyalm.shape, self.lib_skyalm.lsides, xmax)

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


class sim_cmb_len_cluster(ffs_cmbs.sims_cmb_len):
    def __init__(self, lib_dir:str, lib_skyalm:ell_mat.ffs_alm, cls_unl:dict, M200:float, z:float, haloprofile:profile, lib_pha=None, use_Pool=0, cache_lens=False):
        super(sim_cmb_len_cluster, self).__init__(lib_dir, lib_skyalm, cls_unl, lib_pha=lib_pha, use_Pool=0, cache_lens=False)

        self.conc_par = haloprofile.get_concentration(M200,z)
        self.kappa_map = haloprofile.kappa_map(M200, z, lib_skyalm.shape, lib_skyalm.lsides, xmax= 3 * self.conc_par)
        #self.kappa_map = haloprofile.kappa_map(M200, z, lib_skyalm.shape, lib_skyalm.lsides)
        #self.kappa_map = 1e2*haloprofile.kappa_map(M200, z, lib_skyalm.shape, lib_skyalm.lsides, xmax=self.conc_par)
        self.defl_map = haloprofile.kmap2deflmap(self.kappa_map, lib_skyalm.shape, lib_skyalm.lsides) 

        #assert 'p' not in self.fields, "Remove the lensing potential power spectrum from the input cls_unl dictionnary to avoid errors."

    def _get_f(self, idx=None):
        """We refine the displacement field using the convergence map of the cluster"""
        return ffs_deflect.ffs_displacement(self.defl_map[0], self.defl_map[1], self.lib_skyalm.lsides)
    
class sim_cmb_len_lss_plus_cluster:
    def __init__(self, lib_dir:str, lib_skyalm:ell_mat.ffs_alm, cls_unl:dict, M200:float, z:float, haloprofile:profile, lib_pha=None, use_Pool=0, cache_lens=False):
        #super(sim_cmb_len_lss_plus_cluster, self).__init__(lib_dir, lib_skyalm, cls_unl, lib_pha=lib_pha, use_Pool=0, cache_lens=False)

        self.M200 = M200
        self.z = z
        self.haloprofile = haloprofile
        self.conc_par = self.haloprofile.get_concentration(M200,z)
        self.kappa_map = haloprofile.kappa_map(M200, z, lib_skyalm.shape, lib_skyalm.lsides, xmax= 3 * self.conc_par)
        #self.kappa_map = haloprofile.kappa_map(M200, z, lib_skyalm.shape, lib_skyalm.lsides)
        self.defl_map = self.haloprofile.kmap2deflmap(self.kappa_map, lib_skyalm.shape, lib_skyalm.lsides) 
        
        if not os.path.exists(lib_dir) and pbs.rank == 0:
            os.makedirs(lib_dir)
        pbs.barrier()
        self.lib_skyalm = lib_skyalm
        fields = get_fields(cls_unl)
        if lib_pha is None and pbs.rank == 0:
            lib_pha = ffs_phas.ffs_lib_phas(os.path.join(lib_dir, 'phas'), len(fields), lib_skyalm)
        else:  # Check that the lib_alms are compatible :
            assert lib_pha.lib_alm == lib_skyalm
        pbs.barrier()

        self.unlcmbs = sim_cmb_unl(cls_unl, lib_pha)
        self.Pool = use_Pool
        self.cache_lens = cache_lens
        fn_hash = os.path.join(lib_dir, 'sim_hash.pk')
        if not os.path.exists(fn_hash) and pbs.rank == 0:
            pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
        pbs.barrier()
        sims_generic.hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')))
        self.lib_dir = lib_dir
        self.fields = fields

        #assert 'p' not in self.fields, "Remove the lensing potential power spectrum from the input cls_unl dictionnary to avoid errors."

    def _get_f_cluster(self, idx=None):
        """We refine the displacement field using the convergence map of the cluster"""
        return ffs_deflect.ffs_displacement(self.defl_map[0], self.defl_map[1], self.lib_skyalm.lsides)
    
    def hashdict(self):
        return {'unl_cmbs': self.unlcmbs.hashdict(),
                'M200': self.M200,
                'z': self.z}#,
                #'haloprofile': self.haloprofile}

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
            
    def _get_f_lss_plus_cluster(self, idx):
        if 'p' in self.unlcmbs.fields and 'o' in self.unlcmbs.fields:
            plm = self.get_sim_plm(idx)
            olm = self.get_sim_olm(idx)
            return ffs_deflect.displacement_frompolm(self.lib_skyalm, plm, olm, verbose=False) + ffs_deflect.ffs_displacement(self.defl_map[0], self.defl_map[1], self.lib_skyalm.lsides)
        elif 'p' in self.unlcmbs.fields:
            plm = self.get_sim_plm(idx)
            return ffs_deflect.displacement_fromplm(self.lib_skyalm, plm) + ffs_deflect.ffs_displacement(self.defl_map[0], self.defl_map[1], self.lib_skyalm.lsides)
        elif 'o' in self.unlcmbs.fields:
            olm = self.get_sim_olm(idx)
            return ffs_deflect.displacement_fromolm(self.lib_skyalm, olm, verbose=False) + ffs_deflect.ffs_displacement(self.defl_map[0], self.defl_map[1], self.lib_skyalm.lsides)
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
            Tlm = self._get_f_cluster(idx).lens_alm(self.lib_skyalm, Tlm, use_Pool=self.Pool)
            #Tlm = self._get_f_lss_plus_cluster(idx).lens_alm(self.lib_skyalm, self.unlcmbs.get_sim_tlm(idx), use_Pool=self.Pool)
            if not self.cache_lens: return Tlm
            np.save(fname, Tlm)
        return np.load(fname)

    def get_sim_qulm(self, idx):
        fname = os.path.join(self.lib_dir, 'sim_%04d_qulm.npy'%idx)
        if not os.path.exists(fname):
            Qlm, Ulm = self.lib_skyalm.EBlms2QUalms(
                np.array([self.unlcmbs.get_sim_elm(idx), self.unlcmbs.get_sim_blm(idx)]))
            f = self._get_f(idx)
            #f = self._get_f_lss_plus_cluster(idx)
            Qlm = f.lens_alm(self.lib_skyalm, Qlm, use_Pool=self.Pool)
            Ulm = f.lens_alm(self.lib_skyalm, Ulm, use_Pool=self.Pool)
            f_cluster = self._get_f_cluster(idx)
            Qlm = f_cluster.lens_alm(self.lib_skyalm, Qlm, use_Pool=self.Pool)
            Ulm = f_cluster.lens_alm(self.lib_skyalm, Ulm, use_Pool=self.Pool)
            if not self.cache_lens: return np.array([Qlm, Ulm])
            np.save(fname, np.array([Qlm, Ulm]))
        return np.load(fname)
    
    
