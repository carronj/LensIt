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


def get_cluster_libdir(cambinifile, profilename, npix, lpix_amin, ellmax_sky, M200, z, nsims):
    return opj(li._get_lensitdir()[0], 'temp', 'clustermaps', 'camb_%s' % cambinifile, '%s_profile' % profilename, 
    'npix%s_lpix_%samin_lmaxsky%s' % (npix, lpix_amin, ellmax_sky), 'M200_%.6E_z%s' % (M200, z), '%s_sims' % nsims)



class cluster_maps(object):
    def __init__(self, libdir:str, npix:int, lpix_amin:float, nsims:int, cosmo:CAMBdata, profparams:dict, profilename='nfw', ellmax_sky = 6000, cmb_exp='5muKamin_1amin', cache_maps=False):
        """Library for flat-sky CMB simulations lensed by a galaxy cluster.

        Args:
            lib_dir: various things will be cached there
            npix: number of pixels on one side of the square box
            lpix_amin: physical size (in arcmin) of the pixels
            nsims: number of CMB maps to simulate
            cosmo: Instantiated camb.results.CAMBdata object 
            profparams: dict containing the parameters defining the profile 
            profilename: string defining the density profile of the cluster (e.g. nfw)
            ellmax_sky: maximum multipole of CMB spectra used to generate the CMB maps 
    """
        self.libdir = libdir
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

        assert ellmax_sky < cosmo.Params.max_l, "Ask for a higher ellmax in CAMB, see l_max_scalar in param inifile or camb.CAMBparams.set_for_lmax()"
        self.ellmax_sky = ellmax_sky

        num_threads=int(os.environ.get('OMP_NUM_THREADS', 1))
        self.lib_skyalm =  ell_mat.ffs_alm_pyFFTW(self.ellmat, num_threads=num_threads, filt_func=lambda ell: ell <= self.ellmax_sky)

        nfields = 3 

        camb_cls = cosmo.get_unlensed_scalar_cls(CMB_unit='muK', raw_cl=True, lmax=self.ellmax_sky).T
        cls_cmb = {'tt':camb_cls[0], 'ee':camb_cls[1], 'bb':camb_cls[2], 'te':camb_cls[3]}
        
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
        sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = li.get_config(cmb_exp)
        self.len_cmbs = sim_cmb_len_cluster(lencmbs_libdir, self.lib_skyalm, cls_cmb, self.M200, self.z, self.haloprofile, lib_pha=skypha)
        
        lmax_sky = self.len_cmbs.lib_skyalm.ellmax
        cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=lmax_sky)

        # The resolution of the data map could be lower than the resolution of the sky map (=the true map)
        lib_datalm = ell_mat.ffs_alm_pyFFTW(self.ellmat, filt_func=lambda ell: ell <= lmax_sky, num_threads=num_threads)
        
        vcell_amin2 = np.prod(lib_datalm.ell_mat.lsides) / np.prod(lib_datalm.ell_mat.shape) * (180 * 60. / np.pi) ** 2
        nTpix = sN_uKamin / np.sqrt(vcell_amin2)
        nPpix = sN_uKaminP / np.sqrt(vcell_amin2)

        # Generate the noise random phases
        pixpha_libdir = opj(self.libdir, 'pixpha')
        pixpha = ffs_phas.pix_lib_phas(pixpha_libdir, 3, lib_datalm.ell_mat.shape, nsims_max=nsims)
        if not pixpha.is_full() and pbs.rank == 0:
            for _i, idx in enumerate_progress(np.arange(nsims), label='Generating Noise phases'):
                pixpha.get_sim(idx)
        pbs.barrier()
        maps_libdir = opj(self.libdir, 'maps')

        self.maps_lib = ffs_maps.lib_noisemap(maps_libdir, lib_datalm, self.len_cmbs, cl_transf, nTpix, nPpix, nPpix,
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
            field: 't', 'q' or 'u' for temperature, Q and U modes of polarization respectively
        Returns:
            lensed map: numpy array of shape self.lib_datalm.shape
        """
        # TODO: how do we get the observed E and B maps ?
        if field=='t':
            return self.maps_lib.get_sim_tmap(idx)
        elif field=='q':
            return self.maps_lib.get_sim_qumap(idx)[0]
        elif field =='u':
            return self.maps_lib.get_sim_qumap(idx)[1]
        elif field in ['e', 'b']:
            qmap, umap = self.maps_lib.get_sim_qumap(idx)
            qlm = self.lib_skyalm.map2alm(qmap)
            ulm = self.lib_skyalm.map2alm(umap)
            elm, blm = self.lib_skyalm.QUlms2EBalms(np.array([qlm, ulm]))
            if field == 'e':
                return self.lib_skyalm.alm2map(elm)
            elif field == 'b':
                return self.lib_skyalm.alm2map(blm)            

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


    def get_kappa_map(self, M200, z):
        """Get the convergence map of the cluster 
        Args: 
            M200: mass (in Msol) of the cluster in a sphere of density 
                200 times the critical density of the universe
            z: redshift of the cluster
        Returns:
            convergence map: numpy array of shape self.lib_skyalm.shape
        """
        return self.haloprofile.kappa_map(M200, z, self.lib_skyalm.shape, self.lib_skyalm.lsides)



class sim_cmb_len_cluster(ffs_cmbs.sims_cmb_len):
    def __init__(self, lib_dir:str, lib_skyalm:ell_mat.ffs_alm, cls_cmb:dict, M200:float, z:float, haloprofile:profile, lib_pha=None, use_Pool=0, cache_lens=False):
        super(sim_cmb_len_cluster, self).__init__(lib_dir, lib_skyalm, cls_cmb, lib_pha=lib_pha, use_Pool=0, cache_lens=False)

        kappa_map = haloprofile.kappa_map(M200, z, lib_skyalm.shape, lib_skyalm.lsides)
        self.defl_map = haloprofile.kmap2deflmap(kappa_map, lib_skyalm.shape, lib_skyalm.lsides) 

        assert 'p' not in self.fields, "Remove the lensing potential power spectrum from the input cls_cmb dictionnary to avoid errors."

    def _get_f(self, idx=None):
        """We refine the displacement field using the convergence map of the cluster"""
        return ffs_deflect.ffs_displacement(self.defl_map[0], self.defl_map[1], self.lib_skyalm.lsides)
