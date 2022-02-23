from lensit.clusterlens.profile import  profile
import numpy as np
from lensit import ell_mat, pbs
from lensit.sims import ffs_phas, ffs_cmbs
from lensit.misc.misc_utils import enumerate_progress, gauss_beam
from lensit.ffs_deflect import ffs_deflect
import lensit as li
from os.path import join as opj
import os


def get_cluster_libdir(cambinifile, profilename, npix, lpix_amin, ellmax_sky, M200, z, nsims):
    return opj(li._get_lensitdir()[0], 'temp', 'clustermaps', 'camb_%s' % cambinifile, '%s_profile' % profilename, 
    'npix%s_lpix_%samin_lmaxsky%s' % (npix, lpix_amin, ellmax_sky), 'M200_%.6E_z%s' % (M200, z), '%s_sims' % nsims)



class cluster_maps(object):
    def __init__(self, libdir, npix, lpix_amin, nsims, cosmo, profparams, profilename='nfw', ellmax_sky = 6000):
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
        ellmatdir = os.path.join(li._get_lensitdir()[0], 'temp', 'ellmats', 'ellmat_npix%s_lpix_%samin' % (self.npix, self.lpix_amin))
        self.ellmat = ell_mat.ell_mat(ellmatdir, shape, lsides)

        assert ellmax_sky < cosmo.Params.max_l, "Ask for a higher ellmax in CAMB, see l_max_scalar in param inifile or camb.CAMBparams.set_for_lmax()"
        self.ellmax_sky = ellmax_sky

        Beam_FWHM_amin = 1
        self.cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=self.ellmax_sky)
        
        num_threads=int(os.environ.get('OMP_NUM_THREADS', 1))
        self.lib_skyalm =  ell_mat.ffs_alm_pyFFTW(self.ellmat, num_threads=num_threads, filt_func=lambda ell: ell <= self.ellmax_sky)

        nfields = 3 

        camb_cls = cosmo.get_unlensed_scalar_cls(CMB_unit='muK', raw_cl=True, lmax=self.ellmax_sky).T
        cls_cmb = {'tt':camb_cls[0], 'ee':camb_cls[1], 'bb':camb_cls[2], 'te':camb_cls[3]}
        
        skypha_libdir = opj(self.libdir,  'len_alms', 'skypha')
        skypha = ffs_phas.ffs_lib_phas(skypha_libdir, nfields, self.lib_skyalm, nsims_max=nsims)
        if not skypha.is_full() and pbs.rank == 0:
            for i, idx in enumerate_progress(np.arange(nsims, dtype=int), label='Generating CMB phases'):
                skypha.get_sim(int(idx))
        pbs.barrier()

        self.haloprofile = profile(self.cosmo, profilename)

        sims_libdir = opj(self.libdir, 'len_alms')
        self.sims = sim_cmb_len_cluster(sims_libdir, self.lib_skyalm, cls_cmb, self.M200, self.z, self.haloprofile)

    def get_unl_map(self, idx, field='t'):
        """Get the unlensed CMB map
        Args: 
            idx: index of the simulation to return 
            field: 't', 'e' or 'b' for temperature, E and B modes of polarization repsectively
        Returns:
            unlensed map: numpy array of shape self.lib_skyalm.shape
        """
        return self.lib_skyalm.alm2map(self.sims.unlcmbs.get_sim_alm(idx, field))

    def get_len_map(self, idx, field='t'):
        """Get the lensed CMB map
        Args: 
            idx: index of the simulation to return 
            field: 't', 'e' or 'b' for temperature, E and B modes of polarization repsectively
        Returns:
            lensed map: numpy array of shape self.lib_skyalm.shape
        """
        return self.lib_skyalm.alm2map(self.sims.get_sim_alm(idx, field))

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
    def __init__(self, lib_dir, lib_skyalm, cls_cmb, M200, z, haloprofile, lib_pha=None, use_Pool=0, cache_lens=False):
        super(sim_cmb_len_cluster, self).__init__(lib_dir, lib_skyalm, cls_cmb, lib_pha=None, use_Pool=0, cache_lens=False)

        kappa_map = haloprofile.kappa_map(M200, z, lib_skyalm.shape, lib_skyalm.lsides)
        self.defl_map = haloprofile.kmap2deflmap(kappa_map, lib_skyalm.shape, lib_skyalm.lsides) 

        assert 'p' not in self.fields, "Remove the lensing potential power spectrum from the input cls_cmb dictionnary to avoid errors."

    def _get_f(self, idx=None):
        """We refine the displacement field using the convergence map of the cluster"""
        return ffs_deflect.ffs_displacement(self.defl_map[0], self.defl_map[1], self.lib_skyalm.lsides)
