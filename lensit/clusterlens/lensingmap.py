from lensit.clusterlens.profile import profile
import numpy as np
from lensit import ell_mat, pbs
from lensit.sims import ffs_phas, ffs_cmbs, ffs_maps
from lensit.misc.misc_utils import enumerate_progress, gauss_beam, cl_inverse
from lensit.ffs_deflect import ffs_deflect
import lensit as li
from os.path import join as opj
import os
from camb import CAMBdata


def get_cluster_libdir(cambinifile, profilename, key, npix, lpix_amin, ellmax_sky, M200, z, xmaxn, nsims, cmbexp):
    return opj(li._get_lensitdir()[0], 'temp', 'clustermaps', 'camb_%s' % cambinifile, 'cmbexp_%s' % cmbexp,
               '%s_profile' % profilename, 'lensed_by_%s' % key,
               'npix%s_lpix_%samin_lmaxsky%s' % (npix, lpix_amin, ellmax_sky),
               'M200_%.6E_z%s_xmaxn%s' % (M200, z, xmaxn), '%s_sims' % nsims)



class cluster_maps(object):
    def __init__(self, libdir:str, key:str, npix:int, lpix_amin:float, nsims:int, cosmo:CAMBdata,
                 profparams:dict, profilename='nfw', ellmax_sky=6000,
                 cmb_exp='5muKamin_1amin', cache_maps=False):
        """Library for flat-sky CMB simulations lensed by a galaxy cluster.

        Args:
            libdir: various things will be cached there
            key: lensing mode — 'lss' (LSS only), 'cluster' (cluster only), or 'lss_plus_cluster'
            npix: number of pixels on one side of the square box
            lpix_amin: physical size (in arcmin) of the pixels
            nsims: number of CMB maps to simulate
            cosmo: Instantiated camb.results.CAMBdata object
            profparams: dict with keys 'M200c', 'z', 'xmaxn' (truncation in units of r_200)
            profilename: density profile of the cluster (e.g. 'nfw')
            ellmax_sky: maximum multipole of CMB spectra used to generate the CMB maps
        """
        assert key in ['lss', 'cluster', 'lss_plus_cluster'], key
        self.libdir = libdir
        self.key = key
        self.cosmo = cosmo
        if profilename == 'nfw':
            self.M200 = profparams['M200c']
            self.z = profparams['z']
            self.xmaxn = profparams['xmaxn']
        self.npix = npix
        self.lpix_amin = lpix_amin
        lbox_amin = npix * lpix_amin
        self.lbox_rad = (lbox_amin / 60) * (np.pi / 180)
        shape = (self.npix, self.npix)
        lpix_rad = self.lpix_amin * np.pi / 180 / 60
        lsides = (lpix_rad*self.npix, lpix_rad*self.npix)
        ellmatdir = opj(li._get_lensitdir()[0], 'temp', 'ellmats', 'ellmat_npix%s_lpix_%samin' % (self.npix, self.lpix_amin))
        self.ellmat = ell_mat.ell_mat(ellmatdir, shape, lsides)
        self.cmb_exp = cmb_exp

        self.ellmax_sky = ellmax_sky

        self.num_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
        self.lib_skyalm = ell_mat.ffs_alm_pyFFTW(self.ellmat, num_threads=self.num_threads, filt_func=lambda ell: ell <= self.ellmax_sky)

        camb_cls = cosmo.get_unlensed_scalar_cls(CMB_unit='muK', raw_cl=True, lmax=self.ellmax_sky).T
        cpp_fid = cosmo.get_lens_potential_cls(lmax=self.ellmax_sky, raw_cl=True).T[0]

        if key == 'cluster':
            self.cls_unl = {'tt': camb_cls[0], 'ee': camb_cls[1], 'bb': camb_cls[2], 'te': camb_cls[3]}
            nfields = 3
        else:
            self.cls_unl = {'tt': camb_cls[0], 'ee': camb_cls[1], 'bb': camb_cls[2], 'te': camb_cls[3], 'pp': cpp_fid}
            nfields = 4

        for cl_key, cl in self.cls_unl.items():
            if cl_key != 'pp':
                cl[6000:] = 0

        # Generate the CMB random phases
        skypha_libdir = opj(self.libdir, 'len_alms', 'skypha')
        skypha = ffs_phas.ffs_lib_phas(skypha_libdir, nfields, self.lib_skyalm, nsims_max=nsims)
        if not skypha.is_full() and pbs.rank == 0:
            for i, idx in enumerate_progress(np.arange(nsims, dtype=int), label='Generating CMB phases'):
                skypha.get_sim(int(idx))
        pbs.barrier()

        self.haloprofile = profile(self.cosmo, profilename)
        self.xmax = self.xmaxn * self.haloprofile.get_concentration(self.M200, self.z)
        self.kappa0 = self.haloprofile.get_kappa0(self.M200, self.z, self.xmax)


        lencmbs_libdir = opj(self.libdir, 'len_alms')

        # Get the noise and beam of the experiment
        sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = li.get_config(self.cmb_exp)

        if self.key == 'lss':
            self.len_cmbs = ffs_cmbs.sims_cmb_len(lencmbs_libdir, self.lib_skyalm, self.cls_unl, lib_pha=skypha)
        elif self.key == 'cluster':
            self.len_cmbs = sim_cmb_len_cluster(lencmbs_libdir, self.lib_skyalm, self.cls_unl,
                                                self.M200, self.z, self.xmaxn, self.haloprofile, lib_pha=skypha)
        elif self.key == 'lss_plus_cluster':
            self.len_cmbs = sim_cmb_len_lss_plus_cluster(lencmbs_libdir, self.lib_skyalm, self.cls_unl,
                                                         self.M200, self.z, self.xmaxn, self.haloprofile, lib_pha=skypha)

        # Set the lmax of the data maps
        self.ellmax_data = ellmax
        self.ellmin_data = ellmin
        self.cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=self.ellmax_data)

        # The resolution of the data map could be lower than the resolution of the sky map (=the true map)
        self.lib_datalm = ell_mat.ffs_alm_pyFFTW(self.ellmat,
            filt_func=lambda ell: np.logical_and(ell <= self.ellmax_data, ell >= self.ellmin_data),
            num_threads=self.num_threads)

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

    def get_kappa0_from_sim(self, lmin, lmax, phi_obs_lm, phi_template_lm, NL, lib_qlm):
        """Matched-filter kappa_0 estimate from a reconstructed phi map.

        Uses the actual kappa map (with correct center-convention phases) as
        the template, so the estimator works regardless of whether the cluster
        sits at the corner or center of the box.

        Args:
            lmin, lmax: multipole range for the matched filter sum
            phi_obs_lm: lensing potential estimate (flat-sky alm array)
            NL: reconstruction noise power spectrum of the phi estimate (array indexed by ell)
        Returns:
            kappa_0 estimate (scalar)
        """

        ell_mode = lib_qlm.reduced_ellmat().astype(float)
        # Apply ell cuts and inverse-noise weighting per mode
        mask = (ell_mode >= lmin) & (ell_mode <= lmax) & (ell_mode > 0)
        iNL_mode = cl_inverse(NL)[ell_mode.astype(int)]
        weight = np.where(mask, iNL_mode, 0.0)

        # Matched filter: complex cross-correlation
        denom = np.sum(weight * np.abs(phi_template_lm) ** 2)
        numer = np.sum(weight * np.real(np.conj(phi_template_lm) * phi_obs_lm))
        return numer / denom


class sim_cmb_len_cluster(ffs_cmbs.sims_cmb_len):
    def __init__(self, lib_dir:str, lib_skyalm:ell_mat.ffs_alm, cls_unl:dict, M200:float, z:float,
                 xmaxn:float, haloprofile:profile, lib_pha=None, use_Pool=0, cache_lens=False):
        self.M200 = M200
        self.z = z
        self.xmaxn = xmaxn
        super(sim_cmb_len_cluster, self).__init__(lib_dir, lib_skyalm, cls_unl, lib_pha=lib_pha, use_Pool=0, cache_lens=False)

        conc = haloprofile.get_concentration(M200, z)
        self.kappa_map = haloprofile.kappa_map(M200, z, lib_skyalm.shape, lib_skyalm.lsides, xmax=xmaxn * conc)
        self.defl_map = haloprofile.kmap2deflmap(self.kappa_map, lib_skyalm.shape, lib_skyalm.lsides)

        assert 'p' not in self.fields, "Remove the lensing potential power spectrum from the input cls_unl dictionary to avoid errors."

    def hashdict(self):
        hd = super().hashdict()
        hd['M200'] = self.M200
        hd['z'] = self.z
        hd['xmaxn'] = self.xmaxn
        return hd

    def _get_f(self, idx=None):
        """Displacement field from the cluster convergence map only."""
        return ffs_deflect.ffs_displacement(self.defl_map[0], self.defl_map[1], self.lib_skyalm.lsides)


class sim_cmb_len_lss_plus_cluster(ffs_cmbs.sims_cmb_len):
    """CMB lensed by LSS then by a cluster (two-plane lensing approximation).

    The LSS deflection field (from the simulated plm) is applied first, then the
    cluster deflection (from the NFW kappa map). This treats the cluster as the
    nearer lens plane, consistent with a typical z_cluster ~ 0.5-1 inside the
    LSS distribution that extends to z_CMB.
    """
    def __init__(self, lib_dir:str, lib_skyalm:ell_mat.ffs_alm, cls_unl:dict, M200:float, z:float,
                 xmaxn:float, haloprofile:profile, lib_pha=None, use_Pool=0, cache_lens=False):
        self.M200 = M200
        self.z = z
        self.xmaxn = xmaxn
        super().__init__(lib_dir, lib_skyalm, cls_unl, lib_pha=lib_pha, use_Pool=use_Pool, cache_lens=cache_lens)

        conc = haloprofile.get_concentration(M200, z)
        self.kappa_map_cluster = haloprofile.kappa_map(M200, z, lib_skyalm.shape, lib_skyalm.lsides, xmax=xmaxn * conc)
        self.defl_map_cluster = haloprofile.kmap2deflmap(self.kappa_map_cluster, lib_skyalm.shape, lib_skyalm.lsides)

    def hashdict(self):
        hd = super().hashdict()
        hd['M200'] = self.M200
        hd['z'] = self.z
        hd['xmaxn'] = self.xmaxn
        return hd

    def _get_f_cluster(self):
        return ffs_deflect.ffs_displacement(self.defl_map_cluster[0], self.defl_map_cluster[1], self.lib_skyalm.lsides)

    def get_sim_tlm(self, idx):
        fname = os.path.join(self.lib_dir, 'sim_%04d_tlm.npy' % idx)
        if not os.path.exists(fname):
            Tlm = self._get_f(idx).lens_alm(self.lib_skyalm, self.unlcmbs.get_sim_tlm(idx), use_Pool=self.Pool)
            Tlm = self._get_f_cluster().lens_alm(self.lib_skyalm, Tlm, use_Pool=self.Pool)
            if not self.cache_lens: return Tlm
            np.save(fname, Tlm)
        return np.load(fname)

    def get_sim_qulm(self, idx):
        fname = os.path.join(self.lib_dir, 'sim_%04d_qulm.npy' % idx)
        if not os.path.exists(fname):
            Qlm, Ulm = self.lib_skyalm.EBlms2QUalms(
                np.array([self.unlcmbs.get_sim_elm(idx), self.unlcmbs.get_sim_blm(idx)]))
            f_lss = self._get_f(idx)
            Qlm = f_lss.lens_alm(self.lib_skyalm, Qlm, use_Pool=self.Pool)
            Ulm = f_lss.lens_alm(self.lib_skyalm, Ulm, use_Pool=self.Pool)
            f_cluster = self._get_f_cluster()
            Qlm = f_cluster.lens_alm(self.lib_skyalm, Qlm, use_Pool=self.Pool)
            Ulm = f_cluster.lens_alm(self.lib_skyalm, Ulm, use_Pool=self.Pool)
            if not self.cache_lens: return np.array([Qlm, Ulm])
            np.save(fname, np.array([Qlm, Ulm]))
        return np.load(fname)
