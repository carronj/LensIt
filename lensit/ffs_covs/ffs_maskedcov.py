import hashlib
import os
import numpy as np

import ffs_cov
import lensit as fs
from ffs_specmat import get_unlPmat_ij
import ffs_specmat_noBB as SMnoBB
import ffs_specmat as SM
import lensit.pbs
from lensit.misc.misc_utils import timer
from lensit.qcinv import multigrid

_types = ['T', 'QU', 'TQU']
_timed = True


class ffs_isocov_wmask(ffs_cov.ffs_diagcov_alm):
    """
    Full flat sky PL2015CMBlensing like ivf :
    cov_wm = fs.get_isocovwmask('Planck', 13)
    iblms = cov_wm.get_MLlms('T',np.array([libPL.get_sim_tmap(0)]))
    solves the ivf in about 7 mins or so to 1e-5, at res 1.4 amin
    """

    # F M B xi B^t M^t F^t + N
    def __init__(self, lib_dir, lib_datalm, cls_unl, cls_len, cl_transf, sN_uKamins, lib_skyalm=None, mask_list=()):
        cls_noise = {}
        for _k in ['t', 'q', 'u']:
            cls_noise[_k] = (np.mean(sN_uKamins[_k]) * np.pi / 180. / 60.) ** 2 * np.ones(lib_datalm.ellmax + 1,
                                                                                          dtype=float)

        for _m in mask_list:
            assert self._load_map(_m).shape == lib_datalm.ell_mat.shape, (_m.shape, lib_datalm.ell_mat.shape)
        self.mask_list = mask_list

        super(ffs_isocov_wmask, self) \
            .__init__(lib_dir, lib_datalm, cls_unl, cls_len, cl_transf, cls_noise, lib_skyalm=lib_skyalm)

        self.sN_uKamins = sN_uKamins

        lensit.pbs.barrier()

    def hashdict(self):
        hash = {'lib_alm': self.lib_datalm.hashdict(), 'lib_skyalm': self.lib_skyalm.hashdict()}
        for key, cl in self.cls_noise.iteritems():
            hash['cls_noise ' + key] = hashlib.sha1(cl).hexdigest()
        for key, cl in self.cls_unl.iteritems():
            hash['cls_unl ' + key] = hashlib.sha1(cl).hexdigest()
        for key, cl in self.cls_len.iteritems():
            hash['cls_len ' + key] = hashlib.sha1(cl).hexdigest()
        hash['cl_transf'] = hashlib.sha1(self.cl_transf).hexdigest()
        hash['mask'] = hashlib.sha1(self.load_mask()).hexdigest()
        return hash

    def _load_map(self, _map):
        if isinstance(_map, str):
            return np.load(_map, mmap_mode='r')
        else:
            return _map

    def load_mask(self):
        if not os.path.exists(self.lib_dir + '/mask.npy') and lensit.pbs.rank == 0:
            if not os.path.exists(self.lib_dir): os.makedirs(self.lib_dir)
            np.save(self.lib_dir + '/mask.npy', self._build_mask())
            print "Cached : ", self.lib_dir + '/mask.npy'
        lensit.pbs.barrier()
        return np.load(self.lib_dir + '/mask.npy')

    def _build_mask(self):
        ret = np.ones(self.dat_shape, dtype=float)
        for _m in self.mask_list: ret *= self._load_map(_m)
        return ret

    def apply_map(self, field, map, inplace=False):
        if inplace:
            map *= self.load_mask() / self._get_Nell(field.lower())
            return
        else:
            return map * self.load_mask() / self._get_Nell(field.lower())

    def _get_Nell(self, field):
        return (self.sN_uKamins[field.lower()] * np.pi / 180. / 60.) ** 2

    def get_MLlms(self, _type, datmaps, use_Pool=0, use_cls_len=True, chain_descr=None, opfilt=None,
                  soltn=None, dense_file='', **kwargs):
        assert np.all([_m.shape == self.dat_shape for _m in datmaps]), (datmaps.shape, self.dat_shape)
        # FIXME : use noBB only if cls_cmb has no BB
        # lmax_ivf = lmax_ivf or self.lib_skyalm.ellmax
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl

        if chain_descr is None:
            chain_descr = fs.qcinv.chain_samples.get_defaultmgchain(self.lib_skyalm.ellmax, self.lsides,
                                                                    self.dat_shape, dense_file=dense_file)
        opfilt = opfilt or (opfilt_cinv_noBB_old if np.all(cls_cmb['bb'] == 0.) else opfilt_cinv_old)
        opfilt._type = _type
        opfilt._use_Pool = use_Pool
        opfilt._use_cls_len = use_cls_len
        print "This is opfilt ", opfilt._prefix
        # _cov = self if lmax_ivf >= self.lib_skyalm.ellmax else self.degrade(self.dat_shape, lmax_ivf)
        _cov = self
        chain = multigrid.multigrid_chain(opfilt, _type, chain_descr, _cov, **kwargs)
        if soltn is None: soltn = np.zeros((opfilt.TEBlen(_type), _cov.lib_skyalm.alm_size), dtype=complex)
        return chain.solve(soltn, datmaps, finiop='MLIK')

    def get_Reslms(self, _type, datmaps, use_Pool=0, use_cls_len=True, chain_descr=None, opfilt=None,
                   **kwargs):
        assert np.all([_m.shape == self.dat_shape for _m in datmaps]), (datmaps.shape, self.dat_shape)
        lmax_ivf = self.lib_skyalm.ellmax
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl

        if chain_descr is None:
            chain_descr = fs.qcinv.chain_samples.get_defaultmgchain(lmax_ivf, self.lsides, self.dat_shape, **kwargs)
        opfilt = opfilt or (opfilt_cinv_noBB_old if np.all(cls_cmb['bb'] == 0) else opfilt_cinv_old)
        opfilt._type = _type
        opfilt._use_Pool = use_Pool
        opfilt._use_cls_len = use_cls_len
        print "This is opfilt ", opfilt._prefix
        _cov = self if lmax_ivf == self.lib_skyalm.ellmax else self.degrade(self.dat_shape, lmax_ivf)

        chain = multigrid.multigrid_chain(opfilt, _type, chain_descr, _cov)
        soltn = np.zeros((opfilt.TEBlen(_type), _cov.lib_skyalm.alm_size), dtype=complex)
        # d0 = opfilt_cinv.dot_op(opfilt_cinv.p)
        return chain.solve(soltn, datmaps, finiop='BINV')

    def get_MFqlms(self, _type, MFkey, lib_qlm, phas, use_cls_len=True, use_Pool=0, **kwargs):
        """
        xmaps : unit variance maps with correct shape
        We can write the MF as (i k_a (Pi + BNiB)^{-1}P^{-1})^a_a(x,x)
        """
        cls_cmb = self.cls_len if use_cls_len else self.cls_unl
        _wBB = (not np.all(cls_cmb['bb'] == 0.))
        if MFkey == 12:
            assert len(phas) == len(_type) and np.all([_p.shape == self.dat_shape for _p in phas]), phas.shape
            # TODO  isocov_PL143 version.
            # W1 =  B^t M^t
            # W2 = ik_a P B^t Covi. (gradients of MLIK maps)
            norm = np.prod(self.dat_shape) / (np.prod(self.lib_skyalm.lsides))
            MLik = self.get_MLlms(_type, phas, use_cls_len=use_cls_len, use_Pool=use_Pool, **kwargs)
            MLik = SMnoBB.TE2TQUlms(_type, self.lib_skyalm, MLik) if not _wBB else SM.TEB2TQUlms(_type, self.lib_skyalm,
                                                                                                 MLik)

            def Left(id):
                return self._2smap(
                    self.lib_skyalm.almxfl(self.lib_skyalm.map2alm(phas[id] * self.load_mask()), norm * self.cl_transf))

            def Right(id, ax):
                assert ax in [0, 1], ax
                kfunc = self.lib_skyalm.get_ikx if ax == 1 else self.lib_skyalm.get_iky
                return self._2smap(MLik[id] * kfunc())
        elif MFkey == 6:
            # (ik_a l^{-1/2} l^{1/2} P^12(1 + P12 BNiB P12)^{-1}P12 B^t Ni B)^a_a(x,x)
            # W1 = B^t Ni B P12 (1 + P12 BNiB P12)^{-1} P12 l^1/2
            # W2 = ik_a l^{-1/2}
            assert 0, 'Check that'
            assert (len(phas) == len(_type) - (not _wBB)) and np.all(
                [_p.size == self.lib_skyalm.alm_size for _p in phas]), phas.shape
            l_12 = np.arange(self.lib_skyalm.ellmax + 1) + 0.5
            TQU = (SM.TEB2TQUlms if _wBB else SMnoBB.TE2TQUlms)(_type, self.lib_skyalm, phas)
            opfilt = fs.qcinv.opfilt_cinv_alt if _wBB else fs.qcinv.opfilt_cinv_alt_noBB
            Mlik = self.get_MLlms(_type, phas, use_cls_len=use_cls_len,
                                  use_Pool=use_Pool, no_calcprep=True, opfilt=opfilt, **kwargs)

            def Left(id):
                return self._2smap(
                    self.lib_skyalm.almxfl(self.lib_skyalm.map2alm(phas[id] * self.load_mask()), norm * self.cl_transf))

            def Right(id, ax):
                assert ax in [0, 1], ax
                kfunc = self.lib_skyalm.get_ikx if ax == 1 else self.lib_skyalm.get_iky
                return self._2smap(self.lib_skyalm.almxfl(TQU[id], 1. / l_12) * kfunc())

        else:
            assert 0
        retdx = Left(0) * Right(0, 1)
        for i in range(1, len(_type)): retdx += Left(i) * Right(i, 1)
        retdx = lib_qlm.map2alm(retdx)
        retdy = Left(0) * Right(0, 0)
        for i in range(1, len(_type)): retdy += Left(i) * Right(i, 0)
        retdy = lib_qlm.map2alm(retdy)
        return np.array([- retdx * lib_qlm.get_ikx() - retdy * lib_qlm.get_iky(),
                         retdx * lib_qlm.get_iky() - retdy * lib_qlm.get_ikx()])  # N0  * output is normalized qest

    def apply_cond3(self, _type, alms, use_Pool=0):
        assert alms.shape == self._datalms_shape(_type), (alms.shape, self._datalms_shape(_type))
        assert _type == 'T'
        _cl = self.cls_len['tt'][0:self.lib_datalm.ellmax + 1] * self.cl_transf[0:self.lib_datalm.ellmax + 1] ** 2
        _cl += self.cls_noise['t'][0:self.lib_datalm.ellmax + 1]
        # does not work better :
        # _cl *= self.lib_datalm.ell_mat.map2cl(self.load_mask())[0:self.lib_datalm.ellmax + 1]
        ret = np.empty_like(alms)
        ret[0] = self.lib_datalm.almxfl(alms[0], lensit.ffs_covs.ffs_cov.cl_inverse(_cl))
        return ret

    def apply_noise(self, _type, alms, inverse=False):
        # F M N M^t F^t
        assert alms.shape == self._datalms_shape(_type), (alms.shape, self._datalms_shape(_type))
        ret = np.empty_like(alms)
        for _i in range(len(_type)):
            noise_fac = self._get_Nell(_type[_i].lower()) if not inverse else 1. / self._get_Nell(_type[_i].lower())
            ret[_i] = noise_fac * self.lib_datalm.map2alm(self.lib_datalm.alm2map(alms[_i]) * self.load_mask() ** 2)
        return ret

    def degrade(self, LD_shape, ellmax=None, ellmin=None, lib_dir=None, libtodegrade='sky', todiag=False, **kwargs):
        if lib_dir is None: lib_dir = self.lib_dir + \
                                      '/degraded%sx%s_%s_%s' % (LD_shape[0], LD_shape[1], ellmin,
                                                                ellmax)
        if 'sky' in libtodegrade:
            lib_dir += 'sky'
            assert ellmax < self.lib_skyalm.ell_mat.Nyq(0)
            lib_almskyLD = self.lib_skyalm.degrade(LD_shape, ellmax=ellmax, ellmin=ellmin)
        else:
            lib_almskyLD = self.lib_skyalm.clone()
        if 'dat' in libtodegrade:
            lib_dir += 'dat'
            assert ellmax < self.lib_datalm.ell_mat.Nyq(0)
            lib_almdatLD = self.lib_datalm.degrade(LD_shape, ellmax=ellmax, ellmin=ellmin)
            maskLD = (fs.misc.rfft2_utils.degrade_mask(self.load_mask(), LD_shape),)
        else:
            lib_almdatLD = self.lib_datalm.clone()
            maskLD = (self.load_mask(),)
        # print maskLD[0].shape,lib_almdatLD.shape,LD_shape,"+++++++++++"
        if todiag:
            return lensit.ffs_covs.ffs_cov.ffs_diagcov_alm(lib_dir, lib_almdatLD, self.cls_unl, self.cls_len,
                                                           self.cl_transf,
                                                           self.cls_noise, lib_skyalm=lib_almskyLD)
        return ffs_isocov_wmask(lib_dir, lib_almdatLD, self.cls_unl, self.cls_len, self.cl_transf, self.sN_uKamins,
                                lib_skyalm=lib_almskyLD, mask_list=maskLD)

    def set_ffinv(self, f, f_inv):
        return ffs_lencov_alm_wmask(self.lib_dir, self.lib_datalm, self.lib_skyalm, self.cls_unl, self.cls_len,
                                    self.cl_transf, self.sN_uKamins, f, f_inv, mask_list=self.mask_list)


class ffs_lencov_alm_wmask(ffs_cov.ffs_lencov_alm):
    def __init__(self, lib_dir, lib_datalm, lib_skyalm, cls_unl, cls_len, cl_transf, sN_uKamins, f, f_inv,
                 mask_list=()):

        cls_noise = {}
        for _k in ['t', 'q', 'u']:
            cls_noise[_k] = (sN_uKamins[_k] * np.pi / 180. / 60.) ** 2 * np.ones(lib_datalm.ellmax + 1)

        self.lib_skyalm = lib_skyalm
        for _m in mask_list:
            assert self._load_map(_m).shape == lib_datalm.ell_mat.shape, (_m.shape, lib_datalm.ell_mat.shape)
        self.mask_list = mask_list

        super(ffs_lencov_alm_wmask, self) \
            .__init__(lib_dir, lib_datalm, lib_skyalm, cls_unl, cls_len, cl_transf, cls_noise, f, f_inv)

        self.sN_uKamins = sN_uKamins
        self.mask_list = mask_list
        self.LD_shape = lib_datalm.ell_mat.shape

        self.lmax_dat = self.lib_datalm.ellmax
        self.lmax_sky = self.lib_skyalm.ellmax

        self.sky_shape = self.lib_skyalm.ell_mat.shape
        # assert self.lmax_dat <= self.lmax_sky, (self.lmax_dat, self.lmax_sky)

        for cl in self.cls_unl.values():   assert len(cl) > lib_skyalm.ellmax
        assert len(cl_transf) > self.lmax_sky

        assert f.shape == self.sky_shape and f_inv.shape == self.sky_shape, (f.shape, f_inv.shape, self.sky_shape)
        assert f.lsides == self.lsides and f_inv.lsides == self.lsides, (f.lsides, f_inv.lsides, self.lsides)
        self.f_inv = f_inv  # inverse displacement
        self.f = f  # displacement

    def hashdict(self):
        hash = {'lib_alm': self.lib_datalm.hashdict(), 'lib_skyalm': self.lib_skyalm.hashdict()}
        for key, cl in self.cls_noise.iteritems():
            hash['cls_noise ' + key] = hashlib.sha1(cl).hexdigest()
        for key, cl in self.cls_unl.iteritems():
            hash['cls_unl ' + key] = hashlib.sha1(cl).hexdigest()
        for key, cl in self.cls_len.iteritems():
            hash['cls_len ' + key] = hashlib.sha1(cl).hexdigest()
        hash['cl_transf'] = hashlib.sha1(self.cl_transf).hexdigest()
        hash['mask'] = hashlib.sha1(self.load_mask()).hexdigest()
        return hash

    def _load_map(self, _map):
        if isinstance(_map, str):
            return np.load(_map, mmap_mode='r')
        else:
            return _map

    def load_mask(self):
        ret = np.ones(self.lib_datalm.ell_mat.shape, dtype=float)
        for _m in self.mask_list: ret *= self._load_map(_m)
        return ret

    def apply_map(self, field, map, inplace=False):
        if inplace:
            map *= self.load_mask() / self._get_Nell(field.lower())
            return
        else:
            return map * self.load_mask() / self._get_Nell(field.lower())

    def _mask_alm(self, alm, forward=True):
        if forward:
            assert alm.shape == (self.lib_skyalm.alm_size,), (alm.shape, self.lib_skyalm.alm_size)
            return self.lib_datalm.map2alm(self.lib_skyalm.alm2map(alm) * self.load_mask())
        else:
            assert alm.shape == (self.lib_datalm.alm_size,), (alm.shape, self.lib_datalm.alm_size)
            return self.lib_skyalm.map2alm(self.lib_datalm.alm2map(alm) * self.load_mask())

    def apply_noise(self, _type, alms, inverse=False):
        # Input should be masked alms
        assert _type in _types, (_type, _types)
        assert alms.shape == self._datalms_shape(_type), (alms.shape, self._datalms_shape(_type))
        ret = np.empty_like(alms)
        # assert np.all(self._load_mask() == 1.)
        for _i in range(len(_type)):
            noise_fac = self._get_Nell(_type[_i].lower()) if not inverse else 1. / self._get_Nell(_type[_i].lower())
            ret[_i] = noise_fac * self.lib_datalm.map2alm(self.lib_datalm.alm2map(alms[_i]) * self.load_mask())
        return ret

    def _apply_signal(self, _type, alms, use_Pool=0):
        assert alms.shape == self._datalms_shape(_type), (alms.shape, self._datalms_shape(_type))

        t = timer(_timed, prefix=__name__, suffix=' _apply signal')
        t.checkpoint("just started")

        tempalms = np.empty(self._skyalms_shape(_type), dtype=complex)
        ret = np.empty_like(alms)
        for _i in range(len(_type)): tempalms[_i] = self._mask_alm(alms[_i], forward=False)
        tempalms = self._apply_beams(_type, tempalms)
        t.checkpoint(("masked alms and applied beams"))

        if use_Pool <= -100:
            import mllens_GPU.apply_GPU as apply_GPU
            f = self.f
            f_inv = self.f_inv
            apply_GPU.apply_FDxiDtFt_GPU_inplace(_type, self.lib_skyalm, self.lib_skyalm, tempalms, f, f_inv,
                                                 self.cls_unl)
            tempalms = self._apply_beams(_type, tempalms)
            for _i in range(len(_type)):
                ret[_i] = self._mask_alm(tempalms[_i], forward=True)
            return ret + self.apply_noise(_type, alms)

        for _i in range(len(_type)):  # Lens with inverse and mult with determinant magnification.
            tempalms[_i] = self.f_inv.lens_alm(self.lib_skyalm, tempalms[_i],
                                               mult_magn=True, use_Pool=use_Pool)
        # NB : 7 new full sky alms for TQU in this routine - > 4 GB total for full sky lmax_sky =  6000.
        t.checkpoint("backward lens + det magn")

        skyalms = np.zeros_like(tempalms)
        for j in range(len(_type)):
            for i in range(len(_type)):
                skyalms[i] += get_unlPmat_ij(_type, self.lib_skyalm, self.cls_unl, i, j) * tempalms[j]
        del tempalms
        t.checkpoint("mult with Punl mat ")

        for i in range(len(_type)):  # Lens with forward displacement
            skyalms[i] = self.f.lens_alm(self.lib_skyalm, skyalms[i], use_Pool=use_Pool)
        t.checkpoint("Forward lensing mat ")

        skyalms = self._apply_beams(_type, skyalms)
        t.checkpoint("Beams")
        for _i in range(len(_type)):
            ret[_i] = self._mask_alm(skyalms[_i], forward=True)
        t.checkpoint("masking")

        return ret

    def get_MLlms(self, _type, datmaps, use_Pool=0, use_cls_len=True, chain_descr=None, opfilt=None,
                  no_deglensing=False, soltn=None, dense_file=None, **kwargs):
        assert np.all([_m.shape == self.dat_shape for _m in datmaps]), (datmaps.shape, self.dat_shape)
        lmax_ivf = self.lib_skyalm.ellmax
        if chain_descr is None:
            chain_descr = fs.qcinv.chain_samples.get_defaultmgchain(lmax_ivf, self.lsides, self.dat_shape,
                                                                    dense_file=dense_file)
        opfilt = opfilt or (opfilt_cinv_noBB_wl if np.all(self.cls_unl['bb'] == 0) else opfilt_cinv_wl_old)
        opfilt._type = _type
        opfilt._use_Pool = use_Pool
        opfilt._use_cls_len = use_cls_len
        print "This is opfilt ", opfilt._prefix
        _cov = self
        chain = multigrid.multigrid_chain(opfilt, _type, chain_descr, _cov, no_deglensing=no_deglensing)
        if soltn is None:
            soltn = np.zeros((opfilt.TEBlen(_type), _cov.lib_skyalm.alm_size), dtype=complex)
        return chain.solve(soltn, datmaps, finiop='MLIK')

    def get_Reslms(self, _type, datmaps, use_Pool=0, use_cls_len=False, chain_descr=None, opfilt=None,
                   **kwargs):
        assert np.all([_m.shape == self.dat_shape for _m in datmaps]), (datmaps.shape, self.dat_shape)
        lmax_ivf = self.lib_skyalm.ellmax
        if chain_descr is None:
            chain_descr = fs.qcinv.chain_samples.get_defaultmgchain(lmax_ivf, self.lsides, self.dat_shape, **kwargs)
        opfilt = opfilt or (opfilt_cinv_noBB_wl if np.all(self.cls_unl['bb'] == 0) else opfilt_cinv_wl_old)
        opfilt._type = _type
        opfilt._use_Pool = use_Pool
        opfilt._use_cls_len = use_cls_len
        print "This is opfilt ", opfilt._prefix
        _cov = self

        chain = multigrid.multigrid_chain(opfilt, _type, chain_descr, _cov)
        soltn = np.zeros((opfilt.TEBlen(_type), _cov.lib_skyalm.alm_size), dtype=complex)
        # d0 = opfilt_cinv.dot_op(opfilt_cinv.p)
        return chain.solve(soltn, datmaps, finiop='BINV')

    def apply_cond3(self, _type, alms, use_Pool=0):
        """
        (DBxiB ^ tD ^ t + N) ^ -1 \sim D ^ -t(BxiBt + N) ^ -1 D ^ -1
        :param alms:
        :return:
        """
        assert alms.shape == self._datalms_shape(_type), (alms.shape, self._datalms_shape(_type))
        t = timer(_timed)
        t.checkpoint("  cond3::just started")
        tempalms = np.empty(self._skyalms_shape(_type), dtype=complex)
        for _i in range(len(_type)): tempalms[_i] = self._mask_alm(alms[_i], forward=False)
        t.checkpoint("  cond3::masked alms")

        if use_Pool <= -100:
            # Try entire evaluation on GPU :
            from fs.gpu.apply_cond3_GPU import apply_cond3_GPU_inplace as c3GPU
            c3GPU(_type, self.lib_skyalm, tempalms, self.f, self.f_inv, self.cls_unl, self.cl_transf, self.cls_noise)
            ret = np.empty_like(alms)
            for _i in range(len(_type)):
                ret[_i] = self._mask_alm(tempalms[_i], forward=True)
            return ret
        temp = np.empty_like(alms)
        for i in range(len(_type)):  # D^{-1}
            temp[i] = self._deg(self.f_inv.lens_alm(self.lib_skyalm, tempalms[i], use_Pool=use_Pool))

        t.checkpoint("  cond3::Lensing with inverse")

        ret = np.zeros_like(alms)  # (B xi B^t + N)^{-1}
        for i in range(len(_type)):
            for j in range(len(_type)):
                ret[i] += self.get_Pmatinv(_type, i, j) * temp[j]
        del temp
        t.checkpoint("  cond3::Mult. w. inv Pmat")

        for i in range(len(_type)):  # D^{-t}
            tempalms[i] = self.f.lens_alm(self.lib_skyalm, self._upg(ret[i]), use_Pool=use_Pool, mult_magn=True)

        t.checkpoint("  cond3::Lens w. forward and det magn.")
        for _i in range(len(_type)):
            ret[_i] = self._mask_alm(tempalms[_i], forward=True)
        return ret

    def degrade(self, LD_shape, no_lensing=False, ellmax=None, ellmin=None, lib_dir=None, libtodegrade='sky'):
        if lib_dir is None: lib_dir = self.lib_dir + '/%sdegraded%sx%s_%s_%s' % \
                                                     ({True: 'unl', False: 'len'}[no_lensing], LD_shape[0], LD_shape[1],
                                                      ellmin, ellmax)
        if 'sky' in libtodegrade:
            lib_dir += 'sky'
            lib_almskyLD = self.lib_skyalm.degrade(LD_shape, ellmax=ellmax, ellmin=ellmin)
            fLD = self.f.degrade(LD_shape, no_lensing)
            finvLD = self.f_inv.degrade(LD_shape, no_lensing)
        else:
            lib_almskyLD = self.lib_skyalm.clone()
            fLD = self.f
            finvLD = self.f_inv
        if 'dat' in libtodegrade:
            lib_dir += 'dat'
            lib_almdatLD = self.lib_datalm.degrade(LD_shape, ellmax=ellmax, ellmin=ellmin)
            maskLD = (fs.misc.rfft2_utils.degrade_mask(self.load_mask(), LD_shape),)
        else:
            lib_almdatLD = self.lib_datalm.clone()
            maskLD = (self.load_mask(),)

        if no_lensing:
            return ffs_isocov_wmask(lib_dir, lib_almdatLD, self.cls_unl, self.cls_len, self.cl_transf,
                                    self.sN_uKamins, mask_list=maskLD, lib_skyalm=lib_almskyLD)

        return ffs_lencov_alm_wmask(lib_dir, lib_almdatLD, lib_almskyLD, self.cls_unl, self.cls_len, self.cl_transf,
                                    self.sN_uKamins, fLD, finvLD, mask_list=maskLD)
