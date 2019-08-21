from __future__ import print_function

import numpy as np

from lensit.qcinv import template_removal
from lensit.misc.misc_utils import cls_hash, npy_hash
from lensit.ffs_covs import ffs_specmat
from lensit.misc.rfft2_utils import degrade_mask

load_map = lambda _map: np.load(_map) if type(_map) is str else _map


def cl_inv(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

class ffs_ninv_filt(object):
    def __init__(self, lib_datalm, lib_skyalm, len_cls, cl_transf, ninv_rad,
                 marge_maps=None, marge_uptolmin=None,marge_ells = None,eigv_thres = 0., cls_noise=None, verbose=False):
        """
        ninv_rad is the inverse variance map in 1 / rad ** 2, not the pixel variance maps.
        This is the inverse pixel variance map / volume of cell.

        eigenvalues in the template matrix with eigv <= max(eigv) * eigv_thres will be set to zero before inversion
        """
        if marge_maps is None : marge_maps = dict()
        if marge_uptolmin is None : marge_uptolmin = dict()
        if marge_ells is None : marge_ells = dict()

        if cls_noise is None: cls_noise = dict()

        self.ninv_rad = ninv_rad
        self.lib_datalm = lib_datalm
        self.lib_skyalm = lib_skyalm
        self.cl_transf = cl_transf[:lib_skyalm.ellmax + 1]
        self.cls = {}
        for key in len_cls.keys():
            self.cls[key] = len_cls[key][:lib_skyalm.ellmax + 1]
        self.npix = np.prod(lib_datalm.shape)
        # Build mean noise level :
        self._iNoiseCl = {}
        for key in ninv_rad.keys():
            if key not in cls_noise.keys():
                _noiseCl = np.mean(1. / ninv_rad[key][np.where(ninv_rad[key] > 0)])
                self._iNoiseCl[key] = 1. / _noiseCl * np.ones(lib_skyalm.ellmax + 1, dtype=float)
                if verbose:
                    print("ninv_filt::Nlev (%s) in uKamin  %.3f" % (
                        key.upper(), np.sqrt(1. / self._iNoiseCl[key][0]) / np.pi * 180. * 60.))
                    print("      ellsky range (%s - %s)" % (lib_skyalm.ellmin, lib_skyalm.ellmax))
            else:
                # This does not appear to generically improve
                self._iNoiseCl[key] = cl_inv(cls_noise[key][:lib_skyalm.ellmax + 1])

        templates = {key: [] for key in self.ninv_rad.keys()}
        for f in marge_maps.keys():
            if f not in templates.keys(): templates[f] = []
            for tmap in [load_map(m) for m in marge_maps[f]]:
                assert (self.npix == len(tmap))
                templates[f].append(template_removal.template_map(tmap))

        for f in marge_uptolmin.keys():
            if marge_uptolmin[f] >= 0:
                if f not in templates.keys(): templates[f] = []
                templates[f].append(template_removal.template_uptolmin(lib_datalm.ell_mat, marge_uptolmin[f]))

        for f in marge_ells.keys():
            if marge_ells[f] >= 0:
                if f not in templates.keys(): templates[f] = []
                templates[f].append(template_removal.template_ellfilt(lib_datalm.ell_mat, marge_ells[f]))


        assert np.all([f in ninv_rad.keys() for f in templates.keys()]), (ninv_rad.keys(), templates.keys())
        self.Pt_Nn1_P_inv = {}
        for key in templates.keys():
            templs = templates[key]
            if len(templs) > 0:
                nmodes = int(np.sum([t.nmodes for t in templs]))
                modes_idx_t = np.concatenate(([t.nmodes * [int(im)] for im, t in enumerate(templs)]))
                modes_idx_i = np.concatenate(([range(0, t.nmodes) for t in templs]))
                print("   Building %s - %s template projection matrix" % (nmodes, nmodes))
                Pt_Nn1_P = np.zeros((nmodes, nmodes))
                for ir in range(0, nmodes):
                    if np.mod(ir, int(0.1 * nmodes)) == 0: print("   filling TNiT: %4.1f" % (100. * ir / nmodes)), "%"
                    tmap = np.copy(ninv_rad[key])
                    templs[modes_idx_t[ir]].apply_mode(tmap, int(modes_idx_i[ir]))
                    ic = 0
                    for tc in templs[0:modes_idx_t[ir] + 1]:
                        Pt_Nn1_P[ir, ic:(ic + tc.nmodes)] = tc.dot(tmap)
                        Pt_Nn1_P[ic:(ic + tc.nmodes), ir] = Pt_Nn1_P[ir, ic:(ic + tc.nmodes)]
                        ic += tc.nmodes
                eigv, eigw = np.linalg.eigh(Pt_Nn1_P)
                eigv_inv = np.where(eigv <= np.max(eigv) * eigv_thres,np.zeros_like(eigv),1.0 / eigv)
                if eigv_thres > 0: print("Kept %.2f percent of the modes"%( (100. * len(eigv_inv.nonzero())) / len(eigv)))
                self.Pt_Nn1_P_inv[key] = np.dot(np.dot(eigw, np.diag(eigv_inv)), np.transpose(eigw))
        self.templates = templates
        self.marge_uptolmin = marge_uptolmin
        self.cls_noise = cls_noise
        self.marge_maps = marge_maps

    def hashdict(self):
        #FIXME:
        return {'transf':cls_hash({'transf':self.cl_transf}, astype=np.float32),'cls':cls_hash(self.cls, astype=np.float32),
                'ninv':{k:npy_hash(self.ninv_rad[k], astype=np.float32) for k in self.ninv_rad.keys()},
                'marge_lmin':self.marge_uptolmin,'lib_datalm':self.lib_datalm.hashdict(),'lib_skyalm':self.lib_skyalm.hashdict()}

    def Nlev_uKamin(self, field):
        return np.sqrt(np.mean(1. / self.ninv_rad[field.lower()][np.where(self.ninv_rad[field.lower()] > 0)])) \
               * 180. * 60 / np.pi

    def _deg(self, skyalm):
        assert skyalm.shape == (self.lib_skyalm.alm_size,), (skyalm.shape, self.lib_skyalm.alm_size)
        if self.lib_skyalm.iseq(self.lib_datalm, allow_shape=True): return skyalm
        return self.lib_datalm.udgrade(self.lib_skyalm, skyalm)

    def _upg(self, datalm):
        assert datalm.shape == (self.lib_datalm.alm_size,), (datalm.shape, self.lib_datalm.alm_size)
        if self.lib_datalm.iseq(self.lib_skyalm, allow_shape=True): return datalm
        return self.lib_skyalm.udgrade(self.lib_datalm, datalm)

    def get_mask(self, field):
        ret = np.ones(self.lib_datalm.shape, dtype=float)
        ret[np.where(self.ninv_rad[field.lower()] <= 0.)] = 0.
        return ret

    def _get_rmspixnoise(self, field):
        # rms pixel noise map in
        vcell = np.prod(self.lib_datalm.lsides) / np.prod(self.lib_datalm.shape)
        ret = np.zeros(self.lib_datalm.shape)
        ii = np.where(self.ninv_rad[field.lower()] > 0)
        if len(ii[0]) > 0:
            ret[ii] = np.sqrt(1. / self.ninv_rad[field.lower()][ii] / vcell)
        return ret

    def get_cl_transf(self,lab):
        return self.cl_transf

    def get_nTpix(self):
        return self._get_rmspixnoise('T')

    def get_nQpix(self):
        return self._get_rmspixnoise('Q')

    def get_nUpix(self):
        return self._get_rmspixnoise('U')

    def apply_R(self, field, alm):
        """
        Apply transfer function, T Q U skyalm to map.
        B
        """
        assert field.lower() in ['t', 'q', 'u'], field
        assert alm.size == self.lib_skyalm.alm_size, (alm.size, self.lib_skyalm.alm_size)
        return self.lib_datalm.alm2map(self._deg(self.lib_skyalm.almxfl(alm, self.cl_transf)))

    def apply_Rs(self, TQUtype,TEBlms):
        """
        Apply transfer function, T E B skyalm to T Q U map.
        """
        assert len(TQUtype) == len(TEBlms),(len(TQUtype),len(TEBlms))
        return np.array([self.apply_R(f.lower(),alm) for f,alm in zip(TQUtype,ffs_specmat.TEB2TQUlms(TQUtype,self.lib_skyalm,TEBlms))])

    def apply_Rt(self, field, _map):
        """
        Apply tranposed transfer function, from T Q U real space to T Q U skyalm.
        B^t
        """
        assert field.lower() in ['t', 'q', 'u'], field
        assert _map.size == self.npix, (self.npix, _map.shape)
        return self.lib_skyalm.almxfl(self._upg(self.lib_datalm.map2alm(_map)), self.cl_transf)

    def apply_Rts(self, TQUtype,_maps):
        """
        Apply tranposed transfer function, from T Q U real space to T E B skyalm.
        B^t
        """
        assert TQUtype in ['T','QU','TQU']
        assert _maps.shape == (len(TQUtype),self.lib_datalm.shape[0],self.lib_datalm.shape[1]), (self.npix, TQUtype,_maps.shape)
        return ffs_specmat.TQU2TEBlms(TQUtype,self.lib_skyalm,np.array([self.apply_Rt(f.lower(),_map) for f,_map in zip(TQUtype,_maps)]))

    def apply_alm(self, field, alm, inplace=True):
        """
        Applies B^t Ni B to T, Q or U lms.
        """
        assert field.lower() in ['t', 'q', 'u'], field
        assert alm.size == self.lib_skyalm.alm_size, (alm.size, self.lib_skyalm.alm_size)
        assert inplace
        if inplace:
            self.lib_skyalm.almxfl(alm, self.cl_transf, inplace=True)
            _map = self.lib_datalm.alm2map(self._deg(alm))
            self.apply_map(field, _map, inplace=True)
            alm[:] = self._upg(self.lib_datalm.map2alm(_map))
            self.lib_skyalm.almxfl(alm, self.cl_transf, inplace=True)
            return

    def apply_alms(self,TQUtype, TEBalms, inplace=True):
        """
        Applies B^t Ni B. (TEB skyalms to TEB skyalms)
        """
        assert TQUtype in ['T','QU','TQU']
        if inplace:
            TEBalms[:] = self.apply_Rts(TQUtype,self.apply_maps(TQUtype,self.apply_Rs(TQUtype,TEBalms),inplace=False))
            return
        else:
            return self.apply_Rts(TQUtype,self.apply_maps(TQUtype,self.apply_Rs(TQUtype,TEBalms),inplace=False))

    def apply_map(self, field, _map, inplace=True):
        """
        Applies ninv to real space T, Q, or U map, in radians units.
        """
        assert _map.size == self.npix, (self.npix, _map.shape)
        assert field.lower() in ['t', 'q', 'u'], field
        f = field.lower()
        if inplace:
            _map *= self.ninv_rad[f]
            if len(self.templates[f]) > 0:
                coeffs = np.concatenate(([t.dot(_map) for t in self.templates[f]]))
                coeffs = np.dot(self.Pt_Nn1_P_inv[f], coeffs)
                pmodes = np.zeros(self.ninv_rad[f].shape)
                im = 0
                for t in self.templates[f]:
                    t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                    im += t.nmodes
                pmodes *= self.ninv_rad[f]
                _map -= pmodes
            return
        else:
            nmap = _map * self.ninv_rad[f]
            if len(self.templates[f]) > 0:
                coeffs = np.concatenate(([t.dot(nmap) for t in self.templates[f]]))
                coeffs = np.dot(self.Pt_Nn1_P_inv[f], coeffs)
                pmodes = np.zeros(self.ninv_rad[f].shape)
                im = 0
                for t in self.templates[f]:
                    t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                    im += t.nmodes
                pmodes *= self.ninv_rad[f]
                nmap -= pmodes
            return nmap

    def apply_maps(self, TQUtype, _maps, inplace=True):
        """
        Applies ninv to real space T, Q, or U map, in radians units.
        """
        assert _maps.shape == (len(TQUtype),self.lib_datalm.shape[0],self.lib_datalm.shape[1]), (self.npix, TQUtype,_maps.shape)
        assert TQUtype in ['T','QU','TQU']
        if inplace:
            for i, f in enumerate(TQUtype.lower()):
                _maps[i] *= self.ninv_rad[f]
                if len(self.templates[f]) > 0:
                    coeffs = np.concatenate(([t.dot(_maps[i]) for t in self.templates[f]]))
                    coeffs = np.dot(self.Pt_Nn1_P_inv[f], coeffs)
                    pmodes = np.zeros(self.ninv_rad[f].shape)
                    im = 0
                    for t in self.templates[f]:
                        t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                        im += t.nmodesx
                    pmodes *= self.ninv_rad[f]
                    _maps[i] -= pmodes
            return
        else:
            nmaps = np.zeros_like(_maps)
            for i, f in enumerate(TQUtype.lower()):
                nmaps[i] = _maps[i] * self.ninv_rad[f]
                if len(self.templates[f]) > 0:
                    coeffs = np.concatenate(([t.dot(nmaps[i]) for t in self.templates[f]]))
                    coeffs = np.dot(self.Pt_Nn1_P_inv[f], coeffs)
                    pmodes = np.zeros(self.ninv_rad[f].shape)
                    im = 0
                    for t in self.templates[f]:
                        t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                        im += t.nmodes
                    pmodes *= self.ninv_rad[f]
                    nmaps[i] -= pmodes
            return nmaps

    def iNoiseCl(self, field):
        return self._iNoiseCl[field.lower()]

    def degrade(self, shape, ellmax=None, ellmin=None, **kwargs):
        lib_almsky = self.lib_skyalm.degrade(shape, ellmax=ellmax, ellmin=ellmin)
        lib_almdat = self.lib_datalm.degrade(shape, ellmax=ellmax, ellmin=ellmin)
        ninvLD = {}
        for key in self.ninv_rad.keys():
            ninvLD[key] = degrade_mask(self.ninv_rad[key], shape)
        return ffs_ninv_filt(lib_almdat, lib_almsky, self.cls, self.cl_transf, ninvLD,
                             marge_uptolmin=self.marge_uptolmin, cls_noise=self.cls_noise)

    def turn2wlfilt(self, f, fi):
        return ffs_ninv_filt_wl(self.lib_datalm, self.lib_skyalm, self.cls, self.cl_transf, self.ninv_rad, f, fi,
                                marge_maps=self.marge_maps, marge_uptolmin=self.marge_uptolmin,
                                cls_noise=self.cls_noise)

    def turn2isofilt(self):
        """Returns an isotropic (no mask, homog. noise) filter built from the average noise levels.

        """
        ninv_rad = {}
        for key in self.ninv_rad.keys():
            ninv_rad[key] = np.ones(self.lib_datalm.shape, dtype=float) * (1. / (self.Nlev_uKamin(key) / 60 / 180. * np.pi) ** 2)
        return ffs_ninv_filt(self.lib_datalm, self.lib_skyalm, self.cls, self.cl_transf, ninv_rad,
                             marge_maps=self.marge_maps, marge_uptolmin=self.marge_uptolmin)


class ffs_ninv_filt_wl(ffs_ninv_filt):
    def __init__(self, lib_datalm, lib_skyalm, unl_cls, cl_transf, ninv_rad, f, fi,
                 marge_maps=None, marge_uptolmin=None, cls_noise=None, lens_pool=0):
        """
        Same as above, but the transfer functions contain the lensing.
        Note that the degradation will kill the lensing.
        """
        super(ffs_ninv_filt_wl, self).__init__(lib_datalm, lib_skyalm, unl_cls, cl_transf, ninv_rad,
                                    marge_maps=marge_maps, marge_uptolmin=marge_uptolmin, cls_noise=cls_noise)
        # Forward and inverse displacement instances :
        assert self.lib_skyalm.shape == f.shape and self.lib_skyalm.lsides == f.lsides
        assert self.lib_skyalm.shape == fi.shape and self.lib_skyalm.lsides == fi.lsides
        self.f = f
        self.fi = fi
        self.lens_pool = lens_pool

    def set_ffi(self, f, fi):
        assert self.lib_skyalm.shape == f.shape and self.lib_skyalm.lsides == f.lsides
        assert self.lib_skyalm.shape == fi.shape and self.lib_skyalm.lsides == fi.lsides
        self.f = f
        self.fi = fi

    def apply_R(self, field, alm):
        """
        Apply transfer function, T Q U skyalm to map.
        B D
        """
        assert field.lower() in ['t', 'q', 'u'], field
        assert alm.size == self.lib_skyalm.alm_size, (alm.size, self.lib_skyalm.alm_size)
        _alm = self.f.lens_alm(self.lib_skyalm, alm, use_Pool=self.lens_pool)
        return self.lib_datalm.alm2map(self._deg(self.lib_skyalm.almxfl(_alm, self.cl_transf)))

    def apply_Rt(self, field, _map):
        """
        Apply tranposed transfer function, from T Q U real space to T Q U skyalm.
        D^t B^t
        """
        assert field.lower() in ['t', 'q', 'u'], field
        assert _map.size == self.npix, (self.npix, _map.shape)
        skyalm = self.lib_skyalm.almxfl(self._upg(self.lib_datalm.map2alm(_map)), self.cl_transf)
        return self.fi.lens_alm(self.lib_skyalm, skyalm, use_Pool=self.lens_pool, mult_magn=True)

    def apply_alm(self, field, alm, inplace=True):
        """
        Applies D^t B^T Ni B D to T, Q or U lms.
        """
        assert field.lower() in ['t', 'q', 'u'], field
        assert alm.size == self.lib_skyalm.alm_size, (alm.size, self.lib_skyalm.alm_size)
        assert inplace
        if inplace:
            alm[:] = self.f.lens_alm(self.lib_skyalm, alm, use_Pool=self.lens_pool)
            self.lib_skyalm.almxfl(alm, self.cl_transf, inplace=True)
            _map = self.lib_datalm.alm2map(self._deg(alm))
            self.apply_map(field, _map, inplace=True)
            alm[:] = self._upg(self.lib_datalm.map2alm(_map))
            self.lib_skyalm.almxfl(alm, self.cl_transf, inplace=True)
            alm[:] = self.fi.lens_alm(self.lib_skyalm, alm, use_Pool=self.lens_pool, mult_magn=True)
            return

    def degrade(self, shape, no_lensing=False, ellmax=None, ellmin=None, **kwargs):
        lib_almsky = self.lib_skyalm.degrade(shape, ellmax=ellmax, ellmin=ellmin)
        lib_almdat = self.lib_datalm.degrade(shape, ellmax=ellmax, ellmin=ellmin)
        ninvLD = {}
        for key in self.ninv_rad.keys():
            ninvLD[key] = degrade_mask(self.ninv_rad[key], shape)
        print("DEGRADING WITH NO MARGE MAPS")
        if no_lensing:
            return ffs_ninv_filt(lib_almdat, lib_almsky, self.cls, self.cl_transf, ninvLD,
                                 marge_uptolmin=self.marge_uptolmin, cls_noise=self.cls_noise)
        else:
            fLD = self.f.degrade(shape, no_lensing)
            fiLD = self.fi.degrade(shape, no_lensing)
            return ffs_ninv_filt_wl(lib_almdat, lib_almsky, self.cls, self.cl_transf, ninvLD, fLD, fiLD,
                                    marge_uptolmin=self.marge_uptolmin, cls_noise=self.cls_noise)