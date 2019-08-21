# FIXME : might think about passing 2 sets of Cls, the correct one and the preconditioner.
from __future__ import print_function

import numpy as np

from lensit.ffs_covs import ffs_specmat

class ffs_ninv_filt(object):
    """B^t C B + N where everyting is ideal and projected onto subset of modes set by lib_datalm.

    """

    def __init__(self, lib_datalm, lib_skyalm, len_cls, cl_transf, nlev_t, nlev_p, verbose=False):

        self.lib_datalm = lib_datalm
        self.lib_skyalm = lib_skyalm
        self.cl_transf = (cl_transf[:lib_skyalm.ellmax + 1]).copy()
        self.cls = {}
        for k in len_cls.keys():
            self.cls[k] = (len_cls[k][:lib_skyalm.ellmax + 1]).copy()
        self.nlevs = {'t': nlev_t, 'q': nlev_p, 'u': nlev_p}
        self._nlevs_rad2 = {'t': (nlev_t / 180 / 60. * np.pi) ** 2,
                            'q': (nlev_p / 180 / 60. * np.pi) ** 2,
                            'u': (nlev_p / 180 / 60. * np.pi) ** 2}
        self.verbose = verbose

    def hashdict(self):
        return {}

    def set_cls(self, cls):
        """Update the filter cmb cls

        """
        for k in self.cls.keys():
            if k in cls.keys():
                self.cls[k] = (cls[k][:self.lib_skyalm.ellmax + 1]).copy()

    def iNoiseCl(self, field):
        # FIXME : should I put zero after datalm ?
        ret = np.zeros(self.lib_skyalm.ellmax + 1)
        ret[:self.lib_datalm.ellmax + 1] = 1. / (self._nlevs_rad2[field.lower()])
        return ret
        # return np.ones(self.lib_skyalm.ellmax + 1) / (self._nlevs_rad2[field.lower()])

    def Nlev_uKamin(self, field):
        return self.nlevs[field.lower()]

    def _deg(self, skyalm):
        assert skyalm.shape == (self.lib_skyalm.alm_size,), (skyalm.shape, self.lib_skyalm.alm_size)
        if self.lib_skyalm.iseq(self.lib_datalm, allow_shape=True): return skyalm
        return self.lib_datalm.udgrade(self.lib_skyalm, skyalm)

    def _upg(self, datalm):
        assert datalm.shape == (self.lib_datalm.alm_size,), (datalm.shape, self.lib_datalm.alm_size)
        if self.lib_datalm.iseq(self.lib_skyalm, allow_shape=True): return datalm
        return self.lib_skyalm.udgrade(self.lib_datalm, datalm)

    def get_mask(self, field):
        return np.ones(self.lib_datalm.shape, dtype=float)

    def apply_R(self, field, alm):
        """
        Apply transfer function, T Q U skyalm to map.
        B
        """
        assert field.lower() in ['t', 'q', 'u'], field
        assert alm.size == self.lib_skyalm.alm_size, (alm.size, self.lib_skyalm.alm_size)
        return self._deg(self.lib_skyalm.almxfl(alm, self.cl_transf))

    def apply_Rs(self, TQUtype, TEBlms):
        """
        Apply transfer function, T E B skyalm to T Q U map.
        """
        assert len(TQUtype) == len(TEBlms),(len(TQUtype),len(TEBlms))
        return np.array([self.apply_R(f.lower(),alm) for f,alm in zip(TQUtype, ffs_specmat.TEB2TQUlms(TQUtype,self.lib_skyalm,TEBlms))])

    def apply_Rt(self, field, _map):
        """
        Apply tranposed transfer function, from T Q U real space to T Q U skyalm.
        B^t
        """
        assert field.lower() in ['t', 'q', 'u'], field
        assert _map.size == self.lib_datalm.alm_size, (_map.size, self.lib_datalm.alm_size)
        return self.lib_skyalm.almxfl(self._upg(_map), self.cl_transf)

    def apply_Rts(self, TQUtype,_maps):
        """
        Apply tranposed transfer function, from T Q U real space to T E B skyalm.
        B^t
        """
        assert TQUtype in ['T','QU','TQU']
        assert _maps.shape == (len(TQUtype),self.lib_datalm.alm_size), (self.lib_datalm.alm_size,_maps.shape, len(TQUtype))
        return ffs_specmat.TQU2TEBlms(TQUtype,self.lib_skyalm,np.array([self.apply_Rt(f.lower(),_map) for f,_map in zip(TQUtype,_maps)]))

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

    def apply_alm(self, field, alm, inplace=True):
        """
        Applies B^t Ni B to T, Q or U lms.
        """
        assert field.lower() in ['t', 'q', 'u'], field
        assert alm.size == self.lib_skyalm.alm_size, (alm.size, self.lib_skyalm.alm_size)
        assert inplace
        if inplace:
            self.lib_skyalm.almxfl(alm, self.cl_transf, inplace=True)
            _map = self._deg(alm)
            self.apply_map(field, _map, inplace=True)
            alm[:] = self._upg(_map)
            self.lib_skyalm.almxfl(alm, self.cl_transf, inplace=True)
            return

    def apply_map(self, field, _map, inplace=True):
        """
        Applies ninv to real space T, Q, or U map, in radians units.
        """
        assert _map.size == self.lib_datalm.alm_size, (_map.size, self.lib_datalm.alm_size)
        assert field.lower() in ['t', 'q', 'u'], field
        if inplace:
            _map[:] *= (1. / self._nlevs_rad2[field.lower()])
        else:
            return _map * (1. / self._nlevs_rad2[field.lower()])

    def apply_maps(self, TQUtype, _maps, inplace=True):
        """
        Applies ninv to real space T, Q, or U map, in radians units.
        """
        if inplace:
            for i, field in enumerate(TQUtype):
                assert field.lower() in ['t', 'q', 'u'], field
                assert _maps[i].size == self.lib_datalm.alm_size, (_maps[i].size, self.lib_datalm.alm_size)
                _maps[i][:] *= (1. / self._nlevs_rad2[field.lower()])
        else:
            ret = np.zeros_like(_maps)
            for i, field in enumerate(TQUtype):
                assert field.lower() in ['t', 'q', 'u'], field
                assert _maps[i].size == self.lib_datalm.alm_size, (_maps[i].size, self.lib_datalm.alm_size)
                ret[i] =_maps[i] * (1. / self._nlevs_rad2[field.lower()])
            return ret


    def turn2wlfilt(self, f, fi, cls_unl=None, lens_pool=0):
            assert self.nlevs['q'] == self.nlevs['u']
            cls = self.cls if cls_unl is None else cls_unl
            return ffs_ninv_filt_wl(self.lib_datalm, self.lib_skyalm, cls, self.cl_transf, self.nlevs['t'], self.nlevs['q'],
                                    f, fi, verbose=self.verbose, lens_pool=lens_pool)

    def turn2isofilt(self):
        assert self.nlevs['q'] == self.nlevs['u']
        return ffs_ninv_filt(self.lib_datalm, self.lib_skyalm, self.cls, self.cl_transf, self.nlevs['t'],
                             self.nlevs['q'],
                             verbose=self.verbose)

    def get_cl_transf(self,lab):
        return self.cl_transf


    def degrade(self, shape, ellmax=None, ellmin=None, **kwargs):
        lib_almsky = self.lib_skyalm.degrade(shape, ellmax=ellmax, ellmin=ellmin)
        lib_almdat = self.lib_datalm.degrade(shape, ellmax=ellmax, ellmin=ellmin)
        assert self.nlevs['q'] == self.nlevs['u']
        return ffs_ninv_filt(lib_almdat, lib_almsky, self.cls, self.cl_transf, self.nlevs['t'], self.nlevs['q'],verbose=self.verbose)


class ffs_ninv_filt_wl(ffs_ninv_filt):
    def __init__(self, lib_datalm, lib_skyalm, unl_cls, cl_transf, nlev_t, nlev_p, f, fi, verbose=False, lens_pool=0):
        super(ffs_ninv_filt_wl, self).__init__(lib_datalm, lib_skyalm, unl_cls, cl_transf, nlev_t, nlev_p,
                                               verbose=verbose)
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
        return self._deg(self.lib_skyalm.almxfl(_alm, self.cl_transf))

    def apply_Rt(self, field, _map):
        """
        Apply tranposed transfer function, from T Q U real space to T Q U skyalm.
        D^t B^t
        """
        assert field.lower() in ['t', 'q', 'u'], field
        assert _map.size == self.lib_datalm.alm_size, (_map.size, self.lib_datalm.alm_size)
        skyalm = self.lib_skyalm.almxfl(self._upg(_map), self.cl_transf)
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
            _map = self._deg(alm)
            self.apply_map(field, _map, inplace=True)
            alm[:] = self._upg(_map)
            self.lib_skyalm.almxfl(alm, self.cl_transf, inplace=True)
            alm[:] = self.fi.lens_alm(self.lib_skyalm, alm, use_Pool=self.lens_pool, mult_magn=True)
            return

    def degrade(self, shape, no_lensing=False, ellmax=None, ellmin=None, **kwargs):
        lib_almsky = self.lib_skyalm.degrade(shape, ellmax=ellmax, ellmin=ellmin)
        lib_almdat = self.lib_datalm.degrade(shape, ellmax=ellmax, ellmin=ellmin)
        assert self.nlevs['q'] == self.nlevs['u']
        if no_lensing:
            return ffs_ninv_filt(lib_almdat, lib_almsky, self.cls, self.cl_transf, self.nlevs['t'], self.nlevs['q'],
                                 verbose=self.verbose)
        else:
            fLD = self.f.degrade(shape, no_lensing)
            fiLD = self.fi.degrade(shape, no_lensing)
            return ffs_ninv_filt_wl(lib_almdat, lib_almsky, self.cls, self.cl_transf, self.nlevs['t'], self.nlevs['q'],
                                    fLD, fiLD,
                                    verbose=self.verbose)
