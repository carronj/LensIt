"""Module using nufft instead of usual bicubic spline"""
from __future__ import print_function, annotations


from lensit.ffs_covs import ffs_specmat
from lensit.ffs_covs.ell_mat import ffs_alm
from lensit.qcinv.ffs_ninv_filt_ideal import ffs_ninv_filt
from lensit.ffs_deflect.ffs_deflect import ffs_displacement



class ffs_ninv_filt_wl(ffs_ninv_filt):
    def __init__(self, lib_datalm: ffs_alm, lib_skyalm: ffs_alm, unl_cls, cl_transf, nlev_t, nlev_p, f: ffs_displacement,
                 verbose=False, lens_pool=0, kapprox=False):
        super(ffs_ninv_filt_wl, self).__init__(lib_datalm, lib_skyalm, unl_cls, cl_transf, nlev_t, nlev_p,
                                               verbose=verbose)
        assert self.lib_skyalm.shape == f.shape and self.lib_skyalm.lsides == f.lsides
        self.f = f
        self.lens_pool = lens_pool
        self.kapprox=kapprox


    def set_ffi(self, f, *args):
        assert self.lib_skyalm.shape == f.shape and self.lib_skyalm.lsides == f.lsides
        self.f = f

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
        return self.f.lens_alm_adjoint(self.lib_skyalm, skyalm)

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
            alm[:] = self.f.lens_alm_adjoint(self.lib_skyalm, alm)
            return

    def apply_alms(self,TQUtype, TEBalms, inplace=True):
        """
        Applies B^t Ni B. (TEB skyalms to TEB skyalms)
        """
        assert TQUtype in ['T','QU','TQU']
        if inplace:
            if not self.kapprox:
                TEBalms[:] = self.apply_Rts(TQUtype,self.apply_maps(TQUtype,self.apply_Rs(TQUtype,TEBalms),inplace=False))
            else:
                assert 0, 'cant use kapprox here'
                print('kapprox to filt')
                TQUlms = ffs_specmat.TEB2TQUlms(TQUtype, self.lib_skyalm, TEBalms)
                for field, alm in zip(TQUtype, TQUlms):
                    self.lib_skyalm.almxfl(alm, self.cl_transf, inplace=True)
                    alm[:] = self.lib_skyalm.map2alm(self.lib_skyalm.alm2map(alm) * self.fi.get_detmagn())
                    _map = self._deg(alm)
                    self.apply_map(field, _map, inplace=True)
                    alm[:] = self._upg(_map)
                    self.lib_skyalm.almxfl(alm, self.cl_transf, inplace=True)
                TEBalms[:] = ffs_specmat.TQU2TEBlms(TQUtype, self.lib_skyalm, TQUlms)
            return
        else:
            if not self.kapprox:
                return self.apply_Rts(TQUtype,self.apply_maps(TQUtype,self.apply_Rs(TQUtype,TEBalms),inplace=False))
            else:
                assert 0, 'cant use kapprox here'
                print('kapprox to filt')
                TQUlms = ffs_specmat.TEB2TQUlms(TQUtype, self.lib_skyalm, TEBalms)
                for field, alm in zip(TQUtype, TQUlms):
                    self.lib_skyalm.almxfl(alm, self.cl_transf, inplace=True)
                    alm[:] = self.lib_skyalm.map2alm(self.lib_skyalm.alm2map(alm) * self.fi.get_det_magn())
                    _map = self._deg(alm)
                    self.apply_map(field, _map, inplace=True)
                    alm[:] = self._upg(_map)
                    self.lib_skyalm.almxfl(alm, self.cl_transf, inplace=True)
                return ffs_specmat.TQU2TEBlms(TQUtype, self.lib_skyalm, TQUlms)

    def degrade(self, shape, no_lensing=False, ellmax=None, ellmin=None, **kwargs):
        lib_almsky = self.lib_skyalm.degrade(shape, ellmax=ellmax, ellmin=ellmin)
        lib_almdat = self.lib_datalm.degrade(shape, ellmax=ellmax, ellmin=ellmin)
        assert self.nlevs['q'] == self.nlevs['u']
        if no_lensing:
            return ffs_ninv_filt(lib_almdat, lib_almsky, self.cls, self.cl_transf, self.nlevs['t'], self.nlevs['q'],
                                 verbose=self.verbose)
        else:
            fLD = self.f.degrade(shape, no_lensing)
            return ffs_ninv_filt_wl(lib_almdat, lib_almsky, self.cls, self.cl_transf, self.nlevs['t'], self.nlevs['q'],
                                    fLD, verbose=self.verbose, kapprox=self.kapprox)
