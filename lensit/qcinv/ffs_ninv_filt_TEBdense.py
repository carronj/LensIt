import numpy as np
import lensit as fs
from lensit.ffs_covs import ffs_specmat

from lensit.qcinv import template_removal

verbose = True

load_map = lambda _map: np.load(_map) if type(_map) is str else _map

def cl_inv(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

class ffs_ninv_filt(object):
    def __init__(self, lib_datalm, lib_skyalm, len_cls, cltransf_dict, ninv_rad,
                 marge_maps={}, marge_uptolmin={}, cls_noise={}, verbose=False):
        """
        cl_transf dict is {'TT': T -> T tranffer function, 'TE' : T -> E transfer, etc.}
        Zero if not specified.

        Noise matrices still TQU diagonal in this version

        ninv_rad is the inverse variance map in 1 / rad ** 2, not the pixel variance maps.
        This is the inverse pixel variance map / volume of cell.
        """
        self.ninv_rad = ninv_rad
        self.lib_datalm = lib_datalm
        self.lib_skyalm = lib_skyalm
        self.cltransf_dict = cltransf_dict
        self.cls = {}
        for _k, _cls in len_cls.iteritems():
            self.cls[_k] = _cls[:lib_skyalm.ellmax + 1]
        self.npix = np.prod(lib_datalm.shape)
        # Build mean noise level :
        self._iNoiseCl = {}
        for _k, _ni in ninv_rad.iteritems():
            if _k not in cls_noise.keys():
                _noiseCl = np.mean(1. / _ni[np.where(_ni > 0)])
                self._iNoiseCl[_k] = 1. / _noiseCl * np.ones(lib_skyalm.ellmax + 1, dtype=float)
                if verbose:
                    print "ninv_filt::Nlev (%s) in uKamin  %.3f" % (
                        _k.upper(), np.sqrt(1. / self._iNoiseCl[_k][0]) / np.pi * 180. * 60.)
                    print "      ellsky range (%s - %s)" % (lib_skyalm.ellmin, lib_skyalm.ellmax)
            else:
                # This does not appear to generically improve
                self._iNoiseCl[_k] = cl_inv(cls_noise[_k][:lib_skyalm.ellmax + 1])

        templates = {_k: [] for _k in self.ninv_rad.keys()}
        for _f in marge_maps.iterkeys():
            if _f not in templates.keys(): templates[_f] = []
            for tmap in [load_map(m) for m in marge_maps[_f]]:
                assert (self.npix == len(tmap))
                templates[_f].append(template_removal.template_map(tmap))

        for _f in marge_uptolmin.iterkeys():
            if marge_uptolmin[_f] >= 0:
                if _f not in templates.keys(): templates[_f] = []
                templates[_f].append(template_removal.template_uptolmin(lib_datalm.ell_mat, marge_uptolmin[_f]))

        assert np.all([_f in ninv_rad.keys() for _f in templates.keys()]), (ninv_rad.keys(), templates.keys())
        for _f, _templates in templates.iteritems():
            self.Pt_Nn1_P_inv = {}
            if (len(_templates) != 0):
                nmodes = np.sum([t.nmodes for t in _templates])
                modes_idx_t = np.concatenate(([t.nmodes * [int(im)] for im, t in enumerate(_templates)]))
                modes_idx_i = np.concatenate(([range(0, t.nmodes) for t in _templates]))
                print "   Building %s - %s template projection matrix" % (nmodes, nmodes)
                Pt_Nn1_P = np.zeros((nmodes, nmodes))
                for ir in range(0, nmodes):
                    if np.mod(ir, int(0.1 * nmodes)) == 0: print ("   filling TNiT: %4.1f" % (100. * ir / nmodes)), "%"
                    tmap = np.copy(ninv_rad[_f])
                    _templates[modes_idx_t[ir]].apply_mode(tmap, int(modes_idx_i[ir]))
                    ic = 0
                    for tc in _templates[0:modes_idx_t[ir] + 1]:
                        Pt_Nn1_P[ir, ic:(ic + tc.nmodes)] = tc.dot(tmap)
                        Pt_Nn1_P[ic:(ic + tc.nmodes), ir] = Pt_Nn1_P[ir, ic:(ic + tc.nmodes)]
                        ic += tc.nmodes
                eigv, eigw = np.linalg.eigh(Pt_Nn1_P)
                eigv_inv = 1.0 / eigv
                self.Pt_Nn1_P_inv[_f] = np.dot(np.dot(eigw, np.diag(eigv_inv)), np.transpose(eigw))
        self.templates = templates
        self.marge_uptolmin = marge_uptolmin
        self.cls_noise = cls_noise
        self.marge_maps = marge_maps

        if verbose:
            for k,cl in self.cltransf_dict.iteritems():
                print "I see transf %s -> %s"%(k[0],k[1])


    def hashdict(self):
        return {}

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
        return self.cltransf_dict[lab.upper()]

    def get_nTpix(self):
        return self._get_rmspixnoise('T')

    def get_nQpix(self):
        return self._get_rmspixnoise('Q')

    def get_nUpix(self):
        return self._get_rmspixnoise('U')

    def apply_Rs(self, TQUtype,TEBlms):
        """
        Apply transfer function, T E B skyalm to T Q U map.
        """
        assert TQUtype in ['T','QU','TQU']
        TEBtype = {'T':'T','QU':'EB','TQU':'TEB'}[TQUtype]
        assert TEBlms.shape == (len(TEBtype),self.lib_skyalm.alm_size), (TEBlms.shape,TEBtype,self.lib_skyalm.alm_size)
        ret = np.zeros((len(TEBtype),self.lib_datalm.alm_size),dtype = complex)
        for i,f in enumerate(TEBtype):
            for j,g in enumerate(TEBtype):
                cl_t = self.cltransf_dict.get(g + f,None)
                if cl_t is not None : ret[i] += self._deg(self.lib_skyalm.almxfl(TEBlms[j],cl_t))
        return np.array([self.lib_datalm.alm2map(alm) for alm in ffs_specmat.TEB2TQUlms(TQUtype,self.lib_datalm,ret)])

    def apply_Rts(self, TQUtype,_maps):
        """
        Apply tranposed transfer function, from T Q U real space to T E B skyalm.
        B^t X
        """
        assert TQUtype in ['T','QU','TQU']
        assert _maps.shape == (len(TQUtype),self.lib_datalm.shape[0],self.lib_datalm.shape[1]), (self.npix, TQUtype,_maps.shape)
        TEBtype = {'T':'T','QU':'EB','TQU':'TEB'}[TQUtype]

        alms = ffs_specmat.TQU2TEBlms(TQUtype,self.lib_skyalm,np.array([self._upg(self.lib_datalm.map2alm(_m)) for _m in _maps]))
        TEBret = np.zeros_like(alms)
        for i, f in enumerate(TEBtype):
            for j, g in enumerate(TEBtype):
                cl_t = self.cltransf_dict.get(f + g, None) # transposed
                if cl_t is not None: TEBret[i] += self.lib_skyalm.almxfl(alms[j], cl_t)
        return TEBret

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

    def apply_maps(self, TQUtype, _maps, inplace=True):
        """
        Applies ninv to real space T, Q, or U map, in radians units.
        """
        assert _maps.shape == (len(TQUtype),self.lib_datalm.shape[0],self.lib_datalm.shape[1]), (self.npix, TQUtype,_maps.shape)
        assert TQUtype in ['T','QU','TQU']
        if inplace:
            for i, _f in enumerate(TQUtype.lower()):
                _maps[i] *= self.ninv_rad[_f]
                if (len(self.templates[_f]) != 0):
                    coeffs = np.concatenate(([t.dot(_maps[i]) for t in self.templates[_f]]))
                    coeffs = np.dot(self.Pt_Nn1_P_inv[_f], coeffs)
                    pmodes = np.zeros(self.ninv_rad[_f].shape)
                    im = 0
                    for t in self.templates[_f]:
                        t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                        im += t.nmodesx
                    pmodes *= self.ninv_rad[_f]
                    _maps[i] -= pmodes
            return
        else:
            nmaps = np.zeros_like(_maps)
            for i, _f in enumerate(TQUtype.lower()):
                nmaps[i] = _maps[i] * self.ninv_rad[_f]
                if (len(self.templates[_f]) != 0):
                    coeffs = np.concatenate(([t.dot(nmaps[i]) for t in self.templates[_f]]))
                    coeffs = np.dot(self.Pt_Nn1_P_inv[_f], coeffs)
                    pmodes = np.zeros(self.ninv_rad[_f].shape)
                    im = 0
                    for t in self.templates[_f]:
                        t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                        im += t.nmodes
                    pmodes *= self.ninv_rad[_f]
                    nmaps[i] -= pmodes
            return nmaps


    def iNoiseCl(self, field):
        return self._iNoiseCl[field.lower()]

    def degrade(self, shape, ellmax=None, ellmin=None, **kwargs):
        lib_almsky = self.lib_skyalm.degrade(shape, ellmax=ellmax, ellmin=ellmin)
        lib_almdat = self.lib_datalm.degrade(shape, ellmax=ellmax, ellmin=ellmin)
        ninvLD = {}
        for _k, _ni in self.ninv_rad.iteritems():
            ninvLD[_k] = fs.misc.rfft2_utils.degrade_mask(_ni, shape)
        # print "NO TEMPLATES in degrading"
        return ffs_ninv_filt(lib_almdat, lib_almsky, self.cls, self.cltransf_dict, ninvLD,
                             marge_uptolmin=self.marge_uptolmin, cls_noise=self.cls_noise)

    def turn2isofilt(self):
        """
        Returns an isotropic (no mask, homog. noise) filter built from the average noise levels.
        """
        # lib_datalm, lib_skyalm, len_cls, cl_transf, ninv_rad,
        # marge_maps = {}, marge_uptolmin = {}, cls_noise = {}
        ninv_rad = {}
        for _k, ninv in self.ninv_rad.iteritems():
            ninv_rad[_k] = np.ones(self.lib_datalm.shape, dtype=float) * (
                1. / (self.Nlev_uKamin(_k) / 60 / 180. * np.pi) ** 2)
        return ffs_ninv_filt(self.lib_datalm, self.lib_skyalm, self.cls, self.cltransf_dict, ninv_rad,
                             marge_maps=self.marge_maps, marge_uptolmin=self.marge_uptolmin)
