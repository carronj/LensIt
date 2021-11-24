import numpy as np
from lensit.qcinv.utils import ffs_converter



class template:
    """
    Generic template class for template projection,
    in the form of a modifiction of a noise matrix N:
    N^{-1}  -> N^{-1} - N^{-1}  T (T^t Ni T)^{-1} T^t N^{-1}
    where T_ia is the template matrix (template index a, 'pixel' index i)

    Each template T_a (e.g. a dust map, or a noisy mode)
    has the fct attributes:
    'dot': m -> T^t m = sum_i T_ia m_i for a map 'm' in pixel space. (output has dimensions the # of templates)
    'accum' m,c -> m + \sum_a T_ia c_a for template coefficients 'c', pixel map 'm'
    'apply': m,c -> m * (T c) for template coefficients 'c', pixel map 'm' (inplace)
    'apply_mode':  m,a -> mi * T_ia for  pixel map 'm', mode index 'a' (integer) (inplace)

    Atrtibutes include
      'nmodes': the number of templates decribed by the instance

    """
    def __init__(self):
        self.nmodes = 0

    def apply(self, map, coeffs):
        # map -> map*[coeffs combination of templates]
        assert 0,'subclass this'

    def apply_mode(self, map, mode):
        assert (mode < self.nmodes)
        assert (mode >= 0)

        tcoeffs = np.zeros(self.nmodes)
        tcoeffs[mode] = 1.0
        self.apply(map, tcoeffs)

    def accum(self, map, coeffs):
        assert (0)

    def dot(self, map):
        ret = []

        for i in range(0, self.nmodes):
            tmap = np.copy(map)
            self.apply_mode(tmap, i)
            ret.append(np.sum(tmap))

        return ret

    def _build_TtNiT(self,Ni):
        """ return the nmodes x nmodes matrix T^t Ni T """
        TtNiT = np.zeros((self.nmodes,self.nmodes),dtype = float)
        for a in range(self.nmodes):
            _Ni = np.copy(Ni)
            self.apply_mode(_Ni,a)
            TtNiT[:,a] = self.dot(_Ni)
            TtNiT[a,:] = TtNiT[:,a]
        return TtNiT


class template_map(template):
    def __init__(self, map):
        self.nmodes = 1
        self.map = map

    def apply(self, map, coeffs):
        assert (len(coeffs) == self.nmodes)

        map *= self.map * coeffs[0]

    def accum(self, map, coeffs):
        assert (len(coeffs) == self.nmodes)

        map += self.map * coeffs[0]

    def dot(self, map):
        return [(self.map * map).sum()]


class template_uptolmin(template):
    def __init__(self, ellmat, lmin):
        try:
            from lensit.ffs_covs.ell_mat import ffs_alm_pyFFTW as ffs_alm
        except ImportError:
            from lensit.ffs_covs.ell_mat import ffs_alm as ffs_alm
        self.lmin = lmin
        lib_alm = ffs_alm(ellmat, filt_func=lambda ell: (ell <= lmin))
        self.conv = ffs_converter(lib_alm)
        self.nmodes = self.conv.rlms_size
        self.lib_alm = lib_alm

    def _rlm2alm(self, rlm):
        return self.conv.rlms2datalms(1, rlm)[0]

    def _alm2rlm(self, alm):
        return self.conv.datalms2rlms(1, [alm])

    def apply(self, tmap, coeffs):  # V
        assert (len(coeffs) == self.nmodes)
        assert tmap.shape == self.lib_alm.shape
        tmap *= self.lib_alm.alm2map(self._rlm2alm(coeffs))

    def accum(self, tmap, coeffs):
        assert (len(coeffs) == self.nmodes)
        tmap += self.lib_alm.alm2map(self._rlm2alm(coeffs))

    def dot(self, tmap):  # V^t
        assert tmap.shape == self.lib_alm.shape
        return self._alm2rlm(self.lib_alm.map2alm(tmap)) * self.lib_alm.nbar()

class template_ellfilt(template):
    def __init__(self, ellmat,filt_func):
        try:
            from lensit.ffs_covs.ell_mat import ffs_alm_pyFFTW as ffs_alm
        except:
            from lensit.ffs_covs.ell_mat import ffs_alm as ffs_alm
        lib_alm = ffs_alm(ellmat, filt_func=lambda ell: filt_func(ell))
        self.conv = ffs_converter(lib_alm)
        self.nmodes = self.conv.rlms_size
        self.lib_alm = lib_alm

    def _rlm2alm(self, rlm):
        return self.conv.rlms2datalms(1, rlm)[0]

    def _alm2rlm(self, alm):
        return self.conv.datalms2rlms(1, [alm])

    def apply(self, tmap, coeffs):  # V
        assert (len(coeffs) == self.nmodes)
        assert tmap.shape == self.lib_alm.shape
        tmap *= self.lib_alm.alm2map(self._rlm2alm(coeffs))

    def accum(self, tmap, coeffs):
        assert (len(coeffs) == self.nmodes)
        tmap += self.lib_alm.alm2map(self._rlm2alm(coeffs))

    def dot(self, tmap):  # V^t
        assert tmap.shape == self.lib_alm.shape
        return self._alm2rlm(self.lib_alm.map2alm(tmap)) * self.lib_alm.nbar() # (norm. totally irrelevant in principle)


class template_pol:
    """
    Generic template class for template projection in polarization,
    in the form of a modifiction of a noise matrix N:
    N^{-1}  -> N^{-1} - N^{-1}  T (T^t Ni T)^{-1} T^t N^{-1}
    where T_Xia is the template matrix (template index a, 'pixel' and Stokes index Xi, X either Q and U)

    Each template T_a (e.g. a dust map, or a noisy mode)
    has the fct attributes:
    'dot': m -> T^t (Q,U) = sum_i T_iQa Q_i + T_iUa U_i for Q,U maps in pixel space. (output has dimensions the # of templates)
    'accum' (Q,U),c -> (Q + \sum_a T_Qib c_b,U + \sum_a T_Uib c_b) for template coefficients 'c', pixel map 'm'
    'apply': m,c -> m * (T c) for template coefficients 'c', pixel map 'm' (inplace)
    'apply_mode':  (Q,U),a -> (Qi* T_Qia,Ui*TUia) for  pixel map 'Q,U', mode index 'a' (integer) (inplace)

    Atrtibutes include
      'nmodes': the number of templates decribed by the instance
    """
    def __init__(self):
        self.nmodes = 0

    def apply(self, map, coeffs,X):
        # map -> map*[coeffs combination of templates]
        assert 0, 'subclass this'

    def apply_mode(self, Smap, mode,X):
        assert (mode < self.nmodes)
        assert (mode >= 0)

        tcoeffs = np.zeros(self.nmodes)
        tcoeffs[mode] = 1.0
        self.apply(Smap, tcoeffs,X)

    def accum(self, map, coeffs,X):
        assert 0, 'subclass this'

    def dot(self,qumap):
        assert 0, 'subclass this'

    def build_TtNiT(self,Ni):
        """ return the nmodes x nmodes matrix T^t Ni T """
        TtNiT = np.zeros((self.nmodes,self.nmodes),dtype = float)
        for a in range(self.nmodes):
            _Ni = np.copy(Ni)
            self.apply_mode(_Ni,a)
            TtNiT[:,a] = self.dot(_Ni)
            TtNiT[a,:] = TtNiT[:,a]
        return TtNiT


class template_Bfilt(template_pol):
    """ Here only B-modes are set to infinite noise, not E. bfilt_func(ell) returns true if ell is marginalized """
    def __init__(self, ellmat, bfilt_func):
        try:
            from lensit.ffs_covs.ell_mat import ffs_alm_pyFFTW as ffs_alm
        except ImportError:
            from lensit.ffs_covs.ell_mat import ffs_alm as ffs_alm
        lib_blm = ffs_alm(ellmat, filt_func=lambda ell: (bfilt_func(ell) & (ell > 0)))
        self.conv = ffs_converter(lib_blm)
        self.nmodes = self.conv.rlms_size
        self.lib_alm = lib_blm

    def _rlm2blm(self, rlm):
        return self.conv.rlms2datalms(1, rlm)[0]

    def _blm2rlm(self, blm):
        return self.conv.datalms2rlms(1, [blm])

    def apply(self, Smap, coeffs, X):  # RbQ  * Q or  RbU * U
        assert (len(coeffs) == self.nmodes)
        assert Smap.shape == self.lib_alm.shape
        assert X in ['Q','U']
        blm = self._rlm2blm(coeffs)
        elm = np.zeros_like(blm)
        Smap *= self.lib_alm.alm2map(self.lib_alm.EBlms2QUalms(np.array([elm,blm]))[0 if X == 'Q' else 1])

    def accum(self, Smap, coeffs, X):
        """(Q, U), c -> (Q + \sum_b T_Qib c_b, U + \sum_b T_Uib c_b) for template coefficients 'c', pixel map 'Q,U'"""
        assert (len(coeffs) == self.nmodes)
        assert Smap.shape == self.lib_alm.shape
        assert X in ['Q','U']
        blm = self._rlm2blm(coeffs)
        elm = np.zeros_like(blm)
        Smap += self.lib_alm.alm2map(self.lib_alm.EBlms2QUalms(np.array([elm,blm]))[0 if X == 'Q' else 1])

    def dot(self, qumap):  # T_bQ * Q + T_bU * U
        assert len(qumap) == 2 and qumap[0].shape == self.lib_alm.shape and qumap[1].shape == self.lib_alm.shape
        blm = self.lib_alm.QUlms2EBalms(np.array([self.lib_alm.map2alm(S) for S in qumap]))[1]
        return self._blm2rlm(blm) * self.lib_alm.nbar() # (norm. totally irrelevant in principle)

    def build_TtNiT(self, NiQQ_NiUU_NiQU):
        """ return the nmodes x nmodes matrix (T^t Ni T )_{bl bl'}'"""
        NiQQ, NiUU, NiQU = NiQQ_NiUU_NiQU
        TtNiT = np.zeros((self.nmodes,self.nmodes),dtype = float)
        for a in range(self.nmodes):
            _NiQ = np.copy(NiQQ) # Building Ni_{QX} R_bX
            self.apply_mode(_NiQ, a, 'Q')
            _NiU = np.copy(NiUU) # Building Ni_{UX} R_bX
            self.apply_mode(_NiU, a, 'U')
            if NiQU is not None :
                _NiQU = np.copy(NiQU)
                self.apply_mode(_NiQU, a, 'U')
                _NiQ += _NiQU
                _NiUQ = np.copy(NiQU)
                self.apply_mode(_NiUQ, a, 'Q')
                _NiU += _NiQU
            TtNiT[:,a] = self.dot([_NiQ,_NiU])
            TtNiT[a,:] = TtNiT[:,a]
        return TtNiT