"""
(P^-1 + B^t N^{-1} B)^{-1}
There are only dat shaped ffts to perform
=========================================

B^t Cov^-1 d = B^t Ni data - (B^t Ni B) MLIK(data)
             = B^t Ni (data - B MLIK(data))

For non-sing. modes in TEB space this is C^{-1} MLIK(data), but can't use pseudo inverse to get the singular modes.
This should work with and without lensing, in which case B contains beam and deflection.
=========================================
"""

import numpy  as np

import dense
from lensit.ffs_covs import ffs_specmat as SMwBB
from lensit.ffs_covs import ffs_specmat_noBB as SM

_type = 'T'  # in ['T','QU','TQU']
_prefix = 'cinv_noBB'


def TEBlen(_type):
    return {'T': 1, 'QU': 1, 'TQU': 2}[_type]


def TQUlen(_type):
    return len(_type)


# =====================
def calc_prep(maps, cov, *args, **kwargs):
    """
    Pre operation for primordial CMB modes.
    B Ni data projected onto T E alms
    """
    print "This is calc prep for %s W. Filtering" % _type, _prefix
    _TQUalms = np.empty((TQUlen(_type), cov.lib_skyalm.alm_size), dtype=complex)
    for i, f in enumerate(_type):
        _map = cov.apply_map(f, maps[i], inplace=False)
        _TQUalms[i] = cov.apply_Rt(f, _map)
    return SM.TQU2TElms(_type, cov.lib_skyalm, _TQUalms)


def apply_fini_BINV(soltn, cov, maps, **kwargs):
    """
    Output TEB skyalms shaped
    B^t Cov^-1 d = B^t Ni data - (B^t Ni B) MLIK(data)
             = B^t Ni (data - B MLIK(data))
    """
    TQUmlik = SM.TE2TQUlms(_type, cov.lib_skyalm, apply_fini_MLIK(soltn, cov, maps, **kwargs))
    for i, f in enumerate(_type):
        _map = maps[i] - cov.apply_R(f, TQUmlik[i])
        cov.apply_map(f, _map, inplace=True)
        TQUmlik[i] = cov.apply_Rt(f, _map)
    del _map
    return SM.TQU2TElms(_type, cov.lib_skyalm, TQUmlik)


def apply_fini_MLIK(soltn, cov, maps, **kwargs):
    """
    Post operation for max. likelihood primordial CMB modes.
     (P^-1 + B^t Ni B)^{-1}  B^t Ni
     output TEB lms shaped
    """
    return soltn


def MLIK2BINV(soltn, cov, maps, **kwargs):
    """
    Output TEB skyalms shaped
    B^t Cov^-1 d = B^t Ni data - (B^t Ni B) MLIK(data)
             = B^t Ni (data - B MLIK(data))
    """
    assert len(soltn) == TEBlen(_type)
    assert len(maps) == TQUlen(_type), (maps.shape, _type)
    TQUmlik = SM.TE2TQUlms(_type, cov.lib_skyalm, soltn)
    for i, f in enumerate(_type):
        _map = maps[i] - cov.apply_R(f, TQUmlik[i])
        cov.apply_map(f, _map, inplace=True)
        TQUmlik[i] = cov.apply_Rt(f, _map)
    del _map
    return SMwBB.TQU2TEBlms(_type, cov.lib_skyalm, TQUmlik)


def soltn2TQUMlik(soltn, cov):
    assert len(soltn) == TEBlen(_type), (len(soltn), TEBlen(_type))
    return SM.TE2TQUlms(_type, cov.lib_skyalm, soltn)


# =====================

class dot_op():
    def __init__(self):
        pass

    def __call__(self, alms1, alms2, **kwargs):
        return np.sum(alms1.real * alms2.real + alms1.imag * alms2.imag)


class fwd_op():  # (P^-1 + B^t Ni B)^{-1} (skyalms)
    def __init__(self, cov, *args):
        self.cov = cov
        self.lib_alm = self.cov.lib_skyalm

    def __call__(self, TElms):
        # print "This is fwd_op w. no_lensing %s _type %s"%(self.no_lensing,_type)
        TQUlms = SM.TE2TQUlms(_type, self.cov.lib_skyalm, TElms)
        for i, f in enumerate(_type): self.cov.apply_alm(f, TQUlms[i], inplace=True)
        return SM.apply_pinvTEmat(_type, self.lib_alm, self.cov.cls, TElms) + SM.TQU2TElms(_type, self.lib_alm, TQUlms)


# =====================
class pre_op_diag():
    # (1/P + bl G Ni Gt bl)^-1
    def __init__(self, cov, *args):
        inv_cls = SM.get_pinvTEcls(_type, cov.cls)
        if _type == 'T':
            NTi = cov.iNoiseCl(_type[0])
            inv_cls['tt'] += cov.cl_transf ** 2 * NTi
        elif _type == 'QU':
            NPi = 0.5 * (cov.iNoiseCl(_type[0]) + cov.iNoiseCl(_type[1]))
            inv_cls['ee'] += cov.cl_transf ** 2 * NPi
        elif _type == 'TQU':
            NPi = 0.5 * (cov.iNoiseCl(_type[1]) + cov.iNoiseCl(_type[2]))
            NTi = cov.iNoiseCl(_type[0])
            inv_cls['tt'] += cov.cl_transf ** 2 * NTi
            inv_cls['ee'] += cov.cl_transf ** 2 * NPi
        else:
            assert 0, (_type)
        self.inv_cls = inv_cls
        self.cov = cov

    def __call__(self, TEBlms):
        assert TEBlms.shape == (TEBlen(_type, ), self.cov.lib_skyalm.alm_size)
        return SM.apply_pinvTEmat(_type, self.cov.lib_skyalm, self.inv_cls, TEBlms)


def pre_op_dense(cov, no_lensing, cache_fname=None):
    return dense.pre_op_dense(cov, fwd_op(cov, no_lensing), TEBlen(_type), cache_fname=cache_fname)
