"""
=========================================

B^t Cov^-1 d = B^t Ni data - (B^t Ni B) MLIK(data)
             = B^t Ni (data - B MLIK(data))

For non-sing. modes in TEB space this is C^{-1} MLIK(data), but can't use pseudo inverse to get the singular modes.

This should work with and without lensing, in which case contains beam and deflection.
=========================================
"""

import numpy  as np

from lensit.qcinv import dense
from lensit.ffs_covs import ffs_specmat as SM

_type = 'T'  # in ['T','QU','TQU']
_prefix = 'cinv'


def TEBlen(_type):
    return len(_type)

def TEBfields(_type):
    if _type == 'T': return 't'
    if _type == 'QU': return ['e','b']
    if _type == 'TQU': return ['t','e', 'b']
    assert 0,_type


def TQUlen(_type):
    return len(_type)

def filtTEBlms(TEBlms,cov):
    # The inverse matrix acts on the non-zero modes space
    for i,e in enumerate(TEBfields(_type)):
        cov.lib_skyalm.almxfl(TEBlms[i],cov.cls[e + e] > 0,inplace=True)
    return TEBlms

# =====================
def calc_prep(maps, cov, *args, **kwargs):
    """
    Pre operation for primordial CMB modes.
    R^t Ni data projected onto T E alms
    """
    print("This is calc prep for %s W. Filtering" % _type, _prefix)
    return filtTEBlms(cov.apply_Rts(_type,cov.apply_maps(_type, maps, inplace=False)),cov)

def apply_fini_BINV(soltn, cov, maps, **kwargs):
    """
    Output TEB skyalms shaped
    B^t Cov^-1 d = B^t Ni data - (B^t Ni B) MLIK(data)
             = B^t Ni (data - B MLIK(data))
    """
    assert len(soltn) == TEBlen(_type)
    assert len(maps) == TQUlen(_type)
    return cov.apply_Rts(_type,cov.apply_maps(_type,maps - cov.apply_Rs(_type,soltn), inplace=False))


def apply_fini_MLIK(soltn, cov, maps, **kwargs):
    """
    Post operation for max. likelihood primordial CMB modes.
     (P^-1 + b G M^t Ni M Gt bl)^{-1}  bl G M^t Ni
     output TEB lms shaped
    """
    return soltn


def MLIK2BINV(soltn, cov, maps, **kwargs):
    """
    Output TEB skyalms shaped
    R^t Cov^-1 d = R^t Ni data - (R^t Ni R) MLIK(data)
             = R^t Ni (data - R MLIK(data))
    """
    assert len(soltn) == TEBlen(_type)
    assert len(maps) == TQUlen(_type)
    return cov.apply_Rts(_type,cov.apply_maps(_type,maps - cov.apply_Rs(_type,soltn), inplace=False))


def soltn2TQUMlik(soltn, cov):
    assert len(soltn) == TEBlen(_type)
    return SM.TEB2TQUlms(_type, cov.lib_skyalm, soltn)


# =====================
class dot_op:
    def __init__(self, lib_skyalm):
        self.lib_skyalm = lib_skyalm
        self.Nell = lib_skyalm.get_Nell()

    def __call__(self, alms1, alms2, **kwargs):
        ret = 0.
        for alm1, alm2 in zip(alms1, alms2):
            ret += np.sum(self.lib_skyalm.alm2cl(alm1,alm2=alm2) * self.Nell)
        return ret
        #return np.sum(alms1.real * alms2.real + alms1.imag * alms2.imag)

class fwd_op:  # (P^-1 + R^t Ni R)^{-1} (skyalms)
    def __init__(self, cov, *args):
        self.cov = cov
        self.lib_alm = self.cov.lib_skyalm

    def __call__(self, TEBlms):
        return filtTEBlms( SM.apply_pinvTEBmat(_type, self.lib_alm, self.cov.cls, TEBlms) +  self.cov.apply_alms(_type, TEBlms,inplace=False),self.cov)


# =====================
class pre_op_diag:
    # (1/P + bl G Ni Gt bl)^-1
    def __init__(self, cov, *args):
        inv_cls = SM.get_pinvTEBcls(_type, cov.cls)
        if _type == 'T':
            NTi = cov.iNoiseCl(_type[0])
            inv_cls['tt'] += cov.cl_transf ** 2 * NTi * (inv_cls['tt'] > 0)
        elif _type == 'QU':
            NPi = 0.5 * (cov.iNoiseCl(_type[0]) + cov.iNoiseCl(_type[1]))
            inv_cls['ee'] += cov.cl_transf ** 2 * NPi * (inv_cls['ee'] > 0)
            inv_cls['bb'] += cov.cl_transf ** 2 * NPi * (inv_cls['bb'] > 0)
        elif _type == 'TQU':
            NPi = 0.5 * (cov.iNoiseCl(_type[1]) + cov.iNoiseCl(_type[2]))
            NTi = cov.iNoiseCl(_type[0])
            inv_cls['tt'] += cov.cl_transf ** 2 * NTi * (inv_cls['tt'] > 0)
            inv_cls['ee'] += cov.cl_transf ** 2 * NPi * (inv_cls['ee'] > 0)
            inv_cls['bb'] += cov.cl_transf ** 2 * NPi * (inv_cls['bb'] > 0)
        else:
            assert 0, (_type)
        self.inv_cls = inv_cls
        self.cov = cov

    def __call__(self, TEBlms):
        assert TEBlms.shape == (TEBlen(_type, ), self.cov.lib_skyalm.alm_size)
        return SM.apply_pinvTEBmat(_type, self.cov.lib_skyalm, self.inv_cls, TEBlms)


def pre_op_dense(cov, no_lensing, cache_fname=None):
    return dense.pre_op_dense(cov, fwd_op(cov, no_lensing), TEBlen(_type), cache_fname=cache_fname)
