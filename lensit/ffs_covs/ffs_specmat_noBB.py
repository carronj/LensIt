"""
Basically same as spectralmatrices_wtensors but set BB to zero.
"""
import numpy as np

_types = ['T', 'QU', 'TQU']


def _rootCMBcls(cmbCls):
    """
    Symmetric square root of
    (T E B) spectral matrix
    TT TE 0
    TE EE 0
    0 0   BB
    This assumes TB = EB == 0
    """
    s = np.sqrt(cmbCls['tt'] * cmbCls['ee'] - cmbCls['te'] ** 2)
    t = np.sqrt(cmbCls['tt'] + cmbCls['ee'] + 2 * s)
    ctt = np.zeros(len(cmbCls['tt']))
    cee = np.zeros(len(cmbCls['ee']))
    cte = np.zeros(len(cmbCls['te']))
    ii = np.where(t > 0.)
    ctt[ii] = (cmbCls['tt'][ii] + s[ii]) / t[ii]
    cee[ii] = (cmbCls['ee'][ii] + s[ii]) / t[ii]
    cte[ii] = cmbCls['te'][ii] / t[ii]
    return {'tt': ctt, 'ee': cee, 'te': cte}


def _clpinv(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl != 0.)] = 1. / cl[np.where(cl != 0.)]
    return ret


def TE2TQUlms(_type, lib_alm, TElms):
    """
    T = A T
    Q     E
    U     B
    where A is
        1   0   0
        0  cos -sin
        0  sin cos
    """
    assert _type in _types
    if _type == 'T':
        return np.array(TElms).copy()
    elif _type == 'QU':
        cos, sin = lib_alm.get_cossin_2iphi()
        return np.array([cos * TElms[0], sin * TElms[0]])
    elif _type == 'TQU':
        cos, sin = lib_alm.get_cossin_2iphi()
        return np.array([TElms[0], cos * TElms[1], sin * TElms[1]])


def TQU2TElms(_type, lib_alm, TQUlms):
    """
    T = A T
    Q     E
    U     B
    where A is
        1   0   0
        0  cos -sin
        0  sin cos
    This is the inverse relation
    """
    assert _type in _types
    assert len(TQUlms) == len(_type)
    if _type == 'T':
        return np.array(TQUlms).copy()
    elif _type == 'QU':
        cos, sin = lib_alm.get_cossin_2iphi()
        return np.array([cos * TQUlms[0] + sin * TQUlms[1]])
    elif _type == 'TQU':
        cos, sin = lib_alm.get_cossin_2iphi()
        return np.array([TQUlms[0], cos * TQUlms[1] + sin * TQUlms[2]])
    else:
        assert 0, (_type, _types)


def apply_rootTEmat(_type, lib_alm, cmb_cls, TElms):
    """
    Assumes TB = EB = BB = 0
    """
    assert (_type in _types)
    assert ('tb' not in cmb_cls.keys()) and ('eb' not in cmb_cls.keys()), cmb_cls.keys()
    if _type == 'T':
        return np.array([lib_alm.almxfl(TElms[0], np.sqrt(cmb_cls['tt']))])
    elif _type == 'QU':
        return np.array([lib_alm.almxfl(TElms[0], np.sqrt(cmb_cls['ee']))])
    elif _type == 'TQU':
        rootCls = _rootCMBcls(cmb_cls)
        fl = lambda id, _f: lib_alm.almxfl(TElms[id], rootCls[_f])
        return np.array([fl(0, 'tt') + fl(1, 'te'), fl(0, 'te') + fl(1, 'ee')])


def apply_TEmat(_type, lib_alm, cmb_cls, TElms):
    """
    Assumes TB = EB = 0
    """
    assert (_type in _types)
    assert ('tb' not in cmb_cls.keys()) and ('eb' not in cmb_cls.keys()), cmb_cls.keys()
    fl = lambda id, _f: lib_alm.almxfl(TElms[id], cmb_cls[_f])
    if _type == 'T':
        return np.array([fl(0, 'tt')])
    elif _type == 'QU':
        return np.array([fl(0, 'ee'), fl(1, 'bb')])
    elif _type == 'TQU':
        return np.array([fl(0, 'tt') + fl(1, 'te'), fl(0, 'te') + fl(1, 'ee')])
    else:
        assert 0, (_type, _types)


def apply_pinvTEmat(_type, lib_alm, cmb_cls, TElms):
    """
    Assumes TB = EB = 0.
    P^{-1} set to zero when there is no power in the variable (e.g. unl BB or ell = 0,1 in pol)
    """
    assert (_type in _types)
    assert ('tb' not in cmb_cls.keys()) and ('eb' not in cmb_cls.keys()), cmb_cls.keys()
    assert ('tb' not in cmb_cls.keys()) and ('eb' not in cmb_cls.keys()), cmb_cls.keys()
    fl = lambda id, cl: lib_alm.almxfl(TElms[id], cl)
    if _type == 'T':
        return np.array([fl(0, _clpinv(cmb_cls['tt']))])
    elif _type == 'QU':
        return np.array([fl(0, _clpinv(cmb_cls['ee']))])
    elif _type == 'TQU':
        cli = get_pinvTEcls(_type, cmb_cls)
        return np.array([fl(0, cli['tt']) + fl(1, cli['te']), fl(0, cli['te']) + fl(1, cli['ee'])])
    else:
        assert 0


def get_pinvTEcls(_type, cmb_cls):
    if _type == 'T':
        return {'tt': _clpinv(cmb_cls['tt'])}
    elif _type == 'QU':
        return {'ee': _clpinv(cmb_cls['ee'])}
    elif _type == 'TQU':
        ret = {}
        # FIXME rewrite this
        deti = _clpinv(cmb_cls['tt'] * cmb_cls['ee'] - cmb_cls['te'] ** 2)
        ret['tt'] = np.where(deti > 0, cmb_cls['ee'] * deti, _clpinv(cmb_cls['tt']))
        ret['te'] = np.where(deti > 0, -cmb_cls['te'] * deti, np.zeros(len(cmb_cls['te'])))
        ret['ee'] = np.where(deti > 0, cmb_cls['tt'] * deti, _clpinv(cmb_cls['ee']))
        return ret
    else:
        assert 0
