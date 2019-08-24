import numpy as np

typs = ['T', 'QU', 'TQU']


def _rootCMBcls(cmbCls):
    """
    Symmetric square root of
    (T E B) spectral matrix
    TT TE 0
    TE EE 0
    0 0   BB
    This assumes TB = EB == 0
    """
    assert ('tb' not in cmbCls.keys()) and ('eb' not in cmbCls.keys()), cmbCls.keys()
    s = np.sqrt(cmbCls['tt'] * cmbCls['ee'] - cmbCls['te'] ** 2)
    t = np.sqrt(cmbCls['tt'] + cmbCls['ee'] + 2 * s)
    ctt = np.zeros(len(cmbCls['tt']))
    cee = np.zeros(len(cmbCls['ee']))
    cte = np.zeros(len(cmbCls['te']))
    ii = np.where(t > 0.)
    ctt[ii] = (cmbCls['tt'][ii] + s[ii]) / t[ii]
    cee[ii] = (cmbCls['ee'][ii] + s[ii]) / t[ii]
    cte[ii] = cmbCls['te'][ii] / t[ii]
    return {'bb': np.sqrt(cmbCls['bb']), 'tt': ctt, 'ee': cee, 'te': cte}


def _clpinv(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0.)] = 1. / cl[np.where(cl > 0.)]
    return ret


def get_unlPmat_ij(typ, lib_alm, cls_cmb, i, j):
    """
    Get the spectral matrix P(k) w/wo transfer fct, w/wo noise, or its inverse.
    :param typ: 'T','QU' or TQU'
    :param cls_cmb : dictionary with the cls array.
    :return:
    This assumes C_TB = 0, C_EB = 0.
    """
    if i < j: return get_unlPmat_ij(typ, lib_alm, cls_cmb, j, i)
    assert (not 'tb' in cls_cmb.keys() and not 'eb' in cls_cmb.keys())
    ell = lambda: lib_alm.reduced_ellmat()
    if typ == 'T':
        assert i == 0 and j == 0, (i, j, typ, typs)
        return cls_cmb['tt'][ell()]
    elif typ == 'QU':
        cos, sin = lib_alm.get_cossin_2iphi()  # in mmap mode 'r' in principle.
        if i == 0 and j == 0:
            return cls_cmb['ee'][ell()] * cos ** 2 + cls_cmb['bb'][ell()] * sin ** 2
        elif i == 1 and j == 1:
            return cls_cmb['ee'][ell()] * sin ** 2 + cls_cmb['bb'][ell()] * cos ** 2
        elif i == 1 and j == 0:
            return ((cls_cmb['ee'] - cls_cmb['bb'])[ell()]) * cos * sin
        else:
            assert 0, (i, j, typ, typs)
    elif typ == 'TQU':
        if i == 0 and j == 0: return cls_cmb['tt'][ell()]
        cos, sin = lib_alm.get_cossin_2iphi()  # in mmap mode 'r' in principle.
        if i == 1 and j == 1:
            return cls_cmb['ee'][ell()] * cos ** 2 + cls_cmb['bb'][ell()] * sin ** 2
        elif i == 2 and j == 2:
            return cls_cmb['ee'][ell()] * sin ** 2 + cls_cmb['bb'][ell()] * cos ** 2
        elif i == 2 and j == 1:
            return ((cls_cmb['ee'] - cls_cmb['bb'])[ell()]) * cos * sin
        elif i == 1 and j == 0:
            return cls_cmb['te'][ell()] * cos
        elif i == 2 and j == 0:
            return cls_cmb['te'][ell()] * sin
        else:
            assert 0, (i, j, typ, typs)
    else:
        assert 0, (typ, typs)


def get_rootunlPmat_ij(typ, lib_alm, cls_cmb, i, j):
    if i < j: return get_rootunlPmat_ij(typ, lib_alm, cls_cmb, j, i)
    if typ == 'T':
        assert i == 0 and j == 0, (i, j, typ, typs)
        return np.sqrt(cls_cmb['tt'])[lib_alm.reduced_ellmat()]
    elif typ == 'QU':
        cos, sin = lib_alm.get_cossin_2iphi()  # in mmap mode 'r' in principle.
        ell = lambda: lib_alm.reduced_ellmat()
        ree = np.sqrt(cls_cmb['ee'])
        rbb = np.sqrt(cls_cmb['bb'])
        if i == 0 and j == 0:
            return ree[ell()] * cos ** 2 + rbb[ell()] * sin ** 2
        elif i == 1 and j == 1:
            return ree[ell()] * sin ** 2 + rbb[ell()] * cos ** 2
        elif i == 1 and j == 0:
            return ((ree - rbb)[ell()]) * cos * sin
        else:
            assert 0, (i, j, typ, typs)
    elif typ == 'TQU':
        return get_unlPmat_ij(typ, lib_alm, _rootCMBcls(cls_cmb), i, j)
    else:
        assert 0


def get_unlrotPmat_ij(typ, lib_alm, cls_cmb, i, j):
    # Rotated polarization mat :0  0 0
    #                           0  0 1   * Pmat
    #                           0 -1 0
    if ('T' in typ) and ((i == 0) or (j == 0)):
        return np.zeros(lib_alm.alm_size, dtype=complex)
    elif typ == 'QU':
        _i = {0: 1, 1: 0}[i]
        _j = j
        sgn = 1 if _i == 1 else -1
        return sgn * get_unlPmat_ij(typ, lib_alm, cls_cmb, _i, _j)
    elif typ == 'TQU':
        assert i > 0 and j > 0
        _i = {0: 0, 1: 2, 2: 1}[i]
        _j = j
        sgn = 1 if _i == 2 else -1
        return sgn * get_unlPmat_ij(typ, lib_alm, cls_cmb, _i, _j)
    else:
        assert 0, (i, j, typ)


def get_datPmat_ij(typ, lib_alm, cls_cmb, cl_transf, cls_noise, i, j):
    """
    Get the spectral matrix P(k) w/wo transfer fct, w/wo noise, or its inverse.
    :param typ: 'T','QU' or TQU'
    :param cls_cmb : dictionary with the cls array.
    :return:
    This assumes C_TB = 0, C_EB = 0.
    """
    if i < j: return get_datPmat_ij(typ, lib_alm, cls_cmb, cl_transf, cls_noise, j, i)

    sl = slice(0, lib_alm.ellmax + 1)
    if typ == 'T':
        assert i == 0 and j == 0, (i, j, typ, typs)
        return (cls_cmb['tt'][sl] * cl_transf[sl] ** 2 + cls_noise['t'][sl])[lib_alm.reduced_ellmat()]
    elif typ == 'QU':
        ell = lambda: lib_alm.reduced_ellmat()
        cos, sin = lib_alm.get_cossin_2iphi()  # in mmap mode 'r' in principle.
        ee = cls_cmb['ee'][sl] * cl_transf[sl] ** 2
        bb = cls_cmb['bb'][sl] * cl_transf[sl] ** 2

        if i == 0 and j == 0:
            return ee[ell()] * cos ** 2 + bb[ell()] * sin ** 2 + cls_noise['q'][ell()]
        elif i == 1 and j == 1:
            return ee[ell()] * sin ** 2 + bb[ell()] * cos ** 2 + cls_noise['u'][ell()]
        elif i == 1 and j == 0:
            return (ee - bb)[ell()] * cos * sin
        else:
            assert 0, (i, j, typ, typs)

    elif typ == 'TQU':
        ell = lambda: lib_alm.reduced_ellmat()
        tt = cls_cmb['tt'][sl] * cl_transf[sl] ** 2
        ee = cls_cmb['ee'][sl] * cl_transf[sl] ** 2
        bb = cls_cmb['bb'][sl] * cl_transf[sl] ** 2
        te = cls_cmb['te'][sl] * cl_transf[sl] ** 2
        if i == 0 and j == 0: return (tt + cls_noise['t'][sl])[ell()]
        cos, sin = lib_alm.get_cossin_2iphi()  # in mmap mode 'r' in principle.

        if i == 1 and j == 1:
            return ee[ell()] * cos ** 2 + bb[ell()] * sin ** 2 + cls_noise['q'][ell()]
        elif i == 2 and j == 2:
            return ee[ell()] * sin ** 2 + bb[ell()] * cos ** 2 + cls_noise['u'][ell()]
        elif i == 2 and j == 1:
            return (ee - bb)[ell()] * cos * sin
        elif i == 1 and j == 0:
            return te[ell()] * cos
        elif i == 2 and j == 0:
            return te[ell()] * sin
        else:
            assert 0, (i, j, typ, typs)
    else:
        assert 0, (typ, typs)


def get_Pmat(typ, lib_alm, cls_cmb,
             cl_transf=None, cls_noise=None, derivative=None, square_root=False, inverse=False):
    """
    Get the spectral matrix P(k) w/wo transfer fct, w/wo noise, or its inverse.
    :param typ: 'T','QU' or TQU'
    :param w_transf:
    :param w_noise:
    :param inverse:
    :return:
    This assumes C_TB = 0, C_EB = 0.
    """
    assert typ.upper() in typs, typ
    assert derivative in [None, 0, 1], derivative
    if derivative is not None: assert not inverse, 'This is really suspicious'
    if derivative is not None: assert not square_root, 'This is really suspicious'

    w_noise = cls_noise is not None
    w_transf = cl_transf is not None

    ret = np.zeros((lib_alm.alm_size, len(typ), len(typ)), dtype=float if derivative is None else complex)
    ell = lambda: lib_alm.reduced_ellmat()
    sl = slice(0, lib_alm.ellmax + 1)
    if typ == 'TQU':
        tt = cls_cmb['tt'].copy()[sl]
        ee = cls_cmb['ee'].copy()[sl]
        bb = cls_cmb['bb'].copy()[sl]
        te = cls_cmb['te'].copy()[sl]
        if w_transf:
            tt *= cl_transf[sl] ** 2
            ee *= cl_transf[sl] ** 2
            bb *= cl_transf[sl] ** 2
            te *= cl_transf[sl] ** 2

        cos, sin = lib_alm.get_cossin_2iphi()  # in mmap mode 'r' in principle.
        ret[:, 0, 0] = tt[ell()]
        ret[:, 0, 1] = te[ell()] * cos
        ret[:, 0, 2] = te[ell()] * sin
        ret[:, 1, 0] = ret[:, 0, 1]
        ret[:, 2, 0] = ret[:, 0, 2]
        ret[:, 1, 1] = ee[ell()] * cos ** 2 + bb[ell()] * sin ** 2
        ret[:, 2, 2] = ee[ell()] * sin ** 2 + bb[ell()] * cos ** 2
        ret[:, 1, 2] = (ee - bb)[ell()] * cos * sin
        ret[:, 2, 1] = ret[:, 1, 2]
        if w_noise:
            ret[:, 0, 0] += cls_noise['t'][ell()]
            ret[:, 1, 1] += cls_noise['q'][ell()]
            ret[:, 2, 2] += cls_noise['u'][ell()]

    elif typ == 'QU':
        ee = cls_cmb['ee'].copy()[sl]
        bb = cls_cmb['bb'].copy()[sl]
        if w_transf:
            ee *= cl_transf[sl] ** 2
            bb *= cl_transf[sl] ** 2

        cos, sin = lib_alm.get_cossin_2iphi()  # in mmap mode 'r' in principle.
        ret[:, 0, 0] = ee[ell()] * cos ** 2 + bb[ell()] * sin ** 2
        ret[:, 1, 1] = ee[ell()] * sin ** 2 + bb[ell()] * cos ** 2
        ret[:, 0, 1] = cos * sin * (ee - bb)[ell()]
        ret[:, 1, 0] = ret[:, 0, 1]
        if w_noise:
            ret[:, 0, 0] += cls_noise['q'][ell()]
            ret[:, 1, 1] += cls_noise['u'][ell()]

    elif typ == 'T':
        tt = cls_cmb['tt'].copy()[sl]
        if w_transf: tt *= cl_transf[sl] ** 2
        if w_noise: tt += cls_noise['t'][sl]
        ret[:, 0, 0] = tt[ell()]

    if square_root:
        # Square rooting. We avoid cholesky as this fails for singular spectral matrices
        for _ell in range(ret.shape[0]):
            u, t, v = np.linalg.svd(ret[_ell, :, :])
            # assert np.all(t >= 0.), t  # Matrix not positive semidefinite
            ret[_ell, :, :] = np.dot(u, np.dot(np.diag(np.sqrt(t)), v))
            # FIXME : super slow for very large lmax. could use broadcoasting rules, but then memory ?
            # w, v = np.linalg.eigh(ret[_ell, :, :])
            # if not np.all(w >= 0.) :
            #    print " !!! I find negative eigenvalues in %s Pmat !!! : "%typ,w
            #    print " I am  assumng this is numerical noise and setting these to zero"
            #    w[np.where(w <= 0.)] = 0.
            # ret[_ell, :, :] = np.dot(v, np.dot(np.diag(np.sqrt(w)), v.T))

    if derivative is None:
        pass
    elif derivative == 0:
        # Hack to use broadcasting rules :
        ret = np.swapaxes(np.swapaxes(ret, 0, 2) * lib_alm.get_iky(), 0, 2)
    elif derivative == 1:
        # Hack to use broadcasting rules :
        ret = np.swapaxes(np.swapaxes(ret, 0, 2) * lib_alm.get_ikx(), 0, 2)
    else:
        assert 0
    return ret if not inverse else np.linalg.inv(ret)


def get_noisePmat(typ, lib_alm, cls_noise, inverse=False):
    assert typ.upper() in typs, typ
    sl = slice(0, lib_alm.ellmax + 1)
    ret = np.zeros((lib_alm.alm_size, len(typ), len(typ)), dtype=float)
    if typ == 'T':
        ret[:, 0, 0] = cls_noise['t'][sl][lib_alm.reduced_ellmat()]
    elif typ == 'QU':
        ret[:, 0, 0] = cls_noise['q'][sl][lib_alm.reduced_ellmat()]
        ret[:, 1, 1] = cls_noise['u'][sl][lib_alm.reduced_ellmat()]
    elif typ == 'TQU':
        ret[:, 0, 0] = cls_noise['t'][sl][lib_alm.reduced_ellmat()]
        ret[:, 1, 1] = cls_noise['q'][sl][lib_alm.reduced_ellmat()]
        ret[:, 2, 2] = cls_noise['u'][sl][lib_alm.reduced_ellmat()]
    return ret if not inverse else np.linalg.inv(ret)


def TQUPmats2TEBcls(lib_alm, TQUpmat):
    """
    Turns a 3x3x ellmax matrix with T,Q,U ell,m spectra to T,E,B ell only spectra
    """
    assert TQUpmat.shape == (3, 3, lib_alm.alm_size), ((3, 3, lib_alm.alm_size), TQUpmat.shape)
    ret = np.zeros((3, 3, lib_alm.ellmax + 1), dtype=float)
    # E = cos Q + sin U
    # B = -sin Q + cos U
    bin2cl = lambda _alm: lib_alm.bin_realpart_inell(_alm)[0:lib_alm.ellmax + 1]
    ret[0, 0, :] = bin2cl(TQUpmat[0, 0, :])
    # TE : -> cos TQ + sin TU
    cos, sin = lib_alm.get_cossin_2iphi()
    ret[0, 1, :] = bin2cl(TQUpmat[0, 1, :] * cos + TQUpmat[0, 2, :] * sin)
    ret[1, 0, :] = ret[0, 1, :]
    # TB : -> -sin TQ + cos TU
    ret[0, 2, :] = bin2cl(- TQUpmat[0, 1, :] * sin + TQUpmat[0, 2, :] * cos)
    ret[2, 0, :] = ret[0, 2, :]

    # EE : cos2 QQ + sin2 UU + sin cos QU
    ret[1, 1, :] = bin2cl(
        TQUpmat[1, 1, :] * cos ** 2 + TQUpmat[2, 2, :] * sin ** 2 + 2 * cos * sin * TQUpmat[1, 2, :])
    # BB : sin2 QQ + cos2 UU - 2 sin cos QU
    ret[2, 2, :] = bin2cl(
        TQUpmat[2, 2, :] * cos ** 2 + TQUpmat[1, 1, :] * sin ** 2 - 2 * cos * sin * TQUpmat[1, 2, :])

    # EB :  (UU-QQ)*cos*sin + QU * (cos2 -sin2)
    ret[1, 2, :] = bin2cl(
        (TQUpmat[2, 2, :] - TQUpmat[1, 1, :]) * cos * sin + TQUpmat[1, 2, :] * (cos ** 2 - sin ** 2))
    ret[2, 1, :] = ret[1, 2, :]
    return ret


def QUPmats2EBcls(lib_alm, QUpmat):
    """
    Turns a 3x3x ellmax matrix with T,Q,U ell,m spectra to T,E,B ell only spectra
    """
    assert QUpmat.shape == ((2, 2, lib_alm.alm_size)), ((2, 2, lib_alm.alm_size), QUpmat.shape)
    ret = np.zeros((2, 2, lib_alm.ellmax + 1), dtype=float)
    # E = cos Q + sin U
    # B = -sin Q + cos U
    bin2cl = lambda _alm: lib_alm.bin_realpart_inell(_alm)[0:lib_alm.ellmax + 1]
    # TE : -> cos TQ + sin TU
    cos, sin = lib_alm.get_cossin_2iphi()
    # EE : cos2 QQ + sin2 UU + sin cos QU
    ret[0, 0, :] = bin2cl(
        QUpmat[0, 0, :] * cos ** 2 + QUpmat[1, 1, :] * sin ** 2 + 2 * cos * sin * QUpmat[0, 1, :])
    # BB : sin2 QQ + cos2 UU - 2 sin cos QU
    ret[1, 1, :] = bin2cl(
        QUpmat[1, 1, :] * cos ** 2 + QUpmat[0, 0, :] * sin ** 2 - 2 * cos * sin * QUpmat[0, 1, :])

    # EB :  (UU-QQ)*cos*sin + QU * (cos2 -sin2)
    ret[0, 1, :] = bin2cl(
        (QUpmat[1, 1, :] - QUpmat[0, 0, :]) * cos * sin + QUpmat[0, 1, :] * (cos ** 2 - sin ** 2))
    ret[1, 0, :] = ret[0, 1, :]
    return ret

def TEBPmat2TQUPmatij(typ, lib_alm, TEBclmat, i, j):
    """
    Rotates TEB TQU anisotrpoic spec matrix.
    using T Q U = 1 0 0    T E B  = R TEB
                 0  c -s
                 0 s c
            -> R P R^t
    """
    if typ == 'T':
        return TEBclmat[0, 0].copy()
    elif typ == 'QU':
        fl = lambda alm, i, j: alm * TEBclmat[i, j]
        c, s = lib_alm.get_cossin_2iphi()
        if i == 0 and j == 0:
            return fl(c ** 2, 0, 0) + fl(s ** 2, 1, 1) - fl(c * s, 0, 1) - fl(c * s, 1, 0)
        if i == 1 and j == 1:
            return fl(s ** 2, 0, 0) + fl(c ** 2, 1, 1) + fl(c * s, 0, 1) + fl(c * s, 1, 0)
        if i == 0 and j == 1:
            return fl(c * s, 0, 0) - fl(c * s, 1, 1) + fl(c ** 2, 0, 1) - fl(s ** 2, 1, 0)
        if i == 1 and j == 0:
            return fl(c * s, 0, 0) - fl(c * s, 1, 1) - fl(s ** 2, 0, 1) + fl(c ** 2, 1, 0)
        assert 0, (i, j)
    elif typ == 'TQU':
        if i == 0 or j == 0:
            return TEBclmat[i, j].copy()
        fl = lambda alm, i, j: alm * TEBclmat[i, j]
        c, s = lib_alm.get_cossin_2iphi()
        if i == 1 and j == 1:
            return fl(c ** 2, 1, 1) + fl(s ** 2, 2, 2) - fl(c * s, 1, 2) - fl(c * s, 2, 1)
        if i == 2 and j == 2:
            return fl(s ** 2, 1, 1) + fl(c ** 2, 2, 2) + fl(c * s, 1, 2) + fl(c * s, 2, 1)
        if i == 1 and j == 2:
            return fl(c * s, 1, 1) - fl(c * s, 2, 2) + fl(c ** 2, 1, 2) - fl(s ** 2, 2, 1)
        if i == 2 and j == 1:
            return fl(c * s, 1, 1) - fl(c * s, 2, 2) - fl(s ** 2, 1, 2) + fl(c ** 2, 2, 1)
        assert 0, (i, j)
    else:
        assert 0

def TEBcls2TQUPmatij(typ, lib_alm, TEBcls, i, j):
    """
    Rotates TEB TQU anisotrpoic spec matrix.
    using T Q U = 1 0 0    T E B  = R TEB
                 0  c -s
                 0 s c
            -> R P R^t
    """
    if typ == 'T':
        ret = np.ones((1,1,lib_alm.alm_size),dtype = complex)
        return lib_alm.almxfl(ret,TEBcls[0,0])
    elif typ == 'QU':
        fl = lambda alm, i, j: lib_alm.almxfl(alm,TEBcls[i, j])
        c, s = lib_alm.get_cossin_2iphi()
        if i == 0 and j == 0:
            return fl(c ** 2, 0, 0) + fl(s ** 2, 1, 1) - fl(c * s, 0, 1) - fl(c * s, 1, 0)
        if i == 1 and j == 1:
            return fl(s ** 2, 0, 0) + fl(c ** 2, 1, 1) + fl(c * s, 0, 1) + fl(c * s, 1, 0)
        if i == 0 and j == 1:
            return fl(c * s, 0, 0) - fl(c * s, 1, 1) + fl(c ** 2, 0, 1) - fl(s ** 2, 1, 0)
        if i == 1 and j == 0:
            return fl(c * s, 0, 0) - fl(c * s, 1, 1) - fl(s ** 2, 0, 1) + fl(c ** 2, 1, 0)
        assert 0, (i, j)
    elif typ == 'TQU':

        fl = lambda alm, i, j: lib_alm.almxfl(alm,TEBcls[i, j])
        if i == 0 or j == 0:
            ret = np.ones((1, 1, lib_alm.alm_size), dtype=complex)
            return fl(ret,0,0)
        c, s = lib_alm.get_cossin_2iphi()
        if i == 1 and j == 1:
            return fl(c ** 2, 1, 1) + fl(s ** 2, 2, 2) - fl(c * s, 1, 2) - fl(c * s, 2, 1)
        if i == 2 and j == 2:
            return fl(s ** 2, 1, 1) + fl(c ** 2, 2, 2) + fl(c * s, 1, 2) + fl(c * s, 2, 1)
        if i == 1 and j == 2:
            return fl(c * s, 1, 1) - fl(c * s, 2, 2) + fl(c ** 2, 1, 2) - fl(s ** 2, 2, 1)
        if i == 2 and j == 1:
            return fl(c * s, 1, 1) - fl(c * s, 2, 2) - fl(s ** 2, 1, 2) + fl(c ** 2, 2, 1)
        assert 0, (i, j)
    else:
        assert 0


def TQUPmat2TEBPmatij(typ, lib_alm, TQUPmat, i, j):
    """
    Rotates TQU to TEB anisotrpoic spec matrix.
    using T Q U = 1 0 0    T E B  = R TEB
                 0  c -s
                 0 s c
            -> R P R^t
    """
    if typ == 'T':
        return TQUPmat[0, 0].copy()
    elif typ == 'QU':
        fl = lambda alm, i, j: alm * TQUPmat[i, j]
        c, s = lib_alm.get_cossin_2iphi()
        if i == 0 and j == 0:
            return fl(c ** 2, 0, 0) + fl(s ** 2, 1, 1) + fl(c * s, 0, 1) + fl(c * s, 1, 0)
        if i == 1 and j == 1:
            return fl(s ** 2, 0, 0) + fl(c ** 2, 1, 1) - fl(c * s, 0, 1) - fl(c * s, 1, 0)
        if i == 0 and j == 1:
            return -fl(c * s, 0, 0) + fl(c * s, 1, 1) + fl(c ** 2, 0, 1) - fl(s ** 2, 1, 0)
        if i == 1 and j == 0:
            return -fl(c * s, 0, 0) + fl(c * s, 1, 1) - fl(s ** 2, 0, 1) + fl(c ** 2, 1, 0)
        assert 0, (i, j)
    elif typ == 'TQU':
        if i == 0 or j == 0:
            return TQUPmat[i, j].copy()
        fl = lambda alm, i, j: alm * TQUPmat[i, j]
        c, s = lib_alm.get_cossin_2iphi()
        if i == 1 and j == 1:
            return fl(c ** 2, 1, 1) + fl(s ** 2, 2, 2) + fl(c * s, 1, 2) + fl(c * s, 2, 1)
        if i == 2 and j == 2:
            return fl(s ** 2, 1, 1) + fl(c ** 2, 2, 2) - fl(c * s, 1, 2) - fl(c * s, 2, 1)
        if i == 1 and j == 2:
            return -fl(c * s, 1, 1) + fl(c * s, 2, 2) + fl(c ** 2, 1, 2) - fl(s ** 2, 2, 1)
        if i == 2 and j == 1:
            return -fl(c * s, 1, 1) + fl(c * s, 2, 2) - fl(s ** 2, 1, 2) + fl(c ** 2, 2, 1)
        assert 0, (i, j)
    else:
        assert 0


def apply_rootTEBmat(typ, lib_alm, cmb_cls, TEBlms):
    """
    Assumes TB = EB = 0
    """
    assert (typ in typs) and (len(TEBlms) == len(typ))
    assert ('tb' not in cmb_cls.keys()) and ('eb' not in cmb_cls.keys()), cmb_cls.keys()
    if typ == 'T':
        return np.array([lib_alm.almxfl(TEBlms[0], np.sqrt(cmb_cls['tt']))])
    elif typ == 'QU':
        return np.array([lib_alm.almxfl(TEBlms[0], np.sqrt(cmb_cls['ee'])),
                         lib_alm.almxfl(TEBlms[1], np.sqrt(cmb_cls['bb']))])
    elif typ == 'TQU':
        rootCls = _rootCMBcls(cmb_cls)
        fl = lambda id, _f: lib_alm.almxfl(TEBlms[id], rootCls[_f])
        return np.array([fl(0, 'tt') + fl(1, 'te'), fl(0, 'te') + fl(1, 'ee'), fl(2, 'bb')])


def apply_TEBmat(typ, lib_alm, cmb_cls, TEBlms):
    """
    Assumes TB = EB = 0
    """
    assert (typ in typs) and (len(TEBlms) == len(typ)), (len(TEBlms), typ)
    assert ('tb' not in cmb_cls.keys()) and ('eb' not in cmb_cls.keys()), cmb_cls.keys()
    fl = lambda id, _f: lib_alm.almxfl(TEBlms[id], cmb_cls[_f])
    if typ == 'T':
        return np.array([fl(0, 'tt')])
    elif typ == 'QU':
        return np.array([fl(0, 'ee'), fl(1, 'bb')])
    elif typ == 'TQU':
        return np.array([fl(0, 'tt') + fl(1, 'te'), fl(0, 'te') + fl(1, 'ee'), fl(2, 'bb')])
    else:
        assert 0, (typ, typs)


def TQU2TEBlms(typ, lib_alm, TQUlms):
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
    assert typ in typs
    if typ == 'T':
        return np.array(TQUlms).copy()
    elif typ == 'QU':
        cos, sin = lib_alm.get_cossin_2iphi()
        return np.array([cos * TQUlms[0] + sin * TQUlms[1], -sin * TQUlms[0] + cos * TQUlms[1]])
    elif typ == 'TQU':
        cos, sin = lib_alm.get_cossin_2iphi()
        return np.array([TQUlms[0], cos * TQUlms[1] + sin * TQUlms[2], -sin * TQUlms[1] + cos * TQUlms[2]])
    else:
        assert 0, (typ, typs)


def TEB2TQUlms(typ, lib_alm, TEBlms):
    """
    T = A T
    Q     E
    U     B
    where A is
        1   0   0
        0  cos -sin
        0  sin cos
    """
    assert typ in typs
    if typ == 'T':
        return np.array(TEBlms).copy()
    elif typ == 'QU':
        cos, sin = lib_alm.get_cossin_2iphi()
        return np.array([cos * TEBlms[0] - sin * TEBlms[1], sin * TEBlms[0] + cos * TEBlms[1]])
    elif typ == 'TQU':
        cos, sin = lib_alm.get_cossin_2iphi()
        return np.array([TEBlms[0], cos * TEBlms[1] - sin * TEBlms[2], sin * TEBlms[1] + cos * TEBlms[2]])


def get_SlmfromTEBlms(typ, lib_alm, TEBlms, S):
    assert len(TEBlms) == len(typ)
    if typ == 'T':
        if S == 'T': return np.array(TEBlms[0]).copy()
    if typ == 'QU':
        cos, sin = lib_alm.get_cossin_2iphi()
        if S == 'Q':
            return np.array(cos * TEBlms[0] - sin * TEBlms[1])
        elif S == 'U':
            return np.array(sin * TEBlms[0] + cos * TEBlms[1])
    if typ == 'TQU':
        if S == 'T':
            return np.array(TEBlms[0]).copy()
        elif S == 'Q':
            cos, sin = lib_alm.get_cossin_2iphi()
            return np.array(cos * TEBlms[1] - sin * TEBlms[2])
        elif S == 'U':
            cos, sin = lib_alm.get_cossin_2iphi()
            return np.array(sin * TEBlms[1] + cos * TEBlms[2])
    assert 0, (S, typ)


def apply_pinvTEBmat(typ, lib_alm, cmb_cls, TEBlms):
    """
    Assumes TB = EB = 0.
    P^{-1} set to zero when there is no power in the variable (e.g. unl BB or ell = 0,1 in pol)
    """
    assert (typ in typs) and (len(TEBlms) == len(typ)), (len(TEBlms), typ)
    assert ('tb' not in cmb_cls.keys()) and ('eb' not in cmb_cls.keys()), cmb_cls.keys()
    fl = lambda id, cl: lib_alm.almxfl(TEBlms[id], cl)
    if typ == 'T':
        return np.array([fl(0, _clpinv(cmb_cls['tt']))])
    elif typ == 'QU':
        return np.array([fl(0, _clpinv(cmb_cls['ee'])), fl(1, _clpinv(cmb_cls['bb']))])
    elif typ == 'TQU':
        cli = get_pinvTEBcls(typ, cmb_cls)
        return np.array([fl(0, cli['tt']) + fl(1, cli['te']), fl(0, cli['te']) + fl(1, cli['ee']), fl(2, cli['bb'])])
    else:
        assert 0


def get_pinvTEBcls(typ, cmb_cls):
    if typ == 'T':
        return {'tt': _clpinv(cmb_cls['tt'])}
    elif typ == 'QU':
        return {'ee': _clpinv(cmb_cls['ee']), 'bb': _clpinv(cmb_cls['bb'])}
    elif typ == 'TQU':
        ret = {}
        deti = _clpinv(cmb_cls['tt'] * cmb_cls['ee'] - cmb_cls['te'] ** 2)
        ret['tt'] = np.where(deti > 0, cmb_cls['ee'] * deti, _clpinv(cmb_cls['tt']))
        ret['te'] = np.where(deti > 0, -cmb_cls['te'] * deti, np.zeros(len(cmb_cls['te'])))
        ret['ee'] = np.where(deti > 0, cmb_cls['tt'] * deti, _clpinv(cmb_cls['ee']))
        ret['bb'] = _clpinv(cmb_cls['bb'])
        return ret
    else:
        assert 0
