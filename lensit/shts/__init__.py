"""
Based on Duncan Hanson quicklens (NB: changed from single to double precision).
https://github.com/dhanson/quicklens

the shts module contains routines for performing spherical harmonic transforms (SHT)s.
for more details, please see the notes in quicklens/notes/shts
"""
import sys
import numpy as np

import util

try:
    import fsht
except ImportError, exc:
    sys.stderr.write("IMPORT ERROR: " + __file__ + " ({})".format(
        exc) + ". Try running 'python setup.py install' or 'python setup.py build_ext --inplace' from the quicklens directory.\n")


def vlm2map(s, tht, phi, vlm):
    """ perform an inverse spin-s spherical harmonic transform from vlm to a complex map for an (n x m) grid of (theta, phi).
    inputs: * integer spin value s.
            * real vector theta=(theta_1, ..., theta_n).
            * real vector phi=(phi_1, ..., phi_m).
            * complex vector of harmonic coefficients v_lm=v[l*l+l+m] with l \in [0,lmax] and m \in [-l, l].
    output: * (n x m) complex map v(i,j) = \sum_{lm} {}_s Y_{lm}(\theta_i, \theta_j) v_{lm}.
    """
    lmax = int(np.sqrt(len(vlm)) - 1)
    assert (len(vlm) == (lmax + 1) ** 2)

    if s < 0:
        vlmn = vlm.copy()
        for l in xrange(0, lmax + 1):
            vlmn[l ** 2:(l + 1) ** 2] = vlmn[l ** 2:(l + 1) ** 2][::-1] * (-1) ** (np.arange(-l, l + 1.))
        ret = -fsht.vlm2map(lmax, -s, tht, -np.array(phi), vlmn)

    else:
        ret = fsht.vlm2map(lmax, s, tht, phi, vlm)

    return ret


def map2vlm(lmax, s, tht, phi, mp):
    """ perform a spin-s spherical harmonic transform, from a complex map to harmonic coeficients vlm.
    inputs: * maximum multipole lmax.
            * integer spin value s.
            * real vector theta=(theta_1, ..., theta_n).
            * real vector phi=(phi_1, ..., phi_m).
            * complex 2d (n x m) map mp.
    output: * complex vector of harmonic coefficients vlm=v[l*l+l+m] with l \in [0,lmax] and m \in [-l, l], given by \sum_{ij} dtheta_i dphi_j Y_{lm}^{*}(\theta_i, \phi_j) map(i,j)
    """
    assert (mp.shape == (len(tht), len(phi)))

    if s < 0:
        ret = -fsht.map2vlm(lmax, -s, tht, -np.array(phi), mp)

        for l in xrange(0, lmax + 1):
            ret[l ** 2:(l + 1) ** 2] = ret[l ** 2:(l + 1) ** 2][::-1] * (-1) ** (np.arange(-l, l + 1.))
    else:
        ret = fsht.map2vlm(lmax, s, tht, phi, mp)

    return ret


def glm2vtm(s, tht, glm):
    if s == 0:
        lmax = util.nlm2lmax(len(glm))
        ret = np.empty((len(tht), 2 * lmax + 1), dtype=complex)
        ret[:, lmax:] = fsht.glm2vtm_s0(lmax, tht, glm)
        ret[:, 0:lmax] = (ret[:, slice(2 * lmax + 1, lmax, -1)]).conjugate()
        return ret
    return vlm2vtm(s, tht, util.alm2vlm(glm))


def glm2vtm_sym(s, tht, glm):
    """
    Same as above but returns for each 'tht' the result for pi -tht as well, for a factor 1.5 speed-up.
    Output has shape 2 * len(tht) x 2 * lmax + 1
    """
    if s == 0:
        lmax = util.nlm2lmax(len(glm))
        ret = np.empty((2 * len(tht), 2 * lmax + 1), dtype=complex)
        ret[:, lmax:] = fsht.glm2vtm_s0sym(lmax, tht, -glm)
        ret[:, 0:lmax] = (ret[:, slice(2 * lmax + 1, lmax, -1)]).conjugate()
        return ret
    return vlm2vtm_sym(s, tht, util.alm2vlm(glm))


def vlm2vtm(s, tht, vlm):
    """
    This is \sum_l _s\Lambda_{lm} v_{lm}. 0 < tht < np.pi are colatitudes .
    """
    assert s >= 0
    lmax = int(np.sqrt(len(vlm)) - 1)
    assert (len(vlm) == (lmax + 1) ** 2)
    if s == 0: print "Consider using glm2vtm for spin 0 for factor of 2 speed-up"
    return fsht.vlm2vtm(lmax, s, tht, vlm)


def vlm2vtm_sym(s, tht, vlm):
    """
    This is \sum_l _s\Lambda_{lm} v_{lm}.
    tht colatitude.
    """
    assert s >= 0
    lmax = int(np.sqrt(len(vlm)) - 1)
    assert (len(vlm) == (lmax + 1) ** 2)
    if s == 0: print "Consider using glm2vtm_sym for spin 0 for factor of 2 speed-up"
    return fsht.vlm2vtm_sym(lmax, s, tht, vlm)


def vtm2vlm(s, tht, vtm):
    """
    This computes sum_th sLm(tht) vtm.
    This returns original vlm for input vtm * dcos(tht) * (2 pi) on a fine co-latitude grid.
    """
    return fsht.vtm2vlm(s, tht, vtm)
