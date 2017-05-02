# quicklens/shts/util.py
# --
# this module contains utilities for working with harmonic
# coefficients. there are three different formats used here:
#
#   'vlm' = complex coefficients vlm[l*l+l+m], with l \in [0, lmax] and m \in [-l, l]
#   'alm' = complex coefficients alm[m * (2*lmax+1-m)/2 + l] with l in [0, lmax] and m in [0, l].
#                * corresponds to a field which has a real-valued spin-0 map.
#                * this is the format used by healpy for harmonic transforms
#   'rlm' = real coefficients rlm[l*l + 2*m + 0] and rlm[l*l + 2*m + 1]
#                * corresponds to the real and imaginary parts of alm, useful for matrix operations.

import numpy as np


def cls2vlm(clgg, clcc=None, clgc=None, rand=np.random.standard_normal):
    """ generate a set of harmonic coefficients v_{lm}.
         * clgg            = gradient-mode power spectrum, array of length lmax+1.
         * (optional) clcc = curl-mode power spectrum, array of length lmax+1 (defaults to zeros).
         * (optional) clgc = gradient-curl cross spectrum, array of length lmax+1 (defaults to zeros).
         * (optional) rand = function rand(n) which will generate an array with n pulls from a N(0,1) Gaussian.
    """
    lmax = len(clgg) - 1
    if clcc == None:
        clcc = np.zeros(lmax + 1)
    if clgc == None:
        clgc = np.zeros(lmax + 1)

    clgg_inv = np.zeros(len(clgg))
    clgg_inv[np.where(clgg != 0)] = 1. / clgg[np.where(clgg != 0)]

    ms = (
        np.ceil(
            ((2 * lmax + 1) - np.sqrt((2 * lmax + 1) ** 2 - 8 * (np.arange(0, lmax2nlm(lmax)) - lmax))) / 2)).astype(
        int)
    ls = np.arange(0, lmax2nlm(lmax), dtype=np.int) - ms * (2 * lmax + 1 - ms) / 2

    r1 = np.sqrt(0.5) * (rand(lmax2nlm(lmax)) + 1.j * rand(lmax2nlm(lmax)))
    r2 = np.sqrt(0.5) * (rand(lmax2nlm(lmax)) + 1.j * rand(lmax2nlm(lmax)))

    glm = r1 * np.sqrt(clgg[ls])
    clm = r1 * clgc[ls] * np.sqrt(clgg_inv[ls]) + r2 * np.sqrt(clcc[ls] - clgc[ls] ** 2 * clgg_inv[ls])
    del r1, r2, ls

    m0s = np.where(ms == 0);
    del ms
    glm[m0s] = np.sqrt(2.) * glm[m0s].real
    clm[m0s] = np.sqrt(2.) * clm[m0s].real

    return alm2vlm(glm, clm)


def nlm2lmax(nlm):
    """ returns the lmax for an array of alm with length nlm. """
    lmax = int(np.floor(np.sqrt(2 * nlm) - 1))
    assert ((lmax + 2) * (lmax + 1) / 2 == nlm)
    return lmax


def lmax2nlm(lmax):
    """ returns the length of the complex alm array required for maximum multipole lmax. """
    return (lmax + 1) * (lmax + 2) / 2


def alm2vlm(glm, clm=None):
    """
    convert alm format -> vlm format coefficients. glm is gradient mode, clm is curl mode.
    For pure gradients holds vl-m = (-1) ** m vlm^*, half the array is redundant and vlm = - glm,
    with ret[l * l + l + m] = -glm
    with ret[l * l + l - m] = -(-1)^m glm^*

    For pure curls     holds vl-m =-(-1) ** m vlm^*, half the array is redundant and vlm = -i clm,
    with ret[l * l + l + m] = -i clm
    with ret[l * l + l - m] = -i(-1)^m clm^*
    """
    lmax = nlm2lmax(len(glm))
    ret = np.zeros((lmax + 1) ** 2, dtype=np.complex)
    for l in xrange(0, lmax + 1):
        ms = np.arange(1, l + 1)
        ret[l * l + l] = -glm[l]
        ret[l * l + l + ms] = -glm[ms * (2 * lmax + 1 - ms) / 2 + l]
        ret[l * l + l - ms] = -(-1) ** ms * np.conj(glm[ms * (2 * lmax + 1 - ms) / 2 + l])

    if not clm is None:
        assert (len(clm) == len(glm))
        for l in xrange(0, lmax + 1):
            ms = np.arange(1, l + 1)
            ret[l * l + l] += -1.j * clm[l]
            ret[l * l + l + ms] += -1.j * clm[ms * (2 * lmax + 1 - ms) / 2 + l]
            ret[l * l + l - ms] += -(-1) ** ms * 1.j * np.conj(clm[ms * (2 * lmax + 1 - ms) / 2 + l])

    return ret


def vlm2alm(vlm):
    """ convert vlm format coefficients -> alm. returns gradient and curl pair (glm, clm). """
    lmax = int(np.sqrt(len(vlm)) - 1)

    glm = np.zeros(lmax2nlm(lmax), dtype=np.complex)
    clm = np.zeros(lmax2nlm(lmax), dtype=np.complex)

    for l in xrange(0, lmax + 1):
        ms = np.arange(1, l + 1)

        glm[l] = -vlm[l * l + l].real
        clm[l] = -vlm[l * l + l].imag

        glm[ms * (2 * lmax + 1 - ms) / 2 + l] = -0.5 * (vlm[l * l + l + ms] + (-1) ** ms * np.conj(vlm[l * l + l - ms]))
        clm[ms * (2 * lmax + 1 - ms) / 2 + l] = 0.5j * (vlm[l * l + l + ms] - (-1) ** ms * np.conj(vlm[l * l + l - ms]))
    return glm, clm


def alm2rlm(alm):
    """ converts a complex alm to 'real harmonic' coefficients rlm. """

    lmax = nlm2lmax(len(alm))
    rlm = np.zeros((lmax + 1) ** 2)

    ls = np.arange(0, lmax + 1)
    l2s = ls ** 2
    rt2 = np.sqrt(2.)

    rlm[l2s] = alm[ls].real
    for m in xrange(1, lmax + 1):
        rlm[l2s[m:] + 2 * m - 1] = alm[m * (2 * lmax + 1 - m) / 2 + ls[m:]].real * rt2
        rlm[l2s[m:] + 2 * m + 0] = alm[m * (2 * lmax + 1 - m) / 2 + ls[m:]].imag * rt2
    return rlm


def rlm2alm(rlm):
    """ converts 'real harmonic' coefficients rlm to complex alm. """

    lmax = int(np.sqrt(len(rlm)) - 1)
    assert ((lmax + 1) ** 2 == len(rlm))

    alm = np.zeros(lmax2nlm(lmax), dtype=np.complex)

    ls = np.arange(0, lmax + 1, dtype=np.int64)
    l2s = ls ** 2
    ir2 = 1.0 / np.sqrt(2.)

    alm[ls] = rlm[l2s]
    for m in xrange(1, lmax + 1):
        alm[m * (2 * lmax + 1 - m) / 2 + ls[m:]] = (rlm[l2s[m:] + 2 * m - 1] + 1.j * rlm[l2s[m:] + 2 * m + 0]) * ir2
    return alm
