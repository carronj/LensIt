"""
Based on Duncan Hanson quicklens (NB: changed its basic routines from single to double precision).
https://github.com/dhanson/quicklens

the shts module contains routines for performing spherical harmonic transforms (SHT)s.
for more details, please see the notes in quicklens/notes/shts
"""
import sys
import numpy as np
import pyfftw
import util

try:
    import fsht
except ImportError, exc:
    sys.stderr.write("IMPORT ERROR: " + __file__ + " ({})".format(exc)
    + ". Try running 'python setup.py install' or 'python setup.py build_ext --inplace' from the shts directory.\n")

#FIXME: It looks like the setup.py build does not work properly with clang with openMP.
#FIXME: Instead this seems to work: f2py -c -m fsht shts.f90 --f90flags="-fopenmp" -lgomp

#f2py -c -m fsht shts.f90  --f90flags="-fopenmp" -lgomp --fcompiler=gfortran
PYFFTWFLAGS = ['FFTW_MEASURE']

def add_PYFFTWFLAGS(flag):
    PYFFTWFLAGS.append(flag)

def vtm2map(spin, vtm, Nphi, pfftwthreads=4, bicubic_prefilt=False, phiflip=()):
    """ Longitudinal Fourier transform to an ECP grid.
    Sends vtm array to (bicubic prefiltered map) with Nphi points equidistant in [0,2pi).
    With bicubic prefiltering this uses 2lmax + 1 1d Ntheta-sized FFTs and one 2d (Ntheta x Nphi) iFFT.

    Apparently the pyFFTW.FFTW is twice as fast than the pyFFTW.interface.
    But the FFTW wisdom calculation overhead can compensate for the gain if only one map is lensed.

    vtm should come from shts.vlm2vtm which returns a farray, with contiguous vtm[:,ip].
    for spin 0 we have vtm^* = vt_-m, real filtered maps and we may use rffts. (not done)

    vtm should of size 2 * lmax + 1
    Flipping phi amounts to phi -> 2pi - phi -> The phi fft is sent to its transpose.
    """
    lmax = (vtm.shape[1] - 1) / 2
    Nt = vtm.shape[0]
    assert (Nt, 2 * lmax + 1) == vtm.shape, ((Nt, 2 * lmax + 1), vtm.shape)
    assert Nphi % 2 == 0, Nphi
    if bicubic_prefilt:
        #TODO: Could use real ffts for spin 0. For high-res interpolation this task can take about half of the total time.
        a = pyfftw.empty_aligned(Nt, dtype=complex)
        b = pyfftw.empty_aligned(Nt, dtype=complex)
        ret = pyfftw.empty_aligned((Nt, Nphi), dtype=complex, order='C')
        fftmap = pyfftw.empty_aligned((Nt, Nphi), dtype=complex, order='C')
        ifft2 = pyfftw.FFTW(fftmap, ret, axes=(0, 1), direction='FFTW_BACKWARD', threads=pfftwthreads,flags=PYFFTWFLAGS)
        fft1d = pyfftw.FFTW(a, b, direction='FFTW_FORWARD', threads=1,flags = PYFFTWFLAGS)
        fftmap[:] = 0. #NB: sometimes the operations above can result in nan's
        if Nphi > 2 * lmax:
            # There is unique correspondance m <-> kphi where kphi is the 2d flat map frequency
            for ip, m in enumerate(range(-lmax, lmax + 1)):
                fftmap[:, (Nphi + m if m < 0 else m)] = fft1d(vtm[:, ip])
        else:
            # The correspondance m <-> k is not unique anymore, but given by
            # (m - k) = N j for j in 0,+- 1, +- 2 ,etc. -> m = k + Nj
            for ik in range(Nphi):
                # candidates for m index
                ms = lmax + ik + Nphi * np.arange(-lmax / Nphi - 1, lmax / Nphi + 1, dtype=int)
                ms = ms[np.where((ms >= 0) & (ms <= 2 * lmax))]
                fftmap[:, ik] = fft1d(np.sum(vtm[:, ms], axis=1))
        w0 = Nphi * 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(Nt)) + 4.)
        w1 = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(Nphi)) + 4.)
        fftmap[:] *= np.outer(w0, w1)
        retmap = ifft2().real if spin == 0 else ifft2()

    else :
        # Probably no real gain to expect here from pyfftw for the ffts.
        # (Actually a lot in relative terms, could implement that)
        if Nphi > 2 * lmax + 1:
            a = np.zeros((Nt,Nphi),dtype = complex)
            a[:,:2 * lmax + 1] = vtm
            ret = np.fft.ifft(a) * (np.exp(np.arange(Nphi) * (-1j / Nphi * (2. * np.pi) * lmax)) * Nphi)
        else:
            ret = np.fft.ifft(vtm[:,lmax - Nphi/2:lmax + Nphi/2])
            ret *= (np.exp(np.arange(Nphi) * (-1j / Nphi * (2. * np.pi) * Nphi/2)) * Nphi)
        retmap = ret.real if spin == 0 else ret
    retmap[phiflip, :] = retmap[phiflip,::-1]
    return retmap

def map2vtm(spin,lmax,_map,pfftwthreads = 1):
    """ longitudinal Fourier transforms of an ECP grid
    input _map is (Ntheta,Nphi)-shaped.
    Output vtm is (Ntheta,2 * lmax + 1) shaped, with v[t,lmax + m] = vtm.
    This assume the _map samples uniformly the longitude [0,2pi) with Nphi = _map.shape[1] points.
    (e.g. vtm2map output)
    """
    assert _map.shape[1] % 2 == 0,_map.shape
    Nt,Nphi = _map.shape
    a = pyfftw.empty_aligned(Nphi, dtype=complex)
    b = pyfftw.empty_aligned(Nphi, dtype=complex)
    fft1d = pyfftw.FFTW(a, b, direction='FFTW_FORWARD', threads=pfftwthreads,flags = PYFFTWFLAGS)
    vtm = np.zeros( (Nt,2 * lmax + 1),dtype = complex,order = 'C')
    if Nphi <= 2 * lmax:
        sli = slice(lmax - Nphi/2,lmax+Nphi/2)
        for it in range(Nt): vtm[it, sli] = np.fft.fftshift(fft1d(_map[it, :]))
        vtm[:,lmax + Nphi/2] = vtm[:,lmax - Nphi/2]
    else : # We could conceive downsampling in some situations but who cares.
        sli = slice(Nphi/2 - lmax ,Nphi/2 + lmax + 1)
        for it in range(Nt): vtm[it,:] = np.fft.fftshift(fft1d(_map[it, :]))[sli]
    return vtm * (2. * np.pi / Nphi)


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
    This is \sum_l _s\Lambda_{lm} v_{lm}. 0 < tht < pi are colatitudes .
    """
    assert s >= 0
    lmax = int(np.sqrt(len(vlm)) - 1)
    assert (len(vlm) == (lmax + 1) ** 2)
    if s == 0:
        print "Consider using glm2vtm for spin 0 for factor of 2 speed-up"
        return fsht.vlm2vtm(lmax, s, tht, vlm)
    #: resolving poles, since fsht implementation of spin > 0 does not handle them.
    north = np.where(tht <= 0.)[0]
    south = np.where(tht >= np.pi)[0]
    if len(north) == 0 and len(south) == 0:
        return fsht.vlm2vtm(lmax, s, tht, vlm)
    else:
        ret = np.zeros((len(tht), 2 * lmax + 1), dtype=complex)
        if len(north) > 0:
            ret[north] = _vlm2vtm_northpole(s, vlm)
        if len(south) > 0:
            ret[south] = _vlm2vtm_southpole(s, vlm)
        if len(north) + len(south) < len(tht):
            others = np.where((tht < np.pi) & (tht > 0.))[0]
            ret[others] = fsht.vlm2vtm(lmax, s, tht[others], vlm)
        return ret

def vlm2vtm_sym(s, tht, vlm):
    """
    This is \sum_l _s\Lambda_{lm} v_{lm}. tht colatitude.
    """
    assert s >= 0
    tht = np.array(tht)
    lmax = int(np.sqrt(len(vlm)) - 1)
    assert (len(vlm) == (lmax + 1) ** 2)
    if s == 0:
        print "Consider using glm2vtm_sym for spin 0 for factor of 2 speed-up"
        return fsht.vlm2vtm_sym(lmax, s, tht, vlm)
    else:
        #: resolving poles, since fsht implementation does not handle them.
        north = np.where(tht <= 0.)[0]
        south = np.where(tht >= np.pi)[0]
        if len(north) == 0 and len(south) == 0:
            return fsht.vlm2vtm_sym(lmax, s, tht, vlm)
        else:
            nt = len(tht)
            ret = np.zeros( (2 * nt, 2 * lmax + 1), dtype=complex)
            if len(north) > 0:
                ret[north] = _vlm2vtm_northpole(s, vlm)
                ret[nt + north] = _vlm2vtm_southpole(s, vlm)
            if len(south) > 0:
                ret[south] = _vlm2vtm_southpole(s, vlm)
                ret[nt + south] = _vlm2vtm_northpole(s, vlm)
            if len(north) + len(south) < len(tht):
                others = np.where( (tht < np.pi) & (tht > 0.))[0]
                vtm =  fsht.vlm2vtm_sym(lmax, s, tht[others], vlm)
                ret[others] = vtm[:len(others)]
                ret[nt + others] = vtm[len(others):]
            return ret

def _vlm2vtm_northpole(s, vlm):
    """On the north pole, the spherical harmonics obeys _s\Lambda_{l,-s} (-1)^s  \sqrt{ (2l + 1) / 4\pi }
        and zero otherwise.
    """
    assert s >= 0, s
    lmax = int(np.sqrt(len(vlm)) - 1)
    assert (len(vlm) == (lmax + 1) ** 2)
    ret = np.zeros(2 * lmax + 1, dtype=complex)
    l = np.arange(lmax + 1)
    ret[lmax - s] =  np.sum(vlm[l * l + l - s] * np.sqrt((2 * l + 1.))) / np.sqrt(4. * np.pi)  * (-1) ** s
    return ret

def _vlm2vtm_southpole(s, vlm):
    """On the south pole, the spherical harmonics obeys _s\Lambda_{l,s} (-1)^l  \sqrt{ (2l + 1) / 4\pi }
        and zero otherwise.
    """
    assert s >= 0, s
    lmax = int(np.sqrt(len(vlm)) - 1)
    assert (len(vlm) == (lmax + 1) ** 2)
    ret = np.zeros(2 * lmax + 1, dtype=complex)
    l = np.arange(lmax + 1)
    ret[lmax + s] =  np.sum(vlm[l * l + l + s] * (-1) ** l * np.sqrt((2 * l + 1.))) / np.sqrt(4. * np.pi)
    return ret


def vtm2vlm(s, tht, vtm):
    """
    This computes sum_th sLm(tht) vtm.
    This returns original vlm for input vtm * dcos(tht) * (2 pi) on a fine co-latitude grid.

    Recall that \int_1^1 dcost sLlm sLl'm = 1/(2 pi) delta_{ll'}
    """
    return fsht.vtm2vlm(s, tht, vtm)

def vtm2tlm_sym(tht, vtm):
    """
    This computes sum_th sLm(tht) vtm.
    This returns original vlm for input vtm * dcos(tht) * (2 pi) on a fine co-latitude grid, using symetry tricks

    Recall that \int_1^1 dcost sLlm sLl'm = 1/(2 pi) delta_{ll'}
    """
    lmax = (vtm.shape[1] - 1) /2
    assert 2 * lmax + 1 == vtm.shape[1] and vtm.shape[0] == len(tht)
    assert len(tht) % 2 == 0
    assert np.all(np.sort(tht) == tht)
    assert np.allclose(np.pi - tht[:len(tht)/2],tht[len(tht)/2:][::-1])
    _vtm = np.zeros((len(tht),lmax + 1),dtype = complex)
    _vtm[:len(tht) // 2,:]  = vtm[:len(tht) // 2,lmax:] + vtm[slice(len(tht), len(tht) // 2 -1, -1),lmax:]
    _vtm[len(tht) // 2 :,:] = vtm[:len(tht) // 2,lmax:] - vtm[slice(len(tht), len(tht) // 2 -1, -1),lmax:]
    return fsht.vtm2alm_syms0(tht[:len(tht)/2], _vtm)


