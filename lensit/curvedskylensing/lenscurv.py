"""
Accuracy is roughly homogeneous except a ring of ~8 pixels around the pole with ~10 times bigger errors (still small though, 0.2%)
It does not matter, but might try fix that at some point.

ist_lensalm stats  :

lens_alm timings (seconds):
    tms  0
    def  107.468
    ecp  72.3171
    itp  30.7173
    tot  212.794

lens_vlm timings (seconds): pol
    vms  0
    def  100.093
    ecp  256.494
    itp  26.5123
    tot  389.109

This python script stats :
STATS for lmax tlm 2048 lmax dlm 3072
         vtm2filtmap: 13.76
              interp: 2.22
    dx,dy band split: 12.29
    dx,dy (full sky): 43.66
                 vtm: 24.42
                 tot: 96 sec.
excl. defl. angle calc.: 40 sec.
      Tot. int. pix.: 314572800 = 17736.2^2

# ==== Same using cos(t) symmetry (same accuracy) :
STATS for lmax tlm 2048 lmax dlm 3072
         vtm2filtmap: 12.69
              interp: 3.42
    dx,dy band split: 10.99
    dx,dy (full sky): 41.01
                 vtm: 16.19
                 tot: 84 sec.
excl. defl. angle calc.: 32 sec.
      Tot. int. pix.: 311427072 = 17647.3^2
#===== Same in pol :
STATS for lmax glm 2048 lmax dlm 3072
              interp: 7.14
                 vtm: 39.50
         vtm2filtmap: 11.85
    dx,dy (full sky): 42.19
           pol. rot.: 13.34
    dx,dy band split: 11.06
                 tot: 125 sec.
excl. defl. angle calc.: 72 sec.
      Tot. int. pix.: 311427072 = 17647.3^2

MacBook-Pro-de-Julien:lenspix jcarron$ time mpirun -np 8 ./lensmap params.ini ../mllens/temp/test_curved_lensing/testtlm_2048.fits ../mllens/temp/test_curved_lensing/testplm_3072.fits ../mllens/temp/test_curved_lensing/test_lens_map.fits
 (interp 2.0) --> 18432.0 pix around the eq. (crashes for higher interp.)

 ALM2GRADIENTMAP: Sending to farm
 ALM2GRADIENTMAP Time:    39.573000907897949
 SCALALM2LENSEDMAPINTERPCYL: Sending to farm
 SCALALM2LENSEDMAPINTERPCYL Time:    68.965070009231567
real	2m0.315s


acc = lambda m1,m2 : (np.mean(np.abs(m1-m2))/np.std(m2),np.max(np.abs(m1-m2))/np.std(m2))

acc(lenspixmap,lenmap)
Out[71]: (8.0307321881247895e-05, 0.0021220560960932967)

acc(istlenmap,lenmap)
Out[72]: (3.8204375224999072e-06, 2.9529778885108624)

acc(istlenmap,lenspixmap)
Out[73]: (8.107202038382077e-05, 2.9528609003802373)


# Stats for 4096 + 1024 :
     [00:02:33]  (total [00:04:28]) Total exec. time  lenscurv 
STATS for lmax tlm 5120 lmax dlm 5120
              interp: 3.72
                 vtm: 114.25
         vtm2filtmap: 22.27
    dx,dy (full sky): 115.70
           pol. rot.: 0.00
    dx,dy band split: 11.73
                 tot: 268 sec.
excl. defl. angle calc.: 140 sec.
      Tot. int. pix.: 311427072 = 17647.3^2
"""

from lensit import shts
import healpy as hp,numpy as np
from lensit.misc.misc_utils import timer

try:
    import weave
except:
    from scipy import weave
import os,time
import pyfftw

# TODO : use rfft for spin 0 vtm2filtmap ? fac ~ 2 possible gain in vtm2filtmap
# FIXME : polarization rotation.
# FIXME : adapt for curl, send dlm to dlm + dcllm
# NB : dont forget to change clang to local gcc when building it.


#======== couple of lensing routines

class thgrid():
    def __init__(self,th1,th2):
        """
        Co-latitudes th1 and th2 between 0 (N. pole) and pi (S. Pole).
        negative theta values are reflected across the north pole.
        (this allows simple padding to take care of the non periodicty of the pine in theta direction.)
        Same for south pole.
        """
        self.th1 = th1
        self.th2 = th2
    def mktgrid(self,Nt):
        return  self.th1 + np.arange(Nt) * ( (self.th2- self.th1) / (Nt-1))
    def togridunits(self,tht,Nt):
        return  (tht - self.th1) / ((self.th2- self.th1) / (Nt-1))

def _th2colat(th):
    ret = np.abs(th)
    ret[np.where(th > np.pi)] = 2 * np.pi - th[np.where(th > np.pi)]
    return ret

def get_def_hp(nside,dlm,th1,th2):
    # FIXME band only calc. with vtm ?
    pix = hp.query_strip(nside, max(th1, 0.), min(np.pi, th2), inclusive=True)
    Red, Imd = hp.alm2map_spin([dlm, np.zeros_like(dlm)], nside, 1, hp.Alm.getlmax(dlm.size))
    Red = Red[pix]
    Imd = Imd[pix]
    return _buildangles(hp.pix2ang(nside, pix),Red[pix],Imd[pix])

def get_def(nside,dlm,Nphi,thgrid,clm = None):
    pass
    #vtm = shts.vlm2vtm_sym(1, _th2colat(tgrid), shts.util.alm2vlm(dlm, clm=clm))
    # FIXME band only calc. with vtm ?
    #pix = hp.query_strip(nside, max(th1, 0.), min(np.pi, th2), inclusive=True)
    #Red, Imd = hp.alm2map_spin([dlm, np.zeros_like(dlm)], nside, 1, hp.Alm.getlmax(dlm.size))
    #Red = Red[pix]
    #Imd = Imd[pix]
    #return _buildangles(hp.pix2ang(nside, pix),Red[pix],Imd[pix])

#====== Wrappers
def tlm2lensmap(nside,tlm,dlm,verbose = True,nband = 16,facres = 0):
    return lens_glm_sym_timed(0,dlm,-tlm,nside,nband = nband,facres = facres)
def eblm2lensmap(nside,elm,dlm,verbose = True,nband = 16,facres = 0,blm = None):
    return lens_glm_sym_timed(2,dlm,elm,nside,nband = nband,facres=facres,clm = blm)

def lens_tlm(tlm,dlm,lmaxout = None,verbose = True,nside = 2048,nband = 16,facres = 0,iter = 0):
    # FIXME : suppress the stupid intermediate step
    lenmap = tlm2lensmap(nside,tlm,dlm,verbose=verbose,nband=nband,facres=facres)
    return hp.map2alm(lenmap,lmax = (lmaxout or hp.Alm.getlmax(tlm.size)),iter = iter)

def lens_eblm(elm,dlm,blm = None,nside = 2048,lmaxout = None,verbose = True,nband = 16,facres = 0):
    # FIXME : suppress the stupid intermediate step
    lenmap = eblm2lensmap(nside,elm,dlm,blm = blm,verbose=verbose,nband=nband,facres=facres)
    return hp.map2alm_spin([lenmap.real, lenmap.imag], 2, lmax = (lmaxout or hp.Alm.getlmax(elm.size)))

#====== Mains :
def lens_gcband_sym(spin,glm, th1, th2, Nt, (tnewN, phinewN), (tnewS, phinewS),clm = None, Nphi=2 ** 14):
    """
    Returns deflected maps between a co-latitude range in the North Hemisphere and its South hemisphere correspondant.
    :param spin: spin of the field described by gradient glm and curl clm mode.
    :param glm:  gradient lm components of the field, in healpy ordering.
    :param th1:  input co-latitude range is (th1,th2) on North hemisphere and (pi - th2,pi -th1) on South hemisphere.
    :param th2:  input co-latitude range is (th1,th2) on North hemisphere and (pi - th2,pi -th1) on South hemisphere.
    :param Nt:  Number of tht point to use for the interpolation.
    :param clm :  curl lm components of the field, in healpy ordering.
    :param Nphi : Number of phi points to use for the interpolation.
    :param (tnewN,phinewN) : deflected positions on North hemisphere. (colat, long.)
    :param (tnewS,phinewS) : deflected positions on North hemisphere. (colat, long.)
    :return: real and imaginary parts of lensed bands (th1,th2) and (pi - th2,pi -th1) as np array.

    NB : This routine performs the deflection only, and not the parallel-transport rotation of the axes.
    """
    assert len(tnewN) == len(phinewN) and len(tnewS) == len(phinewS),(len(tnewN),len(phinewN),len(tnewS),len(phinewS))
    tgrid = thgrid(th1, th2).mktgrid(Nt)

    # Calls \sum_l sLm(tht) vlm at each tht and m :
    # (first Nt elements are th1 to th2 and last Nt are pi - th1, pi - th2)
    vtm = shts.vlm2vtm_sym(spin, _th2colat(tgrid), shts.util.alm2vlm(glm, clm = clm))
    # Turn this into the bicubic-prefiltered map to interpolate :
    filtmapN = vtm2filtmap(spin, vtm[0:Nt], Nphi,phiflip=np.where((tgrid < 0))[0])
    filtmapS = vtm2filtmap(spin, vtm[slice(2 * Nt - 1, Nt - 1, -1)], Nphi,phiflip=np.where((np.pi - tgrid) > np.pi)[0])
    del vtm
    # Actual interpolation for the 4 real maps :
    bicubicspline = r"for(int j= 0; j < npix; j++ )\
                       {lenmap[j] = bicubiclensKernel_rect(filtmap,phinew[j],tnew[j],Nx,Ny);}"
    header = r' "%s/lensit/gpu/bicubicspline.h" ' % os.path.abspath(os.curdir)

    lenmapNR = np.empty(len(tnewN), dtype=float)
    lenmapNI = np.empty(len(tnewN), dtype=float)
    lenmapSR = np.empty(len(tnewS), dtype=float)
    lenmapSI = np.empty(len(tnewS), dtype=float)
    for N, filtmap, lenmap, (_tnew, _phinew) in zip([1,1,0,0],
                                            [filtmapN.real, filtmapN.imag,filtmapS.real, filtmapS.imag],
                                            [lenmapNR,lenmapNI, lenmapSR,lenmapSI],
                                            [(tnewN, phinewN),(tnewN, phinewN), (tnewS, phinewS), (tnewS, phinewS)]):
        if filtmap.flags.farray: filtmap = np.copy(filtmap, order='C')  # Could not make the following work with F
        npix = len(lenmap)
        tnew = thgrid(th1, th2).togridunits(_tnew, Nt) if N else thgrid(np.pi - th2, np.pi - th1).togridunits(_tnew, Nt)
        phinew = _phinew / ((2. * np.pi) / Nphi)
        Ny, Nx = filtmap.shape
        weave.inline(bicubicspline, ['lenmap', 'filtmap', 'phinew', 'tnew', 'npix', 'Nx', 'Ny'], headers=[header])
    return lenmapNR,lenmapNI,lenmapSR,lenmapSI

def lens_band_sym(glm,th1,th2,Nt,(tnewN,phinewN),(tnewS,phinewS),Nphi = 2 ** 14):
    """
    Same as lens_gcband_sym for spin 0 field, where maps are real.
    Note that in the adopted conventions tlm = -glm for temperature maps.
    """
    assert len(tnewN) == len(phinewN) and len(tnewS) == len(phinewS)
    tgrid = thgrid(th1,th2).mktgrid(Nt)

    # get bicubic pre-filtered map that we will interpolate
    # for spin 0 this is real
    # first Nt elements are th1 to th2 and last Nt are pi - th1, pi - th2
    vtm = shts.glm2vtm_sym(0, _th2colat(tgrid), glm)
    filtmapN = vtm2filtmap(0,vtm[0:Nt],Nphi,phiflip=np.where((tgrid < 0))[0])
    filtmapS = vtm2filtmap(0,vtm[slice(2 * Nt -1,Nt-1,-1)],Nphi,phiflip=np.where( (np.pi - tgrid) > np.pi)[0])
    del vtm

    bicubicspline = r"for(int j= 0; j < npix; j++ )\
                       {lenmap[j] = bicubiclensKernel_rect(filtmap,phinew[j],tnew[j],Nx,Ny);}"
    header = r' "%s/lensit/gpu/bicubicspline.h" ' % os.path.abspath(os.curdir)
    lenmapN = np.empty(len(tnewN),dtype = float)
    lenmapS = np.empty(len(tnewS),dtype = float)

    for N,filtmap,lenmap,(tnew,phinew) in zip([1,0],[filtmapN,filtmapS],[lenmapN,lenmapS],
                                                    [(tnewN,phinewN),(tnewS,phinewS)]):
        if filtmap.flags.farray : filtmap = np.copy(filtmap, order='C') # Could not make the following work with F
        npix = len(lenmap)
        tnew = thgrid(th1,th2).togridunits(tnew,Nt) if N else thgrid(np.pi - th2, np.pi- th1).togridunits(tnew,Nt)
        phinew /= ((2. * np.pi)/ Nphi)
        Ny, Nx = filtmap.shape
        weave.inline(bicubicspline, ['lenmap', 'filtmap', 'phinew', 'tnew', 'npix','Nx','Ny'], headers=[header])
    return lenmapN,lenmapS


def lens_band_sym_timed(glm, th1, th2, Nt, (tnewN, phinewN), (tnewS, phinewS), Nphi=2 ** 14):
    """
    Same as lens_band_sym with some more timing info.
    """
    assert len(tnewN) == len(phinewN) and len(tnewS) == len(phinewS)

    tgrid = thgrid(th1, th2).mktgrid(Nt)
    times = {}
    t0 = time.time()
    vtm = shts.glm2vtm_sym(0, _th2colat(tgrid), glm)
    times['vtm'] = time.time() - t0
    t0 = time.time()
    filtmapN = vtm2filtmap(0, vtm[0:Nt], Nphi,phiflip=np.where((tgrid < 0))[0])
    filtmapS = vtm2filtmap(0, vtm[slice(2 * Nt - 1, Nt - 1, -1)], Nphi,phiflip = np.where((np.pi - tgrid) > np.pi)[0])
    times['vtm2filtmap'] = time.time() - t0
    del vtm
    t0 = time.time()
    lenmapN = np.empty(len(tnewN), dtype=float)
    lenmapS = np.empty(len(tnewS), dtype=float)
    bicubicspline = r"for(int j= 0; j < npix; j++ )\
                       {lenmap[j] = bicubiclensKernel_rect(filtmap,phinew[j],tnew[j],Nx,Ny);}"
    header = r' "%s/lensit/gpu/bicubicspline.h" ' % os.path.abspath(os.curdir)

    for N, filtmap, lenmap, (tnew, phinew) in zip([1, 0], [filtmapN, filtmapS], [lenmapN, lenmapS],
                                                  [(tnewN, phinewN), (tnewS, phinewS)]):
        if filtmap.flags.farray: filtmap = np.copy(filtmap, order='C')  # Could not make the following work with F
        npix = len(lenmap)
        tnew = thgrid(th1, th2).togridunits(tnew, Nt) if N else thgrid(np.pi - th2, np.pi - th1).togridunits(tnew, Nt)
        phinew /= ((2. * np.pi) / Nphi)
        Ny, Nx = filtmap.shape
        weave.inline(bicubicspline, ['lenmap', 'filtmap', 'phinew', 'tnew', 'npix', 'Nx', 'Ny'], headers=[header])
    times['interp'] = time.time() - t0
    return lenmapN, lenmapS,times

def gclm2lensmap_symband_timed(spin, glm, th1, th2, Nt, (tnewN, phinewN), (tnewS, phinewS), clm = None, Nphi=2 ** 14):
    """
    Same as lens_gcband_sym with some more timing info.
    """
    assert len(tnewN) == len(phinewN) and len(tnewS) == len(phinewS)
    tgrid = thgrid(th1, th2).mktgrid(Nt)
    times = {}
    t0 = time.time()
    vtm = shts.vlm2vtm_sym(spin, _th2colat(tgrid), shts.util.alm2vlm(glm, clm = clm))
    times['vtm'] = time.time() - t0
    t0 = time.time()
    filtmapN = vtm2filtmap(spin, vtm[0:Nt], Nphi,phiflip=np.where((tgrid < 0))[0])
    filtmapS = vtm2filtmap(spin, vtm[slice(2 * Nt - 1, Nt - 1, -1)], Nphi,phiflip=np.where((np.pi - tgrid) > np.pi)[0])
    times['vtm2filtmap'] = time.time() - t0
    del vtm
    t0 = time.time()
    lenmapNR = np.empty(len(tnewN), dtype=float)
    lenmapNI = np.empty(len(tnewN), dtype=float)
    lenmapSR = np.empty(len(tnewS), dtype=float)
    lenmapSI = np.empty(len(tnewS), dtype=float)

    bicubicspline = r"for(int j= 0; j < npix; j++ )\
                       {lenmap[j] = bicubiclensKernel_rect(filtmap,phinew[j],tnew[j],Nx,Ny);}"
    header = r' "%s/lensit/gpu/bicubicspline.h" ' % os.path.abspath(os.curdir)

    for N, filtmap, lenmap, (_tnew, _phinew) in zip([1,1,0,0],
                                            [filtmapN.real, filtmapN.imag,filtmapS.real, filtmapS.imag],
                                            [lenmapNR,lenmapNI, lenmapSR,lenmapSI],
                                            [(tnewN, phinewN),(tnewN, phinewN), (tnewS, phinewS), (tnewS, phinewS)]):
        if filtmap.flags.farray: filtmap = np.copy(filtmap, order='C')  # Could not make the following work with F
        npix = len(lenmap)
        tnew = thgrid(th1, th2).togridunits(_tnew, Nt) if N else thgrid(np.pi - th2, np.pi - th1).togridunits(_tnew, Nt)
        phinew = _phinew / ((2. * np.pi) / Nphi)
        Ny, Nx = filtmap.shape
        weave.inline(bicubicspline, ['lenmap', 'filtmap', 'phinew', 'tnew', 'npix', 'Nx', 'Ny'], headers=[header])
    times['interp'] = time.time() - t0
    return lenmapNR,lenmapNI,lenmapSR,lenmapSI,times

def lensgclm_symband_timed(spin,glm, th1, th2, Nt, (tnewN, phinewN), (tnewS, phinewS),clm = None, Nphi=2 ** 14):
    """
    Same as lens_gcband_sym with some more timing info.
    """
    assert len(tnewN) == len(phinewN) and len(tnewS) == len(phinewS)
    tgrid = thgrid(th1, th2).mktgrid(Nt)
    times = {}
    t0 = time.time()
    vtm = shts.vlm2vtm_sym(spin, _th2colat(tgrid), shts.util.alm2vlm(glm, clm = clm))
    times['vtm'] = time.time() - t0
    t0 = time.time()
    filtmapN = vtm2filtmap(spin, vtm[0:Nt], Nphi,phiflip=np.where((tgrid < 0))[0])
    filtmapS = vtm2filtmap(spin, vtm[slice(2 * Nt - 1, Nt - 1, -1)], Nphi,phiflip=np.where((np.pi - tgrid) > np.pi)[0])
    times['vtm2filtmap'] = time.time() - t0
    del vtm
    t0 = time.time()
    lenmapNR = np.empty(len(tnewN), dtype=float)
    lenmapNI = np.empty(len(tnewN), dtype=float)
    lenmapSR = np.empty(len(tnewS), dtype=float)
    lenmapSI = np.empty(len(tnewS), dtype=float)

    bicubicspline = r"for(int j= 0; j < npix; j++ )\
                       {lenmap[j] = bicubiclensKernel_rect(filtmap,phinew[j],tnew[j],Nx,Ny);}"
    header = r' "%s/lensit/gpu/bicubicspline.h" ' % os.path.abspath(os.curdir)

    for N, filtmap, lenmap, (_tnew, _phinew) in zip([1,1,0,0],
                                            [filtmapN.real, filtmapN.imag,filtmapS.real, filtmapS.imag],
                                            [lenmapNR,lenmapNI, lenmapSR,lenmapSI],
                                            [(tnewN, phinewN),(tnewN, phinewN), (tnewS, phinewS), (tnewS, phinewS)]):
        if filtmap.flags.farray: filtmap = np.copy(filtmap, order='C')  # Could not make the following work with F
        npix = len(lenmap)
        tnew = thgrid(th1, th2).togridunits(_tnew, Nt) if N else thgrid(np.pi - th2, np.pi - th1).togridunits(_tnew, Nt)
        phinew = _phinew / ((2. * np.pi) / Nphi)
        Ny, Nx = filtmap.shape
        weave.inline(bicubicspline, ['lenmap', 'filtmap', 'phinew', 'tnew', 'npix', 'Nx', 'Ny'], headers=[header])
    times['interp'] = time.time() - t0
    return lenmapNR,lenmapNI,lenmapSR,lenmapSI,times


def _buildangles((tht,phi),Red,Imd):
    """
    e.g.
        Redtot, Imdtot = hp.alm2map_spin([dlm, np.zeros_like(dlm)], nside, 1, hp.Alm.getlmax(dlm.size))
        pix = hp.query_strip(nside, th1, np.max(tht_patch), inclusive=True)
        costnew,phinew = buildangles(hp.pix2ang(nside, pix),Redtot[pix],Imdtot[pix])
    """
    # FIXME :  poles
    costnew = np.cos(tht)
    phinew = phi.copy()
    norm = np.sqrt(Red ** 2 + Imd ** 2)
    ii = np.where(norm > 0.)
    costnew[ii] = np.cos(norm[ii]) * costnew[ii] - np.sin(norm[ii]) * np.sin(tht[ii]) * (Red[ii] / norm[ii])
    phinew[ii] += np.arcsin((Imd[ii] / norm[ii]) * np.sin(norm[ii]) / (np.sqrt(1. - costnew[ii] ** 2)))
    return np.arccos(costnew),phinew

def polrot(spin,lenmap,tht,Red,Imd):
    # FIXME :  poles

    if spin == 2 :
        ret = np.ones(len(lenmap),dtype = complex)
        norm = np.sqrt(Red ** 2 + Imd ** 2)
        ii = np.where(norm > 0.)
        A  = Imd[ii] / (norm[ii] * np.sin(norm[ii]) * np.cos(tht[ii]) / np.sin(tht[ii]) + Red[ii] * np.cos(norm[ii]))
        ret[ii] = 2 * (Red[ii] + Imd[ii] * A) ** 2 + 2j * (Red[ii] + Imd[ii] * A) * (Imd[ii] - Red[ii] * A)
        ret[ii] /= (norm[ii] ** 2 * (1. + A ** 2))
        ret[ii] -= 1.
        return ret
    else : assert 0

def get_Nphi(th1,th2,facres = 0):
    """ Calculates a phi sampling density at co-latitude theta """
    target_amin = 0.745 # 0.66 corresponds to 2 ** 15 = 32768
    sint = max(np.sin(th1),np.sin(th2))
    for res in np.arange(15,3,-1):
        if 2. * np.pi / (2 ** (res-1)) * 180. * 60 /np.pi * sint >= target_amin : return 2 ** (res + facres)
    assert 0

def lens_glm_sym(spin,dlm,glm,nside,nband = 32,facres = 0,clm = None,rotpol = True):
    """
    Same as lens_alm but lens simultnously a North and South colatitude band,
    to make profit of the symmetries of the spherical harmonics.
    Note that tlm = -glm in the spin 0 case.
    """
    target_nt = 3 ** 1 * 2 ** (11 + facres)
    th1s = np.arange(nband) * (np.pi * 0.5 / nband)
    th2s = np.concatenate((th1s[1:],[np.pi * 0.5]))
    Nt_perband = target_nt / nband
    Redtot, Imdtot = hp.alm2map_spin([dlm, np.zeros_like(dlm)], nside, 1, hp.Alm.getlmax(dlm.size))
    ret = np.zeros(hp.nside2npix(nside),dtype = float if spin == 0 else complex)
    for th1,th2 in zip(th1s,th2s):
        pixN = hp.query_strip(nside, th1, th2, inclusive=True)
        pixS = hp.query_strip(nside, np.pi- th2,np.pi - th1, inclusive=True)
        tnewN,phinewN = _buildangles(hp.pix2ang(nside, pixN),Redtot[pixN],Imdtot[pixN])
        tnewS,phinewS = _buildangles(hp.pix2ang(nside, pixS), Redtot[pixS], Imdtot[pixS])
        matnewN = np.max(tnewN)
        mitnewN = np.min(tnewN)
        matnewS = np.max(tnewS)
        mitnewS = np.min(tnewS)
        buffN = 10 * (matnewN - mitnewN) / (Nt_perband - 1) / (1. - 2. * 10. / (Nt_perband - 1))
        buffS = 10 * (matnewS - mitnewS) / (Nt_perband - 1) / (1. - 2. * 10. / (Nt_perband - 1))
        thup = min(np.pi - (matnewS + buffS),mitnewN - buffN)
        thdown = max(np.pi - (mitnewS - buffS),matnewN + buffN)
        if spin == 0:
            lenN,lenS = lens_band_sym(glm,thup,thdown,Nt_perband,(tnewN,phinewN),(tnewS,phinewS),
                                                Nphi=get_Nphi(thup, thdown, facres=facres))
            ret[pixN] = lenN
            ret[pixS] = lenS
        else :
            lenNR,lenNI,lenSR,lenSI = lens_gcband_sym(spin,glm,thup,thdown, Nt_perband,(tnewN,phinewN),(tnewS,phinewS),
                                                      Nphi=get_Nphi(thup, thdown, facres=facres),clm = clm)
            ret[pixN] = lenNR + 1j * lenNI
            ret[pixS] = lenSR + 1j * lenSI
            if rotpol :
                ret[pixN] *= polrot(spin,ret[pixN], hp.pix2ang(nside, pixN)[0],Redtot[pixN],Imdtot[pixN])
                ret[pixS] *= polrot(spin,ret[pixS], hp.pix2ang(nside, pixS)[0],Redtot[pixS],Imdtot[pixS])
    return ret

def lens_glm_sym_timed(spin,dlm,glm,nside,nband = 32,facres = 0,clm = None,rotpol = True):
    """
    Same as lens_alm but lens simultnously a North and South colatitude band,
    to make profit of the symmetries of the spherical harmonics.
    """
    assert spin >= 0,spin
    t = timer(True,suffix=' ' + __name__)
    target_nt = 3 ** 1 * 2 ** (11 + facres) # on one hemisphere
    times = {}

    #co-latitudes
    th1s = np.arange(nband) * (np.pi * 0.5 / nband)
    th2s = np.concatenate((th1s[1:],[np.pi * 0.5]))
    ret = np.zeros(hp.nside2npix(nside),dtype = float if spin == 0 else complex)
    Nt_perband = target_nt / nband
    t0 = time.time()
    Redtot, Imdtot = hp.alm2map_spin([dlm, np.zeros_like(dlm)], nside, 1, hp.Alm.getlmax(dlm.size))
    times['dx,dy (full sky)'] = time.time() - t0
    times['dx,dy band split'] = 0.
    times['pol. rot.'] = 0.
    t.checkpoint('healpy Spin 1 transform for displacement (full %s map)' % nside)
    _Npix = 0 # Total number of pixels used for interpolation
    def coadd_times(tim):
        for _k,_t in tim.iteritems():
            if _k not in times :
                times[_k] = _t
            else : times[_k] += _t

    for ib,th1,th2 in zip(range(nband),th1s,th2s):
        print "BAND %s in %s :"%(ib,nband)
        t0 = time.time()
        pixN = hp.query_strip(nside, th1, th2, inclusive=True)
        pixS = hp.query_strip(nside, np.pi- th2,np.pi - th1, inclusive=True)
        tnewN,phinewN = _buildangles(hp.pix2ang(nside, pixN),Redtot[pixN],Imdtot[pixN])
        tnewS,phinewS = _buildangles(hp.pix2ang(nside, pixS), Redtot[pixS], Imdtot[pixS])

        # Adding a 10 pixels buffer for new angles to be safely inside interval.
        # th1,th2 is mapped onto pi - th2,pi -th1 so we need to make sure to cover both buffers
        matnewN = np.max(tnewN)
        mitnewN = np.min(tnewN)
        matnewS = np.max(tnewS)
        mitnewS = np.min(tnewS)

        buffN = 10 * (matnewN - mitnewN) / (Nt_perband - 1) / (1. - 2. * 10. / (Nt_perband - 1))
        buffS = 10 * (matnewS - mitnewS) / (Nt_perband - 1) / (1. - 2. * 10. / (Nt_perband - 1))
        _thup = min(np.pi - (matnewS + buffS),mitnewN - buffN)
        _thdown = max(np.pi - (mitnewS - buffS),matnewN + buffN)

        #print "min max tnew (degrees) in the band %.3f %.3f "%(_th1 /np.pi * 180.,_th2 /np.pi * 180.)
        #==== these are the theta and limits. It is ok to go negative or > 180
        print 'input t1,t2 %.3f %.3f in degrees'%(_thup /np.pi * 180,_thdown/np.pi * 180.)
        print 'North %.3f and South %.3f buffers in amin'%(buffN /np.pi * 180 * 60,buffS/np.pi * 180. * 60.)
        Nphi = get_Nphi(_thup, _thdown, facres=facres)
        dphi_patch = (2. * np.pi) / Nphi * max(np.sin(_thup),np.sin(_thdown))
        dth_patch = (_thdown - _thup) / (Nt_perband -1)
        print "cell (theta,phi) in amin (%.3f,%.3f)" % (dth_patch / np.pi * 60. * 180, dphi_patch / np.pi * 60. * 180)
        times['dx,dy band split'] += time.time() - t0
        if spin == 0:
            lenN,lenS,tim = lens_band_sym_timed(glm,_thup,_thdown,Nt_perband,(tnewN,phinewN),(tnewS,phinewS),Nphi=Nphi)
            ret[pixN] = lenN
            ret[pixS] = lenS
        else :
            lenNR,lenNI,lenSR,lenSI,tim = gclm2lensmap_symband_timed(spin, glm, _thup, _thdown, Nt_perband,
                                                            (tnewN, phinewN), (tnewS, phinewS), Nphi=Nphi, clm = clm)
            ret[pixN] = lenNR + 1j * lenNI
            ret[pixS] = lenSR + 1j * lenSI
            t0 = time.time()
            if rotpol and spin > 0 :
                ret[pixN] *= polrot(spin,ret[pixN], hp.pix2ang(nside, pixN)[0],Redtot[pixN],Imdtot[pixN])
                ret[pixS] *= polrot(spin,ret[pixS], hp.pix2ang(nside, pixS)[0],Redtot[pixS],Imdtot[pixS])
                times['pol. rot.'] += time.time() -t0
        coadd_times(tim)

        #coadd_times(tim)
        _Npix += 2 * Nt_perband * Nphi
    t.checkpoint('Total exec. time')

    print "STATS for lmax tlm %s lmax dlm %s"%(hp.Alm.getlmax(glm.size),hp.Alm.getlmax(dlm.size))
    tot= 0.
    for _k,_t in times.iteritems() :
        print '%20s: %.2f'%(_k,_t)
        tot += _t
    print "%20s: %2.f sec."%('tot',tot)
    print "%20s: %2.f sec."%('excl. defl. angle calc.',tot - times['dx,dy (full sky)']-times['dx,dy band split'])
    print '%20s: %s = %.1f^2'%('Tot. int. pix.',int(_Npix),np.sqrt(_Npix * 1.))

    return ret

def vtm2filtmap(spin,vtm,Nphi,threads = 8,phiflip = []):
    return vtm2map(spin,vtm,Nphi,threads=threads,bicubic_prefilt=True,phiflip=phiflip)

def vtm2map(spin,vtm, Nphi,threads = 8,bicubic_prefilt = False,phiflip = []):
        """
        Send vtm array to bicubic prefiltered map to be interpolated with Nphi points equidistant in [0,2pi).
        This uses 2lmax + 1 1d Ntheta-sized FFTs and one 2d (Ntheta x Nphi) iFFT.

        Apparently the pyFFTW.FFTW is twice as fast than the pyFFTW.interface.
        But the FFTW wisdom calculation overhead can compensate for the gain if only one map is lensed.

        vtm should come from shts.vlm2vtm which returns a farray, with contiguous vtm[:,ip].

        for spin 0 we have vtm^* = vt_-m, real filtered maps and we may use rffts.

        vtm should of size 2 * lmax + 1

        Flipping phi amounts to phi -> 2pi - phi -> The phi fft is sent to its transpose.
        """
        # looks like it actually works OK
        # TODO : separate spin 0  (using rffts etc) from spin != 0. (~2 gain)
        # TODO : better wisdom handling ? (~10 sec gain)
        # TODO : put the phi filtering in the vtm calculation ? (Probably nothing)
        lmax = (vtm.shape[1] - 1) / 2
        Nt = vtm.shape[0]
        assert (Nt, 2 * lmax + 1) == vtm.shape, ((Nt, 2 * lmax + 1), vtm.shape)
        # pyFFTW objects and Wisdom. Apparently the pyFFTW.FFTW is twice as fast than the pyFFTW.interface.
        ret = pyfftw.empty_aligned((Nt, Nphi), dtype=complex,order='C')
        ifftmap = pyfftw.empty_aligned((Nt, Nphi),dtype=complex,order = 'C')
        ifft2 = pyfftw.FFTW(ifftmap, ret, axes=(0, 1), direction='FFTW_BACKWARD', threads=threads)

        a = pyfftw.empty_aligned(Nt, dtype=complex)
        b = pyfftw.empty_aligned(Nt, dtype=complex)
        fft1d = pyfftw.FFTW(a, b,direction='FFTW_FORWARD',threads=threads)
        if bicubic_prefilt :
            w0 = (Nphi) * 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(Nt)) + 4.)
            w1 = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(Nphi)) + 4.)
            ifftmap[:] = np.outer(w0, w1)
        else :
            ifftmap[:] = 1.

        #pyfftw.interfaces.cache.enable()
        if Nphi > 2 * lmax :
            # There is unique correspondance m <-> kphi where kphi is the 2d flat map frequency
            for ip, m in enumerate(range(-lmax, lmax + 1)):
                # vtm array goes from -lmax to + lmax
                ifftmap[:, (Nphi + m if m < 0 else m)] *= fft1d(vtm[:, ip])
                #ifftmap[:, (Nphi + m if m < 0 else m)] = pyfftw.interfaces.numpy_fft.fft(vtm[:, ip],
                #                                        threads=threads,overwrite_input=True)
        else :
            # The correspondance m <-> k is not unique anymore, but given by
            # (m - k) = N j for j in 0,+- 1, +- 2 ,etc. -> m = k + Nj
            for ik in range(Nphi):
                # vtm array goes from -lmax to + lmax
                ms = lmax + ik + Nphi * np.arange(-lmax / Nphi - 1,lmax / Nphi + 1,dtype = int) # candidates for m index
                ms = ms[np.where( (ms >= 0) & (ms <= 2 * lmax))]
                ifftmap[:, ik] *= fft1d(np.sum(vtm[:,ms],axis = 1))
                # ifftmap[:, ik] = pyfftw.interfaces.numpy_fft.fft(np.sum(vtm[:,ms],axis = 1),
                #                                        threads=threads,overwrite_input=True)
        ifftmap[phiflip,:].imag *= -1.
        if spin == 0 : return ifft2().real
        return ifft2()
        #return pyfftw.interfaces.numpy_fft.ifft2(ifftmap,threads=threads,overwrite_input=True)
        #wisdom = pyfftw.export_wisdom()

def map2vtm(spin,lmax,_map,threads = 8):
    """
    input _map is (Ntheta,Nphi)-shaped.
    Output vtm is (Ntheta,2 * lmax + 1) shaped, with v[t,lmax + m] = vtm.
    This assume the _map samples uniformly the longitude [0,2pi) with Nphi = _map.shape[1] points.
    """
    assert _map.shape[1] % 2 == 0,_map.shape
    Nt,Nphi = _map.shape
    a = pyfftw.empty_aligned(Nphi, dtype=complex)
    b = pyfftw.empty_aligned(Nphi, dtype=complex)
    fft1d = pyfftw.FFTW(a, b, direction='FFTW_FORWARD', threads=threads)
    vtm = np.zeros( (Nt,2 * lmax + 1),dtype = complex)
    if Nphi <= 2 * lmax:
        sli = slice(lmax - Nphi/2,lmax+Nphi/2)
        for it in range(Nt): vtm[it, sli] = np.fft.fftshift(fft1d(_map[it, :]))
        vtm[:,lmax + Nphi/2] = vtm[:,lmax - Nphi/2]
    else : # We could conceive downsampling in some situations but who cares.
        sli = slice(Nphi/2 - lmax ,Nphi/2 + lmax + 1)
        for it in range(Nt): vtm[it,:] = np.fft.fftshift(fft1d(_map[it, :]))[sli]
    return vtm
