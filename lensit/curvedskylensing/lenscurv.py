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
from . import gauleg
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
    header = r' "%s/lensit/gpu/bicubicspline.h" ' % os.path.abspath(os.environ.get('LENSIT',os.curdir))

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
    header = r' "%s/lensit/gpu/bicubicspline.h" ' % os.path.abspath(os.environ.get('LENSIT',os.curdir))
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
    header = r' "%s/lensit/gpu/bicubicspline.h" ' % os.path.abspath(os.environ.get('LENSIT',os.curdir))

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
    header = r' "%s/lensit/gpu/bicubicspline.h" ' % os.path.abspath(os.environ.get('LENSIT',os.curdir))

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
    header = r' "%s/lensit/gpu/bicubicspline.h" ' % os.path.abspath(os.environ.get('LENSIT',os.curdir))

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
    costnew = np.cos(tht)
    phinew = phi.copy()
    norm = np.sqrt(Red ** 2 + Imd ** 2)
    ii = np.where(norm > 0.)
    costnew[ii] = np.cos(norm[ii]) * costnew[ii] - np.sin(norm[ii]) * np.sin(tht[ii]) * (Red[ii] / norm[ii])
    ii = np.where( (norm > 0.) & (costnew ** 2 < 1.))
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
    else : assert 0,'not implemented'

def get_Nphi(th1,th2,facres = 0,target_amin = 0.745):
    """ Calculates a phi sampling density at co-latitude theta """
    # 0.66 corresponds to 2 ** 15 = 32768
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


def lens_glm_GLth_sym_timed(spin,dlm,glm,lmax_target,
            nband = 16,facres = 0,clm = None,olm = None,rotpol = True):
    """
    Same as lens_alm but lens simultnously a North and South colatitude band,
    to make profit of the symmetries of the spherical harmonics.
    """
    assert spin >= 0,spin
    times = {}
    t0 = time.time()
    tGL,wg = gauleg.get_xgwg(lmax_target + 2)
    times['GL points and weights'] =time.time() -t0
    target_nt = 3 ** 1 * 2 ** (11 + facres) # on one hemisphere (0.87 arcmin spacing)
    th1s = np.arange(nband) * (np.pi * 0.5 / nband)
    th2s = np.concatenate((th1s[1:],[np.pi * 0.5]))
    Nt = target_nt / nband
    tGL = np.arccos(tGL)
    tGL = np.sort(tGL)
    wg = wg[np.argsort(tGL)]
    times['pol. rot.'] = 0.
    times['vtm2defl2ang'] = 0.
    times['vtmdefl'] = 0.

    def coadd_times(tim):
        for _k,_t in tim.iteritems():
            if _k not in times :
                times[_k] = _t
            else : times[_k] += _t

    shapes = []
    shapes_d =[]

    tGLNs = []
    tGLSs = []
    wgs = []
    # Collects (Nt,Nphi) per band and prepare wisdom
    wisdomhash = str(lmax_target) + '_' + str(nband) + '_' + str(facres + 1000) + '.npy'
    assert os.path.exists(os.path.dirname(os.path.realpath(__file__)) + '/pyfftw_cache/')
    t0 = time.time()
    print "building and caching FFTW wisdom, this might take a while"
    for ib,th1,th2 in zip(range(nband),th1s,th2s):
        Np = get_Nphi(th1,th2, facres=facres,target_amin =60. * 90. / target_nt) # same spacing as theta grid
        Np_d = min(get_Nphi(th1,th2,target_amin = 180. * 60. / lmax_target),2 * lmax_target) #Equator point density
        pixN, = np.where((tGL >= th1) & (tGL <= th2))
        pixS, = np.where((tGL >= (np.pi - th2)) & (tGL <= (np.pi - th1)))
        assert np.all(pixN[::-1] == len(tGL) - 1 -pixS),'symmetry of GL points'
        shapes_d.append((len(pixN),Np_d))
        shapes.append((Nt,Np))
        tGLNs.append(tGL[pixN])
        tGLSs.append(tGL[pixS])
        wgs.append(np.concatenate([wg[pixN],wg[pixS]]))
        print "BAND %s in %s. deflection    (%s x %s) pts "%(ib,nband,len(pixN),Np_d)
        print "               interpolation (%s x %s) pts "%(Nt,Np)
        #==== For each block we have the following ffts:
        # (Np_d) complex to complex (deflection map) BACKWARD (vtm2map)
        # (Nt,Np) complex to complex (bicubic prefiltering) BACKWARD (vt2mmap) (4 threads)
        # (Nt) complex to complex (bicubic prefiltering) FORWARD (vt2map)
        # (Np_d) complex to complex  FORWARD (map2vtm)
        # Could rather do a try with FFTW_WISDOM_ONLY
        if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + '/pyfftw_cache/' + wisdomhash):
            a = pyfftw.empty_aligned(Np_d, dtype='complex128')
            b = pyfftw.empty_aligned(Np_d, dtype='complex128')
            fft = pyfftw.FFTW(a, b, direction='FFTW_FORWARD', threads=1)
            fft = pyfftw.FFTW(a, b, direction='FFTW_BACKWARD', threads=1)
            a = pyfftw.empty_aligned(Nt, dtype='complex128')
            b = pyfftw.empty_aligned(Nt, dtype='complex128')
            fft = pyfftw.FFTW(a, b, direction='FFTW_FORWARD', threads=1)
            a = pyfftw.empty_aligned((Nt,Np), dtype='complex128')
            b = pyfftw.empty_aligned((Nt,Np), dtype='complex128')
            fft = pyfftw.FFTW(a, b, direction='FFTW_BACKWARD', axes = (0,1),threads=4)

    if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + '/pyfftw_cache/' + wisdomhash):
        np.save(os.path.dirname(os.path.realpath(__file__)) + '/pyfftw_cache/' + wisdomhash,pyfftw.export_wisdom())
    pyfftw.import_wisdom(np.load(os.path.dirname(os.path.realpath(__file__)) + '/pyfftw_cache/' + wisdomhash))
    shts.PYFFTWFLAGS = ['FFTW_WISDOM_ONLY']
    times['pyfftw_caches'] = time.time() -t0
    print "Total number of interpo points: %s = %s ** 2"%(np.sum([np.prod(s) for s in shapes]),np.sqrt(1. * np.sum([np.prod(s) for s in shapes])))
    print "Total number of deflect points: %s = %s ** 2"%(np.sum([np.prod(s) for s in shapes_d]),np.sqrt(1. * np.sum([np.prod(s) for s in shapes_d])))

    glmout = np.zeros(shts.util.lmax2nlm(lmax_target), dtype=np.complex)
    clmout = np.zeros(shts.util.lmax2nlm(lmax_target), dtype=np.complex)
    for ib,th1,th2 in zip(range(nband),th1s,th2s):

        Nt_d,Np_d = shapes_d[ib]
        Nt,Np = shapes[ib]

        t0 = time.time()
        vtm_def = shts.vlm2vtm_sym(1, _th2colat(tGLNs[ib]), shts.util.alm2vlm(dlm, clm=olm))
        times['vtmdefl'] += time.time() - t0

        #==== gettting deflected positions
        # NB: forward slice to keep theta -> pi - theta correspondance.
        t0 = time.time()
        dmapN = shts.vtm2map(1, vtm_def[:Nt_d,:], Np_d).flatten()
        dmapS = shts.vtm2map(1, vtm_def[slice(Nt_d,2 * Nt_d), :], Np_d).flatten()

        told = np.outer(tGLNs[ib],np.ones(Np_d)).flatten()
        phiold = np.outer(np.ones(Nt_d),np.arange(Np_d) * (2. * np.pi / Np_d)).flatten()

        tnewN, phinewN = _buildangles((told, phiold), dmapN.real, dmapN.imag)
        tnewS, phinewS = _buildangles(( (np.pi -told)[::-1], phiold), dmapS.real, dmapS.imag)
        del vtm_def
        times['vtm2defl2ang'] += time.time() - t0

        #===== Adding a 10 pixels buffer for new angles to be safely inside interval.
        # th1,th2 is mapped onto pi - th2,pi -th1 so we need to make sure to cover both buffers
        matnewN = np.max(tnewN)
        mitnewN = np.min(tnewN)
        matnewS = np.max(tnewS)
        mitnewS = np.min(tnewS)


        buffN = 10 * (matnewN - mitnewN) / (Nt - 1) / (1. - 2. * 10. / (Nt - 1))
        buffS = 10 * (matnewS - mitnewS) / (Nt - 1) / (1. - 2. * 10. / (Nt - 1))
        _thup = min(np.pi - (matnewS + buffS),mitnewN - buffN)
        _thdown = max(np.pi - (mitnewS - buffS),matnewN + buffN)

        #==== these are the theta and limits. It is ok to go negative or > 180

        dphi_patch = (2. * np.pi) / Np * max(np.sin(_thup),np.sin(_thdown))
        dth_patch = (_thdown - _thup) / (Nt -1)

        print 'input t1,t2 %.3f %.3f in degrees'%(_thup /np.pi * 180,_thdown/np.pi * 180.)
        print 'North %.3f and South %.3f buffers in amin'%(buffN /np.pi * 180 * 60,buffS/np.pi * 180. * 60.)
        print "cell (theta,phi) in amin (%.3f,%.3f)" % (dth_patch / np.pi * 60. * 180, dphi_patch / np.pi * 60. * 180)

        if spin == 0:
            lenN,lenS,tim = lens_band_sym_timed(glm,_thup,_thdown,Nt,(tnewN,phinewN),(tnewS,phinewS),Nphi=Np)
            ret = np.zeros((2 * Nt_d,Np_d),dtype = complex)
            ret[:Nt_d,:]  = lenN.reshape((Nt_d, Np_d))
            ret[Nt_d:, :] = lenS.reshape((Nt_d, Np_d))
            vtm = shts.map2vtm(spin, lmax_target, ret)
            glmout -= shts.vtm2tlm_sym(np.concatenate([tGLNs[ib],tGLSs[ib]]),vtm * np.outer(wgs[ib], np.ones(vtm.shape[1])))
        else :
            assert 0,'fix this'
            lenNR,lenNI,lenSR,lenSI,tim = gclm2lensmap_symband_timed(spin, glm, _thup, _thdown, Nt,
                                                            (tnewN, phinewN), (tnewS, phinewS), Nphi=Nphi, clm = clm)
            retN = (lenNR + 1j * lenNI).reshape( (len(pixN),Np_d))
            retS = (lenSR + 1j * lenSI).reshape( (len(pixN),Np_d))
            glm, clm = shts.util.vlm2alm(shts.vtm2vlm(spin, tGL, vtm * np.outer(wg, np.ones(vtm.shape[1]))))
            t0 = time.time()
            if rotpol and spin > 0 :
                ret[pixN,:] *= polrot(spin,retN.flatten(), tnewN,dmapN.real,dmapN.imag)
                ret[pixS,:] *= polrot(spin,retS.flatten(), tnewS,dmapS.real,dmapS.imag)
                times['pol. rot.'] += time.time() -t0
        coadd_times(tim)

    t0 = time.time()
    print "STATS for lmax tlm %s lmax dlm %s"%(hp.Alm.getlmax(glm.size),hp.Alm.getlmax(dlm.size))
    tot= 0.
    for _k,_t in times.iteritems() :
        print '%20s: %.2f'%(_k,_t)
        tot += _t
    print "%20s: %2.f sec."%('tot',tot)
    return glmout,clmout,ret


def vtm2filtmap(spin,vtm,Nphi,threads = 4,phiflip = []):
    return shts.vtm2map(spin,vtm,Nphi,pfftwthreads=threads,bicubic_prefilt=True,phiflip=phiflip)



def _map2gclm(spin,lmax,_map,tht,wg,threads = 1):
    """
    input _map is (Ntheta,Nphi)-shaped.
    Output vtm is (Ntheta,2 * lmax + 1) shaped, with v[t,lmax + m] = vtm.
    This assume the _map samples uniformly the longitude [0,2pi) with Nphi = _map.shape[1] points.
    tht and wg : colatitude and integration weights (e.g. GL zeroes and weights)

    Here is a trick to save fac of 2 time in map2alm:
    \int dcost f(t) sLlm(t) = int_(upperhalf) f+(t) sLlm + int_(upperhalf) f-(t) sLlm
    where f+,f_ are the symmmetric / antisymmetric part, and the first term non zero only for l-m of the same partity and vice vers.

    """
    vtm = shts.map2vtm(spin,lmax,_map,pfftwthreads=threads)
    vlm = shts.vtm2vlm(spin,tht, vtm * np.outer(wg, np.ones(vtm.shape[1])))
    return shts.util.vlm2alm(vlm)



"""
                 vtm: 444.31
            map2gclm: 72.41
              interp: 2.37
         vtm2filtmap: 24.86
        vtm2defl2ang: 12.77
           pol. rot.: 0.00
GL points and weights: 28.04
             vtmdefl: 314.82
                 tot: 900 sec.
      Tot. int. pix.: 311427072 = 17647.3^2
lmax = 4096
lmax_target = 4096
dlmax = 1024
facres = 0
import lensit as li,jc_camb as camb
pl.ion()
clunltt = camb.spectra_fromcambfile('/Users/jcarron/SpyderProjects/jpipe/inputs/FFP10/FFP10_lenspotentialCls.dat')['tt'][:]
cllentt = camb.spectra_fromcambfile('/Users/jcarron/SpyderProjects/jpipe/inputs/FFP10/FFP10_lensedCls.dat')['tt'][:]
clunlpp = camb.spectra_fromcambfile('/Users/jcarron/SpyderProjects/jpipe/inputs/FFP10/FFP10_lenspotentialCls.dat')['pp'][:]
dlm = hp.synalm(clunlpp[:lmax + dlmax + 1],verbose = True)
hp.almxfl(dlm,np.sqrt(np.arange(6001,dtype = float)*np.arange(1,6002)),inplace=True)
unltlm = hp.synalm(clunltt[:lmax + dlmax  + 1],verbose = True)
tlm2,glm,ret = li.curvedskylensing.lenscurv.lens_glm_GLth_sym_timed(0,dlm,-unltlm,lmax_target + dlmax,facres = facres)
del clm
pl.plot(hp.alm2cl(tlm2)[:lmax_target+ dlmax+ 1]/cllentt[:lmax_target+ dlmax+ 1])
pl.ylim(0.9,1.1)
pl.axhline(0.99,color = 'black')


"""
"""

#Testing the thinned routine
lmax = 4096
lmax_target = 4096
dlmax = 0
facres = 0
import lensit as li,jc_camb as camb
pl.ion()
clunltt = camb.spectra_fromcambfile('/Users/jcarron/SpyderProjects/jpipe/inputs/FFP10/FFP10_lenspotentialCls.dat')['tt'][:]
cllentt = camb.spectra_fromcambfile('/Users/jcarron/SpyderProjects/jpipe/inputs/FFP10/FFP10_lensedCls.dat')['tt'][:]
clunlpp = camb.spectra_fromcambfile('/Users/jcarron/SpyderProjects/jpipe/inputs/FFP10/FFP10_lenspotentialCls.dat')['pp'][:]
dlm = hp.synalm(clunlpp[:lmax + dlmax + 1],verbose = True)
hp.almxfl(dlm,np.sqrt(np.arange(6001,dtype = float)*np.arange(1,6002)),inplace=True)
dlm *= 0
unltlm = hp.synalm(clunltt[:lmax + dlmax  + 1],verbose = True)
%time tlm2,clm,ret = li.curvedskylensing.lenscurv.lens_glm_GLth_sym_timed(0,dlm,-unltlm,lmax_target,facres = facres)
%time tlm3,clm,ret = li.curvedskylensing.lenscurv.lens_glm_GL_sym_timed(0,dlm,-unltlm,lmax_target,facres = facres)

del clm
pl.plot(hp.alm2cl(tlm2)[:lmax_target+ 1]/clunltt[:lmax_target+ 1])
pl.axhline(0.99,color = 'black')
"""


"""

import jc_camb as camb
import lensit as li
clunltt = camb.spectra_fromcambfile('/Users/jcarron/SpyderProjects/jpipe/inputs/FFP10/FFP10_lenspotentialCls.dat')['tt'][:]
tlm = hp.synalm(clunltt[:200])
from lensit import shts
from lensit.misc import gausslegendre
xg,wg = gausslegendre.get_xgwg(500)
t = np.arccos(xg)[::-1]
vlm = shts.util.alm2vlm(tlm)
vtm = shts.vlm2vtm(0,t,vlm)
alm2 = -shts.util.vlm2alm(shts.vtm2vlm(0,t,vtm))[0]
alm3 = shts.vtm2tlm_sym(t,vtm)
np.max(np.abs(alm2 - alm3))
"""