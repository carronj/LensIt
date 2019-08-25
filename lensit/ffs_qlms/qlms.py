"""
(B^t F^t Cov^-1 d)^a(z) (D dxi_unl/da D^t B^t F^t Cov^-1 d)_a(z)
Only lib_skys enter this. Sign is correct for pot. estimate, not gradient.
This can written as (D_f (Res lms))(z) (D_f P_a Res lms)(z) * |M_f|(z)

Similarly the mean field can be written as the diagonal
    |M_f|(z) (i k_a P D^t B^t Covi B D)(f(z),f(z))
=   |M_f|(z) (i k_a (Pi + D^tB^tNiBD)^{-1}P^{-1})(f(z),f(z))


"""
from __future__ import print_function
import numpy as np

from lensit.misc.misc_utils import timer
from lensit.ffs_deflect.ffs_deflect import ffs_id_displacement
from lensit.ffs_covs import ffs_specmat as SM

verbose = False
typs = ['T', 'QU', 'TQU']


def get_qlms_wl(typ, lib_sky, TQU_Mlik, ResTQU_Mlik, lib_qlm, f=None,lib_sky2 =None, use_Pool=0, **kwargs):
    """
    Stand alone qlm estimator starting from lib_sky and unlensed Cls
    Likelihood gradient (from the quadratic part).
    (B^t F^t Cov^-1 d)^a(z) (D dxi_unl/da D^t B^t F^t Cov^-1 d)_a(z)
    Only lib_skys enter this.
    Sign is correct for pot. estimate, not gradient.

    This can written as (D_f (Res lms))(z) (D_f P_a Res lms)(z) * |M_f|(z)
    Only forward displacement is needed.

    Res lms is here D^t B^t Cov^-1 data. This can be written in general
    as  D^t B^t Ni (data - B D MLIK(data)) in T E B space. For non-singular modes
    this may be written as P_TEB^{-1} MLIK. (but we can't use pseudo inverse
    for singular modes.)
    P_a Res lms are always the max. likelihood modes however.
    N0  * output is normalized qest for MV estimates

    1/2 (VX WY  +  VY WX)
    1/2 VX WY  +  VY WX

    1/4 (VVt WW^t + VVt WWt + WV^t VW^t + V W^t WV^t)

    We can get something without having to lens any weird maps through
    ( B^t Ni (data - B D Xmap))(z)    (D ika Xmap)(z)
    """
    lib_sky2 = lib_sky if lib_sky2 is None else lib_sky
    if typ in ['EE','EB','BE','BB']:
        TEB_Mlik = lib_sky.QUlms2EBalms(TQU_Mlik)
        TEB_Res = lib_sky.QUlms2EBalms(ResTQU_Mlik)
        TEB_Mlik[{'E':1,'B':0}[typ[0]]] *= 0.
        TEB_Res[{'E':1,'B':0}[typ[1]]] *= 0.
        return get_qlms_wl('QU',lib_sky,lib_sky.EBlms2QUalms(TEB_Mlik),lib_sky2.EBlms2QUalms(TEB_Res),lib_qlm,
                           f = f,use_Pool=use_Pool,lib_sky2 = lib_sky2)

    assert len(TQU_Mlik) == len(typ) and len(ResTQU_Mlik) == len(typ)
    t = timer(verbose, prefix=__name__)
    if f is None: f = ffs_id_displacement(lib_sky.shape, lib_sky.lsides)

    def left(id):
        assert id in range(len(typ)), (id, typ)
        return lib_sky.alm2map(ResTQU_Mlik[id])

    def Right(S_id, axis):
        assert S_id in range(len(typ)), (S_id, typ)
        assert axis in [0, 1]
        kfunc = lib_sky2.get_ikx if axis == 1 else lib_sky2.get_iky
        return f.alm2lenmap(lib_sky2, TQU_Mlik[S_id] * kfunc(), use_Pool=use_Pool)

    retdx = left(0) * Right(0, 1)
    for _i in range(1, len(typ)): retdx += left(_i) * Right(_i, 1)
    retdx = lib_qlm.map2alm(retdx)
    t.checkpoint("get_likgrad::Cart. gr. x done. (%s map(s) lensed, %s fft(s)) " % (len(typ), 2 * len(typ) + 1))

    retdy = left(0) * Right(0, 0)
    for _i in range(1, len(typ)): retdy += left(_i) * Right(_i, 0)
    retdy = lib_qlm.map2alm(retdy)
    t.checkpoint("get_likgrad::Cart. gr. y done. (%s map(s) lensed, %s fft(s)) " % (len(typ), 2 * len(typ) + 1))

    return np.array([- retdx * lib_qlm.get_ikx() - retdy * lib_qlm.get_iky(),
                       retdx * lib_qlm.get_iky() - retdy * lib_qlm.get_ikx()])  # N0  * output is normalized qest


def _Mlik2ResTQUMlik_diag(field, ninv_filt, TQUMlik, data, f, fi):
    """
    Produces B^t Ni (data - B D Mlik) in TQU space,
    that is fed into the qlm estimator.
    """
    assert field in ['T', 'Q', 'U']
    f_id = ffs_id_displacement(ninv_filt.lib_skyalm.shape, ninv_filt.lib_skyalm.lsides)
    ninv_filt.set_ffi(f, fi)
    _map = data - ninv_filt.apply_R(field, TQUMlik)
    ninv_filt.apply_map(f, _map, inplace=True)
    ninv_filt.set_ffi(f_id, f_id)
    return ninv_filt.apply_Rt(field, _map)

def get_response(typ,lib_datalm,cls_len,NlevT_uKamin,NlevP_uKamin,cl_transf,
                 wAl = None,wBl = None,fAl = None,fBl = None,lib_qlm = None):
    """ Q. estimator response """
    assert typ[0] in ['T','E','B'] and typ[1] in ['T','E','B']
    assert typ[0] in ['E','B'] and typ[1] in ['E','B'], "T not implemented"
    assert 'eb' not in cls_len.keys() and 'be' not in cls_len.keys()
    assert 'tb' not in cls_len.keys() and 'bt' not in cls_len.keys()

    lmax = lib_datalm.ellmax
    if wAl is None: wAl = np.ones(lmax + 1,dtype = float)
    if wBl is None: wBl = cls_len[(typ[1] + typ[1]).lower()][:lmax + 1]
    if fAl is None:
        Nlev = NlevT_uKamin if typ[0] == 'T' else NlevP_uKamin
        ii = np.where(cl_transf[:lmax + 1] > 0.)
        fAl = np.zeros(lmax + 1,dtype = float)
        fAl[ii] = 1./ (cls_len[(typ[0] + typ[0]).lower()][ii] + ( (Nlev / 60. /180. * np.pi)/ cl_transf[ii]) ** 2)
    if fBl is None:
        Nlev = NlevT_uKamin if typ[1] == 'T' else NlevP_uKamin
        ii = np.where(cl_transf[:lmax + 1] > 0.)
        fBl = np.zeros(lmax + 1,dtype = float)
        fBl[ii] = 1./ (cls_len[(typ[1] + typ[1]).lower()][ii] + ( (Nlev / 60. /180. * np.pi)/ cl_transf[ii]) ** 2)

    if lib_qlm is None: lib_qlm = lib_datalm

    def get_pmat(A, i, j, clA):
        if A == 'T':
            if i == 0 and j == 0:
                return clA[lib_datalm.reduced_ellmat()]
            else:
                assert 0,('zero',i,j)
        elif A == 'E':
            cos, sin = lib_datalm.get_cossin_2iphi()
            if i == 1 and j == 1:
                return clA[lib_datalm.reduced_ellmat()] * cos ** 2
            elif i == 2 and j == 2:
                return clA[lib_datalm.reduced_ellmat()] * sin ** 2
            elif i == 2 and j == 1:
                return clA[lib_datalm.reduced_ellmat()] * cos * sin
            elif i == 1 and j == 2:
                return clA[lib_datalm.reduced_ellmat()] * cos * sin
            else:
                assert 0,('zero',i,j)
        elif A == 'B':
            cos, sin = lib_datalm.get_cossin_2iphi()
            if i == 1 and j == 1:
                return clA[lib_datalm.reduced_ellmat()] * sin ** 2
            elif i == 2 and j == 2:
                return clA[lib_datalm.reduced_ellmat()] * cos ** 2
            elif i == 1 and j == 2:
                return -clA[lib_datalm.reduced_ellmat()] * cos * sin
            elif i == 2 and j == 1:
                return -clA[lib_datalm.reduced_ellmat()] * cos * sin
            else:
                assert 0,('zero',i,j)
        else:
            assert 0,(A,['T','E','B'])
    retxx = np.zeros(lib_datalm.shape,dtype = float)
    retyy = np.zeros(lib_datalm.shape,dtype = float)
    retxy = np.zeros(lib_datalm.shape,dtype = float)
    retyx = np.zeros(lib_datalm.shape,dtype = float)

    _2map = lambda alm : lib_datalm.alm2map(alm)
    ikx = lambda : lib_datalm.get_ikx()
    iky = lambda: lib_datalm.get_iky()

    clB = wBl * fBl * cls_len[(typ[1] + typ[1]).lower()][:lmax + 1]
    clA = wAl * fAl
    for i, j in [(1, 1),(1, 2),(2, 1),(2, 2)]:
        retxx += _2map(get_pmat(typ[0],i,j, clA))  *  _2map(ikx() ** 2 * get_pmat(typ[1],j,i,clB ))
        retyy += _2map(get_pmat(typ[0],i,j, clA))  *  _2map(iky() ** 2 * get_pmat(typ[1],j,i,clB ))
        retxy += _2map(get_pmat(typ[0],i,j, clA))  *  _2map(ikx() * iky() * get_pmat(typ[1],j,i,clB ))
        retyx += _2map(get_pmat(typ[0],i,j, clA))  *  _2map(ikx() * iky() * get_pmat(typ[1],j,i,clB ))

    clB = wBl * fBl
    clA = wAl * fAl * cls_len[(typ[0] + typ[0]).lower()][:lmax + 1]
    for i, j in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        retxx += _2map(ikx() * get_pmat(typ[0], i, j, clA)) * _2map(ikx() * get_pmat(typ[1], j, i, clB))
        retyy += _2map(iky() * get_pmat(typ[0], i, j, clA)) * _2map(iky() * get_pmat(typ[1], j, i, clB))
        retxy += _2map(ikx() * get_pmat(typ[0], i, j, clA)) * _2map(iky() * get_pmat(typ[1], j, i, clB))
        retyx += _2map(iky() * get_pmat(typ[0], i, j, clA)) * _2map(ikx() * get_pmat(typ[1], j, i, clB))
    fac = 1. / np.sqrt(np.prod(lib_datalm.lsides))
    _2alm = lambda _map : lib_qlm.map2alm(_map)
    retxx = _2alm(retxx)
    retyy = _2alm(retyy)
    retxy = _2alm(retxy)
    retyx = _2alm(retyx)
    ikx = lambda : lib_qlm.get_ikx()
    iky = lambda : lib_qlm.get_iky()
    return  np.array([fac * (retxx * ikx() ** 2 + retyy * iky() ** 2 + (retxy + retyx) * ikx() * iky()),
             fac * (retxx * iky() ** 2 + retyy * ikx() ** 2 - (retxy + retyx) * ikx() * iky()) ])

class MFestimator:
    def __init__(self, ninv_filt, opfilt, mchain, lib_qlm, pix_pha=None, cmb_pha=None, use_Pool=0):

        self.ninv_filt = ninv_filt
        self.opfilt = opfilt
        self.mchain = mchain
        self.lib_qlm = lib_qlm
        self.pix_pha = pix_pha
        self.cmb_pha = cmb_pha
        self.use_Pool = use_Pool

    def npix(self):
        return self.ninv_filt.npix

    def get_MFqlms(self, typ, MFkey, idx, soltn=None):
        lib_sky = self.ninv_filt.lib_skyalm
        lib_dat = self.ninv_filt.lib_datalm
        assert lib_sky.lsides == lib_dat.lsides
        self.opfilt.typ = typ
        if hasattr(self.ninv_filt, 'f'):
            print("******* I am using displacement for ninvfilt in MFest")
        else:
            print("******* Using id displacement in MFest")
        f = getattr(self.ninv_filt, 'f', ffs_id_displacement(lib_sky.shape, lib_sky.lsides))
        if MFkey == 12:
            # B^t M^t X (x) (D ika P D^t B^t Covi X )(x). Second term are just the deflected gradients of the recontructed
            assert self.pix_pha is not None
            if soltn is None:
                soltn = np.zeros((self.opfilt.TEBlen(typ), self.ninv_filt.lib_skyalm.alm_size), dtype=complex)
            phas = self.pix_pha.get_sim(idx)[0:len(typ)]
            for i, _f in enumerate(typ): phas[i] *= self.ninv_filt.get_mask(_f.lower())
            self.mchain.solve(soltn, phas, finiop='MLIK')
            TQUMlik = self.opfilt.soltn2TQUMlik(soltn, self.ninv_filt)
            norm = np.prod(lib_dat.shape) / (np.prod(lib_dat.lsides))

            def Left(id):
                _alm = lib_sky.udgrade(lib_dat, lib_dat.map2alm(phas[id]))
                return lib_sky.alm2map(lib_sky.almxfl(_alm, norm * self.ninv_filt.cl_transf))

            def Right(id, ax):
                assert ax in [0, 1], ax
                kfunc = lib_sky.get_ikx if ax == 1 else lib_sky.get_iky
                return f.alm2lenmap(lib_sky, TQUMlik[id] * kfunc(), use_Pool=self.use_Pool)
        elif MFkey == 2:
            # X unit variance random phases dat map shaped
            # X (x) (D ika P D^t B^t Covi B X )(x). Second term are just the deflected gradients of the recontructed
            # B X given in input

            norm = np.prod(lib_dat.shape) / (np.prod(lib_sky.lsides))
            phas = self.pix_pha.get_sim(idx)[0:len(typ)]
            if soltn is None:
                soltn = np.zeros((self.opfilt.TEBlen(typ), self.ninv_filt.lib_skyalm.alm_size), dtype=complex)
            inp = np.array([lib_sky.almxfl(lib_sky.map2alm(_p), self.ninv_filt.cl_transf) for _p in phas])
            inp = np.array([lib_dat.alm2map(lib_dat.udgrade(lib_sky, _p)) * self.ninv_filt.get_mask(_f) for _p, _f in
                            zip(inp, typ)])
            self.mchain.solve(soltn, inp, finiop='MLIK')
            soltn = self.opfilt.soltn2TQUMlik(soltn, self.ninv_filt)

            def Left(id):
                return phas[id]

            def Right(id, ax):
                assert ax in [0, 1], ax
                kfunc = lib_sky.get_ikx if ax == 1 else lib_sky.get_iky
                return f.alm2lenmap(lib_sky, norm * soltn[id] * kfunc(), use_Pool=self.use_Pool)
        elif MFkey == 22:
            # D ika b X (x) (B^t Covi B D P 1/b X )(x). TEB phas
            assert 0
        else:
            assert 'not implemented'
        retdx = Left(0) * Right(0, 1)
        for i in range(1, len(typ)): retdx += Left(i) * Right(i, 1)
        retdx = self.lib_qlm.map2alm(retdx)
        retdy = Left(0) * Right(0, 0)
        for i in range(1, len(typ)): retdy += Left(i) * Right(i, 0)
        retdy = self.lib_qlm.map2alm(retdy)
        return np.array([- retdx * self.lib_qlm.get_ikx() - retdy * self.lib_qlm.get_iky(),
                         retdx * self.lib_qlm.get_iky() - retdy * self.lib_qlm.get_ikx()])  # N0  * output is normalized qest


def get_MFqlms(typ, MFkey, lib_dat, lib_sky, pix_phas, TQUMlik_pha, cl_transf, lib_qlm, f=None, use_Pool=0):
    assert lib_dat.lsides == lib_sky.lsides
    if f is None: f = ffs_id_displacement(lib_sky.shape, lib_sky.lsides)
    if MFkey == 12:
        # X unit variance random phases ! on the unmasked pixels !
        # B^t M^t X (x) (D ika P D^t B^t Covi X )(x). Second term are just the deflected gradients of the recontructed
        # X
        norm = np.prod(lib_dat.shape) / (np.prod(lib_dat.lsides))

        def Left(id):
            _alm = lib_sky.udgrade(lib_dat, lib_dat.map2alm(pix_phas[id]))
            return lib_sky.alm2map(lib_sky.almxfl(_alm, norm * cl_transf))

        def Right(id, ax):
            assert ax in [0, 1], ax
            kfunc = lib_sky.get_ikx if ax == 1 else lib_sky.get_iky
            return f.alm2lenmap(lib_sky, TQUMlik_pha[id] * kfunc(), use_Pool=use_Pool)
    elif MFkey == 2:
        # X unit variance random phases dat map shaped
        # X (x) (D ika P D^t B^t Covi B X )(x). Second term are just the deflected gradients of the recontructed
        # B X given in input

        norm = np.prod(lib_dat.shape) / (np.prod(lib_sky.lsides))

        def Left(id):
            return pix_phas[id]

        def Right(id, ax):
            assert ax in [0, 1], ax
            kfunc = lib_sky.get_ikx if ax == 1 else lib_sky.get_iky
            return f.alm2lenmap(lib_sky, norm * TQUMlik_pha[id] * kfunc(), use_Pool=use_Pool)
    elif MFkey == 22:
        # FIXME : need TEB pha
        # X unit variance TEB sky-shaped.
        # D ika b X (x) (B^t Covi B D P 1/b X )(x). Second term given by pix_pha
        # X TEB shap
        # TQU_mlik must be TQUskylm shaped for this.
        # pix_phas is TQU dat-shaped and the solution of B^t Covi B D P 1/b X

        norm = 1.
        assert np.all([_m.shape == lib_dat.shape for _m in pix_phas])
        assert np.all([_m.size == lib_sky.alm_size for _m in TQUMlik_pha])

        def Left(id):
            return pix_phas[id]

        def Right(id, ax):
            assert ax in [0, 1], ax
            kfunc = lib_sky.get_ikx if ax == 1 else lib_sky.get_iky
            return f.alm2lenmap(lib_sky, lib_sky.almxfl(TQUMlik_pha[id] * kfunc(), cl_transf * norm), use_Pool=use_Pool)
    else:
        assert 0, 'not implemented'
    retdx = Left(0) * Right(0, 1)
    for i in range(1, len(typ)): retdx += Left(i) * Right(i, 1)
    retdx = lib_qlm.map2alm(retdx)
    retdy = Left(0) * Right(0, 0)
    for i in range(1, len(typ)): retdy += Left(i) * Right(i, 0)
    retdy = lib_qlm.map2alm(retdy)
    return np.array([- retdx * lib_qlm.get_ikx() - retdy * lib_qlm.get_iky(),
                     retdx * lib_qlm.get_iky() - retdy * lib_qlm.get_ikx()])  # N0  * output is unnormalized qest


def get_qlms(typ, lib_sky, Res_TEBlms, cls_unl, lib_qlm, Res_TEBlms2=None, f=None, use_Pool=0, **kwargs):
    # FIXME : Seems to work but D_f to Reslm is a super small scale map in close to in configuration with little noise.
    # FIXME : The map B^t Covi d has spec 1 / (P + N/B^2) which can peak at a farily small scale.
    # FIXME there is probably a better way.
    """
    Stand alone qlm estimator starting from lib_sky and unlensed Cls
    Likelihood gradient (from the quadratic part).
    (B^t F^t Cov^-1 d)^a(z) (D dxi_unl/da D^t B^t F^t Cov^-1 d)_a(z)
    Only lib_skys enter this.
    Sign is correct for pot. estimate, not gradient.

    This can written as (D_f (Res lms))(z) (D_f P_a Res lms)(z) * |M_f|(z)
    Only forward displacement is needed.

    Res lms is here D^t B^t Cov^-1 data. This can be written in general
    as  D^t B^t Ni (data - B D MLIK(data)) in T E B space. For non-singular modes
    this may be written as P_TEB^{-1} MLIK. (but we can't use pseudo inverse
    for singular modes.)
    P_a Res lms are always the max. likelihood modes however.
    N0  * output is normalized qest for MV estimates

    1/2 (VX WY  +  VY WX)
    1/2 VX WY  +  VY WX

    1/4 (VVt WW^t + VVt WWt + WV^t VW^t + V W^t WV^t)

    We can get something without having to lens any weird maps through
    ( B^t Ni (data - B D Xmap))(z)    (D Xmap)(z)
    """
    _Res_TEBlms2 = Res_TEBlms if Res_TEBlms2 is None else Res_TEBlms2
    assert len(Res_TEBlms) == len(typ) and len(_Res_TEBlms2) == len(typ)
    t = timer(verbose, prefix=__name__)
    if f is not None: print(" qlms.py :: consider using get_qlms_wl for qlms with lensing, to avoid lensing noisy maps")
    if f is None: f = ffs_id_displacement(lib_sky.shape, lib_sky.lsides)

    TQUmlik = SM.TEB2TQUlms(typ, lib_sky, SM.apply_TEBmat(typ, lib_sky, cls_unl, _Res_TEBlms2))

    def left(S_id):
        assert S_id in range(len(typ)), (S_id, typ)
        return f.alm2lenmap(lib_sky, SM.get_SlmfromTEBlms(typ, lib_sky, Res_TEBlms, typ[S_id]), use_Pool=use_Pool)

    def Right(S_id, axis):
        assert S_id in range(len(typ)), (S_id, typ)
        assert axis in [0, 1]
        kfunc = lib_sky.get_ikx if axis == 1 else lib_sky.get_iky
        return f.alm2lenmap(lib_sky, TQUmlik[S_id] * kfunc(), use_Pool=use_Pool)

    retdx = left(0) * Right(0, 1)
    for _i in range(1, len(typ)): retdx += left(_i) * Right(_i, 1)
    retdx = lib_qlm.map2alm(f.mult_wmagn(retdx))
    t.checkpoint("get_likgrad::Cart. gr. x done. (%s map(s) lensed, %s fft(s)) " % (2 * len(typ), 2 * len(typ) + 1))

    retdy = left(0) * Right(0, 0)
    for _i in range(1, len(typ)): retdy += left(_i) * Right(_i, 0)
    retdy = lib_qlm.map2alm(f.mult_wmagn(retdy))
    t.checkpoint("get_likgrad::Cart. gr. y done. (%s map(s) lensed, %s fft(s)) " % (2 * len(typ), 2 * len(typ) + 1))

    return np.array([- retdx * lib_qlm.get_ikx() - retdy * lib_qlm.get_iky(),
                     retdx * lib_qlm.get_iky() - retdy * lib_qlm.get_ikx()])  # N0  * output is normalized qest


def get_response_flexible(lib_tlm, lib_elm, lib_blm, cls, cls_transf, cls_noise, lib_qlm, isoN0s = True):
        """
        N0 calc, allowing for abritrary aniso filtering.
        -(xi,a K) (xi,b K) - (K)ab (xi,a K xi,b) with K = B^t Covi B
        """
        assert lib_tlm.ell_mat == lib_elm.ell_mat and lib_tlm.ell_mat == lib_blm.ell_mat
        assert 'tt' in cls_noise.keys() and 'ee' in cls_noise.keys() and 'bb' in cls_noise.keys()
        ellmat = lib_tlm.ell_mat
        lmax = max(lib_tlm.ellmax,lib_elm.ellmax,lib_blm.ellmax)
        Ki_cls  = {}
        w_cls = {}
        t_cls = {}
        for k in ['tt','ee','te','bb']:
            Ki_cls[k] = np.zeros(ellmat.ellmax + 1,dtype = float)
            Ki_cls[k][:lmax + 1] = cls[k][:lmax + 1] * cls_transf[k[0]][:lmax + 1] * cls_transf[k[1]][:lmax + 1]
            if k in cls_noise.keys():  Ki_cls[k][:lmax + 1]  += cls_noise[k][:lmax + 1]
            w_cls[k] = np.zeros(ellmat.ellmax + 1,dtype = float)
            w_cls[k][:lmax + 1] = cls[k][:lmax + 1]
        for k in ['t','e','b']:
            t_cls[k] =  np.zeros(ellmat.ellmax + 1,dtype = float)
            t_cls[k][:lmax + 1] = np.copy(cls_transf[k][:lmax + 1])
        if lib_tlm.ellmax > 0: t_cls['t'][:lib_tlm.ellmax + 1] *= (lib_tlm.get_Nell() > 0)
        if lib_elm.ellmax > 0: t_cls['e'][:lib_elm.ellmax + 1] *= (lib_elm.get_Nell() > 0)
        if lib_blm.ellmax > 0: t_cls['b'][:lib_blm.ellmax + 1] *= (lib_blm.get_Nell() > 0)

        K_almmap = np.zeros((ellmat.rshape[0],ellmat.rshape[1],3,3),dtype = float)
        K_almmap[:,:,0, 0] = Ki_cls['tt'][ellmat()] * lib_tlm._cond()
        K_almmap[:,:,1, 1] = Ki_cls['ee'][ellmat()] * lib_elm._cond()
        K_almmap[:,:,2, 2] = Ki_cls['bb'][ellmat()] * lib_blm._cond()
        K_almmap[:,:,0, 1] = Ki_cls['te'][ellmat()] * lib_tlm._cond()* lib_elm._cond()
        K_almmap[:,:,1, 0] = Ki_cls['te'][ellmat()] * lib_tlm._cond()* lib_elm._cond()
        if np.__version__ >= '1.14':
            K_almmap = np.linalg.pinv(K_almmap) # B^t Covi B
        else:
            for l1 in range(ellmat.rshape[0]):
                for l2 in range(ellmat.rshape[1]):
                    K_almmap[l1,l2,:,:] = np.linalg.pinv(K_almmap[l1,l2,:,:])
        K_almmap[:, :, 0, 0] *= t_cls['t'][ellmat()] ** 2
        K_almmap[:, :, 1, 1] *= t_cls['e'][ellmat()] ** 2
        K_almmap[:, :, 0, 1] *= t_cls['t'][ellmat()] * t_cls['e'][ellmat()]
        K_almmap[:, :, 1, 0] *= t_cls['e'][ellmat()] * t_cls['t'][ellmat()]
        K_almmap[:, :, 2, 2] *= t_cls['b'][ellmat()] ** 2

        xiK_almmap = np.zeros((ellmat.rshape[0],ellmat.rshape[1],3,3),dtype = float)
        xiK_almmap[:, :, 0, 0] = w_cls['tt'][ellmat()] * K_almmap[:, :, 0, 0] + w_cls['te'][ellmat()] * K_almmap[:, :, 1, 0]
        xiK_almmap[:, :, 1, 1] = w_cls['te'][ellmat()] * K_almmap[:, :, 0, 1] + w_cls['ee'][ellmat()] * K_almmap[:, :, 1, 1]
        xiK_almmap[:, :, 2, 2] = w_cls['bb'][ellmat()] * K_almmap[:, :, 2, 2]
        xiK_almmap[:, :, 0, 1] = w_cls['tt'][ellmat()] * K_almmap[:, :, 0, 1] + w_cls['te'][ellmat()] * K_almmap[:, :, 1, 1]
        xiK_almmap[:, :, 1, 0] = w_cls['te'][ellmat()] * K_almmap[:, :, 0, 0] + w_cls['ee'][ellmat()] * K_almmap[:, :, 1, 0]

        xiKxi_almmap = np.zeros((ellmat.rshape[0],ellmat.rshape[1],3,3),dtype = float)
        xiKxi_almmap[:, :, 0, 0] = w_cls['tt'][ellmat()] * xiK_almmap[:, :, 0, 0] + w_cls['te'][ellmat()] * xiK_almmap[:, :, 0, 1]
        xiKxi_almmap[:, :, 1, 1] = w_cls['te'][ellmat()] * xiK_almmap[:, :, 1, 0] + w_cls['ee'][ellmat()] * xiK_almmap[:, :, 1, 1]
        xiKxi_almmap[:, :, 2, 2] = w_cls['bb'][ellmat()] * xiK_almmap[:, :, 2, 2]
        xiKxi_almmap[:, :, 0, 1] = w_cls['ee'][ellmat()] * xiK_almmap[:, :, 0, 1] + w_cls['te'][ellmat()] * xiK_almmap[:, :, 0, 0]
        xiKxi_almmap[:, :, 1, 0] = w_cls['tt'][ellmat()] * xiK_almmap[:, :, 1, 0] + w_cls['te'][ellmat()] * xiK_almmap[:, :, 1, 1]
        cos,sin = ellmat.get_cossin_2iphi_mat()

        def apply_RSX(almmap,iS,iX):
            """ T Q U = (1 0 0
                         0 c  -s
                        0 s c)    T E B"""
            if iS == 0:
                return almmap.copy() if iX == 0 else np.zeros_like(almmap)
            elif iS == 1:
                if iX == 1:
                    return cos * almmap
                elif iX == 2:
                    return -sin * almmap
                else :
                    return np.zeros_like(almmap)
            elif iS == 2:
                if iX == 1:
                    return sin * almmap
                elif iX == 2:
                    return cos * almmap
                else :
                    return np.zeros_like(almmap)
            else:
                assert 0

        def TEB2TQU(iS,jS,TEBmat):
            """ R_sx R_ty Y Pxy """
            assert TEBmat.shape == (ellmat.rshape[0],ellmat.rshape[1],3,3)
            ret = np.zeros(ellmat.rshape)
            for iX in range(3):
                for jX in range(3):
                    ret += apply_RSX(apply_RSX(TEBmat[:,:,iX,jX],iS,iX),jS,jX)
            return ret
        # turn TEB to TQU:
        xiK = np.zeros_like(xiK_almmap)
        for iS in range(3):
            for jS in range(3):
                xiK[:,:,iS,jS] = TEB2TQU(iS,jS,xiK_almmap)
        del xiK_almmap

        xiKxi = np.zeros_like(xiKxi_almmap)
        for iS in range(3):
            for jS in range(3):
                xiKxi[:, :, iS, jS] = TEB2TQU(iS, jS, xiKxi_almmap)
        del xiKxi_almmap

        K = np.zeros_like(K_almmap)
        for iS in range(3):
            for jS in range(3):
                K[:, :, iS, jS] = TEB2TQU(iS, jS, K_almmap)
        del K_almmap

        Fxx = np.zeros(ellmat.shape, dtype=float)
        Fyy = np.zeros(ellmat.shape, dtype=float)
        Fxy = np.zeros(ellmat.shape, dtype=float)
        Fyx = np.zeros(ellmat.shape, dtype=float)

        _2map = lambda almmap:np.fft.irfft2(almmap.astype(complex))
        ikx = lambda :ellmat.get_ikx_mat()
        iky = lambda: ellmat.get_iky_mat()

        #-(xi, a K)(xi, b K) - (K)(xi, a  K xi, b)
        for iS in range(3):
            for jS in range(3):
                Fxx += _2map(xiK[:,:,iS,jS] * ikx()) * _2map(xiK[:,:,jS,iS] * ikx())
                Fyy += _2map(xiK[:,:,iS,jS] * iky()) * _2map(xiK[:,:,jS,iS] * iky())
                Fxy += _2map(xiK[:,:,iS,jS] * ikx()) * _2map(xiK[:,:,jS,iS] * iky())
                Fyx += _2map(xiK[:,:,iS,jS] * iky()) * _2map(xiK[:,:,jS,iS] * ikx())

                Fxx += _2map(K[:, :, iS, jS]) * _2map(xiKxi[:, :, jS, iS] * ikx() * ikx())
                Fyy += _2map(K[:, :, iS, jS]) * _2map(xiKxi[:, :, jS, iS] * iky() * iky())
                Fxy += _2map(K[:, :, iS, jS]) * _2map(xiKxi[:, :, jS, iS] * ikx() * iky())
                Fyx += _2map(K[:, :, iS, jS]) * _2map(xiKxi[:, :, jS, iS] * iky() * ikx())

        fac = 1. / np.sqrt(np.prod(ellmat.lsides)) * lib_tlm.fac_alm2rfft ** 2
        Fxx = lib_qlm.map2alm(Fxx)
        Fyy = lib_qlm.map2alm(Fyy)
        Fxy = lib_qlm.map2alm(Fxy)
        Fyx = lib_qlm.map2alm(Fyx)
        ikx = lambda : lib_qlm.get_ikx()
        iky = lambda : lib_qlm.get_iky()
        assert isoN0s,'implement this (non anisotropic N0 2d cls)' #this affects only the following line:
        return (fac*lib_qlm.bin_realpart_inell(Fxx * ikx() ** 2 + Fyy * iky() ** 2 + (Fxy + Fyx) * ikx() * iky()),
                fac * lib_qlm.bin_realpart_inell( (Fxx * iky() ** 2 + Fyy * ikx() ** 2 - (Fxy + Fyx) * ikx() * iky())))