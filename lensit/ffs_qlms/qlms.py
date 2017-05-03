"""
(B^t F^t Cov^-1 d)^a(z) (D dxi_unl/da D^t B^t F^t Cov^-1 d)_a(z)
Only lib_skys enter this. Sign is correct for pot. estimate, not gradient.
This can written as (D_f (Res lms))(z) (D_f P_a Res lms)(z) * |M_f|(z)

Similarly the mean field can be written as the diagonal
    |M_f|(z) (i k_a P D^t B^t Covi B D)(f(z),f(z))
=   |M_f|(z) (i k_a (Pi + D^tB^tNiBD)^{-1}P^{-1})(f(z),f(z))


"""
import healpy as hp
import numpy as np

import lensit as fs
from lensit.ffs_covs import ffs_specmat as SM

verbose = True
_types = ['T', 'QU', 'TQU']


def get_qlms_wl(_type, lib_sky, TQU_Mlik, ResTQU_Mlik, lib_qlm, f=None, use_Pool=0, **kwargs):
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
    assert len(TQU_Mlik) == len(_type) and len(ResTQU_Mlik) == len(_type)
    t = fs.misc.misc_utils.timer(verbose, prefix=__name__)
    if f is None: f = fs.ffs_deflect.ffs_deflect.ffs_id_displacement(lib_sky.shape, lib_sky.lsides)

    def left(id):
        assert id in range(len(_type)), (id, _type)
        return lib_sky.alm2map(ResTQU_Mlik[id])

    def Right(S_id, axis):
        assert S_id in range(len(_type)), (S_id, _type)
        assert axis in [0, 1]
        kfunc = lib_sky.get_ikx if axis == 1 else lib_sky.get_iky
        return f.alm2lenmap(lib_sky, TQU_Mlik[S_id] * kfunc(), use_Pool=use_Pool)

    retdx = left(0) * Right(0, 1)
    for _i in range(1, len(_type)): retdx += left(_i) * Right(_i, 1)
    retdx = lib_qlm.map2alm(retdx)
    t.checkpoint("get_likgrad::Cart. gr. x done. (%s map(s) lensed, %s fft(s)) " % (len(_type), 2 * len(_type) + 1))

    retdy = left(0) * Right(0, 0)
    for _i in range(1, len(_type)): retdy += left(_i) * Right(_i, 0)
    retdy = lib_qlm.map2alm(retdy)
    t.checkpoint("get_likgrad::Cart. gr. y done. (%s map(s) lensed, %s fft(s)) " % (len(_type), 2 * len(_type) + 1))

    return np.array([- retdx * lib_qlm.get_ikx() - retdy * lib_qlm.get_iky(),
                     retdx * lib_qlm.get_iky() - retdy * lib_qlm.get_ikx()])  # N0  * output is normalized qest


def _Mlik2ResTQUMlik_diag(field, ninv_filt, TQUMlik, data, f, fi):
    """
    Produces B^t Ni (data - B D Mlik) in TQU space,
    that is fed into the qlm estimator.
    """
    assert field in ['T', 'Q', 'U']
    f_id = fs.ffs_deflect.ffs_deflect.ffs_id_displacement(ninv_filt.lib_skyalm.shape, ninv_filt.lib_skyalm.lsides)
    ninv_filt.set_ffi(f, fi)
    _map = data - ninv_filt.apply_R(field, TQUMlik)
    ninv_filt.apply_map(f, _map, inplace=True)
    ninv_filt.set_ffi(f_id, f_id)
    return ninv_filt.apply_Rt(field, _map)


class MFestimator():
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

    def get_MFqlms(self, _type, MFkey, idx, soltn=None):
        lib_sky = self.ninv_filt.lib_skyalm
        lib_dat = self.ninv_filt.lib_datalm
        assert lib_sky.lsides == lib_dat.lsides
        self.opfilt._type = _type
        if hasattr(self.ninv_filt, 'f'):
            print "******* I am using displacement for ninvfilt in MFest"
        else:
            print "******* Using id displacement in MFest"
        f = getattr(self.ninv_filt, 'f', fs.ffs_deflect.ffs_deflect.ffs_id_displacement(lib_sky.shape, lib_sky.lsides))
        if MFkey == 12:
            # B^t M^t X (x) (D ika P D^t B^t Covi X )(x). Second term are just the deflected gradients of the recontructed
            assert self.pix_pha is not None
            if soltn is None:
                soltn = np.zeros((self.opfilt.TEBlen(_type), self.ninv_filt.lib_skyalm.alm_size), dtype=complex)
            phas = self.pix_pha.get_sim(idx)[0:len(_type)]
            for i, _f in enumerate(_type): phas[i] *= self.ninv_filt.get_mask(_f.lower())
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
            phas = self.pix_pha.get_sim(idx)[0:len(_type)]
            if soltn is None:
                soltn = np.zeros((self.opfilt.TEBlen(_type), self.ninv_filt.lib_skyalm.alm_size), dtype=complex)
            inp = np.array([lib_sky.almxfl(lib_sky.map2alm(_p), self.ninv_filt.cl_transf) for _p in phas])
            inp = np.array([lib_dat.alm2map(lib_dat.udgrade(lib_sky, _p)) * self.ninv_filt.get_mask(_f) for _p, _f in
                            zip(inp, _type)])
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
        for i in range(1, len(_type)): retdx += Left(i) * Right(i, 1)
        retdx = self.lib_qlm.map2alm(retdx)
        retdy = Left(0) * Right(0, 0)
        for i in range(1, len(_type)): retdy += Left(i) * Right(i, 0)
        retdy = self.lib_qlm.map2alm(retdy)
        return np.array([- retdx * self.lib_qlm.get_ikx() - retdy * self.lib_qlm.get_iky(),
                         retdx * self.lib_qlm.get_iky() - retdy * self.lib_qlm.get_ikx()])  # N0  * output is normalized qest


def get_MFqlms(_type, MFkey, lib_dat, lib_sky, pix_phas, TQUMlik_pha, cl_transf, lib_qlm, f=None, use_Pool=0):
    assert lib_dat.lsides == lib_sky.lsides
    if f is None: f = fs.ffs_deflect.ffs_deflect.ffs_id_displacement(lib_sky.shape, lib_sky.lsides)
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
            return f.alm2lenmap(lib_sky, hp.almxfl(TQUMlik_pha[id] * kfunc(), cl_transf * norm), use_Pool=use_Pool)
    else:
        assert 0, 'not implemented'
    retdx = Left(0) * Right(0, 1)
    for i in range(1, len(_type)): retdx += Left(i) * Right(i, 1)
    retdx = lib_qlm.map2alm(retdx)
    retdy = Left(0) * Right(0, 0)
    for i in range(1, len(_type)): retdy += Left(i) * Right(i, 0)
    retdy = lib_qlm.map2alm(retdy)
    return np.array([- retdx * lib_qlm.get_ikx() - retdy * lib_qlm.get_iky(),
                     retdx * lib_qlm.get_iky() - retdy * lib_qlm.get_ikx()])  # N0  * output is unnormalized qest
