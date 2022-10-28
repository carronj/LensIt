"""CMB lensing likelihood and posterior curvature matrices

"""
from __future__ import annotations
import numpy as np, sys
import lensit as li
import lensit.ffs_covs.ffs_cov
from lensit import qcinv
from lensit.ffs_deflect import ffs_deflect
from lensit.qcinv import cd_solve, utils, cd_monitors, multigrid, ffs_ninv_filt
from lensit.misc import misc_utils
from lensit.ffs_covs.ffs_specmat import get_unlPmat_ij as unlPmat, TEB2TQUlms, TQU2TEBlms
from lensit.ffs_covs.ffs_cov import cl_inverse as cli

logger_H = (lambda itr, eps, watch=None, **kwargs:
                sys.stdout.write('H logger: rank %s ' % li.pbs.rank + '[' + str(watch.elapsed()) + '] ' + str((itr, eps)) + '\n'))
_BKW_TOL = 1e-5

class H:
    def __init__(self,typ,filt:ffs_ninv_filt.ffs_ninv_filt_wl, mchain:multigrid.multigrid_chain,
                 lib_qlm=None,wMF=False,datmaps=None,
                 datcls=None,TQUWF=None,TQURes=None,
                 datcmbs=None, plm_in=None, H0:None or np.ndarray=None):
        """
            Args:
                typ: 'T', 'QU', or 'TQU' for MV
                filt: filtering instance that contains the lik. model and will perform inverse-variance filtering
                mchain: multigrid chain used in the inverse-evariance filtering
                datcmbs: input sky E-mode if want to calculate responses
                H0: isotropic guess for full curvature, used in preconditioner (typically 1/N0 + 1/Cpp)

        """
        if TQUWF is None or TQURes is None:
            assert datmaps is not None

        self.filt= filt
        self.typ = typ

        self.TQUWF = TQUWF
        self.TQURes = TQURes
        self.datmaps = datmaps

        self.Pool = 0

        self.lib_skyalm = self.filt.lib_skyalm
        self.lib_datalm = self.filt.lib_datalm
        self.lib_qlm = self.lib_skyalm if lib_qlm is None else lib_qlm
        self.H0_i = cli(H0) if H0 is not None else np.ones(self.lib_qlm.ellmax + 1, dtype=float)

        self._datcls = datcls
        self._irootpostcls = None
        self.wMF = wMF  # Includes perturbative mean-field, or not
        self.datcmbs = datcmbs
        self.plm_in = plm_in
        if not np.any(self.filt.cls['bb']):
            from lensit.qcinv import opfilt_cinv_noBB as opfilt
        else:
            from lensit.qcinv import opfilt_cinv as opfilt

        self.opfilt = opfilt
        self.mchain = mchain

        self.f = filt.f.copy()
        self.fi = filt.fi.copy()

    def get_XWF(self,i:int or None=None):
        if self.TQUWF is None:
            assert self.datmaps is not None
            soltn = np.zeros((self.opfilt.TEBlen(self.typ), self.filt.lib_skyalm.alm_size), dtype=complex)
            self.mchain.solve(soltn, self.datmaps, finiop='MLIK')
            self.TQUWF = self.opfilt.soltn2TQUMlik(soltn, self.filt)
        return self.TQUWF[i] if i is not None else self.TQUWF

    def get_TQUres(self,i:int or None=None):
        if self.TQURes is None:
            assert self.datmaps is not None
            self.TQURes = self._mlik2rest_tqumlik(self.datmaps, self.get_XWF())
        return self.TQURes[i] if i is not None else self.TQURes

    def get_Res(self,i):
        return self.lib_skyalm.alm2map(self.get_TQUres(i=i))

    def get_Xmap(self,i,derv = (0,0)):
        alm = self.get_XWF(i).copy()
        if derv[0] > 0: alm *= self.lib_skyalm.get_iky() ** derv[0]
        if derv[1] > 0: alm *= self.lib_skyalm.get_ikx() ** derv[1]
        return self.f.alm2lenmap(self.lib_skyalm,alm,use_Pool=self.Pool)

    def _mlik2rest_tqumlik(self, datmaps, TQUMlik):
        """Produces B^t Ni (data - B D Mlik) in TQU space, that is fed into the qlm estimator.

        """
        f_id = ffs_deflect.ffs_id_displacement(self.filt.lib_skyalm.shape, self.filt.lib_skyalm.lsides)
        temp = TQU2TEBlms(self.typ, self.filt.lib_skyalm, TQUMlik)
        residuals = datmaps - self.filt.apply_Rs(self.typ, temp)
        self.filt.apply_maps(self.typ, residuals, inplace=True)
        self.filt.set_ffi(f_id, f_id)
        temp = self.filt.apply_Rts(self.typ, residuals)
        self.filt.set_ffi(self.f, self.fi)
        return TEB2TQUlms(self.typ, self.filt.lib_skyalm, temp)

    def apply_K(self, alms, **kwargs):
        """Applies inverse-variance filtering operator K = B^dagger Cov^{-1} B


        """
        self.opfilt._type = self.typ
        soltn = np.zeros((self.opfilt.TEBlen(self.typ), self.filt.lib_skyalm.alm_size), dtype=complex)
        maps = np.array([self.lib_datalm.alm2map(self.lib_datalm.almxfl(self.filt._deg(a),self.filt.cl_transf)) for a in alms])
        self.mchain.solve(soltn, maps, finiop='MLIK')
        return self._mlik2rest_tqumlik(maps, self.opfilt.soltn2TQUMlik(soltn, self.filt))


    def apply_dataF(self,alm,**kwargs):
        # Much better version, involving only one filtering of maps
        # See Notes
        # X_,a(x) (V - W)(x) + Res(x) (xi_,a(V - W)(x)
        _2map = lambda _alm: self.lib_skyalm.alm2map(_alm)
        _2lenmap = lambda _alm: self.filt.f.alm2lenmap(self.lib_skyalm,_alm,use_Pool=self.Pool)

        ikx_q = lambda: self.lib_qlm.get_ikx()
        iky_q = lambda: self.lib_qlm.get_iky()

        def vmap(ax):
            assert ax in ['x','y']
            if ax == 'y': return self.lib_qlm.alm2map(iky_q() * alm)
            if ax == 'x': return self.lib_qlm.alm2map(ikx_q() * alm)
            assert 0

        fields  = range(len(self.typ))

        # X,b v^b - xi,b (Res v^b)
        V_W  = np.array([self.lib_skyalm.map2alm(self.get_Xmap(i,derv=(0,1)) * vmap('x')) for i in fields])
        V_W -=  self.apply_xip(np.array([self.lib_skyalm.map2alm(self.get_Res(i) * vmap('x')) for i in fields]),
                           self.filt.cls,derv=(0,1))

        V_W += np.array([self.lib_skyalm.map2alm(self.get_Xmap(i,derv=(1,0)) * vmap('y')) for i in fields])
        V_W -= self.apply_xip(np.array([self.lib_skyalm.map2alm(self.get_Res(i) * vmap('y')) for i in fields]),
                           self.filt.cls,derv=(1,0))
        # K xi_b (Res v^b): inverse Covariance filtering.
        V_W = self.apply_K(V_W)

        # Cunl D^t K (V- W)
        xiVW = self.apply_xip(V_W,self.filt.cls,donotlensback=True)

        retdx = np.zeros(self.lib_skyalm.shape,dtype = float)
        retdy = np.zeros(self.lib_skyalm.shape,dtype = float)
        for i in fields:
            retdx += self.get_Xmap(i, derv= (0, 1)) * _2map(V_W[i])
            retdx += self.get_Res(i) * _2lenmap(xiVW[i] * self.lib_skyalm.get_ikx())

            retdy += self.get_Xmap(i, derv= (1, 0))* _2map(V_W[i])
            retdy += self.get_Res(i) * _2lenmap(xiVW[i] * self.lib_skyalm.get_iky())
        return -ikx_q() * self.lib_qlm.map2alm(retdx) - iky_q() * self.lib_qlm.map2alm(retdy)



    def apply_xip(self,alms,cmb_cls,derv = (0,0),donotlensback = False):
        # D C_unl D^t with optional derivatives in the midle. C_unl D^t if donotlensback is set.
        assert len(alms) == len(self.typ),(alms.shape,self.typ)
        assert np.all([alm.size == self.lib_skyalm.alm_size for alm in alms]),(alms.shape,self.lib_skyalm.alm_size)
        blms = np.array([self.filt.fi.lens_alm(self.lib_skyalm,_a,mult_magn=True,use_Pool=self.Pool) for _a in alms])
        ret = np.zeros(alms.shape,dtype = complex)
        for i in range(len(self.typ)):
            for j in range(len(self.typ)):
                ret[i] += unlPmat(self.typ,self.lib_skyalm,cmb_cls,i,j) * blms[j]
        del blms
        if derv[0] > 0:  # using broadcasting rules
            ret *= self.lib_skyalm.get_iky() ** derv[0]
        if derv[1] > 0:
            ret *= self.lib_skyalm.get_ikx() ** derv[1]
        if donotlensback: return ret
        return np.array([self.filt.f.lens_alm(self.lib_skyalm,_a,use_Pool=self.Pool) for _a in ret])

    def apply_datapart(self, alm, withF = True, withD = True, timed = False, **kwargs):
        assert alm.size == self.lib_qlm.alm_size,(alm.size,self.lib_qlm.alm_size)
        t = li.misc.misc_utils.timer(timed,'apply_datapart')
        ret = np.zeros_like(alm)
        if withF:
            ret += self.apply_dataF(alm,**kwargs)
            t.checkpoint('done F part')
        if withD:
            ret += self.apply_dataC(alm,**kwargs)
            t.checkpoint('done C part')
        return ret

    def apply_dataC(self,alm, **kwargs):
        # Res(x) (xi_ab Resv^b - X_ab V^b)
        _2alm = lambda _map: self.lib_skyalm.map2alm(_map)
        _2lenmap = lambda _alm: self.filt.f.alm2lenmap(self.lib_skyalm, _alm, use_Pool=self.Pool)

        ikx_q = lambda: self.lib_qlm.get_ikx()
        iky_q = lambda: self.lib_qlm.get_iky()

        def vmap(ax):
            assert ax in ['x','y']
            if ax == 'y': return self.lib_qlm.alm2map(iky_q() * alm)
            if ax == 'x': return self.lib_qlm.alm2map(ikx_q() * alm)
            assert 0
        fields  = range(len(self.typ))
        # xi_ab Res v^b, wihtou the 'a'...
        xiVW = self.apply_xip(np.array([_2alm(self.get_Res(i,**kwargs) * vmap('y'))for i in fields]),self.filt.cls,derv = (1,0),donotlensback=True)
        xiVW += self.apply_xip(np.array([_2alm(self.get_Res(i, **kwargs) * vmap('x')) for i in fields]), self.filt.cls, derv=(0, 1), donotlensback=True)
        retdx = np.zeros(self.lib_skyalm.shape, dtype=float)
        retdy = np.zeros(self.lib_skyalm.shape, dtype=float)
        for i in fields:
            retdx += self.get_Res(i, **kwargs) *( _2lenmap(xiVW[i] * self.lib_skyalm.get_ikx())
                                                     - self.get_Xmap(i,derv= (1, 1)) * vmap('y')
                                                     - self.get_Xmap(i,derv= (0, 2)) * vmap('x'))
            retdy += self.get_Res(i, **kwargs) * (_2lenmap(xiVW[i] * self.lib_skyalm.get_iky())
                                                  - self.get_Xmap(i, derv=(1, 1)) * vmap('x')
                                                  - self.get_Xmap(i, derv=(2, 0)) * vmap('y'))
        return -ikx_q() * self.lib_qlm.map2alm(retdx) - iky_q() * self.lib_qlm.map2alm(retdy)

    def apply(self,alm,tdat=1.,**kwargs):
        ret = self.lib_qlm.almxfl(alm,cli(self.filt.cls['pp']))
        if tdat == 0: return ret
        return ret + tdat * self.apply_likpart(alm,**kwargs)


    def apply_likpart(self,alm,**kwargs):
        ret = self.apply_detpart(alm,**kwargs)
        return ret + self.apply_datapart(alm, **kwargs)

    def apply_detpart(self,alm,**kwargs):
        if self.wMF:
            assert 0, 'not implemented'
            #return self.filt.get_mfresplms(self.typ, self.lib_qlm, use_cls_len=False)[0] * alm
        else:
            return alm * 0.

    def apply_R(self, qlm):
        # response term d /Xdat g^QD_phi  d Xdat/dphi
        assert qlm.size == self.lib_qlm.alm_size, (qlm.size, self.lib_qlm.alm_size)
        assert self.datcmbs is not None, 'cant do this without the input cmbs'
        assert self.plm_in is not None, 'cant do this without the input lensing map'

        _2map = lambda _alm: self.lib_skyalm.alm2map(_alm)
        _2mapq = lambda _alm: self.lib_qlm.alm2map(_alm)

        from lensit.ffs_deflect import ffs_deflect
        f_input = ffs_deflect.displacement_fromplm(self.lib_qlm, self.plm_in)
        _2lenmap = lambda _alm: f_input.alm2lenmap(self.lib_skyalm,_alm,use_Pool=self.Pool)
        ikx_q = lambda: self.lib_qlm.get_ikx()
        iky_q = lambda: self.lib_qlm.get_iky()
        ikx_s = lambda: self.lib_skyalm.get_ikx()
        iky_s = lambda: self.lib_skyalm.get_iky()

        assert self.typ == 'QU'
        assert self.datcmbs.size == self.lib_skyalm.alm_size, 'need input unlensed E mode here'
        qulms = self.lib_skyalm.EBlms2QUalms(np.array([self.datcmbs, 0 * self.datcmbs]))
        KXbab =        [ _2lenmap(ikx_s() * qulms[0]) * _2mapq(ikx_q() * qlm) +  \
                         _2lenmap(iky_s() * qulms[0]) * _2mapq(iky_q() * qlm),
                         _2lenmap(ikx_s() * qulms[1]) * _2mapq(ikx_q() * qlm) + \
                         _2lenmap(iky_s() * qulms[1]) * _2mapq(iky_q() * qlm) ]
        KXbab = self.apply_K(np.array([self.lib_skyalm.map2alm(m) for m in KXbab]))
        xiaKxab_0 = self.apply_xip(KXbab, self.filt.cls, derv=(1, 0))
        xiaKxab_1 = self.apply_xip(KXbab, self.filt.cls, derv=(0, 1))
        KXbab = [_2map(m) for m in KXbab]
        xiaKxab_0 = [_2map(m) for m in xiaKxab_0]
        xiaKxab_1 = [_2map(m) for m in xiaKxab_1]
        # fixme, sign?
        ret_0 = self.get_Xmap(0, derv=(1, 0)) * KXbab[0] + self.get_Xmap(1, derv=(1, 0)) * KXbab[1]
        ret_1 = self.get_Xmap(0, derv=(0, 1)) * KXbab[0] + self.get_Xmap(1, derv=(0, 1)) * KXbab[1]
        ret_0 += self.get_Res(0) * xiaKxab_0[0]  + self.get_Res(1) * xiaKxab_0[1]
        ret_1 += self.get_Res(0) * xiaKxab_1[0]  + self.get_Res(1) * xiaKxab_1[1]

        return -iky_q() * self.lib_qlm.map2alm(ret_0) - ikx_q() * self.lib_qlm.map2alm(ret_1)

    def apply_condH0(self,alm,tdat):
        print("**** this is conditioner H0")
        return self.lib_qlm.almxfl(alm,self.H0_i)

    def apply_condNone(self, alm, tdat):
        return alm.copy()

    def cd_solve(self, plm, tdat=1.,cond='H0', maxiter=50, ulm0=None,
                 tol=_BKW_TOL, tr_cd=li.qcinv.cd_solve.tr_cg, cd_roundoff=25, d0=None):
        """Looks like I can invert H^-1 Ha to good accuracy.

        """
        assert plm.size == self.lib_qlm.alm_size,(plm.size,self.lib_qlm.alm_size)
        if ulm0 is not None:
            assert ulm0.size == self.lib_qlm.alm_size, (ulm0.size, self.lib_qlm.alm_size)

        class dot_op:
            def __init__(self):
                pass

            def __call__(self, alms1, alms2, **kwargs):
                return np.sum(alms1.real * alms2.real + alms1.imag * alms2.imag)

        cond_func = getattr(self, 'apply_cond%s' % cond)

        # ! fwd_op and pre_ops must not change their arguments !
        def Hfwd_op(this_plm):
            return self.apply(this_plm.copy(),tdat=tdat)

        def Hpre_ops(this_plm):
            return cond_func(this_plm,tdat=tdat)

        Hdot_op = dot_op()

        if ulm0 is None: ulm0 = np.zeros_like(plm)

        criterion = li.qcinv.cd_monitors.monitor_basic(Hdot_op, iter_max=maxiter, eps_min=tol, d0=d0,logger=logger_H)
        print("++ cd_solve Hlib: starting, cond %s " % cond)
        li.ffs_covs.ffs_cov._timed = False
        iter = li.qcinv.cd_solve.cd_solve(ulm0,plm, Hfwd_op, [Hpre_ops], Hdot_op, criterion,tr_cd, roundoff=cd_roundoff)
        return ulm0, iter

"""
pl.ion()
par = imp.load_source('par','./scripts/rPDF/main.py')
rdata = 0.0
rgrid = par.get_rgrid(rdata,fine = True)
rfid = rgrid[20]
rfidh = rgrid[21]
Hlib = par.get_Hlib(0,rdata,rfid,10)
Hlibh = par.get_Hlib(0,rdata,rfidh,10)
h = rfidh-rfid
dplm = Hlib.get_dplm(tol = 1e-10) 
dphilm = (Hlibh._get_plm() -Hlib._get_plm())/h
ell = np.where(Hlib.lib_skyalm.get_Nell()[:4001] > 0)[0][1:]
pl.figure('dev')
pl.plot(ell,Hlib.lib_skyalm.alm2cl(dplm-dphilm)[ell]/Hlib.lib_skyalm.alm2cl(dphilm)[ell])
pl.loglog()
pl.figure('specs')
pl.plot(ell,Hlib.lib_skyalm.alm2cl(dplm)[ell],label = 'an. derv')
pl.plot(ell,Hlib.lib_skyalm.alm2cl(dphilm)[ell],label = '2p derv')
pl.loglog()
dgQD = Hlib.get_dgQD()
dgMF = Hlib.get_dgMF()
dplmQD = Hlib._get_dplmQD()
pl.figure('dgs')
pl.plot(ell,Hlib.lib_skyalm.alm2cl(dgQD[0])[ell],label = 'dgQD')
pl.plot(ell,Hlib.lib_skyalm.alm2cl(dgMF[0])[ell],label = 'dgMF')
pl.loglog()
pl.legend()
pl.figure('dev')
pl.plot(ell,Hlib.lib_skyalm.alm2cl(dplm-dphilm)[ell]/Hlib.lib_skyalm.alm2cl(dphilm)[ell],label = 'tot')

pl.plot(ell,Hlib.lib_skyalm.alm2cl(dplmQD-dphilm)[ell]/Hlib.lib_skyalm.alm2cl(dphilm)[ell],label = 'QD only')
pl.loglog()
pl.show()
"""
