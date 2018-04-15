import sys, os, re, copy
import numpy as np
import time
import pickle as pk
import cd_solve, cd_monitors


# ===
class dt():
    def __init__(self, _dt):
        self.dt = _dt

    def __str__(self):
        return ('%02d:%02d:%02d' % (np.floor(self.dt / 60 / 60),
                                    np.floor(np.mod(self.dt, 60 * 60) / 60),
                                    np.floor(np.mod(self.dt, 60))))

    def __int__(self):
        return int(self.dt)


class stopwatch():
    def __init__(self):
        self.st = time.time()
        self.lt = self.st

    def lap(self):
        lt = time.time()
        ret = (dt(lt - self.st), dt(lt - self.lt))
        self.lt = lt
        return ret

    def elapsed(self):
        lt = time.time()
        ret = dt(lt - self.st)
        self.lt = lt
        return ret


class multigrid_stage(object):
    def __init__(self, id, pre_ops_descr, lmax, nside, iter_max, eps_min, tr, cache):
        self.depth = id
        self.pre_ops_descr = pre_ops_descr
        self.lmax = lmax
        self.nside = nside
        self.iter_max = iter_max
        self.eps_min = eps_min
        self.tr = tr
        self.cache = cache
        self.pre_ops = []


class multigrid_chain():
    def __init__(self, opfilt, _type, chain_descr, cov, lib_to_split='skydat',
                 debug_log_prefix=None, plogdepth=0, no_deglensing=False):
        self.debug_log_prefix = debug_log_prefix
        self.plogdepth = plogdepth

        self.opfilt = opfilt
        self.chain_descr = chain_descr
        # lmin = cov.lib_datalm.ellmin
        self.cov = cov
        self._type = _type
        self.no_deglensing = no_deglensing

        stages = {}
        for [id, pre_ops_descr, lmax, nside, iter_max, eps_min, tr, cache] in self.chain_descr:
            stages[id] = multigrid_stage(id, pre_ops_descr, lmax, nside, iter_max, eps_min, tr, cache)
            for pre_op_descr in pre_ops_descr:  # recursively add all stages to stages[0]
                stages[id].pre_ops.append(
                    parse_pre_op_descr(pre_op_descr, opfilt=self.opfilt, cov=cov, lmin=0, libtosplit=lib_to_split,
                                       stages=stages, lmax=lmax, nside=nside, chain=self, no_lensing=no_deglensing))
        # TODO : the no_lensing keyword here just switch on or off the lensing operations on the resolution degraded
        # TODO   maps. It might be useful to be more clever, e.g. doing at least the dense part with lensing but switching it then
        # TODO   off, or using a crude lensing method.
        self.bstage = stages[0]  # these are the pre_ops called in cd_solve

    def solve(self, soltn, alms, finiop=None, d0=None):
        self.watch = stopwatch()

        self.iter_tot = 0
        self.prev_eps = None

        logger = (lambda iter, eps, stage=self.bstage, **kwargs:
                  self.log(stage, iter, eps, **kwargs))
        if hasattr(self.opfilt, 'crit_op'):
            print "**** multigrid : setting up criterium operation"
            crit_op = self.opfilt.crit_op(self.cov)
        else:
            crit_op = self.opfilt.dot_op(self.cov.lib_skyalm)
        fwd_op = self.opfilt.fwd_op(self.cov, False)
        _b = self.opfilt.calc_prep(alms, self.cov)
        if d0 is None: d0 = crit_op(_b, _b)
        monitor = cd_monitors.monitor_basic(crit_op, logger=logger, iter_max=self.bstage.iter_max,
                                            eps_min=self.bstage.eps_min, d0=d0)
        cd_solve.cd_solve(soltn, _b, fwd_op, self.bstage.pre_ops, self.opfilt.dot_op(self.cov.lib_skyalm), monitor,
                          tr=self.bstage.tr, cache=self.bstage.cache)
        if finiop is None:
            return
        else:
            assert hasattr(self.opfilt, 'apply_fini_' + finiop)
            return getattr(self.opfilt, 'apply_fini_' + finiop)(soltn, self.cov, alms)

    def log(self, stage, iter, eps, **kwargs):
        self.iter_tot += 1
        elapsed = self.watch.elapsed()

        if stage.depth > self.plogdepth:
            # pass #Uncomment this to print the details of the multigrid convergence
            return

        log_str = ('   ') * stage.depth + '(%4d, %04d) [%s] (%d, %.10f)' % (
            stage.nside, stage.lmax, str(elapsed), iter, eps) + '\n'
        sys.stdout.write(log_str)

        if (self.debug_log_prefix is not None):
            log = open(self.debug_log_prefix + 'stage_all.dat', 'a')
            log.write(log_str)
            log.close()

            if (stage.depth == 0):
                f_handle = file(self.debug_log_prefix + 'stage_soltn_' + str(stage.depth) + '.dat', 'a')
                np.savetxt(f_handle, [[v for v in kwargs['soltn']]])
                f_handle.close()

                f_handle = file(self.debug_log_prefix + 'stage_resid_' + str(stage.depth) + '.dat', 'a')
                np.savetxt(f_handle, [[v for v in kwargs['resid']]])
                f_handle.close()

            log_str = '%05d %05d %10.6e %05d %s\n' % (self.iter_tot, int(elapsed), eps, iter, str(elapsed))
            log = open(self.debug_log_prefix + 'stage_' + str(stage.depth) + '.dat', 'a')
            log.write(log_str)
            log.close()

            if ((self.prev_eps is not None) and (self.prev_stage.depth > stage.depth)):
                log_final_str = '%05d %05d %10.6e %s\n' % (
                    self.iter_tot - 1, int(self.prev_elapsed), self.prev_eps, str(self.prev_elapsed))

                log = open(self.debug_log_prefix + 'stage_final_' + str(self.prev_stage.depth) + '.dat', 'a')
                log.write(log_final_str)
                log.close()

            self.prev_stage = stage
            self.prev_eps = eps
            self.prev_elapsed = elapsed


# ===

def parse_pre_op_descr(pre_op_descr, **kwargs):
    if re.match("split\((.*),\s*(.*),\s*(.*)\)\Z", pre_op_descr):
        (low_descr, lsplit, hgh_descr) = re.match("split\((.*),\s*(.*),\s*(.*)\)\Z", pre_op_descr).groups()
        print 'creating split preconditioner ', (low_descr, lsplit, hgh_descr)

        lsplit = int(lsplit)
        lmax = kwargs['lmax']
        lmin = kwargs['lmin']
        _shape = (kwargs['nside'], kwargs['nside'])

        kwargs_low = copy.copy(kwargs)
        kwargs_hgh = copy.copy(kwargs)
        # FIXME : do we really want to use lmin ?
        kwargs_low['lmax'] = lsplit
        kwargs_hgh['lmin'] = lsplit + 1
        # kwargs_hgh['cov'] = kwargs['cov'].degrade(_shape,kwargs['no_lensing'],ellmax = lmax,ellmin = lsplit + 1)
        # kwargs_low['cov'] = kwargs['cov'].degrade(_shape,kwargs['no_lensing'],ellmax = lsplit, ellmin =lmin)

        pre_op_low = parse_pre_op_descr(low_descr, **kwargs_low)
        pre_op_hgh = parse_pre_op_descr(hgh_descr, **kwargs_hgh)
        # return pre_op_split(kwargs['cov'], pre_op_low, pre_op_hgh)
        split = pre_op_split_sky  # if kwargs['libtosplit'] == 'sky' else pre_op_split
        return split(kwargs['cov'].degrade(_shape, no_lensing=kwargs['no_lensing'], ellmax=lmax, ellmin=lmin,
                                           libtodegrade=kwargs['libtosplit']), pre_op_low, pre_op_hgh)
        # return pre_op_split(kwargs['cov'].degrade(_shape, kwargs['no_lensing'], ellmax=lmax), pre_op_low,pre_op_hgh)

    elif re.match("diag_cl\Z", pre_op_descr):
        lmax = kwargs['lmax']
        lmin = kwargs['lmin']
        _shape = (kwargs['nside'], kwargs['nside'])
        cov = kwargs['cov'].degrade(_shape, no_lensing=True, ellmin=lmin, ellmax=lmax,
                                    libtodegrade=kwargs['libtosplit'])
        return kwargs['opfilt'].pre_op_diag(cov, kwargs['no_lensing'])

    elif re.match("pseuddiag_cl\Z", pre_op_descr):
        lmax = kwargs['lmax']
        lmin = kwargs['lmin']
        _shape = (kwargs['nside'], kwargs['nside'])
        cov = kwargs['cov'].degrade(_shape, no_lensing=True, ellmin=lmin, ellmax=lmax,
                                    libtodegrade=kwargs['libtosplit'])
        return kwargs['opfilt'].pre_op_pseudiag(cov, kwargs['no_lensing'])

    elif re.match("dense\((.*)\)\Z", pre_op_descr):
        (dense_cache_fname,) = re.match("dense\((.*)\)\Z", pre_op_descr).groups()
        if dense_cache_fname == '': dense_cache_fname = None
        lmax = kwargs['lmax']
        lmin = kwargs['lmin']
        no_lensing = kwargs['no_lensing']
        _shape = (kwargs['nside'], kwargs['nside'])

        print 'creating dense preconditioner. (nside = %d, lmax = %d, cache = %s)' \
              '' % (kwargs['nside'], lmax, dense_cache_fname)
        cov = kwargs['cov'].degrade(_shape, no_lensing=no_lensing, ellmin=lmin, ellmax=lmax,
                                    libtodegrade=kwargs['libtosplit'])
        print dense_cache_fname, ' DENSE CACHE FNAME'
        return kwargs['opfilt'].pre_op_dense(cov, no_lensing, cache_fname=dense_cache_fname)

    elif re.match("stage\(.*\)\Z", pre_op_descr):
        (stage_id,) = re.match("stage\((.*)\)\Z", pre_op_descr).groups()
        print 'creating multigrid preconditioner: stage_id = %s, ell range %s %s' % (
            stage_id, kwargs['lmin'], kwargs['lmax'])

        stage = kwargs['stages'][int(stage_id)]
        logger = (lambda iter, eps, stage=stage, chain=kwargs['chain'], **kwargs:
                  chain.log(stage, iter, eps, **kwargs))
        lmax = kwargs['lmax']
        lmin = kwargs['lmin']
        no_lensing = kwargs['no_lensing']
        _shape = (kwargs['nside'], kwargs['nside'])
        cov = kwargs['cov'].degrade(_shape, no_lensing=no_lensing, ellmin=lmin, ellmax=lmax,
                                    libtodegrade=kwargs['libtosplit'])
        assert (stage.lmax == kwargs['lmax'])

        return pre_op_multigrid(kwargs['opfilt'], stage.nside,
                                cov, no_lensing, stage.pre_ops, logger, stage.tr, stage.cache,
                                stage.iter_max, stage.eps_min)
    # opfilt, lmax, nside, cov, no_lensing, _type, pre_ops,
    # logger, tr, cache, iter_max, eps_min
    else:
        print 'pre_op_descr = ', pre_op_descr, ' is unrecognized!'
        assert (0)


# ===
# FIXME : a better flat sky scheme might be good here as well...
class pre_op_split():
    def __init__(self, cov, pre_op_low, pre_op_hgh):
        self.cov = cov  # Base class for lib_alm that we will split
        self.pre_op_low = pre_op_low
        self.pre_op_hgh = pre_op_hgh
        self.lmax_low = self.pre_op_low.cov.lib_datalm.ellmax
        self.lmin_high = self.pre_op_hgh.cov.lib_datalm.ellmin

        # assert lmax_low  < lmin_high,(lmax_low,lmin_high) # Not necessary
        print "    ++ setting up split cov", self.cov.lib_datalm.ellmin, self.cov.lib_datalm.ellmax
        self.iter = 0

    def __call__(self, alms):
        return self.calc(alms)

    def calc(self, alms, low_only=False, high_only=False):
        self.iter += 1
        alms_low = np.array([self.pre_op_low.cov.lib_datalm.udgrade(self.cov.lib_datalm, _alm) for _alm in alms])
        alms_hgh = np.array([self.pre_op_hgh.cov.lib_datalm.udgrade(self.cov.lib_datalm, _alm) for _alm in alms])

        alms_low = self.pre_op_low(alms_low)
        alms_hgh = self.pre_op_hgh(alms_hgh)

        # The low-ell is anyway given by the low-ell calculation, but the low-ell - high-ell correlations are given
        # by the high-ell calculation.
        ret = np.array([self.cov.lib_datalm.udgrade(self.pre_op_hgh.cov.lib_datalm, _a) for _a in alms_hgh])
        if high_only: return ret
        if low_only: ret *= 0

        if self.lmax_low < self.lmin_high:  # can simply sum up result
            ret += np.array([self.cov.lib_datalm.udgrade(self.pre_op_low.cov.lib_datalm, _a) for _a in alms_low])
            return ret
        else:
            idc = np.where(self.cov.lib_datalm.reduced_ellmat() <= self.pre_op_low.cov.lib_datalm.ellmax)
            for _i in xrange(ret.shape[0]):
                ret[_i, idc] = self.cov.lib_datalm.udgrade(self.pre_op_low.cov.lib_datalm, alms_low[_i])[idc]
            return ret


class pre_op_split_sky():
    def __init__(self, cov, pre_op_low, pre_op_hgh):
        self.cov = cov  # Base class for lib_alm that we will split
        self.pre_op_low = pre_op_low
        self.pre_op_hgh = pre_op_hgh
        self.lmax_low = self.pre_op_low.cov.lib_skyalm.ellmax
        self.lmin_high = self.pre_op_hgh.cov.lib_skyalm.ellmin

        # assert lmax_low  < lmin_high,(lmax_low,lmin_high) # Not necessary
        print "    ++ setting up split cov", self.cov.lib_skyalm.ellmin, self.cov.lib_skyalm.ellmax
        self.iter = 0

    def __call__(self, alms):
        return self.calc(alms)

    def calc(self, alms, low_only=False, high_only=False):
        self.iter += 1
        alms_low = np.array([self.pre_op_low.cov.lib_skyalm.udgrade(self.cov.lib_skyalm, _alm) for _alm in alms])
        alms_hgh = np.array([self.pre_op_hgh.cov.lib_skyalm.udgrade(self.cov.lib_skyalm, _alm) for _alm in alms])

        alms_low = self.pre_op_low(alms_low)
        alms_hgh = self.pre_op_hgh(alms_hgh)

        # The low-ell is anyway given by the low-ell calculation, but the low-ell - high-ell correlations are given
        # by the high-ell calculation.
        ret = np.array([self.cov.lib_skyalm.udgrade(self.pre_op_hgh.cov.lib_skyalm, _a) for _a in alms_hgh])
        if high_only: return ret
        if low_only: ret *= 0

        if self.lmax_low < self.lmin_high:  # can simply sum up result
            ret += np.array([self.cov.lib_skyalm.udgrade(self.pre_op_low.cov.lib_skyalm, _a) for _a in alms_low])
            return ret
        else:
            idc = np.where(self.cov.lib_skyalm.reduced_ellmat() <= self.pre_op_low.cov.lib_skyalm.ellmax)
            for _i in xrange(ret.shape[0]):
                ret[_i, idc] = self.cov.lib_skyalm.udgrade(self.pre_op_low.cov.lib_skyalm, alms_low[_i])[idc]
            return ret


class pre_op_multigrid():
    def __init__(self, opfilt, nside, cov, no_lensing, pre_ops,
                 logger, tr, cache, iter_max, eps_min):
        self.opfilt = opfilt
        self.fwd_op = opfilt.fwd_op(cov, no_lensing)
        self.cov = cov
        self.nside = nside

        self.pre_ops = pre_ops

        self.logger = logger

        self.tr = tr
        self.cache = cache

        self.iter_max = iter_max
        self.eps_min = eps_min

    def __call__(self, alms):
        return self.calc(alms)

    def calc(self, alms):
        monitor = cd_monitors.monitor_basic(self.opfilt.dot_op(), iter_max=self.iter_max, eps_min=self.eps_min,
                                            logger=self.logger)
        soltn = alms * 0.0
        cd_solve.cd_solve(soltn, alms.copy(),
                          self.fwd_op, self.pre_ops, self.opfilt.dot_op(), monitor, tr=self.tr, cache=self.cache)

        return soltn
