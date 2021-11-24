import sys
import numpy as np
import time
from lensit.pbs import pbs


class dt:
    def __init__(self, _dt):
        self.dt = _dt

    def __str__(self):
        return ('%02d:%02d:%02d' % (np.floor(self.dt / 60 / 60),
                                    np.floor(np.mod(self.dt, 60 * 60) / 60),
                                    np.floor(np.mod(self.dt, 60))))

    def __int__(self):
        return int(self.dt)


class stopwatch:
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


## monitors
logger_basic = (lambda iter, eps, watch=None, **kwargs:
                sys.stdout.write('rank %s ' % pbs.rank + '[' + str(watch.elapsed()) + '] ' + str((iter, eps)) + '\n'))
logger_none = (lambda iter, eps, watch=None, **kwargs: 0)


class monitor_basic:
    def __init__(self, dot_op, iter_max=1000, eps_min=1.0e-10, d0=None, logger=logger_basic):
        """Basic gonjugate gradient convergence monitor

        :param dot_op:
        :param iter_max:
        :param eps_min:
        :param d0: criterion to step iteration is  self.dot_op(resid, resid) <= self.eps_min ** 2 * self.d0
        :param logger:
        :return:

        """
        self.dot_op = dot_op
        self.iter_max = iter_max
        self.eps_min = eps_min
        self.logger = logger
        self.d0 = d0
        self.watch = stopwatch()

    def criterion(self, iter, soltn, resid):
        delta = self.dot_op(resid, resid)

        if iter == 0 and self.d0 is None:
            # For first iteration typically resid is b where x = A^-1 b is the equations to solve,
            # unless some starting guess is provided.
            self.d0 = delta
        if self.d0 == 0: self.d0 = 1.
        if self.logger is not None: self.logger(iter, np.sqrt(delta / self.d0), watch=self.watch,
                                                  soltn=soltn, resid=resid)

        if (iter >= self.iter_max) or (delta <= self.eps_min ** 2 * self.d0):
            return True

        return False

    def __call__(self, *args):
        return self.criterion(*args)
