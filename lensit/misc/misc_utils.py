# Convenience functions :
import sys
import time

import numpy as np
import os
import hashlib
from .. import pbs


def cls_hash(cls, lmax=None):
    if lmax is None:
        arr = np.concatenate([cls[k] for k in sorted(cls.keys())])
    else:
        arr = np.concatenate([(cls[k])[:lmax + 1] for k in sorted(cls.keys())])
    return hashlib.sha1(arr.copy(order='C')).hexdigest()

class timer():
    def __init__(self, verbose, prefix='', suffix=''):
        self.t0 = time.time()
        self.ti = np.copy(self.t0)
        self.verbose = verbose
        self.prefix = prefix
        self.suffix = suffix

    def checkpoint(self, msg):
        dt = time.time() - self.t0
        self.t0 = time.time()

        if self.verbose:
            dh = np.floor(dt / 3600.)
            dm = np.floor(np.mod(dt, 3600.) / 60.)
            ds = np.floor(np.mod(dt, 60))
            dhi = np.floor((self.t0 - self.ti) / 3600.)
            dmi = np.floor(np.mod((self.t0 - self.ti), 3600.) / 60.)
            dsi = np.floor(np.mod((self.t0 - self.ti), 60))
            sys.stdout.write("\r  %s   [" % self.prefix + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] "
                             + " (total [" + (
                                 '%02d:%02d:%02d' % (dhi, dmi, dsi)) + "]) " + msg + ' %s \n' % self.suffix)


def read_params(paramfile):
    """
    Reads a parameter file with lines of the form key = value as a dictionary
    """
    assert os.path.exists(paramfile), paramfile
    params = {}
    with open(paramfile) as f:
        for line in f:
            (key, equal, val) = line.split()
            params[key] = val
    return params


def legendreP(N, x):
    """
    returns the values of the Legendre polynomials
               up to order N, at the argument x
    """
    x = np.array(x)
    Pn = np.ones(x.size)
    if N == 0: return Pn
    res = np.zeros((N + 1, x.size))
    Pn1 = x
    res[0, :] = Pn
    res[1, :] = Pn1
    if N == 1: return res
    for I in xrange(1, N):
        res[I + 1L, :] = 2. * x * res[I, :] - res[I - 1, :] - (x * res[I, :] - res[I - 1, :]) / (I + 1.)
    return res


def C0_box(Cl, fsky):
    """
    Returns the zero mode variance expected in a cap of the sky of volume fsky.
    """
    assert fsky > 0. and fsky <= 1.
    lmax = len(Cl)
    x = 1. - 2 * fsky
    facl = 0.5 / np.sqrt(4. * np.pi) / fsky / np.sqrt(2 * np.arange(lmax + 1) + 1)
    Pl = legendreP(lmax + 1, x)[:, 0]
    W0 = facl[0] * (1. - x)
    Wl = -Pl[2: len(Pl)] + Pl[0:len(Pl) - 2]
    Wl = np.insert(Wl * facl[1:], 0, W0)
    return np.sum(Cl * Wl[0:lmax] ** 2)


def enumerate_progress(list, label=''):
    # Taken boldly from Duncan Hanson lpipe :
    #  e.g. : for i,v in enumerate_progress(list,label = 'calculating...')
    if pbs.size == 1 or pbs.rank == 0:
        t0 = time.time()
        ni = len(list)
        for i, v in enumerate(list):
            yield i, v
            ppct = int(100. * (i - 1) / ni)
            cpct = int(100. * (i + 0) / ni)
            if cpct > ppct:
                dt = time.time() - t0
                dh = np.floor(dt / 3600.)
                dm = np.floor(np.mod(dt, 3600.) / 60.)
                ds = np.floor(np.mod(dt, 60))
                sys.stdout.write("\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " +
                                 label + " " + int(10. * cpct / 100) * "-" + "> " + ("%02d" % cpct) + r"%")
                sys.stdout.flush()
        sys.stdout.write("\n");
        sys.stdout.flush()
    else:
        for i, v in enumerate(list):
            yield i, v


def IsPowerOfTwo(i):
    """
        Returns true if all entries of i are powers of two.
        False otherwise.
    """
    return (i & (i - 1)) == 0 and i != 0


def Log2ofPowerof2(shape):
    """
    Returns powers of two exponent for each element of shape
    """
    # There must be a better way, such as the first non zero byte.
    res = np.array(shape)
    for i in xrange(res.size):
        n = shape[i]
        assert (IsPowerOfTwo(n)), "Invalid input"
        ix = 0
        while n > 1:
            n /= 2
            ix += 1
        res[i] = ix
    return res


def int_tabulated(x, y, **kwargs):
    # Emulates IDL int_tabulated fct for the moment with scipy.integrate.sims
    from scipy.integrate import simps
    return simps(y, x=x, **kwargs)


class stats():
    """
    Simple minded routines for means and averages of sims .
    Calculates means as 1/N sum()
    and Cov as 1/(N-1)sum(x - mean)(x - mean)^t
    """

    def __init__(self, size, xcoord=None, do_cov=True, dtype=float):
        self.N = 0  # number of samples
        self.size = size  # dim of data vector
        self.sum = np.zeros(self.size, dtype=dtype)  # sum_i x_i
        self.do_cov = do_cov
        if self.do_cov:
            self.mom = np.zeros((self.size, self.size))  # sum_i x_ix_i^t
        self.xcoord = xcoord

    def add(self, v):
        assert (v.shape == (self.size,)), "input not understood"
        self.sum += v
        if self.do_cov:
            self.mom += np.outer(v, v)
        self.N += 1

    def mean(self):
        assert (self.N > 0)
        return self.sum / self.N

    def cov(self):
        """
        1/(N-1) sum_i = 1^N (X_i - bX)(X_i - bX)
        = Mom / (N-1) + N/(N-1) bX bX^t - 2 N / (N-1) bX bX^t
        = Mom / (N-1) - N/(N-1) bX bX^t
        """
        assert (self.N > 0)
        assert self.do_cov
        if self.N == 1: return np.zeros((self.size, self.size))
        mean = self.mean()
        return self.mom / (self.N - 1.) - np.outer(mean, mean * (self.N / (self.N - 1.)))

    def sigmas(self):
        return np.sqrt(np.diagonal(self.cov()))

    def corrcoeffs(self):
        assert self.do_cov
        sigmas = self.sigmas()
        return self.cov() / np.outer(sigmas, sigmas)

    def sigmas_on_mean(self):
        assert (self.N > 0)
        return self.sigmas() / np.sqrt(self.N)

    def inverse(self, bias_p=None):  # inverse cov, using unbiasing a factor following G. statistics
        assert (self.N > self.size), "Non invertible cov.matrix"
        if bias_p is None: bias_p = (self.N - self.size - 2.) / (self.N - 1)
        return bias_p * np.linalg.inv(self.cov())

    def get_chisq(self, data):  # Returns (data -mean)Sig^{-1}(data-mean)
        assert (data.size == self.size), "incompatible input"
        dx = data - self.mean()
        return np.sum(np.outer(dx, dx) * self.inverse())

    def get_chisq_pte(self, data):  # probability to exceed, or survival function
        from scipy.stats import chi2
        return chi2.sf(self.get_chisq(data), self.N - 1)  # 'survival function' of chisq distribution with N -1 dof

    def rebin_that_nooverlap(self, orig_coord, lmins, lmaxs, weights=None):
        # Returns a new stat instance rebinning with non-overlapping weights
        # >= a gauche, <= a droite.
        assert (orig_coord.size == self.size), "Incompatible input"
        assert (lmins.size == lmaxs.size), "Incompatible input"
        assert (np.all(np.diff(np.array(lmins)) > 0.)), "This only for non overlapping bins."
        assert (np.all(np.diff(np.array(lmaxs)) > 0.)), "This only for non overlapping bins."
        assert (np.all(lmaxs - lmins) > 0.), "This only for non overlapping bins."

        if weights is None: weights = np.ones(self.size)
        assert (weights.size == self.size), "incompatible input"
        newsize = len(lmaxs)
        assert (self.size > newsize), "Incompatible dimensions"
        Tmat = np.zeros((newsize, self.size))
        newsum = np.zeros(newsize)
        for k, lmin, lmax in zip(np.arange(newsize), lmins, lmaxs):
            idc = np.where((orig_coord >= lmin) & (orig_coord <= lmax))
            if len(idc) > 0:
                norm = np.sum(weights[idc])
                Tmat[k, idc] = weights[idc] / norm
                newsum[k] = np.sum(weights[idc] * self.sum[idc]) / norm

        newmom = np.dot(np.dot(Tmat, self.mom), Tmat.transpose())  # New mom. matrix is T M T^T
        newstats = stats(newsize, xcoord=0.5 * (lmins[0:len(lmins) - 1] + lmax[1:]))
        # Resets the stats things
        newstats.mom = newmom
        newstats.sum = newsum
        newstats.N = self.N
        return newstats


class binner():
    def __init__(self, bins_l, bins_r):
        """
        Binning routines. Left and right inclusive.
        For most general situation
        :param bins_l: left edges (inclusive)
        :param bins_r: right edges (inclusive)
        """
        assert (len(bins_l) == len(bins_r)), "inconsistent inputs"
        assert (np.all(bins_r - bins_l > 0.)), "inconsistent input"
        self.bins_l = np.array(bins_l)
        self.bins_r = np.array(bins_r)

    def Nbins(self):
        return len(self.bins_l)

    def bin_centers(self):
        return 0.5 * self.bins_l + 0.5 * self.bins_r

    def bin_that(self, x, y, weights=None, return_err=False):
        ret = np.zeros(self.Nbins())
        if weights is None: weights = np.ones(len(x), dtype=float)
        assert (len(x) == len(y) and len(x) == len(weights)), "inconsistent inputs"
        err = np.zeros(self.Nbins())
        for i, bin_l, bin_r in zip(xrange(self.Nbins()), self.bins_l, self.bins_r):
            idc = np.array(np.where((x >= bin_l) & (x <= bin_r)))
            if idc.size > 0.:
                ret[i] = np.sum(y[idc] * weights[idc]) / idc.size
                err[i] = np.std(y[idc] * weights[idc]) / np.sqrt(idc.size)
        if not return_err:
            return ret
        else:
            return ret, err


def mk_session_seed(verbose=False):
    """
    Tries to create a reasonable seed from hostname and time and initialize the
    nump.random rng.
    """
    from decimal import Decimal
    import socket
    import time
    from lensit import pbs
    from hashlib import sha1
    rank = pbs.rank
    hostname = socket.gethostname()
    # if not os.path.exists(fname) :
    time_str = str(Decimal(time.time()))
    if verbose: print 'building seed with Hostname, pbs 111 * rank and 111 * time :', hostname, 111 * rank, time_str
    hash_hex = sha1(hostname + str(111 * rank) + 111 * time_str).hexdigest()  # 160 bits hexadecimal hash string
    seed = np.array([int(hash_hex[i:i + 8], 16) for i in [0, 8, 16, 24, 32]])
    # seed must a array like of 32 bit integers
    np.random.seed(seed)
    # moves the seed by 10^6
    if verbose: print "moving seed 10^6 times:"
    for i in xrange(1000000): a = np.random.random()
    if verbose: print "done"
    return np.random.get_state()


def rfft2_sum(rfft_map):
    """ Implementation of \sum_k map_k when using rfft arrays : (for odd number of points set only [:,0]) """
    assert len(rfft_map.shape) == 2
    if rfft_map.shape[1] % 2 == 0:
        return 2 * np.sum(rfft_map) - np.sum(rfft_map[:, [-1, 0]])
    else:
        2 * np.sum(rfft_map) - np.sum(rfft_map[:, 0])


def PartialDerivativePeriodic(arr, axis, h=1., rule='4pts'):
    """
    Returns the partial derivative of the arr along axis 'axis',
    following a 2pts or 4pts rule, reinventing the wheel.
    Uses periodic boundary conditions.
    """
    if rule == '4pts':  # O(h**4) rule
        idc = [-2, -1, 1, 2]
        weights = np.array((-1., 8., -8., 1.)) / (12. * h)
    elif rule == '2pts':  # O(h**2) rule
        idc = [-1, 1]  # np.rolling by one means g(x) =  f(x -1)
        weights = np.array((1., -1)) / (2. * h)
    else:
        idc = 0
        weights = 0
        assert 0, rule + " not implemented"

    grad = np.roll(arr, idc[0], axis=axis) * weights[0]
    for i, w in zip(idc[1:], weights[1:]): grad += np.roll(arr, i, axis=axis) * w
    return grad


def outerproducts(vs):
    """
    vs is a list of 1d numpy arrays, not necessarily of the same size.
    Return a matrix A_i1_i2..i_ndim = vi1_vi2_..v_indim.
    Use np.outer recursively on flattened arrays.
    """

    # check input and infer new shape  :
    assert (isinstance(vs, list)), "Want list of 1d arrays"
    ndim = len(vs)
    if ndim == 1: return vs[0]
    shape = ()
    for i in xrange(ndim):
        assert (vs[i].ndim == 1), "Want list of 1d arrays"
        shape += (vs[i].size,)

    B = vs[ndim - 1]

    for i in xrange(1, ndim): B = np.outer(vs[ndim - 1 - i], B).flatten()
    return B.reshape(shape)


def square_pixwin_map(shape):
    """
    pixel window function of square top hat for any dimension.
    k*lcell / 2
    """

    vs = []
    for ax in range(len(shape)):
        lcell_ka = 0.5 * Freq(np.arange(shape[ax]), shape[ax]) * (2. * np.pi / shape[ax])
        vs.append(np.insert(np.sin(lcell_ka[1:]) / lcell_ka[1:], 0, 1.))
    return outerproducts(vs)


def Freq(i, N):
    """
     Outputs the absolute integers frequencies [0,1,...,N/2,N/2-1,...,1]
     in numpy fft convention as integer i runs from 0 to N-1.
     Inputs can be numpy arrays e.g. i (i1,i2,i3) with N (N1,N2,N3)
                                  or i (i1,i2,...) with N
     Both inputs must be integers.
     All entries of N must be even.
    """
    assert (np.all(N % 2 == 0)), "This routine only for even numbers of points"
    return i - 2 * (i >= (N / 2)) * (i % (N / 2))


def DirichletKernel(n, dim, d=1.):
    """
    Returns the Dirichlet kernel associated to the fft frequencies.
    dim is the 1d fft dimension, d the real space spacing, int n the width
    of the top-hat. Has value 2n + 1 at 0
    """
    freqs = Freq(np.arange(1, dim), dim) * (2 * np.pi / dim / d)
    return np.insert(np.sin(freqs * (n + 0.5)) / np.sin(freqs * 0.5), 0, (2. * n + 1))


class library_datacube():
    """
        Library for fields represented by numpy arrays
        meant to for Fourier analysis etc, where
        each side may have a different physical length and grid resolution.
        Recall numpy fft  conventions :
        fft(f(x)) (np) = \sum_m a_m e^(-i2\pi m k /N)
            ---> (Npix/V) \int dx f(x) e^(-ikx)
        ifft(fft) = 1 = 1/Npix \sum a_k e^(i 2\pi m k /N)
            ---> (V / Npix) \int dk /(2pi) \tilde a(k) e^(ikx)
    """

    def __init__(self, resolution, lside, verbose=True):

        self.resolution = np.array(resolution)  # the number of points in each side is 2**res
        self.lside = np.array(lside)  # Physical total lengths of the box
        self.ndim = len(self.lside)
        self.verbose = verbose
        assert (len(lside) == len(resolution)), "Inconsistent input"

    def vol(self):
        """
        Returns volume in physical units
        """
        return np.prod(self.lside)

    def cell_vol(self):
        return np.prod(self.rmin())

    def npix(self):
        """
        Returns the number of resolution elements
        """
        return np.prod(self.shape())

    def rmin(self):
        """
        Returns physical cell lengths along each dimensions
        """
        return self.lside / self.shape()

    def rmax(self):
        """
        Returns physical cell lengths along each dimensions
        """
        return self.lside / 2.

    def kmin(self):
        """
        Returns minimal frequencies along each dimension
        """
        return 2. * np.pi / self.lside

    def kmax(self):
        """
        Returns maximal frequencies along each dimension
        """
        return np.pi * self.shape() / self.lside

    def shape(self):
        """
        Numpy shape of the jc_datacube
        """
        return 2 ** self.resolution

    def sqd_uniqfreq(self, return_inverse=False, return_counts=False):
        """
        Returns the sorted array of unique frequencies k**2 = sum_i k_i^2
        together with the counts.
        Output is that of np.unique with the corresponding keywords.
        """
        return np.unique(self.sqd_freqmap(), return_inverse=return_inverse, return_counts=return_counts)

    def sqd_freqmap(self, verbose=None):
        """
        Returns the array of squared frequencies, in physical units.
        Same shape than the datacube.
        """
        s = self.shape()
        # First we check if the cube is regular
        if (len(np.unique(s)) == 1 and len(np.unique(self.lside)) == 1):  # regular hypercube
            l02 = Freq(np.arange(s[0]), s[0]) ** 2 * self.kmin()[0] ** 2
            ones = np.ones(s[0])
            if self.ndim == 1: return l02
            vec = [l02]
            for i in xrange(1, self.ndim):
                vec.append(ones)
            l0x2 = outerproducts(vec)
            sqd_freq = np.zeros(s)
            for i in xrange(0, self.ndim):
                sqd_freq += np.swapaxes(l0x2, 0, i)
            return sqd_freq
        # Ok, let's use a different dumb method.
        idc = np.indices(s)
        kmin2 = self.kmin() ** 2
        mapk = kmin2[0] * Freq(idc[0, :], s[0]) ** 2
        for j in xrange(1, self.ndim):
            mapk += kmin2[j] * Freq(idc[j, :], s[j]) ** 2
        return mapk

    def sqd_int_freqmap(self, verbose=None):
        """
        Returns the array of squared frequencies, in physical units.
        Same shape than the datacube.
        """
        s = self.shape()
        # First we check if the cube is regular
        if (len(np.unique(s)) == 1 and len(np.unique(self.lside)) == 1):  # regular hypercube
            l02 = Freq(np.arange(s[0]), s[0]) ** 2
            ones = np.ones(s[0])
            if self.ndim == 1: return l02
            vec = [l02]
            for i in xrange(1, self.ndim):
                vec.append(ones)
            l0x2 = outerproducts(vec)
            sqd_freq = np.zeros(s)
            for i in xrange(0, self.ndim):
                sqd_freq += np.swapaxes(l0x2, 0, i)
            return sqd_freq
        # Ok, let's use a different dumb method.
        idc = np.indices(s)
        mapk = Freq(idc[0, :], s[0]) ** 2
        for j in xrange(1, self.ndim):
            mapk += Freq(idc[j, :], s[j]) ** 2
        return mapk

    def sqd_distmap(self):
        """
        Returns the array of squared distances, in physical units.
        Same shape than the datacube.
        """
        s = self.shape()
        idc = np.indices(s)
        rmin2 = self.rmin() ** 2
        mapk = rmin2[0] * Freq(idc[0, :], s[0]) ** 2
        for j in xrange(1, len(s)): mapk += rmin2[j] * Freq(idc[j, :], s[j]) ** 2
        return mapk

    def fftTH_filter(self, n):
        """
        Returns the rectangular top-hat filter in Fourier space.
        n is an array of int with the same dimension than shape.
        The top hat filter has width 2*n_i + 1 along dimension i.
        At zero has entry Prod_dimensions (2 n_i + 1)
        """
        assert (len(n) == self.ndim), "Inconsistent input"
        vs = []
        shape = self.shape()
        rmin = self.rmin()
        for i in xrange(self.ndim): vs.append(DirichletKernel(n[i], shape[i], d=rmin[i]))
        return outerproducts(vs)

    def fftGauss_filter(self, sR):
        """
        Returns the Gaussian filter in Fourier space. exp(-1/2 \sum k_i^2s2R_i )
        Equal to unity at zero frequency
        """
        assert (len(sR) == self.ndim), "Want one dispersion per dimension"
        vs = []
        shape = self.shape()
        kmin2 = self.kmin() ** 2
        for i in xrange(self.ndim):
            sqdfreqs = kmin2[i] * Freq(np.arange(shape[i]), shape[i]) ** 2
            vs.append(np.exp(-sqdfreqs * (sR[i] ** 2 * 0.5)))
        return outerproducts(vs)


def check_attributes(par, required_attrs):
    attr_ok = [hasattr(par, attr) for attr in required_attrs]
    if not np.all(attr_ok):
        print "# !! required attributes not found :"
        for attr in required_attrs:
            if not hasattr(par, attr):
                print "  ", attr
        assert 0
    return np.all(attr_ok)

# --------------------------
# Some simple-minded utils for verbose mode on :
# --------------------------

def LevelUp(verbose):
    return verbose + (verbose > 0)


def Offset(verbose):
    offset = ' '
    for i in xrange(verbose - 1):
        offset += '  .../'
    return offset


def PrtAndRstTime(verbose, t0):
    print Offset(verbose), "--- %0.2fs ---" % (time.time() - t0)
    return time.time()


def PrtMsg(msg, verbose):
    print Offset(verbose), msg
