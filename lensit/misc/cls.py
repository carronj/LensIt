import time

import healpy as hp
import numpy as np
from scipy import interpolate
import misc_utils as utils


def legendreP(N, x):
    #
    # PURPOSE : returns the values of prefac * the Legendre polynomials
    #           up to order N, at the argument x
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


class Cl_lminlmax():
    """
    Class for 2d tabulated power spectra (typically from CAMB)
    entries not given in the input are set to zero.
    The array goes from lmin  to lmax.
    Input ell must be contiguous and of the right size
    """

    def __init__(self, ell, Cl, verbose=True):
        self.lmax = ell[-1]
        self.lmin = ell[0]
        self.Cl = Cl
        self.verbose = verbose
        assert (len(Cl) == self.lmax - self.lmin + 1), "Input not understood"
        assert (np.all(self.ell() == ell)), "Input not understood"

    def __add__(self, x):
        if np.isscalar(x):
            Clc = self.clone()
            Clc.Cl += x
            return Clc
        elif isinstance(x, Cl_lminlmax):
            # Extends with zeros if the ranges do not match
            lmin = min([self.lmin, x.lmin])
            lmax = max([self.lmax, x.lmax])
            Cl = np.zeros(lmax - lmin + 1)
            Cl[self.ell() - lmin] += self.Cl
            Cl[x.ell() - lmin] += x.Cl
            return Cl_lminlmax(np.arange(lmax - lmin + 1) + lmin, Cl)
        else:
            assert (0), "Operation not supported"

    def __mul__(self, x):
        if np.isscalar(x):
            Clc = self.clone()
            Clc.Cl *= x
            return Clc
        elif isinstance(x, Cl_lminlmax):
            # Extends with zeros if the ranges do not match
            lmin = min([self.lmin, x.lmin])
            lmax = max([self.lmax, x.lmax])
            Cla = Cl_lmax(self.ell(), self.Cl)
            Cla.extend_lmax(lmax, value=0.)
            Clb = Cl_lmax(x.ell(), x.Cl)
            Clb.extend_lmax(lmax, value=0.)
            return Cl_lminlmax(np.arange(lmax - lmin + 1) + lmin, Cla.Cl[lmin:] * Clb.Cl[lmin:])
        else:
            assert (0), "Operation not supported"

    def __getitem__(self, item):
        Clc = Cl_lmax(self.ell(), self.Cl)
        return Clc.Cl[item]

    def ell(self):
        return np.arange(self.lmax - self.lmin + 1, dtype=int) + self.lmin

    def clone(self):
        return Cl_lminlmax(self.ell(), self.Cl, verbose=self.verbose)

    def plot_Dl(self, **kwargs):  # quick plot on linear scales
        l = self.ell()
        import pylab as pl
        pl.figure('Power spectrum')
        pl.xlabel('$\ell$')
        pl.ylabel('$D_\ell$')
        pl.loglog(l, self.Cl * l * (l + 1) / (2. * np.pi), **kwargs)

    def plot_Cl(self, **kwargs):  # quick plot on linear scales
        l = self.ell()
        import pylab as pl
        pl.figure('Power spectrum')
        pl.xlabel('$\ell$')
        pl.ylabel('$C_\ell$')
        pl.loglog(l, self.Cl, **kwargs)

    def extend_lmax(self, lmax, value=0.):  # Padding with zeroes
        lmax = np.int_(np.round(lmax))
        if lmax > self.lmax:
            self.Cl = np.append(self.Cl, value * np.ones(lmax - self.lmax))
            self.lmax = lmax

    def extend_lmin(self, lmin, value=0.):  # Padding with zeroes
        lmin = np.int_(np.round(lmin))
        if lmin >= 0 and lmin < self.lmin:
            self.Cl = np.append(value * np.ones(self.lmin - lmin), self.Cl)
            self.lmin = lmin

    def realspace_variance(self):
        return np.sum((2. * self.ell() + 1.) * self.Cl) / (4. * np.pi)

    def Cl_log(self, NSIDE=2048, lmax=None):
        """
        Returns the spectrum of the log field assuming lognormal statistics :
        This produces a fake map with map(n) = xi(cost), take the log and get the new alm.
        Input parameter NSIDE is the healpix res of the fake map. (lmax \sim 3*NSIDE -1)
        """
        if self.verbose and self.lmin > 0:
            utils.PrtMsg("Filling ell (0 : " + str(self.lmin) + ") with lowest l entry.", self.verbose)
        Clc = Cl_lmax(self.ell(), self.Cl)
        Clc.Cl[0:self.lmin] = self.Cl[0]
        alm = 1j * np.zeros(self.lmax + 1) + Clc.Cl
        w = np.sqrt((2. * Clc.ell() + 1.) / 4. / np.pi)  # weights -> sum_l Cl/w_l Y_l0(cost) = xi(n)
        # This creates a map xi with xi(hat n) = xi(cos t)
        print "Generating NSIDE " + str(NSIDE) + " xi map"
        xi = hp.alm2map(w * alm, NSIDE, lmax=self.lmax, mmax=0, verbose=False)
        assert (np.all(xi > -1.)), "Procedure ill-defined"
        alm_log = hp.map2alm(np.log(1. + xi), lmax=lmax, mmax=0)
        alm_log /= np.sqrt((2. * np.arange(len(alm_log)) + 1.) / 4. / np.pi)
        if not (np.all(alm_log.real >= 0.)):
            "Warning , negative values in Plog"
        return Cl_lminlmax(np.arange(len(alm_log)), alm_log.real, verbose=self.verbose)

    def laplacian(self):
        """
        :return: A Cl instance corresponding to the spherical Laplacian
        of the original field. e.g. phi-> 2 * kappa
        """
        ell = self.ell()
        return Cl_lminlmax(ell, self.Cl * ell ** 2 * (ell + 1) ** 2)

    def simulate_GaussianSky(self, NSIDE, seed=None):
        """
        Returns a healpy sky map from the Cl with synfast
        """
        if self.lmin > 0:
            if self.verbose: utils.PrtMsg("Filling ell (0 : " + str(self.lmin) + ") with zeros.", self.verbose)
            Cl = Cl_lmax(self.ell(), self.Cl).Cl  # In order to fill in the missing ells
        else:
            Cl = self.Cl
        if seed is not None: np.random.set_state(seed)
        return hp.synfast(Cl, NSIDE, new=True)

    def simulate_LognormalSky(self, NSIDE):
        """
        Returns a healpy sky map obeying lognormal statistics
        with the input delta Cl.
        Ln field has mean unity
        """
        # Get the spectrum of the log : (Use higher Nside to maybe avoid disc. effects.
        Cl_log = self.Cl_log(2 * NSIDE, lmax=3 * NSIDE - 1)
        map = Cl_log.simulate_GaussianSky(NSIDE) - 0.5 * Cl_log.realspace_variance()
        #  :zero mean Gaussian map -0.5s2A
        return np.exp(map)

    def simulate_PoissonLognormalSky(self, NSIDE, NGal_per_sqdeg):
        if self.verbose:
            t0 = time.time()
            utils.PrtMsg("Generating Lognormal field :", self.verbose)
        rho = self.simulate_LognormalSky(NSIDE)
        rate = hp.nside2pixarea(NSIDE, degrees=True) * NGal_per_sqdeg
        if self.verbose:
            t0 = utils.PrtAndRstTime(self.verbose, t0)
            utils.PrtMsg("Poisson sampling and returning : ", self.verbose)
        return np.random.poisson(rho * rate, size=rho.size)

    def simulate_MultinomialLognormalSky(self, NSIDE, Npts_per_sqdeg):
        if self.verbose:
            t0 = time.time()
            utils.PrtMsg("Generating Lognormal field :", self.verbose)
        rho = self.simulate_LognormalSky(NSIDE)
        if self.verbose:
            t0 = utils.PrtAndRstTime(self.verbose, t0)
            utils.PrtMsg("Multinomial sampling and returning: ", self.verbose)
        Ntot = hp.nside2pixarea(NSIDE, degrees=True) * Npts_per_sqdeg * hp.nside2npix(NSIDE)
        return np.random.multinomial(Ntot, rho / np.sum(rho))

    def get_two_flatsky_sim(self, res):
        """
        simulate a Gaussian on a periodic square of 4pi volume.
        Number of points on each side is 2**res.
        """
        N = 2 ** res
        # build frequency maps
        Cl0 = self.clone()
        Cl0.extend_lmin(0, value=0.)
        freqmap = np.outer(utils.Freq(np.arange(N), N) ** 2, np.ones(N, dtype=int))
        freqmap = np.sqrt(freqmap + np.transpose(freqmap)).flatten()
        minfreq = np.sqrt(np.pi)  # (2 pi  / sqrt(4 pi))
        nbar = N * N / 4. / np.pi
        spl = interpolate.InterpolatedUnivariateSpline((Cl0.ell() - 0.5) / minfreq, N * np.sqrt(Cl0.Cl * nbar),
                                                       ext='zeros')
        map = 1j * np.random.normal(size=(N, N)) + np.random.normal(size=(N, N))
        map *= spl(freqmap).reshape(N, N)
        map = np.fft.ifft2(map)
        return [map.real, map.imag]

    def to_realspace(self, cos_t):
        """
        Performs the Legendre transform to get the real space 2pcf.
        """
        cos_t = np.array(cos_t)
        if self.lmin > 0:
            if self.verbose: utils.PrtMsg("Filling ell (0 : " + str(self.lmin) + ") with zeros.", self.verbose)
            Cl_temp = Cl_lmax(self.ell(), self.Cl)  # In order to fill in the missing ells
            Cl = Cl_temp.Cl
            ell = Cl_temp.ell()
        else:
            Cl = self.Cl
            ell = self.ell()

        Clell2_4pi = Cl * (2. * ell + 1.) / (4. * np.pi)
        return np.polynomial.legendre.legval(cos_t, Clell2_4pi)


class Cl_lmax(Cl_lminlmax):
    """
    The array in this subclass always goes from lmin = 0 to lmax.
    Fills unspecified values of Cls with zeros.
    """

    def __init__(self, ell, Cl, verbose=True):
        assert (len(ell) == len(Cl)), "Input not understood"
        self.lmin = 0
        self.lmax = np.max(ell)
        self.Cl = np.zeros(self.lmax + 1)
        self.Cl[ell] = Cl
        self.verbose = verbose


class Cls_lminlmax():
    """
    Class for Cl spectral matrices. Similar to Cl_lminlmax
    expect for the matrix caracter. The input is checked
    for semi positivity of each ell matrix.
    Give keys for description
    """

    def __init__(self, ell, Cls, nCls, keys=None, verbose=True):
        self.nCls = nCls
        self.lmax = ell[-1]
        self.lmin = ell[0]
        self.Cls = Cls
        if keys is None: keys = list(str(i) for i in np.arange(nCls))
        self.dict = {}
        assert (len(keys) == self.nCls), "Inconsistent input"
        for i, key in zip(np.arange(self.nCls), keys):
            self.dict[key] = i

        assert (nCls > 1), "Try Cl_lminlmax instead"
        assert (Cls.shape == (nCls, nCls, len(ell))), "Input not understood"
        assert (np.all(self.ell() == ell)), "Input not understood"
        if verbose: print "Checking positivity of all spectral matrices "
        for l in xrange(len(ell)):
            assert (np.all(np.linalg.eigvals(self.getCls(l)) >= 0)), "Inconsistent input"

    def ell(self):
        return np.arange(self.lmax - self.lmin + 1) + self.lmin

    def getCls(self, ell):
        return self.Cls[:, :, ell - self.lmin]

    def getCl(self, ell, keyA, keyB=None):
        if keyB is None: keyB = keyA
        return self.Cls[self.dict[keyA], self.dict[keyB], ell - self.lmin]
