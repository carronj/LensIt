import numpy as np
from lensit.clusterlens import constants as const
from scipy.special import sici
from lensit.misc.misc_utils import Freq
from camb import CAMBdata

class profile(object):
    def __init__(self, cosmo:CAMBdata, pname='nfw'):
        r"""Lensing convergence field from a dark matter halo.

            The profile follows the definitions fo Geach and Peacock 2017

            Args:
                cosmo: an instantiated camb.CAMBdata object
                pname: String, could use different halo profile (for now only NFW)
        """
        # TODO: everything is defined in terms of M200c, but we might want to use M500c (~0.7 M200c), or M200m (~1.4 M200c)
        # In that case we should update the concentration mass relation accordingly
        
        self.pname = pname
        self.cosmo = cosmo
        self.h = self.cosmo.Params.H0 / 100  
        chi_cmb = cosmo.conformal_time(0)- cosmo.tau_maxvis
        self.zcmb = cosmo.redshift_at_comoving_radial_distance(chi_cmb)

        if pname == 'nfw':
            self.kappa = self.kappa_nfw

    def kappa_nfw(self, M200, z, R, xmax=None):
        """Convergence profile of a cluster for a NFW density profile
            Args:
                M200: mass (in Msol) of the cluster in a sphere of density 
                    200 times the critical density of the universe
                z: redshift of the cluster
                R: array, distance from the center of the cluster in Mpc 
        """
        return self.sigma_nfw(M200, z, R, xmax=xmax) / self.sigma_crit(z)

    def get_rho_s(self, M200, z, const_c=None):
        c = self.get_concentration(M200, z, const_c)
        return 200* self.rho_crit(z)/3 * c**3 / (np.log(1+c) - c/(1+c))

    def get_rs(self, M200, z, const_c=None):
        return self.get_r200(M200, z) / self.get_concentration(M200, z, const_c)
    
    def get_thetas_amin(self, M200, z, const_c=None):
        return self.r_to_theta(z, self.get_rs(M200, z, const_c))

    def get_r200(self, M200, z):
        """Get r200 in Mpc"""
        mthreshold = 200
        return (M200 * 3 / (4 * np.pi * mthreshold * self.rho_crit(z)))**(1/3)        

    def rho_crit(self, z):
        """The critical density of the universe at redshift z
        in units of :math:`M_{\\odot} / {\\rm Mpc}^3
        """
        return const.rhocrit * (self.cosmo.hubble_parameter(z))**2

    def get_concentration(self, M200, z, const_c=None):
        """Get the concentration parameter as in Geach and Peacock eq. 5.
        M200: In units of solar mass."""
        if const_c is None:
            const_c = 5.71 * (1 + z)**(-0.47) * (M200 / (2E12 / self.h))**(-0.084)

        return const_c

    def sigma_crit(self, z):
        """Critical surface mass density, in Msun/Mpc^2"""
        Dang_OS = self.cosmo.angular_diameter_distance(self.zcmb)
        Dang_OL = self.cosmo.angular_diameter_distance(z)
        Dang_LS = self.cosmo.angular_diameter_distance2(z, self.zcmb)
        return const.c_Mpcs**2 /(4*np.pi*const.G_Mpc3_pMsol_ps2) *  Dang_OS/Dang_OL/Dang_LS
    
    def get_kappa0(self, M200, z, xmax=None):
        """Return the value of the lensing convergence profile for x = 1, i.e. R = rs"""
        rs = self.get_rs(M200, z)
        if xmax is not None:
            assert xmax>1, 'xmax should be larger than 1 (i.e. truncation should be larger than the scale radius rs)'
        return self.kappa(M200, z, rs, xmax=xmax)[0]
    

    def sigma_nfw(self, M200, z, R, xmax=None):
        """Analytic expression for the surface mass desinty of a NFW profile 
        From Equation 7 of Bartelmann 1996
        Args:
            M200: mass (in Msol) of the cluster in a shpere of density 200 times the critical density
            z: redshift of the cluster
            R: distance from the center of the cluster in Mpc
        Returns:
            sigma: surface mass density  in Msun / Mpc^2
        """
        rs = self.get_rs(M200, z)
        rhos = self.get_rho_s(M200, z)
        R = np.atleast_1d(R)
        x = R / rs 
        if xmax is None:
            f = self.fx(x)
        else:
            f = self.gx(x, xmax)
        sigma = 2*rhos*rs*f
        return sigma

    def fx(self, x, tol=6):
        """This integral of the NFW profile along the line of sight is integrated up to infinity
        See Bartelmann 1996 or Wright et al 1999"""
        f = np.zeros_like(x)
        xp = np.where(x>1)
        xo = np.where(np.abs(x-1.) < 10**(-tol)) 
        xm = np.where(x<1)
        f[xp] = (1 - 2/np.sqrt(x[xp]**2 - 1) * np.arctan(np.sqrt((x[xp] - 1)/(x[xp] + 1))))/(x[xp]**2-1)
        f[xm] = (1 - 2/np.sqrt(1 - x[xm]**2) * np.arctanh(np.sqrt((1 - x[xm])/(1 + x[xm]))))/(x[xm]**2-1)
        f[xo] = 1/3
        return f


    def gx(self, x, xmax, tol=5):
        """We apply a cutoff in the halo profile for x>xmax, i.e. R>rs*xmax
        See equation 27 of Takada and Jain 2003"""
        g = np.zeros_like(x)
        xp = np.where(np.logical_and(x>1, x<xmax))
        xo = np.where(np.abs(x-1.) < 10**(-tol))       
        xm = np.where(np.logical_and(x<1, x>0))
        g[xp] = (np.sqrt(xmax**2 - x[xp]**2)/(1+xmax) - 1/np.sqrt(x[xp]**2 - 1) * np.arccos((x[xp]**2 + xmax) / (x[xp] * (1+xmax))) )/(x[xp]**2-1)
        g[xm] = (np.sqrt(xmax**2 - x[xm]**2)/(1+xmax) - 1/np.sqrt(1-x[xm]**2) * np.arccosh((x[xm]**2 + xmax) / (x[xm] * (1+xmax))) )/(x[xm]**2-1)
        g[xo] = np.sqrt(xmax**2 - 1)/(3*(1+xmax)) * (1+ 1/ (1+xmax))
        return g



    def rho_nfw(self, M200, z, r):
        r"""Navarro Frenk and White density profile 
            Args:
                M200: mass (in Msol) of the cluster in a sphere of density 
                    200 times the critical density of the universe
                z: redshift of the cluster
                r: array, distance from the center of the cluster in Mpc 
        """
        rho_s = self.get_rho_s(M200, z)
        rs = self.get_rs(M200, z)
        rho = rho_s / ((r/rs)*(1+r/rs)**2)
        return rho

    def sigma_int(self, M200, z, R, xmax = 100, npoints=1000):
        """Integrate density over the line of sight 
        Args:
            M200: M200c mass of the cluster in Msol
            z: redshift of the cluster
            R: distance from the center of the cluster in Mpc
            xmax: maximum radius of the integration in factors of rs
            npoints: number of points in the integration
        Returns:
            sigma: surface mass density  in Msun / Mpc^2
        """
        rs = self.get_rs(M200, z)
        # assert rs*xmax >= np.max(R), "Need to increase integration limit"
        assert R[0] > 0, "Can't integrate on center of NFW profile, please give R>0"
        R = np.atleast_1d(R)
        sigma = np.zeros_like(R)
        for i, iR in enumerate(R):
            if iR>rs*xmax:
                sigma[i] = 0
            else:
                r_arr = np.geomspace(iR, xmax*rs, num = npoints)
                dr_arr = np.diff(r_arr, axis=0)
                # I skip the point r= R because the integrand is not defined, should check if corect?
                r_arr = r_arr[1:]
                rho_arr = self.rho_nfw(M200, z, r_arr)
                sigma[i] = np.trapz( 2 * r_arr * rho_arr / np.sqrt(r_arr**2 - iR**2), r_arr)
        return sigma

    def analitic_kappa_ft(self, M200, z, xmax, ell, const_c=None):
        """Analytic Fourier transform of the convergence fiels for a NFW profile
            from Oguri&Takada 2010, Eq.28
            Oguri and Takada assumes a truncation at r_vir, so c=r_vir/rs, 
            but we can use any truncation radius xmax=r_trunc/rs"""
        c = self.get_concentration(M200, z, const_c)
        mu_nfw = np.log(1. + c) - c / (1. + c)
        rs = self.get_rs(M200, z)
        chi = self.cosmo.comoving_radial_distance(z)
        k = ell / chi
        x = ((1. + z) * k * rs)
        x = np.asarray(x, dtype=float)
        u0 = np.empty_like(x, dtype=float)
        nz = x != 0
        if np.any(nz):
            Six, Cix = sici(x[nz])
            Sixpc, Cixpc = sici(x[nz] * (1. + xmax))
            Sidiff = Sixpc - Six
            Cidiff = Cixpc - Cix
            u0[nz] = np.sin(x[nz]) * Sidiff + np.cos(x[nz]) * Cidiff - np.sin(x[nz] * xmax) / (x[nz] * (1. + xmax))
        u0[~nz] = np.log(1. + xmax) - xmax / (1. + xmax)
        ufft = 1. / mu_nfw * u0

        kappaft = M200 * ufft * (1+z)**2 /chi**2 / self.sigma_crit(z)
        return kappaft

    def kappa_theta(self, M200, z, theta, xmax=None):
        """Get the convergence profile
            Args:
                M200: Cluster mass defined as in a sphere 200 times the critical density 
                z: redshift of teh cluster
                theta: amgle in arcminute
                
            Returns:
                kappa: convergence profile
        """
        R = self.theta_amin_to_r(z, theta)
        return self.kappa(M200, z, R, xmax=xmax)

    def r_to_theta(self, z, R):
        """Convert a transverse distance into an angle
            Args:
                R: transverse distance in Mpc
                z: redshift of the cluster
            
            Returns:
                theta: angle on the sky in arcmin, with R = Dang(z) * theta"""

        Dang = self.cosmo.angular_diameter_distance(z)
        theta_rad = R / Dang
        return theta_rad *180*60/np.pi

    def theta_amin_to_r(self, z, theta_amin):
        """Convert a transverse distance into an angle
        Args:
            theta: angle on the sky in arcmin
            z: redshift of the cluster
        
        Returns:
            R: transverse distance in Mpc, with R = Dang(z) * theta"""

        Dang = self.cosmo.angular_diameter_distance(z)
        R = Dang * theta_amin * np.pi/180/60
        return R


    def x_to_theta_amin(self,M200, z, x, const_c=None):
        """Angle substended at chararcteric scale x = r/rs

            Args:
                M200: cluster M200 mass in solar masses (?)
                z: cluster redshift
                x:  dimensionless R / Rs

        """
        return self.r_to_theta(z, x * self.get_rs(M200, z, const_c=const_c))

    def pix_to_theta(self, x, y, dtheta, c0):
        """Return the angle between the center of the map and the pixel (x,y)
        Args:
            x, y: pixel coordinates of the map
            dtheta: size 2 array, physical size of the pixels (dx, dy) 
            c0: size 2 array, central pixel of the map
        Returns:
            angle between the pixel and the center 
            """
        return np.sqrt((x-c0[0])**2 * dtheta[0]**2 + (y-c0[1])**2 * dtheta[1]**2)


    def kappa_map(self, M200, z, shape, lsides, xmax=None, nsub=16, center='center'):
        """Get the convergence map of the cluster
            Args:
                M200: Cluster mass defined as in a sphere 200 times the critical density
                z: redshift of the cluster
                shape(2-tuple): pair of int defining the number of pixels on each side of the box
                lsides(2-tuple): physical size (in radians) of the box sides
                xmax: cutoff scale in factors of rs
                nsub: linear sub-pixel resolution used to average kappa over the
                    central pixel (which is otherwise divergent since theta=0)
                center: if 'corner', the center of the cluster is in the corner of the map (for periodicity),
                    if 'center' the center of the cluster is in the center of the map
            Returns:
                kappa_map: numpy array defining the convergence field
        """
        dtheta_x = lsides[0]/shape[0] * 180/np.pi*60
        dtheta_y = lsides[1]/shape[1] * 180/np.pi*60


        if center == 'corner':
            # Center of the cluster in the corner of the patch, for periodicity
            x0 = 0
            y0 = 0

            if shape[0] % 2 == 0 and shape[1] % 2 == 0:
                X, Y =  np.meshgrid(np.concatenate((np.arange(0,shape[0]//2), np.arange(-shape[0]//2,0))), np.concatenate((np.arange(0,shape[1]//2), np.arange(-shape[1]//2,0))))
            else:
                assert 0, "pixel size is not right"

        elif center =='center':
            # Center of the cluster in the center of the patch (periodic if truncated 
            # profile, almost periordic in any case since kappa is small at the edges 
            # of the patch)
            x0 = shape[0]//2
            y0 = shape[1]//2
            X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        
        theta_amin = self.pix_to_theta(X, Y, (dtheta_x,dtheta_y),  (x0, y0))
        kappa = self.kappa_theta(M200, z, theta_amin, xmax=xmax)

        # Average kappa over the central pixel to avoid the theta=0 divergence.
        # Cell-centred subsamples keep every offset away from the singular point.
        sub = (np.arange(nsub) + 0.5) / nsub - 0.5
        sX, sY = np.meshgrid(sub, sub)
        sub_theta = np.sqrt((sX * dtheta_x)**2 + (sY * dtheta_y)**2)
        kappa[y0, x0] = np.mean(self.kappa_theta(M200, z, sub_theta.ravel(), xmax=xmax))
        return kappa


    def kmap2deflmap(self, kappamap, shape, lsides):
        """Transforms a kappa map into a deflection map 
            Args: 
                kappamap: numpy array defining the convergence field
                shape(2-tuple): pair of int defining the number of pixels on each side of the box
                lsides(2-tuple): physical size (in radians) of the box sides
            Returns:
                (dx, dy): 2 numpy arrays defining the deflection field
        """

        rfft_kappa = np.fft.rfft2(kappamap)
        rs = rfft_kappa.shape

        ky = (2. * np.pi) / lsides[0] * Freq(np.arange(shape[0]), shape[0])
        ky[int(shape[0] / 2):] *= -1.
        kx = (2. * np.pi) / lsides[1] * Freq(np.arange(rs[1]), shape[1])
        KX, KY = np.meshgrid(kx, ky)
        k2 = KX**2 + KY**2
        dx_lm = np.zeros_like(rfft_kappa, dtype=complex)
        dy_lm = np.zeros_like(rfft_kappa, dtype=complex)
        np.divide(2 * rfft_kappa * 1.j * KX, k2, out=dx_lm, where=k2 != 0)
        np.divide(2 * rfft_kappa * 1.j * KY, k2, out=dy_lm, where=k2 != 0)

        dx = np.fft.irfft2(dx_lm, shape)
        dy = np.fft.irfft2(dy_lm, shape)
        return (dx, dy)



    def phimap2kappamap(self, phimap, shape, lsides):
        """Transforms a phi map into a kappa map 
            Args: 
                phimap: numpy array defining the convergence field
                shape(2-tuple): pair of int defining the number of pixels on each side of the box
                lsides(2-tuple): physical size (in radians) of the box sides
            Returns:
                kappa_map: numpy array of the convergence field
        """

        rfft_phi = np.fft.rfft2(phimap)
        rs = rfft_phi.shape

        ky = (2. * np.pi) / lsides[0] * Freq(np.arange(shape[0]), shape[0])
        ky[int(shape[0] / 2):] *= -1.
        kx = (2. * np.pi) / lsides[1] * Freq(np.arange(rs[1]), shape[1])
        KX, KY = np.meshgrid(kx, ky)

        rfft_kappa = 1/2 * (KX**2 + KY**2) * rfft_phi

        return np.fft.irfft2(rfft_kappa, shape)
