try :
    import weave
except:
    try :
        from scipy import weave
    except:
        print "could not import weave"

import numpy as np

def get_xgwg(n,x1 = -1.,x2 = 1.):
    """
    Gauss-Legendre integration function, gauleg, from "Numerical Recipes in C"
    (Cambridge Univ. Press) by W.H. Press, S.A. Teukolsky, W.T. Vetterling, and
    B.P. Flannery
    Given the lower and upper limits of integration x1 and x2, and given n, this
    routine returns arrays x and w of length n, containing the abscissas
    and weights of the Gauss-Legendre n-point quadrature formula.
    """
    header = ["<stdlib.h>","<math.h>"]
    support_code = "#define EPS 3.0e-11 /* EPS is the relative precision. */"
    gauleg = """
    //{
    	int m,j,i;
    	double z1,z,xm,xl,pp,p3,p2,p1;
    	m=(n+1)/2; /* The roots are symmetric, so we only find half of them. */
    	xm=0.5*(x2+x1);
    	xl=0.5*(x2-x1);
    	for (i=1;i<=m;i++) { /* Loop over the desired roots. */
    		z=cos(3.141592654*(i-0.25)/(n+0.5));
    		/* Starting with the above approximation to the ith root, we enter */
    		/* the main loop of refinement by Newton's method.                 */
    		do {
    			p1=1.0;
    			p2=0.0;
    			for (j=1;j<=n;j++) { /* Recurrence to get Legendre polynomial. */
    				p3=p2;
    				p2=p1;
    				p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
    			}
    			/* p1 is now the desired Legendre polynomial. We next compute */
    			/* pp, its derivative, by a standard relation involving also  */
    			/* p2, the polynomial of one lower order.                     */
    			pp=n*(z*p1-p2)/(z*z-1.0);
    			z1=z;
    			z=z1-p1/pp;  /* Newton's method. */
    		} while (fabs(z-z1) > EPS);
    		x[i-1]=xm-xl*z;      /* Scale the root to the desired interval, */
    		x[n-i]=xm+xl*z;  /* and put in its symmetric counterpart.   */
    		w[i-1]=2.0*xl/((1.0-z*z)*pp*pp); /* Compute the weight             */
    		w[n-i]=w[i-1];                 /* and its symmetric counterpart. */
    	}
    //}
    """
    n = int(n)
    w = np.empty(n,dtype = float)
    x = np.empty(n,dtype = float)
    x1 = float(x1)
    x2 = float(x2)
    weave.inline(gauleg, ['x1','x2','x','w','n'], headers=header,support_code=support_code)
    return x,w

def get_Pn(N,x,norm = False):
    """ Legendre Pol. up to order N at points x. norm to get orthonormal Poly. output N + 1,Nx shaped """
    x =  np.array(x)
    Pn = np.ones(x.size)
    if N == 0 : return Pn
    res = np.zeros( (N + 1,x.size))
    Pn1 = x
    res[0,:] = Pn
    res[1,:] = Pn1
    if N == 1 : return res
    for I in xrange(1,N) :
        res[I + 1L,:] = 2.*x*res[I,:] - res[I-1,:] - (x*res[I,:]- res[I-1,:] )/(I + 1.)
    if not norm :
        return res
    return res * np.outer(np.sqrt(np.arange(res.shape[0]) + 0.5),np.ones(x.shape))

def get_rspace(cl,cost):
    """ alpha(cost) =  sum_l (2l + 1)/ 4pi alpha_l P_l(cost) """
    return np.polynomial.Legendre(cl * (2. * np.arange(len(cl)) + 1.)/ (4. * np.pi))(cost)

def get_alphasq(cl):
    """ Legendre coeff of xi(cost) ** 2, given Legendre coeff. of xi. GL quadrature. """
    lmax = len(cl) - 1
    xg,wg = get_xgwg(2 * lmax + 1)
    return (2. * np.pi) * np.dot(get_Pn(2 *  lmax,xg), wg * get_rspace(cl,xg) ** 2)

