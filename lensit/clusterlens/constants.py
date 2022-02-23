import numpy as np
# Some useful constants

# Solar mass 
Msol_kg = 1.98840987e+30 # in kg

# 1 Megaparsec in meter 
Mpc_m = 30856775814671914*1e6


# Speed of light in Mpc / s
c_Mpcs =  299792458 / Mpc_m


# Gravitational constant 

G_m3_pkg_ps2 = 6.67430 *1e-11 # In m^3 / kg / s^2

G_Mpc3_pMsol_ps2 = G_m3_pkg_ps2 / Mpc_m**3  * Msol_kg  # In Mpc^3 / Msun / s^2

G_Mpc_km2_pMsol_ps2 = G_m3_pkg_ps2 / Mpc_m /1000**2 * Msol_kg # In Mpc km^2 / Msun / s^2


# Critical density of the universe 3/(8 pi G)
rhocrit = 3/(8*np.pi*G_Mpc_km2_pMsol_ps2) # In units of M_sun s2 / Mpc / km2 


amin_to_rad = np.pi/180/60 
