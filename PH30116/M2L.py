#!/usr/bin/env python

"""
Simulate sample of galaxy redshifts, luminosities, stellar masses and associated stellar mass uncertainty
"""

import numpy as np
import pylab
from astropy.cosmology import WMAP9 as cosmo

Mpc=3.086e24 # cm2
flux_limit = 10.**-16 # erg/cm2/s
Msol = 1.998e33 # grams
Lsol = 3.828e33 # erg/s

def Mlim(z,M2L):
    """
    Given some mass-to-light ratio, M2L, and redshift, z, use the flux_limit=10**-16 erg/cm2/s defined above to determine the corresponding stellar mass limit
    """
    Mlimit = (4. * np.pi * (cosmo.luminosity_distance(z).value * Mpc)**2 * flux_limit * M2L)/Msol
   
    return Mlimit

def sample_sim(n=500,Msigma=1.0,M2Llo=2,M2Lhi=10):
    """
    Simulate sample of galaxy redshifts, luminosities, stellar masses and associated stellar mass uncertainty

    Arguments:
    n: number of data points to simulate
    Msigma: sigma of logarithmic mass distribution from which select random stellar mass values. The larger the value, the broader the range of simulated stellar mass values (default=1)
    M2Llo: the lower limit on the mass-to-light ratio selected by random (default=2)
    M2hi: the upper limit on the mass-to-light ratio selected by random (default=10)
    """
    mbin = 9.5
    redshift,Mass,Lumin,Merr = np.array([]),np.array([]),np.array([]),np.array([])
    for zmax in [1,2,3,4]:
        for i in np.arange(n):
            m = 10.**(np.random.normal(mbin,Msigma))
            #munc = m*(np.random.uniform(0.2,0.8))
            munc = 2/np.log10(m)
            z = np.random.uniform(zmax-1,zmax)
            M2L = np.random.uniform(M2Llo,M2Lhi)
            if (m>Mlim(z,M2L)) and (m<10**12.5):
                redshift = np.append(redshift,z)
                Mass = np.append(Mass,m)
                Merr = np.append(Merr,munc)
                M2L = np.random.normal(20,19)
                while (M2L<0):
                    M2L = np.random.normal(20,19)
                Lumin = np.append(Lumin,(m*Msol/M2L))
        mbin+=0.4
        Msigma-=0.1

    return redshift,np.log10(Mass),Merr,np.log10(Lumin/Lsol)
