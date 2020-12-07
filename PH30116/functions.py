import pandas as pd 
import numpy as np 
import glob
import pylab
mykepler=2
from astropy.io import fits
from scipy.signal import medfilt
import matplotlib.pyplot as plt

def import_and_flatten_lightcurve():
    "function to combine all the data from different files and tidy up (sort data, remove nans and outliers). Flatten lightcurve     by dividing through by the median filter"
    'returns:  sorted dataframe containing the time, original flux, error and new normalised, flattened flux lightcurve' 
    
    alldata = pd.DataFrame()
    #looping through each dataset - normalises the flux and the error in the flux 
    for lcfile in glob.glob('Data/Object%slc/kplr*.fits' %(mykepler)):
        tmp = fits.open(lcfile)
        tmptime = (tmp[1].data['TIME'])
        tmpflux = (tmp[1].data['PDCSAP_FLUX'])
        normalflux = tmpflux/np.nanmedian(tmp[1].data['PDCSAP_FLUX'])
        tmperror = (tmp[1].data['PDCSAP_FLUX_ERR'])/np.nanmedian(tmp[1].data['PDCSAP_FLUX'])
    
        #combining all the separate datasets into one dataframe 
        df = pd.DataFrame({
            'time': tmptime,
            'flux': normalflux,
            'error':tmperror
             })
        alldata = alldata.append(df) 

    #sort values in chronological order 
    alldata.sort_values(by='time', inplace=True)
    
    #remove any nan values from the dataset
    alldata.dropna(inplace=True)

    #removes any visible outliers 
    alldata = alldata[alldata['flux']<1.001]
    alldata = alldata[alldata['flux']>0.998]

    #calculates the median filter
    median = medfilt(alldata['flux'], kernel_size=101) ##kernel_size is the size of the window over which the median is calculated 
    #new flux is the cleaned and flattened lightcurve - outliers removed and flattened by divided by the median flux filter 
    alldata['new flux'] = alldata['flux']/median
    
    
    #print the first 5 lines of the dataframe to check it looks reasonable
    return alldata
    #plot the flattened lightcurve 
    return pylab.plot(alldata['time'], flattened_lightcurve)



def fold_lightcurve(filename, period,*args, **kwargs):
    obj_name = kwargs.get('obj_name', None)
    outdata = kwargs.get('output_file', 'folded_lc_data.csv')
    plotname = kwargs.get('plot_file', 'folded_lc.pdf')
    
    ## Read in the data. Should be comma separated, header row (if present) should have # at the start.
    data = pd.read_csv(filename, usecols=[0,1,2], names=('JD', 'mag', 'error'), comment='#')
    if len(data.columns) < 3:
        print("File format should be \n\
              (M)JD, magnitude, uncertainty\n")
        exit(1)
    ## Folding the lightcurve:
    ## Phase = JD/period - floor(JD/period)
    ## The floor function is there to make sure that the phase is between 0 and 1.
    
    data['Phase'] = data.apply(lambda x: ((x.JD/ period) - np.floor(x.JD / period)), axis=1)
    
    ## concatenating the arrays to make phase -> 0 - 3
    ## This makes it easier to see if periodic lightcurves join up as expected
    
    #phase_long = np.concatenate((data.Phase, data.Phase + 1.0, data.Phase + 2.0))
    phase_long = np.concatenate((data.Phase, data.Phase, data.Phase))
    mag_long = np.concatenate((data.mag, data.mag, data.mag))
    err_long = np.concatenate((data.error, data.error, data.error))
    
    data.sort_values(by='Phase', inplace=True)
    data.to_csv(outdata, header=True, index=False, sep=',')

 
    return data

stellar_info = pd.DataFrame({'radius':1.065, 'radius err':0.02, 'mass':0.961, 'mass err':0.025, 'temp':5657}, index=[0])

def planet_radius(popt, pcov):
    'Using the optimised values for base flux and transit flux from the lightcurve the formula below is used to calculate planet      radius: change in flux/flux = planet radius/stellar radius'
    'Inputs: popt and pcov from the curve fit function on the transit'
    'Returns: radius of the planet and associated error'
    
    #Solar radius in metres
    solar_radius = 6.96e8
    #Jupiter radius in metres
    jupiter_radius = 71.492e6
    #Earth radius in metres 
    earth_radius = 6.371e6
    
    star_radius = stellar_info['radius'][0] * solar_radius
    star_radius_unc = stellar_info['radius err'][0] * solar_radius
    
    flux_base = popt[0]
    flux_transit = popt[1]
    
    uncertainty = np.sqrt(np.diag(pcov))
    
    flux_base_unc = uncertainty[0]
    flux_transit_unc = uncertainty[1]
    
    planet_radius = np.sqrt((star_radius**2) * ((flux_base-flux_transit)/flux_base)) #in metres 
    planet_radius_mj = planet_radius/jupiter_radius #in Jupiter masses
    planet_radius_me = planet_radius/earth_radius  #in earth masses
    
    #assume uncertainty in flux_transit >> flux_base  (units solar radii)
    planet_radius_unc_mj = planet_radius_mj * (np.sqrt((flux_transit_unc / flux_transit)**2 + (star_radius_unc /             star_radius)**2))  #jupiter masses 
    planet_radius_unc_me = planet_radius_me * (np.sqrt((flux_transit_unc / flux_transit)**2 + (star_radius_unc / star_radius)**2))  #earth masses

    print("Radius of planet in Jupiter masses is %.4e +/- %.4e" % (planet_radius_mj,planet_radius_unc_mj))
    print("Radius of planet in Earth masses is %.4e +/- %.4e" % (planet_radius_me,planet_radius_unc_me))
        
    
def semi_major_axis(period):
    "calculate semi major axis of planet (a) using Kepler's third law"
    'Inputs:  period of planet in DAYS'
    'Returns: semi-major axis of planet in astronomical units'
    
    period_secs = period * 24 * 3600
    G = 6.67e-11
    solar_mass = 1.989e30
    #assume mass of star >> mass of planet 
    mass = 0.961 *solar_mass
    AU = 1.496e11
    
    a = ((period_secs**2) * G * mass * (1/4*np.pi))**(1/3)
    return a/AU

def planet_temperature(semi_major_axis):
    'function first calculates the luminosity of the star, and then uses this to calculate the temperature of the planets            surface'
    'Assumptions: there is no albedo afffect on the planet; there is no atmosphere altering temperature'
    'Inputs: semi-major axis of planet in astronomical units'
    'Returns:  temperature of planet in Kelvin'
    
    stellar_temp = stellar_info['temp'][0]
    
    #Stefan-Boltzmann constant 
    s_const = 5.67e-8
    #Solar radius 
    solar_radius = 6.96e8
    #1 AU 
    AU = 1.496e11
    
    stellar_luminosity = 4*np.pi * (stellar_info['radius'][0]*solar_radius)**2 * s_const * stellar_temp**4
    
    planet_temp = (stellar_luminosity/(16 * np.pi * s_const * (semi_major_axis*AU)**2))**(1/4)
    
    return planet_temp