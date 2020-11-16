#!/usr/bin/env python

"""
Linear model fits and determining confidence levels for some given 2D distribution 

"""

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.interpolate import interpn

def linfit(xdata, a):
    """
    Apply linear fit to data (with intercept=0)

    Arguments:
    xdata: 1D array containing x-axis data points
    a: model parameter to linear fit defining slope of relation
    """
    return xdata*a

def conflevels(x,y,nbins,confints=[0.99,0.95,0.68]):
    """
    Determine contour regions containing a certain input fraction of the population
    Argument:
    x: 1D array containing x-axis data points
    y: 1D array containing y-axis data points
    confints: 3-element array containing confidence regions to be calculated (default set to [0.99, 0.95, 0.69])
    """

    # Make a 2d normed histogram
    H,xedges,yedges=np.histogram2d(x,y,bins=nbins,normed=True)

    norm=H.sum() # Find the norm of the sum
    # Set contour levels
    contour1=confints[0]
    contour2=confints[1]
    contour3=confints[2]

    # Set target levels as percentage of norm
    target1 = norm*contour1
    target2 = norm*contour2
    target3 = norm*contour3

    # Take histogram bin membership as proportional to Likelihood
    # This is true when data comes from a Markovian process
    def objective(limit, target):
        w = np.where(H>limit)
        count = H[w]
        return count.sum() - target

    # Find levels by summing histogram to objective
    level1= optimize.bisect(objective, H.min(), H.max(), args=(target1,))
    level2= optimize.bisect(objective, H.min(), H.max(), args=(target2,))
    level3= optimize.bisect(objective, H.min(), H.max(), args=(target3,))

    # For nice contour shading with seaborn, define top level
    level4=H.max()
    levels=[level1,level2,level3]

    return levels

def density_scatter(x , y, ax=None, sort=True, bins=20,):
    """
    Scatter plot colored by 2d histogram
    """
    #if ax is None :
    #    fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    #ax.scatter( x, y, c=z, **kwargs )
    return z
