"""Spectra Methods"""
import os
import sys

import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.constants import h, c, e



def find_near(a, Near):
    """
    find_near(x, Near)
    
    Returns
    ___________________________________________________________________________ 
    Int
        The "x" array position with the nearest value to "Near" value
    """
    a = np.asarray(a)
    nearpos = (np.abs(a-Near)).argmin()
    return nearpos

def array_region(a, *intervals, index=False, dim=0):
    """
    Returns the selected region selected with the *intervals pos variable

    Parameters
    ___________________________________________________________________________   
    a: array_like
    intervals: np.ndarray
                like interval1, interval2... intervaln ,where interval are 
                has the form [a1, a2] a2>a1 or  *[[a1,a2], [b1,b2]]...
    index: bool
           True: intervals are the position array elementes
           False: intervals are values in array 
    dim: int, default 0
            In case you use index=False choose what dimension represent
            the intervals values.
    Returns
    ___________________________________________________________________________ 
    res:
        The concatenated array with intervasl chosen.

    """
    def __region(a, begin, end):
        if begin<=end: 
            return a[begin:end]
        else:
            return a[begin:end:-1]

    a = np.asarray(a)
    for i, interval in enumerate(intervals):
        if index: begin, end = a[interval[0], interval[1]]
        else : begin, end= (find_near(a[:, dim], interval[0]),
                            find_near(a[:, dim], interval[1])
                            )
        if i==0: region = __region(a, begin, end)
        else: region = np.concatenate((region,__region(a, begin, end)))
    return region

def nm_to_ev(unit):
    np.seterr(divide='ignore', invalid='ignore')  # Suppress divide-by-zero warnings
    try:
        to_unit = h * c / (unit * e * 1e-9)
    except ZeroDivisionError as err:
        print('nanometers or eV cant be zero:', err)
        return np.nan
    finally:
        np.seterr(divide='warn', invalid='warn')  # Reset warning behavior
    return np.where(unit != 0, to_unit, np.nan)

###############################################################################
#MODELS
###############################################################################
def linear_model(x, m=1.0, b=0.0): 
    """Linear model
    linear_model(x , m = slope, b= intercept)
    return mx+b 
    """
    return x*m + b

def sommerfel_broadening(x, amplitude, center, sigma, rydberg):
    return ((amplitude/(1+np.exp((-x+center)/sigma)))
            *(2/(1+np.exp(-2*np.pi*np.sqrt(rydberg/abs(x-center))))))
###############################################################################
#Fits
###############################################################################

def fit_baseline(spectra, baseline, *intervals,
        model= linear_model, parameters= (1.0, 0.0), index = False, dim=0):
    """
    Return the optimal baseline for a spectra
 
    Parameters:
    ___________________________________________________________________________  
    spectra: array_like
             [[x1, y1], [x2, y2]...]
    baseline: array_like
             [[x1, yb1], [x2, yb2]...]. 
             Note: The abscissa axis in spectra and baseline must correspond
    intervals: np.ndarray
                like interval1, interval2... intervaln ,where interval are 
                has the form [a1, a2] a2>a1 or  *[[a1,a2], [b1,b2]]...
                check ArrayRegiom() function.
    model: func
           default linear_model(x, m=slope, b=intercept)
    parameters: model parameters
                default (m=1.0, b=0.0)
            
    Returns
    ___________________________________________________________________________ 
    res: OptimizeResult 
         References https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
    """

    spectra = array_region(spectra, *intervals, index=index, dim=dim)
    baseline = array_region(baseline, *intervals, index=index, dim=dim)

    def __baseline_model(x, *parameters):
        y = model(baseline[:,1], *parameters) 
        return y

    def __res(parameters, x, y):
        return y - __baseline_model(x, *parameters)

    result = least_squares(__res, parameters,
            args = (spectra[:,0], spectra[:,1]))
    return result


