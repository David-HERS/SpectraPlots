"""Spectra Methods"""
import os
import sys

import h5py
import numpy as np
from scipy.optimize import least_squares


def FindNear(a, Near):
    """
    FindNear(x, Near)
    
    Returns
    ___________________________________________________________________________ 
    Int
        The "x" array position with the nearest value to "Near" value
    """
    a = np.asarray(a)
    nearpos = (np.abs(a-Near)).argmin()
    return nearpos

def ArrayRegion(a, *intervals, index=False, dim=0):
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
        The concatenated array with intervasl choosen.

    """
    def __region(a, begin, end):
        if begin<=end: 
            return a[begin:end]
        else:
            return a[begin:end:-1]

    a = np.asarray(a)
    for i, interval in enumerate(intervals):
        if index: begin, end = a[interval[0], interval[1]]
        else : begin, end= (FindNear(a[:, dim], interval[0]),
                            FindNear(a[:, dim], interval[1])
                            )
        if i==0: region = __region(a, begin, end)
        else: region = np.concatenate((region,__region(a, begin, end)))
    return region

   
###############################################################################
#MODELS
###############################################################################
def LinearModel(x, m=1.0, b=0.0): 
    """Linear model
    LinearModel(x , m = slope, b= intercept)
    return mx+b 
    """
    return x*m + b

###############################################################################
#Fits
###############################################################################

def FitBaseline(spectra, baseline, *intervals,
        model= LinearModel, parameters= (1.0, 0.0), index = False, dim=0):
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
           default LinearModel(x, m=slope, b=intercept)
    parameters: model parameters
                default (m=1.0, b=0.0)
            
    Returns
    ___________________________________________________________________________ 
    res: OptimizeResult 
         References https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
    """

    spectra = ArrayRegion(spectra, *intervals, index=index, dim=dim)
    baseline = ArrayRegion(baseline, *intervals, index=index, dim=dim)

    def __BaselineModel(x, *parameters):
        y = model(baseline[:,1], *parameters) 
        return y 

    def __res(parameters, x, y):
        return y - __BaselineModel(x, *parameters)

    result = least_squares(__res, parameters,
            args = (spectra[:,0], spectra[:,1]))
    return result


