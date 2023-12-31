"""Spectra Methods"""
import os
import sys

import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.constants import h, c, e
import peakutils

from .h5utils import h5Utils, criteria_name, is_dataset, is_group, _default_func


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
            return a[end:begin]

    a = np.asarray(a)
    if intervals:
        for i, interval in enumerate(intervals):
            if index: begin, end = a[interval[0], interval[1]]
            else : begin, end= (find_near(a[:, dim], interval[0]),
                                find_near(a[:, dim], interval[1])
                                )
            if i==0: region = __region(a, begin, end)
            else: region = np.concatenate((region,__region(a, begin, end)))
    else: region=a
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


def folder_average(path, folder_name='', fmt='%1.5f', display=False):
    """
    folder average
    Gives the average number of folders that have the same data structure

    Parameters
    --------------------------------------------------------------------------- 
    path: str
          A path that contains folders with same structure

    folder_name: str
                 General name folder 

    Returns
    --------------------------------------------------------------------------- 
    returns: None 
             A average files in the main path with the general name in files  
    """
    if path == '':
        path = str(input('Add principal path:'))

    with os.scandir(path) as folders:
        path_folders = ['/'+folder.name for folder in folders
                if (folder.is_dir() and (folder_name in folder.name))]

    if path_folders:
        if display: print(f'Folders:{path_folders}')
    else:
        if display: print('No Folders')
        return None 

    #The folder[0] is used to give the name files
    with os.scandir(path+path_folders[0]) as files:
        path_files = ['/'+file.name for file in files if file.is_file()]

    if path_files:
        if display: print(f'Name files:{path_files}')
    else:  
        if display: print('No files')
        return None

    for file in path_files:
        for count,  folder  in enumerate(path_folders):
            data = np.loadtxt(path+folder+file) 
            if count ==0: average = np.zeros_like(data)
            average = data + average 
        np.savetxt(path+file, average/(count+1), fmt=fmt)
        if display: print(f'Successful file:{path+file}')


    return None 

def baseline(dataset, interval=[], name= '', suffix='',prefix='',
             deg=None, max_it=None, tol=None):
    """
    Create a baseline for h5py.Dataset object and save as attribute

    Parameters
    --------------------------------------------------------------------------- 
    dataset: array_like
        [[x1, y1], [x2, y2]...]
    interval: array_like
        [a, b] interval for baseline
    name: str, default:'Baseline'
        name for baseline attribute 
    suffix: str, default:''
        suffix for name 
    prefix: str, default:''
        prefix for name
    deg: int, default:3
        Degree of the polynomial that will estimate the data baseline. A low degree may fail to detect all the baseline present, while a high degree may make the data too oscillatory, especially at the edges.  
    max_it: int, default:100
        Maximum number of iterations to perform.
    tol: float, default:1e-03
        Tolerance to use when comparing the difference between the current fit coefficients and the ones from the last iteration. The iteration procedure will stop when the difference between them is lower than tol.
    
    Returns
    --------------------------------------------------------------------------- 
    Array with the baseline amplitude for every original point in y and saved  as attribute in dataset
    """
    if name: name = name
    else: name = 'Baseline' 
    if prefix: name = f'{prefix}name'
    if suffix: name = f'name{suffix}'
    if is_dataset(dataset):
        data = np.array(dataset)
        if interval: data = array_region(data, interval)
        _baseline =  peakutils.baseline(data[:,1], 
                                        deg=deg, max_it=max_it, tol=tol)
        data[:,1] = data[:,1]-_baseline
        base  = data.copy()
        base[:,1] = _baseline
        dataset.attrs.create(f'{name}', data=np.array(base))
        dataset.attrs.create(f'{name}.Interval', data = np.array(interval))
        dataset.attrs.create(f'{name}.Substract', data = np.array(data))


def mk_map(file_name_or_object, name='Map', mode='r+',
           name_criteria=None, object_criteria=None, func=None,
           xattr = 'OssilaX2000.SMU1 Voltage(V)', baseline='', attributes=[]): 
    """
    Create a photoluminescence map (X, Y, Z).

    Parameters
    ----------
    file_name_or_object : str or h5py object
        The name of the HDF5 file or the h5py object representing the file.

    name : str, optional, default: 'Map'
        The name for the dataset that will be created.

    mode : str, optional, default: 'r+'
        The mode for reading the HDF5 file.

    name_criteria : callable, optional
        A function that filters dataset names. Only datasets matching this criteria will be considered.

    object_criteria : callable, optional
        A function that filters datasets based on their attributes. Only datasets matching this criteria will be considered.

    func : callable, optional
        A function for sorting keys. This function should accept an h5py object and key and return a sortable value.

    xattr : str, optional, default: 'OssilaX2000.SMU1 Voltage(V)'
        The attribute name used to obtain the X-axis data.

    baseline : str, optional, default: ''
        The name of the baseline attribute in the dataset, if available.

   attributes : list, optional, default: []
        A list of numeric attribute names to be aggregated and stored in the output dataset. The function computes the average of these attributes across all datasets.
        

    Returns
    -------
    None

    Example Usage
    -------------
    >>> mk_map('your_file.h5', name='PhotoluminescenceMap', attributes=['Sample', 'Temperature'])
    """
    sample = h5Utils(file_name_or_object,mode=mode,
                     name_criteria=name_criteria, object_criteria=object_criteria,
                     func=func)
    keys = sample.mk_keys()
    sample.func = _default_func
    x = np.zeros(len(keys))

    for count, dataset in enumerate(sample.apply_keys(keys)):
        #suppose that all data is similar in size
        if baseline: array = np.array(dataset.attrs.get(baseline))
        else: array = np.array(dataset)

        if count == 0:
            x[count] = dataset.attrs.get(xattr)
            attributes_dict = {}
            for attribute in attributes:
                    attributes_dict[attribute] = np.float64(dataset.attrs.get(attribute))
               
            y = array[:,0]
            X, Y  = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            Z[:,count] = array[:,1]

        else:

            for attribute in attributes:
                attributes_dict[attribute] += np.float64(dataset.attrs.get(attribute))

            x[count] = dataset.attrs.get(xattr)
            X[:,count] = x[count]
            Z[:,count] = array[:,1]
        if count==len(keys)-1:
            try:
                root = dataset
                while True:
                    root = root.parent
                    if root.name == '/':
                        break
                mesh = root.create_dataset(name, data=np.array([X, Y, Z]))
                for attribute in attributes:
                    mesh.attrs.create(attribute, attributes_dict[attribute]/(count+1))
            except ValueError:
                print(ValueError, 'try another name')
    return None


def mk_profile(file_name_or_object, keys, profile_value, name='Profile',
        xattr = 'Wavelenght(nm)', baseline='', attributes=[], 
        mode='r+', func=None):
    """
    Create a profile dataset based on specified values in HDF5 datasets. 
    Used for PLE
    Parameters
    ----------
    file_name_or_object : str or h5py object
        The name of the HDF5 file or the h5py object representing the file.

    keys : list
        A list of dataset keys to extract profile values from.

    profile_value : float
        The value at which the profile is extracted from each dataset.

    name : str, optional, default: 'Profile'
        The name for the profile dataset that will be created.

    xattr : str, optional, default: 'Wavelength(nm)'
        The attribute name used to obtain the X-axis data.

    baseline : str, optional, default: ''
        The name of the baseline attribute in the dataset, if available.

    attributes : list, optional, default: []
        A list of attribute names to be aggregated and stored in the output dataset. The function computes the average of these attributes across all datasets.

    mode : str, optional, default: 'r+'
        The mode for reading the HDF5 file.

    func : callable, optional
        A function to process the profile values before storing them in the dataset. This function should accept the profile value and the dataset as parameters and return the processed value.

    Returns
    -------
    None

    """

    sample = h5Utils(file_name_or_object,mode=mode)
    x = np.zeros(len(keys))
    y = np.zeros(len(keys))

    for count, dataset in enumerate(sample.apply_keys(keys)):
        x[count] = float(dataset.attrs.get(xattr))
        point = find_near(dataset[:,0], profile_value)

        if baseline: y[count] = dataset.attrs.get(baseline)[point, 1]
        else: y[count] = dataset[point, 1]
        if func: y[count] = func(y[count], dataset) 

        if count==len(keys)-1:
            try:
                root = dataset
                while True:
                    root = root.parent
                    if root.name == '/':
                        break
                profile = np.stack((x,y), axis=-1)
                dataset_profile = root.create_dataset(name, data=profile)
                for attribute in attributes:
                    dataset_profile.attrs.create(attribute, attributes_dict[attribute]/(count+1))
            except ValueError:
                print(ValueError, 'try another name')

    return None




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


