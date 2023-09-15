from os import access
import sys
import string

import numpy as np
import h5py

###############################################################################
#Acces and Open h5 files
###############################################################################
class h5FileContext:
    def __init__(self, file_name_or_object, **kwargs):
        self.file_name_or_object = file_name_or_object
        self.kwargs = kwargs

    def __enter__(self):
        if isinstance(self.file_name_or_object, h5py.File):
            # User provided an already open HDF5 file
            return self.file_name_or_object
        elif isinstance(self.file_name_or_object, h5py.Group):
            # User provided an already open HDF5 group
            return self.file_name_or_object
        elif isinstance(self.file_name_or_object, str):
            # User provided a file name, so open it
            self.h5_file = h5py.File(self.file_name_or_object, **self.kwargs)
            return self.h5_file
        else:
            raise TypeError("Unsupported type for 'file_name_or_object'. Must be an H5 file or a string file name.")

    def __exit__(self, exc_type, exc_value, traceback):
        if isinstance(self.file_name_or_object, str):
            self.h5_file.close()


def __acces_h5(file_name_or_object, mode='r'):
    return h5FileContext(file_name_or_object, mode=mode)


def __all_keys(file_name_or_object, deep = 10, mode = 'r' ):
    "Recursively find all keys in an h5py.Group."
    with __acces_h5(file_name_or_object, mode = mode) as h5_object:
        keys = (h5_object.name,)
        deep  = abs(deep) #for bool(0jj)
        if bool(deep):
            deep -= 1
            for key, value in h5_object.items():
                if isinstance(value, h5py.Group):
                    keys = keys + __all_keys(value, deep, mode=mode)
                elif isinstance(value, h5py.Dataset):
                    keys = keys + (value.name,)
    return keys


def yield_keys(file_name_or_object, name_criteria = None,
        object_criteria= None, deep = 10, mode = 'r' , func= None):
    "Yields the keys for the h5 file  with some cirterias"
    def __default_criteria(path):
        return True
    name_criteria = name_criteria or __default_criteria
    object_criteria = object_criteria or __default_criteria
    func = func or __default_criteria

    with __acces_h5(file_name_or_object, mode = mode) as h5_object:
        if object_criteria(h5_object) and name_criteria(h5_object.name):
            func(h5_object)
            yield  h5_object.name
        deep  = abs(deep) #for bool(0jj)
        if bool(deep):
            deep -= 1
            for key, value in h5_object.items():
                key = value.name
                if isinstance(value, h5py.Group):
                    yield from yield_keys(value, deep=deep, mode=mode,
                            name_criteria=name_criteria,
                            object_criteria=object_criteria, func=func)
                elif (isinstance(value, h5py.Dataset) and
                        object_criteria(value) and
                        name_criteria(key)):
                    func(value)
                    yield key

def __h5_keys(file_name_or_object, name_criteria = None, object_criteria = None,
        deep = 10):
    """Return names of the h5_obj with criteria name and object"""
    def __default_criteria(path):
        return True
    name_criteria = name_criteria or __default_criteria
    object_criteria = object_criteria or __default_criteria

    names = __all_keys(file_name_or_object, deep = deep)
    with acces_h5(file_name_or_object, mode='r') as h5_object:
        names_criteria = [name for name in names
                if object_criteria(h5_object.get(name)) and name_criteria(name)]
    return names_criteria

###############################################################################
#criterias for hdf5 files
###############################################################################
def __default_criteria(file_name_or_object):
    return True

def is_dataset(h5_object):
    return isinstance(h5_object, h5py.Dataset)


def is_group(h5_object):
    return isinstance(h5_object, h5py.Group)

def criteria_name(
        path, in_path=[] , not_in_path=[],
        starts = [], ends = [], not_starts = [], not_ends = [],
        operator = 'and'):
    """
    """
    if operator == 'and':
        operator= np.prod
    elif operator == 'or':
        operator= np.sum
    else:
        operator= np.sum

    if in_path:
        __in_path = bool(operator(np.array(
            [(include in path) for include in in_path])))
    else: __in_path = True
    if starts:
        __starts = bool(operator(np.array(
            [path.startswith(criteria) for criteria in starts])))
    else: __starts = True
    if ends:
        __ends =  bool(operator(np.array(
            [path.endswith(criteria) for criteria in ends])))
    else: __ends = True
    if not_in_path:
        __not_in_path = bool(operator(np.array(
            [not(not_inc in path) for not_inc in not_in_path])))
    else: __not_in_path = True
    if not_starts:
        __not_starts = bool(operator(np.array(
            [not(path.startswith(criteria)) for criteria in not_starts])))
    else: __not_starts = True 
    if not_ends:
        __not_ends =  bool(operator(np.array(
            [not(path.endswith(criteria)) for criteria in not_ends])) )
    else: __not_ends = True 
    
    satisfy = (__in_path and __not_in_path and  __starts and  __not_starts
            and __ends and __not_ends)
    return satisfy

###############################################################################
#for create or modify hdf5 files 
###############################################################################
def string_to_float(number_string, dot='_'):
    
    #clean number_string
    alphabet = list(string.ascii_letters)
    symbols = list(string.punctuation.replace(dot,''))
    name = number_string
    for letter in alphabet:
        number_string = number_string.replace(letter, '')
    for symbol in symbols:
        number_string = number_string.replace(symbol, '')
        
    number_string = number_string.replace(dot, '.')

    #avoid starting dots
    starts = 0
    ends = len(number_string)
    for i, s in enumerate(number_string):
        if s in string.digits:
            starts=i
            break
    #avoid multiple dots
    dots=0
    for i, s in enumerate(number_string[starts:]):
        if s in '.' and dots==1:
            ends=i
            dots+=1
        elif s in '.':
            dots += 1

    try:
        number = float(number_string[starts:starts+ends])
    except:
        print(f' String to float failed in "{name}"\n',
                f'String try:{number_string[starts:ends]}')
        number = float('Nan')
    return number

def __apply_attributes(file_name_or_object, keys,  attribute_name,
        criteria=False, func=string_to_float, start=None, end=None, step=None):
    """
    attr_name
    Create a hdf5 attribute with the file name with a rule the default 
    is replace the '_' with '.' and seatch the number_string 
    Parameters:
    ---------------------------------------------------------------------------   
    h5_obj: h5py.Group, h5py.File
            Contains Datasets 
    keys: Array of strings
         ['key1', 'key2'...,'keyn']
    search_interval: array like
                     [interval1, interval2] where interval1>interval2
    rule: func
          Method to choose the attribute value with key file 
    """
    with __acces_h5(file_name_or_object, mode='r+') as h5_object:
        if not(criteria): criteria= __default_criteria
        slide = slice(start, end, step)
        
        for key in keys:
            key_short = key[slide]
            attribute = func(key_short)
            h5_object.get(key).attrs.create(attribute_name, np.float64(attribute))
    return file_name_or_object

def apply_name_attribute(h5_object, attribute_name, func=string_to_float,
        start = None, end=None, step=None, confirm=True):
   """

   """
   slide = slice(start, end, step)
   attribute = func(h5_object.name[slide])
   if confirm:
       h5_object.attrs.create(attribute_name, np.float64(attribute))
   return attribute


class h5Utils():
    def __init__(self, file_name_or_object, mode = 'r', deep = 10, 
            name_criteria=None, object_criteria=None, func=None):
       self.file_name_or_object = file_name_or_object
       self.mode = mode
       self.name_criteria = name_criteria
       self.object_criteria = object_criteria
       self.deep = deep
       self.func = func

    @property
    def file_name_or_object(self):
        return self._file_name_or_object

    @file_name_or_object.setter
    def file_name_or_object(self, file_name_or_object):
        if isinstance(file_name_or_object, h5py.File):
            # User provided an already open HDF5 file
            self._file_name_or_object = file_name_or_object
        elif isinstance(file_name_or_object, h5py.Group):
            # User provided an already open HDF5 group
            self._file_name_or_object = file_name_or_object
        elif isinstance(file_name_or_object, str):
            # User provided a file name, so open it
            self._file_name_or_object = file_name_or_object
        else:
            raise TypeError("Unsupported type for 'file_name_or_object'. Must be an H5 object (File or Group) or a string file name.")

    @property
    def mode(self):
        return self._mode

    @mode.setter 
    def mode(self, mode):
        if mode in ['r','r+', 'w', 'w-', 'x', 'a']:
            self._mode = mode
        else:
            raise NameError("mode must be ['r','r+', 'w', 'w-', 'x', 'a']")

    def set_default(self, mode = 'r', name_criteria=None,
            object_criteria= None, deep=10, func=None):
        self.mode = mode
        self.name_criteria = name_criteria 
        self.object_criteria = object_criteria 
        self.deep = deep
        self.func = None

    def acces_h5(self):
        return __acces_h5(self.file_name_or_object, mode=self.mode)

    def select_keys(self):
        return yield_keys(self.file_name_or_object,
                name_criteria=self.name_criteria,
                object_criteria=self.object_criteria,
                deep=self.deep, mode=self.mode, func=self.func)

    def apply_keys(self, keys):
        with acces_h5(self) as h5_object:
            for  key in keys:
                if self.func:
                    self.func(h5_object.get(keys))
                    yield h5_object.get(key)
