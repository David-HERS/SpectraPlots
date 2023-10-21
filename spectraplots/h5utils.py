import sys
import string

import numpy as np
import h5py

###############################################################################
#access and Open h5 files
###############################################################################
class h5FileContext:
    """
    A recursive context manager for working with HDF5 files and groups using the 'with' statement.

    This context manager is designed to handle the opening and closing of HDF5 files and groups while
    ensuring proper resource management. It can be used recursively, making it suitable for working
    with nested HDF5 structures.

    Parameters:
    -----------
    file_name_or_object : str or h5py.File or h5py.Group
        The name of the HDF5 file, an open h5py File, or an open h5py Group object.

    **kwargs : additional keyword arguments
        Additional keyword arguments to be passed when opening the HDF5 file (if applicable).

    Example:
    --------
    # Using 'h5FileContext' to open an HDF5 file and a nested group
    with h5FileContext('data.h5', mode='r') as h5_file:
        # Work with the open HDF5 file
        data = h5_file['my_dataset'][:]

        # Nested context to open a group
        with h5FileContext(h5_file['nested_group']) as h5_group:
            # Work with the open HDF5 group

    # The file is automatically closed when exiting the outermost context.
    # See yield_items for example
    """
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


def _access_h5(file_name_or_object, mode='r', **kwargs):
    return h5FileContext(file_name_or_object, mode=mode, **kwargs)


def _all_keys(file_name_or_object, deep = 10, mode = 'r' ):
    """
    Recursively find all keys (dataset paths) in an h5py Group.

    This function iterates through an h5py Group in an HDF5 file and recursively finds all the keys (dataset paths)
    within that group. The function returns a tuple of dataset paths.

    Parameters:
    -----------
    file_name_or_object : str or h5py.Group
        The name of the HDF5 file or an open h5py Group object to traverse.

    deep : int, optional
        The maximum depth to traverse within the HDF5 group. A depth of 0 means only the current level.
        Default is 10. Use a negative value for unlimited depth.

    mode : str, optional
        The file access mode for reading the HDF5 file. Default is 'r' (read-only).

    Returns:
    --------
    tuple
        A tuple containing all the dataset paths within the specified h5py Group.

    Example:
    --------
    # Get all dataset paths within a specific HDF5 group
    keys = _all_keys('data.h5', deep=3)
    for key in keys:
        print(f'Dataset Path: {key}')

    # Alternatively, you can use an already opened h5py Group object
    group = h5py.File('data.h5', 'r')['my_group']
    keys = _all_keys(group, deep=-1)
    for key in keys:
        print(f'Dataset Path: {key}')
    """
    with __access_h5(file_name_or_object, mode = mode) as h5_object:
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


def yield_items(file_name_or_object, name_criteria = None,
        object_criteria= None, deep = 10, mode = 'r' , func= None, **kwargs):
    """
    Yield items from an HDF5 file based on specified criteria.

    This function iterates through items (Groups and Datasets) in an HDF5 file or Group and yields items
    that meet specified criteria, such as name and object-based conditions.

    Parameters:
    -----------
    file_name_or_object : str or h5py.File or h5py.Group
        The name of the HDF5 file or an open h5py File/Group object to traverse.

    name_criteria : function, optional
        A user-defined function that determines whether a dataset or group's name meets specific criteria.
        If not provided, the default criteria always return True.

    object_criteria : function, optional
        A user-defined function that determines whether a dataset or group's object itself meets specific criteria.
        If not provided, the default criteria always return True.

    deep : int, optional
        The maximum depth to traverse within the HDF5 file. A depth of 0 means only the current level.
        Default is 10. Use a negative value for unlimited depth.

    mode : str, optional
        The file access mode for reading the HDF5 file. Default is 'r' (read-only).

    func : function, optional
        A user-defined function to apply to each matching dataset or group.

    **kwargs : additional keyword arguments
        Additional keyword arguments to pass to the underlying `_access_h5` function.

    Yields:
    -------
    h5py.Dataset or h5py.Group
        Datasets and groups that meet the specified criteria.

    Example:
    --------
    # Define custom criteria functions
    def custom_name_criteria(name):
        return 'data' in name

    def custom_object_criteria(obj):
        return 'temperature' in obj.attrs and obj.attrs['temperature'] > 25

    # Yield all datasets and groups with names containing 'data' and temperature above 25
    for item in yield_items('data.h5', name_criteria=custom_name_criteria,
                            object_criteria=custom_object_criteria, deep=3):
        print(f'Found item: {item.name}')

    # Define a custom function to process matching items
    def process_item(item):
        if isinstance(item, h5py.Dataset):
            print(f'Dataset: {item.name}')
        else:
            print(f'Group: {item.name}')

    # Apply the custom function during iteration
    for item in yield_items('data.h5', deep=-1, func=process_item):
        pass
    """
    def __default_criteria(path):
        return True
    name_criteria = name_criteria or __default_criteria
    object_criteria = object_criteria or __default_criteria
    func = func or __default_criteria

    with _access_h5(file_name_or_object, mode = mode, **kwargs) as h5_object:
        if object_criteria(h5_object) and name_criteria(h5_object.name):
            func(h5_object)
            yield  h5_object
        deep  = abs(deep) #for bool(0jj)
        if bool(deep):
            deep -= 1
            for key, value in h5_object.items():
                key = value.name
                if isinstance(value, h5py.Group):
                    yield from yield_items(value, deep=deep, mode=mode,
                            name_criteria=name_criteria,
                            object_criteria=object_criteria, func=func)
                elif (isinstance(value, h5py.Dataset) and
                        object_criteria(value) and
                        name_criteria(key)):
                    func(value)
                    yield value


def _h5_keys(file_name_or_object, name_criteria = None, object_criteria = None,
        deep = 10):
    """Return names of the h5_obj with criteria name and object"""
    def __default_criteria(path):
        return True
    name_criteria = name_criteria or __default_criteria
    object_criteria = object_criteria or __default_criteria

    names = __all_keys(file_name_or_object, deep = deep)
    with access_h5(file_name_or_object, mode='r') as h5_object:
        names_criteria = [name for name in names
                if object_criteria(h5_object.get(name)) and name_criteria(name)]
    return names_criteria

###############################################################################
#criterias for hdf5 files
###############################################################################
def _default_criteria(file_name_or_object):
    return True


def is_dataset(h5_object):
    return isinstance(h5_object, h5py.Dataset)


def is_group(h5_object):
    return isinstance(h5_object, h5py.Group)


def is_root(name):
    return name == '/'


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
def _default_func(h5_object):
    return None


def string_to_float(number_string, dot='_'):
    """
    Convert a formatted number string into a float.

    This function takes a string representing a number and attempts to convert it to a float.
    It performs cleaning on the input string to remove non-numeric characters, replace the dot
    (or a specified character) with the decimal point, and handles cases where the number
    may be improperly formatted.

    Parameters:
    -----------
    number_string : str
        The string containing the number to be converted.

    dot : str, optional
        The character used as a decimal point in the string. Default is underscore ('_').

    Returns:
    --------
    float
        The converted float value. If the conversion fails, it returns 'NaN' (Not-a-Number).

    Example:
    --------
    # Handle a poorly formatted number string
    result = string_to_float("abc12345_67def")
    print(result)  # Output: 12345.64
    """
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

def _apply_attributes(file_name_or_object, keys,  attribute_name,
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
    with __access_h5(file_name_or_object, mode='r+') as h5_object:
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
   apply_name_attribute
   Create a hdf5 attribute with the h5_object name with a rule the default 
   is replace the '_' with '.' and search the number_string 
   Parameters:
   ---------------------------------------------------------------------------   
   h5_obj: h5py.Group, h5py.Dataset

   attribute_name: string

   func: func
         Function to choose the attribute value with name of h5 object the 
         the default is the string_to_float method

   start:Int
         Starts of character in h5_obj.name 

   end:Int
       Ends of character in h5_obj.name
   step:Int
        Step of character in h5_obj.name
   confirm:Bool
           Confirms the creation of the attribute
   """
   slide = slice(start, end, step)
   attribute = func(h5_object.name[slide])
   if confirm:
       h5_object.attrs.create(attribute_name, np.float64(attribute))
   return attribute


class h5Utils():
    def __init__(self, file_name_or_object, *, mode = 'r', deep = 10, 
            name_criteria=None, object_criteria=None, func=None, keys=(), **kwargs):
       self.file_name_or_object = file_name_or_object
       self.mode = mode
       self.name_criteria = name_criteria or _default_criteria
       self.object_criteria = object_criteria or _default_criteria
       self.deep = deep
       self.func = func or _default_func 
       self.keys = keys
       self.kwargs = kwargs

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

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, keys):
        if isinstance(keys, tuple) or isinstance(keys, list):
            self._keys = keys
        else:
            print('It is not possible to set the keys. Maybe Tuple or List ')

    def set_default(self):
        self.mode = 'r'
        self.name_criteria = _default_criteria
        self.object_criteria = _default_criteria
        self.deep = 10
        self.func = _default_func
        return self

    def set_kwargs(self, *,mode = None, name_criteria=None,
            object_criteria= None, deep=None, func=None, keys=(), **kwargs):
        self.mode = mode or self.mode
        self.name_criteria = name_criteria or self.name_criteria
        self.object_criteria = object_criteria or self.object_criteria
        self.deep = deep or self.deep
        self.func = func or self.func
        self.keys = keys or self.keys 
        self.kwargs = kwargs
        return self

    def access_h5(self, *,mode = None):
        self.mode = mode or self.mode
        return _access_h5(self.file_name_or_object, mode=self.mode, **self.kwargs)

    def select_items(self, *,name_criteria=None,
            object_criteria=None, deep=None,
            mode=None, func= None):
        """
        See yield_items
        """
        file_name_or_object = self.file_name_or_object
        mode = mode or self.mode
        name_criteria = name_criteria or self.name_criteria
        object_criteria = object_criteria or self.object_criteria
        deep = deep or self.deep
        func = func or self.func

        return yield_items(file_name_or_object,
                name_criteria=name_criteria,
                object_criteria=object_criteria,
                deep=deep, mode=mode, func=func)

    def mk_keys(self, *,name_criteria=None,
            object_criteria=None, deep=None,
            mode=None, func= None):
        """
        set_keys 
        Returns the string name (key) of h5 objects that satisfies name_criteria,
        object_criteria and deep.
        Parameters:
        ---------------------------------------------------------------------------   
        name_criteria: Bool function
        object_criteria: Bool function
        func: func 
              is the criteria to sort like func(h5_obj, key), where key is string type
              and h5_obj contain objects related to key, the sort function must
              return a numeric value
        """
        self.set_kwargs(mode=mode, name_criteria=name_criteria, 
                object_criteria=object_criteria, deep=deep, func=func)
        keys = ()
        for h5_object in self.select_items(func=_default_criteria):
            keys = keys + (h5_object.name,)

        if self.func != _default_func:
            with self.access_h5() as h5_obj:
                _func = lambda key: self.func(h5_obj, key)
                keys = sorted(keys, key=_func)
        self.keys = keys
        return keys 

    def apply_keys(self, keys=None, mode=None ,func=None):
        keys = keys or self.keys
        func = func or self.func
        mode = mode or self.mode
        with self.access_h5(mode=mode) as h5_object:
            for  key in keys:
                if func:
                    func(h5_object.get(key))
                    yield h5_object.get(key)
                else:
                    yield h5_object.get(key)
