from . import *

import sys; sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser
import numpy as np
import gt4py as gt
from copy import deepcopy


# Read serialized data at a specific tile and savepoint
def read_data(path, tile, ser_count, is_in):
    
    mode_str = "in" if is_in else "out"
    vars     = IN_VARS if is_in else OUT_VARS
    
    if is_in:
        
        serializer = ser.Serializer(ser.OpenModeKind.Read, path, "Generator_rank" + str(tile))
        savepoint  = ser.Savepoint(f"cloud_mp-{mode_str}-{ser_count:0>6d}")
        
    else:
        
        serializer = ser.Serializer(ser.OpenModeKind.Read, path, "Serialized_rank" + str(tile))
        savepoint  = ser.Savepoint(f"cloud_mp-{mode_str}-x-{ser_count:0>6d}")
    
    return data_dict_from_var_list(vars, serializer, savepoint)
    
    
# Read given variables from a specific savepoint in the given serializer
def data_dict_from_var_list(vars, serializer, savepoint):
    
    data_dict = {}
    
    for var in vars:
        data_dict[var] = serializer.read(var, savepoint)
        
    searr_to_scalar(data_dict)
    
    return data_dict
    

# Convert single element arrays (searr) to scalar values of the correct 
# type
def searr_to_scalar(data_dict):
    
    for var in data_dict:
        
        if data_dict[var].size == 1:
            
            if var in INT_VARS: data_dict[var] = DTYPE_INT(data_dict[var][0])
            if var in FLT_VARS: data_dict[var] = DTYPE_FLT(data_dict[var][0])
            

# Transform a dictionary of numpy arrays into a dictionary of gt4py 
# storages of shape (iie-iis+1, jje-jjs+1, kke-kks+1)
def numpy_dict_to_gt4py_dict(np_dict):
    
    shape      = np_dict["qi"].shape
    gt4py_dict = {}
    
    for var in np_dict:
        
        data = np_dict[var]
        ndim = data.ndim
        
        if (ndim > 0) and (ndim <= 3) and (data.size >= 2):
            
            reshaped_data = np.empty(shape)
            
            if ndim == 1:       # 1D array (i-dimension)
                reshaped_data[...] = data[:, np.newaxis, np.newaxis]
            elif ndim == 2:     # 2D array (i-dimension, j-dimension)
                reshaped_data[...] = data[:, :, np.newaxis]
            elif ndim == 3:     # 3D array (i-dimension, j-dimension, k-dimension)
                reshaped_data[...] = data[...]
            
            dtype           = DTYPE_INT if var in INT_VARS else DTYPE_FLT
            gt4py_dict[var] = gt.storage.from_array(reshaped_data, BACKEND, DEFAULT_ORIGIN, dtype=dtype)
            
        else:   # Scalars
            
            gt4py_dict[var] = deepcopy(data)
            
    return gt4py_dict


# Cast a dictionary of gt4py storages into dictionary of numpy arrays
def view_gt4py_storage(gt4py_dict):
    
    np_dict = {}
    
    for var in gt4py_dict:
        
        data = gt4py_dict[var]
        
        # ~ if not isinstance(data, np.ndarray): data.synchronize()
        if BACKEND == "gtcuda": data.synchronize()
        
        np_dict[var] = data.view(np.ndarray)
        
    return np_dict


# Compare two dictionaries of numpy arrays, raise error if one array in 
# data does not match the one in ref_data
def compare_data(data, ref_data, explicit=True, blocking=True):
    
    wrong = []
    flag  = True
    
    for var in data:
        
        if not np.allclose(data[var], ref_data[var], rtol=1e-11, atol=1.e-13, equal_nan=True):
            
            wrong.append(var)
            flag = False
            
        else:
            
            if explicit: print(f"Successfully validated {var}!")
            
    if blocking:
        assert flag, f"Output data does not match reference data for field {wrong}!"
    else:
        if not flag: print(f"Output data does not match reference data for field {wrong}!")


# Scale the input dataset for benchmarks
def scale_dataset(data, factor):
    
    divider    = factor[0]
    multiplier = factor[1]
    
    do_divide = divider < 1.
    
    scaled_data = {}
    
    for var in data:
        
        data_var = data[var]
        ndim     = data_var.ndim
        
        if ndim == 3:
            
            if do_divide:
                data_var = data_var[:DTYPE_INT(len(data_var)*divider), :, :]
            
            scaled_data[var] = np.tile(data_var, (multiplier, 1, 1))
            
        elif ndim == 2:
            
            if do_divide:
                data_var = data_var[:DTYPE_INT(len(data_var)*divider), :]
            
            scaled_data[var] = np.tile(data_var, (multiplier, 1))
            
        elif ndim == 1:
            
            if do_divide:
                data_var = data_var[:DTYPE_INT(len(data_var)*divider)]
            
            scaled_data[var] = np.tile(data_var, multiplier)
            
        elif ndim == 0:
            
            if var == "iie":
                scaled_data[var] = DTYPE_INT(data[var] * multiplier * divider)
            else:
                scaled_data[var] = data[var]
    
    return scaled_data

def scale_dataset_to_N(data, CN):

    scaled_data = {}
    orig = DTYPE_INT(data['iie'])
    CN2 = CN*CN
    multiplier = DTYPE_INT(np.ceil(CN2/orig))

    for var in data:
        
        data_var = data[var]
        ndim     = data_var.ndim
        
        if ndim == 3:
            
            scaled_data[var] = np.tile(data_var, (multiplier, 1, 1))[0:CN2,:,:]
            
        elif ndim == 2:
            
            scaled_data[var] = np.tile(data_var, (multiplier, 1))[0:CN2,:]
            
        elif ndim == 1:
            
            scaled_data[var] = np.tile(data_var, multiplier)[CN2]
            
        elif ndim == 0:
            
            if var == "iie":
                scaled_data[var] = DTYPE_INT(CN2)
            else:
                scaled_data[var] = data[var]
    
    return scaled_data