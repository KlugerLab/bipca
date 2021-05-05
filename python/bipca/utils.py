import numpy as np
import inspect
import scipy.sparse as sparse
from sklearn.base import BaseEstimator
from collections.abc import Iterable
from itertools import count
import tasklogger
from sklearn.base import BaseEstimator
from sklearn import set_config





def _is_vector(x):
    return (x.ndim == 1 or x.shape[0] == 1 or x.shape[1] == 1)

def _xor(lst, obj):
    condeval = [ele==obj for ele in lst]
    condeval = sum(condeval)
    return condeval==1

def _zero_pad_vec(nparray, final_length):
    # pad a vector (nparray) to have length final_length by adding zeros
    # adds to the largest axis of the vector if it has 2 dimensions
    # requires the input to have 1 dimension or at least one dimension has length 1.
    if (nparray.shape[0]) == final_length:
        z = nparray
    else:
        axis = np.argmax(nparray.shape)
        pad = final_length - nparray.shape[axis]
        if nparray.ndim>1:
            if not 1 in nparray.shape:
                raise ValueError('Input nparray is not a vector')
        padshape = list(nparray.shape)
        padshape[axis] = pad
        z = np.concatenate((nparray,np.zeros(padshape)),axis=axis)
    return z

def filter_dict(dict_to_filter, thing_with_kwargs,negate=False):
    """
    Modified from 
    https://stackoverflow.com/a/44052550    
    User "Adviendha"

    """
    sig = inspect.signature(thing_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD and param.name in dict_to_filter.keys()]
    if negate:
        filtered_dict = {key:dict_to_filter[key] for key in dict_to_filter.keys() if key not in filter_keys}
    else:
        filtered_dict = {filter_key:dict_to_filter[filter_key] for filter_key in filter_keys}
    return filtered_dict

def ischanged_dict(old_dict, new_dict, keys_ignore = []):
    ischanged = False

    #check for adding or updating arguments
    for k in new_dict:
        ischanged = k not in old_dict or old_dict[k] != new_dict[k]#catch values that are new
        if ischanged:
            break
    #now check for removing arguments
    if not ischanged:
        for k in old_dict:
            if k not in keys_ignore and k not in new_dict:
                ischanged = True
                break
    return ischanged

    # pre-processing function that removes 0 rows and columns with the option of adding a small eps to matrix
# to allow for sinkhorn to converge faster/better
def stabilize_matrix(mat, read_cts = None, add_eps = False, return_zero_indices = False):
    
    
    # Might need a method here that determines what the tolerance value is
    # Since in this experiment we are generating count data, the tolerance value can be 0. We will set to 1e-6 just in case
    
    tol = 1E-6
    if sparse.issparse(mat):
        mat = mat[mat.getnnz(1)>0][:,mat.getnnz(0)>0]
    else:
        zero_rows = np.all(np.abs(mat)<=tol, axis = 1)
        zero_cols = np.all(np.abs(mat)<=tol, axis = 0)

        mat = mat[~zero_rows,:]
        mat = mat[:,~zero_cols]

    if return_zero_indices == True:
        if read_cts is not None:     
            # if we have read counts to prune as well, we do that here
            read_cts = read_cts[~zero_cols]
            return mat, read_cts, [zero_rows, zero_cols]
        
        return mat, [zero_rows, zero_cols]
    
    if add_eps == True:
        pass
    
    
    return mat