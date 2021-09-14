"""Summary
"""
import numpy as np
import inspect
import scipy.sparse as sparse
from sklearn.base import BaseEstimator
from collections.abc import Iterable
from itertools import count
import tasklogger
from sklearn.base import BaseEstimator
from sklearn import set_config
import torch




###Some functions the user may want

def write_to_adata(obj, adata):
    """Store the main outputs of a bipca object in place into an AnnData object.
        Note that if `obj.conserve_memory is True`, then adata.layers['Z'] cannot be written directly.
        In this case, the function applies `obj.get_Z(adata.X)` to obtain `Z`
    Parameters
    ----------
    obj : bipca.BiPCA
        BiPCA object to extract results from
    adata : AnnData
        AnnData object to store results in.

    Returns
    -------
    adata : AnnData
        The modified adata object. 
    
    Raises
    ------
    ValueError
        
    """
    target_shape = adata.shape
    if target_shape != (obj.M, obj.N) and target_shape != (obj.N, obj.M):
        raise ValueError("Invalid shape passed. Adata must have shape " + str((obj.M,obj.N)) +
            " or " + str((obj.N,obj.M)))
    with obj.logger.task("Writing bipca to anndata"):
        try:
            adata.uns['bipca']
        except KeyError as e:
            adata.uns['bipca'] = {}
        Y_scaled = obj.transform(unscale = False)
        if target_shape != Y_scaled.shape:
            Y_scaled = Y_scaled.T
        try:
            if obj.conserve_memory:
                adata.layers['Z_biwhite'] = obj.get_Z(adata.X)
            else:
                adata.layers['Z_biwhite'] = obj.Z
            adata.layers['Y_biwhite'] = Y_scaled
        except ValueError:
            if obj.conserve_memory:
                adata.layers['Z_biwhite'] = obj.get_Z(adata.X.T).T
            else:
                adata.layers['Z_biwhite'] = obj.Z.T
            adata.layers['Y_biwhite'] = Y_scaled.T

        if target_shape == (obj.M,obj.N):
            adata.varm['V_biwhite'] = obj.V_Z
            adata.obsm['U_biwhite'] = obj.U_Z
            adata.uns['bipca']['left_biwhite'] = obj.left_biwhite
            adata.uns['bipca']['right_biwhite'] = obj.right_biwhite
        else:
            adata.varm['V_biwhite'] = obj.U_Z
            adata.obsm['U_biwhite'] = obj.V_Z
            adata.uns['bipca']['left_biwhite'] = obj.right_biwhite
            adata.uns['bipca']['right_biwhite'] = obj.left_biwhite

        adata.uns['bipca']['S'] = obj.S_Z
        adata.uns['bipca']['rank'] = obj.mp_rank
        try:
            adata.uns['bipca']['q'] = obj.q
        except:
            pass
        try:
            adata.uns['bipca']['kst'] = obj.kst
        except:
            pass
        try:
            adata.uns['bipca']['plotting_spectrum'] = obj.get_plotting_spectrum()
        except:
            pass
        adata.uns['bipca']['sigma'] = obj.shrinker.sigma
    return adata
###Other functions that the user may not want.



def _is_vector(x):
    """Summary
    
    Parameters
    ----------
    x : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    return (x.ndim == 1 or x.shape[0] == 1 or x.shape[1] == 1)

def _xor(lst, obj):
    """Summary
    
    Parameters
    ----------
    lst : TYPE
        Description
    obj : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    condeval = [ele==obj for ele in lst]
    condeval = sum(condeval)
    return condeval==1

def zero_pad_vec(nparray, final_length):
    """Summary
    
    Parameters
    ----------
    nparray : TYPE
        Description
    final_length : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    
    Raises
    ------
    ValueError
        Description
    """
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
    
    Parameters
    ----------
    dict_to_filter : TYPE
        Description
    thing_with_kwargs : TYPE
        Description
    negate : bool, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    
    """
    sig = inspect.signature(thing_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD and param.name in dict_to_filter.keys()]
    if negate:
        filtered_dict = {key:dict_to_filter[key] for key in dict_to_filter.keys() if key not in filter_keys}
    else:
        filtered_dict = {filter_key:dict_to_filter[filter_key] for filter_key in filter_keys}
    return filtered_dict

def ischanged_dict(old_dict, new_dict, keys_ignore = []):
    """Summary
    
    Parameters
    ----------
    old_dict : TYPE
        Description
    new_dict : TYPE
        Description
    keys_ignore : list, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
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
def issparse(X, check_torch= True, check_scipy = True):
    """Summary
    
    Parameters
    ----------
    X : TYPE
        Description
    check_torch : bool, optional
        Description
    check_scipy : bool, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    #Checks if X is a sparse tensor or matrix
    #returns False if not sparse
    #if sc
    if check_torch:
        if isinstance(X,torch.Tensor):
            return 'sparse' in str(X.layout)
    if check_scipy:
        return sparse.issparse(X)

    return False
def is_nparray_or_sparse(X):
    """Summary
    
    Parameters
    ----------
    X : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    valid_X = [isinstance(X,np.ndarray),sparse.issparse(X)]
    if not any(valid_X):
        return False
    else:
        return True
def attr_exists_not_none(obj,attr):
    """Summary
    
    Parameters
    ----------
    obj : TYPE
        Description
    attr : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    return hasattr(obj,attr) and not getattr(obj, attr) is None

def make_tensor(X,keep_sparse=True):
    """Summary
    
    Parameters
    ----------
    X : TYPE
        Description
    keep_sparse : bool, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    
    Raises
    ------
    TypeError
        Description
    """
    if sparse.issparse(X):
        if keep_sparse:
            coo = sparse.coo_matrix(X)
            values = coo.data
            indices = np.vstack((coo.row, coo.col))
            i = torch.LongTensor(indices)
            v = torch.DoubleTensor(values)
            shape = coo.shape
            y = torch.sparse.DoubleTensor(i, v, torch.Size(shape))
        else:
            y = torch.from_numpy(X.toarray()).double()                
    elif isinstance(X, np.ndarray):
            y = torch.from_numpy(X).double()
    elif isinstance(X, torch.Tensor):
            y = X
    else:
        raise TypeError("Input matrix x is not sparse,"+
                 "np.array, or a torch tensor")
    return y

def stabilize_matrix(mat, read_cts = None, threshold = 0):
    """Summary
    
    Parameters
    ----------
    mat : TYPE
        Description
    read_cts : None, optional
        Description
    threshold : int, optional
        Description
    return_zero_indices : bool, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    
    if sparse.issparse(mat):
        nixs = mat.getnnz(0)>threshold # cols
        mixs = mat.getnnz(1)>threshold # rows
    else:
        nixs = nz_along(mat,axis=0) > threshold # cols
        mixs = nz_along(mat,axis=1) > threshold # rows
        
    mat = mat[mixs,:]
    mat = mat[:,nixs]

    nixs = np.argwhere(nixs).flatten()
    mixs = np.argwhere(mixs).flatten()
    
    return mat, mixs, nixs

def nz_along(M,axis=0):
    """
    Count the nonzeros along an axis of a `scipy.sparse.spmatrix` or `numpy.ndarray`.
    
    
    Parameters
    ----------
    M : scipy.sparse.spmatrix or numpy.ndarray
        M x N matrix to count the nonzeros in
    axis : int, default 0
        Axis to count nonzeros along. Follows numpy standard of directions:
        axis=0 moves down the rows, thus returning the number of nonzeros in each column,
        while axis=1 moves over the columns, returning the number of nonzeros in each row.
    
    Returns
    -------
    numpy.array
        Nonzeros along `axis` of `M`
    
    Raises
    ------
    TypeError
        Description
    ValueError
        Description
    TypeError
    ValueError
    """

    if not is_nparray_or_sparse(M):
        raise TypeError('M must be an np.ndarray or scipy.spmatrix, not %s' % str(type(M)))
    iaxis = int(axis)
    if iaxis != axis:
        raise TypeError('axis must be an integer, not %s' % str(type(axis)))
    else:
        axis = iaxis


    ndim = M.ndim

    if axis < 0:
        axis = ndim + axis
    if axis > M.ndim-1 or axis < 0:
        raise ValueError("axis out of range")
    if sparse.issparse(M):
        def countfun(m):
            """Summary
            
            Parameters
            ----------
            m : TYPE
                Description
            
            Returns
            -------
            TYPE
                Description
            """
            try:
                return m.getnnz(axis)
            except NotImplementedError as err:
                return m.tocsr().getnnz(axis)
    else:
        countfun = lambda m:  np.count_nonzero(m,axis=axis) #the number of nonzeros in each col
    return countfun(M)


def check_row_bound(X,gamma,nzs):
    """Summary
    
    Parameters
    ----------
    X : TYPE
        Description
    gamma : TYPE
        Description
    nzs : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    n = X.shape[1]
    zs = X.shape[1]-nzs
    for k in np.arange(np.floor(n/2),0,-1):
        bound = np.ceil(k*gamma)
        if not (np.where(zs>=n-k,1,0).sum() < bound):
            return True
    return False

def check_column_bound(X,gamma,nzs):
    """Summary
    
    Parameters
    ----------
    X : TYPE
        Description
    gamma : TYPE
        Description
    nzs : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    n = X.shape[1]
    zs = X.shape[0]-nzs
    for k in np.arange(np.floor((n*gamma)/2),0,-1):
        bound = np.ceil(k/gamma)

        if not (np.where(zs>=n*gamma-k,1,0).sum() < bound):
            return True
    return False

def farey(x, N):
    """Summary
    
    Parameters
    ----------
    x : TYPE
        Description
    N : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    #best rational approximation to X given a denominator no larger than N
    #obtained from https://www.johndcook.com/blog/2010/10/20/best-rational-approximation/
    a, b = 0, 1
    c, d = 1, 1
    while (b <= N and d <= N):
        mediant = float(a+c)/(b+d)
        if x == mediant:
            if b + d <= N:
                return a+c, b+d
            elif d > b:
                return c, d
            else:
                return a, b
        elif x > mediant:
            a, b = a+c, b+d
        else:
            c, d = a+c, b+d

    if (b > N):
        return c, d
    else:
        return a, b