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
        adata.uns['bipca']['variance_estimator'] = obj.variance_estimator
        try:
            adata.uns['bipca']['fits']['kst'] = obj.kst
            adata.uns['bipca']['fits']['kst_pvals'] = obj.kst_pvals
        except:
            pass
        try:
            adata.uns['bipca']['plotting_spectrum'] = obj.plotting_spectrum
        except:
            pass
    return adata
###Other functions that the user may not want.


def fill_missing(X):
    if sparse.issparse(X):
        typ = type(X)
        X = sparse.coo_matrix(X)
        missing_entries = np.isnan(X.data)
        rows = X.row[missing_entries]
        cols = X.col[missing_entries]
        X.data[missing_entries] = 0
        observed_entries = np.ones_like(X)
        observed_entries[rows,cols] = 0
        X.eliminate_zeros()
        X = typ(X)
    else:
        missing_entries = np.isnan(X)
        observed_entries = np.ones_like(X)
        observed_entries[missing_entries] = 0
        X = np.where(missing_entries, 0, X)
    return X
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

def stabilize_matrix(X,*,order=False,threshold=None,
                    row_threshold=None,column_threshold=None,n_iters=0):
    """Filter the rows and/or columns of input matrix `mat` based on the number of
    nonzeros in each element
    
    Parameters
    ----------
    X : np.ndarray or scipy.spmatrix
        m x n input matrix
    order : bool or int, default False
        Perform filtering sequentially and specify order. Must be in [False, True, 0, 1].
        If False, filtering is performed using the original matrix.
        If True, perform filtering sequentially, starting with the rows.
        If integer, the integer specifies the first dimension to filter: 0 implies rows first,
            and 1 implies columns first.
    threshold : int, optional
        Global nonzero threshold for the rows and columns of the matrix.
        When `row_threshold` and `column_threshold` are not `None`,
        `threshold` is not used. Otherwise, sets the default condition for
        `row_threshold` and `column_threshold`
        If `threshold`, `row_threshold`, and `column_threshold` are None,
        defaults to 1.
    row_threshold, column_threshold  : int, optional
        Row (column) nonzero threshold of the matrix.
        Defaults to `threshold`. If `threshold` is `None`,
        defaults to 1.
    Returns
    -------
    Y : np.ndarray or scipy.spmatrix
        Output filtered matrix.
    indices : (np.ndarray(int), np.ndarray(int))
        Original indices in `X` used to produce `Y`,i.e.
        `X[indices[0],:][:,indices[1]] = Y`
    """
    if all([ele is None for ele in [threshold, row_threshold, column_threshold]]):
        threshold=1
    if row_threshold is None:
        row_threshold=1 if threshold is None else threshold
    if column_threshold is None:
        column_threshold=1 if threshold is None else threshold
    assert row_threshold is not None, "`row_threshold` somehow is not set. Please file a bug report."
    assert column_threshold is not None, "`column_threshold` somehow is not set. Please file a bug report."
    assert order in [False, True, 0, 1], "`order` must be in [False, True, 0, 1]."
    
    if order is not False:
        first_dimension = 0 if order is True else order #cast true to first dimension
        assert first_dimension in [0,1]
        second_dimension=1-first_dimension
        indices=[np.ones([X.shape[0]],bool), np.ones([X.shape[1]],bool)]
        threshold=[row_threshold,column_threshold]
        indices[first_dimension]= nz_along(X,axis=second_dimension) >= threshold[first_dimension]
        Y=X[indices[0],:][:,indices[1]]
        indices[second_dimension] = nz_along(Y,axis=first_dimension) >= threshold[second_dimension]
        Y=X[indices[0],:][:,indices[1]]
    else:
        indices=[nz_along(X,axis=1) >= row_threshold,
                nz_along(X,axis=0) >= column_threshold] # cols
        Y = X[indices[0],:]
        Y = Y[:,indices[1]]
    indices= [np.argwhere(ele).flatten() for ele in indices]
    
    if n_iters>0:
        niters=n_iters
        assert isinstance(n_iters,int), "n_iters must be an integer"
        converged = lambda indices: all([
            np.all(nz_along(X[indices[0],:][:,indices[1]],axis=0)>=column_threshold),
            np.all(nz_along(X[indices[0],:][:,indices[1]],axis=1)>=row_threshold)])
        while not converged(indices) and n_iters>0:
            n_iters-=1
            Y,indices2 = stabilize_matrix(Y,order=order,threshold=threshold,
                row_threshold=row_threshold,column_threshold=column_threshold,
                n_iters=0)
            indices = [inds0[indsnu] for inds0,indsnu in zip(indices,indices2)] #rebuild the indices
        if n_iters == 0 and not converged(indices):
            #didn't converge, recommend to user to increase n_iters.
            print(f"** Iterative filtering did not converge to target thresholds after {niters} iterations; "
                    "inspect output Y and indices and consider repeating `stabilize_matrix`.\n"
                    "\tTo start from the current filter, run \n"
                    f"\t\t`Y2, indices2 = stabilize_matrix(X=Y, order={order}, threshold={threshold},"
                    f"row_threshold={row_threshold}, column_threshold={column_threshold},"
                     " n_iters=extra_iterations).\n\tRemap `indices2` to original indices by noting that"
                     " Y2 = X[indices[0][indices2[0]],:][:,indices[1][indices2[1]]]")
    return Y, indices

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
def feature_scale(x,axis=-1):
    if axis==-1:
        return (x - np.min(x)) / (np.max(x)-np.min(x))
    else:
        return (x - np.min(x, axis=axis)[:,None]) / (np.max(x,axis=axis)[:,None]-np.min(x,axis=axis)[:,None])

class CachedFunction(object):
    def __init__(self,f, num_outs=1):
        self.f = f
        self.cache = {}
        self.num_outs = num_outs
    def compute_f(self, x):
        if isinstance(x,Iterable):
            x = tuple(x)
        if x in self.cache.keys(): #the value is already cached
            pass
        else: # compute the value
            self.cache[x] = self.f(x)
        return self.cache[x]
    def keys(self):
        return self.cache.keys()
    def __call__(self, x):
        if isinstance(x,Iterable):
            if isinstance(x,np.ndarray):
                typef = lambda z: np.array(z)
            else: #return eg a list
                typef = lambda z: type(x)(z)
            # evaluate the function
            # check that the number of outputs is stable
            outs = [[] for _ in range(self.num_outs)]
            for xx in x:
                y = self.compute_f(xx)
                if not(isinstance(y,Iterable)):
                    y = [y]
                if len(y) != self.num_outs:
                    raise ValueError("Number of outputs ({}) did not match ".format(len(y))+
                        "CachedFunction.num_outs ({})".format(self.num_outs))                    
                for yx in range(self.num_outs):
                    outs[yx] += [y[yx]]

            for lx in range(self.num_outs):
                outs[lx] = typef(outs[lx])
            outs = tuple(outs)
            if self.num_outs == 1:
                return outs[0]
            return outs

        return self.compute_f(x)