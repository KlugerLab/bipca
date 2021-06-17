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
def stabilize_matrix(mat, read_cts = None, threshold = 0, return_zero_indices = False):
    
    
    # Might need a method here that determines what the tolerance value is
    # Since in this experiment we are generating count data, the tolerance value can be 0. We will set to 1e-6 just in case
    
    tol = 1E-6
    if sparse.issparse(mat):
        nixs = mat.getnnz(1)>threshold
        mixs = mat.getnnz(0)>threshold
    else:
        nixs = nz_along(M,axis=1) > threshold
        mixs = nz_along(M,axis=0) > threshold
        
    mat = mat[nixs,:]
    mat = mat[:,mixs]

    nixs = np.argwhere(nixs).flatten()
    mixs = np.argwhere(mixs).flatten()

    if return_zero_indices == True:
        if read_cts is not None:     
            # if we have read counts to prune as well, we do that here
            read_cts = read_cts[~zero_cols]
            return mat, read_cts, [zero_rows, zero_cols]
        
        return mat, [zero_rows, zero_cols]

    
    return mat, nixs, mixs

def resample_matrix_safely(matrix,target_large_axis, seed = 42):
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
    m,n = matrix.shape
    gamma = m/n
    ny = int(target_large_axis)
    my = int(gamma * ny)
    rsubs = np.random.RandomState(seed=seed).permutation(m)
    csubs = np.random.RandomState(seed=seed).permutation(n)


    nixs = csubs[:ny]
    mixs = rsubs[:my]
    if sparse.issparse(matrix):
        nzrows = lambda m: np.diff(m.indptr)
        nzcols = lambda m: np.diff(m.T.indptr)
    else:
        nzcols = lambda m:  np.count_nonzero(m,axis=0) #the number of nonzeros in each col
        nzrows = lambda m:  np.count_nonzero(m,axis=1) #the number of nonzeros in each row

    new_submatrix = matrix[mixs,:][:,nixs]
    approximate_columns_per_row = np.round(1/gamma).astype(int)
    nz_cols = nzcols(new_submatrix)
    nz_rows = nzrows(new_submatrix)

    while check_column_bound(new_submatrix, gamma, nz_cols) or check_row_bound(new_submatrix, gamma, nz_rows):
        sparsest_cols = nixs[np.argsort(nz_cols)[:approximate_columns_per_row]]
        sparsest_col_bool = ~np.in1d(nixs,sparsest_cols)
        nixs = nixs[sparsest_col_bool]
        sparsest_rows = mixs[np.argsort(nz_rows)[0]]
        sparsest_row_bool = ~np.in1d(mixs,sparsest_rows)
        mixs = mixs[sparsest_row_bool]
        new_submatrix = matrix[mixs,:][:,nixs]
        approximate_columns_per_row = np.round(1/gamma).astype(int)
        nz_cols = nzcols(new_submatrix)
        nz_rows = nzrows(new_submatrix)
    return mixs,nixs

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
    
    """
    if sparse.issparse(M):
        nzrows = lambda m: np.diff(m.tocsr().indptr)
        nzcols = lambda m: np.diff(m.T.tocsr().indptr)
    else:
        nzcols = lambda m:  np.count_nonzero(m,axis=0) #the number of nonzeros in each col
        nzrows = lambda m:  np.count_nonzero(m,axis=1) #the number of nonzeros in each row
    if axis==0: #columns
        return nzcols(M)
    else:
        return nzrows(M)

# def resample_matrix(X,desired_size):
#     X_row_nzs = nz_along(X,axis=1)
#     X_col_nzs = nz_along(X,axis=0)

#     X_row_min = np.min(X_row_nzs)
#     X_col_min = np.min(X_col_min)
# def resample_matrix(X, desired_size, dim=1):
#     #get the aspect ratio of the wide matrix
#     #this function assumes that the input is already wide.
#     if X.shape[0] > X.shape[1]:
#         X = X.T
#         transposed = True
#     else:
#         transposed = False
#     M_X, N_X = X.shape
#     gamma = M_X/N_X # the aspect ratio we want to approximate
#     num,denom = farey(gamma, desired_size)
#     column_densities = np.count_nonzero(X, axis=0)
#     row_densities = np.count_nonzero(X,axis=1)
#     n_idx0 = np.random.permutation(N_X).astype(int)
#     m_idx0 = np.random.permutation(M_X).astype(int)

#     nixs = np.array([],dtype=int)
#     mixs = np.array([],dtype=int)
#     n_sampled = lambda : len(nixs)
#     #the initial set of columns
#     nixs = np.concatenate((nixs,n_idx0[:denom]))
#     n_idx = n_idx0[~np.in1d(n_idx0, nixs)]
#     m_idx = m_idx0[~np.in1d(m_idx0, mixs)]
#     current_submatrix = X[m_idx,:][:,nixs]
#     # choose the densest `num` rows 
#     densest = m_idx[np.argpartition(np.count_nonzero(current_submatrix,axis=1),num )[::-1][:num]]
#     mixs = np.hstack((mixs,densest)) #update to the current rows
#     m_idx = m_idx0[~np.in1d(m_idx0, mixs)]

#     current_submatrix = X[mixs,:][:,n_idx] # the submatrix consisting of the current rows and remaining columns
#     densest = n_idx[np.argpartition(np.count_nonzero(current_submatrix,axis=0),denom)[::-1]]

#     #check the columns and rows we have
#     new_submatrix = X[mixs,:][:,nixs]
#     new_gamma = new_submatrix.shape[0]/new_submatrix.shape[1]
#     nzs = np.count_nonzero(new_submatrix,axis=0) #the number of nonzeros in each column
#     while check_column_bound(new_submatrix, new_gamma, nzs):
#         #if we failed the column bound, then we swap columns
#         sparsest_idx = nixs[np.argmin(nzs)]
#         sparsest_bool = ~np.in1d(nixs,sparsest_idx)
#         nixs = nixs[sparsest_bool]
#         nixs = np.append(nixs,densest[0])
#         densest = densest[1:]
#         new_submatrix = X[mixs,:][:,nixs]
#         nzs = np.count_nonzero(new_submatrix,axis=0)

#     while n_sampled() < desired_size:
#         current_submatrix = X[m_idx,:][:,nixs]
#         # choose the densest `num` rows 
#         densest = m_idx[np.argpartition(np.count_nonzero(current_submatrix,axis=1),num )[::-1][:num]]
#         mixs = np.hstack((mixs,densest)) #update to the current rows
#         m_idx = m_idx0[~np.in1d(m_idx0, mixs)]

#         current_submatrix = X[mixs,:][:,n_idx] # the submatrix consisting of the current rows and remaining columns
#         densest = n_idx[np.argpartition(np.count_nonzero(current_submatrix,axis=0),denom)[::-1]]

#         #check the columns and rows we have
#         new_submatrix = X[mixs,:][:,nixs]
#         new_gamma = new_submatrix.shape[0]/new_submatrix.shape[1]
#         nzs = np.count_nonzero(new_submatrix,axis=0) #the number of nonzeros in each column
#         while check_column_bound(new_submatrix, new_gamma, nzs):
#             #if we failed the column bound, then we swap columns
#             sparsest_idx = nixs[np.argmin(nzs)]
#             sparsest_bool = ~np.in1d(nixs,sparsest_idx)
#             nixs = nixs[sparsest_bool]
#             nixs = np.append(nixs,densest[0])
#             densest = densest[1:]
#             new_submatrix = X[mixs,:][:,nixs]
#             nzs = np.count_nonzero(new_submatrix,axis=0)

#         current_submatrix = X[m_idx,:][:,nixs] # the submatrix consisting of the current columns and remaining rows
#         densest = m_idx[np.argpartition(np.count_nonzero(current_submatrix,axis=1),num)[::-1]]
#         #check the rows now
#         new_submatrix = X[mixs,:][:,nixs]
#         new_gamma = new_submatrix.shape[0]/new_submatrix.shape[1]
#         nzs = np.count_nonzero(new_submatrix,axis=1) #the number of nonzeros in each row
#         while check_row_bound(new_submatrix,new_gamma,nzs):
#             sparsest_idx = mixs[np.argmin(nzs)]
#             sparsest_bool = ~np.in1d(mixs,sparsest_idx)
#             mixs = mixs[sparsest_bool]
#             mixs = np.append(mixs,densest[0])
#             densest = densest[1:]
#             new_submatrix = X[mixs,:][:,nixs]
#             nzs = np.count_nonzero(new_submatrix,axis=1)

#         current_submatrix = X[mixs,:][:,n_idx] # the submatrix consisting of the current rows and remaining columns
#         densest = n_idx[np.argpartition(np.count_nonzero(current_submatrix,axis=0),denom)[::-1][:denom]]
#         nixs = np.hstack((nixs,densest))

#     return nixs,mixs


def check_row_bound(X,gamma,nzs):
    n = X.shape[1]
    zs = X.shape[1]-nzs
    for k in np.arange(np.floor(n/2),0,-1):
        bound = np.ceil(k*gamma)
        if not (np.where(zs>=n-k,1,0).sum() < bound):
            return True
    return False

def check_column_bound(X,gamma,nzs):
    n = X.shape[1]
    zs = X.shape[0]-nzs
    for k in np.arange(np.floor((n*gamma)/2),0,-1):
        bound = np.ceil(k/gamma)

        if not (np.where(zs>=n*gamma-k,1,0).sum() < bound):
            return True
    return False

def farey(x, N):
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