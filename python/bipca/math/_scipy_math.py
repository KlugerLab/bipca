"""
math functions required for bipca when the input is a numpy array or scipy sparse matrix.
"""
from typing import Optional, Tuple,Literal, Union,TypeVar
import numpy as np
import numpy.typing as npt

import scipy
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator

from sklearn.utils.extmath import randomized_svd
ArrayLike = TypeVar("ArrayLike",npt.NDArray,  sparse.spmatrix)

def _double_offset_matrix(A: sparse.spmatrix, 
    grand_mean:float, 
    row_mean:npt.NDArray, 
    column_mean:npt.NDArray)->LinearOperator:

    if row_mean.ndim == 1:
        row_mean = row_mean[:,None]
        
    if column_mean.ndim == 1:
        column_mean = column_mean[None,:]


    def _double_centered_mm_mv(A, row_mean, column_mean, grand_mean, x):
        if x.ndim == 1:
            x = x[:,None]
        xs0 = x.sum(0)[None,:]
        return A @ x - row_mean @ xs0 - column_mean @ x + xs0 * grand_mean

    return LinearOperator(
        matvec=lambda x: _double_centered_mm_mv(A, row_mean, column_mean, grand_mean, x),
        matmat=lambda x: _double_centered_mm_mv(A, row_mean, column_mean, grand_mean, x),
        rmatvec=lambda x: _double_centered_mm_mv(A.T, column_mean.T, row_mean.T, grand_mean, x),
        rmatmat=lambda x: _double_centered_mm_mv(A.T, column_mean.T, row_mean.T, grand_mean, x),
        dtype=A.dtype,
        shape=A.shape,
    )
def _column_offset_matrix(X: sparse.spmatrix, offset:npt.NDArray)->LinearOperator:
    """Create an implicitly offset linear operator.

    This is used by PCA on sparse data to avoid densifying the whole data
    matrix.

    Params
    ------
        X : sparse matrix of shape (n_samples, n_features)
        offset : ndarray of shape (n_features,)

    Returns
    -------
    centered : LinearOperator
    """
    if offset.ndim == 1:
        offset = offset[:, None]
    XT = X.T
    return LinearOperator(
        matvec=lambda x: X @ x - offset @ x,
        matmat=lambda x: X @ x - offset @ x,
        rmatvec=lambda x: XT @ x - (offset * x.sum()),
        rmatmat=lambda x: XT @ x - offset.T @ x.sum(axis=0)[None, :],
        dtype=X.dtype,
        shape=X.shape,
    )

def _row_offset_matrix(X: sparse.spmatrix, offset:npt.NDArray)->LinearOperator:
    """Create an implicitly offset linear operator.

    This is used by PCA on sparse data to avoid densifying the whole data
    matrix.

    Params
    ------
        X : sparse matrix of shape (n_samples, n_features)
        offset : ndarray of shape (n_features,)

    Returns
    -------
    centered : LinearOperator
    """
    if offset.ndim == 1:
        offset = offset[:, None]
    XT = X.T

    return LinearOperator(
        matvec=lambda x: X @ x - (offset * x.sum()).squeeze(),
        matmat=lambda x: X @ x - offset * x.sum(axis=0)[None,:],
        rmatvec=lambda x: XT @ x - offset.T @ x,
        rmatmat=lambda x: XT @ x - offset.T @ x,
        dtype=X.dtype,
        shape=X.shape,
    )


def _centered_matrix(X: ArrayLike, 
    centering: Literal[False,'row','column','both']) -> Union[LinearOperator, npt.NDArray]:
    """Center a matrix along the rows and/or columns.

    If sparse, a LinearOperator is used. Otherwise, the centered matrix is returned.
    Params
    ------
        X (sparse.spmatrix or npt.NDArray) of shape (n_samples, n_features)

    Returns
    -------
    centered : LinearOperator or ndarray
    """
    if centering is False:
        return X
    if centering == 'row':
        row_mean = np.asarray(X.mean(axis=1))
        
        if sparse.issparse(X):
            return _row_offset_matrix(X, row_mean)
        else:
            return X - row_mean[:, None]
    elif centering == 'column':
        column_mean = np.asarray(X.mean(axis=0))
        if sparse.issparse(X):
            return _column_offset_matrix(X, column_mean)
        else:
            return X - column_mean[None, :]
    elif centering == 'both':
        row_mean = np.asarray(X.mean(axis=1))
        column_mean = np.asarray(X.mean(axis=0))
        grand_mean = row_mean.mean().squeeze()
        if sparse.issparse(X):
            return _double_offset_matrix(X, grand_mean, row_mean, column_mean)
        else:
            return X - row_mean[:,None] - column_mean[None,:] + grand_mean
    else:
        raise ValueError(f"Invalid centering: {centering}")
def _svd_lowrank(X: ArrayLike, 
    n_components: int = 6,
    random_state: int = 42,
    centering: Literal[False,'row', 'column', 'both'] = False,
    vals_only:bool=False) -> Tuple[
        Union[npt.NDArray,None], 
        npt.NDArray, 
        Union[npt.NDArray,None]]:
    """
    Compute the SVD of a matrix X with centering.

    Parameters:
        A (sparse.spmatrix): The input sparse matrix.
        n_components (int): The number of singular values and vectors to compute.
        random_state (int): The random seed to use.
        centering (str): The type of centering to use. If 'row', 'column', or 'both', 
                        the matrix will be centered along the corresponding axis.
        vals_only: If True, only the singular values are returned. U and V will be None.

    Returns:
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: The left singular vectors, singular values, and right singular vectors.
    """

    
    X_centered = _centered_matrix(X,centering)
    
    u, s, v = scipy.sparse.linalg.svds(
            X_centered,
            k=n_components,
            random_state = random_state,
            return_singular_vectors=not(vals_only),
            solver='propack'
        )

    return u,s,v.T

def _svd(X: ArrayLike, 
    centering: Literal[False,'row', 'column', 'both'] = True,
    vals_only: bool = False) -> Tuple[
        Union[npt.NDArray,None],
        npt.NDArray,
        Union[npt.NDArray,None]]:
    """
    Compute the SVD of a matrix X with (optional) centering.

    Parameters:
        A (sparse.spmatrix or npt.NDArray): The input sparse matrix.
        centering (str): The type of centering to use. If 'row', 'column', or 'both', 
                        the matrix will be centered along the corresponding axis.
        vals_only (bool): If True, only the singular values are returned.

    Returns:
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: The left singular vectors, singular values, and right singular vectors.
    """
    X_centered = _centered_matrix(X,centering)
    if vals_only:
        S = scipy.linalg.svd(X_centered,full_matrices=False,compute_uv=False)
        return None, S, None
    else:
        return scipy.linalg.svd(X_centered,full_matrices=False)

def _sinkhorn(X: npt.NDArray,
row_sums: Optional[npt.NDArray],
column_sums: Optional[npt.NDArray],
n_iter:int=1000,
tol:float=1e-6) -> Tuple[npt.NDArray, npt.NDArray]:
    """numpy/scipy implementation of sinkhorn biscaling"""
    row_sums = np.full((X.shape[0]),X.shape[1],dtype=X.dtype) if row_sums is None else row_sums
    col_sums = np.full((X.shape[1]),X.shape[0],dtype=X.dtype) if column_sums is None else column_sums
    rk = np.ones_like(row_sums)
    for i in range(n_iter):
        rk = np.divide(row_sums, X.dot(np.divide(col_sums, X.T.dot(rk))))

        if (i + 1) % 10 == 0 and tol > 0:
            rkp1 = X.dot(np.divide(col_sums, X.T.dot(rk)))
            err = np.abs(rk*rkp1 -row_sums)
            if np.any(err>tol):
                rk = np.divide(row_sums, rkp1)
                continue
            else:
                break

    ck = np.divide(col_sums,X.T.dot(rk))
    return rk, ck