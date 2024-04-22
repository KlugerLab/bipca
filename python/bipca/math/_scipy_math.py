"""
math functions required for bipca when the input is a numpy array or scipy sparse matrix.
"""
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt

import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator

from sklearn.utils.extmath import randomized_svd

def _double_offset_matrix(A: sparse.spmatrix, 
    grand_mean:float, 
    row_mean:npt.NDArray, 
    column_mean:npt.NDArray):

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

def _double_centered_matrix(A: sparse.spmatrix):
    """
    Returns a linear operator which computes double-centered matrix-vector and matrix-matrix
    products from a sparse A.

    Parameters:
        A (sparse.spmatrix): The input sparse matrix.

    Returns:
        LinearOperator: A linear operator that performs double centering on the input matrix.
    """
    row_mean = np.asarray(A.mean(axis=1))
    column_mean = np.asarray(A.mean(axis=0))
    grand_mean = row_mean.mean().squeeze()
    return _double_offset_matrix(A, grand_mean, row_mean, column_mean)

def _svd_lowrank(A: sparse.spmatrix, 
    k: Optional[int] = 6,
    n_oversamples: Optional[int] = 10,
    n_iter: Optional[int] = 5,
    random_state: Optional[int] = 42,
    double_centering: bool = True) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Compute the SVD of a sparse matrix A after double centering.

    Parameters:
        A (sparse.spmatrix): The input sparse matrix.
        k (int): The number of singular values and vectors to compute.
        n_oversamples (int): The number of oversamples to use in the randomized SVD.
        n_iter (int): The number of iterations to use in the randomized SVD.
        random_state (int): The random seed to use.


    Returns:
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: The left singular vectors, singular values, and right singular vectors.
    """
    if double_centering:
        A_double_centered = _double_centered_matrix(A)
        u, s, v = randomized_svd(
                A_double_centered,
                n_components=k,
                n_iter = n_iter,
                n_oversamples=n_oversamples,
                random_state = random_state,
            )
    else:
        u, s, v = randomized_svd(
                A,
                n_components=k,
                n_iter = n_iter,
                n_oversamples=n_oversamples,
                random_state = random_state,
            )
    return u[...,:k],s[...,:k],v.T[...,:k]


def _sinkhorn(X: npt.NDArray, row_sums: npt.NDArray, column_sums: npt.NDArray, n_iter:int=1000,
tol:float=1e-6) -> Tuple[npt.NDArray, npt.NDArray]:
    """numpy/scipy implementation of sinkhorn biscaling"""
    
    rk = np.ones_like(row_sums)
    for i in range(n_iter):
        rk = np.divide(row_sums, X.dot(np.divide(col_sums, X.T.dot(rk))))

        if (i + 1) % 10 == 0 and self.tol > 0:
            rkp1 = X.dot(np.divide(col_sums, X.T.dot(rk)))
            err = np.abs(rk*rkp1 -row_sums)
            if np.any(err>tol):
                rk = np.divide(row_sums, rkp1)
                continue
            else:
                break

    ck = np.divide(col_sums,X.T.dot(rk))

    return rk, ck