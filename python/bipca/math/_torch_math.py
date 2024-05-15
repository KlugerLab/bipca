from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch import _linalg_utils as _utils
from torch.overrides import handle_torch_function, has_torch_function
from functools import reduce
import numpy as np

@torch.jit.script
def _parse_centering(A: torch.Tensor,
                    A_t: torch.Tensor,
                    centering:Union[bool,str]=False
                    )->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """ Compute the centering vectors for the input matrix A and its transpose A_t. If 
    A is dense, then the centered matrix is returned and the centering vectors are 0.
    Otherwise, the centering vectors are computed and the uncentered matrix is returned.
    """
    grand_mean = column_mean = row_mean = torch.tensor([0],dtype=A.dtype)
    if isinstance(centering,str):
        m, n = A.shape[-2:]
        if centering == 'both':
            if A.layout == torch.sparse_csr:
                row_mean = torch._sparse_csr_sum(A,-1,keepdim=True).to_dense()/n
                column_mean = torch._sparse_csr_sum(A,-2,keepdim=True).to_dense()/m
            else:
                row_mean = (A.sum(-1,keepdim=True)).to_dense()/n
                column_mean = (A.sum(-2,keepdim=True).to_dense()/m)
            grand_mean = row_mean.mean(-2)
        elif centering == 'row':
            if A.layout == torch.sparse_csr:
                row_mean = torch._sparse_csr_sum(A,-1,keepdim=True).to_dense()/n
            else:
                row_mean = (A.sum(-1,keepdim=True)).to_dense()/n
        elif centering == 'column':
            if A.layout == torch.sparse_csr:
                column_mean = torch._sparse_csr_sum(A,-2,keepdim=True).to_dense()/m
            else:
                column_mean = (A.sum(-2,keepdim=True)).to_dense()/m
    else:
        return A,A_t, row_mean,column_mean,grand_mean
    
    if A.layout != torch.strided:
        return A, A_t, row_mean, column_mean, grand_mean
    else:
        M = A - row_mean - column_mean + grand_mean
        return M,M.mT,row_mean,column_mean,grand_mean

@torch.jit.script
def _centered_matmul(A: torch.Tensor, grand_mean:torch.Tensor, row_mean:torch.Tensor, column_mean:torch.Tensor, M:torch.Tensor):
    """
    Performs a centered matrix multiplication.

    Args:
        A (torch.Tensor): The input matrix A of shape (M, N).
        grand_mean : The grand mean of shape (1, 1).
        row_mean (torch.Tensor): The row mean of shape (M, 1).
        column_mean (torch.Tensor): The column mean of shape (1, N).
        M (torch.Tensor): The centered matrix M of shape (N,P).

    Returns:
        torch.Tensor: The (M,P) result of the centered matrix multiplication.

    Raises:
        ValueError: If the shapes of the input tensors are not compatible.

        If the input tensor `A` is sparse, consider converting it to Compressed Sparse Row (CSR) format
        for faster computation.

    """

    matmul = _utils.matmul
    Ms0 = M.sum(0).unsqueeze(0)
    #compute each term
    # term 1 is just the matrix-vector
    term1 = matmul(A, M)
    #term2 is the row mean times the sum of the columns of M.
    # if row mean is 0, then we don't do this
    if torch.all(row_mean==0):
        term2 = row_mean
    else:
        term2 = matmul(row_mean, Ms0)
    #term3 is the column mean times the matrix
    if torch.all(column_mean==0):
        term3 = column_mean
    else:
        term3 = matmul(column_mean, M)
    #term4 is the grand mean times the sum of the columns of M
    if torch.all(grand_mean==0):
        term4 = grand_mean
    else:
        term4 = Ms0 * grand_mean
    return term1 - term2 - term3 + term4

@torch.jit.script
def _get_approximate_basis_double_centering(
    A: Tensor, q: int = 6, 
    n_iter: int = 5, 
    random_state: int = 42,
    grand_mean:Tensor=torch.tensor([0]),
    row_mean:Tensor=torch.tensor([0]),
    column_mean:Tensor=torch.tensor([0]),
    A_t: Optional[Tensor] = None,
) -> Tensor:
    """Return tensor :math:`Q` with :math:`q` orthonormal columns such
    that :math:`Q Q^H A` approximates :math:`A`. If `grand_mean`, 
    `row_mean`, and `column_mean` are specified,
    then :math:`Q` is such that :math:`Q Q^H (A - M)`
    approximates :math:`A - M` and :math:`M` is the double centering
    matrix formed using `grand_mean`, `row_mean`, and `column_mean`.

    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al, 2009.

    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              choosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Args::
        A (Tensor): the input tensor of size :math:`(m, n)`. If A is sparse, then
        it is recommended to use a sparse_csr matrix.

        q (int): the dimension of subspace spanned by :math:`Q`
                 columns.

        n_iter (int, optional): the number of subspace iterations to
                               conduct; ``n_iter`` must be a
                               nonnegative integer. In most cases, the
                               default value 2 is more than enough.

        grand_mean (Tensor, optional): the input tensor's mean of size :math:`(1,)`.

        row_mean (Tensor, optional): the input tensor's row means of size :math:`(m,1)`.
        
        column_mean (Tensor, optional): the input tensor's column means of size :math:`(1,n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """

    not_centering:bool = torch.all(grand_mean==0) and torch.all(row_mean==0) and torch.all(column_mean==0)
    m, n = A.shape[-2:]
    dtype = _utils.get_floating_dtype(A)
    matmul = _utils.matmul
    torch.manual_seed(random_state)
    R = torch.randn(n, q, dtype=dtype, device=A.device)
    
    # The following code could be made faster using torch.geqrf + torch.ormqr
    # but geqrf is not differentiable
    A_H = _utils.transjugate(A) if A_t is None else _utils.conjugate(A_t)
    if not_centering:
        Q = torch.linalg.qr(matmul(A, R)).Q
        for i in range(n_iter):
            Q = torch.linalg.qr(matmul(A_H, Q)).Q
            Q = torch.linalg.qr(matmul(A, Q)).Q
    else:
        gm_H = _utils.transjugate(grand_mean)
        cm_H = _utils.transjugate(row_mean)
        rm_H = _utils.transjugate(column_mean)
        Q = torch.linalg.qr(_centered_matmul(A,grand_mean,row_mean,column_mean,R)).Q
        for i in range(n_iter):
            Q = torch.linalg.qr(_centered_matmul(A_H,gm_H,rm_H,cm_H,Q)).Q
            Q = torch.linalg.qr(_centered_matmul(A,grand_mean,row_mean,column_mean,Q)).Q
    return Q
    

@torch.jit.script
def _svd_lowrank(
    A: Tensor,
    n_components: int = 6,
    n_oversamples: int = 10,
    n_iter: int = 5,
    random_state: int = 42,
    A_t: Optional[Tensor] = None,
    centering: Union[bool,str] = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Return the singular value decomposition ``(U, S, V)`` of a matrix,
    batches of matrices, or a sparse matrix :math:`A` such that
    :math:`A \approx U diag(S) V^T`. If `centering` is not False,
    SVD is computed for the matrix :math:`A - M` where :math:`M` is the
    double centering matrix formed using `grand_mean`, `row_mean`, and `column_mean`.

    .. note:: The implementation is based on the Algorithm 5.1 from
              Halko et al, 2009.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    .. note:: The input is assumed to be a low-rank matrix.

    .. note:: In general, use the full-rank SVD implementation
              :func:`torch.linalg.svd` for dense matrices due to its 10-fold
              higher performance characteristics. The low-rank SVD
              will be useful for huge sparse matrices that
              :func:`torch.linalg.svd` cannot handle.

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`. If A is sparse,
                    then it is recommended to use a sparse_csr matrix.  

        k (int, optional): desired rank of SVD.

        n_oversamples (int, optional): the number of oversamples to use.

        n_iter (int, optional): the number of subspace iterations to
                               conduct; n_iter must be a nonnegative
                               integer, and defaults to 5.
        
        random_state (int, optional): the random seed to use.

        A_t (Tensor, optional): the transpose of A. If A is sparse, then some speedups may be had by
                                providing the transpose of A in csr format. Defaults to None.

        double_centering (bool, optional): whether to double center the input matrix. Defaults to True.


    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <https://arxiv.org/abs/0909.4061>`_).

    """
    q = n_components+n_oversamples
    m, n = A.shape[-2:]
    matmul = _utils.matmul

    A_t = _utils.transpose(A) if A_t is None else A_t
    
    A, A_t,row_mean, column_mean, grand_mean = _parse_centering(A,A_t,centering)
    
    centering:bool = (torch.all(grand_mean==0)
                        and torch.all(row_mean==0)
                        and torch.all(column_mean==0))
    centering = not(centering)

    # Algorithm 5.1 in Halko et al 2009, slightly modified to reduce
    # the number conjugate and transpose operations
    if m < n or n > q:
        # computing the SVD approximation of a transpose in
        # order to keep B shape minimal (the m < n case) or the V
        # shape small (the n > q case)
        if centering:
            Q = _get_approximate_basis_double_centering(A_t, q, n_iter=n_iter, random_state=random_state,
            row_mean=_utils.transpose(column_mean),column_mean=_utils.transpose(row_mean), grand_mean=grand_mean, A_t=A)
        else:
            Q = _get_approximate_basis_double_centering(A_t, q, n_iter=n_iter, random_state=random_state, A_t=A)
        Q_c = _utils.conjugate(Q)
        if centering:
            B_t = _centered_matmul(A,grand_mean,row_mean,column_mean, Q_c)
        else:
            B_t = matmul(A, Q_c)
        U, S, Vh = torch.linalg.svd(B_t, full_matrices=False)
        V = Vh.mH
        V = Q.matmul(V)
    else:
        if centering:
            Q = _get_approximate_basis_double_centering(A, q, n_iter=n_iter, random_state=random_state,
            row_mean=row_mean,column_mean=column_mean, grand_mean=grand_mean, A_t=A_t)
        else:
            Q = _get_approximate_basis_double_centering(A, q, n_iter=n_iter, random_state=random_state, A_t=A_t)
        Q_c = _utils.conjugate(Q)
        if centering:
            B = _centered_matmul(A_t,grand_mean,_utils.transpose(column_mean),_utils.transpose(row_mean), Q_c)
        else:
            B = matmul(A_t, Q_c)
        B_t = _utils.transpose(B)
       
        U, S, Vh = torch.linalg.svd(B_t, full_matrices=False)
        V = Vh.mH
        U = Q.matmul(U)

    return U[...,:n_components], S[...,:n_components], V[...,:n_components]

@torch.jit.script
def _svd(A:torch.Tensor,
        vals_only:bool=False,
        centering:Union[bool,str]=False
        )->Tuple[Union[torch.Tensor,None],torch.Tensor,Union[torch.Tensor,None]]:
    """
    Compute the SVD of a matrix A with (optional) centering.

    Args:
        A (torch.Tensor): The input matrix A of shape (M, N).
        centering (Union[bool,str]): The type of centering to use. If 'row', 'column', or 'both', 
                        the matrix will be centered along the corresponding axis. Defaults to False.
        vals_only (bool): If True, only the singular values are returned.

    Returns:
        Tuple[torch.Tensor,torch.Tensor,torch.Tensor]: The left singular vectors, singular values, and right singular vectors.

    """
    A_t = _utils.transpose(A)
    A,_,_,_,_ = _parse_centering(A,A_t,centering)
    if vals_only:
        S = torch.linalg.svdvals(A)
        return None, S, None
    else:
        U,S,V = torch.linalg.svd(A)
        return U,S,V.T
    
@torch.jit.script
def _sinkhorn(T:Tensor,  row_sums:Optional[Tensor]=None, column_sums:Optional[Tensor]=None, T_t: Optional[Tensor]=None, n_iter:int=1000,tol:float=1e-6):
    """
    Applies the Sinkhorn algorithm to compute the row and column scaling factors of a matrix T.

    Args:
        T (Tensor): The input matrix of shape (M, N).
        row_sums (Tensor): The row sums of the matrix T of shape (M,).
        column_sums (Tensor): The column sums of the matrix T of shape (N,).
        T_t (Optional[Tensor]): The transpose of T. If T is sparse, it is best that T is a CSR matrix and T_t 
            is its transpose computed in CSR format. This is used for speedups in the case that T is sparse. 
            Defaults to None, in which T_t is computed at runtime.
        n_iter (int): The maximum number of iterations for the Sinkhorn algorithm. Defaults to 1000.
        tol (float): The tolerance for convergence. If the error between the current and previous row scaling factors is below this tolerance, the algorithm stops. Defaults to 1e-6.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the row scaling factors (rk) and column scaling factors (ck) of the matrix T.

    """
    if row_sums is None:
        row_sums = torch.full((T.shape[-2],),T.shape[-1],dtype=T.dtype,device=T.device)
    if column_sums is None:
        column_sums = torch.full((T.shape[-1],),T.shape[-2],dtype=T.dtype,device=T.device)
    rk = torch.ones_like(row_sums)
    ckp1:Tensor=torch.tensor([1])
    ck:Tensor=torch.tensor([1])
    if T_t is None:
        for i in range(n_iter):
            
            rk = torch.div(
                            row_sums, T.mv(torch.div(column_sums, _utils.transpose(T).mv(rk)))
                        )
            if (i+1) % 10 == 0 and tol > 0:
                rkp1= T.mv(torch.div(column_sums,_utils.transpose(T).mv(rk)))
                err = torch.abs(rk*rkp1-row_sums)
                if torch.any(err>tol):
                    rk = torch.div(row_sums,rkp1)
                    continue
                else:
                    break
        ck = torch.div(column_sums,_utils.transpose(T).mv(rk))
    else:
        for i in range(n_iter):
            
            rk = torch.div(
                            row_sums, T.mv(torch.div(column_sums, T_t.mv(rk)))
                        )
            if (i+1) % 10 == 0 and tol > 0:
                rkp1= T.mv(torch.div(column_sums,T_t.mv(rk)))
                err = torch.abs(rk*rkp1-row_sums)
                if torch.any(err>tol):
                    rk = torch.div(row_sums,rkp1)
                    continue
                else:
                    break
        ck = torch.div(column_sums,T_t.mv(rk))

        
    return rk,ck