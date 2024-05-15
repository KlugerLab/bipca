"""Subroutines used to compute a BiPCA transform
"""
from functools import singledispatchmethod
from typing import Union, TypeVar, Literal,Optional, Tuple,Union
import numpy as np
import numpy.typing as npt
import sklearn as sklearn
import scipy as scipy
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import scipy.integrate as integrate
import scipy.sparse as sparse
import tasklogger
from anndata._core.anndata import AnnData
from scipy.stats import rv_continuous, kstest, gaussian_kde
import torch
from bipca.utils import (
    _is_vector,
    filter_dict_with_kwargs,
    ischanged_dict,
    nz_along,
    make_tensor,
    make_scipy,
    issparse,
    attr_exists_not_none,
)
from bipca.safe_basics import *
from bipca.base import *

ArrayLike = TypeVar("ArrayLike",npt.NDArray, torch.Tensor, sparse.spmatrix)
DenseArray = TypeVar("DenseArray",npt.NDArray, torch.Tensor)

class Sinkhorn(BiPCAEstimator):
    """
    Wrapper for Sinkhorn biscaling.

    Parameters
    ----------
    row_sums : array, optional
        Target row sums. Defaults to 1.
    column_sums : array, optional
        Target column sums. Defaults to 1.
    tol : float, default 1e-6
        Sinkhorn tolerance
    n_iter : int, default 100
        Number of Sinkhorn iterations.
    verbose : {0, 1, 2}, default 0
        Logging level
    logger : :log:`tasklogger.TaskLogger < >`, optional
        Logging object. By default, write to new logger.

    Attributes
    ----------
    row_sums : array
        Row sums used during fitting.
    column_sums : array
        Column sums used during fitting. 
    converged : bool
        True if the Sinkhorn algorithm converged.
    fit_ : bool
        True if the estimator has been fit.
    left : array
        Left (row-wise) scaling vector
    right : array
        Right (column-wise) scaling vector
    """

    def __init__(
        self,
        row_sums=None,
        column_sums=None,
        tol=1e-6,
        n_iter=100,
        logger=None,
        verbose=1,
        **kwargs,
    ):

        super().__init__(False,logger, verbose, **kwargs)

        self.row_sums = row_sums
        self.column_sums = column_sums
        self.tol = tol
        self.n_iter = n_iter
        self.converged = False
        self.fit_ = False
       
    @fitted_property
    def right(self) -> DenseArray:
        """Right scaling vector.
        """
        if attr_exists_not_none(self, "right_"):
            return self.right_
        else:
            raise NotFittedError("Estimator must be fit before accessing right scaling vector.")

    @right.setter
    def right(self, right: DenseArray):
        """Summary

        Parameters
        ----------
        right : TYPE
            Description
        """
        self.right_ = right

    @fitted_property
    def left(self) -> DenseArray:
        """Left scaling vector
        """
        if attr_exists_not_none(self, "left_"):
            return self.left_
        else:
            raise NotFittedError("Estimator must be fit before accessing left scaling vector.")

    @left.setter
    def left(self, left : DenseArray):
        """Summary

        Parameters
        ----------
        left : TYPE
            Description
        """
        self.left_ = left


    def fit_transform(self, X: ArrayLike)-> ArrayLike:
        """Fit the estimator and transform the input matrix X.
        """
        return self.fit(X).transform(X)
    
    @fitted
    def transform(self, X: ArrayLike) -> ArrayLike:
        """Scale the input by left and right Sinkhorn vectors. 

        Parameters
        ----------
        A : None, optional
            Description

        Returns
        -------
        type(X)
            Biscaled matrix of same type as input.

        """
        with self.logger.task(f"biscaling transform"):
            return self.scale(X)
    
    @fitted
    def scale(self, X: ArrayLike) -> ArrayLike:
        """Rescale matrix by Sinkhorn scalers.
        Estimator must be fit.

        Parameters
        ----------
        X : array, optional
            Matrix to rescale by Sinkhorn scalers.

        Returns
        -------
        array
            Matrix scaled by Sinkhorn scalerss.
        """
        return multiply(multiply(X, self.right), self.left[:, None])
    
    @fitted
    def unscale(self, X: ArrayLike) -> ArrayLike:
        """Applies inverse Sinkhorn scalers to input X.
        Estimator must be fit.

        Parameters
        ----------
        X : array, optional
            Matrix to unscale
        """
        return multiply(
            multiply(X, 1 / self.right), 1 / self.left[:, None]
        )
        
    def fit(self, X: ArrayLike):
        """
        Fits the left and right Sinkhorn matrix scalers to the input X

        Parameters
        ----------
        X : array, optional
            Matrix to scale

        Returns
        -------
        self
        """
        return self._sinkhorn(X)

    @singledispatchmethod
    def _sinkhorn(self,X: ArrayLike):
        raise NotImplementedError(f"Cannot fit type {type(X)}")

    @_sinkhorn.register(np.ndarray)
    @_sinkhorn.register(sparse.spmatrix)
    def _(self, X: Union[npt.NDArray, sparse.spmatrix]):
        from ._scipy_math import _sinkhorn
        # set the row and column sums if they have not been set already.
        if self.row_sums is None:
            self.row_sums = np.full((X.shape[0],), X.shape[1], dtype=X.dtype)
        if self.column_sums is None:
            self.column_sums = np.full((X.shape[1],), X.shape[0], dtype=X.dtype)
        #compute the sinkhorn scalers
        self.left, self.right = _sinkhorn(X,
        column_sums=self.column_sums,
        row_sums=self.row_sums, 
        n_iter=self.n_iter,
        tol=self.tol)
        #check if we converged
        if np.abs(self.scale(X).sum(0)-X.shape[0]).max()<=self.tol:
            self.converged = True
            self.fit_ = True
        else:
            self.converged = False
            self.fit_ = False
        return self

    @_sinkhorn.register(torch.Tensor)
    def _(self, X):
        from ._torch_math import _sinkhorn
        # check to see if X is a sparse tensor, if it's coo, convert to csr and also csr(T)
        # if it's dense don't do anything
        #set the row and column sums if they have not been set already.
        if self.row_sums is None:
            self.row_sums = torch.full((X.shape[0],), X.shape[1], dtype=X.dtype)
        if self.column_sums is None:
            self.column_sums = torch.full((X.shape[1],), X.shape[0], dtype=X.dtype)
        if issparse(X):
            X = X.to_sparse_coo()
            Xcsr = X.to_sparse_csr()
            XTcsr = X.T.to_sparse_csr()
            self.left, self.right = _sinkhorn(Xcsr, column_sums=self.column_sums,
                row_sums=self.row_sums,n_iter=self.n_iter, tol=self.tol,T_t = XTcsr)
        else:
            self.left, self.right = _sinkhorn(X, column_sums=self.column_sums,
                row_sums=self.row_sums,n_iter=self.n_iter, tol=self.tol)
        #check if we converged
        if issparse(X):
            criteria = torch.abs(self.scale(X).sum(0).to_dense()-X.shape[0]).max()
        else:
            criteria = torch.abs(self.scale(X).sum(0)-X.shape[0]).max()
        if criteria<=self.tol:
            self.converged = True
            self.fit_ = True
        else:
            self.converged = False
            self.fit_ = False
        return self

    # def _update_quadratic_parameters(self,sigma_nu,bhat,chat):
    #     #update the object to reflect changes in sigma
    #     # this occurs when sigma is updated in BiPCA by the shrinker.
    #     # a modification by sigma_nu is multiplying the variance matrix by sigma_nu
    #     # update the right scaling factor. Due to the update order, this is 
    #     # the only change that propagates from an update to the variance matrix
    #     self.right *= 1/sigma_nu
    #     if attr_exists_not_none(self, "_Z"):
    #         self.Z *= 1/sigma_nu
    #     if attr_exists_not_none(self, "_var"):
    #         self._var *= sigma_nu**2
    #     #update the variance parameters 
    #     if self.variance_estimator == "quadratic_2param":
    #         self.bhat = bhat
    #         self.chat = chat
    #         self.c = self.compute_c(self.chat)
    #         self.b = self.compute_b(self.bhat, self.c)
    #         self.bhat = (self.b * self.P) / (1 + self.c)
    #         self.chat = (1 + self.c - self.P) / (1 + self.c)


 

    # def estimate_variance(
    #     self, X, dist=None, q=None, bhat=None, chat=None, read_counts=None, **kwargs
    # ):
    #     """Estimate the element-wise variance in the matrix X

    #     Parameters
    #     ----------
    #     X : TYPE
    #         Description
    #     dist : str, optional
    #         Description
    #     q : int, optional
    #         Description
    #     **kwargs
    #         Description

    #     Returns
    #     -------
    #     TYPE
    #         Description
    #     """
    #     self.__set_operands(X)

    #     if dist is None:
    #         dist = self.variance_estimator
    #     if read_counts is None:
    #         read_counts = self.read_counts
    #     if read_counts is None:
    #         read_counts = X.sum(0)
    #     if q is None:
    #         q = self.q
    #     if bhat is None:
    #         bhat = self.bhat
    #     if chat is None:
    #         chat = self.chat

    #     if dist == "binomial":
    #         var = binomial_variance(X, read_counts)
    #     elif dist == "normalized":
    #         var = normalized_binomial(
    #             X,
    #             self.P,
    #             read_counts,
    #             mult=multiply,
    #             square=square,
    #         )
    #     elif dist == "quadratic_convex":
    #         var = quadratic_variance_convex(X, q=q)
    #     elif dist == "quadratic_2param":
    #         var = quadratic_variance_2param(X, bhat=bhat, chat=chat)
    #     elif dist == None:  # vanilla biscaling
    #         var = X
    #     else:
    #         var = general_variance(X)
    #     return var, read_counts

    #should this be a class method?
    def _extend_scalers(self, var_Y, axis, l0,r0):
        """ Extend scalers l0 and r0 to a new matrix with the same row or column identities as the original matrix
        """
        if axis == 0:
            #extend the rows 
            r1 = r0
            l1 = var_Y.shape[1]/sum(multiply(var_Y,r0[None,:]),axis=1)
        else:
            #extend the columns
            l1 = l0
            r1 = var_Y.shape[0]/sum(multiply(var_Y,l1[:,None]),axis=0)
        return l1,r1

    @fitted
    def predict(self, Y, prediction_axis=0, return_scalers=True):
        """predict: Given an input set of new points Y which share either the same row or column identities of the
        original matrix X, predict the transformed value of Y using new sinkhorn scalers
        
        *** BETA: Does not work with matrices with unknown entries (i.e. b and c are matrices)
        prediction_axis: int, optional
            The axis along which to predict the transformed values.  If 0, predict the row values.  If 1, predict the column values.
            The shape of Y must correspond to the prediction axis correctly.
        """
        #first, check if Y is the right shape
        if not([sz in (self.M,self.N) for sz in Y.shape]):
            raise ValueError("Input matrix Y must have the same number of rows or columns as the fitted matrix.")
        #now, interpret the prediction axis.
        if prediction_axis == 0:
            if Y.shape[1]!=self.N:
                raise ValueError("Prediction axis is 0 (rows) but Y does not have the same number of columns as the fitted matrix.")
        elif prediction_axis == 1:
            if Y.shape[0]!=self.M:
                raise ValueError("Prediction axis is 1 (columns) but Y does not have the same number of rows as the fitted matrix.")
        else:
            raise ValueError("Prediction axis must be 0 or 1.")
        var_Y = Y
        l = self.left
        r = self.right
 
        
        lnu,rnu = self._extend_scalers(Y, prediction_axis, self.left, self.right)
        if return_scalers:
            return multiply(multiply(Y, rnu), lnu[:,None]), lnu, rnu
        else:
            return multiply(multiply(Y, rnu), lnu[:,None])

class SVD(BiPCAEstimator):
    """
    Type-aware API for (optionally-centered) singular value decomposition. 
    Computes and stores the SVD of an `(M, N)` matrix `X = U@S@V^T`.
    Wraps torch, numpy, and scipy interfaces to SVD.

    Parameters
    ----------
    n_components : int > 0, optional
        Number of singular tuples to compute
        (By default the entire decomposition is performed).
    vals_only : bool, default False
        Only compute singular values.
    oversample_factor: int, default 10
        Oversampling factor for randomized SVD.
    random_state : int, default 42
        Random seed for randomized SVD.
    n_iter : int, default 5
        Number of iterations for randomized SVD.
    verbose : {0, 1, 2}, default 1
        Logging level
    logger : :log:`tasklogger.TaskLogger < >`, optional
        Logging object. By default, write to new logger.
    **kwargs
        Arguments for downstream SVD algorithm.

    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        centering: Literal[False,"row","column","double"] = False,
        vals_only: bool = False,
        oversample_factor: int = 10,
        random_state: int = 42,
        n_iter: int = 5,
        logger: Optional[tasklogger.TaskLogger] = None,
        verbose: Literal[0,1,2] = 1,
        **kwargs,
    ):

        super().__init__(False, logger, verbose, **kwargs)

        self.vals_only = vals_only
        self.oversample_factor = oversample_factor
        self.n_components = n_components
        self.centering = centering
        self.random_state = random_state
        self.n_iter = n_iter
    @fitted_property
    def U(self) -> DenseArray:
        """Return the left singular vectors that correspond to the largest `n_components` singular values of the fitted matrix

        .. Warning:: The object must be fit before requesting this attribute.
        """
        return self.U_

    @U.setter
    def U(self, U: DenseArray):
        self.U_ = U

    @fitted_property
    def V(self)-> DenseArray:
        """Return the right singular vectors that correspond to the largest `n_components` singular values of the fitted matrix

        .. Warning:: The object must be fit before requesting this attribute.

        """
        return self.V_

    @V.setter
    def V(self, V) -> DenseArray:
        self.V_ = V

    @fitted_property
    def S(self) -> DenseArray:
        """Return the largest `n_components` singular values of the fitted matrix

        .. Warning:: The object must be fit before requesting this attribute.
        """
        return self.S_

    @S.setter
    def S(self, S):
        self.S_ = S

    def fit(self, X:ArrayLike):
        """Run SVD on the input matrix X.
        """

        #first, check if X and n_components are valid
        if self.n_components in [0,-1,None] or self.n_components >= np.min(X.shape):
            self.n_components = np.min(X.shape)

        self.n_oversamples = self.oversample_factor * self.n_components
        assert self.n_components > 0, "n_components must be greater than 0."
        assert self.n_components <= np.min(X.shape), "n_components must be less than the minimum dimension of X."


        
        with self.logger.task('SVD'):
            if self.n_components < np.min(X.shape):
                U,S,V = self._partial_svd(X)
            else:
                U,S,V = self._exact_svd(X)
            ix = argsort(S, descending=True)

            self.S = S[ix]
            if U is not None:
                self.U = U[:, ix]
                ncols = X.shape[1]
                nS = len(S)
                if V.shape == (nS, ncols):
                    self.V = V[ix, :].T
                else:
                    self.V = V[:, ix]
        self.fit_ = True
        return self

    @singledispatchmethod
    def _partial_svd(self, X: ArrayLike):
        raise NotImplementedError(f"Cannot fit type {type(X)}")

    @_partial_svd.register(np.ndarray)
    @_partial_svd.register(sparse.spmatrix)
    def _(self, X: Union[npt.NDArray, sparse.spmatrix])-> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        from ._scipy_math import _svd_lowrank
        return _svd_lowrank(X, n_components=self.n_components, 
                               random_state=self.random_state,
                               centering=self.centering,
                               vals_only=self.vals_only)

    @_partial_svd.register(torch.Tensor)
    def _(self, X: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from ._torch_math import _svd_lowrank
        if X.layout == torch.sparse_coo:
            Xt = X.mT
            X = X.to_sparse_csr()
            Xt = Xt.to_sparse_csr()
        else:
            Xt = X.mT
        return _svd_lowrank(X, n_components=self.n_components, 
                               n_oversamples=self.n_oversamples,
                               n_iter=self.n_iter,
                               random_state=self.random_state,
                               centering=self.centering,A_t = Xt)
    @singledispatchmethod
    def _exact_svd(self, X: ArrayLike):
        raise NotImplementedError(f"Cannot fit type {type(X)} using exact SVD")

    @_exact_svd.register(np.ndarray)
    @_exact_svd.register(sparse.spmatrix)
    def _(self, X: Union[npt.NDArray, sparse.spmatrix])-> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        from ._scipy_math import _svd
        if sparse.issparse(X):
            raise NotImplementedError("Full-rank SVD is not supported for sparse " 
            "matrices. Please either supply a dense matrix or specify a rank smaller "
            "than the minimum dimension of the matrix.")
        else:
            return _svd(X, centering=self.centering,vals_only=self.vals_only)

    @_exact_svd.register(torch.Tensor)
    def _(self, X: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from ._torch_math import _svd
        # if it's sparse, then we raise an error and tell the user to convert to dense or
        # specify low rank
        if X.layout != torch.strided:
            raise NotImplementedError("Full-rank SVD is not supported for sparse " 
            "tensors. Please either supply a dense tensor or specify a rank smaller "
            "than the minimum dimension of the matrix.")
        else:
            return _svd(X, centering=self.centering, vals_only=self.vals_only)

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        """Compute an SVD and return the projection of the columns of X onto the first `N_components` singular vectors.

        Parameters
        ----------
        X : array
          The matrix to approximate using SVD.
        """
        return self.fit(X).transform()

    @fitted
    def transform(self, X:Optional[ArrayLike] = None):
        if X is None:
            return self.U[...,:self.n_components]*self.S[...,:self.n_components]
        else:
            return self.X @ self.V[...,:self.n_components]
  

class Shrinker(BiPCAEstimator):
    """
    Optimal singular value shrinkage

    Parameters
    ----------
    default_shrinker : {'frobenius','fro','operator','op','nuclear','nuc','hard','hard threshold','soft','soft threshold'}, default 'frobenius'
        shrinker to use when Shrinker.transform is called with no argument `shrinker`.
    rescale_svs : bool, default True
        Rescale the shrunk singular values back to the original domain using the noise variance.
    verbose : int, optional
        Description
    logger : :log:`tasklogger.TaskLogger < >`, optional
        Description

    Attributes
    ----------
    A : TYPE
        Description
    cov_eigs_ : TYPE
        Description
    default_shrinker : str,optional
        Description
    emp_qy_ : TYPE
        Description
    gamma_ : TYPE
        Description
    M_ : TYPE
        Description
    MP : TYPE
        Description
    N_ : TYPE
        Description
    quantile_ : TYPE
        Description
    rescale_svs : TYPE
        Description
    scaled_cov_eigs_ : TYPE
        Description
    scaled_cutoff_ : TYPE
        Description
    scaled_mp_rank_ : TYPE
        Description
    sigma_ : TYPE
        Description
    theory_qy_ : TYPE
        Description
    unscaled_mp_rank_ : TYPE
        Description
    y_ : TYPE
        Description
    nuclear

    Methods
    -------
    fit_transform : array
        Apply Sinkhorn algorithm and return biscaled matrix
    fit : array

    Deleted Attributes
    ------------------
    logger : TYPE
        Description

    """

    """How many `Shrinker` objects are there?"""

    def __init__(
        self,
        default_shrinker="frobenius",
        logger=None,
        verbose=1,
        **kwargs,
    ):
        super().__init__(False, logger, verbose, suppress, **kwargs)
        self.default_shrinker = default_shrinker
        self.rescale_svs = rescale_svs

    @fitted_property
    def sigma(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.sigma_

    @sigma.setter
    def sigma(self, sigma):
        """Summary

        Parameters
        ----------
        sigma : TYPE
            Description
        """
        self.sigma_ = sigma

    @fitted_property
    def scaled_mp_rank(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.scaled_mp_rank_

    @scaled_mp_rank.setter
    def scaled_mp_rank(self, scaled_mp_rank):
        """Summary

        Parameters
        ----------
        scaled_mp_rank : TYPE
            Description
        """
        self.scaled_mp_rank_ = scaled_mp_rank

    @fitted_property
    def scaled_cutoff(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.scaled_cutoff_

    @scaled_cutoff.setter
    def scaled_cutoff(self, scaled_cutoff):
        """Summary

        Parameters
        ----------
        scaled_cutoff : TYPE
            Description
        """
        self.scaled_cutoff_ = scaled_cutoff

    @fitted_property
    def unscaled_mp_rank(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.unscaled_mp_rank_

    @unscaled_mp_rank.setter
    def unscaled_mp_rank(self, unscaled_mp_rank):
        """Summary

        Parameters
        ----------
        unscaled_mp_rank : TYPE
            Description
        """
        self.unscaled_mp_rank_ = unscaled_mp_rank

    @fitted_property
    def gamma(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.gamma_

    @gamma.setter
    def gamma(self, gamma):
        """Summary

        Parameters
        ----------
        gamma : TYPE
            Description
        """
        self.gamma_ = gamma

    @fitted_property
    def emp_qy(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.emp_qy_

    @emp_qy.setter
    def emp_qy(self, emp_qy):
        """Summary

        Parameters
        ----------
        emp_qy : TYPE
            Description
        """
        self.emp_qy_ = emp_qy

    @fitted_property
    def theory_qy(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.theory_qy_

    @theory_qy.setter
    def theory_qy(self, theory_qy):
        """Summary

        Parameters
        ----------
        theory_qy : TYPE
            Description
        """
        self.theory_qy_ = theory_qy

    @fitted_property
    def quantile(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.quantile_

    @quantile.setter
    def quantile(self, quantile):
        """Summary

        Parameters
        ----------
        quantile : TYPE
            Description
        """
        self.quantile_ = quantile

    @fitted_property
    def scaled_cov_eigs(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.scaled_cov_eigs_

    @scaled_cov_eigs.setter
    def scaled_cov_eigs(self, scaled_cov_eigs):
        """Summary

        Parameters
        ----------
        scaled_cov_eigs : TYPE
            Description
        """
        self.scaled_cov_eigs_ = scaled_cov_eigs

    @fitted_property
    def cov_eigs(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.cov_eigs_

    @cov_eigs.setter
    def cov_eigs(self, cov_eigs):
        """Summary

        Parameters
        ----------
        cov_eigs : TYPE
            Description
        """
        self.cov_eigs_ = cov_eigs

    def fit(self, y, shape=None, sigma=None, theory_qy=None, q=None, suppress=None):
        """Summary

        Parameters
        ----------
        y : TYPE
            Description
        shape : None, optional
            Description
        sigma : None, optional
            Description
        theory_qy : None, optional
            Description
        q : None, optional
            Description
        suppress : None, optional
            Description

        Raises
        ------
        ValueError
            Description

        Returns
        -------
        TYPE
            Description
        """
        super().fit()
        if suppress is None:
            suppress = self.suppress
        if isinstance(y, AnnData):
            self.A = y
            if "SVD" in self.A.uns.keys():
                y = self.A.uns["SVD"]["S"]
            else:
                y = self.A.uns["bipca"]["S"]
        try:
            check_is_fitted(self)
            try:
                assert np.allclose(y, self.y_)  # if this fails, then refit
            except:
                self.__suppressable_logs__(
                    "Refitting to new input y", level=1, suppress=suppress
                )
                raise
        except:
            with self.logger.task("Shrinker fit"):
                if shape is None:
                    if _is_vector(y):
                        raise ValueError("Fitting requires shape parameter")
                    else:
                        if y.shape[0] > y.shape[1]:
                            y = y.T
                        shape = y.shape
                        y = np.diag(y)
                if shape[0] > shape[1]:
                    shape = (shape[1], shape[0])
                assert np.all(y.shape <= shape)
                y = np.sort(y)[::-1]
                # mp_rank, sigma, scaled_cutoff, unscaled_cutoff, gamma, emp_qy, theory_qy, q
                self.MP = MarcenkoPastur(gamma=shape[0] / shape[1])
                params = self._estimate_MP_params(
                    y=y, N=shape[1], M=shape[0], sigma=sigma, theory_qy=theory_qy, q=q
                )
                (
                    self.sigma,
                    self.scaled_mp_rank,
                    self.scaled_cutoff,
                    self.gamma,
                    self.quantile,
                    self.scaled_cov_eigs,
                    self.cov_eigs,
                ) = params
                self.M_ = shape[0]
                self.N_ = shape[1]
                self.y_ = y
                if (
                    self.scaled_mp_rank == len(y)
                    and sigma is not None
                    and len(y) != np.min(shape)
                ):
                    return self, False  # not converged, needs more eigs?
                else:
                    return self, True

        return self, True

    def _estimate_noise_variance(self, y=None, M=None, N=None, compensate_bulk=False):
        # estimate the noise variance sigma in y
        # key parameters change the behavior of this function:
        # compensate_bulk: extract the empirical median of the bulk data AFTER adjusting for the previous rank estimate.
        if y is None:
            y = self.y_
        if M is None:
            M = self._M
        if N is None:
            N = self._N

        y = np.sort(y)
        cutoff = np.sqrt(N) + np.sqrt(M)
        r = (y > cutoff).sum()
        if r == len(y):
            r = 1
            compensate_bulk = False
        sigma0 = 0
        sigma1 = 1

        i = 0
        obj = lambda sigma0, sigma1: ((sigma0) / (sigma1)) != 1
        while obj(sigma0, sigma1):
            i += 1
            sigma0 = sigma1
            bulk_size = M - r  # the actual size of the current bulk = # svs - rank
            if compensate_bulk:
                # compensate_bulk: consider a smaller set of singular values that is composed of only the current estimate of bulk.
                #  this changes the indexing and thus the quantile we are matching
                # this effect is particularly acute when the rank is large and we are given a partial estimate of the singular values.
                # suppose M = 200, len(y) = 20, and r = 19, bulk_size = 181.
                # in the normal setting, z_size = len(y), and the algorithm assumes that the empirical quantile is computed from the (1-len(z)/M) = (1-20/181) ~= 89th quantile
                # when compensate_bulk = True, z_size = len(y) - r = 1, bulk_size = 181.
                # then the empirical median is assumed to be computed from the (1-1/181) ~= 99th quantile
                # it is unclear to me which is more appropriate!
                z = y[:-r].copy()  # grab all but the largest r elements
            else:
                z = y

            emp_qy = None
            if len(z) >= bulk_size // 2:
                # we can compute the exact median, no need for iteration
                emp_qy = np.median(z)
                q = 0.5

            if emp_qy is None:
                # we need to compute a quantile from the lowest number in y
                # we have len(z) singular values. len(z)-1 are greater than this value
                q = (
                    bulk_size - len(z)
                ) / bulk_size  # assuming that there are no duplicates, then len(z)-1 values are greater than z[0],
                # and bulk_size -len(z) are less than z[0]
                emp_qy = z[0]  # recall that y is sorted, thus z is sorted.

            mp = MarcenkoPastur(M / N)
            # compute the theoretical qy
            theory_qy = mp.ppf(q)

            sigma1 = emp_qy / np.sqrt(N * theory_qy)

            scaled_svs = (y / (np.sqrt(N) * sigma1)) ** 2
            r = (scaled_svs > mp.b).sum()
            if r == len(y):
                break
            if i % 20 == 0:
                # hack to break early in event of runaway loop due to r not changing but sigma not stabilized.
                # if r doesn't change and q stays stable, then the problem can runaway.
                # this happens when a partial svd is supplied.
                # to fix: change algorithm to be on sigma?
                break
        return scaled_svs, sigma1, q

    def _estimate_MP_params(
        self, y=None, M=None, N=None, theory_qy=None, q=None, sigma=None
    ):
        """Summary

        Parameters
        ----------
        y : None, optional
            Description
        M : None, optional
            Description
        N : None, optional
            Description
        theory_qy : None, optional
            Description
        q : None, optional
            Description
        sigma : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        P *
        ------
        ValueError
            Description
        """
        with self.logger.task("MP Parameter estimate"):
            if np.any([y, M, N] == None):
                check_is_fitted(self)
            if y is None:
                y = self.y_
            if M is None:
                M = self._M
            if N is None:
                N = self._N
            if theory_qy is not None and q is None:
                raise ValueError("If theory_qy is specified then q must be specified.")
            assert M <= N

            # computing the noise variance
            if sigma is None:  # precomputed sigma
                _, sigma, q = self._estimate_noise_variance(y=y, M=M, N=N)
                self.logger.info(
                    "Estimated noise variance computed from the {:.0f}th percentile is {:.3f}".format(
                        np.round(q * 100), sigma**2
                    )
                )

            else:
                self.logger.info(
                    "Pre-computed noise variance is {:.3f}".format(sigma**2)
                )
            n_noise = np.sqrt(N) * sigma
            # scaling svs and cutoffs
            cov_eigs = (y / np.sqrt(N)) ** 2
            scaled_cov_eigs = y / n_noise
            scaled_cutoff = self.MP.b
            scaled_mp_rank = (scaled_cov_eigs**2 > scaled_cutoff).sum()
            if scaled_mp_rank == len(y):
                self.logger.info(
                    "\n ****** It appears that too few singular values were supplied to Shrinker. ****** \n ****** All supplied singular values are signal. ****** \n ***** It is suggested to refit this estimator with larger `n_components`. ******\n "
                )

            self.logger.info("Scaled Marcenko-Pastur rank is " + str(scaled_mp_rank))

        return (
            sigma,
            scaled_mp_rank,
            scaled_cutoff,
            self.MP.gamma,
            q,
            scaled_cov_eigs**2,
            cov_eigs,
        )

    def fit_transform(self, y=None, shape=None, shrinker=None, rescale=None):
        """Summary

        Parameters
        ----------
        y : None, optional
            Description
        shape : None, optional
            Description
        shrinker : None, optional
            Description
        rescale : None, optional,q
            Description

        Returns
        -------
        TYPE
            Description
        """
        self.fit(y, shape)
        if shrinker is None:
            shrinker = self.default_shrinker
        return self.transform(y=y, shrinker=shrinker)
    @fitted
    def transform(self, y=None, shrinker=None, rescale=None):
        """Summary

        Parameters
        ----------
        y : None, optional
            Description
        shrinker : None, optional
            Description
        rescale : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        if y is None:
            # the alternative is that we transform a non-fit y.
            y = self.y_
        if shrinker is None:
            shrinker = self.default_shrinker
        if rescale is None:
            rescale = self.rescale_svs
        with self.logger.task(
            "Shrinking singular values according to " + str(shrinker) + " loss"
        ):
            return _optimal_shrinkage(
                y,
                self.sigma_,
                self.M_,
                self.N_,
                self.gamma_,
                scaled_cutoff=self.scaled_cutoff_,
                shrinker=shrinker,
                rescale=rescale,
            )


def general_variance(X):
    """
    Estimated variance under a general model.

    Parameters
    ----------
    X : array-like
        Description

    Returns
    -------
    np.array
        Description

    """
    Y = MeanCenteredMatrix().fit_transform(X)
    if issparse(X, check_torch=False):
        Y = Y.toarray()
    Y = np.abs(Y) ** 2
    return Y


def quadratic_variance_convex(X, q=0):
    """
    Estimated variance under the quadratic variance count model with convex `q` parameter.

    Parameters
    ----------
    X : TYPE
        Description
    q : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    if issparse(X, check_torch=False):
        Y = X.copy()
        Y.data = (1 - q) * X.data + q * X.data**2
        return Y
    return (1 - q) * X + q * X**2


def quadratic_variance_2param(X, bhat=1.0, chat=0):
    """
    Estimated variance under the quadratic variance count model with 2 parameters.

    Parameters
    ----------
    X : TYPE
        Description
    q : int, optional
        Description

    Returns
    -------
    TYPE
        Description

    """
    if issparse(X, check_torch=False):
        Y = X.copy()
        Y.data = bhat * X.data + chat * X.data**2
        return Y
    return multiply(X, bhat) + multiply(square(X), chat)


def binomial_variance(X, counts):
    """
    Estimated variance under the binomial count model.

    Parameters
    ----------
    X : TYPE
        Description
    counts : TYPE
        Description
    mult : TYPE, optional
        Description
    square : TYPE, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    if np.any(counts <= 1):
        raise ValueError("Counts must be greater than 1.")
    if sparse.issparse(X) and isinstance(counts, int):
        var = X.copy()
        div = np.divide(counts, counts - 1)
        var.data = (var.data * div) - (var.data**2 * (1 / (counts - 1)))
        var.data = abs(var.data)
        var.eliminate_zeros()
    else:
        var = multiply(X, np.divide(counts, counts - 1)) - multiply(
            square(X), (1 / (counts - 1))
        )
        var = abs(var)

    return var


def normalized_binomial(X, p, counts, mult=lambda x, y: x * y, square=lambda x: x**2):
    mask = np.where(counts >= 2, True, False)
    X = X.copy()
    counts = counts.copy()
    X[mask] /= counts[mask]  # fill the elements where counts >= 2 with Xij / nij
    X[np.logical_not(mask)] = 0  # elmements where counts < 2 = 0. This is \bar{Y}
    counts[np.logical_not(mask)] = 2  # to fix nans, we truncate counts to 2.
    var = mult(p / counts, X) + mult(1 - 1 / counts - p, square(X))
    var /= 1 - 1 / counts

    return var


from scipy.stats import rv_continuous


class MarcenkoPastur(rv_continuous):
    """ "marcenko-pastur

    Attributes
    ----------
    gamma : TYPE
        Description
    """

    def __init__(self, gamma):
        """Summary

        Parameters
        ----------
        gamma : TYPE
            Description
        """
        if gamma > 1:
            gamma = 1 / gamma
        a = (1 - gamma**0.5) ** 2
        b = (1 + gamma**0.5) ** 2

        super().__init__(a=a, b=b)
        self.gamma = gamma

    def _pdf(self, x):
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
        m0 = lambda a: np.clip(a, 0, None)
        m0b = self.b - x
        m0b = np.core.umath.maximum(m0b, 0)
        m0a = x - self.a
        m0a = np.core.umath.maximum(m0a, 0)

        return np.sqrt(m0b * m0a) / (2 * np.pi * self.gamma * x)

    def cdf(self, x, which="analytical"):
        which = which.lower()
        if which not in ["analytical", "numerical"]:
            raise ValueError(
                f"which={which} is invalid."
                " MP.cdf requires which in"
                " ['analytical, 'numerical']."
            )
        if which == "numerical":
            return super()._cdf(x)
        else:
            return self.cdf_analytical(x)

    def cdf_analytical(self, x):
        with np.errstate(all="ignore"):
            x = np.asarray(x)
            const = 1 / (2 * np.pi * self.gamma)
            m0b = self.b - x
            m0a = x - self.a
            rx = np.sqrt(m0b / m0a, where=m0b / m0a > 0)
            term1 = np.pi * self.gamma
            term2 = np.sqrt(m0b * m0a, where=m0b * m0a > 0)
            term3 = -(1 + self.gamma) * np.arctan((rx**2 - 1) / (2 * rx))
            term4_numerator = self.a * rx**2 - self.b
            term4_denominator = 2 * (1 - self.gamma) * rx
            term4 = (1 - self.gamma) * np.arctan(term4_numerator / term4_denominator)
            output = const * (term1 + term2 + term3 + term4)
            output = np.where(x <= self.a, 0, output)
            output = np.where(x >= self.b, 1, output)
        return output


def L2(x, func1, func2):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    func1 : TYPE
        Description
    func2 : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    return np.square(func1(x) - func2(x))


def normalized_KS(y, mp, m, r):
    y_above_mask = y > mp.b
    y_below_mask = y < mp.a
    y_inrange_mask = np.logical_not(y_above_mask) * np.logical_not(y_below_mask)
    y_inrange = y[y_inrange_mask]

    return (y_above_mask.sum() + y_below_mask.sum()) / m + kstest(y_inrange, mp.cdf)[0]


def KS(y, mp):
    """Summary

    Parameters
    ----------
    y : TYPE
        Description
    mp : TYPE
        Description
    num : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    # x = np.linspace(mp.a*0.8, mp.b*1.2, num = num)
    # yesd = np.interp(x, np.flip(y), np.linspace(0,1,num=len(y),endpoint=False))
    # mpcdf = mp.cdf(x)
    # return np.amax(np.absolute(mpcdf - yesd))
    return kstest(y, mp.cdf)[0]


def L1(x, func1, func2):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    func1 : TYPE
        Description
    func2 : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    return np.absolute(func1(x) - func2(x))


# evaluate given loss function on a pdf and an empirical pdf (histogram data)
def emp_pdf_loss(pdf, epdf, loss=L2, start=0):
    """Summary

    Parameters
    ----------
    pdf : TYPE
        Description
    epdf : TYPE
        Description
    loss : TYPE, optional
        Description
    start : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    # loss() should have three arguments: x, func1, func2
    # note 0 is the left limit because our pdfs are strictly supported on the non-negative reals, due to the nature of sv's

    val = integrate.quad(lambda x: loss(x, pdf, epdf), start, np.inf, limit=100)[0]

    return val


def emp_mp_loss(mat, gamma=0, loss=L2, precomputed=True, M=None, N=None):
    """Summary

    Parameters
    ----------
    mat : TYPE
        Description
    gamma : int, optional
        Description
    loss : TYPE, optional
        Description
    precomputed : bool, optional
        Description
    M : None, optional
        Description
    N : None, optional
        Description

    Returns
    -------
    TYPE
        Description

    Raises
    ------
    RuntimeError
        Description
    """
    if precomputed:
        if M is None or N is None:
            raise RuntimeError()
    else:
        M = np.shape(mat)[0]
        N = np.shape(mat)[1]
    if gamma == 0:
        gamma = M / N

    if gamma >= 1:
        # have to pad singular values with 0
        if not precomputed:
            svs = np.linalg.svd(mat)[1]
            cov_eig = np.append(1 / N * svs**2, np.zeros(M - N))
        else:
            cov_eig = mat
        # hist = np.histogram(cov_eig, bins=np.minimum(np.int(N/4),60))
        hist = np.histogram(cov_eig, bins=np.int(4 * np.log2(5 * N)))
        esd = sp.stats.rv_histogram(hist).pdf

        # error at 0 is the difference between the first bin of the histogram and (1 - 1/gamma) = (M - N)/N
        err_at_zero = np.absolute(hist[0][0] - (1 - 1 / gamma))
        if loss == L2:
            err_at_zero = err_at_zero**2

        # we now start integrating AFTER the bin that contains the zeros
        start = hist[1][1]
        u_edge = (1 + np.sqrt(gamma)) ** 2
        # we integrate a little past the upper edge of MP, or the last bin of the histogram, whichever one is greater.
        end = 1.2 * np.maximum(u_edge, hist[1][-1])
        # end = 20
        val = (
            integrate.quad(
                lambda x: loss(x, lambda y: mp_pdf(y, gamma), esd), start, end
            )[0]
            + err_at_zero
        )

    else:
        if not precomputed:
            svs = np.linalg.svd(mat)[1]
            cov_eig = 1 / N * svs**2
        else:
            cov_eig = mat
        # hist = np.histogram(cov_eig, bins=np.minimum(np.int(N/4),60))
        hist = np.histogram(cov_eig, bins=np.int(4 * np.log2(5 * N)))
        esd = sp.stats.rv_histogram(hist).pdf

        u_edge = (1 + np.sqrt(gamma)) ** 2
        end = 1.2 * np.maximum(u_edge, hist[1][-1])
        # end = 20
        val = integrate.quad(
            lambda x: loss(x, lambda y: mp_pdf(y, gamma), esd), 0, end
        )[0]

    return val


def debias_singular_values(y, m, n, gamma=None, sigma=1):
    """Summary

    Parameters
    ----------
    y : TYPE
        Description
    m : TYPE
        Description
    n : TYPE
        Description
    gamma : None, optional
        Description
    sigma : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    # optimal shrinker derived by boris for inverting singular values to remove noise
    # if sigma is 1, then y may be normalized
    # if sigma is not 1, then y is unnormalized
    if gamma is None:
        gamma = m / n
    sigma2 = sigma**2
    threshold = sigma * (np.sqrt(n) + np.sqrt(m))

    nsigma2 = n * sigma2
    s = np.sqrt((y**2 + nsigma2 * (1 - gamma)) ** 2 - 4 * y**2 * nsigma2)
    s = y**2 - nsigma2 * (1 + gamma) + s
    s = np.sqrt(s / 2)
    return np.where(y > threshold, s, 0)


def _optimal_shrinkage(
    unscaled_y,
    sigma,
    M,
    N,
    gamma,
    scaled_cutoff=None,
    shrinker="frobenius",
    rescale=True,
    logger=None,
):
    """Summary

    Parameters
    ----------
    unscaled_y : TYPE
        Description
    sigma : TYPE
        Description
    M : TYPE
        Description
    N : TYPE
        Description
    gamma : TYPE
        Description
    scaled_cutoff : None, optional
        Description
    shrinker : str, optional
        Description
    rescale : bool, optional
        Description
    logger : None, optional
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
    if scaled_cutoff is None:
        scaled_cutoff = scaled_mp_bound(gamma)
    shrinker = shrinker.lower()

    ##defining the shrinkers
    frobenius = lambda y: 1 / y * np.sqrt((y**2 - gamma - 1) ** 2 - 4 * gamma)
    operator = (
        lambda y: 1
        / np.sqrt(2)
        * np.sqrt(y**2 - gamma - 1 + np.sqrt((y**2 - gamma - 1) ** 2 - 4 * gamma))
    )

    soft = lambda y: y - np.sqrt(scaled_cutoff)
    hard = lambda y: y

    # compute the scaled svs for shrinking
    n_noise = (np.sqrt(N)) * sigma
    scaled_y = unscaled_y / n_noise
    # assign the shrinker
    cond = scaled_y >= np.sqrt(scaled_cutoff)
    with np.errstate(
        invalid="ignore", divide="ignore"
    ):  # the order of operations triggers sqrt and x/0 warnings that don't matter.
        # this needs a refactor
        if shrinker in ["frobenius", "fro"]:
            shrunk = lambda z: np.where(cond, frobenius(z), 0)
        elif shrinker in ["operator", "op"]:
            shrunk = lambda z: np.where(cond, operator(z), 0)
        elif shrinker in ["soft", "soft threshold"]:
            shrunk = lambda z: np.where(cond, soft(z), 0)
        elif shrinker in ["hard", "hard threshold"]:
            shrunk = lambda z: np.where(cond, hard(z), 0)
        # elif shrinker in ['boris']:
        #     shrunk = lambda z: np.where(unscaled_y>)
        elif shrinker in ["nuclear", "nuc"]:
            x = operator(scaled_y)
            x2 = x**2
            x4 = x2**2
            bxy = np.sqrt(gamma) * x * scaled_y
            nuclear = (x4 - gamma - bxy) / (x2 * scaled_y)
            # special cutoff here
            cond = x4 >= gamma + bxy
            shrunk = lambda z: np.where(cond, nuclear, 0)
        else:
            raise ValueError("Invalid Shrinker")
        z = shrunk(scaled_y)
        if rescale:
            z = z * n_noise

    return z


def scaled_mp_bound(gamma):
    """Summary

    Parameters
    ----------
    gamma : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    scaled_bound = (1 + np.sqrt(gamma)) ** 2
    return scaled_bound


class KDE(rv_continuous):
    def __init__(self, x):
        a, b = np.min(x), np.max(x)
        self.kde = gaussian_kde(x.squeeze())
        super().__init__(a=a, b=b)

    def _pdf(self, x):
        return self.kde(x)

    def _cdf(self, x):
        from scipy.special import ndtr

        cdf = tuple(
            ndtr(np.ravel(item - self.kde.dataset) / self.kde.factor).mean()
            for item in x
        )
        return cdf


class MeanCenteredMatrix(BiPCAEstimator):
    """
    Mean centering and decentering
    
    Parameters
    ----------
    maintain_sparsity : bool, optional
        Only center the nonzero elements of the input. Default False
    consider_zeros : bool, optional
        Include zeros when computing mean. Default True
    conserve_memory : bool, default True
        Only store centering factors.
    verbose : {0, 1, 2}
        Logging level, default 1\
    logger : :log:`tasklogger.TaskLogger < >`, optional
        Logging object. By default, write to new logger
    suppress : Bool, optional.
        Suppress some extra warnings that logging level 0 does not suppress.
    
    Attributes
    ----------
    consider_zeros : TYPE
        Description
    fit_ : bool
        Description
    M : TYPE
        Description
    maintain_sparsity : TYPE
        Description
    N : TYPE
        Description
    X_centered : TYPE
        Description
    row_means
    column_means
    grand_mean
    X_centered
    maintain_sparsity
    consider_zeros
    force_type
    conserve_memory
    verbose
    logger
    suppress
    
    Deleted Attributes
    ------------------
    X__centered : TYPE
        Description
    """

    def __init__(
        self,
        maintain_sparsity=False,
        consider_zeros=True,
        conserve_memory=False,
        logger=None,
        verbose=1,
        suppress=True,
        **kwargs,
    ):
        """Summary

        Parameters
        ----------
        maintain_sparsity : bool, optional
            Description
        consider_zeros : bool, optional
            Description
        conserve_memory : bool, optional
            Description
        logger : None, optional
            Description
        verbose : int, optional
            Description
        suppress : bool, optional
            Description
        **kwargs
            Description
        """
        super().__init__(conserve_memory, logger, verbose, suppress, **kwargs)
        self.maintain_sparsity = maintain_sparsity
        self.consider_zeros = consider_zeros

    @memory_conserved_property
    @fitted
    def X_centered(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self._X_centered

    @X_centered.setter
    def X_centered(self, Mc):
        """Summary

        Parameters
        ----------
        Mc : TYPE
            Description
        """
        if not self.conserve_memory:
            self._X_centered = Mc

    @fitted_property
    def row_means(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self._row_means

    @fitted_property
    def column_means(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self._column_means

    @fitted_property
    def grand_mean(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self._grand_mean

    @fitted_property
    def rescaling_matrix(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        # Computes and returns the dense rescaling matrix
        mat = -1 * self._grand_mean * np.ones((self.N, self.M))
        mat += self._row_means[:, None]
        mat += self._column_means[None, :]
        return mat

    def __compute_grand_mean(self, X, consider_zeros=True):
        """Summary

        Parameters
        ----------
        X : TYPE
            Description
        consider_zeros : bool, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        if issparse(X, check_torch=False):
            nz = lambda x: x.nnz
            D = X.data
        else:
            nz = lambda x: np.count_nonzero(x)
            D = X
        if consider_zeros:
            nz = lambda x: np.prod(x.shape)

        return np.sum(D) / nz(X)

    def __compute_dim_means(self, X, axis=0, consider_zeros=True):
        """Summary

        Parameters
        ----------
        X : TYPE
            Description
        axis : int, optional
            Description
        consider_zeros : bool, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        # axis = 0 gives you the column means
        # axis = 1 gives row means
        if not consider_zeros:
            nzs = nz_along(X, axis)
        else:
            nzs = X.shape[axis]

        means = X.sum(axis) / nzs
        if issparse(X, check_torch=False):
            means = np.array(means).flatten()
        return means

    def fit_transform(self, X):
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
        self.fit(X)
        return self.transform(X)

    def fit(self, X):
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
        # let's compute the grand mean first.
        self._grand_mean = self.__compute_grand_mean(X, self.consider_zeros)
        self._column_means = self.__compute_dim_means(
            X, axis=0, consider_zeros=self.consider_zeros
        )
        self._row_means = self.__compute_dim_means(
            X, axis=1, consider_zeros=self.consider_zeros
        )
        self.N = X.shape[0]
        self.M = X.shape[1]
        self.fit_ = True
        return self

    @fitted
    def transform(self, X):
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
        # remove the means learned from .fit() from the input X.
        if self.maintain_sparsity:
            dense_rescaling_matrix = self.rescaling_matrix
            if issparse(X, check_torch=False):
                X = sparse.csr_matrix(X)
                X_nzindices = X.nonzero()
                X_c = X
                X_c.data = X.data - dense_rescaling_matrix[X_nzindices]
            else:
                X_nzindices = np.nonzero(X)
                X_c = X
                X_c[X_nzindices] = (
                    X_c[X_nzindices] - dense_rescaling_matrix[X_nzindices]
                )
        else:
            X_c = X - self.rescaling_matrix
        if isinstance(X_c, np.matrix):
            X_c = np.array(X_c)
        self.X_centered = X_c
        return X_c

    def scale(self, X):
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
        # Convenience synonym for transform
        return self.transform(X)

    def center(self, X):
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
        # Convenience synonym for transform
        return self.transform(X)

    @fitted
    def invert(self, X):
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
        # Subtract means from the data
        if self.maintain_sparsity:
            dense_rescaling_matrix = self.rescaling_matrix
            if issparse(X, check_torch=False):
                X = sparse.csr_matrix(X)
                X_nzindices = X.nonzero()
                X_c = X
                X_c.data = X.data + dense_rescaling_matrix[X_nzindices]
            else:
                X_nzindices = np.nonzero(X)
                X_c = X
                X_c[X_nzindices] = (
                    X_c[X_nzindices] + dense_rescaling_matrix[X_nzindices]
                )
        else:
            X_c = X + self.rescaling_matrix
        if isinstance(X_c, np.matrix):
            X_c = np.array(X_c)
        return X_c

    def uncenter(self, X):
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
        # Convenience synonym for invert
        return self.invert(X)

    def unscale(self, X):
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
        # Convenience synonym for invert
        return self.invert(X)

def library_normalize(X, scale='median'):
    """library_normalize:
    Normalize the data so that the rows sum to 1.

    Parameters
    ----------
    X : array-like or AnnData
        The input data to process.
    scale : {numbers.Number, 'median'}, default 1
        The scale factor to apply to the data
    Returns
    -------
    Y : array-like or AnnData"""

    libsize = np.asarray(sum(X, dim=1)).squeeze()
    if scale == "median":
        scale = np.median(libsize)
    
    
    scale = scale / libsize[:,None]

    return multiply(X, scale)
def minimize_chebfun(p, domain=[0, 1]):
    start, stop = domain
    pd = p.differentiate()
    pdd = pd.differentiate()
    try:
        q = pd.roots()  # the zeros of the derivative
        # minima are zeros of the first derivative w/ positive second derivative
        mi = q[pdd(q) > 0]
        if mi.size == 0:
            mi = np.linspace(start, stop, 100000)
        x = np.linspace(start, stop, 100000)
        x_ix = np.argmin(p(x))
        mi_ix = np.argmin(p(mi))
        if p(x)[x_ix] <= p(mi)[mi_ix]:
            q = x[x_ix]
        else:
            q = mi[mi_ix]
    except IndexError:
        x = np.linspace(start, stop, 100000)
        x_ix = np.argmin(p(x))
        q = x[x_ix]

    return q


def find_linearly_dependent_columns(R):
    # generate the matrix of inner products btwn columns of R

    inner = R.T @ R

    norm = np.linalg.norm(R, ord=2, axis=0)

    return np.argwhere(
        np.abs(np.outer(norm, norm) - inner) + np.eye(inner.shape[0]) < 1e-4
    )
