"""Subroutines used to compute a BiPCA transform
"""
from collections.abc import Iterable
from collections import defaultdict

import numpy as np
import sklearn as sklearn
import scipy as scipy
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import randomized_svd
import scipy.integrate as integrate
import scipy.sparse as sparse
import scipy.linalg
import tasklogger
from sklearn.base import clone
from anndata._core.anndata import AnnData
from scipy.stats import rv_continuous, kstest, gaussian_kde
import torch
from .utils import (
    _is_vector,
    filter_dict_with_kwargs,
    ischanged_dict,
    nz_along,
    make_tensor,
    make_scipy,
    issparse,
    attr_exists_not_none,
)
from .safe_basics import *
from .base import *


class Sinkhorn(BiPCAEstimator):
    """
    Sinkhorn biwhitening and biscaling.

     By default (`variance_estimator` is one of `binomial`, `normalized`,
     `poisson`, or `empirical`), this class performs biwhitening:
      1) A variance matrix is estimated for the input data,
      2) Left and right scaling factors that biscale the matrix to have constant
      row and column sums
     is biscaled, and the left and right scaling factors


    Parameters
    ----------
    variance : array, optional
        variance matrix for input data to be biscaled
        (default variance is estimated from data using the model).
    variance_estimator : {'binomial', 'quadratic_convex','quadratic_2param',
    'empirical',`normalized`, None}, optional

    row_sums : array, optional
        Target row sums. Defaults to 1.
    col_sums : array, optional
        Target column sums. Defaults to 1.
    read_counts : array
        The expected `l1` norm of each column.
        Used when `variance_estimator=='binomial'`.
        (Defaults to the sum of the input data).
    tol : float, default 1e-6
        Sinkhorn tolerance
    n_iter : int, default 100
        Number of Sinkhorn iterations.

    conserve_memory : bool, optional
        NotImplemented. Save output scaled matrix as a factor.
    backend : {'scipy', 'torch', 'torch_gpu'}, optional
        Computation engine. Default torch.
    verbose : {0, 1, 2}, default 0
        Logging level
    logger : :log:`tasklogger.TaskLogger < >`, optional
        Logging object. By default, write to new logger.

    Attributes
    ----------
    backend : str
        Computation engine used for Sinkhorn iterations.
    col_sums : ndarray or None
        Target column sums for biscaling.
    column_error : float
        Column-wise Sinkhorn convergence error after fitting.
    column_error_ : float
        Column-wise Sinkhorn error (stored attribute).
    converged : bool
        Whether the Sinkhorn iteration converged within tolerance.
    fit_ : bool
        Whether the estimator has been fitted.
    left : ndarray
        Left (row) scaling vector. Square root of raw scalers when
        using a variance estimator, raw scalers otherwise.
    left_ : ndarray
        Left scaling vector (stored attribute).
    n_iter : int
        Maximum number of Sinkhorn iterations.
    q : float
        Convex combination parameter for the quadratic variance model.
        Set to 0 when ``variance_estimator`` is None.
    read_counts : ndarray or int
        Expected column sums (library sizes) used by the binomial
        variance estimator.
    right : ndarray
        Right (column) scaling vector. Square root of raw scalers when
        using a variance estimator, raw scalers otherwise.
    right_ : ndarray
        Right scaling vector (stored attribute).
    row_error : float
        Row-wise Sinkhorn convergence error after fitting.
    row_error_ : float
        Row-wise Sinkhorn error (stored attribute).
    row_sums : ndarray or None
        Target row sums for biscaling.
    tol : float
        Convergence tolerance for Sinkhorn iterations.
    var : ndarray
        Entry-wise variance matrix estimated from the data.
    variance_estimator : str or None
        Name of the variance model used to estimate entry-wise variances.
    X : ndarray or sparse matrix
        Input data matrix stored after fitting.
    X_ : ndarray or sparse matrix
        Input data (stored attribute).
    Z : ndarray or sparse matrix
        Biscaled (transformed) data matrix.

    """

    def __init__(
        self,
        variance=None,
        variance_estimator="quadratic_convex",
        row_sums=None,
        col_sums=None,
        read_counts=None,
        tol=1e-6,
        P=1,
        q=1,
        b=None,
        bhat=1.0,
        c=None,
        chat=0,
        n_iter=100,
        conserve_memory=False,
        backend="torch",
        logger=None,
        verbose=1,
        suppress=True,
        **kwargs,
    ):
        """Initialize the Sinkhorn biscaling estimator.

        Parameters
        ----------
        variance : array-like or None, optional
            Pre-computed entry-wise variance matrix. If None, the variance
            is estimated from the data using ``variance_estimator``.
        variance_estimator : str or None, optional
            Variance model to use. One of ``'binomial'``, ``'quadratic_convex'``,
            ``'quadratic_2param'``, ``'normalized'``, ``'empirical'``, or None
            for vanilla biscaling without variance estimation.
        row_sums : array-like or None, optional
            Target row sums for biscaling. Defaults to the number of columns.
        col_sums : array-like or None, optional
            Target column sums for biscaling. Defaults to the number of rows.
        read_counts : array-like or int or None, optional
            Expected column sums (library sizes) for the binomial variance
            estimator. Defaults to the column sums of the input data.
        tol : float, optional
            Convergence tolerance for the Sinkhorn iteration.
        q : float, optional
            Convex combination parameter for the quadratic variance model.
        n_iter : int, optional
            Maximum number of Sinkhorn iterations.
        conserve_memory : bool, optional
            If True, intermediate matrices are not stored after fitting.
        backend : str, optional
            Computation engine. One of ``'scipy'``, ``'torch'``, or
            ``'torch_gpu'``.
        logger : tasklogger.TaskLogger or None, optional
            Logging object. If None, a new logger is created.
        verbose : int, optional
            Logging verbosity level.
        suppress : bool, optional
            If True, suppress warnings from redundant fit calls.
        **kwargs
            Additional keyword arguments passed to the base estimator.
        """
        super().__init__(conserve_memory, logger, verbose, suppress, **kwargs)

        self.read_counts = read_counts
        self.row_sums = row_sums
        self.col_sums = col_sums
        self.tol = tol
        self.n_iter = n_iter
        self.variance_estimator = variance_estimator
        if variance_estimator is None:
            q = 0
        self.q = q
        self.init_quadratic_params(b, bhat, c, chat)
        self.P = P
        self.backend = backend
        self.converged = False
        self._issparse = None
        self.__typef_ = (
            lambda x: x
        )  # we use this for type matching in the event the input is sparse.
        self._Z = None
        self.X_ = None
        self._var = variance
        self.__xtype = None
        self.fit_ = False

    def init_quadratic_params(self, b, bhat, c, chat):
        if self.variance_estimator == "quadratic_2param":
            if b is not None:
                ## A b value was specified
                if c is None:
                    raise ValueError(
                        "Quadratic variance parameter b was"
                        + " specified, but c was not. Both must be specified."
                    )
                else:
                    bhat_tmp = b / (1 + c)
                    # check that if bhat was specified that they match b
                    bhat = bhat_tmp
                    chat_tmp = c / (1 + c)
                    chat = chat_tmp
        self.bhat = bhat
        self.chat = chat
        self.b = b
        self.c = c

    def compute_b(self, bhat, c):
        if bhat is None:
            return None
        return bhat * (1 + c)

    def compute_c(self, chat):
        if chat is None:
            return None
        return chat / (1 - chat)

    @property
    def var(self):
        """Return the entry-wise variance matrix estimated by estimate_variance.

        Returns
        -------
        ndarray or sparse matrix
            The estimated variance matrix with the same shape as the input data.

        Raises
        ------
        RuntimeError
            If ``conserve_memory`` is True, since the variance matrix is not
            stored in that mode.
        """
        if not self.conserve_memory:
            return self._var
        else:
            raise RuntimeError(
                "Since conserve_memory is true, var can only be obtained by "
                + "calling Sinkhorn.estimate_variance(X, Sinkhorn.variance_estimator, q = Sinkhron.q)"
            )

    @var.setter
    def var(self, var):
        """Set the variance matrix.

        Only stores the value when ``conserve_memory`` is False.

        Parameters
        ----------
        var : ndarray or sparse matrix
            The entry-wise variance matrix to store.
        """
        if not self.conserve_memory:
            self._var = var

    @property
    def variance(self):
        """Alias for :attr:`var`. Return the entry-wise variance matrix.

        Returns
        -------
        ndarray or sparse matrix
            The estimated variance matrix with the same shape as the input data.

        Raises
        ------
        RuntimeError
            If ``conserve_memory`` is True, since the variance matrix is not
            stored in that mode.
        """
        if not self.conserve_memory:
            return self._var
        else:
            raise RuntimeError(
                "Since conserve_memory is true, variance can only be obtained by "
                + "calling Sinkhorn.estimate_variance(X, Sinkhorn.variance_estimator, q = Sinkhron.q)"
            )

    @fitted_property
    def Z(self):
        """Return the biscaled (transformed) data matrix.

        Computed lazily as ``scale(X)`` if not already cached.

        Returns
        -------
        ndarray or sparse matrix
            The biscaled data matrix, same type as the input.

        Raises
        ------
        RuntimeError
            If ``conserve_memory`` is True, since the transformed matrix is
            not stored in that mode.
        """
        if not self.conserve_memory:
            if self._Z is None:
                return self.__type(self.scale(self.X))
            return self._Z
        else:
            raise RuntimeError(
                "Since conserve_memory is true, Z can only be obtained by "
                + "calling Sinkhorn.transform(X)"
            )

    @Z.setter
    def Z(self, Z):
        """Set the biscaled data matrix.

        Only stores the value when ``conserve_memory`` is False.

        Parameters
        ----------
        Z : ndarray or sparse matrix
            The biscaled data matrix to store.
        """
        if not self.conserve_memory:
            self._Z = Z

    @fitted_property
    def right(self):
        """Return the right (column) scaling vector.

        Returns
        -------
        ndarray
            The right scaling vector of length N (number of columns).
        """
        if attr_exists_not_none(self, "right_"):
            return self.right_
        return self.right_

    @right.setter
    def right(self, right):
        """Set the right (column) scaling vector.

        Parameters
        ----------
        right : ndarray
            The right scaling vector to store.
        """
        self.right_ = right

    @fitted_property
    def left(self):
        """Return the left (row) scaling vector.

        Returns
        -------
        ndarray or None
            The left scaling vector of length M (number of rows), or None
            if not yet computed.
        """
        if attr_exists_not_none(self, "left_"):
            return self.left_
        else:
            return None

    @left.setter
    def left(self, left):
        """Set the left (row) scaling vector.

        Parameters
        ----------
        left : ndarray
            The left scaling vector to store.
        """
        self.left_ = left

    @fitted_property
    def row_error(self):
        """Return the row-wise Sinkhorn convergence error.

        Returns
        -------
        float
            Maximum absolute deviation of row sums from their targets.
        """
        return self.row_error_

    @row_error.setter
    def row_error(self, row_error):
        """Set the row-wise Sinkhorn convergence error.

        Parameters
        ----------
        row_error : float
            The row convergence error to store.
        """
        self.row_error_ = row_error

    @fitted_property
    def column_error(self):
        """Return the column-wise Sinkhorn convergence error.

        Returns
        -------
        float
            Maximum absolute deviation of column sums from their targets.
        """
        return self.row_error_

    @column_error.setter
    def column_error(self, column_error):
        """Set the column-wise Sinkhorn convergence error.

        Parameters
        ----------
        column_error : float
            The column convergence error to store.
        """
        self.column_error_ = column_error

    def __is_valid(self, X, row_sums, col_sums):
        """Verify input data is non-negative and shapes match target sums.

        Parameters
        ----------
        X : ndarray or sparse matrix
            The input data matrix of shape ``(M, N)``.
        row_sums : ndarray
            Target row sums of length M.
        col_sums : ndarray
            Target column sums of length N.
        """
        eps = 1e-3
        assert amin(X)>=0, "Matrix is not non-negative"
        assert np.shape(X)[0] == np.shape(row_sums)[0], "Row dimensions mismatch"
        assert np.shape(X)[1] == np.shape(col_sums)[0], "Column dimensions mismatch"

        # sum(row_sums) must equal sum(col_sums), at least approximately
        # assert (
        #     np.abs(np.sum(row_sums) - np.sum(col_sums)) < eps
        # ), "Rowsums and colsums do not add up to the same number"

    def __type(self, M):
        """Cast matrix to the same type as the original fitted input.

        Parameters
        ----------
        M : ndarray or sparse matrix
            Matrix to cast.

        Returns
        -------
        ndarray or sparse matrix
            The matrix cast to the type of the original input data.
        """
        check_is_fitted(self)
        if isinstance(M, self.__xtype):
            return M
        else:
            return self.__typef_(M)

    def fit_transform(self, X=None):
        """Fit the Sinkhorn estimator and return the biscaled data.

        If already fitted, attempts to transform first; refits on failure.

        Parameters
        ----------
        X : array-like or AnnData or None, optional
            Input data matrix. If None, uses the previously fitted data.

        Returns
        -------
        ndarray or sparse matrix
            The biscaled data matrix.
        """
        if X is None:
            check_is_fitted(self)

        if self.fit_:
            try:
                return self.transform(A=X)
            except:
                self.fit(X)
        else:
            self.fit(X)
        return self.transform(A=X)

    def transform(self, A=None):
        """Scale the input by left and right Sinkhorn vectors.

        Parameters
        ----------
        A : array-like, AnnData, or None, optional
            Data matrix to transform. If None, uses the stored fitted data.

        Returns
        -------
        ndarray or sparse matrix
            Biscaled matrix of same type as input.
        """
        check_is_fitted(self)

        if isinstance(A, AnnData):
            X = A.X
        else:
            X = A
        if X is None:
            if not self.conserve_memory:
                X = self.X
        sparsestr = ""
        if X is not None:
            if sparse.issparse(X):
                sparsestr = "sparse"
            else:
                sparsestr = "dense"
        with self.logger.task(f"{sparsestr} Biscaling transform"):
            if X is not None:
                self.__set_operands(X)
                if self.conserve_memory:
                    return self.__type(self.scale(X))
                else:
                    self.Z = self.__type(self.scale(X))
            output = self.Z
        return output

    def scale(self, X=None):
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
        check_is_fitted(self)
        if X is None:
            X = self.X
        self.__set_operands(X)

        if X.shape[0] == self.M: 
            return multiply(multiply(X, self.right), self.left[:, None])
        else:
            return multiply(
                multiply(X, self.right[:, None]), self.left[None, :]
            )

    def unscale(self, X=None):
        """Applies inverse Sinkhorn scalers to input X.
        Estimator must be fit.

        Parameters
        ----------
        X : array, optional
            Matrix to unscale

        Returns
        -------
        array
            Matrix unscaled by the inverse Sinkhorn scalers
        """
        check_is_fitted(self)
        if X is None:
            return self.X
        self.__set_operands(X)

        if X.shape[0] == self.M:
            return multiply(
                multiply(X, 1 / self.right), 1 / self.left[:, None]
            )
        else:
            return multiply(
                multiply(X, 1 / self.right[:, None]), 1 / self.left[None, :]
            )

    @property
    def M(self):
        return len(self.left)

    @property
    def N(self):
        return len(self.right)

    def fit(self, A):
        """Fit the Sinkhorn biscaling estimator to the input data.

        Estimates the variance matrix, computes left and right scaling
        vectors via iterative Sinkhorn balancing, and stores the results.

        Parameters
        ----------
        A : array-like or AnnData
            Input non-negative data matrix of shape ``(M, N)``.

        Returns
        -------
        Sinkhorn
            The fitted estimator.
        """
        super().fit()

        X, A = self.process_input_data(A)
        self._issparse = issparse(X, check_scipy=True, check_torch=True)
        self.__set_operands(X)

        self._M = X.shape[0]
        self._N = X.shape[1]
        if self._issparse or (
            self.variance_estimator == "binomial" and isinstance(self.read_counts, int)
        ):
            sparsestr = "sparse"
        else:
            sparsestr = "dense"

        with self.logger.task(
            "Sinkhorn biscaling with {} {} backend".format(sparsestr, str(self.backend))
        ):
            if self.fit_:
                self.row_sums = None
                self.col_sums = None
            row_sums, col_sums = self.__compute_dim_sums()
            self.c = self.compute_c(self.chat)
            self.b = self.compute_b(self.bhat, self.c)
            self.bhat = None if self.c is None else (self.b * self.P) / (1 + self.c)
            self.chat = None if self.b is None else (1 + self.c - self.P) / (1 + self.c)
            self.__is_valid(X, row_sums, col_sums)
            if self._var is None:
                var, rcs = self.estimate_variance(
                    X,
                    q=self.q,
                    bhat=self.bhat,
                    chat=self.chat,
                    read_counts=self.read_counts,
                )
            else:
                var = self.var
                rcs = self.read_counts

            l, r, re, ce = self.__sinkhorn(var, row_sums, col_sums)
            self.__xtype = type(X)

            # now set the final fit attributes.
            if not self.conserve_memory:
                self.X = X
                self.var = var
                self.read_counts = rcs
                self.row_sums = row_sums
                self.col_sums = col_sums
            else:
                del X, var, rcs, row_sums, col_sums
            if (
                self.variance_estimator == None
            ):  # vanilla biscaling, we're just rescaling the original matrix.
                self.left = l
                self.right = r
            else:
                self.left = np.sqrt(l)
                self.right = np.sqrt(r)
            self.row_error = re
            self.column_error = ce
            self.fit_ = True
        return self
    def _update_quadratic_parameters(self,sigma_nu,bhat,chat):
        #update the object to reflect changes in sigma
        # this occurs when sigma is updated in BiPCA by the shrinker.
        # a modification by sigma_nu is multiplying the variance matrix by sigma_nu
        # update the right scaling factor. Due to the update order, this is 
        # the only change that propagates from an update to the variance matrix
        self.right *= 1/sigma_nu
        if attr_exists_not_none(self, "_Z"):
            self.Z *= 1/sigma_nu
        if attr_exists_not_none(self, "_var"):
            self._var *= sigma_nu**2
        #update the variance parameters 
        if self.variance_estimator == "quadratic_2param":
            self.bhat = bhat
            self.chat = chat
            self.c = self.compute_c(self.chat)
            self.b = self.compute_b(self.bhat, self.c)
            self.bhat = (self.b * self.P) / (1 + self.c)
            self.chat = (1 + self.c - self.P) / (1 + self.c)
    def __set_operands(self, X=None):
        """DEPRECATED"""
        # changing the operators to accomodate for sparsity
        # allows us to have uniform API for elemientwise operations
        if X is None:
            isssparse = self._issparse
        else:
            isssparse = issparse(X, check_torch=True)
        if isinstance(X, torch.Tensor):
            if isssparse:
                self.__typef_ = lambda x: make_tensor(X).to_sparse()
                self.__ispos = lambda t: (t.values() >= 0).item()
                self.__dimsum = lambda t, dim: torch.sparse.sum(t, dim=dim).numpy()
            else:
                self.__typef_ = lambda x: make_tensor(X)
                self.__ispos = lambda t: (t.amin() >= 0).item()
                self.__dimsum = lambda t, dim: torch.sum(t, dim=dim).numpy()

            self.__mem = lambda x, y: x * y
            self.__mesq = lambda x: x**2
        else:
            if isssparse:
                self.__typef_ = type(X)
                self.__mem = lambda x, y: x.multiply(y)
                self.__mesq = lambda x: x.power(2)
            else:
                self.__typef_ = lambda x: x
                self.__mem = lambda x, y: np.multiply(x, y)
                self.__mesq = lambda x: np.square(x)
            self.__ispos = lambda x: np.amin(x) >= 0
            self.__dimsum = lambda x, dim: x.sum(dim)

    def __compute_dim_sums(self):
        """Compute target row and column sums for Sinkhorn iteration.

        Uses stored ``row_sums`` and ``col_sums`` if set, otherwise
        defaults to constant vectors (N for rows, M for columns).

        Returns
        -------
        row_sums : ndarray
            Target row sums of length M.
        col_sums : ndarray
            Target column sums of length N.
        """
        if self.row_sums is None:
            row_sums = np.full(self._M, self._N)
        else:
            row_sums = self.row_sums
        if self.col_sums is None:
            col_sums = np.full(self._N, self._M)
        else:
            col_sums = self.col_sums
        return row_sums, col_sums

    def estimate_variance(
        self, X, dist=None, q=None, bhat=None, chat=None, read_counts=None, **kwargs
    ):
        """Estimate the element-wise variance matrix for the input data.

        Dispatches to the appropriate variance function based on the
        selected variance estimator model.

        Parameters
        ----------
        X : ndarray or sparse matrix
            Input data matrix of shape ``(M, N)``.
        dist : str or None, optional
            Variance model name. If None, uses ``self.variance_estimator``.
        q : float or None, optional
            Quadratic variance convex parameter. If None, uses ``self.q``.
        bhat : float or None, optional
            Quadratic 2-parameter model scale parameter.
        chat : float or None, optional
            Quadratic 2-parameter model curvature parameter.
        read_counts : ndarray, int, or None, optional
            Column-wise library sizes for the binomial model.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        var : ndarray or sparse matrix
            Estimated entry-wise variance matrix.
        read_counts : ndarray or int
            The read counts used for estimation.
        """
        self.__set_operands(X)

        if dist is None:
            dist = self.variance_estimator
        if read_counts is None:
            read_counts = self.read_counts
        if read_counts is None:
            read_counts = X.sum(0)
        if q is None:
            q = self.q
        if bhat is None:
            bhat = self.bhat
        if chat is None:
            chat = self.chat

        if dist == "binomial":
            var = binomial_variance(X, read_counts)
        elif dist == "normalized":
            var = normalized_binomial(
                X,
                self.P,
                read_counts,
                mult=multiply,
                square=square,
            )
        elif dist == "quadratic_convex":
            var = quadratic_variance_convex(X, q=q)
        elif dist == "quadratic_2param":
            var = quadratic_variance_2param(X, bhat=bhat, chat=chat)
        elif dist == None:  # vanilla biscaling
            var = X
        else:
            var = general_variance(X)
        return var, read_counts

    def __sinkhorn(self, X, row_sums, col_sums, n_iter=None):
        """Run the Sinkhorn iteration to find left and right scaling vectors.

        Iteratively computes vectors ``u`` (left) and ``v`` (right) such that
        ``diag(u) @ X @ diag(v)`` has the prescribed row and column sums.

        Parameters
        ----------
        X : ndarray or sparse matrix
            The matrix to biscale, typically the variance matrix.
        row_sums : ndarray
            Target row sums of length M.
        col_sums : ndarray
            Target column sums of length N.
        n_iter : int or None, optional
            Number of iterations. If None, uses ``self.n_iter``.

        Returns
        -------
        u : ndarray
            Left (row) scaling vector of length M.
        v : ndarray
            Right (column) scaling vector of length N.
        row_error : float or None
            Final row convergence error.
        col_error : float or None
            Final column convergence error.
        """
        n_row = X.shape[0]
        row_error = None
        col_error = None
        if n_iter is None:
            n_iter = self.n_iter
        if self._N > 100000 and self.verbose >= 1:
            print_progress = True
            print("Sinkhorn progress: ", end="")
        else:
            print_progress = False
        if self.backend.startswith("torch"):
            y = make_tensor(X, keep_sparse=True)
            if isinstance(row_sums, np.ndarray):
                row_sums = torch.from_numpy(row_sums).double()
                col_sums = torch.from_numpy(col_sums).double()
            with torch.no_grad():
                if torch.cuda.is_available() and (
                    self.backend.endswith("gpu") or self.backend.endswith("cuda")
                ):
                    try:
                        y = y.cuda()
                        row_sums = row_sums.cuda()
                        col_sums = col_sums.cuda()
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            self.logger.warning(
                                "GPU cannot fit the matrix in memory. Falling back to CPU."
                            )
                        else:
                            raise e

                u = torch.ones_like(row_sums).double()
                for i in range(n_iter):
                    if print_progress:
                        print("|", end="")
                    u = torch.div(
                        row_sums, y.mv(torch.div(col_sums, y.transpose(0, 1).mv(u)))
                    )
                    if (i + 1) % 10 == 0 and self.tol > 0:
                        v = torch.div(col_sums, y.transpose(0, 1).mv(u))
                        u = torch.div(row_sums, (y.mv(v)))
                        row_converged, col_converged, _, _ = self.__check_tolerance(
                            y, u, v
                        )
                        if row_converged and col_converged:
                            self.logger.info(
                                "Sinkhorn converged early after "
                                + str(i + 1)
                                + " iterations."
                            )
                            break
                        else:
                            del v

                v = torch.div(col_sums, y.transpose(0, 1).mv(u))
                u = torch.div(row_sums, (y.mv(v)))
                v = v.cpu().numpy()
                u = u.cpu().numpy()
                del y
                del row_sums
                del col_sums
            torch.cuda.empty_cache()
        else:
            u = np.ones_like(row_sums)
            for i in range(n_iter):
                u = np.divide(row_sums, X.dot(np.divide(col_sums, X.T.dot(u))))
                if print_progress:
                    print("|", end="")
                if (i + 1) % 10 == 0 and self.tol > 0:
                    v = np.divide(col_sums, X.T.dot(u))
                    u = np.divide(row_sums, X.dot(v))
                    u = np.array(u).flatten()
                    v = np.array(v).flatten()
                    row_converged, col_converged, _, _ = self.__check_tolerance(X, u, v)
                    if row_converged and col_converged:
                        self.logger.info(
                            "Sinkhorn converged early after "
                            + str(i + 1)
                            + " iterations."
                        )
                        break
                    else:
                        del v

            v = np.array(np.divide(col_sums, X.T.dot(u))).flatten()
            u = np.array(np.divide(row_sums, X.dot(v))).flatten()

        if self.tol > 0:
            row_converged, col_converged, row_error, col_error = self.__check_tolerance(
                X, u, v
            )
            del X
            self.converged = all([row_converged, col_converged])
            if not self.converged:
                raise Exception(
                    "At least one of (row, column) errors: "
                    + str((row_error, col_error))
                    + " exceeds requested tolerance: "
                    + str(self.tol)
                    + f" after {i} iterations."
                )

        return u, v, row_error, col_error

    def __check_tolerance(self, X, u, v):
        """Check if the Sinkhorn iteration has converged for a given set of biscalers

        Parameters
        ----------
        X : (M, N) array
            The matrix being biscaled
        u : (M,) array
            The left (row) scaling vector
        v : (N,) array
            The right (column) scaling vector

        Returns
        -------
        row_converged : bool
            The status of row convergence
        col_converged : bool
            The status of column convergence
        row_error : float
            The current error in the row scaling
        col_error : float
            The current in the column scaling
        """
        ZZ = multiply(multiply(X, v.squeeze()), u.squeeze()[:, None])
        
        row_error = amax(abs(self._M - sum(ZZ, 0)))
        col_error = amax(abs(self._N - sum(ZZ, 1)))
        
        if isnan(row_error) + isnan(col_error):
            self.converged = False
            raise Exception(
                "NaN value detected.  Check that the input matrix"
                + " is properly filtered of sparse rows and columns."
            )
        del ZZ
        return row_error < self.tol, col_error < self.tol, row_error, col_error

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

    def predict(self, Y, prediction_axis=0):
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
        is_transposed=False
        if prediction_axis == 0:
            if Y.shape[1]!=self.N:
                raise ValueError("Prediction axis is 0 (rows) but Y does not have the same number of columns as the fitted matrix.")
        elif prediction_axis == 1:
            if Y.shape[0]!=self.M:
                raise ValueError("Prediction axis is 1 (columns) but Y does not have the same number of rows as the fitted matrix.")
        else:
            raise ValueError("Prediction axis must be 0 or 1.")
        if self.variance_estimator is None:
            var_Y = Y
            l = self.left
            r = self.right
        else:
            var_Y = self.estimate_variance(Y)[0]
            l = self.left**2
            r = self.right**2
        
        lnu,rnu = self._extend_scalers(var_Y, prediction_axis, l, r)

        if self.variance_estimator is None:
            return multiply(multiply(Y,rnu[None,:]),lnu[:,None])
        else:
            return multiply(multiply(Y,np.sqrt(rnu[None,:])),np.sqrt(lnu[:,None]))


class SVD(BiPCAEstimator):
    """
    Type-efficient singular value decomposition and storage.


    Computes and stores the SVD of an `(M, N)` matrix `X = US*V^T`.

    Parameters
    ----------
    n_components : int > 0, optional
        Number of singular pairs to compute
        (By default the entire decomposition is performed).
    algorithm : callable, optional
        SVD function accepting arguments `(X, n_components, kwargs)` and returning `(U,S,V)`
        By default, the most efficient algorithm to apply is inferred from the structure of the input data.
    exact : bool, default True
        Only consider exact singular value decompositions.
    conserve_memory : bool, default True
        Remove unnecessary data matrices from memory after fitting a transform.
    suppress : bool, default True
        Suppress helpful interrupts due to suspected redundant calls to SVD.fit()
    verbose : {0, 1, 2}, default 0
        Logging level
    logger : :log:`tasklogger.TaskLogger < >`, optional
        Logging object. By default, write to new logger.
    **kwargs
        Arguments for downstream SVD algorithm.

    Attributes
    ----------
    A : array-like
        The raw input data passed to :meth:`fit`.
    backend : {'scipy', 'torch', 'torch_gpu'}
        The computation backend used for the SVD.
    exact : bool
        Whether exact or approximate SVD algorithms are used.
    fit_ : bool
        True if the estimator has been fit.
    k : int
        The rank of the computed singular value decomposition.
    kwargs : dict
        Keyword arguments passed to the underlying SVD algorithm.
    S : numpy.ndarray
        The singular values of the fitted matrix, sorted in descending order.
    S_ : numpy.ndarray
        Internal storage for singular values.
    U : numpy.ndarray
        The left singular vectors of the fitted matrix.
    U_ : numpy.ndarray
        Internal storage for left singular vectors.
    V : numpy.ndarray
        The right singular vectors of the fitted matrix (stored column-wise).
    V_ : numpy.ndarray
        Internal storage for right singular vectors.
    X : array-like
        The processed input matrix.
    svd : tuple of numpy.ndarray
        The full decomposition ``(U, S, V)``.
    algorithm : callable
        The SVD algorithm selected for the current data and settings.
    n_components : int
        Alias for :attr:`k`.
    conserve_memory : bool
        If True, intermediate data matrices are removed after fitting.

    """

    def __init__(
        self,
        n_components=None,
        exact=True,
        use_eig=False,
        force_dense=False,
        vals_only=False,
        oversample_factor=10,
        conserve_memory=False,
        logger=None,
        verbose=1,
        suppress=True,
        backend="scipy",
        **kwargs,
    ):

        super().__init__(conserve_memory, logger, verbose, suppress, **kwargs)
        self._kwargs = {}
        self.kwargs = kwargs

        self.__k_ = None
        self.vals_only = vals_only
        self.use_eig = use_eig
        self.force_dense = force_dense
        self.oversample_factor = oversample_factor
        self._exact = exact
        self.k = n_components
        self.backend = backend

    @property
    def kwargs(self):
        """
        Return the keyword arguments used to compute the SVD by :meth:`fit`

        .. Warning:: Updating :attr:`kwargs` does not force a new transform; to obtain a new representation of the data, :meth:`fit` must be called.

        .. Important:: This property returns only the arguments that match the function signature of :meth:`algorithm`. :attr:`_kwargs` contains the complete dictionary of keyword arguments.

        Returns
        -------
        dict
            SVD keyword arguments
        """
        hasalg = attr_exists_not_none(self, "_algorithm")
        if hasalg:
            kwargs = filter_dict_with_kwargs(self._kwargs, self._algorithm)
        else:
            kwargs = self._kwargs
        return kwargs

    @kwargs.setter
    def kwargs(self, args):
        """Set the keyword arguments for the SVD algorithm.

        Parameters
        ----------
        args : dict
            Dictionary of keyword arguments for the SVD algorithm.
        """
        # do some logic to check if we are truely changing the arguments.
        fit_ = hasattr(self, "U_")
        if fit_ and ischanged_dict(self.kwargs, args):
            self.logger.warning(
                "Keyword arguments have been updated. The estimator must be refit."
            )
            # there is a scenario in which kwargs is updated with things that do not match the function signature.
            # this code still warns the user
        if "full_matrices" not in args:
            args["full_matrices"] = False
        self._kwargs = args

    @fitted_property
    def svd(self):
        """Return the entire singular value decomposition

        .. Warning:: The object must be fit before requesting this attribute.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            (U,S,V) : The left singular vectors, singular values, and right singular vectors such that USV^T = X

        Raises
        ------
        NotFittedError
        """
        return (self.U, self.S, self.V)

    @fitted_property
    def U(self):
        """Return the left singular vectors that correspond to the largest `n_components` singular values of the fitted matrix

        .. Warning:: The object must be fit before requesting this attribute.

        Returns
        -------
        numpy.ndarray
            The left singular vectors of the fitted matrix.

        Raises
        ------
        NotFittedError
        """
        return self.U_

    @U.setter
    def U(self, U):
        """Set the left singular vectors.

        Parameters
        ----------
        U : numpy.ndarray
            Left singular vectors matrix of shape ``(M, k)``.
        """
        self.U_ = U

    @fitted_property
    def V(self):
        """Return the right singular vectors that correspond to the largest `n_components` singular values of the fitted matrix

        .. Warning:: The object must be fit before requesting this attribute.

        Returns
        -------
        numpy.ndarray
            The right singular vectors of the fitted matrix.

        Raises
        ------
        NotFittedError
        """
        return self.V_

    @V.setter
    def V(self, V):
        """Set the right singular vectors.

        Parameters
        ----------
        V : numpy.ndarray
            Right singular vectors matrix of shape ``(N, k)``.
        """
        self.V_ = V

    @fitted_property
    def S(self):
        """Return the largest `n_components` singular values of the fitted matrix

        .. Warning:: The object must be fit before requesting this attribute.

        Returns
        -------
        numpy.ndarray
            The singular values of the fitted matrix.

        Raises
        ------
        NotFittedError
        """
        return self.S_

    @S.setter
    def S(self, S):
        """Set the singular values.

        Parameters
        ----------
        S : numpy.ndarray
            Singular values array of length ``k``.
        """
        self.S_ = S

    @property
    def exact(self):
        """
        Return whether this object computes exact or approximate SVDs.
        When this attribute is updated, a new best algorithm for the data is computed.

        .. Warning:: Updating :attr:`exact` does not force a new transform; to obtain a new representation of the data, :meth:`fit` must be called.

        Returns
        -------
        bool
            If true, the transforms produced by this object will use exact algorithms.
        """
        return self._exact

    @exact.setter
    def exact(self, val):
        """Set whether to use exact SVD algorithms.

        Parameters
        ----------
        val : bool
            If True, only exact SVD algorithms will be used.
        """
        self._exact = val

    @property
    def backend(self):
        """Return the computation backend for the SVD.

        Returns
        -------
        str
            One of ``'scipy'``, ``'torch'``, or ``'torch_gpu'``.
        """
        if not attr_exists_not_none(self, "_backend"):
            self._backend = "scipy"
        return self._backend

    @backend.setter
    def backend(self, val):
        """Set the computation backend and recompute the best algorithm.

        Parameters
        ----------
        val : str
            One of ``'scipy'``, ``'torch'``, or ``'torch_gpu'``.
        """
        val = self.isvalid_backend(val)
        if self.backend != val:
            self._backend = val
            self.__best_algorithm()

    @property
    def algorithm(self):
        """
        Return the algorithm used for factoring the fitted data.
        The keyword arguments used with this algorithm are returned by :attr:`kwargs`.

        Returns
        -------
        callable
            single argument lambda function wrapping the underlying algorithm.

        No Longer Raises
        ----------------
        NotFittedError
            If a correct algorithm cannot be determined or set, the estimator has not been fit.

        """
        ###Implicitly determines and sets algorithm by wrapping __best_algorithm
        best_alg = self.__best_algorithm()
        if attr_exists_not_none(self, "U_"):
            ### We've already run a transform and we need to change our logic a bit.
            if self._algorithm != best_alg:
                self.logger.warning(
                    "The new optimal algorithm does not match the current transform. "
                    + "Recompute the transform for accuracy."
                )
        self._algorithm = best_alg

        # if self._algorithm is None:
        #     raise NotFittedError()
        return self._algorithm

    def __best_algorithm(self, X=None):
        """Select the most efficient SVD algorithm for the current data and settings.

        Chooses between full, partial, and randomized SVD implementations
        based on the backend, whether exact decomposition is required, and the
        ratio of requested rank to matrix dimensions.

        Parameters
        ----------
        X : array-like, optional
            Input matrix. If None, uses the stored matrix ``self.X``.

        Returns
        -------
        callable
            The selected SVD algorithm function.
        """
        if not attr_exists_not_none(self, "_algorithm"):
            self._algorithm = None
        if X is None:
            if attr_exists_not_none(self, "X_"):
                X = self.X
            else:
                return self._algorithm

        sparsity = issparse(X)
        if "torch" in self.backend:
            algs = [
                self.__compute_torch_svd,
                self.__compute_randomized_svd,
                self.__compute_partial_torch_svd,
            ]
        else:
            algs = [
                self.__compute_scipy_svd,
                self.__compute_randomized_svd,
                sklearn.utils.extmath.randomized_svd,
            ]

        if self.exact:
            if self.k <= np.min(X.shape) * 0.75:
                alg = algs[1]  # returns the partial svds in the exact case
            else:
                alg = algs[0]
        else:
            if self.k >= np.min(X.shape) / 5:
                if self.k <= np.min(X.shape) * 0.75:
                    alg = algs[1]
                else:
                    alg = algs[0]
            else:  # only use the randomized algorithms when k is less than one fifth of the size of the input. I made this number up.
                alg = algs[-1]

        if alg == self.__compute_torch_svd:
            self.k = np.min(
                X.shape
            )  ### THIS CAN LEAD TO VERY LARGE SVDS WHEN EXACT IS TRUE AND TORCH
        self._algorithm = alg
        return self._algorithm

    def __compute_randomized_svd(self, X, k):
        self.k = k
        X = make_scipy(X)
        u, s, v = sklearn.utils.extmath.randomized_svd(
            X,
            n_components=k,
            n_oversamples=int(self.oversample_factor * k),
            random_state=None,
        )
        return u, s, v

    def __compute_scipy_svd(self, X, k):
        self.k = np.min(X.shape)
        X = make_scipy(X)
        if self.k >= 27000 and not self.vals_only:
            raise Exception(
                "The optimal workspace size is larger than allowed "
                "by 32-bit interface to backend math library. "
                "Use a partial SVD or set vals_only=True"
            )
        if self.use_eig:
            if X.shape[0] <= X.shape[1]:
                XXt = X @ X.T
                XTX = False
            else:
                XXt = X.T @ X
                XTX = True
            if sparse.issparse(XXt):
                XXt = XXt.toarray()
            if self.vals_only:
                s = np.sqrt(np.abs(scipy.linalg.eigvalsh(XXt, check_finite=False)))
                s.sort()
                s = s[::-1]
                u = None
                v = None
            else:
                if XTX:
                    s, v = scipy.linalg.eigh(XXt)
                    s = np.sqrt(np.abs(s))
                    six = np.argsort(s)
                    s = s[six]
                    v = v[:, six]
                    v = v[:, ::-1]
                    s = s[::-1]

                    u = X @ ((1 / s) * v)
                    v = v.T
                else:
                    s, u = scipy.linalg.eigh(XXt)
                    s = np.sqrt(np.abs(s))
                    six = np.argsort(s)
                    s = s[six]
                    u = u[:, six]
                    u = u[:, ::-1]
                    s = s[::-1]
                    v = (((1 / s) * u).T @ X).T
        else:
            if self.vals_only:
                s = scipy.linalg.svdvals(X)
                u = None
                v = None
            else:
                u, s, v = scipy.linalg.svd(X, full_matrices=False, check_finite=False)
        return u, s, v

    def __compute_partial_torch_svd(self, X, k):
        """Compute a partial (low-rank) SVD using ``torch.svd_lowrank``.

        Falls back to CPU if the GPU runs out of memory.

        Parameters
        ----------
        X : array-like
            Input matrix.
        k : int
            Number of singular triplets to compute.

        Returns
        -------
        tuple of torch.Tensor
            ``(U, S, V)`` — left singular vectors, singular values,
            and right singular vectors.

        Raises
        ------
        RuntimeError
            If a CUDA error other than out-of-memory occurs.
        """
        y = make_tensor(X, keep_sparse=True)
        with torch.no_grad():
            if torch.cuda.is_available() and (
                self.backend.endswith("gpu") or self.backend.endswith("cuda")
            ):
                try:
                    y = y.cuda()
                except RuntimeError as e:
                    if "CUDA error: out of memory" in str(e):
                        self.logger.warning(
                            "GPU cannot fit the matrix in memory. Falling back to CPU."
                        )
                    else:
                        raise e
            outs = torch.svd_lowrank(y, q=k)
            u, s, v = [ele.cpu() for ele in outs]
            torch.cuda.empty_cache()
        return u, s, v

    def __compute_torch_svd(self, X, k=None):
        """Compute SVD using PyTorch, dispatching to partial or full decomposition.

        Delegates to :meth:`__compute_partial_torch_svd` for sparse matrices or
        when ``k`` is small relative to the matrix dimensions; otherwise computes
        a full SVD (or eigendecomposition) on CPU or GPU.

        Parameters
        ----------
        X : array-like or sparse matrix
            Input data matrix.
        k : int, optional
            Desired rank of the decomposition.

        Returns
        -------
        tuple of (Tensor or None, Tensor, Tensor or None)
            Left singular vectors U, singular values S, and right singular
            vectors V.  U and V are ``None`` when ``vals_only`` is True.

        Raises
        ------
        RuntimeError
            If a CUDA out-of-memory error occurs that cannot be recovered from.
        Exception
            If the matrix dimensions exceed the 32-bit workspace limit and
            ``vals_only`` is False.
        """
        y = make_tensor(X, keep_sparse=True)
        if issparse(X) or k <= np.min(X.shape) / 10:
            return self.__compute_partial_torch_svd(X, k)
        else:
            with torch.no_grad():
                if torch.cuda.is_available() and (
                    self.backend.endswith("gpu") or self.backend.endswith("cuda")
                ):
                    try:
                        y = y.cuda()
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            self.logger.warning(
                                "GPU cannot fit the matrix in memory. Falling back to CPU."
                            )
                        else:
                            raise e
                self.k = np.min(X.shape)
                if self.k >= 27000 and not self.vals_only:
                    raise Exception(
                        "The optimal workspace size is larger than allowed "
                        "by 32-bit interface to backend math library. "
                        "Use a partial SVD or set vals_only=True"
                    )
                if self.use_eig:
                    if y.shape[0] <= y.shape[1]:
                        yyt = torch.matmul(y, y.T)
                        yTy = False
                    else:
                        yyt = torch.matmul(y.T, y)
                        yTy = True
                    if self.vals_only:
                        s, _ = torch.sqrt(torch.abs(torch.linalg.eigvalsh(yyt))).sort(
                            descending=True
                        )
                        s = s.cpu().numpy()
                        u = None
                        v = None
                    else:
                        if yTy:
                            e, v = torch.linalg.eigh(yyt)
                            s, indices = torch.sqrt(torch.abs(e)).sort(descending=True)
                            v = v[:, indices]
                            u = torch.matmul(y, (1 / s) * v)
                            v = v.T
                        else:
                            e, u = torch.linalg.eigh(yyt)
                            s, indices = torch.sqrt(torch.abs(e)).sort(descending=True)
                            u = u[:, indices]
                            v = torch.matmul(((1 / s) * u).T, y).T
                        u = u.cpu()
                        s = s.cpu()
                        v = v.cpu()
                else:
                    if self.vals_only:
                        outs = torch.linalg.svdvals(y)
                        s = outs.cpu()
                        u = None
                        v = None
                    else:
                        outs = torch.linalg.svd(y, full_matrices=False)
                        u, s, v = [ele.cpu() for ele in outs]
                torch.cuda.empty_cache()
            return u, s, v

    @property
    def n_components(self):
        """Return the rank of the singular value decomposition
        This property does the same thing as `k`.

        .. Warning:: Updating :attr:`n_components` does not force a new transform; to obtain a new representation of the data, :meth:`fit` must be called.

        Returns
        -------
        int

        No Longer Raises
        ----------------
        NotFittedError
            In the event that `n_components` is not specified on object initialization,
            this attribute is not valid until fit.
        """
        return self.k

    @n_components.setter
    def n_components(self, val):
        """Set the number of components (rank) for the decomposition.

        Parameters
        ----------
        val : int
            Desired number of components.
        """
        self.k = val

    @property
    def k(self):
        """Return the rank of the singular value decomposition
        This property does the same thing as `n_components`.

        .. Warning:: Updating :attr:`k` does not force a new transform; to obtain a new representation of the data, :meth:`fit` must be called.

        Returns
        -------
        int

        Raises
        ------
        NotFittedError
            In the event that `n_components` is not specified on object initialization,
            this attribute is not valid until fit.
        """
        if self.__k_ is None or 0:
            raise NotFittedError()
        else:
            return self.__k()

    @k.setter
    def k(self, k):
        """Set the rank of the singular value decomposition.

        Parameters
        ----------
        k : int
            Desired rank for the decomposition.
        """
        self.__k(k=k)

    def __k(self, k=None, X=None, suppress=None):
        """
        ### REFACTOR INTO A PROPERTY
        Reset k if necessary and return the rank of the SVD.

        Parameters
        ----------
        k : int, optional
            Desired rank. If None, uses the stored rank.
        X : array-like, optional
            Data matrix to infer rank from if ``k`` is None.
        suppress : bool, optional
            If True, suppress warnings about rank adjustments.

        Returns
        -------
        int
            The validated rank for the decomposition.
        """

        if k is None or 0:
            k = self.__k_
        if k is None:
            if attr_exists_not_none(self, "X_"):
                k = np.min(self.X.shape)
            else:
                if X is not None:
                    k = np.min(X.shape)
        if X is None:
            if hasattr(self, "X_"):
                X = self.X
        if X is not None:
            if k > np.min(X.shape):
                self.logger.warning(
                    "Specified rank k is greater than the minimum dimension of the input."
                )
        if k is None or k <= 0:
            if X is not None:
                k = np.min(X.shape)
            else:
                k = 0
        if k != self.__k_:
            msgs = []
            if self.__k_ is not None:
                msg = (
                    "Updating number of components from k="
                    + str(self.__k_)
                    + " to k="
                    + str(k)
                )
                level = 2
                msgs.append((msg, level))
            if self.fit_:
                # check that our new k matches
                msg = ""
                level = 0
                if k >= np.min(self.U_.shape):
                    msg = (
                        "More components specified than available. "
                        + "Transformation must be recomputed."
                    )
                    level = 1
                elif k <= np.min(self.U_.shape):
                    msg = (
                        "Fewer components specified than available. "
                        + "Output transforms will be lower rank than precomputed."
                    )
                    level = 2
                if level:
                    msgs.append((msg, level))
            super().__suppressable_logs__(msgs, suppress=suppress)

            self.__k_ = k
        self._kwargs["n_components"] = self.__k_
        self._kwargs["k"] = self.__k_
        return self.__k_

    def __check_k_(self, k=None):
        """Validate and return the rank, falling back to the current rank.

        Parameters
        ----------
        k : int, optional
            Requested rank. If ``None``, defaults to ``self.k``.

        Returns
        -------
        int
            Validated rank value.

        Raises
        ------
        ValueError
            If ``k`` exceeds the fitted rank or is non-positive.
        """
        ### helper to check k and raise errors when it is bad
        if k is None or 0:
            k = self.k
        else:
            if k > self.k:
                raise ValueError(
                    "Requested rank requires a higher rank decomposition. "
                    + "Re-fit the estimator at the desired rank."
                )
            if k <= 0:
                raise ValueError("Cannot use a rank 0 or negative rank.")
        return k

    def fit(self, A=None, k=None, exact=None):
        """Compute the singular value decomposition of the input matrix.

        Parameters
        ----------
        A : array-like or sparse matrix, optional
            Input data matrix. If ``None``, uses the previously stored matrix.
        k : int, optional
            Desired rank of the decomposition. If ``None``, uses the current
            value of :attr:`k`.
        exact : bool, optional
            If provided, overrides :attr:`exact` to force an exact or
            approximate decomposition.

        Returns
        -------
        self
            The fitted SVD object.

        Raises
        ------
        ValueError
            If ``k`` is invalid for the given matrix dimensions.
        NotFittedError
            If no input data is available.
        RuntimeError
            If the backend encounters a computational error.
        """

        super().fit()

        if exact is not None:
            self.exact = exact
        if A is None:
            X = self.X
        else:
            X, A = self.process_input_data(A)

        if self.force_dense:
            if sparse.issparse(X):
                X = X.toarray()
            if issparse(X):
                X = X.to_dense()

        self.__k(X=X, k=k)
        if self.k == 0 or self.k is None:
            self.k = np.min(A.shape)
        if self.k >= 27000 and not self.vals_only:
            raise Exception(
                "The optimal workspace size is larger than allowed "
                "by 32-bit interface to backend math library. "
                "Use a partial SVD or set vals_only=True"
            )
        self.__best_algorithm(X=X)
        logstr = "rank k=%d %s %s singular value decomposition using %s."
        logvals = [self.k]
        if sparse.issparse(X):
            logvals += ["sparse"]
        else:
            logvals += ["dense"]
        if self.exact or self.k == np.min(A.shape):
            logvals += ["exact"]
        else:
            logvals += ["approximate"]
        if self.use_eig == "auto":
            aspect_ratio = X.shape[0] / X.shape[1]
            if aspect_ratio > 1:
                aspect_ratio = 1 / aspect_ratio
            if aspect_ratio <= 0.5:
                # rectangular matrix, use eig!
                self.use_eig = True
            else:
                self.use_eig = False
        alg = (
            self.algorithm
        )  # this sets the algorithm implicitly, need this first to get to the fname.
        logvals += [self._algorithm.__name__]
        with self.logger.task(logstr % tuple(logvals)):
            U, S, V = alg(X, **self.kwargs)
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

    def approximate(self, k=None):
        """Rank k approximation of the fitted matrix

        .. Warning:: The object must be fit before calling this method.

        Parameters
        ----------
        k : int, optional
            Desired rank. Defaults to :attr:`k`

        Returns
        -------
        array
            Rank k approximation of the fitted matrix.

        Raises
        ------
        NotFittedError
        """
        check_is_fitted(self)
        k = self.__check_k_(k)

        logstr = "rank k = %s approximation of fit data"
        logval = k
        with self.logger.task(logstr % logval):
            return (self.U[:, :k] * self.S[:k]) @ self.V[:, :k].T

    def get_factors(self):
        if self.vals_only:
            return None, self.S, None
        return self.U, self.S, self.V

    def factorize(self, X=None, k=None, exact=None):
        self.fit(X, k, exact)
        return self.get_factors()

    def fit_transform(self, X=None, k=None, exact=None):
        """Compute an SVD and return the rank `k` approximation of `X`


        .. Error:: If called with ``X = None`` and :attr:`conserve_memory <bipca.math.SVD>` is true,
                    this method will fail as there is no underlying matrix to transform.

        Parameters
        ----------
        X : array-like, optional
            Data matrix to decompose. If None, uses the stored matrix.
        k : int, optional
            Rank of the approximation. If None, uses the stored rank.
        exact : bool, optional
            If True, use exact (full) SVD instead of randomized.

        Returns
        -------
        ndarray
            The rank-``k`` approximation ``U @ diag(S) @ V^T``.

        Raises
        ------
        ValueError
            If ``k`` exceeds matrix dimensions.
        NotFittedError
            If called without data and no prior ``fit``.
        RuntimeError
            If ``conserve_memory`` is True and no matrix is available.

        """
        return self.factorize(X=X, k=k, exact=exact)

    def PCA(self, k=None):
        """Return the rank-k principal component scores (U * S).

        Parameters
        ----------
        k : int, optional
            Number of components to retain. If ``None``, defaults to
            :attr:`k`.

        Returns
        -------
        numpy.ndarray
            Principal component score matrix of shape ``(n_samples, k)``.
        """
        k = self.__check_k_(k)
        return self.U[:, :k] * self.S[:k]

    def compute_element(self, index=None, U=None, S=None, V=None, rank=None):
        """compute_element: Compute an element (or row, col, resp.) of a matrix from its singular value decomposition.
        Optionally, U, S, and V can be swapped to introduce transformation of the singular triplets.

        This function can be used to selectively compute rows or columns of the low-rank approximated matrix
        without placing the full matrix in memory.

        See `np.s_` for creating inputs to this function.

        Parameters
        ----------
        index : `tuple` [`slice`, `int`, `None`] , optional.
            (row, col) index of element to compute.
            To compute an entire row (column resp.), use `index=np.s_[row,:]` (`index=np.s_[:,col]`).
            By default, no index is used, i.e. `U[:,:rank]*S[:rank])@V[:,:rank].T` is returned.
        U : `np.ndarray` or `torch.Tensor`, (m, rank) optional
            Left singular vector(s).
            By default, this method uses the singular vectors stored in `self.U`.
        S : `np.ndarray` or `torch.Tensor`, (rank,), optional.
            Singular values.
            By default, this method uses the singular vectors stored in `self.S`.
        V : `np.ndarray` or `torch.Tensor`, (n, rank), optional
            Right singular vector(s).
            By default, this method uses the singular vectors stored in `self.V`.
        rank : int or None, optional
            Approximation rank of element.
            By default, infer rank from the size of `S`.

        Returns
        -------
        np.ndarray or torch.Tensor
            The requested elements from the decomposition.
            Type specified inferred from inputs.
        """

        ## check inputs, validate rank.
        if U is None:
            U = self.U
        if S is None:
            S = self.S
        if V is None:
            V = self.V
        if rank is None:
            rank = np.min([len(S), U.shape[1], V.shape[1]])
        if len(S) < rank:
            raise ValueError(
                "`rank` was larger than the number of singular values contained in `S`. "
                "`len(S)` must be larger than `rank`."
            )
        if U.shape[1] < rank:
            # there are fewer left singular vectors than requested.
            raise ValueError(
                "`rank` was larger than column dimension of `U`, "
                "thus an approximation at the requested rank is impossible. "
                "U.shape[1] must be less than or equal to `rank`."
            )
        if V.shape[1] < rank:
            # there are fewer right singular vectors than requested.
            raise ValueError(
                "`rank` was larger than column dimension of `V`, "
                "thus an approximation at the requested rank is impossible. "
                "V.shape[1] must be less than or equal to `rank`."
            )

        # we have now ensured that we have a U, S, and V with second dimension at least `rank`.

        U = U[:, :rank]
        S = S[:rank]
        V = V[:, :rank]
        # now check `index` and return.
        if index is not None:
            try:
                U = U[index[0], :]
            except Exception as e:
                raise IndexError(
                    "The row index into `U` was invalid. See traceback for type hints."
                ) from e
            try:
                V = V[index[1], :]
            except Exception as e:
                raise IndexError(
                    "The column index into `V` was invalid. See traceback for type hints."
                ) from e
        if U.ndim == 1:
            U = U[None, :]
        if V.ndim == 1:
            V = V[None, :]
        return multiply(U, S) @ V.T

  

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
        Verbosity level.
    logger : :log:`tasklogger.TaskLogger < >`, optional
        Logger instance for progress tracking.

    Attributes
    ----------
    A : ndarray
        The rank-``k`` approximation matrix after shrinkage.
    cov_eigs_ : ndarray
        Covariance eigenvalues (squared singular values divided by N).
    default_shrinker : str
        Default shrinkage loss function name.
    emp_qy_ : float
        Empirical quantile of the singular values.
    gamma_ : float
        Fitted aspect ratio ``M / N``.
    M_ : int
        Number of rows of the fitted data.
    MP : MarcenkoPastur
        Fitted Marcenko-Pastur distribution instance.
    N_ : int
        Number of columns of the fitted data.
    quantile_ : float
        Quantile used for noise variance estimation.
    rescale_svs : bool
        Whether to rescale shrunk singular values.
    scaled_cov_eigs_ : ndarray
        Covariance eigenvalues scaled by the noise variance.
    scaled_cutoff_ : float
        Scaled Marcenko-Pastur upper edge cutoff.
    scaled_mp_rank_ : int
        Number of singular values above the scaled MP cutoff.
    sigma_ : float
        Estimated noise standard deviation.
    theory_qy_ : float
        Theoretical quantile for the noise bulk boundary.
    unscaled_mp_rank_ : int
        Number of singular values above the unscaled MP cutoff.
    y_ : ndarray
        Fitted singular values.
    nuclear

    Methods
    -------
    fit_transform : array
        Apply Sinkhorn algorithm and return biscaled matrix
    fit : array

    Deleted Attributes
    ------------------
    logger : object
        Previously used logger attribute (removed).

    """

    """How many `Shrinker` objects are there?"""

    def __init__(
        self,
        default_shrinker="frobenius",
        rescale_svs=True,
        conserve_memory=False,
        logger=None,
        verbose=1,
        suppress=True,
        **kwargs,
    ):
        """Initialize the Shrinker with shrinkage and estimation options.

        Parameters
        ----------
        default_shrinker : str, optional
            Shrinkage method to use by default in ``transform``.
        rescale_svs : bool, optional
            If True, rescale shrunk singular values back to the original
            domain using the estimated noise variance.
        conserve_memory : bool, optional
            If True, reduce memory usage at the cost of speed.
        logger : tasklogger.TaskLogger or None, optional
            Logger instance for status messages.
        verbose : int, optional
            Verbosity level for log output.
        suppress : bool, optional
            If True, suppress non-critical log messages.
        **kwargs
            Additional keyword arguments passed to the parent estimator.
        """
        super().__init__(conserve_memory, logger, verbose, suppress, **kwargs)
        self.default_shrinker = default_shrinker
        self.rescale_svs = rescale_svs

    # some properties for fetching various shrinkers when the object has been fitted.
    # these are just wrappers for transform.
    @fitted_property
    def frobenius(self):
        """Shrunk singular values using the Frobenius-norm optimal shrinker.

        Returns
        -------
        ndarray
            Singular values after Frobenius-norm optimal shrinkage.
        """
        check_is_fitted(self)
        return self.transform(shrinker="fro")

    @fitted_property
    def operator(self):
        """Shrunk singular values using the operator-norm optimal shrinker.

        Returns
        -------
        ndarray
            Singular values after operator-norm optimal shrinkage.
        """
        check_is_fitted(self)
        return self.transform(shrinker="op")

    @fitted_property
    def hard(self):
        """Shrunk singular values using hard thresholding.

        Returns
        -------
        ndarray
            Singular values after hard threshold shrinkage.
        """
        check_is_fitted(self)
        return self.transform(shrinker="hard")

    @fitted_property
    def soft(self):
        """Shrunk singular values using soft thresholding.

        Returns
        -------
        ndarray
            Singular values after soft threshold shrinkage.
        """
        check_is_fitted(self)
        return self.transform(shrinker="soft")

    @fitted_property
    def nuclear(self):
        """Shrunk singular values using the nuclear-norm optimal shrinker.

        Returns
        -------
        ndarray
            Singular values after nuclear-norm optimal shrinkage.
        """
        return self.transform(shrinker="nuc")

    @fitted_property
    def sigma(self):
        """Estimated noise standard deviation.

        Returns
        -------
        float
            The estimated noise level from the Marcenko-Pastur fit.
        """
        return self.sigma_

    @sigma.setter
    def sigma(self, sigma):
        """Set the estimated noise standard deviation.

        Parameters
        ----------
        sigma : float
            Noise standard deviation value.
        """
        self.sigma_ = sigma

    @fitted_property
    def scaled_mp_rank(self):
        """Marcenko-Pastur rank estimated from the scaled singular values.

        Returns
        -------
        int
            Number of signal singular values above the scaled MP threshold.
        """
        return self.scaled_mp_rank_

    @scaled_mp_rank.setter
    def scaled_mp_rank(self, scaled_mp_rank):
        """Set the scaled Marcenko-Pastur rank.

        Parameters
        ----------
        scaled_mp_rank : int
            Scaled MP rank value.
        """
        self.scaled_mp_rank_ = scaled_mp_rank

    @fitted_property
    def scaled_cutoff(self):
        """Singular value cutoff for the scaled spectrum.

        Returns
        -------
        float
            Threshold above which singular values are considered signal.
        """
        return self.scaled_cutoff_

    @scaled_cutoff.setter
    def scaled_cutoff(self, scaled_cutoff):
        """Set the scaled singular value cutoff.

        Parameters
        ----------
        scaled_cutoff : float
            Scaled cutoff value.
        """
        self.scaled_cutoff_ = scaled_cutoff

    @fitted_property
    def unscaled_mp_rank(self):
        """Marcenko-Pastur rank estimated from the unscaled singular values.

        Returns
        -------
        int
            Number of signal singular values above the unscaled MP threshold.
        """
        return self.unscaled_mp_rank_

    @unscaled_mp_rank.setter
    def unscaled_mp_rank(self, unscaled_mp_rank):
        """Set the unscaled Marcenko-Pastur rank.

        Parameters
        ----------
        unscaled_mp_rank : int
            Unscaled MP rank value.
        """
        self.unscaled_mp_rank_ = unscaled_mp_rank

    @fitted_property
    def gamma(self):
        """Aspect ratio of the data matrix (rows / columns).

        Returns
        -------
        float
            The aspect ratio gamma = M / N used in Marcenko-Pastur fitting.
        """
        return self.gamma_

    @gamma.setter
    def gamma(self, gamma):
        """Set the aspect ratio.

        Parameters
        ----------
        gamma : float
            Aspect ratio value.
        """
        self.gamma_ = gamma

    @fitted_property
    def emp_qy(self):
        """Empirical quantile function of the singular values.

        Returns
        -------
        ndarray
            Empirical quantile values of the observed singular value distribution.
        """
        return self.emp_qy_

    @emp_qy.setter
    def emp_qy(self, emp_qy):
        """Set the empirical quantile function.

        Parameters
        ----------
        emp_qy : ndarray
            Empirical quantile values.
        """
        self.emp_qy_ = emp_qy

    @fitted_property
    def theory_qy(self):
        """Theoretical quantile function from the Marcenko-Pastur distribution.

        Returns
        -------
        ndarray
            Theoretical MP quantile values used to estimate noise.
        """
        return self.theory_qy_

    @theory_qy.setter
    def theory_qy(self, theory_qy):
        """Set the theoretical quantile function.

        Parameters
        ----------
        theory_qy : ndarray
            Theoretical MP quantile values.
        """
        self.theory_qy_ = theory_qy

    @fitted_property
    def quantile(self):
        """Quantile points used for the QQ-plot fitting.

        Returns
        -------
        ndarray
            Quantile evaluation points.
        """
        return self.quantile_

    @quantile.setter
    def quantile(self, quantile):
        """Set the quantile points.

        Parameters
        ----------
        quantile : ndarray
            Quantile evaluation points.
        """
        self.quantile_ = quantile

    @fitted_property
    def scaled_cov_eigs(self):
        """Covariance eigenvalues estimated from the scaled singular values.

        Returns
        -------
        ndarray
            Scaled covariance eigenvalues (squared singular values / N).
        """
        return self.scaled_cov_eigs_

    @scaled_cov_eigs.setter
    def scaled_cov_eigs(self, scaled_cov_eigs):
        """Set the scaled covariance eigenvalues.

        Parameters
        ----------
        scaled_cov_eigs : ndarray
            Scaled covariance eigenvalue array.
        """
        self.scaled_cov_eigs_ = scaled_cov_eigs

    @fitted_property
    def cov_eigs(self):
        """Covariance eigenvalues estimated from the unscaled singular values.

        Returns
        -------
        ndarray
            Unscaled covariance eigenvalues.
        """
        return self.cov_eigs_

    @cov_eigs.setter
    def cov_eigs(self, cov_eigs):
        """Set the unscaled covariance eigenvalues.

        Parameters
        ----------
        cov_eigs : ndarray
            Unscaled covariance eigenvalue array.
        """
        self.cov_eigs_ = cov_eigs

    def fit(self, y, shape=None, sigma=None, theory_qy=None, q=None, suppress=None):
        """Estimate Marcenko-Pastur parameters and prepare for shrinkage.

        Parameters
        ----------
        y : ndarray or AnnData
            Singular values (1-D) or a data matrix. If an AnnData object,
            singular values are read from ``uns['SVD']['S']`` or
            ``uns['bipca']['S']``.
        shape : tuple of int or None, optional
            Shape ``(M, N)`` of the original data matrix. Required when
            ``y`` is a 1-D vector of singular values.
        sigma : float or None, optional
            Known noise standard deviation. If None, estimated from the
            data via QQ-plot fitting against the Marcenko-Pastur distribution.
        theory_qy : ndarray or None, optional
            Pre-computed theoretical quantile values for the MP distribution.
            If None, computed internally.
        q : ndarray or None, optional
            Quantile evaluation points for the QQ-plot. If None, chosen
            automatically.
        suppress : bool or None, optional
            Override the instance-level suppress flag for this call.

        Raises
        ------
        ValueError
            If ``y`` is a 1-D vector and ``shape`` is not provided.

        Returns
        -------
        Shrinker
            The fitted Shrinker instance (self).
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

        yx = np.argsort(y)
        yxx = np.argsort(yx) # yxx is the inverse permutation of yx
        y = y[yx]  # sort the singular values
        cutoff = np.sqrt(N) + np.sqrt(M)
        r = (y > cutoff).sum()
        if r == len(y):
            #handle the extreme case where all singular values are signal
            r = 1 
            compensate_bulk = False
        sigma0 = 0
        sigma1 = 1

        i = 0
        obj = lambda sigma0, sigma1: ((sigma0) / (sigma1)) != 1
        while obj(sigma0, sigma1):
            i += 1
            sigma0 = sigma1
            if compensate_bulk:
                # compensate_bulk: consider a smaller set of singular values that is composed of only the current estimate of bulk.
                # this changes the indexing and thus the quantile we are matching
                # this effect is particularly acute when the rank is large and we are given a partial estimate of the singular values.
                # suppose M = 200, len(y) = 20, and r = 19, bulk_size = 181.
                # in the normal setting, z_size = len(y), and the algorithm assumes that the empirical quantile is computed from the (1-len(z)/M) = (1-20/181) ~= 89th quantile
                # when compensate_bulk = True, z_size = len(y) - r = 1, bulk_size = 181.
                # then the empirical median is assumed to be computed from the (1-1/181) ~= 99th quantile
                # it is unclear to me which is more appropriate!
                z = y[:-r].copy()  # grab all but the largest r elements
                bulk_size = M - r  # the actual size of the current empirical bulk = svs - rank
            else:
                z = y
                bulk_size = M   


            emp_qy = None
            if len(z) >= bulk_size // 2 +1:
                # we have enough singular values to compute the median exactly.
                q = 0.5
                if (bulk_size % 2) == 1:
                    # odd case
                    # because integer division M//2 is floor, we need to add 1 to get the correct index when M is odd.
                    emp_qy = z[-(bulk_size//2+1)]
                else:
                    emp_qy = (z[-(bulk_size//2)] + z[-(bulk_size//2+1)]) / 2

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
        return scaled_svs[yxx], sigma1, q

    def _estimate_MP_params(
        self, y=None, M=None, N=None, theory_qy=None, q=None, sigma=None
    ):
        """Estimate Marcenko-Pastur distribution parameters from singular values.

        Parameters
        ----------
        y : array-like, optional
            Singular values. If None, uses the fitted values.
        M : int, optional
            Number of rows. If None, uses the fitted value.
        N : int, optional
            Number of columns. If None, uses the fitted value.
        theory_qy : float, optional
            Theoretical quantile for the noise bulk boundary.
        q : float, optional
            Quantile parameter for noise variance estimation.
        sigma : float, optional
            Precomputed noise standard deviation.

        Returns
        -------
        tuple
            Estimated MP parameters including sigma, gamma, cutoff, and
            covariance eigenvalues.

        Raises
        ------
        ValueError
            If ``theory_qy`` is specified without ``q``.
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
        """Fit the model and return optimally shrunk singular values.

        Parameters
        ----------
        y : array-like, optional
            Singular values. If None, uses the fitted values.
        shape : tuple, optional
            Shape ``(M, N)`` of the original data matrix.
        shrinker : str, optional
            Shrinkage loss function to apply. If None, uses the default.
        rescale : bool, optional
            Whether to rescale shrunk values back to the original scale.

        Returns
        -------
        ndarray
            Optimally shrunk singular values.
        """
        self.fit(y, shape)
        if shrinker is None:
            shrinker = self.default_shrinker
        return self.transform(y=y, shrinker=shrinker)

    def transform(self, y=None, shrinker=None, rescale=None):
        """Apply optimal shrinkage to singular values.

        Parameters
        ----------
        y : array-like, optional
            Singular values to shrink. If None, uses the fitted values.
        shrinker : str, optional
            Shrinkage loss function. If None, uses the default.
        rescale : bool, optional
            Whether to rescale shrunk values back to the original scale.

        Returns
        -------
        ndarray
            Optimally shrunk singular values.
        """
        check_is_fitted(self)
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
    Estimate entry-wise variance under a general (empirical) model.

    Computes the squared absolute values of the mean-centered data as
    a plug-in variance estimate.

    Parameters
    ----------
    X : ndarray or sparse matrix
        Input non-negative data matrix of shape ``(M, N)``.

    Returns
    -------
    ndarray
        Entry-wise variance estimates of the same shape as ``X``.
    """
    Y = MeanCenteredMatrix().fit_transform(X)
    if issparse(X, check_torch=False):
        Y = Y.toarray()
    Y = np.abs(Y) ** 2
    return Y


def quadratic_variance_convex(X, q=0):
    """
    Estimate entry-wise variance under the quadratic variance model.

    Computes ``(1 - q) * X + q * X**2``, a convex combination of Poisson-like
    (linear) and overdispersed (quadratic) variance terms.

    Parameters
    ----------
    X : ndarray or sparse matrix
        Input non-negative data matrix of shape ``(M, N)``.
    q : float, optional
        Convex combination parameter in ``[0, 1]``. When ``q=0``, the
        variance equals ``X`` (Poisson); when ``q=1``, it equals ``X**2``.

    Returns
    -------
    ndarray or sparse matrix
        Entry-wise variance estimates of the same shape as ``X``.
    """
    if issparse(X, check_torch=False):
        Y = X.copy()
        Y.data = (1 - q) * X.data + q * X.data**2
        return Y
    return (1 - q) * X + q * X**2


def quadratic_variance_2param(X, bhat=1.0, chat=0):
    """
    Estimate entry-wise variance under the two-parameter quadratic model.

    Computes ``bhat * X + chat * X**2``, allowing independent control of the
    linear and quadratic variance terms.

    Parameters
    ----------
    X : ndarray or sparse matrix
        Input non-negative data matrix of shape ``(M, N)``.
    bhat : float, optional
        Coefficient of the linear (Poisson-like) variance term.
    chat : float, optional
        Coefficient of the quadratic (overdispersion) variance term.

    Returns
    -------
    ndarray or sparse matrix
        Entry-wise variance estimates of the same shape as ``X``.
    """
    if issparse(X, check_torch=False):
        Y = X.copy()
        Y.data = bhat * X.data + chat * X.data**2
        return Y
    return multiply(X, bhat) + multiply(square(X), chat)


def binomial_variance(X, counts):
    """
    Estimate entry-wise variance under the binomial count model.

    Computes ``X * counts / (counts - 1) - X**2 / (counts - 1)`` and takes
    the absolute value to ensure non-negativity.

    Parameters
    ----------
    X : ndarray or sparse matrix
        Input non-negative data matrix of shape ``(M, N)``.
    counts : ndarray or int
        Column-wise library sizes (read counts). Must be greater than 1.

    Returns
    -------
    ndarray or sparse matrix
        Entry-wise variance estimates of the same shape as ``X``.
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
    """Marcenko-Pastur distribution for random matrix theory.

    Attributes
    ----------
    gamma : float
        Aspect ratio of the data matrix (values > 1 are inverted).
    """

    def __init__(self, gamma):
        """Initialize the Marcenko-Pastur distribution.

        Parameters
        ----------
        gamma : float
            Aspect ratio ``M / N`` (values > 1 are automatically inverted).
        """
        if gamma > 1:
            gamma = 1 / gamma
        a = (1 - gamma**0.5) ** 2
        b = (1 + gamma**0.5) ** 2

        super().__init__(a=a, b=b)
        self.gamma = gamma

    def _pdf(self, x):
        """Evaluate the Marcenko-Pastur probability density function.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the PDF.

        Returns
        -------
        ndarray
            Density values at ``x``.
        """
        m0 = lambda a: np.clip(a, 0, None)
        m0b = self.b - x
        m0b = np.maximum(m0b, 0)
        m0a = x - self.a
        m0a = np.maximum(m0a, 0)

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


class SamplingMatrix(object):
    __array_priority__ = 1

    def __init__(self, X=None):
        self.ismissing = False
        if X is not None:
            self.M, self.N = X.shape
            self.compute_probabilities(X)

    def compute_probabilities(self, X):
        if issparse(X):
            self.coords = self.__build_coordinates_sparse(X)
        else:
            self.coords = self.__build_coordinates_dense(X)
        self.__compute_probabilities_from_coordinates(*self.coords)

    @property
    def shape(self):
        return (self.M, self.N)

    def __build_coordinates_sparse(self, X):
        X = make_scipy(X).tocoo()
        coordinates = np.where(np.isnan(X.data))
        rows = X.row[coordinates]
        cols = X.col[coordinates]
        return rows, cols

    def __build_coordinates_dense(self, X):
        rows, cols = np.where(np.isnan(X))
        return rows, cols

    def __compute_probabilities_from_coordinates(self, rows, cols):
        m, n = self.shape
        n_samples = m * n - len(rows)
        grand_mean = 1 / (m * n) * n_samples
        self.row_p = np.ones(
            (m, 1),
        )
        self.row_p = self.row_p / np.sqrt(grand_mean)

        self.col_p = np.ones(
            (1, n),
        )
        self.col_p = self.col_p / np.sqrt(grand_mean)

        if n_samples < m * n:

            unique, counts = np.unique(rows, return_counts=True)
            self.row_p[unique.astype(int), :] = (
                ((n - counts) / n) / np.sqrt(grand_mean)
            )[:, None]

            unique, counts = np.unique(cols, return_counts=True)
            self.col_p[:, unique.astype(int)] = (
                ((m - counts) / m) / np.sqrt(grand_mean)
            )[None, :]

            self.ismissing = True

    def __getitem__(self, pos):
        if isinstance(pos, tuple):
            row, col = pos
        else:
            if isinstance(pos, slice):
                start, stop, step = pos.start, pos.stop, pos.step
                if start is None:
                    start = 0
                if stop is None:
                    stop = np.prod(self.shape)
                if step is None:
                    step = 1
                pos = np.arange(start, stop, step)
            row, col = np.unravel_index(pos, self.shape)
        return np.minimum(self.get_row(row) * self.get_col(col), 1)

    @property
    def T(self):
        obj = SamplingMatrix()
        obj.M, obj.N = self.N, self.M
        obj.coords = self.coords[1], self.coords[0]
        obj.row_p = self.col_p.T
        obj.col_p = self.row_p.T
        obj.ismissing = self.ismissing
        return obj

    def __call__(self):
        return np.minimum(self.row_p * self.col_p, 1)

    def __add__(self, val):
        return val + self()

    def __radd__(self, val):
        return self + val

    def __sub__(self, val):
        return -1 * val + self()

    def __rsub__(self, val):
        return val + -1 * self

    def __mul__(self, val):
        return val * self()

    def __rmul__(self, val):
        return val * self()

    def __repr__(self):
        return f"SamplingMatrix({self.row_p},{self.col_p})"

    def get_row(self, row):
        return self.row_p[row, :].squeeze()

    def get_col(self, col):
        return self.col_p[:, col].squeeze()


def L2(x, func1, func2):
    """Compute the squared difference (L2 loss) between two functions at ``x``.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate the functions.
    func1 : callable
        First function.
    func2 : callable
        Second function.

    Returns
    -------
    ndarray
        Element-wise squared difference ``(func1(x) - func2(x))^2``.
    """
    return np.square(func1(x) - func2(x))


def normalized_KS(y, mp, m, r):
    y_above_mask = y > mp.b
    y_below_mask = y < mp.a
    y_inrange_mask = np.logical_not(y_above_mask) * np.logical_not(y_below_mask)
    y_inrange = y[y_inrange_mask]

    return (y_above_mask.sum() + y_below_mask.sum()) / m + kstest(y_inrange, mp.cdf)[0]


def KS(y, mp):
    """Compute the Kolmogorov-Smirnov statistic against an MP distribution.

    Parameters
    ----------
    y : array-like
        Observed eigenvalues or singular values.
    mp : rv_continuous
        Marcenko-Pastur distribution instance with a ``cdf`` method.

    Returns
    -------
    float
        The KS test statistic.
    """
    # x = np.linspace(mp.a*0.8, mp.b*1.2, num = num)
    # yesd = np.interp(x, np.flip(y), np.linspace(0,1,num=len(y),endpoint=False))
    # mpcdf = mp.cdf(x)
    # return np.amax(np.absolute(mpcdf - yesd))
    return kstest(y, mp.cdf)[0]


def L1(x, func1, func2):
    """Compute the absolute difference (L1 loss) between two functions at ``x``.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate the functions.
    func1 : callable
        First function.
    func2 : callable
        Second function.

    Returns
    -------
    ndarray
        Element-wise absolute difference ``|func1(x) - func2(x)|``.
    """
    return np.absolute(func1(x) - func2(x))


# evaluate given loss function on a pdf and an empirical pdf (histogram data)
def emp_pdf_loss(pdf, epdf, loss=L2, start=0):
    """Evaluate a loss between a theoretical and empirical PDF by integration.

    Parameters
    ----------
    pdf : callable
        Theoretical probability density function.
    epdf : callable
        Empirical probability density function.
    loss : callable, optional
        Loss function ``loss(x, func1, func2)``. Defaults to ``L2``.
    start : float, optional
        Lower integration bound. Defaults to 0.

    Returns
    -------
    float
        Integrated loss value.
    """
    # loss() should have three arguments: x, func1, func2
    # note 0 is the left limit because our pdfs are strictly supported on the non-negative reals, due to the nature of sv's

    val = integrate.quad(lambda x: loss(x, pdf, epdf), start, np.inf, limit=100)[0]

    return val


def emp_mp_loss(mat, gamma=0, loss=L2, precomputed=True, M=None, N=None):
    """Compute the loss between empirical spectral distribution and MP law.

    Parameters
    ----------
    mat : ndarray
        Data matrix or precomputed covariance eigenvalues.
    gamma : float, optional
        Aspect ratio ``M / N``. If 0, computed from ``M`` and ``N``.
    loss : callable, optional
        Loss function ``loss(x, func1, func2)``. Defaults to ``L2``.
    precomputed : bool, optional
        If True, ``mat`` contains precomputed eigenvalues.
    M : int, optional
        Number of rows. Required when ``precomputed`` is True.
    N : int, optional
        Number of columns. Required when ``precomputed`` is True.

    Returns
    -------
    float
        Integrated loss between the empirical and MP distributions.

    Raises
    ------
    RuntimeError
        If ``precomputed`` is True and ``M`` or ``N`` is None.
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
    """Remove noise bias from singular values using an optimal shrinker.

    Values below ``sigma * (sqrt(n) + sqrt(m))`` are set to zero.

    Parameters
    ----------
    y : array-like
        Observed singular values.
    m : int
        Number of rows.
    n : int
        Number of columns.
    gamma : float, optional
        Aspect ratio ``m / n``. If None, computed from ``m`` and ``n``.
    sigma : float, optional
        Noise standard deviation. Defaults to 1.

    Returns
    -------
    ndarray
        Debiased singular values.
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
    """Apply optimal singular value shrinkage for a given loss function.

    Parameters
    ----------
    unscaled_y : array-like
        Singular values on the original (unscaled) scale.
    sigma : float
        Estimated noise standard deviation.
    M : int
        Number of rows of the data matrix.
    N : int
        Number of columns of the data matrix.
    gamma : float
        Aspect ratio ``M / N``.
    scaled_cutoff : float, optional
        Scaled MP upper bound. If None, computed from ``gamma``.
    shrinker : str, optional
        Shrinkage type: ``'frobenius'``, ``'operator'``, ``'soft'``,
        ``'hard'``, or ``'nuclear'``. Defaults to ``'frobenius'``.
    rescale : bool, optional
        If True, rescale shrunk values back to the original scale.
    logger : object, optional
        Logger instance.

    Returns
    -------
    ndarray
        Shrunk singular values.

    Raises
    ------
    ValueError
        If ``shrinker`` is not a recognized shrinkage type.
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
    """Compute the upper edge of the scaled Marcenko-Pastur distribution.

    Parameters
    ----------
    gamma : float
        Aspect ratio ``M / N``.

    Returns
    -------
    float
        Upper bound ``(1 + sqrt(gamma))^2``.
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
    consider_zeros : bool
        Whether to include zeros when computing means.
    fit_ : bool
        Whether the instance has been fitted.
    M : int
        Number of columns of the fitted data.
    maintain_sparsity : bool
        Whether to preserve sparsity during centering.
    N : int
        Number of rows of the fitted data.
    X_centered : array-like
        The mean-centered data matrix (stored if ``conserve_memory`` is False).
    row_means : ndarray
        Mean of each row.
    column_means : ndarray
        Mean of each column.
    grand_mean : float
        Grand mean of the matrix.
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
        """Initialize the mean centering transformer.

        Parameters
        ----------
        maintain_sparsity : bool, optional
            If True, only center nonzero elements to preserve sparsity.
        consider_zeros : bool, optional
            If True, include zeros when computing means.
        conserve_memory : bool, optional
            If True, only store centering factors, not the centered matrix.
        logger : tasklogger.TaskLogger, optional
            Logger instance for status messages.
        verbose : int, optional
            Verbosity level (0, 1, or 2).
        suppress : bool, optional
            If True, suppress extra warnings beyond the logging level.
        **kwargs
            Additional keyword arguments passed to the parent estimator.
        """
        super().__init__(conserve_memory, logger, verbose, suppress, **kwargs)
        self.maintain_sparsity = maintain_sparsity
        self.consider_zeros = consider_zeros

    @memory_conserved_property
    @fitted
    def X_centered(self):
        """The mean-centered data matrix from the last call to ``transform``.

        Returns
        -------
        array-like
            The centered data matrix.
        """
        return self._X_centered

    @X_centered.setter
    def X_centered(self, Mc):
        """Set the centered data matrix, stored only when memory is not conserved.

        Parameters
        ----------
        Mc : array-like
            The centered data matrix to store.
        """
        if not self.conserve_memory:
            self._X_centered = Mc

    @fitted_property
    def row_means(self):
        """Mean of each row computed during ``fit``.

        Returns
        -------
        ndarray
            Array of row means.
        """
        return self._row_means

    @fitted_property
    def column_means(self):
        """Mean of each column computed during ``fit``.

        Returns
        -------
        ndarray
            Array of column means.
        """
        return self._column_means

    @fitted_property
    def grand_mean(self):
        """Grand mean of the matrix computed during ``fit``.

        Returns
        -------
        float
            The scalar grand mean.
        """
        return self._grand_mean

    @fitted_property
    def rescaling_matrix(self):
        """Dense matrix of row, column, and grand mean corrections.

        Returns
        -------
        ndarray
            An ``(N, M)`` array where each entry is
            ``row_mean + column_mean - grand_mean``.
        """
        # Computes and returns the dense rescaling matrix
        mat = -1 * self._grand_mean * np.ones((self.N, self.M))
        mat += self._row_means[:, None]
        mat += self._column_means[None, :]
        return mat

    def __compute_grand_mean(self, X, consider_zeros=True):
        """Compute the grand mean of a matrix.

        Parameters
        ----------
        X : array-like or sparse matrix
            Input data matrix.
        consider_zeros : bool, optional
            If True, divide by total element count; otherwise by nonzero count.

        Returns
        -------
        float
            The grand mean.
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
        """Compute means along a given axis.

        Parameters
        ----------
        X : array-like or sparse matrix
            Input data matrix.
        axis : int, optional
            0 for column means, 1 for row means.
        consider_zeros : bool, optional
            If True, divide by the full dimension; otherwise by nonzero count.

        Returns
        -------
        ndarray
            Array of means along the specified axis.
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
        """Fit centering parameters and return the centered matrix.

        Parameters
        ----------
        X : array-like or sparse matrix
            Input data matrix of shape ``(N, M)``.

        Returns
        -------
        array-like
            The mean-centered data matrix.
        """
        self.fit(X)
        return self.transform(X)

    def fit(self, X):
        """Compute row means, column means, and grand mean from the data.

        Parameters
        ----------
        X : array-like or sparse matrix
            Input data matrix of shape ``(N, M)``.

        Returns
        -------
        MeanCenteredMatrix
            The fitted instance.
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
        """Subtract the fitted means from the input matrix.

        Parameters
        ----------
        X : array-like or sparse matrix
            Data matrix of shape ``(N, M)`` to center.

        Returns
        -------
        array-like
            The mean-centered data matrix.
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
        """Alias for ``transform``. Center the input matrix.

        Parameters
        ----------
        X : array-like or sparse matrix
            Data matrix of shape ``(N, M)`` to center.

        Returns
        -------
        array-like
            The mean-centered data matrix.
        """
        # Convenience synonym for transform
        return self.transform(X)

    def center(self, X):
        """Alias for ``transform``. Center the input matrix.

        Parameters
        ----------
        X : array-like or sparse matrix
            Data matrix of shape ``(N, M)`` to center.

        Returns
        -------
        array-like
            The mean-centered data matrix.
        """
        # Convenience synonym for transform
        return self.transform(X)

    @fitted
    def invert(self, X):
        """Add back the fitted means to reverse centering.

        Parameters
        ----------
        X : array-like or sparse matrix
            A previously centered data matrix of shape ``(N, M)``.

        Returns
        -------
        array-like
            The uncentered data matrix.
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
        """Alias for ``invert``. Reverse the centering transformation.

        Parameters
        ----------
        X : array-like or sparse matrix
            A previously centered data matrix of shape ``(N, M)``.

        Returns
        -------
        array-like
            The uncentered data matrix.
        """
        # Convenience synonym for invert
        return self.invert(X)

    def unscale(self, X):
        """Alias for ``invert``. Reverse the centering transformation.

        Parameters
        ----------
        X : array-like or sparse matrix
            A previously centered data matrix of shape ``(N, M)``.

        Returns
        -------
        array-like
            The uncentered data matrix.
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
