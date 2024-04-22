"""BiPCA: BiStochastic Principal Component Analysis
"""
import numpy as np
from numbers import Number
import sklearn as sklearn
import scipy as scipy
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import scipy.sparse as sparse
from scipy.stats import kstest
import tasklogger
from anndata._core.anndata import AnnData
from pychebfun import Chebfun
from torch.multiprocessing import Pool
import torch
from .math import (
    Sinkhorn,
    SVD,
    Shrinker,
    MarcenkoPastur,
    KS,
    normalized_KS,
    SamplingMatrix,
    minimize_chebfun,
)
from .math import library_normalize as lib_normalize
from .utils import (
    stabilize_matrix,
    filter_dict_with_kwargs,
    filter_dict_with_kwargs,
    nz_along,
    attr_exists_not_none,
    write_to_adata,
    CachedFunction,
    make_tensor,
    make_scipy,
    fill_missing,
    typecast
)
from .base import *
from .safe_basics import (abs, 
                          add,
                          less_equal,
                          mean,
                          multiply,
                          divide,
                          quantile,
                          square,
                          subtract,
                          sum,
                          where)


class BiPCA(BiPCAEstimator):

    """Bistochastic PCA:
    Biscale and denoise according to the paper

    Parameters
    ----------
    center : bool, optional
        Center the biscaled matrix before denoising. `True` introduces a dense matrix to the problem,
        which can lead to memory problems and slow results for large problems. This option is not recommended for large problems.
        Default False.
    variance_estimator : {'quadratic','binomial'}, default 'quadratic'
        Variance estimator to use when Sinkhorn biscaling.
    q : int, default 0
        Precomputed quadratic variance for generalized Poisson sinkhorn biwhitening. Used when `qits <= 1`
    qits : int, default 21
        Number of variance fitting cycles to run per subsample when `variance_estimator` is `'quadratic'`.
        If `qits <= 1`, then no variance fitting is performed.
    n_subsamples : int, default 5
        Number of subsamples to consider when `variance_estimator` is `'quadratic'`
    approximate_sigma : bool, optional
        Estimate the noise variance for the Marcenko Pastur model using a submatrix of the original data
        Default True for inputs with small axis larger than 2000.
    compute_full_approx : bool, optional
        Compute the complete singular value decomposition of subsampled matrices when `approximate_sigma=True`.
        Useful for pre-computing singular values computed by `get_plotting_spectrum()` by saving a repeated SVD.
        Default True.
    default_shrinker : {'frobenius','fro','operator','op','nuclear','nuc','hard','hard threshold','soft','soft threshold'}, default 'frobenius'
        shrinker to use when bipca.transform is called with no argument `shrinker`.
    sinkhorn_tol : float, default 1e-6
        Sinkhorn tolerance threshold
    n_iter : int, default 500
        Number of sinkhorn iterations before termination.
    n_components : None, optional
        Number of singular vectors to compute for denoising. By default, 200 are computed.
    pca_method : str, optional
        Description
    exact : bool, optional
        Compute SVD using any of the full, exact decompositions from the 'torch' or backend,
        or the partial decomposition provided by scipy.sparse.linalg.svds.
        Default True
    conserve_memory : bool, optional
        Save memory footprint by storing fewer matrices in memory, instead computing them at runtime.
        Default False.
    logger : None, optional
        Description
    verbose : int, optional
        Description
    suppress : bool, optional
        Description
    subsample_size : None, optional
        Description
    backend : {'scipy', 'torch'}, optional
        Engine to use as the backend for sinkhorn and SVD computations. Overwritten by `sinkhorn_backend` and `svd_backend`.
        Default 'scipy'
    svd_backend : None, optional
        Description
    sinkhorn_backend : None, optional
        Description
    **kwargs
        Description


    Attributes
    ----------
    approximate_sigma : TYPE
        Description
    backend : TYPE
        Description
    default_shrinker : TYPE
        Description
    exact : TYPE
        Description
    k : TYPE
        Description
    keep_aspect : TYPE
        Description
    kst : float
        The ks-test score achieved by the best fitting variance estimate.
    n_iter : TYPE
        Description
    pca_method : TYPE
        Description
    q : float
        The q-value used in the biwhitening variance estimate.
    qits : TYPE
        Description
    S_X : TYPE
        Description
    shrinker : TYPE
        Description
    sinkhorn : TYPE
        Description
    sinkhorn_backend : TYPE
        Description
    sinkhorn_kwargs : TYPE
        Description
    sinkhorn_tol : TYPE
        Description
    subsample_gamma : TYPE
        Description
    submatrix_indices : dict
        Description
    subsample_M : TYPE
        Description
    subsample_N : TYPE
        Description
    subsample_sinkhorn : TYPE
        Description
    subsample_size : TYPE
        Description
    svd : TYPE
        Description
    svd_backend : TYPE
        Description
    svdkwargs : TYPE
        Description
    variance_estimator : TYPE
        Description
    X : TYPE
        Description
    Y : TYPE
        Description
    Z : TYPE
        Description
    S_Y : TYPE
        Description
    """

    def __init__(
        self,
        variance_estimator="quadratic",
        qits=51,
        P=None,
        normalized_KS=False,
        minimize_mean=True,
        fit_sigma=True,
        seed=42,
        n_subsamples=5,
        oversample_factor=10,
        b=None,
        bhat=None,
        c=None,
        chat=None,
        keep_aspect=False,
        read_counts=None,
        use_eig="auto",
        dense_svd=True,
        default_shrinker="frobenius",
        sinkhorn_tol=1e-6,
        n_iter=500,
        n_components=None,
        exact=True,
        subsample_threshold=None,
        conserve_memory=False,
        logger=None,
        verbose=1,
        suppress=True,
        subsample_size=5000,
        backend="torch",
        svd_backend=None,
        sinkhorn_backend=None,
        njobs=1,
        **kwargs,
    ):
        # build the logger first to share across all subprocedures
        super().__init__(conserve_memory, logger, verbose, suppress, **kwargs)

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # initialize the subprocedure classes
        self.k = n_components
        self.sinkhorn_tol = sinkhorn_tol
        self.default_shrinker = default_shrinker
        self.n_iter = n_iter
        self.exact = exact
        self.minimize_mean = minimize_mean
        self.variance_estimator = variance_estimator
        self.qits = qits
        self.use_eig = use_eig
        self.dense_svd = dense_svd
        self.n_subsamples = n_subsamples
        self.njobs = njobs
        self.fit_sigma = fit_sigma
        self.backend = backend
        self.svd_backend = svd_backend
        self.oversample_factor = oversample_factor
        self.sinkhorn_backend = sinkhorn_backend
        self.keep_aspect = keep_aspect
        self.read_counts = read_counts
        if read_counts is None and variance_estimator == "binomial":
            raise ValueError(
                "read_counts must be specified for binomial variance estimator"
            )
        self.subsample_threshold = subsample_threshold
        self.normalized_KS = normalized_KS
        self.P = P
        self.X_ = None

        self.init_quadratic_params(b, bhat, c, chat)
        self.reset_submatrices(subsample_size=subsample_size)
        self.reset_plotting_spectrum()
        # remove the kwargs that have been assigned by super.__init__()

        # hotfix to remove tol collisions
        self.svdkwargs = kwargs

        self.sinkhorn_kwargs = kwargs.copy()
        if "tol" in kwargs:
            del self.sinkhorn_kwargs["tol"]
        if "P" in kwargs:
            del self.sinkhorn_kwargs["P"]

    ###Properties that construct the objects that we use to compute a bipca###
    @property
    def sinkhorn(self):
        """Return the Sinkhorn matrix scaling instance used by this BiPCA instance.
        """
        if not attr_exists_not_none(self, "_sinkhorn"):
            self._sinkhorn = Sinkhorn(
                tol=self.sinkhorn_tol,
                n_iter=self.n_iter,
                read_counts=self.read_counts,
                variance_estimator=self.variance_estimator,
                bhat=self.bhat,
                chat=self.chat,
                b=self.b,
                c=self.c,
                P=self.P,
                relative=self,
                backend=self.sinkhorn_backend,
                conserve_memory=self.conserve_memory,
                suppress=self.suppress,
                **self.sinkhorn_kwargs,
            )
        return self._sinkhorn

    @sinkhorn.setter
    def sinkhorn(self, val):
        """
        """
        if isinstance(val, Sinkhorn):
            self._sinkhorn = val
        else:
            raise ValueError("Cannot set self.sinkhorn to non-Sinkhorn estimator")

    @property
    def svd(self):
        """Return the SVD instance used by this BiPCA instance.
        """
        if not attr_exists_not_none(self, "_svd"):
            self._svd = SVD(
                n_components=self.k,
                exact=self.exact,
                oversample_factor=self.oversample_factor,
                backend=self.svd_backend,
                relative=self,
                conserve_memory=self.conserve_memory,
                force_dense=self.dense_svd,
                use_eig=self.use_eig,
                suppress=self.suppress,
            )
        return self._svd

    @svd.setter
    def svd(self, val):
        """
        """
        if isinstance(val, SVD):
            self._svd = val
        else:
            raise ValueError("Cannot set self.svd to non-SVD estimator")

    @property
    def shrinker(self):
        """Return the Shrinker instance used by this BiPCA instance.
        """
        if not attr_exists_not_none(self, "_shrinker"):
            self._shrinker = Shrinker(
                default_shrinker=self.default_shrinker,
                rescale_svs=True,
                relative=self,
                suppress=self.suppress,
            )
        return self._shrinker

    @shrinker.setter
    def shrinker(self, val):
        """
        """
        if isinstance(val, Shrinker):
            self._shrinker = val
        else:
            raise ValueError("Cannot set self.shrinker to non-Shrinker estimator")

    @property
    def svd_backend(self):
        """
        """
        if not attr_exists_not_none(self, "_svd_backend"):
            return self.backend
        else:
            return self._svd_backend

    @svd_backend.setter
    def svd_backend(self, val):
        """
        """
        val = self.isvalid_backend(val)
        if attr_exists_not_none(self, "_svd_backend"):
            if val != self._svd_backend:
                # its a new backend
                self._svd_backend = val
        else:
            self._svd_backend = val
        self.reset_backend()

    @property
    def sinkhorn_backend(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        if not attr_exists_not_none(self, "_sinkhorn_backend"):
            return self.backend
        return self._sinkhorn_backend

    @sinkhorn_backend.setter
    def sinkhorn_backend(self, val):
        """Summary

        Parameters
        ----------
        val : TYPE
            Description
        """
        val = self.isvalid_backend(val)
        if attr_exists_not_none(self, "_sinkhorn_backend"):
            if val != self._sinkhorn_backend:
                # its a new backend
                self._sinkhorn_backend = val
        else:
            self._sinkhorn_backend = val
        self.reset_backend()

    ###general properties###
    @fitted_property
    def mp_rank(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self._mp_rank

    @property
    def n_components(self):
        """Convenience function for :attr:`k`

        Returns
        -------
        TYPE
            Description
        """
        return self.k

    @n_components.setter
    def n_components(self, val):
        """Summary

        Parameters
        ----------
        val : TYPE
            Description
        """
        self.k = val

    @property
    def k(self):
        """Return the rank of the base singular value decomposition
        .. Warning:: Updating :attr:`k` does not force a new transform; to obtain a new representation of the data, :meth:`fit` must be called.

        Returns
        -------
        int

        """
        return self._k

    @k.setter
    def k(self, k):
        """Summary

        Parameters
        ----------
        k : TYPE
            Description
        """
        self._k = k

    @fitted_property
    def U_Y(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.svd.U

    @fitted_property
    def S_Y(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.svd.S

    @fitted_property
    def V_Y(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """

        return self.svd.V

    @property
    def Y(self):
        """Summary

        Returns
        -------
        TYPE
            Description

        Raises
        ------
        RuntimeError
            Description
        """
        if not self.conserve_memory:
            Y = self._Y
            return Y
        else:
            raise RuntimeError(
                "Since conserve_memory is true, Y can only be obtained by calling .get_Y(X)"
            )

    @Y.setter
    def Y(self, Y):
        """Summary

                    fill_missing)
        ----------
        Y : TYPE
            Description
        """
        if not self.conserve_memory:
            self._Y = Y

    @property
    def Y(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        check_is_fitted(self)
        if not self.conserve_memory:
            if self._Y is None:
                self._Y = self.transform()

            return self._Y
        else:
            return self.transform()

    @Y.setter
    def Y(self, Y):
        """Summary

        Parameters
        ----------
        Y : TYPE
            Description
        """
        if not self.conserve_memory:
            self._Y = Y

    def get_Y(self, X=None):
        """Summary

        Parameters
        ----------
        X : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        check_is_fitted(self)
        if X is None:
            return self.Y
        else:
            return self.sinkhorn.transform(X)

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
        with self.logger.task("Transform unscaling"):
            return self.sinkhorn.unscale(X)

    @property
    def right_biwhite(self):
        """Summary
            if self.P.ismissing:
                    X = fill_missing(X)
                    if not self.conserve_memory:
                        self.X = X
                else:
                    self.P = 1
        Returns
        -------
        TYPE
            Description
        """
        return self.sinkhorn.right

    @property
    def left_biwhite(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """

        return self.sinkhorn.left

    @property
    def aspect_ratio(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        m = np.min([self.M, self.N])
        n = np.max([self.M, self.N])
        return m / n

    @property
    def N(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self._N

    @property
    def M(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self._M

    def fit(self, A):
        """Summary

        Parameters
        ----------
        A : TYPE
            Description

        Deleted Parameters
        ------------------
        X : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        # bug: sinkhorn needs to be reset when the model is refit.
        super().fit()
        with self.logger.task("BiPCA fit"):
            self._M, self._N = A.shape

            X, A = self.process_input_data(A)
            self.library_size = np.median(np.asarray(sum(X,axis =1)))
            self.reset_submatrices(X=X)
            if self.k == -1:  # k is determined by the minimum dimension
                self.k = np.min([self.M, self.N])
            elif self.k is None or self.k == 0:  # automatic k selection
                self.k = np.min([200, np.min([self.M, self.N]) // 2])
                if self.variance_estimator == "binomial":
                    # if it is binomial, we don't need to estimate parameters
                    # of the distribution, so we only need to take enough
                    # singular values to cover the data
                    self.k = np.min([200, np.min([self.M, self.N])])

            self.k = np.min(
                [self.k, *X.shape]
            )  # ensure we are not asking for too many SVs
            self.svd.k = self.k
            if self.P is None:
                self.P = SamplingMatrix(X)
            if isinstance(self.P, SamplingMatrix):
                if self.P.ismissing:
                    X = fill_missing(X)
                    if not self.conserve_memory:
                        self.X = X
                else:
                    self.P = 1
            if self.variance_estimator == "quadratic":
                self.bhat, self.chat = self.fit_quadratic_variance(X=X)
                self.sinkhorn = Sinkhorn(
                    tol=self.sinkhorn_tol,
                    n_iter=self.n_iter,
                    bhat=self.bhat,
                    chat=self.chat,
                    read_counts=self.read_counts,
                    P=self.P,
                    variance_estimator="quadratic_2param",
                    relative=self,
                    backend=self.sinkhorn_backend,
                    conserve_memory=self.conserve_memory,
                    suppress=self.suppress,
                    **self.sinkhorn_kwargs,
                )
            elif self.variance_estimator == "normalized":
                self.sinkhorn = Sinkhorn(
                    tol=self.sinkhorn_tol,
                    n_iter=self.n_iter,
                    read_counts=self.read_counts,
                    variance_estimator="normalized",
                    bhat=self.bhat,
                    chat=self.chat,
                    b=self.b,
                    c=self.c,
                    P=self.P,
                    relative=self,
                    backend=self.sinkhorn_backend,
                    conserve_memory=self.conserve_memory,
                    suppress=self.suppress,
                    **self.sinkhorn_kwargs,
                )

            else:
                b = 1
                c = -1 / self.read_counts
                self.init_quadratic_params(b=b, c=c, bhat=None, chat=None)
                self.sinkhorn = Sinkhorn(
                    tol=self.sinkhorn_tol,
                    n_iter=self.n_iter,
                    read_counts=self.read_counts,
                    variance_estimator="quadratic_2param",
                    bhat=self.bhat,
                    chat=self.chat,
                    b=self.b,
                    c=self.c,
                    P=self.P,
                    relative=self,
                    backend=self.sinkhorn_backend,
                    conserve_memory=self.conserve_memory,
                    suppress=self.suppress,
                    **self.sinkhorn_kwargs,
                )

            self.sinkhorn.fit(X)
            if self.variance_estimator == "normalized":
                X = np.where(self.read_counts >= 2, X / self.read_counts, 0)
            M = self.sinkhorn.transform(X)
            self.Y = M
            if (
                self.variance_estimator == "binomial"
            ):  # no variance estimate needed when binomial is used.
                sigma_estimate = 1
            else:
                if self.fit_sigma:
                    sigma_estimate = None
                else:
                    sigma_estimate = 1

            converged = False

            while not converged:
                self.svd.fit(M)
                toshrink = self.S_Y
                _, converged = self.shrinker.fit(
                    toshrink, shape=X.shape, sigma=sigma_estimate
                )
                self._mp_rank = self.shrinker.scaled_mp_rank_
                if not converged:
                    self.k = int(np.min([self.k * 1.5, *X.shape]))
                    self.svd.k = self.k
                    self.logger.info(
                        "Full rank partial decomposition detected,"
                        " fitting with a larger k = {}".format(self.k)
                    )
            if self.variance_estimator == "quadratic" and self.fit_sigma == True:
                self._update_quadratic_parameters(self.shrinker.sigma)
                
                

            del M
            del X

            return self

    @fitted
    def _transform(self, 
        U=None, 
        S=None, 
        V=None, 
        counts=True,
        which="left",
        unscale=False,
        library_normalize=True,
        shrinker=None,
        truncate=0,
        truncation_axis=0,):
        """ Given U, S, and V, apply denoising. This is a backend function that is called by `transform` and `predict`.
        """

        if U is None:
            U = self.U_Y
        if S is None:
            S = self.S_Y
        if V is None:
            V = self.V_Y
        sshrunk = self.shrinker.transform(S, shrinker=shrinker)
        if counts:
            Z = U[:, : self.mp_rank].numpy() * sshrunk[: self.mp_rank]
            Z = Z @ V[:, : self.mp_rank].T.numpy()
            if truncate is not False:
                if truncate == 0 or truncate is True:
                    thresh = 0 
                else:
                    thresh = abs(quantile(Z, truncate, axis=truncation_axis))
                    if truncation_axis == 1:
                        thresh = thresh[:, None]
                Z = where(less_equal(Z, thresh), 0, Z)
            if unscale:
                Z = self.unscale(Z)
            if library_normalize:
                if library_normalize is True:
                    scale = self.library_size
                elif isinstance(library_normalize, Number):
                    scale = library_normalize
                Z = lib_normalize(Z,scale = scale)
            return Z
        else:
            # return the PCs
            if which == "left":
                return (
                    U[:, : self.mp_rank]
                    * self.shrinker.transform(S, shrinker=shrinker)[
                        : self.mp_rank
                    ]
                )
            else:
                return (
                    V[:, : self.mp_rank]
                    * self.shrinker.transform(S, shrinker=shrinker)[
                        : self.mp_rank
                    ]
                )
    @fitted
    def transform(
        self,
        X=None,
        counts=True,
        which="left",
        unscale=False,
        library_normalize=True,
        shrinker=None,
        denoised=True,
        truncate=0,
        truncation_axis=0,
    ):
        """Return a denoised version of the data.

        Parameters
        ----------
        X : array, optional
            If `BiPCA.conserve_memory` is True, then X must be provided in order to obtain
            the solely biwhitened transform, i.e., for unscale=False, denoised=False.
        counts : bool, default True
            Return the reconstructed counts matrix. If False, return the principal components.
        which : {'left','right'}, default 'left'
            Which principal components to return. By default, the left (row-wise) PCs are returned.
        unscale : bool, default False
            Unscale the output matrix so that it is in the original input domain.
        library_normalize : float or bool, default True
            Perform library normalization on the output matrix. If a float, then the 
            output matrix is divided by the sum of its column counts and multiplied by 
            the float. If False, no library normalization is performed.
            If True, the output matrix is divided by the sum of its column counts and 
            multiplied by the median of the input matrix column sums.
        shrinker : {'hard','soft', 'frobenius', 'operator','nuclear'}, optional
            Shrinker to use for denoising
            (Defaults to `obj.default_shrinker`)
        denoised : bool, default True
            Return denoised output.
        truncate : numeric or bool, default 0
            Truncate the transformed data. If 0 or true, then the output is thresholded at 0.
            If nonzero, the truncate-th quantile along `truncation_axis` is used to adaptively
            threshold the output.
        truncation_axis : {0,1}, default 0.
            Axis to gather truncation thresholds from. Uses numpy axes:
            truncation_axis==0 gathers thresholds down the rows, therefore is column-wise
            truncation_axis==1 gathers thresholds across the columns, therefore is row-wise

        Returns
        -------
        np.array
            (N x M) transformed array
        """
        if denoised:
            return self._transform(self.U_Y, self.S_Y, self.V_Y,
                                counts=counts, which=which, unscale=unscale, 
                                library_normalize=library_normalize, shrinker=shrinker,
                                truncate=truncate, truncation_axis=truncation_axis)
        else:
            if not self.conserve_memory:
                X = self.Y  # the full rank, biwhitened matrix.
            else:
                X = self.get_Y(X)
            return X
  
    def fit_transform(self, X=None, shrinker=None, **kwargs):
        """Fit the estimator, then return a denoised version of the data.

        Parameters
        ----------
        X : None, optional
            Description
        shrinker : None, optional
            Description
        **kwargs
            Description

        Returns
        -------
        TYPE
            Description
        """
        if X is None:
            check_is_fitted(self)
        else:
            self.fit(X)
        return self.transform(shrinker=shrinker, **kwargs)

    def predict(self, 
        Y, 
        prediction_axis=0,
        counts=True,
        which="left",
        unscale=False,
        library_normalize=True,
        shrinker=None,
        denoised=True,
        truncate=0,
        truncation_axis=0,):
            """Denoise new data matrix Y using the fitted BiPCA model.

            Parameters:
            ----------
            Y : array-like, shape (n_samples, n_features)
                The new data matrix to be denoised. One of the axes must have the same
                dimension as the data that was used to fit the model.
            prediction_axis : int, optional (default=0)
                The axis along which the denoised data will be predicted.
            counts : bool, default True
                Return the reconstructed counts matrix. If False, return the principal components.
            which : {'left','right'}, default 'left'
                Which principal components to return. By default, the left (row-wise) PCs are returned.
            unscale : bool, default False
                Unscale the output matrix so that it is in the original input domain.
            library_normalize : float or bool, default True
                Perform library normalization on the output matrix. If a float, then the 
                output matrix is divided by the sum of its column counts and multiplied by 
                the float. If False, no library normalization is performed.
                If True, the output matrix is divided by the sum of its column counts and 
                multiplied by the median of the input matrix column sums.
            shrinker : {'hard','soft', 'frobenius', 'operator','nuclear'}, optional
                Shrinker to use for denoising
                (Defaults to `obj.default_shrinker`)
            denoised : bool, default True
                Return denoised output.
            truncate : numeric or bool, default 0
                Truncate the transformed data. If 0 or true, then the output is thresholded at 0.
                If nonzero, the truncate-th quantile along `truncation_axis` is used to adaptively
                threshold the output.
            truncation_axis : {0,1}, default 0.
                Axis to gather truncation thresholds from. Uses numpy axes:
                truncation_axis==0 gathers thresholds down the rows, therefore is column-wise
                truncation_axis==1 gathers thresholds across the columns, therefore is row-wise

            Returns
            -------
            np.array
                (n_samples x n_features) transformed array

            """
            type_Y = type(Y)
            if self.sinkhorn_backend=='torch':
                Y = make_tensor(Y)
            else:
                Y = make_scipy(Y)
            Y = self.sinkhorn.predict(Y, prediction_axis=prediction_axis)
            # next, project onto the principal components appropriately
            if self.svd_backend == 'torch':
                Y = make_tensor(Y)
            else:
                Y = make_scipy(Y)
            U = None
            V = None
            if prediction_axis == 0:
                #we're adding rows, so we need to project onto the right singular vectors
                U = Y @ self.V_Y[:, : self.mp_rank]
                U /= self.S_Y[:self.mp_rank]
            else:
                #we're adding columns, so we need to project onto the left singular vectors
                V = Y.T@self.U_Y[:, : self.mp_rank] 
                V /= self.S_Y[:self.mp_rank]
            Y = self._transform(U=U, V=V, counts=counts, which=which, unscale=unscale, 
                                library_normalize=library_normalize, shrinker=shrinker,
                                truncate=truncate, truncation_axis=truncation_axis)
            return typecast(Y, type_Y)
            
    def write_to_adata(self, adata):
        """Convenience wrapper for `bipca.utils.write_to_adata`.

        Parameters
        ----------
        adata : AnnData
            The AnnData object to write into.

        Returns
        -------
        TYPE
            Description
        """
        return write_to_adata(self, adata)

    #### SUBMATRIX RELATED METHODS
    def reset_submatrices(self, X=None, subsample_size=None):
        """Reset the state of the submatrices.
        Clears submatrix_indices, rebuilds subsample_size.
        This function operates on the dimensions of the original input data.
        """
        self.submatrix_indices = []
        if not hasattr(self, "subsample_size") or subsample_size is not None:
            self.subsample_size = subsample_size
        if X is None:
            X = self.X
        if X is not None:
            (
                M,
                N,
            ) = (
                X.shape
            )  # note that these are not the same M and N as the M and N in the model!
            if isinstance(self.subsample_size, (tuple, list, np.ndarray)):
                if len(self.subsample_size) > 2:
                    raise ValueError(
                        "Subsample size must be \
                        iterable of length 2"
                    )

                sub_M, sub_N = self.subsample_size
                ## 1, sanitize Nones, which get converted to missing dimensions
                if sub_M is None:
                    sub_M = 0
                if sub_N is None:
                    sub_N = 0

                ## 2, check that no dimension is larger than feasible
                sub_M = np.min([sub_M, M])
                sub_N = np.min([sub_N, N])
                # next, fill in any missing dimensions
                if sub_M < 1 and sub_N >= 1:
                    if self.keep_aspect:
                        sub_M = np.max([np.floor(M / N * sub_N).astype(int), 1])
                    else:
                        sub_M = np.min([sub_N, M])
                if sub_N < 1:
                    # note that because we didn't check against sub_M,
                    # this gets called when sub_M and sub_N < 1, which
                    # defaults to no subsampling.
                    # reset and go through this method again
                    if self.keep_aspect:
                        sub_N = np.max([np.floor(N / M * sub_M).astype(int), 1])
                    else:
                        sub_N = np.min([sub_M, N])
                self.subsample_size = (sub_M, sub_N)

            elif isinstance(self.subsample_size, Number):
                # self.subsample_size is assumed to be the limiting (minimum) target
                # dimension
                self.subsample_size = int(self.subsample_size)

                if M > N:
                    minDim = N
                    maxDim = M
                else:
                    minDim = M
                    maxDim = N
                if (
                    (self.subsample_size > minDim)
                    or (self.subsample_size == minDim and not self.keep_aspect)
                    or self.subsample_size < 1
                ):
                    self.subsample_size = (M, N)
                else:
                    if self.keep_aspect:
                        sub_Max = np.max(
                            [
                                np.floor(maxDim / minDim * self.subsample_size).astype(
                                    int
                                ),
                                1,
                            ]
                        )
                    else:
                        sub_Max = np.min([self.subsample_size, maxDim])
                    if maxDim == N:
                        self.subsample_size = (self.subsample_size, sub_Max)
                    else:
                        self.subsample_size = (sub_Max, self.subsample_size)
            else:
                self.subsample_size = (M, N)

    def get_submatrix_indices(
        self, reset=False, X=None, n_subsamples=None, threshold=None
    ):
        """Compute `n_subsamples` submatrices of approximate minimum dimension `subsample_size`
        of the matrix `X` with minimum number of nonzeros per row or column given by threshold.

        Parameters
        ----------
        X : None, optional
            Description
        n_subsamples : None, optional
        subsample_size : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """

        if X is None:
            X = self.X
        if n_subsamples is None:
            n_subsamples = self.n_subsamples
        if threshold is None:
            threshold = self.subsample_threshold
        if self.variance_estimator == "quadratic":
            variance_estimator = "quadratic_convex"
        else:
            variance_estimator = self.variance_estimator

        if threshold is None:
            # compute the threshold by the minimum nnz in the variance estimate
            sinkhorn = Sinkhorn(
                read_counts=self.read_counts,
                tol=self.sinkhorn_tol,
                n_iter=self.n_iter,
                variance_estimator=variance_estimator,
                backend=self.sinkhorn_backend,
                verbose=0,
                **self.sinkhorn_kwargs,
            )
            varX = sinkhorn.estimate_variance(X)[0]
            cols = nz_along(varX, axis=0)
            rows = nz_along(varX, axis=1)
            cols = np.min(cols) - 1
            rows = np.min(rows) - 1
            threshold = self.subsample_threshold = np.min([rows, cols])

        if (
            reset
            or not attr_exists_not_none(self, "submatrix_indices")
            or self.subsample_size is None
        ):
            self.reset_submatrices(X=X)

        if n_subsamples == 0 or self.subsample_size == (X.shape[0], X.shape[1]):
            rixs = np.arange(X.shape[0])
            cixs = np.arange(X.shape[1])
            self.submatrix_indices.append({"rows": rixs, "columns": cixs})
        else:
            for n_ix in range(n_subsamples):
                rixs = self.rng.permutation(X.shape[0])
                cixs = self.rng.permutation(X.shape[1])
                rixs = rixs[: self.subsample_size[0]]
                cixs = cixs[: self.subsample_size[1]]
                xsub = X[rixs, :][:, cixs]
                # instantiate a sinkhorn instance to get a proper variance estimate
                # we have to stabilize the matrix based on the sparsity of the variance estimate
                sinkhorn = Sinkhorn(
                    read_counts=self.read_counts,
                    tol=self.sinkhorn_tol,
                    n_iter=self.n_iter,
                    variance_estimator=variance_estimator,
                    backend=self.sinkhorn_backend,
                    verbose=0,
                    **self.sinkhorn_kwargs,
                )
                varX = sinkhorn.estimate_variance(xsub)[0]
                cols = nz_along(varX, axis=0)
                rows = nz_along(varX, axis=1)
                cols = np.max(cols)
                rows = np.max(rows)
                if cols < threshold or rows < threshold:
                    threshold_proportion = threshold / np.min(X.shape)
                    thresh_temp = threshold_proportion * np.min(xsub.shape)
                    threshold = int(np.max([np.floor(thresh_temp), 1]))
                _, (mixs, nixs) = stabilize_matrix(varX, threshold=threshold)

                rixs = rixs[mixs]
                cixs = cixs[nixs]
                self.submatrix_indices.append({"rows": rixs, "columns": cixs})

        return self.submatrix_indices

    def get_submatrix(
        self, index=0, reset=False, X=None, n_subsamples=None, threshold=None
    ):
        if X is None:
            X = self.X
        if self.backend == "torch":
            X = make_tensor(X)
        if threshold is None:
            threshold = self.subsample_threshold

        if (reset is True) or (self.submatrix_indices == []):
            self.get_submatrix_indices(
                X=X, reset=reset, n_subsamples=n_subsamples, threshold=threshold
            )

        if len(self.submatrix_indices) == 0:
            return X
        else:
            if abs(index) > len(self.submatrix_indices) or index >= len(
                self.submatrix_indices
            ):
                raise IndexError(
                    "Index larger than number of \
                     available submatrices."
                )
        ixs = self.submatrix_indices[index]

        return X[ixs["rows"], :][:, ixs["columns"]]

    def reset_plotting_spectrum(self):
        self.plotting_spectrum = {}

    def get_plotting_spectrum(
        self, subsample=False, get_raw=False, dense_svd=None, reset=False, X=None
    ):
        """
        Return (and compute, if necessary) the eigenvalues of the covariance
        matrices associated with 1) the unscaled data and 2) the biscaled,
        normalized data.

        Parameters
        ----------
        subsample : bool, optional
            Compute the covariance eigenvalues over a subset of the data
            Default False.
        X : None, optional
            Description

        Returns
        -------
        dict
            Description
        """
        if X is None:
            X = self.X
        if reset:
            self.reset_plotting_spectrum()
        if dense_svd is None:
            dense_svd = self.dense_svd
        if attr_exists_not_none(self, "plotting_spectrum"):
            written_keys = self.plotting_spectrum.keys()
            if "Y" not in written_keys or (get_raw and "X" not in written_keys):
                with self.logger.task("plotting spectra"):
                    if attr_exists_not_none(self, "kst"):
                        # The variance has already been fit
                        # Get the indices associated with the best fit
                        best_flat_idx = np.argmin(self.kst)
                        best_submtx_idx, best_q_idx = np.unravel_index(
                            best_flat_idx, self.kst.shape
                        )
                        best_subsample_idxs = {
                            "rows": self.submatrix_indices["rows"][best_submtx_idx],
                            "columns": self.submatrix_indices["columns"][
                                best_submtx_idx
                            ],
                        }
                    else:
                        # The variance has not been fit.
                        # This occurs in the binomial case
                        if subsample:
                            _ = self.get_submatrix_indices(X=X)
                            best_submtx_idx = 0
                            best_subsample_idxs = {
                                "rows": self.submatrix_indices[best_submtx_idx]["rows"],
                                "columns": self.submatrix_indices[best_submtx_idx][
                                    "columns"
                                ],
                            }
                        else:
                            best_submtx_idx = 0
                            best_subsample_idxs = {
                                "rows": np.arange(X.shape[0]),
                                "columns": np.arange(X.shape[1]),
                            }

                    # subsample the matrix
                    if subsample:
                        xsub = X[best_subsample_idxs["rows"], :]
                        xsub = xsub[:, best_subsample_idxs["columns"]]
                    else:
                        xsub = X
                    Msub = np.min(xsub.shape)
                    Nsub = np.max(xsub.shape)
                    self.plotting_spectrum["shape"] = np.array([Msub, Nsub])
                    if Msub == np.min(X.shape):
                        # this is the raw data
                        not_a_submtx = True
                    else:
                        not_a_submtx = False

                    if self.svd_backend == "scipy" and Msub >= 27000:
                        raise Exception(
                            "The optimal workspace size is larger than allowed "
                            "by 32-bit interface to backend math library. "
                            "Use a partial SVD or set vals_only=True"
                        )
                    if get_raw:
                        with self.logger.task("spectrum of raw data"):
                            # get the spectrum of the raw data
                            svd = SVD(
                                k=Msub,
                                backend=self.svd_backend,
                                exact=True,
                                vals_only=True,
                                force_dense=dense_svd,
                                use_eig=True,
                                relative=self,
                                verbose=self.verbose,
                            )
                            svd.fit(xsub)
                            self.plotting_spectrum["X"] = (svd.S / np.sqrt(Nsub)) ** 2

                    with self.logger.task("spectrum of biwhitened data"):
                        if not_a_submtx:
                            # we're working with a full matrix
                            if len(self.S_Y) == Msub:
                                # if we already have the entire SVD, we don't
                                # need to recompute
                                pass
                            else:
                                # if we don't already have the entire SVD,
                                # we need to get the biwhitened matrix
                                # and compute its spectrum
                                msub = self.get_Y(X)

                                svd = SVD(
                                    k=self.M,
                                    backend=self.svd_backend,
                                    relative=self,
                                    exact=True,
                                    vals_only=True,
                                    force_dense=dense_svd,
                                    use_eig=True,
                                    verbose=self.verbose,
                                )
                                svd.fit(msub)
                                self.svd.S = svd.S
                                self.logger.warning(
                                    "Resetting shrinker given new complete set of eigenvalues."
                                    " Recommend obtaining new estimates of downstream matrices."
                                )
                                self.shrinker = Shrinker(
                                    default_shrinker=self.default_shrinker,
                                    rescale_svs=True,
                                    relative=self,
                                    suppress=self.suppress,
                                )
                                if (
                                    self.variance_estimator == "binomial"
                                ):  # no variance estimate needed when binomial is used.
                                    sigma_estimate = 1
                                else:
                                    if self.fit_sigma:
                                        sigma_estimate = None
                                    else:
                                        sigma_estimate = 1

                                _, _ = self.shrinker.fit(
                                    self.S_Y, shape=msub.shape, sigma=sigma_estimate
                                )
                                self._mp_rank = self.shrinker.scaled_mp_rank_
                            self.plotting_spectrum["Y"] = self.shrinker.scaled_cov_eigs

                        else:
                            # biwhiten the submatrix using the fitted & averaged parameters
                            if self.variance_estimator == "quadratic":
                                sinkhorn = Sinkhorn(
                                    tol=self.sinkhorn_tol,
                                    n_iter=self.n_iter,
                                    bhat=self.bhat,
                                    chat=self.chat,
                                    read_counts=self.read_counts,
                                    variance_estimator="quadratic_2param",
                                    relative=self,
                                    backend=self.sinkhorn_backend,
                                    conserve_memory=self.conserve_memory,
                                    suppress=self.suppress,
                                    **self.sinkhorn_kwargs,
                                )
                            if self.variance_estimator == "binomial":
                                sinkhorn = Sinkhorn(
                                    tol=self.sinkhorn_tol,
                                    n_iter=self.n_iter,
                                    read_counts=self.read_counts,
                                    variance_estimator=self.variance_estimator,
                                    relative=self,
                                    backend=self.sinkhorn_backend,
                                    conserve_memory=self.conserve_memory,
                                    suppress=self.suppress,
                                    **self.sinkhorn_kwargs,
                                )
                            msub = sinkhorn.fit_transform(xsub)
                            # get the spectrum of the biwhitened matrix
                            svd = SVD(
                                k=Msub,
                                backend=self.svd_backend,
                                exact=True,
                                vals_only=True,
                                force_dense=dense_svd,
                                use_eig=True,
                                verbose=self.verbose,
                            )
                            svd.fit(msub)
                            subshrinker = Shrinker(
                                default_shrinker=self.default_shrinker,
                                rescale_svs=True,
                                relative=self,
                                suppress=self.suppress,
                            )
                            _, _ = subshrinker.fit(
                                svd.S, shape=msub.shape, sigma_estimate=None
                            )
                            self.plotting_spectrum["Y"] = subshrinker.scaled_cov_eigs

                        MP = MarcenkoPastur(gamma=Msub / Nsub)
                        kst = KS(self.plotting_spectrum["Y"], MP)

                        self.plotting_spectrum["kst"] = kst
                        self.plotting_spectrum["normalized_kst"] =  (
                            kst - (self.plotting_spectrum["Y"] >= MP.b).sum() / Msub
                        ) ** 2

                        if self.variance_estimator == "quadratic":
                            self.plotting_spectrum["b"] = self.b
                            self.plotting_spectrum["c"] = self.c
                            self.plotting_spectrum["bhat"] = self.bhat
                            self.plotting_spectrum["chat"] = self.chat
                            self.plotting_spectrum["bhat_var"] = np.var(self.best_bhats)
                            self.plotting_spectrum["chat_var"] = np.var(self.best_chats)
                            if hasattr(self, "f_vals"):
                                self.plotting_spectrum["fits"] = {
                                    str(n): {} for n in range(len(self.f_vals))
                                }
                                for dix, outs in enumerate(self.f_vals):
                                    fitdict = self.plotting_spectrum["fits"][str(dix)]
                                    sigma = outs[0]
                                    kst = outs[1]
                                    fitdict["q"] = self.f_nodes
                                    fitdict["sigma"] = sigma
                                    fitdict["kst"] = kst
                                    bhat = self.compute_bhat(self.f_nodes, sigma)
                                    chat = self.compute_chat(self.f_nodes, sigma)
                                    c = self.compute_c(chat)
                                    b = self.compute_b(bhat, chat)
                                    fitdict["bhat"] = bhat
                                    fitdict["chat"] = chat
                                    fitdict["b"] = b
                                    fitdict["c"] = c
            return self.plotting_spectrum

    def _quadratic_bipca(self, X, q):
        shp = (np.min(X.shape), np.max(X.shape))
        if not self.suppress:
            verbose = self.verbose
        else:
            verbose = 0
        sinkhorn = Sinkhorn(
            read_counts=self.read_counts,
            tol=self.sinkhorn_tol,
            n_iter=self.n_iter,
            q=q,
            variance_estimator="quadratic_convex",
            P=1,
            backend=self.sinkhorn_backend,
            verbose=verbose,
            conserve_memory=True,
            **self.sinkhorn_kwargs,
        )

        m = sinkhorn.fit_transform(X)
        del sinkhorn
        del X
        svd = SVD(
            k=np.min(m.shape),
            backend=self.svd_backend,
            exact=True,
            vals_only=True,
            force_dense=True,
            use_eig=True,
            verbose=verbose,
            conserve_memory=True,
        )
        svd.fit(m)
        del m
        s = svd.S
        del svd
        shrinker = Shrinker(verbose=0)

        shrinker.fit(s, shape=shp)
        MP = MarcenkoPastur(gamma=np.min(shp) / np.max(shp))
        if self.normalized_KS:
            kst = (
                normalized_KS(
                    shrinker.scaled_cov_eigs, MP, np.min(shp), shrinker.scaled_mp_rank_
                )
                ** 2
            )
        else:
            kst = KS(shrinker.scaled_cov_eigs, MP)
        output = (shrinker.scaled_cov_eigs, shrinker.sigma, kst)
        del shrinker
        del MP
        return output

    def _fit_chebyshev(self, sub_ix):
        if isinstance(sub_ix, int):
            xsub = self.get_submatrix(sub_ix)
        else:
            xsub = sub_ix

        f = CachedFunction(lambda q: self._quadratic_bipca(xsub, q)[1:], num_outs=2)
        p = Chebfun.from_function(lambda x: f(x)[1], domain=[0, 1], N=self.qits)
        coeffs = p.coefficients()
        nodes = np.array(list(f.keys()))  # q
        vals = np.asarray(
            f(nodes)
        )  # (sigma, kst) - don't use chebfun here because it only has kst

        del f
        ncoeffs = len(coeffs)
        approx_ratio = coeffs[-1] ** 2 / np.linalg.norm(coeffs) ** 2
        self.logger.info(
            f"Chebyshev approximation of KS reached {approx_ratio} with {ncoeffs} "
            "coefficients"
        )
        if ncoeffs == 1:
            # instead minimize in terms of sigma
            self.logger.info(
                "Because KS was constant, minimizing in terms of sigma instead of KS"
            )
            p = Chebfun.from_data(np.abs(1 - vals[0]), domain=[0, 1])
            coeffs = p.coefficients()
            ncoeffs = len(coeffs)
            approx_ratio = coeffs[-1] ** 2 / np.linalg.norm(coeffs) ** 2
            self.logger.info(
                f"Chebyshev approximation of sigma reached {approx_ratio} with {ncoeffs} "
                "coefficients"
            )
        q = minimize_chebfun(p)

        _, sigma, kst = self._quadratic_bipca(xsub, q)

        if vals is None:
            vals = (sigma, kst)
            nodes = np.array([0.5])
        bhat = self.compute_bhat(q, sigma)
        chat = self.compute_chat(q, sigma)
        kst = kst
        c = self.compute_c(chat)
        b = self.compute_b(bhat, c)

        self.logger.info("Estimated b={}, c={}, KS={}".format(b, c, kst))

        return (
            nodes,
            vals,
            bhat,
            chat,
            kst,
        )

    def init_quadratic_params(self, b, bhat, c, chat):
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
                if bhat is None:
                    bhat = bhat_tmp
                else:  # a bhat was specified and it is not clear if they match
                    if np.abs(bhat_tmp - bhat) <= 1e-6:  # they match close enough
                        pass
                    else:
                        raise ValueError(
                            "Quadratic parameters b and bhat "
                            + "were specified but did not match. Specify only"
                            + " one, or ensure that they match."
                        )
                # Now do the same matching for c
                chat_tmp = c / (1 + c)
                if chat is None:
                    chat = chat_tmp
                else:
                    if np.abs(chat_tmp - chat) <= 1e-6:
                        pass
                    else:
                        raise ValueError(
                            "Quadratic parameters c and chat "
                            + "were specified but did not match. Specify only"
                            + " one, or ensure that they match."
                        )

        self.bhat = bhat
        self.chat = chat

        if bhat is not None:
            self.best_bhats = np.array([bhat])
            self.best_chats = np.array([chat])
            self.q = chat / (bhat + chat)
            self.sigma = np.sqrt(chat + bhat)

    def fit_quadratic_variance(self, X=None):
        """Fit the quadratic variance parameter for Poisson variance estimator
        using a subsample of the data.
        Returns
        -------
        TYPE
            Description

        Parameters
        ----------
        X : None, optional
            Description
        """
        if self.bhat is not None and self.chat is not None:
            return self.bhat, self.chat

        else:
            self.bhat = 0
            self.chat = 0

        if X is None:
            X = self.X
        if self.n_subsamples == 0 or self.subsample_size == X.shape:
            task_string = "variance fit over entire input"
        else:
            task_string = "variance fit over {:d} submatrices".format(self.n_subsamples)
        self.qits = np.max([1, self.qits])

        with self.logger.task(task_string):
            nsubs = len(self.get_submatrix_indices(X=X))
            # the grid of qs we will resample the function over
            # the qs in the space of x

            self.best_bhats = np.zeros((nsubs,))
            self.best_chats = np.zeros_like(self.best_bhats)
            self.best_kst = np.zeros_like(self.best_bhats)
            self.f_nodes = None
            self.f_vals = [None] * nsubs

            submtx_generator = (self.get_submatrix(i) for i in range(nsubs))
            if self.njobs not in [1, 0]:
                if self.njobs < 1:
                    njobs = nsubs
                else:
                    njobs = self.njobs
            else:
                njobs = self.njobs
            if njobs not in [1, 0]:
                try:
                    with Pool(processes=njobs) as pool:
                        results = pool.map(self._fit_chebyshev, submtx_generator)
                except:
                    print("Unable to use multiprocessing")
                    results = map(self._fit_chebyshev, submtx_generator)
            else:
                results = map(self._fit_chebyshev, submtx_generator)

            for sub_ix, result in enumerate(results):
                (
                    nodes,
                    vals,
                    bhat,
                    chat,
                    kst,
                ) = result
                if self.f_nodes is None:
                    self.f_nodes = nodes
                else:
                    # this should never happen, as we're using the same quadrature for
                    # every submatrix
                    assert len(self.f_nodes) == len(nodes), "Nodes should be the same"
                self.f_vals[sub_ix] = vals
                # get a chebfun object to differentiate
                self.best_bhats[sub_ix] = bhat
                self.best_chats[sub_ix] = chat
                self.best_kst[sub_ix] = kst
            if self.minimize_mean:  # compute the minimizer of the means
                self.logger.info("Approximating the mean of all submatrices")
                if self.f_vals is not None:
                    mean_values = np.mean(self.f_vals, axis=0)
                    ks_p = Chebfun.from_data(
                        mean_values[1], domain=[0, 1]
                    )  # mean(ks(q))

                    sigma_p = Chebfun.from_data(
                        mean_values[0], domain=[0, 1]
                    )  # mean(sigma(q))
                    mean_kst = ks_p(self.f_nodes)
                    mean_sigma = sigma_p(self.f_nodes)
                    self.f_vals.append(np.asarray((mean_sigma, mean_kst)))

                    approx_ratio = (ks_p.coefficients()[-1] / 
                                np.linalg.norm(ks_p.coefficients()))**2
                    self.logger.info(
                        f"Approximation ratio is "
                        f"{approx_ratio} with {len(ks_p.coefficients())} coefficients"
                    )
                    if len(ks_p.coefficients()) == 1 or np.allclose(
                        ks_p.coefficients()[1:], 0
                    ):  # KS is constant
                        self.logger.info(
                            "KS is constant, computing q by minimizing ell1(1-sigma)"
                        )
                        sigma_p2 = Chebfun.from_data(
                           np.abs(1 - mean_values[0]), domain=[0, 1]
                        )
                        q = self.q = minimize_chebfun(
                            sigma_p2
                        )  # the q that minimizes sigma
                    else:
                        q = self.q = minimize_chebfun(ks_p)
                    self.sigma = sigma_p(q)

                    self.bhat = self.compute_bhat(q, self.sigma)
                    self.chat = self.compute_chat(q, self.sigma)

                else:
                    # cannot compute the mean of all submatrices
                    self.logger.info(
                        "Unable to compute mean. Computing b and c as the mean of the minimizers"
                    )
                    self.bhat = mean(self.best_bhats)
                    self.chat = mean(self.best_chats)

            else:  # compute the mean of the minimizer!
                self.logger.info("Computing b and c as the mean of the minimizers")
                self.bhat = mean(self.best_bhats)
                self.chat = mean(self.best_chats)
                self.logger.info(f"b={self.b}, c={self.c}")
            self.logger.info(f"b={self.b}, c={self.c}")
            return self.bhat, self.chat

    def _update_quadratic_parameters(self, sigma_nu):
        #update internal parameters
        self.sigma = np.sqrt(self.sigma**2 * sigma_nu**2)
        self.bhat = self.compute_bhat(self.q, self.sigma)
        self.chat = self.compute_chat(self.q, self.sigma)
        #update sinkhorn 
        self.sinkhorn._update_quadratic_parameters(sigma_nu,self.bhat,self.chat)
        #update SVD
        if attr_exists_not_none(self.svd,"S_"):
            self.svd.S_ /= sigma_nu
        if attr_exists_not_none(self.svd,"X_"):
            self.svd.X_ /= sigma_nu
        #update shrinker
        self.shrinker.sigma = 1.

        
    def compute_bhat(self, q, sigma):
        return multiply(subtract(1, q), square(sigma))

    def compute_chat(self, q, sigma):
        return multiply(q, square(sigma))

    def compute_b(self, bhat, c):
        return multiply(bhat,add(1,c))

    def compute_c(self, chat):
        return divide(chat,subtract(1, chat))

    @property
    def c(self):
        if attr_exists_not_none(self, "chat"):
            return self.compute_c(self.chat)  # (q*sigma^2) / (1-q*sigma^2)

    @property
    def b(self):
        if attr_exists_not_none(self, "bhat"):
            return self.compute_b(self.bhat, self.c)
