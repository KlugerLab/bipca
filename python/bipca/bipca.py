"""BiPCA: BiStochastic Principal Component Analysis
"""
import numpy as np
from dataclasses import dataclass
from typing import Union
from numbers import Number
from functools import partial
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
from .math import Sinkhorn, SVD, Shrinker, MarcenkoPastur, KS, SamplingMatrix
from .utils import (is_valid,
                    stabilize_matrix,
                    filter_dict,
                    nz_along,
                    attr_exists_not_none,
                    write_to_adata,
                    CachedFunction,
                    fill_missing)
from .base import *


class BiPCA(BiPCAEstimator):

    """Bistochastic PCA:
    Biscale and denoise according to the paper
    
    Parameters
    ----------
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
    subsample_indices : dict
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
    S_Z : TYPE
        Description
    """
    @dataclass
    class TransformParameters(ParameterSet):
        unscale: bool = ValidatedField(bool,[],False)
        shrinker: str = ValidatedField(str,[Shrinker.__is_valid_shrinker__], 'frobenius')
        denoised: bool = ValidatedField(bool,[],True)
        truncate: Union[bool,float,int] = ValidatedField((bool,Number),[],0)
        truncation_axis: int = ValidatedField((int),
                                [partial(is_valid,lambda x: x in [-1,0,1])],0)

    @dataclass 
    class FitParameters(ParameterSet):

        ## variance parameters (for precomputed fits)
        ## parameters for the QVF estimators
        b: Union[Number, None] = ValidatedField((Number, type(None)),
                            [],
                            None)
        bhat: Union[Number,None] = ValidatedField((Number, type(None)),
                            [],
                            None)
        c: Union[Number, None] = ValidatedField((Number, type(None)),
                            [],
                            None)
        chat: Union[Number,None] = ValidatedField((Number, type(None)),
                            [],
                            None)
        ## variance fitting:
        ### number of subsamples to take for computing variance
        n_subsamples: int = ValidatedField(int,
                            [partial(is_valid, lambda x: x>=1)],
                            5)
        subsample_threshold: Union[int,None] = ValidatedField((int, type(None)),
                                                [partial(is_valid, lambda x: x is None or x>=0)],
                                                None)
        subsample_size: int = ValidatedField(int, 
                              [partial(is_valid, lambda x: x>0)],
                              5000)
        
        ### chebyshev parameters
        chebyshev_iterations: int = ValidatedField(int, 
                                    [partial(is_valid, lambda x: x>=0)],
                                    51) #this parameter used to be qits
        ## final adjustment to noise variance?
        tweak_sigma: bool = ValidatedField(bool, [], False) #used to be called fit_sigma
        

    _parameters = BiPCAEstimator._parameters + ['fit_parameters',
                                                'transform_parameters',
                                                'sinkhorn_parameters',
                                                'svd_parameters',
                                                'shrinker_parameters']
    def __init__(self,fit_parameters=FitParameters(),
                    transform_parameters=TransformParameters(),
                    sinkhorn_parameters=Sinkhorn.FitParameters(),
                    logging_parameters=LoggingParameters(),
                    compute_parameters=ComputeParameters(),
                    oversample_factor=10,
                    b = None, bhat = None, c = None, chat = None,
                    keep_aspect=False, use_eig=True, dense_svd=True,
                    n_components = None, exact = True, subsample_threshold=None, backend = 'torch',
                    svd_backend=None,sinkhorn_backend='scipy', njobs=1,**kwargs):
        #build the logger first to share across all subprocedures
        super().__init__(compute_parameters=compute_parameters, 
                logging_parameters=logging_parameters,
                **kwargs)
        for parameter_set in self._parameters:
            params=eval(parameter_set)
            if parameter_set not in self.__dict__.keys():
                    self.__dict__[parameter_set] = replace_dataclass(params, **{key:value for key, value in kwargs.items() if key in params.__dataclass_fields__})
                    for field in params.__dataclass_fields__:
                        self.__dict__[field]=self.__dict__[parameter_set]
        #initialize the subprocedure classes
        self.k = n_components
        self.sinkhorn_tol = sinkhorn_tol
        self.default_shrinker=default_shrinker
        self.n_iter = n_iter
        self.exact = exact
        self.variance_estimator = variance_estimator
        self.subsample_size = subsample_size
        self.qits = qits
        self.use_eig=use_eig
        self.dense_svd=dense_svd
        self.n_subsamples=n_subsamples
        self.njobs = njobs
        self.fit_sigma=fit_sigma
        self.backend = backend
        self.svd_backend = svd_backend
        self.oversample_factor = oversample_factor
        self.sinkhorn_backend = sinkhorn_backend
        self.keep_aspect=keep_aspect
        self.read_counts = read_counts
        self.subsample_threshold = subsample_threshold
        self.init_quadratic_params(b,bhat,c,chat)
        self.reset_submatrices()
        self.reset_plotting_spectrum()
        #remove the kwargs that have been assigned by super.__init__()
        self._X = None

        #hotfix to remove tol collisions
        self.svdkwargs = kwargs

        self.sinkhorn_kwargs = kwargs.copy()
        if 'tol' in kwargs:
            del self.sinkhorn_kwargs['tol']




    ###Properties that construct the objects that we use to compute a bipca###
    @property
    def sinkhorn(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if not attr_exists_not_none(self,'_sinkhorn'):
            self._sinkhorn = Sinkhorn(tol = self.sinkhorn_tol, n_iter = self.n_iter, 
                variance_estimator = self.variance_estimator, 
                relative = self, backend=self.sinkhorn_backend,
                                **self.sinkhorn_kwargs)
        return self._sinkhorn
    @sinkhorn.setter
    def sinkhorn(self,val):
        """Summary
        
        Parameters
        ----------
        val : TYPE
            Description
        
        Raises
        ------
        ValueError
            Description
        """
        if isinstance(val, Sinkhorn):
            self._sinkhorn = val
        else:
            raise ValueError("Cannot set self.sinkhorn to non-Sinkhorn estimator")

    @property
    def svd(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if not attr_exists_not_none(self,'_svd'):
            self._svd =  SVD(n_components = self.k, exact=self.exact,
                    oversample_factor=self.oversample_factor,
                    backend = self.svd_backend, relative = self, 
                    conserve_memory = self.conserve_memory,force_dense=self.dense_svd,
                    use_eig=self.use_eig,suppress=self.suppress)
        return self._svd
    @svd.setter
    def svd(self,val):
        """Summary
        
        Parameters
        ----------
        val : TYPE
            Description
        
        Raises
        ------
        ValueError
            Description
        """
        if isinstance(val, SVD):
            self._svd = val
        else:
            raise ValueError("Cannot set self.svd to non-SVD estimator")


    @property
    def shrinker(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if not attr_exists_not_none(self,'_shrinker'):
            self._shrinker = Shrinker(default_shrinker=self.default_shrinker,
                    rescale_svs = True, relative = self,suppress=self.suppress)
        return self._shrinker
    @shrinker.setter
    def shrinker(self,val):
        """Summary
        
        Parameters
        ----------
        val : TYPE
            Description
        
        Raises
        ------
        ValueError
            Description
        """
        if isinstance(val, Shrinker):
            self._shrinker = val
        else:
            raise ValueError("Cannot set self.shrinker to non-Shrinker estimator")

    ###backend properties and resetters###

    @property
    def backend(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if not attr_exists_not_none(self,'_backend'):
            self._backend = 'scipy'
        return self._backend
    @backend.setter
    def backend(self, val):
        """Summary
        
        Parameters
        ----------
        val : TYPE
            Description
        """
        val = self.isvalid_backend(val)
        if attr_exists_not_none(self,'_backend'):
            if val != self._backend:
                if attr_exists_not_none(self,'_svd_backend'): #we check for none here. If svd backend is none, it follows self.backend, and there's no need to warn.
                    if val != self.svd_backend:
                        self.logger.warning("Changing the global backend is overwriting the SVD backend. \n" + 
                            "To change this behavior, set the global backend first by obj.backend = 'foo', then set obj.svd_backend.")
                        self.svd_backend=val
                if attr_exists_not_none(self,'_sinkhorn_backend'):
                    if val != self.sinkhorn_backend:
                        self.logger.warning("Changing the global backend is overwriting the sinkhorn backend. \n" + 
                            "To change this behavior, set the global backend first by obj.backend = 'foo', then set obj.sinkhorn_backend.")
                        self.sinkhorn_backend=val
                #its a new backend
                self._backend = val
        else:
            self._backend=val
        self.reset_backend()

    @property 
    def svd_backend(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if not attr_exists_not_none(self,'_svd_backend'):
            return self.backend
        else:
            return self._svd_backend

    @svd_backend.setter
    def svd_backend(self, val):
        """Summary
        
        Parameters
        ----------
        val : TYPE
            Description
        """
        val = self.isvalid_backend(val)
        if attr_exists_not_none(self, '_svd_backend'):
            if val != self._svd_backend:
                #its a new backend
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
        if not attr_exists_not_none(self,'_sinkhorn_backend'):
            return self.backend
        return self._sinkhorn_backend
    
    @sinkhorn_backend.setter
    def sinkhorn_backend(self,val):
        """Summary
        
        Parameters
        ----------
        val : TYPE
            Description
        """
        val = self.isvalid_backend(val)
        if attr_exists_not_none(self, '_sinkhorn_backend'):
            if val != self._sinkhorn_backend:
                #its a new backend
                self._sinkhorn_backend = val
        else:
            self._sinkhorn_backend = val
        self.reset_backend()

    def reset_backend(self):
        """Summary
        """
        #Must be called after setting backends.
        attrs = self.__dict__.keys()
        for attr in attrs:
            obj = self.__dict__[attr]
            objname =  obj.__class__.__name__.lower()
            if hasattr(obj, 'backend'):
                #the object has a backend attribute to set
                if 'svd' in objname:
                    obj.backend = self.svd_backend
                elif 'sinkhorn' in objname:
                    obj.backend = self.sinkhorn_backend
                else:
                    obj.backend = self.backend
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
    def n_components(self,val):
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
    def U_Z(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self.svd.U

    @fitted_property
    def S_Z(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self.svd.S

    @fitted_property
    def V_Z(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """

        return self.svd.V

    @property
    def Z(self):
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
            Z = self._Z
            if self._istransposed:
                Z = Z.T
            return Z
        else:
            raise RuntimeError("Since conserve_memory is true, Z can only be obtained by calling .get_Z(X)")
    @Z.setter
    def Z(self,Z):
        """Summary
        
        Parameters
        ----------
        Z : TYPE
            Description
        """
        if not self.conserve_memory:
            self._Z = Z
    
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
                if self._Y.shape != (self._M,self._N):
                    self._Y = self._Y.T
            return self._Y
        else:
            return self.transform()
    @Y.setter
    def Y(self,Y):
        """Summary
        
        Parameters
        ----------
        Y : TYPE
            Description
        """
        if not self.conserve_memory:
            self._Y = Y
    
    def get_Z(self,X = None):
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
            return self.Z
        else:
            return self.sinkhorn.transform(X)

    def unscale(self,X):
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
        if self._istransposed:
            return self._N/self._M
        return self._M/self._N
    @property
    def N(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if self._istransposed:
            return self._M
        return self._N
    @property
    def M(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if self._istransposed:
            return self._N
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
        #bug: sinkhorn needs to be reset when the model is refit.
        super().fit()
        with self.logger.task("BiPCA fit"):
            self._M, self._N = A.shape
            if self._M/self._N>1:
                self._istransposed = True
                A = A.T
            else:
                self._istransposed = False
            if isinstance(A, AnnData):
                A = A.X
            if not self.conserve_memory:
                self.X = A
            X = A
            if self.k == -1: # k is determined by the minimum dimension
                self.k = self.M
            elif self.k is None or self.k == 0: #automatic k selection
                       
                    self.k = np.min([200,self.M//2])
                    if self.variance_estimator == 'binomial':
                        #if it is binomial, we don't need to estimate parameters
                        #of the distribution, so we only need to take enough
                        #singular values to cover the data
                        self.k = np.min([200,self.M])

            self.k = np.min([self.k, *X.shape]) #ensure we are not asking for too many SVs
            self.svd.k = self.k
            self.P = SamplingMatrix(X)
            if self.P.ismissing:
                X = fill_missing(X)
                if not self.conserve_memory:
                    self.X = X
            else:
                self.P = 1
            if self.variance_estimator == 'quadratic':
                self.bhat,self.chat = self.fit_quadratic_variance(X=X)
                self.sinkhorn = Sinkhorn(tol = self.sinkhorn_tol, n_iter = self.n_iter,
                                bhat = self.bhat, chat = self.chat,
                                read_counts=self.read_counts, P = self.P,
                                variance_estimator = 'quadratic_2param', 
                                relative = self, backend=self.sinkhorn_backend,
                                conserve_memory = self.conserve_memory, suppress=self.suppress,
                                **self.sinkhorn_kwargs)
            else:
                b = 1
                c = -1/self.read_counts
                self.init_quadratic_params(b=b,c=c,bhat=None,chat=None)
                self.sinkhorn = Sinkhorn(tol = self.sinkhorn_tol, n_iter = self.n_iter,
                        read_counts=self.read_counts, variance_estimator = 'quadratic_2param',
                        bhat = self.bhat,chat=self.chat,
                        b = self.b, c = self.c, P = self.P,
                        relative = self, backend=self.sinkhorn_backend,
                        conserve_memory = self.conserve_memory, suppress=self.suppress,
                        **self.sinkhorn_kwargs)
        
            M = self.sinkhorn.fit_transform(X)
            self.Z = M
            if self.variance_estimator =='binomial': # no variance estimate needed when binomial is used.
                sigma_estimate = 1
            else:
                if self.fit_sigma:
                    sigma_estimate = None
                else:
                    sigma_estimate = 1
                

            converged = False

            while not converged:
                
                self.svd.fit(M)
                toshrink = self.S_Z
                _, converged = self.shrinker.fit(toshrink, shape = X.shape,sigma=sigma_estimate)
                self._mp_rank = self.shrinker.scaled_mp_rank_
                if not converged:
                    self.k = int(np.min([self.k*1.5, *X.shape]))
                    self.svd.k = self.k
                    self.logger.info("Full rank partial decomposition detected,"
                                    " fitting with a larger k = {}".format(self.k))
            del M
            del X

            return self

        
    @fitted
    def transform(self,  X = None):
        """Return a denoised version of the data.
        
        Parameters
        ----------
        X : array, optional
            If `BiPCA.conserve_memory` is True, then X must be provided in order to obtain 
            the solely biwhitened transform, i.e., for unscale=False, denoised=False.
        unscale : bool, default False
            Unscale the output matrix so that it is in the original input domain.
        shrinker : {'hard','soft', 'frobenius', 'operator','nuclear'}, optional
            Shrinker to use for denoising
            (Defaults to `obj.default_shrinker`)
        denoised : bool, default True
            Return denoised output.
        truncate : numeric or bool, default 0
            Truncate the transformed data. If <=0 or true, then the output is thresholded at 0.
            If nonzero, the truncate-th quantile along `truncation_axis` is used to adaptively 
            threshold the output.
        truncation_axis : {-1, 0, 1}, default 0.
            Axis to gather truncation thresholds from. Uses numpy axes:
            truncatioon_axis==-1 applies a single threshold from the entire matrix.
            truncation_axis==0 gathers thresholds down the rows, therefore is column-wise
            truncation_axis==1 gathers thresholds across the columns, therefore is row-wise
        
        Returns
        -------
        np.array
            (N x M) transformed array
        """
        if denoised:
            sshrunk = self.shrinker.transform(self.S_Z, shrinker=shrinker)
            if self.U_Z.shape[1] == self.k:
                Y = (self.U_Z[:,:self.mp_rank]*sshrunk[:self.mp_rank])
            else:
                Y = (self.U_Z[:self.mp_rank,:].T*sshrunk[:self.mp_rank])
            if self.V_Z.shape[0] == self.k:
                Y = Y@self.V_Z[:self.mp_rank,:]
            else:
                Y = Y@self.V_Z[:,:self.mp_rank].T
        else:
            if not self.conserve_memory:
                Y = self.Z #the full rank, biwhitened matrix.
            else:
                Y = self.get_Z(X)
        if truncate is not False:
            if not denoised: # There's a bug here when Y is a sparse matrix. This only happens when Y is Z
                pass
            else:
                truncate = 0 if truncate is True else truncate
                if truncate <= 0:
                    Y = np.where(Y<=truncate, 0,Y)
                else:
                    if truncation_axis>=0:
                        if self._istransposed: #if transposed, we need to fix the truncation_axis
                            #truncation_axis==0 -> threshold on columns of input, which are rows of internal if transposed
                            truncation_axis=[1,0][truncation_axis]
                        
                        thresh = np.abs(np.minimum(np.percentile(Y, truncate, axis=truncation_axis),0))
                        if truncation_axis==1:
                            thresh=thresh[:,None]
                    else:
                        thresh = np.abs(np.min(np.percentile(Y,truncate),0))
                    Y = np.where(np.less_equal(Y,thresh), 0, Y)
        if unscale:
            Y = self.unscale(Y)
        if self._istransposed:
            Y = Y.T

        self.Y = Y
        return Y
    def fit_transform(self, X = None, shrinker = None,**kwargs):
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
        return self.transform(shrinker=shrinker,**kwargs)
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
        return write_to_adata(self,adata)
    def reset_submatrices(self):
        self.subsample_indices = {'rows':[],
                                'columns':[]}


    def get_submatrices(self, reset=False, X = None, n_subsamples = None,
        subsample_size = None, threshold=None):
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
        if subsample_size is None:
            subsample_size = self.subsample_size
        if subsample_size is None:
            subsample_size = 2000
        if threshold is None:
            threshold = self.subsample_threshold
        if self.variance_estimator == 'quadratic':
            variance_estimator = 'quadratic_convex'
        else:
            variance_estimator = self.variance_estimator

        if threshold is None: 
            # compute the threshold by the minimum nnz in the variance estimate
            sinkhorn = Sinkhorn(read_counts=self.read_counts,
                        tol = self.sinkhorn_tol, n_iter = self.n_iter,
                        variance_estimator = variance_estimator, 
                        backend = self.sinkhorn_backend,
                        verbose=0, **self.sinkhorn_kwargs) 
            varX = sinkhorn.estimate_variance(X)[0]
            cols = nz_along(varX,axis=0)
            rows = nz_along(varX,axis=1)
            cols = np.min(cols)-1
            rows = np.min(rows)-1
            threshold = np.min([rows,cols])


        if reset or not attr_exists_not_none(self, 'subsample_indices'):
            self.reset_submatrices()
        
        sub_M = subsample_size
        if self.keep_aspect:
            sub_N = np.floor(1/self.aspect_ratio * sub_M).astype(int)
        else:
            sub_N = np.min([subsample_size,X.shape[1]])
        if self.subsample_indices['rows'] == []:
            if n_subsamples == 0 or subsample_size >= np.min(X.shape):
                rixs = np.arange(X.shape[0])
                cixs = np.arange(X.shape[1])
                self.subsample_indices['rows'].append(rixs)
                self.subsample_indices['columns'].append(cixs)
            else:
                for n_ix in range(n_subsamples):
                    rng = np.random.default_rng()
                    rixs = rng.permutation(X.shape[0])
                    cixs = rng.permutation(X.shape[1])
                    rixs = rixs[:sub_M]
                    cixs = cixs[:sub_N]
                    xsub = X[rixs,:][:,cixs]
                    # instantiate a sinkhorn instance to get a proper variance estimate
                    # we have to stabilize the matrix based on the sparsity of the variance estimate
                    sinkhorn = Sinkhorn(read_counts=self.read_counts,
                        tol = self.sinkhorn_tol, n_iter = self.n_iter,
                        variance_estimator = variance_estimator, 
                        backend = self.sinkhorn_backend,
                        verbose=0, **self.sinkhorn_kwargs)
                    varX = sinkhorn.estimate_variance(xsub)[0]
                    cols = nz_along(varX,axis=0)
                    rows = nz_along(varX,axis=1)
                    cols = np.max(cols)
                    rows = np.max(rows)
                    if cols<threshold or rows < threshold:
                        threshold_proportion = threshold / np.min(X.shape)
                        thresh_temp = threshold_proportion * np.min(xsub.shape)
                        threshold = int(np.max([np.floor(thresh_temp),1]))
                    _, (mixs, nixs) = stabilize_matrix(
                        varX,
                        threshold = threshold)

                    rixs = rixs[mixs]
                    cixs = cixs[nixs]
                    self.subsample_indices['rows'].append(rixs)
                    self.subsample_indices['columns'].append(cixs)

        return [X[rixs,:][:,cixs] for rixs, cixs in zip(
            self.subsample_indices['rows'], self.subsample_indices['columns'])]
    def reset_plotting_spectrum(self):
        self.plotting_spectrum = {}    
    def get_plotting_spectrum(self,  subsample = False, get_raw=True, dense_svd=None, reset = False, X = None):
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
        if attr_exists_not_none(self,'plotting_spectrum'):
            written_keys = self.plotting_spectrum.keys()
            if 'Y' not in written_keys or (get_raw and 'X' not in written_keys):
                with self.logger.task("plotting spectra"):
                        if attr_exists_not_none(self,'kst'):
                            #The variance has already been fit
                            #Get the indices associated with the best fit
                            best_flat_idx = np.argmin(self.kst)
                            best_submtx_idx, best_q_idx = np.unravel_index(
                                best_flat_idx,self.kst.shape)
                            best_subsample_idxs = {
                                'rows':self.subsample_indices['rows'][best_submtx_idx],
                                'columns':self.subsample_indices['columns'][best_submtx_idx]
                                }
                        else:
                            # The variance has not been fit. 
                            # This occurs in the binomial case
                            if subsample:
                                _ = self.get_submatrices(X=X)
                                best_submtx_idx = 0
                                best_subsample_idxs = {
                                    'rows':self.subsample_indices['rows'][best_submtx_idx],
                                    'columns':self.subsample_indices['columns'][best_submtx_idx]
                                    }
                            else:
                                best_submtx_idx = 0
                                best_subsample_idxs = {'rows':np.arange(X.shape[0]),
                                                        'columns': np.arange(X.shape[1])}

                        #subsample the matrix
                        if subsample:
                            xsub = X[best_subsample_idxs['rows'],:]
                            xsub = xsub[:,best_subsample_idxs['columns']]
                        else:
                            xsub = X
                        Msub = np.min(xsub.shape)
                        Nsub = np.max(xsub.shape)
                        self.plotting_spectrum['shape'] = np.array([Msub, Nsub])
                        if Msub == np.min(X.shape):
                            #this is the raw data
                            not_a_submtx = True
                        else:
                            not_a_submtx = False

                        if self.svd_backend=='scipy' and Msub >= 27000:
                            raise Exception("The optimal workspace size is larger than allowed "
                                "by 32-bit interface to backend math library. "
                                "Use a partial SVD or set vals_only=True")
                        if get_raw:
                            with self.logger.task("spectrum of raw data"):
                                #get the spectrum of the raw data
                                svd = SVD(k = Msub, backend=self.svd_backend, 
                                    exact = True,vals_only=True, force_dense=dense_svd,
                                    use_eig=True,relative=self,verbose=self.verbose)
                                svd.fit(xsub)
                                self.plotting_spectrum['X'] = (svd.S /
                                                                np.sqrt(Nsub))**2

                        with self.logger.task("spectrum of biwhitened data"):
                            if not_a_submtx:
                                # we're working with a full matrix
                                if len(self.S_Z) == Msub:
                                    # if we already have the entire SVD, we don't
                                    # need to recompute
                                    self.plotting_spectrum['Y'] = (self.S_Z/np.sqrt(Nsub))**2
                                else:
                                    # if we don't already have the entire SVD,
                                    # we need to get the biwhitened matrix
                                    # and compute its spectrum
                                    msub = self.get_Z(X)

                                    svd = SVD(k = self.M, 
                                        backend=self.svd_backend, relative=self,
                                        exact=True, vals_only=True, force_dense=dense_svd,
                                        use_eig=True,verbose = self.verbose)
                                    svd.fit(msub)
                                    self.plotting_spectrum['Y'] = (svd.S /
                                                             np.sqrt(Nsub))**2


                            else:
                                #biwhiten the submatrix using the fitted & averaged parameters
                                if self.variance_estimator == 'quadratic':
                                    sinkhorn = Sinkhorn(tol = self.sinkhorn_tol, 
                                            n_iter = self.n_iter,
                                            bhat = self.bhat, chat = self.chat,
                                            read_counts=self.read_counts,
                                            variance_estimator = 'quadratic_2param', 
                                            relative = self, 
                                            backend=self.sinkhorn_backend,
                                            conserve_memory = self.conserve_memory, 
                                            suppress=self.suppress,
                                            **self.sinkhorn_kwargs)
                                if self.variance_estimator == 'binomial':
                                    sinkhorn = Sinkhorn(tol = self.sinkhorn_tol, 
                                            n_iter = self.n_iter,
                                            read_counts=self.read_counts,
                                            variance_estimator = self.variance_estimator,
                                            relative = self, 
                                            backend=self.sinkhorn_backend,
                                            conserve_memory = self.conserve_memory, 
                                            suppress=self.suppress,
                                            **self.sinkhorn_kwargs)                  
                                msub = sinkhorn.fit_transform(xsub)
                                #get the spectrum of the biwhitened matrix
                                svd = SVD(k = Msub, backend=self.svd_backend,
                                    exact=True, vals_only=True, force_dense=dense_svd,
                                    use_eig=True,verbose = self.verbose)
                                svd.fit(msub)
                                self.plotting_spectrum['Y'] = (svd.S /
                                                             np.sqrt(Nsub))**2
                            MP = MarcenkoPastur(gamma = Msub/Nsub)
                            kst = KS(self.plotting_spectrum['Y'],
                                                            MP)
                                                            
                            self.plotting_spectrum['kst'] = kst

                            if self.variance_estimator == 'quadratic':
                                self.plotting_spectrum['b'] = self.b
                                self.plotting_spectrum['c'] = self.c
                                self.plotting_spectrum['bhat'] = self.bhat
                                self.plotting_spectrum['chat'] = self.chat
                                self.plotting_spectrum['bhat_var'] = np.var(
                                                                self.best_bhats)
                                self.plotting_spectrum['chat_var'] = np.var(
                                                                self.best_chats)
                                if hasattr(self,'chebfun'):
                                    self.plotting_spectrum['fits'] = {n:{} for n in range(len(self.chebfun))}
                                    for q,outs,dix,chebfun in zip(self.f_nodes,
                                                                self.f_vals,
                                                                range(len(self.chebfun)),
                                                                self.chebfun):

                                        fitdict = self.plotting_spectrum['fits'][dix]
                                        sigma = outs[0]
                                        kst = outs[1]
                                        fitdict['q'] = q
                                        fitdict['sigma'] = sigma
                                        fitdict['kst'] = kst
                                        bhat = is_valid_self.compute_bhat(q,sigma)
                                        chat = self.compute_chat(q,sigma)
                                        c = self.compute_c(chat)
                                        b = self.compute_b(bhat,chat)
                                        fitdict['bhat'] = bhat
                                        fitdict['chat'] = chat
                                        fitdict['b'] = b
                                        fitdict['c'] = c
                                        fitdict['coefficients'] = None
                                        if chebfun is not None:
                                            fitdict['coefficients'] = chebfun.coefficients()
            return self.plotting_spectrum
    def _quadratic_bipca(self, X, q):
        if X.shape[1]<X.shape[0]:
            X = X.T
        if not self.suppress:
            verbose = self.verbose
        else:
            verbose = 0
        sinkhorn = Sinkhorn(read_counts=self.read_counts,
                        tol = self.sinkhorn_tol, n_iter = self.n_iter, q = q,
                        variance_estimator = 'quadratic_convex', 
                        backend = self.sinkhorn_backend,
                        verbose=verbose, **self.sinkhorn_kwargs)
                    
        m = sinkhorn.fit_transform(X)
        svd = SVD(k = np.min(X.shape), backend=self.svd_backend, 
            exact = True,vals_only=True, force_dense=True,use_eig=True,verbose=verbose)
        svd.fit(m)
        s = svd.S
        shrinker = Shrinker(verbose=0)

        shrinker.fit(s,shape = X.shape) 
        kst = KS(shrinker.scaled_cov_eigs,MP)
        return shrinker.scaled_cov_eigs,shrinker.sigma, kst
    def _fit_chebyshev(self, sub_ix):
        xsub = self.get_submatrices()[sub_ix]

        if xsub.shape[1]<xsub.shape[0]:
            xsub = xsub.T
        f = CachedFunction(lambda q: self._quadratic_bipca(xsub, q)[1:],num_outs=2)
        p = Chebfun.from_function(lambda x: f(x)[1],domain=[0,1],N=self.qits)
        coeffs = p.coefficients()
        nodes = np.array(list(f.keys()))
        vals = f(nodes)
        ncoeffs = len(coeffs)
        approx_ratio = coeffs[-1]**2/np.linalg.norm(coeffs)**2

        #compute the minimum
        pd = p.differentiate()
        pdd = pd.differentiate()
        try:
            q = pd.roots() # the zeros of the derivative
            #minima are zeros of the first derivative w/ positive second derivative
            mi = q[pdd(q)>0]
            if mi.size == 0:
                mi = np.linspace(0,1,100000)

            x = np.linspace(0,1,100000)
            x_ix = np.argmin(p(x))
            mi_ix = np.argmin(p(mi))
            if p(x)[x_ix] <= p(mi)[mi_ix]:
                q = x[x_ix]
            else:
                q = mi[mi_ix]
        except IndexError:
            x = np.linspace(0,1,100000)
            x_ix = np.argmin(p(x))
            q = x[x_ix]

        totest, sigma, kst = self._quadratic_bipca(xsub, q)

        if vals is None:
            vals = (sigma,kst)
            nodes=np.array([0.5])
        bhat = self.compute_bhat(q,sigma)
        chat = self.compute_chat(q,sigma)
        kst = kst
        c = self.compute_c(chat)
        b = self.compute_b(bhat,c)
        self.logger.info("Chebyshev approximation ratio reached {} with {} coefficients".format(approx_ratio,ncoeffs))
        self.logger.info("Estimated b={}, c={}, KS={}".format(b,c,kst))

        return nodes, vals, coeffs, approx_ratio, ncoeffs, bhat, chat, kst, b, c
    def init_quadratic_params(self,b,bhat,c,chat):
        if b is not None:
            ## A b value was specified
            if c is None:
                raise ValueError("Quadratic variance parameter b was"+
                    " specified, but c was not. Both must be specified.")
            else:
                bhat_tmp = b/(1+c)
                #check that if bhat was specified that they match b
                if bhat is None:
                    bhat = bhat_tmp
                else: #a bhat was specified and it is not clear if they match
                    if np.abs(bhat_tmp - bhat) <= 1e-6: #they match close enough
                        pass
                    else:
                        raise ValueError("Quadratic parameters b and bhat "+
                            "were specified but did not match. Specify only"+
                            " one, or ensure that they match.")
                # Now do the same matching for c
                chat_tmp = c/(1+c)
                if chat is None:
                    chat = chat_tmp
                else:
                    if np.abs(chat_tmp - chat) <= 1e-6:
                        pass
                    else:
                        raise ValueError("Quadratic parameters c and chat "+
                            "were specified but did not match. Specify only"+
                            " one, or ensure that they match.")

        self.bhat = bhat
        self.chat = chat
        if bhat is not None:
            self.best_bhats = np.array([bhat])
            self.best_chats = np.array([chat])
    def fit_quadratic_variance(self, X = None):
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
        if self.n_subsamples == 0 or self.subsample_size >= np.min(X.shape):
            task_string = "variance fit over entire input"
        else:
            task_string = "variance fit over {:d} submatrices".format(self.n_subsamples)
        self.qits = np.max([1,self.qits])



        with self.logger.task(task_string):
            submatrices = self.get_submatrices(X=X)
                #the grid of qs we will resample the function over
                #the qs in the space of x
            self.best_bhats = np.zeros((len(submatrices),))
            self.best_chats = np.zeros_like(self.best_bhats)
            self.best_kst = np.zeros_like(self.best_bhats)
            self.chebfun = [None] * len(submatrices)
            self.f_nodes = [None] * len(submatrices)
            self.f_vals = [None] * len(submatrices)
            self.approx_ratio = np.zeros_like(self.best_bhats)
            if self.njobs not in [1,0]:
                if self.njobs<1:
                    njobs = len(submatrices)
                else:
                    njobs = self.njobs
            else:
                njobs = self.njobs
            if njobs not in [1,0]:
                try:
                    with Pool(processes=njobs) as pool:
                        results = pool.map(self._fit_chebyshev, range(len(submatrices)))
                except:
                    print("Unable to use multiprocessing")
                    results = map(self._fit_chebyshev,range(len(submatrices)))
            else:
                results = map(self._fit_chebyshev,range(len(submatrices)))
            for sub_ix, result in enumerate(results):
                nodes, vals, coeffs, approx_ratio, ncoeffs, bhat, chat, kst, b, c = result
                if coeffs is not None:
                    self.chebfun[sub_ix] = Chebfun.from_coeff(coeffs, domain=[0,1])
                else:
                    self.chebfun[sub_ix] = None
                self.approx_ratio[sub_ix] = approx_ratio
                self.f_nodes[sub_ix] = nodes
                self.f_vals[sub_ix] = vals
                #get a chebfun object to differentiate
                self.best_bhats[sub_ix] = bhat
                self.best_chats[sub_ix] = chat
                self.best_kst[sub_ix] = kst
            self.bhat = np.mean(self.best_bhats)
            self.chat = np.mean(self.best_chats)

            #write the chebyshev fits to the plotting spectrum
            self.plotting_spectrum['b'] = self.b
            self.plotting_spectrum['c'] = self.c
            self.plotting_spectrum['bhat'] = self.bhat
            self.plotting_spectrum['chat'] = self.chat
            self.plotting_spectrum['bhat_var'] = np.var(
                                            self.best_bhats)
            self.plotting_spectrum['chat_var'] = np.var(
                                            self.best_chats)
            self.plotting_spectrum['fits'] = {n:{} for n in range(len(self.chebfun))}
            for q,outs,dix,chebfun in zip(self.f_nodes,
                                        self.f_vals,
                                        range(len(self.chebfun)),
                                        self.chebfun):

                fitdict = self.plotting_spectrum['fits'][dix]
                sigma = outs[0]
                kst = outs[1]
                fitdict['q'] = q
                fitdict['sigma'] = sigma
                fitdict['kst'] = kst
                bhat = self.compute_bhat(q,sigma)
                chat = self.compute_chat(q,sigma)
                c = self.compute_c(chat)
                b = self.compute_b(bhat,chat)
                fitdict['bhat'] = bhat
                fitdict['chat'] = chat
                fitdict['b'] = b
                fitdict['c'] = c
                fitdict['coefficients'] = None
                if chebfun is not None:
                    fitdict['coefficients'] = chebfun.coefficients()

            return self.bhat, self.chat
    def compute_bhat(self,q,sigma):
        return (1-q) * sigma ** 2
    def compute_chat(self,q,sigma):
        return q * sigma ** 2
    def compute_b(self,bhat,c):
        return bhat * (1+c)
    def compute_c(self,chat):
        return chat/(1-chat)
    @property
    def c(self):
        if attr_exists_not_none(self,'chat'):
            return self.compute_c(self.chat) #(q*sigma^2) / (1-q*sigma^2)
    @property
    def b(self):
        if attr_exists_not_none(self,'bhat'):
            return self.compute_b(self.bhat,self.c)

