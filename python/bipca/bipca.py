"""BiPCA: BiStochastic Principal Component Analysis
"""
import numpy as np
import sklearn as sklearn
import scipy as scipy
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import scipy.sparse as sparse
from scipy.stats import kstest
import tasklogger
from anndata._core.anndata import AnnData
from .math import Sinkhorn, SVD, Shrinker, MarcenkoPastur, KS, MeanCenteredMatrix
from .utils import stabilize_matrix, filter_dict, nz_along,attr_exists_not_none,write_to_adata
from .base import *

class BiPCA(BiPCAEstimator):

    """Bistochastic PCA:
    Biscale and denoise according to the paper
    
    Parameters
    ----------
    center : bool, optional
        Center the biscaled matrix before denoising. `True` introduces a dense matrix to the problem,
        which can lead to memory problems and slow results for large problems. This option is not recommended for large problems.
        Default False.
    variance_estimator : {'poisson','binomial'}, default 'poisson'
        Variance estimator to use when Sinkhorn biscaling.
    q : int, default 0
        Precomputed quadratic variance for generalized Poisson sinkhorn biwhitening. Used when `qits <= 1`
    qits : int, default 5
        Number of variance fitting cycles to run when `variance_estimator` is `'poisson'`.
        If `qits <= 1`, then no variance fitting is performed.
    approximate_sigma : bool, optional
        Estimate the noise variance for the Marcenko Pastur model using a submatrix of the original data
        Default True for inputs with small axis larger than 2000.
    compute_full_approx : bool, optional
        Compute the complete singular value decomposition of subsampled matrices when `approximate_sigma=True`. 
        Useful for pre-computing singular values computed by `get_plotting_data()` by saving a repeated SVD.
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
        Compute SVD using any of the full, exact decompositions from the 'torch' or 'dask' backend, 
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
    backend : {'scipy', 'torch', 'dask'}, optional
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
    
    def __init__(self, variance_estimator = 'poisson', q=0, qits=21,
                    approximate_sigma = True, keep_aspect=False, read_counts = None,
                    default_shrinker = 'frobenius', sinkhorn_tol = 1e-6, n_iter = 500, 
                    n_components = None, exact = True,
                    conserve_memory=False, logger = None, verbose=1, suppress=True,
                    subsample_size = 2000, backend = 'torch',
                    svd_backend=None,sinkhorn_backend=None, **kwargs):
        """Summary
        
        Parameters
        ----------
        variance_estimator : str, optional
            Description
        q : int, optional
            Description
        qits : int, optional
            Description
        approximate_sigma : bool, optional
            Description
        keep_aspect : bool, optional
            Description
        default_shrinker : str, optional
            Description
        sinkhorn_tol : float, optional
            Description
        n_iter : int, optional
            Description
        n_components : None, optional
            Description
        exact : bool, optional
            Description
        conserve_memory : bool, optional
            Description
        logger : None, optional
            Description
        verbose : int, optional
            Description
        suppress : bool, optional
            Description
        subsample_size : int, optional
            Description
        backend : str, optional
            Description
        svd_backend : None, optional
            Description
        sinkhorn_backend : None, optional
            Description
        **kwargs
            Description
        """
        #build the logger first to share across all subprocedures
        super().__init__(conserve_memory, logger, verbose, suppress,**kwargs)
        #initialize the subprocedure classes
        self.k = n_components
        self.sinkhorn_tol = sinkhorn_tol
        self.default_shrinker=default_shrinker
        self.n_iter = n_iter
        self.exact = exact
        self.variance_estimator = variance_estimator
        self.approximate_sigma = approximate_sigma
        self.subsample_size = subsample_size
        self.q = q
        self.qits = qits
        self.backend = backend
        self.svd_backend = svd_backend
        self.sinkhorn_backend = sinkhorn_backend
        self.keep_aspect=keep_aspect
        self.read_counts = read_counts
        self.reset_subsample()
        self.reset_plotting_data()
        #remove the kwargs that have been assigned by super.__init__()
        self._X = None

        #hotfix to remove tol collisions
        self.svdkwargs = kwargs

        self.sinkhorn_kwargs = kwargs.copy()
        if 'tol' in kwargs:
            del self.sinkhorn_kwargs['tol']

        self.sinkhorn = Sinkhorn(tol = sinkhorn_tol, n_iter = n_iter, q=self.q, read_counts=self.read_counts,
                                variance_estimator = variance_estimator, 
                                relative = self, backend=self.sinkhorn_backend,
                                conserve_memory = self.conserve_memory, suppress=suppress,
                                **self.sinkhorn_kwargs)
        
        self.svd = SVD(n_components = self.k, exact=self.exact, 
                    backend = self.svd_backend, relative = self, 
                    conserve_memory = self.conserve_memory, suppress=suppress)

        self.shrinker = Shrinker(default_shrinker=self.default_shrinker, rescale_svs = True, relative = self,suppress=suppress)


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
                q=self.q, variance_estimator = self.variance_estimator, 
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
            self._svd = SVD(n_components = self.k, exact=self.exact, backend = self.svd_backend, relative = self)
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
            self._shrinker = Shrinker(default_shrinker=self.default_shrinker, rescale_svs = True, relative = self)
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
            if self._istransposed:
                return self.sinkhorn.transform(X.T).T
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


    @property
    def subsample_sinkhorn(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if not hasattr(self, '_subsample_sinkhorn') or self._subsample_sinkhorn is None:
            self._subsample_sinkhorn = Sinkhorn(read_counts=self.read_counts,tol = self.sinkhorn_tol, n_iter = self.n_iter, q = self.q,
             variance_estimator = self.variance_estimator, backend = self.sinkhorn_backend, relative = self, **self.sinkhorn_kwargs)
        return self._subsample_sinkhorn

    @subsample_sinkhorn.setter
    def subsample_sinkhorn(self,val):
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
            self._subsample_sinkhorn = val
        else:
            raise ValueError("Cannot set subsample_sinkhorn to non-Sinkhorn estimator")

    @property
    def subsample_svd(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if not hasattr(self, '_subsample_svd') or self._subsample_svd is None:
            self._subsample_svd = SVD(exact=self.exact, relative = self, backend=self.svd_backend, **self.svdkwargs)
        return self._subsample_svd
    
    @subsample_svd.setter
    def subsample_svd(self,val):
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
            self._subsample_svd = val
        else:
            raise ValueError("Cannot set subsample_svd to non-SVD estimator")
    
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
                    if self.approximate_sigma:
                        self.k = np.min([200,self.M])
                    else:
                        self.k = np.ceil(self.M/2).astype(int)
                # oom = np.floor(np.log10(np.min(X.shape)))
                # self.k = np.max([int(10**(oom-1)),10])
            self.k = np.min([self.k, *X.shape]) #ensure we are not asking for too many SVs
            self.svd.k = self.k
            if self.variance_estimator == 'poisson':
                self.q, self.sinkhorn = self.fit_variance(X=X)
            M = self.sinkhorn.fit_transform(X)
            self.Z = M

            sigma_estimate = None
            if self.approximate_sigma and self.k != self.M: # if self.k is the minimum dimension, then the user requested a full decomposition.
                sigma_estimate = self.subsample_estimate_sigma(X = X)

            converged = False

            while not converged:
                
                self.svd.fit(M)
                toshrink = self.S_Z
                _, converged = self.shrinker.fit(toshrink, shape = X.shape,sigma=sigma_estimate)
                self._mp_rank = self.shrinker.scaled_mp_rank_
                if not converged:
                    self.k = int(np.min([self.k*1.5, *X.shape]))
                    self.svd.k = self.k
                    self.logger.warning("Full rank partial decomposition detected, fitting with a larger k = {}".format(self.k))
            del M
            del X

            return self

        
    @fitted
    def transform(self,  X = None, unscale=True, shrinker = None, denoised=True, truncate=True):
        """Summary
        
        Parameters
        ----------
        X : array, optional
            If `BiPCA.conserve_memory` is True, then X must be provided in order to obtain 
            the solely biwhitened transform, i.e., for unscale=False, denoised=False.
        unscale : bool, default True
            Unscale the output matrix so that it is in the original input domain.
        shrinker : {'hard','soft', 'frobenius', 'operator','nuclear'}, optional
            Shrinker to use for denoising
            (Defaults to `obj.default_shrinker`)
        denoised : bool, default True
            Return denoised output.
        truncate : bool, default True
            Truncate the transformed data at 0. 
        
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
        if truncate:
            if not denoised: # There's a bug here when Y is a sparse matrix. This only happens when Y is Z
                pass
            else:
                Y = np.where(Y<0, 0,Y)
        if unscale:
            Y = self.unscale(Y)
        if self._istransposed:
            Y = Y.T

        self.Y = Y
        return Y
    def fit_transform(self, X = None, shrinker = None,**kwargs):
        """Summary
        
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

    def subsample(self, X = None, reset = False, subsample_size = None, force_sinkhorn_convergence = True):
        """Summary
        
        Parameters
        ----------
        X : None, optional
            Description
        reset : bool, optional
            Description
        subsample_size : None, optional
            Description
        force_sinkhorn_convergence : bool, optional
            Description
        
        
        Returns
        -------
        TYPE
            Description
        """
        if X is None:
            X = self.X 
        if reset or self.subsample_indices == {}:
            self.reset_subsample()
            if subsample_size is None:
                subsample_size = self.subsample_size
                   


            #the "master" shape
            M,N = X.shape
            if M > N:
                X = X.T
                M,N = X.shape
            aspect_ratio = M/N

            #get the downsampled approximate shape. This can grow and its aspect ratio may shift.
            if subsample_size is None:
                subsample_size = 2000
    
            sub_M = np.min([subsample_size,M])
            sub_N = np.floor(1/aspect_ratio * sub_M).astype(int)
            if sub_N>10000:
                if not self.keep_aspect:
                    sub_N = 5000
            self.subsample_gamma = sub_M/sub_N
            with self.logger.task("identifying a valid {:d} x {:d} submatrix".format(sub_M,sub_N)):

                nixs0 = np.random.choice(np.arange(N),replace=False,size=sub_N)
                mixs0 = np.random.choice(np.arange(M), replace=False, size = sub_M)

                thresh = 1
                xsub,mixs,nixs = stabilize_matrix(self.subsample_sinkhorn.estimate_variance(X[mixs0,:][:,nixs0])[0],threshold = thresh)
                nixs0 = nixs0[nixs]
                mixs0 = mixs0[mixs]
                if force_sinkhorn_convergence:
                    sinkhorn_estimator = self.subsample_sinkhorn
                    it = 0.05
                    while not sinkhorn_estimator.converged:
                        try:
                            msub = sinkhorn_estimator.fit(X[mixs0,:][:,nixs0])
                        except Exception as e:
                            print(e)
                            nixs0 = np.random.choice(np.arange(N),replace=False,size=sub_N)

                            mixs0 = np.random.choice(np.arange(M), replace=False, size = sub_M)
                            thresh *= 2
                            xsub, mixs,nixs = stabilize_matrix(self.subsample_sinkhorn.estimate_variance(X[mixs0,:][:,nixs0])[0],threshold=thresh)
                            nixs0 = nixs0[nixs]
                            mixs0 = mixs0[mixs]

                self.subsample_gamma = xsub.shape[0]/xsub.shape[1]
                self.subsample_indices['rows'] = mixs0
                self.subsample_indices['cols'] = nixs0
                # self.subsample_indices['permutation'] = np.unravel_index(np.random.permutation(sub_M*sub_N).reshape((sub_M,sub_N)))
                self.subsample_N = len(nixs0)
                self.subsample_M = len(mixs0)
                

        mixs = self.subsample_indices['rows']
        nixs = self.subsample_indices['cols']

        return X[mixs,:][:,nixs]
    def reset_subsample(self):
        """Reset all of the subsample attributes to None.
        """
        self.subsample_N = None
        self.subsample_M = None
        self.subsample_gamma = None
        self.subsample_indices = {}
        self._subsample_sinkhorn = None
        self._subsample_spectrum = {'X': None,
                                    'Y': None, 
                                    'Y_normalized': None,
                                    'Y_permuted': None}
        self._subsample_sinkhorn =  None
        self._subsample_svd = None
    def reset_plotting_data(self):
        """Reset the dictionary of plotting spectra.
        """
        self._plotting_spectrum = {}

    def compute_subsample_spectrum(self, X=None, M = 'Y', k = None, reset = False):
        """Compute and set the subsampled spectrum
        
        Returns
        -------
        TYPE
            Description
        
        Parameters
        ----------
        X : None, optional
            Description
        M : str, optional
            Description
        k : None, optional
            Description
        reset : bool, optional
            Description
        
        Raises
        ------
        ValueError
            Description
        """
        if M not in ['X', 'Y', 'Y_normalized']:
            raise ValueError("Invalid matrix requested. M must be in ['X','Y','Y_normalized']")
        
        if reset:
            self.reset_subsample()

        if M in self._subsample_spectrum.keys():
            if self._subsample_spectrum[M] is not None:
                return self._subsample_spectrum[M]
        xsub = self.subsample(X=X)
        self.subsample_svd.k = k
        if self.subsample_svd.k in [None, 0]:
            self.subsample_svd.k = self.subsample_M
        k = self.subsample_svd.k

        with self.logger.task("Computing subsampled spectrum of {}".format(M)):
                if M == 'Y_normalized':
                    if self._subsample_spectrum['Y'] is not None:
                        self._subsample_spectrum[M] = (self._subsample_spectrum['Y']/(self.shrinker.sigma)) # collect everything and store it
                    else:
                        try:
                            msub = self.subsample_sinkhorn.transform(xsub)
                        except:
                            msub = self.subsample_sinkhorn.fit_transform(xsub)


                        self.subsample_svd.fit(msub/self.shrinker.sigma) 
                        self._subsample_spectrum['Y_normalized'] = self.subsample_svd.S
                        self._subsample_spectrum[M] = (self._subsample_spectrum['Y_normalized']) # collect everything and store it
                if M == 'Y':
                    try:

                        msub = self.subsample_sinkhorn.transform(xsub)
                    except:
                        msub = self.subsample_sinkhorn.fit_transform(xsub)

                    self.subsample_svd.fit(msub) 
                    self._subsample_spectrum['Y'] = self.subsample_svd.S
                if M == 'X':

                    self.subsample_svd.fit(xsub)
                    self._subsample_spectrum['X'] =  self.subsample_svd.S
        return self._subsample_spectrum[M]

    def subsample_estimate_sigma(self,X = None):
        """Estimate the noise variance for the model using a subsample of the input matrix.
        
        Returns
        -------
        sigma_estimate : float
            An estimate of the standard deviation of the noise in the matrix
        
        Other Parameters
        ----------------
        X : array-like, optional
            Not implemented. The matrix to subsample and estimate the noise variance of
        """
        xsub = self.subsample(X=X)
        sub_M, sub_N = xsub.shape

        with self.logger.task("noise variance approximation by subsampling"):
             # we will compute the whole deocmposition so that we can use it later for plotting MP fit.
            self.subsample_svd.k = sub_M

            S = self.compute_subsample_spectrum(M = 'Y', k = self.subsample_svd.k,X=X)
            if xsub.shape[0]>xsub.shape[1]:
                xsub = xsub.T
            self.shrinker.fit(S,shape = xsub.shape)
            sigma_estimate = self.shrinker.sigma_
            totest = self.shrinker.scaled_cov_eigs 
            MP = MarcenkoPastur(gamma = xsub.shape[0]/xsub.shape[1])
            self.kst = KS(totest, MP)
            self.logger.info("Subsampled KS test = " + str(self.kst))
        return sigma_estimate

    @property
    def plotting_spectrum(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if not hasattr(self, '_plotting_spectrum') or self._plotting_spectrum is None:
            self.get_plotting_data()
        return self._plotting_spectrum
    
    def get_plotting_data(self,  subsample = None, reset = False, X = None):
        """
        Return (and compute, if necessary) the eigenvalues of the covariance matrices associated with 1) the unscaled data and 2) the biscaled, normalized data.
        
        Parameters
        ----------
        subsample : bool, optional
            Compute the covariance eigenvalues over a subset of the data
            Defaults to `obj.approximate_sigma`, which in turn is default `True`.
        X : None, optional
            Description
        cov_eigs : bool, optional
            Compute the eigenvalues of the M x M covariance matrix, rather than the singular values of the M x N X.
            Default False.
        
        Returns
        -------
        TYPE
            Description
        """
        if subsample is None:
            subsample = self.approximate_sigma
        if reset:
            self.reset_plotting_data()

        if self._plotting_spectrum == {}:
            if X is None:
                X = self.X

            if X.shape[1] <= 2000: #if the matrix is sufficiently small, we want to just compute the decomposition on the whole thing. 
                subsample = False
            if not subsample:
                if len(self.S_Z)<=self.M*0.95: #we haven't computed more than 95% of the svs, so we're going to recompute them
                    self.svd.k = self.M
                    M = self.sinkhorn.fit_transform(X)
                    self.svd.fit(M)
                if not attr_exists_not_none(self,"S_X"):    
                    svd = SVD(n_components = len(self.S_Z),backend=self.svd_backend, exact= self.exact,relative = self, **self.svdkwargs)
                    svd.fit(X)
                    self.S_X = svd.S
                self._plotting_spectrum['X'] = (self.S_X / np.sqrt(self.N))**2
                self._plotting_spectrum['Y'] = (self.S_Z/ np.sqrt(self.N))**2
                self._plotting_spectrum['Y_normalized'] = (self.S_Z/(self.shrinker.sigma * np.sqrt(self.N)))**2
                self._plotting_spectrum['shape'] = X.shape
            else:
                xsub = self.subsample(X=X)
                self._plotting_spectrum['Y'] = (self.compute_subsample_spectrum(X=X,M = 'Y', k = self.subsample_M) / np.sqrt(self.subsample_N))**2
                self._plotting_spectrum['Y_normalized'] = (self.compute_subsample_spectrum(X=X,M ='Y_normalized', k = self.subsample_M)/ np.sqrt(self.subsample_N))**2
                self._plotting_spectrum['X'] = (self.compute_subsample_spectrum(X=X,M = 'X', k = self.subsample_M) / np.sqrt(self.subsample_N))**2
                self._plotting_spectrum['shape'] = xsub.shape
        return self._plotting_spectrum

    def fit_variance(self, X = None):
        """Fit the quadratic variance parameter for Poisson variance estimator using a subsample of the data.
        
        Returns
        -------
        TYPE
            Description
        
        Parameters
        ----------
        X : None, optional
            Description
        """
        if self.qits>1:
            if X is None:
                X = self.X
            with self.logger.task('variance estimator q'):
                if self.approximate_sigma is False or self.subsample_size >= np.min(X.shape):
                    subsampled=False
                    xsub = X
                else:
                    subsampled = True
                    xsub = self.subsample(X=X)

                if xsub.shape[1]<xsub.shape[0]:
                    xsub = xsub.T
                q_grid = np.round(np.linspace(0.0,1.0,self.qits),2)
                bestq = 0
                bestqval = 10000000
                bestvd  = 0
                MP = MarcenkoPastur(gamma = xsub.shape[0]/xsub.shape[1])
                for q in q_grid:
                    sinkhorn = Sinkhorn(tol = self.sinkhorn_tol, backend=self.sinkhorn_backend, n_iter = self.n_iter, variance_estimator = "poisson", q = q,
                                        **self.sinkhorn_kwargs,relative=self,verbose = self.verbose)
                    try:
                        m = sinkhorn.fit_transform(xsub)
                    except Exception as e:
                        print(e)
                        continue
                    svd = SVD(k = np.min(xsub), backend=self.svd_backend, exact = True,relative=self,verbose=self.verbose)
                    svd.fit(m)
                    s = svd.S
                    shrinker = Shrinker(relative=self,verbose=self.verbose)
                    shrinker.fit(s,shape = xsub.shape)
                    totest = shrinker.scaled_cov_eigs
                    kst = KS(totest, MP)
                    if bestqval<kst:
                        break
                    if bestqval-kst>0:
                        bestq=q
                        bestqval = kst
                        self.logger.info("New best K-S score reached for q = " + str(bestq) + ", K-S = " + str(bestqval))
                        bestvd = s

                self.logger.info('Using q = {}'.format(bestq))
                if subsampled:
                    self._subsample_spectrum['Y'] = bestvd
                self.q  = bestq
                self.kst = bestqval
                self.subsample_sinkhorn = Sinkhorn(tol = self.sinkhorn_tol,backend=self.sinkhorn_backend, n_iter = self.n_iter, variance_estimator = "poisson", q = bestq, relative = self,
                                        **self.sinkhorn_kwargs)

                return bestq, Sinkhorn(tol = self.sinkhorn_tol, backend=self.sinkhorn_backend, n_iter = self.n_iter, variance_estimator = "poisson", q = bestq, relative = self,
                                        **self.sinkhorn_kwargs)
        else:
            return self.q,self.sinkhorn


