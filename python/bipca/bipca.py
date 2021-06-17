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
from .math import Sinkhorn, SVD, Shrinker, MarcenkoPastur, KS
from .utils import stabilize_matrix, filter_dict,resample_matrix_safely,nz_along
from .base import *

class BiPCA(BiPCAEstimator):

    """Summary
    
    Attributes
    ----------
    approximate_sigma : TYPE
        Description
    center : TYPE
        Description
    compute_full_approx : TYPE
        Description
    default_shrinker : TYPE
        Description
    exact : TYPE
        Description
    fit_dist : TYPE
        Description
    k : TYPE
        Description
    n_iter : TYPE
        Description
    pca_type : TYPE
        Description
    q : TYPE
        Description
    qits : TYPE
        Description
    refit : TYPE
        Description
    S_X : TYPE
        Description
    S_Y : TYPE
        Description
    shrinker : TYPE
        Description
    sinkhorn : TYPE
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
    subsample_size : TYPE
        Description
    svd : TYPE
        Description
    svdkwargs : TYPE
        Description
    U_Y : TYPE
        Description
    U_Y_ : TYPE
        Description
    V_Y : TYPE
        Description
    V_Y_ : TYPE
        Description
    variance_estimator : TYPE
        Description
    X : TYPE
        Description
    Y : TYPE
        Description
    
    Deleted Attributes
    ------------------
    build_plotting_data : TYPE
        Description
    biscaled_normalized_covariance_eigenvalues : TYPE
        Description
    data_covariance_eigenvalues : TYPE
        Description
    fit_ : bool
        Description
    n_sigma_estimates : TYPE
        Description
    S_Y_ : TYPE
        Description
    """
    
    def __init__(self, center = True, variance_estimator = 'poisson', q=0, qits=5,
                    approximate_sigma = False, compute_full_approx = True,
                    default_shrinker = 'frobenius', sinkhorn_tol = 1e-6, n_iter = 100, 
                    n_components = None, pca_type ='rotate', exact = True,
                    conserve_memory=False, logger = None, verbose=1, suppress=True,
                    subsample_size = None, refit = True, backend = 'scipy', **kwargs):
        """Summary
        
        Parameters
        ----------
        center : bool, optional
            Description
        variance_estimator : str, optional
            Description
        fit_dist : bool, optional
            Description
        qits : int, optional
            Description
        approximate_sigma : bool, optional
            Description
        compute_full_approx : bool, optional
            Description
        default_shrinker : str, optional
            Description
        sinkhorn_tol : float, optional
            Description
        n_iter : int, optional
            Description
        n_components : None, optional
            Description
        pca_type : str, optional
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
        subsample_size : None, optional
            Description
        refit : bool, optional
            Refit annData objects
        backend : {'scipy', 'dask'}, optional
            Computaton engine to use.  Dask is recommended for large problems.
        **kwargs
            Description
        
        Deleted Parameters
        ------------------
        build_plotting_data : bool, optional
            Description
        n_sigma_estimates : int, optional
            Description
        """
        #build the logger first to share across all subprocedures
        super().__init__(conserve_memory, logger, verbose, suppress,**kwargs)
        #initialize the subprocedure classes
        self.center = center
        self.k = n_components
        self.pca_type = pca_type
        self.sinkhorn_tol = sinkhorn_tol
        self.default_shrinker=default_shrinker
        self.n_iter = n_iter
        self.exact = exact
        self.variance_estimator = variance_estimator
        self.approximate_sigma = approximate_sigma
        self.subsample_size = subsample_size
        self.refit = refit
        self.compute_full_approx = compute_full_approx
        self.q = q
        self.qits = qits
        self.backend = backend
        self.reset_subsample()
        self.reset_plotting_data()
        #remove the kwargs that have been assigned by super.__init__()
        self._X = None

        #hotfix to remove tol collisions
        self.svdkwargs = kwargs

        self.sinkhorn_kwargs = kwargs.copy()
        if 'tol' in kwargs:
            del sinkhorn_kwargs['tol']

        self.sinkhorn = Sinkhorn(tol = sinkhorn_tol, n_iter = n_iter, q=self.q, variance_estimator = variance_estimator, relative = self, backend=self.backend,
                                **self.sinkhorn_kwargs)
        
        self.svd = SVD(n_components = n_components, exact=exact, relative = self, **kwargs)

        self.shrinker = Shrinker(default_shrinker=default_shrinker, rescale_svs = True, relative = self,**kwargs)

    @property
    def scaled_svd(self):
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
        if hasattr(self,'_denoised_svd'):
            return self._denoised_svd
        else:
            if hasattr(self,'_mp_rank'):
                self._denoised_svd = SVD(n_components = self.mp_rank, exact=self.exact, logger=self.logger, conserve_memory=self.conserve_memory)
            else:
                raise RuntimeError("Scaled SVD is only feasible after the marcenko pastur rank is known.")
        return self._denoised_svd

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
    def S_denoised(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if hasattr(self,'_denoised_svd'):
            S = self.scaled_svd.S
        return S
    @fitted_property
    def U_denoised(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if hasattr(self,'_denoised_svd'):
            U = self.scaled_svd.U
            if self._istransposed:
                U = self.scaled_svd.V
            return U

    @fitted_property
    def V_denoised(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if hasattr(self,'_denoised_svd'):
            V = self.scaled_svd.V
            if self._istransposed:
                V = self.scaled_svd.U
            return V

    @fitted_property
    def U_Y(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        U = self.U_Y_
        if self._istransposed:
            U = self.V_Y_
        return U
    @U_Y.setter
    @stores_to_ann(target='obsm')
    def U_Y(self,val):
        """Summary
        
        Parameters
        ----------
        val : TYPE
            Description
        """
        self.U_Y_ = val

    @fitted_property
    def S_Y(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self._S_Y
    @S_Y.setter
    @stores_to_ann(target='uns')
    def S_Y(self,val):
        """Summary
        
        Parameters
        ----------
        val : TYPE
            Description
        """
        self._S_Y = val

    @fitted_property
    def V_Y(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """

        V = self.V_Y_
        if self._istransposed:
            V = self.U_Y_
        return V.T
    @V_Y.setter
    @stores_to_ann(target='varm')
    def V_Y(self,val):
        """Summary
        
        Parameters
        ----------
        val : TYPE
            Description
        """
        self.V_Y_ = val
   
           
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
        if self._istransposed:
            X = X.T
        return self.sinkhorn.unscale(X)
    @property
    def right_scaler(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if self._istransposed:
            return self.sinkhorn.left
        else:
            return self.sinkhorn.right
    @property
    def left_scaler(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if self._istransposed:
            return self.sinkhorn.right
        else:
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
            self._subsample_sinkhorn = Sinkhorn(tol = self.sinkhorn_tol, n_iter = self.n_iter, q = self.q, variance_estimator = self.variance_estimator, relative = self, **self.sinkhorn_kwargs)
        return self._subsample_sinkhorn

    @subsample_sinkhorn.setter
    def subsample_sinkhorn(self,val):
        self._subsample_sinkhorn = val

    @property
    def subsample_svd(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if not hasattr(self, '_subsample_svd') or self._subsample_svd is None:
            self._subsample_svd = SVD(exact=self.exact, relative = self,  **self.svdkwargs)
        return self._subsample_svd
    

    
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
            self.X = A
            X = A
            if self.k is None or self.k == 0 and self.M>=2000: #automatic k selection
                    if self.approximate_sigma:
                        self.k = np.min([500,self.M])
                    else:
                        self.k = np.ceil(self.M/2).astype(int)
                # oom = np.floor(np.log10(np.min(X.shape)))
                # self.k = np.max([int(10**(oom-1)),10])
            self.k = np.min([self.k, *X.shape]) #ensure we are not asking for too many SVs
            self.svd.k = self.k

            if self.variance_estimator == 'poisson':
                q, self.sinkhorn = self.fit_variance()
            M = self.sinkhorn.fit_transform(X)
            self._Z = M

            sigma_estimate = None
            if self.approximate_sigma and self.M>=2000 and self.k != self.M: # if self.k is the minimum dimension, then the user requested a full decomposition.
                sigma_estimate = self.subsample_estimate_sigma(X, compute_full = self.compute_full_approx)

            # if self.mean_rescale:
            converged = False
            while not converged:
                self.svd.fit(M)
                self.U_Y = self.svd.U
                self.S_Y = self.svd.S
                self.V_Y = self.svd.V
                toshrink = self.A if isinstance(A, AnnData) else self.S_Y
                _, converged = self.shrinker.fit(toshrink, shape = X.shape,sigma=sigma_estimate)
                self._mp_rank = self.shrinker.scaled_mp_rank_
                if not converged:
                    self.k = int(np.min([self.k*1.5, *X.shape]))
                    self.svd.k = self.k
                    self.logger.warning("Full rank partial decomposition detected, fitting with a larger k = {}".format(self.k))
            return self

        
    @fitted
    def transform(self, unscale=True, shrinker = None):
        """Summary
        
        Parameters
        ----------
        shrinker : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        sshrunk = self.shrinker.transform(self.S, shrinker=shrinker)
        Y = (self.U[:,:self.mp_rank]*sshrunk[:self.mp_rank])@self.V[:,:self.mp_rank].T
        if unscale:
            Y = self.unscale(Y)
            self.Y = Y
        if self._istransposed:
            Y = Y.T
        return Y
    def fit_transform(self, X = None, shrinker = None):
        """Summary
        
        Parameters
        ----------
        X : None, optional
            Description
        shrinker : None, optional
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
        return self.transform(shrinker=shrinker)

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
        
        Deleted Parameters
        ------------------
        sinkhorn_stable : bool, optional
            Repeat subnsampling until Sinkhorn converges
        
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
                subsample_size = 1000
    
            sub_M = np.min([subsample_size,M])
            sub_N = np.floor(1/aspect_ratio * sub_M).astype(int)
            self.subsample_gamma = sub_M/sub_N
            with self.logger.task("identifying a valid {:d} x {:d} submatrix".format(sub_M,sub_N)):

                #compute some probability distributions to sample from
                # col_density = nz_along(X,axis=0)

                ## get the width of the distribution that we need minimum
                #order = col_density.argsort()
                #ranks = order.argsort()
                #the cols in the middle of the distribution
                # cols = np.nonzero((ranks>=(sub_N*0.9)/2) * (ranks<=(N+sub_N*1.1)/2))[0]
                nixs0 = np.random.choice(np.arange(N),replace=False,size=sub_N)

                #now preferentially sample genes that are dense in this region
                # rows_in_col_density = nz_along(X,axis=1)
                # pdist = rows_in_col_density/rows_in_col_density.sum()
                mixs0 = np.random.choice(np.arange(M), replace=False, size = sub_M)
                thresh = 1
                xsub,mixs,nixs = stabilize_matrix(X[mixs0,:][:,nixs0],threshold = thresh)
                nixs0 = nixs0[nixs]
                mixs0 = mixs0[mixs]
                if force_sinkhorn_convergence:
                    sinkhorn_estimator = self.subsample_sinkhorn
                    it = 0.05
                    while not sinkhorn_estimator.converged:
                        try:
                            msub = sinkhorn_estimator.fit(X[mixs0,:][:,nixs0])
                        except:
                            #resample again,slide the distribution up
                            # it *= 2
                            # cols = np.nonzero((ranks>=(sub_N*(0.9+it))/2) * (ranks<=(N+sub_N*(1.1+it))/2))[0]

                            # nixs = np.random.choice(cols,replace=False,size=sub_N)

                            # rows_in_col_density = nz_along(X,axis=1)
                            # pdist = rows_in_col_density/rows_in_col_density.sum()
                            nixs0 = np.random.choice(np.arange(N),replace=False,size=sub_N)

                            mixs0 = np.random.choice(np.arange(M), replace=False, size = sub_M)
                            thresh *= 2
                            xsub, mixs,nixs = stabilize_matrix(X[mixs0,:][:,nixs0],threshold=thresh)
                            nixs0 = nixs0[nixs]
                            mixs0 = mixs0[mixs]
                            
                self.subsample_gamma = xsub.shape[0]/xsub.shape[1]
                self.subsample_indices['rows'] = mixs0
                self.subsample_indices['cols'] = nixs0
                # self.subsample_indices['permutation'] = np.unravel_index(np.random.permutation(sub_M*sub_N).reshape((sub_M,sub_N)))
                self.subsample_N = sub_N
                self.subsample_M = sub_M
                

        mixs = self.subsample_indices['rows']
        nixs = self.subsample_indices['cols']

        return X[mixs,:][:,nixs]
    def reset_subsample(self):
        """Summary
        """
        self.subsample_N = None
        self.subsample_M = None
        self.subsample_gamma = None
        self.subsample_indices = {}
        self._subsample_svd = None
        self._subsample_sinkhorn = None
        self._subsample_spectrum = {'X': None,
                                    'Y': None, 
                                    'Y_normalized': None,
                                    'Y_permuted': None}
        self._subsample_sinkhorn = None
        self._subsample_svd = None
    def reset_plotting_data(self):
        """Summary
        """
        self._plotting_spectrum = {}

    def compute_subsample_spectrum(self, M = 'Y', k = None):
        """Compute and set the subsampled spectrum
        
        Returns
        -------
        TYPE
            Description
        
        Parameters
        ----------
        M : str, optional
            Description
        k : None, optional
            Description
        
        Raises
        ------
        ValueError
            Description
        """
        if M not in ['X', 'Y', 'Y_normalized']:
            raise ValueError("Invalid matrix requested. M must be in ['X','Y','Y_normalized']")
        
        if M in self._subsample_spectrum.keys():
            if self._subsample_spectrum[M] is not None:
                return self._subsample_spectrum[M]
        xsub = self.subsample()

        if k is None:
            if self.subsample_svd.k in [None, 0]:
                self.subsample_svd.k = self.subsample_M
            k = self.subsample_svd.k

        self.subsample_svd.k = k
        with self.logger.task("Computing subsampled spectrum of {}".format(M)):
                if M == 'Y_normalized':
                    if self._subsample_spectrum['Y'] is not None and len(self._subsample_spectrum['Y'])>=k: # we have a spectrum for Y and we have enough svs for what was requested
                        if not hasattr(self.shrinker, 'sigma'): #we don't have a sigma
                            self.shrinker.fit(self.subsample_spectrum['Y'], shape = (self.subsample_M, self.subsample_N))
                        
                    else:
                        #we either don't have a spectrum for Y or we don't have enough svs.
                        try:
                            msub = self.subsample_sinkhorn.transform(xsub)
                        except:
                            msub = self.subsample_sinkhorn.fit_transform(xsub)

                        if sparse.issparse(msub):
                            msub = msub.toarray()
                        self.subsample_svd.fit(msub) 
                        self._subsample_spectrum['Y'] = self.subsample_svd.S
                        self.shrinker.fit(self._subsample_spectrum['Y'], shape = (self.subsample_M, self.subsample_N)) # compute the sigma

                    self._subsample_spectrum[M] = (self._subsample_spectrum['Y']/(self.shrinker.sigma_)) # collect everything and store it

                if M == 'Y':
                    try:
                        msub = self.subsample_sinkhorn.transform(xsub)
                    except:
                        msub = self.subsample_sinkhorn.fit_transform(xsub)

                    if sparse.issparse(msub):
                        msub = msub.toarray()
                    self.subsample_svd.fit(msub) 
                    self._subsample_spectrum['Y'] = self.subsample_svd.S
                if M == 'X':
                    if sparse.issparse(xsub):
                        xsub = xsub.toarray()
                    self.subsample_svd.fit(xsub)
                    self._subsample_spectrum['X'] =  self.subsample_svd.S

                if M == 'Y_permuted':
                    try:
                        msub = self.subsample_sinkhorn.transform(xsub)
                    except:
                        msub = self.subsample_sinkhorn.fit_transform(xsub)
                    msub = msub[self.subsample_indices['permutation']]
                    self.subsample_svd.fit(msub)
                    self._subsample_spectrum['Y_permuted'] = self.subsample_svd.S

        return self._subsample_spectrum[M]

    def subsample_estimate_sigma(self, X = None, compute_full=True):
        """Summary
        
        Parameters
        ----------
        X : None, optional
            Description
        compute_full : bool, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Deleted Parameters
        ------------------
        subsample_size : None, optional
            Description
        store_svs : bool, optional
            Description
        """
        xsub = self.subsample()
        sub_M, sub_N = xsub.shape

        with self.logger.task("noise variance approximation by subsampling"):
            if compute_full: # we will compute the whole deocmposition so that we can use it later for plotting MP fit.
                self.subsample_svd.k = sub_M
            else:
                self.subsample_svd.k = np.ceil(sub_M/2)
        S = self.compute_subsample_spectrum(M = 'Y', k = self.subsample_svd.k)
        self.shrinker.fit(S,shape = xsub.shape)
        sigma_estimate = self.shrinker.sigma_
        return sigma_estimate

    def PCA(self,shrinker = None, pca_type = None):
        """
        Project the denoised data onto its Marcenko Pastur rank column space. Provides dimensionality reduction.
        If pca-type is 'traditional', traditional PCA is performed on the full denoised data.  
        The resulting PCs have the traditional interpretation as directions of maximal variance. 
        This approach suffers requires a singular value decomposition on a (possibly large) dense matrix. 
        
        If pca-type is 'rotate', the full denoised data is projected onto its column space using QR orthogonalization of the half-rescaled right bistochastic principal components. 
        
        Parameters
        ----------
        shrinker : None, optional
            Description
        pca_type : {'traditional', 'rotate'}, optional
            The type of PCA to perform. 
        
        Returns
        -------
        numpy.array
            Description
        """
        check_is_fitted(self)
        if pca_type is None:
            pca_type = self.pca_type
        with self.logger.task("Scaled domain PCs"):
            Y = self.transform(shrinker = shrinker)#need logic here to prevent redundant calls onto SVD and .transform()
            if pca_type == 'traditional':
                YY = self.scaled_svd.fit(Y)
                PCs = self.U_denoised[:,:self.mp_rank]*self.S_denoised[:self.mp_rank]
            elif pca_type == 'rotate':
                #project  the data onto the columnspace
                rot = 1/self.right_scaler[:,None]*self.V_Y[:,:self.mp_rank]
                PCs = scipy.linalg.qr_multiply(rot, Y)[0]
        return PCs


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
            Description
        reset : bool, optional
            Description
        X : None, optional
            Description
        
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
                if len(self.S_Y)<=self.M*0.95: #we haven't computed more than 95% of the svs, so we're going to recompute them
                    self.svd.k = self.M
                    M = self.sinkhorn.fit_transform(X)
                    self.svd.fit(M)
                    self.S_Y = self.svd.S
                if not hasattr(self, 'S_X') or self.S_X is None:    
                    svd = SVD(n_components = len(self.S_Y), exact= self.exact,relative = self, **self.svdkwargs)
                    svd.fit(X)
                    self.S_X = svd.S
                self._plotting_spectrum['X'] = (self.S_X / np.sqrt(self.N))**2
                self._plotting_spectrum['Y'] = (self.S_Y/ np.sqrt(self.N))**2
                self._plotting_spectrum['Y_normalized'] = (self.S_Y/(self.shrinker.sigma * np.sqrt(self.N)))**2
                self._plotting_spectrum['shape'] = X.shape
            else:
                xsub = self.subsample()
                self._plotting_spectrum['Y'] = (self.compute_subsample_spectrum(M = 'Y', k = self.subsample_M) / np.sqrt(self.subsample_N))**2
                self._plotting_spectrum['Y_normalized'] = (self.compute_subsample_spectrum(M ='Y_normalized', k = self.subsample_M)/ np.sqrt(self.subsample_N))**2
                self._plotting_spectrum['X'] = (self.compute_subsample_spectrum(M = 'X', k = self.subsample_M) / np.sqrt(self.subsample_N))**2
                self._plotting_spectrum['shape'] = xsub.shape
        return self._plotting_spectrum

    def fit_variance(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if self.qits>1:
            if np.min(self.X.shape)<2000:
                subsampled=False
                xsub = self.X
            else:
                subsampled = True
                xsub = self.subsample()
            q_grid = np.round(np.linspace(0.0,1.0,self.qits),2)
            bestq = 0
            bestqval = 10000000
            bestvd  = 0
            MP = MarcenkoPastur(gamma = xsub.shape[0]/xsub.shape[1])
            for q in q_grid:
                sinkhorn = Sinkhorn(tol = self.sinkhorn_tol, n_iter = self.n_iter, variance_estimator = "poisson", q = q, verbose=0,
                                    **self.sinkhorn_kwargs)
                m = sinkhorn.fit_transform(xsub)
                if sparse.issparse(m):
                    m = m.toarray()
                svd = SVD(k = np.min(xsub), exact = True,verbose=0)
                svd.fit(m)
                s = svd.S
                shrinker = Shrinker(verbose=1)
                shrinker.fit(s,shape = xsub.shape)
                totest = shrinker.scaled_cov_eigs#*shrinker.sigma**2
                # totest = totest[totest<=MP.b]
                # totest = totest[MP.a<=totest]
                # kst = kstest(totest, MP.cdf, mode='exact')
                kst = KS(totest, MP)
                if bestqval-kst>1e-5:
                    bestq=q
                    bestqval = kst
                    print(kst)
                    bestsinkhorn = sinkhorn
                    bestvd = s
            print('q = {}'.format(bestq))
            self.subsample_sinkhorn = Sinkhorn(tol = self.sinkhorn_tol, n_iter = self.n_iter, variance_estimator = "poisson", q = bestq, relative = self,
                                    **self.sinkhorn_kwargs)
            if subsampled:
                self._subsample_spectrum['Y'] = bestvd
            self.q  = bestq
            return bestq, Sinkhorn(tol = self.sinkhorn_tol, n_iter = self.n_iter, variance_estimator = "poisson", q = bestq, relative = self,
                                    **self.sinkhorn_kwargs)
        else:
            return self.q,self.sinkhorn

