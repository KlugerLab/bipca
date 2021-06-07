"""BiPCA: BiStochastic Principal Component Analysis
"""
import numpy as np
import sklearn as sklearn
import scipy as scipy
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import scipy.sparse as sparse
import tasklogger
from anndata._core.anndata import AnnData
from .math import Sinkhorn, SVD, Shrinker
from .utils import stabilize_matrix, filter_dict,resample_matrix_safely,nz_along
from .base import *

class BiPCA(BiPCAEstimator):

    """Summary
    
    Attributes
    ----------
    approximate_sigma : TYPE
        Description
    approximating_gamma : TYPE
        Description
    biscaled_normalized_covariance_eigenvalues : TYPE
        Description
    build_plotting_data : TYPE
        Description
    center : TYPE
        Description
    data_covariance_eigenvalues : TYPE
        Description
    default_shrinker : TYPE
        Description
    exact : TYPE
        Description
    fit_ : bool
        Description
    k : TYPE
        Description
    n_iter : TYPE
        Description
    n_sigma_estimates : TYPE
        Description
    pca_type : TYPE
        Description
    resample_size : TYPE
        Description
    shrinker : TYPE
        Description
    sinkhorn : TYPE
        Description
    sinkhorn_tol : TYPE
        Description
    svd : TYPE
        Description
    svdkwargs : TYPE
        Description
    variance_estimator : TYPE
        Description
    Y : TYPE
        Description
    """
    
    def __init__(self, center = True, variance_estimator = 'binomial', approximate_sigma = False, n_sigma_estimates = 1,
                    default_shrinker = 'frobenius', sinkhorn_tol = 1e-6, n_iter = 100, 
                    n_components = None, pca_type='traditional',exact = True,
                    conserve_memory=False, logger = None, verbose=1, suppress=True, resample_size = None, refit = True,**kwargs):
        """Summary
        
        Parameters
        ----------
        center : bool, optional
            Description
        variance_estimator : str, optional
            Description
        approximate_sigma : bool, optional
            Description
        n_sigma_estimates : int, optional
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
        build_plotting_data : bool, optional
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
        resample_size : None, optional
            Description
        refit : bool, optional
            Refit annData objects
        **kwargs
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
        self.n_sigma_estimates = n_sigma_estimates
        self.resample_size = resample_size
        self.data_covariance_eigenvalues = None
        self.biscaled_normalized_covariance_eigenvalues = None
        self.refit = refit
        #remove the kwargs that have been assigned by super.__init__()
        self._X = None
        #hotfix to remove tol collisions
        self.svdkwargs = kwargs

        sinkhorn_kwargs = kwargs.copy()
        if 'tol' in kwargs:
            del sinkhorn_kwargs['tol']

        self.sinkhorn = Sinkhorn(tol = sinkhorn_tol, n_iter = n_iter, variance_estimator = variance_estimator, relative = self,
                                **sinkhorn_kwargs)
        
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
    def U_mp(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        U = self.U_mp_
        if self._istransposed:
            U = self.V_mp_
        return U
    @U_mp.setter
    @stores_to_ann
    def U_mp(self,val):
        self.U_mp_ = val

    @fitted_property
    def S_mp(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self.S_mp_
    @S_mp.setter
    @stores_to_ann
    def S_mp(self,val):
        self.S_mp_ = val
    @fitted_property
    def V_mp(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """

        V = self.V_mp_
        if self._istransposed:
            V = self.U_mp_
        return V
    @V_mp.setter
    @stores_to_ann
    def V_mp(self,val):
        self.V_mp_ = val
   
           
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




    def fit(self, A):
        """Summary
        
        Parameters
        ----------
        X : TYPE
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
            if self.k is None:
                oom = np.floor(np.log10(np.min(X.shape)))
                self.k = np.max([int(10**(oom-1)),10])
            self.k = np.min([self.k, *X.shape]) #ensure we are not asking for too many SVs
            self.svd.k = self.k
            M = self.sinkhorn.fit_transform(X)
            self._Z = M

            sigma_estimate = None
            if self.approximate_sigma and X.shape[1]>1000:
                sigma_estimate, self.data_covariance_eigenvalues, self.biscaled_normalized_covariance_eigenvalues = self.shuffle_estimate_sigma(X)

            # if self.mean_rescale:

            self.svd.fit(M)
            self.U_mp = self.svd.U
            self.S_mp = self.svd.S
            self.V_mp = self.svd.V
            toshrink = self.A if isinstance(A, AnnData) else self.S_mp
            self.shrinker.fit(toshrink, shape = X.shape,sigma=sigma_estimate)
            self._mp_rank = self.shrinker.scaled_mp_rank_
            self.fit_ = True
    
    @fitted
    def transform(self, shrinker = None):
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

    def shuffle_estimate_sigma(self,X = None, resample_size = None):
        """Summary
        
        Parameters
        ----------
        X : None, optional
            Description
        resample_size : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        #this needs to be reworked for a flag for computing the pre-svds
        sigma_estimate = 0
        if resample_size is None:
            resample_size = self.resample_size
        if X is None:
            X = self._X        
        data_covariance_eigenvalues = []
        biscaled_normalized_covariance_eigenvalues = []

        #the "master" shape
        M,N = X.shape
        if M > N:
            X = X.T
            M,N = X.shape
        aspect_ratio = M/N

        #get the downsampled approximate shape. This can grow and its aspect ratio may shift.
        if resample_size is None:
            if N <= 5000:
                sub_N = 1000
            elif N>5000:
                sub_N = 5000
        else:
            sub_N = np.min([resample_size,N])
        sub_M = np.floor(aspect_ratio * sub_N).astype(int)
        self.approximating_gamma = sub_M/sub_N

        with self.logger.task("noise variance approximation by subsampling a {:d} x {:d} submatrix".format(sub_M,sub_N)):
            svd_sigma = SVD(exact=self.exact, relative = self, **self.svdkwargs) # the  svd operator we will use to compute the downsampled svds
            for kk in range(self.n_sigma_estimates):
                sinkhorn_estimator = Sinkhorn(tol = self.sinkhorn_tol, n_iter = self.n_iter, variance_estimator = self.variance_estimator, relative = self) #the downsampled biscaler
                while not sinkhorn_estimator.converged:
                    nidx = np.random.permutation(self.N) #grab a random column set
                    nixs = nidx[:sub_N] # the current choice of rows
                    valid_rows = np.arange(X.shape[0])[nz_along(X[:,nixs],axis=1)>5] # these rows make our columns valid
                    mixs = np.random.choice(valid_rows, replace=False, size=sub_M)
                    xsub = X[mixs,:][:,nixs]
                    try:
                        msub = sinkhorn_estimator.fit_transform(xsub)
                    except:
                        #make the resample slightly bigger
                        sub_N = np.min([sub_N + int(sub_N*1.025),self.N])
                        sub_M = np.floor(self.aspect_ratio * sub_N).astype(int)
                        self.approximating_gamma = sub_M/sub_N 

                svd_sigma.k = np.min(msub.shape)    
                svd_sigma.fit(msub)
                S = svd_sigma.S
                biscaled_normalized_covariance_eigenvalues.append(S)  #these will ultimately be adjusted to actually be noise variance adjusted
                svd_sigma.k = np.min(msub.shape) #this just suppressed an annoying printout for sparse matrices. 
                svd_sigma.fit(xsub)
                covS= (svd_sigma.S/np.sqrt(xsub.shape[1]))**2
                data_covariance_eigenvalues.append(covS)

                self.shrinker.fit(S,shape = msub.shape)
                biscaled_normalized_covariance_eigenvalues[-1] = (biscaled_normalized_covariance_eigenvalues[-1]/(np.sqrt(msub.shape[1])*self.shrinker.sigma_))**2
                sigma_estimate += self.shrinker.sigma_/self.n_sigma_estimates

            self.logger.set_level(self.verbose)
        return sigma_estimate, data_covariance_eigenvalues, biscaled_normalized_covariance_eigenvalues

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
                rot = 1/self.right_scaler[:,None]*self.V_mp[:,:self.mp_rank]
                PCs = scipy.linalg.qr_multiply(rot, Y)[0]
        return PCs


    
    def get_histogram_data(self,  X = None):
        """
        Return (and compute, if necessary) the eigenvalues of the covariance matrices associated with 1) the unscaled data and 2) the biscaled, normalized data.
        
        Parameters
        ----------
        X : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        if self.data_covariance_eigenvalues is None:
            if X is None:
                X = self.X
            if len(self.S_mp)>=self.M-1: 
                with self.logger.task("Getting singular values of input data"):
                    svd = SVD(n_components = self.M, exact=self.exact, relative = self, **self.svdkwargs)
                    svd.fit(X)
                    self.data_covariance_eigenvalues = (svd.S / np.sqrt(self.N))**2
                    self.biscaled_normalized_covariance_eigenvalues = (self.S_mp / (np.sqrt(self.N)*self.shrinker.sigma_))**2
                    self.approximating_gamma = self.M/self.N
            else:
                with self.logger.task("Recording pre and post SVs by downsampling"):
                    _, self.data_covariance_eigenvalues, self.biscaled_normalized_covariance_eigenvalues = self.shuffle_estimate_sigma(X)
        return self.data_covariance_eigenvalues, self.biscaled_normalized_covariance_eigenvalues, self.shrinker.sigma_

