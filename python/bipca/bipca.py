"""BiPCA: Stochastic Principal Component Analysis
"""
import numpy as np
import sklearn as sklearn
import scipy as scipy
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import scipy.sparse as sparse
import tasklogger

from .math import Sinkhorn, SVD, Shrinker

class BiPCA(BaseEstimator):
    __log_instance = 0
    def __init__(self, variance_estimator = 'binomial', sigma_estimate = 'shuffle',
                    default_shrinker = 'frobenius', tol = 1e-6, n_iter = 100, 
                    n_components = None, scaled_PCA='full_scaled', exact = True, conserve_memory= True, 
                    verbose = 1, logger = None):
        #build the logger first to share across all subprocedures
        self.verbose = verbose
        if logger == None:
            BiPCA.__log_instance += 1
            self.logger = tasklogger.TaskLogger(name='BiPCA ' 
                + str(BiPCA.__log_instance),level = verbose)
        else:
            self.logger = logger
        #initialize the subprocedure classes
        self.k = n_components
        self.pca_type = pca_type
        self.tol = tol
        self.default_shrinker=default_shrinker
        self.n_iter = n_iter
        self.exact = exact
        self.conserve_memory=conserve_memory
        self.variance_estimator = variance_estimator
        self.sigma_estimate = sigma_estimate
        self.sinkhorn = Sinkhorn(tol = tol, n_iter = n_iter, verbose = verbose, logger = self.logger,conserve_memory=conserve_memory, variance_estimator = variance_estimator)
        self.svd = SVD(n_components = n_components, exact=exact, logger=self.logger, conserve_memory=conserve_memory)
        self.shrinker = Shrinker(default_shrinker=default_shrinker, 
                                    rescale_svs = True, verbose=verbose, logger = self.logger)

    @property
    def scaled_svd(self):
        if hasattr(self,'_scaled_svd'):
            return self._scaled_svd
        else:
            if hasattr(self,'_mp_rank'):
                self._scaled_svd = SVD(n_components = self.mp_rank, exact=self.exact, logger=self.logger, conserve_memory=self.conserve_memory)
            else:
                raise RuntimeError("Scaled SVD is only feasible after the marcenko pastur rank is known.")
        return self._scaled_svd

    @property
    def mp_rank(self):
        return self._mp_rank
    
    @property
    def n_components(self):
        """Convenience function for :attr:`k`
        """
        return self.k
        
    @n_components.setter
    def n_components(self,val):
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
        self._k = k
    @property
    def mp_rank(self):
        check_is_fitted(self)
        return self._mp_rank
    @property
    def U(self):
        check_is_fitted(self)
        if hasattr(self,'_scaled_svd'):
            U = self.U_scaled
        else:
            U = self.U_mp
        return U
    @property
    def S(self):
        check_is_fitted(self)
        if hasattr(self,'_scaled_svd'):
            S = self.S_scaled
        else:
            S = self.S_mp
        return S
    @property
    def V(self):
        check_is_fitted(self)
        if hasattr(self,'_scaled_svd'):
            V = self.V_scaled
        else:
            V = self.V_mp
        return V
    @property 
    def S_scaled(self):
        check_is_fitted(self)
        if hasattr(self,'_scaled_svd'):
            self.scaled_svd.S
        return S
    @property
    def U_scaled(self):
        if hasattr(self,'_scaled_svd'):
            U = self.scaled_svd.U
            if self._istransposed:
                U = self.scaled_svd.V
            return U
    @property
    def V_scaled(self):
        if hasattr(self,'_scaled_svd'):
            V = self.scaled_svd.V
            if self._istransposed:
                V = self.scaled_svd.U
            return V
    @property
    def U_mp(self):
        U = self.svd.U
        if self._istransposed:
            U = self.svd.V
        return U
    @property
    def S_mp(self):
        return self.svd.S
    @property
    def V_mp(self):
        V = self.svd.V
        if self._istransposed:
            V = self.svd.U
        return V
    @property
    def Z(self):
        if not self.conserve_memory:
            Z = self._Z
            if self._istransposed:
                Z = Z.T
            return Z
        else:
            raise RuntimeError("Since conserve_memory is true, Z can only be obtained by calling .get_Z(X)")

    @property
    def Y(self):
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
        if not self.conserve_memory:
            self._Y = Y
    
    def get_Z(self,X = None):
        check_is_fitted(self)
        if X is None:
            return self.Z
        else:
            if self._istranposed:
                return self.sinkhorn.transform(X.T).T
            else:
                return self.sinkhorn.transform(X)

    def unscale(self,X):
        if self._istransposed:
            X = X.T
        return self.sinkhorn.unscale(X)

    def fit(self, X):
        #bug: sinkhorn needs to be reset when the model is refit.
        self._M, self._N = X.shape

        if self._M/self._N>1:
            self._istransposed = True
            X = X.T
        else:
            self._istransposed = False
        if self.k is None:
            oom = np.floor(np.log10(np.min(X.shape)))
            self.k = np.max([int(10**(oom-1)),10])
        self.k = np.min([self.k, *X.shape]) #ensure we are not asking for too many SVs
        self.logger.task('BiPCA estimator fit')
        M = self.sinkhorn.fit_transform(X,return_scalers=False)
        self._Z = M

        # if self.sigma_estimate is 'shuffle':# Start shuffling SVDs... need to choose a sensible amount of points and also stabilize the matrix.
        #     aspect_ratio = self._M/self._N

        self.svd.k = self.k
        self.svd.fit(M)
        self.shrinker.fit(self.S, shape = X.shape)
        self._mp_rank = self.shrinker.scaled_mp_rank_
        self.fit_ = True

    def transform(self, shrinker = None):
        check_is_fitted(self)
        sshrunk = self.shrinker.transform(self.S, shrinker=shrinker)
        Y = (self.U[:,:self.mp_rank]*sshrunk[:self.mp_rank])@self.V[:,:self.mp_rank].T
        Y = self.unscale(Y)
        self.Y = Y
        if self._istransposed:
            Y = Y.T
        return Y
    def fit_transform(self, X = None, shrinker = None):
        if X is None:
            check_is_fitted(self)
        else:
            self.fit(X)
        return self.transform(shrinker=shrinker)

    def PCA(self,shrinker = None, pca_type = None, pcs=None):
        check_is_fitted(self)
        if pca_type is None:
            pca_type = self.pca_type
        with self.logger.task("Scaled domain PCs"):
                Y = self.transform(shrinker = shrinker)#need logic here to prevent redundant calls onto SVD and .transform()
            if pca_type == 'full_scaled':
                YY = self.scaled_svd.fit(Y)
                PCs = self.U_scaled[:,:self.mp_rank]*self.S_scaled[:self.mp_rank]
            elif pca_type == 'rotate':
                #project  the data onto the rowspace
                rot = self.sinkhorn.right*self.V_mp[:,:self.mp_rank]
                PCs = scipy.linalg.qr_multiply(rot, Y) 
        return PCs



                #     #should we rescale the rows??
    #     check_is_fitted(self)
    #     sshrunk = self.shrinker.transform(self.S,shrinker=shrinker)
    #     return self._unscale()
