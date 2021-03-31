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
    def __init__(self, default_shrinker = 'frobenius', tol = 1e-6, n_iter = 100, 
                    n_components = 1000, exact = True, conserve_memory= True, 
                    verbose = 1, logger = None):
        #build the logger first to share across all subprocedures
        if logger == None:
            BiPCA.__log_instance += 1
            self.logger = tasklogger.TaskLogger(name='BiPCA ' 
                + str(BiPCA.__log_instance),level = verbose)
        else:
            self.logger = logger
        #initialize the subprocedure classes
        self.k = n_components
        self._sinkhorn = Sinkhorn(tol = tol, n_iter = n_iter, verbose = verbose, logger = self.logger)
        self._svd = SVD(n_components = n_components, exact=exact, logger=self.logger)
        self._shrinker = Shrinker(default_shrinker=default_shrinker, 
                                    rescale_svs = True, verbose=verbose, logger = self.logger)
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
        return self._svd.U
    @property
    def S(self):
        return self._svd.S
    @property
    def V(self):
        return self._svd.V
    @property
    def Z(self):
        return self._Z
    def _unscale(self,X):
        return self._sinkhorn.unscale(X)
    def fit(self, X):
        #bug: sinkhorn needs to be reset when the model is refit.
        self.k = np.min([self.k, *X.shape]) #ensure we are not asking for too many SVs
        self._Z = self._sinkhorn.fit_transform(X,return_scalers=False)
        self._svd.k = self.k
        self._svd.fit(self.Z)
        self._shrinker.fit(self.S, shape = X.shape)
        self._mp_rank = self._shrinker.scaled_mp_rank_
        self.fit_ = True
    def transform(self, shrinker = None):
        check_is_fitted(self)
        sshrunk = self._shrinker.transform(self.S, shrinker=shrinker)
        Y = (self.U[:,self.mp_rank]*sshrunk[:,self.mp_rank])@self.V[self.mp_rank,:].T
        Y = self._unscale(Y)
        return Y
    def fit_transform(self, X = None, shrinker = None):
        if X is None:
            check_is_fitted(self)
        else:
            self.fit(X)
        return self.transform(shrinker=shrinker)


