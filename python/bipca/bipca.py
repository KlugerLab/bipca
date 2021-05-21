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
from .utils import stabilize_matrix, filter_dict,resample_matrix_safely,nz_along
from .base import BiPCAEstimator,__memory_conserved__

class BiPCA(BiPCAEstimator):
    def __init__(self, center = True, variance_estimator = 'binomial', sigma_estimate = 'shuffle', n_sigma_estimates = 5,
                    default_shrinker = 'frobenius', sinkhorn_tol = 1e-6, n_iter = 100, 
                    n_components = None, pca_type='full_scaled', build_plotting_data = True, exact = True,
                    conserve_memory=True, logger = None, verbose=1, suppress=True, resample_size = None, **kwargs):
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
        self.sigma_estimate = sigma_estimate
        self.n_sigma_estimates = n_sigma_estimates
        self.build_plotting_data = build_plotting_data
        self.resample_size = resample_size
        self.pre_svs = None
        self.post_svs = None
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
        return self._mp_rank

    @property
    def U(self):
        if hasattr(self,'_scaled_svd'):
            U = self.U_scaled
        else:
            U = self.U_mp
        return U
    @property
    def S(self):
        if hasattr(self,'_scaled_svd'):
            S = self.S_scaled
        else:
            S = self.S_mp
        return S
    @property
    def V(self):
        if hasattr(self,'_scaled_svd'):
            V = self.V_scaled
        else:
            V = self.V_mp
        return V
    @property 
    def S_scaled(self):
        if hasattr(self,'_scaled_svd'):
            S = self.scaled_svd.S
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
    def X(self):
        if not self.conserve_memory:
            X = self._X
            if self._istransposed:
                X = X.T
            return X
        else:
            raise RuntimeError("Since conserve memory is true, X is not stored")
    @X.setter
    def X(self):
        if not self.conserve_memory:
            self._X = X
           
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
            if self._istransposed:
                return self.sinkhorn.transform(X.T).T
            else:
                return self.sinkhorn.transform(X)

    def unscale(self,X):
        if self._istransposed:
            X = X.T
        return self.sinkhorn.unscale(X)
    @property
    def right_scaler(self):
        if self._istransposed:
            return self.sinkhorn.left
        else:
            return self.sinkhorn.right
    @property
    def left_scaler(self):
        if self._istransposed:
            return self.sinkhorn.right
        else:
            return self.sinkhorn.left
    @property
    def aspect_ratio(self):
        if self._istransposed:
            return self._N/self._M
        return self._M/self._N
    @property
    def N(self):
        if self._istransposed:
            return self._M
        return self._N
    @property
    def M(self):
        if self._istransposed:
            return self._N
        return self._M

    def fit(self, X):
        #bug: sinkhorn needs to be reset when the model is refit.
        super().fit()
        with self.logger.task("BiPCA fit"):
            self._M, self._N = X.shape
            if self._M/self._N>1:
                self._istransposed = True
                X = X.T
            else:
                self._istransposed = False
            self._X = X
            if self.k is None:
                oom = np.floor(np.log10(np.min(X.shape)))
                self.k = np.max([int(10**(oom-1)),10])
            self.k = np.min([self.k, *X.shape]) #ensure we are not asking for too many SVs
            self.svd.k = self.k
            M = self.sinkhorn.fit_transform(X,return_scalers=False)[0]
            self._Z = M

            sigma_estimate = None
            if self.sigma_estimate=='shuffle':
                sigma_estimate, self.pre_svs, self.post_svs = self.shuffle_estimate_sigma(M,X,self.build_plotting_data)

            # if self.mean_rescale:

            self.svd.fit(M)
            self.shrinker.fit(self.S, shape = X.shape,sigma=sigma_estimate)
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

    def shuffle_estimate_sigma(self, M, X=None, compute_both=False):

        sigma_estimate = None        
        if compute_both:
            if X is None:
                X = self._X
            pre_svs = []
            post_svs = []
        else:
            pre_svs = None
            post_svs = None
        if M is None:
            M = self._Z
        with self.logger.task("noise variance estimate by submatrix shuffling"):
            if self.resample_size is None:
                if 3000<self.N <=5000:
                    sub_N = 1000
                elif self.N>5000:
                    sub_N = 10000
                else: 
                    sub_N = 100
            else:
                sub_N = self.resample_size
            sub_M = np.floor(self.aspect_ratio * sub_N).astype(int)
            self.approximating_gamma = sub_M/sub_N
            sigma_estimate = 0 
            ##We used to just use the self.svd object for this task, but issues with changing k and resetting the estimator w/ large matrices
            ## broke that.  For now, this hotfix just builds a new svd estimator for the specific task of computing the shuffled SVDs
            ## The old method could be fixed by writing an intelligent reset method for bipca.SVD
            svd_sigma = SVD(n_components = int(np.floor(sub_M/2)+1), exact=False, relative = self, **self.svdkwargs)
            for kk in range(self.n_sigma_estimates):
                print(kk)
                #mixs,nixs = resample_matrix_safely(M,sub_N,seed=kk)
                nidx = np.random.permutation(sub_N)
                nixs = nidx[:sub_N]
                mixs = np.argsort(nz_along(X[:,nixs],axis=1))[::-1][:sub_M]
                xsub = X[mixs,:][:,nixs]
                sinkhorn_estimator = Sinkhorn(tol = self.sinkhorn_tol, n_iter = self.n_iter, variance_estimator = self.variance_estimator, relative = self)
                msub =sinkhorn_estimator.fit_transform(xsub,return_scalers=False)[0]
                svd_sigma.k = np.min(msub.shape)    
                svd_sigma.fit(msub)
                S = svd_sigma.S
                post_svs.append(S)
                svd_sigma.k = np.min(msub.shape)
                svd_sigma.fit(xsub)
                covS= (svd_sigma.S/np.sqrt(xsub.shape[1]))**2
                pre_svs.append(covS)

                self.shrinker.fit(S,shape = msub.shape)
                post_svs[-1] = (post_svs[-1]/(np.sqrt(msub.shape[1])*self.shrinker.sigma_))**2
                sigma_estimate += self.shrinker.sigma_/self.n_sigma_estimates

            self.logger.set_level(self.verbose)
        return sigma_estimate, pre_svs, post_svs

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
                #project  the data onto the columnspace
                rot = 1/self.right_scaler[:,None]*self.V_mp[:,:self.mp_rank]
                PCs = scipy.linalg.qr_multiply(rot, Y)[0]
        return PCs

                #     #should we rescale the rows??
    #     check_is_fitted(self)
    #     sshrunk = self.shrinker.transform(self.S,shrinker=shrinker)
    #     return self._unscale()
    
    def get_histogram_data(self, Z = None, X = None):
        if self.pre_svs is None:
            if X is None:
                X = self._X
            if len(self.S_mp)>=self.M-1: 
                with self.logger.task("Getting singular values of input data"):
                    svd = SVD(n_components = self.M, exact=self.exact, relative = self, **self.svdkwargs)
                    svd.fit(X)
                    self.pre_svs = (svd.S / np.sqrt(self.N))**2
                    self.post_svs = (self.S_mp / (np.sqrt(self.N)*self.shrinker.sigma_))**2
                    self.approximating_gamma = self.M/self.N
            else:
                if Z is None:
                    Z = self._Z
                with self.logger.task("Recording pre and post SVs by downsampling"):
                    _, self.pre_svs, self.post_svs = self.shuffle_estimate_sigma(Z, X = X, compute_both = True)
        return self.pre_svs, self.post_svs, self.shrinker.sigma_

