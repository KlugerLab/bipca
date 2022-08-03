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
from scipy.stats import rv_continuous,kstest,gaussian_kde
import torch
from .utils import (zero_pad_vec,
                    filter_dict,
                    ischanged_dict,
                    nz_along,
                    make_tensor,make_scipy,
                    issparse, 
                    attr_exists_not_none,
                    safe_dim_sum,
                    safe_all_non_negative,
                    safe_hadamard,safe_elementwise_square)
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
    backend : TYPE
        Description
    col_sums : TYPE
        Description
    column_error : TYPE
        Description
    column_error_ : float
        Column-wise Sinkhorn error.
    converged : bool
        Description
    fit_ : bool
        Description
    
    left : TYPE
        Description
    left_ : array
        Left scaling vector.
    n_iter
    
    poisson_kwargs : TYPE
        Description
    q : TYPE
        Description
    read_counts : TYPE
        Description
    right : TYPE
        Description
    right_ : array
        Right scaling vector.
    row_error : TYPE
        Description
    row_error_ : float
        Row-wise Sinkhorn error.
    row_sums : TYPE
        Description
    tol : TYPE
        Description
    var : TYPE
        Description
    variance_estimator : TYPE
        Description
    X : TYPE
        Description
    X_ : array
        Input data.
    Z : TYPE
        Description
    var
    col_sums
    row_sums
    read_counts
    tol
    verbose
    
    """


    def __init__(self, variance = None, variance_estimator = 'quadratic_convex',
        row_sums = None, col_sums = None, read_counts = None, tol = 1e-6, P = 1,
        q = 1, b = None,bhat=1.0, c= None, chat=0,  n_iter = 100, conserve_memory=False, backend = 'torch', 
        logger = None, verbose=1, suppress=True,
         **kwargs):
        """Summary
        
        Parameters
        ----------
        variance : None, optional
            Description
        variance_estimator : str, optional
            Description
        row_sums : None, optional
            Description
        col_sums : None, optional
            Description
        read_counts : None, optional
            Description
        tol : float, optional
            Description
        q : int, optional
            Description
        n_iter : int, optional
            Description
        conserve_memory : bool, optional
            Description
        backend : str, optional
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
        super().__init__(conserve_memory, logger, verbose, suppress,**kwargs)

        self.read_counts = read_counts
        self.row_sums = row_sums
        self.col_sums = col_sums
        self.tol = tol
        self.n_iter = n_iter
        self.variance_estimator = variance_estimator
        if variance_estimator is None:
            q = 0
        self.q = q
        self.init_quadratic_params(b,bhat,c,chat)
        self.P = P
        self.backend = backend
        self.poisson_kwargs = {'q': self.q}
        self.converged = False
        self._issparse = None
        self.__typef_ = lambda x: x #we use this for type matching in the event the input is sparse.
        self._Z = None
        self.X_ = None
        self._var = variance
        self.__xtype = None
        self.fit_=False
    def init_quadratic_params(self,b,bhat,c,chat):
        if self.variance_estimator == 'quadratic_2param':
            if b is not None:
                ## A b value was specified
                if c is None:
                    raise ValueError("Quadratic variance parameter b was"+
                        " specified, but c was not. Both must be specified.")
                else:
                    bhat_tmp = b/(1+c)
                    #check that if bhat was specified that they match b
                    bhat = bhat_tmp
                    chat_tmp = c/(1+c)
                    chat = chat_tmp
        self.bhat = bhat
        self.chat = chat
        self.b=b
        self.c=c

    def compute_b(self,bhat,c):
        if bhat is None:
            return None
        return bhat * (1+c)
    def compute_c(self,chat):
        if chat is None:
            return None
        return chat/(1-chat)


    @property
    def var(self):  
        """Returns the entry-wise variance matrix estimated by estimate_variance.
        
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
            return self._var
        else:
            raise RuntimeError("Since conserve_memory is true, var can only be obtained by " +
                "calling Sinkhorn.estimate_variance(X, Sinkhorn.variance_estimator, q = Sinkhron.q)")
    @var.setter
    def var(self,var):
        """Summary
        
        Parameters
        ----------
        var : TYPE
            Description
        """
        if not self.conserve_memory:
            self._var = var
    @property
    def variance(self):
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
            return self._var
        else:
            raise RuntimeError("Since conserve_memory is true, variance can only be obtained by " +
                "calling Sinkhorn.estimate_variance(X, Sinkhorn.variance_estimator, q = Sinkhron.q)")
    @fitted_property
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
            if self._Z is None:
                return self.__type(self.scale(self.X))
            return self._Z
        else:
            raise RuntimeError("Since conserve_memory is true, Z can only be obtained by " +
                "calling Sinkhorn.transform(X)")
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

    @fitted_property
    def right(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if attr_exists_not_none(self,'right_'):
            return self.right_
        return self.right_
    @right.setter
    def right(self,right):
        """Summary
        
        Parameters
        ----------
        right : TYPE
            Description
        """
        self.right_ = right
    @fitted_property
    def left(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if attr_exists_not_none(self,'left_'):
            return self.left_
        else:
            return None
    @left.setter
    def left(self,left):
        """Summary
        
        Parameters
        ----------
        left : TYPE
            Description
        """
        self.left_ = left

    @fitted_property
    def row_error(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self.row_error_
    @row_error.setter
    def row_error(self, row_error):
        """Summary
        
        Parameters
        ----------
        row_error : TYPE
            Description
        """
        self.row_error_ = row_error

    @fitted_property
    def column_error(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self.row_error_
    @column_error.setter
    def column_error(self, column_error):
        """Summary
        
        Parameters
        ----------
        column_error : TYPE
            Description
        """
        self.column_error_ = column_error
        
    def __is_valid(self, X,row_sums,col_sums):
        """Verify input data is non-negative and shapes match.
        
        Parameters
        ----------
        X : TYPE
            Description
        row_sums : TYPE
            Description
        col_sums : TYPE
            Description
        X : array
        row_sums : array
        col_sums : array
        """
        eps = 1e-3
        assert safe_all_non_negative(X), "Matrix is not non-negative"
        assert np.shape(X)[0] == np.shape(row_sums)[0], "Row dimensions mismatch"
        assert np.shape(X)[1] == np.shape(col_sums)[0], "Column dimensions mismatch"
        
        # sum(row_sums) must equal sum(col_sums), at least approximately
        assert np.abs(np.sum(row_sums) - np.sum(col_sums)) < eps, "Rowsums and colsums do not add up to the same number"

    def __type(self,M):
        """Typecast data matrix M based on fitted type __typef_
        
        Parameters
        ----------
        M : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        check_is_fitted(self)
        if isinstance(M, self.__xtype):
            return M
        else:
            return self.__typef_(M)

    def fit_transform(self, X = None):
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

    
    def transform(self, A = None):
        """Scale the input by left and right Sinkhorn vectors.  Compute 
        
        Parameters
        ----------
        A : None, optional
            Description
        
        Returns
        -------
        type(X)
            Biscaled matrix of same type as input.
        
        """
        check_is_fitted(self)


        if isinstance(A,AnnData):
            X = A.X
        else:
            X = A
        if X is None:
            if not self.conserve_memory:
                X = self.X
        sparsestr = ''
        if X is not None:
            if sparse.issparse(X):
                sparsestr = 'sparse'
            else:
                sparsestr = 'dense'
        with self.logger.task(f"{sparsestr} Biscaling transform"):
            if X is not None:
                self.__set_operands(X)
                if self.conserve_memory:
                    return (self.__type(self.scale(X)))
                else:
                    self.Z = (self.__type(self.scale(X)))
            output = self.Z                
        return output

    def scale(self,X = None):
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

        
        if X.shape[0] == self.M: #why is this function so slow
            return safe_hadamard(safe_hadamard(X,self.right),self.left[:,None])
        else:
            return safe_hadamard(safe_hadamard(X,self.right[:,None]),self.left[None,:])

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
            return safe_hadamard(safe_hadamard(X,1/self.right),1/self.left[:,None])
        else:
            return safe_hadamard(safe_hadamard(X,1/self.right[:,None]),1/self.left[None,:])

    @property
    def M(self):
        return len(self.left)
    @property
    def N(self):
        return len(self.right)
    
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
        super().fit()

            
        X,A=self.process_input_data(A)
        self._issparse = issparse(X,check_scipy=True,check_torch=True)
        self.__set_operands(X)

        self._M = X.shape[0]
        self._N = X.shape[1]
        if (
            self._issparse or
             (
             self.variance_estimator == 'binomial' and isinstance(self.read_counts,int)
             )):
            sparsestr = 'sparse'
        else:
            sparsestr = 'dense'

        with self.logger.task('Sinkhorn biscaling with {} {} backend'.format(sparsestr,str(self.backend))):
            if self.fit_:
                self.row_sums = None
                self.col_sums = None
            row_sums, col_sums = self.__compute_dim_sums()
            self.c = self.compute_c(self.chat)
            self.b = self.compute_b(self.bhat, self.c)
            self.bhat = None if self.c is None else (self.b * self.P) / (1+self.c) 
            self.chat = None if self.b is None else (1+self.c - self.P) / (1+self.c)
            self.__is_valid(X,row_sums,col_sums)
            if self._var is None:
                var, rcs = self.estimate_variance(X,
                    q=self.q, bhat=self.bhat, 
                    chat=self.chat, read_counts=self.read_counts)
            else:
                var = self.var
                rcs = self.read_counts

            
            l,r,re,ce = self.__sinkhorn(var,row_sums, col_sums)
            self.__xtype = type(X)

            # now set the final fit attributes.
            if not self.conserve_memory:
                self.X = X
                self.var = var
                self.read_counts = rcs
                self.row_sums = row_sums
                self.col_sums = col_sums
            else:
                del X,var,rcs,row_sums,col_sums
            if self.variance_estimator == None: #vanilla biscaling, we're just rescaling the original matrix.
                self.left = l
                self.right = r
            else:
                self.left = np.sqrt(l)
                self.right = np.sqrt(r)
            self.row_error = re
            self.column_error = ce
            self.fit_ = True
        return self


    def __set_operands(self, X=None):
        """DEPRECATED
        """
        # changing the operators to accomodate for sparsity 
        # allows us to have uniform API for elemientwise operations
        if X is None:
            isssparse = self._issparse
        else:
            isssparse = issparse(X,check_torch=True)
        if isinstance(X, torch.Tensor):
            if isssparse:
                self.__typef_ = lambda x: make_tensor(X).to_sparse()
                self.__ispos = lambda t: (t.values()>=0).item()
                self.__dimsum = lambda t,dim: torch.sparse.sum(t,dim=dim).numpy()
            else:
                self.__typef_ = lambda x: make_tensor(X)
                self.__ispos = lambda t: (t.amin()>=0).item()
                self.__dimsum = lambda t,dim: torch.sum(t,dim=dim).numpy()

            self.__mem = lambda x,y : x*y
            self.__mesq = lambda x : x**2
        else:
            if isssparse:
                self.__typef_ = type(X)
                self.__mem = lambda x,y : x.multiply(y)
                self.__mesq = lambda x : x.power(2)
            else: 
                self.__typef_ = lambda x: x
                self.__mem= lambda x,y : np.multiply(x,y)
                self.__mesq = lambda x : np.square(x)
            self.__ispos = lambda x: np.amin(x)>=0
            self.__dimsum = lambda x,dim: x.sum(dim)

    def __compute_dim_sums(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
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

    def estimate_variance(self, X, dist=None, q=0,bhat=1.0,chat=0,read_counts=None, **kwargs):
        """Estimate the element-wise variance in the matrix X
        
        Parameters
        ----------
        X : TYPE
            Description
        dist : str, optional
            Description
        q : int, optional
            Description
        **kwargs
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        self.__set_operands(X)

        if dist is None:
            dist = self.variance_estimator
        if read_counts is None:
            read_counts = self.read_counts
        if read_counts is None:
            read_counts = X.sum(0)
        if dist=='binomial':
            var = binomial_variance(X,read_counts,
                mult = safe_hadamard, square = safe_elementwise_square)
            self.__set_operands(var)
        elif dist == 'normalized':
            var = normalized_binomial(X, self.P, read_counts,
            mult=safe_hadamard, square=safe_elementwise_square)
        elif dist =='quadratic_convex':
            var = quadratic_variance_convex(X, q = q)
        elif dist =='quadratic_2param':
            var = quadratic_variance_2param(X,bhat=bhat,chat=chat)
        elif dist == None: #vanilla biscaling
            var = X 
        else:
            var = general_variance(X)
        return var,read_counts

    def __sinkhorn(self, X, row_sums, col_sums, n_iter = None):
        """
        Execute Sinkhorn algorithm X mat, row_sums,col_sums for n_iter
        
        Parameters
        ----------
        X : TYPE
            Description
        row_sums : TYPE
            Description
        col_sums : TYPE
            Description
        n_iter : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        n_row = X.shape[0]
        row_error = None
        col_error = None
        if n_iter is None:
            n_iter = self.n_iter
        if self._N > 100000 and self.verbose>=1:
            print_progress = True
            print("Sinkhorn progress: ",end='')
        else:
            print_progress = False
        if self.backend.startswith('torch'):
            y = make_tensor(X,keep_sparse=True)
            if isinstance(row_sums,np.ndarray):
                row_sums = torch.from_numpy(row_sums).double()
                col_sums = torch.from_numpy(col_sums).double()
            with torch.no_grad():
                if torch.cuda.is_available() and (self.backend.endswith('gpu') or self.backend.endswith('cuda')):
                    try:
                        y = y.to('cuda')
                        row_sums = row_sums.to('cuda')
                        col_sums = col_sums.to('cuda')
                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            self.logger.warning('GPU cannot fit the matrix in memory. Falling back to CPU.')
                        else:
                            raise e

                u = torch.ones_like(row_sums).double()
                for i in range(n_iter):
                    if print_progress:
                        print("|", end = '')
                    u = torch.div(row_sums,y.mv(torch.div(col_sums,y.transpose(0,1).mv(u))))
                    if (i+1) % 10 == 0 and self.tol>0:
                        v = torch.div(col_sums,y.transpose(0,1).mv(u))
                        u = torch.div(row_sums,(y.mv(v)))
                        row_converged, col_converged,_,_ = self.__check_tolerance(X,u,v)
                        if row_converged and col_converged:
                            self.logger.info("Sinkhorn converged early after "+str(i+1) +" iterations.")
                            break
                        else:
                            del v


                v = torch.div(col_sums,y.transpose(0,1).mv(u))
                u = torch.div(row_sums,(y.mv(v)))
                v = v.cpu().numpy()
                u = u.cpu().numpy()
                del y
                del row_sums
                del col_sums
            torch.cuda.empty_cache()
        else:
            u = np.ones_like(row_sums)
            for i in range(n_iter):
                u = np.divide(row_sums,X.dot(np.divide(col_sums,X.T.dot(u))))
                if print_progress:
                    print("|", end = '')
                if (i+1) % 10 == 0 and self.tol > 0:
                    v = np.divide(col_sums,X.T.dot(u))
                    u = np.divide(row_sums,X.dot(v))
                    u = np.array(u).flatten()
                    v = np.array(v).flatten()
                    row_converged, col_converged,_,_ = self.__check_tolerance(X,u,v)
                    if row_converged and col_converged:
                        self.logger.info("Sinkhorn converged early after "+str(i+1) +" iterations.")
                        break
                    else:
                        del v

            v = np.array(np.divide(col_sums,X.T.dot(u))).flatten()
            u = np.array(np.divide(row_sums,X.dot(v))).flatten()

        if self.tol>0 :
            row_converged, col_converged,row_error,col_error = self.__check_tolerance(X,u,v)
            del X
            self.converged = all([row_converged,col_converged])
            if not self.converged:
                raise Exception("At least one of (row, column) errors: " + str((row_error,col_error))
                    + " exceeds requested tolerance: " + str(self.tol) + \
                        f" after {i} iterations.")
            
        return u, v, row_error, col_error
    def __check_tolerance(self,X, u, v):
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
        ZZ = safe_hadamard(safe_hadamard(X,v.squeeze()), u.squeeze()[:,None])
        row_error  = np.amax(np.abs(self._M - safe_dim_sum(ZZ,0)))
        col_error =  np.amax(np.abs(self._N - safe_dim_sum(ZZ,1)))
        if np.any([np.isnan(row_error), np.isnan(col_error)]):
            self.converged = False
            raise Exception("NaN value detected.  Check that the input matrix"+
                " is properly filtered of sparse rows and columns.")
        del ZZ
        return row_error <  self.tol, col_error < self.tol, row_error,col_error

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
    A : TYPE
        Description
    backend : TYPE
        Description
    exact : TYPE
        Description
    fit_ : bool
        Description
    k : TYPE
        Description
    kwargs : TYPE
        Description
    S : TYPE
        Description
    S_ : TYPE
        Description
    U : TYPE
        Description
    U_ : TYPE
        Description
    V : TYPE
        Description
    V_ : TYPE
        Description
    X : TYPE
        Description
    U : array
    S : array
    V : array
    svd : array
    algorithm : callable
    k : int
    n_components : int
    exact : bool
    kwargs : dict
    conserve_memory : bool
    
    
    """


    def __init__(self, n_components = None, 
                exact = True, use_eig = False, force_dense=False, vals_only=False,
                oversample_factor = 10,
                conserve_memory=False, logger = None, verbose=1, suppress=True,backend='scipy',
                **kwargs):

        super().__init__(conserve_memory, logger, verbose, suppress,**kwargs)
        self._kwargs = {}
        self.kwargs = kwargs
        
        self.__k_ = None
        self.vals_only=vals_only
        self.use_eig = use_eig
        self.force_dense = force_dense
        self.oversample_factor = oversample_factor
        self._exact = exact
        self.k=n_components
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
        hasalg = attr_exists_not_none(self, '_algorithm')
        if hasalg:
            kwargs = filter_dict(self._kwargs, self._algorithm)
        else:
            kwargs = self._kwargs
        return kwargs

    @kwargs.setter
    def kwargs(self,args):
        """Summary
        
        Parameters
        ----------
        args : TYPE
            Description
        """
        #do some logic to check if we are truely changing the arguments.
        fit_ = hasattr(self,'U_')
        if fit_ and ischanged_dict(self.kwargs, args):
            self.logger.warning('Keyword arguments have been updated. The estimator must be refit.')
            #there is a scenario in which kwargs is updated with things that do not match the function signature.
            #this code still warns the user
        if 'full_matrices' not in args:
            args['full_matrices'] = False
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
        return (self.U,self.S,self.V)
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
    def U(self,U):
        """Summary
        
        Parameters
        ----------
        U : TYPE
            Description
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
    def V(self,V):
        """Summary
        
        Parameters
        ----------
        V : TYPE
            Description
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
    def S(self,S):
        """Summary
        
        Parameters
        ----------
        S : TYPE
            Description
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
    def exact(self,val):
        """Summary
        
        Parameters
        ----------
        val : TYPE
            Description
        """
        self._exact = val
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
                self.logger.warning("The new optimal algorithm does not match the current transform. " +
                    "Recompute the transform for accuracy.")
        self._algorithm = best_alg

        # if self._algorithm is None:
        #     raise NotFittedError()
        return self._algorithm

    def __best_algorithm(self, X=None):
        """Summary
        
        Parameters
        ----------
        X : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        
        No Longer Raises
        ----------------
        AttributeError
            Description
        """
        if not attr_exists_not_none(self,'_algorithm'):
            self._algorithm = None
        if X is None:
            if attr_exists_not_none(self,"X_"):
                X = self.X
            else:
                return self._algorithm

        sparsity = issparse(X)
        if 'torch' in self.backend:
            algs = [self.__compute_torch_svd, self.__compute_randomized_svd, self.__compute_partial_torch_svd]
        else:
            algs =  [self.__compute_scipy_svd, self.__compute_randomized_svd, sklearn.utils.extmath.randomized_svd]

        if self.exact:
            if (self.k<=np.min(X.shape)*0.75):
                alg = algs[1] #returns the partial svds in the exact case
            else:
                alg = algs[0]
        else:
            if self.k>=np.min(X.shape)/5:
                if self.k<= np.min(X.shape)*0.75:
                    alg = algs[1]
                else:
                    alg = algs[0]
            else: # only use the randomized algorithms when k is less than one fifth of the size of the input. I made this number up.
                alg = algs[-1] 

        if alg == self.__compute_torch_svd:
            self.k = np.min(X.shape) ### THIS CAN LEAD TO VERY LARGE SVDS WHEN EXACT IS TRUE AND TORCH
        self._algorithm = alg
        return self._algorithm
    def __compute_randomized_svd(self,X,k):
        self.k = k
        X = make_scipy(X)
        u,s,v = sklearn.utils.extmath.randomized_svd(X,n_components=k,
                                                    n_oversamples=int(self.oversample_factor*k),
                                                    random_state=None)
        return u,s,v
    def __compute_scipy_svd(self,X,k):
        self.k = np.min(X.shape)
        X=make_scipy(X)
        if self.k >= 27000 and not self.vals_only:
            raise Exception("The optimal workspace size is larger than allowed "
                "by 32-bit interface to backend math library. "
                "Use a partial SVD or set vals_only=True")
        if self.use_eig:
            if X.shape[0]<=X.shape[1]:
                XXt = X@X.T
                XTX = False
            else:
                XXt = X.T@X
                XTX = True
            if sparse.issparse(XXt):
                XXt = XXt.toarray()
            if self.vals_only:
                s = np.sqrt(np.abs(scipy.linalg.eigvalsh(XXt,check_finite=False)))
                s.sort()
                s = s[::-1]
                u = None
                v = None
            else:
                if XTX:
                    s,v = scipy.linalg.eigh(XXt)
                    s = np.sqrt(np.abs(s))
                    six = np.argsort(s)
                    s = s[six]
                    v = v[:,six]
                    v = v[:,::-1]
                    s = s[::-1]

                    u = (X@((1/s)*v))
                    v = v.T
                else:
                    s,u = scipy.linalg.eigh(XXt)
                    s = np.sqrt(np.abs(s))
                    six = np.argsort(s)
                    s = s[six]
                    u = u[:,six]
                    u = u[:,::-1]
                    s = s[::-1]
                    v = (((1/s)*u).T@X).T
        else:
            if self.vals_only:
                s = scipy.linalg.svdvals(X)
                u = None
                v = None
            else:
                u,s,v = scipy.linalg.svd(X,full_matrices=False,check_finite=False)
        return u,s,v
    def __compute_partial_torch_svd(self,X,k):
        """Summary
        
        Parameters
        ----------
        X : TYPE
            Description
        k : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        e
            Description
        """
        y = make_tensor(X,keep_sparse = True)
        with torch.no_grad():
            if torch.cuda.is_available() and (self.backend.endswith('gpu') or self.backend.endswith('cuda')):
                try:
                    y = y.to('cuda')
                except RuntimeError as e:
                    if 'CUDA error: out of memory' in str(e):
                        self.logger.warning('GPU cannot fit the matrix in memory. Falling back to CPU.')
                    else:
                        raise e
            outs = torch.svd_lowrank(y,q=k)
            u,s,v = [ele.cpu().numpy() for ele in outs]
            torch.cuda.empty_cache()
        return u,s,v

    def __compute_torch_svd(self,X,k=None):
        """Summary
        
        Parameters
        ----------
        X : TYPE
            Description
        k : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        e
            Description
        """
        y = make_tensor(X,keep_sparse = True)
        if issparse(X) or k <= np.min(X.shape)/10:
            return self.__compute_partial_torch_svd(X,k)
        else:
            with torch.no_grad():
                if torch.cuda.is_available() and (self.backend.endswith('gpu') or self.backend.endswith('cuda')):
                    try:
                        y = y.to('cuda')
                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            self.logger.warning('GPU cannot fit the matrix in memory. Falling back to CPU.')
                        else:
                            raise e
                self.k = np.min(X.shape)
                if self.k >= 27000 and not self.vals_only:
                    raise Exception("The optimal workspace size is larger than allowed "
                        "by 32-bit interface to backend math library. "
                        "Use a partial SVD or set vals_only=True")
                if self.use_eig:
                    if y.shape[0]<=y.shape[1]:
                        yyt = torch.matmul(y,y.T)
                        yTy = False
                    else:
                        yyt = torch.matmul(y.T,y)
                        yTy = True
                    if self.vals_only:
                        s,_=torch.sqrt(torch.abs(torch.linalg.eigvalsh(yyt))).sort(descending=True)
                        s = s.cpu().numpy()
                        u = None
                        v = None
                    else:
                        if yTy:
                            e,v = torch.linalg.eigh(yyt)
                            s,indices = torch.sqrt(torch.abs(e)).sort(descending=True)
                            v = v[:,indices]
                            u = torch.matmul(y,(1/s)*v)
                            v = v.T
                        else:
                            e,u = torch.linalg.eigh(yyt)
                            s,indices = torch.sqrt(torch.abs(e)).sort(descending=True)
                            u = u[:,indices]
                            v = torch.matmul(((1/s)*u).T,y).T
                        u = u.cpu().numpy()
                        s = s.cpu().numpy()
                        v = v.cpu().numpy()
                else:
                    if self.vals_only:
                        outs=torch.linalg.svdvals(y)
                        s = outs.cpu().numpy()
                        u = None
                        v = None
                    else:
                        outs = torch.linalg.svd(y,full_matrices=False)
                        u,s,v = [ele.cpu().numpy() for ele in outs]
                torch.cuda.empty_cache()
            return u,s,v
           
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
        """Summary
        
        Parameters
        ----------
        k : TYPE
            Description
        """
        self.__k(k=k)


    def __k(self, k = None, X = None,suppress=None):
        """
        ### REFACTOR INTO A PROPERTY
        Reset k if necessary and return the rank of the SVD.
        
        Parameters
        ----------
        k : None, optional
            Description
        X : None, optional
            Description
        suppress : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        
        if k is None or 0:
            k = self.__k_
        if k is None:
            if attr_exists_not_none(self,'X_'):
                k = np.min(self.X.shape)
            else:
                if X is not None:
                    k = np.min(X.shape)
        if X is None:
            if hasattr(self,'X_'):
                X = self.X
        if X is not None:
            if k > np.min(X.shape):
                self.logger.warning("Specified rank k is greater than the minimum dimension of the input.")
        if k is None or k<=0:
            if X is not None:
                k = np.min(X.shape)
            else:
                k = 0
        if k != self.__k_: 
            msgs = []
            if self.__k_ is not None: 
                msg = "Updating number of components from k="+str(self.__k_) + " to k=" + str(k)
                level = 2
                msgs.append((msg,level))
            if self.fit_:
                #check that our new k matches
                msg = ''
                level = 0
                if k >= np.min(self.U_.shape):
                    msg = ("More components specified than available. "+ 
                          "Transformation must be recomputed.")
                    level = 1
                elif k<= np.min(self.U_.shape):
                    msg = ("Fewer components specified than available. " + 
                           "Output transforms will be lower rank than precomputed.")
                    level = 2
                if level:
                    msgs.append((msg,level))
            super().__suppressable_logs__(msgs,suppress=suppress)

            self.__k_ = k
        self._kwargs['n_components'] = self.__k_
        self._kwargs['k'] = self.__k_
        return self.__k_ 

    def __check_k_(self,k = None):
        """Summary
        
        Parameters
        ----------
        k : None, optional
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
        ### helper to check k and raise errors when it is bad
        if k is None or 0:
            k = self.k
        else:
            if k > self.k:
                raise ValueError("Requested rank requires a higher rank decomposition. " + 
                    "Re-fit the estimator at the desired rank.")
            if k <=0 : 
                raise ValueError("Cannot use a rank 0 or negative rank.")
        return k

    def fit(self, A = None,k=None,exact=None):
        """Summary
        
        Parameters
        ----------
        A : None, optional
            Description
        k : None, optional
            Description
        exact : None, optional
            Description
        
        Raises
        ------
        ValueError
        NotFittedError
        RuntimeError
        
        Deleted Parameters
        ------------------
        X : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """

        super().fit()

        if exact is not None:
            self.exact = exact
        if A is None:
            X = self.X
        else:
            X,A=self.process_input_data(A)

        if self.force_dense:
            if sparse.issparse(X):
                X = X.toarray()
            if issparse(X):
                X = X.to_dense()

        self.__k(X=X,k=k)
        if self.k == 0 or self.k is None:
            self.k = np.min(A.shape)
        if self.k >= 27000 and not self.vals_only:
                    raise Exception("The optimal workspace size is larger than allowed "
                        "by 32-bit interface to backend math library. "
                        "Use a partial SVD or set vals_only=True")
        self.__best_algorithm(X = X)
        logstr = "rank k=%d %s %s singular value decomposition using %s."
        logvals = [self.k]
        if sparse.issparse(X):
            logvals += ['sparse']
        else:
            logvals += ['dense']
        if self.exact or self.k == np.min(A.shape):
            logvals += ['exact']
        else:
            logvals += ['approximate']
        if self.use_eig=='auto':
            aspect_ratio = X.shape[0]/X.shape[1]
            if aspect_ratio > 1:
                aspect_ratio = 1/aspect_ratio
            if aspect_ratio <= 0.5:
                #rectangular matrix, use eig!
                self.use_eig=True
            else:
                self.use_eig=False
        alg = self.algorithm # this sets the algorithm implicitly, need this first to get to the fname.
        logvals += [self._algorithm.__name__]
        with self.logger.task(logstr % tuple(logvals)):
            U,S,V = alg(X, **self.kwargs)
            ix = np.argsort(S)[::-1]

            self.S = S[ix]
            if U is not None:
                self.U = U[:,ix]

                ncols = X.shape[1]
                nS = len(S)
                if V.shape == (nS,ncols):
                    self.V = V[ix,:].T
                else:
                    self.V = V[:,ix]
        self.fit_ = True
        return self
    def transform(self, k = None):
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
        with self.logger.task(logstr%logval):
            return (self.U[:,:k]*self.S[:k]) @ self.V[:,:k].T

    def get_factors(self):
        if self.vals_only:
            return None, self.S, None
        return self.U, self.S, self.V
    def factorize(self,X=None,k=None,exact=None):
        self.fit(X,k,exact)
        return self.get_factors()

    def fit_transform(self, X = None, k=None, exact=None):
        """Compute an SVD and return the rank `k` approximation of `X`
        
        
        .. Error:: If called with ``X = None`` and :attr:`conserve_memory <bipca.math.SVD>` is true, 
                    this method will fail as there is no underlying matrix to transform.  
        
        Parameters
        ----------
        X : array
            Description
        k : None, optional
            Description
        exact : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        ValueError
        NotFittedError
        RuntimeError
        
        """
        self.fit(X,k,exact)
        return self.transform()

    def PCA(self, k = None):
        """Summary
        
        Parameters
        ----------
        k : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        k = self.__check_k_(k)
        return self.U[:,:k]*self.S[:k]

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
    def __init__(self, default_shrinker = 'frobenius',rescale_svs = True,
        conserve_memory=False, logger = None, verbose=1, suppress=True,
        **kwargs):
        """Summary
        
        
        Parameters
        ----------
        default_shrinker : str, optional
            Description
        rescale_svs : bool, optional
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
        super().__init__(conserve_memory, logger, verbose, suppress,**kwargs)
        self.default_shrinker = default_shrinker
        self.rescale_svs = rescale_svs

    #some properties for fetching various shrinkers when the object has been fitted.
    #these are just wrappers for transform.
    @fitted_property
    def frobenius(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        check_is_fitted(self)
        return self.transform(shrinker = 'fro')
    @fitted_property
    def operator(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        check_is_fitted(self)
        return self.transform(shrinker = 'op')
    @fitted_property
    def hard(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        check_is_fitted(self)
        return self.transform(shrinker = 'hard')
    @fitted_property
    def soft(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        check_is_fitted(self)
        return self.transform(shrinker = 'soft')
    @fitted_property
    def nuclear(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self.transform(shrinker = 'nuc')

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
    

    def fit(self, y, shape=None, sigma = None, theory_qy = None, q = None, suppress = None):
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
        if isinstance(y,AnnData):
            self.A = y
            if 'SVD' in self.A.uns.keys():
                y = self.A.uns['SVD']['S']
            else:
                y = self.A.uns['BiPCA']['S_Z']
        try:
            check_is_fitted(self)
            try:
                assert np.allclose(y,self.y_) #if this fails, then refit
            except: 
                self.__suppressable_logs__("Refitting to new input y",level=1,suppress=suppress)
                raise
        except:
            with self.logger.task("Shrinker fit"):
                if shape is None:
                    if _is_vector(y):
                        raise ValueError("Fitting requires shape parameter")
                    else:
                        assert y.shape[0]<=y.shape[1]
                        shape = y.shape
                        y = np.diag(y)
                assert shape[0]<=shape[1]
                assert (np.all(y.shape<=shape))
                y = np.sort(y)[::-1]
                # mp_rank, sigma, scaled_cutoff, unscaled_cutoff, gamma, emp_qy, theory_qy, q
                self.MP = MarcenkoPastur(gamma = shape[0]/shape[1])
                params = self._estimate_MP_params(y=y, N = shape[1], M = shape[0], sigma = sigma, theory_qy = theory_qy, q = q)
                self.sigma, self.scaled_mp_rank, self.scaled_cutoff, self.unscaled_mp_rank, self.unscaled_cutoff, self.gamma, self.emp_qy, self.theory_qy, self.quantile , self.scaled_cov_eigs, self.cov_eigs = params
                self.M_ = shape[0]
                self.N_ = shape[1]
                self.y_ = y
                if self.scaled_mp_rank == len(y) and sigma is not None and len(y)!=np.min(shape):
                    return self, False #not converged, needs more eigs?
                else:
                    return self, True

        return self, True

    def _estimate_MP_params(self, y = None,
                            M = None,N = None, theory_qy = None, q = None,sigma = None):
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
            if np.any([y,M,N]==None):
                check_is_fitted(self)
            if y is None:
                y = self.y_
            if M is None:
                M = self._M
            if N is None:
                N = self._N
            if theory_qy is not None and q is None:
                raise ValueError("If theory_qy is specified then q must be specified.")        
            assert M<=N
            unscaled_cutoff = np.sqrt(N) + np.sqrt(M)


            rank = (y>=unscaled_cutoff).sum()
            if rank == len(y):
                self.logger.info("Approximate Marcenko-Pastur rank is full rank")
                mp_rank = len(y)
            else:
                self.logger.info("Approximate Marcenko-Pastur rank is "+ str(rank))
                mp_rank = rank
            #quantile finding and setting

            ispartial = len(y)<M
            if ispartial: #We assume that we receive the top k sorted singular values. The objective is to pick the closest value to the median.
                self.logger.info("A fraction of the total singular values were provided")
                if len(y) >= np.ceil(M/2): #len(y) >= ceil(M/2), then 
                    if M%2: #M is odd and emp_qy is exactly y[ceil(M/2)-1] (due to zero indexing)
                        qix = int(np.ceil(M/2))
                        emp_qy = y[qix-1]
                    else:
                        #M is even.  We need 1/2*(y[M/2]+y[M/2-1]) (again zero indexing)
                        # we don't necessarily have y[M/2].
                        qix = int(M/2)        
                        if len(y)>M/2:
                            emp_qy = y[qix]+y[qix-1]
                            emp_qy = emp_qy/2;
                        else: #we only have the lower value, len(y)==M/2.
                            emp_qy = y[qix-1]
                            qix-=1
                else:
                    # we don't have the median. we need to grab the smallest number in y.
                    qix = len(y)
                    emp_qy = y[qix-1]
                #now we compute the actual quantile.
                q = 1-qix/M
                z = zero_pad_vec(y,M) #zero pad here for uniformity.
            else:
                z = y
                if q is None:
                    q = 0.5
                emp_qy = np.percentile(z,q*100)

            if q>=1:
                q = q/100
                assert q<=1
            #grab the empirical quantile.
            assert(emp_qy != 0 and emp_qy >= np.min(z))
            #computing the noise variance
            if sigma is None: #precomputed sigma
                if theory_qy is None: #precomputed theory quantile
                    theory_qy = self.MP.ppf(q)
                sigma = emp_qy/np.sqrt(N*theory_qy)
                self.logger.info("Estimated noise variance computed from the {:.0f}th percentile is {:.3f}".format(np.round(q*100),sigma**2))

            else:
                self.logger.info("Pre-computed noise variance is {:.3f}".format(sigma**2))
            n_noise = np.sqrt(N)*sigma
            #scaling svs and cutoffs
            scaled_emp_qy = (emp_qy/n_noise)
            cov_eigs = (z/np.sqrt(N))**2
            scaled_cov_eigs = (z/n_noise)
            scaled_cutoff = self.MP.b
            scaled_mp_rank = (scaled_cov_eigs**2>=scaled_cutoff).sum()
            if scaled_mp_rank == len(y):
                self.logger.info("\n ****** It appears that too few singular values were supplied to Shrinker. ****** \n ****** All supplied singular values are signal. ****** \n ***** It is suggested to refit this estimator with larger `n_components`. ******\n ")

            self.logger.info("Scaled Marcenko-Pastur rank is "+ str(scaled_mp_rank))

        return sigma, scaled_mp_rank, scaled_cutoff, mp_rank, unscaled_cutoff, self.MP.gamma, emp_qy, theory_qy, q, scaled_cov_eigs**2, cov_eigs


    def fit_transform(self, y = None, shape = None, shrinker = None, rescale = None):
        """Summary
        
        Parameters
        ----------
        y : None, optional
            Description
        shape : None, optional
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
        self.fit(y,shape)
        if shrinker is None:
            shrinker = self.default_shrinker
        return self.transform(y = y, shrinker = shrinker)

    def transform(self, y = None,shrinker = None,rescale=None):
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
        check_is_fitted(self)
        if y is None:
            #the alternative is that we transform a non-fit y.
            y = self.y_
        if shrinker is None:
            shrinker = self.default_shrinker
        if rescale is None:
            rescale = self.rescale_svs
        with self.logger.task("Shrinking singular values according to " + str(shrinker) + " loss"):
            return  _optimal_shrinkage(y, self.sigma_, self.M_, self.N_, self.gamma_, scaled_cutoff = self.scaled_cutoff_,shrinker  = shrinker,rescale=rescale)

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
    if issparse(X,check_torch=False):
        Y = Y.toarray()
    Y = np.abs(Y)**2
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
    if issparse(X,check_torch=False):
        Y = X.copy()
        Y.data = (1-q)*X.data + q*X.data**2
        return Y
    return (1-q) * X + q * X**2

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
    if issparse(X,check_torch=False):
        Y = X.copy()
        Y.data = bhat*X.data + chat*X.data**2
        return Y
    return safe_hadamard(X,bhat) + safe_hadamard(safe_elementwise_square(X),chat)

def binomial_variance(X, counts, 
    mult = lambda x,y: x*y, 
    square = lambda x: x**2):
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
    if sparse.issparse(X) and isinstance(counts,int):
        var = X.copy()
        div = np.divide(counts,counts-1)
        var.data = (var.data*div) - (var.data**2 * (1/(counts-1)))
        var.data = abs(var.data)
        var.eliminate_zeros()
    else:
        var = mult(X,np.divide(counts, counts - 1)) - mult(square(X), (1/(counts-1)))
        var = abs(var)
    if isinstance(counts,int):
        if not sparse.issparse(var):
            var = sparse.csr_matrix(var)
    return var

def normalized_binomial(X,p, counts,
        mult=lambda x,y: x*y, square = lambda x: x**2):
    mask = np.where(counts>=2,True,False)
    X = X.copy()
    counts = counts.copy()
    X[mask] /= counts[mask] #fill the elements where counts >= 2 with Xij / nij
    X[np.logical_not(mask)] = 0 #elmements where counts < 2 = 0. This is \bar{Y}
    counts[np.logical_not(mask)] = 2 #to fix nans, we truncate counts to 2.
    var = mult(p/counts, X) + mult(1-1/counts - p,square(X))
    var /= (1 - 1/counts)
    
    return var

class MarcenkoPastur(rv_continuous): 
    """"marcenko-pastur
    
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
            gamma = 1/gamma
        a = (1-gamma**0.5)**2
        b = (1+gamma**0.5)**2
        
        super().__init__(a=a,b=b)
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
        m0 = lambda a: np.clip(a, 0,None)
        m0b = self.b - x
        m0b = np.core.umath.maximum(m0b,0)
        m0a = x-self.a
        m0a = np.core.umath.maximum(m0a,0)
        
        return np.sqrt( m0b * m0a) / ( 2*np.pi*self .gamma*x)

    def cdf(self,x,which='analytical'):
        which = which.lower()
        if which not in ['analytical', 'numerical']:
            raise ValueError(f"which={which} is invalid." 
                " MP.cdf requires which in"
                " ['analytical, 'numerical'].")
        if which=='numerical':
            return super()._cdf(x)
        else:
            return self.cdf_analytical(x)
    def cdf_analytical(self,x):
        with np.errstate(all='ignore'):
            isarray = isinstance(x,np.ndarray)
            typ = type(x)
            x = np.asarray(x)
            const = 1 / (2*np.pi * self.gamma)
            m0b = self.b - x
            m0a = x-self.a
            rx = np.sqrt(m0b/m0a,where=m0b/m0a>0)
            term1 = np.pi * self.gamma
            term2 = np.sqrt( m0b*m0a,where=m0b*m0a>0)
            term3 = -(1+self.gamma) * np.arctan( (rx**2-1) / (2*rx))
            term4_numerator = self.a*rx**2 - self.b
            term4_denominator = 2*(1-self.gamma)*rx
            term4 = (1-self.gamma) * np.arctan( term4_numerator / 
                                                term4_denominator  )
            output = const * ( term1 + term2 + term3 + term4 )
            output = np.where(x>self.a,output,0)
            output = np.where(x>=self.b, 1,output)
        if isarray:
            return output
        else:
            return typ(output)

class SamplingMatrix(object):
    __array_priority__ = 1
    def __init__(self, X = None):
        self.ismissing=False
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

    def __build_coordinates_sparse(self,X):
        X = make_scipy(X).tocoo()
        coordinates = np.where(np.isnan(X.data))
        rows = X.row[coordinates]
        cols = X.col[coordinates]
        return rows, cols

    def __build_coordinates_dense(self, X):
        rows,cols = np.where(np.isnan(X))
        return rows, cols

    def __compute_probabilities_from_coordinates(self, rows, cols):
        m,n = self.shape
        n_samples = m*n - len(rows)
        grand_mean = 1 / (m*n) * n_samples
        self.row_p =  np.ones((m,1),)
        self.row_p = self.row_p / np.sqrt(grand_mean)

        self.col_p =  np.ones((1,n),)
        self.col_p = self.col_p / np.sqrt(grand_mean)


        if n_samples < m*n:
            
            
            unique, counts = np.unique(rows,return_counts=True)
            self.row_p[unique.astype(int),:] =  (((n-counts)/n)/np.sqrt(grand_mean))[:,None]
                                        
            
            unique, counts = np.unique(cols,return_counts=True)
            self.col_p[:,unique.astype(int)] =  (((m-counts)/m)/np.sqrt(grand_mean))[None,:]

            self.ismissing = True
    def __getitem__(self,pos):
        if isinstance(pos,tuple):
            row,col = pos
        else:
            if isinstance(pos,slice):
                start,stop,step = pos.start,pos.stop,pos.step
                if start is None:
                    start = 0
                if stop is None:
                    stop = np.prod(self.shape)
                if step is None:
                    step = 1
                pos = np.arange(start,stop,step)
            row,col = np.unravel_index(pos, self.shape)
        return np.core.umath.minimum(self.get_row(row)*self.get_col(col),1)
    @property
    def T(self):
        obj = SamplingMatrix()
        obj.M,obj.N = self.N,self.M
        obj.coords = self.coords[1],self.coords[0]
        obj.row_p = self.col_p.T
        obj.col_p = self.row_p.T
        obj.ismissing = self.ismissing
        return obj
    def __call__(self):
        return np.core.umath.minimum(self.row_p*self.col_p,1)
    def __add__(self, val):
        return val + self()
    def __radd__(self, val):
        return self + val
    def __sub__(self, val):
        return -1 * val + self()
    def __rsub__(self, val):
        return val + -1*self
    def __mul__(self, val):
        return val * self()
    def __rmul__(self, val):
        return val * self()
    def __repr__(self):
        return f"SamplingMatrix({self.row_p},{self.col_p})"
    def get_row(self,row):
        return self.row_p[row,:].squeeze()
    def get_col(self,col):
        return self.col_p[:,col].squeeze()

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

def normalized_KS(y,mp, m, r):

    return  KS(y,mp) - r/m

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
    return kstest(y,mp.cdf)[0]

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
def emp_pdf_loss(pdf, epdf, loss = L2, start = 0):
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
    
    val = integrate.quad(lambda x: loss(x, pdf, epdf), start, np.inf,limit=100)[0]
    
    
    return val

def emp_mp_loss(mat, gamma = 0, loss = L2, precomputed=True,M=None, N = None):
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
        if (M is None or N is None):
            raise RuntimeError()
    else:
        M = np.shape(mat)[0]
        N = np.shape(mat)[1]
    if gamma == 0:
        gamma = M/N

    if gamma >= 1:
        # have to pad singular values with 0
        if not precomputed:
            svs = np.linalg.svd(mat)[1]
            cov_eig = np.append(1/N*svs**2, np.zeros(M-N))
        else:
            cov_eig = mat
        # hist = np.histogram(cov_eig, bins=np.minimum(np.int(N/4),60))
        hist = np.histogram(cov_eig, bins = np.int(4*np.log2(5*N)))
        esd = sp.stats.rv_histogram(hist).pdf

        # error at 0 is the difference between the first bin of the histogram and (1 - 1/gamma) = (M - N)/N
        err_at_zero = np.absolute(hist[0][0] - (1 - 1 / gamma))
        if loss == L2:
            err_at_zero = err_at_zero**2

        # we now start integrating AFTER the bin that contains the zeros
        start = hist[1][1]
        u_edge = (1 + np.sqrt(gamma))**2
        # we integrate a little past the upper edge of MP, or the last bin of the histogram, whichever one is greater.
        end = 1.2*np.maximum(u_edge, hist[1][-1])
        # end = 20
        val = integrate.quad(lambda x: loss(x, lambda y: mp_pdf(y, gamma), esd), start, end)[0] + err_at_zero
    
    else:
        if not precomputed:
            svs = np.linalg.svd(mat)[1]
            cov_eig = 1/N*svs**2
        else:
            cov_eig = mat
        # hist = np.histogram(cov_eig, bins=np.minimum(np.int(N/4),60))
        hist = np.histogram(cov_eig, bins = np.int(4*np.log2(5*N)))
        esd = sp.stats.rv_histogram(hist).pdf
        
        u_edge = (1 + np.sqrt(gamma))**2
        end = 1.2*np.maximum(u_edge, hist[1][-1])
        # end = 20
        val = integrate.quad(lambda x: loss(x, lambda y: mp_pdf(y, gamma), esd), 0, end)[0]

    return val


def debias_singular_values(y,m,n,gamma=None, sigma=1):
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
    #optimal shrinker derived by boris for inverting singular values to remove noise
    #if sigma is 1, then y may be normalized
    #if sigma is not 1, then y is unnormalized
    if gamma is None:
        gamma = m/n
    sigma2 = sigma**2
    threshold = sigma*(np.sqrt(n)+np.sqrt(m))

    nsigma2 = n*sigma2
    s = np.sqrt((y**2 + nsigma2 * (1-gamma))**2 - 4*y**2*nsigma2)
    s = y**2 - nsigma2 * (1+gamma) + s
    s = np.sqrt(s/2)
    return np.where(y>threshold,s,0)

def _optimal_shrinkage(unscaled_y, sigma, M,N, gamma, scaled_cutoff = None, shrinker = 'frobenius',rescale=True,logger=None):
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
    frobenius = lambda y: 1/y * np.sqrt((y**2-gamma-1)**2-4*gamma)
    operator = lambda y: 1/np.sqrt(2) * np.sqrt(y**2-gamma-1+np.sqrt((y**2-gamma-1)**2-4*gamma))

    soft = lambda y: y-np.sqrt(scaled_cutoff)
    hard = lambda y: y
  
    #compute the scaled svs for shrinking
    n_noise = (np.sqrt(N))*sigma
    scaled_y = unscaled_y / n_noise
    # assign the shrinker
    cond = scaled_y>=np.sqrt(scaled_cutoff)
    with np.errstate(invalid='ignore',divide='ignore'): # the order of operations triggers sqrt and x/0 warnings that don't matter.
        #this needs a refactor
        if shrinker in ['frobenius','fro']:
            shrunk = lambda z: np.where(cond,frobenius(z),0)
        elif shrinker in ['operator','op']:
            shrunk =  lambda z: np.where(cond,operator(z),0)
        elif shrinker in ['soft','soft threshold']:
            shrunk = lambda z: np.where(cond,soft(z),0)
        elif shrinker in ['hard','hard threshold']:
            shrunk = lambda z: np.where(cond,hard(z),0)
        # elif shrinker in ['boris']:
        #     shrunk = lambda z: np.where(unscaled_y>)
        elif shrinker in ['nuclear','nuc']:
            x = operator(scaled_y)
            x2 = x**2
            x4 = x2**2
            bxy = np.sqrt(gamma)*x*scaled_y
            nuclear = (x4-gamma-bxy)/(x2*scaled_y)
            #special cutoff here
            cond = x4>=gamma+bxy
            shrunk = lambda z: np.where(cond,nuclear,0)
        else:
            raise ValueError('Invalid Shrinker') 
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
    scaled_bound = (1+np.sqrt(gamma))**2
    return scaled_bound

class KDE(rv_continuous):
    def __init__(self,x):
        a,b = np.min(x),np.max(x)
        self.kde = gaussian_kde(x.squeeze())
        super().__init__(a=a,b=b)
    def _pdf(self, x):
        return self.kde(x)
    def _cdf(self, x):
        from scipy.special import ndtr
        cdf = tuple(ndtr(np.ravel(item - self.kde.dataset) / self.kde.factor).mean()
                for item in x)
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
    def __init__(self, maintain_sparsity = False, consider_zeros = True, conserve_memory=False, logger = None, verbose=1, suppress=True,
         **kwargs):
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
        super().__init__(conserve_memory, logger, verbose, suppress,**kwargs)
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
    def X_centered(self,Mc):
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
        #Computes and returns the dense rescaling matrix
        mat = -1* self._grand_mean*np.ones((self.N,self.M))
        mat += self._row_means[:,None]
        mat += self._column_means[None,:]
        return mat
    def __compute_grand_mean(self,X,consider_zeros=True):
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
        if issparse(X,check_torch=False):
            nz = lambda  x : x.nnz
            D = X.data
        else:
            nz = lambda x : np.count_nonzero(x)
            D = X
        if consider_zeros:
            nz = lambda x : np.prod(x.shape)

        return np.sum(D)/nz(X)

    def __compute_dim_means(self,X,axis=0,consider_zeros=True):
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
            nzs = nz_along(X,axis)
        else:
            nzs = X.shape[axis] 

        means = X.sum(axis)/nzs
        if issparse(X,check_torch=False):
            means = np.array(means).flatten()
        return means

    def fit_transform(self,X):
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
        #let's compute the grand mean first.
        self._grand_mean = self.__compute_grand_mean(X, self.consider_zeros)
        self._column_means = self.__compute_dim_means(X,axis=0, consider_zeros=self.consider_zeros)
        self._row_means = self.__compute_dim_means(X,axis=1, consider_zeros=self.consider_zeros)
        self.N = X.shape[0]
        self.M = X.shape[1]
        self.fit_ = True
        return self

    @fitted 
    def transform(self,X):
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
        #remove the means learned from .fit() from the input X.
        if self.maintain_sparsity:
            dense_rescaling_matrix = self.rescaling_matrix
            if issparse(X,check_torch = False):
                X = sparse.csr_matrix(X)
                X_nzindices = X.nonzero()
                X_c = X
                X_c.data = X.data - dense_rescaling_matrix[X_nzindices]
            else:
                X_nzindices = np.nonzero(X)
                X_c = X
                X_c[X_nzindices] = X_c[X_nzindices] - dense_rescaling_matrix[X_nzindices]
        else:
            X_c = X - self.rescaling_matrix
        if isinstance(X_c,np.matrix):
            X_c = np.array(X_c)
        self.X_centered = X_c
        return X_c

    def scale(self,X):
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
        #Subtract means from the data
        if self.maintain_sparsity:
            dense_rescaling_matrix = self.rescaling_matrix
            if issparse(X,check_torch=False):
                X = sparse.csr_matrix(X)
                X_nzindices = X.nonzero()
                X_c = X
                X_c.data = X.data + dense_rescaling_matrix[X_nzindices]
            else:
                X_nzindices = np.nonzero(X)
                X_c = X
                X_c[X_nzindices] = X_c[X_nzindices] + dense_rescaling_matrix[X_nzindices]
        else:
            X_c = X + self.rescaling_matrix
        if isinstance(X_c,np.matrix):
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
   

def minimize_chebfun(p,domain=[0,1]):
    start,stop = domain
    pd = p.differentiate()
    pdd = pd.differentiate()
    try:
        q = pd.roots() # the zeros of the derivative
        #minima are zeros of the first derivative w/ positive second derivative
        mi = q[pdd(q)>0]
        if mi.size == 0:
            mi = np.linspace(start,stop,100000)
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
    
    return q