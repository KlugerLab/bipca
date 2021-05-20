"""Subroutines used to compute a BiPCA transform
"""



import numpy as np
import sklearn as sklearn
import scipy as scipy
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import randomized_svd
import scipy.integrate as integrate
import scipy.sparse as sparse
import tasklogger
from sklearn.base import clone
from .utils import _is_vector, _xor, _zero_pad_vec,filter_dict,ischanged_dict
from .base import BiPCAEstimator,__memory_conserved__

class Sinkhorn(BiPCAEstimator):
    """
    Sinkhorn biscaling
    
    Parameters
    ----------
    centered : Bool, optional
        Return mean-centered matrix
    var : array, optional
        variance matrix for input data
        (default variance is estimated from data using binomial model).
    variance_estimator : {'binomial', 'poisson'}, optional
    row_sums : array, optional
        Target row sums. Defaults to 1.
    col_sums : array, optional
        Target column sums. Defaults to 1.
    read_counts : array
        The expected `l1` norm of each column
        (Defaults to the sum of the input data).
    tol : float, default 1e-6
        Sinkhorn tolerance
    n_iter : int, default 30
        Number of Sinkhorn iterations.
    return_scalers : bool, default False
        Return left and right scaling vectors from transform methods.
    force_sparse : bool, default False
        Cast outputs as `scipy.sparse.csr_matrix`.
    conserve_memory : bool, default True
        Save output scaled matrix as a factor.  
    verbose : {0, 1, 2}, default 0
        Logging level
    logger : :log:`tasklogger.TaskLogger < >`, optional
        Logging object. By default, write to new logger.
    
    Attributes
    ----------
    col_error_ : float
        Column-wise Sinkhorn error.
    col_sums : TYPE
        Description
    force_sparse : TYPE
        Description
    left_ : array
        Left scaling vector.
    logger : TYPE
        Description
    n_iter : TYPE
        Description
    read_counts : TYPE
        Description
    return_errors : TYPE
        Description
    return_scalers : TYPE
        Description
    right_ : array
        Right scaling vector.
    row_error_ : float
        Row-wise Sinkhorn error.
    row_sums : TYPE
        Description
    tol : TYPE
        Description
    var : TYPE
        Description
    verbose : TYPE
        Description
    X_ : array
        Input data.
    var
    col_sums
    row_sums
    read_counts
    tol
    n_iter
    return_scalers
    return_errors
    force_sparse
    verbose
    logger
    """

    __log_instance = 0
    """How many `Sinkhorn` objects are there?"""

    def __init__(self, variance = None, variance_estimator = 'binomial',
        row_sums = None, col_sums = None, read_counts = None, tol = 1e-6, 
        n_iter = 30, return_scalers = True,  force_sparse = False, return_errors = False, 
        conserve_memory=True, logger = None, verbose=1, suppress=True,
         **kwargs):
        
        super().__init__(conserve_memory, logger, verbose, suppress,**kwargs)

        self.return_scalers = return_scalers
        self.read_counts = read_counts
        self.row_sums = row_sums
        self.col_sums = col_sums
        self.tol = tol
        self.n_iter = n_iter
        self.return_errors = return_errors
        self.force_sparse = force_sparse
        self.variance_estimator = variance_estimator
        self._issparse = None
        self.__typef_ = lambda x: x #we use this for type matching in the event the input is sparse.
        self._Z = None
        self._X = None
        self._var = variance
        self.__xtype = None
        self.fit_ = False

    @property
    @__memory_conserved__
    def X(self):
        return self._X

    @X.setter
    def X(self,X):
        if not self.conserve_memory:
            self._X = X

    @property
    @__memory_conserved__
    def var(self):  
        return self._var
    @var.setter
    def var(self,var):
        if not self.conserve_memory:
            self._var = var
    @property
    @__memory_conserved__
    def variance(self):
        return self._var
    
    @property
    @__memory_conserved__
    def Z(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if self.Z is None:
            self.Z = self.__type(self.scale(self.X))
        return self._Z
    @Z.setter
    def Z(self,Z):
        if not self.conserve_memory:
            self._Z = Z

    @property
    def right(self):
        return self.right_
    @property
    def left(self):
        return self.left_
    

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
        assert np.amin(X) >= 0, "Matrix is not non-negative"
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
        if self.force_sparse and not self._issparse:
            return sparse.csr_matrix(M)
        else:
            if isinstance(M, self.__xtype):
                return M
            else:
                return self.__typef_(M)

    def fit_transform(self, X = None, return_scalers = None, return_errors = None):
        """Summary
        
        Parameters
        ----------
        X : None, optional
            Description
        return_scalers : None, optional
            Description
        return_errors : None, optional
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
        return self.transform(X=X, return_scalers = return_scalers, return_errors = return_errors)

    
    def transform(self, X = None, return_scalers = None, return_errors= None):
        """Scale the input by left and right Sinkhorn vectors.  Compute 
        
        Parameters
        ----------
        X : None, optional
            Input matrix to scale
        return_scalers : None, optional
            Description
        return_errors : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        check_is_fitted(self)
        if return_scalers is None:
            return_scalers = self.return_scalers
        if return_errors is None:
            return_errors = self.return_errors
        with self.logger.task('Biscaling transform'):
            if X is None:
                output = [self.Z,]
            else:
                output = [self.__type(self.scale(X)),]
            if return_scalers:
                output += [self.left_,self.right_]
            if return_errors:
                output += [self.row_error_, self.col_error_]

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
        return self.__mem(self.__mem(X,self.right),self.left[:,None])

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
        return self.__mem(self.__mem(X,1/self.right),1/self.left[:,None])

    def fit(self, X):
        """Summary
        
        Parameters
        ----------
        X : TYPE
            Description
        """
        super().fit()

        with self.logger.task('Sinkhorn biscaling'):

            self._issparse = sparse.issparse(X)
            if self.force_sparse and self._issparse:
                X = sparse.csr_matrix(X)
            self.__set_operands(X)

            self._M = X.shape[0]
            self._N = X.shape[1]
            if self.fit_:
                self.row_sums = None
                self.col_sums = None
            row_sums, col_sums = self.__compute_dim_sums()
            self.__is_valid(X,row_sums,col_sums)

            if self._var is None:
                var, rcs = self.estimate_variance(X,self.variance_estimator)
            else:
                var = self.var
                rcs = self.read_counts

            l,r,re,ce = self.__sinkhorn(var,row_sums, col_sums)
            # now set the final fit attributes.
            self.X = X
            self.var = var
            self.__xtype = type(X)
            self.read_counts = rcs
            self.row_sums = row_sums
            self.col_sums = col_sums
            self.left_ = np.sqrt(l)
            self.right_ = np.sqrt(r)
            self.row_error_ = re
            self.col_error_ = ce
            self.fit_ = True
    def __set_operands(self, X=None):
        """Summary
        
        Parameters
        ----------
        X : TYPE
            Description
        """
        # changing the operators to accomodate for sparsity 
        # allows us to have uniform API for elemientwise operations
        if X is None:
            issparse = self._issparse
        else:
            issparse = sparse.issparse(X)
        if issparse:
            self.__typef_ = type(X)
            self.__mem = lambda x,y : x.multiply(y)
            self.__mesq = lambda x : x.power(2)
        else:
            self.__typef_ = lambda x: x
            self.__mem= lambda x,y : np.multiply(x,y)
            self.__mesq = lambda x : np.square(x)

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

    def estimate_variance(self, X, dist='binomial'):
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
        read_counts = (X.sum(0))
        if dist=='binomial':
            var = binomial_variance(X,read_counts,
                mult = self.__mem, square = self.__mesq)
        else:
            var = poisson_variance(X)
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
        a = np.ones(n_row,)
        for i in range(n_iter):
            b = np.divide(col_sums, X.T.dot(a))
            a = np.divide(row_sums, X.dot(b))    
        a = np.array(a).flatten()
        b = np.array(b).flatten()
        if self.tol>0:
            ZZ = self.__mem(self.__mem(X,b), a[:,None])
            row_error  = np.amax(np.abs(self._M - ZZ.sum(0)))
            col_error =  np.amax(np.abs(self._N - ZZ.sum(1)))
            if row_error > self.tol:
                self.logger.warning("Col error: " + str(row_error)
                    + " exceeds requested tolerance: " + str(self.tol))
            if col_error > self.tol:
                self.logger.warning("Row error: " + str(col_error)
                    + " exceeds requested tolerance: " + str(self.tol))
            
        return a, b, row_error, col_error


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
    suppress: bool, default True
        Suppress helpful interrupts due to suspected redundant calls to SVD.fit()
    verbose : {0, 1, 2}, default 0
        Logging level
    logger : :log:`tasklogger.TaskLogger < >`, optional
        Logging object. By default, write to new logger.
    **kwargs
        Arguments for downstream SVD algorithm.
        
    Attributes
    ----------
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
    logger : :log:`tasklogger.TaskLogger < >`
        Associated logging objecs
    _kwargs : dict
        All SVD-related keyword arguments stored in the object.

    
    """

    __log_instance = 0
    """How many `SVD` objects are there?"""

    def __init__(self, n_components = None, algorithm = None, exact = True, 
                conserve_memory=True, logger = None, verbose=1, suppress=True,
                **kwargs):

        super().__init__(conserve_memory, logger, verbose, suppress,**kwargs)
        self._kwargs = {}
        self.kwargs = kwargs
        self.__k_ = None
        self._algorithm = None
        self._exact = exact
        self.__feasible_algorithm_functions = []
        self.k=n_components
        self.__reset_feasible_algorithms(algorithm, exact)

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
        hasalg = hasattr(self, '_algorithm')
        if hasalg:
            kwargs = filter_dict(self._kwargs, self._algorithm)
        else:
            kwargs = self._kwargs
        return kwargs

    @kwargs.setter
    def kwargs(self,args):
        #do some logic to check if we are truely changing the arguments.
        isfit = hasattr(self,'U_')
        if isfit and ischanged_dict(self.kwargs, args):
            self.logger.warning('Keyword arguments have been updated. The estimator must be refit.')
            #there is a scenario in which kwargs is updated with things that do not match the function signature.
            #this code still warns the user
        self._kwargs = args
    @property
    def svd(self):
        """Return the entire singular value decomposition

        .. Warning:: The object must be fit before requesting this attribute. 

        Returns
        -------
        (numpy.ndarray,numpy.ndarray,numpy.ndarray)
            (U,S,V) : The left singular vectors, singular values, and right singular vectors such that USV^T = X

        Raises 
        ------
        NotFittedError
        """
        check_is_fitted(self)
        return (self.U_,self.S_,self.V_)
    @property
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
        check_is_fitted(self)
        return self.U_
    @property
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
        check_is_fitted(self)
        return self.V_

    @property
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
        check_is_fitted(self)
        return self.S_
    
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
        self._exact = val
        self.__reset_feasible_algorithms(exact =val)

    @property
    def algorithm(self):
        """
        Return the algorithm used for factoring the fitted data. 
        The keyword arguments used with this algorithm are returned by :attr:`kwargs`.
        
        Returns
        -------
        callable
            single argument lambda function wrapping the underlying algorithm.
        
        Raises
        ------
        NotFittedError
            If a correct algorithm cannot be determined or set, the estimator has not been fit.

        """
        ###Implicitly determines and sets algorithm by wrapping __best_algorithm
        best_alg = self.__best_algorithm()
        if hasattr(self, "U_"):
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
        
        Raises
        ------
        AttributeError
            Description
        """
        if self.__feasible_algorithm_functions == []:
            raise AttributeError("No feasible algorithms to search")
        elif len(self.__feasible_algorithm_functions) == 1:
            return self.__feasible_algorithm_functions[0]

        if X is None:
            if hasattr(self,"X_"):
                X = self.X_
            else:
                return self._algorithm

        if self.k==np.min(X.shape):
            if sparse.issparse(X):
                self.logger.warning("Computing a full SVD on a sparse matrix is often inefficient." +
                    "Consider a truncated factorization with k slightly less than "+
                    "the smallest dimension of X.")
                self.k = np.min(X.shape)-1
                return self.__feasible_algorithm_functions[-1]
            return self.__feasible_algorithm_functions[0]
        else:
            return self.__feasible_algorithm_functions[-1]

        
    def __reset_feasible_algorithms(self, algorithm = None, exact = None):
        """
        Parses the input parameters that determine potential SVD algorithms
        
        Parameters
        ----------
        algorithm : None, optional
            Description
        exact : None, optional
            Description
        """
        #clear the old 
        self.__feasible_algorithm_functions = []
        if exact is None:
            exact = self.exact
        # pre_specified single algorithm
        if callable(algorithm):
            self.__feasible_algorithm_functions = [algorithm]
        else:
            self.__feasible_algorithm_functions = [scipy.linalg.svd, scipy.sparse.linalg.svds]
            if not exact:
                self.__feasible_algorithm_functions += [sklearn.utils.extmath.randomized_svd]
    @property
    def n_components(self):
        """Return the rank of the singular value decomposition
        This property does the same thing as `k`.

        .. Warning:: Updating :attr:`n_components` does not force a new transform; to obtain a new representation of the data, :meth:`fit` must be called.

        Returns
        -------
        int
        
        Raises
        ------
        NotFittedError
            In the event that `n_components` is not specified on object initialization,
            this attribute is not valid until fit.
        """
        return self.k
    @n_components.setter
    def n_components(self,val):
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
        if self.__k_ is None:
            raise NotFittedError()
        else:
            return self.__k()
    @k.setter
    def k(self, k):
        self.__k(k=k)


    def __k(self, k = None, X = None,suppress=None):
        """
        ### REFACTOR INTO A PROPERTY
        Reset k if necessary and return the rank of the SVD.
        """
        
        if k is None:
            k = self.__k_
        if k is None:
            if hasattr(self,'X_'):
                k = np.min(self.X_.shape)
        if X is None:
            if hasattr(self,'X_'):
                X = self.X_
        if X is not None:
            if k > np.min(X.shape):
                raise ValueError("Specified rank k is greater than the minimum dimension of the input.")
        if k == 0:
            raise ValueError("Cannot compute an SVD with zero components.")
        if k != self.__k_: #removed as this is noisy
            msgs = []
            if self.__k_ is not None: 
                msg = "Updating number of components from k="+str(self.__k_) + " to k=" + str(k)
                level = 2
                msgs.append((msg,level))
            if hasattr(self,'U_'):
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
        ### helper to check k and raise errors when it is bad
        if k is None:
            k = self.k
        else:
            if k > self.k:
                raise ValueError("Requested rank requires a higher rank decomposition. " + 
                    "Re-fit the estimator at the desired rank.")
            if k <=0 : 
                raise ValueError("Cannot use a rank 0 or negative rank.")
        return k

    def fit(self,X = None,k=None,exact=None):
        """Summary
        
        Parameters
        ----------
        X : TYPE
            Description
        k : None, optional
            Description
        exact : None, optional
            Description
        Raises
        -------
        ValueError
        NotFittedError
        RuntimeError
        """

        super().fit()

        if exact is not None:
            self.exact = exact
        if X is None:
            if not hasattr(self, 'X_'):
                if hasattr(self, 'U_') and self.conserve_memory:
                    raise ValueError("SVD has been fit previously, "+
                        "but the input matrix was not stored as SVD.conserve_memory is True. " + 
                        "To perform a new decomposition, call SVD.fit(X) with an input X to be factored. " +
                        "Additionally, set SVD.conserve_memory=True to store X " +
                        "for subsequent decomposition.")
                else:
                    raise NotFittedError('Cannot fit estimator without an input matrix X')
        else:
            self.X_ = X
        self.k = k
        if hasattr(self,'U_') and self.k<=self.U_.shape[1] and X is None:
            msg = ('Requested decomposition appears to be contained ' +
                 'in the previously fitted transform. It may be more efficient to call '+
                 'SVD.transform(k=k) to obtain the new decomposition. To suppress this error '+ 
                 'Set SVD.suppress = True.')
            super().__suppressable_logs__(msg,RuntimeError,suppress=suppress)

        logstr = "rank k=%d %s singular value decomposition using %s."
        logvals = [self.k]
        if self.exact or self.k == np.min(self.X_.shape):
            logvals += ['exact']
        else:
            logvals += ['approximate']
        alg = self.algorithm # this sets the algorithm implicitly, need this first to get to the fname.
        logvals += [self._algorithm.__name__]
        with self.logger.task(logstr % tuple(logvals)):
            U,S,V = alg(X, **self.kwargs)
            ix = np.argsort(S)[::-1]
            self.U_ = U[:,ix]
            self.S_ = S[ix]
            self.V_ = V[ix,:]
            self.V_ = self.V_.T
        if self.conserve_memory:
            del self.X_
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
        -------
        NotFittedError
        """
        check_is_fitted(self)
        k = self.__check_k_(k)
        
        logstr = "rank k = %s approximation of fit data"
        logval = k
        with self.logger.task(logstr%logval):
            return (self.U[:,:k]*self.S[:k]) @ self.V[:,:k].T

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
        -------
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
    default_shrinker : str,optional
        Description
    logger : TYPE
        Description
    M_ : TYPE
        Description
    N_ : TYPE
        Description
    rescale_svs : TYPE
        Description
    y_ : TYPE
        Description
    nuclear
    
    Methods
    -------
    fit_transform : array
        Apply Sinkhorn algorithm and return biscaled matrix
    fit : array
    
    """


    """How many `Shrinker` objects are there?"""
    def __init__(self, default_shrinker = 'frobenius',rescale_svs = True,
        conserve_memory=True, logger = None, verbose=1, suppress=True,
        **kwargs):
        """Summary
        
        
        Parameters
        ----------
        default_shrinker : str, optional
            Description
        rescale_svs : bool, optional
            Description
        verbose : int, optional
            Description
        logger : None, optional
            Description
        
        
        """
        super().__init__(conserve_memory, logger, verbose, suppress,**kwargs)
        self.default_shrinker = default_shrinker
        self.rescale_svs = rescale_svs

    #some properties for fetching various shrinkers when the object has been fitted.
    #these are just wrappers for transform.
    @property 
    def frobenius(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        check_is_fitted(self)
        return self.transform(shrinker = 'fro')
    @property 
    def operator(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        check_is_fitted(self)
        return self.transform(shrinker = 'op')
    @property 
    def hard(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        check_is_fitted(self)
        return self.transform(shrinker = 'hard')
    @property 
    def soft(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        check_is_fitted(self)
        return self.transform(shrinker = 'soft')
    @property 
    def nuclear(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        check_is_fitted(self)
        return self.transform(shrinker = 'nuc')

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
        
        Raises
        ------
        ValueError
            Description
        """
        super().fit()
        if suppress is None:
            suppress = self.suppress
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
                params = self._estimate_MP_params(y=y, N = shape[1], M = shape[0], sigma = sigma, theory_qy = theory_qy, q = q)
                self.sigma_, self.scaled_mp_rank_, self.scaled_cutoff_, self.unscaled_mp_rank_, self.unscaled_cutoff_, self.gamma_, self.emp_qy_, self.theory_qy_, self.q_ , self.scaled_cov_eigs_, self.cov_eigs_ = params
                self.M_ = shape[0]
                self.N_ = shape[1]
                self.y_ = y

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
        
        Raises
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
            gamma = M/N
            rank = (y>=unscaled_cutoff).sum()
            if rank == len(y):
                self.logger.warning("Approximate Marcenko-Pastur rank is full rank")
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
                z = _zero_pad_vec(y,M) #zero pad here for uniformity.
            else:
                z = y
                if q is None:
                    q = 0.5
            if q>=1:
                q = q/100
                assert q<=1
            #grab the empirical quantile.
            emp_qy = np.percentile(z,q*100)
            assert(emp_qy != 0 and emp_qy >= np.min(z))

            #computing the noise variance
            if sigma is None: #precomputed sigma
                if theory_qy is None: #precomputed theory quantile
                    theory_qy = mp_quantile(gamma,q = q,logger=self.logger)
                sigma = emp_qy/np.sqrt(N*theory_qy)
                self.logger.info("Estimated noise variance computed from the {:.0f}th percentile is {:.3f}".format(np.round(q*100),sigma))

            else:
                self.logger.info("Pre-computed noise variance is {:.3f}".format(sigma))
            n_noise = np.sqrt(N)*sigma
            #scaling svs and cutoffs
            scaled_emp_qy = (emp_qy/n_noise)
            cov_eigs = (z/np.sqrt(N))**2
            scaled_cov_eigs = (z/n_noise)
            scaled_cutoff = scaled_mp_bound(gamma)
            scaled_mp_rank = (scaled_cov_eigs>=scaled_cutoff).sum()
            if scaled_mp_rank == len(y):
                self.logger.warning("\n ****** It appears that too few singular values were supplied to Shrinker. ****** \n ****** All supplied singular values are signal. ****** \n ***** It is suggested to refit this estimator with larger `n_components`. ******\n ")
            self.logger.info("Scaled Marcenko-Pastur rank is "+ str(scaled_mp_rank))

        return sigma, scaled_mp_rank, scaled_cutoff, mp_rank, unscaled_cutoff, gamma, emp_qy, theory_qy, q, scaled_cov_eigs**2, cov_eigs


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
# class MeanScaler(BaseEstimator):
#     # """
#     # Adaptive mean-centering and decentering
    
        
#     # Parameters
#     # ----------
#     # masked : bool, optional
#     #     For sparse inputs, preserve the non-zero pattern by centering only the nonzero entries
#     #     (True by default).
#     # verbose : {0, 1, 2}, default 0
#     #     Logging level
#     # logger : :log:`tasklogger.TaskLogger < >`, optional
#     #     Logging object. By default, write to new logger.
#     # **kwargs
#     #     Arguments for downstream SVD algorithm.
        
#     # Attributes
#     # ----------
#     # U : array
#     # S : array
#     # V : array
#     # svd : array
#     # algorithm : callable
#     # k : int
#     # n_components : int
#     # exact : bool
#     # kwargs : dict
#     # conserve_memory : bool
#     # logger : :log:`tasklogger.TaskLogger < >`
#     #     Associated logging objecs
#     # _kwargs : dict
#     #     All SVD-related keyword arguments stored in the object.

    
#     # """
#     # def __init__(self, masked = True):

#     # @property
#     # def mean_rows(self):
#     #     """Return the row mean for the biscaled transform"""
#     #     check_is_fitted(self)
#     #     if self._mean_rows is None:
#     #         self.__set_means(self.Z)
#     #     return self._mean_rows

#     # @property
#     # def mean_cols(self):
#     #     """Return the column mean for the biscaled transform"""
#     #     check_is_fitted(self)
#     #     if self._mean_cols is None:
#     #         self.__set_means(self.Z)
#     #     return self._mean_cols

#     # @property
#     # def mean(self):
#     #     """Return the global mean for the biscaled transform"""
#     #     check_is_fitted(self)
#     #     if self._mean is None:
#     #         self.__set_means(self.Z)
#     #     return self._mean

#     # def __set_means(self,Z):
#     #     """Set the row, column, and global means for the current transformed output"""
#     #     self._mean = np.sum(Z) / np.prod(Z.shape)
#     #     self._mean_cols = np.sum(self.Z,axis=0) / self._N
#     #     self._mean_rows = np.sum(self.Z,axis=1) / self._M
#     # def centering_matrix(self):
#     # def centered_matrix(self,Z=None):
#     #     if Z is None:
#     #         Z = self.Z
#     #     return Z - self.mean_rows[:,None]-self.mean_cols[None,:] - self.mean
#     # def center_matrix(self,Y):
#     #     return Y + self.mean_rows[:,None] +self.mean_cols[None,:]-self.mean
def poisson_variance(X):
    return X
def binomial_variance(X, counts, 
    mult = lambda x,y: X*y, 
    square = lambda x,y: x**2):
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
    var = mult(X,np.divide(counts, counts - 1)) - mult(square(X), (1/(counts-1)))
    var = abs(var)
    return var


def mp_pdf(x, g):
    """Summary
    
    Parameters
    ----------
    x : TYPE
        Description
    g : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    # Marchenko-Pastur distribution pdf
    # g is aspect ratio gamma
    m0 = lambda a: np.maximum(a, np.zeros_like(a))
    gplus=(1+g**0.5)**2
    gminus=(1-g**0.5)**2
    return np.sqrt(  m0(gplus  - x) *  m0(x- gminus)) / ( 2*np.pi*g*x)


# find location of given quantile of standard marchenko-pastur
def mp_quantile(gamma,  q = 0.5, eps = 1E-9,logger = tasklogger, mp = mp_pdf):
    """Compute quantiles of the standard Marchenko-pastur
    
    Compute a quantile `q` from the Marcenko-pastur PDF `mp` with aspect ratio `gamma`
    
    Parameters
    ----------
    gamma : float
        Marcenko-Pastur aspect ratio
    q : float
        Quantile to compute. Must satsify `0 < q < 1`
    eps : float
        Integration tolerance
    logger : {tasklogger, tasklogger.TaskLogger, false}, default tasklogger
        Logging interface.
        tasklogger => Use the default logging parameters as defined by tasklogger module
        False => disable logging.
    
    Returns
    -------
    cent : float
        Computed quantile of Marcenko-Pastur distribution
    
    Other Parameters
    ----------------
    mp : callable, default = bipca.math.mp_pdf
        Marcenko-Pastur PDF accepting two arguments: `x` and `gamma` (aspect ratio)
    
    Examples
    --------
    Compute the median for the Marcenko-Pastur with aspect ratio 0.5:
    
    >>> from bipca.math import mp_quantile
    >>> gamma = 0.5
    >>> quant = mp_quantile(gamma)
    Calculating Marcenko Pastur quantile search...
      Number of MP iters: 28
      MP Error: 5.686854320785528e-10
    Calculated Marcenko Pastur quantile search in 0.10 seconds.
    >>> print(quant)
    0.8304658803921712
    
    Compute the 75th percentile from the same distribution:
    
    >>> q = 0.75
    >>> quant = mp_quantile(gamma, q=q)
    Calculating Marcenko Pastur quantile search...
      Number of MP iters: 28
      MP Error: 2.667510656806371e-10
    Calculated Marcenko Pastur quantile search in 0.11 seconds.
    >>> print(quant)
    1.4859216144349212
    
    Compute the 75th percentile from the same distribution at a lower tolerance:
    
    >>> q = 0.75
    >>> eps = 1e-3
    >>> quant = mp_quantile(gamma, q=q, eps=eps)
    Calculating Marcenko Pastur quantile search...
      Number of MP iters: 8
      MP Error: 0.0009169163809685799
    Calculated Marcenko Pastur quantile search in 0.07 seconds.
    >>> print(quant)
    1.48895145654396
    
    Compute the Marcenko-Pastur median with no logging:
    
    >>> quant = mp_quantile(gamma, q, logger=False)
    >>> print(quant)
    0.8304658803921712
    
    Compute the Marcenko-Pastur median with a custom logger:
    
    >>> import tasklogger
    >>> logger = tasklogger.TaskLogger(name='foo', level=1, timer='wall')
    >>> quant = mp_quantile(gamma, logger=logger)
    Calculating Marcenko Pastur quantile search...
      Number of MP iters: 28
      MP Error: 5.686854320785528e-10
    Calculated Marcenko Pastur quantile search in 0.12 seconds.
    >>> print(quant)
    0.8304658803921712
    >>> logger.set_timer('cpu') ## change the logger to compute cpu time
    >>> quant = mp_quantile(0.5,logger=logger)
    Calculating Marcenko Pastur quantile search...
      Number of MP iters: 28
      MP Error: 5.686854320785528e-10
    Calculated Marcenko Pastur quantile search in 0.16 seconds.
    >>> print(quant)
    0.8304658803921712
    >>> logger.set_level(0) ## mute the logger
    >>> quant = mp_quantile(0.5,logger=logger)
    >>> print(quant)
    0.8304658803921712
    
    
    """
    l_edge = (1 - np.sqrt(gamma))**2
    u_edge = (1 + np.sqrt(gamma))**2
    
    if logger is False:
        loginfo = lambda x: x
        logtask = lambda x: x
    elif logger == tasklogger:
        loginfo = tasklogger.log_info
        logtask = tasklogger.log_task
    else:
        loginfo = logger.info
        logtask = logger.task
    
    # binary search
    nIter = 0
    error = 1
    left = l_edge
    right = u_edge
    cent = left
    with logtask("Marcenko Pastur quantile search"):
        while error > eps:
            cent = (left + right)/2
            val = integrate.quad(lambda x: mp(x, gamma), l_edge, cent)[0]
            error = np.absolute(val - q)
            if val < q:
                left = cent
            elif val > q:
                right = cent
            else:
                # integral is exactly equal to quantile
                return cent
            
            nIter+=1

        loginfo("Number of MP iters: "+ str(nIter))
        loginfo("MP Error: "+ str(error))
        
    return cent


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
        val = integrate.quad(lambda x: loss(x, lambda y: mp(y, gamma), esd), start, end)[0] + err_at_zero
    
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
        val = integrate.quad(lambda x: loss(x, lambda y: mp(y, gamma), esd), 0, end)[0]

    return val


def debias_singular_values(y,m,n,gamma=None, sigma=1):
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

    soft = lambda y: y-scaled_cutoff
    hard = lambda y: y
  
    #compute the scaled svs for shrinking
    n_noise = (np.sqrt(N))*sigma
    scaled_y = unscaled_y / n_noise
    # assign the shrinker
    cond = scaled_y>=scaled_cutoff
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
    scaled_bound = 1+np.sqrt(gamma)
    return scaled_bound
