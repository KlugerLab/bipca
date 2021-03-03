import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import scipy.sparse as sparse
import tasklogger

class Sinkhorn(BaseEstimator):
    """
    Sinkhorn Algorithm implementation. 
    ...

    Attributes
    ----------
    return_scalers: bool, Default True
        Return scaling vectors from Sinkhorn.transform
    var: ndarray or None, Default None. 
        variance matrix for input data
       None = > binomial count model estimates underlying variance.
    read_counts:  ndarray or None, default None 
        vector of total counts of each column, or alternatively the expected counts of each column
        None    => read_counts = column sums of X.
    row_sums : ndarray or None, default None
        None    => Target row sums are 1.
        ndarray => Target row sums are row_sums.
    col_sums : ndarray or None, default None
        None    => Target col sums are 1.
        ndarray => Target col sums are col_sums.
    tol :   float, default 1e-6
        Sinkhorn tolerance
    n_iter : int, default 30
        Number of Sinkhorn iterations.
    force_sparse : bool, default False
        False   => maintain input data type (ndarray or sparse matrix)
        True    => impose sparsity on inputs and outputs; 
    verbose : {0, 1, 2}, default 0
        Logging level

    Methods
    -------
    fit_transform : ndarray
        Apply Sinkhorn algorithm and return biscaled matrix
    fit : ndarray 


    """
    def __init__(self,  return_scalers = True, var = None, read_counts = None,
        row_sums = None, col_sums = None, tol = 1e-6,
        n_iter = 30, force_sparse = False, verbose=0):
        self.return_scalers = return_scalers
        self.read_counts = read_counts


        self.row_sums = row_sums
        self.col_sums = col_sums
        self.tol = tol
        self.n_iter = n_iter
        self.force_sparse = force_sparse
        self.verbose = verbose
        self.var = None

        self._issparse = None
        self.__type = lambda x: x #we use this for type matching in the event the input is sparse.
        tasklogger.set_level(verbose)

    def __is_valid(self, X,row_sums,col_sums):
        eps = 1e-3
        assert np.amin(X) >= 0, "Matrix is not non-negative"
        assert np.shape(X)[0] == np.shape(row_sums)[0], "Row dimensions mismatch"
        assert np.shape(X)[1] == np.shape(col_sums)[0], "Column dimensions mismatch"
        
        # sum(row_sums) must equal sum(col_sums), at least approximately
        assert np.abs(np.sum(row_sums) - np.sum(col_sums)) < eps, "Rowsums and colsums do not add up to the same number"
    
    @property
    def Z(self):
        return self.transform()
    
    def fit_transform(self, X = None):
        if X is None:
            check_is_fitted(self)
        else:
            self.fit(X)
        return self.transform()

    
    def transform(self, X = None):
        check_is_fitted(self)
        if X is None:
            X = self.X_
        with tasklogger.log_task('Transform'):
            Z = self.__mem(self.__mem(X,self.right_),self.left_[:,None])
            ZZ = self.__mem(self.__mem(Z,self.right_),self.left_[:,None])
            row_error  = np.amax(np.abs(self._N - ZZ.sum(1)))
            col_error =  np.amax(np.abs(self._M - ZZ.sum(0)))
            if row_error > self.tol:
                tasklogger.log_warning("Row error: " + str(row_error)
                    + " exceeds requested tolerance: " + str(self.tol))
            if col_error > self.tol:
                tasklogger.log_warning("Column error: " + str(col_error)
                    + " exceeds requested tolerance: " + str(self.tol))
            if self.return_scalers:
                Z = (self.__type(Z),self.left_,self.right_)
            return Z

    def fit(self, X):
        with tasklogger.log_task('Fit'):

            self._issparse = sparse.issparse(X)
            if self.force_sparse and self._issparse:
                X = sparse.csr_matrix(X)
            self.__set_operands(X)

            self._M = X.shape[0]
            self._N = X.shape[1]

            row_sums, col_sums = self.__compute_row_sums()
            self.__is_valid(X,row_sums,col_sums)

            if self.var is None:
                var, rcs = self.__variance(X)
            else:
                var = self.var

            l,r = self.__sinkhorn(var,row_sums, col_sums)
            # now set the final fit attributes.
            self.X_ = X
            self.var = var
            self.read_counts = rcs
            self.row_sums = row_sums
            self.col_sums = col_sums
            self.left_ = np.sqrt(l)
            self.right_ = np.sqrt(r)

    def __set_operands(self, X):
        # changing the operators to accomodate for sparsity 
        # allows us to have uniform API for elemientwise operations

        if self._issparse:
            self.__type = type(X)
            self.__mem = lambda x,y : x.multiply(y)
            self.__mesq = lambda x : x.power(2)
        else:
            self.__mem= lambda x,y : np.multiply(x,y)
            self.__mesq = lambda x : np.square(x)
    def __compute_dim_sums(self):
            if self.row_sums is None:
                row_sums = np.full(self._M, self._N)
            else:
                row_sums = self.row_sums
            if self.col_sums is None:
                col_sums = np.full(self._N, self._M)
            else:
                col_sums = self.col_sums
        return row_sums, col_sums
    def __variance(self, X):
        read_counts = np.sum(X, axis = 0)
        var = binomial_variance(X,read_counts,
            mult = self.__mem, square = self.__mesq)
        return var,read_counts

    def __sinkhorn(self, X, row_sums, col_sums, n_iter = None):
        """
        Execute Sinkhorn algorithm X mat, row_sums,col_sums for n_iter
        """
        n_row = X.shape[0]
        with tasklogger.log_task("Sinkhorn iteration"): 
            if n_iter is None:
                n_iter = self.n_iter
            a = np.ones(n_row,)
            for i in range(n_iter):
                b = np.divide(col_sums, X.T.dot(a))
                a = np.divide(row_sums, X.dot(b))    
            a = np.asarray(a).flatten()
            b = np.array(b).flatten()
        return a, b

def binomial_variance(X, counts, 
    mult = lambda x,y: X*y, 
    square = lambda x,y: x**2):
    """
        Estimated variance under the binomial count model.
    """
    var = mult(X,np.divide(counts, counts - 1)) - mult(square(X), (1/(counts-1)))
    var = abs(var)
    return var