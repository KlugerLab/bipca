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
 	var: ndarray or None, Default None. 
 		variance matrix for input data
       None = > binomial count model estimates underlying variance.
	read_counts:  ndarray or None, default None 
		vector of total counts of each column, or alternatively the expected counts of each column
        None 	=> read_counts = column sums of X.
	row_sums : ndarray or None, default None
		None 	=> Target row sums are 1.
		ndarray => Target row sums are row_sums.
	col_sums : ndarray or None, default None
		None 	=> Target col sums are 1.
		ndarray => Target col sums are col_sums.
    tol : 	float, default 1e-6
    	Sinkhorn tolerance
	n_iter : int, default 30
		Number of Sinkhorn iterations.
    force_sparse : bool, default False
    	False 	=> maintain input data type (ndarray or sparse matrix)
    	True 	=> impose sparsity on inputs and outputs; 
	verbose : {0, 1, 2}, default 0
		Logging level

    Methods
    -------
    fit_transform : ndarray or scipy.sparse.csr_matrix
        Apply Sinkhorn algorithm and return biscaled matrix
    fit : ndarray or scipy.sparse.csr_matrix


    """
	def __init__(self,  var = None, read_counts = None, 
		row_sums = None, col_sums = None, tol = 1e-6,
		n_iter = 30, force_sparse = False, verbose=0):

		self.read_counts = read_counts
		self.issparse = None
		self.row_sums = row_sums
		self.col_sums = col_sums
		self.tol = tol
		self.n_iter = n_iter
		self.force_sparse = force_sparse
		self.verbose = verbose
		self.var = None
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

	
	def transform(self):
		check_is_fitted(self)
		with tasklogger.task('Transform')
		    Z = (self.X_ * self.right_) * self.left_[:,None]
		    ZZ = Z * self.right_ * self.left_[:,None]
			row_error  = np.amax(np.abs(self._N - np.sum(ZZ, axis = 1)))
			col_error =  np.amax(np.abs(self._M - np.sum(ZZ, axis = 0)))
			if row_error > self.tol:
				logger.log_warning("Row error: " + str(row_error)
					+ " exceeds requested tolerance: " + self.tol)
			if col_error > self.tol:
				logger.log_warning("Column error: " + str(col_error)
					+ " exceeds requested tolerance: " + self.tol)
			return Z

    def fit(self, X):
    	with tasklogger.task('Fit'):
			self.issparse = sparse.issparse(X)
			self._M = X.shape[0]
			self._N = X.shape[1]
			if self.row_sums is None:
				row_sums = np.full(self._N, self._M)
			else:
				row_sums = self.row_sums
			if self.col_sums is None:
				col_sums = np.full(self._M, self._N)
			else:
				col_sums = self.col_sums
			else:
			self.__check_valid(X,row_sums,col_sums)
			if self.var is None:
				var, rcs = self.__variance(X)
			else:
				var = self.var
			l,r = self.__fit(var,row_sums, col_sums)
			self.X_ = X
			self.var = var
			self.read_counts = rcs
			self.row_sums = row_sums
			self.col_sums = col_sums
			self.left_ = np.sqrt(l)
			self.right_ = np.sqrt(r)

	def __variance(self, X):

		read_counts = np.sum(X, axis = 0)
		if not self.issparse:
			var = dense_binomial_variance(X,counts)
		return var,read_counts

	def __sinkhorn(self, X, row_sums, col_sums, n_iter = None):
	'''
		Execute Sinkhorn algorithm X mat, row_sums,col_sums for n_iter
	'''
		with tasklogger.log_task("Sinkhorn iteration"):	
			if n_iter is None:
				n_iter = self.n_iter
			if not self.issparse:
			    a = np.ones(n_row)
			    for i in range(nIter):

			        b = np.divide(col_sums, X.T @ a)
			        a = np.divide(row_sums, X @ b)    
	    return a, b

def dense_binomial_variance(X, counts):
	'''
		Estimated variance under the binomial count model.
	'''
	var = X * np.divide(counts, counts - 1) - X**2 * (1/(counts-1))
	var = np.abs(var)
	return var