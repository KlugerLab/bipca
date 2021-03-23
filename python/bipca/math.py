"""Summary
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import scipy.integrate as integrate
import scipy.sparse as sparse
import tasklogger
from .utils import _is_vector, _xor, _zero_pad_vec

class Sinkhorn(BaseEstimator):
    """
    Sinkhorn Algorithm implementation. 
    ...
    Parameters
    ----------
    var : ndarray or None, Default None.
        variance matrix for input data
        None = > binomial count model estimates underlying variance.
    row_sums : ndarray or None, default None
        None    => Target row sums are 1.
        ndarray => Target row sums are row_sums.
    col_sums : ndarray or None, default None
        None    => Target col sums are 1.
        ndarray => Target col sums are col_sums.
    read_counts : ndarray or None, default None
        vector of total counts of each column, or alternatively the expected counts of each column
        Bone    => read_counts = column sums of X.
    tol : float, default 1e-6
        Sinkhorn tolerance
    n_iter : int, default 30
        Number of Sinkhorn iterations.
    return_scalers : bool, default False
        Return left and right scaling vectors from transform methods.
    force_sparse : bool, default False
        False   => maintain input data type (ndarray or sparse matrix)
        True    => impose sparsity on inputs and outputs; 
    verbose : {0, 1, 2}, default 0
        Logging level
    logger : tasklogger.TaskLogger, optional
        Logging object. By default, write to new logger.
    Attributes
    ----------

    left_ : ndarray
        Left scaling vector
    right_ : ndarray
        Right scaling vector
    X_ : ndarray
        Input data
    var : ndarray
        variance matrix for input data
    col_sums : ndarray or None
        Target column sums
    row_sums : ndarray or None
        Target row sums
    read_counts : ndarray or None
        vector of total counts of each column, or alternatively the expected counts of each column
    tol : float
        Sinkhorn tolerance
    n_iter : int
        Sinkhorn iterations
    return_scalers : bool, Default True
        Return scaling vectors from Sinkhorn.transform
    return_errors : bool, Default False
        Return the Sinkhorn biscaling errors from transform methods
    force_sparse : bool
        Maintain input data type
    verbose : int
        Logging level
    logger : tasklogger.TaskLogger
        Logging object
    
    """
    log_instance = 0
    def __init__(self,  var = None, 
        row_sums = None, col_sums = None, read_counts = None, tol = 1e-6, 
        n_iter = 30, return_scalers = True,  force_sparse = False, return_errors = False, verbose=1, logger = None):
        self.return_scalers = return_scalers
        self.read_counts = read_counts


        self.row_sums = row_sums
        self.col_sums = col_sums
        self.tol = tol
        self.n_iter = n_iter
        self.return_errors = return_errors
        self.force_sparse = force_sparse
        self.verbose = verbose
        self.var = None

        self._issparse = None
        self.__typef_ = lambda x: x #we use this for type matching in the event the input is sparse.
        if logger == None:
            Sinkhorn.log_instance += 1
            self.logger = tasklogger.TaskLogger(name='Sinkhorn ' + str(Sinkhorn.log_instance),level = verbose)
        else:
            self.logger = logger


    def __is_valid(self, X,row_sums,col_sums):
        """Verify input data is non-negative and shapes match.
        
        Parameters
        ----------
        X : ndarray
        row_sums : ndarray
        col_sums : ndarray
        """
        eps = 1e-3
        assert np.amin(X) >= 0, "Matrix is not non-negative"
        assert np.shape(X)[0] == np.shape(row_sums)[0], "Row dimensions mismatch"
        assert np.shape(X)[1] == np.shape(col_sums)[0], "Column dimensions mismatch"
        
        # sum(row_sums) must equal sum(col_sums), at least approximately
        assert np.abs(np.sum(row_sums) - np.sum(col_sums)) < eps, "Rowsums and colsums do not add up to the same number"
    
    @property
    def Z(self):
        """ndarray: Biscaled matrix
        """
        check_is_fitted(self)
        Z = self.__type(self.scale())
        return Z

    def __type(self,M):
        """ Typecast data matrix M based on fitted type __typef_
        """
        check_is_fitted(self)
        if self.force_sparse and not self._issparse:
            return sparse.csr_matrix(M)
        else:
            if isinstance(M, type(self.X_)):
                return M
            else:
                return self.__typef_(M)

    def fit_transform(self, X = None, return_scalers = None, return_errors = None):
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
        else:
            self.fit(X)
        return self.transform(X=X, return_scalers = return_scalers, return_error = return_errors)

    
    def transform(self, X = None, return_scalers = None, return_errors= None):
        """ Scale the input by left and right Sinkhorn vectors.  Compute 
        
        Parameters
        ----------
        X : None, optional
            Input matrix to scale
        
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
        with self.logger.task('Transform'):
            if X is None:
                output = (self.Z,)
            else:
                output = self.__type(self.scale(X))
            if return_scalers:
                output += (self.left_,self.right_)
            if return_errors:
                output += (self.row_error_, self.col_error_)
        return output
    @property
    def left(self):
        """ndarray: left scaling vector"""
        check_is_fitted(self)
        return self.left_
    @property
    def right(self):
        check_is_fitted(self)
        return self.right_

    def scale(self,X = None):
        """Rescale matrix by Sinkhorn scalers.
        Estimator must be fit.
        
        Parameters
        ----------
        X : ndarray, optional
            Matrix to rescale by Sinkhorn scalers.
        
        Returns
        -------
        ndarray
            Matrix scaled by Sinkhorn scalerss.
        """
        check_is_fitted(self)
        if X is None:
            X = self.X_
        return self.__mem(self.__mem(X,self.right),self.left[:,None])

    def unscale(self, X=None):
        """Applies inverse Sinkhorn scalers to input X.
        Estimator must be fit.
        Parameters
        ----------
        X : ndarray, optional
            Matrix to unscale
        
        Returns
        -------
        ndarray
            Matrix unscaled by the inverse Sinkhorn scalers
        """
        check_is_fitted(self)
        if X is None:
            return self.X_
        return self.__mem(self.__mem(X,1/self.right),1/self.left[:,None])

    def fit(self, X):
        """Summary
        
        Parameters
        ----------
        X : TYPE
            Description
        """
        with self.logger.task('Fit'):

            self._issparse = sparse.issparse(X)
            if self.force_sparse and self._issparse:
                X = sparse.csr_matrix(X)
            self.__set_operands(X)

            self._M = X.shape[0]
            self._N = X.shape[1]

            row_sums, col_sums = self.__compute_dim_sums()
            self.__is_valid(X,row_sums,col_sums)

            if self.var is None:
                var, rcs = self.__variance(X)
            else:
                var = self.var

            l,r,re,ce = self.__sinkhorn(var,row_sums, col_sums)
            # now set the final fit attributes.
            self.X_ = X
            self.var = var
            self.read_counts = rcs
            self.row_sums = row_sums
            self.col_sums = col_sums
            self.left_ = np.sqrt(l)
            self.right_ = np.sqrt(r)
            self.row_error_ = re
            self.col_error_ = ce

    def __set_operands(self, X):
        """Summary
        
        Parameters
        ----------
        X : TYPE
            Description
        """
        # changing the operators to accomodate for sparsity 
        # allows us to have uniform API for elemientwise operations

        if self._issparse:
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

    def __variance(self, X):
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
        var = binomial_variance(X,read_counts,
            mult = self.__mem, square = self.__mesq)
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
        with self.logger.task("Sinkhorn iteration"): 
            if n_iter is None:
                n_iter = self.n_iter
            a = np.ones(n_row,)
            for i in range(n_iter):
                b = np.divide(col_sums, X.T.dot(a))
                a = np.divide(row_sums, X.dot(b))    
            a = np.asarray(a).flatten()
            b = np.array(b).flatten()
            if self.tol>0:
                ZZ = (self.var * self.right_**2)* self.left_[:,None]**2
                row_error  = np.amax(np.abs(self._M - ZZ.sum(0)))
                col_error =  np.amax(np.abs(self._N - ZZ.sum(1)))
                if row_error > self.tol:
                    self.logger.warning("Col error: " + str(row_error)
                        + " exceeds requested tolerance: " + str(self.tol))
                if col_error > self.tol:
                    self.logger.warning("Row error: " + str(col_error)
                        + " exceeds requested tolerance: " + str(self.tol))
            
        return a, b, row_error, col_error


class Shrinker(BaseEstimator):
    """
    Optimal Shrinkage class
    ...
    Parameters
    ----------
    default_shrinker : str, default "frobenius"
        shrinker to use when Shrinker.transform is called with no argument `shrinker`.
        Must satisfy `default_shrinker in ['frobenius','fro','operator','op','nuclear','nuc','hard','hard threshold','soft','soft threshold']`
    rescale_svs : bool, optional
        Description
    verbose : int, optional
        Description
    logger : None, optional
        Description
    Attributes
    ----------
    default_shrinker : str,optional
        Description
    log_instance : int
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
    
    Methods
    -------
    fit_transform : ndarray
        Apply Sinkhorn algorithm and return biscaled matrix
    fit : ndarray
    
    """
    log_instance = 0
    def __init__(self, default_shrinker = 'frobenius',rescale_svs = True,verbose = 1,logger = None):
        """Summary
        

        """
        self.default_shrinker = default_shrinker
        self.rescale_svs = rescale_svs
        if logger == None:
            Shrinker.log_instance += 1
            self.logger = tasklogger.TaskLogger(name='Shrinker ' + str(Shrinker.log_instance),level = verbose)
        else:
            self.logger = logger
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

    def fit(self, y, shape=None, sigma = None, theory_qy = None, q = None):
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
        try:
            check_is_fitted(self)
            try:
                assert np.allclose(y,self.y_) #if this fails, then refit
            except: 
                self.logger.info("Refitting to new input y")
                raise
        except:
            with self.logger.task("Fit"):
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
        rank = np.where(y < unscaled_cutoff)[0]
        if np.shape(rank)[0] == 0:
            self.logger.warning("Approximate Marcenko-Pastur rank is full rank")
        else:
            self.logger.info("Approximate Marcenko-Pastur rank is "+ str(rank[0]))
        mp_rank = rank
        #quantile finding and setting
        with self.logger.task("MP Parameter estimate"):
            ispartial = len(y)<M
            if ispartial:
                tasklogger.info("A fraction of the total singular values were provided")
                assert mp_rank!= len(y) #check that we have enough to compute a quantile
                if q is None: 
                    q = (M - (len(y)-1))/M # grab the smallest value in y
                y = _zero_pad_vec(y,M) #zero pad here for uniformity.
            else:
                if q is None:
                    q = 0.5
            if q>=1:
                q = q/100
                assert q<=1
            #grab the empirical quantile.
            emp_qy = np.percentile(y,q*100)
            assert(emp_qy != 0 and emp_qy >= np.min(y))

            #computing the noise variance
            if sigma is None: #precomputed sigma
                if theory_qy is None: #precomputed theory quantile
                    theory_qy = mp_quantile(gamma,q = q,logger=self.logger)
                sigma = emp_qy/np.sqrt(N*theory_qy)
            n_noise = np.sqrt(N)*sigma
            #scaling svs and cutoffs
            scaled_emp_qy = (emp_qy/n_noise)
            cov_eigs = (y/np.sqrt(N))**2
            scaled_cov_eigs = (y/n_noise)
            scaled_cutoff = scaled_mp_bound(gamma)
            scaled_mp_rank = (scaled_cov_eigs>=scaled_cutoff).sum()
            self.logger.info("Estimated noise variance is "+ str(sigma))
            self.logger.info("Scaled Marcenko-Pastur rank is "+ str(scaled_mp_rank))

        return sigma, scaled_mp_rank, scaled_cutoff, mp_rank, unscaled_cutoff, gamma, emp_qy, theory_qy, q, scaled_cov_eigs**2, cov_eigs


    def fit_transform(self, y = None, shape = None, shrinker = None):
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
            return  _optimal_shrinkage(y, self.sigma_, self.N_, self.gamma_, scaled_cutoff = self.scaled_cutoff_,shrinker  = shrinker,rescale=rescale)


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

def _optimal_shrinkage(unscaled_y, sigma, N, gamma, scaled_cutoff = None, shrinker = 'frobenius',rescale=True,logger=None):
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
