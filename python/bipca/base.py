import numpy as np
from sklearn.base import BaseEstimator
from collections.abc import Iterable
from itertools import count
import tasklogger
from sklearn.base import BaseEstimator
from sklearn import set_config
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from .utils import filter_dict,attr_exists_not_none,make_tensor,issparse,make_scipy
from functools import wraps
from sklearn.base import clone
from anndata._core.anndata import AnnData
from torch import Tensor
##### MODULE LEVEL PARAMETERS ######
set_config(print_changed_only=False)


##### DECORATORS USED THROUGH BIPCA ###### 


def memory_conserved(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        if args[0].conserve_memory:
            args[0].__suppressable_log__('`conserve_memory=True`,' +
            ' so '+func.__name__+' is returned as `None`. ' +
            '\n This warning may be disabled by letting suppress=True.',
            level=0,suppress=args[0].suppress)
            result=None
        else:
            result = func(*args,**kwargs)
        return result
    return wrapper

def fitted(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        #This needs to be unified across all fit states. Current implementation is fucked
        #try to return the property and catch attribute errors with a notfittederror
        try: 
            return func(*args,**kwargs)
        except AttributeError:
            raise NotFittedError
        # if hasattr(args[0],'fit_'):
        #     if args[0].fit_:
        #         return func(*args, **kwargs)
        #     else:
        #         raise NotFittedError
    return wrapper

def memory_conserved_property(func):
    return property(memory_conserved(func))
def fitted_property(func):
    return property(fitted(func))


def stores_to_ann(f_py = None, prefix = '', target = ''):
    #decorator to make properties automatically store to an associated anndata object, if there is one

    #https://stackoverflow.com/a/60832711
    assert callable(f_py) or f_py is None

    if prefix == '':
        storefun = lambda obj,func, args: store_ann_attr(obj.A,func.__name__, args[1], 
            prefix=obj.__class__.__name__, target = target)
    else:
        storefun = lambda obj,func, args: store_ann_attr(obj.A,func.__name__, args[1], 
            prefix=prefix, target = target)
    def _decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            obj = args[0]
            if attr_exists_not_none(obj,'A_'):
                storefun(obj,func,args)
            return func(*args,**kwargs)
        return wrapper
    return _decorator(f_py) if callable(f_py) else _decorator



#### BASE CLASSES #####

class _BiPCALogger(tasklogger.TaskLogger):
    def __repr__(self):
        return self.name

class BiPCAEstimator(BaseEstimator):
    _ids=count(0)

    def __init__(self, conserve_memory=True, logger = None, verbose=1, suppress=True, relative = None, **kwargs):
        if isinstance(relative, BiPCAEstimator):
            self.conserve_memory = relative.conserve_memory
            self.suppress = relative.suppress
            self.verbose = relative.verbose
            self.id = relative.id
            self.logger = relative.logger
        else:
            self.conserve_memory = conserve_memory
            self.suppress = suppress
            self.verbose = verbose
            self.id = next(self._ids)
            if logger == None:
                log_name = self.__class__.__name__ + str(self.id)
                try:
                    self.logger = _BiPCALogger(name=self.__class__.__name__ + str(self.id),level = self.verbose)
                except:
                    self.logger = tasklogger.get_tasklogger(log_name)
            else:
                self.logger = logger
        self.fit_ = False
        self._clone = None
## data / input stuff
    def process_input_data(self,A):
        if isinstance(A, AnnData):
            X=A.X
        else:
            X = A
        if self.backend=='torch':
            if issparse(X):
                self.logger.warning('Sparse tensors are not supported. Switching to sparse scipy backend. \n'
                                    'If you wish to use torch as a backend, supply a dense matrix.')
                self.backend='scipy'
                X = make_scipy(X)
            else:
                if isinstance(A,AnnData) and not isinstance(A.X, Tensor):
                    self.logger.warning('Tensor backend was specified with AnnData input. \n'
                                    'This will generate a new matrix in memory that is not bound to A.X \n'
                                    'To conserve memory, consider pre-converting A.X to a tensor.')
                X = make_tensor(X)
        self.X = X
        self.A = A
        return X, A


    @memory_conserved_property
    def X(self):
        return self.X_
    @X.setter
    def X(self, value):
        if not self.conserve_memory:
            if self.backend=='torch':
                self.X_=make_tensor(value)
            else:
                self.X_ = value
    @memory_conserved_property
    def A(self):
        return self.A_
    @A.setter
    def A(self, value):
        if not self.conserve_memory:
            if isinstance(value, AnnData):
                self.A_ = value
##
    ###backend properties and resetters##
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
        if attr_exists_not_none(self,'_backend'):
            if val != self._backend:
                if attr_exists_not_none(self,'_svd_backend'): #we check for none here. If svd backend is none, it follows self.backend, and there's no need to warn.
                    if val != self.svd_backend:
                        self.logger.warning("Changing the global backend is overwriting the SVD backend. \n" + 
                            "To change this behavior, set the global backend first by obj.backend = 'foo', then set obj.svd_backend.")
                        self.svd_backend=val
                if attr_exists_not_none(self,'_sinkhorn_backend'):
                    if val != self.sinkhorn_backend:
                        self.logger.warning("Changing the global backend is overwriting the sinkhorn backend. \n" + 
                            "To change this behavior, set the global backend first by obj.backend = 'foo', then set obj.sinkhorn_backend.")
                        self.sinkhorn_backend=val
                #its a new backend
                self._backend = val
        else:
            self._backend=val
        self.reset_backend()
    def reset_backend(self):
        """Summary
        """
        #Must be called after setting backends.
        attrs = self.__dict__.keys()
        for attr in attrs:
            obj = self.__dict__[attr]
            objname =  obj.__class__.__name__.lower()
            if hasattr(obj, 'backend'):
                #the object has a backend attribute to set
                if 'svd' in objname:
                    obj.backend = self.svd_backend
                elif 'sinkhorn' in objname:
                    obj.backend = self.sinkhorn_backend
                else:
                    obj.backend = self.backend

##
    def reset_estimator(self):
        return clone(self,safe=True)
    def fit(self):
        pass
    def __suppressable_logs__(self,msgs,level = None,suppress = None):
        if suppress is None:
            suppress = self.suppress
        if level is None:
            #we need to get the levels from the list
            if not isinstance(msgs,Iterable):
                # the msgs are either a single entry or invalid
                if isinstance(msgs, str):#BLOCKID#1
                     #single entry with no level 
                    #still pass the message, but default to warning
                    msgs = msgs + "\n This message was passed as a warning as no level was supplied"
                    self.__suppressable_log__(msgs,suppress=suppress)
                else: # invalid, neither a string or an iterable
                    raise ValueError("The supplied message was neither an iterable or a string.")
            else: #we either got an iterator of tuples or a tuple
                if len(msgs)<1: #empty iterator
                    return 0 
                if len(msgs) == 1: #hopefully a tuple; call again. If it's a list [str] 
                    # then recursion will go to the previous BLOCKID#1
                    self.__suppressable_logs__(msgs[0])
                else: #we either have a tuple or a list
                    if isinstance(msgs,tuple):
                        self.__suppressable_logs__(msgs[0],level=msgs[1])
                    elif isinstance(msgs,list): #we have a list. we need to verify it contains tuples
                        if not all([isinstance(ele,tuple) for ele in msgs]):
                            #verify they are tuples
                            raise ValueError("Inputs must be a list of msg,level tuples")
                        if not all([isinstance(ele[0],str) and (ele[1] in [1,2,3] or ele[1] in ['warning','debug','info'] or isinstance(ele[1], Exception)) for ele in msgs]):
                            #verify we have msg,level pairs.
                            raise ValueError("Invalid msg,level pairs supplied")
                        (self.__suppressable_logs__(msgs[0][0], level = msgs[0][1], suppress=suppress), 
                            self.__suppressable_logs__(msgs[1:], suppress=suppress))
                    else:
                        raise ValueError("Invalid msgs iterable suplied to log.")


        else: # the logging level is uniform for the whole set of msgs
            if not isinstance(msgs,str):
                if isinstance(msgs,Iterable):
                    # if it is iterable, we will just recurse through the msgs
                    (self.__suppressable_logs__(msgs[0],level = level,suppress = suppress) ,
                        self.__suppressable_logs__(msgs[1:],level=level, suppress=suppress))
                #at this point we should only be getting strings!
                raise ValueError("Messages can only be supplied as strings.")
            self.__suppressable_log__(msgs,level = level,suppress = suppress)

    def __suppressable_log__(self,msg,level = None,suppress = None):
        if suppress is None:
            suppress = self.suppress
        if not suppress:
            if level == 0 or level == 'warning' or level == None:
                self.logger.warning('****WARNING*****: '+msg)
            elif level == 1 or level == 'info':
                self.logger.info(msg)
            elif level == 2 or level == 'debug':
                self.logger.debug(msg)
            elif isinstance(level, Exception):
                raise level(msg)

    def isvalid_backend(self,backend_val):
        if backend_val is None:
            return backend_val
        else:
            if isinstance(backend_val,str):
                ## valid backend_val
                backend_val = backend_val.lower()
                if backend_val in ['','scipy', 'torch', 'torch_cpu', 'torch_gpu', 'dask']:
                    return backend_val
                else:
                    raise_error = True
            else:
                raise_error = True
        if raise_error:
            raise ValueError("Backend values must be None or a string in {'', 'torch', 'torch_cpu', 'torch_gpu', 'dask'}")
        else:
            return backend_val

def store_ann_attr(adata, attr,val, prefix = '', target = None):
    if isinstance(adata, AnnData):
        if target is None or target == '':
            if hasattr(val,'shape'):
                if np.any(adata.n_vars in val.shape): 
                    if np.any(adata.n_obs in val.shape):
                        target = 'layers'
                    elif np.any( 1 in val.shape) or len(val.shape) == 1:
                        target = 'var'
                    else:
                        target = 'varm'
                elif np.any(adata.n_obs in val.shape):
                    if np.any(1 in val.shape) or len(val.shape) == 1:
                        target = 'obs'
                    else:
                        target = 'obsm'
        if target is None or target == '':
            target = 'uns'
        if target == 'uns':
            if prefix not in getattr(adata,target).keys():
                getattr(adata,target)[prefix] = {}
            getattr(adata,target)[prefix][attr] = val
        else:
            if prefix != '':
                prefix = prefix +'_'
            write_str = prefix+attr
            getattr(adata,target)[write_str] = val