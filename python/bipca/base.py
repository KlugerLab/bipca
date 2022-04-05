import numpy as np
from sklearn.base import BaseEstimator
from inspect import signature
from collections.abc import Iterable
from itertools import count
from typing import Union
from functools import partial
from dataclasses import dataclass, is_dataclass
from dataclasses import replace as replace_dataclass

import tasklogger
from sklearn import set_config
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from .utils import is_valid, filter_dict,attr_exists_not_none
from functools import wraps
from sklearn.base import clone
from anndata._core.anndata import AnnData

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
        if hasattr(args[0],'fit_'):
            if args[0].fit_:
                return func(*args, **kwargs)
            else:
                raise NotFittedError(f"Requested {type(func).__name__} "
                f"{func.__name__}"
                f" must belong to a fitted parent. "
                f"Try calling {args[0].__class__.__name__}"
                f".fit{signature(args[0].fit)}")
        else:
            raise AttributeError(f"The requested function,{func.__name__} "
            "was decorated as a fitted function, but the parent object, "
            f"{args[0]}, does not have a fitting state attribute.")
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

#### FIELDS FOR DATACLASSES #####

class ValidatedField:
    def __init__(self, typ, validators=(), default=None):
        if not isinstance(typ, type):
            if isinstance(typ, tuple) and all([isinstance(t,type) for t in typ]):
                pass
            else:
                raise TypeError(f"'typ' must be a {type(type)!r} or {type(tuple())!r}` of {type(type)!r}")
        else:
            typ=(typ,)
        self.type = typ
        self.name = f"MyAttr_{self.type!r}"
        self.validators = validators
        self.default=default
        if self.default is not None or type(None) in typ:
            self.__validate__(self.default)
        
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if not instance: return self
        return instance.__dict__[self.name]

    def __delete__(self, instance):
        del instance.__dict__[self.name]
        
    def __validate__(self, value):
        for validator in self.validators:
            validator(self.name, value)
            
    def __set__(self, instance, value):
        if value is self:
            value = self.default
        if not isinstance(value, self.type):
            raise TypeError(f"{self.name!r} values must be of type {self.type!r}")

        instance.__dict__[self.name] = value
    


#### BASE CLASSES #####

class ParameterSet:
    __isfrozen = False
    def __post_init__(self):
        self.__isfrozen=True
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

class _BiPCALogger(tasklogger.TaskLogger):
    def __repr__(self):
        return self.name
class BiPCAEstimator():
    pass
@dataclass
class LoggingParameters(ParameterSet):
    logger: Union[None,_BiPCALogger] = ValidatedField((type(None), _BiPCALogger),[], None)
    verbose: int = ValidatedField(int,[partial(is_valid, lambda x: x>=0)], 1)
    suppress: bool = ValidatedField(bool,[], True)
    relative: Union[BiPCAEstimator,None] = ValidatedField((type(None), BiPCAEstimator),[], None)
@dataclass
class ComputeParameters(ParameterSet):
    conserve_memory: bool = ValidatedField(bool, [], True)
    refit: bool = ValidatedField(bool, [], False)
class BiPCAEstimator(BaseEstimator,BiPCAEstimator):
    _ids=count(0)
    _parameters=['logging_parameters','compute_parameters']
    _backup_parameters=[]
    def __getattribute__(self, attr):
        if attr == '__dict__':
             return super().__getattribute__(attr)
        else:
            if attr in self.__dict__.keys():
                if is_dataclass(self.__dict__[attr]):
                    try:
                        return self.__dict__[attr].__getattribute__(attr)
                    except:
                        return self.__dict__[attr]
            return super().__getattribute__(attr)
        
    def __setattr__(self, attr, value):
        #first, handle dataclass setting & expansion
        if is_dataclass(value):
            super().__setattr__(attr,value) # set the attribute to the dataclass
            if attr in self._parameters: # if the attribute is one of the pre-specified parameters
                # then we expand it by linking this class's attributes to it.
                for field in value.__dataclass_fields__: 
                    self.__dict__[field] = self.__dict__[attr]
        else:
            if attr in self.__dict__: # if the attribute already exists
                if is_dataclass(self.__dict__[attr]): # if the attribute points to a dataclass
                    # then write its new value into the old dataclass
                    setattr(self.__dict__[attr],attr,value)
                else:            
                    super().__setattr__(attr,value)
            else:
                super().__setattr__(attr,value)

    def __expand_parameters__(self,**kwargs):
        dataclass_kwargs = {}
        external_kwargs = {}
        for k,v in kwargs.items():

            if is_dataclass(v):
                dataclass_kwargs[k] = v
            else:
                external_kwargs[k] = v
        for dc,params in dataclass_kwargs.items():
            self.__setattr__(dc, params)
        for attr, val in external_kwargs.items():
            self.__setattr__(attr,val)
    def __grab_backup_parameters__(self):
        if '__backup__' not in self.__dict__:
            self.__dict__['__backup__'] = {}
        for parameter_name in self._backup_parameters:
            self.__dict__['__backup__'][parameter_name] = getattr(self,
                                                            parameter_name)
    def __init__(self, logging_parameters=None, compute_parameters=None, **kwargs):
        if logging_parameters is None:
            logging_parameters = LoggingParameters()
        if compute_parameters is None:
            compute_parameters = ComputeParameters()
        self.__expand_parameters__(logging_parameters=logging_parameters,
                                    compute_parameters=compute_parameters,
                                    **kwargs)
        if self.__ischild:
            self.conserve_memory = self.relative.conserve_memory
            self.suppress = self.relative.suppress
            self.verbose = self.relative.verbose
            self.id = self.relative.id
            self.logger = self.relative.logger
        else:
            self.id = next(self._ids)
            if self.logger == None:
                log_name = self.__class__.__name__ + str(self.id)
                try:
                    self.logger = _BiPCALogger(name=self.__class__.__name__ + str(self.id),level = self.verbose)
                except:
                    self.logger = tasklogger.get_tasklogger(log_name)
        self.fit_ = False
        self.__grab_backup_parameters__()
    @property
    def __ischild(self):
        return isinstance(self.relative, BiPCAEstimator)
    @fitted_property
    @memory_conserved    
    def X(self):
        if hasattr(self, 'X_'):
            if self.X_ is self.relative:
                return self.relative.X
            else:
                return self.X_
        else:
            return self.A.X
    @X.setter
    def X(self, value):
        def set_X(self,value):
            if not self.conserve_memory:
                if isinstance(value, AnnData):
                    self.A_ = value
                else:
                    self.X_ = value
        if self.__ischild: #point to the relative!
            if value is self.relative.X or value is self.relative.A:
                self.A_ = self.relative
                self.X_ = self.relative
            else:
                set_X(self,value)
        else:
            set_X(self,value)
        
    @fitted_property
    @memory_conserved   
    def A(self):
        if hasattr(self,'A_'):
            if self.A_ is self.relative:
                return self.relative.A
            else:
                return self.A_
        else:
            raise AttributeError
    @A.setter
    def A(self, value):
        def set_A(self, value):
            if not self.conserve_memory:
                if isinstance(value, AnnData):
                    self.A_ = value
                else:
                    raise ValueError("Cannot set A with non-AnnData object.")
        if self.__ischild: #point to the relative!
            if value is self.relative.X or value is self.relative.A:
                self.A_ = self.relative
                self.X_ = self.relative
            else:
                set_A(self,value)
        else:
            set_A(self,value)

    def __extract_input_matrix__(self,A):
        if A is not None:
            self.X = A
            if isinstance(A, AnnData):
                X = A.X
            else:
                X = A
                A = None
            return X, A
        else:
            return None, None
        
    def reset_estimator(self,inplace=False):
        for parameter_name in self._backup_parameters:
            setattr(self, parameter_name, 
                    self.__dict__['__backup__'][parameter_name])
        obj=clone(self)
        if inplace:
            self.__dict__=obj.__dict__
            return self
        return obj
    def fit(self):
        self.fit_=True
        return self
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
                if backend_val in ['','scipy', 'torch', 'torch_cpu', 'torch_gpu']:
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