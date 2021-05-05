import numpy as np
from sklearn.base import BaseEstimator
from collections.abc import Iterable
from itertools import count
import tasklogger
from sklearn.base import BaseEstimator
from sklearn import set_config
from .utils import filter_dict
from functools import wraps
from sklearn.base import clone

def __memory_conserved__(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        if args[0].conserve_memory:
            args[0].__suppressable_log__('`conserve_memory=True`,' +
            ' so '+func.__name__+' is returned as `None`. ' +
            '\n This warning may be disabled by letting suppress=True.',
            level=0,suppress=args[0].suppress)

        result = func(*args,**kwargs)
        return result
    return wrapper

set_config(print_changed_only=False)

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
        self._clone = None
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

