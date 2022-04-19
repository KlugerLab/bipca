"""Subroutines used to compute a BiPCA transform
"""
from collections.abc import Iterable
from collections import defaultdict
from dataclasses import dataclass

from typing import Union,Optional
from functools import partial
from numbers import Number
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
from .utils import (get_args,
                    is_valid,
                    all_equal,
                    zero_pad_vec,
                    filter_dict,
                    ischanged_dict,
                    nz_along,
                    make_tensor,
                    issparse, 
                    attr_exists_not_none)
from .base import *

class QuadraticParameters:
    """
    Store and convert between quadratic variance function (QVF) formulations.

    This class converts the coefficients of the three equivalent \
    |QVFS|:

    **Formulation 1 - convex formulation**:
       |QVF1|,

    **Formulation 2 - theoretical formulation**:
        |QVF2|, and 

    **Formulation 3 - numerical formulation**:
        |QVF3|.
        


    These parameters are accessible as runtime-computed attributes. 

    .. Warning:: For some values of `q`, `sigma`, `bhat`,or `chat`, the \
        theoretical coefficients are not well-behaved. In these regimes, \
        `b` and `c` will return |None|, although \
        `q`, `sigma`, `bhat`,or `chat`  are defined and can be used in their \
        respective |QVFS|

    - On initialization, the class will compute the parameter set from all \
    keyword arguments and attempt to validate them against one another. If \
    conflicting parameter sets are supplied, the class throws \
    |AssertionError|.
    - At least two parameters are required to compute the attributes of the \
    entire class, however certain pairs are invalid. 
    - If insufficient parameters are available, unsupplied attributes will \
        return |None|.
    - Outside of initialization, this class uses an update stack in order to \
        keep track of parameters. When an attribute is updated by the user, \
        the new attribute is pushed onto the update stack. If the stack is longer than 2\
        parameters, the stack is truncated to either 2 or 3 parameters: 3 if \
        the latest 2 members of the stack contain both `b` and `q` or `sigma`, \
        and 2 otherwise. Then, all non-stacked attributes in the class are recomputed \
        to ensure that the new parameter set agrees. One can disable this behavior \
        by toggling `~bipca.math.QuadraticParameters.update` ``=`` |False|\
        during initialization to allow multiple parameters to be set \
        simultaneously. `~bipca.math.QuadraticParameters.update` ``=`` |True| during normal \
        runtime.


    Parameters
    ----------
    q : |Number|, optional
        Convex coefficient for formulation 1. Must satisfy :math:`q\in[0,1]`.

    sigma : |Number|, optional
        Noise deviation coefficient for formulation 1. Must satisfy \
        :math:`\sigma>0`

    b : |Number|, optional
        Theoretical linear coefficient for formulation 2.

    c : |Number|, optional
        Theoretical quadratic coefficient for formulation 2.

    bhat : |Number|, optional
        Experimental linear coefficient for formulation 3.

    chat : |Number|, optional
        Experimental quadratic coefficient for formulation 3.

    Raises
    ------
    |AssertionError|
        Raised when multiple parameters are set that do not agree.

    """
    def __init__(self,q: Optional[Number]=None,
                sigma: Optional[Number]=None,
                b: Optional[Number]=None,
                bhat: Optional[Number]=None,
                c: Optional[Number]=None,
                chat: Optional[Number]=None):
        self.__update_stack__=[]
        self.update=False
        self.q=q
        self.sigma=sigma
        self.b=b
        self.bhat=bhat
        self.c=c
        self.chat=chat

        
        for param in ['_q','_sigma','_b','_bhat','_c','_chat']:
            if getattr(self, param) is not None:
                self.__update_stack__.append(param)
        self.update=True

    @property
    def update(self):
        """The update behavior of a :class:`~bipca.math.QuadraticParameters` instance.

        |False| during initialization and |True| afterwards, when \
        `update` ``is`` |True|, the :class:`~bipca.math.QuadraticParameters` instance maintains \
        an update stack \
        and validates the stack when a new parameter is set. When \
        `update` ``is`` |False|, \
        no parameters are checked, though an update stack is kept unmonitored.\
        The `update` ``is`` |False| state allows one to change many \
        attributes at once \
        without |AssertionError|.
        
        Returns
        -------
        :class:`bool`
            The update state.

        Raises
        ------
        |TypeError|
            `~bipca.math.QuadraticParameters.update` must be a :class:`bool`
        """
        if not attr_exists_not_none(self,'_update'):
            self._update=True
        return self._update
    
    @update.setter
    def update(self,val:bool):

        if not isinstance(val, bool):
            raise TypeError("QuadraticParameters.update must be a boolean.")
        if self.update != val:
            if val == True:
                self.compute(q=self._q,sigma=self._sigma,b=self._b,
                bhat=self._bhat,c=self._c,chat=self._chat)
            else:
                pass
            self._update=val


    def __reset_parameters__(self,ignore=None):
        parameters=['_q','_sigma','_b','_bhat','_c','_chat']
        if not isinstance(ignore, list):
            ignore=[ignore]
        for param in parameters:
            if param not in ignore:
                setattr(self,param,None)

    def __update__(self,attr):
        if len(self.__update_stack__)>0:
            if self.__update_stack__[-1] == attr:
                self.compute()
                return
            elif attr in ['_c','_chat'] \
                and self.__update_stack__[-1] in ['_c','_chat']:
                setattr(self,self.__update_stack__[-1],None)
                self.__update_stack__[-1]=attr
            else:
                self.__update_stack__.append(attr)
        else:
            self.__update_stack__.append(attr)
        if self.update is True:
            if '_b' in self.__update_stack__[-2:]:
                if '_sigma' in self.__update_stack__[-2:] or \
                    '_q' in self.__update_stack__[-2:]:
                    self.__update_stack__=self.__update_stack__[-3:]
                else:
                    self.__update_stack__=self.__update_stack__[-2:]
            else:
                self.__update_stack__=self.__update_stack__[-2:]
            
            self.__reset_parameters__(ignore=self.__update_stack__)
            self.compute()
    @property
    def q(self):
        """The convex coefficient in the convex formulation (#1) of a quadratic \
        variance function.

        Computes (if not specified) or accesses the convex coefficient `q` for \
        the convex :abbr:`QVF (quadratic variance function)` formulation:

        :math:`\widehat{\mathtt{var}}[Y_{ij}]=\sigma^2\left((1-q)Y_{ij}+qY_{ij}^2\\right)`.

        `q` must be a  |Number| satisfying :math:`q \in [0,1]` \
        or |None|.

        .. Warning:: When `~bipca.math.QuadraticParameters.update` ``is`` |True|, this \
        property pushes onto the update stack, releasing all but the most \
        recent 2-3 (depending on identity) quadratic parameters from the \
        stack. See `update` for details.

        Returns
        -------
        |Number| or |None|
            If `q` is computable, a |Number| is returned, otherwise |None|.

        Raises
        ------
        |ValueError|
            If `q` is a |Number| outside of :math:`[0,1]`
        |TypeError|
            If `q` is neither a |Number| nor |None|.

        """
        if attr_exists_not_none(self,'_q'):
            return self._q
        else:
            return self.compute_q(q=None,
                                sigma=self._sigma,
                                b=self._b,
                                bhat=self._bhat,
                                c=self._c,
                                chat=self._chat
                                )
    @q.setter
    def q(self,value:Union[None, Number]):
        if isinstance(value,(type(None),Number)):
            if value is None or (value <= 1 and value >= 0):
                self._q = value
            else:
                raise ValueError("QuadraticParameters.q must be in [0,1]"
                " or None.")
        else:
            raise TypeError("QuadraticParameters.q must be non-negative " 
            "Number of None.")
        if self._q is not None:
            self.__update__('_q')

    @property
    def sigma(self):
        """The noise deviation coefficient in the convex formulation (#1)\
        of a quadratic variance function.

        Computes (if not specified) or accesses the noise deviation \
        coefficient `sigma` \
        for the convex :abbr:`QVF (quadratic variance function)` formulation:

        :math:`\widehat{\mathtt{var}}[Y_{ij}]=\sigma^2\left((1-q)Y_{ij}+qY_{ij}^2\\right)`.

        `sigma` must be a  |Number| satisfying :math:`\sigma>0` \
        or |None|.

        .. Warning:: When `~bipca.math.QuadraticParameters.update` ``is`` |True|, this \
        property pushes onto the update stack, releasing all but the most \
        recent 2-3 (depending on identity) quadratic parameters from the \
        stack. See `update` for details.

        Returns
        -------
        |Number| or |None|
            If `sigma` is computable, a |Number| is returned, otherwise |None|.
            
        Raises
        ------
        |ValueError|
            If `sigma` is a non-positive |Number|.
        |TypeError|
            If `sigma` is neither a |Number| nor |None|.
        """
        if attr_exists_not_none(self,'_sigma'):
            return self._sigma
        else:
            return self.compute_sigma(q=self._q,
                                sigma=None,
                                b=self._b,
                                bhat=self._bhat,
                                c=self._c,
                                chat=self._chat
                                )
    @sigma.setter
    def sigma(self,value:Union[None, Number]):
        if isinstance(value,(type(None),Number)):
            if value is None or value > 0:
                self._sigma = value
            else:
                raise ValueError("QuadraticParameters.sigma must be positive"
                " or None.")
        else:
            raise TypeError("QuadraticParameters.sigma must be positive " 
            "Number of None.")
        if self._sigma is not None:
            self.__update__('_sigma')

    @property
    def b(self):
        """The linear coefficient in the theoretical formulation (#2)\
        of a quadratic variance function.

        Computes (if not specified) or accesses the linear coefficient `b` \
        for the theoretical |QVF| \
        formulation:

        |QVF2|

        `b` must be a  |Number| or |None|.

        .. Warning:: For some values of `q`, `sigma`, `bhat`,or `chat`, the \
        theoretical coefficients are not well-behaved. In these regimes, \
        `b` and `c` will return |None|, although \
        `q`, `sigma`, `bhat`,or `chat`  are defined and can be used in their \
        respective |QVFS|

        .. Warning:: When `~bipca.math.QuadraticParameters.update` ``is`` |True|, this \
        property pushes onto the update stack, releasing all but the most \
        recent 2-3 (depending on identity) quadratic parameters from the \
        stack. See `update` for details.

        Returns
        -------
        |Number| or |None|
            If `b` is computable, a |Number| is returned, otherwise |None|.

        Raises
        ------
        |TypeError|
            If `b` is neither a |Number| nor |None|.
        """
        if attr_exists_not_none(self,'_b'):
            return self._b
        else:
            return self.compute_b(q=self._q, 
                                  sigma=self._sigma,
                                  b=None,
                                  bhat=self._bhat,
                                  c=self._c,
                                  chat=self._chat)
    @b.setter
    def b(self,value:Union[None, Number]):
        if isinstance(value,(type(None),Number)):
            self._b = value
        else:
            raise TypeError("QuadraticParameters.b must be " 
            "Number of None.")
        if self._b is not None:
            self.__update__('_b')

    @property
    def bhat(self):
        """The linear coefficient in the numerical formulation (#3)\
        of a quadratic variance function.

        Computes (if not specified) or accesses the linear coefficient `bhat` \
        for the numerical |QVF| \
        formulation:

        |QVF3|

        `bhat` must be a  |Number| or |None|.

        .. Warning:: When `~bipca.math.QuadraticParameters.update` ``is`` |True|, this \
        property pushes onto the update stack, releasing all but the most \
        recent 2-3 (depending on identity) quadratic parameters from the \
        stack. See `update` for details.

        Returns
        -------
        |Number| or |None|
            If `bhat` is computable, a |Number| is returned, otherwise |None|.

        Raises
        ------
        |TypeError|
            If `bhat` is neither a |Number| nor |None|.
        """
        if attr_exists_not_none (self,'_bhat'):
            return self._bhat
        else:
            return self.compute_bhat(q=self._q, 
                                  sigma=self._sigma,
                                  b=self._b,
                                  bhat=None,
                                  c=self._c,
                                  chat=self._chat)
    @bhat.setter
    def bhat(self,value:Union[None, Number]):
        if isinstance(value,(type(None),Number)):
            self._bhat = value
        else:
            raise TypeError("QuadraticParameters.bhat must be " 
            "Number or None.")         
        if self._bhat is not None:
            self.__update__('_bhat')

    @property
    def c(self):
        """The quadratic coefficient in the theoretical formulation (#2)\
        of a quadratic variance function.

        Computes (if not specified) or accesses the quadratic coefficient `c` \
        for the theoretical |QVF| \
        formulation:

        |QVF2|

        `c` must be a  |Number| or |None|.

        .. Warning:: For some values of `q`, `sigma`, `bhat`,or `chat`, the \
        theoretical coefficients are not well-behaved. In these regimes, \
        `b` and `c` will return |None|, although \
        `q`, `sigma`, `bhat`,or `chat`  are defined and can be used in their \
        respective |QVFS|

        .. Warning:: When `~bipca.math.QuadraticParameters.update` ``is`` |True|, this \
        property pushes onto the update stack, releasing all but the most \
        recent 2-3 (depending on identity) quadratic parameters from the \
        stack. See `update` for details.

        Returns
        -------
        |Number| or |None| 
            If `c` is computable, a |Number| is returned, otherwise |None|.
            
        Raises
        ------
        |TypeError|
            If `c` is neither a |Number| nor |None|.
        """
        if attr_exists_not_none(self,'_c'):
            return self._c
        else:
            return self.compute_c(q=self._q, 
                                  sigma=self._sigma,
                                  b=self._b,
                                  bhat=self._bhat,
                                  c=None,
                                  chat=self._chat)
    @c.setter
    def c(self,value):
        if isinstance(value,(type(None),Number)):
            self._c = value
        else:
            raise TypeError("QuadraticParameters.c must be " 
            "Number or None.")               
        if self._c is not None:
            self.__update__('_c')

    @property
    def chat(self:Union[None, Number]):
        """The quadratic coefficient in the numerical formulation (#3)\
        of a quadratic variance function.

        Computes (if not specified) or accesses the quadratic coefficient `chat` \
        for the numerical |QVF| \
        formulation:

        |QVF3|

        `chat` must be a  |Number| or |None|.

        .. Warning:: When `~bipca.math.QuadraticParameters.update` ``is`` |True|, this \
        property pushes onto the update stack, releasing all but the most \
        recent 2-3 (depending on identity) quadratic parameters from the \
        stack. See `update` for details.

        Returns
        -------
        |Number| or |None| 
            If `chat` is computable, a |Number| is returned, otherwise |None|.
            
        Raises
        ------
        |TypeError|
            If `chat` is neither a |Number| nor |None|.
        """
        if attr_exists_not_none (self,'_chat'):
            return self._chat
        else:
            return self.compute_chat(q=self._q, 
                                  sigma=self._sigma,
                                  b=self._b,
                                  bhat=self._bhat,
                                  c=self._c,
                                  chat=None)
    @chat.setter
    def chat(self,value):
        if isinstance(value,(type(None),Number)):
            self._chat = value
        else:
            raise TypeError("QuadraticParameters.chat must be " 
            "Number or None.")     
        if self._chat is not None:
            self.__update__('_chat')


    def compute(self, q:Optional[Number]=None,
                sigma:Optional[Number]=None,
                b:Optional[Number]=None,
                bhat:Optional[Number]=None,
                c:Optional[Number]=None,
                chat:Optional[Number]=None):
        """Compute and validate the parameters for every quadratic variance \
        function formulation.

        .. Warning:: For some values of `q`, `sigma`, `bhat`,or `chat`, the \
        theoretical coefficients are not well-behaved. In these regimes, \
        `b` and `c` will return |None|, although \
        `q`, `sigma`, `bhat`,or `chat`  are defined and can be used in their \
        respective |QVFS|

        If feasible, this function computes `q`, `sigma`, `b`, `bhat`, `c`, and \
        `chat` using optional arguments. If an argument is not supplied, the \
        method uses any parameters stored in the class.

        If a parameter is supplied (or retrieved from the instance) \
        and additionally computable using other parameters, the input \
        is validated by comparison against the value computed using the other \
        parameters.

        .. Warning:: This method does not update `~bipca.math.QuadraticParameters` \
        attributes in place.


        Returns
        -------
        (q, sigma, b, bhat, c, chat) : \
(|Number| or |None|, |Number| or |None|, |Number| or |None|, \
|Number| or |None|, |Number| or |None|, |Number| or |None|)
            The computed parameters or |None| if insufficiently many arguments \
            were supplied to complete the parameter set.

            
        Raises
        ------
        |AssertionError|
            If an input parameter is not compatible with other inputs.
        """
        if q is None:
            q = self._q
        if sigma is None:
            sigma = self._sigma
        if b is None:
            b = self._b
        if bhat is None:
            bhat = self._bhat
        if c is None:
            c = self._c
        if chat is None:
            chat = self._chat
        qout = QuadraticParameters.compute_q(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat)
        assert q is None or np.isclose(qout,q), \
            'Input q does not match other parameters.'
        q=qout 

        sigmaout = QuadraticParameters.compute_sigma(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat)
        assert sigma is None or np.isclose(sigmaout,sigma), \
            'Input sigma does not match other parameters.'
        sigma=sigmaout 

        chatout = QuadraticParameters.compute_chat(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat)
        assert chat is None or np.isclose(chatout,chat), \
            'Input chat does not match other parameters.'
        chat=chatout
        cout = QuadraticParameters.compute_c(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat)
        assert c is None or np.isclose(cout, c), \
            'Input c does not match other parameters.'
        c = cout
        bhatout = QuadraticParameters.compute_bhat(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat)
        assert bhat is None or np.isclose(bhatout, bhat), \
            'Input bhat does not match other parameters.'
        bhat = bhatout

        bout = QuadraticParameters.compute_b(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat)
        assert b is None or np.isclose(bout, b), \
            'Input b does not match other parameters'
        b = bout

        return q,sigma,b,bhat,c,chat
    @staticmethod
    def compute_q(q=None,
                sigma=None,
                b=None,
                bhat=None,
                c=None,
                chat=None):
        """Compute the convex coefficient ``q`` (:math:`q`).

        The convex coefficient :math:`q` is used in the |QVF| formulation:
        
        |QVF1|.

        If feasible, this static method computes :math:`q`. Otherwise, \
        it returns |None|. 
        
        .. note:: This is a static method. If an argument is not supplied, \
        the method does NOT use any parameters stored in the class.

        If ``q`` is supplied as an argument\
        and additionally computable using other arguments, the input \
        is validated by comparison against the value computed using the other \
        arguments.

        .. Warning:: This method does not update `~bipca.math.QuadraticParameters` \
        attributes in place.

        .. seealso:: `bipca.math.QuadraticParameters.q`

        Returns
        -------
        q : |Number| or |None|
            The convex coefficient or |None| if \
            insufficiently many arguments were supplied to compute :math:`q`, \
            or |None| if :math:`q` is not well-defined as a function of the \
            the input arguments.

            
        Raises
        ------
        |AssertionError|
            If a computed :math:`q` does not match other computed :math:`q` s, i.e., \
            the input arguments did not yield a consistent set of parameters.
        """
        answers = []
        if q is not None:
            answers.append(q)
        nargs = [ele is None for ele in [q,sigma,b,bhat,c,chat]]
        if all(nargs) or len(nargs)-np.sum(nargs) == 1:
            pass
        else:
            
            if sigma is not None: 
                if bhat is not None:
                    answers.append(1-bhat/(sigma**2))

                if c is not None:
                    answers.append( c / ( (1+c) *sigma**2 ) )

                if chat is not None:
                    answers.append(chat/(sigma**2))

            if b is not None:
                if bhat is not None: #this was solved for by combining
                    #b(q,sigma), c(q,sigma), bhat(q,sigma)
                    if not np.isclose(b,0) and np.isclose(bhat,0): #the limit as bhat -> 0
                        answers.append(1)
                    elif np.isclose(b,0) and not np.isclose(bhat,0): #the limit as b->0
                        answers.append(1)
                    elif np.isclose(b,0) and np.isclose(b,0): 
                        #the simultaneous limit is indeterminate, but the nested one isn't.
                        #can we assume the nested limit?
                        answers.append(1)
                    else:
                        answers.append(((-1*bhat)+b) / ((-1*bhat)+b+bhat*b))


                if c is not None: 
                    answers.append(QuadraticParameters.compute_q(b=b,
                                    chat=QuadraticParameters.compute_chat(b=b,c=c)))

                if chat is not None: #found by combining 
                    #b(q,sigma) and chat(q,sigma)

                    answers.append( (chat - b*chat ) / ( b + chat - b*chat - \
                                    b * (b + chat - b * chat ) ))

            if bhat is not None:
                if c is not None: #combining
                #bhat(q,sigma) and c(q,sigma)
                    answers.append( -1 * ( ( 1 + c ) * \
                    ( bhat - ( bhat + c + bhat * c ) / ( 1 + c ) ) ) / \
                    ( bhat + c + bhat * c ) )

                if chat is not None:
                    answers.append(chat/(bhat+chat))
        answers = list(filter(lambda ele: ele is not None, answers))
        if len(answers)==0:
            return None
        assert all_equal(answers), str(answers)
        return answers[0]

    @staticmethod
    def compute_sigma(q=None,
                sigma=None,
                b=None,
                bhat=None,
                c=None,
                chat=None):
        """Compute the noise deviation ``sigma`` (:math:`\sigma`).

        The noise deviation :math:`\sigma` is used in the |QVF| formulation:
        
        |QVF1|.


        If feasible, this static method computes :math:`\sigma`. Otherwise, \
        it returns |None|.

        .. note:: This is a static method. If an argument is not supplied, \
        the method does NOT use any parameters stored in the class.

        If ``sigma`` is supplied as an argument \
        and additionally computable using other arguments, the input \
        is validated by comparison against the value computed using the other \
        arguments.

        .. Warning:: This method does not update `~bipca.math.QuadraticParameters` \
        attributes in place.

        .. seealso:: `bipca.math.QuadraticParameters.sigma`

        Returns
        -------
        sigma : |Number| or |None|
            The noise deviation coefficient or |None| if \
            insufficiently many arguments were supplied to compute 
            :math:`\sigma`, \
            or |None| if :math:`\sigma` is not well-defined as a function of the \
            the input arguments.
            
        Raises
        ------
        |AssertionError|
            If a computed :math:`\sigma` does not match other computed \
            :math:`\sigma` s, i.e., \
            the input arguments did not yield a consistent set of parameters.
        """
        answers = []
        if sigma is not None:
            answers.append(sigma)
        nargs = [ele is None for ele in [q,sigma,b,bhat,c,chat]]
        if all(nargs) or len(nargs)-np.sum(nargs) == 1:
            pass
        else:
            if q is not None:
                if b is not None:
                    if np.isclose(q,1):
                        answers.append(None)
                    elif np.isclose(b,0) and not np.isclose(q,1):
                        answers.append(0)
                    else:
                        answers.append(  np.abs(np.real(1j*np.sqrt(0j+b) / np.sqrt( -1+q - b*q +0j)) ))

                if bhat is not None:
                    if np.isclose(q,1) and np.isclose(bhat,0):
                        pass #indeterminate
                    elif q==1 and not np.isclose(bhat,0):
                        answers.append(np.sign(bhat)*np.Inf)
                    elif not np.isclose(q,1) and np.isclose(bhat,0):
                        answers.append(0)
                    else:
                        answers.append(np.sqrt(bhat/(1-q)))

                if c is not None:
                    if q==0 and np.isclose(c,np.Inf):
                        answers.append(None)
                    elif q==0:
                        if c!=0:
                            answers.append(np.Inf)
                        else:
                            answers.append(None)
                    elif np.isclose(c,np.Inf) and q!=0:
                        answers.append(np.Inf)
                    else:
                        answers.append(np.real(np.sqrt(c+0j) / np.sqrt( q + c * q + 0j)))

                if chat is not None:
                    if q==0:
                        if not np.isclose(chat,0):
                            answers.append(np.Inf)
                        else:
                            answers.append(None)
                    else:
                        answers.append(np.sqrt(chat)/np.sqrt(q))

            if b is not None:
                if np.isclose(b,np.Inf):
                    answers.append(np.Inf)
                if bhat is not None:
                    if np.isclose(b,0) and np.isclose(bhat,0):
                        pass #indeterminate
                    elif np.isclose(b,0) and not np.isclose(bhat,0):
                        answers.append(np.sqrt( -1 * np.sign(bhat) * np.Inf))
                    elif not np.isclose(b,0) and np.isclose(bhat,0):
                        answers.append(1)
                    else:
                        answers.append(np.real(np.sqrt(-1*bhat+b+bhat*b+0j)/np.sqrt(b+0j)))
                if c is not None:
                    if np.isclose(c,np.Inf):
                        answers.append(np.Inf)
                    else:
                        answers.append( np.sqrt( (b+c) / (1+c) ))
                if chat is not None:
                    answers.append( np.sqrt( b + chat - b * chat) ) 
                
            if bhat is not None:
                if c is not None:
                    answers.append( np.sqrt( (bhat + c + bhat * c ) / ( 1 + c )))
                if chat is not None:
                    answers.append(np.sqrt(chat+bhat))

        answers = list(filter(lambda ele: ele is not None, answers))
        if len(answers)==0:
            return None
        assert all_equal(answers), str(answers)
        return answers[0]

    @staticmethod
    def compute_b(q = None,
                sigma=None,
                b = None,
                bhat=None,
                c = None,
                chat=None):
        """Compute the theoretical linear coefficient ``b`` (:math:`b`).

        The theoretical linear coefficient :math:`b` is used in the |QVF| \
        formulation:
        
        |QVF2|.

        .. Warning:: For some values of ``q``, ``sigma``, ``bhat``,or ``chat``, \
        the \
        theoretical coefficients are not well-behaved. In these regimes, \
        `compute_b` and `compute_c` will return |None|, although \
        ``q``, ``sigma``, ``bhat``, and ``chat``  are defined and can be used \
        in their \
        respective |QVFS|

        If feasible, this static method computes :math:`b`. Otherwise, \
        it returns |None|.

        .. note:: This is a static method. If an argument is not supplied, \
        the method does NOT use any parameters stored in the class.

        If ``b`` is supplied as an argument\
        and additionally computable using other arguments, the input \
        is validated by comparison against the value computed using the other \
        arguments.

        .. Warning:: This method does not update `~bipca.math.QuadraticParameters` \
        attributes in place.

        .. seealso:: `bipca.math.QuadraticParameters.b`

        Returns
        -------
        b : |Number| or |None|
            The theoretical linear coefficient or |None| if \
            insufficiently many arguments were supplied to compute :math:`b`, \
            or |None| if :math:`b` is not well-defined as a function of the \
            the input arguments.
            
        Raises
        ------
        |AssertionError|
            If a computed :math:`b` does not match other computed :math:`b` s, \
            i.e., \
            the input arguments did not yield a consistent set of parameters.
        """
        answers = []
        if b is not None:
            answers.append(b)
        nargs = [ele is None for ele in [q,sigma,bhat,c,chat]]
        if all(nargs) or len(nargs)-np.sum(nargs) == 1:
            pass
        else:
            if all(ele is not None for ele in [q,sigma]):
                if q==1 and not np.isclose(q*sigma**2,1):
                    answers.append(0)
                elif np.isclose(q*sigma**2, 1):
                    answers.append(None)
                elif q*sigma**2 > 1:
                    answers.append(QuadraticParameters.compute_b(
                        bhat=QuadraticParameters.compute_bhat(q=q,sigma=sigma),
                        chat=QuadraticParameters.compute_chat(q=q,sigma=sigma)
                    ))
                else:
                    answers.append( ((1-q) * sigma**2) / (1- q*sigma**2 ))

            if bhat is not None:
                if np.isclose(bhat,0):
                    answers.append(0)
                else:
                    if c is not None:
                        answers.append(bhat*(1+c))
                    if chat is not None:
                        if np.isclose(chat,1):
                            answers.append(None)
                        else:                
                            answers.append(bhat*(chat/(1-chat) + 1))

            if (len(answers) - int(b is not None)) < 1:
                answers.append(QuadraticParameters.compute_b(
                                q=QuadraticParameters.compute_q(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat),
                                sigma=QuadraticParameters.compute_sigma(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat),
                ))
            
        answers = list(filter(lambda ele: ele is not None, answers))
        if len(answers)==0:
            return None
        assert all_equal(answers), str(answers)
        return answers[0]

    @staticmethod
    def compute_bhat(q=None,
                    sigma=None,
                    b=None,
                    bhat=None,
                    c=None,
                    chat=None):
        """Compute the numerical linear coefficient ``bhat`` (:math:`\hat{b}`).

        The numerical linear coefficient :math:`\hat{b}` is used in the |QVF| \
        formulation:
        
        |QVF3|.

        If feasible, this static method computes :math:`\hat{b}` Otherwise, \
        it returns |None|.

        .. note:: This is a static method. If an argument is not supplied, \
        the method does NOT use any parameters stored in the class.

        If ``bhat`` is supplied as an argument\
        and additionally computable using other arguments, the input \
        is validated by comparison against the value computed using the other \
        arguments.

        .. Warning:: This method does not update `~bipca.math.QuadraticParameters` \
        attributes in place.

        .. seealso:: `bipca.math.QuadraticParameters.bhat`


        Returns
        -------
        bhat : |Number| or |None|
            The numerical linear coefficient or |None| if \
            insufficiently many arguments were supplied to compute \
            :math:`\hat{b}`, \
            or |None| if `:math:`\hat{b}` is not well-defined as a function  \
            of the input arguments.
            
        Raises
        ------
        |AssertionError|
            If a computed :math:`\hat{b}` does not match other computed \
            `:math:`\hat{b}`` s, i.e., \
            the input arguments did not yield a consistent set of parameters.
        """
        answers = []
        if bhat is not None:
            answers.append(bhat)
        nargs = [ele is None for ele in [q,sigma,b,c,chat]]
        if all(nargs) or len(nargs)-np.sum(nargs) == 1:
            pass
        else:
            if all(ele is not None for ele in [q,sigma]):
                answers.append((1-q)*sigma**2)
            if all(ele is not None for ele in [b, c]):
                answers.append(b / (1+c))
            if (len(answers) - int(bhat is not None)) < 1:
                answers.append(QuadraticParameters.compute_bhat(
                                q=QuadraticParameters.compute_q(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat),
                                sigma=QuadraticParameters.compute_sigma(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat),
                ))
        answers = list(filter(lambda ele: ele is not None, answers))
        if len(answers)==0:
            return None
        assert all_equal(answers), str(answers)
        return answers[0]

    @staticmethod
    def compute_c(q=None,
                  sigma=None,
                  b=None,
                  bhat=None,
                  c=None,
                  chat=None):
        """Compute the theoretical quadratic coefficient ``c`` (:math:`c`).

        The theoretical quadratic coefficient :math:`c` is used in the |QVF| \
        formulation:
        
        |QVF2|.

        .. Warning:: For some values of ``q``, ``sigma``, ``bhat``, \
        or ``chat``, the \
        theoretical coefficients are not well-behaved. In these regimes, \
        `compute_b` and `compute_c` will return |None|, although \
        ``q``, ``sigma``, ``bhat``, and ``chat``  are defined and can be used \
        in their respective |QVFS|.

        If feasible, this static method computes :math:`c`. Otherwise, \
        it returns |None|.

        .. note:: This is a static method. If an argument is not supplied, \
        the method does NOT use any parameters stored in the class.

        If ``c`` is supplied as an argument\
        and additionally computable using other arguments, the input \
        is validated by comparison against the value computed using the other \
        arguments.

        .. Warning:: This method does not update `~bipca.math.QuadraticParameters` \
        attributes in place.

        .. seealso:: `bipca.math.QuadraticParameters.c`

        Returns
        -------
        c : |Number| or |None|
            The theoretical quadratic coefficient or |None| if \
            insufficiently many arguments were supplied to compute :math:`c`, \
            or |None| if :math:`c` is not well-defined as a function of the \
            the input arguments.
            
        Raises
        ------
        |AssertionError|
            If a computed :math:`c` does not match other computed :math:`c` s, i.e., \
            the input arguments did not yield a consistent set of parameters.
        """
        answers = []
        if c is not None:
            answers.append(c)
        nargs = [ele is None for ele in [q,sigma,b,bhat,chat]]

        if all(nargs) or len(nargs)-np.sum(nargs) == 1:
            pass
        else:
            if all(ele is not None for ele in [q,sigma]):
                if (q==1 and sigma==1):
                    answers.append(None)  
                elif q==0 and np.isclose(sigma,np.Inf):
                    answers.append(0)
                elif np.isclose(q*sigma**2,1):
                    answers.append(None)
                elif q*sigma**2 > 1:
                    answers.append(QuadraticParameters.compute_c(
                        bhat=QuadraticParameters.compute_bhat(q=q,sigma=sigma),
                        chat=QuadraticParameters.compute_chat(q=q,sigma=sigma)
                    ))
                else:
                    answers.append((q*sigma**2)/(1-q*sigma**2))
            
            if chat is not None:
                if np.isclose(chat,1):
                    answers.append(None)
                else:
                    answers.append(chat/(1-chat))
            if (len(answers) - int(b is not None)) < 1:
                answers.append(QuadraticParameters.compute_c(
                                q=QuadraticParameters.compute_q(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat),
                                sigma=QuadraticParameters.compute_sigma(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat),
                ))
        answers = list(filter(lambda ele: ele is not None, answers))
        if len(answers)==0:
            return None
        assert all_equal(answers), str(answers)
        return answers[0]


    @staticmethod
    def compute_chat(q=None,
                    sigma=None,
                    b=None,
                    bhat=None,
                    c=None,
                    chat=None):
        """Compute the numerical quadratic coefficient ``chat`` (:math:`\hat{c}`).

        The numerical quadratic coefficient :math:`\hat{c}` is used in the |QVF| formulation:
        
        |QVF3|.

        If feasible, this static method computes :math:`\hat{c}`. Otherwise, \
        it returns |None|.
        

        .. note:: This is a static method. If an argument is not supplied, \
        the method does NOT use any parameters stored in the class.

        If ``chat`` is supplied as an argument\
        and additionally computable using other arguments, the input \
        is validated by comparison against the value computed using the other \
        arguments.

        .. Warning:: This method does not update `~bipca.math.QuadraticParameters` \
        attributes in place.

        .. seealso:: `bipca.math.QuadraticParameters.chat`

        Returns
        -------
        chat : |Number| or |None|
            The numerical quadratic coefficient :math:`\hat{c}` or |None| if \
            insufficiently many arguments were supplied to compute \
            :math:`\hat{c}`, \
            or |None| if :math:`\hat{c}` is not well-defined as a function of the \
            the input arguments.
            
        Raises
        ------
        |AssertionError|
            If a computed :math:`\hat{c}` does not match other computed \
            :math:`\hat{c}` s, i.e., \
            the input arguments did not yield a consistent set of parameters.
        """
        answers = []
        if chat is not None:
            answers.append(chat)
        nargs = [ele is None for ele in [q,sigma,b,bhat,c]]
        if all(nargs) or len(nargs)-np.sum(nargs) == 1:
            pass
        else:
            if all(ele is not None for ele in [q,sigma]):
                answers.append(q*sigma**2)
            
            if c is not None:
                if np.isclose(c,np.Inf):
                    answers.append(1)
                else:
                    answers.append( c / (1+c) )
            if (len(answers) - int(chat is not None)) < 1:
                answers.append(QuadraticParameters.compute_chat(
                                q=QuadraticParameters.compute_q(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat),
                                sigma=QuadraticParameters.compute_sigma(q=q,
                                sigma=sigma,
                                b=b,
                                bhat=bhat,
                                c=c,
                                chat=chat),
                ))

    
        answers = list(filter(lambda ele: ele is not None, answers))
        if len(answers)==0:
            return None
        assert all_equal(answers), str(answers)
        return answers[0]

class Sinkhorn(BiPCAEstimator):
    """
    Sinkhorn biwhitening and biscaling. 
    """

    @dataclass
    class FitParameters(ParameterSet):
        """Dataclass that houses the fitting parameters of \
        :class:`bipca.math.Sinkhorn`. 

        Parameters
        ----------
            variance : {np.ndarray, sparse matrix} of shape (M, N), optional.
                variance matrix for input data to be biscaled
                (default variance is estimated from data using the model).
            
            variance_estimator : {'binomial', 'quadratic','precomputed',\
'general', None}, default 'quadratic'.
                The variance estimator to use. Determines the biwhitened \
                or biscaled output.  Default ``'quadratic'``.
                
                - If `variance_estimator` ``=='binomial'``, uses a binomial \
                model according to `read_counts`. 
                - If `variance_estimator` ``=='quadratic'``, uses either \
                the convex or 2 parameter model,\
                (depending on which of `q`, `sigma`, `b`, `bhat`, `c`, `chat`) \
                are supplied. 
                - If `variance_estimator` ``=='general'``, use the empirical \
                variance of the input.
                - If `variance_estimator` ``=='precomputed'``, uses the precomputed \
                variance contained in `variance`.
                - If `variance_estimator` ``is None``, Sinkhorn biscaling \
                is performed, rather than biwhitening.
                
            row_sums : Number or np.ndarray of shape (M,), optional.
                Legacy parameter for pre-set row sums in Sinkhorn optimization.

            col_sums : Number or np.ndarray of shape (N,), optional.
                Legacy parameter for pre-set column sums in Sinkhorn \
                optimization.

            read_counts : Number, optional.
                Read counts of binomial distribution. "Number of coin flips". \
                Used when `variance_estimator` ``=='binomial'``.
                Defaults to the sum along the columns of the input.

            q : Number, optional.
                Convex parameter for quadratic variance. \
                Defaults to 0, i.e. the variance is entirely linear in the mean.\
                Used when 
                :math:`\widehat{\mathtt{var}}[Y_{ij}]=\sigma((1-q)Y_{ij}+qY_{ij}^2)`.
            
            sigma : Number, optional.
                Scaling parameter for quadratic variance. \
                Defaults to 1, i.e., the data is assumed Poisson \
                with zero noise variance. \
                Used when \
                :math:`\widehat{\mathtt{var}}[Y_{ij}]=\sigma((1-q)Y_{ij}+qY_{ij}^2)`.

            b : Number, optional.
                Linear parameter for quadratic variance. Used when
                :math:`\widehat{\mathtt{var}}[Y_{ij}]=(1+c)^{-1}(bY_{ij}+cY_{ij}^2)`.

            bhat : Number, optional.
                Pre-normalized linear parameter for quadratic variance. \
                Used when
                :math:`\widehat{\mathtt{var}}[Y_{ij}]=(\hat{b}Y_{ij}+\hat{c}Y_{ij}^2)`.

            c : Number, optional.
                Quadratic parameter for quadratic variance. Used when
                :math:`\widehat{\mathtt{var}}[Y_{ij}]=(1+c)^{-1}(bY_{ij}+cY_{ij}^2)`.

            chat : Number, optional.
                Pre-normalized quadratic parameter for quadratic variance. \
                Used when
                :math:`\widehat{\mathtt{var}}[Y_{ij}]=(\hat{b}Y_{ij}+\hat{c}Y_{ij}^2)`.

            tol : Number, default 1e-6.
                Sinkhorn convergence tolerance. Must be larger than 0.

            n_iter : Number, default 100.
                Maximum Sinkhorn iterations. Must be larger than 0.

            backend : {'torch', 'scipy', 'torch_gpu','torch_cuda'}, default \
'torch'.
                Computation engine to run Sinkhorn.

        """

        _qp = QuadraticParameters()
        #: Documentation for variance
        variance: Union[np.ndarray,sparse.spmatrix, None] = ValidatedField(
                            (type(None),np.ndarray,sparse.spmatrix),
                            [],
                            None)
                            
        #: parameters related to variance estimation
        variance_estimator: Union[str, None]= ValidatedField((type(None),str),
                            [partial(is_valid,
                            lambda x : x in ['precomputed', 'general',
                                             'quadratic','binomial',None])],
                            'quadratic')
        #__variance_estimator: Union[str, None] 
        """Documentation for variance_estimator""" 

        #: precomputed row / column sums:
        row_sums: Union[None, Number, np.ndarray] = ValidatedField(
                            (type(None), Number, np.ndarray),
                            [],
                            None)
        col_sums: Union[None, Number, np.ndarray] = ValidatedField(
                            (type(None), Number, np.ndarray),
                            [],
                            None)
        #: read counts parameter (binomial):
        read_counts: Union[None, Number] = ValidatedField(
                            (type(None), Number),
                            [],
                            None)
        #:  the convex QVF estimator
        q: Union[Number, None] = ValidatedField((Number, type(None)),
                            [],
                            None)
        sigma: Union[Number, None] = ValidatedField((Number, type(None)),
                            [],
                            None)
        #: parameters for the QVF estimators
        b: Union[Number, None] = ValidatedField((Number, type(None)),
                            [],
                            None)
        bhat: Union[Number, None] = ValidatedField((Number, type(None)),
                            [],
                            None)
        c: Union[Number, None] = ValidatedField((Number, type(None)),
                            [],
                            None)
        chat: Union[Number, None] = ValidatedField((Number, type(None)),
                            [],
                            None)
        #: parameters that control the sinkhorn algorithm
        tol: Number = ValidatedField(Number, 
                    [partial(is_valid, lambda x: x>0)], 
                    1e-6)
        n_iter: int = ValidatedField(int, 
                    [partial(is_valid, lambda x: x>=0)],
                    100)
        backend: str = ValidatedField(str,
                    [partial(is_valid, lambda x: x in ['torch', 'scipy', 'torch_gpu','torch_cuda'])],
                    'torch')
        
        def __post_init__(self):
            super().__post_init__()
            self.__init_variance_parameters__()
            if isinstance(self.variance, np.ndarray):
                self.variance_estimator = 'precomputed'
        # @property
        # def variance_estimator(self):
        #     return self.__variance_estimator
        # @variance_estimator.setter
        # def variance_estimator(self, value):
        #     if type(value) is property:
        #         value=self.__class__.variance_estimator
        #     self.__variance_estimator=value
        #     print(self.variance_estimator)
        #     try: #this is wrapped in a try block.. We want to use this callback
        #         #but if it's during __init__ then the callback will fail
        #         #because all of the parameters have not been set.
        #         self.__init_variance_parameters__()
        #     except:
        #         pass
        

        # @property
        # def q(self):
        #     return self._qp.q

        # @q.setter
        # def q(self, value):
        #     if type(value) is property:
        #         value=self.__class__.q
        #         print(self.__class__.variance_estimator)
        #     self._qp.q=value

        # @property
        # def sigma(self):
        #     return self._qp.sigma

        # @sigma.setter
        # def sigma(self, value):
        #     if type(value) is property:
        #         value=self.__class__.sigma
        #     self._qp.sigma=value

        # @property
        # def b(self):
        #     return self._qp.b

        # @b.setter
        # def b(self, value):
        #     if type(value) is property:
        #         value=self.__class__.b
        #     self._qp.b=value
         
        # @property
        # def bhat(self):
        #     return self._qp.bhat

        # @bhat.setter
        # def bhat(self, value):
        #     if type(value) is property:
        #         value=self.__class__.bhat
        #     self._qp.bhat=value
        
        # @property
        # def c(self):
        #     return self._qp.c

        # @c.setter
        # def c(self, value):
        #     if type(value) is property:
        #         value=self.__class__.c
        #     self._qp.c=value
                    
        # @property
        # def chat(self):
        #     return self._qp.chat

        # @chat.setter
        # def chat(self, value):
        #     if type(value) is property:
        #         value=self.__class__.chat
        #     self._qp.chat=value

        def __init_variance_parameters__(self):
            if self.variance_estimator == 'precomputed':
                if self.variance is None:
                    raise ValueError("Precomputed variance estimator requires input to variance.")
            # if self.variance_estimator in ['precomputed',
            #                                 'general',
            #                                 'binomial',
            #                                 None]:
            #      self.b = self.bhat = self.c = self.chat = self.q = None
            # else: #  a form of quadratic
            #     self._qp=QuadraticParameters(q=self.q,sigma=self.sigma,
            #                             b=self.b,
            #                             bhat=self.bhat,
            #                             c=self.c,
            #                             chat=self.chat)
            #     try:
            #         self.q,self.sigma,self.b,self.bhat,self.c,self.chat = self._qp.compute()
            #         failed=False
            #     except:
            #         failed=True
            #     if any([ele is None for ele in [self.bhat, self.chat]]) or failed:
            #         self.b = self.bhat = self.c = self.chat = None
            #         self.q = 0
            #         self.sigma = 1
            #         self._qp=QuadraticParameters(q=self.q,sigma=self.sigma,
            #                         b=self.b,
            #                         bhat=self.bhat,
            #                         c=self.c,
            #                         chat=self.chat)   
                

    _parameters=BiPCAEstimator._parameters + ['fit_parameters']
    _backup_parameters=BiPCAEstimator._parameters+['variance', 
                                                    'row_sums',
                                                    'col_sums']
    def __init__(self, fit_parameters=None,
        compute_parameters=None,
        logging_parameters=None, 
        **kwargs):
        if fit_parameters is None:
            fit_parameters=Sinkhorn.FitParameters()
        super().__init__(**get_args(self.__init__, locals(), kwargs))

        self.converged = False
        self._issparse = None
        self.__typef_ = lambda x: x #we use this for type matching in the event the input is sparse.
        self.__xtype = None

    @memory_conserved_property
    def variance(self):  
        """Returns the entry-wise variance matrix estimated by estimate_variance.
        
        .. Warning:: This attribute is memory conserved. 

        """

        return self.fit_parameters.variance

    @variance.setter
    def variance(self,val):
        if not self.conserve_memory:
            self.fit_parameters.variance = val
    
    @fitted_property
    @memory_conserved
    def Y(self):
        """Returns the biwhitened(scaled) matrix stored in memory.

       .. Warning:: This attribute is memory conserved.
       .. Warning:: The Sinkhorn estimator must be fit to retrieve this attribute. 

        Returns
        -------
        
        """
        if attr_exists_not_none(self, '_Y'):
            return self._Y
        else:
            _Y = self.__type(self.scale(self.Y))
            self.Y = _Y
            return _Y

    @Y.setter
    def Y(self,val):
        if not self.conserve_memory:
            self._Y = val

    @fitted_property
    def right(self):
        """Returns the right-hand (column-wise) scaling vector.

        Returns
        -------
        np.ndarray of shape (N,)
            Row-wise scaling vectors
        """
        if attr_exists_not_none(self,'right_'):
            return self.right_
        return self.right_

    @right.setter
    def right(self,right):
        self.right_ = right

    @fitted_property
    def left(self):
        """Returns the left-hand (row-wise) scaling vector.
        
        Returns
        -------
        np.ndarray of shape (M,)
            Row-wise scaling vectors
        """
        if attr_exists_not_none(self,'left_'):
            return self.left_
        else:
            return None
    @left.setter
    def left(self,left):
        self.left_ = left

    @fitted_property
    def row_error(self):
        """The row errors resulting from Sinkhorn optimization.
        
        Returns
        -------
        TYPE
            Description
        """
        return self.row_error_
    @row_error.setter
    def row_error(self, row_error):
        self.row_error_ = row_error

    @fitted_property
    def column_error(self):
        """The column errors resulting from Sinkhorn optimization
        
        Returns
        -------
        TYPE
            Description
        """
        return self.row_error_
    @column_error.setter
    def column_error(self, column_error):
        self.column_error_ = column_error

    @fitted_property
    def M(self):
        """The number of rows in the input."""
        return self._M
    @fitted_property
    def N(self):
        """The number of columns in the input."""
        return self._N
        
    def __is_valid(self, X,row_sums,col_sums):
        """Verify input data is non-negative and shapes match.
        """
        eps = 1e-3
        if np.amin(X) < 0:
            self.reset_estimator(inplace=True)
            raise ValueError("Input matrix is not non-negative")
        assert np.shape(X)[0] == np.shape(row_sums)[0], "Row dimensions mismatch"
        assert np.shape(X)[1] == np.shape(col_sums)[0], "Column dimensions mismatch"
        
        # sum(row_sums) must equal sum(col_sums), at least approximately
        assert np.abs(np.sum(row_sums) - np.sum(col_sums)) < eps, "Rowsums and colsums do not add up to the same number"

    def __type(self,M):
        """Typecast data matrix M based on fitted type __typef_
        """
        check_is_fitted(self)
        if isinstance(M, self.__xtype):
            return M
        else:
            return self.__typef_(M)

    def fit(self, A):
        """Summary
        
        Parameters
        ----------
        A : TYPE
            Description
        Returns
        -------
        TYPE
            Description
        """
        if self.fit_ and self.refit:
            self.reset_estimator(inplace=True)
        if self.fit_:
            return self
        else:
            X,A = self.__extract_input_matrix__(A)
            if X is None:
                X = self.X
            if X is None:
                raise ValueError("No matrix to fit.")

            self._issparse = issparse(X,check_scipy=True,check_torch=False)
            self.__set_operands(X)
            self._M = X.shape[0]
            self._N = X.shape[1]
            row_sums, col_sums = self.__compute_dim_sums()
            self.__is_valid(X,row_sums,col_sums)
            self.row_sums = row_sums
            self.col_sums = col_sums
            if (
                self._issparse or
                (
                self.variance_estimator == 'binomial' and isinstance(self.read_counts,int)
                )):
                sparsestr = 'sparse'
            else:
                sparsestr = 'dense'

            with self.logger.task('Sinkhorn biscaling with {} {} backend'.format(sparsestr,str(self.backend))):
                
                
                if self.variance is None:
                    self.fit_parameters.__init_variance_parameters__()
                    var, rcs = self.estimate_variance(X,
                        q=self.q, bhat=self.bhat, 
                        chat=self.chat, read_counts=self.read_counts)
                    self.variance = var
                    self.read_counts = rcs
                else:
                    var = self.variance
                    rcs = self.read_counts
                
                
                l,r,re,ce = self.__sinkhorn(var,row_sums, col_sums)
                self.read_counts = rcs
                self.row_error = re
                self.column_error = ce
                # now set the final fit attributes.
                self.__xtype = type(X)
                if self.variance_estimator == None: #vanilla biscaling, we're just rescaling the original matrix.
                    self.left = l
                    self.right = r
                else:
                    self.left = np.sqrt(l)
                    self.right = np.sqrt(r)

                super().fit()
            return self

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
        
        return self.fit(X).transform(A=X)

    @fitted
    def transform(self, A=None):
        """Scale the input by left and right Sinkhorn vectors.  

        .. Warning:: The Sinkhorn estimator must be fit before transforming. 

        Parameters
        ----------
        A : None, optional
            Description
        
        Returns
        -------
        type(A) or type(A.X)
            Biscaled matrix of same type as input.
        
        """
        X,_ = self.__extract_input_matrix__(A)
        if X is None:
            X = self.X
        if X is None:
            raise ValueError("No matrix is available to transform.")
        sparsestr = 'sparse' if sparse.issparse(X) else 'dense'
        with self.logger.task(f"{sparsestr} Biscaling transform"):
            self.__set_operands(X)
            Y = self.__type(self.scale(X))
            self.Y = Y
            return Y

    @fitted
    def scale(self,A=None):
        """Rescale matrix by Sinkhorn scalers.

        .. Warning:: The Sinkhorn estimator must be fit in order to scale. 
        
        Parameters
        ----------
        X : array, optional
            Matrix to rescale by Sinkhorn scalers.
        
        Returns
        -------
        array
            Matrix scaled by Sinkhorn scalerss.
        """
        X,_ = self.__extract_input_matrix__(A)
        if X is None:
            X = self.X
        if X is None:
            raise ValueError("No matrix is available to transform.")
        if X.shape[0] == self.M:
            return self.__mem(self.__mem(X,self.right),self.left[:,None])
        else:
            return self.__mem(self.__mem(X,self.right[:,None]),self.left[None,:])
    @fitted
    def unscale(self, A=None):
        """Applies inverse Sinkhorn scalers to input X.
        
        .. Warning:: The Sinkhorn estimator must be fit in order to scale.
             
        Parameters
        ----------
        X : array, optional
            Matrix to unscale
        
        Returns
        -------
        array
            Matrix unscaled by the inverse Sinkhorn scalers
        """
        X,_ = self.__extract_input_matrix__(A)
        if X is None:
            X = self.X
        if X is None:
            raise ValueError("No matrix is available to transform.")
        if X.shape[0] == self.M:
            return self.__mem(self.__mem(X,1/self.right),1/self.left[:,None])
        else:
            return self.__mem(self.__mem(X,1/self.right[:,None]),1/self.left[None,:])


    def __set_operands(self, X=None):
        """Learn the correct operands for matrix math according to type.
        """
        # changing the operators to accomodate for sparsity 
        # allows us to have uniform API for elemientwise operations
        if X is None:
            isssparse = self._issparse
        else:
            isssparse = issparse(X,check_torch=False)
        if isssparse:
            self.__typef_ = type(X)
            self.__mem = lambda x,y : x.multiply(y)
            self.__mesq = lambda x : x.power(2)
        else:
            self.__typef_ = lambda x: x
            self.__mem= lambda x,y : np.multiply(x,y)
            self.__mesq = lambda x : np.square(x)

    def __compute_dim_sums(self):
        """Compute the sum along each dimension of the matrix
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

    def estimate_variance(self, X, dist=None, bhat=None,chat=None,read_counts=None, **kwargs):
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
        if dist=='binomial':
            if read_counts is None:
                read_counts = self.read_counts
            if read_counts is None:
                read_counts = X.sum(0)
            var = binomial_variance(X,read_counts,
                mult = self.__mem, square = self.__mesq, **kwargs)
            self.__set_operands(var)
        elif dist =='quadratic':
            if bhat is None:
                bhat = self.bhat
            if chat is None:
                chat = self.chat
            var = quadratic_variance(X,bhat=bhat,chat=chat)
        elif dist == 'general': #vanilla biscaling
            var = general_variance(X)
        else:
            var = X
        return var,read_counts

    def __sinkhorn(self, X, row_sums, col_sums, n_iter = None):
        """
        Execute Sinkhorn algorithm X mat, row_sums,col_sums for n_iter
        
        Parameters
        ----------
        X : TYPE              col_sums = torch.from_numpy(col_sums).double()
            with torch.no_grad():
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
                        a = u.cpu().numpy()
                        b = v.cpu().numpy()
                        row_converged, col_converged,_,_ = self.__check_tolerance(X,a,b)
                        if row_converged and col_converged:
                            self.logger.info("Sinkhorn converged early after "+str(i+1) +" iterations.")
                            break
                        else:
                            del v
                            del a
                            del b

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
                    + " exceeds requested tolerance: " + str(self.tol))
            
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
        ZZ = self.__mem(self.__mem(X,v), u[:,None])
        row_error  = np.amax(np.abs(self._M - ZZ.sum(0)))
        col_error =  np.amax(np.abs(self._N - ZZ.sum(1)))
        if np.any([np.isnan(row_error), np.isnan(col_error)]):
            self.converged = False
            raise Exception("NaN value detected.  Check that the input matrix"+
                " is properly filtered of sparse rows and columns.")
        del ZZ
        return row_error <  self.tol, col_error < self.tol, row_error,col_error

class SVD(BiPCAEstimator):


    @dataclass
    class FitParameters(ParameterSet):
        n_components: Union[int, None] = ValidatedField((int,type(None)), 
                                    [partial(is_valid, lambda x: x is None or x>=-1)],
                                    None) 
        exact: bool = ValidatedField(bool,
                        [],
                        True)
        use_eig: bool = ValidatedField(bool,
                        [],
                        False)
        force_dense: bool = ValidatedField(bool,
                        [],
                        False)
        vals_only: bool = ValidatedField(bool,
                        [],
                        False)
        oversample_factor: Number = ValidatedField(Number, 
                                    [partial(is_valid, lambda x: x>=1)],
                                    10)
        backend: str = ValidatedField(str,
                    [partial(is_valid, lambda x: x in ['torch', 'scipy', 'torch_gpu','torch_cuda'])],
                    'torch')
    _parameters = BiPCAEstimator._parameters + ['fit_parameters']
    def __init__(self, fit_parameters=FitParameters(),
                logging_parameters=LoggingParameters(),
                compute_parameters=ComputeParameters(),
                **kwargs):

        super().__init__(**get_args(self.__init__, locals(), kwargs))

        self._kwargs = {}
        self.kwargs = kwargs
        
        self.__k_ = None
        self._algorithm = None

        self.k=self.n_components
        
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
        u,s,v = sklearn.utils.extmath.randomized_svd(X,n_components=k,
                                                    n_oversamples=int(self.oversample_factor*k),
                                                    random_state=None)
        return u,s,v
    def __compute_scipy_svd(self,X,k):
        self.k = np.min(X.shape)
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
            self.fit_parameters.n_components=self.__k()
            return self.fit_parameters.n_components
    @k.setter
    def k(self, k):
        """Summary
        
        Parameters
        ----------
        k : TYPE
            Description
        """
        self.__k(k=k)
        self.fit_parameters.n_components=self.__k()


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
        if k is None or k < 0:
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
        if k == 0:
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

        

        if exact is not None:
            self.exact = exact
        if A is None:
            if not self.conserve_memory:
                X = self.X
                A = self.A
        else:
            if isinstance(A,AnnData):
                if not self.conserve_memory:
                    self.A = A
                X = A.X
            else:
                if not self.conserve_memory:
                    self.X = A
                X = A
        if self.force_dense:
            if sparse.issparse(X):
                X = X.toarray()
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
   
    @dataclass
    class FitParameters(ParameterSet):
        default_shrinker: str = ValidatedField(str,
                [partial(is_valid, lambda x: x in ['frobenius',
                'fro',
                'operator',
                'op',
                'nuclear',
                'nuc',
                'hard',
                'hard threshold',
                'hard_threshold',
                'soft',
                'soft threshold',
                'soft_threshold'])], 'frobenius')
        rescale_svs: bool = ValidatedField(bool, [], True)


    _parameters = BiPCAEstimator._parameters + ['fit_parameters']
    def __init__(self,  fit_parameters=FitParameters(),
                logging_parameters=LoggingParameters(),
                compute_parameters=ComputeParameters(),
                **kwargs):
        super().__init__(**get_args(self.__init__, locals(), kwargs))


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
        -------sigma
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
    ----------quadratic_variance
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

def quadratic_variance(X, bhat=1.0, chat=0):
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
    return bhat * X + chat * X**2

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
        if sparse.issparse(X):
            self.coords = self.__build_coodinates_sparse(X)
        else:
            self.coords = self.__build_coodinates_dense(X)
        self.__compute_probabilities_from_coordinates(*self.coords)
    @property
    def shape(self):
        return (self.M, self.N)

    def __build_coodinates_sparse(self,X):
        X = sparse.coo_matrix(X)
        coordinates = np.where(np.isnan(X.data))
        rows = X.row[coordinates]
        cols = X.col[coordinates]
        return rows, cols

    def __build_coodinates_dense(self, X):
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
 
    """
    return np.square(func1(x) - func2(x))


def KS(y, mp, num=500):
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
    -------:
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
        -------sigma
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
   
