import numpy as np
import torch
from .utils import filter_dict_with_kwargs


@singledispatch
def less_equal(X, Y, **kwargs):
    kwargs = filter_dict_with_kwargs(kwargs, np.less_equal)
    return np.less_equal(x, y, **kwargs)

@less_equal.register(torch.Tensor)
def less_equal_tensor(x, y, **kwargs):
    kwargs = filter_dict_with_kwargs(kwargs, torch.le)
    return torch.le(x,y, **kwargs)

@singledispatch
def where(condition, x, y, **kwargs):
    return np.where(condition,x,y)

@where.register(torch.Tensor)
def where_tensor(condition,x,y):
    return torch.where(condition,x,y)

@singledispatch
def quantile(X, q, **kwargs):
    if 'dim' in kwargs and 'axis' not in kwargs:
        kwargs['axis'] = kwargs.pop('dim')
    return np.quantile(X, q, **kwargs)

@quantile.register(torch.Tensor)
def quantile_tensor(X, q, **kwargs):
    if 'axis' in kwargs and 'dim' not in kwargs:
        kwargs['dim'] = kwargs.pop('axis')
    return torch.quantile(X, q, **kwargs)

@singledispatch
def amax(X, **kwargs):
    return np.amax(X, **kwargs)

@amax.register(torch.Tensor)
def amax_tensor(*args,**kwargs):
    return torch.amax(*args, **kwargs)

@singledispatch
def abs(X, **kwargs):
    return np.abs(X,**kwargs)

@abs.register(torch.Tensor)
def abs_tensor(*args,**kwargs):
    return torch.abs(*args, **kwargs)

@singledispatch
def isnan(X, **kwargs):
    return np.isnan(X,**kwargs)

@isnan.register(torch.Tensor)
def isnan_tensor(*args,**kwargs):
    return torch.isnan(*args, **kwargs)

@singledispatch
def any(X, **kwargs):
    return np.any(X,**kwargs)

@any.register(torch.Tensor)
def any_tensor(*args,**kwargs):
    return torch.any(*args, **kwargs)