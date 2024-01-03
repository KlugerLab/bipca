from functools import singledispatch, wraps
import numpy as np
import torch
from scipy.sparse import spmatrix
from .utils import (filter_dict, 
                   rename_keys_in_dict)

# kwarg mappings:
numpy_to_torch = {'keepdim': 'keepdims','dim':'axis', 'correction':'ddof'}
torch_to_numpy = {value:key for key,value in numpy_to_torch.items()}
# wrappers for translation between numpy and torch
def translate_kwargs(synonyms):
    translater = lambda x: rename_keys_in_dict(x,synonyms)
    def wrapper(func):
        @wraps(func)
        def decorated_func(*args, **kwargs):
            kwargs = translater(kwargs)
            return func(*args, **kwargs)
        return decorated_func
    return wrapper

# element-wise booleans
@singledispatch
@translate_kwargs(torch_to_numpy)
def all(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'out', 'where'])
    return np.all(*args,**kwargs)

@all.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def all_tensor(*args,**kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'out'])
    return torch.all(*args, **kwargs)

@singledispatch
@translate_kwargs(torch_to_numpy)
def any(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'out', 'where'])
    return np.any(*args,**kwargs)

@any.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def any_tensor(*args,**kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'out'])
    return torch.any(*args, **kwargs)

@singledispatch
@translate_kwargs(torch_to_numpy)
def isnan(*args, **kwargs):
    kwargs = filter_dict(kwargs, [])
    return np.isnan(*args, **kwargs)

@isnan.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def isnan_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, [])
    return torch.isnan(*args, **kwargs)

@singledispatch
@translate_kwargs(torch_to_numpy)
def less_equal(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out', 'where'])
    return np.less_equal(*args, y, **kwargs)

@less_equal.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def less_equal_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.le(*args, **kwargs)

@singledispatch
@translate_kwargs(torch_to_numpy)
def where(*args, **kwargs):
    kwargs = filter_dict(kwargs, [])
    return np.where(*args, **kwargs)

@where.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def where_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.where(condition,*args,y)

# element-wise math
@singledispatch
@translate_kwargs(torch_to_numpy)
def abs(*args, **kwargs):
    kwargs = filter_dict(kwargs, [])
    return np.abs(*args, **kwargs)

@abs.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def abs_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, [])
    return torch.abs(*args, **kwargs)

@singledispatch
@translate_kwargs(torch_to_numpy)
def multiply(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out', 'where', 'casting', 'order', 'dtype', 'subok',
    'signature'])
    return np.multiply(*args, **kwargs)

@multiply.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def multiply_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    if not isinstance(args[1], torch.Tensor):
        args[1] = torch.Tensor(args[1])
    return torch.mul(*args, **kwargs)

@multiply.register(spmatrix)
def multiply_sparse(*args, **kwargs):
    return type(args[0])(args[0].multiply(args[1]))

@singledispatch
@translate_kwargs(torch_to_numpy)
def power(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out', 'where', 'casting', 'order', 'dtype', 'subok',
    'signature'])
    return np.power(*args, **kwargs)

@power.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def power_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.pow(*args, **kwargs)

@power.register(spmatrix)
def power_sparse(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dtype'])
    return args[0].power(*args[1:], **kwargs)

@singledispatch
@translate_kwargs(torch_to_numpy)
def sqrt(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out', 'where', 'casting', 'order', 'dtype', 'subok',
    'signature'])
    return np.sqrt(*args, **kwargs)

@sqrt.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def sqrt_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.sqrt(*args, **kwargs)

def square(*args, **kwargs):
    return power(*args, 2)

@singledispatch
@translate_kwargs(torch_to_numpy)
def exp(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out', 'where', 'casting', 'order', 'dtype', 'subok',
    'signature'])
    return np.exp(*args, **kwargs)

@exp.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def exp_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out', 'dtype'])
    return torch.exp(*args, **kwargs)

# dimensional & element-wise statistics

@singledispatch
@translate_kwargs(torch_to_numpy)
def amax(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'initial', 'where'])  
    return np.amax(*args, **kwargs)

@amax.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def amax_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdim', 'out'])  
    return torch.amax(*args, **kwargs)

@amax.register(spmatrix)
def amax_sparse(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'out'])  
    return args[0].max(**kwargs)

@singledispatch
@translate_kwargs(torch_to_numpy)
def amin(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'initial', 'where'])  
    return np.amin(*args, **kwargs)

@amin.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def amin_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdim', 'out'])  
    return torch.amin(*args, **kwargs)

@amin.register(spmatrix)
def amin_sparse(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'out'])  
    return args[0].min(**kwargs)

@singledispatch
@translate_kwargs(torch_to_numpy)
def argsort(*args, **kwargs):
    #add support for descending kwarg from torch
    if 'descending' in kwargs:
        descending = kwargs.pop('descending')
    else:
        descending = False
    if stable in kwargs:
        if kwargs[stable]:
            if kind in kwargs:
                assert (kwargs[kind] in ['stable','mergesort'])
            else:
                kwargs[kind] = 'stable'
        else:
            if kind in kwargs:
                assert (kwargs[kind] not in ['stable','mergesort'])
            else:
                kwargs[kind] = 'quicksort'
            
    kwargs = filter_dict(kwargs, ['axis', 'kind', 'order'])
    ix = np.argsort(*args, **kwargs)
    if descending:
        return ix[::-1]
    else:
        return ix

@argsort.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def argsort_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'descending','stable'])
    return torch.argsort(*args, **kwargs)

@singledispatch
@translate_kwargs(torch_to_numpy)
def quantile(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['q', 'axis', 'out', 'overwrite_input', 
    'interpolation', 'keepdims'])
    return np.quantile(*args, **kwargs)

@quantile.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def quantile_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs,['dim','keepdim','interpolation','out'])
    return torch.quantile(*args, **kwargs)

@singledispatch
@translate_kwargs(torch_to_numpy)
def mean(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'dtype', 'out','where'])
    return np.mean(*args, **kwargs)

@mean.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def mean_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'dtype','out'])
    return torch.mean(*args, **kwargs)

@singledispatch
@translate_kwargs(torch_to_numpy)
def max(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'initial', 'where'])
    return np.max(*args, **kwargs)

@max.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def max_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'out'])
    return torch.max(*args, **kwargs)

@max.register(spmatrix)
def max_sparse(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'out'])
    return args[0].max(**kwargs)


@singledispatch
@translate_kwargs(torch_to_numpy)
def min(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'initial', 'where'])
    return np.min(*args, **kwargs)

@min.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def min_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'out'])
    return torch.min(*args, **kwargs)

@min.register(spmatrix)
def min_sparse(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'out'])
    return args[0].min(**kwargs)


@singledispatch
@translate_kwargs(torch_to_numpy)
def std(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'dtype', 'out', 'ddof', 'where'])
    return np.std(*args, **kwargs)

@std.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def std_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'out', 'correction'])
    return torch.std(*args, **kwargs)

@singledispatch
@translate_kwargs(torch_to_numpy)
def sum(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'dtype', 'out','initial','where'])
    return np.sum(*args, **kwargs)

@sum.register(torch.Tensor)
@translate_kwargs(numpy_to_torch)
def sum_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'dtype'])
    return torch.sum(*args, **kwargs)

