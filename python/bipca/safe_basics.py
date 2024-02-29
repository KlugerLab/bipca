"""safe_basics: Type-safe generic wrappers for performing math and array operations on
torch and numpy arrays.
"""
from functools import singledispatch, wraps
import numpy as np
import torch
from scipy.sparse import spmatrix
from .utils import (filter_dict, 
                   rename_keys_in_dict)

###################################################
#   TRANSLATION BETWEEN TORCH AND NUMPY           #
###################################################
# kwarg mappings:
torch_to_numpy = {'keepdim': 'keepdims','dim':'axis', 'correction':'ddof'}
numpy_to_torch = {value:key for key,value in torch_to_numpy.items()}
numpy_ufunc_kwargs = ['out', 'where', 'casting', 'order', 'dtype', 'subok', 'signature']
# wrapper to automate argument translation
def translate_args(synonyms):
    kwarg_translater = lambda x: rename_keys_in_dict(x,synonyms)
    def wrapper(func):
        @wraps(func)
        def decorated_func(*args, **kwargs):
            kwargs = kwarg_translater(kwargs)
            if synonyms == numpy_to_torch:
                #cast the remaining arguments to torch tensors.
                new_args = [args[0]]
                for arg in args[1:]:
                    
                    if isinstance(arg,spmatrix):
                        raise TypeError('tensor by sparse matrix operations are not '
                        'supported')
                    else:
                        if hasattr(arg,'__len__'):
                            arg = torch.as_tensor(arg)
                        else:
                            pass
                        new_args.append(arg)
            else:
                new_args = args
            return func(*new_args, **kwargs)
        return decorated_func
    return wrapper

###################################################
#   BOOLEAN OPERATIONS                            #
###################################################
@singledispatch
@translate_args(torch_to_numpy)
def all(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'out', 'where'])
    return np.all(*args,**kwargs)

@all.register(torch.Tensor)
@translate_args(numpy_to_torch)
def all_tensor(*args,**kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'out'])
    return torch.all(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def any(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'out', 'where'])
    return np.any(*args,**kwargs)

@any.register(torch.Tensor)
@translate_args(numpy_to_torch)
def any_tensor(*args,**kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'out'])
    return torch.any(*args, **kwargs)

@singledispatch
def array_equal(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['equal_nan'])
    return np.array_equal(*args,**kwargs)

@array_equal.register(torch.Tensor)
def array_equal_tensor(*args,**kwargs):
    #implement equal nan behavior in torch
    #this is probably slow, but it works.
    equal_nan = kwargs.pop('equal_nan',False)
    kwargs = filter_dict(kwargs, ['out'])
    return torch.equal(*args, **kwargs)
    

@singledispatch
@translate_args(torch_to_numpy)
def equal(*args, **kwargs):
    kwargs = filter_dict(kwargs, numpy_ufunc_kwargs)
    return np.equal(*args, **kwargs)

@equal.register(torch.Tensor)
@translate_args(numpy_to_torch)
def equal_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.eq(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def greater(*args, **kwargs):
    kwargs = filter_dict(kwargs, numpy_ufunc_kwargs)
    return np.greater(*args, **kwargs)

@greater.register(torch.Tensor)
@translate_args(numpy_to_torch)
def greater_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.gt(*args, **kwargs)


@singledispatch
@translate_args(torch_to_numpy)
def isnan(*args, **kwargs):
    kwargs = filter_dict(kwargs, [])
    return np.isnan(*args, **kwargs)

@isnan.register(torch.Tensor)
@translate_args(numpy_to_torch)
def isnan_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, [])
    return torch.isnan(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def less_equal(*args, **kwargs):
    kwargs = filter_dict(kwargs, numpy_ufunc_kwargs)
    return np.less_equal(*args, **kwargs)

@less_equal.register(torch.Tensor)
@translate_args(numpy_to_torch)
def less_equal_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.le(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def less(*args, **kwargs):
    kwargs = filter_dict(kwargs, numpy_ufunc_kwargs)
    return np.less(*args, **kwargs)

@less_equal.register(torch.Tensor)
@translate_args(numpy_to_torch)
def less_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.lt(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def not_equal(*args, **kwargs):
    kwargs = filter_dict(kwargs, numpy_ufunc_kwargs)
    return np.not_equal(*args, **kwargs)

@not_equal.register(torch.Tensor)
@translate_args(numpy_to_torch)
def not_equal_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.ne(*args, **kwargs)


@singledispatch
@translate_args(torch_to_numpy)
def where(*args, **kwargs):
    kwargs = filter_dict(kwargs, [])
    return np.where(*args, **kwargs)

@where.register(torch.Tensor)
@translate_args(numpy_to_torch)
def where_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.where(*args,**kwargs)

###################################################
#   ELEMENT-WISE MATHEMATICS                      #
###################################################
@singledispatch
@translate_args(torch_to_numpy)
def abs(*args, **kwargs):
    kwargs = filter_dict(kwargs, [])
    return np.abs(*args, **kwargs)

@abs.register(torch.Tensor)
@translate_args(numpy_to_torch)
def abs_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, [])
    return torch.abs(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def add(*args, **kwargs):
    if 'alpha' in kwargs:
        args = (args[0],multiply(args[1],kwargs.pop('alpha')))
    kwargs = filter_dict(kwargs, numpy_ufunc_kwargs)
    return np.add(*args, **kwargs)

@add.register(torch.Tensor)
@translate_args(numpy_to_torch)
def add_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out','alpha'])
    return torch.add(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def divide(*args, **kwargs):
    kwargs = filter_dict(kwargs, numpy_ufunc_kwargs)
    return np.divide(*args, **kwargs)

@divide.register(torch.Tensor)
@translate_args(numpy_to_torch)
def divide_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out','rounding_mode'])
    return torch.div(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def multiply(*args, **kwargs):
    kwargs = filter_dict(kwargs, numpy_ufunc_kwargs)
    return np.multiply(*args, **kwargs)

@multiply.register(torch.Tensor)
@translate_args(numpy_to_torch)
def multiply_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.mul(*args, **kwargs)

@multiply.register(spmatrix)
def multiply_sparse(*args, **kwargs):
    return type(args[0])(args[0].multiply(args[1]))

@singledispatch
@translate_args(torch_to_numpy)
def power(*args, **kwargs):
    kwargs = filter_dict(kwargs, numpy_ufunc_kwargs)
    return np.power(*args, **kwargs)

@power.register(torch.Tensor)
@translate_args(numpy_to_torch)
def power_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.pow(*args, **kwargs)

@power.register(spmatrix)
def power_sparse(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dtype'])
    return args[0].power(*args[1:], **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def subtract(*args, **kwargs):
    if 'alpha' in kwargs:
        args = (args[0],multiply(args[1],kwargs.pop('alpha')))
    kwargs = filter_dict(kwargs, numpy_ufunc_kwargs)
    return np.subtract(*args, **kwargs)

@subtract.register(torch.Tensor)
@translate_args(numpy_to_torch)
def subtract_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out','alpha'])
    return torch.sub(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def sqrt(*args, **kwargs):
    kwargs = filter_dict(kwargs, numpy_ufunc_kwargs)
    return np.sqrt(*args, **kwargs)

@sqrt.register(torch.Tensor)
@translate_args(numpy_to_torch)
def sqrt_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.sqrt(*args, **kwargs)

def square(*args, **kwargs):
    return power(*args, 2)

@singledispatch
@translate_args(torch_to_numpy)
def exp(*args, **kwargs):
    kwargs = filter_dict(kwargs, numpy_ufunc_kwargs)
    return np.exp(*args, **kwargs)

@exp.register(torch.Tensor)
@translate_args(numpy_to_torch)
def exp_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out', 'dtype'])
    return torch.exp(*args, **kwargs)

###################################################
#   DIMENSIONAL & ELEMENT-WISE STATISTICS         #
###################################################

@singledispatch
@translate_args(torch_to_numpy)
def amax(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'initial', 'where'])  
    return np.amax(*args, **kwargs)

@amax.register(torch.Tensor)
@translate_args(numpy_to_torch)
def amax_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdim', 'out'])  
    return torch.amax(*args, **kwargs)

@amax.register(spmatrix)
def amax_sparse(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'out'])  
    return args[0].max(**kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def amin(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'initial', 'where'])  
    return np.amin(*args, **kwargs)

@amin.register(torch.Tensor)
@translate_args(numpy_to_torch)
def amin_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdim', 'out'])  
    return torch.amin(*args, **kwargs)

@amin.register(spmatrix)
def amin_sparse(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'out'])  
    return args[0].min(**kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def argsort(*args, **kwargs):
    #add support for descending kwarg from torch
    if 'descending' in kwargs:
        descending = kwargs.pop('descending')
    else:
        descending = False
    if 'stable' in kwargs:
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
@translate_args(numpy_to_torch)
def argsort_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'descending','stable'])
    return torch.argsort(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def quantile(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['q', 'axis', 'out', 'overwrite_input', 
    'interpolation', 'keepdims'])
    return np.quantile(*args, **kwargs)

@quantile.register(torch.Tensor)
@translate_args(numpy_to_torch)
def quantile_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs,['dim','keepdim','interpolation','out'])
    return torch.quantile(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def mean(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'dtype', 'out','where'])
    return np.mean(*args, **kwargs)

@mean.register(torch.Tensor)
@translate_args(numpy_to_torch)
def mean_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'dtype','out'])
    return torch.mean(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def max(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'initial', 'where'])
    return np.max(*args, **kwargs)

@max.register(torch.Tensor)
@translate_args(numpy_to_torch)
def max_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'out'])
    return torch.max(*args, **kwargs)

@max.register(spmatrix)
def max_sparse(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'out'])
    return args[0].max(**kwargs)


@singledispatch
@translate_args(torch_to_numpy)
def min(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'initial', 'where'])
    return np.min(*args, **kwargs)

@min.register(torch.Tensor)
@translate_args(numpy_to_torch)
def min_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'out'])
    return torch.min(*args, **kwargs)

@min.register(spmatrix)
def min_sparse(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'out'])
    return args[0].min(**kwargs)


@singledispatch
@translate_args(torch_to_numpy)
def std(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'dtype', 'out', 'ddof', 'where'])
    return np.std(*args, **kwargs)

@std.register(torch.Tensor)
@translate_args(numpy_to_torch)
def std_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'out', 'correction'])
    return torch.std(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def sum(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['axis', 'keepdims', 'dtype', 'out','initial','where'])
    return np.sum(*args, **kwargs)

@sum.register(torch.Tensor)
@translate_args(numpy_to_torch)
def sum_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['dim', 'keepdim', 'dtype'])
    return torch.sum(*args, **kwargs)

###################################################
#   LINEAR ALGEBRA                                #
###################################################

@singledispatch
@translate_args(torch_to_numpy)
def dot(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return np.dot(*args, **kwargs)

@dot.register(torch.Tensor)
@translate_args(numpy_to_torch)
def dot_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.dot(*args, **kwargs)

@singledispatch
@translate_args(torch_to_numpy)
def matmul(*args, **kwargs):
    kwargs = filter_dict(kwargs, numpy_ufunc_kwargs)
    return np.matmul(*args, **kwargs)

@matmul.register(torch.Tensor)
@translate_args(numpy_to_torch)
def matmul_tensor(*args, **kwargs):
    kwargs = filter_dict(kwargs, ['out'])
    return torch.matmul(*args, **kwargs)

@translate_args(torch_to_numpy)
def parse_transpose_args(*args, **kwargs):
    """parses the kwargs for transpose and returns the axes kwarg,
    or raises an error if the kwargs are invalid.
    facilitates inter-operable torch and numpy transpose behavior.
    Note that by default torch will use numpy behavior now. 
    """
    #first, check for parameter collisions. these occur anytime that axes is supplied
    # and dim0 or dim1 are supplied, or if dim0 and dim1 are supplied positionally
    # and also as kwargs.
    if ( ('axes' in kwargs) # kwarg axes match or
        or (len(args) > 1 # call has positional arguments and 
        and hasattr(args[1],'__len__') # positional arguments support len and
        and len(args[1]) == args[0].ndim)): # positional axes match:
        #axes has been specified.
        #check to see if dim0 and dim1 have been specified as kwargs or positionally
        if len(args)>2:
            raise TypeError('transpose received too many positional arguments with axes')
        if ('dim0' in kwargs) or ('dim1' in kwargs):
            raise TypeError('transpose received both `axes` and kwarg dim0 and/or'
            'dim1')
        
        if 'axes' in kwargs:
            if ( len(args)>1 and # positional axes
                 any( not_equal(args[1],kwargs['axes']) )
               ):
                raise ValueError(f'axes kwarg {kwargs["axes"]} does not match the '
                f'positional axes {args[1]}')
            axes = kwargs['axes']
        else:
            axes = args[1]
    else: #axes are not specified by kwargs or positionally
        axes = list(range(args[0].ndim))
        #first, check to see if dim0 or dim1 appear in the arguments
        if len(args) > 1:
            if (len(args) == 2):  
                #dim0 appears positionally
                if ('dim0' in kwargs):
                    return TypeError('transpose received both positional and kwarg '
                    'dim0')
                else:
                    if 'dim1' in kwargs:
                        # we can work with this
                        dim0 = args[1]
                        dim1 = kwargs['dim1'].pop()
                    else:
                        # we can't work with this
                        raise TypeError('transpose received only one positional arg, '
                        'but dim1 was not specified as a kwarg')
            elif (len(args) == 3):
                #dim0 and dim1 appear positionally
                if ('dim0' in kwargs) or ('dim1' in kwargs):
                    return TypeError('transpose received both positional and kwarg dim0' 
                    'and/or dim1')
                else:
                    dim0 = args[1]
                    dim1 = args[2]
            else:
                raise TypeError('transpose received too many positional arguments')
            axes[dim0] = dim1
            axes[dim1] = dim0
        else:
            #default behavior. in numpy this will be to reverse the order of the axes.
            axes = axes[::-1]
    
    return args[0],axes

@singledispatch
@translate_args(torch_to_numpy)
def transpose(*args, **kwargs):
    #this function should transpose the input array, and optionally support torch-style
    #transpose behavior
    M, axes = parse_transpose_args(*args, **kwargs)
    return np.transpose(M, axes)

@transpose.register(torch.Tensor)
@translate_args(torch_to_numpy)
def transpose_tensor(*args, **kwargs):
    #this function defaults to numpy style transpose behavior, but also supports
    #torch.transpose(x, dim0,dim1) style behavior.
    #uses torch_to_numpy translation style to make parse_transpose_args work.
    #since args,kwargs are directly supplied to torch.permute, there's no need to
    #translate the other direction.
    M, axes = parse_transpose_args(*args, **kwargs)
    return torch.permute(M, axes)
