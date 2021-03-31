import numpy as np
import inspect
def _is_vector(x):
	return (x.ndim == 1 or x.shape[0] == 1 or x.shape[1] == 1)

def _xor(lst, obj):
	condeval = [ele==obj for ele in lst]
	condeval = sum(condeval)
	return condeval==1

def _zero_pad_vec(nparray, final_length):
	# pad a vector (nparray) to have length final_length by adding zeros
	# adds to the largest axis of the vector if it has 2 dimensions
	# requires the input to have 1 dimension or at least one dimension has length 1.
	if (nparray.shape[0]) == final_length:
		z = nparray
	else:
		axis = np.argmax(nparray.shape)
		pad = final_length - nparray.shape[axis]
		if nparray.ndim>1:
			if not 1 in nparray.shape:
				raise ValueError('Input nparray is not a vector')
		padshape = list(x.shape)
		padshape[axis] = padamt
		z = np.concatenate((nparray,np.zeros(padshape)),axis=axis)
	return z

def filter_dict(dict_to_filter, thing_with_kwargs):
	"""
	Modified from 
	https://stackoverflow.com/a/44052550    
	User "Adviendha"

	"""
	sig = inspect.signature(thing_with_kwargs)
	filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD and param.name in dict_to_filter.keys()]
	filtered_dict = {filter_key:dict_to_filter[filter_key] for filter_key in filter_keys}
	return filtered_dict

def ischanged_dict(old_dict, new_dict, keys_ignore = []):
	ischanged = False

	#check for adding or updating arguments
	for k in new_dict:
		ischanged = k not in old_dict or old_dict[k] != new_dict[k]#catch values that are new
		if ischanged:
			break
	#now check for removing arguments
	if not ischanged:
		for k in old_dict:
			if k not in keys_ignore and k not in new_dict:
				ischanged = True
				break
	return ischanged