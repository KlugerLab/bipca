import numpy as np

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
		z = np.concatenate((x,np.zeros(padshape)),axis=axis)
	return z