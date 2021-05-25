"""This file includes code for generating example data
"""
import numpy as np
import numpy.matlib as matlib


def multinomial_data(nrows=500, ncols=1000, rank=10, sample_rate=100, simple=False):
	"""
	Generate multinomial distributed data of prescribed rank.
	
	Parameters
	----------
	nrows : int, default 500
	    Description
	ncols : int, default 1000
	    Description
	rank : int, default 10
	    Description
	sample_rate : int, default 100
	    Description
	simple : bool, optional
	    Description
	
	Returns
	-------
	TYPE
	    Description
	"""
	#the probability basis for the data
	p = np.random.multinomial(nrows,[1/nrows]*nrows,rank) / nrows
	if simple:
		cluster_size = np.floor(ncols/rank).astype(int)
		PX = matlib.repmat(p.T,1,cluster_size)
		rem = ncols-PX.shape[1]
		if rem > 0:
			PX = np.hstack((matlib.repmat(p.T[:,0],1,rem),PX))
	else:
		#draw random loadings
		loading = np.random.multinomial(rank, [1/rank]*rank, ncols) / rank
		#the ground truth probability matrix
		PX = (loading @ p).T
	#the data
	X = np.vstack([np.random.multinomial(sample_rate,PX[:,i]) for i in range(ncols)])
	return X, PX