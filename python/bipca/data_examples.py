import numpy as np



def multinomial_data(nrows=500, ncols=1000, rank=10, sample_rate=100):
	#the probability basis for the data
	p = np.random.multinomial(nrows,[1/nrows]*nrows,rank) / nrows
	#draw random loadings
	loading = np.random.multinomial(rank, [1/rank]*rank, ncols) / rank
	#the ground truth probability matrix
	PX = (loading @ p).T
	#the data
	X = np.vstack([np.random.multinomial(sample_rate,PX[:,i]) for i in range(ncols)])
	return X, PX