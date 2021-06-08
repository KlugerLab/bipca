import numpy as np
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sparse
from scipy.sparse import linalg as spla
import scipy.linalg
import numpy.matlib
import time


#number of trials per instance
ntrials = 3
#densities to check
min_nnz = 0.01
densities = np.arange(0,8)
densities = min_nnz*2**densities
densities[-1] = 1.0

#aspect ratios and matrix dimension
nel = 500000
aspect_ratios = np.linspace(0.25,1,2,endpoint=False)
aspect_ratios = np.hstack((aspect_ratios,np.array(1), 1/aspect_ratios[::-1]))


results = np.zeros((len(densities),len(aspect_ratios),4))

algorithm = [np.linalg.svd, scipy.linalg.svd, spla.svds, randomized_svd]

for rix, density in enumerate(densities):
	print(rix)
	for cix, aspect_ratio in enumerate(aspect_ratios):
		for trial in range(ntrials):
			m = np.floor(np.sqrt(nel * aspect_ratio)).astype(int)
			n = np.floor(nel/np.sqrt(nel*aspect_ratio)).astype(int)
			spmat = sparse.random(m, n, density=density, format='csr')
			mat = spmat.toarray()
			k = np.min(mat.shape)
			for algx, alg in enumerate(algorithm):
				X = mat if algx<2 else spmat
				start = time.process_time()
				if algx < 2:
					u = alg(X)
				elif alg == spla.svds:
					u = alg(spmat,k = int(k/2))
				elif alg==randomized_svd:
					u = alg(mat, n_components = int(k/2))
				results[rix,cix, algx] += (time.process_time()-start)/ntrials


print(np.argmin(results,2))