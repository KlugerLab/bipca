import scanpy as sc
from bipca import BiPCA
from bipca.math import binomial_variance
import unittest
from nose2.tools import params
import numpy as np

##make the data to be tested in this file
data = sc.datasets.pbmc3k()
adata = sc.read_h5ad('data/pbmc3k_raw.h5ad')
adata = adata[:,:1000]
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.filter_genes(adata, min_cells=10)
X = adata.X.toarray()
op = BiPCA(n_components=0,n_subsamples=5,subsample_size=200,subsample_threshold=10,qits=21, q=0.25,verbose = 1)

op.fit(X)


class Test_BiPCA(unittest.TestCase):

	def test_write_to_adata(self):
		op.write_to_adata(adata)
	def test_binomial_variance(self):
		op = BiPCA(variance_estimator='binomial',read_counts=2,
			qits=0, q=0.26,approximate_sigma = False,verbose = 0,sinkhorn_tol=2e-3,n_iter=1000)
		
		X = np.array([[1,1,2],[2,1,1],[0,1,2]])
		op.fit(X)
		assert op.sinkhorn.read_counts == 2
		assert np.allclose(op.sinkhorn.var, binomial_variance(X,counts=2)) 
		