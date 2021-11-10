import scanpy as sc
from bipca import BiPCA
from bipca.math import binomial_variance
from scipy.sparse import csr_matrix
import unittest
from nose2.tools import params
import numpy as np
import torch
##make the data to be tested in this file
from test_data import filtered_adata
X = filtered_adata.X.toarray()

class Test_BiPCA(unittest.TestCase):

	def test_plotting_spectrum_submtx(self):
		op = BiPCA(n_components=0,n_subsamples=2,subsample_size=200,qits=2,verbose = 0,suppress=False, njobs=1)
		X_sparse = csr_matrix(X)
		op.fit(X_sparse)
		op.get_plotting_spectrum()
		assert len(op.plotting_spectrum['Y']) == op.plotting_spectrum['shape'][0]
		assert len(op.plotting_spectrum['fits']) == 2
	def test_sparse(self):
		op = BiPCA(n_components = 0 , n_subsamples = 2, subsample_size=200, qits = 2, verbose =0, njobs=1)
		X_sparse = csr_matrix(X)
		op.fit(X_sparse)
	def test_transposed_get_Z(self):
		op = BiPCA(n_components=0,n_subsamples=2,subsample_size=200,qits=2,verbose = 0,njobs=1)
		op.fit(X.T)
		op.get_Z(X.T)
		op.get_Z(X)
	def test_plotting_spectrum_fullmtx_size_enforced(self):
		op = BiPCA(n_components=0,n_subsamples=2,subsample_size=1000,qits=2,verbose = 0,njobs=1)
		op.fit(X)
		op.get_plotting_spectrum()
		assert len(op.plotting_spectrum['Y']) == np.min(X.shape)
	def test_plotting_spectrum_fullmtx_nsubs_enforced(self):
		op = BiPCA(n_components=0,n_subsamples=0,subsample_size=200,qits=2, verbose = 0,njobs=1)
		op.fit(X)
		op.get_plotting_spectrum()
		assert len(op.plotting_spectrum['Y']) == np.min(X.shape)
	def test_plotting_spectrum_fullmtx_subsample_False(self):
		op = BiPCA(n_components=0,n_subsamples=2,subsample_size=200,qits=2, verbose = 0,njobs=1)
		op.fit(X)
		op.get_plotting_spectrum(subsample=False)
		assert len(op.plotting_spectrum['Y']) == np.min(X.shape)

	def test_write_to_adata(self):
		op = BiPCA(n_components=0,n_subsamples=2,subsample_size=200,qits=2, verbose = 0,njobs=1)
		op.fit(X)
		op.write_to_adata(filtered_adata)
	def test_binomial_variance(self):
		op = BiPCA(variance_estimator='binomial',read_counts=2,
			verbose = 0,sinkhorn_tol=2e-3,n_iter=1000)
		
		X = np.array([[1,1,2],[2,1,1],[0,1,2]])
		op.fit(X)
		assert op.sinkhorn.read_counts == 2
		assert np.allclose(op.sinkhorn.var.toarray(), binomial_variance(X,counts=2).toarray()) 
	def test_plotting_spectrum_binomial_submtx(self):
		op = BiPCA(variance_estimator='binomial',read_counts=2,
			n_subsamples=2,subsample_size=200,approximate_sigma = False,verbose = 0,sinkhorn_tol=2e-3,n_iter=1000,njobs=1)
		
		X = np.random.binomial(2,0.5,size=(1000,1000))
		op.fit(X)
		op.get_plotting_spectrum()
		assert len(op.plotting_spectrum['Y']) == op.plotting_spectrum['shape'][0]

	def test_plotting_spectrum_fullmtx_size_enforced(self):
		op = BiPCA(variance_estimator='binomial',read_counts=3,
			n_subsamples=2,subsample_size=1000,verbose = 0,sinkhorn_tol=2e-3,n_iter=1000,njobs=1)
		X = np.random.binomial(2,0.5,size=(500,500))

		op.fit(X)
		op.get_plotting_spectrum()
		assert len(op.plotting_spectrum['Y']) == 500
	def test_plotting_spectrum_fullmtx_nsubs_enforced(self):
		op = BiPCA(variance_estimator='binomial',read_counts=3,
			n_subsamples=0,subsample_size=200,verbose = 0,sinkhorn_tol=2e-3,n_iter=1000,njobs=1)
		X = np.random.binomial(2,0.5,size=(500,500))

		op.fit(X)
		op.get_plotting_spectrum()
		assert len(op.plotting_spectrum['Y']) == 500
	def test_plotting_spectrum_fullmtx_subsample_False(self):
		op = BiPCA(variance_estimator='binomial',read_counts=3,
			n_subsamples=2,subsample_size=200,verbose = 0,sinkhorn_tol=2e-3,n_iter=1000,njobs=1)
		X = np.random.binomial(2,0.5,size=(500,500))

		op.fit(X)
		op.get_plotting_spectrum(subsample=False)
		assert len(op.plotting_spectrum['Y']) == 500