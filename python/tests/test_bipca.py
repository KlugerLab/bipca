import scanpy as sc
from bipca import BiPCA
from bipca.math import binomial_variance
import unittest
from nose2.tools import params
import numpy as np

##make the data to be tested in this file
from test_data import filtered_adata
X = filtered_adata.X.toarray()

class Test_BiPCA(unittest.TestCase):

	def test_plotting_spectrum_submtx(self):
		op = BiPCA(n_components=0,n_subsamples=5,subsample_size=200,qits=21, q=0.25,verbose = 1)
		op.fit(X)
		op.get_plotting_spectrum()
		assert len(op.plotting_spectrum['Y']) == op.plotting_spectrum['shape'][0]
	def test_plotting_spectrum_fullmtx_size_enforced(self):
		op = BiPCA(n_components=0,n_subsamples=5,subsample_size=1000,qits=21, q=0.25,verbose = 0)
		op.fit(X)
		op.get_plotting_spectrum()
		assert len(op.plotting_spectrum['Y']) == np.min(X.shape)
	def test_plotting_spectrum_fullmtx_nsubs_enforced(self):
		op = BiPCA(n_components=0,n_subsamples=0,subsample_size=200,qits=21, q=0.25,verbose = 0)
		op.fit(X)
		op.get_plotting_spectrum()
		assert len(op.plotting_spectrum['Y']) == np.min(X.shape)
	def test_plotting_spectrum_fullmtx_subsample_False(self):
		op = BiPCA(n_components=0,n_subsamples=5,subsample_size=200,qits=21, q=0.25,verbose = 0)
		op.fit(X)
		op.get_plotting_spectrum(subsample=False)
		assert len(op.plotting_spectrum['Y']) == np.min(X.shape)

	def test_write_to_adata(self):
		op = BiPCA(n_components=0,n_subsamples=5,subsample_size=200,qits=21, q=0.25,verbose = 0)
		op.fit(X)
		op.write_to_adata(filtered_adata)
	def test_binomial_variance(self):
		op = BiPCA(variance_estimator='binomial',read_counts=2,
			qits=0, q=0.26,approximate_sigma = False,verbose = 0,sinkhorn_tol=2e-3,n_iter=1000)
		
		X = np.array([[1,1,2],[2,1,1],[0,1,2]])
		op.fit(X)
		assert op.sinkhorn.read_counts == 2
		assert np.allclose(op.sinkhorn.var, binomial_variance(X,counts=2)) 
	def test_plotting_spectrum_binomial_submtx(self):
		op = BiPCA(variance_estimator='binomial',read_counts=2,
			n_subsamples=5,subsample_size=200,approximate_sigma = False,verbose = 0,sinkhorn_tol=2e-3,n_iter=1000)
		
		X = np.random.binomial(2,0.5,size=(1000,1000))
		op.fit(X)
		op.get_plotting_spectrum()
		assert len(op.plotting_spectrum['Y']) == op.plotting_spectrum['shape'][0]

	def test_plotting_spectrum_fullmtx_size_enforced(self):
		op = BiPCA(variance_estimator='binomial',read_counts=3,
			n_subsamples=5,subsample_size=1000,approximate_sigma = False,verbose = 0,sinkhorn_tol=2e-3,n_iter=1000)
		X = np.random.binomial(2,0.5,size=(500,500))

		op.fit(X)
		op.get_plotting_spectrum()
		assert len(op.plotting_spectrum['Y']) == 500
	def test_plotting_spectrum_fullmtx_nsubs_enforced(self):
		op = BiPCA(variance_estimator='binomial',read_counts=3,
			n_subsamples=0,subsample_size=200,approximate_sigma = False,verbose = 0,sinkhorn_tol=2e-3,n_iter=1000)
		X = np.random.binomial(2,0.5,size=(500,500))

		op.fit(X)
		op.get_plotting_spectrum()
		assert len(op.plotting_spectrum['Y']) == 500
	def test_plotting_spectrum_fullmtx_subsample_False(self):
		op = BiPCA(variance_estimator='binomial',read_counts=3,
			n_subsamples=5,subsample_size=200,approximate_sigma = False,verbose = 0,sinkhorn_tol=2e-3,n_iter=1000)
		X = np.random.binomial(2,0.5,size=(500,500))

		op.fit(X)
		op.get_plotting_spectrum(subsample=False)
		assert len(op.plotting_spectrum['Y']) == 500