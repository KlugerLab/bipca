import scanpy as sc
from bipca import BiPCA
from bipca.math import binomial_variance
from bipca.utils import stabilize_matrix
from scipy.sparse import csr_matrix
import unittest
from nose2.tools import params
import numpy as np
##make the data to be tested in this file
from test_data import filtered_adata
X = filtered_adata.X.toarray()

class Test_BiPCA(unittest.TestCase):

	def test_plotting_spectrum_submtx(self):
		op = BiPCA(n_components=0,n_subsamples=2,
		subsample_size=(400,600),qits=2,verbose = 0,suppress=False, njobs=1)
		op.fit(X)
		op.get_plotting_spectrum()
		assert len(op.plotting_spectrum['Y']) == op.plotting_spectrum['shape'][0]
		assert len(op.plotting_spectrum['fits']) == 2
	def test_sparse(self):
		op = BiPCA(n_components = 0 ,
		 n_subsamples = 2, 
		subsample_size=(400,600), qits = 2, verbose =0, njobs=1)
		X_sparse = csr_matrix(X)
		op.fit(X_sparse)
	def test_transposed_get_Z(self):
		op = BiPCA(n_components=0,
		n_subsamples=2,subsample_size=(400,600),qits=2,verbose = 0,njobs=1)
		op.fit(X.T)
		op.get_Z(X.T)
		op.get_Z(X)
	def test_plotting_spectrum_fullmtx_size_enforced(self):
		op = BiPCA(n_components=0,
		n_subsamples=2,subsample_size=1000,qits=2,verbose = 0,njobs=1)
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
		op = BiPCA(n_components=0,subsample_threshold=1,
		n_subsamples=2,subsample_size=200,qits=2, verbose = 0,njobs=1)
		op.fit(X)
		op.write_to_adata(filtered_adata)
	def test_binomial_variance(self):
		op = BiPCA(variance_estimator='binomial',read_counts=2,
			verbose = 0,sinkhorn_tol=2e-3,n_iter=1000)
		
		X = np.array([[1,1,2],[2,1,1],[0,1,2]])
		op.fit(X)
		assert op.sinkhorn.read_counts == 2
		assert np.allclose(op.sinkhorn.var, binomial_variance(X,counts=2).toarray()) 
	def test_missing_entries(self):
		op = BiPCA(read_counts=2,
			verbose = 0,sinkhorn_tol=2e-3,n_iter=1000)
		nrows = 1000
		rank = 5
		ncols=500
		seed =42
		rng = np.random.default_rng(seed = seed)
		S = np.exp(2*rng.standard_normal(size=(nrows,rank)));
		coeff = rng.uniform(size=(rank,ncols));
		X = S@coeff;
		X = np.where(np.random.binomial(1,0.9,size=X.shape),X,np.NaN)

		op.fit(X)
		assert op.mp_rank == 5
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


	def test_subsample_sizing(self):
		#smoketest with valid sizes
		xx= X[:400,:][:,:600]
		xx,_ = stabilize_matrix(xx,threshold=10)
		print(xx.shape)
		op = BiPCA(subsample_size=(xx.shape[0]-1,xx.shape[1]-1),verbose = 0) 
		op.reset_submatrices(xx)
		assert op.subsample_size == (xx.shape[0]-1,xx.shape[1]-1), op.subsample_size

		#smoketest w/ invalid sizes

		op.subsample_size=(xx.shape[0]+1,xx.shape[1]+1)
		op.reset_submatrices(xx)
		assert op.subsample_size == (xx.shape[0],xx.shape[1]), op.subsample_size

		#smoketest on first dimension
		op.subsample_size=(None,xx.shape[1]-10)
		op.reset_submatrices(xx)
		assert op.subsample_size == ((xx.shape[1]-10),xx.shape[1]-10),op.subsample_size


		op.subsample_size=(None,xx.shape[1]-10)
		op.keep_aspect=True
		op.reset_submatrices(xx)
		assert op.subsample_size == (np.floor(xx.shape[0]/xx.shape[1] * (xx.shape[1]-10)),xx.shape[1]-10),op.subsample_size

		#smoketest on second dimension
		op.subsample_size=(xx.shape[0]-10,0)
		op.reset_submatrices(xx)
		assert op.subsample_size == (xx.shape[0]-10,np.floor(xx.shape[1]/xx.shape[0] * (xx.shape[0]-10))),op.subsample_size