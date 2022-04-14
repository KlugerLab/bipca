import scanpy as sc
from bipca import BiPCA, denoise_means
from bipca.math import binomial_variance, Sinkhorn
from bipca.utils import nz_along
from scipy.sparse import csr_matrix
import unittest
from nose2.tools import params
import numpy as np
import torch
import numpy.matlib as matlib

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


class test_denoise_means:

	m = 50
	n = 100
	H = np.r_[np.ones((n//2,1)), np.zeros((n//2,1))]
	H = np.c_[H,np.flipud(H)]
	X = np.r_[matlib.repmat(H[:,0][None,:],m//2,1),
			  matlib.repmat(H[:,1][None,:],m//2,1)]
	mean=10
	var=10000
	sigma2=np.log(var/(mean**2)+1)
	mu = np.log(mean)-sigma2/2
	sigma=np.sqrt(sigma2)
	libsize=np.ceil(np.exp(np.random.randn(n)*sigma+mu)).astype(int)
	genesize=np.ceil(np.exp(np.random.randn(m)*sigma+mu)).astype(int)

	X = genesize[:,None] * X * libsize[None,:]
	Y = np.random.poisson(X)
	XH = X@H

	bhat, chat = 1, 0


	def test_vanilla(self):
		outputs = denoise_means(X=self.X, Y=self.Y, H = self.H,verbose=False)


	def test_precomputed_biwhite(self):
		op = Sinkhorn(bhat=1, chat=0,variance_estimator='quadratic_2param',verbose=False)
		Xhat = op.fit_transform(self.X)
		op = Sinkhorn(bhat=1, chat=0,variance_estimator='quadratic_2param',verbose=False)
		Yhat = op.fit_transform(self.Y)

		outputs = denoise_means(X=Xhat, Y=Yhat,
								H = self.H,
								precomputed=True,verbose=False)

	def test_precomputed_u_v(self):
		op = Sinkhorn(bhat=1, chat=0,verbose=False,variance_estimator='quadratic_2param')
		Xhat = op.fit_transform(self.X)
		op = BiPCA(bhat=1, chat=0,verbose=False)
		op.fit(self.Y)
		U = op.U_Z
		V = op.V_Z
		r = op.mp_rank
		Yhat = op.Z
		U = U[:,:r]
		V = V[:,:r]
		outputs = denoise_means(X=self.X, Y = self.Y, H=self.H,
								U=U, V=V, verbose=False) #precomputed u and v smoketest
								
		outputs2 = denoise_means(X=self.X, Y = self.Y, H=self.H, #precomputed u,v, bhat,chat
								U=U, V=V,bhat=1, chat=0, verbose=False)

		outputs3 = denoise_means(X=Xhat, Y = Yhat, H=self.H,
								U=U, V=V,bhat=1, chat=0, 
								precomputed=True, verbose=False)

		assert np.allclose(outputs2[0],outputs3[0])


	def test_identity(self):
		H = np.eye(self.n)
		op = BiPCA(bhat=1, chat=0,verbose=False)
		op.fit(self.Y)
		U = op.U_Z
		V = op.V_Z
		r = op.mp_rank

		s = op.S_Z[:r]
		U = U[:,:r]
		V = V[:,:r]
		outputs = denoise_means(X=self.Y, Y = self.Y, H=H,bhat=1,chat=0,
								verbose=False)
		assert np.allclose(outputs[0], ((U*s)@V.T))

	def test_tall_matrix(self):
		m = 100
		n = 50
		H = np.r_[np.ones((n//2,1)), np.zeros((n//2,1))]
		H = np.c_[H,np.flipud(H)]
		X = np.r_[matlib.repmat(H[:,0][None,:],m//2,1),
				matlib.repmat(H[:,1][None,:],m//2,1)]
		mean=10
		var=10000
		sigma2=np.log(var/(mean**2)+1)
		mu = np.log(mean)-sigma2/2
		sigma=np.sqrt(sigma2)
		libsize=np.ceil(np.exp(np.random.randn(n)*sigma+mu)).astype(int)
		genesize=np.ceil(np.exp(np.random.randn(m)*sigma+mu)).astype(int)

		X = genesize[:,None] * X * libsize[None,:]
		Y = np.random.poisson(X)
		XH = X@H
		outputs = denoise_means(X=Y, Y = Y, H=H,
								verbose=False)


	def test_H_maps_rows(self):

		outputs = denoise_means(X=self.X.T, Y = self.Y.T, H=self.H,
								verbose=False)
