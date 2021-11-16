from bipca.math import SVD, binomial_variance
from utils import raises

import numpy as np
import unittest

from nose2.tools import params

class Test_SVD(unittest.TestCase):


	@params(('scipy',True),('torch_cpu',True),('torch_gpu',True),
		('scipy',False),('torch_cpu',False),('torch_gpu',False))
	def test_output_shape_full(self, backend,exact):
		#the proper output of SVD will be such that X = (U*S)@V.T
		#different underlying algorithms lead to different transposes, especially of V.
		test_mat = np.random.randn(200,50)
		op = SVD(n_components=50, backend=backend,exact=exact,verbose=0)
		op.fit(test_mat)
		assert op.U.shape == (200,50)
		assert op.S.shape == (50,)
		assert op.V.shape == (50,50)
		test_mat = np.random.randn(50,200)
		op.fit(test_mat)
		assert op.U.shape == (50,50)
		assert op.S.shape == (50,)
		assert op.V.shape == (200,50)

	@params(('scipy',True),('torch_cpu',True),('torch_gpu',True),
		('scipy',False),('torch_cpu',False),('torch_gpu',False))
	def test_output_shape_partial(self,backend,exact):
		test_mat = np.random.randn(200,50)
		op = SVD(n_components=10, exact=exact, backend=backend,verbose=0) # we want 10 components out now.
		op.fit(test_mat)
		assert op.U.shape == (200,10)
		assert op.S.shape == (10,)
		assert op.V.shape == (50,10)
		test_mat = np.random.randn(50,200)
		op.fit(test_mat)
		assert op.U.shape == (50,10)
		assert op.S.shape == (10,)
		assert op.V.shape == (200,10)

	@params(('scipy',True),('torch_cpu',True),('torch_gpu',True),
		('scipy',False),('torch_cpu',False),('torch_gpu',False))
	def test_output_shape_partial_switches_to_full(self,backend,exact):
		import scipy
		test_mat = np.random.randn(200,50)
		op = SVD(n_components=25, exact=exact, backend=backend,verbose=0) # we want 25 components out now.
		op.fit(test_mat)
		assert op.algorithm == scipy.sparse.linalg.svds
		assert op.U.shape == (200,25)
		assert op.S.shape == (25,)
		assert op.V.shape == (50,25)
		test_mat = np.random.randn(50,200)
		op.fit(test_mat)
		assert op.U.shape == (50,25)
		assert op.S.shape == (25,)
		assert op.V.shape == (200,25)

	def test_svd(self):
		test_mat = np.random.randn(200,50)
		opsvd = SVD(backend='torch',verbose=0)
		U,S,V = opsvd.factorize(X=test_mat)
		assert np.allclose((U*S)@V.T,test_mat)

	def test_eigs(self):
		test_mat = np.random.randn(200,50)
		opsvd = SVD(backend='torch',vals_only=True,verbose=0)
		opsvd.fit(test_mat)
		ssvd = opsvd.S
		opeigs = SVD(backend='torch',vals_only=True,use_eig=True,verbose=0)
		opeigs.fit(test_mat)
		seig = opeigs.S
		assert np.allclose(seig,ssvd)
		
		opeigs = SVD(backend='torch',vals_only=False,use_eig=True,verbose=0)
		opeigs.fit(test_mat)
		ueigs = opeigs.U
		seigs = opeigs.S
		veigs = opeigs.V
		assert np.allclose((ueigs*seigs)@veigs.T, test_mat)
		opeigs = SVD(backend='torch',vals_only=False,use_eig=True,verbose=0)
		opeigs.fit(test_mat.T)
		ueigs = opeigs.U
		seigs = opeigs.S
		veigs = opeigs.V
		assert np.allclose((ueigs*seigs)@veigs.T, test_mat.T)

		opsvd = SVD(backend='scipy',vals_only=True, verbose=0,use_eig=False)
		opsvd.fit(test_mat)
		opeigs = SVD(backend='scipy',vals_only=True, verbose=0,use_eig=True)
		opeigs.fit(test_mat)
		assert np.allclose(opeigs.S, opsvd.S)

		opeigs = SVD(backend='scipy',vals_only=False,use_eig=True,verbose=0)
		opeigs.fit(test_mat)
		ueigs = opeigs.U
		seigs = opeigs.S
		veigs = opeigs.V
		assert np.allclose((ueigs*seigs)@veigs.T, test_mat)
		opeigs = SVD(backend='scipy',vals_only=False,use_eig=True,verbose=0)
		opeigs.fit(test_mat.T)
		ueigs = opeigs.U
		seigs = opeigs.S
		veigs = opeigs.V
		assert np.allclose((ueigs*seigs)@veigs.T, test_mat.T)

class Test_Binomial_Variance(unittest.TestCase):
	@raises(ValueError)
	def test_counts_leq_1(self):
		X = np.eye(3)
		counts=1
		binomial_variance(X,counts)

	def test_counts_eq_2(self):
		X = np.eye(3)*2
		counts = 2
		Y = binomial_variance(X,counts)
		assert np.allclose(np.zeros((3,3)),Y.toarray())
		