from bipca.math import (Sinkhorn, SVD,QuadraticParameters,
						binomial_variance,
						MarcenkoPastur,
						quadratic_variance_2param,
						SamplingMatrix
						)
import scipy.sparse as sparse
from utils import raises
import warnings
import numpy as np
import unittest

from nose2.tools import params

class Test_QuadraticParameters(unittest.TestCase):

	def test_equivalent_parameters(self):
		x = np.random.randn(100,20)**2
		sigma=2
		q=0.5
		varX0 = sigma**2*((1-q)*x+q*x**2)
		quad_params_operator = QuadraticParameters(q=q,sigma=sigma)
class Test_Sinkhorn(unittest.TestCase):
	x = np.random.uniform(100,size=[100,100]).astype(int)

	def test_smoketest(self):
		op = Sinkhorn(variance_estimator=None,refit=True)
		Y = op.fit_transform(self.x)
		assert np.allclose(100,Y.sum(1))
		assert np.allclose(100,Y.sum(0))
		op.backend='scipy'
		Y = op.fit_transform(self.x)
		assert np.allclose(100,Y.sum(1))
		assert np.allclose(100,Y.sum(0))
		op.variance_estimator='quadratic'
		print(op.variance_estimator)
		op.b=1
		op.c=0.5
		Y=op.fit_transform(self.x)
	@raises(ValueError, startswith="Input matrix")
	def test_non_negative_fails(self):
		Sinkhorn().fit(-1*self.x)

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
	def test_consistency_with_quadratic(self):
		X = np.eye(3)*2
		counts = 2
		Y = binomial_variance(X,counts)
		b = 1
		c = -(1/counts)
		bhat = b/(1+c)
		chat = (1+c)/(1+c)
		Z = quadratic_variance_2param(X,bhat=bhat,chat=chat)
		assert np.allclose(np.zeros((3,3)),Y.toarray())
class Test_MP(unittest.TestCase):
	def test_cdf(self):
		aspect_ratios = np.linspace(0,1,10)
		test_vals = np.linspace(-10,10,100)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			for mp in map(MarcenkoPastur,aspect_ratios):
				np.allclose(mp.cdf(test_vals,which='analytical'),
					mp.cdf(test_vals,which='numerical'),
					)
class Test_SamplingMatrix(unittest.TestCase):
	def __init__(self,*args,**kwargs):
		self.X = np.ones((3,2))
		self.X[0,0] = np.nan
		super().__init__(*args,**kwargs)
	def test_dense(self):
		self.M = SamplingMatrix(self.X)
		self._assert_correct()
	def test_sparse(self):
		self.M = SamplingMatrix(sparse.csr_matrix(self.X))
		self._assert_correct()

	def _assert_correct(self):
		assert np.allclose(self.M[0,0],0.4)
		assert np.allclose(self.M[1,1], 1)
		assert np.max(self.M()) == 1