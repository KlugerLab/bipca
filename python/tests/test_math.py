from bipca.math import (SVD,Sinkhorn,
						binomial_variance,
						MarcenkoPastur,
						quadratic_variance_2param,
						SamplingMatrix
						)
import scipy.sparse as sparse
from testing_utils import raises, TestingError
import warnings
import numpy as np
import unittest

from nose2.tools import params

class Test_SVD(unittest.TestCase):
	
	def __init__(self,*args,**kwargs):
		self.M,self.N = 200,50
		self.X = np.random.randn(self.M,self.N)

		super().__init__(*args,**kwargs)

	@params(('scipy',True),('torch_cpu',True),('torch_gpu',True),
		('scipy',False),('torch_cpu',False),('torch_gpu',False))
	def test_output_shape_full(self, backend,exact):
		#the proper output of SVD will be such that X = (U*S)@V.T
		#different underlying algorithms lead to different transposes, especially of V.
		op = SVD(n_components=self.N, backend=backend,exact=exact,verbose=0)
		op.fit(self.X)
		assert op.U.shape == (self.M,self.N)
		assert op.S.shape == (self.N,)
		assert op.V.shape == (self.N,self.N)
		op.fit(self.X.T)
		assert op.U.shape == (self.N,self.N)
		assert op.S.shape == (self.N,)
		assert op.V.shape == (self.M,self.N)

	@params(('scipy',True),('torch_cpu',True),('torch_gpu',True),
		('scipy',False),('torch_cpu',False),('torch_gpu',False))
	def test_output_shape_partial(self,backend,exact):
		op = SVD(n_components=10, exact=exact, backend=backend,verbose=0) # we want 10 components out now.
		op.fit(self.X)
		assert op.U.shape == (200,10)
		assert op.S.shape == (10,)
		assert op.V.shape == (50,10)
		op.fit(self.X.T)
		assert op.U.shape == (50,10)
		assert op.S.shape == (10,)
		assert op.V.shape == (200,10)

	@params(('scipy',True),('torch_cpu',True),('torch_gpu',True),
		('scipy',False),('torch_cpu',False),('torch_gpu',False))
	def test_output_shape_partial_switches_to_full(self,backend,exact):
		op = SVD(n_components=25, exact=exact, backend=backend,verbose=0) # we want 25 components out now.
		op.fit(self.X)
		assert op.U.shape == (200,25)
		assert op.S.shape == (25,)
		assert op.V.shape == (50,25)
		op.fit(self.X.T)
		assert op.U.shape == (50,25)
		assert op.S.shape == (25,)
		assert op.V.shape == (200,25)

	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_svd(self,backend):
		opsvd = SVD(backend=backend,verbose=0)
		U,S,V = opsvd.factorize(X=self.X)
		assert np.allclose((U*S)@V.T,self.X)

	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_eigvalues_match_svs(self,backend):
		opsvd = SVD(backend=backend,vals_only=True,verbose=0)
		opsvd.fit(self.X)
		ssvd = opsvd.S
		opeigs = SVD(backend=backend,vals_only=True,use_eig=True,verbose=0)
		opeigs.fit(self.X)
		seig = opeigs.S
		assert np.allclose(seig,ssvd)

	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_eigdecomposition_match_og_matrix(self,backend):
		opeigs = SVD(backend=backend,use_eig=True,verbose=0)
		opeigs.fit(self.X)
		ueigs = opeigs.U
		seigs = opeigs.S
		veigs = opeigs.V
		assert np.allclose((ueigs*seigs)@veigs.T, self.X)
		opeigs = SVD(backend=backend,use_eig=True,verbose=0)
		opeigs.fit(self.X.T)
		ueigs = opeigs.U
		seigs = opeigs.S
		veigs = opeigs.V
		assert np.allclose((ueigs*seigs)@veigs.T, self.X.T)

### TESTS FOR SVD.compute_element:
class Test_SVD_compute_element(unittest.TestCase):
	def __init__(self,*args,**kwargs):
		self.M,self.N = 200,50
		self.X = np.random.randn(self.M,self.N)
		self.backends = ['scipy','torch_cpu', 'torch_gpu']
		self.svds = {b:SVD(backend=b,verbose=0).fit(self.X) for b in self.backends}
		super().__init__(*args,**kwargs)

	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_individual_valid_element(self,backend):
		#test if we can get an individual element from full rank.
		opsvd = self.svds[backend]
		index = (0,0)
		assert np.allclose(opsvd.compute_element(index),self.X[0,0])

	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_individual_valid_element_negative(self,backend):
		#test if we can get an individual element from full rank with a negative index
		opsvd = self.svds[backend]
		index = (-1,-1)
		assert np.allclose(opsvd.compute_element(index),self.X[-1,-1])
	
	@raises(IndexError)
	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_individual_invalid_element(self,backend):
		#make sure that invalid individual indices raise IndexError.
		opsvd = self.svds[backend]
		index = (self.M+1,0)
		opsvd.compute_element(index)
	
	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_full_matrix(self,backend):
		#compute_element should return the full matrix when no params are specified and the object is fit.
		opsvd = self.svds[backend]
		y = opsvd.compute_element()
		assert isinstance(y, type(opsvd.U))
		assert np.allclose(y,self.X)

	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_rows(self,backend):
		opsvd = self.svds[backend]
		target_rows = [0,2]
		#compute_element should return the full matrix when no params are specified and the object is fit.
		y = opsvd.compute_element(index=np.s_[target_rows,:])
		
		assert isinstance(y, type(opsvd.U))
		assert np.allclose(y,self.X[target_rows,:])

	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_cols(self,backend):
		opsvd = self.svds[backend]
		target_cols = [0,2]
		#compute_element should return the full matrix when no params are specified and the object is fit.
		y = opsvd.compute_element(index=np.s_[:,target_cols])
		
		assert isinstance(y, type(opsvd.U))
		assert np.allclose(y,self.X[:,target_cols])


	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_rank(self,backend):
		opsvd = self.svds[backend]

		rank = 2
		X = (opsvd.U[:,:rank]*opsvd.S[:rank])@opsvd.V[:,:rank].T
		target_cols = [0,2]
		#compute_element should return the full matrix when no params are specified and the object is fit.
		y = opsvd.compute_element(index=np.s_[:,target_cols],rank=rank)
		
		assert isinstance(y, type(opsvd.U))
		assert np.allclose(y,X[:,target_cols])

	@raises(ValueError)
	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_rank_too_large(self,backend):
		opsvd = self.svds[backend]

		rank = 2000
		#X = (opsvd.U[:,:rank]*opsvd.S[:rank])@opsvd.V[:,:rank].T
		target_cols = [0,2]
		#compute_element should return the full matrix when no params are specified and the object is fit.
		opsvd.compute_element(index=np.s_[:,target_cols],rank=rank)
		
	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_custom_S(self,backend):
		opsvd = self.svds[backend]

		custom_S = np.ones_like(opsvd.S)
		X = (opsvd.U*custom_S)@opsvd.V.T
		target_cols = [0,2]
		y = opsvd.compute_element(index=np.s_[:,target_cols], S = custom_S)
		assert isinstance(y, type(opsvd.U))
		assert np.allclose(y,X[:,target_cols])

	@raises(ValueError)
	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_custom_S_wrong_rank(self,backend):
		opsvd = self.svds[backend]

		custom_S = np.ones_like(opsvd.S)
		target_cols = [0,2]
		#should panicdue to wrong size
		y = opsvd.compute_element(index=np.s_[:,target_cols], S = custom_S, rank=2000)
	
	
	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_custom_V(self,backend='torch_cpu'):
		opsvd = self.svds[backend]

		custom_V = opsvd.V[:,:2]
		#print(custom_V)
		target_cols = [0,2]
		#we should obtain the rank 2 approximation of the first and third rows
		y = opsvd.compute_element(index=np.s_[:,target_cols], V = custom_V)
		X = (opsvd.U[:,:2]*opsvd.S[:2])@custom_V.T
		assert isinstance(y, type(opsvd.U))
		assert np.allclose(y,X[:,target_cols])

	@raises(ValueError)
	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_custom_V_wrong_rank(self,backend='torch_cpu'):
		opsvd = self.svds[backend]

		custom_V = opsvd.V[:,:2]
		#print(custom_V)
		target_cols = [0,2]
		y = opsvd.compute_element(index=np.s_[:,target_cols], V = custom_V,rank=2000)

	
	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_custom_V_cols(self,backend='torch_cpu'):
		opsvd = self.svds[backend]

		custom_V = opsvd.V[[0,2],:10]
		target_cols = [0,2]
		y = opsvd.compute_element(V = custom_V)
		X = (opsvd.U[:,:10]*opsvd.S[:10])@custom_V.T
		assert isinstance(y, type(opsvd.U))
		assert np.allclose(y,X)

	@raises(ValueError)
	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_custom_U_wrong_rank(self,backend='torch_cpu'):
		opsvd = self.svds[backend]

		custom_U = opsvd.U[:,:2]
		target_cols = [0,2]
		y = opsvd.compute_element(index=np.s_[:,target_cols], U = custom_U,rank=2000)

	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_custom_U(self,backend='torch_cpu'):
		opsvd = self.svds[backend]

		custom_U = opsvd.U[[0,2],:10]
		
		y = opsvd.compute_element(U=custom_U)
		X = (custom_U*opsvd.S[:10])@opsvd.V[:,:10].T
		assert isinstance(y, type(opsvd.U))
		assert np.allclose(y,X)

	@params(('scipy'),('torch_cpu'),('torch_gpu'))
	def test_compute_element_type_mismatch(self,backend='torch_cpu'):
		opsvd = self.svds[backend]

		custom_U = np.asarray(opsvd.U[[0,2],:10])
		
		y = opsvd.compute_element(U=custom_U)
		X = (custom_U*np.asarray(opsvd.S[:10]))@np.asarray(opsvd.V[:,:10].T)
		assert isinstance(y, type(opsvd.U))
		assert np.allclose(y,X)


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

	
	def test_counts_matrix(self):
		X = np.eye(3)*2
		counts = np.ones_like(X)*2
		Y = binomial_variance(X,counts)
		assert np.allclose(np.zeros((3,3)),Y)
		op = Sinkhorn(variance_estimator='binomial', read_counts=counts)
		var=op.estimate_variance(X)
		assert np.allclose(np.zeros((3,3)),var[0])
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