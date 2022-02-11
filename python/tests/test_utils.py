from utils import raises
import bipca
import numpy as np
import unittest

def test_stabilize_matrix():
	from bipca.utils import stabilize_matrix,nz_along
	X=np.ones(11)
	X=np.triu(X)
	def test_indices_correct(Y,indices,X):
		assert np.allclose(Y, X[indices[0],:][:,indices[1]])

	#test that all rows+columns get returned
	Y,indices= stabilize_matrix(X,threshold=1)
	assert np.allclose(Y, X)
	test_indices_correct(Y,indices,X)
	Y,indices= stabilize_matrix(X,threshold=6)
	assert np.allclose(Y, np.ones(6))
	test_indices_correct(Y,indices,X)

	### test fitering of all-zero rows and columns

	X2r=np.r_[X, [np.zeros(11)]] #rows
	Y,indices= stabilize_matrix(X2r)
	test_indices_correct(Y,indices,X)
	assert np.allclose(Y, X)

	X2c=np.c_[X,np.zeros(11)]#columns
	Y,indices= stabilize_matrix(X2c)
	assert np.allclose(Y, X)
	test_indices_correct(Y,indices,X)

	X3=np.r_[X2c,[np.zeros(12)]]#columns and rows
	Y,indices= stabilize_matrix(X3) #filter both axes
	assert np.allclose(Y, X)
	test_indices_correct(Y,indices,X3)

	Y,indices= stabilize_matrix(X3,row_threshold=1,threshold=0) #filter only rows
	assert np.allclose(Y, X2c)
	test_indices_correct(Y,indices,X2c)

	Y,indices= stabilize_matrix(X3,row_threshold=0,column_threshold=1) #filter only columns
	assert np.allclose(Y, X2r)
	test_indices_correct(Y,indices,X2r)

	#test differential filtering: rows only
	Y,indices = stabilize_matrix(X, threshold=1, row_threshold=2)
	Y2,_ = stabilize_matrix(X, threshold=2, column_threshold=1)
	assert np.allclose(Y,Y2)
	assert np.allclose(Y[-1,:], X[-2,:])
	assert 10 not in indices[0]
	test_indices_correct(Y,indices,X)

	#test differential filtering: cols only
	Y,indices = stabilize_matrix(X, threshold=1, column_threshold=2)
	Y2,_ = stabilize_matrix(X, threshold=2, row_threshold=1)
	assert np.allclose(Y,Y2)
	assert np.allclose(Y[:,0], X[:,1])
	assert 0 not in indices[1]
	test_indices_correct(Y,indices,X)
	#test differential filtering : both direction
	Y,indices = stabilize_matrix(X, threshold=2,  column_threshold=5)
	Y2,_ = stabilize_matrix(X, threshold=5,  row_threshold=2)
	assert Y.shape == (10,7)
	test_indices_correct(Y,indices,X)
	assert np.allclose(Y,Y2)


	### Test sequential filtering
	#basic situation: filtering out zero rows
	Y,indices = stabilize_matrix(X, order=0)
	assert np.allclose(Y,X)
	test_indices_correct(Y,indices,X)
	Y,indices = stabilize_matrix(X, order=0)
	assert np.allclose(Y,X)
	test_indices_correct(Y,indices,X)

	#now a matrix with some zeros
	Y,indices = stabilize_matrix(X3, order=0)
	assert np.allclose(Y,X)
	test_indices_correct(Y,indices,X)
	Y,indices = stabilize_matrix(X3, order=1)
	assert np.allclose(Y,X)
	test_indices_correct(Y,indices,X)
	# now a slightly  more complicated result in which the order matters
	X = np.triu(np.ones(4))
	X = np.c_[np.zeros(4),X]
	X[-1,0]=1
	Y,indices = stabilize_matrix(X,order=0,threshold=2)
	test_indices_correct(Y,indices,X)
	Z=np.array([[1., 1., 1.],
	[1., 1., 1.],
	[0., 1., 1.],
	[0., 0., 1.]])
	assert np.allclose(Y,Z)
	Y,indices = stabilize_matrix(X,order=1,threshold=2)
	test_indices_correct(Y,indices,X)
	Z=np.array([[1., 1., 1.],
       [1., 1., 1.],
       [0., 1., 1.]])
	assert np.allclose(Y,Z)

def test_iterative_stabilize_matrix():
	def test_indices_correct(Y,indices,X):
		assert np.allclose(Y, X[indices[0],:][:,indices[1]])
	from bipca.utils import stabilize_matrix,nz_along
	X = np.c_[np.zeros(4),np.triu(np.ones(4))]
	X = np.r_[np.c_[np.zeros(4),np.triu(np.ones(4))],[np.zeros(5)]]
	X[-2,0]=1
	X[-1,1]=1
	#smoke test to make sure it even works
	_,_ = stabilize_matrix(X,n_iters=5)
	_,_ = stabilize_matrix(X,order=0,n_iters=5)
	_,_ = stabilize_matrix(X,order=1,n_iters=5)
	# now validate that we converge and we have the right indices
	for o in [False,0,1]:
		for t in range(0,4):
			Y,indices=stabilize_matrix(X,order=o,threshold=t,n_iters=5)
			test_indices_correct(Y,indices,X)
			assert np.all(nz_along(Y,axis=0) >= t )
			assert np.all(nz_along(Y,axis=1) >= t )
	for _ in range(100):
		X = np.random.choice([0, 1], size=100, p=[0.5, .5]).reshape([10,10])
		for o in [False,0,1]:
				for t in range(0,10):
					Y,indices=stabilize_matrix(X,order=o,threshold=t,n_iters=20)
					test_indices_correct(Y,indices,X)
					assert np.all(nz_along(Y,axis=0) >= t ), nz_along(Y,axis=0).min()
					assert np.all(nz_along(Y,axis=1) >= t ), nz_along(Y,axis=0).min()
def test_nz_along():
	#nz_along is intended to check the nonzeros along an axis in a type-independent way: it can check
	#numpy arrays or scipy.sparse matrices
	from bipca.utils import nz_along
	import scipy.sparse as sparse

	#the basic check: a dense identity matrix for which the nonzeros are the same on the rows and columns.
	eyemat = np.eye(100)
	valid_answer = np.ones((100,))

	assert np.allclose(nz_along(eyemat,1),nz_along(eyemat,0))
	assert np.allclose(nz_along(eyemat,0),valid_answer)


	#another check: a matrix with a single non zero entry

	zmat = np.zeros((100,100))
	valid_answer = np.zeros((100,))
	valid_answer[0] = 1
	zmat[0,0] = 1
	assert np.allclose(nz_along(zmat,1),nz_along(zmat,0))
	assert np.allclose(nz_along(zmat,0),valid_answer)

	#now let column 0 have more entries in it.
	zmat = np.zeros((100,100))
	valid_answer = np.zeros((100,2)) #multidimensional answers
	valid_answer[0,0] = 2
	valid_answer[0,1] = 1
	valid_answer[1,1] = 1
	zmat[0,0] = 1
	zmat[1,0] = 1
	assert np.allclose(nz_along(zmat,0),valid_answer[:,0])
	assert np.allclose(nz_along(zmat,1),valid_answer[:,1])

	#now do a tensor 
	zmat = np.zeros((100,100,10))
	valid_answer = np.zeros((100,3)) #multidimensional answers
	valid_answer[0,0] = 2
	valid_answer[0,1] = 1
	valid_answer[1,1] = 1
	zmat[0,0,0] = 1
	zmat[1,0,0] = 1

	assert np.allclose(nz_along(zmat,0)[:,0],valid_answer[:,0])
	assert np.allclose(nz_along(zmat,1)[:,0],valid_answer[:,1])
	assert np.allclose(nz_along(zmat,2)[:,0],valid_answer[:,1])
	zmat[0,0,1] = 1
	assert np.allclose(nz_along(zmat,2)[0,:],valid_answer[:,0])

	def test_sparse_formats():
	#verify that it works for sparse.
		possible_sparse_formats = [sparse.bsr_matrix, sparse.csr_matrix, sparse.csc_matrix,
									sparse.coo_matrix, sparse.dia_matrix, 
									sparse.lil_matrix, sparse.dok_matrix]
		for tipe in possible_sparse_formats:
			speyemat = tipe(eyemat)
			assert np.allclose(nz_along(speyemat,1),nz_along(speyemat,0))
			assert np.allclose(nz_along(speyemat,0),valid_answer)
		return True
	assert test_sparse_formats

	@raises(TypeError)
	def test_incompatible_type():
		x = np.matrix([1,2])
		return nz_along(x,0)
	assert test_incompatible_type()
	@raises(TypeError)
	def test_incompatible_type2():
		import torch
		x = torch.Tensor([1,2])
		return nz_along(x,0)
	assert test_incompatible_type()
	# Does passing an invalid index raise an exception?
	@raises(ValueError)
	def test_nz_along_bad_axis():
		return nz_along(eyemat,2)
	assert test_nz_along_bad_axis()
	@raises(ValueError)
	def test_nz_along_bad_negative_axis():
		return nz_along(eyemat,-3)
	assert test_nz_along_bad_negative_axis()

def test_attr_exists_not_none():
#the attr_exists_not_none is intended for checking if a variable needs to be instantiated first.
# it is often used when a class instantiates without an attribute that is married to a property
# if the property is called, it must check if the attribute is there, and if it's not or it is none, 
# the attribute must be written.
	from bipca.utils import attr_exists_not_none
	class fakeObj(object):
		def __init__(self,kwarg1='foo',kwarg2='bar',kwarg3=None):
			self.kwarg1 = kwarg1
			self._kwarg3 = kwarg3
		@property
		def kwarg3(self):
			return self._kwarg3
		@kwarg3.setter
		def kwarg3(self,val):
			self._kwarg3 = val
	obj = fakeObj()
	# kwarg 1 exists and is not none
	assert attr_exists_not_none(obj, 'kwarg1')
	# kwarg 2 does not exist
	assert not attr_exists_not_none(obj,'kwarg2')
	# kwarg 3 exists in a property, but it's None
	assert not attr_exists_not_none(obj, '_kwarg3')
	#check that being a property doesn't break our reference
	assert not attr_exists_not_none(obj,'kwarg3')
	obj.kwarg3 = 'bar'
	#now it exists and it's not none
	assert attr_exists_not_none(obj, '_kwarg3')
	#check the property
	assert attr_exists_not_none(obj,'kwarg3')

	return True
class Test_CachedFunction(unittest.TestCase):

	def test_single_output(self):
		x = np.arange(10)
		f = lambda x: np.power(x,2)
		f_cached = bipca.utils.CachedFunction(f)
		assert np.all(f_cached(x) == f(x))
		assert all([f_cached.cache[xx] == f(xx) for xx in x])
		assert isinstance(f_cached(x),np.ndarray)
		x = list(x)
		assert isinstance(f_cached(x),list)
	@raises(KeyError)
	def test_not_cached(self):
		x = np.arange(10)
		f = lambda x: np.power(x,2)
		f_cached = bipca.utils.CachedFunction(f)
		y = f_cached(x)
		f_cached.cache[0]
	def test_multiple_output(self):

		def f(x):
			return np.power(x,2), np.power(x,3)

		f_cached = bipca.utils.CachedFunction(f,num_outs=2)
		x = np.arange(10)
		assert np.all(f_cached(x)[0] == np.power(x,2))
		assert f_cached.cache[2][1] == np.power(2,3)
		x = list(x)
		y1,y2 = f_cached(x)
		assert isinstance(y1,list)
	@raises(ValueError)
	def test_incorrect_num_outs(self):
		def f(x):
			if x == 0:
				return True,False
			else:
				return True,False,True
		f_cached = bipca.utils.CachedFunction(f,num_outs=2)
		x = np.arange(10)

	def test_unhashable_input(self):
		x = np.array([[1,2],[3,4],[5,6]])
		f_cached = bipca.utils.CachedFunction(lambda x: np.sum(x),num_outs=1)
		assert isinstance(f_cached(x),np.ndarray)
		x = list([[1,2],[3,4]])
		assert isinstance(f_cached(x),list)
