from utils import raises
import bipca
import numpy as np
import unittest


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
