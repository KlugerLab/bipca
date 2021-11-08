# Tests in here for speed comparisons
# Does not run unless nose2 is specifically pointed at it


from bipca.math import SVD, binomial_variance,Sinkhorn

import numpy as np
import scipy.sparse as sparse
import unittest
from functools import partial
import timeit

repeats = 3
n = 5
def test_SVD_speed_cpu_dense():
	def run_eigvals(X):
		op = SVD(use_eig=True,backend='torch',vals_only=True,verbose=0)
		u,s,v=op.factorize(X=X)
	def run_svdvals(X):
		op = SVD(use_eig=False,backend='torch',vals_only=True,verbose=0)
		u,s,v=op.factorize(X=X)
	def run_eigh(X):
		op = SVD(use_eig=True,backend='torch',vals_only=False,verbose=0)
		u,s,v=op.factorize(X=X)
	def run_svd(X):
		op = SVD(use_eig=False,backend='torch',vals_only=False,verbose=0)
		u,s,v=op.factorize(X=X)

	print("*****testing dense torch cpu SVD speed*****")
	print(" Running square matrix test")
	X = np.random.randn(1000,1000)
	f = partial(run_eigvals,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for eigvals:",min(times)/n)
	f = partial(run_svdvals,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for svdvals:",min(times)/n)
	f = partial(run_eigh,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for eig:",min(times)/n)
	f = partial(run_svd,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for svd:",min(times)/n)

	print(" Running skinny matrix test")
	X = np.random.randn(1000,50)
	f = partial(run_eigvals,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for eigvals:",min(times)/n)
	f = partial(run_svdvals,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for svdvals:",min(times)/n)
	f = partial(run_eigh,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for eig:",min(times)/n)
	f = partial(run_svd,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for svd:",min(times)/n)

def test_SVD_speed_cpu_sparse():
	def run_eigvals(X):
		op = SVD(use_eig=True,force_dense=False,backend='torch',vals_only=True,verbose=0)
		u,s,v=op.factorize(X=X)
	
	print("*****testing sparse torch cpu SVD speed*****")

	print(" Running square sparse matrix test")
	X = sparse.random(1000,1000,density=0.1)
	f = partial(run_eigvals,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for SVD:",min(times)/n)

	print(" Running skinny sparse matrix test")
	X = sparse.random(1000,50,density=0.1)
	f = partial(run_eigvals,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for SVD:",min(times)/n)

def test_SVD_speed_gpu_dense():
	def run_eigvals(X):
		op = SVD(use_eig=True,backend='torch_gpu',vals_only=True,verbose=0)
		u,s,v=op.factorize(X=X)
	def run_svdvals(X):
		op = SVD(use_eig=False,backend='torch_gpu',vals_only=True,verbose=0)
		u,s,v=op.factorize(X=X)
	def run_eigh(X):
		op = SVD(use_eig=True,backend='torch_gpu',vals_only=False,verbose=0)
		u,s,v=op.factorize(X=X)
	def run_svd(X):
		op = SVD(use_eig=False,backend='torch_gpu',vals_only=False,verbose=0)
		u,s,v=op.factorize(X=X)

	print("*****testing dense gpu SVD speed*****")
	print(" Running square matrix test")
	X = np.random.randn(1000,1000)
	f = partial(run_eigvals,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for eigvals:",min(times)/n)
	f = partial(run_svdvals,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for svdvals:",min(times)/n)
	f = partial(run_eigh,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for eig:",min(times)/n)
	f = partial(run_svd,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for svd:",min(times)/n)

	
	print(" Running skinny matrix test")
	X = np.random.randn(1000,50)
	f = partial(run_eigvals,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for eigvals:",min(times)/n)
	f = partial(run_svdvals,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for svdvals:",min(times)/n)
	f = partial(run_eigh,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for eig:",min(times)/n)
	f = partial(run_svd,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for svd:",min(times)/n)



def test_SVD_speed_gpu_sparse():
	def run_eigvals(X):
		op = SVD(use_eig=True,force_dense=False,backend='torch_gpu',vals_only=True,verbose=0)
		u,s,v=op.factorize(X=X)
	def run_svdvals(X):
		op = SVD(use_eig=False,force_dense=False,backend='torch_gpu',vals_only=True,verbose=0)
		u,s,v=op.factorize(X=X)
	def run_eigh(X):
		op = SVD(use_eig=True,force_dense=False,backend='torch_gpu',vals_only=False,verbose=0)
		u,s,v=op.factorize(X=X)
	def run_svd(X):
		op = SVD(use_eig=False,force_dense=False,backend='torch_gpu',vals_only=False,verbose=0)
		u,s,v=op.factorize(X=X)

	print("*****testing sparse gpu SVD speed*****")

	print(" Running square sparse matrix test")
	X = sparse.random(1000,1000,density=0.1)
	f = partial(run_eigvals,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for SVD:",min(times)/n)

	print(" Running skinny sparse matrix test")
	X = sparse.random(1000,50,density=0.1)
	f = partial(run_eigvals,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for SVD:",min(times)/n)
	

def test_Sinkhorn_speed():
	def run_sinkhorn_scipy(X):
		op = Sinkhorn(backend='scipy',verbose=0)
		op.fit_transform(X)
	def run_sinkhorn_torch(X):
		op = Sinkhorn(backend='torch',verbose=0)
		op.fit_transform(X)
	def run_sinkhorn_torch_gpu(X):
		op = Sinkhorn(backend='torch_gpu',verbose=0)
		op.fit_transform(X)
	print("*****Testing dense Sinkhorn speed*****")
	X = sparse.random(10000,10000,density=0.01)
	Y = X.toarray()
	f = partial(run_sinkhorn_scipy,Y)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for scipy:",min(times)/n)
	f = partial(run_sinkhorn_torch,Y)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for torch:",min(times)/n)
	f = partial(run_sinkhorn_torch_gpu,Y)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for torch_gpu:",min(times)/n)
	print("*****Testing sparse Sinkhorn speed*****")
	f = partial(run_sinkhorn_scipy,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for scipy:",min(times)/n)
	f = partial(run_sinkhorn_torch,X)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for torch:",min(times)/n)
	f = partial(run_sinkhorn_torch_gpu,Y)
	times = timeit.Timer(f).repeat(repeats,n)
	print("  Average time for torch_gpu:",min(times)/n)