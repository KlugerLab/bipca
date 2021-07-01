"""This file includes code for generating example data
"""
import numpy as np
import numpy.matlib as matlib
import scanpy as sc
from anndata import AnnData
from numpy.random import default_rng


def multinomial_data(nrows=500, ncols=1000, rank=10, sample_rate=100, simple=False):
	"""
	Generate multinomial distributed data of prescribed rank.
	
	Parameters
	----------
	nrows : int, default 500
	    Description
	ncols : int, default 1000
	    Description
	rank : int, default 10
	    Description
	sample_rate : int, default 100
	    Description
	simple : bool, optional
	    Description
	
	Returns
	-------
	TYPE
	    Description
	"""
	#the probability basis for the data
	p = np.random.multinomial(nrows,[1/nrows]*nrows,rank) / nrows
	if simple:
		cluster_size = np.floor(ncols/rank).astype(int)
		PX = matlib.repmat(p.T,1,cluster_size)
		rem = ncols-PX.shape[1]
		if rem > 0:
			PX = np.hstack((matlib.repmat(p.T[:,0],1,rem),PX))
	else:
		#draw random loadings
		loading = np.random.multinomial(rank, [1/rank]*rank, ncols) / rank
		#the ground truth probability matrix
		PX = (loading @ p).T
	#the data
	X = np.vstack([np.random.multinomial(sample_rate,PX[:,i]) for i in range(ncols)])
	return X, PX

def poisson_data(nrows=500, ncols=1000, rank=10, noise = 1, seed = 42):
	"""Summary
	
	Parameters
	----------
	nrows : int, optional
	    Description
	ncols : int, optional
	    Description
	rank : int, optional
	    Description
	noise : int, optional
	    Description
	seed : int, optional
	    Description
	
	Returns
	-------
	TYPE
	    Description
	"""
	rng = default_rng(seed = seed)
	S = np.exp(2*rng.standard_normal(size=(nrows,rank)));
	coeff = rng.uniform(size=(rank,ncols));
	X = S@coeff;
	X = X/X.mean() * noise; # Normalized to have average SNR = 1
	Y = rng.poisson(lam=X);  # Poisson sampling

	return Y, X


class ScanpyPipeline(object):

	"""Load an .h5ad raw dataset into ScanPy and run the standard transformations to it.
	"""
	
	def __init__(self, fname, readfun = sc.read_h5ad, adata=None):
		self.fname = fname
		if isinstance(adata,AnnData):
			self.adata_raw = adata
		else:
			self.adata_raw = readfun(fname)
		writename = fname.split('.')
		writename = writename[:-1]
		if readfun is not sc.read_h5ad:
			self.adata_raw.write('.'.join(writename)+'.h5ad')

		self.results_file = '.'.join(writename) + '_standard.h5ad'
	def fit(self, min_genes = 100, min_cells = 100, 
		max_n_genes_by_counts = 2500, mt_pct_counts = 5, 
		target_sum = 1e4,reset=False):

		adata = self.adata_raw.copy()
		##filter cells and genes
		sc.pp.filter_cells(adata, min_genes=min_genes)
		sc.pp.filter_genes(adata, min_cells=min_cells)

		adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
		sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

		adata = adata[adata.obs.n_genes_by_counts < max_n_genes_by_counts, :]
		adata = adata[adata.obs.pct_counts_mt < mt_pct_counts, :]

		sc.pp.normalize_total(adata, target_sum=target_sum)
		sc.pp.log1p(adata)
		sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

		adata.raw = adata

		adata = adata[:, adata.var.highly_variable]

		sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

		sc.pp.scale(adata, max_value=10)

		self.adata = adata
		self.adata.write(self.results_file)