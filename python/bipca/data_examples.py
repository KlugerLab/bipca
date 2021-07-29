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
			self.basename ='.'.join(writename)+'.h5ad'
		else:
			self.basename = fname

		self.results_file = '.'.join(writename) + '_standard.h5ad'
	def fit(self, min_genes = 100, min_cells = 100, 
		max_n_genes_by_counts = 2500, max_n_cells_by_counts=100000, mt_pct_counts = 5, 
		target_sum = 1e4,log_normalize= False, write=False,reset=False):

		adata = self.adata_raw.copy()
		##filter cells and genes
		self.cells_kept,self.n_genes = sc.pp.filter_cells(adata, min_genes=min_genes,inplace=False)
		self.genes_kept,self.n_cells= sc.pp.filter_genes(adata[self.cells_kept,:], min_cells=min_cells,inplace=False)

		adata.obs['n_genes'] = self.n_genes
		adata.var['n_cells'] = self.n_cells
		adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
		sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
		self.cells_kept = self.cells_kept & (adata.obs.n_genes_by_counts < max_n_genes_by_counts)
		self.cells_kept = self.cells_kept & (adata.obs.pct_counts_mt < mt_pct_counts)
		self.genes_kept = self.genes_kept & (adata.var.n_cells_by_counts < max_n_cells_by_counts)

		for k in adata.obs.keys():
			if k not in self.adata_raw.obs.keys():
				try:
					self.adata_raw.obs[k] = adata.obs[k]
				except Exception as e:
					print(e)
		for k in adata.var.keys():
			if k not in self.adata_raw.var.keys():
				try:
					self.adata_raw.var[k] = adata.var[k]
				except Exception as e:
					print(e)

		adata = adata[self.cells_kept, self.genes_kept]
		self.adata_filtered = adata.copy()
		if log_normalize:
			sc.pp.normalize_total(adata, target_sum=target_sum)
			sc.pp.log1p(adata)
			sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
			self.highly_variable = adata.var.highly_variable
			adata.raw = adata

			adata = adata[:, adata.var.highly_variable]

			sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

			sc.pp.scale(adata, max_value=10)

		self.adata = adata
		if write:
			self.write()
	def write(self, adata = None, fname = None):
		if adata is None:
			if isinstance(fname, str):
				fname_raw = fname+'raw'
				fname_standard = fname + 'standard'
			else:
				fname_raw = self.basename
				fname_standard = self.results_file


			self.adata.write(fname_standard)
			self.adata_raw.write(fname_raw)
		else:
			adata.write(fname)
