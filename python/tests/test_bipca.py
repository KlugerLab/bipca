import scanpy as sc
from bipca import BiPCA
import unittest
from nose2.tools import params
import numpy as np

##make the data to be tested in this file
data = sc.datasets.pbmc3k()
adata = sc.read_h5ad('data/pbmc3k_raw.h5ad')
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs.pct_counts_mt < 5, :]
X = adata.X.toarray()

class Test_BiPCA(unittest.TestCase):

	@params(('scipy',True),('torch_cpu',True),('torch_gpu',True),('dask',True),
		('scipy',False),('torch_cpu',False),('torch_gpu',False),('dask',False))
	def test_PCA_shape_property_instantiation(self,backend,exact):
		op = BiPCA(exact=exact,qits=0, q=0.26, backend=backend,sinkhorn_backend='',approximate_sigma = False,verbose = 0)
		assert op.rotated_right_pcs is None
		assert op.rotated_left_pcs is None
		assert op.traditional_left_pcs is None
		assert op.traditional_right_pcs is None
		assert op.biwhitened_right_pcs is None
		assert op.biwhitened_left_pcs is None
		op.fit(X)

		pca_out = op.PCA(which='right',pca_method='rotate')
		assert  pca_out.shape == (X.shape[1],op.mp_rank)
		assert np.allclose(op.rotated_right_pcs, pca_out)
		pca_out = op.PCA(which='left',pca_method='rotate')
		assert  pca_out.shape == (X.shape[0],op.mp_rank)
		assert np.allclose(op.rotated_left_pcs, pca_out)

		pca_out = op.PCA(which='right',pca_method='biwhitened')
		assert  pca_out.shape == (X.shape[1],op.mp_rank)
		assert np.allclose(op.biwhitened_right_pcs, pca_out)
		pca_out = op.PCA(which='left',pca_method='biwhitened')
		assert  pca_out.shape == (X.shape[0],op.mp_rank)
		assert np.allclose(op.biwhitened_left_pcs, pca_out)

		pca_out = op.PCA(which='right',pca_method='traditional')
		assert  pca_out.shape == (X.shape[1],op.mp_rank)
		assert np.allclose(op.traditional_right_pcs, pca_out)
		pca_out = op.PCA(which='left',pca_method='traditional')
		assert  pca_out.shape == (X.shape[0],op.mp_rank)
		assert np.allclose(op.traditional_left_pcs, pca_out)
