from os.path import exists
from os import remove
import scanpy as sc
import numpy as np
import bipca
import scipy.sparse as sparse
import pathlib
test_datadir = str(pathlib.Path(__file__).parent.resolve())
sc.settings.datasetdir = test_datadir
path_to_raw_data = str(test_datadir) + '/pbmc3k_raw.h5ad'
path_to_filtered_data =  str(test_datadir) + '/pbmc3k_filtered.h5ad'
path_to_filtered_sparse_data =  str(test_datadir) + '/pbmc3k_filtered_sparse.h5ad'

test_output_path = str(test_datadir) + '/test_pbmc3k_bipca_output.h5ad'

raw_adata = sc.datasets.pbmc3k()
if exists(path_to_filtered_data):
	filtered_adata = sc.read_h5ad(path_to_filtered_data)
else:
	
	filtered_adata = raw_adata[:,:1000]
	sc.pp.filter_cells(filtered_adata, min_genes=10)
	sc.pp.filter_genes(filtered_adata, min_cells=10)
	sparse_filtered_adata = filtered_adata.copy()
	filtered_adata.write(path_to_filtered_data)
	sparse_filtered_adata.X = sparse.csr_matrix(sparse_filtered_adata.X)
	sparse_filtered_adata.write(path_to_filtered_sparse_data)
