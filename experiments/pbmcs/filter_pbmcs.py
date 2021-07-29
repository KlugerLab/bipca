#!/usr/bin/env python3
##Load the full PBMC dataset from ../data/, filter it and logtransform.  Save the filtered h5 file.

##### FILTERING PARAMETERS
min_genes = 100
min_cells = 100
max_n_genes_by_counts = 100000
mt_pct_counts = 10

##### MODULES
import numpy as np
from scipy.io import mmread
import scanpy as sp
from os.path import exists as file_exists
import scipy.sparse as sparse
import scipy.linalg as spla
import pandas as pd
import h5sparse
from pathlib import Path
import anndata
from bipca.data_examples import ScanpyPipeline


spipe = ScanpyPipeline(fname = data_h5_file,adata = dataset)
path_to_data = '../data/'
data_mtx_file = path_to_data + 'sparse_purified_PBMC_filtered_big.mtx'
data_h5_file = path_to_data + 'purified_PBMC_filtered_big.h5'
labels_csv_file = path_to_data + 'purified_PBMC_filtered_labels_nofilt.csv'
genes_csv_file = path_to_data + 'purified_PBMC_filtered_genes_nofilt.csv'
labels  = pd.read_csv(labels_csv_file).to_numpy()
genenames = pd.read_csv(genes_csv_file).to_numpy()
#load NPZ of data if available
if not file_exists(data_h5_file):
    X = mmread(data_mtx_file).tocsr()
    with h5sparse.File(data_h5_file,'a') as h5f:
        h5f.create_dataset('X', data=X.toarray())
        h5f.create_dataset('labels', data=labels)
        h5f.create_dataset('col_names', data=genenames)
labels  = pd.read_csv(labels_csv_file).to_numpy()
genenames = pd.read_csv(genes_csv_file).to_numpy()
dataset = sp.read_hdf(data_h5_file, 'X')

#map the clusters
clusters = labels.astype('U').flatten()
genenames = genenames.astype('U').flatten()
dataset.var_names = genenames
dataset.var_names_make_unique()
dataset.obs['clusters'] = clusters

spipe = ScanpyPipeline(fname = 'pbmcs.h5ad',adata = dataset)

spipe.fit(min_genes = min_genes, min_cells = min_cells, 
          max_n_genes_by_counts = max_n_genes_by_counts,
          mt_pct_counts = mt_pct_counts, log_normalize=False)

spipe.write()