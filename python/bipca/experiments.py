"""Summary
"""
import numpy as np
from scipy.sparse import linalg
import scanpy as sc
import pandas as pd
from . import bipca, math
from .data_examples import ScanpyPipeline


def gene_set_experiment(sp, algorithms, label = "clusters", 
                        magnitude=True, negative=False, fig=None, 
                        k=None, verbose = True,**kwargs):
    """gene_set_experiment
    
    Parameters
    ----------
    sp : bipca.data_examples.ScanpyPipeline 
        Data to run the experiment on.
        `sp.adata_raw.obs[label]` must encode the clusters to extract gene sets from.
    algorithms : list of str
        List of algorithms to run method on
    label : str, optional
        Default "clusters"
    magnitude : bool, optional
        Use the magnitude of the principal components to learn genes.
        Default `True`.
    negative : bool, optional
        Compute top negative genes. Only relevant when `magnitude==False`.
        Default `False`. 
    k : int, optional 
        Number of PCs to compute. 
        By default, if 'bipca' is an algorithm, use BiPCA.mp_rank.
    verbose : bool, optional
        Print experiment status to the console.
    Returns
    -------
    gene_sets : dict of dicts of sets of strings
        For each cluster in `sp`, for each algorithm, 
        unique genes in the top k principal components of the data.
    k_used : dict
        The k used for each cluster.
    fig : matplotlib.Figure 
    axes : 
        if fig is True, a figure object and accompanying axes are returned

    Raises
    ------
    TypeError
    ValueError

    """
    def get_genes_from_adata_v(adata, v, k):
        if magnitude:
            v = abs(v)
        else:
            if negative:
                v = -v
        genes = set()
        for i in range(k):
            spc = np.argsort(signal_pcs[:,i])[::-1]
            for gene in list(adata.var_names[spc])[:10]: 
                genes.add(gene)
        return genes

    algorithms_are_strings = all([isinstance(ele,str) for ele in algorithms])
    if not algorithms_are_strings:
        raise TypeError("All passed algorithms must be strings")
    algorithms = [algorithm.lower() for algorithm in algorithms]



    unique_clusters=pd.unique(sp.adata_raw.obs['clusters'])
    gene_sets = {clust:{alg:set() for alg in algorithms} for clust in unique_clusters}
    k_used = {clust: None for clust in unique_clusters}

    if k is None: #we will use the bipca rank
        if 'bipca' not in algorithms:
            raise ValueError("If k is None, then 'bipca' must be in algorithms in order to estimate the rank of the data.")
        algorithms.remove('bipca')

    for clust in unique_clusters:
        gate = sp.adata_raw.obs['clusters'] == clust
        if verbose:
            print("Processing cluster %s" % clust)
        cluster_adata = ScanpyPipeline('ERROR', adata = sp.adata_raw[gate,:]) # we dont actually ever write so fname should never get triggered
        cluster_adata.fit(**kwargs)
        if k is None:
            ##we know that bipca has been removed, so we do it first to get a k.
            bipcaop = bipca.BiPCA(exact=True,approximate_sigma=True,
                sinkhorn_backend='torch', svd_backend='torch',
                subsample_size = 2500, n_components=50, qits=11,verbose = verbose)
            adata = cluster_adata.adata_filtered
            bipcaop.fit(adata.X)
            k_used[clust] = bipcaop.mp_rank
            gene_sets[clust]['bipca'] = get_genes_from_adata_v(adata, bipcaop.V_mp, k_used[clust])
        else:
            k_used[clust] = k 
        for alg in algorithms:
            if alg == 'log1p':
                adata = cluster_adata.adata.raw.to_adata()
                X = adata.X
            elif alg == 'hvg':
                adata = cluster_adata.adata
                X = adata.X
            _,_,v = linalg.svds(X,k=k)
            gene_sets[clust][alg] = get_genes_from_adata_v(adata, v, k_used[clust])

    if not fig:
        return gene_sets, k_used
