"""Summary
"""
import numpy as np
from scipy.sparse import linalg
import scanpy as sc
import pandas as pd
from statsmodels import robust
import warnings
from . import bipca, math
from .data_examples import ScanpyPipeline


def gene_set_experiment(sp, algorithms=['bipca','log1p','hvg'], label = "clusters", 
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
    fig : None, optional
        Description
    k : int, optional
        Number of PCs to compute. 
        By default, if 'bipca' is an algorithm, use BiPCA.mp_rank.
    verbose : bool, optional
        Print experiment status to the console.
    **kwargs
        Description
    
    Returns
    -------
    gene_sets : dict of dicts of sets of strings
        For each cluster in `sp`, for each algorithm, 
        unique genes in the top k principal components of the data.
    k_used : dict
        The k used for each cluster.
    fig : matplotlib.Figure 
    axes
        if fig is True, a figure object and accompanying axes are returned
    
    Raises
    ------
    TypeError
        Description
    ValueError
        Description
    TypeError
    ValueError
    
    """
    def get_genes_from_adata_v(adata, v, k):
        """Summary
        
        Parameters
        ----------
        adata : TYPE
            Description
        v : TYPE
            Description
        k : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        if magnitude:
            v = abs(v)
        else:
            if negative:
                v = -v
        genes = set()
        for i in range(k):
            spc = np.argsort(v[:,i])[::-1]
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
            bipcaop = bipca.BiPCA(exact=True, approximate_sigma=True,
                sinkhorn_backend='torch', svd_backend='torch',
                subsample_size = 2500, n_components=50, qits=11, verbose = verbose)
            adata = cluster_adata.adata_filtered
            bipcaop.fit(adata.X)
            k_used[clust] = bipcaop.mp_rank
            if bipcaop.V_Z.shape[0]!=adata.shape[1]:
                gene_sets[clust]['bipca'] = get_genes_from_adata_v(adata, bipcaop.U_Z, k_used[clust])
            else:
                gene_sets[clust]['bipca'] = get_genes_from_adata_v(adata, bipcaop.V_Z, k_used[clust])
        else:
            k_used[clust] = k 


        for alg in algorithms:
            if alg == 'bipca':
                bipcaop = bipca.BiPCA(exact=True, approximate_sigma=True,
                sinkhorn_backend='torch', svd_backend='torch',
                subsample_size = 2500, n_components=50, qits=11, verbose = verbose)
                adata = cluster_adata.adata_filtered
                bipcaop.fit(adata.X)
                if bipcaop.V_Z.shape[0]!=adata.shape[1]:
                    v = bipcaop.U_Z
                else:
                    v = bipcaop.V_Z
            else:
                if alg == 'log1p':
                    adata = cluster_adata.adata.raw.to_adata()
                    X = adata.X
                elif alg == 'hvg':
                    adata = cluster_adata.adata
                    X = adata.X
                _,_,v = linalg.svds(X,k=k_used[clust])
                v = v.T
            gene_sets[clust][alg] = get_genes_from_adata_v(adata, v, k_used[clust])

    if not fig:
        return gene_sets, k_used

def knn_mixing(data_list, batch_labels, N = None):
    """knn_mixing:
    Compute batch effect mixing by comparing local neighborhoods to global proportions
    using chi-squared goodness of fit.
    
    Parameters
    ----------
    data_list : TYPE
        Description
    batch_labels : TYPE
        Description
    N : None, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    num_datasets = len(data_list)
    num_samples = data_list[0].shape[0]
    
    if N is None: # Get the number of nearest neighbors to embed.
        N = np.round(np.logspace(np.log10(50),np.log10(num_samples))).astype(int)
        
    batches, counts = np.unique(batch_labels,return_counts=True)
    k = len(batches) #the number of possible batches
    pi = counts/num_samples #the "theoretical" probabilities
    
    
    labels = np.zeros((num_samples, num_samples, num_datasets, k))
    output = np.zeros((num_samples, len(N), num_datasets))
    
    for data_ix, data in enumerate(data_list):
        dists = squareform(pdist(data)) #get the distances to the neighbors
        argsorted_points = np.argsort(dists) #sort them
        labels_bulk = batch_labels[argsorted_points] #sort the labels using the distances
        for k_ix, batch_label in enumerate(batches):
            labels[:,:,data_ix, k_ix] = labels_bulk == batch_label
            
    for n_ix, n in enumerate(N): #for nearest neighbor width
        for k_ix, pi_k_ix in enumerate(pi): #for items
            for data_ix, _ in enumerate(data_list): #for datasets
                # the point-wise number of labels that match the current item
                x_k_ix = labels[:,:n, data_ix, k_ix].sum(1) 
                #compute the marginal chisquared test statistic for the current item
                E_k_ix = n * pi_k_ix
                cs_k_ix = x_k_ix - E_k_ix
                cs_k_ix = cs_k_ix**2
                cs_k_ix = cs_k_ix / E_k_ix
                #the statistic is summed into `output` over the items
                output[:,n_ix,data_ix] += cs_k_ix 
    output = np.sum(output>=stats.chi2.ppf(q=0.95,df=k-1),axis=0)/num_samples
    return output

def get_mean_var(X,axis=0,mean=None,var=None):
    if mean is None:
        mean = np.mean(X,axis=axis,dtype=np.float64)
    if var is None:
        mean_sq = np.multiply(X,X).mean(axis=axis,dtype=np.float64)
        var = mean_sq - mean ** 2
        var *= X.shape[axis]/(X.shape[axis]-1)
    return mean, var
def get_normalized_dispersion(X,axis=0,mean=None,var=None):
    #copied from scanpy
    mean, var = get_mean_var(X,axis=axis,mean=mean,var=var)
    mean[mean==0] = 1e-12
    dispersion = var / mean

    df = pd.DataFrame()
    df['means'] = mean
    df['dispersions'] = dispersion

    df['mean_bin'] = pd.cut(
        df['means'],
        np.r_[-np.inf,np.percentile(df['means'],np.arange(10,105,5)), np.inf])

    disp_grouped = df.groupby('mean_bin')['dispersions']
    disp_median_bin = disp_grouped.median()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        disp_mad_bin = disp_grouped.apply(robust.mad)
        df['dispersions_norm'] = (
                df['dispersions'].values - disp_median_bin[df['mean_bin'].values].values
            ) / disp_mad_bin[df['mean_bin'].values].values
    dispersion_norm = df['dispersions_norm'].values

    return dispersion_norm

def get_top_n(arr,n):
    ixs = np.argsort(arr)[::-1]
    return ixs[:n]