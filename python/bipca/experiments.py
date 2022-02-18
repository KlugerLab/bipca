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
from collections.abc import Iterable
from sklearn.model_selection import KFold
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def knn_classifier(X=None,labels_true=None,k_cv=5,train_ratio=0.8,
                    K=None,train_metric=None,metrics=None,
                    KNeighbors_kwargs={},train_metric_kwargs={},
                    metrics_kwargs={},**kwargs):
    #check we have enough labels
    labels = np.asarray(labels_true)
    N=len(labels)
    assert len(labels) in X.shape
    #put label dimension on the rows
    if len(labels)==X.shape[1]:
        X=X.T
    #parse specified Ks
    if K is None:
        K = [2, 5,10,20,40,80,160]
    if not isinstance(K, Iterable):
        #coerce it to an iterable
        K=[K]
    K=np.asarray(K)
    k_score=np.zeros_like(K)

    #parse the validation & cv metrics
    #user-specified metrics must accept y_true and y_pred:
    #if the function you want to use does not support that, then make a lambda function
    #if you make a lambda function, make sure that it accepts the dictionary **metrics_kwargs[fun]
    #and passes it as **kwargs
    if metrics is None:
        metrics=[train_metric]
    for fun in metrics:
        if fun not in metrics_kwargs:
            if fun==train_metric:
                metrics_kwargs[fun]=train_metric_kwargs
            else:
                metrics_kwargs[fun]={}

    #split the data into train & validate sets 
    (X_train,X_validate), (Y_train,Y_validate) = split_arrays([X,labels],train_ratio=train_ratio)
    #start the training - get k by cross validation
    ##this could be abstracted a lot by placing it into a separate cv function
    KNeighbors_kwargs.pop('n_neighbors', None)
    for kx, k in enumerate(K): 
        neigh=KNeighborsClassifier(n_neighbors=k,**KNeighbors_kwargs)
        for train, test in KFold(k_cv).split(X_train, Y_train):
            neigh.fit(X_train[train,:], Y_train[train])
            if train_metric is None:
                k_score[kx]+=neigh.score(X_train[test,:], Y_train[test])
            else:
                test_pred=neigh.predict(X_train[test,:])
                k_score[kx]+=train_metric(y_true=Y_train[test],y_pred=test_pred,  **train_metric_kwargs)
    k_score/=k_cv #take the average
    k=K[np.argmax(k_score)]
    neigh=KNeighborsClassifier(n_neighbors=k, **KNeighbors_kwargs)
    validate_pred=neigh.predict(X_validate)
    scores={}
    for metric in metrics:
        if metric is None:
            scores['score']=neigh.score(X_validate,Y_validate)
        else:
            scores[metric]=metric(y_true=Y_validate,y_pred=validate_pred, **metrics_kwargs[metric])
    if len(scores.keys())==1:
        #collapse the scores to a single number
        scores=scores[list(scores.keys())[0]]
    #return the scores, the classifier, and the converged_k
    return scores, neigh, k

def cluster_quality(X=None,labels_true=None,labels_pred=None,
                    algorithm=KMeans,
                    metrics=None,algorithm_kwargs={},metrics_kwargs={},
                    **kwargs):
    #algorithm should be a function that accepts X and kwargs, and outputs cluster labels,
    #for example,
    #let algorithm_kwargs={n_clusters:8}
    #algorithm=lambda X, **algorithm_kwargs: KMeans(**algorithm_kwargs).fit_predict(X)
    #then algorithm(X,**algorithm_kwargs) runs kmeans with n_clusters=8 on X and returns the fitted labels.
    #metrics should be a single metric or a list of metrics, each of which 
    #accepts X, labels_true, labels_pred. if you need a metric that doesn't accept these,
    #then wrap it in a lambda function that accepts X, labels_true, labels_pred, and **metrics_kwargs
    # the keys of metrics_kwargs should be the actual functions that you pass into metrics
    # so if you're passing a lambda function you'll need to use that lambda function as the key for kwargs
    
    if metrics==None:#default metrics
        metrics=[lambda X,labels_true,labels_pred, **metrics_kwargs: adjusted_rand_score(labels_true,labels_pred)]
    if not isinstance(metrics,Iterable):
        metrics=[metrics]
    for metric in metrics:
        #build the metric kwarg dictionary
        if metric not in metrics_kwargs.keys():
            metrics_kwargs[metric]={}
    if algorithm==KMeans:#default algorithm
        algorithm=lambda X, **algorithm_kwargs: algorithm(**algorithm_kwargs).fit_predict(X)
        if 'n_clusters' not in algorithm_kwargs.keys():
            algorithm_kwargs['n_clusters']=algorithm_kwargs.pop('k', len(np.unique(labels)))
    if algorithm==None:
        pass
    else:
        labels_pred=algorithm(X,**algorithm_kwargs)
    scores={metric:metric(X=X, labels_true=labels_true,labels_pred=labels_pred,**metrics_kwargs[metric]) for metric in metrics}
    if len(scores.keys())==1:
        #collapse the scores to a single number
        scores=scores[list(scores.keys())[0]]
    return scores
    
def split_arrays(arrays, train_ratio=0.8):
    # yield train & test indices given a ratio
    # this function expects the first dimension (the len) of all arrays to be equal.
    if isinstance(arrays, np.ndarray): #single array
        arrays=[arrays]
    arrays=[np.asarray(array) for array in arrays] #cast everything to np.array
    lens=[array.shape[0] for array in arrays]
    if not all([lens[0]==length for length in lens]):
        raise ValueError("Not all arrays are the same length along the first dimension")
    N=lens[0]
    N_train=np.ceil(train_ratio * N).astype(int)
    idx=np.random.permutation(N)
    train_idx=idx[:N_train]
    validate_idx=idx[N_train:]
    return tuple([(array[train_idx], array[validate_idx]) for array in arrays])


def quantify_data(X,labels_true=None,labels_pred=None,npca=100,pcafun=PCA,
            method=knn_classifier,pca_kwargs={},**kwargs):
    #pcafun can be sklearn.decomposition.PCA or a function that accepts
    #X as the first arg and npca as the second, positional argument,
    #as well as a dictionary pca_kwargs
    #an example is 
    #lambda x, npca,pca_kwargs: sklearn.decomposition.TruncatedSVD(n_components=npca,**pca_kwargs).fit_transform(x)
    if labels_true is not None:
        labels_true = np.asarray(labels_true)
        N=len(labels_true)
        assert len(labels_true) in X.shape
        #put label dimension on the rows
        if len(labels_true)==X.shape[1]:
            X=X.T
    #run pca
    if npca>0:
        if npca<np.min(X.shape[0]):
            if pcafun==PCA:
                pcafun=lambda X, npca, **pca_kwargs: PCA(npca,**pca_kwargs).fit_transform(X)
            X = pcafun(X, npca,**pca_kwargs)
        else:
            raise ValueError("npca was larger than the minimum dimension of X.")
    return method(X=X,labels_true=labels_true,labels_pred=labels_pred,**kwargs)


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