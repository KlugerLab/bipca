"""Summary
"""
from typing import Union, Tuple, Optional, Any, Literal
import numpy as np
from scipy.sparse import linalg
from bipca.utils import is_tensor, issparse
from bipca.safe_basics import *
from torch import log as tensor_log
from numbers import Number
import scanpy as sc
from anndata import AnnData
import pandas as pd
from statsmodels import robust
import warnings
import bipca
import bipca.math as math
from bipca.data_examples import ScanpyPipeline
from collections.abc import Iterable
from sklearn.model_selection import KFold
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from collections import OrderedDict
from sklearn.metrics.pairwise import euclidean_distances
from openTSNE import TSNE
from scipy.stats import mannwhitneyu

from bipca.experiments.utils import (
    parse_number_less_than,
    parse_number_greater_than,
    get_rng,
    parse_mrows_ncols_rank,
)


def knn_classifier(
    X=None,
    labels_true=None,
    k_cv=5,
    train_ratio=0.8,
    K=None,
    train_metric=None,
    metrics=None,
    random_state=34,
    KNeighbors_kwargs={},
    train_metric_kwargs={},
    metrics_kwargs={},
):
    """knn_classifier:
    Train and validate a KNN classifier with k-fold cross validation.
    Parameters
    ----------
    X : array-like of shape (M,N)
        Data to run the classifier on.
    labels_true : array-like of shape (M,)
        Ground truth labels to test.
    k_cv : int, default 5
        Number of cross-validation folds to use to learn the k-NN parameter `k` during training
    train_ratio : 0 < float < 1.0, default 0.8
        Ratio of training split to final validation split.
    K : array-like of ints, optional
        Integer k-neighborhood sizes to cross validate.
        Default [2, 5, 10, 20, 40, 80, 160]
    train_metric : function accepting `y_true` and `y_pred` kwargs, optional
        Metric to use during cross-validation to select `k`. Consider classifier metrics from `sklearn.metrics`.
        User-designed functions must accept `y_true` and `y_pred` kwargs for labels.
        Defaults to `sklearn.neighbors.KNeighborsClassifier.score`
    metrics : list of functions accepting `y_true` and `y_pred` kwargs, optional
        Metrics to report validation results. Consider classifier metrics from `sklearn.metrics`.
        User-designed functions must accept `y_true` and `y_pred` kwargs for labels.
        Defaults to `train_metric`.
    KNeighbors_kwargs : dict, default {}
        Keyword arguments for `sklearn.neighbors.KNeighborsClassifier` constructor
    train_metric_kwargs : dict, default {}
        Keyword arguments for the training metric
    metrics_kwargs : dict of dicts, default {}
        Dictionary of dictionaries. The first level must contain keys that correspond to the function *handles* contained in `metrics`.
        Each function handle key points to a dictionary of its corresponding kwargs.
    Returns
    -------
    scores : list
        validation scores
    neigh : `sklearn.neighbors.KNeighborsClassifier`
        Trained classifier.
    k : int
        The `k` parameter learned from cross validation.
    """
    # check we have enough labels
    labels = np.asarray(labels_true)
    N = len(labels)
    assert len(labels) in X.shape
    # put label dimension on the rows
    if len(labels) == X.shape[1]:
        X = X.T
    # parse specified Ks
    if K is None:
        K = [2, 5, 10, 20, 40, 80, 160]
    if not isinstance(K, Iterable):
        # coerce it to an iterable
        K = [K]
    K = np.asarray(K)
    k_score = np.zeros(K.shape)
    # parse the validation & cv metrics
    # user-specified metrics must accept y_true and y_pred:
    # if the function you want to use does not support that, then make a lambda function
    # if you make a lambda function, make sure that it accepts the dictionary **metrics_kwargs[fun]
    # and passes it as **kwargs
    if metrics is None:
        metrics = [train_metric]
    for fun in metrics:
        if fun not in metrics_kwargs:
            if fun == train_metric:
                metrics_kwargs[fun] = train_metric_kwargs
            else:
                metrics_kwargs[fun] = {}
    # split the data into train & validate sets
    (X_train, X_validate), (Y_train, Y_validate) = split_arrays(
        [X, labels], train_ratio=train_ratio, random_state=random_state
    )
    # start the training - get k by cross validation
    ##this could be abstracted a lot by placing it into a separate cv function
    KNeighbors_kwargs.pop("n_neighbors", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kx, k in enumerate(K):
            neigh = KNeighborsClassifier(n_neighbors=k, **KNeighbors_kwargs)
            for train, test in KFold(
                k_cv, shuffle=True, random_state=random_state
            ).split(X_train, Y_train):
                neigh.fit(X_train[train, :], Y_train[train])
                if train_metric is None:
                    k_score[kx] += neigh.score(X_train[test, :], Y_train[test])
                else:
                    test_pred = neigh.predict(X_train[test, :])
                    k_score[kx] += train_metric(
                        y_true=Y_train[test], y_pred=test_pred, **train_metric_kwargs
                    )
        k_score /= float(k_cv)  # take the average
        k = K[np.argmax(k_score)]
        neigh = KNeighborsClassifier(n_neighbors=k, **KNeighbors_kwargs).fit(
            X_train, Y_train
        )
        validate_pred = neigh.predict(X_validate)
        scores = {}
        for metric in metrics:
            if metric is None:
                scores["score"] = neigh.score(X_validate, Y_validate)
            else:
                scores[metric] = metric(
                    y_true=Y_validate, y_pred=validate_pred, **metrics_kwargs[metric]
                )
    if len(scores.keys()) == 1:
        # collapse the scores to a single number
        scores = scores[list(scores.keys())[0]]
    # return the scores, the classifier, and the converged_k
    return scores, neigh, k


def split_arrays(arrays, train_ratio=0.8, random_state=34):
    # yield train & test indices given a ratio
    # this function expects the first dimension (the len) of all arrays to be equal.
    if isinstance(arrays, np.ndarray):  # single array
        arrays = [arrays]
    arrays = [np.asarray(array) for array in arrays]  # cast everything to np.array
    lens = [array.shape[0] for array in arrays]
    if not all([lens[0] == length for length in lens]):
        raise ValueError("Not all arrays are the same length along the first dimension")
    N = lens[0]
    N_train = np.ceil(train_ratio * N).astype(int)
    idx = np.random.RandomState(seed=random_state).permutation(N)
    train_idx = idx[:N_train]
    validate_idx = idx[N_train:]
    return tuple([(array[train_idx], array[validate_idx]) for array in arrays])


def quantify_data(
    X,
    labels_true=None,
    labels_pred=None,
    npca=100,
    pcafun=PCA,
    method=knn_classifier,
    pca_kwargs={},
    **kwargs,
):
    # pcafun can be sklearn.decomposition.PCA or a function that accepts
    # X as the first arg and npca as the second, positional argument,
    # as well as a dictionary pca_kwargs
    # an example is
    # lambda x, npca,pca_kwargs: sklearn.decomposition.TruncatedSVD(n_components=npca,**pca_kwargs).fit_transform(x)
    if labels_true is not None:
        labels_true = np.asarray(labels_true)
        N = len(labels_true)
        assert len(labels_true) in X.shape
        # put label dimension on the rows
        if len(labels_true) == X.shape[1]:
            X = X.T
    # run pca
    if npca > 0:
        if npca < np.min(X.shape[0]):
            if pcafun == PCA:
                pcafun = lambda X, npca, **pca_kwargs: PCA(
                    npca, **pca_kwargs
                ).fit_transform(X)
            X = pcafun(X, npca, **pca_kwargs)
        else:
            raise ValueError("npca was larger than the minimum dimension of X.")
    return method(X=X, labels_true=labels_true, labels_pred=labels_pred, **kwargs)


def gene_set_experiment(
    sp,
    algorithms=["bipca", "log1p", "hvg"],
    label="clusters",
    magnitude=True,
    negative=False,
    fig=None,
    k=None,
    verbose=True,
    **kwargs,
):
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
            spc = np.argsort(v[:, i])[::-1]
            for gene in list(adata.var_names[spc])[:10]:
                genes.add(gene)
        return genes

    algorithms_are_strings = all([isinstance(ele, str) for ele in algorithms])
    if not algorithms_are_strings:
        raise TypeError("All passed algorithms must be strings")
    algorithms = [algorithm.lower() for algorithm in algorithms]

    unique_clusters = pd.unique(sp.adata_raw.obs["clusters"])
    gene_sets = {clust: {alg: set() for alg in algorithms} for clust in unique_clusters}
    k_used = {clust: None for clust in unique_clusters}

    if k is None:  # we will use the bipca rank
        if "bipca" not in algorithms:
            raise ValueError(
                "If k is None, then 'bipca' must be in algorithms in order to estimate the rank of the data."
            )
        algorithms.remove("bipca")

    for clust in unique_clusters:
        gate = sp.adata_raw.obs["clusters"] == clust
        if verbose:
            print("Processing cluster %s" % clust)
        cluster_adata = ScanpyPipeline(
            "ERROR", adata=sp.adata_raw[gate, :]
        )  # we dont actually ever write so fname should never get triggered
        cluster_adata.fit(**kwargs)
        if k is None:
            ##we know that bipca has been removed, so we do it first to get a k.
            bipcaop = bipca.BiPCA(
                exact=True,
                approximate_sigma=True,
                sinkhorn_backend="torch",
                svd_backend="torch",
                subsample_size=2500,
                n_components=50,
                qits=11,
                verbose=verbose,
            )
            adata = cluster_adata.adata_filtered
            bipcaop.fit(adata.X)
            k_used[clust] = bipcaop.mp_rank
            if bipcaop.V_Z.shape[0] != adata.shape[1]:
                gene_sets[clust]["bipca"] = get_genes_from_adata_v(
                    adata, bipcaop.U_Z, k_used[clust]
                )
            else:
                gene_sets[clust]["bipca"] = get_genes_from_adata_v(
                    adata, bipcaop.V_Z, k_used[clust]
                )
        else:
            k_used[clust] = k

        for alg in algorithms:
            if alg == "bipca":
                bipcaop = bipca.BiPCA(
                    exact=True,
                    approximate_sigma=True,
                    sinkhorn_backend="torch",
                    svd_backend="torch",
                    subsample_size=2500,
                    n_components=50,
                    qits=11,
                    verbose=verbose,
                )
                adata = cluster_adata.adata_filtered
                bipcaop.fit(adata.X)
                if bipcaop.V_Z.shape[0] != adata.shape[1]:
                    v = bipcaop.U_Z
                else:
                    v = bipcaop.V_Z
            else:
                if alg == "log1p":
                    adata = cluster_adata.adata.raw.to_adata()
                    X = adata.X
                elif alg == "hvg":
                    adata = cluster_adata.adata
                    X = adata.X
                _, _, v = linalg.svds(X, k=k_used[clust])
                v = v.T
            gene_sets[clust][alg] = get_genes_from_adata_v(adata, v, k_used[clust])

    if not fig:
        return gene_sets, k_used


def libsize_normalize(X, scale=1):
    """libsize_normalize:
    Normalize the data so that the rows sum to 1.

    Parameters
    ----------
    X : array-like or AnnData
        The input data to process.
    scale : {numbers.Number, 'median'}, default 1
        The scale factor to apply to the data
    Returns
    -------
    Y : array-like or AnnData"""

    libsize = sum(X, dim=1)
    if scale == "median":
        scale = np.median(libsize)
    scale /= libsize
    return multiply(X, scale[:, None])


def log1p(A, scale="median"):
    """log1p:
    Compute log1p transform commonly applied to single cell data. This procedure computes the sum along the rows of the data as the "library size".
    Then, the data is normalized so that the rows sum to 1. Next, a scale factor (by default the median of the library size) is multiplied by the normalized data. Finally,
    the natural log of 1 + the normalized and scaled data is computed.
    Parameters
    ----------
    A : array-like or AnnData
        The input data to process.
    scale : {numbers.Number, 'median'}, default 'median'
        The scale factor to apply to the data
    """
    if isinstance(A, AnnData):
        X = A.X
    else:
        X = A
    if scale != "median" and not isinstance(scale, Number):
        raise ValueError("`scale` must be 'median' or a Number")
    to_log = libsize_normalize(X, scale=scale)
    if is_tensor(X):
        return tensor_log(to_log + 1)
    else:
        if issparse(X):
            to_log.data += 1
            to_log.data = np.log(to_log.data)
            return to_log
        else:
            return np.log(to_log + 1)


def knn_matching(original_data, batched_data, batch_label=None, N=None):
    """knn_matching:
    Compute the percentage of neighbors in batched_data that are neighbors in original_data.
    If batch_label is supplied, the batch-wise percentages are computed.
    """
    pass


def knn_mixing(data_list, batch_labels, N=None):
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

    if N is None:  # Get the number of nearest neighbors to embed.
        N = np.round(np.logspace(np.log10(2), np.log10(num_samples))).astype(int)

    batches, counts = np.unique(batch_labels, return_counts=True)
    k = len(batches)  # the number of possible batches
    pi = counts / num_samples  # the "theoretical" probabilities

    labels = np.zeros((num_samples, num_samples, num_datasets, k))
    output = np.zeros((num_samples, len(N), num_datasets))

    for data_ix, data in enumerate(data_list):
        dists = squareform(pdist(data))  # get the distances to the neighbors
        argsorted_points = np.argsort(dists)  # sort them
        labels_bulk = batch_labels[
            argsorted_points
        ]  # sort the labels using the distances
        for k_ix, batch_label in enumerate(batches):
            labels[:, :, data_ix, k_ix] = labels_bulk == batch_label

    for n_ix, n in enumerate(N):  # for nearest neighbor width
        for k_ix, pi_k_ix in enumerate(pi):  # for items
            for data_ix, _ in enumerate(data_list):  # for datasets
                # the point-wise number of labels that match the current item
                x_k_ix = labels[:, :n, data_ix, k_ix].sum(1)
                # compute the marginal chisquared test statistic for the current item
                E_k_ix = n * pi_k_ix
                cs_k_ix = x_k_ix - E_k_ix
                cs_k_ix = cs_k_ix**2
                cs_k_ix = cs_k_ix / E_k_ix
                # the statistic is summed into `output` over the items
                output[:, n_ix, data_ix] += cs_k_ix
    output = np.sum(output >= chi2.ppf(q=0.95, df=k - 1), axis=0) / num_samples
    return output


def get_mean_var(X, axis=0, mean=None, var=None):
    if mean is None:
        mean = np.mean(X, axis=axis, dtype=np.float64)
    if var is None:
        mean_sq = np.multiply(X, X).mean(axis=axis, dtype=np.float64)
        var = mean_sq - mean**2
        var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var


def get_normalized_dispersion(X, axis=0, mean=None, var=None):
    # copied from scanpy
    mean, var = get_mean_var(X, axis=axis, mean=mean, var=var)
    mean[mean == 0] = 1e-12
    dispersion = var / mean

    df = pd.DataFrame()
    df["means"] = mean
    df["dispersions"] = dispersion

    df["mean_bin"] = pd.cut(
        df["means"],
        np.r_[-np.inf, np.percentile(df["means"], np.arange(10, 105, 5)), np.inf],
    )

    disp_grouped = df.groupby("mean_bin")["dispersions"]
    disp_median_bin = disp_grouped.median()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp_mad_bin = disp_grouped.apply(robust.mad)
        df["dispersions_norm"] = (
            df["dispersions"].values - disp_median_bin[df["mean_bin"].values].values
        ) / disp_mad_bin[df["mean_bin"].values].values
    dispersion_norm = df["dispersions_norm"].values

    return dispersion_norm


def get_top_n(arr, n):
    ixs = np.argsort(arr)[::-1]
    return ixs[:n]


def rank_to_sigma(rank, singular_values, shape):
    """rank_to_sigma:
    Compute the sigma (noise deviation) required to shrink a vector of singular values to a particular rank.

    Parameters
    ----------
    rank : `int` or array-like of length R
        Desired rank(s)
    singular_values : array-like
        Singular values to compute sigma for
    shape : tuple of ints
        Shape of matrix to shrink

    Returns
    -------
    sig_lower : float or np.ndarray of shape (R,)
        Lower bound on sigma required to shrink to the desired rank
    sig_upper : float or np.ndarray of shape (R,)
        Upper bound on sigma required to shrink to the desired rank
    mean_sigma : float or np.ndarray of shape (R,)
        Midpoints between upper and lower sigmas.
    """
    if isinstance(rank, Iterable):
        rank = np.asarray(rank)
        assert (rank <= len(singular_values)).all()
    else:
        assert isinstance(rank, int)
        assert rank <= len(singular_values)
    singular_values = np.asarray(singular_values)
    n = np.max(shape)
    MP = math.MarcenkoPastur(np.min(shape) / np.max(shape))
    sig_lower = (
        singular_values[rank] / (np.sqrt(n) * np.sqrt(MP.b))
        + 2 * np.finfo(np.float64).eps
    )
    sig_upper = singular_values[rank - 1] / (np.sqrt(n) * np.sqrt(MP.b))

    mean_sigma = (sig_lower + sig_upper) / 2
    return sig_lower, sig_upper, mean_sigma


def random_nonnegative_matrix(
    mrows: int,
    ncols: int,
    rank: int,
    libsize_mean: Number = 1000,
    entrywise_mean: Union[Literal[False], Number] = False,
    minimum_singular_value: Union[Literal[False], Number] = False,
    rng: Union[np.random._generator.Generator, Number] = 42,
) -> np.ndarray:
    """Generate a random nonnegative matrix."""
    rng = get_rng(rng)
    mrows, ncols, rank = parse_mrows_ncols_rank(mrows, ncols, rank)
    libsize = rng.lognormal(np.log(libsize_mean), sigma=0.1, size=(mrows,)).astype(int)
    # "modules"
    coeff = np.geomspace(0.0001, 0.05, num=rank * ncols)
    coeff = np.random.permutation(coeff).reshape(rank, ncols)
    loadings = rng.multinomial(libsize, pvals=[1 / rank] * rank)

    X = loadings @ coeff
    if entrywise_mean is not False:
        X /= X.mean()
        X *= entrywise_mean
    if minimum_singular_value is not False:
        S = (
            math.SVD(
                backend="torch",
                use_eig=True,
                n_components=-1,
                vals_only=True,
                verbose=False,
            )
            .fit(X)
            .S
        )
        min_sv = np.argsort(S)[::-1][rank - 1]
        min_sv = S[min_sv]
        factor = minimum_singular_value / min_sv
        X *= factor
    return X


def random_nonnegative_orthonormal_matrix(
    m: int, r: int, rng: Union[np.random._generator.Generator, Number] = 42
):
    """Generate a random nonnegative orthonormal matrix."""
    rng = get_rng(rng)
    m, _, r = parse_mrows_ncols_rank(m, m, r)
    nnzs = np.full(r + 1, m // r)
    if diff := int(m - sum(nnzs[1:])):
        nnzs[1 : diff + 1] += 1  # add 1 so that we have every basis element covered
    norm = 1 / np.sqrt(nnzs)
    nnzs[0] = 0
    nnzs = np.cumsum(nnzs)
    idxs = rng.permutation(m)
    output = np.zeros((m, r))
    for ix in range(1, len(nnzs)):
        idxs_select = idxs[nnzs[ix - 1] : nnzs[ix]]
        output[idxs_select, ix - 1] = norm[ix]
    return output


def random_nonnegative_factored_matrix(
    mrows: int,
    ncols: int,
    rank: int,
    minimum_singular_value: Number = 0,
    constant_singular_value: bool = False,
    entrywise_mean: Union[Literal[False], Number] = False,
    rng: Union[np.random._generator.Generator, Number] = 42,
    **kwargs,
) -> np.ndarray:
    """Generate a random nonnegative matrix from non-negative orthonormal factors with a
    given rank and minimum singular value."""
    rng = get_rng(rng)
    mrows, ncols, rank = parse_mrows_ncols_rank(mrows, ncols, rank)
    minimum_singular_value = parse_number_greater_than(
        minimum_singular_value, 0, "minimum_singular_value", equal_to=True, typ=Number
    )

    # generate m x r non-negative orthonormal basis for rows
    U = random_nonnegative_orthonormal_matrix(mrows, rank, rng)
    # generate r x n non-negative orthonormal basis for columns
    V = random_nonnegative_orthonormal_matrix(ncols, rank, rng)

    if entrywise_mean is False:
        S = minimum_singular_value
    else:
        S = (
            entrywise_mean
            * np.sqrt(np.count_nonzero(U, axis=0))
            * np.sqrt(np.count_nonzero(V, axis=0))
        )  # gets you pretty close to entrywise mean,
        # provided there aren't huge gaps in the nnzs across rows and columns
        if constant_singular_value:
            S = S.mean()
            if S <= minimum_singular_value:
                S = minimum_singular_value
        else:
            S = np.clip(S, minimum_singular_value, None)

    return (U * S) @ V.T


def knn_graph(X,k,Dist_mat):
    # number of samples
    n = X.shape[0]
    if Dist_mat is None:
        D = euclidean_distances(X)
    else:
        D = Dist_mat
    D_cp = D.copy()
    D_cp = D_cp + np.diag(np.repeat(float("inf"), D_cp.shape[0]))
    nn = D_cp.argsort(axis = 1)[:,:k]
    #W = np.zeros((n,n))
    #for i in range(n):
    #    W[i,nn[i,:k]] = 1
    return nn
    

# return the knn statistics
def knn_test_k(X,y,k,Dist_mat=None):
    
    # number of samples
    n = X.shape[0]
    
    # compute the knn mat
    nn = knn_graph(X,k,Dist_mat)
    
    # compute the knn test statistics
    ## check whether the nearest neighbors of each sample are of the same group
    T_s = np.sum(np.array([yi == y[nn[ix,:k]] for ix,yi in enumerate(y)]))/(n*k)
    
    return T_s
    
def compute_affine_coordinates_PCA(X, r, axis=0):
    # compute the orthogonal affine coordinates of X using PCA
    mx = np.expand_dims(np.mean(X, axis),axis)
    Xc = X - mx
    U,S,V = bipca.math.SVD(use_eig=True,backend='torch').fit_transform(Xc)
    M = U if axis==1 else V
    M = M[:,:r]
    
    mx_orthogonal = mx - (M@(M.T@mx.squeeze())).numpy()
    gamma = 1/(np.sqrt(1+np.sum(mx_orthogonal**2)))

    return M.numpy(), mx_orthogonal, gamma

def compute_stiefel_coordinates_from_affine(M,mx_orthogonal,gamma):
    mx_orthogonal = mx_orthogonal.reshape(-1,1)
    return np.block([
        [M , gamma * mx_orthogonal],
        [np.zeros(M.shape[1]), np.array([gamma])]
    ])
def compute_stiefel_coordinates_from_data(X,r, axis=0):
    args = compute_affine_coordinates_PCA(X,r,axis)
    return compute_stiefel_coordinates_from_affine(*args)


def libnorm(X):
    return X/X.sum(axis=1)[:,None]

def new_svd(X,r,which="left",**kwargs):
    if backend not in kwargs:
        kwargs['backend'] = 'torch'
    if use_eig not in kwargs:
        kwargs['use_eig'] = True
    svd_op = bipca.math.SVD(n_components=-1,**kwargs)
    U,S,V = svd_op.fit_transform(X)
    if which == "left":
        return (np.asarray(U)[:,:r])*(np.asarray(S)[:r])
        #org_mat = (op.U_Y[:,:ext_r] * ext_s[:ext_r]) @ op.V_Y.T[:ext_r,:]
    else:
        return np.asarray(U[:,:r]),np.asarray(S[:r]),np.asarray(V.T[:r,:])

def manwhitneyu_de(data,astrocyte_mask, batch_mask):
    test_p_all = mannwhitneyu(data[astrocyte_mask & (batch_mask),:],
        data[astrocyte_mask & (~batch_mask),:],axis=0)[1]

    test_padj_all = test_p_all * len(test_p_all)
    test_padj_all[test_padj_all > 1] = 1

    return test_padj_all
