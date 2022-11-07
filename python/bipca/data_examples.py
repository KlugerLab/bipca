"""This file includes code for generating example data
"""
import numpy as np
import numpy.matlib as matlib
import scipy.stats as stats
import scanpy as sc
from .math import Sinkhorn, find_linearly_dependent_columns
from anndata import AnnData
from numpy.random import default_rng


def get_cluster_sizes(nclusters, ncells,  seed=42,**kwargs):
    """Randomly draw `nclusters` sizes that sum to `ncells`.
    
    Parameters
    ----------
    nclusters : TYPE
        Description
    ncells : TYPE
        Description
    seed : int, optional
        Description
    **kwargs
        Description
    
    Returns
    -------
    TYPE
        Description
    """

    if isinstance(seed, np.random._generator.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)


    p = [1/nclusters] * nclusters
    cluster_sizes = rng.multinomial(ncells, p)

    return cluster_sizes
    
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
    cluster_size = np.floor(ncols/rank).astype(int)

    if simple:
        PX = []
        for r in range(rank):
            PX.append(matlib.repmat(p.T[:,r][:,None],1,cluster_size))
        PX = np.hstack(PX)
        rem = ncols-PX.shape[1]
        if rem > 0:
            PX = np.hstack((matlib.repmat(p.T[:,0][:,None],1,rem),PX))
    else:
        #draw random loadings
        PX = []
        for r in range(rank):
            loading = np.random.multinomial(rank, [1/rank]*rank, cluster_size) / rank
            #the ground truth probability matrix
            PXr = (loading @ p).T
            PX.append(PXr)
        PX = np.hstack(PX)
        rem = ncols-PX.shape[1]
        if rem > 0:
            PX = np.hstack((matlib.repmat(p.T[:,0][:,None],1,rem),PX))
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
def generate_poisson_clustered_data(mrows, ncols, k_clusters, rank, subspace_SNR=1, sampling_SNR = 1, seed=42, **kwargs):
    """generate_poisson_clustered_data: Generate an mrows x ncols matrix of \
        Poisson sampled data with the following properties:
    1. It forms `k_clusters` number of clusters
    2. The total matrix is of rank `rank`
    3. Given b=1, c=0, the noise variance of the matrix is approximately `noise`
    
    To generate this data,
    1) a basis for the ambient data is created by 
        drawing`rank` linear independent vectors of length 
        `mrows` from an element-wise exponential distribution.
    2) each of 0,1,...`k_clusters-1` clusters are assigned to a 
        "signal subspace" of vectors from the basis. These vectors will be the 
        signal span of each cluster. First, the dimension of each subspace is 
        determined by drawing values from a multinomial with 
        parameters `n=rank` and uniform probability. Then, the dimensions are 
        assigned sequentially to each cluster based on their size. For instance,
        if cluster i is of signal dimension d_i>1, then cluster 0 is assigned to
        the first 0,...,d_0-1 basis vectors, cluster 1 is assigned to
        d0,...,d_1-1, etc.
    3) columns of the data are assigned to each cluster using the same 
        process as basis vector assignment. The size of cluster i is s_i.
    4) the loadings of each cell in each cluster onto the global basis
        are generated randomly.
        4.a) a set of loadings is drawn for the signal subspace of each cluster.
            Each column in cluster d_i is assigned loadings by drawing d_i 
            values from a multinomial with n = Poisson(100) and uniform
            probability. This generates loadings for each cluster that 
            are non-uniform within a cluster. 
            The signal loadings are then l2 normalized.
        4.b) a set of loadings for the noise subspace of each column, i.e., 
            the remaining basis vectors that were assigned to the other clusters,
            are drawn in a similar manner and l2 normalized. Thus, at this stage 
            each column is composed of a set of signal loadings over the cluster 
            subspace and a set of noise loadings over the remaining basis vectors. 
            The SNR at this point is 1.
        4.c), the signal loadings from 4.a are scaled by subspace_SNR to 
            create a target SNR.
        4.d) The signal and noise loadings are combined into a single matrix of
            size (s_i, rank)
    5) the underlying matrix of Poisson parameters for each cluster is generated
        by multiplying the basis vectors by the loadings to produce a matrix
        of size (mrows, s_i). The mean value of this matrix is normalized to 1
        and susequently multiplied by `sampling_SNR**2` to create a cluster 
        of means with a desired sampling SNR.
    6) The latent cluster matrices are concatenated into `X`
    7) The data is drawn from a Poisson with entry-wise parameters from `X`,
        i.e., `Y[i,j]=Poisson(X[i,j])`


    Parameters
    ----------
    mrows : int
        The number of rows ("genes") to simulate
    ncols : int
        The number of columns ("cells") to simulate
    k_clusters : int
        The number of clusters ("cell types")
    rank : int
        The rank of the matrix. Must be smaller than `nrows`.
    subspace_SNR : Number, default 1
        Signal to noise ratio (SNR) of signal basis vectors to non-signal basis 
        for each cluster. 
    sampling_SNR : Number, default 1
        SNR of the Poisson sampling. Small values lead to sparsity.
    seed : int or `np.random._generator.Generator`, default 42
        A random number generator or a seed to np.random.default_rng
    **kwargs
        Arguments to pass to `bipca.data_examples.get_cluster_sizes`
    
    Returns
    -------
    Y : (mrows, ncols) array
        The sampled simulation data
    X : (mrows, ncols) array
        The Poisson parameters used to sample Y
    cluster_indicator : (ncols,) array
        The indicator vector for the clusters
    """
    #instantiate random generator
    if isinstance(seed, np.random._generator.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)    
        
        
    #generate the cluster sizes in terms of their inner dimensions and their number of columns
    cluster_sizes = get_cluster_sizes(k_clusters, ncols, seed=rng, **kwargs)
    cluster_dimensions = get_cluster_sizes(k_clusters, rank, seed=rng, **kwargs)
    
    #transform cluster sizes to assignments
    cluster_assignments = np.cumsum(cluster_sizes)
    cluster_assignments = np.r_[0,cluster_assignments]
    cluster_assignments = [(cluster_assignments[k],cluster_assignments[k+1]) for k in range(k_clusters)]
    #transform cluster dimensions into assignments
    dimension_assignments = np.cumsum(cluster_dimensions)
    dimension_assignments = np.r_[0,dimension_assignments]
    dimension_assignments = [(dimension_assignments[k],dimension_assignments[k+1]) for k in range(k_clusters)]
    
    #build basis vectors; 
    #these are likely linearly independent to begin with, but
    #we check to make sure that they actually are
    basis_vecs = [np.exp(2*rng.standard_normal(size=(mrows,1)))]
    while len(basis_vecs) < rank:
        new_vec = np.exp(2*rng.standard_normal(size=(mrows,1)))
        test_vecs = basis_vecs.copy()
        test_vecs.append(new_vec)
        if len(find_linearly_dependent_columns(np.hstack(test_vecs))) == 0:
            basis_vecs.append(new_vec)
        else:
            pass
    basis_vecs = np.hstack(basis_vecs)
    basis_vecs /= np.linalg.norm(basis_vecs,ord=2, axis=0) #normalize the basis vectors
    
    cluster_data = []
    cluster_indicator = []
    
    for kix,(basis_indices, basis_size, cluster_size) in enumerate(zip(dimension_assignments, 
                                                      cluster_dimensions,
                                                      cluster_sizes)):
        
        cluster_loadings = np.zeros((cluster_size,rank))

        #draw the loadings
        signal_subspace_loadings = rng.multinomial(
            [rng.poisson(100)]*cluster_size,
            [1/basis_size]*basis_size).astype(np.float64)
        signal_subspace_loadings /= np.linalg.norm(signal_subspace_loadings,
            axis=1)[:,None] # the l2 norm of the cluster loadings is now 1
        signal_subspace_loadings *= subspace_SNR # make the L2 norm  = subspac_SNR
        
        noise_subspace_loadings = rng.multinomial(
            [rng.poisson(100)]*cluster_size,
            [1/(rank-basis_size)]*(rank-basis_size)).astype(np.float64)
        noise_subspace_loadings /= np.linalg.norm(noise_subspace_loadings,
            axis=1)[:,None] # the l2 norm of the extra cluster loadings is now 1

        cluster_loadings[:,basis_indices[0]:basis_indices[1]] = signal_subspace_loadings
        noise_mask = np.logical_not(np.isin(
            np.arange(rank),np.arange(basis_indices[0],basis_indices[1]),))
        cluster_loadings[:,noise_mask] = noise_subspace_loadings
        Xi = basis_vecs@cluster_loadings.T
        Xi /= Xi.mean()  # the average entry of the matrix is 1, so the SNR is 1
        Xi *= sampling_SNR**2 # scale to the desired sampling SNR
        cluster_data.append(Xi)
        cluster_indicator.append(kix*np.ones(cluster_size))

    cluster_indicator = np.hstack(cluster_indicator).astype(int)
    X = np.hstack(cluster_data)
    Y = rng.poisson(X)
    
    
    return Y, X, cluster_indicator

def clustered_poisson(nrows=500,ncols=1000, nclusters = 5, 
                        cluster_rank = 4, rank = 20, prct_noise = 0.5,
                        row_samplefun = lambda nrows: stats.loguniform.rvs(0.1,1000, size=nrows),
                        col_samplefun = lambda ncols: stats.loguniform.rvs(0.1,1000, size=ncols),
                        norm_mean = 0,
                        seed=42, **kwargs):
    """Generate a set of Poisson sampled data with the following properties:
    1. It forms `nclusters` number of clusters
    2. Each cluster is of rank `cluster_rank`
    3. The total matrix is of rank `rank`
    4. `prct_noise` proportion of the rows of the data are sampled from a constant poisson
    5. The Poisson parameters are based on a biscaled matrix that is then multiplied by sampled row and column factors.
        
    
    Parameters
    ----------
    nrows : int, default 500
        The number of rows ("genes") to simulate
    ncols : int, default 1000
        The number of columns ("cells") to simulate
    nclusters : int, default 5
        The number of clusters ("cell types")
    cluster_rank : int, default 4
        The rank of the subspace each cluster lies in
    rank : int, default 20
        The rank of the matrix. Must be greater than `cluster_rank` and smaller than `nrows` * `prct_noise`.
    prct_noise : float, default 0.5
        The proportion of constant noise dimensions to include in the output.
    row_samplefun : callable, default scipy.stats.loguniform.rvs
        Function to generate row scale factors. Must accept `nrows` as sole argument
    col_samplefun : callable, default scipy.stats.loguniform.rvs
        Function to generate column scale factors. Must accept `ncols` as sole argument.
    seed : float, default 42
        Random seed.
    **kwargs
        Arguments to pass to `bipca.data_examples.get_cluster_sizes`
    
    Returns
    -------
    X : (nrows, ncols) array
        The sampled simulation data
    lambdas : (nrows, ncols) array
        The Poisson parameters used to sample X
    PX2 : (nrows, ncols) array
        The underlying biscaled, column stochastic matrix of cell probabilities
    PX : (nrows, ncols) array
        The underlying matrix of cell parameters before biscaling and column normalization
    row_factors : (nrows,) array
        The ground truth row factors
    col_factors : (ncols,) array
        The ground truth column factors
    cluster_indicator : (ncols,) array
        The indicator vector for the clusters
    """
    cluster_sizes = get_cluster_sizes(nclusters, ncols, seed=seed,**kwargs)
    rng = np.random.default_rng(seed)
    S = np.exp(2*rng.standard_normal(size=(np.floor((1-prct_noise)*nrows).astype(int),rank))) #draw `rank` basis vectors in `nrows` dimensions
    cluster_loadings = rng.uniform(size=(cluster_rank,rank,nclusters)) #draw `nclusters` matrices of size `rank x rank`. 
    # these describe the subspace the clusters lie in by the following..
    cluster_vectors = S@cluster_loadings # This is `rank` x `nrows` x `nclusters` : the span of each cluster


    PX = [] #store the actual cells in here before stacking them together
    cluster_indicator = []
    for cix in range(nclusters):
        cell_loading = rng.uniform(size=(cluster_rank,cluster_sizes[cix])) #make loadings for each cell in the cluster
        PX.append((cell_loading.T@cluster_vectors[:,:,cix]).T) #get the actual coordinates for each cell
        cluster_indicator.append(cix*np.ones((cluster_sizes[cix])))
    PX = np.hstack(PX) #concatenate all of the clusters together to make the final matrix
    cluster_indicator = np.hstack(cluster_indicator)
    noise_dims = np.ones(((nrows-PX.shape[0]),PX.shape[1]))
    PX = np.vstack((PX,noise_dims)) #The final parameter matrix before biscaling

    sinkhorn_operator = Sinkhorn(variance_estimator=None)
    PX2 = sinkhorn_operator.fit_transform(PX)
    PX2 = PX2/np.sum(PX2,axis=0) #make the matrix column stochastic: the cells form probability distributions over the genes
    row_factors = row_samplefun(nrows)
    col_factors = col_samplefun(ncols)
    lambdas = PX2 * row_factors[:,None] * col_factors[None,:] #the matrix of Poisson parameters
    if norm_mean>0:
        lambdas = lambdas * norm_mean/np.mean(lambdas)

    X = rng.poisson(lam=lambdas) #the sampled data matrix

    return X, lambdas, PX2, PX, row_factors, col_factors,cluster_indicator

class ScanpyPipeline(object):

    """Load an .h5ad raw dataset into ScanPy and run the standard transformations to it.
    
    Attributes
    ----------
    adata : TYPE
        Description
    adata_filtered : TYPE
        Description
    adata_raw : TYPE
        Description
    basename : TYPE
        Description
    cells_kept : TYPE
        Description
    fname : TYPE
        Description
    genes_kept : TYPE
        Description
    highly_variable : TYPE
        Description
    results_file : TYPE
        Description
    """
    
    def __init__(self, fname, readfun = sc.read_h5ad, adata=None):
        """Summary
        
        Parameters
        ----------
        fname : TYPE
            Description
        readfun : TYPE, optional
            Description
        adata : None, optional
            Description
        """
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
        """Summary
        
        Parameters
        ----------
        min_genes : int, optional
            Description
        min_cells : int, optional
            Description
        max_n_genes_by_counts : int, optional
            Description
        max_n_cells_by_counts : int, optional
            Description
        mt_pct_counts : int, optional
            Description
        target_sum : float, optional
            Description
        log_normalize : bool, optional
            Description
        write : bool, optional
            Description
        reset : bool, optional
            Description
        """
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
        """Summary
        
        Parameters
        ----------
        adata : None, optional
            Description
        fname : None, optional
            Description
        """
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
