from collections.abc import Iterable
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import numpy as np
from scipy import stats
from .math import mp_pdf,mp_quantile,emp_pdf_loss, L2, L1
from matplotlib.offsetbox import AnchoredText

def MP_histogram(svs,gamma, cutoff = None,  theoretical_median = None,  
    loss_fun = [L1, L2], evaluate_on_bin = True, where='center', ax = None, bins=100, histkwargs = {}):
    """
    Histogram of covariance eigenvalues compared to the theoretical Marcenko-Pastur law.

    Compute a density-normalized histogram of the covariance eigenvalues in `svs` 
    and plot the histogram alongside the theoretical Marcenko-Pastur law.
    If multiple sets of eigenvalues are provided (by passing a list of arrays to `svs`),
    an average over many histograms is performed.

    Parameters
    ----------
    svs : array or list of arrays
        Covariance eigenvalues. If a list is provided, 
        the output histograms will be the average of individual histograms.
    gamma : float
        Aspect ratio of the corresponding wide data matrix (gamma <= 1).
    cutoff : float, optional
        The Marcenko-Pastur rank cutoff. Defaults to (1+np.sqrt(gamma))**2
    theoretical_median : float, optional
        Theoretical median of the Marcenko-Pastur distribution. By default this is computed from the input.
    loss_fun : list of callable or False, optional
        Default L2. Compute and print loss according to `bipca.math.loss_fun`
    evaluate_on_bin : bool, optional
        Default True. Evaluate the theoretical Marcenko-Pastur distribution on the bins computed for the histogram, rather than a tiling.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Matplotlib axis object to plot the histograms in. Defaults to new axis.
    Other Parameters
    ----------------
    histkwargs : dict, optional
        Keyword arguments to np.histogram.
    """
    
    if cutoff is None:
        cutoff = (1+np.sqrt(gamma))**2
    if ax is None:
        ax = plt.axes()
    if not isinstance(svs,list):
        sv = svs
    else:
        sv = svs[0]
    if theoretical_median is None:
        theoretical_median = mp_quantile(gamma, mp = lambda x,gamma: mp_pdf(x,gamma))

    n, bins = np.histogram(sv[sv<=cutoff*2], bins=bins, range = [0, cutoff*2],density = True,*histkwargs)
    actual_median = np.median(sv)
    if isinstance(svs,list):
        for sv in svs[1:]:
            nn, _ = np.histogram(sv[sv<=cutoff*2],bins=bins,density = True)
            actual_median += np.median(sv)

            n+=nn
        n = n / len(svs)
        actual_median = actual_median /len(svs)
    w = bins[:-1]-bins[1:]
    ax.hist(bins[:-1], bins, weights=n)
    est_dist = stats.rv_histogram([n, bins])

    if evaluate_on_bin:
        if where =='center':
            xx = (bins[1:]+bins[:-1])/2
        else:
            xx = bins
        #add points outside the bins
        xx= np.hstack((bins[0]*0.9, xx, bins[-1]*1.1))
    else:
        xx=np.linspace(bins[0]*0.5, bins[-1]*1.1, 1000)
    ax.plot(xx,mp_pdf(xx,gamma), 'g-', markersize = 1)
    ax.axvline(theoretical_median, c='r')
    ax.axvline(actual_median, c='y')
    if loss_fun:
        if isinstance(loss_fun,list):
            est_loss = [emp_pdf_loss(lambda x: mp_pdf(x,gamma),est_dist.pdf, loss = loss) for loss in loss_fun]
        else:
            est_loss = [emp_pdf_loss(lambda x: mp_pdf(x,gamma),est_dist.pdf,loss=loss_fun)]
            loss_fun = [loss_fun]
        loss_str = 'Error:'
        for val, fun in zip(est_loss,loss_fun):
            loss_str += '\n' 
            loss_str += str(fun.__name__) + ': {:.3f}'.format(val)
        anchored_text = AnchoredText(loss_str, loc='upper right',frameon=False)
        ax.add_artist(anchored_text)

    return ax

def MP_histograms_from_bipca(bipcaobj, bins = 100, avg= False, ix=0, 
    fig = None, axes = None, figsize = (15,5), dpi=300, title='',output = '', histkwargs = {}, **kwargs):
    """
    Spectral density before and after bipca biscaling and noise variance normalization from a single BiPCA object.
    
    Plot the spectral density of 
    1) the unscaled, non-normalized 
    2) scaled, non-normalized, and 
    3) scaled, noise-variance normalized
    covariance matrices and the corresponding Marcenko-Pastur law learned by a `BiPCA` object. 

    Parameters
    ----------
    bipcaobj : bipca.bipca.BiPCA
        A fit BiPCA estimator that contains `data_covariance_eigenvalues` and `biscaled_normalized_covariance_eigenvalues` attributes.
        These attributes may be set by bipcaobj.get_histogram_data().
    avg : bool, optional 
        If multiple sets of pre and post eigenvalues exist (such as obtained by shuffled resampling), 
        plot the average of histograms over these spectra. (default False)
    ix : int, optional
        The index of the set of eigenvalues to consider from `data_covariance_eigenvalues` and `biscaled_normalized_covariance_eigenvalues`. 
        Only relevant if `avg` is False and `data_covariance_eigenvalues` is a list.


    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Matplotlib axis containing histogram.
    
    Other Parameters
    ----------------
    fig : matplotlib.figure.Figure, optional
        Figure object to plot the 3 new histograms in. 
    axes : list of 3 or more matplotlib.axes._subplots.AxesSubplot, optional
    figsize : tuple, optional
        Figure size in inches 
    """

    if fig is None:
        if axes is None: # neither fig nor axes was supplied.
            fig,axes = plt.subplots(1,3,dpi=dpi,figsize=figsize)
        fig = axes[0].figure
    if axes is None:
        axes = add_rows_to_figure(fig,ncols=3)
    if len(axes) != 3:
        raise ValueError("Number of axes must be 3")
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]   

    sigma2 = bipcaobj.shrinker.sigma_**2
    if isinstance(bipcaobj.data_covariance_eigenvalues,list): #this needs to be cleaned
        if not avg:
            presvs = bipcaobj.data_covariance_eigenvalues[ix]
            postsvs = bipcaobj.biscaled_normalized_covariance_eigenvalues[ix]
            postsvs_noisy = postsvs* sigma2
        else:
            presvs = bipcaobj.data_covariance_eigenvalues
            postsvs = bipcaobj.biscaled_normalized_covariance_eigenvalues
            postsvs_noisy = [ele * sigma2 for ele in postsvs]
    else:
        presvs = bipcaobj.data_covariance_eigenvalues
        postsvs=bipcaobj.biscaled_normalized_covariance_eigenvalues
        postsvs_noisy =  postsvs* sigma2
    gamma = bipcaobj.approximating_gamma
    theoretical_median = mp_quantile(gamma, mp = lambda x,gamma: mp_pdf(x,gamma))
    cutoff = bipcaobj.shrinker.scaled_cutoff_

    ax1 = MP_histogram(presvs, gamma, cutoff, theoretical_median, bins=bins,ax=ax1,histkwargs=histkwargs,**kwargs)
    ax1.set_title('Unscaled covariance \n' r'$\frac{1}{N}XX^T$')
    ax1.grid(True)

    ax2 = MP_histogram(postsvs_noisy, gamma, cutoff, theoretical_median, bins=bins, ax= ax2,histkwargs=histkwargs,**kwargs)
    ax2.set_title('Biscaled covariance \n' r'$\frac{1}{N}YY^T$')
    ax2.grid(True)

    ax3 = MP_histogram(postsvs, gamma, cutoff, theoretical_median, bins=bins, ax=ax3, histkwargs=histkwargs,**kwargs)
    ax3.set_title('Biscaled, noise corrected covariance \n' r'$\frac{1}{N\sigma^{2}}YY^T$' + '\n' + r'$\sigma^2 = {:.2f} $'.format(sigma2))
    ax3.grid(True)
    fig.tight_layout()
    ax2.legend(["Marcenko-Pastur PDF","Theoretical Median", "Actual Median"],bbox_transform=ax2.transAxes,loc='center',bbox_to_anchor=(0.5,-0.2),ncol=3)
    ax2.text(0.5,1.25,title,fontsize=16,ha='center',transform=ax2.transAxes)
    #fig.tight_layout()
    if output != '':
        plt.savefig(output, bbox_inches="tight")
    return (ax1,ax2,ax3,fig)

def spectra_from_bipca(bipcaobj, semilogy = True, zoom = True, zoomfactor = 10, ix = 0, ax = None, dpi=300,figsize = (15,5), title = '', output = ''):
    #this function does not plot from averages.
    fig, axes = plt.subplots(1,3,dpi=dpi,figsize = figsize)

    scaled_cutoff = bipcaobj.shrinker.scaled_cutoff_**2
    sigma2 = bipcaobj.shrinker.sigma_**2

    if isinstance(bipcaobj.data_covariance_eigenvalues,list):
        presvs = bipcaobj.data_covariance_eigenvalues[ix]
        postsvs = bipcaobj.biscaled_normalized_covariance_eigenvalues[ix]
    else:       
        presvs = bipcaobj.data_covariance_eigenvalues
        postsvs=bipcaobj.biscaled_normalized_covariance_eigenvalues
    presvs = -np.sort(-np.round(presvs, 4))
    postsvs = -np.sort(-np.round(postsvs,4))
    postsvs_noisy =  -np.sort(-postsvs * sigma2)
    svs = [presvs, postsvs_noisy,postsvs]

    pre_rank = (presvs>=scaled_cutoff).sum()
    biscaled_noisy_rank = (postsvs_noisy>=scaled_cutoff).sum()
    postrank = (postsvs>=scaled_cutoff).sum()
    ranks = np.array([pre_rank, biscaled_noisy_rank,postrank],dtype=int)

    if zoom:
        if isinstance(zoom, int):
            #user supplied max-SVs to plot
            high =  zoom 
            low = 1
        elif isinstance(zoom,tuple): #user supplied upper and lower range of SVs
            low = zoom[0]
            high = zoom[1]
        else:
            # compute the number of eigenvalues by selecting the range within a factor of the MP cutoff.
            low = np.zeros((3,))
            high = np.zeros((3,))

            lower_cutoff = scaled_cutoff/zoomfactor
            upper_cutoff = scaled_cutoff*zoomfactor
            for ix,sv in enumerate(svs):
                #get the indices that lie within the range
                valid_pts = np.argwhere(sv>=lower_cutoff and sv<=upper_cutoff)
                #record them with a 1-offset, rather than the python indices. 
                low[ix] = np.min(valid_pts)+1
                high[ix] = np.max(valid_pts)+1
            # dims = np.array([len(ele) for ele in [presvs,postsvs,postsvs_noisy]])
            # minimum_dim = np.min(dims)
            # maximum_rank = np.max(ranks)
            # num_to_plot = int(np.min([minimum_dim, 1.2*maximum_rank]))

    else:
        #no x-zoom, plot the whole spectrum
        high = len(postsvs)
        low = 1

    if isinstance(high,int): 
        x = [np.arange(low, high+1) for _ in range(3)]#+1 because of exclusive arange
    else:
        x = [np.arange(lo, hi+1) for lo,hi in zip(low,high)]
    #now truncate the svs appropriately - remembering that our xs are 1-indexed
    svs = [ele[xx-1] for xx,ele in zip(x,svs)]

    if semilogy:
        plotfun = lambda ax, x, svs: ax.semilogy(x,svs)
    else:
        plotfun = lambda ax, x, svs: ax.plot(x,svs)
        #needs some code for truncation or axis splitting

    for ix,ax in enumerate(axes):
        #the plotting loop
        plotfun(ax,x[ix],svs[ix])
        ax.fill_between(x[ix],0,svs[ix])
        ax.axvline(x=ranks[ix],c='xkcd:light orange',linestyle='--',linewidth=1)
        ax.axhline(y=scaled_cutoff,c='xkcd:light red',linestyle='--',linewidth=1)
        ax.grid(True)
        ax.legend([r'$\frac{\lambda_X(k)^2}{N}$','selected rank = '+str(ranks[ix]),r'MP threshold $(1 + \sqrt{\gamma})^2$'])
        ax.set_xlabel('Eigenvalue index k')
        ax.set_ylabel('Eigenvalue')
    axes[0].set_title('Unscaled covariance \n' r'$\frac{1}{N}XX^T$')
    axes[1].set_title('Biscaled covariance \n' r'$\frac{1}{N}YY^T$')
    axes[2].set_title('Biscaled, noise corrected covariance \n' r'$\frac{1}{N\sigma^{2}}YY^T$' + '\n' + r'$\sigma^2 = {:.2f} $'.format(sigma2))


    fig.suptitle(title)
    fig.tight_layout()
    if output !='':
        plt.savefig(output, bbox_inches="tight")

    return (axes[0],axes[1],axes[2],fig)


def add_rows_to_figure(fig, ncols = None, nrows = 1):
    """
    Add rows to a figure.  
    Returns ncols*nrows axes, which defaults to 
    the number of columns in the current figure, i.e. a single row with no spanning elements.
    

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add rows to
    ncols : int
        Number of columns in the row, defaults to the current columns in the figure.
    nrows : int
        Number of rows to add

    Returns
    -------
    list(matplotlib.axes._subplots.AxesSubplot)
        The new ncols*nrows axes
    
    """
    new_axes = []
    if len(fig.axes)==0:
        if ncols is None:
            ncols = 1
        new_gs = fig.add_gridspec(nrows=nrows,ncols=ncols)
        for j in range(-1*nrows,0):
            for i in range(ncols):
                newpos = new_gs[j, i]
                new_axes.append(fig.add_subplot(newpos))
    else:
        old_gs = fig.axes[0].get_gridspec() # get the old gridspec
        curnrows = old_gs.nrows
        curncols = old_gs.ncols
        if ncols is None:
            ncols = curncols
        div = ncols
        if curncols % div != 0: #we need to resize the grid
            tgtncols = curncols*div
        else:
            tgtncols = curncols
        histstride = tgtncols//div
        ogstride = tgtncols//curncols
        new_gs = gridspec.GridSpec(nrows=curnrows+nrows, ncols=tgtncols)
        for ax in fig.axes:
            currentposition = ax.get_subplotspec()
            r0 = currentposition.rowspan
            c0 = currentposition.colspan
            newposition = new_gs[ r0.start:r0.stop,c0.start*ogstride:c0.stop*ogstride]
            ax.set_position(newposition.get_position(fig))
            ax.set_subplotspec(newposition)
        for j in range(-1*nrows,0):
            for i in range(ncols):
                newpos = new_gs[j, i*histstride:(i+1)*histstride]
                new_axes.append(fig.add_subplot(newpos))
    return new_axes



# def plot_sparsity_statistics
# fig,ax = plt.subplots(1,2,figsize=(18,4))

# currentix = 1
# current_dset = datasets[currentix][1]
# M,N = current_dset.X.shape
# obs_n_genes = current_dset.obs['n_genes_by_counts']
# var_n_cells = current_dset.var['n_cells_by_counts']

# ##get the percentiles
# bottom_percentile_ncells = var_n_cells<=np.percentile(var_n_cells,25)
# bottom_percentile_ngenes = obs_n_genes<=np.percentile(obs_n_genes,25)
# n,bins,patches =ax[0].hist(var_n_cells,bins=100,density=False)
# ax[0].axvline(np.mean(var_n_cells),color='r')
# ax[0].axvline(np.median(var_n_cells),color='orange')

# ax[0].legend(['mean = {:.0f}'.format(np.mean(var_n_cells)),'median = {:.0f}'.format(np.median(var_n_cells))],loc=4)

# ax[0].set_title('Nonzeros per row (gene)')
# n20,bins20,_=ax[1].hist(obs_n_genes,bins=100,density=False)
# ax[1].axvline(np.mean(obs_n_genes),color='r')
# ax[1].axvline(np.median(obs_n_genes),color='orange')

# ax[1].legend(['mean = {:.0f}'.format(np.mean(obs_n_genes)),'median = {:.0f}'.format(np.median(obs_n_genes))],loc=4)

# axin1.set_title(r'$\leq$ 25th percentile')

# ax[1].set_title("Nonzeros per column (cell)")
# fig.suptitle("Distribution of zeros in unfiltered {:.0f} x {:.0f} 10X_".format(N,M)+str(datasets[currentix][0]))