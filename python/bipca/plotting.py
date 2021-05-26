from collections.abc import Iterable
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import numpy as np
from scipy import stats
from .math import mp_pdf,mp_quantile,emp_pdf_loss

def MP_histogram(svs,gamma, cutoff = None,  theoretical_median = None, ax = None, bins=100, histkwargs = {}):
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
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Matplotlib axis object to plot the histograms in. Defaults to new axis.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Matplotlib axis containing histogram.

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
    ax.bar(bins[:-1],n,width = w, align='edge')
    est_dist = stats.rv_histogram([n, bins])
    est_loss = emp_pdf_loss(lambda x: mp_pdf(x,gamma),est_dist.pdf)
    xx=np.linspace(0.000001, bins[-1]*1.1, 10000)
    ax.plot(xx,mp_pdf(xx,gamma), 'go-', markersize = 2)
    ax.plot(theoretical_median*np.ones(50),np.linspace(0,max(n),50),'rx--',alpha=1,markersize=2)
    ax.plot(actual_median*np.ones(50),np.linspace(0,max(n),50),'ys--',alpha=0.5,markersize=2)

    return ax

def MP_histograms_from_bipca(bipcaobj, bins = 100, avg= False, ix=0, fig = None, axes = None, figsize = (15,5), dpi=300, title='',output = '', histkwargs = {}):
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

    ax1 = MP_histogram(presvs, gamma, cutoff, theoretical_median, bins=bins,ax=ax1)
    ax1.set_title('Unscaled covariance \n' r'$\frac{1}{N}XX^T$')
    ax1.grid(True)

    ax2 = MP_histogram(postsvs_noisy, gamma, cutoff, theoretical_median, bins=bins, ax= ax2)
    ax2.set_title('Biscaled covariance \n' r'$\frac{1}{N}YY^T$')
    ax2.grid(True)

    ax3 = MP_histogram(postsvs, gamma, cutoff, theoretical_median, bins=bins, ax=ax3)
    ax3.set_title('Biscaled, noise corrected covariance \n' r'$\frac{1}{N\sigma^{2}}YY^T$' + '\n' + r'$\sigma^2 = {:.2f} $'.format(sigma2))
    ax3.grid(True)
    fig.tight_layout()
    ax2.legend(["Marcenko-Pastur PDF","Theoretical Median", "Actual Median"],bbox_transform=ax2.transAxes,loc='center',bbox_to_anchor=(0.5,-0.2),ncol=3)
    ax2.text(0.5,1.25,title,fontsize=16,ha='center',transform=ax2.transAxes)
    #fig.tight_layout()
    if output != '':
        plt.savefig(output, bbox_inches="tight")
    return (ax1,ax2,ax3,fig)

def spectra_from_bipca(bipcaobj, ix = 0, ax = None, dpi=300,figsize = (15,5),title = '', output = ''):
    #this function does not plot from averages.
    fig, (ax0,ax1,ax2) = plt.subplots(1,3,dpi=dpi,figsize = figsize)

    scaled_cutoff = bipcaobj.shrinker.scaled_cutoff_**2
    sigma2 = bipcaobj.shrinker.sigma_**2

    if isinstance(bipcaobj.data_covariance_eigenvalues,list):
        presvs = bipcaobj.data_covariance_eigenvalues[ix]
        postsvs = bipcaobj.biscaled_normalized_covariance_eigenvalues[ix]
    else:       
        presvs = bipcaobj.data_covariance_eigenvalues
        postsvs=bipcaobj.biscaled_normalized_covariance_eigenvalues
    presvs = np.round(presvs, 4)
    postsvs = np.round(postsvs,4)
    postsvs_noisy =  postsvs * sigma2

    pre_rank = (presvs>=scaled_cutoff).sum()
    x = np.arange(1,len(postsvs)+1)
    biscaled_noisy_rank = (postsvs_noisy>=scaled_cutoff).sum()
    postrank = (postsvs>=scaled_cutoff).sum()

    ax0.semilogy(x,presvs)
    ax0.fill_between(x,0, presvs)
    ax0.axvline(x=pre_rank,c='xkcd:light orange',linestyle='--',linewidth=1)
    ax0.axhline(y=scaled_cutoff,c='xkcd:light red',linestyle='--',linewidth=1)
    ax0.grid(True)
    ax0.legend([r'$\frac{\lambda_X(k)^2}{N}$','selected rank = '+str(pre_rank),r'MP threshold $(1 + \sqrt{\gamma})^2$'])
    ax0.set_xlabel('Eigenvalue index k')
    ax0.set_ylabel('Eigenvalue')
    ax0.set_title('Unscaled covariance \n' r'$\frac{1}{N}XX^T$')
    ax1.semilogy(x,postsvs_noisy)
    ax1.fill_between(x,0, postsvs_noisy)
    ax1.axvline(x=biscaled_noisy_rank,c='xkcd:light orange',linestyle='--',linewidth=1)
    ax1.axhline(y=scaled_cutoff,c='xkcd:light red',linestyle='--',linewidth=1)
    ax1.grid(True)
    ax1.legend([r'$\frac{\lambda_Y(k)^2}{N}$','selected rank = '+str(biscaled_noisy_rank),r'MP threshold $(1 + \sqrt{\gamma})^2$'])
    ax1.set_xlabel('Eigenvalue index k')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Biscaled covariance \n' r'$\frac{1}{N}YY^T$')


    ax2.semilogy(x,postsvs)
    ax2.fill_between(x,0, postsvs)
    ax2.axvline(x=postrank,c='xkcd:light orange',linestyle='--',linewidth=1)
    ax2.axhline(y=scaled_cutoff,c='xkcd:light red',linestyle='--',linewidth=1)
    ax2.grid(True)
    ax2.legend([r'$\frac{\lambda_Y(k)^2}{N\sigma^2}$','selected rank = '+str(postrank),r'MP threshold $(1 + \sqrt{\gamma})^2$'])
    ax2.set_xlabel('Eigenvalue index k')
    ax2.set_ylabel('Eigenvalue')
    ax2.set_title('Biscaled, noise corrected covariance \n' r'$\frac{1}{N\sigma^{2}}YY^T$' + '\n' + r'$\sigma^2 = {:.2f} $'.format(sigma2))


    fig.suptitle(title)
    fig.tight_layout()
    if output !='':
        plt.savefig(output, bbox_inches="tight")

    return (ax0,ax1,ax2,fig)


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



# def parse_figure_axes_inputs(nax, nrows, ncols, axes_geom = None, fig = None, axes = None, figparams = {}):
#   """
#   Parse various pre-supplied figure-axis combinations, return as a single figure
    

#     Parameters
#     ----------
#     nax : int
#       Number of requested axes
#   nrows : int
#       Number of rows in total geometry
#   ncols : int
#       Number of columns in total geometry
#   axes_geom : list of tuples, optional
#       Row and column span of each requested axis

#     Returns
#     -------
   
    
#     Other Parameters
#     ----------------
#     """
#     if nax > nrows*ncols:
#       raise ValueError("Requested number of axes is greater than the amount allowed by supplied (nrows, ncols)")



#     # now we check that we have pre-built axes
#   if isinstance(axes, np.ndarray):
#       axes = axes.flatten()
#   if isinstance(axes,Iterable):
#       axes = [ax for ax in axes if isinstance(ax,mpl.axes.Axes)]
#       if len(axes) == 0:
#           axes = None
#   elif isinstance(axes, mpl.axes.Axes):
#       axes = [axes]       

#   #how many axes do we need to make?
#   if axes is None:
#       naxes_to_make = nax
#   else:
#       naxes_to_make = nax - len(axes)

#   if isinstance(axes_geom, list): # now check the supplied geometries to see if we have the right amount
#       if len(axes_geom) not in (naxes_to_make, nax): 
#           invalid = True
#       if any([ele[0]>nrows or ele[1] > ncols for ele in axes_geom]): # if any of them are too large
#           invalid = True
#       if invalid:
#           raise ValueError("Supplied geometry is invalid")
#   elif axes_geom is None or isinstance(axes_geom,tuple):
#       axes_geom = [axes_geom]

#   #did we get a figure?
#     isfig = isinstance(fig, mpl.figure.Figure)
#     #if we didn't get a figure, then we will either add one or use the figure associated with the supplied axes.
#   if not isfig:
#       if axes is None:
#           fig = plt.figure(*figparams)
#       else:
#           fig = axes[0].figure
#   if naxes_to_make == 0:
#       return fig, axes

#   #now we have a figure.
#   #if it has subplots, then we need to add to them naxes_to_make 
#   if len(fig.axes) == 0 : #the figure has no subplots
#       gs = fig.add_gridspec(1, 3)
#       axes = gs.subplots()
#       else: #the figure already has subplots
#           gs = fig.axes[0].get_gridspec()
#           if axes is True: #in this case we append new axes
#               #we need to figure out 


