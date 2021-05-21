
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from .math import mp_pdf,mp_quantile,emp_pdf_loss

def MP_histogram(svs,gamma,cutoff,  ax = None, histkwargs = {}):
	if ax is None:
		ax = plt.axes()
	if not isinstance(svs,list):
		sv = svs
	else:
		sv = svs[0]
	theoretical_median = mp_quantile(gamma, mp = lambda x,gamma: mp_pdf(x,gamma))
	n, bins = np.histogram(sv[sv<=cutoff*2], bins=100, range = [0, cutoff*2],density = True,*histkwargs)
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

def MP_histograms_from_bipca(bipcaobj, avg= False, ix=0, ax = None, figsize = (15,5), title='',output = '', histkwargs = {}):
	fig,axes = plt.subplots(1,3,dpi=300,figsize=figsize)
	sigma2 = bipcaobj.shrinker.sigma_**2
	if isinstance(bipcaobj.pre_svs,list): #this needs to be cleaned
		if not avg:
			presvs = bipcaobj.pre_svs[ix]
			postsvs = bipcaobj.post_svs[ix]
			postsvs_noisy = postsvs* sigma2
		else:
			presvs = bipcaobj.pre_svs
			postsvs = bipcaobj.post_svs
			postsvs_noisy = [ele * sigma2 for ele in postsvs]
	else:
		presvs = bipcaobj.pre_svs
		postsvs=bipcaobj.post_svs
		postsvs_noisy =  postsvs* sigma2

	ax1 = MP_histogram(presvs,bipcaobj.approximating_gamma,bipcaobj.shrinker.scaled_cutoff_,axes[0])
	ax1.set_title('Unscaled covariance \n' r'$\frac{1}{N}XX^T$')
	ax1.grid(True)

	ax2 = MP_histogram(postsvs_noisy,bipcaobj.approximating_gamma,bipcaobj.shrinker.scaled_cutoff_,axes[1])
	ax2.set_title('Biscaled covariance \n' r'$\frac{1}{N}YY^T$')
	ax2.grid(True)

	ax3 = MP_histogram(postsvs,bipcaobj.approximating_gamma,bipcaobj.shrinker.scaled_cutoff_,axes[2])
	ax3.set_title('Biscaled, noise corrected covariance \n' r'$\frac{1}{N\sigma^{2}}YY^T$' + '\n' + r'$\sigma^2 = {:.2f} $'.format(sigma2))
	ax3.grid(True)
	fig.legend(["Marcenko-Pastur PDF","Theoretical Median", "Actual Median"],bbox_to_anchor=(0.65,0.05),ncol=3)
	fig.suptitle(title)
	fig.tight_layout()
	if output is not '':
		plt.savefig(output, bbox_inches="tight")
	return (ax1,ax2,ax3,fig)

def spectra_from_bipca(bipcaobj, ix = 0, ax = None, dpi=300,figsize = (15,5),title = '', output = ''):
	#this function does not plot from averages.
	fig, (ax0,ax1,ax2) = plt.subplots(1,3,dpi=dpi,figsize = figsize)

	scaled_cutoff = bipcaobj.shrinker.scaled_cutoff__**2
	sigma2 = bipcaobj.shrinker.sigma_**2

	if isinstance(bipcaobj.pre_svs,list):
		presvs = bipcaobj.pre_svs[ix]
		postsvs = bipcaobj.post_svs[ix]
	else:		
		presvs = bipcaobj.pre_svs
		postsvs=bipcaobj.post_svs
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
	ax2.set_title('Biscaled, noise corrected covariance \n' r'$\frac{1}{N\sigma^{2}}YY^T$' '\n' r'$\sigma^2 = $' '{:.2f}'.format(sigma2))


	fig.suptitle(title)
	fig.tight_layout()
	if output is not '':
		plt.savefig(output, bbox_inches="tight")

	return (ax0,ax1,ax2,fig)