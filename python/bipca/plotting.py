
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

def MP_histograms_from_bipca(bipcaobj, avg= False, ix=0, ax = None, figsize = (15,4), title='',output = '', histkwargs = {}):
	fig,axes = plt.subplots(1,3,dpi=300,figsize=figsize)
	if isinstance(bipcaobj.pre_svs,list): #this needs to be cleaned
		if not avg:
			presvs = bipcaobj.pre_svs[ix]
			postsvs = bipcaobj.post_svs[ix]
			postsvs_noisy = postsvs* (bipcaobj.shrinker.sigma_**2)
		else:
			presvs = bipcaobj.pre_svs
			postsvs = bipcaobj.post_svs
			postsvs_noisy = [ele * (bipcaobj.shrinker.sigma_**2) for ele in postsvs]
	else:
		presvs = bipcaobj.pre_svs
		postsvs=bipcaobj.post_svs
		postsvs_noisy =  postsvs* (bipcaobj.shrinker.sigma_**2)

	ax1 = MP_histogram(presvs,bipcaobj.approximating_gamma,bipcaobj.shrinker.scaled_cutoff_,axes[0])
	ax1.set_title('Unscaled covariance \n' r'$\frac{1}{N}XX^T$')
	ax2 = MP_histogram(postsvs_noisy,bipcaobj.approximating_gamma,bipcaobj.shrinker.scaled_cutoff_,axes[1])
	ax2.set_title('Biscaled covariance \n' r'$\frac{1}{N}YY^T$')

	ax3 = MP_histogram(postsvs,bipcaobj.approximating_gamma,bipcaobj.shrinker.scaled_cutoff_,axes[2])
	ax3.set_title('Biscaled, noise corrected covariance \n' r'$\frac{1}{N\sigma^{2}}YY^T$')
	fig.legend(["Marcenko-Pastur PDF","Theoretical Median", "Actual Median"],bbox_to_anchor=(0.65,0.05),ncol=3)
	fig.suptitle(title)
	fig.tight_layout()
	plt.savefig(output, bbox_inches="tight")
	return (ax1,ax2,ax3,fig)
