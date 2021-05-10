
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from .math import mp_pdf,mp_quantile,emp_pdf_loss

def MP_histogram(svs,gamma,cutoff,  ax = None, histkwargs = {}):
	if ax is None:
		ax = plt.axes()
	if not isinstance(svs,list):
		svs = [svs]
	sv = svs[0]
	theoretical_median = mp_quantile(gamma, mp = lambda x,gamma: mp_pdf(x,gamma))
	n, bins = np.histogram(sv[sv<=cutoff*2], bins=50, range = [0, cutoff*2],density = True,*histkwargs)
	actual_median = np.median(sv)
	for sv in svs[1:]:
		nn, _ = np.histogram(sv[sv<=cutoff*2],bins=bins,density = True)
		actual_median += np.median(sv)

		n+=nn
	n = n / len(svs)
	w = bins[:-1]-bins[1:]
	actual_median = actual_median /len(svs)
	ax.bar(bins[:-1],n,width = w, align='edge')
	est_dist = stats.rv_histogram([n, bins])
	est_loss = emp_pdf_loss(lambda x: mp_pdf(x,gamma),est_dist.pdf)
	xx=np.linspace(0.000001, bins[-1]*1.1, 10000)
	ax.plot(xx,mp_pdf(xx,gamma), 'go-', markersize = 2)
	ax.plot(theoretical_median*np.ones(50),np.linspace(0,max(n),50),'rx--',alpha=1,markersize=2)
	ax.plot(actual_median*np.ones(50),np.linspace(0,max(n),50),'ys--',alpha=0.5,markersize=2)

	return ax

def MP_histograms_from_bipca(bipcaobj, ax = None, figsize = (), histkwargs = {}):
	fig,axes = plt.subplots(1,2,dpi=300,figsize=figsize)

	ax1 = MP_histogram(bipcaobj.pre_svs,bipcaobj.approximating_gamma,bipcaobj.shrinker.scaled_cutoff_,axes[0])
	ax1.set_title('Before biPCA')
	ax2 = MP_histogram(bipcaobj.post_svs,bipcaobj.approximating_gamma,bipcaobj.shrinker.scaled_cutoff_,axes[1])
	ax2.set_title('After biPCA')
	fig.legend(["Marcenko-Pastur PDF","Theoretical Median", "Actual Median"],bbox_to_anchor=(0.65,0.05),ncol=3)
	return ax
