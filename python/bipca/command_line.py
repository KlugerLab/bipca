#!/usr/bin/env python

import bipca
from os.path import exists
import argparse
import anndata as ad
import sys
import torch
import numpy as np
from threadpoolctl import threadpool_limits
import faulthandler
def bipca_main(args = None):
	args = bipca_parse_args(args)
	adata = ad.read_h5ad(args.X)

	torch.set_num_threads(args.threads)
	bipca_operator = bipca.BiPCA(n_iter = args.n_iter,
		backend=args.backend,
		variance_estimator=args.variance_estimator,
		n_components = args.ncomponents, 
		exact = args.randomized, 
		dense_svd=args.sparse_svd,
		conserve_memory=args.conserve_memory,
		njobs = args.njobs,
		qits = args.qits,
		n_subsamples = args.n_subsamples,
		subsample_size = args.subsample_size,
		read_counts=args.read_counts,
		use_eig = args.no_eig,
		oversample_factor=args.oversample_factor,
		b = args.quadratic_b,
		c = args.quadratic_c,
		bhat = args.quadratic_bhat,
		chat = args.quadratic_chat,
		verbose=args.verbose)
	faulthandler.enable()
	with threadpool_limits(limits=args.threads):
		bipca_operator.fit(adata.X)
		if args.no_plotting_spectrum:
			pass
		else:
			if args.subsample_plotting_spectrum:
				bipca_operator.get_plotting_spectrum(subsample=True)
				cutoff = (1+np.sqrt(bipca_operator.aspect_ratio))**2
				rank = (rank >= cutoff).sum()
				if bipca_operator.mp_rank < rank:
					bipca_operator.logger.log_warning(f"**** The rank of the"
						" fitted operator ({bipca_operator.mp_rank})"
						" did not match the rank of the"
						" plotting spectrum ({rank}). Recommend refitting with"
						" by increasing -k, setting k = -1 "
						" or increasing --oversample_factor. ****")
			else:
				bipca_operator.get_plotting_spectrum(subsample=False)
				rank = bipca_operator.plotting_spectrum['Y']
				cutoff = (1+np.sqrt(bipca_operator.aspect_ratio))**2
				rank = (rank >= cutoff).sum()
				if bipca_operator.mp_rank != rank:
					bipca_operator.logger.log_warning(f"**** The rank of the"
						" fitted operator ({bipca_operator.mp_rank})"
						" did not match the rank of the"
						" plotting spectrum ({rank}). Recommend refitting with"
						" by increasing -k, setting k = -1 "
						" or increasing --oversample_factor. ****")
	bipca_operator.write_to_adata(adata)
	adata.write(args.Y)

def bipca_parse_args(args):

	parser = argparse.ArgumentParser(prog='BiPCA', 
		description = "Run bistochastic PCA on count data.")
	## Basic arguments
	parser.add_argument('X', metavar='input_file',type=str, 
		help='Path to the input .h5ad file, \n '+
		'which stores a non-negative count matrix in its .X key.')
	parser.add_argument('Y', metavar='output_file',type=str, help='Output path')
	parser.add_argument('-v','--verbose',type=int, default = 1, 
		choices = [0,1,2,3],help="Logging level {0,1,2,3} to use.")

	parser.add_argument('-t','--threads',type=int, default=None,
		help = "Number of threads to use in Torch. Defaults to numcores/4")
	parser.add_argument('-njobs','--njobs',type=int, default=-1,
		help = "Number of jobs to use when computing chebyshev approximation")
	## Backend arguments
	backend_group = parser.add_mutually_exclusive_group()
	backend_group.add_argument('-torch_gpu', action = 'store_true',
		help="Use the experimental torch GPU implementation of BiPCA."+
		" Default is torch CPU.")
	backend_group.add_argument('-scipy', action = 'store_true', 
		help="Use scipy implementation of BiPCA. Default is torch CPU.")
	## Sinkhorn argumnets
	parser.add_argument('-n','--n_iter',type=int, default=500, 
		help="Maximum of sinkhorn iterations")
	## SVD arguments
	parser.add_argument('-r','--randomized',action='store_false', 
		help="Use randomized SVD to compute bipca." + 
		"Recommended for large sparse matrices.")
	parser.add_argument('-k','--ncomponents', default=0, type=int,
		help='Number of PCs to compute during denoising.'+ 
		' Choosing a small number accelerates the algorithm, '+
		'but can lead to slowdown due to underestimating the rank.')
	parser.add_argument('-sparse_svd','--sparse_svd', action='store_false',
		help='Use a sparse SVD for sparse inputs. By default,' +
		'Dense SVD is used to optimize for speed. '+ 
		'Enable this option to help with memory usage.')
	parser.add_argument('-no_eig', '--no_eig', action='store_false',
		help="Use a direct SVD, rather than computing the dense "
		" eigendecomposition. Enable this option when forming X@X.T or X.T@X" 
		" is inaccurate or leads to memory problems.")
	parser.add_argument('-o', '--oversample_factor', type=float,default=10,
		help="Oversampling ratio for randomized svd. Only used when k is "
		"not -1 and less than the minimum dimension of the data"
		"and a partial decomposition is requested.")
	parser.add_argument('-conserve_memory','--conserve_memory',action='store_true',
		help='Conserve memory usage. Use in combination with -sparse_svd')
	## variance estimation arguments
	### binomial arguments
	parser.add_argument('-var','--variance_estimator',type=str,
		default='quadratic',choices = ['quadratic','binomial'], 
		help='Variance estimator to use.')
	parser.add_argument('-rc','--read-counts', type=int, default=None, 
		help="Binomial read counts. Use with -var binomial.")

	### arguments for the quadratic variance estimate fitting
	parser.add_argument('-qits','--qits',type=int, default = 51,
		help="Number of iterations for quadratic variance estimation.")
	parser.add_argument('-nsubs','--n_subsamples',type=int, default=5, 
		help="Number of subsamples to use when computing quadratic variance."+
		" Set to 0 to fit the full matrix.")
	parser.add_argument('-subsize', '--subsample_size',type=int,default=5000,
		help="Size of subsamples to use when computing quadratic variance." +
		" Set to greater than the minimum dimension of the input to fit the " +
		"full matrix.")
	### arguments for pre-specified quadratic fits
	parser.add_argument('-b','--quadratic_b', type=float,default=None,
		help="Linear variance term to use when computing quadratic variance.")
	parser.add_argument('-bhat','--quadratic_bhat',type=float, default = None,
		help="Underlying linear variance term to use when computing quadratic"+ 
		" variance.")
	parser.add_argument('-c','--quadratic_c', type=float,default=None,
		help="Quadratic variance term to use when computing quadratic variance.")
	parser.add_argument('-chat','--quadratic_chat',type=float, default = None,
		help="Underlying quadratic variance term to use when computing quadratic"+ 
		" variance.")

	## Argument for getting plotting data
	plotting_group = parser.add_mutually_exclusive_group()

	plotting_group.add_argument('--subsample_plotting_spectrum', action='store_true',
		help='Write a subsampled plotting spectrum to the file. By default '+
		'the full plotting spectrum is written.')
	plotting_group.add_argument('--no_plotting_spectrum',action='store_true',
		help='Disable plotting spectrum. ' +
		'By default, the full plotting spectrum is written')
	args = parser.parse_args(args)
	if args.torch_gpu:
		args.backend='torch_gpu'
	elif args.scipy:
		args.backend= 'scipy'
	else:
		args.backend='torch'
	if not exists(args.X):
		raise ValueError("Input file {} does not exist.".format(args.X))
	if args.threads is None:
		args.threads = torch.get_num_threads()
		args.threads = int(np.ceil(args.threads/4))
	return args

def bipca_plot_parse_args(args):
	parser = argparse.ArgumentParser(prog='BiPCA_plot', description = "Plot the Marcenko-Pastur fit from a biPCA object.")
	parser.add_argument('X', metavar='input_file',type=str, help='Path to the input .h5ad file, \n '+
		'which has been fit previously using biPCA.')
	parser.add_argument('Y', metavar='output_directory',type=str, help='Output path. '+
		'The output path will be appended directly to this path.')
	parser.add_argument('-f','--format',type=str,default='jpg',help='Output file format')
	parser.add_argument('-n','--nbins', type=int, default=100, 
		help='Number of bins to use when generating the histograms.')
	parser.add_argument('-minus','--minus',nargs="*",type=int,default=[10,10],
		help="Number of singular values to plot less than the rank-th "
		"singular value. Pass two arguments to control the pre and post biPCA "
		"plots separately.")
	parser.add_argument('-plus','--plus',nargs="*",type=int,default=[10,10],
		help="Number of singular values to plot greater than the rank-th "
		"singular value. Pass two arguments to control the pre and post biPCA "
		"plots separately.")
	parser.add_argument('-scale','--scale', type=str,default = 'linear',
		choices = ['log','linear','symlog'],
		help="Plot spectrum using scale.")
	args = parser.parse_args(args)
	if not exists(args.X):
		raise ValueError("Input file {} does not exist.".format(args.X))
	return args
def bipca_plot(args = None):
	from bipca import plotting

	args = bipca_plot_parse_args(args)

	adata = ad.read_h5ad(args.X)
	output_dir = args.Y

	MP_output = output_dir + 'histogram.'+args.format
	spectrum_output = output_dir + 'spectrum.'+args.format
	KS_output = output_dir + 'KS.'+args.format

	plotting.MP_histograms_from_bipca(adata,bins=args.nbins,output=MP_output)
	plotting.spectra_from_bipca(adata,scale = args.scale,
		plus=args.plus,minus=args.minus,
		output=spectrum_output)
	try: #KS_from_bipca throws a value error if the bipcaobj is not quadratic
		plotting.KS_from_bipca(adata,output=KS_output)
	except:
		pass

