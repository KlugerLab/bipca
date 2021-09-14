#!/usr/bin/env python

import bipca
from bipca import plotting
import argparse
import scanpy as sc




def main():
	parser = argparse.ArgumentParser(prog='BiPCA', description = "Run bistochastic PCA on count data.")
	parser.add_argument('X', metavar='input_file',type=str, help='Path to the input .h5ad file, \n '+
		'which stores a non-negative count matrix in its .X key.')
	parser.add_argument('Y', metavar='output_file',type=str, help='Output path')
	parser.add_argument('-v','--verbose',type=int, default = 1, 
		choices = [0,1,2,3],help="Logging level {0,1,2,3} to use.")
	backend_group = parser.add_mutually_exclusive_group()

	backend_group.add_argument('-torch_gpu', action = 'store_true',help="Use the experimental torch GPU implementation of BiPCA. Default is torch CPU.")
	backend_group.add_argument('-scipy', action = 'store_true', help="Use scipy implementation of BiPCA. Default is torch CPU.")
	parser.add_argument('-n','--n_iter',type=int, default=500, help="Maximum of sinkhorn iterations")
	# parser.add_argument('-w','--writedebug', action='store_true', help="Write all of the associated matrices to the output."+
	# 	"By default, only the re-scaled output is written along with its rank.")
	parser.add_argument('-r','--randomized',action='store_false', help="Use randomized SVD to compute bipca." + 
		"Recommended for large sparse matrices.")
	parser.add_argument('-a','--exactsigma', action ='store_false', help='Exactly compute the noise variance by using the full matrix.'+
		' Not recommended for large matrices.')
	parser.add_argument('-k','--ncomponents', default=0, type=int,help='Number of PCs to compute during denoising.'+ 
		' Choosing a small number accelerates the algorithm, but can lead to slowdown due to underestimating the rank.')
	parser.add_argument('-q','--q', type=float, default=0.0, help='Pre-estimated q-value. Used with -qits 0.')
	parser.add_argument('-qits','--qits',type=int, default = 21,help="Number of iterations for variance estimation.")
	args = parser.parse_args()

	adata = sc.read_h5ad(args.X)

	if args.torch_gpu:
		backend='torch_gpu'
	elif args.scipy:
		backend= 'scipy'
	else:
		backend='torch'
	bipca_operator = bipca.BiPCA(n_iter = args.n_iter,
		backend=backend, 
		n_components = args.ncomponents, 
		exact = args.randomized, 
		approximate_sigma = args.exactsigma,
		q = args.q,
		qits = args.qits,
		verbose=args.verbose)
	# if not args.writedebug:
	# 	Y = bipca_operator.fit_transform(adata.X)
	# 	adata.layers['Y_bipca'] = Y
	# 	adata.uns['bipca_rank'] = bipca_operator.mp_rank
	# 	adata.uns['bipca_q'] = bipca_operator.
	bipca_operator.fit(adata.X)
	bipca_operator.write_to_adata(adata)
	adata.write(args.Y)


def main_plot():
	parser = argparse.ArgumentParser(prog='BiPCA_plot', description = "Plot the Marcenko-Pastur fit from a biPCA object.")
	parser.add_argument('X', metavar='input_file',type=str, help='Path to the input .h5ad file, \n '+
		'which has been fit previously using biPCA.')
	parser.add_argument('Y', metavar='output_directory',type=str, help='Output path. '+
		'The output path will be appended directly to this path.')
	parser.add_argument('-f','--format',type=str,default='jpg',help='Output file format')
	parser.add_argument('-n','--nbins', type=int, default=100, 
		help='Number of bins to use when generating the histograms.')
	args = parser.parse_args()

	adata = sc.read_h5ad(args.X)
	output_dir = args.Y

	MP_output = output_dir + 'histogram.'+args.format
	spectrum_output = output_dir + 'spectrum.'+args.format

	plotting.MP_histograms_from_bipca(adata,bins=args.nbins,output=MP_output)
	plotting.spectra_from_bipca(adata,output=spectrum_output)

	