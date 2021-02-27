import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd
import seaborn as sns

import numpy as np
from numpy import random as rand
import scipy as sp

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

import scipy.sparse as sparse
from sklearn.utils.extmath import randomized_svd


# function returns a (dim x len(read_counts)) matrix of probabilities
def gen_prob_vecs(dim, n_vecs, variety = 'unif'):
    
    if variety == 'unif':
        probs = np.zeros((dim, n_vecs))
        for i in range(n_vecs):
            prob = rand.uniform(0, 20, dim)
            prob = prob / np.sum(prob)
            probs[:, i] = prob
            
    elif variety == 'exp':
        probs = np.zeros((dim, n_vecs))
        for i in range(n_vecs):
            prob = np.exp(rand.uniform(0, 5, dim))
            prob = prob / np.sum(prob)
            probs[:, i] = prob
    
    return probs



# generate a probability matrix of dim M x N from the cell-type probability matrix given by gen_prob_vecs
def gen_prob_mat(N, cell_prob_vecs, sampling_probs = None):
    # sampling_probs is an optional parameter where one can give the probabilities that given cell types are sampled
    # default is uniform over the cell types
    
    n_vecs = np.shape(cell_prob_vecs)[1]
    M = np.shape(cell_prob_vecs)[0]
    prob_mat = np.zeros((M,N))
    if sampling_probs == None:
        for i in range(N):
            cell_type = rand.randint(0, n_vecs) #"high" is exclusive, so we get cell_type in 0:n_vecs-1
            prob_mat[:,i] = cell_prob_vecs[:, cell_type]
#         print("dim(prob_mat) = ", np.shape(prob_mat.T))
    
    return prob_mat


# function that given a probability matrix and the read counts of each col, 
# returns a matrix where each entry is a binomial sample
def gen_binom_mat(prob_mat, read_counts, scaled_to_prob = False):
    # check that n_cols of prob_mat equals read_counts
    assert np.shape(prob_mat)[1] == np.shape(read_counts)[0], "Dimension mismatch"
    N = np.shape(read_counts)[0]
    
    binom_mat = np.array([rand.binomial(read_counts[i], prob_mat[:,i]) for i in range(N)])
    
    if scaled_to_prob == True:
        # if we want the sample matrix in the form of a probability matrix, divide each column by its sum
        binom_mat = binom_mat/np.sum(binom_mat, axis = 0)
    
    return binom_mat.T

def gen_multi_mat(prob_mat, read_counts, scaled_to_prob = False):
    # check that n_cols of prob_mat equals read_counts
    assert np.shape(prob_mat)[1] == np.shape(read_counts)[0], "Dimension mismatch"
    N = np.shape(read_counts)[0]
    
    prob_mat = np.append(prob_mat, [np.sum(prob_mat, axis = 0)], axis = 0)    
    
    multi_mat = (np.array([rand.multinomial(read_counts[i], prob_mat[:,i], size=1)[0] for i in range(N)])).T
#     for i in range(N):
#         # take nTrials samples from a multinom model using the i-th column of pmat (padded with 1-colsum) as prob vector
#         measurements[:,i] = np.random.multinomial(read_counts[i], pmat[:,i], size=1)[0]
    
    # take out the last row
    multi_mat = multi_mat[:-1,:]
    return multi_mat

# Helper function: Sinkhorn algorithm: unaltered and probably numerically unstable
def sinkhorn(mat, r_sums, c_sums, nIter = 100, eps = 0.001, verbose = False):
    
    assert np.amin(mat) >= 0, "Matrix is not non-negative"
    assert np.shape(mat)[0] == np.shape(r_sums)[0], "Row dimensions mismatch"
    assert np.shape(mat)[1] == np.shape(c_sums)[0], "Column dimensions mismatch"
    
    # sum(r_sums) must equal sum(c_sums), at least approximately
    assert np.abs(np.sum(r_sums) - np.sum(c_sums)) < eps, "Rowsums and colsums do not add up to the same number"
    
    
    n_row = np.shape(mat)[0]
    n_col = np.shape(mat)[1]
    
    a = np.ones(n_row)
    for i in range(nIter):
        b = np.divide(c_sums, mat.T @ a)
        a = np.divide(r_sums, mat @ b)
    
    scaled_mat = (mat * b) * a[:,None]
    
    if verbose:
        print("Row error: ", np.amax(np.abs(r_sums - np.sum(scaled_mat, axis = 1))))
        print("Col error: ", np.amax(np.abs(c_sums - np.sum(scaled_mat, axis = 0))))
    # returned values are the sinkhorn scaled matrix, and the corresponding left and right weights s.t.
    # mat = diag(a) * scaled_mat * diag(b)
    return scaled_mat, a, b


# Function that does the whitening
def biscaling(meas_mat, read_counts=None, return_H = False, H_0 = None, verbose = False):
    assert np.amin(meas_mat) >= 0, "Matrix is not non-negative"
    
    # assumes that measurements are counts, not scaled probabilities
    if read_counts is None:
        read_counts = np.sum(meas_mat, axis = 0)
        
    n_row = np.shape(meas_mat)[0]
    n_col = np.shape(meas_mat)[1]
    
    if H_0 is None:
        # compute the "scaled" sample variance matrix
        print("Binomial model")
        H_0 = meas_mat * np.divide(read_counts, read_counts - 1) - meas_mat**2 * (1/(read_counts-1))
    
    sk_scale = sinkhorn(H_0, np.full(n_row, n_col), np.full(n_col, n_row), verbose = verbose)
    
    left = np.sqrt(sk_scale[1])
    right = np.sqrt(sk_scale[2])
    
    scaled_meas_mat = (meas_mat * right) * left[:,None]
    
    # after bi-scaling, the columns of the matrix should have variance 1. The next step is to mean-center the columns
    col_means = np.mean(scaled_meas_mat, axis = 0)
    mean_center_mat = scaled_meas_mat - col_means
    
    # Given matrix A, returns D_1^(1/2)*A*D_2^(1/2), as well as the scaling matrices in diagonal form
    
    if return_H == True:
        return scaled_meas_mat, left, right, mean_center_mat, col_means, H_0, sk_scale
    else:
        return scaled_meas_mat, left, right, mean_center_mat, col_means
   
    
    
# pre-processing function that removes 0 rows and columns with the option of adding a small eps to matrix
# to allow for sinkhorn to converge faster/better
def stabilize_matrix(mat, read_cts = None, add_eps = False, return_zero_indices = False):
    
    print("Old dimension: ", np.shape(mat))
    
    # Might need a method here that determines what the tolerance value is
    # Since in this experiment we are generating count data, the tolerance value can be 0. We will set to 1e-6 just in case
    
    tol = 1E-6
    zero_rows = np.all(np.abs(mat)<=tol, axis = 1)
    zero_cols = np.all(np.abs(mat)<=tol, axis = 0)
    
    mat = mat[~zero_rows,:]
    mat = mat[:,~zero_cols]
    
    print("New dimension: ", np.shape(mat))
    
    if return_zero_indices == True:
        if read_cts is not None:     
            # if we have read counts to prune as well, we do that here
            read_cts = read_cts[~zero_cols]
            return mat, read_cts, [zero_rows, zero_cols]
        
        return mat, [zero_rows, zero_cols]
    
    if add_eps == True:
        pass
    
    
    return mat
    
# helper function that reconstructs matrix given sinkhorn-scaled matrix and its left and right weights
def reconstruct(data, revert = False):
    # in our code, data[0] is the sinkhorn-scaled matrix, data[1] is the left weights, and data[2] is the right weights
    
    if revert == True:
        l = 1/data[1]
        r = 1/data[2]
        mat = (data[0] * r) * l[:,None]
    else:
        l = data[1]
        r = data[2]
        mat = (data[0] * r) * l[:,None]
    return mat