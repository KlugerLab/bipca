import os
import numpy as np
from numpy.linalg import svd
import scipy as sp
from sklearn.utils.extmath import randomized_svd

# eventually will implement sparse matrix computation
import scipy.sparse as sparse

def parallelanalysis(mat, transform = lambda X: permute(X, axis = 0), means = None, quantile = 1, nIter = 5, sv_cutoff = 1, rule = 'pairwise'):
    dim = np.shape(mat)
    max_rank = int(sv_cutoff*np.amin(dim))
    
    mat_sv = svd(mat)[1]
    if means is not None:
        centered_mat = mat - means
        trans_sv = [svd(transform(centered_mat))[1] for i in range(nIter)]
    else:
        trans_sv = [svd(transform(mat))[1] for i in range(nIter)]
    
    trans_sv_quantiles = np.percentile(trans_sv, 100*quantile, axis = 0)
    
    if rule == 'pairwise':
        # pairwise comparisons of the measurement svs vs the svs of the transformed matrices
        # returns the indices where the measurement svs are smaller than the corresponding transformed mat svs
        index = np.where(mat_sv <= trans_sv_quantiles)[0]
    elif rule == 'top':
        # retuns the indices of where the measurement svs are smaller than the top transformed mat sv
        index = np.where(mat_sv <= trans_sv_quantiles[0])[0]
        
    print(rule, "PA rank is: ", index[0])
    
    if rule == 'top':
        return index[0] if index.size > 0 else -1, trans_sv_quantiles[0]
    else:
        return index[0] if index.size > 0 else -1

def permute(mat, axis = 0):
    # axis = 0 corresponds to permuting the columns independently, 1 corresponds to permuting the rows independently
    return np.apply_along_axis(np.random.permutation, axis, mat)

#     if axis == 0:
#         return np.array([np.random.permutation(mat[i,:]) for i in range(np.shape(mat)[0])])
#     elif axis == 1:
#         return np.transpose(np.array([np.random.permutation(mat[:,j]) for j in range(np.shape(mat)[1])]))
    
def signflip(mat):
    dim = np.shape(mat)
    bern_mat = 2*np.random.binomial(1,0.5,size=dim)-1
    return np.multiply(mat, bern_mat)
    
def deflated_PA(mat, transform = lambda X: permute(X, axis = 1), quantile = 1, sv_cutoff = 1, nIter = 10):
    pcs = 0
    while True:
        index = parallelanalysis(mat, transform, quantile, sv_cutoff, nIter)
        
        # iteratively subtract the top PC if PA determines at least one PC is significant  
        if index >= 0:
            u,s,v = randomized_svd(mat, 1)
            mat = mat - s*(u@v)
            pcs+=1
        else:
            break
            
        # commented out is a slightly different version that subtracts ALL significant PCs at each iteration
        # tbd if that makes a difference
#         if index >= 0:
#             U,s,V = randomized_svd(mat, index+1)
#             mat = mat - U @ np.diag(s) @ V
#             pcs+=np.shape(s)[0]
#         else:
#             break
            
    print("#PCs chosen by deflation: ", pcs)
    return pcs - 1

