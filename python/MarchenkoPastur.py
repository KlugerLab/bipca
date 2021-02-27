import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd
import seaborn as sns

import numpy as np
from numpy import random as rand
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import scipy.sparse as sparse
import scipy.integrate as integrate
from sklearn.utils.extmath import randomized_svd


import biscaling_PCA as biPCA
import parallel_analysis as pa


def mp(x, g):
    # Marchenko-Pastur distribution pdf
    # g is aspect ratio gamma
    def m0(a):
        "Element wise maximum of (a,0)"
        return np.maximum(a, np.zeros_like(a))
    gplus=(1+g**0.5)**2
    gminus=(1-g**0.5)**2
    return np.sqrt(  m0(gplus  - x) *  m0(x- gminus)) / ( 2*np.pi*g*x)


# find location of given quantile of standard marchenko-pastur
def mp_quantile(mp, gamma, q = 0.5, eps = 1E-9):
    
    l_edge = (1 - np.sqrt(gamma))**2
    u_edge = (1 + np.sqrt(gamma))**2
    
    print(integrate.quad(lambda x: mp(x, gamma), l_edge, u_edge)[0])
    
    # binary search
    nIter = 0
    error = 1
    left = l_edge
    right = u_edge
    cent = left
    
    while error > eps:
        cent = (left + right)/2
        val = integrate.quad(lambda x: mp(x, gamma), l_edge, cent)[0]
        error = np.absolute(val - q)
        if val < q:
            left = cent
        elif val > q:
            right = cent
        else:
            # integral is exactly equal to quantile
            return cent
        
        nIter+=1
    
    print("Number of iters: ", nIter)
    print("Error: ", error)
    
    return cent


def L2(x, func1, func2):
    
    return np.square(func1(x) - func2(x))


def L1(x, func1, func2):
    
    return np.absolute(func1(x) - func2(x))


# evaluate given loss function on a pdf and an empirical pdf (histogram data)
def emp_pdf_loss(pdf, epdf, loss = L2, start = 0):
    
    # loss() should have three arguments: x, func1, func2
    # note 0 is the left limit because our pdfs are strictly supported on the non-negative reals, due to the nature of sv's
    
    val = integrate.quad(lambda x: loss(x, pdf, epdf), start, np.inf)[0]
    
    
    return val

def emp_hist_loss(n, bins, gamma):
    
    nhist = len(n)
    val = 0
    
    u_edge = (1 + np.sqrt(gamma))**2
    
    for i in range(nhist):
        seg_len = bins[i+1] - bins[i]
        mp_segment_int = integrate.quad(lambda x: mp(x,gamma), bins[i], bins[i+1])[0]
        diff = (mp_segment_int - n[i]*seg_len)**2
        val+=diff
    
    if bins[-1] < u_edge:
        val+= integrate.quad(lambda x: mp(x,gamma), bins[-1], u_edge)[0]**2
    
    val = np.sqrt(val)
    
    return val

def emp_mp_loss(mat, gamma = 0, loss = L2):
    
    
    M = np.shape(mat)[0]
    N = np.shape(mat)[1]
    if gamma == 0:
        gamma = M/N

    if gamma >= 1:
        # have to pad singular values with 0
        svs = np.linalg.svd(mat)[1]
        cov_eig = np.append(1/N*svs**2, np.zeros(M-N))
        # hist = np.histogram(cov_eig, bins=np.minimum(np.int(N/4),60))
        hist = np.histogram(cov_eig, bins = np.int(4*np.log2(5*N)))
        esd = sp.stats.rv_histogram(hist).pdf

        # error at 0 is the difference between the first bin of the histogram and (1 - 1/gamma) = (M - N)/N
        err_at_zero = np.absolute(hist[0][0] - (1 - 1 / gamma))
        if loss == L2:
            err_at_zero = err_at_zero**2

        # we now start integrating AFTER the bin that contains the zeros
        start = hist[1][1]
        u_edge = (1 + np.sqrt(gamma))**2
        # we integrate a little past the upper edge of MP, or the last bin of the histogram, whichever one is greater.
        end = 1.2*np.maximum(u_edge, hist[1][-1])
        # end = 20
        val = integrate.quad(lambda x: loss(x, lambda y: mp(y, gamma), esd), start, end)[0] + err_at_zero
    
    else:
        svs = np.linalg.svd(mat)[1]
        cov_eig = 1/N*svs**2
        # hist = np.histogram(cov_eig, bins=np.minimum(np.int(N/4),60))
        hist = np.histogram(cov_eig, bins = np.int(4*np.log2(5*N)))
        esd = sp.stats.rv_histogram(hist).pdf
        
        u_edge = (1 + np.sqrt(gamma))**2
        end = 1.2*np.maximum(u_edge, hist[1][-1])
        # end = 20
        val = integrate.quad(lambda x: loss(x, lambda y: mp(y, gamma), esd), 0, end)[0]

    return val