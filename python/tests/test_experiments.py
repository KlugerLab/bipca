from testing_utils import raises
from bipca import utils,experiments
import numpy as np
import unittest
import scipy.sparse as sparse
import torch

class Test_log1p(unittest.TestCase):
    base_X = np.array([[1,1,1],[0,1,1],[2,0,3]])
    base_libsize = np.sum(base_X,axis=0)
    base_median = np.median(base_libsize)
    X_log1p_median = np.log(base_X/base_libsize[:,None] * base_median + 1)
    X_log1p_noscale = np.log(base_X/base_libsize[:,None] + 1)
    def test_tensor(self):
        X = utils.make_tensor(self.base_X)
        Y = experiments.log1p(X)
        assert np.allclose(Y,self.X_log1p_median)
        Y = experiments.log1p(X,scale=1)
        assert np.allclose(Y,self.X_log1p_noscale)
    def test_np(self):
        X = self.base_X
        Y = experiments.log1p(X)

        assert np.allclose(Y,self.X_log1p_median)
        Y = experiments.log1p(X,scale=1)
        assert np.allclose(Y,self.X_log1p_noscale)
    def test_scipy(self):
        X = sparse.csr_matrix(self.base_X)
        Y = experiments.log1p(X)
        assert np.allclose(Y.toarray(),self.X_log1p_median)
        Y = experiments.log1p(X,scale=1)
        assert np.allclose(Y.toarray(),self.X_log1p_noscale)