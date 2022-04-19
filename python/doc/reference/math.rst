.. _math:
.. module:: bipca.math

***********************************************************
Operator subcomponents of biPCA (:ref:`bipca.math <math>`)
***********************************************************

.. currentmodule:: bipca.math

This module encapsulates the component algorithms that underly biPCA. For a complete end-to-end class to perform bistochastic PCA, see :ref:`its module documentation <bipca>`. 

biPCA is fundamentally a three step algorithm.
For an input X, we:

1. Apply the Sinkhorn algorithm to learn left and right scaling vectors that rescale its row and column variances to be on average 1.

2. Perform singular value decomposition on the scaled matrix X.


For steps 1 and 3, the :class:`math.Sinkhorn` class exposes methods for learning scaling vectors and applying them to matrices.
Step 2 is accomplished by the  class, which wraps a variety of singular value decomposition algorithms for optimal performance for particular data types, as well as storing these decompositions for ease of use. Then, the :class:`math.Shrinker` class can be used to apply a variety of optimal shrinkage techniques.

.. include:: defs.hrst

Variance Estimation
==================================
.. autosummary::
   :toctree: generated
   :caption: Variance Estimation
   
   bipca.math.QuadraticParameters
   bipca.math.quadratic_variance
   bipca.math.binomial_variance
   bipca.math.general_variance

   
Sinkhorn Biscaling & Biwhitening
=================================
.. autosummary::
   :toctree: generated
   :caption: Sinkhorn Biscaling & Biwhitening

   bipca.math.Sinkhorn
   bipca.math.Sinkhorn.FitParameters

Singular Value Decomposition
============================
.. autosummary::
   :toctree: generated
   :caption: Singular Value Decomposition

   bipca.math.SVD
   bipca.math.SVD.FitParameters


Singular Value Shrinkage & Marcenko-Pastur Analysis
===================================================
.. autosummary::
   :toctree: generated
   :caption: Singular Value Shrinkage & Marcenko-Pastur Analysis


   bipca.math.Shrinker
   bipca.math.Shrinker.FitParameters
   bipca.math._optimal_shrinkage
   bipca.math.MarcenkoPastur
   bipca.math.scaled_mp_bound

Matrix representations
======================
.. autosummary::
   :toctree: generated
   :caption: Matrix representations

   bipca.math.MeanCenteredMatrix
   bipca.math.SamplingMatrix

Utilities
=========
.. autosummary::
   :toctree: generated
   :caption: Utilities

   bipca.math.KDE
   bipca.math.emp_mp_loss
   bipca.math.emp_pdf_loss
   bipca.math.KS
   bipca.math.L2
   bipca.math.L1
