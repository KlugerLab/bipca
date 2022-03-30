.. _math:: 
.. module:: bipca.math

Operator subcomponents of biPCA (:mod:`bipca.math`)
***************************************************

.. currentmodule:: bipca

This module encapsulates the component algorithms that underly biPCA. For a complete end-to-end class to perform bistochastic PCA, see :ref:`its module documentation <bipca>`. 

biPCA is fundamentally a three step algorithm.
For an input X, we:

1. Apply the Sinkhorn algorithm to learn left and right scaling vectors that rescale its row and column variances to be on average 1.

2. Perform singular value decomposition on the scaled matrix X.


For steps 1 and 3, the :class:`math.Sinkhorn` class exposes methods for learning scaling vectors and applying them to matrices.
Step 2 is accomplished by the  class, which wraps a variety of singular value decomposition algorithms for optimal performance for particular data types, as well as storing these decompositions for ease of use. Then, the :class:`math.Shrinker` class can be used to apply a variety of optimal shrinkage techniques.


.. toctree::
   math.Sinkhorn
   