"""BiPCA: Biwhitened Principal Component Analysis.

A method for denoising and dimensionality reduction of count data.
BiPCA uses Sinkhorn bistochastic scaling to biwhiten the input matrix,
followed by singular value decomposition and optimal singular value
shrinkage under a Marcenko-Pastur noise model. This produces denoised
count matrices and principal components suitable for downstream analysis
of high-dimensional count data such as single-cell RNA sequencing.
"""

from __future__ import absolute_import

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("biPCA")
except PackageNotFoundError:
    __version__ = "unknown"
  
from .bipca import BiPCA
