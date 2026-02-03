from __future__ import absolute_import

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("biPCA")
except PackageNotFoundError:
    __version__ = "unknown"
  
from .bipca import BiPCA
