"""
pychebfun - Python Chebyshev Polynomials

Vendored from https://github.com/pychebfun/pychebfun (v0.3.0)
See LICENSE in this directory for copyright and license information.
"""

from .chebfun import Chebfun
from .plotting import compare, plot

__all__ = ["Chebfun", "compare", "plot"]
