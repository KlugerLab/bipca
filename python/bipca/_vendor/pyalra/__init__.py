"""
.. module:: pyalra
   :platform: Unix, Windows
   :synopsis: Translation of the ALRA R package

.. moduleauthor:: Miles Smith <miles-smith@omrf.org>

Vendored from https://github.com/milescsmith/pyalra (v1.6.2)
License: BSD-3-Clause
"""

from loguru import logger

from bipca._vendor.pyalra.alra import alra
from bipca._vendor.pyalra.choose_k import choose_k

__version__ = "1.6.2"

logger.disable("bipca._vendor.pyalra")

__all__ = ["alra", "choose_k"]
