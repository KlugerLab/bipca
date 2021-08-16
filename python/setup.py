#!/usr/bin/env python
import sys

from setuptools import find_packages, setup
from bipca import __version__ as version
setup(name='biPCA',
      version=version,
      description='Python Distribution Utilities',
      author='Jay S. Stanley III, Junchen Yang, Thomas Zhang, Boris Landa, Yuval Kluger',
      author_email='jay.s.stanley.3@gmail.com',
      license="GNU General Public License Version 2",
      packages=find_packages(),
      scripts=['bin/bipca']
     )