#!/usr/bin/env python
import sys

from setuptools import find_packages, setup
from bipca import __version__ as version
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='biPCA',
      version=version,
      description='Bistochastic PCA',
      author='Jay S. Stanley III, Junchen Yang, Thomas Zhang, Boris Landa, Yuval Kluger',
      author_email='jay.s.stanley.3@gmail.com',
      license="GNU General Public License Version 2",
      packages=find_packages(),
      entry_points = {
        'console_scripts': ['bipca=bipca.command_line:bipca_main','bipca-plot=bipca.command_line:bipca_plot'],
    },
      install_requires = requirements
     )
