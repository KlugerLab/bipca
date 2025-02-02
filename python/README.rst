Biwhitened Principal Component Analysis (BiPCA)
===============================================

BiPCA is a Python package for processing high-dimensional omics count data, such as
scRNAseq, spatial transcriptomics, scATAC-seq, and many others.
BiPCA first scales the rows and columns of the data to make the noise approximately
homoscedastic (biwhitening step), which reveals the underlying rank of the data
(based on MP distribution). Then, BiPCA performs optimal shrinkage of singular
values to recover the biological signal (denoising step).

Installation
------------

Pip installation
~~~~~~~~~~~~~~~~

You can install BiPCA using this command:

.. code-block:: bash

   pip install -e 'git+https://github.com/KlugerLab/bipca.git#egg=bipca&subdirectory=python'

Docker installation
~~~~~~~~~~~~~~~~~~~

Alternatively, we recommend installing BiPCA with the accompanied
``bipca-experiment`` Docker environment. This image reproduces the environment
we used to make the BiPCA manuscript. The pre-built docker image can be downloaded using:

.. code-block:: bash

   docker pull jyc34/bipca-experiment:lastest

and to run the Docker container:

.. code-block:: bash

   docker run -it --rm -e USER=john -e USERID=$(id -u) --name bipca -p 8080:8080 -p 8029:8787 \
     -v /data/:/data/ docker.io/jyc34/bipca-experiment:lastest /bin/bash

Here, change ``/data/:/data`` to ``<your_local_data_directory>:/data``.
A JupyterLab will be launched on host port 8080 and an RStudio will be on port 8029.
If you would like to link a local bipca installation (for package development, for instance),
you can use ``-v <your_local_bipca_directory>:/bipca``.

See detailed `descriptions <https://github.com/KlugerLab/bipca-experiment>`_ regarding
the Docker image usage and information on the corresponding Dockerfile.

Getting Started
---------------

- Running BiPCA with a built-in dataset:
  `tutorial-0-quick_start.ipynb <tutorials/tutorial-0-quick_start.ipynb>`_

- Running BiPCA with an unfiltered dataset from scanpy:
  `tutorial-1-pbmc_scrna_scanpy.ipynb <tutorials/tutorial-1-pbmc_scrna_scanpy.ipynb>`_

Reproducing figures
-------------------

Codes for generating the figures used in the manuscript are documented as individual
functions in `figure.py <bipca/experiments/figures/figures.py>`_. For example, run
the following to reproduce the marker gene figure:

.. code-block:: python

   from bipca.experiments.figures import Figure_marker_genes
   fig_obj = Figure_marker_genes(base_plot_directory="./result/",
                                 output_dir="./data/",
                                 formatstr="png")
   fig_obj.plot_figure(save=True)

`Figure1_Suppfig1.ipynb <bipca/experiments/figures/Figure1_Suppfig1.ipynb>`_ regenerates
Fig1 and Supplemental Fig1 used in the manuscript.

Reference
---------

*If you use BiPCA for your research, please cite accordingly.*
