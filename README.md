# Biwhitened Principal Component Analysis (BiPCA) # 
![](python/tutorials/Figure1.png)

BiPCA is a python package for processing high-dimensional omics count data, such as scRNAseq, spatial transcriptomics, scATAC-seq, and many others. 
BiPCA first scales the rows and columns of the data to make the noise approximately homoscedastic (biwhitening step), which reveals the underlying rank of the data (based on MP distribution). Then, BiPCA performs optimal shrinkage of singular values to recover the biological signal (denoising step). 

## Installation ##

You can pull the bipca package through

```
git clone https://github.com/KlugerLab/bipca.git
```

We recommend installing BiPCA with the accompanied ```bipca-experiment``` docker environment. The pre-built docker image can be downloaded using 

```
docker pull jyc34/bipca-experiment:lastest
```

and to run the docker container:

```
docker run -it --rm -e USER=john -e USERID=$(id -u) --name bipca -p 8080:8080 -p 8029:8787  -v /data/:/data/   -v /bipca/:/bipca/  docker.io/jyc34/bipca-experiment:lastest  /bin/bash
```

Here, change ```/data:/data``` to ```<your_local_data_directory>:/data``` and ```/bipca/:/bipca/``` to ```<your_local_bipca_directory>:/bipca/```. This will automatically installs bipca when initializing the docker environment. A jupyter-lab will be launched on host port 8080 and a rstudio will be on port 8029. 

See for detailed [descritions](https://github.com/KlugerLab/bipca-experiment) regarding the docker image usage and the information of the corresponding dockerfile.


## Getting Started ##

- Running BiPCA with a built-in dataset: [tutorial-0-quick_start.ipynb](python/tutorials/tutorial-0-quick_start.ipynb)
- Running BiPCA with unfiltered dataset from scanpy: [tutorial-1-pbmc_scrna_scanpy.ipynb](python/tutorials/tutorial-1-pbmc_scrna_scanpy.ipynb)

## Reproducing figures ##

Codes for the generating the figures used in the manuscript are documented as individual functions in [figure.py](python/bipca/experiments/figures/figures.py). For example, run the following to reproduce the marker gene figure:

```
from bipca.experiments.figures import Figure_marker_genes
fig_obj = Figure_marker_genes(base_plot_directory="./result/",output_dir="./data/",formatstr="png")
fig_obj.plot_figure(save=True)
```
[Figure1_Suppfig1.ipynb](python/bipca/experiments/figures/Figure1_Suppfig1.ipynb) regenerates Fig1 and Supplemental fig1 used in the manuscript.

## Reference ##