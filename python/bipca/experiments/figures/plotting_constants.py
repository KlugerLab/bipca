import numpy as np
from matplotlib.colors import LinearSegmentedColormap
# Orders, cmaps, etc used for all figures
from .utils import npg_cmap

algorithm_color_index = {
    "log1p": 0,
    "log1p+z": 4,
    "Pearson": 3,
    "Sanity": 1,
    "ALRA": 8,
    "BiPCA": 2,
}
algorithm_fill_color = {
    algorithm: npg_cmap(0.85)(index)
    for algorithm, index in algorithm_color_index.items()
}

modality_color_index = {
    "SingleCellRNASeq": 2,
    "SingleCellATACSeq": 1,
    "SpatialTranscriptomics": 0,
    "ChromatinConformationCapture": 4,
    "SingleNucleotidePolymorphism": 3,
}
modality_fill_color = {
    modality: npg_cmap(0.85)(index) for modality, index in modality_color_index.items()
}
modality_label = {
    "SingleCellRNASeq": "scRNA-seq",
    "SingleCellATACSeq": "scATAC-seq",
    "SpatialTranscriptomics": "spatial transcriptomics",
    "ChromatinConformationCapture": "Hi-C",
    "SingleNucleotidePolymorphism": "genomics",
}
dataset_label = {"TenX": "10X"}

line_cmap = npg_cmap(alpha=1)
fill_cmap = npg_cmap(alpha=0.5)

marker_experiment_colors = {r'$+$ cluster': 7,
                            r'$-$ cluster': 3}

gradient = []
for alpha in np.linspace(1,0,101):
    gradient.append(npg_cmap(alpha)(3))
for alpha in np.linspace(0,1,101):
    gradient.append(npg_cmap(alpha)(7))
heatmap_cmap = LinearSegmentedColormap.from_list('npg_heatmap',gradient)