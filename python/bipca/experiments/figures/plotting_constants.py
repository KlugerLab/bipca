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
    "ChromatinConformationCapture": 3,
    "CalciumImaging":5,
    "SingleCellMethylomics":8
}
modality_fill_color = {
    modality: npg_cmap(0.85)(index) for modality, index in modality_color_index.items()
}
modality_label = {
    "SingleCellRNASeq": "scRNA-seq",
    "SingleCellATACSeq": "scATAC-seq",
    "SpatialTranscriptomics": "spatial transcriptomics",
    "ChromatinConformationCapture": "Hi-C",
    #"SingleNucleotidePolymorphism": "genomics",
    "GEXATAC_Multiome":"Multiome",
    "CalciumImaging":"CalciumImaging",
    "SingleCellMethylomics":"SingleCellMethylomics"
}


RNA_color_index = {
    'TenXChromiumRNAV1':0,
    'TenXChromiumRNAV2':1,
       'TenXChromiumRNAV3':2,
    'TenXChromiumRNAV3_1':2,
    'SmartSeqV3':8,
    'SmartSeqV3xpress':6,
    'CITEseq_rna':4,
    'Multiome_rna':5}
atac_color_index = {
    'Buenrostro2015Protocol':0,
    'TenXChromiumATACV1':1,
       'TenXChromiumATACV1_1':1,
    'Multiome_ATAC':2,
"Multiome_ATAC_fragment":3,
"10x_ATAC_fragment":5}
ST_color_index = {
    'TenXVisium':0,
    'DBiTSeq':1,
    'SpatialTranscriptomicsV1':2, 
    'CosMx':3, 
    'SeqFISHPlus':4
       }

RNA_fill_color = {
    modality: npg_cmap(0.85)(index) for modality, index in RNA_color_index.items()
}
ST_fill_color = {
    modality: npg_cmap(0.85)(index) for modality, index in ST_color_index.items()
}
atac_fill_color = {
    modality: npg_cmap(0.85)(index) for modality, index in atac_color_index.items()
}

tech_label = {
    "TenXChromiumRNAV1": "10xChromV1",
    "TenXChromiumRNAV2": "10xChromV2",
    "TenXChromiumRNAV3": "10xChromV3",
    "TenXChromiumRNAV3_1": "10xChromV3",
    "SmartSeqV3":"SmartSeqV3",
    "SmartSeqV3xpress":"SmartSeqV3xpress",
    "CITEseq_rna":"CITEseq (RNA)",
    "Multiome_rna":"Multiome (RNA)",

    "Buenrostro2015Protocol":"Buenrostro2015Protocol",
    "TenXChromiumATACV1":"10xChromV1 - reads",
     "TenXChromiumATACV1_1":"10xChromV1 - reads",
    "Multiome_ATAC":"Multiome (ATAC) - reads",
    "Multiome_ATAC_fragment": "Multiome (ATAC) - fragments",
    "10x_ATAC_fragment":"10xChromV1 - fragments",

    'TenXVisium':"10xVisium",
    'DBiTSeq':"DBiTSeq",
    'SpatialTranscriptomicsV1':"ST", 
    'CosMx':"CosMx", 
    'SeqFISHPlus':"SeqFISHPlus"
    
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