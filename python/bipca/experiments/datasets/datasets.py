#standard libraries
import tarfile
import gzip
import zipfile
import sys
import inspect
import re
import subprocess
import os #refactor this out?
import json
from shutil import move as mv, rmtree, copyfileobj
from numbers import Number
from typing import Dict
from pathlib import Path
from itertools import combinations

#scipy libraries
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.io import loadmat, mmread

#anndata/scanpy
import scanpy as sc
import anndata as ad
from anndata import AnnData, read_h5ad

#ALLCools
from ALLCools.mcds import MCDS
from ALLCools.mcds.mcds import _make_obs_df_var_df

#other libs
from pandas_plink import read_plink
import pyreadr
from imageio import imread
#bipca
from bipca.math import MarcenkoPastur
from bipca.experiments.datasets.base import AnnDataFilters, Dataset
from bipca.experiments.datasets.utils import (
    get_ensembl_mappings,
    read_csv_pyarrow_bad_colnames,
)
from bipca.experiments.datasets.modalities import *
from bipca.experiments.utils import get_rng




from bipca.utils import nz_along


def get_all_datasets():
    return [
        ele[1]
        for ele in inspect.getmembers(
            sys.modules[__name__],
            lambda member: inspect.isclass(member)
            and member.__module__ == __name__
            and issubclass(member, Dataset)
            and not issubclass(member, Simulation),
        )
    ]


# SIMULATIONS #
class RankRPoisson(LowRankSimulation):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def _default_session_directory(self) -> str:
        return (
            f"seed{self.seed}"
            f"rank{self.rank}"
            f"entrywise_mean{self.entrywise_mean}"
            f"libsize_mean{self.libsize_mean}"
            f"minimum_singular_value{self.minimum_singular_value}"
            f"constant_singular_value{self.constant_singular_value}"
            f"mrows{self.mrows}"
            f"ncols{self.ncols}"
        )

    def _compute_simulated_data(self):
        # Generate a random matrix with rank r
        rng = get_rng(self.seed)
        X = self.get_low_rank_matrix(rng)
        Y = rng.poisson(lam=X)  # Poisson sampling
        adata = AnnData(Y, dtype=float)
        adata.layers["ground_truth"] = X
        adata.uns["rank"] = self.rank
        adata.uns["seed"] = self.seed

        adata.uns["entrywise_mean"] = self.entrywise_mean
        adata.uns["library_size_mean"] = self.libsize_mean
        adata.uns["b"] = 1
        adata.uns["c"] = 0
        adata.uns["minimum_singular_value"] = self.minimum_singular_value
        adata.uns["constant_singular_value"] = self.constant_singular_value

        return adata


class QVFNegativeBinomial(LowRankSimulation):
    def __init__(
        self,
        b: Number = 1,
        c: Number = 0.00001,
        **kwargs,
    ):
        self.b = b
        self.c = c
        super().__init__(**kwargs)

    def _default_session_directory(self) -> str:
        return (
            f"seed{self.seed}"
            f"rank{self.rank}"
            f"entrywise_mean{self.entrywise_mean}"
            f"libsize_mean{self.libsize_mean}"
            f"minimum_singular_value{self.minimum_singular_value}"
            f"constant_singular_value{self.constant_singular_value}"
            f"b{self.b}"
            f"c{self.c}"
            f"mrows{self.mrows}"
            f"ncols{self.ncols}"
        )

    def _compute_simulated_data(self):
        # the variance of a negative binomial is
        # mu^2 / n  + mu
        # therefore b = mu
        # c = 1 / n
        rng = get_rng(self.seed)
        X = self.get_low_rank_matrix(rng)
        theta = 1 / self.c
        nb_p = theta / (theta + X)
        Y0 = rng.negative_binomial(theta, nb_p)
        Y = self.b * Y0
        adata = AnnData(Y, dtype=float)
        adata.layers["ground_truth"] = X
        adata.uns["rank"] = self.rank
        adata.uns["seed"] = self.seed
        adata.uns["entrywise_mean"] = self.entrywise_mean
        adata.uns["library_size_mean"] = self.libsize_mean
        adata.uns["minimum_singular_value"] = self.minimum_singular_value
        adata.uns["constant_singular_value"] = self.constant_singular_value
        adata.uns["b"] = self.b
        adata.uns["c"] = self.c

        return adata


###################################################
#   Real Data                                     #
###################################################
###################################################
#   Hi-C                                          #
###################################################
class Johanson2018(ChromatinConformationCapture):
    _citation = (
        "@article{johanson2018genome,\n"
        "   title={Genome-wide analysis reveals no evidence of trans chromosomal "
        "regulation of mammalian immune development},\n"
        "   author={Johanson, Timothy M and Coughlan, Hannah D and Lun, Aaron TL and "
        "Bediaga, Naiara G and Naselli, Gaetano and Garnham, Alexandra L and Harrison, "
        "Leonard C and Smyth, Gordon K and Allan, Rhys S},\n"
        "   journal={PLoS genetics},\n"
        "   volume={14},\n"
        "   number={6},\n"
        "   pages={e1007431},\n"
        "   year={2018},\n"
        "   publisher={Public Library of Science San Francisco, CA USA}\n"
        "}"
        "@article{bediaga2021multi,\n"
        "    title={Multi-level remodelling of chromatin underlying activation "
        "of human T cells},\n"
        "    author={Bediaga, Naiara G and Coughlan, Hannah D and "
        "Johanson, Timothy M and Garnham, Alexandra L and Naselli, Gaetano "
        "and Schr{\"o}der, Jan and Fearnley, Liam G and Bandala-Sanchez, Esther "
        "and Allan, Rhys S and Smyth, Gordon K and others},\n"
        "    journal={Scientific reports},\n"
        "    volume={11},\n"
        "    number={1},\n"
        "    pages={528},\n"
        "    year={2021},\n"
        "    publisher={Nature Publishing Group UK London}\n"
        "}"
    )
    _raw_urls = {
        "CD4T1.mtx.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2827nnn/GSM2827786/"
            "suppl/GSM2827786_CD4T1_hg_t.mtx.gz"
        ),
        "CD4T2.mtx.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2827nnn/GSM2827787/"
            "suppl/GSM2827787_CD4T2_hg_t.mtx.gz"
        ),
        "CD8T1.mtx.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2827nnn/GSM2827788/"
            "suppl/GSM2827788_CD8T1_hg_t.mtx.gz"
        ),
        "CD8T2.mtx.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2827nnn/GSM2827789/"
            "suppl/GSM2827789_CD8T2_hg_t.mtx.gz"
        ),
        "B1.mtx.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2827nnn/GSM2827790/"
            "suppl/GSM2827790_HB1_hg_t.mtx.gz"
        ),
        "B2.mtx.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2827nnn/GSM2827791/"
            "suppl/GSM2827791_HB2_hg_t.mtx.gz"
        ),
        "activatedCD4T1.mtx.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3591nnn/GSM3591809/"
            "suppl/GSM3591809_CD4T_Ac1.mtx.gz"
        ),
        "activatedCD4T2.mtx.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3591nnn/GSM3591810/"
            "suppl/GSM3591810_CD4T_Ac2.mtx.gz"
        ),
        "activatedCD8T1.mtx.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3591nnn/GSM3591811/"
            "suppl/GSM3591811_CD8T_Ac1.mtx.gz"
        ),
        "activatedCD8T2.mtx.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3591nnn/GSM3591812/"
            "suppl/GSM3591812_CD8T_Ac2.mtx.gz"
        ),
        "regions.bed.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE105nnn/GSE105776/"
            "suppl/GSE105776_GenomicRegions.bed.gz"
        ),
    }

    _unfiltered_urls = [
        f"{sample.replace('.mtx.gz','')}" for sample in _raw_urls if ".mtx" in sample
    ]
    _unfiltered_urls = [
        f"{sample}_{i}_{j}.h5ad"
        for sample in _unfiltered_urls
        for i, j in combinations(range(1, 23), 2)
    ]
    _unfiltered_urls = {sample: None for sample in _unfiltered_urls}
    _hidden_samples = [sample for sample in _unfiltered_urls if "_1_2." not in sample]
    _filters = AnnDataFilters(
        obs={"total_contacts": {"min": 10}}, var={"total_contacts": {"min": 10}}
    )

    def __init__(self, intersect_vars=False, *args, **kwargs):
        # change default here so that it doesn't intersect between samples.
        kwargs["intersect_vars"] = intersect_vars
        super().__init__(*args, **kwargs)

    def _process_raw_data(self) -> Dict[str, AnnData]:
        # first, split the data into samples based on activation
        # then, split the data into chromosome pairs.

        df = pd.read_csv(
            self.raw_files_paths["regions.bed.gz"],
            compression="gzip",
            delimiter="\t",
            header=None,
        )
        # get the target chromosomes
        chrs = df[0].unique()
        targets = [f"{x}" for x in range(1, 23)]
        chrom_map = {chrom: {} for chrom in chrs}
        for chrom in chrs:
            inds = df.where(df[0] == chrom).dropna()
            chrom_map[chrom]["start"] = inds.index.values.min()
            chrom_map[chrom]["stop"] = inds.index.values.max()
            chrom_map[chrom]["inds"] = inds.index.values
            chrom_map[chrom]["len"] = len(inds)
            chrom_map[chrom]["region"] = (
                chrom
                + "-"
                + inds[1].astype(int).astype(str)
                + "-"
                + inds[2].astype(int).astype(str)
            ).values
        adatas = {}
        for state in ["activated", "naive"]:
            # build the filter criteria for iterating over the mtx files.
            if state == "naive":
                criteria = lambda s: ("activated" not in s) and (".bed.gz" not in s)
            else:
                criteria = lambda s: "activated" in s
            for key in filter(criteria, self.raw_files_paths.keys()):
                sample = key.split(".")[0]
                mtx_path = self.raw_files_paths[key]
                X = mmread(str(mtx_path)).tocsr()
                for row, col in combinations(targets, 2):
                    current_sample = f"{sample}_{row}_{col}"
                    row_map = chrom_map[f"chr{row}"]
                    col_map = chrom_map[f"chr{col}"]
                    ad = AnnData(X[row_map["inds"], :][:, col_map["inds"]])
                    ad.obs_names = row_map["region"]
                    ad.var_names = col_map["region"]
                    adatas[current_sample] = ad
        return adatas


###################################################
#   DNA                                           #
###################################################
###################################################
#                 1000 Genome Phase3              #
###################################################
class Byrska2022(SingleNucleotidePolymorphism):
    """
    Dataset class to obtain 1000 genome phase3 SNP data
    Note: This class is dependent on plink/plink2 (bash) and a python package
    pandas_plink.

        plink and plink2 need to be added to the environment variables

        plink: conda install -c bioconda plink
        plink2: conda install -c bioconda plink2
        pandas_plink: conda install -c conda-forge pandas-plink
    """

    _citation = (
        "@article{byrska2022high,\n"
        "  title={High-coverage whole-genome sequencing of the expanded "
        "1000 Genomes Project cohort including 602 trios},\n"
        "  author={Byrska-Bishop, Marta and Evani, Uday S and Zhao, Xuefang and  "
        "Basile, Anna O and Abel, Haley J and Regier, Allison A and "
        "Corvelo, Andr{'e} and Clarke, Wayne E and Musunuri, Rajeeva and "
        "Nagulapalli, Kshithija and others},\n"
        "  journal={Cell},\n"
        "  volume={185},\n"
        "  number={18},\n"
        "  pages={3426--3440},\n"
        "  year={2022},\n"
        "  publisher={Elsevier},\n"
        "}"
    )

    _raw_urls = {
        "all_hg38.psam": (
            "https://www.dropbox.com/s/2e87z6nc4qexjjm/hg38_corrected.psam?dl=1"
        ),
        "20130606_g1k_3202_samples_ped_population.txt": (
            "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
            "1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt"
        ),
        "deg2_hg38.king.cutoff.out.id": (
            "https://www.dropbox.com/s/4zhmxpk5oclfplp/deg2_hg38.king.cutoff.out.id?dl=1"
        ),
        "all_hg38.pgen.zst": (
            "https://www.dropbox.com/s/j72j6uciq5zuzii/all_hg38.pgen.zst?dl=1"
        ),
        "all_hg38.pvar.zst": (
            "https://www.dropbox.com/s/vx09262b4k1kszy/all_hg38.pvar.zst?dl=1"
        ),
    }

    _unfiltered_urls = {
        None: None #"/banach2/jyc/bipca/data/1000Genome/bipca/datasets/"
        #"SingleNucleotidePolymorphism/Phase3_1000Genome/"
        #"unfiltered/Phase3_1000Genome.h5ad"
    }

    _filters = AnnDataFilters(
        obs={"total_SNPs": {"min": -np.Inf}},
        var={"total_obs": {"min": -np.Inf}},
    )
    _bipca_kwargs = {"variance_estimator": "binomial", "read_counts": 2}

    def _process_raw_data(self) -> AnnData:
        self._run_bash_processing()

        # read the processed files as adata
        (bim, fam, bed) = read_plink(
            str(self.raw_files_directory) + "/all_phase3_pruned", verbose=True
        )

        adata = AnnData(X=bed.compute().transpose())

        # read the metadata and store the metadata in adata
        metadata = pd.read_csv(
            str(self.raw_files_directory)
            + "/20130606_g1k_3202_samples_ped_population.txt",
            sep=" ",
        )
        metadata["iid"] = metadata["SampleID"]
        metadata = fam.merge(metadata, on="iid", how="left")
        adata.obs[["iid"]] = metadata[["iid"]].values
        adata.obs[["Population"]] = metadata[["Population"]].values
        adata.obs.index = adata.obs.index.astype(str)

        return adata

    def _run_bash_processing(self):
        # run plink preprocessing
        subprocess.run(
            ["/bin/bash", "/bipca/python/bipca/experiments/datasets/plink_preprocess.sh"], # TODO: the path need to change to an internal location
            cwd=str(self.raw_files_directory),
        )


###################################################
#   Spatial and imaging                           #
###################################################
###################################################
#   CosMx                                         #
###################################################
class Kluger2023Melanoma(CosMx):
    _citation = "undefined"

    _raw_urls = {
        "31767.h5ad": ("/banach1/jay/bipca_raw_data/cosmx/bipca_split/31767.h5ad"),
        "31778.h5ad": ("/banach1/jay/bipca_raw_data/cosmx/bipca_split/31778.h5ad"),
        "31790.h5ad": ("/banach1/jay/bipca_raw_data/cosmx/bipca_split/31790.h5ad"),
    }

    _unfiltered_urls = {"31767.h5ad": None, "31778.h5ad": None, "31790.h5ad": None}

    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 100}}, var={"total_obs": {"min": 100}}
    )

    def _process_raw_data(self) -> Dict[str, AnnData]:
        adata = {
            k.rstrip(".h5ad"): read_h5ad(str(v))
            for k, v in self.raw_files_paths.items()
        }

        return adata


class FrontalCortex6k(CosMx):
    _citation = (
        "@misc{FrontalCortex6k,\n"
        "   author={NanoString Technologies},\n"
        "   title={CosMx Human Frontal Cortex FFPE Dataset"
        "},\n"
        "   howpublished="
        "Available at \\url{https://nanostring.com/products/"
        "cosmx-spatial-molecular-imager/ffpe-dataset/"
        "human-frontal-cortex-ffpe-dataset/},\n"
        "   year={2023},\n"
        "   month={September},\n"
        '   note = "[Online; accessed 30-July-2024]"\n'
        "}"
    )

    # count data and metadata are converted from the seurat object SeuratObj_withTranscripts.RDS in R 
    # count_data <- obj.seurat@assays$RNA@counts
    # Matrix::writeMM(count_data,file = "/banach2/jyc/data/cosmx6k/count_data.txt")
    # write.csv(obj.seurat@meta.data,file = "/banach2/jyc/data/cosmx6k/metadata.csv")
    # write.csv(rownames(count_data),file = "/banach2/jyc/data/cosmx6k/genes.csv")

    _raw_urls = {
        "count_data.txt": ("/banach2/jyc/data/cosmx6k/count_data.txt"),
        "metadata.csv": ("/banach2/jyc/data/cosmx6k/metadata.csv"),
        "genes.csv": ("/banach2/jyc/data/cosmx6k/genes.csv"),
    }

    _unfiltered_urls = {None: None}

    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 100}}, var={"total_obs": {"min": 1}}
    )

    def _process_raw_data(self) -> Dict[str, AnnData]:
        count_data = mmread((self.raw_files_directory / "count_data.txt").resolve())
        metadata = pd.read_csv((self.raw_files_directory / "metadata.csv").resolve(),index_col=0)
        gene_names = pd.read_csv((self.raw_files_directory / "genes.csv").resolve(),index_col=0)
        adata = sc.AnnData(X=count_data.tocsr().T,
                           obs=metadata,
                           var=gene_names)
        adata.var_names = gene_names['x']
        # take the fov with the most cells
        fov2keep = 63
        adata = adata[[fov == fov2keep for fov in adata.obs['fov']],:].copy()

        return adata


###################################################
#   DBiT-Seq                                      #
###################################################
class Liu2020(DBiTSeq):
    _citation = DBiTSeq.technology_citation
    _raw_urls = {
        "dbit-Seq.tsv.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/"
            "GSM4096nnn/GSM4096261/suppl/GSM4096261_10t.tsv.gz"
        )
    }
    _unfiltered_urls = {None: None}

    def _process_raw_data(self) -> AnnData:
        # get the raw data file name
        path = next(iter(self.raw_files_paths.values()))
        filename = path.stem
        output_filename = self.raw_files_directory / filename
        # read the raw data
        # unzip the file in context
        with gzip.open(path, "rb") as f:
            with open(output_filename, "wb") as f_out:
                copyfileobj(f, f_out)
        adata = read_csv_pyarrow_bad_colnames(
            output_filename, delimiter="\t", index_col=0
        )
        # remove the extracted tsv
        output_filename.unlink()
        obs_names = adata.index.values
        var_names = adata.columns
        adata = AnnData(
            csr_matrix(adata.values, dtype=int),
            dtype=int,
        )
        adata.obs_names = obs_names
        adata.var_names = var_names
        return adata

###################################################
#   Calcium Imaging                               #
###################################################
class Neurofinder2016(CalciumImaging):
    _citation = (
        "@misc{neurofinder2016,\n"
        "   title={neurofinder: benchmarking challenge for finding neurons in calcium "
        "imaging data}, \n"
        "   author={Peron, Simon and Sofroniew, Nicholas and Svoboda, Karel and "
        "Packer, Adam and Russell, Lloyd and Häusser, Michael and Zaremba, Jeff and "
        "Kaifosh, Patrick and Losonczy, Attila and Chettih, Selmaan and "
        "Minderer, Matthias and Harvey, Chris and Rebo, Maxwell and "
        "Conlen, Matthew and Freeman, Jeffrey}, \n"
        "   howpublished="
        '"Available at \\url{https://github.com/codeneuro/neurofinder}",\n'
        "   year={2016},\n"
        "   month={March},\n"
        '   note = "[Online; accessed 02-January-2024]"\n'
        "}"
    )
    _sample_ids = ["02.00"]
    _raw_urls = {
        f"{sample}.zip": (
            "https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/"
            f"neurofinder.{sample}.zip"
        ) for sample in _sample_ids
    }
    _unfiltered_urls = {f"{sample}.h5ad": None for sample in _sample_ids}
    
    @classmethod 
    def _annotate(cls, adata:AnnData) -> AnnDataAnnotations:
        annotations = AnnDataAnnotations.from_other(adata)
        return annotations
    def _process_raw_data(self) -> AnnData:
        adata = {}
        images = []
        for key, pth in self.raw_files_paths.items():
            dataset = '.'.join(key.split('.')[:2])
            with zipfile.ZipFile(pth,'r') as archive:
                for file in archive.infolist():
                    filename = file.filename
                    if filename.endswith('.tiff'): # open images,
                    # store them with their frame number
                        code = filename.replace('.tiff','')[-5:]
                        with archive.open(file) as f:
                            img = imread(f)
                        images.append((code,img))
                    if filename.endswith('regions.json'):
                        # if it's a training dataset, it has a regions annotation
                        # these encode interesting neurons
                        with archive.open(file) as f:
                            regions = json.load(f)
            images = np.asarray([ele[1] for ele in sorted(images, key=lambda x: x[0])])
            nframes,dims = images.shape[0],images.shape[1:]
            images = images.reshape(images.shape[0], -1).astype(np.int16)
            #extract coordinates of neurons into sparse array
            rows,cols = tuple(np.r_[tuple(
                np.asarray(ele['coordinates']) 
                for ele in regions)].T)
            vals = np.r_[tuple(
                np.full((len(ele['coordinates']),),ix+1) 
                for ix,ele in enumerate(regions))]
            neuron_coordinates = csr_matrix((vals,(rows,cols)),
                                shape=(dims)).toarray().reshape(-1,).astype(np.float32)
            neuron_coordinates[neuron_coordinates==0] = np.nan
            adata[dataset] = AnnData(images)
            adata[dataset].var['neuron_id'] = neuron_coordinates
        
        return adata
###################################################
#   SeqFISH+                                      #
###################################################
class Eng2019(SeqFISHPlus):
    _citation = SeqFISHPlus.technology_citation
    _raw_urls = {
        "raw.zip": "https://github.com/CaiGroup/seqFISH-PLUS/raw/master/sourcedata.zip"
    }
    _unfiltered_urls = {"subventricular_zone.h5ad": None, "olfactory_bulb.h5ad": None}

    def __init__(self, intersect_vars=False, *args, **kwargs):
        # change default here so that it doesn't intersect between samples.
        kwargs["intersect_vars"] = intersect_vars
        super().__init__(*args, **kwargs)

    def _process_raw_data(self) -> AnnData:
        sources = {
            "subventricular_zone": "cortex_svz_counts.csv",
            "olfactory_bulb": "ob_counts.csv",
        }
        with zipfile.ZipFile(self.raw_files_directory / "raw.zip", "r") as zip_ref:
            for member in sources.values():
                target = self.raw_files_directory / member
                member = f"sourcedata/{member}"
                with zip_ref.open(member) as source, open(target, "wb") as target:
                    copyfileobj(source, target)
        adata = {}
        for name, path in sources.items():
            pth = self.raw_files_directory / path
            df = read_csv_pyarrow_bad_colnames(pth, delimiter=",", index_col=None)
            var_names = df.columns
            adata[name] = AnnData(csr_matrix(df.values, dtype=int), dtype=int)
            adata[name].var_names = var_names
            pth.unlink()
        return adata


###################################################
#   Spatial Transcriptomics                       #
###################################################
class Asp2019(SpatialTranscriptomicsV1):
    _citation = (
        "@article{asp2019spatiotemporal,\n"
        "  title={A spatiotemporal organ-wide gene expression and cell atlas of the "
        "developing human heart},\n"
        "  author={Asp, Michaela and Giacomello, Stefania and Larsson, Ludvig and "
        'Wu, Chenglin and F{"u}rth, Daniel and Qian, Xiaoyan and W{"a}rdell, Eva '
        "and Custodio, Joaquin and Reimeg{\aa}rd, Johan and Salm{'e}n, Fredrik and "
        "others},\n"
        "  journal={Cell},\n"
        "  volume={179},\n"
        "  number={7},\n"
        "  pages={1647--1660},\n"
        "  year={2019},\n"
        "  publisher={Elsevier}\n}"
        "}"
    )
    _raw_urls = {
        "raw.zip": (
            "https://data.mendeley.com/public-files/datasets/mbvhhf8m62/files/"
            "f76ec6ad-addd-41c3-9eec-56e31ddbac71/file_downloaded"
        )
    }
    _unfiltered_urls = {None: None}

    def _process_raw_data(self) -> AnnData:
        metadata_output = self.raw_files_directory / "meta_data.tsv"
        counts_output = self.raw_files_directory / "counts.tsv"

        with zipfile.ZipFile(self.raw_files_directory / "raw.zip", "r") as zip_ref:
            with gzip.open(
                zip_ref.open("filtered_ST_matrix_and_meta_data/meta_data.tsv.gz")
            ) as meta_data_ref:
                with open(metadata_output, "wb") as f_out:
                    copyfileobj(meta_data_ref, f_out)
            with gzip.open(
                zip_ref.open("filtered_ST_matrix_and_meta_data/filtered_matrix.tsv.gz")
            ) as counts_ref:
                with open(counts_output, "wb") as f_out:
                    copyfileobj(counts_ref, f_out)
        adata = read_csv_pyarrow_bad_colnames(
            counts_output, delimiter="\t", index_col=0
        )
        meta = read_csv_pyarrow_bad_colnames(
            metadata_output, delimiter="\t", index_col=0
        )

        var_names = adata.index.values
        obs_names = adata.columns
        adata = AnnData(csr_matrix(adata.values.T, dtype=int), dtype=int)
        adata.obs_names = obs_names
        adata.var_names = var_names
        adata.obs = adata.obs.join(meta)
        counts_output.unlink()
        metadata_output.unlink()
        return adata


class Thrane2018(SpatialTranscriptomicsV1):
    _citation = (
        "@article{thrane2018spatially,\n"
        "  title={Spatially resolved transcriptomics enables dissection of genetic "
        "heterogeneity in stage III cutaneous malignant melanoma},\n"
        "  author={Thrane, Kim and Eriksson, Hanna and Maaskola, Jonas and Hansson, "
        "Johan and Lundeberg, Joakim},\n"
        "  journal={Cancer research},\n"
        "  volume={78},\n"
        "  number={20},\n"
        "  pages={5970--5979},\n"
        "  year={2018},\n"
        "  publisher={AACR}\n"
        "}"
    )

    _raw_urls = {
        "raw.zip": (
            "https://9b0ce2.p3cdn1.secureserver.net/wp-content/uploads/2019/03/"
            "ST-Melanoma-Datasets_1.zip"
        )
    }

    _unfiltered_urls = {
        "mel1_1.h5ad": None,
        "mel1_2.h5ad": None,
        "mel2_1.h5ad": None,
        "mel2_2.h5ad": None,
        "mel3_1.h5ad": None,
        "mel3_2.h5ad": None,
        "mel4_1.h5ad": None,
        "mel4_2.h5ad": None,
    }

    def __init__(self, intersect_vars=False, *args, **kwargs):
        # change default here so that it doesn't intersect between samples.
        kwargs["intersect_vars"] = intersect_vars
        super().__init__(*args, **kwargs)

    def _process_raw_data(self) -> AnnData:
        extracted_raw_files = [
            "ST_mel1_rep1_counts.tsv",
            "ST_mel1_rep2_counts.tsv",
            "ST_mel2_rep1_counts.tsv",
            "ST_mel2_rep2_counts.tsv",
            "ST_mel3_rep1_counts.tsv",
            "ST_mel3_rep2_counts.tsv",
            "ST_mel4_rep1_counts.tsv",
            "ST_mel4_rep2_counts.tsv",
        ]
        extracted_raw_files = {
            (pth := self.raw_files_directory / f).name: pth for f in extracted_raw_files
        }
        with zipfile.ZipFile(self.raw_files_directory / "raw.zip", "r") as zip_ref:
            for name, path in extracted_raw_files.items():
                with zip_ref.open(f"{name}") as source:
                    with open(path, "wb") as target:
                        copyfileobj(source, target)
        adata = {}
        for pth in extracted_raw_files.values():
            key = (
                pth.stem.replace("ST_mel", "mel")
                .replace("_counts", "")
                .replace("_rep", "_")
            )
            val = read_csv_pyarrow_bad_colnames(pth, delimiter="\t")
            obs_names = val.columns
            var_names = val.index
            adata[key] = AnnData(csr_matrix(val.values, dtype=int).T, dtype=int)
            adata[key].obs_names = obs_names
            adata[key].var_names = var_names
        return adata


###################################################
#   Visium                                        #
###################################################
class Maynard2021(TenXVisium):
    _citation = (
        "@article{maynard2021transcriptome,\n"
        "   title={Transcriptome-scale spatial gene expression in the human "
        "dorsolateral prefrontal cortex},\n"
        "   author={Maynard, Kristen R and Collado-Torres, Leonardo and Weber, "
        "Lukas M and Uytingco, Cedric and Barry, Brianna K and Williams, "
        "Stephen R and Catallini, Joseph L and Tran, Matthew N and Besich, "
        "Zachary and Tippani, Madhavi and others},\n"
        "   journal={Nature neuroscience},\n"
        "   volume={24},\n"
        "   number={3},\n"
        "   pages={425--436},\n"
        "   year={2021},\n"
        "   publisher={Nature Publishing Group US New York}\n"
        "}\n"
    )
    _sample_ids = [
        151507,
        151508,
        151509,
        151510,
        151669,
        151670,
        151671,
        151672,
        151673,
        151674,
        151675,
        151676,
    ]
    _raw_urls = {
        f"{key}.h5": (
            f"https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/{key}"
            "_filtered_feature_bc_matrix.h5"
        )
        for key in _sample_ids
    }
    _unfiltered_urls = {f"{sample}.h5ad": None for sample in _sample_ids}

    def __init__(self, intersect_vars=False, *args, **kwargs):
        # change default here so that it doesn't intersect between samples.
        kwargs["intersect_vars"] = intersect_vars
        super().__init__(*args, **kwargs)

    def _process_raw_data(self) -> Dict[str, AnnData]:
        adata = {
            path.stem: sc.read_10x_h5(str(path))
            for path in self.raw_files_paths.values()
        }
        for value in adata.values():
            value.X = csr_matrix(value.X, dtype=int)
            value.var_names_make_unique()
            value.obs_names_make_unique()
        return adata


class TenX2020HumanBreastCancer(TenXVisium):
    _citation = (
        "@misc{10x2020humanbreastcancer,\n"
        "  author = {10x Genomics},\n"
        "  title = {{Human Breast Cancer} (Block A Sections 1 and 2)},\n"
        '  howpublished = "Available at \\url{https://www.10xgenomics.com/'
        "resources/datasets/human-breast-cancer-block-a-section-1-1-"
        "standard-1-1-0} and \\url{https://www.10xgenomics.com/resources/"
        'datasets/human-breast-cancer-block-a-section-2-1-standard-1-1-0}",\n'
        "  year = {2020},\n"
        "  month = {June},\n"
        '  note = "[Online; accessed 17-April-2023]"\n'
        "}"
    )
    _raw_urls = {
        "section1.h5": (
            "https://cf.10xgenomics.com/samples/spatial-exp/"
            "1.1.0/V1_Breast_Cancer_Block_A_Section_1/"
            "V1_Breast_Cancer_Block_A_Section_1_"
            "filtered_feature_bc_matrix.h5"
        ),
        "section2.h5": (
            "https://cf.10xgenomics.com/samples/spatial-exp/"
            "1.1.0/V1_Breast_Cancer_Block_A_Section_2/"
            "V1_Breast_Cancer_Block_A_Section_2_"
            "filtered_feature_bc_matrix.h5"
        ),
    }
    _unfiltered_urls = {None: None}

    def _process_raw_data(self) -> AnnData:
        return self._process_raw_data_10X()


class TenX2020HumanHeart(TenXVisium):
    _citation = (
        "@misc{10x2020humanheart,\n"
        "  author = {10x Genomics},\n"
        "  title = {{Human Heart}},\n"
        "  howpublished = Available at \\url{https://www.10xgenomics.com/"
        "resources/datasets/human-heart-1-standard-1-1-0},\n"
        "  year = {2020},\n"
        "  month = {June},\n"
        "  note = {[Online; accessed 17-April-2023]}\n"
        "}"
    )

    _raw_urls = {
        "raw.h5": (
            "https://cf.10xgenomics.com/samples/spatial-exp/"
            "1.1.0/V1_Human_Heart/V1_Human_Heart_filtered_feature_bc_matrix.h5"
        )
    }
    _unfiltered_urls = {None: None}

    def _process_raw_data(self) -> AnnData:
        return self._process_raw_data_10X()


class TenX2020HumanLymphNode(TenXVisium):
    _citation = (
        "@misc{10x2020humanlymphnode,\n"
        "  author = {10x Genomics},\n"
        "  title = {{Human Lymph Node}},\n"
        "  howpublished = Available at \\url{https://www.10xgenomics.com/"
        "resources/datasets/human-lymph-node-1-standard-1-1-0},\n"
        "  year = {2020},\n"
        "  month = {June},\n"
        "  note = {[Online; accessed 17-April-2023]}\n"
        "}"
    )

    _raw_urls = {
        "raw.h5": (
            "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/"
            "V1_Human_Lymph_Node/V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5"
        )
    }
    _unfiltered_urls = {None: None}

    def _process_raw_data(self) -> AnnData:
        return self._process_raw_data_10X()


class TenX2022MouseBrain(TenXVisium):
    _citation = (
        "@misc{10x2022mousebrain,\n"
        "  author = {10x Genomics},\n"
        "  title = {Aggregate of Mouse Brain Sections: Visium Fresh Frozen, "
        "Whole Transcriptome},\n"
        "  howpublished = Available at \\url{https://www.10xgenomics.com/resources/"
        "datasets/aggregate-of-mouse-brain-sections-visium-fresh-frozen-whole-"
        "transcriptome-1-standard},\n"
        "  year = {2022},\n"
        "  month = {July},\n"
        "  note = {[Online; accessed 17-April-2023]}\n"
        "}"
    )

    _raw_urls = {
        "raw.h5": (
            "https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/"
            "Aggregate_Visium_Mouse_Brain_Sagittal/"
            "Aggregate_Visium_Mouse_Brain_Sagittal_filtered_feature_bc_matrix.h5"
        )
    }
    _unfiltered_urls = {None: None}

    def _process_raw_data(self) -> AnnData:
        return self._process_raw_data_10X()


###################################################
#   Single Cell Data                              #
###################################################
###################################################
#   ATAC                                          #
###################################################
class Buenrostro2018(Buenrostro2015Protocol):
    _citation = (
        "@article{buenrostro2018integrated,\n"
        "  title={Integrated single-cell analysis maps the continuous regulatory "
        "landscape of human hematopoietic differentiation},\n"
        "  author={Buenrostro, Jason D and Corces, M Ryan and Lareau, Caleb A and "
        "Wu, Beijing and Schep, Alicia N and Aryee, Martin J and Majeti, Ravindra and "
        "Chang, Howard Y and Greenleaf, William J},\n"
        "  journal={Cell},\n"
        "  volume={173},\n"
        "  number={6},\n"
        "  pages={1535--1548},\n"
        "  year={2018},\n"
        "  publisher={Elsevier}\n"
        "}"
    )
    _raw_urls = {
        "raw.zip": (
            "/banach1/jay/bipca_raw_data/buenrostro2018/raw.zip"
            # the old link was broken by cell.
            # "https://www.cell.com/cms/10.1016/j.cell.2018.03.074/"
            # "attachment/2a72a316-33cc-427d-8019-dfc83bd220ca/mmc4.zip"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_sites": {"min": 1000}},  # these are from the episcanpy tutorial.
        var={"total_cells": {"min": 100}}, #50
    )

    def _process_raw_data(self) -> AnnData:
        with tarfile.open(self.raw_files_directory / "raw.zip") as f:
            f.extract(
                "supplementary_code/input/single_cell.mat", self.raw_files_directory
            )

        # mv the scATACseq data

        zipdir = (self.raw_files_directory / "supplementary_code").resolve()
        data_path = (zipdir / "input" / "single_cell.mat").resolve()
        new_data_path = (self.raw_files_directory / "raw.mat").resolve()
        mv(data_path, new_data_path)
        # rm the unzipped directory
        rmtree(zipdir)

        # load the data
        mat = loadmat(str(new_data_path))
        mat = mat["scDat"][0][0]
        X = mat[0].tocsr()
        cellnames = mat[2]
        celltypes = mat[3]

        adata = AnnData(X=X, dtype=int)
        adata.obs_names = [ele[0][0] for ele in cellnames]
        adata.obs["cell_type"] = [ele[0] for ele in celltypes.flatten()]
        adata.obs["facs_label"] = [
            "MEP"
            if "MEP" in line
            else line.split(".bam")[0].lstrip("singles-").split("BM")[-1].split("-")[1]
            for line in adata.obs_names.tolist()
        ]
        return adata


class TenX2019PBMCATAC(TenXChromiumATACV1):
    _citation = (
        "@misc{10x2019pbmcatac,\n"
        "   author={10X Genomics},\n"
        "   title={10k Peripheral Blood Mononuclear Cells (PBMCs) from a Healthy Donor"
        "},\n"
        "   howpublished="
        '"Available at \\url{https://www.10xgenomics.com/resources/datasets/'
        "10-k-peripheral-blood-mononuclear-cells-pbm-cs-from-a-healthy-donor-"
        '1-standard-1-2-0}",\n'
        "   year={2019},\n"
        "   month={November},\n"
        '   note = "[Online; accessed 17-April-2023]"\n'
        "}"
    )
    _raw_urls = {
        "raw.h5": (
            "https://cf.10xgenomics.com/samples/cell-atac/1.2.0/"
            "atac_v1_pbmc_10k/atac_v1_pbmc_10k_filtered_peak_bc_matrix.h5"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_sites": {"min": 1000}},  # these are from the episcanpy tutorial.
        var={"total_cells": {"min": 100}}, #50
    )

    def _process_raw_data(self) -> AnnData:
        adata = {
            path.stem: sc.read_10x_h5(str(path), gex_only=False)
            for path in self.raw_files_paths.values()
        }
        for value in adata.values():
            value.X = csr_matrix(value.X, dtype=int)
        return next(iter(adata.values()))


class TenX2019MouseBrainATAC(TenXChromiumATACV1):
    _citation = (
        "@misc{10x2019mousebrainatac,\n"
        "   author={10X Genomics},\n"
        "   title={Fresh Cortex, Hippocampus, and Ventricular Zone from "
        "Embryonic Mouse Brain (E18)},\n"
        "   howpublished="
        '"Available at \\url{https://www.10xgenomics.com/resources/datasets/'
        "fresh-cortex-hippocampus-and-ventricular-zone-from-embryonic-mouse-brain-e-18-"
        '1-standard-1-2-0}",\n'
        "   year={2019},\n"
        "   month={November},\n"
        '   note = "[Online; accessed 17-April-2023]"\n'
        "}"
    )
    _raw_urls = {
        "raw.h5": (
            "https://cf.10xgenomics.com/samples/cell-atac/1.2.0/"
            "atac_v1_E18_brain_fresh_5k/"
            "atac_v1_E18_brain_fresh_5k_filtered_peak_bc_matrix.h5"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_sites": {"min": 1000}},  # these are from the episcanpy tutorial.
        var={"total_cells": {"min": 100}}, #50
    )

    def _process_raw_data(self) -> AnnData:
        adata = {
            path.stem: sc.read_10x_h5(str(path), gex_only=False)
            for path in self.raw_files_paths.values()
        }
        for value in adata.values():
            value.X = csr_matrix(value.X, dtype=int)
        return next(iter(adata.values()))


class TenX2022MouseCortexATAC(TenXChromiumATACV1_1):
    _citation = (
        "@misc{10x2022mousecortexatac,\n"
        "   author={10X Genomics},\n"
        "   title={8k Adult Mouse Cortex Cells, ATAC v1.1, Chromium X},\n"
        "   howpublished="
        '"Available at \\url{https://www.10xgenomics.com/resources/datasets/'
        '8k-adult-mouse-cortex-cells-atac-v1-1-chromium-x-1-1-standard}",\n'
        "   year={2022},\n"
        "   month={March},\n"
        '   note = "[Online; accessed 17-April-2023]"\n'
        "}"
    )
    _raw_urls = {
        "raw.h5": (
            "https://cf.10xgenomics.com/samples/cell-atac/2.1.0/"
            "8k_mouse_cortex_ATACv1p1_nextgem_Chromium_X/8k_mouse_cortex_"
            "ATACv1p1_nextgem_Chromium_X_filtered_peak_bc_matrix.h5"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_sites": {"min": 1000}},  # these are from the episcanpy tutorial.
        var={"total_cells": {"min": 100}}, #50
    )

    def _process_raw_data(self) -> AnnData:
        adata = {
            path.stem: sc.read_10x_h5(str(path), gex_only=False)
            for path in self.raw_files_paths.values()
        }
        for value in adata.values():
            value.X = csr_matrix(value.X, dtype=int)
        return next(iter(adata.values()))


class TenX2022PBMCATAC(TenXChromiumATACV1_1):
    _citation = (
        "@misc{10x2022pbmcatac,\n"
        "   author={10X Genomics},\n"
        "   title={10k Human PBMCs, ATAC v1.1, Chromium X},\n"
        "   howpublished="
        '"Available at \\url{https://www.10xgenomics.com/resources/datasets/'
        '10k-human-pbmcs-atac-v1-1-chromium-x-1-1-standard}",\n'
        "   year={2022},\n"
        "   month={March},\n"
        '   note = "[Online; accessed 17-April-2023]"\n'
        "}"
    )
    _raw_urls = {
        "raw.h5": (
            "https://cf.10xgenomics.com/samples/cell-atac/2.1.0/"
            "10k_pbmc_ATACv1p1_nextgem_Chromium_X/10k_pbmc_ATACv1p1_nextgem"
            "_Chromium_X_filtered_peak_bc_matrix.h5"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_sites": {"min": 1000}},  # these are from the episcanpy tutorial.
        var={"total_cells": {"min": 100}}, #50
    )

    def _process_raw_data(self) -> AnnData:
        adata = {
            path.stem: sc.read_10x_h5(str(path), gex_only=False)
            for path in self.raw_files_paths.values()
        }
        for value in adata.values():
            value.X = csr_matrix(value.X, dtype=int)
        return next(iter(adata.values()))


###################################################
#   RNA                                           #
###################################################
###################################################
#   SmartSeq                                      #
###################################################
class HagemannJensen2020(SmartSeqV3):
    _citation = (
        "@article{hagemann2020single,\n"
        "title={Single-cell RNA counting at allele and isoform resolution using Smart-seq3},\n"
        "author={Hagemann-Jensen, Michael and Ziegenhain, Christoph and Chen, "
        'Ping and Ramsk{"o}ld, Daniel and Hendriks, Gert-Jan and Larsson, '
        "Anton JM and Faridani, Omid R and Sandberg, Rickard},\n"
        "journal={Nature Biotechnology},\n"
        "volume={38},\n"
        "number={6},\n"
        "pages={708--714},\n"
        "year={2020},\n"
        "publisher={Nature Publishing Group US New York}\n"
        "}"
    )

    _raw_urls = {
        "UMIs.txt": (
            "https://www.ebi.ac.uk/biostudies/files/E-MTAB-8735/"
            "HCA.UMIcounts.PBMC.txt"
        ),
        "reads.txt": (
            "https://www.ebi.ac.uk/biostudies/files/E-MTAB-8735/"
            "HCA.readcounts.PBMC.txt"
        ),
        "annotations.txt": (
            "https://www.ebi.ac.uk/biostudies/files/E-MTAB-8735/"
            "Smart3.PBMC.annotated.txt"
        ),
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={
            "pct_mapped_reads": {"min": 0.75},
            "pct_MT_reads": {"min": -np.Inf},  # 0.15 for the other dataset
            "total_reads": {"min": 1e5},
            "pct_MT_UMIs": {"min": -np.Inf},  # get rid of this extra UMI filter.
            "total_genes": {"min": 500},  # 500
            "isHEK": {"max": 0},  # Remove HEK cells
        },
        var={"total_cells": {"min": 250}}, #250 # 10 for the other dataset
    )

    def __init__(self, n_filter_iters=1, *args, **kwargs):
        # change default here so that it doesn't filter twice.
        kwargs["n_filter_iters"] = n_filter_iters
        super().__init__(*args, **kwargs)

    def _process_raw_data(self) -> AnnData:
        # first load the umis, read counts, and the barcode annotations.
        base_files = [v for k, v in self.raw_files_paths.items()]

        data = {
            fname: df
            for fname, df in map(
                lambda f: (
                    f.stem,
                    read_csv_pyarrow_bad_colnames(
                        f, delimiter="\t", logger=self.logger
                    ),
                ),
                base_files,
            )
        }

        # process the matrix files into the adata.
        adata = AnnData(csr_matrix(data["UMIs"].values.T, dtype=int), dtype=int)
        adata.obs_names = data["UMIs"].columns.values
        adata.var_names = data["UMIs"].index
        del data["UMIs"]
        adata.layers["reads"] = csr_matrix(data["reads"].values.T)
        adata.obs["total_reads"] = np.asarray(adata.layers["reads"].sum(1)).squeeze()
        del data["reads"]

        for col in data["annotations"].columns:
            if data["annotations"][col].dtype == "object":
                data["annotations"][col] = data["annotations"][col].astype("category")

        data["annotations"] = data["annotations"].rename(
            columns={"clusterName": "cell_types"}
        )
        adata.obs = pd.concat([adata.obs, data["annotations"]], axis=1)
        adata.obs["isHEK"] = (adata.obs["cell_types"] == "HEK") | (
            adata.obs["cell_types"] == "HEK cells"
        )
        #gene_dict = get_ensembl_mappings(adata.var_names.tolist(), logger=self.logger) # comment out because it throws an error
        #var_df = pd.DataFrame.from_dict(gene_dict, orient="index")
        #var_df["gene_biotype"] = var_df["gene_biotype"].astype("category")
        #adata.var = var_df
        return adata


class HagemannJensen2022(SmartSeqV3xpress):
    _citation = (
        "@article{hagemannjensen2022,\n"
        "  title={Scalable single-cell RNA sequencing from full transcripts "
        "with Smart-seq3xpress},\n"
        "  author={Hagemann-Jensen, Michael and Ziegenhain, Christoph and "
        "Sandberg, Rickard},\n"
        "  journal={Nature Biotechnology},\n"
        "  volume={40},\n"
        "  number={10},\n"
        "  pages={1452--1457},\n"
        "  year={2022},\n"
        "  publisher={Nature Publishing Group US New York}\n"
        "}"
    )

    _raw_urls = {
        "raw.zip": "/banach1/jay/bipca_raw_data/hagemannjensen2022/raw.zip"
        # obtained from a redirect of
        # ("https://www.ebi.ac.uk/biostudies/files/E-MTAB-11452/zip")
        # the original files are here:
        # "UMIs.txt": (
        #     "https://www.ebi.ac.uk/biostudies/files/E-MTAB-11452/"
        #     "PBMCs.allruns.umicounts_intronexon.txt"
        # ),
        # "reads.txt": (
        #     "https://www.ebi.ac.uk/biostudies/files/E-MTAB-11452/"
        #     "PBMCs.allruns.readcounts_intronexon.txt"
        # ),
        # "annotations.txt": (
        #     "https://www.ebi.ac.uk/biostudies/files/E-MTAB-11452/"
        #     "PBMCs.allruns.barcode_annotation.txt"
        # ),
    }
    _sample_ids = [
        'donor1', 'donor2', 'donor3', 'donor4', 'donor6', 'donor7','donor8'
    ]
    _unfiltered_urls = {f"{sample}.h5ad": None for sample in _sample_ids}
    #_unfiltered_urls["full.h5ad"] = None
    _filters = AnnDataFilters(
        obs={
            "pct_mapped_reads": {"min": 0.5},
            "pct_MT_reads": {"max": 0.15},
            "total_reads": {"min": 2e4},
            "passed_qc": {
                "min": 1
            },  # passed_qc is a boolean annotation for this dataset.
            "pct_MT_UMIs": {"min": -np.Inf},  # get rid of this extra UMI filter.
            "total_genes": {"min": 500}, #500
        },
        var={"total_cells": {"min": 50}}, #10 is too sparse
    )

    def __init__(self, n_filter_iters=1, intersect_vars = False, *args, **kwargs):
        # change default here so that it doesn't filter twice.
        kwargs["n_filter_iters"] = n_filter_iters
        # not to keep the same genes
        kwargs["intersect_vars"] = intersect_vars
        super().__init__(*args, **kwargs)

    def _process_raw_data(self) -> AnnData:
        # first load the umis, read counts, and the barcode annotations.
        base_files = {
            "annotations.txt": "PBMCs.allruns.barcode_annotation.txt",
            "reads.txt": "PBMCs.allruns.readcounts_intronexon.txt",
            "UMIs.txt": "PBMCs.allruns.umicounts_intronexon.txt",
        }
        base_files = {self.raw_files_directory / k: v for k, v in base_files.items()}
        with zipfile.ZipFile(self.raw_files_directory / "raw.zip", "r") as zip_ref:
            for target, member in base_files.items():
                with zip_ref.open(member) as source, open(target, "wb") as target:
                    copyfileobj(source, target)

        data = {
            fname: df
            for fname, df in map(
                lambda f: (
                    Path(f).stem,
                    read_csv_pyarrow_bad_colnames(
                        f, delimiter="\t", logger=self.logger
                    ),
                ),
                base_files.keys(),
            )
        }

        # process the matrix files into the adata.
        adata = AnnData(csr_matrix(data["UMIs"].values.T, dtype=int), dtype=int)
        adata.obs_names = data["UMIs"].columns.values
        adata.var_names = data["UMIs"].index
        del data["UMIs"]
        adata.layers["reads"] = csr_matrix(data["reads"].values.T)
        del data["reads"]

        for col in data["annotations"].columns:
            if data["annotations"][col].dtype in ["object", "category"]:
                data["annotations"][col] = data["annotations"][col].astype(str)

        data["annotations"] = data["annotations"].rename(
            columns={"nReads": "total_reads", "QC_status": "passed_qc"}
        )
        data["annotations"]["passed_qc"] = data["annotations"]["passed_qc"] == "QCpass"
        adata.obs = pd.concat([adata.obs, data["annotations"]], axis=1)
        #gene_dict = get_ensembl_mappings(adata.var_names.tolist(), logger=self.logger) # comment out because it throws an error
        #var_df = pd.DataFrame.from_dict(gene_dict, orient="index")

        #adata.var = var_df
        #for c in adata.var.columns:
        #    if adata.var[c].dtype in ["object", "category"]:
        #        adata.var[c] = adata.var[c].astype(str)
        #for c in adata.obs.columns:
        #    if adata.obs[c].dtype in ["object", "category"]:
        #        adata.obs[c] = adata.obs[c].astype(str)

        # for each sample
        adata = {
            bid: adata[adata.obs["donor"] == bid,:]
            for bid in self._sample_ids
        }
        return adata


###################################################
#   Chromium                                      #
###################################################
class TenX2016PBMC(TenXChromiumRNAV1):
    _citation = (
        "@misc{10x2016pbmc,\n"
        "   author={10X Genomics},\n"
        "   title={33k PBMCs from a Healthy Donor},\n"
        "   howpublished="
        '"Available at \\url{https://www.10xgenomics.com/resources/datasets/'
        '33-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0}",\n'
        "   year={2016},\n"
        "   month={September},\n"
        '   note = "[Online; accessed 17-April-2023]"\n'
        "}"
    )
    _raw_urls = {
        "pbmc33k.tar.gz": (
            "https://cf.10xgenomics.com/samples/cell-exp/"
            "1.1.0/pbmc33k/pbmc33k_filtered_gene_bc_matrices.tar.gz"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.1}},
        var={"total_cells": {"min": 100}},
    )

    def _process_raw_data(self) -> AnnData:
        targz = next(iter(self.raw_files_paths.values()))
        with self.logger.log_task(f"extracting {targz.name}"):
            tarfile.open(str(targz)).extractall(self.raw_files_directory)
        matrix_dir = self.raw_files_directory / "filtered_gene_bc_matrices" / "hg19"
        with self.logger.log_task(f"reading {matrix_dir}"):
            adata = sc.read_10x_mtx(matrix_dir)
        rmtree((self.raw_files_directory / "filtered_gene_bc_matrices").resolve())
        return adata


class TenX2017MouseBrain(TenXChromiumRNAV2):
    _citation = (
        "@misc{10x2017mousebrain,\n"
        "   author={10X Genomics},\n"
        "   title={9k Brain Cells from an E18 Mouse}, \n"
        "   howpublished="
        '"Available at \\url{https://www.10xgenomics.com/resources/datasets/'
        '9-k-brain-cells-from-an-e-18-mouse-2-standard-1-3-0}",\n'
        "   year={2017},\n"
        "   month={February},\n"
        '   note = "[Online; accessed 17-April-2023]"\n'
        "}"
    )
    _raw_urls = {
        "mouse9k.tar.gz": (
            "https://cf.10xgenomics.com/samples/cell-exp/1.3.0/"
            "neuron_9k/neuron_9k_filtered_gene_bc_matrices.tar.gz"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.1}},
        var={"total_cells": {"min": 100}},
    )

    def _process_raw_data(self) -> AnnData:
        targz = next(iter(self.raw_files_paths.values()))
        with self.logger.log_task(f"extracting {targz.name}"):
            tarfile.open(str(targz)).extractall(self.raw_files_directory)
        matrix_dir = self.raw_files_directory / "filtered_gene_bc_matrices" / "mm10"
        with self.logger.log_task(f"reading {matrix_dir}"):
            adata = sc.read_10x_mtx(matrix_dir)
        rmtree((self.raw_files_directory / "filtered_gene_bc_matrices").resolve())
        return adata


class TenX2017PBMC(TenXChromiumRNAV2):
    _citation = (
        "@misc{10x2017pbmc,\n"
        "   author={10X Genomics},\n"
        "   title={8k PBMCs from a Healthy Donor}, \n"
        "   howpublished="
        '"Available at \\url{https://www.10xgenomics.com/resources/datasets/'
        '8-k-pbm-cs-from-a-healthy-donor-2-standard-1-3-0}",\n'
        "   year={2017},\n"
        "   month={February},\n"
        '   note = "[Online; accessed 17-April-2023]"\n'
        "}"
    )
    _raw_urls = {
        "pbmc8k.tar.gz": (
            "https://cf.10xgenomics.com/samples/cell-exp/1.3.0/"
            "pbmc8k/pbmc8k_filtered_gene_bc_matrices.tar.gz"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.1}},
        var={"total_cells": {"min": 100}},
    )

    def _process_raw_data(self) -> AnnData:
        targz = next(iter(self.raw_files_paths.values()))
        with self.logger.log_task(f"extracting {targz.name}"):
            tarfile.open(str(targz)).extractall(self.raw_files_directory)
        matrix_dir = self.raw_files_directory / "filtered_gene_bc_matrices" / "GRCh38"
        with self.logger.log_task(f"reading {matrix_dir}"):
            adata = sc.read_10x_mtx(matrix_dir)
        rmtree((self.raw_files_directory / "filtered_gene_bc_matrices").resolve())
        return adata


class TenX2018MouseBrain(TenXChromiumRNAV3):
    _citation = (
        "@misc{10X2018MouseBrain,\n"
        "   author={10X Genomics},\n"
        "   title={10k Brain Cells from an E18 Mouse (v3 chemistry)}, \n"
        "   howpublished="
        '"Available at \\url{https://www.10xgenomics.com/resources/datasets/'
        '10-k-brain-cells-from-an-e-18-mouse-v-3-chemistry-3-standard-3-0-0}",\n'
        "   year={2018},\n"
        "   month={November},\n"
        '   note = "[Online; accessed 17-April-2023]"\n'
        "}"
    )
    _raw_urls = {
        "mouse10k.h5": (
            "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/"
            "neuron_10k_v3/neuron_10k_v3_filtered_feature_bc_matrix.h5"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.1}},
        var={"total_cells": {"min": 100}},
    )

    def _process_raw_data(self) -> AnnData:
        adata = {
            path.stem: sc.read_10x_h5(str(path))
            for path in self.raw_files_paths.values()
        }
        for value in adata.values():
            value.X = csr_matrix(value.X, dtype=int)
            value.var_names_make_unique()
            value.obs_names_make_unique()
        return next(iter(adata.values()))


class TenX2018PBMC(TenXChromiumRNAV3):
    _citation = (
        "@misc{10x2018pbmc,\n"
        "   author={10X Genomics},\n"
        "   title={10k PBMCs from a Healthy Donor (v3 chemistry)}, \n"
        "   howpublished="
        '"Available at \\url{https://www.10xgenomics.com/resources/datasets/'
        '10-k-pbm-cs-from-a-healthy-donor-v-3-chemistry-3-standard-3-0-0}",\n'
        "   year={2018},\n"
        "   month={November},\n"
        '   note = "[Online; accessed 17-April-2023]"\n'
        "}"
    )
    _raw_urls = {
        "pbmc10k.h5": (
            "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/"
            "pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.1}},
        var={"total_cells": {"min": 100}},
    )

    def _process_raw_data(self) -> AnnData:
        adata = {
            path.stem: sc.read_10x_h5(str(path))
            for path in self.raw_files_paths.values()
        }
        for value in adata.values():
            value.X = csr_matrix(value.X, dtype=int)
            value.var_names_make_unique()
            value.obs_names_make_unique()
        return next(iter(adata.values()))


class TenX2020MouseBrain(TenXChromiumRNAV3_1):
    _citation = (
        "@misc{10x2020mousebrain,\n"
        "   author={10X Genomics},\n"
        "   title={10k Mouse E18 Combined Cortex, Hippocampus and Subventricular "
        "Zone Cells, Dual Indexed}, \n"
        "   howpublished="
        '"Available at \\url{https://www.10xgenomics.com/resources/datasets/'
        "10-k-mouse-e-18-combined-cortex-hippocampus-and-subventricular-zone-cells-dual"
        '-indexed-3-1-standard-4-0-0}",\n'
        "   year={2020},\n"
        "   month={July},\n"
        '   note = "[Online; accessed 17-April-2023]"\n'
        "}"
    )
    _raw_urls = {
        "mousebrain.h5": (
            "https://cf.10xgenomics.com/samples/cell-exp/"
            "4.0.0/SC3_v3_NextGem_DI_Neuron_10K/"
            "SC3_v3_NextGem_DI_Neuron_10K_filtered_feature_bc_matrix.h5"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.1}},
        var={"total_cells": {"min": 100}},
    )

    def _process_raw_data(self) -> AnnData:
        adata = {
            path.stem: sc.read_10x_h5(str(path))
            for path in self.raw_files_paths.values()
        }
        for value in adata.values():
            value.X = csr_matrix(value.X, dtype=int)
            value.var_names_make_unique()
            value.obs_names_make_unique()
        return next(iter(adata.values()))


class TenX2021PBMC(TenXChromiumRNAV3_1):
    _citation = (
        "@misc{10x2021pbmc,\n"
        "   author={10X Genomics},\n"
        "   title={20k Human PBMCs, 3' HT v3.1, Chromium X}, \n"
        "   howpublished="
        '"Available at \\url{https://www.10xgenomics.com/resources/datasets/'
        '20-k-human-pbm-cs-3-ht-v-3-1-chromium-x-3-1-high-6-1-0}",\n'
        "   year={2021},\n"
        "   month={August},\n"
        '   note = "[Online; accessed 17-April-2023]"\n'
        "}"
    )
    _raw_urls = {
        "pbmc20k.h5": (
            "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/"
            "20k_PBMC_3p_HT_nextgem_Chromium_X/"
            "20k_PBMC_3p_HT_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.1}},
        var={"total_cells": {"min": 60}},
    )

    def _process_raw_data(self) -> AnnData:
        adata = {
            path.stem: sc.read_10x_h5(str(path))
            for path in self.raw_files_paths.values()
        }
        for value in adata.values():
            value.X = csr_matrix(value.X, dtype=int)
            value.var_names_make_unique()
            value.obs_names_make_unique()
        return next(iter(adata.values()))


class Zheng2017(TenXChromiumRNAV1):
    _citation = (
        "@article{zheng2017,\n"
        "   title={Massively parallel digital transcriptional profiling of single"
        " cells},\n"
        "   author={Zheng, Grace XY and Terry, Jessica M and Belgrader, Phillip and "
        "Ryvkin, Paul and Bent, Zachary W and Wilson, Ryan and Ziraldo, Solongo B and "
        "Wheeler, Tobias D and McDermott, Geoff P and Zhu, Junjie and others},\n"
        "   journal={Nature communications},\n"
        "   volume={8},\n"
        "   number={1},\n"
        "   pages={14049},\n"
        "   year={2017},\n"
        "   publisher={Nature Publishing Group UK London}\n"
        "}"
    )

    _raw_urls = {
        "CD19+ B cells.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "b_cells/b_cells_filtered_gene_bc_matrices.tar.gz"
        ),
        "CD4+ T cells.tar.gz": (
            "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "cd4_t_helper/cd4_t_helper_filtered_gene_bc_matrices.tar.gz"
        ),
        "CD14+ cells.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "cd14_monocytes/cd14_monocytes_filtered_gene_bc_matrices.tar.gz"
        ),
        "CD34+ cells.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "cd34/cd34_filtered_gene_bc_matrices.tar.gz"
        ),
        "CD4+CD25+ regulatory T cells.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "regulatory_t/regulatory_t_filtered_gene_bc_matrices.tar.gz"
        ),
        "CD4+CD45RA+CD25- naive T cells.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "naive_t/naive_t_filtered_gene_bc_matrices.tar.gz"
        ),
        "CD4+CD45RO+ memory T cells.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "memory_t/memory_t_filtered_gene_bc_matrices.tar.gz"
        ),
        "CD56+ natural killer cells.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "cd56_nk/cd56_nk_filtered_gene_bc_matrices.tar.gz"
        ),
        "CD8+ cytotoxic T cells.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "cytotoxic_t/cytotoxic_t_filtered_gene_bc_matrices.tar.gz"
        ),
        "CD8+CD45RA+ naive cytotoxic T cells.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "naive_cytotoxic/naive_cytotoxic_filtered_gene_bc_matrices.tar.gz"
        ),
        "DC_barcodes.csv": ("/banach1/jay/bipca_raw_data/zheng2017/DC_barcodes.csv"),
    }
    #FROM THE ZHENG PAPER METHODS: CD4+ T-Helpers include all CD4+ cells.
    """For example, CD4+ T-helper cells include all CD4+ cells. 
    This means that cells from this sample will overlap with cells from samples
    that contain CD4+ cells, including CD4+/CD25+ T reg, CD4+/CD45RO+ T memory,
    CD4+/CD45RA+/CD25− naive T. Thus, when a cell was assigned the ID of CD4+ T-helper 
    cell based on the correlation score, the next highest correlation was checked to
    see if it was one of the CD4+ samples. If it was, the cell’s ID was updated to the
    cell type with the next highest correlation. The same procedure was performed for
    CD8+ cytotoxic T and CD8+/CD45RA+ naive cytotoxic T (which is a subset of CD8+
    cytotoxic T).
    """
    _unfiltered_urls = {f'{k.split(".")[0]}.h5ad': None for k in _raw_urls.keys()}
    _unfiltered_urls["full.h5ad"] = None
    _unfiltered_urls["markers.h5ad"] = None  # this is the marker genes figure sample
    _unfiltered_urls["classifier.h5ad"] = None  # this is the classifier sample
    # from the paper.
    _hidden_samples = ["markers.h5ad", "classifier.h5ad"]
    _not_a_sample = ['DC_barcodes.csv']
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.1}},
        var={"total_cells": {"min": 100}},
    )

    def __init__(self, intersect_vars=False, *args, **kwargs):
        # change default here so that it doesn't intersect between samples.
        kwargs["intersect_vars"] = intersect_vars
        super().__init__(*args, **kwargs)

 
    def _process_raw_data(self) -> AnnData:
        targz = [v for k, v in self.raw_files_paths.items() if k.endswith('.tar.gz')]
        data = {}
        with open(self.raw_files_paths["DC_barcodes.csv"], "r") as f:
            DC_barcodes = f.read().splitlines()
        for filepath in targz:
            cell_type = filepath.stem.split(".")[0]
            with self.logger.log_task(f"extracting {filepath.name}"):
                tarfile.open(str(filepath)).extractall(self.raw_files_directory)
            matrix_dir = self.raw_files_directory / "filtered_matrices_mex" / "hg19"
            with self.logger.log_task(f"reading {filepath.name}"):
                data[cell_type] = sc.read_10x_mtx(matrix_dir)
            
            if cell_type == 'CD14+ cells':
                data[cell_type].obs["cluster"] = 'CD14+CLEC9A- monocytes'
                data[cell_type].obs.loc[
                    data[cell_type].obs_names.isin(
                        DC_barcodes),'cluster'] = 'CD14+CLEC9A+ dendritic cells'
            else:
                data[cell_type].obs["cluster"] = cell_type
     
        

        # rm any extracted data.
        rmtree((self.raw_files_directory / "filtered_matrices_mex").resolve())
        # merge all the data into one adata.
        data["full"] = ad.concat([data for label, data in data.items()])
        data["full"].obs_names_make_unique()
        # get the marker gene figure clusters
        adata = data["full"]
        adata.obs['super_cluster'] = adata.obs['cluster'].astype(str).apply(
            lambda x: 'CD4+ T cells' if 'CD4+' in x else x).apply(
            lambda x: 'CD8+ T cells' if 'CD8+' in x else x)  
        mask = adata.obs["super_cluster"].isin(
            ["CD4+ T cells", "CD8+ T cells",
            "CD56+ natural killer cells", "CD19+ B cells"]
        ) 
        adata = adata[mask, :].copy()
       
        data["markers"] = adata

        data["classifier"] = data["full"][
            ~data["full"].obs["cluster"].isin(["CD4+ T cells","CD8+ cytotoxic T cells"]), :
        ].copy()
        return data


class TenX2021HekMixtureV2(TenXChromiumRNAV2):
    _citation = (
        "@misc{10x2021hek_mixture_v2,\n"
        "   author={10X Genomics},\n"
        "   title={20k 1:1 Mixture of Human HEK293T and Mouse NIH3T3 cells, 5' HT v2.0}, \n"
        "   howpublished="
        "Available at \\url{https://www.10xgenomics.com/resources/datasets/"
        "20-k-1-1-mixture-of-human-hek-293-t-and-mouse-nih-3-t-3-cells-5-ht-v-2-0-2-high-6-1-0}"
        "   year={2021},\n"
        "   month={August},\n"
        '   note = "[Online; accessed 30-May-2023]"\n'
        "}"
    )
    _raw_urls = {
        "human_mouse_v2_20k.h5": (
            "https://cf.10xgenomics.com/samples/cell-vdj/"
            "6.1.0/20k_hgmm_5pv2_HT_nextgem_Chromium_X/"
            "20k_hgmm_5pv2_HT_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.1}},
        var={"total_cells": {"min": 100}},
    )

    def _process_raw_data(self) -> AnnData:
        adata = {
            path.stem: sc.read_10x_h5(str(path))
            for path in self.raw_files_paths.values()
        }
        for value in adata.values():
            value.X = csr_matrix(value.X, dtype=int)
            value.var_names_make_unique()
            value.obs_names_make_unique()
        return next(iter(adata.values()))


class TenX2021HekMixtureV3(TenXChromiumRNAV3_1):
    _citation = (
        "@misc{10x2021hek_mixture_v3,\n"
        "   author={10X Genomics},\n"
        "   title={20k 1:1 Mixture of Human HEK293T and Mouse NIH3T3 cells, 3' HT v3.1}, \n"
        "   howpublished="
        "Available at \\url{https://www.10xgenomics.com/resources/datasets/"
        "20-k-1-1-mixture-of-human-hek-293-t-and-mouse-nih-3-t-3-cells-3-ht-v-3-1-3-1-high-6-1-0}"
        "   year={2021},\n"
        "   month={August},\n"
        '   note = "[Online; accessed 30-May-2023]"\n'
        "}"
    )
    _raw_urls = {
        "human_mouse_v3_20k.h5": (
            "https://cf.10xgenomics.com/samples/"
            "cell-exp/6.1.0/20k_hgmm_3p_HT_nextgem_Chromium_X/"
            "20k_hgmm_3p_HT_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.1}},
        var={"total_cells": {"min": 100}},
    )

    def _process_raw_data(self) -> AnnData:
        adata = {
            path.stem: sc.read_10x_h5(str(path))
            for path in self.raw_files_paths.values()
        }
        for value in adata.values():
            value.X = csr_matrix(value.X, dtype=int)
            value.var_names_make_unique()
            value.obs_names_make_unique()
        return next(iter(adata.values()))


# TODO: add citations
# TODO: to be replaced by a permenant online path
class SCORCH_INS(TenXChromiumRNAV3):
    _citation = (
        "@article{ament2024single,\n"
        "title={The single-cell opioid responses in the context of HIV (SCORCH) consortium},\n"
        "author={Ament, Seth A and Campbell, Rianne R and Lobo, Mary Kay and Receveur, \n"
        "Joseph P and Agrawal, Kriti and Borjabad, Alejandra and Byrareddy, Siddappa N and Chang, \n"
        "Linda and Clarke, Declan and Emani, Prashant and others},\n"
        "journal={Molecular Psychiatry},\n"
        "pages={1--12},\n"
        "year={2024},\n"
        "publisher={Nature Publishing Group UK London}}")
    _raw_urls = {
        "scorch_ins_nih1889.tar.gz": (
            "/banach2/SCORCH/data/raw/10xChromiumV3_Nuclei-INS-CTR_OUD-5pairs-05242021/"
            "cellranger/NIH1889_OUD/filtered_feature_bc_matrix.tar.gz"
        ),
        "metadata.csv" : (
             "/banach2/jyc/bipca/data/um1/batch_effect_oud/NIH1889_OUD_all_metadata.csv"
        ),
        "scDblFinder.csv":(
         "/banach2/jyc/bipca/data/um1/batch_effect_oud/scDblFinder.csv"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 500, "max": 7500}, "pct_MT_UMIs": {"max": 0.02},"doublet":{"max":0.5}},
        var={"total_cells": {"min": 100}},
    )

    def _process_raw_data(self) -> AnnData:
        targz = next(iter(self.raw_files_paths.values()))
        with self.logger.log_task(f"extracting {targz.name}"):
            tarfile.open(str(targz)).extractall(self.raw_files_directory)
        matrix_dir = self.raw_files_directory / "filtered_feature_bc_matrix"
        with self.logger.log_task(f"reading {matrix_dir}"):
            adata = sc.read_10x_mtx(matrix_dir)

        meta_info = pd.read_csv(self.raw_files_directory / "metadata.csv",index_col=0)
        doublet_info =  pd.read_csv(self.raw_files_directory / "scDblFinder.csv",index_col=0)
        adata.obs["replicate_id"] = meta_info.loc[adata.obs_names,"replicate_id"]
        adata.obs['doublet'] = 1 
        cell2keep = doublet_info[doublet_info["scDblFinder.class"].values == "singlet"].index
            
        adata.obs.loc[cell2keep,'doublet'] = 0
            
        return adata


# PFC data

class SCORCH_PFC(Multiome_rna):
    _citation = ()

    _sample_ids = [
        "s1",
        "s2",
        "s3"
    ]
    _raw_urls = {
        "s1.h5": (
            "/banach2/SCORCH/data/raw/10xMultiome-PFC-CTR_HIV-8pairs-10172023/cellranger_arc/HCTXJ_CTR_PFC_MAH/outs/"
            "/filtered_feature_bc_matrix.h5"
        ),
        "s2.h5": (
            "/banach2/SCORCH/data/raw/10xMultiome-PFC-CTR_HIV-6pairs-08102022/cellranger_arc/HCtNZ_CTR_PFC_MAH/outs/"
            "/filtered_feature_bc_matrix.h5"
        ),

        "s3.h5": (
            "/banach2/SCORCH/data/raw/10xMultiome-PFC-HIVOUD_OUD-8pairs-09232023//cellranger_arc/NIH1564_OUD_PFC_MAH//outs/"
            "/filtered_feature_bc_matrix.h5"
        ),
    
    }
    for sid in _sample_ids:
        _raw_urls[sid+"_scDblFinder.csv"] = "/banach1/jyc/bipca/biPCA_copy_Dec8_2023/biPCA/scripts/um1_data/small_PCs_experiment/um1_cleaned_new/"+sid+"_scDblFinder.csv"

    _unfiltered_urls = {f"{sample}.h5ad": None for sample in _sample_ids}
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 500, "max": 7500}, "pct_MT_UMIs": {"max": 0.02},"doublet":{"max":0.5}},
        var={"total_cells": {"min": 100}},
    )

    def __init__(self, intersect_vars=False, n_filter_iters=1 ,*args, **kwargs):
        # change default here so that it doesn't intersect between samples.
        kwargs["intersect_vars"] = intersect_vars
        kwargs["n_filter_iters"] = n_filter_iters
        super().__init__(*args, **kwargs)

    def _process_raw_data(self) -> Dict[str, AnnData]:

        adata = {
            sid:sc.read_10x_h5(self.raw_files_directory / (sid+".h5"),gex_only=True)
            for sid in self._sample_ids
        }
        for value in adata.values():
            value.X = csr_matrix(value.X, dtype=int)
            value.var_names_make_unique()
            value.obs_names_make_unique()
        for sid in self._sample_ids:
            
            scDbl_df = pd.read_csv(self.raw_files_directory / (sid+"_scDblFinder.csv"),index_col=0)
            adata[sid].obs['doublet'] = 1 
            cell2keep = scDbl_df[scDbl_df["scDblFinder.class"].values == "singlet"].index
            
            adata[sid].obs.loc[cell2keep,'doublet'] = 0
            
            
        return adata

# class SCORCH_PFC_ATAC(Multiome_ATAC):
#     _citation = ()

#     _sample_ids = [
#         "s1",
#         "s2",
#         "s3"
#     ]
#     _raw_urls = {
#         "s1.h5": (
#             "/banach2/SCORCH/data/raw/10xMultiome-PFC-CTR_HIV-8pairs-10172023/cellranger_arc/HCTXJ_CTR_PFC_MAH/outs/"
#             "/filtered_feature_bc_matrix.h5"
#         ),
#         "s2.h5": (
#             "/banach2/SCORCH/data/raw/10xMultiome-PFC-CTR_HIV-6pairs-08102022/cellranger_arc/HCtNZ_CTR_PFC_MAH/outs/"
#             "/filtered_feature_bc_matrix.h5"
#         ),

#         "s3.h5": (
#             "/banach2/SCORCH/data/raw/10xMultiome-PFC-HIVOUD_OUD-8pairs-09232023//cellranger_arc/NIH1564_OUD_PFC_MAH//outs/"
#             "/filtered_feature_bc_matrix.h5"
#         ),
    
#     }
#     for sid in _sample_ids:
#         _raw_urls[sid+"_scDblFinder.csv"] = "/banach1/jyc/bipca/biPCA_copy_Dec8_2023/biPCA/scripts/um1_data/small_PCs_experiment/um1_cleaned_new/"+sid+"_scDblFinder.csv"

#     _unfiltered_urls = {f"{sample}.h5ad": None for sample in _sample_ids}
#     _filters = AnnDataFilters(
#         obs={"total_sites": {"min": 1000},"doublet":{"max":0.5}},
#         var={"total_cells": {"min": 50}},
#     )

#     def __init__(self, intersect_vars=False, n_filter_iters=1 ,*args, **kwargs):
#         # change default here so that it doesn't intersect between samples.
#         kwargs["intersect_vars"] = intersect_vars
#         kwargs["n_filter_iters"] = n_filter_iters
#         super().__init__(*args, **kwargs)

#     def _process_raw_data(self) -> Dict[str, AnnData]:

#         adata = {
#             sid:sc.read_10x_h5(self.raw_files_directory / (sid+".h5"),gex_only=False)
#             for sid in self._sample_ids
#         }
#         for value in adata.values():
#             value.X = csr_matrix(value.X, dtype=int)
#             value.var_names_make_unique()
#             value.obs_names_make_unique()
#         for sid in self._sample_ids:
#             adata[sid] = adata[sid][:,adata[sid].var['feature_types'] == "Peaks"]
#             scDbl_df = pd.read_csv(self.raw_files_directory / (sid+"_scDblFinder.csv"),index_col=0)
#             adata[sid].obs['doublet'] = 1 
#             cell2keep = scDbl_df[scDbl_df["scDblFinder.class"].values == "singlet"].index
            
#             adata[sid].obs.loc[cell2keep,'doublet'] = 0
            
        
            
#         return adata

####################################################
#               CITE-seq (RNA)                     #
####################################################


class Stoeckius2017(CITEseq_rna):
    _citation = (
        " @article{stoeckius2017simultaneous,\n"
        " title={Simultaneous epitope and transcriptome measurement in single cells},\n"
        " author={Stoeckius, Marlon and Hafemeister, Christoph and Stephenson, "
        " William and Houck-Loomis, Brian and Chattopadhyay, Pratip K and Swerdlow, "
        " Harold and Satija, Rahul and Smibert, Peter},\n"
        " journal={Nature methods},\n"
        " volume={14},\n"
        " number={9},\n"
        " pages={865--868},\n"
        " year={2017},\n"
        " publisher={Nature Publishing Group}\n"
        "}"
    )
    _raw_urls = {
        "cbmc_rna.csv.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/"
            "GSE100866/suppl/GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz"
        ),
        "cbmc.SeuratData.tar.gz": (
            "http://seurat.nygenome.org/src/contrib/cbmc.SeuratData_3.1.4.tar.gz"
        ),
    }

    _unfiltered_urls = {None: None}

    def __init__(self, n_filter_iters=1, *args, **kwargs):
        # change default here so that it doesn’t intersect between samples.
        kwargs["n_filter_iters"] = n_filter_iters
        super().__init__(*args, **kwargs)

    def _process_raw_data(self) -> AnnData:
        counts_rna_in = self.raw_files_directory / "cbmc_rna.csv.gz"
        metadata_in_parent = self.raw_files_directory / "cbmc.SeuratData.tar.gz"

        counts_rna_output = self.raw_files_directory / "cbmc_rna.csv"

        # unzip each file
        # read in the rna data as adata, add the metadata inside
        with gzip.open(counts_rna_in) as counts_rna:
            with open(counts_rna_output, "wb") as f_out:
                copyfileobj(counts_rna, f_out)
        adata = read_csv_pyarrow_bad_colnames(
            counts_rna_output, delimiter=",", index_col=0
        )
        # note: data is tranposed
        var_names = adata.index.values
        obs_names = adata.columns
        adata = AnnData(
            csr_matrix(adata.values.T, dtype=int),
            dtype=int,
        )
        adata.obs_names = obs_names
        adata.var_names = var_names

        with self.logger.log_task(f"extracting {metadata_in_parent.name}"):
            tarfile.open(str(metadata_in_parent)).extractall(self.raw_files_directory)
        cbmc_meta = pyreadr.read_r(
            str(self.raw_files_directory)
            + "/cbmc.SeuratData/inst/extdata/annotations/annotations.Rds"
        )[None]

        adata.obs["protein_annotations"] = cbmc_meta.loc[
            adata.obs_names.values, "protein_annotations"
        ]
        adata.obs["rna_annotations"] = cbmc_meta.loc[
            adata.obs_names.values, "rna_annotations"
        ]

        # remove some cell types
        adata.obs["passQC"] = (
            adata.obs["protein_annotations"].notna()
            & (
                ~adata.obs["rna_annotations"].isin(
                    ["T/Mono doublets"]
                )
            )
            & (~adata.obs["protein_annotations"].isin(["T/Mono doublets"]))
        )

        adata.var["isMouse"] = adata.var_names.str.contains("MOUSE")
        adata.var["total_UMIs"] = np.asarray(adata.X.sum(0)).squeeze()

        
        
        # remove mouse genes
        adata = adata[:, ~adata.var["isMouse"]]
        adata = adata[adata.obs["passQC"], :]

        return adata


class Stuart2019(CITEseq_rna):
    _citation = (
        " @article{stuart2019comprehensive,\n"
        " title={Comprehensive integration of single-cell data},\n"
        " author={Stuart, Tim and Butler, Andrew and Hoffman, "
        " Paul and Hafemeister, Christoph and Papalexi, Efthymia and Mauck, "
        " William M and Hao, Yuhan and Stoeckius, Marlon and Smibert, "
        " Peter and Satija, Rahul},\n"
        " journal={Cell},"
        " volume={177},"
        " number={7},"
        " pages={1888--1902},"
        " year={2019},"
        " publisher={Elsevier}"
        " }"
    )
    _raw_urls = {
        "bmcite_rna.tsv.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3681nnn/"
            "GSM3681518/suppl/GSM3681518_MNC_RNA_counts.tsv.gz"
        ),
        "bmcite_metadata.tsv": (
            "https://www.dropbox.com/s/g2jm3m3uupqwdlc/bmcite_metadata.tsv?dl=1"
        ),
    }

    _unfiltered_urls = {None: None}

    def __init__(self, n_filter_iters=1, *args, **kwargs):
        kwargs["n_filter_iters"] = n_filter_iters
        super().__init__(*args, **kwargs)

    def _process_raw_data(self) -> AnnData:
        counts_rna_in = self.raw_files_directory / "bmcite_rna.tsv.gz"
        metadata_in = self.raw_files_directory / "bmcite_metadata.tsv"

        counts_rna_output = self.raw_files_directory / "bmcite_rna.tsv"

        # unzip each file
        # read in the rna data as adata, add the metadata inside
        with gzip.open(counts_rna_in) as counts_rna:
            with open(counts_rna_output, "wb") as f_out:
                copyfileobj(counts_rna, f_out)
        adata = read_csv_pyarrow_bad_colnames(
            counts_rna_output, delimiter="\t", index_col=0
        )
        # note: data is transposed
        var_names = adata.index.values
        obs_names = adata.columns
        adata = AnnData(
            csr_matrix(adata.values.T, dtype=int),
            dtype=int,
        )
        adata.obs_names = obs_names
        adata.var_names = var_names

        bm_meta = read_csv_pyarrow_bad_colnames(
            metadata_in, delimiter="\t", index_col=0
        )

        bm_meta.index = bm_meta.index.str.replace("-", ".")

        cells2keep = np.intersect1d(bm_meta.index, adata.obs_names)
        adata = adata[cells2keep, :]
        bm_meta = bm_meta.loc[cells2keep, :]
        adata.obs["cell_types"] = bm_meta.loc[adata.obs_names.values, "celltype.l2"]

        return adata

#########################################################
#                 Open challenge Data                   #
#########################################################

class OpenChallengeCITEseqData(CITEseq_rna):
    _citation = (
        "@inproceedings{luecken2021sandbox, \n"
        "title={A sandbox for prediction and integration of \n"
        "DNA, RNA, and proteins in single cells}, \n"
        "author={Luecken, Malte D and Burkhardt, Daniel Bernard and \n"
        "Cannoodt, Robrecht and Lance, Christopher and Agrawal, Aditi and \n"
        "Aliee, Hananeh and Chen, Ann T and Deconinck, Louise and Detweiler, \n"
        "Angela M and Granados, Alejandro A and others},\n"
        "booktitle={Thirty-fifth conference on neural information \n"
        "processing systems datasets and benchmarks track (Round 2)},\n"
        "year={2021}"
        "}" 
    )
    _raw_urls = {
        "cite_BMMC_processed.h5ad.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/"
            "GSE194122/suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"
        )
    }
    _sample_ids = [
        's1d1', 's1d2', 's1d3', 's2d1', 's2d4', 's2d5', 's3d1', 's3d6', 's3d7', 's4d1', 's4d8', 's4d9'
    ]
    _unfiltered_urls = {f"{sample}.h5ad": None for sample in _sample_ids}
    
    def __init__(self, intersect_vars = False, n_filter_iters=10, *args, **kwargs):
        # change default here so that it doesn’t intersect between samples.
        kwargs["intersect_vars"] = intersect_vars
        kwargs["n_filter_iters"] = n_filter_iters
        super().__init__(*args, **kwargs)

    def _process_raw_data(self) -> Dict[str, AnnData]:
        cite_data_in = self.raw_files_directory / "cite_BMMC_processed.h5ad.gz"
        cite_data_output = self.raw_files_directory / "cite_BMMC_processed.h5ad"

        # unzip each file
        with gzip.open(cite_data_in) as cite_data:
            with open(cite_data_output, "wb") as f_out:
                copyfileobj(cite_data, f_out)

        cite_adata = sc.read_h5ad(str(cite_data_output))
        # keep the count data to X
        cite_adata.X = cite_adata.layers['counts'].toarray().copy()
        # keep only RNA data
        cite_adata = cite_adata[:,cite_adata.var['feature_types'] == "GEX"]
        cite_adata.var.drop('gene_id',axis=1,inplace=True)
        adata = {
            bid: cite_adata[cite_adata.obs["batch"] == bid,:]
            for bid in self._sample_ids
        }
        
        return adata


class OpenChallengeMultiomeData(Multiome_rna):
    _citation = (
        "@inproceedings{luecken2021sandbox, \n"
        "title={A sandbox for prediction and integration of \n"
        "DNA, RNA, and proteins in single cells}, \n"
        "author={Luecken, Malte D and Burkhardt, Daniel Bernard and \n"
        "Cannoodt, Robrecht and Lance, Christopher and Agrawal, Aditi and \n"
        "Aliee, Hananeh and Chen, Ann T and Deconinck, Louise and Detweiler, \n"
        "Angela M and Granados, Alejandro A and others},\n"
        "booktitle={Thirty-fifth conference on neural information \n"
        "processing systems datasets and benchmarks track (Round 2)},\n"
        "year={2021}"
        "}" 
    )
    _raw_urls = {
         "multiome_BMMC_processed.h5ad.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/"
            "GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz"
        )
    }
    _sample_ids = [
        's1d1', 's1d2', 's1d3', 's2d1', 's2d4', 's2d5', 's3d10','s3d3', 's3d6', 's3d7', 's4d1', 's4d8', 's4d9'
    ]
    _unfiltered_urls = {f"{sample}.h5ad": None for sample in _sample_ids}
    def __init__(self, intersect_vars = False, n_filter_iters=10, *args, **kwargs):
        # change default here so that it doesn’t intersect between samples.
        kwargs["intersect_vars"] = intersect_vars
        kwargs["n_filter_iters"] = n_filter_iters
        super().__init__(*args, **kwargs)

    def _process_raw_data(self) -> Dict[str, AnnData]:
        multi_data_in = self.raw_files_directory / "multiome_BMMC_processed.h5ad.gz"
        multi_data_output = self.raw_files_directory / "multiome_BMMC_processed.h5ad"

        # unzip each file
        with gzip.open(multi_data_in) as multi_data:
            with open(multi_data_output, "wb") as f_out:
                copyfileobj(multi_data, f_out)

        multi_adata = sc.read_h5ad(str(multi_data_output))
        # keep the count data to X
        multi_adata.X = multi_adata.layers['counts'].toarray().copy()
        # keep only RNA data
        multi_adata = multi_adata[:,multi_adata.var['feature_types'] == "GEX"]
        # drop the gene_id column -> which would cause an error for annotate
        multi_adata.var.drop('gene_id',axis=1,inplace=True)
        adata = {
            bid: multi_adata[multi_adata.obs["batch"] == bid,:]
            for bid in self._sample_ids
        }
        
        return adata        

class OpenChallengeMultiomeData_ATAC(Multiome_ATAC):
    _citation = (
        "@inproceedings{luecken2021sandbox, \n"
        "title={A sandbox for prediction and integration of \n"
        "DNA, RNA, and proteins in single cells}, \n"
        "author={Luecken, Malte D and Burkhardt, Daniel Bernard and \n"
        "Cannoodt, Robrecht and Lance, Christopher and Agrawal, Aditi and \n"
        "Aliee, Hananeh and Chen, Ann T and Deconinck, Louise and Detweiler, \n"
        "Angela M and Granados, Alejandro A and others},\n"
        "booktitle={Thirty-fifth conference on neural information \n"
        "processing systems datasets and benchmarks track (Round 2)},\n"
        "year={2021}"
        "}" 
    )
    _raw_urls = {
         "multiome_BMMC_processed.h5ad.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/"
            "GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz"
        )
    }
    _sample_ids = [
        's1d1', 's1d2', 's1d3', 's2d1', 's2d4', 's2d5', 's3d10','s3d3', 's3d6', 's3d7', 's4d1', 's4d8', 's4d9'
    ]
    _unfiltered_urls = {f"{sample}.h5ad": None for sample in _sample_ids}
    _filters = AnnDataFilters(
        obs={"total_sites": {"min": 100}},  # 1000
        var={"total_cells": {"min": 150}}, #50
    )
    def __init__(self, intersect_vars = False, n_filter_iters=10, *args, **kwargs):
        # change default here so that it doesn’t intersect between samples.
        kwargs["intersect_vars"] = intersect_vars
        kwargs["n_filter_iters"] = n_filter_iters
        super().__init__(*args, **kwargs)

    def _process_raw_data(self) -> Dict[str, AnnData]:
        multi_data_in = self.raw_files_directory / "multiome_BMMC_processed.h5ad.gz"
        multi_data_output = self.raw_files_directory / "multiome_BMMC_processed.h5ad"

        # unzip each file
        with gzip.open(multi_data_in) as multi_data:
            with open(multi_data_output, "wb") as f_out:
                copyfileobj(multi_data, f_out)

        multi_adata = sc.read_h5ad(str(multi_data_output))
        # keep the count data to X
        multi_adata.X = multi_adata.layers['counts'].toarray().copy()
        # keep only RNA data
        multi_adata = multi_adata[:,multi_adata.var['feature_types'] == "ATAC"]
        multi_adata.var.drop('gene_id',axis=1,inplace=True)
        adata = {
            bid: multi_adata[multi_adata.obs["batch"] == bid,:]
            for bid in self._sample_ids
        }
        
        return adata  
####################################################
#                  10 X Multiome                   #
####################################################
# class TenX2021PBMCMultiome(TenXMultiome):
#     _citation = (
#         "@misc{10x2021pbmcmultiome,\n"
#         "  author = {10x Genomics},\n"
#         "  title = { {PBMC from a Healthy Donor} - Granulocytes Removed Through Cell Sorting (10k)},\n"
#         '  howpublished = "Available at \\url{https://www.10xgenomics.com/resources/'
#         "datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-"
#         '10-k-1-standard-2-0-0}",\n'
#         "  year = {2021},\n"
#         "  month = {May},\n"
#         '  note = "[Online; accessed 31-July-2023]"\n'
#         "}"
#     )
#     _raw_urls = {
#         "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5": (
#             "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/"
#             "pbmc_granulocyte_sorted_10k/"
#             "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"
#         )
#     }
#     _unfiltered_urls = {None: None}
#     _filters = AnnDataFilters(
#         obs={"total_genes": {"min": 100},
#             "total_sites":{"min":100},
#             "pct_MT_UMIs": {"max": 0.1}},
#         var={"ATAC":{"total_cells": {"min": 100}},#5
#             "GEX":{"total_cells":{"min": 100}}},#50
#     )

     

# class SCORCH_PFC_HIVCTR_Multiome(TenXMultiome):
#     _citation = ()

#     _sample_ids = [
#         "6801066772_HIV",
#         "6801187468_HIV",
#         "7100518287_HIV",
#         "7101847783_HIV",
#         "7102096765_HIV",
#         "7200776574_HIV",
#         "HCcPL_CTR",
#         "HCTKN_CTR",
#         "HCtME_CTR",
#         "HCTMW_CTR",
#         "HCtNZ_CTR",
#         "HCTTS_CTR",
#     ]
#     _raw_urls = {
#         f"{key}.h5": (
#             f"/banach2/SCORCH/data/raw//10xMultiome-PFC-CTR_HIV-6pairs-08102022/cellranger/{key}_PFC_MAH/outs/"
#             "/filtered_feature_bc_matrix.h5"
#         )
#         for key in _sample_ids
#     }
#     _unfiltered_urls = {f"{sample}.h5ad": None for sample in _sample_ids}
#     _filters = AnnDataFilters(
#         obs={"total_genes": {"min": 500, "max": 7500}, "total_sites":{"min":100}},
#         var={
#             "ATAC":{"total_cells":{"min":100}},
#             "GEX":{"total_cells":{"min":50}}
#         }
#     )

#     def __init__(self, intersect_vars=False, *args, **kwargs):
#         # change default here so that it doesn't intersect between samples.
#         kwargs["intersect_vars"] = intersect_vars
#         super().__init__(*args, **kwargs)

#     def _process_raw_data(self) -> Dict[str, AnnData]:
#         adata = {
#             path.stem: sc.read_10x_h5(str(path), gex_only=False)
#             for path in self.raw_files_paths.values()
#         }
#         for value in adata.values():
#             value.X = csr_matrix(value.X, dtype=int)
#             value.var_names_make_unique()
#             value.obs_names_make_unique()
#         return adata


####################################################
#                    snmCseq2                      #
####################################################

class RufZamojski2021(SnmCseq2):
    """
    Note: This class is dependent on a python package ALLCools. 
    Please refer to
    https://lhqing.github.io/ALLCools/start/installation.html 
    for install instructions.
    """
    _citation = (
        "@article{ruf2021single,\n"
        "  title={Single nucleus multi-omics regulatory landscape of the murine pituitary},\n"
        "  author={Ruf-Zamojski, Frederique and Zhang, Zidong and Zamojski, Michel and Smith, Gregory R and Mendelev, Natalia and Liu, Hanqing and Nudelman, German and Moriwaki, Mika and Pincas, Hanna and Castanon, Rosa Gomez and others},\n"
        "  journal={Nature communications},\n"
        "  volume={12},\n"
        "  number={1},\n"
        "  pages={2677},\n"
        "  year={2021},\n"
        "  publisher={Nature Publishing Group UK London}\n"
        "}"
    )
    _raw_urls = {
        "mm10.chrom.sizes":(
            "http://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes"
        ),
        "mm10-blacklist.v2.bed.gz":(
            "https://github.com/Boyle-Lab/Blacklist/blob/master/lists/mm10-blacklist.v2.bed.gz"
        ),
        "PIT.CellMetadata.csv.gz":(
            "http://neomorph.salk.edu/hanqingliu/PIT/PIT.CellMetadata.csv.gz"
        ),
        "filelist.txt":(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE152nnn/GSE152011/suppl/filelist.txt"
        ),
        "GSE152011_RAW.tar":(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE152nnn/GSE152011/suppl/GSE152011_RAW.tar"
        ),
    }
    _sample_names = ['mc','cov']
    _unfiltered_urls = dict(zip([sample + ".h5ad" for sample in _sample_names], 
                                [
                                    ("/banach2/ruiqi/bipca/datasets/SnmCseq2/"
                                     "RufZamojski2021/unfiltered/"
                                     "chrom5k_mcg.h5ad"),
                                    ("/banach2/ruiqi/bipca/datasets/SnmCseq2/"
                                     "RufZamojski2021/unfiltered/"
                                     "chrom5k_cov.h5ad"),
                                ]))

    @classmethod 
    def _annotate(cls, adata:AnnData) -> AnnDataAnnotations:
        annotations = super()._annotate(adata)
        # Subsampling
        row_sample_num = len(annotations.obs) * 5
        if len(annotations.var) > row_sample_num:
            selected_rows = annotations.var.sample(n = row_sample_num, random_state = 1)
            annotations.var["subsample"] = annotations.var.index.isin(selected_rows.index)
        else:
            annotations.var["subsample"] = True
        annotations.var['subsample'] = annotations.var['subsample'].replace({True: 2, False: 0})

        col_sample_num = len(annotations.var) * 5
        if len(annotations.obs) > col_sample_num:
            selected_cols = annotations.obs.sample(n = col_sample_num, random_state = 1)
            annotations.obs["subsample"] = annotations.obs.index.isin(selected_cols.index)
        else:
            annotations.obs["subsample"] = True
        annotations.var['subsample'] = annotations.var['subsample'].replace({True: 2, False: 0})
        annotations.obs['subsample'] = annotations.obs['subsample'].replace({True: 2, False: 0})
        return annotations

    # Sampling the rows & cols & stabilizing
    _filters = AnnDataFilters(
        obs = {
            "subsample":{'min':1},
            "total_bins": {'min':5},
        },
        var = {
            "subsample":{'min':1},
            "total_obs": {'min':5},
        }
    )
    
    def _process_raw_data(self) -> AnnData:
        # process chrom.sizes file, only keep the main chroms
        if "mm10.main.chrom.sizes" not in os.listdir(self.raw_files_directory):
            pattern = re.compile(r'^chr(?:[1-9]|1[0-9]|X|Y|M)\t')
            with open(self.raw_files_directory / "mm10.chrom.sizes", "r") as file:
                kept_lines = [line for line in file if pattern.match(line) is not None]
            with open(self.raw_files_directory / "mm10.main.chrom.sizes", "w") as file:
                file.writelines(kept_lines)
      
        # Process raw file
        ## Unzip raw file
        output_dic = self.raw_files_directory / "raw_list"
        if not os.path.exists(output_dic):
            os.makedirs(output_dic)
        expected_content = pd.read_csv(self.raw_files_directory / "filelist.txt",sep = "\t").drop(0)["Name"].tolist()
        if any(item not in os.listdir(output_dic) for item in expected_content):
            with tarfile.open(self.raw_files_directory / "GSE152011_RAW.tar", 'r') as archive:
                archive.extractall(path=output_dic)

        ## Tabix raw file
        files = os.listdir(output_dic)
        tsv_files = [f for f in files if f.endswith('.allc.tsv.gz')]
        for file in tsv_files:
            tbi_file = file + ".tbi"
            # If corresponding .tbi file does not exist, run tabix
            if tbi_file not in files:
                cmd = ['tabix', '-b', '2', '-e', '2', '-s', '1', os.path.join(output_dic, file)]
                subprocess.run(cmd)
    
        # Create ALLC table
        if "allc_table.tsv" not in os.listdir(self.raw_files_directory):
            ALLC_table = pd.read_csv(self.raw_files_directory / "filelist.txt",sep = "\t")
            ALLC_table = ALLC_table.drop(0)
            ALLC_table = ALLC_table[["Name"]]
            ALLC_table["sample"] = ALLC_table["Name"].str.extract(r'_(.+)\.allc\.tsv\.gz')
            ALLC_table = ALLC_table[["sample", "Name"]]
            ALLC_table[["Name"]] = str(output_dic) + "/" + ALLC_table[["Name"]]
            ALLC_table.to_csv(str(self.raw_files_directory / "allc_table.tsv"), sep='\t', index=False, header = False)

        # ALLCools to generate MCDS file
        if not any(folder.endswith('.mcds') for folder in os.listdir(self.raw_files_directory)):
            self._run_bash_processing()
        #TODO: MOVE THIS MCDS PROCESSING TO ADATA FILTERS?
        # Load MCDS & Meta file and preprocess by ALLCools
        mcds = MCDS.open(self.raw_files_directory / "RufZamojski2021NC.mcds", var_dim = 'chrom5k')
        metadata = pd.read_csv(self.raw_files_directory / 'PIT.CellMetadata.csv.gz', index_col=0)
        # Basic filtering parameters
        mapping_rate_cutoff = 0.5
        mapping_rate_col_name = 'MappingRate'  
        final_reads_cutoff = 500000
        final_reads_col_name = 'FinalmCReads'  
        mccc_cutoff = 0.03
        mccc_col_name = 'mCCCFrac'  
        mch_cutoff = 0.2
        mch_col_name = 'mCHFrac'  
        mcg_cutoff = 0.5
        mcg_col_name = 'mCGFrac' 
        
        judge = (metadata[mapping_rate_col_name] > mapping_rate_cutoff) & \
                (metadata[final_reads_col_name] > final_reads_cutoff) & \
                (metadata[mccc_col_name] < mccc_cutoff) & \
                (metadata[mch_col_name] < mch_cutoff) & \
                (metadata[mcg_col_name] > mcg_cutoff)
        metadata = metadata[judge].copy()
        mcds = mcds.sel(cell = metadata.index)
        mcds.add_cell_metadata(metadata)

        # remove blacklist regions
        # Encode The ENCODE blacklist can be downloaded from https://github.com/Boyle-Lab/Blacklist/blob/master/lists/mm10-blacklist.v2.bed.gz
        black_list_path = self.raw_files_directory / 'mm10-blacklist.v2.bed.gz'
        mcds = mcds.remove_black_list_region(black_list_path=black_list_path)
        # remove chromosomes
        exclude_chromosome = ['chrM', 'chrY']
        mcds = mcds.remove_chromosome(var_dim='chrom5k', exclude_chromosome=exclude_chromosome)

        # create anndata
        var_dim = 'chrom5k'
        obs_dim = 'cell'
        # count_types = ['mc','cov']
        adatas = {}
        for count_type in _sample_names:
            use_data = mcds[f'{var_dim}_da'].sel({'count_type':count_type}).squeeze()
            obs_df, var_df = _make_obs_df_var_df(use_data, obs_dim, var_dim)
            ad = anndata.AnnData(
                X = use_data.transpose(obs_dim,var_dim).values.astype(int),
                obs = obs_df,
                var = var_df
            )
            adatas[count_type] = ad
        return adatas
        
        
    def _run_bash_processing(self):
        subprocess.run(
            ["/bin/bash","/bipca/bipca/experiments/datasets/allcools_preprocess.sh"],
            cwd=str(self.raw_files_directory),
        )