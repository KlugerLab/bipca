import tarfile
import gzip
import zipfile
import sys
import inspect
from itertools import product
from shutil import move as mv, rmtree, copyfileobj
from numbers import Number
from typing import Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.io import loadmat

import scanpy as sc
import anndata as ad
from anndata import AnnData, read_h5ad

from bipca.experiments.datasets.base import DataFilters, Dataset
from bipca.experiments.datasets.utils import (
    get_ensembl_mappings,
    read_csv_pyarrow_bad_colnames,
)
from bipca.experiments.datasets.modalities import *

import subprocess
from pandas_plink import read_plink
from bipca.math import binomial_variance


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
class RankRPoisson(Simulation):
    def __init__(
        self,
        rank: int = 1,
        mean: Number = 1,
        mrows: int = 500,
        ncols: int = 1000,
        **kwargs,
    ):
        self.rank = rank
        self.mean = mean
        super().__init__(mrows=mrows, ncols=ncols, **kwargs)

    def _default_session_directory(self) -> str:
        return (
            f"seed{self.seed}"
            f"rank{self.rank}"
            f"mean{self.mean}"
            f"mrows{self.mrows}"
            f"ncols{self.ncols}"
        )

    def _compute_simulated_data(self):
        # Generate a random matrix with rank r
        rng = np.random.default_rng(self.seed)
        S = np.exp(2 * rng.standard_normal(size=(self.mrows, self.rank)))
        coeff = rng.uniform(size=(self.rank, self.ncols))
        X = S @ coeff
        X = X / X.mean()  # Normalized to have average SNR = 1

        X *= self.mean  # Scale to desired mean
        Y = rng.poisson(lam=X)  # Poisson sampling
        adata = AnnData(Y, dtype=int)
        adata.layers["ground_truth"] = X
        adata.uns["rank"] = self.rank
        adata.uns["seed"] = self.seed
        adata.uns["mean"] = self.mean
        return adata


class QVFNegativeBinomial(Simulation):
    def __init__(
        self,
        rank: int = 1,
        mean: Number = 1,
        b: Number = 1,
        c: Number = 0.00001,
        mrows: int = 500,
        ncols: int = 1000,
        **kwargs,
    ):
        self.rank = rank
        self.mean = mean
        self.b = b
        self.c = c
        super().__init__(mrows=mrows, ncols=ncols, **kwargs)

    def _default_session_directory(self) -> str:
        return (
            f"seed{self.seed}"
            f"rank{self.rank}"
            f"mean{self.mean}"
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
        rng = np.random.default_rng(seed=self.seed)

        S = np.exp(2 * rng.standard_normal(size=(self.mrows, self.rank)))
        coeff = rng.uniform(size=(self.rank, self.ncols))
        X = S @ coeff
        X = X / X.mean()
        # Normalized to have average SNR = 1
        X *= self.mean

        theta = 1 / self.c
        nb_p = theta / (theta + X)
        Y0 = rng.negative_binomial(theta, nb_p)
        Y = self.b * Y0

        adata = AnnData(Y, dtype=int)
        adata.layers["ground_truth"] = X
        adata.uns["rank"] = self.rank
        adata.uns["seed"] = self.seed
        adata.uns["mean"] = self.mean
        adata.uns["b"] = self.b
        adata.uns["c"] = self.c
        return adata


###################################################
#   Real Data                                     #
###################################################
###################################################
#   Spatial transcriptomics                       #
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

    _filters = DataFilters(
        obs={"total_genes": {"min": 100}}, var={"total_obs": {"min": 100}}
    )

    def _process_raw_data(self) -> Dict[str, AnnData]:
        adata = {
            k.rstrip(".h5ad"): read_h5ad(str(v))
            for k, v in self.raw_files_paths.items()
        }

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
#   SeqFISH+                                      #
###################################################
class Eng2019(SeqFISHPlus):
    _citation = SeqFISHPlus.technology_citation
    _raw_urls = {
        "raw.zip": "https://github.com/CaiGroup/seqFISH-PLUS/raw/master/sourcedata.zip"
    }
    _unfiltered_urls = {"subventricular_zone.h5ad": None, "olfactory_bulb.h5ad": None}

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
        print(adata)
        for value in adata.values():
            value.X = csr_matrix(value.X, dtype=int)
            value.var_names_make_unique()
            value.obs_names_make_unique()
        return adata


class TenX2020HumanBreastCancer(TenXVisium):
    _citation = (
        "@misc{humanbreastcancer,\n"
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
class Buenrostro2018ATAC(Buenrostro2015Protocol):
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
    _filters = DataFilters(
        obs={"total_sites": {"min": 1000}},  # these are from the episcanpy tutorial.
        var={"total_cells": {"min": 5}},
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


###################################################
#   RNA                                           #
###################################################
###################################################
#   SmartSeq                                      #
###################################################
class HagemannJensen2022(SmartSeqV3):
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
        "UMIs.txt": (
            "https://www.ebi.ac.uk/biostudies/files/E-MTAB-11452/"
            "PBMCs.allruns.umicounts_intronexon.txt"
        ),
        "reads.txt": (
            "https://www.ebi.ac.uk/biostudies/files/E-MTAB-11452/"
            "PBMCs.allruns.readcounts_intronexon.txt"
        ),
        "annotations.txt": (
            "https://www.ebi.ac.uk/biostudies/files/E-MTAB-11452/"
            "PBMCs.allruns.barcode_annotation.txt"
        ),
    }
    _unfiltered_urls = {None: None}
    _filters = DataFilters(
        obs={
            "pct_mapped_reads": {"min": 0.5},
            "pct_MT_reads": {"max": 0.15},
            "total_reads": {"min": 2e4},
            "passed_qc": {
                "min": 1
            },  # passed_qc is a boolean annotation for this dataset.
            "pct_MT_UMIs": {"min": -np.Inf},  # get rid of this extra UMI filter.
            "total_genes": {"min": 500},
        },
        var={"total_cells": {"min": 10}},
    )

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
        del data["reads"]

        for col in data["annotations"].columns:
            if data["annotations"][col].dtype == "object":
                data["annotations"][col] = data["annotations"][col].astype("category")

        data["annotations"] = data["annotations"].rename(
            columns={"nReads": "total_reads", "QC_status": "passed_qc"}
        )
        data["annotations"]["passed_qc"] = data["annotations"]["passed_qc"] == "QCpass"
        adata.obs = pd.concat([adata.obs, data["annotations"]], axis=1)
        gene_dict = get_ensembl_mappings(adata.var_names.tolist(), logger=self.logger)
        var_df = pd.DataFrame.from_dict(gene_dict, orient="index")
        var_df["gene_biotype"] = var_df["gene_biotype"].astype("category")
        adata.var = var_df
        return adata


###################################################
#   Chromium                                      #
###################################################
class Pbmc33k(TenXChromiumRNAV1):
    _citation = (
        "@misc{pbmc33k,\n"
        "   author={10X Genomics},\n"
        "   title=\{\{33k PBMCs from a Healthy Donor\}\},\n"
        "   howpublished="
        '"\\url{https://www.10xgenomics.com/resources/datasets/'
        '33-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0}",\n'
        "   year={2016},\n"
        "   month={September},\n"
        '   note = "[Online; accessed 17-April-2023]"'
    )
    _raw_urls = {
        "pbmc33k.tar.gz": (
            "https://cf.10xgenomics.com/samples/cell-exp/"
            "1.1.0/pbmc33k/pbmc33k_filtered_gene_bc_matrices.tar.gz"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = DataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.05}},
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
        "B.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "b_cells/b_cells_filtered_gene_bc_matrices.tar.gz"
        ),
        "Helper T.tar.gz": (
            "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "cd4_t_helper/cd4_t_helper_filtered_gene_bc_matrices.tar.gz"
        ),
        "CD14+.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "cd14_monocytes/cd14_monocytes_filtered_gene_bc_matrices.tar.gz"
        ),
        "CD34+.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "cd34/cd34_filtered_gene_bc_matrices.tar.gz"
        ),
        "Treg.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "regulatory_t/regulatory_t_filtered_gene_bc_matrices.tar.gz"
        ),
        "Naive T.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "naive_t/naive_t_filtered_gene_bc_matrices.tar.gz"
        ),
        "Memory T.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "memory_t/memory_t_filtered_gene_bc_matrices.tar.gz"
        ),
        "CD56+ NK.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "cd56_nk/cd56_nk_filtered_gene_bc_matrices.tar.gz"
        ),
        "Cytotoxic T.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "cytotoxic_t/cytotoxic_t_filtered_gene_bc_matrices.tar.gz"
        ),
        "Naive cytotoxic T.tar.gz": (
            "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
            "naive_cytotoxic/naive_cytotoxic_filtered_gene_bc_matrices.tar.gz"
        ),
    }

    _unfiltered_urls = {f'{k.split(".")[0]}.h5ad': None for k in _raw_urls.keys()}
    _unfiltered_urls["full.h5ad"] = None
    _filters = DataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.05}},
        var={"total_cells": {"min": 100}},
    )

    def __init__(self, intersect_vars=False, *args, **kwargs):
        # change default here so that it doesn't intersect between samples.
        kwargs["intersect_vars"] = intersect_vars
        super().__init__(*args, **kwargs)

    def _process_raw_data(self) -> AnnData:
        targz = [v for k, v in self.raw_files_paths.items()]
        data = {}
        for filepath in targz:
            cell_type = filepath.stem.split(".")[0]
            with self.logger.log_task(f"extracting {filepath.name}"):
                tarfile.open(str(filepath)).extractall(self.raw_files_directory)
            matrix_dir = self.raw_files_directory / "filtered_matrices_mex" / "hg19"
            with self.logger.log_task(f"reading {filepath.name}"):
                data[cell_type] = sc.read_10x_mtx(matrix_dir)
            data[cell_type].obs["cluster"] = cell_type
        # rm any extracted data.
        rmtree((self.raw_files_directory / "filtered_matrices_mex").resolve())
        data["full"] = ad.concat([data for label, data in data.items()])
        return data


#############################################################
###               1000 Genome Phase3                      ###
#############################################################


class Phase3_1000Genome(SingleNucleotidePolymorphism):
    """
    Dataset class to obtain 1000 genome phase3 SNP data
    Note: This class is dependent on plink/plink2 (bash) and a python package pandas_plink.

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
        None: "/banach2/jyc/bipca/data/1000Genome/bipca/datasets/"
        "SingleNucleotidePolymorphism/Phase3_1000Genome/"
        "unfiltered/Phase3_1000Genome.h5ad"
    }

    _filters = DataFilters(
        obs={"total_SNPs": {"min": -np.Inf}},
        var={"binomal_var_non0": {"min": 80}, "total_obs": {"min": -np.Inf}},
    )

    def _process_raw_data(self) -> AnnData:
        self._run_bash_processing()

        # read the processed files as adata
        (bim, fam, bed) = read_plink(
            str(self.raw_files_directory) + "/all_phase3_pruned", verbose=True
        )

        adata = AnnData(X=bed.compute().transpose())
        binomial_var = binomial_variance(adata.X, 2)

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

        # add the binomal_var sparisity
        adata.var["binomal_var_non0"] = np.sum(binomial_var != 0, axis=0)

        return adata

    def _run_bash_processing(self):
        # run plink preprocessing
        subprocess.run(
            ["/bin/bash", "/bipca/bipca/experiments/datasets/plink_preprocess.sh"],
            cwd=str(self.raw_files_directory),
        )
