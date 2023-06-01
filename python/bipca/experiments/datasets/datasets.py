import tarfile
import gzip
import zipfile
import sys
import inspect
from shutil import move as mv, rmtree, copyfileobj
from numbers import Number
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.io import loadmat

import scanpy as sc
import anndata as ad
from anndata import AnnData, read_h5ad

from bipca.math import MarcenkoPastur
from bipca.experiments.datasets.base import AnnDataFilters, Dataset
from bipca.experiments.datasets.utils import (
    get_ensembl_mappings,
    read_csv_pyarrow_bad_colnames,
)
from bipca.experiments.datasets.modalities import *
from bipca.experiments.experiments import random_nonnegative_orthonormal_matrix
import subprocess
from pandas_plink import read_plink
import pyreadr


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
        mean: Number = 1,  # also controls the SV amplitude when random_signal = False
        random_signal: bool = True,
        minimum_signal_singular_value: Number = None,
        mrows: int = 500,
        ncols: int = 1000,
        **kwargs,
    ):
        self.rank = rank
        self.mean = mean
        self.random_signal = random_signal
        super().__init__(mrows=mrows, ncols=ncols, **kwargs)
        if minimum_signal_singular_value is None:
            self.minimum_signal_singular_value = (
                MarcenkoPastur(np.minimum(mrows, ncols) / np.maximum(mrows, ncols)).b
                * 2
            )
        else:
            self.minimum_signal_singular_value = minimum_signal_singular_value

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
        if self.random_signal:
            S = np.exp(2 * rng.standard_normal(size=(self.mrows, self.rank)))
            coeff = rng.uniform(size=(self.rank, self.ncols))
            X = S @ coeff
            X = X / X.mean()  # Normalized to have average SNR = 1
            X *= self.mean  # Scale to desired mean
        else:
            # generate m x r non-negative orthonormal basis for rows
            U = random_nonnegative_orthonormal_matrix(self.mrows, self.rank, rng)
            # generate r x n non-negative orthonormal basis for columns
            V = random_nonnegative_orthonormal_matrix(self.ncols, self.rank, rng)
            S = (
                self.mean
                * np.sqrt(np.count_nonzero(U, axis=0))
                * np.sqrt(np.count_nonzero(V, axis=0))
            ).mean()  # gets you pretty close to entrywise mean, provided there aren't huge gaps in the nnzs across rows and columns
            if self.minimum_signal_singular_value is not False:
                if S <= self.minimum_signal_singular_value:
                    self.logger.log_warning(
                        f"Entrywise mean of {self.mean} yields a matrix with "
                        f"a small minimum signal norm ({S:.3f}). Clamping signal "
                        f"singular values to  {self.minimum_signal_singular_value:.3f}"
                    )
                    S = self.minimum_signal_singular_value

            X = (U * S) @ V.T
        Y = rng.poisson(lam=X)  # Poisson sampling
        adata = AnnData(Y, dtype=float)
        adata.layers["ground_truth"] = X
        adata.uns["rank"] = self.rank
        adata.uns["seed"] = self.seed
        adata.uns["mean"] = self.mean
        adata.uns["minimum_signal_singular_value"] = self.minimum_signal_singular_value

        return adata


class QVFNegativeBinomial(Simulation):
    def __init__(
        self,
        rank: int = 1,
        mean: Number = 1000,
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

        libsize = rng.lognormal(
            np.log(self.mean), sigma=0.1, size=(self.mrows,)
        ).astype(int)
        # "modules"
        coeff = np.geomspace(0.0001, 0.05, num=self.rank * self.ncols)
        coeff = np.random.permutation(coeff).reshape(self.rank, self.ncols)
        loadings = rng.multinomial(libsize, pvals=[1 / self.rank] * self.rank)

        X = loadings @ coeff
        theta = 1 / self.c
        nb_p = theta / (theta + X)
        Y0 = rng.negative_binomial(theta, nb_p)
        Y = self.b * Y0

        adata = AnnData(Y, dtype=float)
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

    _filters = AnnDataFilters(
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
    _filters = AnnDataFilters(
        obs={"total_sites": {"min": 1000}},  # these are from the episcanpy tutorial.
        var={"total_cells": {"min": 50}},
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
        var={"total_cells": {"min": 50}},
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
        var={"total_cells": {"min": 50}},
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
        var={"total_cells": {"min": 50}},
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
        var={"total_cells": {"min": 50}},
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
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
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

    def __init__(self, n_filter_iters=1, *args, **kwargs):
        # change default here so that it doesn't filter twice.
        kwargs["n_filter_iters"] = n_filter_iters
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
        gene_dict = get_ensembl_mappings(adata.var_names.tolist(), logger=self.logger)
        var_df = pd.DataFrame.from_dict(gene_dict, orient="index")

        adata.var = var_df
        for c in adata.var.columns:
            if adata.var[c].dtype in ["object", "category"]:
                adata.var[c] = adata.var[c].astype(str)
        for c in adata.obs.columns:
            if adata.obs[c].dtype in ["object", "category"]:
                adata.obs[c] = adata.obs[c].astype(str)
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
    _unfiltered_urls["markers.h5ad"] = None  # this is the marker genes figure sample
    _unfiltered_urls["classifier.h5ad"] = None  # this is the classifier sample
    # from the paper.
    _hidden_samples = ["markers.h5ad", "classifier.h5ad"]
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.1}},
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
        data["full"].obs_names_make_unique()
        data["full"].obs["cluster"] = data["full"].obs["cluster"].astype("category")
        # get the fig4 4 clusters
        adata = data["full"]
        mask = ~adata.obs["cluster"].isin(
            ["CD14+", "CD34+"]
        )  # CD14 and C34 cells are not in marker genes figure
        adata = adata[mask, :].copy()
        # reannotate for the marker gene clusters
        adata.obs["cluster"] = adata.obs["cluster"].cat.add_categories(
            ["CD4+ T", "CD8+ T"]
        )

        adata.obs["cluster"][
            adata.obs["cluster"].isin(
                [
                    "Naive T",
                    "Treg",
                    "Helper T",
                    "Memory T",
                ]
            )
        ] = "CD4+ T"
        adata.obs["cluster"][
            adata.obs["cluster"].isin(["Cytotoxic T", "Naive cytotoxic T"])
        ] = "CD8+ T"
        adata.obs["cluster"][adata.obs["cluster"].isin(["B"])] = "B"
        adata.obs["cluster"][adata.obs["cluster"].isin(["CD56+ NK"])] = "CD56+ NK"

        adata.obs["cluster"] = adata.obs["cluster"].cat.remove_unused_categories()
        data["markers"] = adata

        data["classifier"] = data["full"][
            data["full"].obs["cluster"] != "Helper T", :
        ].copy()
        return data


# TODO: add citations
# TODO: to be replaced by a permenant online path
class SCORCH_INS_OUD(TenXChromiumRNAV3):
    _citation = ()
    _raw_urls = {
        "scorch_ins_nih1889.tar.gz": (
            "/banach2/SCORCH/data/raw/10xChromiumV3_Nuclei-INS-CTR_OUD-5pairs-05242021/"
            "cellranger/NIH1889_OUD/filtered_feature_bc_matrix.tar.gz"
        )
    }
    _unfiltered_urls = {None: None}
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 500, "max": 7500}, "pct_MT_UMIs": {"max": 0.1}},
        var={"total_cells": {"min": 100}},
    )

    def _process_raw_data(self) -> AnnData:
        targz = next(iter(self.raw_files_paths.values()))
        with self.logger.log_task(f"extracting {targz.name}"):
            tarfile.open(str(targz)).extractall(self.raw_files_directory)
        matrix_dir = self.raw_files_directory / "filtered_feature_bc_matrix"
        with self.logger.log_task(f"reading {matrix_dir}"):
            adata = sc.read_10x_mtx(matrix_dir)

        return adata


#########################################################
###               CITE-seq (RNA)                     ###
#########################################################


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
                    ["Eryth", "Mk", "DC", "T/Mono doublets"]
                )
            )
            & (~adata.obs["protein_annotations"].isin(["T/Mono doublets"]))
        )

        adata.var["isMouse"] = adata.var_names.str.contains("MOUSE")
        adata.var["total_UMIs"] = np.asarray(adata.X.sum(0)).squeeze()

        # only keep the top 100 most highly expressed mouse genes
        adata.var["Genes2keep"] = adata.var_names.isin(
            adata.var.loc[adata.var["isMouse"], "total_UMIs"]
            .sort_values(ascending=False)[:100]
            .index
        ) | (~adata.var["isMouse"])

        adata = adata[:, adata.var["Genes2keep"]]
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


#############################################################
###               1000 Genome Phase3                      ###
#############################################################


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
        None: "/banach2/jyc/bipca/data/1000Genome/bipca/datasets/"
        "SingleNucleotidePolymorphism/Phase3_1000Genome/"
        "unfiltered/Phase3_1000Genome.h5ad"
    }

    _filters = AnnDataFilters(
        obs={"total_SNPs": {"min": -np.Inf}},
        var={"total_obs": {"min": -np.Inf}},
    )

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
            ["/bin/bash", "/bipca/bipca/experiments/datasets/plink_preprocess.sh"],
            cwd=str(self.raw_files_directory),
        )
