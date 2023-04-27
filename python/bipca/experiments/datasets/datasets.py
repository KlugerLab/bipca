import tarfile
import gzip
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

from bipca.experiments.datasets.base import DataFilters
from bipca.experiments.datasets.utils import (
    get_ensembl_mappings,
    read_csv_pyarrow_bad_colnames,
)
from bipca.experiments.datasets.modalities import *


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
        rng = np.random.default_rng(seed=seed)

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

    _filtered_urls = {"31767.h5ad": None, "31778.h5ad": None, "31790.h5ad": None}

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
    _citation = (
        "@article{liu2020high,\n"
        "  title={High-spatial-resolution multi-omics sequencing via deterministic "
        "barcoding in tissue},\n"
        "  author={Liu, Yang and Yang, Mingyu and Deng, Yanxiang and Su, Graham and "
        "Enninful, Archibald and Guo, Cindy C and Tebaldi, Toma and Zhang, Di and "
        "Kim, Dongjoo and Bai, Zhiliang and others},\n"
        "  journal={Cell},\n"
        "  volume={183},\n"
        "  number={6},\n"
        "  pages={1665--1681},\n"
        "  year={2020},\n"
        "  publisher={Elsevier}\n}"
    )
    _raw_urls = {
        "dbit-Seq.tsv.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/"
            "GSM4096nnn/GSM4096261/suppl/GSM4096261_10t.tsv.gz"
        )
    }
    _filtered_urls = {None: None}
    _filters = DataFilters(
        obs={"total_genes": {"min": -np.Inf}}, var={"total_obs": {"min": 100}}
    )

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
    _filtered_urls = {None: None}
    _filters = DataFilters(
        obs={"total_genes": {"min": -np.Inf}}, var={"total_obs": {"min": 100}}
    )

    def _process_raw_data(self) -> AnnData:
        with tarfile.open(self.raw_files_directory / "raw.zip") as f:
            f.extractall(self.raw_files_directory)


###################################################
#   Visium                                        #
###################################################
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
        "}\n"
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
    _filtered_urls = {None: None}
    _filters = DataFilters(
        obs={"total_genes": {"min": -np.Inf}}, var={"total_obs": {"min": 100}}
    )

    def _process_raw_data(self) -> AnnData:
        adata = {
            path.stem: sc.read_10x_h5(str(path))
            for path in self.raw_files_paths.values()
        }
        for section_name, adat in adata.items():
            adat.obs["section"] = int(section_name[-1])
            adat.X = csr_matrix(adat.X, dtype=int)
            adat.var_names_make_unique()
            adat.obs_names_make_unique()
        adata = ad.concat(adata.values())
        adata.obs_names_make_unique()
        return adata


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
    _filtered_urls = {None: None}
    _filters = DataFilters(
        obs={"total_genes": {"min": -np.Inf}}, var={"total_obs": {"min": 100}}
    )

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


# SINGLE CELL DATA #
# ATAC #
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
            "https://www.cell.com/cms/10.1016/j.cell.2018.03.074/"
            "attachment/2a72a316-33cc-427d-8019-dfc83bd220ca/mmc4.zip"
        )
    }
    _filtered_urls = {None: None}
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


# RNA #
# SMART-SEQ #
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
    _filtered_urls = {None: None}
    _filters = DataFilters(
        obs={
            "pct_mapped_reads": {"min": 0.5},
            "pct_MT_reads": {"max": 0.15},
            "total_reads": {"min": 2e4},
            "passed_qc": {
                "min": 0
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


# 10X #
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
    _filtered_urls = {None: None}
    _filters = DataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.05}},
        var={"total_cells": {"min": 100}},
    )

    def _process_raw_data(self) -> AnnData:
        targz = next(iter(self.raw_files_paths.values()))
        with self.logger.log_task(f"extracting {targz.name}"):
            tarfile.open(str(targz)).extractall(self.raw_files_directory)
        matrix_dir = self.raw_files_directory / "filtered_matrices_mex" / "hg19"
        with self.logger.log_task(f"reading {filepath.name}"):
            adata = sc.read_10x_mtx(matrix_dir)
        rmtree((self.raw_files_directory / "filtered_matrices_mex").resolve())
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

    _filtered_urls = {None: None}

    _filters = DataFilters(
        obs={"total_genes": {"min": 100}, "pct_MT_UMIs": {"max": 0.05}},
        var={"total_cells": {"min": 100}},
    )

    def _process_raw_data(self) -> AnnData:
        targz = [v for k, v in self.raw_files_paths.items()]
        data = {}
        for filepath in targz:
            cell_type = filepath.stem
            with self.logger.log_task(f"extracting {filepath.name}"):
                tarfile.open(str(filepath)).extractall(self.raw_files_directory)
            matrix_dir = self.raw_files_directory / "filtered_matrices_mex" / "hg19"
            with self.logger.log_task(f"reading {filepath.name}"):
                data[cell_type] = sc.read_10x_mtx(matrix_dir)
            data[cell_type].obs["cluster"] = cell_type
        # rm any extracted data.
        rmtree((self.raw_files_directory / "filtered_matrices_mex").resolve())
        adata = ad.concat([data for label, data in data.items()])
        return adata
