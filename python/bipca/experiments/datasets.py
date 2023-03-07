import tarfile
from shutil import move as mv, rmtree
import tarfile
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.io import loadmat


import scanpy as sc
import anndata as ad
from anndata import AnnData

from .base import DataFilters
from .utils import get_ensembl_mappings, read_csv_pyarrow_bad_colnames
from .modalities import *


class Buenrostro2018ATAC(Buenrostro2015Protocol):
    _citation = (
        "@article{buenrostro2018integrated,\n"
        "title={Integrated single-cell analysis maps the continuous regulatory "
        "landscape of human hematopoietic differentiation},\n"
        "author={Buenrostro, Jason D and Corces, M Ryan and Lareau, Caleb A and "
        "Wu, Beijing and Schep, Alicia N and Aryee, Martin J and Majeti, Ravindra and "
        "Chang, Howard Y and Greenleaf, William J},\n"
        "journal={Cell},\n"
        "volume={173},\n"
        "number={6},\n"
        "pages={1535--1548},\n"
        "year={2018},\n"
        "publisher={Elsevier}\n"
        "}"
    )
    _raw_urls = {
        "raw.zip": (
            "https://www.cell.com/cms/10.1016/j.cell.2018.03.074/"
            "attachment/2a72a316-33cc-427d-8019-dfc83bd220ca/mmc4.zip"
        )
    }
    _filtered_url = ""
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

        adata = AnnData(X=X)
        adata.obs_names = [ele[0][0] for ele in cellnames]
        adata.obs["cell_type"] = [ele[0] for ele in celltypes.flatten()]
        adata.obs["facs_label"] = [
            "MEP"
            if "MEP" in line
            else line.split(".bam")[0].lstrip("singles-").split("BM")[-1].split("-")[1]
            for line in adata.obs_names.tolist()
        ]
        return adata


class HagemannJensen2022(SmartSeqV3):
    _citation = (
        "@article{hagemannjensen2022,\n"
        "title={Scalable single-cell RNA sequencing from full transcripts with "
        "Smart-seq3xpress},\n"
        "author={Hagemann-Jensen, Michael and Ziegenhain, Christoph and Sandberg, \n"
        "Rickard},\n"
        """journal={Nature Biotechnology},
        volume={40},
        number={10},
        pages={1452--1457},
        year={2022},
        publisher={Nature Publishing Group US New York
        }"""
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
    _filtered_url = ""
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
        adata = AnnData(csr_matrix(data["UMIs"].values.T))
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


class Kluger2023Melanoma(CosMx):
    _citation = "undefined"

    _raw_urls = {
        "panel1.csv": (
            "/banach1/CosMx_melanoma/SMI-0168_HarrietKluger_Yale/8 Metadata/"
            "Run5612_TMA1_exprMat_file_annotated.csv"
        ),
        "panel1_metadata.csv": (
            "/banach1/CosMx_melanoma/SMI-0168_HarrietKluger_Yale/5 Raw data/"
            "Run5612_TMA1/Run5612_TMA1_metadata_file.csv"
        ),
        "panel2.csv": (
            "/banach1/CosMx_melanoma/SMI-0168_HarrietKluger_Yale/8 Metadata/"
            "Run5619_TMA2_exprMat_file_annotated.csv"
        ),
        "panel2_metadata.csv": (
            "/banach1/CosMx_melanoma/SMI-0168_HarrietKluger_Yale/5 Raw data/"
            "Run5619_TMA2/Run5619_TMA2_metadata_file.csv"
        ),
        "clinical_metadata.csv": (
            "/banach1/CosMx_melanoma/SMI-0168_HarrietKluger_Yale/8 Metadata/"
            "TMA_376_clinical_metadata.csv"
        ),
        "sample_mapping.csv": (
            "/banach1/CosMx_melanoma/SMI-0168_HarrietKluger_Yale/8 Metadata/"
            "TMA_376_sample_mapping_simplified.csv"
        ),
    }

    _filtered_url = ""

    _filters = DataFilters(
        obs={"total_genes": {"min": 100}}, var={"total_obs": {"min": 100}}
    )

    def _process_raw_data(self) -> AnnData:
        base_files = [v for k, v in self.raw_files_paths.items()]

        data = {
            fname: df
            for fname, df in map(
                lambda f: (
                    f.stem,
                    read_csv_pyarrow_bad_colnames(
                        f, logger=self.logger, index_col=None
                    ),
                ),
                base_files,
            )
        }
        Xs = []
        obs = []
        sample_mapping = data["sample_mapping"].rename(
            columns={"TMA2_5619": 2, "TMA1_5612": 1}
        )
        for i in range(1, 3):
            key = f"panel{i}"
            counts = data[key]
            metadata = data[key + "_metadata"]
            # the count data has an extra row per person that is ECM
            # we need to add this to the metadata.

            ecm = counts.loc[counts["cell_ID"] == 0][["fov", "cell_ID"]]
            metadata = pd.concat([metadata, ecm])

            for df in [counts, metadata]:
                df.set_index(
                    f"{i}-" + df.fov.astype(str) + "-" + df.cell_ID.astype(str),
                    inplace=True,
                )
                df.fov = df.fov.astype(str)
                df.cell_ID = df.cell_ID.astype(str)
            # sort the metadata so it is the same order as the counts
            metadata = metadata.loc[counts.index]
            # add an extra "is_extracellular" column to the metadata
            metadata = metadata.assign(is_extracellular=lambda x: x.cell_ID == 0)
            # load the correct? CPIDs
            print(len(np.unique(counts['CPID'])))

            panel_df = sample_mapping.loc[:, ["CPID", i]]
            panel_df = panel_df.loc[panel_df[i] != "X"]
            panel_df = panel_df.set_index(i, drop=True)
            counts = counts.drop(columns='CPID').join(panel_df, on="fov")
            print(len(np.unique(panel_df['CPID'])))
            metadata["CPID"] = counts["CPID"]
            metadata["run"] = i
            obs.append(metadata)
            Xs.append(csr_matrix(counts.iloc[:, 3:]))
        X = vstack(Xs)
        var_names = counts.iloc[:, 3:].columns
        del Xs
        adata = AnnData(X)
        adata.var_names = var_names
        obs = pd.concat(obs)
        obs.index = obs["CPID"].astype(str) + "-" + obs["cell_ID"].astype(str)
        obs = obs.join(data["clinical_metadata"].set_index("CPID"), on="CPID")
        adata.obs = obs
        return adata


class Pbmc33k(TenXChromiumRNAV1):
    pass


class Zheng2017(TenXChromiumRNAV1):
    _citation = (
        "@article{zheng2017,\n"
        "title={Massively parallel digital transcriptional profiling of single cells},"
        "\n"
        "author={Zheng, Grace XY and Terry, Jessica M and Belgrader, Phillip and "
        "Ryvkin, Paul and Bent, Zachary W and Wilson, Ryan and Ziraldo, Solongo B and "
        "Wheeler, Tobias D and McDermott, Geoff P and Zhu, Junjie and others},\n"
        "journal={Nature communications},\n"
        "volume={8},\n"
        "number={1},\n"
        "pages={14049},\n"
        "year={2017},\n"
        "publisher={Nature Publishing Group UK London}\n"
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

    _filtered_url = ""

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
