from numbers import Number
from typing import Literal, Optional, Union, Callable, Tuple
import numpy as np
from pandas import DataFrame
import anndata as ad
from anndata import AnnData
import scanpy as sc
from scipy.sparse import csr_matrix

from bipca.utils import nz_along
from bipca.math import MarcenkoPastur
from bipca.experiments.base import abstractmethod
from bipca.experiments.experiments import (
    random_rank_R_nonnegative_matrix,
    random_rank_R_nonnegative_matrix_minimum_singular_value,
)
from bipca.experiments.utils import parse_mrows_ncols_rank, get_rng
from bipca.experiments.datasets.base import (
    Modality,
    Technology,
    AnnDataFilters,
    AnnDataAnnotations,
)

###################################################
#   Simulations                                   #
###################################################


class Simulation(Modality, Technology):
    _citation = None
    _raw_urls = None
    _unfiltered_urls = {"simulation.h5ad": None}
    _filters = AnnDataFilters(
        obs={"total_nz": {"min": 10}},
        var={"total_nz": {"min": 10}},
    )

    def __init__(
        self,
        mrows: int = 500,
        ncols: int = 1000,
        seed: Number = 42,
        store_filtered_data: bool = False,
        store_unfiltered_data: bool = False,
        **kwargs,
    ):
        self.seed = seed
        self.mrows, self.ncols, _ = parse_mrows_ncols_rank(mrows, ncols)
        super().__init__(
            store_filtered_data=store_filtered_data,
            store_unfiltered_data=store_unfiltered_data,
            store_raw_files=False,
            **kwargs,
        )

    @classmethod
    def _annotate(cls, adata: AnnData) -> AnnDataAnnotations:
        annotations = AnnDataAnnotations.from_other(adata)
        annotations.obs["total_nz"] = np.asarray(nz_along(adata.X, 1))
        annotations.var["total_nz"] = np.asarray(nz_along(adata.X, 0))

        return annotations

    @property
    @abstractmethod
    def _default_session_directory(self) -> str:
        pass

    def acquire_raw_data(self, *args, **kwargs):
        # override this so that we don't download anything for simulations
        raise NotImplementedError

    def acquire_filtered_data(self, *args, **kwargs):
        # override so we don't download anything for simulations
        raise NotImplementedError

    @abstractmethod
    def _compute_simulated_data(self):
        pass

    def _process_raw_data(self):
        return self._compute_simulated_data()


class LowRankSimulation(Simulation):
    def __init__(
        self,
        rank: int = 1,
        entrywise_mean: Union[Number, Literal[False]] = False,
        libsize_mean: Number = 1000,
        minimum_singular_value: Union[Number, bool] = False,
        constant_singular_value: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        _, _, self.rank = parse_mrows_ncols_rank(self.mrows, self.ncols, rank)

        (
            self.generating_function,
            self.minimum_singular_value,
            self.constant_singular_value,
        ) = self._parse_generating_function_parameters(
            minimum_singular_value, constant_singular_value
        )
        self.entrywise_mean = entrywise_mean
        self.libsize_mean = libsize_mean
        self.constant_singular_value = constant_singular_value

    # todo: use @overload from typing with Literal to change this signature.
    def _parse_generating_function_parameters(
        self, minimum_singular_value: Union[Number, bool], constant_singular_value: bool
    ) -> Tuple[Callable, Union[Number, Literal[False]], bool]:
        if minimum_singular_value is True:
            # infer what the minimum singular value should be
            minimum_singular_value = (
                MarcenkoPastur(
                    np.minimum(self.mrows, self.ncols)
                    / np.maximum(self.mrows, self.ncols)
                ).b
                * 2
            )
        if not isinstance(minimum_singular_value, (Number, bool)):
            raise TypeError("minimum_singular_value must be a number or bool")
        if not isinstance(constant_singular_value, bool):
            raise TypeError("constant_singular_value must be a bool")

        if minimum_singular_value is not False:
            # use a generating function that will ensure a fixed min singular value
            generating_function = (
                random_rank_R_nonnegative_matrix_minimum_singular_value
            )
        else:
            # use a generating function that does not ensure a fixed min singular value
            generating_function = random_rank_R_nonnegative_matrix
        return generating_function, minimum_singular_value, constant_singular_value

    def get_low_rank_matrix(self, rng: np.random.Generator = None) -> np.ndarray:
        # override this to get a low rank matrix
        rng = get_rng(rng)
        if self.generating_function == random_rank_R_nonnegative_matrix:
            return self.generating_function(
                self.mrows,
                self.ncols,
                self.rank,
                entrywise_mean=self.entrywise_mean,
                libsize_mean=self.libsize_mean,
                rng=rng,
            )
        elif (
            self.generating_function
            == random_rank_R_nonnegative_matrix_minimum_singular_value
        ):
            return self.generating_function(
                self.mrows,
                self.ncols,
                self.rank,
                minimum_singular_value=self.minimum_singular_value,
                constant_singular_value=self.constant_singular_value,
                entrywise_mean=self.entrywise_mean,
                rng=rng,
            )


###################################################
#   Hi-C                                          #
###################################################


class ChromatinConformationCapture(Modality, Technology):
    pass


###################################################
#   scATACseq and technologies                    #
###################################################


class SingleCellATACSeq(Modality):
    """SingleCellRNASeq: Base Modality subclass for variants of SingleCellATACSeq.

    Implements SingleCellATACSeq specific annotations and filters for sites, and peaks.

    """

    _filters = AnnDataFilters(
        obs={"total_sites": None},  # these are from the episcanpy tutorial.
        var={"total_cells": None},
    )

    @classmethod
    def _annotate(cls, adata: AnnData) -> AnnDataAnnotations:
        annotations = AnnDataAnnotations.from_other(adata)
        annotations.obs["total_sites"] = np.asarray(nz_along(adata.X, 1))
        annotations.obs["total_peaks"] = np.asarray(adata.X.sum(1)).squeeze()
        annotations.var["total_cells"] = np.asarray(nz_along(adata.X, 0))
        annotations.var["total_peaks"] = np.asarray(adata.X.sum(0)).squeeze()
        return annotations


class Buenrostro2015Protocol(SingleCellATACSeq, Technology):
    """Buenrostro15Protocol: Technology proposed by Buenrostro15."""

    _technology_citation = (
        "@article{buenrostro2015single,\n"
        "title={Single-cell chromatin accessibility reveals principles of regulatory "
        "variation},\n"
        "author={Buenrostro, Jason D and Wu, Beijing and Litzenburger, Ulrike M and "
        "Ruff, Dave and Gonzales, Michael L and Snyder, Michael P and Chang, Howard Y "
        "and Greenleaf, William J},\n"
        "journal={Nature},\n"
        "volume={523},\n"
        "number={7561},\n"
        "pages={486--490},\n"
        "year={2015},\n"
        "publisher={Nature Publishing Group UK London}\n"
        "}"
    )


class TenXChromiumATACV1(SingleCellATACSeq, Technology):
    pass


class TenXChromiumATACV1_1(SingleCellATACSeq, Technology):
    pass


class TenXChromiumATACV2(SingleCellATACSeq, Technology):
    pass


###################################################
#   scRNAseq and technologies                     #
###################################################


class SingleCellRNASeq(Modality):
    """SingleCellRNASeq: Base Modality subclass for variants of SingleCellRNASeq.

    Implements SingleCellRNASeq specific annotations and filters for UMIs and gene names.

    """

    _filters = AnnDataFilters(
        obs={"total_genes": None, "pct_MT_UMIs": None}, var={"total_cells": None}
    )

    @classmethod
    def _annotate(cls, adata: AnnData) -> AnnDataAnnotations:
        annotations = AnnDataAnnotations.from_other(adata)

        try:
            gn = annotations.var["gene_name"]
        except Exception:
            gn = adata.var_names
        # gene annotations
        annotations.var["total_UMIs"] = np.asarray(adata.X.sum(0)).squeeze()
        annotations.var["total_cells"] = np.asarray(nz_along(adata.X, 0))
        annotations.var["is_MT"] = gn.str.lower().str.startswith("mt-").astype(bool)
        # cell annotations
        annotations.obs["total_UMIs"] = np.asarray(adata.X.sum(1)).squeeze()
        annotations.obs["total_MT_UMIs"] = np.asarray(
            adata[:, annotations.var.is_MT].X.sum(1)
        )
        annotations.obs["pct_MT_UMIs"] = (
            annotations.obs["total_MT_UMIs"] / annotations.obs["total_UMIs"]
        )
        annotations.obs["total_genes"] = np.asarray(nz_along(adata.X, 1))
        return annotations


class DropSeq(SingleCellRNASeq, Technology):
    pass


class CITEseq_rna(SingleCellRNASeq, Technology):
    """CITEseq_rna: The RNA modality of the CITE-seq technology."""

    _filters = AnnDataFilters(
        obs={
            "total_genes": {"min": 100},
            "pct_MT_UMIs": {"min": -np.Inf},
        },  # get rid of this extra UMI filter.
        var={"total_cells": {"min": 100}},
    )


class SmartSeqV3(SingleCellRNASeq, Technology):
    """SmartSeqV3: SingleCellRNASeq technology with support for read-based features.

    Implements SmartSeqV3 specific annotations and filters for reads stored in
    adata.layers['reads'].

    Implementing Datasets can disable read-based filters by setting them to `np.Inf`
    and `-np.Inf`.

    Does not annotate when `adata.layers['reads']` is not present.
    """

    _filters = AnnDataFilters(
        obs={
            "pct_mapped_reads": None,
            "pct_MT_reads": None,
            "total_reads": None,
            "pct_MT_UMIs": {"min": -np.Inf},  # get rid of this extra UMI filter.
        },
        var={"total_cells": None},
    )

    @classmethod
    def _annotate(cls, adata: AnnData) -> AnnDataAnnotations:
        annotations = AnnDataAnnotations.from_other(adata)
        if (reads := adata.layers.get("reads", None)) is not None:
            assert hasattr(adata.obs, "total_reads")
            # read annotations of genes
            annotations.var["mapped_reads"] = np.asarray(reads.sum(0)).squeeze()

            # read annotations of cells
            annotations.obs["mapped_reads"] = np.asarray(reads.sum(1)).squeeze()
            annotations.obs["pct_mapped_reads"] = (
                annotations.obs["mapped_reads"] / adata.obs["total_reads"]
            )
            annotations.obs["MT_reads"] = np.asarray(
                reads[:, annotations.var.is_MT].sum(1)
            ).squeeze()
            annotations.obs["pct_MT_reads"] = (
                annotations.obs["MT_reads"] / annotations.obs["mapped_reads"]
            )

        else:
            # do not apply read annotations
            pass

        return annotations


class TenXChromiumRNAV1(SingleCellRNASeq, Technology):
    """TenXChromiumRNAV1: SingleCellRNASeq Technology.

    Does not extend base UMI features of scRNAseq.
    """

    pass


class TenXChromiumRNAV2(SingleCellRNASeq, Technology):
    """TenXChromiumRNAV2: SingleCellRNASeq Technology.

    Does not extend base UMI features of scRNAseq.
    """

    pass


class TenXChromiumRNAV3(SingleCellRNASeq, Technology):
    """TenXChromiumRNAV3: SingleCellRNASeq Technology.

    Does not extend base UMI features of scRNAseq.
    """

    pass


class TenXChromiumRNAV3_1(SingleCellRNASeq, Technology):
    """TenXChromiumRNAV3_1: SingleCellRNASeq Technology.

    Does not extend base UMI features of scRNAseq.
    """

    pass


###################################################
###                   SNPs                      ###
###################################################


class SingleNucleotidePolymorphism(Modality, Technology):
    _filters = AnnDataFilters(obs={"total_SNPs": None}, var={"total_obs": None})

    @classmethod
    def _annotate(cls, adata: AnnData) -> AnnDataAnnotations:
        annotations = AnnDataAnnotations.from_other(adata)
        annotations.obs["total_SNPs"] = np.asarray(nz_along(adata.X, axis=1))
        annotations.obs["total_counts"] = np.asarray(adata.X.sum(1)).squeeze()
        annotations.var["total_obs"] = np.asarray(nz_along(adata.X, axis=0))
        annotations.var["total_counts"] = np.asarray(adata.X.sum(0)).squeeze()

        return annotations


###################################################
#   Spatial Transcriptomics and technologies      #
###################################################


class SpatialTranscriptomics(Modality):
    _filters = AnnDataFilters(obs={"total_genes": None}, var={"total_obs": None})
    _filters = AnnDataFilters(
        obs={"total_genes": {"min": 20}}, var={"total_obs": {"min": 100}}
    )  # added this to generalize to all spatial transcriptomics datasets.

    @classmethod
    def _annotate(cls, adata: AnnData) -> AnnDataAnnotations:
        annotations = AnnDataAnnotations.from_other(adata)
        annotations.obs["total_genes"] = np.asarray(nz_along(adata.X, axis=1))
        annotations.obs["total_counts"] = np.asarray(adata.X.sum(1)).squeeze()
        annotations.var["total_obs"] = np.asarray(nz_along(adata.X, axis=0))
        annotations.var["total_counts"] = np.asarray(adata.X.sum(0)).squeeze()

        return annotations


class CosMx(SpatialTranscriptomics, Technology):
    _technology_citation = (
        "@article{he2022high,\n"
        "   title={High-plex imaging of RNA and proteins at subcellular resolution in "
        "fixed tissue by spatial molecular imaging},\n"
        "   author={He, Shanshan and Bhatt, Ruchir and Brown, Carl and Brown, Emily A "
        "and Buhr, Derek L and Chantranuvatana, Kan and Danaher, Patrick and "
        "Dunaway, Dwayne and Garrison, Ryan G and Geiss, Gary and others}, \n"
        "   journal={Nature Biotechnology},\n"
        "   volume={40},\n"
        "   number={12},\n"
        "   pages={1794--1806},\n"
        "   year={2022},\n"
        "   publisher={Nature Publishing Group US New York}\n"
        "}"
    )


class DBiTSeq(SpatialTranscriptomics, Technology):
    _technology_citation = (
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


class SpatialTranscriptomicsV1(SpatialTranscriptomics, Technology):
    _technology_citation = (
        "@article{staahl2016visualization,\n"
        "   title={Visualization and analysis of gene expression in tissue sections by "
        "spatial transcriptomics},\n"
        "   author={St{\aa}hl, Patrik L and Salm{'e}n, Fredrik and Vickovic, Sanja and"
        " Lundmark, Anna and Navarro, Jos{'e} Fern{'a}ndez and Magnusson, Jens and "
        "Giacomello, Stefania and Asp, Michaela and Westholm, Jakub O and Huss, "
        "Mikael and others},\n"
        "   journal={Science},"
        "   volume={353},"
        "   number={6294},"
        "   pages={78--82},"
        "   year={2016},"
        "   publisher={American Association for the Advancement of Science}"
        "}"
    )


class SeqFISHPlus(SpatialTranscriptomics, Technology):
    _technology_citation = (
        "@article{eng2019transcriptome,\n"
        "   title={Transcriptome-scale super-resolved imaging in tissues by RNA "
        "seqFISH+},\n"
        "   author={Eng, Chee-Huat Linus and Lawson, Michael and Zhu, Qian and "
        "Dries, Ruben and Koulena, Noushin and Takei, Yodai and Yun, Jina and "
        "Cronin, Christopher and Karp, Christoph and Yuan, Guo-Cheng and others},\n"
        "   journal={Nature},\n"
        "   volume={568},\n"
        "   number={7751},\n"
        "   pages={235--239},\n"
        "   year={2019},\n"
        "   publisher={Nature Publishing Group}\n"
        "}"
    )


class TenXVisium(SpatialTranscriptomics, Technology):
    def _process_raw_data_10X(self) -> AnnData:
        """_process_raw_data_10X process h5 files obtained from 10X.

        Returns
        -------
        ad.AnnData
            AnnData object with all the data from the 10X Visium dataset.
        """ """"""
        adata = {
            path.stem: sc.read_10x_h5(str(path))
            for path in self.raw_files_paths.values()
        }
        if len(adata) > 1:
            # merge all the adata objects into one
            for section_name, adat in adata.items():
                adat.obs["section"] = int(section_name[-1])
                adat.X = csr_matrix(adat.X, dtype=int)
                adat.var_names_make_unique()
                adat.obs_names_make_unique()
            adata = ad.concat(adata.values())
            adata.obs_names_make_unique()
        else:
            adat = list(adata.values())[0]
            adat.X = csr_matrix(adat.X, dtype=int)
            adat.var_names_make_unique()
            adat.obs_names_make_unique()
            adata = adat
        return adata
