from numbers import Number
import numpy as np
import anndata as ad
from anndata import AnnData
import scanpy as sc
from scipy.sparse import csr_matrix

from bipca.utils import nz_along
from bipca.experiments.base import abstractmethod
from bipca.experiments.datasets.base import Modality, Technology, DataFilters

###################################################
#   Simulations                                   #
###################################################


class Simulation(Modality, Technology):
    _citation = None
    _raw_urls = None
    _unfiltered_urls = {"simulation.h5ad": None}
    _filters = DataFilters(
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
        self.mrows = mrows
        self.ncols = ncols
        super().__init__(
            store_filtered_data=store_filtered_data,
            store_unfiltered_data=store_unfiltered_data,
            store_raw_files=False,
            **kwargs,
        )

    @classmethod
    def _annotate(cls, adata: AnnData) -> AnnData:
        adata.obs["total_nz"] = nz_along(adata.X, 1)
        adata.var["total_nz"] = nz_along(adata.X, 0)
        return adata

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

    _filters = DataFilters(
        obs={"total_sites": None},  # these are from the episcanpy tutorial.
        var={"total_cells": None},
    )

    @classmethod
    def _annotate(cls, adata: AnnData) -> AnnData:
        adata.obs["total_sites"] = nz_along(adata.X, 1)
        adata.obs["total_peaks"] = np.asarray(adata.X.sum(1)).squeeze()
        adata.var["total_cells"] = nz_along(adata.X, 0)
        adata.var["total_peaks"] = np.asarray(adata.X.sum(0)).squeeze()
        return adata


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


###################################################
#   scRNAseq and technologies                     #
###################################################


class SingleCellRNASeq(Modality):
    """SingleCellRNASeq: Base Modality subclass for variants of SingleCellRNASeq.

    Implements SingleCellRNASeq specific annotations and filters for UMIs and gene names.

    """

    _filters = DataFilters(
        obs={"total_genes": None, "pct_MT_UMIs": None}, var={"total_cells": None}
    )

    @classmethod
    def _annotate(cls, adata: AnnData) -> AnnData:
        try:
            gn = adata.var["gene_name"]
        except Exception:
            gn = adata.var_names
        # gene annotations
        adata.var["total_UMIs"] = np.asarray(adata.X.sum(0)).squeeze()
        adata.var["total_cells"] = nz_along(adata.X, 0)
        adata.var["is_MT"] = gn.str.lower().str.startswith("mt-")
        # cell annotations
        adata.obs["total_UMIs"] = np.asarray(adata.X.sum(1)).squeeze()
        adata.obs["total_MT_UMIs"] = np.asarray(adata[:, adata.var.is_MT].X.sum(1))
        adata.obs["pct_MT_UMIs"] = adata.obs["total_MT_UMIs"] / adata.obs["total_UMIs"]
        adata.obs["total_genes"] = nz_along(adata.X, 1)
        return adata


class DropSeq(SingleCellRNASeq, Technology):
    pass


class SmartSeqV3(SingleCellRNASeq, Technology):
    """SmartSeqV3: SingleCellRNASeq technology with support for read-based features.

    Implements SmartSeqV3 specific annotations and filters for reads stored in
    adata.layers['reads'].

    Implementing Datasets can disable read-based filters by setting them to `np.Inf`
    and `-np.Inf`.

    Does not annotate when `adata.layers['reads']` is not present.
    """

    _filters = DataFilters(
        obs={
            "pct_mapped_reads": None,
            "pct_MT_reads": None,
            "total_reads": None,
            "pct_MT_UMIs": {"min": -np.Inf},  # get rid of this extra UMI filter.
        },
        var={"total_cells": None},
    )

    @classmethod
    def _annotate(cls, adata: AnnData) -> AnnData:
        if (reads := adata.layers.get("reads", None)) is not None:
            assert hasattr(adata.obs, "total_reads")
            # read annotations of genes
            adata.var["mapped_reads"] = np.asarray(reads.sum(0)).squeeze()

            # read annotations of cells
            adata.obs["mapped_reads"] = np.asarray(reads.sum(1)).squeeze()
            adata.obs["pct_mapped_reads"] = (
                adata.obs["mapped_reads"] / adata.obs["total_reads"]
            )
            adata.obs["MT_reads"] = np.asarray(
                reads[:, adata.var.is_MT].sum(1)
            ).squeeze()
            adata.obs["pct_MT_reads"] = (
                adata.obs["MT_reads"] / adata.obs["mapped_reads"]
            )

        else:
            # do not apply read annotations
            pass

        return adata


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
    _filters = DataFilters(obs={"total_SNPs": None}, var={"total_obs": None})

    @classmethod
    def _annotate(cls, adata: AnnData) -> AnnData:
        adata.obs["total_SNPs"] = nz_along(adata.X, axis=1)
        adata.obs["total_counts"] = np.asarray(adata.X.sum(1)).squeeze()
        adata.var["total_obs"] = nz_along(adata.X, axis=0)
        adata.var["total_counts"] = np.asarray(adata.X.sum(0)).squeeze()

        return adata


###################################################
#   Spatial Transcriptomics and technologies      #
###################################################


class SpatialTranscriptomics(Modality):
    _filters = DataFilters(obs={"total_genes": None}, var={"total_obs": None})
    _filters = DataFilters(
        obs={"total_genes": {"min": 20}}, var={"total_obs": {"min": 100}}
    )  # added this to generalize to all spatial transcriptomics datasets.

    @classmethod
    def _annotate(cls, adata: AnnData) -> AnnData:
        adata.obs["total_genes"] = nz_along(adata.X, axis=1)
        adata.obs["total_counts"] = np.asarray(adata.X.sum(1)).squeeze()
        adata.var["total_obs"] = nz_along(adata.X, axis=0)
        adata.var["total_counts"] = np.asarray(adata.X.sum(0)).squeeze()

        return adata


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
