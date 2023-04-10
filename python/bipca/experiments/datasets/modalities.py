import numpy as np
from anndata import AnnData

from bipca.utils import nz_along
from bipca.experiments.datasets.base import Modality, Technology, DataFilters


###################################################
### Hi-C                                        ###
###################################################


class ChromatinConformationCapture(Modality, Technology):
    pass


###################################################
### scATACseq and technologies                  ###
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
### scRNAseq and technologies                   ###
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


###################################################
###                   SNPs                      ###
###################################################


class SingleNucleotidePolymorphism(Modality, Technology):
    pass


###################################################
### Spatial Transcriptomics and technologies    ###
###################################################


class SpatialTranscriptomics(Modality):
    _filters = DataFilters(obs={"total_genes": None}, var={"total_obs": None})

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
        "title={High-plex imaging of RNA and proteins at subcellular resolution in "
        "fixed tissue by spatial molecular imaging},\n"
        "author={He, Shanshan and Bhatt, Ruchir and Brown, Carl and Brown, Emily A "
        "and Buhr, Derek L and Chantranuvatana, Kan and Danaher, Patrick and "
        "Dunaway, Dwayne and Garrison, Ryan G and Geiss, Gary and others}, \n"
        """journal={Nature Biotechnology},
        volume={40},
        number={12},
        pages={1794--1806},
        year={2022},
        publisher={Nature Publishing Group US New York}
        }"""
    )


class DBiTSeq(SpatialTranscriptomics, Technology):
    pass


class SpatialTranscriptomicsTechnology(SpatialTranscriptomics, Technology):
    pass


class SeqFishPlus(SpatialTranscriptomics, Technology):
    pass


class TenXVisium(SpatialTranscriptomics, Technology):
    pass
