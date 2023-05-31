import os, errno
import itertools
import subprocess
from pathlib import Path
from functools import partial, singledispatch
from typing import Dict, Union, Optional, List


import numpy as np
import pandas as pd
from anndata import AnnData, read_h5ad
import scanpy as sc
from scipy import sparse
from scipy.io import mmwrite
from scipy.stats import zscore
from ALRA import ALRA

import bipca
from bipca import BiPCA
from bipca.utils import issparse
from bipca.plotting import set_spine_visibility

from bipca.experiments.figures.base import (
    Figure,
    is_subfigure,
    plots,
    label_me,
    plt,
    mpl,
    algorithm_to_npg_cmap_index,
)
from .utils import (
    parameter_estimation_plot,
    compute_minor_log_ticks,
    npg_cmap,
    compute_axis_limits,
)
import bipca.experiments.datasets as bipca_datasets
from bipca.experiments.experiments import log1p


def extract_dataset_parameters(
    dataset: bipca.experiments.datasets.base.Dataset,
    samples: Optional[List[str]] = None,
    **bipca_kwargs,
):
    if "seed" not in bipca_kwargs:
        bipca_kwargs["seed"] = 42
    if "verbose" not in bipca_kwargs:
        bipca_kwargs["verbose"] = 0
    if "n_components" not in bipca_kwargs:
        bipca_kwargs["n_components"] = -1
    if "backend" not in bipca_kwargs:
        bipca_kwargs["backend"] = "torch"
    # get the parameters of a dataset
    name = dataset.__class__.__name__
    modality = dataset.modality
    technology = dataset.technology
    if samples is None:
        samples = dataset.samples
    else:
        samples = [sample for sample in dataset.samples if sample in samples]

    adatas = dataset.get_unfiltered_data(samples=samples)
    parameters = []

    for sample in samples:
        unfiltered_N, unfiltered_M = adatas[sample].shape
        filtered_adata = dataset.filter(dataset.annotate(adatas[sample]))
        filtered_N, filtered_M = filtered_adata.shape
        if issparse(filtered_adata.X):
            X = filtered_adata.X.toarray()
        else:
            X = filtered_adata.X
        op = BiPCA(**bipca_kwargs, logger=dataset.logger).fit(X)
        op.get_plotting_spectrum()

        parameters.append(
            {
                "Dataset": name,
                "Sample": sample,
                "Modality": modality,
                "Technology": technology,
                "Unfiltered # observations": unfiltered_N,
                "Unfiltered # features": unfiltered_M,
                "Filtered # observations": filtered_N,
                "Filtered # features": filtered_M,
                "Kolmogorov-Smirnov distance": op.plotting_spectrum["kst"],
                "Rank": op.mp_rank,
                "Linear coefficient (b)": op.b,
                "Quadratic coefficient (c)": op.c,
            }
        )
    return parameters


def run_all(
    csv_path="/bipca_data/results/dataset_parameters.csv", overwrite=False
) -> pd.DataFrame:
    """run_all Apply biPCA to all datasets and save the results to a csv file."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        if not overwrite:
            # extract the already written datasets / samples
            df = pd.read_csv(csv_path)
            written_datasets_samples = (
                (df["Dataset"] + "-" + df["Sample"]).str.split("-").values.tolist()
            )
    if not csv_path.exists() or overwrite:
        df = pd.DataFrame(
            columns=[
                "Dataset-Sample",
                "Dataset",
                "Sample",
                "Modality",
                "Technology",
                "Unfiltered # observations",
                "Unfiltered # features",
                "Filtered # observations",
                "Filtered # features",
                "Kolmogorov-Smirnov distance",
                "Rank",
                "Linear coefficient (b)",
                "Quadratic coefficient(c)",
            ]
        )
        df.set_index("Dataset-Sample", inplace=True)
        df.to_csv(csv_path, mode="a", header=True)
        written_datasets_samples = []
    datasets = bipca_datasets.get_all_datasets()
    for dataset in datasets:
        to_compute = []
        d = dataset()
        for sample in d.samples:
            if sample in d.hidden_samples:
                continue
            if [d.__class__.__name__, sample] in written_datasets_samples:
                continue
            else:
                print(d.__class__.__name__, sample)
                to_compute.append(sample)
        if len(to_compute) > 0:
            parameters = extract_dataset_parameters(d, samples=to_compute)
            df = pd.DataFrame(parameters)
            df["Dataset-Sample"] = df["Dataset"] + "-" + df["Sample"]
            df.set_index("Dataset-Sample", inplace=True)
            # df.drop(["Dataset", "Sample"], axis="columns")
            df.to_csv(csv_path, mode="a", header=False)
    return pd.read_csv(csv_path)


def apply_normalizations(adata_path: Union[Path, str], n_threads=32, no=[]):
    """
    adata_path: path to the input anndata file which stores the raw count as adata.X
    output_path: path to store the output adata that will store the normalized data matrices,
                 and a tmp folder that store the intermediate files from sanity
    output_adata: output name for the normalized adata
    n_threads: number of threads to run Sanity Default: 10
    no: a array of which methods not to run, including log1p, sctransform, alra, sanity Default: empty array
    """

    # Read data
    print("Loading count data ...\n")
    try:
        adata = read_h5ad(adata_path)
    except FileNotFoundError:
        print("Error: Unable to find the h5ad file")

    # convert to sparse matrix
    if issparse(adata.X):
        X = adata.X
    else:
        X = sparse.csr_matrix(adata.X)

    # If no, else run log1p
    if ("log" in no) | ("log1p" in no) | ("logtransform" in no):
        pass
    else:
        print("Running log normalization ...\n")
        adata.layers["log1p"] = log1p(X)
        adata.layers["log1p+z"] = sparse.csr_matrix(
            zscore(adata.layers["log1p"].toarray(), axis=0)
        )
    # If no, else run sctransform
    if "sct" in no:
        pass
    else:
        print("Running analytical pearson residuals ...\n")
        result_dict = sc.experimental.pp.normalize_pearson_residuals(
            adata, inplace=False
        )
        adata.layers["Pearson"] = result_dict["X"]
    # If no, else run alra
    if "alra" in no:
        pass
    else:
        print("Running ALRA ...\n")
        adata.layers["ALRA"] = ALRA(X)

    # If no, else run sanity
    if "sanity" in no:
        pass
    else:
        print("Running Sanity ...\n")
        # Mounted to where sanity is installed
        sanity_installation_path = "/Sanity/bin/Sanity"
        # Specify the temporary folder that will store the output from intermediate outputs from Sanity
        tmp_path_sanity = Path(adata_path).parent / "tmp"
        tmp_path_sanity.mkdir(parents=True, exist_ok=True)
        # write intermediate files from sanity
        sanity_counts_path = tmp_path_sanity / "count.mtx"
        sanity_cells_path = tmp_path_sanity / "barcodes.tsv"
        sanity_genes_path = tmp_path_sanity / "genes.tsv"
        mmwrite(str(sanity_counts_path), X.T)
        pd.Series(adata.obs_names).to_csv(
            sanity_cells_path,
            sep="\t",
            index=False,
            header=None,
        )
        pd.Series(adata.var_names).to_csv(
            sanity_genes_path, sep="\t", index=False, header=None
        )

        # hack the count mtx because sanity can't handle 2nd % from mmwrite
        # with open((tmp_path_sanity + "/count_tmp.mtx"), "r") as file_input:
        #    file_orig = file_input.readlines()
        #    file_rev = file_orig[::-1]
        #    with open((tmp_path_sanity + "/count.mtx"), "w") as output:
        #        output.write(file_orig[0])
        #        output.write(file_orig[2])
        #        for i,line in enumerate(file_rev[:-3]):
        #                output.write(line)

        sanity_command = (
            sanity_installation_path
            + " -f "
            + str(sanity_counts_path)
            + " -mtx_genes "
            + str(sanity_genes_path)
            + " -mtx_cells "
            + str(sanity_cells_path)
            + " -d "
            + str(tmp_path_sanity)
            + " -n "
            + str(n_threads)
        )
        subprocess.run(sanity_command.split())
        # print(error)
        sanity_output = tmp_path_sanity / "log_transcription_quotients.txt"
        adata.layers["Sanity"] = (
            pd.read_csv(
                sanity_output,
                sep="\t",
                index_col=0,
            )
            .to_numpy()
            .T
        )

    adata.write(adata_path)
    return adata


class Figure2(Figure):
    _figure_layout = [
        ["A", "A", "B", "B", "C", "C", "D", "E", "F"],
        ["G", "G", "G", "H", "H", "H", "I", "I", "I"],
        ["K", "K", "K", "L", "L", "L", "M", "M", "M"],
    ]

    def __init__(
        self,
        seed=42,
        mrows=10000,
        ncols=5000,
        ranks=2 ** np.arange(0, 10),
        bs=2.0 ** np.arange(-7, 7),
        cs=2.0 ** np.arange(-7, 7),
        n_iterations=10,
        *args,
        **kwargs,
    ):
        self.seed = seed
        self.mrows = mrows
        self.ncols = ncols
        self.ranks = ranks
        self.bs = bs
        self.cs = cs
        self.n_iterations = n_iterations
        self.results = {}
        self.printing_params = [
            "mrows",
            "ncols",
            "ranks",
            "seed",
            "bs",
            "cs",
            "n_iterations",
        ]
        super().__init__(*args, **kwargs)

    def parameter_estimation_plot(self, axis, results):
        axis = parameter_estimation_plot(
            axis,
            results,
            mean=True,
            errorbars=False,
            scatter_kwargs=dict(
                marker="o",
                s=10,
                facecolor=npg_cmap(0.85)(algorithm_to_npg_cmap_index["BiPCA"]),
                linewidth=0.1,
                edgecolor="k",
            ),
        )

        axis.set_xscale("log", **{"base": 2})
        axis.set_yscale("log", **{"base": 2})

        set_spine_visibility(axis, which=["top", "right"], status=False)
        # set the axis limits
        axis.set_aspect("equal")
        axis.set_box_aspect(1)
        xlim = compute_axis_limits(results["x"], "log", {"base": 2})
        ylim = compute_axis_limits(results["y"], "log", {"base": 2})
        lim_min = min(xlim[0], ylim[0])
        lim_max = max(xlim[1], ylim[1])
        axis.set_xlim([lim_min, lim_max])

        axis.set_ylim([lim_min, lim_max])
        return axis

    @property
    def parameters(self) -> str:
        """parameters Print the parameters of the figure."""
        return {
            param: param_value
            for param in self.printing_params
            if (param_value := getattr(self, param, None)) is not None
        }

    @is_subfigure(label="A")
    def compute_A(self):
        """compute_A Generate subfigure 2A, simulating the rank recovery in BiPCA."""
        seeds = [self.seed + i for i in range(self.n_iterations)]
        FixedPoisson = partial(
            bipca_datasets.RankRPoisson,
            mean=4,
            mrows=self.mrows,
            ncols=self.ncols,
            verbose=0,
        )
        parameters = itertools.product(self.ranks, seeds)
        datasets = map(
            lambda ele: FixedPoisson(rank=ele[0], seed=ele[1]).get_filtered_data()[
                "simulation"
            ],
            parameters,
        )
        results = np.array(
            list(
                map(
                    lambda x: (
                        x.uns["rank"],
                        BiPCA(backend="torch", n_components=-1, seed=42, verbose=0)
                        .fit(x.X)
                        .mp_rank,
                    ),
                    datasets,
                ),
            )
        )
        return results

    @is_subfigure(label="A")
    @plots
    @label_me
    def plot_A(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        """plot_A Plot the results of subfigure 2A."""
        assert "A" in self.results
        results = {"x": self.results["A"][:, 0], "y": self.results["A"][:, 1]}
        axis = self.parameter_estimation_plot(axis, results)
        yticks = [
            2**i
            for i in np.arange(
                np.min(np.log2(self.ranks).astype(int)),
                max(np.log2(self.ranks).astype(int)) + 1,
            )
        ]
        yticklabels = [
            rf"${{{i}}}$" if i % 2 == 0 else None
            for i in np.arange(
                np.min(np.log2(self.ranks).astype(int)),
                max(np.log2(self.ranks).astype(int)) + 1,
            )
        ]
        minorticks = compute_minor_log_ticks(yticks, 2)

        axis.set_yticks(yticks, labels=yticklabels, minor=False)
        axis.set_yticks(minorticks, minor=True)
        axis.set_xticks(yticks, labels=yticklabels, minor=False)
        axis.set_xticks(minorticks, minor=True)
        axis.set_xlabel(r"$r$ ($\mathrm{log}_2$)", wrap=True)
        axis.set_ylabel(r"$\hat{r}$ ($\mathrm{log}_2$)", wrap=True)

        return axis

    @is_subfigure(label="B")
    def compute_B(self):
        """compute_B Generate subfigure 2B, simulating the recovery of
        QVF parameter b in BiPCA."""

        seeds = [self.seed + i for i in range(self.n_iterations)]
        # use partial to make a QVFNegativeBinomial with fixed rank, c, and mean
        FixedNegativeBinomial = partial(
            bipca_datasets.QVFNegativeBinomial,
            rank=1,
            c=0.000001,
            mean=1000,
            mrows=self.mrows,
            ncols=self.ncols,
            verbose=0,
        )
        # generate the parameter set as combinations of b and seeds
        parameters = itertools.product(self.bs, seeds)

        # map the experiment over the parameters
        datasets = map(
            lambda ele: FixedNegativeBinomial(
                b=ele[0], seed=ele[1]
            ).get_filtered_data()["simulation"],
            parameters,
        )

        # run biPCA on the datasets, extract the b parameter, and store the results
        results = np.array(
            list(
                map(
                    lambda x: (
                        x.uns["b"],
                        BiPCA(backend="torch", seed=42, verbose=0, n_components=-1)
                        .fit(x.X)
                        .b,
                    ),
                    datasets,
                ),
            )
        )
        return results

    @is_subfigure(label="B")
    @plots
    @label_me
    def plot_B(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        """plot_B Plot the results of subfigure 2B."""
        assert "B" in self.results
        results = {"x": self.results["B"][:, 0], "y": self.results["B"][:, 1]}

        axis = self.parameter_estimation_plot(
            axis,
            results,
        )
        yticks = [
            2.0**i
            for i in np.arange(
                np.min(np.log2(self.bs).astype(int)),
                max(np.log2(self.bs).astype(int)) + 1,
            )
        ]
        yticklabels = [
            rf"${{{i}}}$" if i % 2 == 0 else None
            for i in np.arange(
                np.min(np.log2(self.bs).astype(int)),
                max(np.log2(self.bs).astype(int)) + 1,
            )
        ]

        minorticks = compute_minor_log_ticks(yticks, 2)
        axis.set_yticks(yticks, labels=yticklabels, minor=False)
        axis.set_yticks(minorticks, labels=[None for _ in minorticks], minor=True)
        axis.set_xticks(yticks, labels=yticklabels, minor=False)
        axis.set_xticks(minorticks, labels=[None for _ in minorticks], minor=True)
        axis.set_xlabel(r"$b$ ($\mathrm{log}_2$)", wrap=True)
        axis.set_ylabel(r"$\hat{b}$ ($\mathrm{log}_2$)", wrap=True)
        return axis

    @is_subfigure(label="C")
    def compute_C(self):
        """compute_C generate subfigure 2C, simulating the recovery of QVF
        parameter c"""
        # generate seeds
        seeds = [self.seed + i for i in range(self.n_iterations)]
        # use partial to make a QVFNegativeBinomial with fixed rank, b, and mean
        FixedNegativeBinomial = partial(
            bipca_datasets.QVFNegativeBinomial,
            rank=1,
            b=1,
            mean=1000,
            mrows=self.mrows,
            ncols=self.ncols,
            verbose=0,
        )
        # generate the parameter set as combinations of c and seeds
        parameters = itertools.product(self.cs, seeds)
        # map the experiment over the parameters
        datasets = map(
            lambda ele: FixedNegativeBinomial(
                c=ele[0], seed=ele[1]
            ).get_filtered_data()["simulation"],
            parameters,
        )

        # run biPCA on the datasets, extract the c parameter, and store the results
        results = np.array(
            list(
                map(
                    lambda x: (
                        x.uns["c"],
                        BiPCA(backend="torch", seed=42, n_components=-1, verbose=0)
                        .fit(x.X)
                        .c,
                    ),
                    datasets,
                ),
            )
        )

        return results

    @is_subfigure(label="C")
    @plots
    @label_me
    def plot_C(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        """plot_C Plot the results of subfigure 2C."""
        assert "C" in self.results
        results = {"x": self.results["C"][:, 0], "y": self.results["C"][:, 1]}

        axis = self.parameter_estimation_plot(
            axis,
            results,
        )
        yticks = [
            2.0**i
            for i in np.arange(
                np.min(np.log2(self.cs).astype(int)),
                max(np.log2(self.cs).astype(int) + 1),
            )
        ]
        yticklabels = [
            rf"${{{i}}}$" if i % 2 == 0 else None
            for i in np.arange(
                np.min(np.log2(self.cs).astype(int)),
                max(np.log2(self.cs).astype(int) + 1),
            )
        ]
        minorticks = compute_minor_log_ticks(yticks, 2)
        axis.set_yticks(yticks, labels=yticklabels, minor=False)
        axis.set_xticks(yticks, labels=yticklabels, minor=False)
        axis.set_yticks(minorticks, labels=[None for _ in minorticks], minor=True)
        axis.set_xticks(minorticks, labels=[None for _ in minorticks], minor=True)
        axis.set_xlabel(r"$c$ ($\mathrm{log}_2$)", wrap=True)
        axis.set_ylabel(
            r"$\hat{c}$ ($\mathrm{log}_2$)",
            wrap=True,
        )
        return axis

    @is_subfigure(label=["D", "E", "F"])
    def compute_D_E_F(self):
        datasets = [
            bipca_datasets.TenX2016PBMC,  # 10xV1
            bipca_datasets.TenX2021PBMC,  # 10xV3
            bipca_datasets.HagemannJensen2022,  # Smartseq3
            bipca_datasets.TenX2022MouseBrain,  # visium
            bipca_datasets.Asp2019,  # spatial transcriptomics
            bipca_datasets.Buenrostro2018ATAC,  # Buenrostro ATAC
            bipca_datasets.TenX2022PBMCATAC,  # 10x ATAC v1.1
        ]
        seeds = [self.seed + i for i in range(self.n_iterations)]
        rngs = list(map(lambda seed: np.random.default_rng(seed), seeds))
        r = np.ndarray((len(datasets), self.n_iterations + 2), dtype=object)
        b = np.ndarray((len(datasets), self.n_iterations + 2), dtype=object)
        c = np.ndarray((len(datasets), self.n_iterations + 2), dtype=object)

        def subset_data(adata, prct, rng):
            n = int(prct * adata.shape[0])
            inds = rng.permutation(adata.shape[0])[:n]
            return adata[inds, :]

        for dset_ix, dataset in enumerate(datasets):
            data_operator = dataset(base_data_directory=self.base_plot_directory)
            adata = data_operator.get_unfiltered_data(samples="full")["full"]
            # get the dataset name
            name = data_operator.__class__.__name__
            r[dset_ix, 0] = name
            b[dset_ix, 0] = name
            c[dset_ix, 0] = name
            for seed_ix, (rng, seed_n) in enumerate(zip(rngs, seeds)):
                # subset the data
                adata_sub = subset_data(adata, 0.75, rng)
                adata_sub = data_operator.filter(adata_sub, n_filter_iters=3)
                # run biPCA
                if issparse(adata_sub.X):
                    X = adata_sub.X.toarray()
                else:
                    X = adata_sub.X
                op = BiPCA(
                    backend="torch",
                    seed=seed_n,
                    n_components=-1,
                    verbose=0,
                    logger=data_operator.logger,
                ).fit(X)
                # store the results
                r[dset_ix, seed_ix + 2] = op.mp_rank
                b[dset_ix, seed_ix + 2] = op.b
                c[dset_ix, seed_ix + 2] = op.c
            # run biPCA on the full data
            adata = data_operator.get_filtered_data(samples="full")["full"]
            if issparse(adata.X):
                X = adata.X.toarray()
            else:
                X = adata.X
            op = BiPCA(
                backend="torch",
                seed=42,
                n_components=-1,
                verbose=0,
                logger=data_operator.logger,
            ).fit(X)
            r[dset_ix, 1] = op.mp_rank
            b[dset_ix, 1] = op.b
            c[dset_ix, 1] = op.c
        results = {"D": r, "E": b, "F": c}
        return results

    @is_subfigure("D")
    @plots
    @label_me
    def plot_D(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        # rank plot w/ resampling
        pass

    @is_subfigure("E")
    @plots
    @label_me
    def plot_E(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        # b plot w/ resampling
        pass

    @is_subfigure("F")
    @plots
    @label_me
    def plot_F(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        # c plot w/ resampling
        pass

    @is_subfigure(label=["G", "H", "I"])
    def compute_G_H_I(self):
        df = run_all(
            csv_path=self.base_plot_directory / "results" / "dataset_parameters.csv"
        )
        G = np.ndarray((len(df), 3), dtype=object)
        G[:, 0] = df.loc[:, "Modality"].values
        G[:, 1] = df.loc[:, "Linear coefficient (b)"].values
        G[:, 2] = df.loc[:, "Quadratic coefficient (c)"].values
        H = np.ndarray((len(df), 4), dtype=object)
        H[:, 0] = df.loc[:, "Modality"].values
        H[:, 1] = df.loc[:, "Filtered # observations"].values
        H[:, 2] = df.loc[:, "Filtered # features"].values
        H[:, 3] = df.loc[:, "Rank"].values
        I = np.ndarray((len(df), 3), dtype=object)
        I[:, 0] = df.loc[:, "Modality"].values
        I[:, 1] = df.loc[:, "Rank"].values / df.loc[
            :, ["Filtered # observations", "Filtered # features"]
        ].values.min(1)
        I[:, 2] = df.loc[:, "Kolmogorov-Smirnov distance"].values
        results = {"G": G, "H": H, "I": I}
        return results

    @is_subfigure("G")
    @plots
    @label_me
    def plot_G(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        # c plot w/ resampling
        assert "G" in self.results
        df = pd.DataFrame(self.results["G"], columns=["Modality", "b", "c"])
        label_mapping = {
            "SingleCellRNASeq": "scRNA-seq",
            "SingleCellATACSeq": "scATAC-seq",
            "SpatialTranscriptomics": "Spatial Transcriptomics",
            "SingleNucleotidePolymorphism": "Genomics",
        }
        groups = df.groupby("Modality").groups
        for ix, (key, label) in enumerate(label_mapping.items()):
            group_inds = groups[key]
            dg = df.loc[group_inds, :]
            x, y = dg.loc[:, ["b", "c"]].values.T
            c = npg_cmap(0.85)(ix)
            axis.scatter(
                x[0],
                y[0],
                label=label,
                s=10,
                color=c,
                marker="o",
                linewidth=0.1,
                edgecolor="k",
            )  # plot the first point for the legend labeling
            # add the color
            df.loc[group_inds, ["r", "g", "bb", "a"]] = c
        # shuffle the points
        idx = np.random.default_rng(self.seed).permutation(df.shape[0])
        df_shuffled = df.iloc[idx, :]
        x, y = df_shuffled.loc[:, ["b", "c"]].values.T

        c = df_shuffled.loc[:, ["r", "g", "bb", "a"]].values
        axis.scatter(
            x,
            y,
            s=10,
            c=c,
            marker="o",
            linewidth=0.1,
            edgecolor="k",
        )
        axis.set_yscale("symlog", linthresh=1e-5)
        yticks = [0] + [10**i for i in range(-5, 3)]
        axis.set_yticks(
            yticks,
            labels=[
                0,
                r"$10^{-5}$",
                None,
                r"$10^{-3}$",
                None,
                r"$10^{-1}$",
                None,
                r"$10^{1}$",
                None,
            ],
        )

        xticks = [0, 0.5, 1.0, 1.5, 2.0]
        axis.set_xticks(
            xticks,
            labels=[0, 0.5, 1, 1.5, 2],
        )
        axis.set_yticks(
            mpl.ticker.LogLocator(base=10, subs="auto").tick_values(1e-4, 10),
            minor=True,
        )
        axis.set_ylim(-3e-6, 100)

        axis.set_xlabel(r"Estimated linear variance $\hat{b}$")
        axis.set_ylabel(r"Estimated quadratic variance $\hat{c}$")
        set_spine_visibility(axis, which=["top", "right"], status=False)
        axis.legend()
        return axis

    @is_subfigure("H")
    @plots
    @label_me
    def plot_H(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        # rank vs number of features and observations
        assert "H" in self.results
        df = pd.DataFrame(
            self.results["H"],
            columns=["Modality", "# observations", "# features", "rank"],
        )
        label_mapping = {
            "SingleCellRNASeq": "scRNA-seq",
            "SingleCellATACSeq": "scATAC-seq",
            "SpatialTranscriptomics": "Spatial Transcriptomics",
            "SingleNucleotidePolymorphism": "Genomics",
        }
        groups = df.groupby("Modality").groups

        axis.set_xticks([])
        axis.set_yticks([])
        set_spine_visibility(axis, status=False)

        lower_axis = axis.inset_axes([0, 0, 1, 0.4], zorder=2)
        upper_axis = axis.inset_axes([0, 0.6, 1, 0.4], zorder=2)

        for ix, (key, label) in enumerate(label_mapping.items()):
            group_inds = groups[key]
            dg = df.iloc[group_inds, :]
            lower_x = dg.loc[:, ["# observations"]].values
            upper_x = dg.loc[:, ["# features"]].values
            c = npg_cmap(0.85)(ix)

            y = dg.loc[:, "rank"].values
            lower_axis.scatter(
                lower_x[0],
                y[0],
                label=label,
                s=10,
                color=c,
                marker="o",
                linewidth=0.1,
                edgecolor="k",
            )
            upper_axis.scatter(
                upper_x[0],
                y[0],
                label=label,
                s=10,
                color=c,
                marker="o",
                linewidth=0.1,
                edgecolor="k",
            )  # plot the first point for the legend labeling
            # add the color
            df.loc[group_inds, ["r", "g", "bb", "a"]] = c
        # shuffle the points
        idx = np.random.default_rng(self.seed).permutation(df.shape[0])
        df_shuffled = df.iloc[idx, :]

        lower_x = df_shuffled.loc[:, ["# observations"]].values
        upper_x = df_shuffled.loc[:, ["# features"]].values
        y = df_shuffled.loc[:, "rank"].values
        c = df_shuffled.loc[:, ["r", "g", "bb", "a"]].values
        for x, ax in zip([lower_x, upper_x], [lower_axis, upper_axis]):
            ax.scatter(
                x,
                y,
                s=10,
                c=c,
                marker="o",
                linewidth=0.1,
                edgecolor="k",
            )
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlabel(r"\# features" if ax == upper_axis else r"\# observations")
            ax.set_ylabel(r"Estimated rank $\hat{r}$")
            set_spine_visibility(ax, which=["top", "right"], status=False)

        return axis

    @is_subfigure("I")
    @plots
    @label_me
    def plot_I(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        df = pd.DataFrame(
            self.results["I"],
            columns=["Modality", "r/m", "KS"],
        )
        label_mapping = {
            "SingleCellRNASeq": "scRNA-seq",
            "SingleCellATACSeq": "scATAC-seq",
            "SpatialTranscriptomics": "Spatial Transcriptomics",
            "SingleNucleotidePolymorphism": "Genomics",
        }
        groups = df.groupby("Modality").groups
        for ix, (key, _) in enumerate(label_mapping.items()):
            group_inds = groups[key]
            c = npg_cmap(0.85)(ix)
            # add the color
            df.loc[group_inds, ["r", "g", "bb", "a"]] = c
        # shuffle the points
        idx = np.random.default_rng(self.seed).permutation(df.shape[0])
        df_shuffled = df.iloc[idx, :]
        x = df_shuffled.loc[:, "r/m"].values
        y = df_shuffled.loc[:, "KS"].values
        c = df_shuffled.loc[:, ["r", "g", "bb", "a"]].values
        axis.scatter(
            x,
            y,
            s=10,
            c=c,
            marker="o",
            linewidth=0.1,
            edgecolor="k",
        )
        axis.set_yscale("log")
        axis.set_xscale("log")
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()

        max_pt = np.maximum(xlim[1], ylim[1])
        min_pt = np.minimum(xlim[0], ylim[0])
        axis.plot(
            [min_pt, max_pt],
            [min_pt, max_pt],
            linewidth=1,
            linestyle="--",
            color="k",
            label="Optimal KS",
        )
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        set_spine_visibility(axis, which=["top", "right"], status=False)

        axis.set_xlabel(r"Estimated rank fraction $\frac{\hat{r}}{m}$")
        axis.set_ylabel(r"Kolmogorov-Smirnov distance")
        axis.legend()
        return axis
