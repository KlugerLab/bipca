import os, errno
import itertools
import subprocess
from pathlib import Path
from functools import partial, singledispatch
from typing import Dict, Union, Optional, List, Any


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


import bipca.experiments.datasets as bipca_datasets
from bipca.experiments.experiments import log1p

from tqdm.contrib.concurrent import thread_map,process_map
import torch
from threadpoolctl import threadpool_limits
from sklearn.utils.extmath import cartesian

from collections import OrderedDict

from sklearn.metrics import balanced_accuracy_score
from bipca.experiments.figures.figures import apply_normalizations
from bipca.experiments import rank_to_sigma
from bipca.experiments.experiments import knn_classifier


from .base import (
    Figure,
    is_subfigure,
    plots,
    label_me,
    plt,
    mpl,
)
from .utils import (
    parameter_estimation_plot,
    compute_minor_log_ticks,
    compute_axis_limits,
    plot_y_equals_x,
    npg_cmap,
)
from .plotting_constants import (
    algorithm_color_index,
    algorithm_fill_color,
    modality_color_index,
    modality_fill_color,
    modality_label,
    dataset_label,
)


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
    dataset_args = dataset.bipca_kwargs.copy()
    for sample in dataset_args.keys():
        for key in bipca_kwargs.keys():
            if key in dataset_args[sample]:
                pass  # don't overwrite.
            else:
                dataset_args[sample][key] = bipca_kwargs[key]
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
        op = BiPCA(**dataset_args[sample], logger=dataset.logger).fit(X)
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
    csv_path="/bipca_data/results/dataset_parameters.csv",
    overwrite=False,
    logger=None,
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
        d = dataset(logger=logger)
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
        adata.layers["ALRA"] = ALRA(log1p(X))

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
        ["G", "G", "G", "J", "J", "J", "I", "I", "I"],
        ["N", "N", "N", "O", "O", "O", "P", "P", "P"],
    ]

    def __init__(
        self,
        seed=42,
        mrows=5120,
        ncols=5120,
        minimum_singular_value=True,
        constant_singular_value=True,
        entrywise_mean=100,
        libsize_mean=1000,
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
        self.minimum_singular_value = minimum_singular_value
        self.constant_singular_value = constant_singular_value
        self.entrywise_mean = entrywise_mean
        self.libsize_mean = libsize_mean
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
            "minimum_singular_value",
            "constant_singular_value",
            "entrywise_mean",
            "libsize_mean",
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
                facecolor=algorithm_fill_color["BiPCA"],
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

    @property
    def fixed_simulation_args(self) -> dict[str, Any]:
        return {
            "mrows": self.mrows,
            "ncols": self.ncols,
            "entrywise_mean": self.entrywise_mean,
            "libsize_mean": self.libsize_mean,
            "minimum_singular_value": self.minimum_singular_value,
            "constant_singular_value": self.constant_singular_value,
            "verbose": 0,
        }

    @property
    def fixed_simulation_bipca_args(self) -> dict[str, Any]:
        return dict(
            backend="torch",
            n_components=-1,
            verbose=0,
            n_subsamples=0,
            logger=self.logger,
        )

    @is_subfigure(label="A")
    def _compute_A(self):
        """compute_A Generate subfigure 2A, simulating the rank recovery in BiPCA."""
        seeds = [self.seed + i for i in range(self.n_iterations)]
        FixedPoisson = partial(
            bipca_datasets.RankRPoisson,
            **self.fixed_simulation_args,
        )
        parameters = itertools.product(self.ranks, seeds)

        results = np.ndarray((len(self.ranks) * len(seeds), 2))
        for ix, (r, seed) in enumerate(parameters):
            with self.logger.log_task(
                f"r={r}, iteration {(ix % self.n_iterations)+1}:"
            ):
                dataset = FixedPoisson(rank=r, seed=seed).get_filtered_data()[
                    "simulation"
                ]
                results[ix, 0] = r
                results[ix, 1] = (
                    BiPCA(seed=seed, **self.fixed_simulation_bipca_args)
                    .fit(dataset.X)
                    .mp_rank
                )

        return results

    @is_subfigure(label="A")
    @plots
    @label_me
    def _plot_A(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
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
    def _compute_B(self):
        """compute_B Generate subfigure 2B, simulating the recovery of
        QVF parameter b in BiPCA."""

        seeds = [self.seed + i for i in range(self.n_iterations)]
        # use partial to make a QVFNegativeBinomial with fixed rank, c, and mean
        FixedNegativeBinomial = partial(
            bipca_datasets.QVFNegativeBinomial,
            rank=1,
            c=0.000001,
            **self.fixed_simulation_args,
        )
        # generate the parameter set as combinations of b and seeds
        parameters = itertools.product(self.bs, seeds)
        # run bipca over the parameters
        results = np.ndarray((len(self.bs) * len(seeds), 2))
        for ix, (b, seed) in enumerate(parameters):
            with self.logger.log_task(
                f"b={b}, iteration {(ix % self.n_iterations)+1}:"
            ):
                dataset = FixedNegativeBinomial(b=b, seed=seed).get_filtered_data()[
                    "simulation"
                ]
                results[ix, 0] = b
                results[ix, 1] = (
                    BiPCA(seed=seed, **self.fixed_simulation_bipca_args)
                    .fit(dataset.X)
                    .b
                )

        return results

    @is_subfigure(label="B")
    @plots
    @label_me
    def _plot_B(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
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
    def _compute_C(self):
        """compute_C generate subfigure 2C, simulating the recovery of QVF
        parameter c"""
        # generate seeds
        seeds = [self.seed + i for i in range(self.n_iterations)]
        # use partial to make a QVFNegativeBinomial with fixed rank, b, and mean
        FixedNegativeBinomial = partial(
            bipca_datasets.QVFNegativeBinomial,
            rank=1,
            b=1,
            **self.fixed_simulation_args,
        )
        # generate the parameter set as combinations of c and seeds
        parameters = itertools.product(self.cs, seeds)

        # run bipca over the parameters
        results = np.ndarray((len(self.cs) * len(seeds), 2))
        for ix, (c, seed) in enumerate(parameters):
            with self.logger.log_task(
                f"c={c}, iteration {(ix % self.n_iterations)+1}:"
            ):
                dataset = FixedNegativeBinomial(c=c, seed=seed).get_filtered_data()[
                    "simulation"
                ]
                results[ix, 0] = c
                results[ix, 1] = (
                    BiPCA(seed=seed, **self.fixed_simulation_bipca_args)
                    .fit(dataset.X)
                    .c
                )

        return results

    @is_subfigure(label="C")
    @plots
    @label_me
    def _plot_C(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
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
    def _compute_D_E_F(self):
        datasets = [
            (bipca_datasets.Zheng2017, "full"),  # 10xV1
            (bipca_datasets.TenX2021PBMC, "full"),  # 10xV3
            (bipca_datasets.Buenrostro2018, "full"),  # Buenrostro ATAC
            (
                bipca_datasets.TenX2022PBMCATAC,
                "full",
            ),  # 10x ATAC v1.1
            (bipca_datasets.Asp2019, "full"),  # spatial transcriptomics
            (bipca_datasets.Maynard2021, "151507"),  # visium
        ]
        seeds = [self.seed + i for i in range(self.n_iterations)]
        rngs = list(map(lambda seed: np.random.default_rng(seed), seeds))
        r = np.ndarray((len(datasets), self.n_iterations + 1), dtype=object)
        b = np.ndarray((len(datasets), self.n_iterations + 1), dtype=object)
        c = np.ndarray((len(datasets), self.n_iterations + 1), dtype=object)

        def subset_data(adata, prct, rng):
            n = int(prct * adata.shape[0])
            inds = rng.permutation(adata.shape[0])[:n]
            return adata[inds, :]

        for dset_ix, (dataset, sample) in enumerate(datasets):
            with self.logger.log_task(f"{dataset.__name__}-{sample}"):
                data_operator = dataset(
                    base_data_directory=self.base_plot_directory, logger=self.logger
                )
                adata = data_operator.get_unfiltered_data(samples=sample)[sample]
                # get the dataset name
                name = data_operator.__class__.__name__
                r[dset_ix, 0] = name
                b[dset_ix, 0] = name
                c[dset_ix, 0] = name
                for seed_ix, (rng, seed_n) in enumerate(zip(rngs, seeds)):
                    with self.logger.log_task(f"{dataset.__name__}-{sample}-{seed_n}"):
                        # subset the data
                        adata_sub = subset_data(adata, 0.75, rng)
                        adata_sub = data_operator.filter(adata_sub)
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
                            logger=self.logger,
                        )
                        op.fit(X)

                        # store the results
                        r[dset_ix, seed_ix + 1] = op.mp_rank
                        b[dset_ix, seed_ix + 1] = op.b
                        c[dset_ix, seed_ix + 1] = op.c
            # run biPCA on the full data
        results = {"D": r, "E": b, "F": c}
        return results

    @is_subfigure("D")
    @plots
    @label_me
    def _plot_D(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        # rank plot w/ resampling
        pass

    @is_subfigure("E")
    @plots
    @label_me
    def _plot_E(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        # b plot w/ resampling
        pass

    @is_subfigure("F")
    @plots
    @label_me
    def _plot_F(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        # c plot w/ resampling
        pass

    @is_subfigure(label=["G", "H", "I", "J"])
    def _compute_G_H_I_J(self):
        df = run_all(
            csv_path=self.base_plot_directory / "results" / "dataset_parameters.csv",
            logger=self.logger,
        )
        G = np.ndarray((len(df), 3), dtype=object)
        G[:, 0] = df.loc[:, "Modality"].values
        G[:, 1] = df.loc[:, "Linear coefficient (b)"].values
        G[:, 2] = df.loc[:, "Quadratic coefficient (c)"].values
        H = np.ndarray((len(df), 3), dtype=object)
        H[:, 0] = df.loc[:, "Modality"].values
        H[:, 1] = df.loc[:, "Filtered # observations"].values
        H[:, 2] = df.loc[:, "Rank"].values

        I = np.ndarray((len(df), 3), dtype=object)
        I[:, 0] = df.loc[:, "Modality"].values
        I[:, 1] = df.loc[:, "Rank"].values / df.loc[
            :, ["Filtered # observations", "Filtered # features"]
        ].values.min(1)
        I[:, 2] = df.loc[:, "Kolmogorov-Smirnov distance"].values
        J = np.ndarray((len(df), 3), dtype=object)
        J[:, 0] = df.loc[:, "Modality"].values
        J[:, 1] = df.loc[:, "Filtered # features"].values
        J[:, 2] = df.loc[:, "Rank"].values
        results = {"G": G, "H": H, "I": I, "J": J}
        return results

    @is_subfigure("G")
    @plots
    @label_me
    def _plot_G(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        # c plot w/ resampling
        assert "G" in self.results
        df = pd.DataFrame(self.results["G"], columns=["Modality", "b", "c"])
        # shuffle the points
        idx = np.random.default_rng(self.seed).permutation(df.shape[0])
        df_shuffled = df.iloc[idx, :]
        x, y = df_shuffled.loc[:, ["b", "c"]].values.T
        c = df_shuffled["Modality"].map(modality_fill_color).apply(pd.Series).values
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

        # build the legend handles and labels
        label2color = {
            modality_label[name]: color for name, color in modality_fill_color.items()
        }
        axis.legend(
            [
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="o",
                    color=color,
                    linewidth=0,
                    markeredgecolor="k",
                    markeredgewidth=0.1,
                    label=label,
                    markersize=5,
                )
                for label, color in label2color.items()
            ],
            list(label2color.keys()),
            # loc="center",
            # bbox_to_anchor=(1.65, -0.5),
            # ncol=3,
            columnspacing=0,
            handletextpad=0,
        )
        return axis

    @is_subfigure("H")
    @plots
    @label_me
    def _plot_H(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        # rank vs number of observations
        assert "H" in self.results
        df = pd.DataFrame(
            self.results["H"],
            columns=["Modality", "# observations", "rank"],
        )

        # shuffle the points
        idx = np.random.default_rng(self.seed).permutation(df.shape[0])
        df_shuffled = df.iloc[idx, :]

        x = df_shuffled.loc[:, "# observations"].values
        y = df_shuffled.loc[:, "rank"].values
        c = df_shuffled["Modality"].map(modality_fill_color).apply(pd.Series).values
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
        axis.set_xlabel(r"\# observations")
        axis.set_ylabel(r"Estimated rank $\hat{r}$")
        set_spine_visibility(axis, which=["top", "right"], status=False)

        return axis

    @is_subfigure("I")
    @plots
    @label_me
    def _plot_I(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        # KS vs r/m
        df = pd.DataFrame(
            self.results["I"],
            columns=["Modality", "r/m", "KS"],
        )
        # shuffle the points
        idx = np.random.default_rng(self.seed).permutation(df.shape[0])
        df_shuffled = df.iloc[idx, :]
        x = df_shuffled.loc[:, "r/m"].values
        y = df_shuffled.loc[:, "KS"].values
        c = df_shuffled["Modality"].map(modality_fill_color).apply(pd.Series).values

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
        # xlim = axis.get_xlim()
        # ylim = axis.get_ylim()
        plot_y_equals_x(
            axis,
            linewidth=1,
            linestyle="--",
            color="k",
            label=r"Optimal KS",
        )
        # axis.set_xlim(xlim)
        # axis.set_ylim(ylim)
        set_spine_visibility(axis, which=["top", "right"], status=False)

        axis.set_xlabel(r"Estimated rank fraction $\frac{\hat{r}}{m}$")
        axis.set_ylabel(r"Kolmogorov-Smirnov distance")
        axis.legend()
        return axis

    @is_subfigure("J")
    @plots
    @label_me
    def _plot_J(self, axis):
        assert "J" in self.results
        df = pd.DataFrame(
            self.results["J"],
            columns=["Modality", "# features", "rank"],
        )

        # shuffle the points
        idx = np.random.default_rng(self.seed).permutation(df.shape[0])
        df_shuffled = df.iloc[idx, :]

        x = df_shuffled.loc[:, "# features"].values
        y = df_shuffled.loc[:, "rank"].values
        c = df_shuffled["Modality"].map(modality_fill_color).apply(pd.Series).values
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
        axis.set_xlabel(r"\# features")
        axis.set_ylabel(r"Estimated rank $\hat{r}$")
        set_spine_visibility(axis, which=["top", "right"], status=False)

        return axis


class Figure3(Figure):
    _figure_layout = [
        ["A", "A", "A", "A", "A", "A"],
        ["A", "A", "A", "A", "A", "A"],
        ["B", "B", "B", "B", "B", "B"],
        ["B", "B", "B", "B", "B", "B"],
    ]

    def __init__(
        self,
         
        upper_r = 1000,
        lower_r = 5,
        test_p_range = 20,
        additional_r = 50,
        output_dir  = "./",    
        n_iterations=10,
        *args,
        **kwargs,
    ):
        self.seed = seed
        
        self.upper_r = upper_r
        self.lower_r = lower_r
        self.test_p_range = test_p_range
        self.additional_r = additional_r
        self.output_dir = output_dir
        
        self.results = {}
        
        super().__init__(*args, **kwargs)


    def new_svd(X,r,which="left"):
    
        svd_op = bipca.math.SVD(n_components=-1,backend='torch',use_eig=True)
        U,S,V = svd_op.fit_transform(X)
        if which == "left":
            return (np.asarray(U)[:,:r])*(np.asarray(S)[:r])
        else:
            return np.asarray(U[:,:r]),np.asarray(S[:r]),np.asarray(V.T[:r,:])
    
    def libnorm(X):
        return X/X.sum(axis=1)[:,None]

    
    def load_Stoeckius2017(self):
        """load processed and normalized cbmc cite-seq data (Stoeckius2017)"""
        data = bipca_datasets.Stoeckius2017(base_data_directory = self.output_dir)
        adata = data.get_filtered_data()['full']

        # run bipca
        torch.set_num_threads(36)
        with threadpool_limits(limits=36):
            op = bipca.BiPCA(n_components=-1,seed=42)
            Z = op.fit_transform(adata.X)
            op.write_to_adata(adata)
        adata.write_h5ad(self.output_dir+"cbmc_bipca.h5ad")
        # run other normalization methods
        adata_others = apply_normalizations(self.output_dir+"cbmc_bipca.h5ad")

        return op,adata_others
    
    def load_Stuart2019(self):
        """load processed and normalized cbmc cite-seq data (Stuart2019)"""
        adata = bipca_datasets.Stuart2019(base_data_directory = self.output_dir).get_filtered_data()['full']
        # run bipca
        torch.set_num_threads(36)
        with threadpool_limits(limits=36):
            op = bipca.BiPCA(n_components=-1,seed=42)
            Z = op.fit_transform(adata.X)
            op.write_to_adata(adata)
        adata.write_h5ad(self.output_dir+"bm_bipca.h5ad")
        # run other normalization methods
        adata_others = apply_normalizations(self.output_dir+"bm_bipca.h5ad")

        return op, adata_others
        

        
    def get_r_shrink_s(self,op,adata):

        r_list = np.sort(np.hstack((np.array([2**p for p in range(self.test_p_range)])[
                                    np.array([2**p <= self.upper_r for p in range(self.test_p_range)])],
                   self.upper_r,
                    op.mp_rank,
                    self.additional_r
                   )))
        _,_,sigma_grids = rank_to_sigma(r_list,op.S_Y,adata.X.shape)

        external_rank_grids = []
        external_s_list = []
        for sig in sigma_grids:
    
            shrinker = bipca.math.Shrinker().fit(op.S_Y, shape=adata.T.shape, sigma = sig)
            r = shrinker[0].scaled_mp_rank    
            shrink_s = shrinker[0].transform()
            external_rank_grids.append(r)
            external_s_list.append(shrink_s)
        external_s_list = np.array(external_s_list)
        estimated_r_pos = np.where(np.array(external_rank_grids) == op.mp_rank)[0][0]

        # add the original rank
        shrinker = bipca.math.Shrinker().fit(op.S_Y, shape=adata.T.shape)
        shrink_s = shrinker[0].transform()
        external_s_list[estimated_r_pos,:] = shrink_s

        return r_list, external_s_list
        
    def getPCs(self,op,r_list,external_s_list,adata):
        bipca_data_list = []
        for idx, external_rank in enumerate(r_list):
    
            ext_r = external_rank
            ext_s = external_s_list[idx,:]
    
            org_mat = (op.U_Y[:,:ext_r] * ext_s[:ext_r]) @ op.V_Y.T[:ext_r,:]
    
            # 0-thresholding
            org_mat[org_mat<0] = 0
    
            # lib-normalization
            lib_mat = self.libnorm(org_mat)

            svd_mat = self.new_svd(lib_mat,r=ext_r)
    
            bipca_data_list.append(svd_mat)

        dataset = OrderedDict()
        dataset["bipca"] = bipca_data_list
        dataset["log1p"] = self.new_svd(adata.layers['log1p'].toarray(),self.upper_r)
        dataset["log1p_z"] = self.new_svd(adata.layers['log1p+z'].toarray(),self.upper_r)
        dataset["SCT"] = self.new_svd(adata.layers['Pearson'],self.upper_r)
        dataset["ALRA"] =  self.new_svd(adata.layers['ALRA'],self.upper_r)
        dataset["Sanity"] = self.new_svd(adata.layers['Sanity'],self.upper_r)

        return dataset
        
    def runClassification_cbmc(self,adata,dataset,r_list):
        sample_names  = np.array(["citeseq-cbmc-classification"])
        methods = np.array(list(dataset.keys()))
        ranks = np.array([str(rank) for rank in r_list])
        rounds = np.arange(n_iterations)
        acc_df = pd.DataFrame(cartesian((sample_names,methods, ranks,rounds)))
        acc_df.rename(columns={0:"sample_name",1:"methods",2:"rank",3:"rounds"},inplace=True)
        acc_df["ACC"] = 0

        annotations_all = np.unique(adata.obs['protein_annotations'].values)
        label_convertor = {k:i for i,k in enumerate(annotations_all)}
        y = np.array([label_convertor[label] for label in adata.obs['protein_annotations']])
        grid_list = np.arange(2,21)

        # at each rank regime
        for i,rank2keep in enumerate(r_list):
            print("-----------------------")
            print("Rank regime: rank = {}".format(rank2keep))
            # repeat 10 times 
            for rounds in range(n_iterations):
                print("- Iteration: {}".format(rounds))
                # run classification for each method
                for ix,method in enumerate(dataset):
            
                    if method == "bipca":
                        X_input = dataset[method][i]
            
                    else:
                        X_input = dataset[method][:,:rank2keep]
            
                    test_acc,_,k = knn_classifier(X_input,y,train_ratio=0.2,train_metric=balanced_accuracy_score,K=grid_list,k_cv=5,random_state=rounds,KNeighbors_kwargs={"n_jobs":30})

                    acc_df.loc[(acc_df["methods"] == method) & (acc_df["rank"] == ranks[i]) & (acc_df["rounds"] == str(rounds)),"ACC"] = test_acc
        
                    print("---- Method: {}, Optimized k : {}, Test acc: {:.4f}".format(method,k,test_acc))

        acc_df_mean = acc_df.drop(columns=['sample_name']).groupby(['rank','methods']).mean()
        acc_df_mean['ranks'] =  np.array([i[0] for i in acc_df_mean.index.values])
        acc_df_mean = acc_df_mean.pivot_table(index=['ranks'], columns='methods',values="ACC")
        acc_df_mean = acc_df_mean.loc[ranks,:]

        acc_df_std = acc_df.drop(columns=['sample_name']).groupby(['rank','methods']).std()
        acc_df_std['ranks'] =  np.array([i[0] for i in acc_df_std.index.values])
        acc_df_std = acc_df_std.pivot_table(index=['ranks'], columns='methods',values="ACC")
        acc_df_std = acc_df_std.loc[ranks,:]

        return acc_df_mean,acc_df_std

    def runClassification_bm(self,adata,dataset,r_list):
        sample_names  = np.array(["citeseq-bone-marrow-classification"])
        methods = np.array(list(dataset.keys()))
        ranks = np.array([str(rank) for rank in r_list])
        rounds = np.arange(n_iterations)
        acc_df = pd.DataFrame(cartesian((sample_names,methods, ranks,rounds)))
        acc_df.rename(columns={0:"sample_name",1:"methods",2:"rank",3:"rounds"},inplace=True)
        acc_df["ACC"] = 0

        annotations_all = np.unique(adata.obs['cell_types'].values)
        label_convertor = {k:i for i,k in enumerate(annotations_all)}
        y = np.array([label_convertor[label] for label in adata.obs['cell_types']])
        grid_list = np.arange(2,21)

        # at each rank regime
        for i,rank2keep in enumerate(r_list):
            print("-----------------------")
            print("Rank regime: rank = {}".format(rank2keep))
            # repeat 10 times 
            for rounds in range(10):
                print("- Iteration: {}".format(rounds))
                # run classification for each method
                for ix,method in enumerate(dataset):
            
                    if method == "bipca":
                        X_input = dataset[method][i]
            
                    else:
                        X_input = dataset[method][:,:rank2keep]
            
                    test_acc,_,k = knn_classifier(X_input,y,train_ratio=0.6,train_metric=balanced_accuracy_score,K=grid_list,k_cv=5,random_state=rounds,KNeighbors_kwargs={"n_jobs":30})

                    acc_df.loc[(acc_df["methods"] == method) & (acc_df["rank"] == ranks[i]) & (acc_df["rounds"] == str(rounds)),"ACC"] = test_acc
        
                    print("---- Method: {}, Optimized k : {}, Test acc: {:.4f}".format(method,k,test_acc))

        acc_df_mean = acc_df.drop(columns=['sample_name']).groupby(['rank','methods']).mean()
        acc_df_mean['ranks'] =  np.array([i[0] for i in acc_df_mean.index.values])
        acc_df_mean = acc_df_mean.pivot_table(index=['ranks'], columns='methods',values="ACC")
        acc_df_mean = acc_df_mean.loc[ranks,:]

        acc_df_std = acc_df.drop(columns=['sample_name']).groupby(['rank','methods']).std()
        acc_df_std['ranks'] =  np.array([i[0] for i in acc_df_std.index.values])
        acc_df_std = acc_df_std.pivot_table(index=['ranks'], columns='methods',values="ACC")
        acc_df_std = acc_df_std.loc[ranks,:]

        return acc_df_mean,acc_df_std
        
    @is_subfigure(label="A")
    def _compute_A(self):
        """compute_A Generate subfigure 2A, simulating the rank recovery in BiPCA."""
        op, adata = self.load_Stoeckius2017()
        self.bipca_rank_A = op.mp_rank
        r_list, external_s_list = self.get_r_shrink_s(op, adata)
        dataset = self.getPCs(op,r_list,external_s_list,adata)
        acc_df_mean,acc_df_std = self.runClassification_cbmc(adata,dataset,r_list)
        results = np.concatenate((r_list.reshape(-1,1),np.concatenate((acc_df_mean.values,acc_df_std.values),axis=1)),axis=1)
        return results

    @is_subfigure(label="A")
    @plots
    @label_me
    def _plot_A(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        """plot_A Plot the results of subfigure 3A."""
        assert "A" in self.results
        
        # to replace
        r_list = self.result["A"][:,0].reshape(-1)
        r_list2show = r_list[3:]
        r_list2show_idx = np.array([np.where(rank == r_list)[0][0] for rank in r_list2show])
        acc_df_mean = pd.DataFrame(self.results["A"][:,1:7],columns=["ALRA","SCT","Sanity","bipca","log1p","log1p_z"]) # change to 1:7
        acc_df_std = pd.DataFrame(self.results["A"][:,7:],columns=["ALRA","SCT","Sanity","bipca","log1p","log1p_z"]) # change to 7:
        ymin = 0.82
        ymax = 0.95

        y_bipca_annotation_calibration = np.hstack(([0.02]*2,[0.005]*(len(r_list2show_idx)-2)))

        axis.errorbar(r_list2show,acc_df_mean.log1p[r_list2show_idx],acc_df_std.log1p[r_list2show_idx],label="log1p",c="deepskyblue")
        axis.errorbar(r_list2show,acc_df_mean.log1p_z[r_list2show_idx],acc_df_std.log1p_z[r_list2show_idx],label="log1p_z",c="blue")
        for i,r2show in enumerate(r_list2show_idx):
            axis.annotate("%.4f" % acc_df_mean.bipca[r2show], (r_list2show[i], acc_df_mean.bipca[r2show]+y_bipca_annotation_calibration[i]),c="red")

        axis.vlines(50,ymin,ymax,linestyles="dashed",colors="lightpink",label="default rank")
        axis.vlines(self.bipca_rank_A,ymin,ymax,linestyles="dashed",colors="deeppink",label="estimated rank") # change to op.mp_rank


        axis.errorbar(r_list2show,acc_df_mean.SCT[r_list2show_idx],acc_df_std.SCT[r_list2show_idx],label="SCT",c="green")
        axis.errorbar(r_list2show,acc_df_mean.ALRA[r_list2show_idx],acc_df_std.ALRA[r_list2show_idx],label="ALRA",c="purple")

        axis.errorbar(r_list2show,acc_df_mean.Sanity[r_list2show_idx],acc_df_std.Sanity[r_list2show_idx],label="Sanity",c="orange")
        axis.errorbar(r_list2show,acc_df_mean.bipca[r_list2show_idx],acc_df_std.bipca[r_list2show_idx],label="bipca",c="red")

        axis.set_xscale('log',base=2)
        axis.set_xticks(r_list2show)
        axis.set_xticklabels(r_list2show)

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
    
        axis.set_ylim(ymin,ymax)
        axis.set_xlabel("Rank")
        axis.set_ylabel("Balanced accuracy")
        axis.legend(loc="lower right")
        
        

        return axis


    @is_subfigure(label="B")
    def _compute_B(self):
        """compute_B Generate subfigure 3B, cell type classification using bone marrow cite-seq data."""
        op, adata = self.load_Stuart2019()
        self.bipca_rank_B = op.mp_rank
        r_list, external_s_list = self.get_r_shrink_s(op, adata)
        dataset = self.getPCs(op,r_list,external_s_list,adata)
        acc_df_mean,acc_df_std = self.runClassification_bm(adata,dataset,r_list)
        results = np.concatenate((r_list.reshape(-1,1),np.concatenate((acc_df_mean.values,acc_df_std.values),axis=1)),axis=1)
        return results


    @is_subfigure(label="B")
    @plots
    @label_me
    def _plot_B(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        """plot_B Plot the results of subfigure 3B."""
        assert "B" in self.results
        
        # to replace
        r_list = self.result["B"][:,0].reshape(-1)
        r_list2show = r_list[4:]
        r_list2show_idx = np.array([np.where(rank == r_list)[0][0] for rank in r_list2show])
        acc_df_mean = pd.DataFrame(self.results["B"][:,1:7],columns=["ALRA","SCT","Sanity","bipca","log1p","log1p_z"]) # change to 1:7
        acc_df_std = pd.DataFrame(self.results["B"][:,1:7],columns=["ALRA","SCT","Sanity","bipca","log1p","log1p_z"]) # change to 7:
        ymin = 0.7
        ymax = 0.85

        y_bipca_annotation_calibration = np.hstack(([0.02]*2,[0.005]*(len(r_list2show_idx)-2)))

        axis.errorbar(r_list2show,acc_df_mean.log1p[r_list2show_idx],acc_df_std.log1p[r_list2show_idx],label="log1p",c="deepskyblue")
        axis.errorbar(r_list2show,acc_df_mean.log1p_z[r_list2show_idx],acc_df_std.log1p_z[r_list2show_idx],label="log1p_z",c="blue")
        for i,r2show in enumerate(r_list2show_idx):
            ax.annotate( "%.4f" % acc_df_mean.bipca[r2show], (r_list2show[i], acc_df_mean.bipca[r2show]+y_bipca_annotation_calibration[i]),c="red")

        axis.vlines(50,ymin,ymax,linestyles="dashed",colors="lightpink",label="default rank")
        axis.vlines(self.bipca_rank_B,ymin,ymax,linestyles="dashed",colors="deeppink",label="estimated rank")


        axis.errorbar(r_list2show,acc_df_mean.SCT[r_list2show_idx],acc_df_std.SCT[r_list2show_idx],label="SCT",c="green")
        axis.errorbar(r_list2show,acc_df_mean.ALRA[r_list2show_idx],acc_df_std.ALRA[r_list2show_idx],label="ALRA",c="purple")

        axis.errorbar(r_list2show,acc_df_mean.Sanity[r_list2show_idx],acc_df_std.Sanity[r_list2show_idx],label="Sanity",c="orange")
        axis.errorbar(r_list2show,acc_df_mean.bipca[r_list2show_idx],acc_df_std.bipca[r_list2show_idx],label="bipca",c="red")

        axis.set_xscale('log',base=2)
        axis.set_xticks(r_list2show)
        axis.set_xticklabels(r_list2show)

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
    
        axis.set_ylim(ymin,ymax)
        axis.set_xlabel("Rank")
        axis.set_ylabel("Balanced accuracy")
        axis.legend(loc="lower right")
        
        

        return axis
