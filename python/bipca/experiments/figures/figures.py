import itertools
from pathlib import Path
from functools import partial, singledispatch
from typing import Dict, Union, Optional, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from anndata import AnnData

import bipca
from bipca import BiPCA
from bipca.utils import issparse
from bipca.plotting import set_spine_visibility

from bipca.experiments.figures.base import (
    Figure,
    is_subfigure,
    plots,
)
from .utils import parameter_estimation_plot, compute_minor_log_ticks
import bipca.experiments.datasets as bipca_datasets


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

    adatas = dataset.get_unfiltered_data()
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
                "Quadratic coefficient(c)": op.c,
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


class Figure2(Figure):
    _figure_layout = [
        ["A", "B", "C", "D"],
        ["E", "E", "F", "F"],
        ["G", "H", "M", "M"],
        ["I", "J", "M", "M"],
    ]

    def __init__(
        self,
        seed=42,
        mrows=4000,
        ncols=5000,
        ranks=2 ** np.arange(0, 8),
        bs=2.0 ** np.arange(-7, 1),
        cs=2.0 ** np.arange(-7, 1),
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
        super().__init__(*args, **kwargs)

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
        results = {"x": results[:, 0], "y": results[:, 1]}
        return results

    @is_subfigure(label="A")
    @plots
    def plot_A(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        """plot_A Plot the results of subfigure 2A."""
        assert "A" in self.results
        axis = parameter_estimation_plot(
            axis,
            self.results["A"],
            parameter_name="rank",
            parameter_var="r",
            xscale="log",
            yscale="log",
            xscale_params={"base": 2},
            yscale_params={"base": 2},
        )
        yticks = [2**i for i in range(0, 8, 1)]
        yticklabels = [rf"$2^{i}$" if i % 2 == 1 else None for i in range(0, 8, 1)]
        minorticks = compute_minor_log_ticks(yticks, 2)

        axis.set_yticks(yticks, labels=yticklabels, minor=False)
        axis.set_yticks(minorticks, minor=True)
        axis.set_xticks(yticks, labels=yticklabels, minor=False)
        axis.set_xticks(minorticks, minor=True)

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
            mean=100,
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
        results = {"x": results[:, 0], "y": results[:, 1]}
        return results

    @is_subfigure(label="B")
    @plots
    def plot_B(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        """plot_B Plot the results of subfigure 2B."""
        assert "B" in self.results
        axis = parameter_estimation_plot(
            axis,
            self.results["B"],
            parameter_name="linear variance",
            parameter_var="b",
            xscale="log",
            yscale="log",
            xscale_params={"base": 2},
            yscale_params={"base": 2},
        )
        yticks = [2**i for i in range(0, 8, 1)]
        yticklabels = [rf"$2^{i}$" if i % 2 == 1 else None for i in range(0, 8, 1)]
        minorticks = compute_minor_log_ticks(yticks, 2)
        axis.set_yticks(yticks, labels=yticklabels, minor=False)
        axis.set_yticks(minorticks, minor=True)
        axis.set_xticks(yticks, labels=yticklabels, minor=False)
        axis.set_xticks(minorticks, minor=True)

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
            mean=100,
            mrows=self.mrows,
            ncols=self.ncols,
            verbose=0,
        )
        # generate the parameter set as combinations of c and seeds
        parameters = itertools.product(self.cs, seeds)
        # map the experiment over the parameters
        datasets = map(
            lambda ele: FixedNegativeBinomial(c=ele[0], seed=ele[1])["simulation"],
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
        results = {"x": results[:, 0], "y": results[:, 1]}
        return results

    @is_subfigure(label="C")
    @plots
    def plot_C(self, axis: mpl.axes.Axes) -> mpl.axes.Axes:
        """plot_C Plot the results of subfigure 2C."""
        assert "C" in self.results
        axis = parameter_estimation_plot(
            axis,
            self.results["C"],
            parameter_name="quadratic variance",
            parameter_var="c",
            xscale="log",
            yscale="log",
            xscale_params={"base": 2},
            yscale_params={"base": 2},
        )
        yticks = [2**i for i in range(0, 8, 1)]
        yticklabels = [rf"$2^{i}$" if i % 2 == 1 else None for i in range(0, 8, 1)]
        minorticks = compute_minor_log_ticks(yticks, 2)

        axis.set_yticks(yticks, labels=yticklabels, minor=False)
        axis.set_yticks(minorticks, minor=True)
        axis.set_xticks(yticks, labels=yticklabels, minor=False)
        axis.set_xticks(minorticks, minor=True)

        return axis
