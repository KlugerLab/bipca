import os, errno
import itertools
import subprocess
import time
from numbers import Number
from pathlib import Path
from functools import partial, singledispatch, reduce
from typing import Dict, Union, Optional, List, Any, Callable

import seaborn as sns

import numpy as np
import pandas as pd
from anndata import AnnData, read_h5ad
import scanpy as sc
from scipy import sparse
from scipy.io import mmwrite
import torch
from torch.multiprocessing import Pool

from sklearn.preprocessing import scale as zscore
from ALRA import ALRA

import bipca
from bipca import BiPCA
from bipca.math import SVD, KDE
from bipca.utils import issparse,feature_scale
from bipca.plotting import set_spine_visibility,ridgeline,plot_density


import bipca.experiments.datasets as bipca_datasets

from tqdm.contrib.concurrent import thread_map, process_map

from threadpoolctl import threadpool_limits
from sklearn.utils.extmath import cartesian

from collections import OrderedDict

from sklearn.metrics import balanced_accuracy_score, average_precision_score,roc_auc_score
from scipy.cluster.hierarchy import linkage,dendrogram
from bipca.experiments import rank_to_sigma
from bipca.experiments import knn_classifier, get_mean_var
from bipca.experiments.utils import uniques

from bipca.experiments import (knn_test_k,
                              compute_affine_coordinates_PCA,
                              compute_stiefel_coordinates_from_affine,
                              compute_stiefel_coordinates_from_data,
                              libnorm,
                                new_svd,
                                mannwhitneyu_de)
from bipca.experiments.normalizations import library_normalize, log1p,apply_normalizations

from bipca.experiments.datasets.base import Dataset

from .base import (
    Figure,
    is_subfigure,
    label_me,
    plt,
    mpl,
    log_func_with,
    SMALL_SIZE,
    MEDIUM_SIZE,
    BIGGER_SIZE
)
from .utils import (
    mean_var_plot,
    parameter_estimation_plot,
    compute_minor_log_ticks,
    compute_axis_limits,
    plot_y_equals_x,
    boxplot,
    npg_cmap,
    compute_latex_ticklabels,
    replace_from_dict,
    download_url,
    download_urls,
    get_files,flatten,
    set_spine_visibility
)

from .plotting_constants import (
    algorithm_color_index,
    algorithm_fill_color,
    modality_color_index,
    modality_fill_color,
    modality_label,
    dataset_label,
    marker_experiment_colors,
    line_cmap,
    fill_cmap,
    heatmap_cmap
)

def _parallel_compute_metric(matrix, binarized_labels):
    results = []
    labs = [np.round(np.sum(lab)/len(lab),4) for lab in binarized_labels.T]
    if not np.allclose(labs,labs[0]):
        uniques,unique_counts = np.unique(labs,
                                                return_counts=True,)
        if len(uniques) == 1:
            new_binarized_labels = binarized_labels.T
        else:
            mode = uniques[np.argmax(unique_counts)]
            new_binarized_labels = []
            for ix,lab in enumerate(binarized_labels.T):
                where_pos = np.where(lab)[0]
                where_neg = np.where(~lab)[0]
                if labs[ix] != mode:
                    # we need to resample this data to match the mode
                    to_sample = len(where_pos)/len(binarized_labels)
                    to_sample = to_sample*mode*len(binarized_labels)
                    samplex = np.random.permutation(len(where_pos))[:int(to_sample)]
                    pos_samples = where_pos[samplex]
                    to_sample = to_sample/mode - to_sample
                    sampley = np.random.permutation(len(where_neg))[:int(to_sample)]
                    neg_samples = where_neg[sampley]
                    samps = np.concatenate([pos_samples,neg_samples]),
                    new_binarized_labels.append((lab[samps],samps))
                    
                else:
                    new_binarized_labels.append(lab)
    else:
        new_binarized_labels = binarized_labels.T
    for g in matrix.T:
        res = []
        for lab in new_binarized_labels:
            if isinstance(lab, tuple):
                idxs = lab[1]
                lab = lab[0]
                g = g[idxs]
            res.append(roc_auc_score(lab,g))
        results.extend(res)
    return results

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
                "Quadratic coefficient (c)",
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





class Figure2(Figure):
    _figure_layout = [
        ["A", "A", "B", "B", "C", "C", "D", "D2", "D3"],
        ["E", "E", "E", "F", "F", "F", "G", "G", "G"],
        ["E", "E", "E", "H", "H", "H", "G", "G", "G"],
        # ["I", "I", "I", "J", "J", "J", "K", "K", "K"],
        # ["L", "L", "L", "M", "M", "M", "N", "N", "N"],
    ]

    def __init__(
        self,
        seed=42,
        mrows=5000,
        ncols=5000,
        minimum_singular_value=False,
        constant_singular_value=False,
        entrywise_mean=10,
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
        xlim = compute_axis_limits(results["x"], "log", {"base": 2})
        ylim = compute_axis_limits(results["y"], "log", {"base": 2})
        lim_min = min(xlim[0], ylim[0])
        lim_max = max(xlim[1], ylim[1])
        axis.set_xlim([lim_min, lim_max])

        axis.set_ylim([lim_min, lim_max])
        return axis

    def _layout(self):
        # subplot mosaic was not working for me, so I'm doing it manually
        figure_left = 0.
        figure_right = 1
        figure_top = 0
        figure_bottom = 0
        super_row_pad = 0.07
        sub_row_pad = 0.05
        sub_column_pad = 0.05
        # full page width figure with ~1 in margin
        original_positions = {
            label: self[label].axis.get_position() for label in self._subfigures
        }
        new_positions = {label: pos for label, pos in original_positions.items()}
        # adjust super columns for whole figure
        # these are set by E, F/H, G.
        left = figure_left
        right = figure_right
        pad = sub_column_pad
        width = (right - left - 2 * pad) / 3
        new_positions["E"].x0 = left
        new_positions["E"].x1 = left + width
        new_positions["F"].x0 = left + width + pad
        new_positions["F"].x1 = left + 2 * width + pad
        new_positions["H"].x0 = left + width + pad
        new_positions["H"].x1 = left + 2 * width + pad
        new_positions["G"].x0 = left + 2 * width + 2 * pad
        new_positions["G"].x1 = right
        # adjust first row super columns
        # the super columns are [A,A,B,B,C,C] and [D, D2,D3]
        # we need [A - C] to key on E-(F/H), while [D,D1,D2] keys on G
        # we also want each [A-C] to be "square"ish, while [D,D2,D3] is a rectangle
        # start with [A-C]
        left = figure_left
        right = new_positions["G"].x1
        pad = sub_column_pad
        # the minimum y0 of [A-F] to get a reasonable whitespace between the rows is 0.75
        # therefore the maximum height of [A-F] is 0.88-0.75 = 0.13
        width = (right - left - 2 * pad) / 3
        height = 0.13
        square_dimension = np.minimum(width, height)
        # now we have the square dimension, we can compute x0 and x1 for A-C.
        new_positions["A"].x0 = left
        new_positions["A"].x1 = left + square_dimension
        new_positions["B"].x0 = left + square_dimension + pad
        new_positions["B"].x1 = left + 2 * square_dimension + pad
        new_positions["C"].x0 = left + 2 * square_dimension + 2 * pad
        new_positions["C"].x1 = left + 3 * square_dimension + 2 * pad
        # now we can compute the positions of [D,D2,D3]
        # the ticklabels on D take a lot of room, so we need to adjust the left
        left = new_positions["G"].x0 + 0.07
        right = figure_right
        pad = 0.01  # this pads between the shared axes
        width = (right - left - 2 * pad) / 3
        # these can be rectangular, but have the same height as A-C
        new_positions["D"].x0 = left
        new_positions["D"].x1 = left + width
        new_positions["D2"].x0 = left + width + pad
        new_positions["D2"].x1 = left + 2 * width + pad
        new_positions["D3"].x0 = left + 2 * width + 2 * pad
        new_positions["D3"].x1 = right
        # finally, set the height of the first row
        for label in ["A", "B", "C"]:
            new_positions[label].y0 = 0.88 - square_dimension
            new_positions[label].y1 = 0.88
        for label in ["D", "D2", "D3"]:  # give a little extra room for the legend
            new_positions[label].y0 = 0.88 - square_dimension
            new_positions[label].y1 = 0.88

        # next, we need to adjust the height of the second row / super row.
        # it will be of height 2 * square_dimension + a row pad
        first_row_offset = figure_top - square_dimension - super_row_pad
        second_row_height = 2 * square_dimension
        pad = 0
        y1 = first_row_offset
        y0 = first_row_offset - second_row_height
        new_positions["E"].y0 = y0
        new_positions["E"].y1 = y1
        new_positions["G"].y0 = y0
        new_positions["G"].y1 = y1
        # [F and H] will split their height and have a pad
        H_J_height = (second_row_height - sub_row_pad) / 2
        new_positions["F"].y1 = y1
        new_positions["F"].y0 = y1 - H_J_height
        new_positions["H"].y1 = new_positions["F"].y0 - sub_row_pad
        new_positions["H"].y0 = y0

        # # set the height of the third row
        # third_row_offset = new_positions["H"].y0 - super_row_pad
        # third_row_height = square_dimension
        # new_positions["I"].y0 = third_row_offset - third_row_height
        # new_positions["I"].y1 = third_row_offset
        # new_positions["J"].y0 = third_row_offset - third_row_height
        # new_positions["J"].y1 = third_row_offset
        # new_positions["K"].y0 = third_row_offset - third_row_height
        # new_positions["K"].y1 = third_row_offset
        # # set the columns of the third row
        # for second, third in zip(["E", "F", "G"], ["I", "J", "K"]):
        #     new_positions[third].x0 = new_positions[second].x0
        #     new_positions[third].x1 = new_positions[second].x1

        # # repeat for fourth row
        # fourth_row_offset = new_positions["I"].y0 - super_row_pad
        # fourth_row_height = square_dimension
        # new_positions["L"].y0 = fourth_row_offset - fourth_row_height
        # new_positions["L"].y1 = fourth_row_offset
        # new_positions["M"].y0 = fourth_row_offset - fourth_row_height
        # new_positions["M"].y1 = fourth_row_offset
        # new_positions["N"].y0 = fourth_row_offset - fourth_row_height
        # new_positions["N"].y1 = fourth_row_offset
        # # set the columns of the fourth row
        # for second, fourth in zip(["E", "F", "G"], ["L", "M", "N"]):
        #     new_positions[fourth].x0 = new_positions[second].x0
        #     new_positions[fourth].x1 = new_positions[second].x1

        # set the positions
        for label, pos in new_positions.items():
            self[label].axis.set_position(pos)
            self[label].axis.patch.set_alpha(0)

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

    @is_subfigure(label="A",plots=True)
    @label_me
    def _plot_A(self, axis: mpl.axes.Axes, results:np.ndarray) -> mpl.axes.Axes:
        """plot_A Plot the results of subfigure 2A."""
        results = {"x": results[:, 0], "y": results[:, 1]}
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
        axis.set_xlabel(r"true $r$ ($\mathrm{log}_2$)", wrap=True)
        axis.set_ylabel(r"estimated $\hat{r}$ ($\mathrm{log}_2$)", wrap=True)
        axis.legend(
            [
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="None",
                    color="k",
                    linewidth=1,
                    linestyle="--",
                    label=r"$y=x$",
                    markersize=0,
                ),
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="o",
                    color=algorithm_fill_color["BiPCA"],
                    markeredgewidth=0.1,
                    markeredgecolor="k",
                    linewidth=0,
                    linestyle="None",
                    label=r"BiPCA",
                    markersize=3,
                ),
            ],
            [r"$y=x$", r"BiPCA"],
            loc="upper left",
            frameon=False,
        )
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

    @is_subfigure(label="B", plots=True)
    @label_me
    def _plot_B(self, axis: mpl.axes.Axes, results:np.ndarray) -> mpl.axes.Axes:
        """plot_B Plot the results of subfigure 2B."""
        results = {"x": results[:, 0], "y": results[:, 1]}

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
        axis.set_xlabel(r"true $b$ ($\mathrm{log}_2$)", wrap=True)
        axis.set_ylabel(r"estimated $\hat{b}$ ($\mathrm{log}_2$)", wrap=True)
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

    @is_subfigure(label="C", plots=True)
    @label_me
    def _plot_C(self, axis: mpl.axes.Axes,  results:np.ndarray) -> mpl.axes.Axes:
        """plot_C Plot the results of subfigure 2C."""
        results = {"x": results[:, 0], "y": results[:, 1]}

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
        axis.set_xlabel(r"true $c$ ($\mathrm{log}_2$)", wrap=True)
        axis.set_ylabel(
            r"estimated $\hat{c}$ ($\mathrm{log}_2$)",
            wrap=True,
        )
        return axis

    @is_subfigure(label=["D", "D2", "D3"])
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
        results = {"D": r, "D2": b, "D3": c}
        return results

    @is_subfigure(label="D", plots=True)
    @label_me(12)
    def _plot_D(self, axis: mpl.axes.Axes,  results:np.ndarray) -> mpl.axes.Axes:
        # rank plot w/ resampling
        data = results
        dataset_names = data[:, 0]
        # extract the display labels for the datasets
        dataset_labels = [
            replace_from_dict(name, dataset_label) for name in dataset_names
        ]
        # get the modalities and colors for the boxes and legend
        dataset_modalities = [
            getattr(bipca_datasets, cls_name).modality for cls_name in dataset_names
        ]
        colors = np.asarray([modality_fill_color[name] for name in dataset_modalities])
        label2color = {}
        for name in modality_fill_color:
            if name in dataset_modalities:
                label = replace_from_dict(name, modality_label)
                color = modality_fill_color[name]
                if label == "spatial transcriptomics":
                    label = "spatial\n transcriptomics"
                label2color[label] = color
        dataset_distribution = data[:, 1:].astype(float).T
        boxplot(axis, dataset_distribution, colors=colors)
        axis.set_xlabel(r"$\hat{r}$ ($\mathrm{log}_{10}$)")
        # axis.set_xscale('log',base=2)
        axis.set_yticks(axis.get_yticks(), labels=dataset_labels)
        leg = axis.legend(
            [
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="s",
                    color=color,
                    linewidth=0,
                    label=label,
                    markersize=5,
                )
                for label, color in label2color.items()
            ],
            list(label2color.keys()),
            loc="lower left",
            markerfirst=False,
            bbox_to_anchor=(-2.5, -0.445),
            ncol=1,
            columnspacing=0,
            handletextpad=0,
            frameon=False,
        )
        for t in leg.get_texts():
            t.set_ha("right")
        axis.fill_between([0, 10**3], 1.35, 2, fc=colors[0], ec=colors[0])
        axis.fill_between([0, 10**3], 1.15, 1.35, fc=colors[2], ec=colors[2])
        axis.fill_between([0, 10**3], 0.95, 1.15, fc=colors[4], ec=colors[4])

        axis.set_xscale("log", base=10)
        axis.set_xlim([10**1, 10**2 + 1 * 10**2])
        axis.set_xticks(
            [10**1, 10**2],
            labels=compute_latex_ticklabels([10**1, 10**2], 10, skip=False),
        )
        return axis

    @is_subfigure(label="D2", plots=True)
    def _plot_D2(self, axis: mpl.axes.Axes,  results:np.ndarray) -> mpl.axes.Axes:
        # b plot w/ resampling
        data = results
        dataset_names = data[:, 0]
        # get the modalities and colors for the boxes and legend
        colors = np.asarray(
            [
                modality_fill_color[getattr(bipca_datasets, cls_name).modality]
                for cls_name in dataset_names
            ]
        )
        dataset_distribution = data[:, 1:].astype(float).T
        boxplot(axis, dataset_distribution, colors=colors)
        axis.fill_between([0, 10**3], 1.35, 2, fc=colors[0], ec=colors[0])
        axis.fill_between([0, 10**3], 1.15, 1.35, fc=colors[2], ec=colors[2])
        axis.fill_between([0, 10**3], 0.95, 1.15, fc=colors[4], ec=colors[4])

        axis.set_xlabel(r"$\hat{b}$")
        axis.set_xlim([0.875, 1.625])
        axis.set_xticks([0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], minor=True)
        axis.sharey(self["D"].axis)
        axis.tick_params(axis="y", left=False, labelleft=False)
        return axis

    @is_subfigure(label="D3", plots=True)
    def _plot_D3(self, axis: mpl.axes.Axes,  results:np.ndarray) -> mpl.axes.Axes:
        # c plot w/ resampling
        data = results
        dataset_names = data[:, 0]
        # get the modalities and colors for the boxes and legend
        colors = np.asarray(
            [
                modality_fill_color[getattr(bipca_datasets, cls_name).modality]
                for cls_name in dataset_names
            ]
        )
        dataset_distribution = data[:, 1:].astype(float).T
        boxplot(axis, dataset_distribution, colors=colors)
        axis.fill_between([0, 10**3], 1.35, 2, fc=colors[0], ec=colors[0])
        axis.fill_between([0, 10**3], 1.15, 1.35, fc=colors[2], ec=colors[2])
        axis.fill_between([0, 10**3], 0.95, 1.15, fc=colors[4], ec=colors[4])
        axis.set_xlim([0.125, 8])
        axis.set_xlabel(r"$\hat{c}$ ($\mathrm{log}_2$)")
        axis.set_xscale("log", base=2)
        major_ticks = 2.0 ** np.arange(-3, 4)

        axis.set_xticks(
            major_ticks, labels=[r"$-3$", None, None, r"$0$", None, None, r"$3$"]
        )
        axis.set_xticks(compute_minor_log_ticks(major_ticks, 2), minor=True)
        axis.sharey(self["D"].axis)
        axis.tick_params(axis="y", left=False, labelleft=False)
        return axis

    @is_subfigure(label=["E", "F", "G", "H"])
    def _compute_E_F_G_H(self):
        df = run_all(
            csv_path=self.base_plot_directory / "results" / "dataset_parameters.csv",
            logger=self.logger,
        )
        E = np.ndarray((len(df), 3), dtype=object)
        E[:, 0] = df.loc[:, "Modality"].values
        E[:, 1] = df.loc[:, "Linear coefficient (b)"].values
        E[:, 2] = df.loc[:, "Quadratic coefficient (c)"].values
        F = np.ndarray((len(df), 3), dtype=object)
        F[:, 0] = df.loc[:, "Modality"].values
        F[:, 1] = df.loc[:, "Filtered # observations"].values
        F[:, 2] = df.loc[:, "Rank"].values

        G = np.ndarray((len(df), 3), dtype=object)
        G[:, 0] = df.loc[:, "Modality"].values
        G[:, 1] = df.loc[:, "Rank"].values / df.loc[
            :, ["Filtered # observations", "Filtered # features"]
        ].values.min(1)
        G[:, 2] = df.loc[:, "Kolmogorov-Smirnov distance"].values
        H = np.ndarray((len(df), 3), dtype=object)
        H[:, 0] = df.loc[:, "Modality"].values
        H[:, 1] = df.loc[:, "Filtered # features"].values
        H[:, 2] = df.loc[:, "Rank"].values
        results = {"E": E, "F": F, "G": G, "H": H}
        return results

    @is_subfigure(label="E", plots=True)
    @label_me
    def _plot_E(self, axis: mpl.axes.Axes,  results:np.ndarray) -> mpl.axes.Axes:
        # c plot w/ resampling
        df = pd.DataFrame(results, columns=["Modality", "b", "c"])
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
        axis.set_yscale("symlog", linthresh=1e-2, linscale=0.5)
        pos_log_ticks = 10.0 ** np.arange(-2, 3)
        neg_log_ticks = 10.0 ** np.arange(
            -2,
            1,
        )
        yticks = np.hstack((-1 * neg_log_ticks, 0, pos_log_ticks))

        axis.set_yticks(
            yticks, labels=compute_latex_ticklabels(yticks, 10, include_base=True)
        )
        xticks = [0, 0.5, 1.0, 1.5, 2.0]
        axis.set_xticks(
            xticks,
            labels=[0, 0.5, 1, 1.5, 2],
        )
        axis.set_yticks(
            np.hstack(
                (
                    -1 * compute_minor_log_ticks(neg_log_ticks, 10),
                    compute_minor_log_ticks(pos_log_ticks, 10),
                )
            ),
            minor=True,
        )
        axis.set_ylim(-1, 100)

        axis.set_xlabel(r"estimated linear variance $\hat{b}$")
        axis.set_ylabel(r"estimated quadratic variance $\hat{c}$")
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
                    markersize=3,
                )
                for label, color in label2color.items()
            ],
            list(label2color.keys()),
            loc="lower left",
            # bbox_to_anchor=(1.65, -0.5),
            # ncol=3,
            columnspacing=0,
            handletextpad=0,
            frameon=False,
        )
        return axis

    @is_subfigure(label="F", plots=True)
    @label_me
    def _plot_F(self, axis: mpl.axes.Axes,  results:np.ndarray) -> mpl.axes.Axes:
        # rank vs number of observations
        df = pd.DataFrame(
           results,
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
        axis.set_xticks(
            [10**3, 10**4, 10**5],
            labels=compute_latex_ticklabels([10**3, 10**4, 10**5], 10, skip=True),
        )
        axis.set_yticks(
            [10**1, 10**2],
            labels=compute_latex_ticklabels([10**1, 10**2], 10, skip=False),
        )
        axis.set_xlabel(r"\# observations ($\mathrm{log}_{10}$)")
        axis.set_ylabel(r"estimated $\hat{r}$ ($\mathrm{log}_{10}$)")
        set_spine_visibility(axis, which=["top", "right"], status=False)

        return axis

    @is_subfigure(label="G", plots=True)
    @label_me
    def _plot_G(self, axis: mpl.axes.Axes,  results: np.ndarray) -> mpl.axes.Axes:
        # KS vs r/m
        df = pd.DataFrame(
            results,
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

        plot_y_equals_x(
            axis,
            linewidth=1,
            linestyle="--",
            color="k",
            label=r"optimal K-S",
        )
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()

        set_spine_visibility(axis, which=["top", "right"], status=False)
        axis.set_yticks(
            axis.get_yticks(), labels=compute_latex_ticklabels(axis.get_yticks(), 10)
        )
        axis.set_xticks(
            axis.get_xticks(), labels=compute_latex_ticklabels(axis.get_xticks(), 10)
        )
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_xlabel(
            r"estimated rank fraction $\frac{\hat{r}}{m}$ ($\mathrm{log}_{10}$)"
        )
        axis.set_ylabel(r"Kolmogorov-Smirnov distance ($\mathrm{log}_{10}$)")
        axis.legend(
            frameon=False,
        )
        return axis

    @is_subfigure(label="H", plots=True)
    @label_me
    def _plot_H(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        df = pd.DataFrame(
            results,
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
        axis.set_xticks(
            [10**3, 10**4, 10**5],
            labels=compute_latex_ticklabels([10**3, 10**4, 10**5], 10, skip=True),
        )
        axis.set_yticks(
            [10**1, 10**2],
            labels=compute_latex_ticklabels([10**1, 10**2], 10, skip=False),
        )
        axis.set_xlabel(r"\# features ($\mathrm{log}_{10}$)")
        axis.set_ylabel(r"Estimated $\hat{r}$  ($\mathrm{log}_{10}$)")
        set_spine_visibility(axis, which=["top", "right"], status=False)

        return axis

    # 

class SupplementaryFigure1(Figure):
    _figure_layout = [
        ["A", "A", "B", "B","B","B"],
        ["C","C","D","D","E","E",],
        
    ]
    """This SupplementaryFigure shows the mean-variance relationships for different
    normalizations BEFORE low rank approximation"""
    """The top panel will how mean-variance before low rank approximation"""
    """they are A: raw, B: log1p, C: log1p+z
    D: Pearson, E: Sanity, F: Biwhitening"""

    def __init__(self,mean_cdf=False,
                      figure_kwargs: dict = dict(dpi=300, figsize=(8.5, 4.25)),
                      *args,**kwargs):
        self.mean_cdf = mean_cdf
        kwargs['figure_kwargs'] = figure_kwargs
        super().__init__(*args,**kwargs)
    
    @is_subfigure(label=["A", "B", "C", "D", "E",])
    def _compute_A(self):
        dataset = bipca_datasets.TenX2016PBMC(
            store_filtered_data=True, logger=self.logger
        )
        adata = dataset.get_filtered_data(samples=["full"])["full"]
        path = dataset.filtered_data_paths["full.h5ad"]
        todo = ["log1p",  "Pearson", "Sanity", "ALRA", "BiPCA"]
        bipca_kwargs = dict(n_components=-1,backend='torch', dense_svd=True,use_eig=True)
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        adata = apply_normalizations(adata, write_path = path,
                                    n_threads=64, apply=todo,
                                    normalization_kwargs={'BiPCA':bipca_kwargs},
                                    logger=self.logger)
        layers_to_process = [
            "raw data",
            "Biwhitened",
            "log1p",
            "Pearson",
            "Sanity",
            
        ]

        means = np.asarray(adata.X.mean(0)).squeeze()
        results = np.ndarray(
            (means.shape[0] + 1, len(layers_to_process) + 2), dtype=object
        )
        results[1:, 0] = adata.var_names.values
        results[0, 0] = "gene"
        results[1:, 1] = means
        results[0, 1] = "mean"
        for ix, layer in enumerate(layers_to_process):
            if layer == "Biwhitened":
                layer_select = "Y_biwhite"
                Y = adata.layers[layer_select]
            elif layer == "raw data":
                Y = adata.X
            else:
                Y = adata.layers[layer]
            results[0, ix + 2] = layer

            if issparse(Y):
                Y = Y.toarray()
            _, results[1:, ix + 2] = get_mean_var(Y, axis=0)
        results = {
            "A": results[:, [0, 1, 2]],
            "B": results[:, [0, 1, 3]],
            "C": results[:, [0, 1, 4]],
            "D": results[:, [0, 1, 5]],
            "E": results[:, [0, 1, 6]],
        }
        return results

    @is_subfigure(label="A", plots=True)
    @label_me
    def _plot_A(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        # raw data mean-var
        df = pd.DataFrame(results[1:, :], columns=["gene", "mean", "var"])
        df["var"] = df["var"].astype(float)
        df["mean"] = df["mean"].astype(float)
        axis = mean_var_plot(axis, df,mean_cdf=self.mean_cdf)

        axis.set_title(results[0, 2])

        return axis

    @is_subfigure(label="B", plots=True)
    @label_me
    def _plot_B(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        # biwhitening mean-var
        df = pd.DataFrame(results[1:, :], columns=["gene", "mean", "var"])
        df["var"] = df["var"].astype(float)
        df["mean"] = df["mean"].astype(float)
        axis = mean_var_plot(axis, df,mean_cdf=self.mean_cdf)

        axis.set_title(results[0, 2])
        axis.set_yticks([10**0,], labels=[r"$0$",])
        return axis

    @is_subfigure(label="C", plots=True)
    @label_me
    def _plot_C(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        # log1p  mean-var
        df = pd.DataFrame(results[1:, :], columns=["gene", "mean", "var"])
        df["var"] = df["var"].astype(float)
        df["mean"] = df["mean"].astype(float)
        axis = mean_var_plot(axis, df,mean_cdf=self.mean_cdf)

        axis.set_yticks([10**-3, 10**-2, 10**-1, 10**0], labels=[r"$-3$",None, r"$-1$",None])
        axis.set_title(results[0, 2])

        return axis

    @is_subfigure(label="D", plots=True)
    @label_me
    def _plot_D(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        # Pearson  mean-var
        df = pd.DataFrame(results[1:, :], columns=["gene", "mean", "var"])
        df["var"] = df["var"].astype(float)
        df["mean"] = df["mean"].astype(float)
        axis = mean_var_plot(axis, df,mean_cdf=self.mean_cdf)

        axis.set_title(results[0, 2])
        axis.set_yticks([10**0, 10**1, 10**2], labels=[r"$0$", None, r"$2$"])

        return axis

    @is_subfigure(label="E", plots=True)
    @label_me
    def _plot_E(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        # Sanity  mean-var
        df = pd.DataFrame(results[1:, :], columns=["gene", "mean", "var"])
        df["var"] = df["var"].astype(float)
        df["mean"] = df["mean"].astype(float)
        axis = mean_var_plot(axis, df,mean_cdf=self.mean_cdf)
        axis.set_title(results[0, 2])
        yticks = np.arange(-4,1,)
        axis.set_yticks(10.0**yticks, labels=[fr"${t}$" if t%2==0 else None for t in yticks])


        return axis

class SupplementaryFigure2(Figure):
    _figure_layout = [
        ["A", "A", "B", "B","B","B"],
        ["C","C","D","D","E","E",],
        
    ]
    """This SupplementaryFigure shows the mean-variance relationships for different
    normalizations BEFORE low rank approximation"""
    """The  panel will show mean-variance after low rank approximation"""
    """they are A: raw, B: BiPCA,
    C: log1p, D: log1p+z, E: ALRA"""

    def __init__(self,mean_cdf=False,
                      figure_kwargs: dict = dict(dpi=300, figsize=(8.5, 4.25)),
                      *args,**kwargs):
        self.mean_cdf = mean_cdf
        kwargs['figure_kwargs'] = figure_kwargs
        super().__init__(*args,**kwargs)

    @is_subfigure(label=["A", "B", "C", "D", "E",])
    def _compute_A(self):
        dataset = bipca_datasets.TenX2016PBMC(
            store_filtered_data=True, logger=self.logger
        )
        adata = dataset.get_filtered_data(samples=["full"])["full"]
        path = dataset.filtered_data_paths["full.h5ad"]
        todo = ["log1p", "log1p+z", "ALRA", "BiPCA"]
        bipca_kwargs = dict(n_components=-1,backend='torch', dense_svd=True,use_eig=True)
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        adata = apply_normalizations(adata, write_path = path,
                                    n_threads=64, apply=todo,
                                    normalization_kwargs={'BiPCA':bipca_kwargs},
                                    logger=self.logger)
        layers_to_process = [
            "raw data",
            "BiPCA",
            "log1p",
            "log1p+z",
            "ALRA",            
        ]

        means = np.asarray(adata.X.mean(0)).squeeze()
        results = np.ndarray(
            (means.shape[0] + 1, len(layers_to_process) + 2), dtype=object
        )
        results[1:, 0] = adata.var_names.values
        results[0, 0] = "gene"
        results[1:, 1] = means
        results[0, 1] = "mean"
        for ix, layer in enumerate(layers_to_process):
            if layer == "BiPCA":
                layer_select = "Z_biwhite"
                Y = adata.layers[layer_select]
                libsizes = np.asarray(adata.X.sum(1)).squeeze()
                scale = np.median(libsizes)
                Y = library_normalize(Y,scale)
            elif layer == "raw data":
                Y = adata.X
            else:
                Y = adata.layers[layer]
            
            
            if issparse(Y):
                Y = Y.toarray()
            if layer not in ['BiPCA', 'ALRA']:
                #apply low rank approximation
                U,S,V = SVD(backend='torch',n_components=50).fit_transform(Y)
                Y = (U*S)@V.T
                results[0, ix + 2] = f'rank 50 approximation of {layer}'
            else:
                results[0, ix + 2] = layer


            _, results[1:, ix + 2] = get_mean_var(Y, axis=0)
        results = {
            "A": results[:, [0, 1, 2]],
            "B": results[:, [0, 1, 3]],
            "C": results[:, [0, 1, 4]],
            "D": results[:, [0, 1, 5]],
            "E": results[:, [0, 1, 6]],
        }
        return results

    @is_subfigure(label="A", plots=True)
    @label_me
    def _plot_A(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        # raw data mean-var
        df = pd.DataFrame(results[1:, :], columns=["gene", "mean", "var"])
        df["var"] = df["var"].astype(float)
        df["mean"] = df["mean"].astype(float)
        axis = mean_var_plot(axis, df,mean_cdf=self.mean_cdf)

        axis.set_title(results[0, 2])
        yticks = np.arange(-5,4,)
        axis.set_yticks(10.0**yticks, labels=[fr"${t}$" if t%2==0 else None for t in yticks])

        return axis

    @is_subfigure(label="B", plots=True)
    @label_me
    def _plot_B(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        # biwhitening mean-var
        df = pd.DataFrame(results[1:, :], columns=["gene", "mean", "var"])
        df["var"] = df["var"].astype(float)
        df["mean"] = df["mean"].astype(float)
        axis = mean_var_plot(axis, df,mean_cdf=self.mean_cdf)

        axis.set_title(results[0, 2])
        yticks = np.arange(-3,2,)

        axis.set_yticks(10.0**yticks, labels=[fr"${t}$" if t%2==0 else None for t in yticks])
        return axis

    @is_subfigure(label="C", plots=True)
    @label_me
    def _plot_C(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        # log1p  mean-var
        df = pd.DataFrame(results[1:, :], columns=["gene", "mean", "var"])
        df["var"] = df["var"].astype(float)
        df["mean"] = df["mean"].astype(float)
        axis = mean_var_plot(axis, df,mean_cdf=self.mean_cdf)

        yticks = np.arange(-5,1,)

        axis.set_yticks(10.0**yticks, labels=[fr"${t}$" if t%2==0 else None for t in yticks])
        axis.set_title(results[0, 2])

        return axis

    @is_subfigure(label="D", plots=True)
    @label_me
    def _plot_D(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        # Pearson  mean-var
        df = pd.DataFrame(results[1:, :], columns=["gene", "mean", "var"])
        df["var"] = df["var"].astype(float)
        df["mean"] = df["mean"].astype(float)
        axis = mean_var_plot(axis, df,mean_cdf=self.mean_cdf)

        yticks = np.arange(-2,1,)

        axis.set_yticks(10.0**yticks, labels=[fr"${t}$" if t%2==0 else None for t in yticks])
        axis.set_title(results[0, 2])

        return axis

    @is_subfigure(label="E", plots=True)
    @label_me
    def _plot_E(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        # Sanity  mean-var
        df = pd.DataFrame(results[1:, :], columns=["gene", "mean", "var"])
        df["var"] = df["var"].astype(float)
        df["mean"] = df["mean"].astype(float)
        axis = mean_var_plot(axis, df,mean_cdf=self.mean_cdf)

        axis.set_title(results[0, 2])
        yticks = np.arange(-3,1,)
        axis.set_yticks(10.0**yticks, labels=[fr"${t}$" if t%2==0 else None for t in yticks])


        return axis

class Figure3(Figure):
    """Marker genes figure"""
    _figure_layout = [
        ["A", "A", "A", "A2", "A2", "A2","A3", "A3", "A3", "A4", "A4", "A4", "A5", "A5", "A5", "A6", "A6", "A6"],
        # ["A", "A", "A", "A2", "A2", "A2","A3", "A3", "A3", "A4", "A4", "A4", "A5", "A5", "A5", "A6", "A6", "A6"],
        ["B","B2", "B3","B4","B5","B6","C","C2","C3","C4","C5","C6","D","D2","D3","D4","D5","D6"],
        ["E","E","E","E2","E2","E2","E3","E3","E3","E4","E4","E4","D","D2","D3","D4","D5","D6"],

        # ["E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E"],
    ]
    _ix_to_layer_mapping = {ix:key for ix,key in enumerate(algorithm_color_index.keys())}
    _subfig_to_celltype = {'B':['CD4+ T cells','CD8+ T cells'],'C':['CD56+ natural killer cells'],'D':['CD19+ B cells']}
    _celltype_ordering = ['CD8+ T cells','CD4+ T cells','CD56+ natural killer cells','CD19+ B cells']
    _default_marker_annotations_url = {"reference_HPA.tsv":(
                                    "https://www.proteinatlas.org/search?format=tsv&download=yes"
                                    ),
                                    # "panglaoDB.tsv.gz":(
                                    #     "https://panglaodb.se/markers/"
                                    #     "PanglaoDB_markers_27_Mar_2020.tsv.gz"
                                    # ),
                                    # "cellmarker.xlsx":(
                                    #     "http://bio-bigdata.hrbmu.edu.cn/CellMarker/"
                                    #     "CellMarker_download_files/file/"
                                    #     "Cell_marker_Human.xlsx"
                                    # ),
                                    }
    _default_marker_annotations_to_df_kwargs = {}

    _default_map_marker_annotations_to_celltypes_kwargs = {'celltype_to_marker_reference_map':
        {'CD56+ natural killer cells':['NK cell'], 
        'CD4+ T cells':['CD4 T cell'],
        'CD8+ T cells':['CD8 T cell'],
        'CD19+ B cells':['B cell'],
        }}

    def __init__(self, marker_annotations_url: Optional[Dict[str,str]] = None,
    marker_annotations_to_df_func: Optional[Callable[[Path, ...], pd.DataFrame]] = None,
    marker_annotations_to_df_kwargs: Dict[str, Any] = {},
    map_marker_annotations_to_celltypes_func: Optional[Callable[[pd.DataFrame, ...], pd.DataFrame]] = None,
    map_marker_annotations_to_celltypes_kwargs: Dict[str, Any] = {},
    fig_A_markers_to_celltypes: Dict[str,str] = {'CD40':'CD19+ B cells',
                                                 'NCAM1':'CD56+ natural killer cells',
                                                 'CD4':'CD4+ T cells',
                                                 'CD8A':'CD8+ T cells',
                                                 },
    npts_kde: int = 1000,
    group_size: int = 6000,
    niter: int = 10,
    seed: Number = 42,
    *args, **kwargs):

        # experimental parameters
        self.kde_x = np.linspace(-0.1, 1, npts_kde)
        self.group_size = group_size
        self.niter = niter
        self.seed = seed
        self.fig_A_markers_to_celltypes = fig_A_markers_to_celltypes
        #parse the marker annotations functions
        self._marker_annotations_url = self._parse_marker_annotations_url(marker_annotations_url)

        #function for parsing marker annotations into a dataframe
        if marker_annotations_to_df_func is None:
            marker_annotations_to_df_func = self._default_marker_annotations_to_df
        if marker_annotations_to_df_func == self._default_marker_annotations_to_df:
            # parse the marker_annotaitons_to_df_kwargs to match defaults where appropriate
            for key in self._default_marker_annotations_to_df_kwargs:
                if key not in marker_annotations_to_df_kwargs:
                    marker_annotations_to_df_kwargs[key] = self._default_marker_annotations_to_df_kwargs[key]
        self._marker_annotations_to_df_kwargs = marker_annotations_to_df_kwargs

        #function and kwarg parsing for mapping marker annotations to cell types
        if map_marker_annotations_to_celltypes_func is None:
            map_marker_annotations_to_celltypes_func = self._default_map_marker_annotations_to_celltypes
        if map_marker_annotations_to_celltypes_func == self._default_map_marker_annotations_to_celltypes:
            # parse the marker annotations_to_celltypes_kwargs to match defaults where appropriate
            for key in self._default_map_marker_annotations_to_celltypes_kwargs:
                if key not in map_marker_annotations_to_celltypes_kwargs:
                    map_marker_annotations_to_celltypes_kwargs[key] = self._default_map_marker_annotations_to_celltypes_kwargs[key]
        self._map_marker_annotations_to_celltypes_kwargs = map_marker_annotations_to_celltypes_kwargs
        
        
        super().__init__(*args, **kwargs)
        #do some final things to register the loggers
        self._map_marker_annotations_to_celltypes_func = log_func_with(map_marker_annotations_to_celltypes_func, 
            self.logger.log_task, 'mapping marker annotations to cell types')
        self._marker_annotations_to_df_func = log_func_with(marker_annotations_to_df_func, 
            self.logger.log_task, 'parsing marker annotations into dataframe')

    def _layout(self):
        # subplot mosaic was not working for me, so I'm doing it manually
        figure_left = 0
        figure_right = 1
        figure_top = 1
        figure_bottom = 0.
        super_row_pad = 0.05
        super_column_pad = 0.06
        sub_row_pad = 0.025
        sub_column_pad = 0.005
        # full page width figure with ~1 in margin
        original_positions = {
            label: self[label].axis.get_position() for label in self._subfigures
        }
        new_positions = {label: pos for label, pos in original_positions.items()}
        #figure out super_column_pad by splitting the figure into 3 columns
        # super_column_pad = (figure_right - figure_left - 2*super_column_pad)/2
        #ignore A for now
        # Adjust B, C, and D to fill the space
        # B - B6 should be very close to one another as they share y axes

        # C - C6 should be very close to one another as they share y axes
        # D - D6 should be very close to one another as they share y axes
        # all of these will be padded by sub_column pad, and then (B-B6):(C-C6):(D-D6) will be padded by super_column_pad
        # compute the total space for B, C, and D
        A_space = (figure_right-figure_left-5*sub_column_pad)/6
        new_positions['A'].x0 = figure_left
        new_positions['A'].x1 = new_positions['A'].x0 + A_space
        for i in range(2,7):
            cur_label = f"A{i}"
            last_label = f"A{i-1}" if i > 2 else "A"
            cur = new_positions[cur_label]
            last = new_positions[last_label]
            cur.x0 = last.x1 + sub_column_pad
            cur.x1 = cur.x0 + A_space
            new_positions[cur_label] = cur
        BCD_space = (figure_right - figure_left-2*super_column_pad)
        # compute the space for each of B, C, and D
        col_space = (BCD_space)/3
        new_positions['B'].x0 = figure_left
        # new_positions['B'].x1 = figure_left + col_space
        new_positions['C'].x0 = new_positions['B'].x0 + col_space + super_column_pad
        # new_positions['C'].x1 = new_positions['C'].x0 + col_space
        new_positions['D'].x0 = new_positions['C'].x0 + col_space + super_column_pad

        # new_positions['D'].x1 = new_positions['D'].x0 + col_space
        #this is the space occupied by a subcolumn
        sub_space = (col_space - 5*sub_column_pad)/6
        # compute the new positions for B,C,D
        cols = ["B","C","D"]
        for ix,label in enumerate(cols):
            # set x0 and x1 for each subcolumn of label
            new_positions[label].x1 = new_positions[label].x0 + sub_space
            new_positions[label].y1 = new_positions['A'].y0 - super_row_pad
            for i in range(2,7):
                cur_label = f"{label}{i}"
                last_label = f"{label}{i-1}" if i > 2 else label
                cur = new_positions[cur_label]
                last = new_positions[last_label]
                cur.x0 = last.x1 + sub_column_pad
                cur.x1 = cur.x0 + sub_space
                cur.y1 = last.y1
                new_positions[f"{label}{i}"] = cur
                
        
        EFGH_space = new_positions['D'].x0-super_column_pad - figure_left - 3*sub_column_pad
        col_space = (EFGH_space)/4
        new_positions['E'].x0 = figure_left
        new_positions['E'].x1 = new_positions['E'].x0 + col_space
        EFGH = ['E','E2','E3','E4']
        for ix in range(1,len(EFGH)):
            cur = EFGH[ix]
            last = EFGH[ix-1]
            new_positions[cur].x0 = new_positions[last].x0 + col_space + sub_column_pad
            new_positions[cur].x1 = new_positions[cur].x0 + col_space
        for label in ['D','D2','D3','D4','D5','D6','E','E2','E3','E4']:
            new_positions[label].y0 = figure_bottom
        for label, pos in new_positions.items():
            self[label].axis.set_position(pos)
            self[label].axis.patch.set_alpha(0)

    def _parse_marker_annotations_url(self,
        marker_annotations_url: Optional[Dict[str,str]] = None) -> Dict[str,str]:
        """parse input marker annotations url and ensure it is valid. Pass back the 
        dictionary of {local filename: remote url}"""
        if marker_annotations_url is None:
            marker_annotations_url = self._default_marker_annotations_url
        else:
            if marker_annotations_url is not dict:
                raise ValueError("marker_annotations_url should be a dictionary")
            if not all([isinstance(v,str) for v in marker_annotations_url.values()]):
                raise ValueError("values of marker_annotations_url should be strings")
            if not all([isinstance(k,str) for k in marker_annotations_url.keys()]):
                raise ValueError("keys of marker_annotations_url should be strings")
            if len(marker_annotations_url) == 0:
                raise ValueError("marker_annotations_url should not be empty")
        return marker_annotations_url

    def _default_marker_annotations_to_df(self, marker_annotations_path: List[Path]) -> Dict[str,pd.DataFrame]:
        """default function to parse marker annotations into a dataframe"""
        marker_annotations = {}
        for filepath in marker_annotations_path:
            file = str(filepath)
            match file.lower():
                case str(s) if '.csv' in s:
                    delimiter = ','
                case str(s) if '.tsv' in s:
                    delimiter = '\t'
            match file.lower():
                case str(s) if '.xls' in s:
                    kwargs = dict()
                    read_func = pd.read_excel
                case str(s):
                    kwargs = dict(delimiter=delimiter)
                    read_func = pd.read_csv
            match file.lower():
                case str(s) if 'reference' in s:
                    key = 'reference'
                case str(s):
                    key = s.split('.')[0]
            match file.lower():
                case str(s) if 'panglao' in s:
                    parse_func = self._parse_panglao_markers
                case str(s) if 'cellmarker' in s:
                    parse_func = self._parse_cellmarker_markers
                case str(s) if 'hpa' in s:
                    parse_func = self._parse_hpa_markers
            if len(marker_annotations_path) == 1:
                key = 'reference'
            marker_annotations[key] = parse_func(read_func(filepath,**kwargs))
        return marker_annotations

    def _parse_hpa_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('Gene')
        df.index = pd.Index(df.index.str.normalize('NFC').str.upper().str.replace('-',''),name='gene')
        reference_df = df.copy()

        
        #there are three places where the cell type is listed.
        # 1. RNA single cell type specific nTPM
        # 2. RNA blood cell specific nTPM
        # 3. RNA blood lineage specific nTPM
        # we need to parse all of these and combine them into a single dataframe.
        #first, remove based on low specificity in blood cells or immune.
        mask1 = df['RNA blood cell specificity'].isin(['Not detected in immune cells',
                                                       'Low immune cell specificity'])
        mask2 = df['RNA blood lineage specificity'].isin(['Not detected',
                                                          'Low lineage specificity'])
        mask3 = (
            df['RNA blood lineage distribution'].isin(['Detected in all'])
            & ~df['RNA blood lineage specificity'].isin(['Lineage enriched',
                                                         'Group enriched'])
            )
        mask4 = ( 
            df['RNA blood cell distribution'].isin(['Detected in all'])
            & ~df['RNA blood cell specificity'].isin(['Immune cell enhanced',
                                                      'Immune cell enriched',
                                                      'Group enriched'])
            )
        mask = ~(mask1 | mask2 | mask3 | mask4)
        df = df[mask]
        # next, gather the different specificity columns
        vals = df[['RNA single cell type specific nTPM','RNA blood cell specific nTPM','RNA blood lineage specific nTPM']]
        # iterate over the columns to extract specific cell types
        vals_lst = []
        for col in vals.columns:
            if col == 'RNA blood cell specific nTPM':
                matchstr = r'(?P<B>B-cell)|(?P<NK>NK-cell)|(?P<CD4>CD4)|(?P<CD8>CD8)'
                add_cols = ['T']
            else:
                matchstr = r'(?P<B>B-cell)|(?P<NK>NK-cell)|(?P<T>T-cell)'
                add_cols = ['CD4','CD8']
            vals_c = vals[col].str.extractall(matchstr)
            vals_c = vals_c.fillna(False)
            vals_c = vals_c!=False
            vals_c = vals_c.groupby('gene',sort=True).any()
            for col in add_cols:
                if col == 'T':
                    vals_c[col] = vals_c['CD4'] & vals_c['CD8']
                else:
                    vals_c[col] = vals_c['T']
            vals_lst.append(vals_c)
            
        ordf = reduce(lambda x,y: x | y, vals_lst)
        # split up the T cell genes into CD4 and CD8
        ordf[['CD4','CD8']] = ordf[['CD4','CD8']] & vals_lst[1][['CD4','CD8']]
        # only grab genes that are annotated for a single cell type (exluding CD4 and CD8)
        bad_genes = ordf[(ordf.drop(columns=['CD4','CD8']).sum(axis=1)>1)].index 
        df = ordf.drop(bad_genes)

        df = df.drop(columns='T')
        not_annotated = ~df.any(axis=1)
        
        df = df.rename(columns=
                        {
                            'B': 'B cell',
                            'NK': 'NK cell',
                            'CD4': 'CD4 T cell',
                            'CD8': 'CD8 T cell'
                            })
        # manually remove genes that are not annotated correctly
        genes_removed = ['IGFLR1','PADI4','TLR1','IRGM',"CXCR6","RGS1"] 
        marker_df = df.drop(genes_removed,errors='ignore')

        return marker_df
    def _parse_panglao_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        """parse the panglao markers dataframe"""
        df = df.set_index('official gene symbol')
        df = df[
        (df['cell type'].isin(['B cells','T cells','T cytotoxic cells','T helper cells', 'NK cells'])) & 
        (df.species.str.contains('Hs'))].copy()
        df['value'] = True
        df = df.pivot_table(values='value',index='official gene symbol',columns=['cell type'],aggfunc=lambda x: True, fill_value=False)
        df.index = pd.Index(df.index.str.normalize('NFC').str.upper().str.replace('-',''),name='gene')
        
        df.loc[df[df['T cells']].index, ['T cytotoxic cells','T helper cells']] = True
        df = df.drop(columns=['T cells'])
        df.rename(columns={
                    'T cytotoxic cells':'CD8 T cell',
                    'T helper cells':'CD4 T cell',
                    'NK cells':'NK cell',
                    'B cells':'B cell'
                    },
                    inplace=True)
        return df

    def _parse_cellmarker_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        synonyms = {'NK cell': 
                    [
                        'Natural killer cell',
                        'CD56+ natural killer cell',
                        'CD56bright Natural killer cell',
                        'Natural killer CD56 bright cell'
                    ],
                    'T/NK cell': 
                    [
                        'T cell & Natural killer cell',
                        'NK and T cell'
                    ],
                    'T cell':
                    [
                        'T cell'
                    ],
                    'B cell':
                    [
                        'B cell lineage',
                        'B cell'
                    ],
                    'CD4 T cell':
                    [
                        'CD4+ T cell',
                        'CD4 T cell',
                        'CD4+ T helper cell',
                        'Conventional CD4+ T cell',
                        'Conventional CD4 T cell',
                        'CD40LG+ T helper cell',
                        'T helper cell',
                        'T helper(Th) cell',
                    ],
                    'CD8 T cell':
                    [
                        'CD8+ T cell',
                        'CD8 T cell',
                        'CD8+ T cytotoxic cell',
                        'Cytotoxic CD8+ T cell',
                        'Cytotoxic CD8 T cell',
                    ]
                    }
        # flip the synonym dictionary
        celltype_to_synonyms = {v.upper():k for k,values in synonyms.items() for v in values}
        celltypes_grab = list(celltype_to_synonyms.keys())
        # grab only normal cells from humans
        df = df[(df['cell_type']=='Normal cell') & (df['species'] == 'Human')]
        df = df.loc[:,['cell_name','marker','Symbol']]
        # convert the df to upper case, remove dashes, normalize unicode
        for col in df.columns:
            df[col] = df[col].str.normalize('NFC').str.upper().replace('-','')
        df.rename(columns={'cell_name':'cell type'},inplace=True)
        df = df[df['cell type'].isin(celltypes_grab)]
        #convert the cell names to their synonym
        df['cell type'] = df['cell type'].map(celltype_to_synonyms)
        # build a map between markers and symbols
        markers = df['marker']
        symbols = df['Symbol']
        symbol_mapping = {marker:symbol for marker,symbol in zip(markers,symbols) if not pd.isna(symbol)}  
        markers = markers[markers.isin(symbol_mapping.keys())]
        markers = markers.replace(symbol_mapping)
        df = df.loc[markers.index,:]
        df['gene'] = markers
        df['value'] = True
        df = df.pivot_table(values='value',index='gene',columns='cell type',aggfunc=lambda x: True,fill_value=False).astype(bool)
        df['NK cell'] = df['NK cell'] | df['T/NK cell']
        df['T cell'] = df['T cell'] | df['T/NK cell']
        df = df.drop(columns=['T/NK cell'])
        df['CD4 T cell'] = df['CD4 T cell'] | df['T cell']
        df['CD8 T cell'] = df['CD8 T cell'] | df['T cell']
        df = df.drop(columns=['T cell'])
        df = df[df.any(axis=1)]
        return df

    def _default_map_marker_annotations_to_celltypes(self,
        marker_annotations: Dict[str,pd.DataFrame],
        celltype_to_marker_reference_map: Optional[Dict[str, List[str]]] = None,
        ) -> pd.DataFrame:
        """default function to map marker annotations to cell types
        
        Parameters
        ----------
        marker_annotations: dict[str, pd.DataFrame]
            marker annotations dataframe, genes x cell types
        celltype_to_marker_reference_map: Optional[Dict[str, List[str]]]
            dictionary of cell types in data to reference cell types. If None,
            the cell types in the columns of marker_annotations are used.
        Returns
        -------
        pd.DataFrame
            genes x cell types boolean dataframe of markers. An entry is True if
            the gene is a marker for the cell type.
        """
        if len(marker_annotations) == 0:
            raise ValueError("marker_annotations should not be empty")
        if not all([isinstance(v,pd.DataFrame) for v in marker_annotations.values()]):
            raise ValueError("values of marker_annotations should be dataframes")
        if len(marker_annotations) == 1:
            marker_annotations = next(iter(marker_annotations.values()))
            other_dfs = False
        else:
            if 'reference' not in marker_annotations:
                raise ValueError("marker_annotations should have a reference key")
            else:
                other_dfs = {k:v for k,v in marker_annotations.items() if k != 'reference'}
                marker_annotations = marker_annotations['reference']
        #first, parse the optional kwargs
        if celltype_to_marker_reference_map is None:
            celltype_to_marker_reference_map = {c:[c] for c in marker_annotations.columns}
        else:
            if not isinstance(celltype_to_marker_reference_map, dict):
                raise ValueError("celltype_to_marker_reference_map should be a dictionary")
            if len(celltype_to_marker_reference_map) == 0:
                raise ValueError("celltype_to_marker_reference_map should not be empty")
            for key,value in celltype_to_marker_reference_map.items():
                # check that the values are lists of strings
                # if they are a string, make it a one element list
                if not isinstance(value,list):
                    if isinstance(value, str):
                        celltype_to_marker_reference_map[key] = [value]
                    else:
                        raise ValueError("values of celltype_to_marker_reference_map should be lists of strings")
                else:
                    if len(value) == 0:
                        raise ValueError("values of celltype_to_marker_reference_map should not be empty lists")
                    # check that the list of cell types are strings
                    if not all([isinstance(v,str) for v in value]):
                        raise ValueError("values of celltype_to_marker_reference_map should be lists of strings")
                if not isinstance(key,str):
                    raise ValueError("keys of celltype_to_marker_reference_map should be strings")
            
            if not all([v in marker_annotations.columns for val in celltype_to_marker_reference_map.values() 
                        for v in val]):
                raise ValueError("all values of celltype_to_marker_reference_map should be in the columns of marker_annotations")

        # flip the celltype_to_marker_reference_map so that the reference cell types are the keys
        marker_reference_to_celltypes_map = {}
        for celltype,reference_celltypes in celltype_to_marker_reference_map.items():
            if isinstance(reference_celltypes, str):
                reference_celltypes = [reference_celltypes]
            for reference_celltype in reference_celltypes:
                if (isinstance(already_saved_ct:=marker_reference_to_celltypes_map.get(reference_celltype,False),str) 
                    and (already_saved_ct != celltype)):
                    raise ValueError(f"reference cell type {reference_celltype} is repeated")
                else:
                    marker_reference_to_celltypes_map[reference_celltype] = celltype
        # rename the reference cell types in the marker_annotations
        marker_annotations = marker_annotations.rename(columns = marker_reference_to_celltypes_map)
        # if we have more things to compare against, we need to rename the columns in the other dataframes
        if other_dfs:
            other_dfs = [df.rename(columns = marker_reference_to_celltypes_map) for df in other_dfs]
            marker_annotations_reduced = reduce(lambda x,y: x&y, [marker_annotaions]+other_dfs)
            # if a gene was a marker in the reference and all the other dataframes, it is a marker
            # however, we want to keep genes that are markers in the reference that got filtered out
            # the filtering was only to remove ambiguities
            missing_genes = marker_annotations_reduced[~marker_annotations_reduced.any(axis=1)].index
            marker_annotations_reduced.loc[missing_genes,:] = marker_annotations.loc[missing_genes,:]
        else:
            marker_annotations_reduced = marker_annotations
        return marker_annotations_reduced
        
    def acquire_marker_annotations(self, marker_annotations_url: Optional[Dict[str,str]] = None):
        """download the marker annotations from the remote url to the local path"""
        if marker_annotations_url is None:
            marker_annotations_url = self._marker_annotations_url
        local_path,remote_path =  next(iter(self._marker_annotations_url.items()))
        local_path = str(self.figure_dir / local_path)
        get_files({local_path: remote_path},logger=self.logger)

    def extract_marker_annotations_to_df(self, marker_annotations_path: Optional[Path] = None,
        marker_annotations_to_df_func: Optional[Callable[[Path], pd.DataFrame]]= None, **kwargs) -> pd.DataFrame:
        """ extract the marker annotations into a genes x cell types dataframe.
        By default, the path to the marker annotations is assumed to be in the results directory. """
        if marker_annotations_to_df_func is None:
            marker_annotations_to_df_func = self._marker_annotations_to_df_func
        else:
            if not callable(marker_annotations_to_df_func):
                raise ValueError("marker_annotations_to_df_func should be a callable")
            else:
                marker_annotations_to_df_func = log_func_with(marker_annotations_to_df_func, 
                    self.logger.log_task, 'parsing marker annotations into dataframe')
        if marker_annotations_path is None:
            marker_annotations_path = [self.figure_dir / k for k in self._marker_annotations_url.keys()]
        paths_non_existant = list(filter(lambda path: not path.exists(), marker_annotations_path))
        if len(paths_non_existant) > 0:
            raise FileNotFoundError(f"{paths_non_existant} does not exist")
        return marker_annotations_to_df_func(marker_annotations_path, **kwargs)

    def map_marker_annotations_to_celltypes(self, marker_annotations: pd.DataFrame,
        map_marker_annotations_to_celltypes_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        **kwargs) -> pd.DataFrame:
        """map the marker annotations to cell types"""
        if map_marker_annotations_to_celltypes_func is None:
            map_marker_annotations_to_celltypes_func = self._map_marker_annotations_to_celltypes_func
        else:
            if not callable(map_marker_annotations_to_celltypes_func):
                raise ValueError("map_marker_annotations_to_celltypes_func should be a callable")
            else:
                map_marker_annotations_to_celltypes_func = log_func_with(map_marker_annotations_to_celltypes_func, 
                    self.logger.log_task, 'mapping marker annotations to cell types')
        return map_marker_annotations_to_celltypes_func(marker_annotations, **kwargs)
    

   
    def _compute_marker_annotations(self) -> pd.DataFrame:
        # compute marker annotations
        self.acquire_marker_annotations()
        marker_annotations = self.extract_marker_annotations_to_df(**self._marker_annotations_to_df_kwargs)
        marker_annotations = self.map_marker_annotations_to_celltypes(marker_annotations,
            **self._map_marker_annotations_to_celltypes_kwargs)
        return marker_annotations

 

    def _compute_subgroups(self, adata, group_size, niter, seed):
        with self.logger.log_task("subsampling groups for marker analysis"):
            # generate the group indices for each comparison
            rng = np.random.default_rng(self.seed)
            #get the super group sizes to check for any that are too small
            super_cluster_sizes = adata.obs['super_cluster'].value_counts()
            super_clusters = super_cluster_sizes.index
            bad_super_clusters_mask = super_cluster_sizes < self.group_size
            bad_super_clusters = super_cluster_sizes[bad_super_clusters_mask].index
            if (bad_super_clusters_mask).any():
                self.logger.log_info(f"{','.join(list(bad_super_clusters))} have fewer "
                                    f"cells than the group size of {self.group_size}. "
                                    "The figure will not subsample these clusters.")

            # for each super_cluster in the adata, we need to know how many sub clusters it contains
            group_indices = {super_cluster: [[] for _ in range(self.niter)] 
                            for super_cluster in super_clusters}

            for super_cluster in super_clusters:
                for i in range(self.niter):
                    if super_cluster in bad_super_clusters:
                        group_indices[super_cluster][i] = list(
                            np.where(adata.obs['super_cluster'] == super_cluster)[0])
                    else:
                        super_cluster_idxs = np.where(adata.obs['super_cluster'] == super_cluster)[0]
                        sub_cluster_sizes = adata.obs['cluster'].iloc[
                            super_cluster_idxs].cat.remove_unused_categories().value_counts()
                        sub_group_size = self.group_size//len(sub_cluster_sizes)
                        for ix,sub_cluster in enumerate(sub_cluster_sizes.sort_values().index):
                            sub_cluster_indices = np.where(adata.obs['cluster'] == sub_cluster)[0]
                            if len(sub_cluster_indices) < sub_group_size:
                                self.logger.log_info(f"{sub_cluster} has fewer "
                                                    f"cells than the group size of {sub_group_size}. "
                                                    "The figure will not subsample this cluster.")
                                group_indices[super_cluster][i].extend(sub_cluster_indices)
                                #update the sub_group_size so that we take more cells from future clusters.
                                sub_group_size = (
                                    self.group_size - len(group_indices[super_cluster][i])
                                    )//(len(sub_cluster_sizes) - (ix+1))
                            else:
                                
                                group_indices[super_cluster][i].extend(
                                    rng.choice(sub_cluster_indices, size = sub_group_size, replace = False))
                        if (remaining:=self.group_size-len(group_indices[super_cluster][i]))>0:
                            # grab some random, yet unsampled cells from the group
                            unsampled = np.setdiff1d(super_cluster_idxs,group_indices[super_cluster][i])
                            group_indices[super_cluster][i].extend(
                                rng.choice(unsampled, size = remaining, replace = False))
        return group_indices

    def _load_normalize_data(self, dataset=bipca_datasets.Zheng2017)->AnnData:
        #load the data
        dataset = bipca_datasets.Zheng2017(store_filtered_data=True, logger=self.logger
        )
        if (adata:=getattr(self, 'adata', None)) is None:
            adata = dataset.get_filtered_data(samples=["markers"])["markers"]
        path = dataset.filtered_data_paths["markers.h5ad"]
        todo = ["log1p", "log1p+z", "Pearson", "Sanity", "ALRA", "BiPCA"]
        bipca_kwargs = dict(n_components=-1,backend='torch', dense_svd=True,use_eig=True)
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        adata = apply_normalizations(adata, write_path = path,
                                    n_threads=64, apply=todo,
                                    normalization_kwargs={'BiPCA':bipca_kwargs},
                                    logger=self.logger)
        self.adata = adata
        return adata

    def _compute_marker_scores(self, adata, marker_annotations, group_indices,
        todo=["log1p", "log1p+z", "Pearson", "Sanity", "ALRA", "BiPCA"])->np.ndarray:
        # compute marker annotations
        results = []
        super_clusters = marker_annotations.columns
        markers = (marker_annotations.loc[
            marker_annotations.index.intersection(adata.var_names)
            ,:].sum(1) > 0)
        marker_annotations = marker_annotations.loc[markers[markers].index,:]
        adata.var['numerical_index'] = np.arange(len(adata.var_names), dtype=int)
        
        super_clusters = [[super_cluster] 
                        if super_cluster!='T cells' else ['CD4+ T cells','CD8+ T cells']
                         for super_cluster in super_clusters]
                
        mats = {}
        with self.logger.log_task('computing marker scores'):
            for layer in todo:
                if layer != "BiPCA":
                    layer_key = layer
                    process_func = lambda x: x if not issparse(x) else x.toarray()
                else:
                    layer_key = "Z_biwhite"
                    process_func = lambda x: x if not issparse(x) else x.toarray()
                    # original_scale = np.median(np.asarray(adata.X.sum(1)))
                    # process_func = lambda x: library_normalize(x,scale=original_scale)
                mats[layer] = np.asarray(process_func(adata.layers[layer_key]))
            
            indices = [np.hstack([group_indices[super_cluster[0]][i] 
                        for super_cluster in filter(lambda x: len(x)==1,super_clusters)])
                        for i in range(self.niter)]
            gxs = adata.var.reindex(marker_annotations.index)['numerical_index']
            with Pool(processes = 5) as pool:
                result = pool.starmap(_parallel_compute_metric, 
                map(
                    lambda tupl: (
                        #gene expression of group
                        mats[tupl[0]][tupl[1],:][:,gxs], 
                        #binarized labels for the group
                        np.c_[
                            tuple(
                                adata.obs.iloc[tupl[1]]['super_cluster'].
                                isin(super_cluster).values
                                for super_cluster in super_clusters)
                            ]
                        ),
                    itertools.product(todo,indices))
                )
            results = []
            super_clusters = [super_cluster[0]
                        if super_cluster!= ['CD4+ T cells','CD8+ T cells'] else 'T cells'
                         for super_cluster in super_clusters]
            for ix, res in enumerate(result):
                layer = todo[ix//self.niter]
                layer_results = np.vstack((
                        np.repeat(layer,len(res)),
                        np.repeat(marker_annotations.index,len(super_clusters)),
                        np.tile(super_clusters, len(marker_annotations.index)),np.repeat(ix%self.niter,len(res)),
                        res),
                    dtype=object
                ).T
                results.append(layer_results)
            results = np.vstack(results)
        return results
    
    def _split_A_results(self, results:np.ndarray) -> Dict[str,np.ndarray]:
        results = pd.DataFrame(results, columns=['method','gene', 'super_cluster','iter', 'score'])
        results = pd.pivot_table(results,index='gene',columns=['method','super_cluster','iter'])
        results.columns = results.columns.droplevel(0)
        results = results.astype(float)
        gms = results.T.groupby(level=['method','super_cluster']).mean()

        # for each gene and method, grab its AUC for the cell type it is mapped to
        with self.logger.log_task('splitting results into subfigures'):
            label=['A','A2','A3','A4','A5','A6']
            #now, iterate through the subfigure labels and map them to the correct data
            results_out = {lab:[['method','celltype','gene','AUC']] for lab in label}
            for lab in label:
                if len(lab) == 1:
                    subfig_ix = 0
                else:
                    subfig_ix = int(lab[1])-1
                method = self._ix_to_layer_mapping[subfig_ix]
                for gene, celltype in self.fig_A_markers_to_celltypes.items():
                    results_out[lab].append(np.hstack([method,celltype, gene, gms.loc[method,celltype].loc[gene]],dtype=object))
        results_out = {k:np.vstack(v,dtype=object) for k,v in results_out.items()}
        return results_out

    def _compute_kernel_density_estimates(self, adata):
        results_out = {}
        genes = list(self.fig_A_markers_to_celltypes.keys())
        gene2dataix = {
            gene:np.where(adata.var_names == gene)[0] 
            for gene in genes
            }
        celltype2dataix = {
            (celltype:=self.fig_A_markers_to_celltypes[gene]):
            np.where(adata.obs['super_cluster'] == celltype)[0] 
            for gene in genes
            }
        with self.logger.log_task('computing kernel density estimates'):
            for ix, method in self._ix_to_layer_mapping.items():
                layer = 'Z_biwhite' if method == 'BiPCA' else method
                y = [['method','celltype','gene']]
                y[0].extend(['fg']*len(self.kde_x))
                y[0].extend(['bg']*len(self.kde_x))
                

                with self.logger.log_task(f'computing KDE for {method}'):
                    data = adata.layers[layer]
                    for markerix, gene in enumerate(genes): 
                        genedata = feature_scale(data[:,gene2dataix[gene]])
                        genedata += np.abs(np.random.randn(*genedata.shape))*0.025
                        xmin = genedata.min()
                        xmax = genedata.max()
                        celltype = self.fig_A_markers_to_celltypes[gene]
                        other_celltypes = [ct for ct in celltype2dataix.keys() if ct != celltype]
                        ctix = celltype2dataix[celltype]
                        otherctix = np.hstack([celltype2dataix[ct] for ct in other_celltypes]).flatten()
                
                        y.append(np.hstack([
                            method, celltype, gene,
                            (
                                KDE(genedata[ctix,:].flatten())
                                .pdf(self.kde_x)
                            ),
                            (
                                KDE(genedata[otherctix,:].flatten())
                                .pdf(self.kde_x)
                            )
                            ], dtype=object
                            )
                        )
            
                        
                if ix == 0:
                    lab = 'A'
                else:
                    lab = f'A{ix+1}'
                results_out[lab] = np.vstack(y,dtype=object)
        return results_out
                    
    @is_subfigure(label=['A','A2','A3','A4','A5','A6'])
    def _compute_A(self):
        # compute marker annotations
        adata = self._load_normalize_data()
        groups = self._compute_subgroups(adata, self.group_size, self.niter, self.seed)
        marker_annotations = pd.DataFrame(index=self.fig_A_markers_to_celltypes.keys(),
                                        columns=self.fig_A_markers_to_celltypes.values())
        for key,item in self.fig_A_markers_to_celltypes.items():
            marker_annotations.loc[key,item] = True
        marker_annotations = marker_annotations.fillna(False)
        #compute AUCs and then split the results
        AUC_results = self._split_A_results(self._compute_marker_scores(adata, marker_annotations, groups))
        #next, we need to compute KDEs for every gene and method
        KDE_results = self._compute_kernel_density_estimates(adata)
        #merge the AUC results with the KDE results
        results = {}
        for key in AUC_results.keys():
            AR_tmp = pd.DataFrame(AUC_results[key][1:,:], columns=AUC_results[key][0,:])
            KR_tmp = pd.DataFrame(KDE_results[key][1:,:], columns=KDE_results[key][0,:])
            results[key] = KR_tmp.join(AR_tmp.set_index(['method','celltype','gene']),on=['method','celltype','gene'],how='inner')
            results[key] = np.vstack([results[key].columns,results[key].values])
        
        return results
    
    def _process_KDE_AUC_results(self, results)->pd.DataFrame:
        results = pd.DataFrame(results[1:,:],columns=results[0,:])
        results = results.set_index(['gene','method','celltype'])
        results = results.astype(float)
        return results
    def _plot_ridgeline(self, axis: mpl.axes.Axes, results: pd.DataFrame,sharey_label:Optional[str]=None)-> mpl.axes.Axes:
        axis.clear()
        method = results.index.get_level_values('method')[0]
        ct = results.index.get_level_values('celltype').values[::-1]
        genes = results.index.get_level_values('gene').values[::-1]
        ct = [f"{c}".replace('natural killer', 'NK').replace('+ ', '+\n') for c in ct]
        aucs = results.AUC.values[::-1]
        cix = marker_experiment_colors[r'$+$ cluster']
        ridgeline(results.fg.values.astype(float)[::-1,:], axis, plot_density,
            fill_color = [fill_cmap(cix)]*len(results), color= [line_cmap(cix)]*len(results),
            linewidth=1, apply_kde=False,vanish_at=False,overlap=0,xmin=np.min(self.kde_x),xmax=np.max(self.kde_x))
        cix = marker_experiment_colors[r'$-$ cluster']
        ridgeline(results.bg.values.astype(float)[::-1,:], axis, plot_density,
            fill_color = [fill_cmap(cix)]*len(results) , color= [line_cmap(cix)]*len(results),
            linewidth=1, apply_kde=False,vanish_at=False,overlap=0,xmin=np.min(self.kde_x),xmax=np.max(self.kde_x))
        axis.set_xlim(-0.05,1)
        set_spine_visibility(axis,status=False)
        axis.xaxis.set_label_position('top')
        axis.set_xlabel(method,fontsize=BIGGER_SIZE,verticalalignment='bottom')
        yticks = np.arange(0,len(results),1)

        if sharey_label is not None:
            axis.sharey(self[sharey_label].axis)
            axis.tick_params(axis="y", left=False, labelleft=False,which='both')
        else:
            axis.tick_params(axis="y", left=False, labelleft=True,pad=-4,which='both')
            axis.tick_params(axis="y", pad=-6)
        axis.set_yticks(yticks+0.5, genes,
                        fontsize=MEDIUM_SIZE,verticalalignment='bottom',
                        horizontalalignment='right')
        axis.set_yticks(yticks+0.1, ct, minor=True,
                        fontsize=SMALL_SIZE,
                        verticalalignment='bottom',
                        horizontalalignment='right')
        for text in axis.get_yticklabels(minor=True):
            bb = text.get_tightbbox().transformed(axis.transData.inverted())
            rect = mpl.patches.Rectangle((bb.x0-0.01, bb.y0+0.01),
                        abs(bb.x1-bb.x0)+0.01,abs(bb.y1-bb.y0)+0.02,
                        linewidth=0.5,edgecolor=line_cmap(marker_experiment_colors[r'$+$ cluster']),
                        facecolor=fill_cmap(marker_experiment_colors[r'$+$ cluster']),
                        zorder=-1,clip_on=False)
            axis.add_patch(rect)
        axis.axvline(0,color='k',linewidth=1)

        for ix in range(len(aucs)):
            string = r"$["+(r"%.2f]$" % aucs[ix]).lstrip('0')
            if string == '$[1.00]$':
                string = r'$[1]$'
            axis.text(1.0,1+yticks[ix]-0.35,string,fontsize=MEDIUM_SIZE,
                    verticalalignment='center',horizontalalignment='right')
        return axis
    #plotting routines for A
    @is_subfigure(label='A',plots=True)
    @label_me
    def _plot_A(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        results = self._process_KDE_AUC_results(results)
        axis = self._plot_ridgeline(axis, results)
        color_function = lambda x: line_cmap(x) if x not in [7,3] else fill_cmap(x)
        marker_function = lambda x: 's' if x in [r'$+$ cluster',r'$-$ cluster'] else "_"

        handles, tab10map = bipca.plotting.generate_custom_legend_handles(marker_experiment_colors,
                                                                        color_function = color_function,
                                                                        marker_function = marker_function)
        handles.append(mpl.lines.Line2D(
                [],
                [],
                marker=' ',
                linewidth=0,
                color='k',
                label = r'$[\mathrm{AUC}]$'))
        self.figure.legend(handles=handles,
            bbox_to_anchor=[0.85,axis.get_position().y0+0.002],fontsize=SMALL_SIZE,frameon=True,ncols=3,
            loc='upper center',handletextpad=0,columnspacing=0)
        self.figure.text(0.5,axis.get_position().y0-0.002,r'Scaled expression',fontsize=MEDIUM_SIZE,ha='center',va='top')
        return axis
    
    @is_subfigure(label='A2',plots=True)
    def _plot_A2(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        results = self._process_KDE_AUC_results(results)
        return self._plot_ridgeline(axis, results,sharey_label='A')
    
    @is_subfigure(label='A3',plots=True)
    def _plot_A3(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        results = self._process_KDE_AUC_results(results)
        return self._plot_ridgeline(axis, results,sharey_label='A')
    
    @is_subfigure(label='A4',plots=True)
    def _plot_A4(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        results = self._process_KDE_AUC_results(results)
        return self._plot_ridgeline(axis, results,sharey_label='A')
    
    @is_subfigure(label='A5',plots=True)
    def _plot_A5(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        results = self._process_KDE_AUC_results(results)
        return self._plot_ridgeline(axis, results,sharey_label='A')

    @is_subfigure(label='A6',plots=True)
    def _plot_A6(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        results = self._process_KDE_AUC_results(results)
        return self._plot_ridgeline(axis, results,sharey_label='A')


    def _split_BCDEFGH_results(self, results:np.ndarray,marker_annotations:pd.DataFrame)->Dict[str,np.ndarray]:
        results = pd.DataFrame(results, columns=['method','gene', 'super_cluster','iter', 'score'])
        results = pd.pivot_table(results,index='gene',columns=['method','super_cluster','iter'])
        results.columns = results.columns.droplevel(0)
        results = results.astype(float)
        gms = results.T.groupby(level=['method','super_cluster']).mean()
        vmin = gms.min().min()
        vmax = gms.max().max()
        # next, compute the correct orderings for the rows and columns
        with self.logger.log_task('reordering rows and columns'):
            idx = pd.IndexSlice
            super_clusters_reordered = self._celltype_ordering
            gms = gms.reindex(super_clusters_reordered,level='super_cluster',)
            marker_annotations = marker_annotations.reindex(gms.columns)
            gene_orders = {}
            for lab, target_cluster in self._subfig_to_celltype.items():
                target_cluster = [target_cluster] if not isinstance(target_cluster,list) else target_cluster
                annotated_genes = marker_annotations[marker_annotations[target_cluster].any(axis=1)]
                marks_this_cluster_mask = ~annotated_genes.drop(columns=target_cluster).any(axis=1)
                marks_this_cluster = annotated_genes[marks_this_cluster_mask].index
                col_inds = dendrogram(linkage(gms.loc[idx['BiPCA',:],marks_this_cluster].values.T),no_plot=True)['leaves']
                gene_orders[lab] = marks_this_cluster[col_inds]
            results = results.T.reindex(super_clusters_reordered,level='super_cluster')
        with self.logger.log_task('splitting results into subfigures'):
            label=['B','B2','B3','B4','B5','B6',
                    'C','C2','C3','C4','C5','C6',
                    'D','D2','D3','D4','D5','D6',
                    ]
            #now, iterate through the subfigure labels and map them to the correct data
            results_out = {}
            for lab in label:
                subfig = lab[0]
                if len(lab) == 1:
                    subfig_ix = 0
                else:
                    subfig_ix = int(lab[1])-1
                method = self._ix_to_layer_mapping[subfig_ix]
                # map the subfigure label to the correct subset of the genes
                marks_this_cluster = gene_orders[subfig]
                #extract the scores for the genes that are annotated to mark the super cluster
                scores = (
                    results.loc[idx[method,:],marks_this_cluster]
                    .melt(ignore_index=False).reset_index()[['super_cluster','gene','value']]
                    ).values
                results_out[lab] = np.vstack([np.round([vmin,vmax,0],2),
                                            ['cluster','gene',method],
                                            scores
                                                ])
            
        #next, for each super_cluster, gather all of its AUCs for the genes that are annotated to mark it
        resultsE = [np.array(['method','gene','celltype','indicator','AUC'],dtype=object)[None,:]]
        for method,group in gms.groupby('method'):
            aucs = []
            genes = []
            celltypes = []
            indicators = []

            for target_cluster in self._celltype_ordering:
                target_cluster = [target_cluster] if not isinstance(target_cluster,list) else target_cluster
                subgroup = group.loc[group.index.get_level_values('super_cluster').isin(target_cluster)]
                marks_this_cluster = marker_annotations[marker_annotations[target_cluster].any(axis=1)].index
                pos_aucs = subgroup[marks_this_cluster].values
                
                does_not_mark_this_cluster = marker_annotations[~marker_annotations[target_cluster].any(axis=1)].index
                neg_aucs = subgroup[does_not_mark_this_cluster].values
                super_cluster = target_cluster[0]
                marks_this_cluster = marks_this_cluster.values
                does_not_mark_this_cluster = does_not_mark_this_cluster.values
                genes.extend(marks_this_cluster.flatten())
                genes.extend(does_not_mark_this_cluster.flatten())
                
                celltypes.extend([super_cluster]*len(marks_this_cluster))
                celltypes.extend([super_cluster]*len(does_not_mark_this_cluster))
                aucs.extend(pos_aucs.flatten())
                aucs.extend(neg_aucs.flatten())
                indicators.extend(['+'] * len(pos_aucs.flatten()))
                indicators.extend(['-'] * len(neg_aucs.flatten()))
            genes = np.asarray(genes,dtype=object)[:,None]
            celltypes = np.asarray(celltypes,dtype=object)[:,None]
            indicators = np.asarray(indicators,dtype=object)[:,None]
            aucs = np.asarray(aucs,dtype=object)[:,None]
            meth_indicators = np.asarray([method] * (len(indicators)),dtype=object)[:,None]
            output = np.hstack([meth_indicators,genes,celltypes,indicators,aucs])
            resultsE.append(np.hstack([meth_indicators,genes,celltypes,indicators,aucs]))
        resultsE = np.vstack(resultsE)
        for label, target_cluster in zip(['E','E2','E3','E4'],self._celltype_ordering):
            results_out[label] = np.vstack([resultsE[0], resultsE[(resultsE[:,2] == target_cluster) &(resultsE[:,3] == '+')]])
        return results_out
    
    @is_subfigure(label=['B','B2','B3','B4','B5','B6',
                        'C','C2','C3','C4','C5','C6',
                        'D','D2','D3','D4','D5','D6',
                        'E','E2','E3','E4'
                        ])
    def _compute_BCDEFGH(self):
        # compute marker annotations
        marker_annotations = self._compute_marker_annotations()
        adata = self._load_normalize_data()
        group_indices = self._compute_subgroups(adata, self.group_size, self.niter, self.seed)
        results = self._compute_marker_scores(adata, marker_annotations, group_indices)
        # make a dataframe from the results, then split the dataframe according to the mapping
        results = self._split_BCDEFGH_results(results,marker_annotations)
        return results
    
    
    def _process_heatmaps(self, axis: mpl.axes.Axes, results: np.ndarray, 
    sharey_label:Optional[str] = None,
    return_im:bool=False)-> mpl.axes.Axes:
        # process the results into a heatmap
        axis.clear()
        vmin,vmax = 0,1
        method = results[1,-1]
        df = (
            pd.DataFrame(results[2:,:],columns=['super_cluster','gene','score'])
            .groupby(['super_cluster','gene'],sort=False)
            .mean().astype(float).unstack()).droplevel(0,axis=1).reindex(self._celltype_ordering).T
       
        im=axis.imshow(df.values, aspect='auto',
        interpolation='none',
        cmap=heatmap_cmap,
        norm=mpl.colors.CenteredNorm(vcenter=0.5,halfrange=0.5),
        )
        
        xlabels = df.columns.str.split(' ').str[0].values
        # xlabels[-1] = 'T cells'
        # xlabels = [f'{val}' for val in xlabels if val != 'T cells']
        axis.set_xticks(np.arange(len(xlabels)))
        axis.set_xticklabels(xlabels,rotation=90,rotation_mode='anchor',ha='right',va='center')
        axis.xaxis.set_label_position('top')
        axis.set_xlabel(method,fontsize=SMALL_SIZE,verticalalignment='bottom')

        axis.tick_params(axis="x", bottom=False, labelbottom=True,pad=-3,labelsize=5)

        ylabels = df.index.values
        ylabels = [f'{val}' for val in ylabels]
        axis.set_yticks(np.arange(len(ylabels)))
        axis.set_yticklabels(ylabels)
        if sharey_label is not None:
            axis.sharey(self[sharey_label].axis)
            axis.tick_params(axis="y", left=False, labelleft=False)
        else:
            axis.tick_params(axis="y", left=False, labelleft=True,pad=-2,labelsize=4.5)
        set_spine_visibility(axis,status=False)
        if return_im:
            return axis,im
        else:
            return axis
    
    def _add_AUC_colorbar(self,axes: List[mpl.axes.Axes],pad:Number=0.04)-> mpl.colorbar.Colorbar:
        # cax = self.figure.add_axes([axis_left.get_position().x0, axis_left.get_position().y0-0.045, 
                                   #axis_right.get_position().x1-axis_left.get_position().x0, 0.015])
        poss = [ax.get_position() for ax in axes]
        left = min([p.x0 for p in poss])
        right = max([p.x1 for p in poss])
        bottom = min([p.y0 for p in poss])
        top = max([p.y1 for p in poss])

        width = right - left
        bar_height = 0.015

        new_bottom = bottom + bar_height +pad 
        for pos,ax in zip(poss,axes):
            pos.y0 = new_bottom

            ax.set_position(pos)
        cbar_ax = self.figure.add_axes([left, bottom, width, bar_height])
        cbar = self.figure.colorbar(mpl.cm.ScalarMappable(
                norm=mpl.colors.CenteredNorm(vcenter=0.5,halfrange=0.5,),
                cmap=heatmap_cmap), cax=cbar_ax,
                orientation = 'horizontal')

        
        cbar.set_label(r'AUC',labelpad=-13)
        cbar.set_ticks(ticks = [0,0.25,0.5,0.75,1.0])
        cbar.set_ticklabels([r'$0$',r'$.25$',r'$.5$',r'$.75$',r'$1$'],horizontalalignment='center')
        cbar.ax.tick_params(axis='x', direction='out',length=2.5,pad=1)
        return cbar
    @is_subfigure(label=['B'], plots=True)
    @label_me(4)
    def _plot_B(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        axis,im = self._process_heatmaps(axis, results,return_im=True)
        # add the colorbar
        cbar = self._add_AUC_colorbar([axis, *[self[f'B{i}'].axis for i in range(2,7)]])
        title = self._subfig_to_celltype['B']
        title = title.replace('natural killer','NK').replace('cells', 'cell') if title != ['CD4+ T cells','CD8+ T cells'] else 'T cell'
        title += ' markers'
        self.figure.text((self['B3'].axis.get_position().x0+self['B4'].axis.get_position().x1)/2,axis.get_position().y1+0.025,title,fontsize=MEDIUM_SIZE,ha='center',va='top')

        return axis

    @is_subfigure(label=['B2'], plots=True)
    def _plot_B2(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="B")

    @is_subfigure(label=['B3'], plots=True)
    def _plot_B3(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="B")

    @is_subfigure(label=['B4'], plots=True)
    def _plot_B4(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="B")

    @is_subfigure(label=['B5'], plots=True)
    def _plot_B5(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="B")

    @is_subfigure(label=['B6'], plots=True)
    def _plot_B6(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="B")

    @is_subfigure(label=['C'], plots=True)
    @label_me(4)
    def _plot_C(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        axis,im = self._process_heatmaps(axis, results,return_im=True)
        label='C'
        #colorbar
        cbar = self._add_AUC_colorbar([axis, *[self[f'C{i}'].axis for i in range(2,7)]])
        title = self._subfig_to_celltype[label]
        title = title[0] if len(title) == 1 else title
        title = title.replace('natural killer','NK').replace('cells', 'cell') if title != ['CD4+ T cells','CD8+ T cells'] else 'T cell'
        title += ' markers'
        title = title.replace('CD56+ ','').replace('CD19+ ','')
        self.figure.text((self[f'{label}3'].axis.get_position().x0+self[f'{label}4'].axis.get_position().x1)/2,axis.get_position().y1+0.025,title,fontsize=MEDIUM_SIZE,ha='center',va='top')
        return axis

    @is_subfigure(label=['C2'], plots=True)
    def _plot_C2(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="C")

    @is_subfigure(label=['C3'], plots=True)
    def _plot_C3(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="C")

    @is_subfigure(label=['C4'], plots=True)
    def _plot_C4(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="C")

    @is_subfigure(label=['C5'], plots=True)
    def _plot_C5(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="C")

    @is_subfigure(label=['C6'], plots=True)
    def _plot_C6(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="C")

    @is_subfigure(label=['D'], plots=True)
    @label_me(4)
    def _plot_D(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        label='D'
        axis,im = self._process_heatmaps(axis, results,return_im=True)
        # add the colorbar
        cbar = self._add_AUC_colorbar([axis, *[self[f'D{i}'].axis for i in range(2,7)]])
        title = self._subfig_to_celltype[label]
        title = title[0] if len(title) == 1 else title
        title = title.replace('natural killer','NK').replace('cells', 'cell') if title != ['CD4+ T cells','CD8+ T cells'] else 'T cell'
        title += ' markers'
        title = title.replace('CD56+ ','').replace('CD19+ ','')
        self.figure.text((self[f'{label}3'].axis.get_position().x0+self[f'{label}4'].axis.get_position().x1)/2,axis.get_position().y1+0.025,title,fontsize=MEDIUM_SIZE,ha='center',va='top')

        return axis

    @is_subfigure(label=['D2'], plots=True)
    def _plot_D2(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="D")

    @is_subfigure(label=['D3'], plots=True)
    def _plot_D3(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="D")

    @is_subfigure(label=['D4'], plots=True)
    def _plot_D4(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="D")

    @is_subfigure(label=['D5'], plots=True)
    def _plot_D5(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="D")

    @is_subfigure(label=['D6'], plots=True)
    def _plot_D6(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="D")

    def _plot_EFGH(self, axis: mpl.axes.Axes, results: np.ndarray, sharey_label:Optional[str]=None)-> mpl.axes.Axes:
        results = pd.DataFrame(results[1:,:],columns=results[0]).set_index('method')
        celltype = results['celltype'].values[0].replace('natural killer','NK')
        group_df = results.groupby('method')

        to_plot = np.asarray([ np.asarray( group_df.get_group(method).AUC) for method in algorithm_color_index.keys()]).T
        boxplot(axis,to_plot,colors=list(algorithm_fill_color.values()))
        if sharey_label is not None:
            axis.sharey(self[sharey_label].axis)
            axis.tick_params(axis="y", left=False, labelleft=False)
        else:
            axis.set_yticklabels(list(algorithm_fill_color.keys()),fontsize=MEDIUM_SIZE,horizontalalignment='right')
            axis.tick_params(axis='y',pad=1,length=0)
        axis.set_xlabel('AUC')
        axis.set_xticks([0,0.2,0.4,0.6,0.8,1.0],labels = [0,.2,.4,.6,.8,1])
        axis.set_xticks([0.1,0.3,0.5,0.7,0.9],minor=True)
        axis.set_title(celltype)
        return axis
    @is_subfigure(label='E', plots=True)
    @label_me
    def _plot_E(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._plot_EFGH(axis, results, sharey_label=None)
    @is_subfigure(label='E2', plots=True)
    def _plot_E2(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._plot_EFGH(axis, results, sharey_label='E')
    @is_subfigure(label='E3', plots=True)

    def _plot_E3(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._plot_EFGH(axis, results, sharey_label='E')
    @is_subfigure(label='E4', plots=True)
    def _plot_E4(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._plot_EFGH(axis, results, sharey_label='E')

class Figure4(Figure):
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
        seed=42,
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


    def new_svd(self,X,r,which="left"):
    
        svd_op = bipca.math.SVD(n_components=-1,backend='torch',use_eig=True)
        U,S,V = svd_op.fit_transform(X)
        if which == "left":
            return (np.asarray(U)[:,:r])*(np.asarray(S)[:r])
        else:
            return np.asarray(U[:,:r]),np.asarray(S[:r]),np.asarray(V.T[:r,:])
    
    def libnorm(self,X):
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
        #adata_others = sc.read_h5ad(self.output_dir+"cbmc_bipca.h5ad")
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
        #adata_others = sc.read_h5ad(self.output_dir+"bm_bipca.h5ad")
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
            #print(org_mat.shape)
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
        
    def runClassification_cbmc(self,adata,dataset,r_list,n_iterations=10):
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
            for round in range(n_iterations):
                print("- Iteration: {}".format(round))
                # run classification for each method
                for ix,method in enumerate(dataset):
            
                    if method == "bipca":
                        X_input = dataset[method][i]
            
                    else:
                        X_input = dataset[method][:,:rank2keep]
            
                    test_acc,_,k = knn_classifier(X_input,y,train_ratio=0.2,train_metric=balanced_accuracy_score,K=grid_list,k_cv=5,random_state=round,KNeighbors_kwargs={"n_jobs":30})

                    acc_df.loc[(acc_df["methods"] == method) & (acc_df["rank"] == ranks[i]) & (acc_df["rounds"] == str(round)),"ACC"] = test_acc
        
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

    def runClassification_bm(self,adata,dataset,r_list,n_iterations=10):
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
            for round in range(n_iterations):
                print("- Iteration: {}".format(round))
                # run classification for each method
                for ix,method in enumerate(dataset):
            
                    if method == "bipca":
                        X_input = dataset[method][i]
            
                    else:
                        X_input = dataset[method][:,:rank2keep]
            
                    test_acc,_,k = knn_classifier(X_input,y,train_ratio=0.6,train_metric=balanced_accuracy_score,K=grid_list,k_cv=5,random_state=round,KNeighbors_kwargs={"n_jobs":30})

                    acc_df.loc[(acc_df["methods"] == method) & (acc_df["rank"] == ranks[i]) & (acc_df["rounds"] == str(round)),"ACC"] = test_acc
        
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
        """compute_A Generate subfigure 4A"""
        op, adata = self.load_Stoeckius2017()
        
        self.bipca_rank_A = op.mp_rank
        r_list, external_s_list = self.get_r_shrink_s(op, adata)
        dataset = self.getPCs(op,r_list,external_s_list,adata)
        acc_df_mean,acc_df_std = self.runClassification_cbmc(adata,dataset,r_list)
        results = np.concatenate((r_list.reshape(-1,1),np.concatenate((acc_df_mean.values,acc_df_std.values),axis=1)),axis=1)
        return results

    @is_subfigure(label="A", plots=True)
    @label_me
    def _plot_A(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        """plot_A Plot the results of subfigure 4A."""
        
        # to replace
        r_list = results[:,0].reshape(-1)
        r_list2show = r_list[3:]
        r_list2show_idx = np.array([np.where(rank == r_list)[0][0] for rank in r_list2show])
        acc_df_mean = pd.DataFrame(results[:,1:7],columns=["ALRA","SCT","Sanity","bipca","log1p","log1p_z"]) # change to 1:7
        acc_df_std = pd.DataFrame(results[:,7:],columns=["ALRA","SCT","Sanity","bipca","log1p","log1p_z"]) # change to 7:
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
        """compute_B Generate subfigure 4B, cell type classification using bone marrow cite-seq data."""
        op, adata = self.load_Stuart2019()
        self.bipca_rank_B = op.mp_rank
        r_list, external_s_list = self.get_r_shrink_s(op, adata)
        dataset = self.getPCs(op,r_list,external_s_list,adata)
        acc_df_mean,acc_df_std = self.runClassification_bm(adata,dataset,r_list)
        results = np.concatenate((r_list.reshape(-1,1),np.concatenate((acc_df_mean.values,acc_df_std.values),axis=1)),axis=1)
        return results


    @is_subfigure(label="B", plots=True)
    @label_me
    def _plot_B(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        """plot_B Plot the results of subfigure 4B."""
        
        # to replace
        r_list = results[:,0].reshape(-1)
        r_list2show = r_list[4:]
        r_list2show_idx = np.array([np.where(rank == r_list)[0][0] for rank in r_list2show])
        acc_df_mean = pd.DataFrame(results[:,1:7],columns=["ALRA","SCT","Sanity","bipca","log1p","log1p_z"]) # change to 1:7
        acc_df_std = pd.DataFrame(results[:,7:],columns=["ALRA","SCT","Sanity","bipca","log1p","log1p_z"]) # change to 7:
        ymin = 0.7
        ymax = 0.85

        y_bipca_annotation_calibration = np.hstack(([0.02]*2,[0.005]*(len(r_list2show_idx)-2)))

        axis.errorbar(r_list2show,acc_df_mean.log1p[r_list2show_idx],acc_df_std.log1p[r_list2show_idx],label="log1p",c="deepskyblue")
        axis.errorbar(r_list2show,acc_df_mean.log1p_z[r_list2show_idx],acc_df_std.log1p_z[r_list2show_idx],label="log1p_z",c="blue")
        for i,r2show in enumerate(r_list2show_idx):
            axis.annotate( "%.4f" % acc_df_mean.bipca[r2show], (r_list2show[i], acc_df_mean.bipca[r2show]+y_bipca_annotation_calibration[i]),c="red")

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
        
class Figure5(Figure):
    _figure_layout = [
        ["A", "A", "A2", "A2", "A3", "A3"],
        ["A4", "A4", "A5", "A5", "A6", "A6"],
        ["B", "B", "B2", "B2", "B3", "B3"],
        ["B4", "B4", "B5", "B5", "B6", "B6"],
        ["C", "C", "C2", "C2", "C3", "C3"],
    ]

    def __init__(
        self,
        output_dir = "./",
        n_repeats = 10, # number of repeats for computing knn stats
        n_neighbors = 50, # number of neighbors for computing knn stats
        seed = 42,
        *args,
        **kwargs,
    ):
        self.output_dir = output_dir
        self.n_repeats = n_repeats
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.results = {}
        self.baseline_methods_order = np.array([ 'log1p', 'log1p+z', 'Pearson', 'Sanity','ALRA', 'BiPCA'])
        super().__init__(*args, **kwargs)

    
    
    def load_data(self):

        adata = bipca_datasets.SCORCH_INS_OUD(base_data_directory = self.output_dir).get_filtered_data()['full']
        # run bipca
        torch.set_num_threads(36)
        with threadpool_limits(limits=36):
            op = bipca.BiPCA(n_components=-1,seed=self.seed)
            Z = op.fit_transform(adata.X.toarray())
            op.write_to_adata(adata)

        #
        adata.write_h5ad(self.output_dir+"ins_oud_bipca.h5ad")
        # run other normalization methods
        adata_others = apply_normalizations(self.output_dir+"ins_oud_bipca.h5ad")

        #adata_others = sc.read_h5ad(self.output_dir+"ins_oud_bipca.h5ad")
        self.r2keep = adata_others.uns['bipca']['rank']
        return adata_others

    def runPCA(self,adata): 
        
        
        dataset = OrderedDict()
        PCset = OrderedDict()

        dataset["BiPCA"] = library_normalize(adata.layers['Z_biwhite'])
        dataset["log1p"] = adata.layers["log1p"].toarray()
        dataset["log1p+z"] = adata.layers["log1p+z"].toarray()
        dataset["Pearson"] = adata.layers['Pearson']
        dataset["ALRA"] =  adata.layers['ALRA']
        dataset["Sanity"] = adata.layers['Sanity']

        PCset["BiPCA"] = new_svd(dataset["BiPCA"],self.r2keep)
        PCset["log1p"] = new_svd(dataset["log1p"],self.r2keep)
        PCset["log1p+z"] = new_svd(dataset["log1p+z"],self.r2keep)
        PCset["Pearson"] = new_svd(dataset["Pearson"],self.r2keep)
        PCset["ALRA"] =  new_svd(dataset["ALRA"],self.r2keep)
        PCset["Sanity"] = new_svd(dataset["Sanity"],self.r2keep)

        return dataset, PCset

    def runKNNstats(self, PCset, adata, astrocyte_mask, batch_mask):

        n_methods = 6 # all baselines
        #knn_list = np.zeros((n_methods,self.n_repeats,1))
        knn_list = {m:np.zeros(self.n_repeats) for m in self.baseline_methods_order}
        
        for rix,method in enumerate(PCset):
    
            X1 = PCset[method][astrocyte_mask & batch_mask,:]
            X_others = PCset[method][astrocyte_mask & (~batch_mask),:]
        
            # subsample the data to the same number
            n_cell2keep = np.min([X1.shape[0],X_others.shape[0]])
            rng = np.random.default_rng(self.seed)
            g12keep = rng.choice(X1.shape[0],n_cell2keep,replace=False)
            rng = np.random.default_rng(self.seed)
            g22keep = rng.choice(X_others.shape[0],n_cell2keep,replace=False)
            X_input = np.concatenate((X1[g12keep,:],X_others[g22keep,:]),axis=0)    
            y_input = np.concatenate(([0]*n_cell2keep,[1]*n_cell2keep))
            
        
            
            Dist_mat = euclidean_distances(X_input)
        
            # 80% of the data, repeated 10 times
            for rand_ix in range(self.n_repeats): 
                new_rng = np.random.default_rng(rand_ix)
                sample2keep = new_rng.choice(X_input.shape[0],int(X_input.shape[0]*0.8),replace=False)

                # compute the knn stats on the tsne coords
                knn_list[method][rand_ix] = knn_test_k(X=X_input[sample2keep,:],
                                          y=y_input[sample2keep],
                                          k=self.n_neighbors,
                                          Dist_mat=Dist_mat[sample2keep,:][:,sample2keep])

        return knn_list

    def runAffineGrassman(self,dataset, astrocyte_mask, batch_mask):

        ag_results = {}
        for k,data in dataset.items():
    
            Y0 = compute_stiefel_coordinates_from_data(data[(~batch_mask) & astrocyte_mask,:],self.r2keep,0)
            Y1 = compute_stiefel_coordinates_from_data(data[(batch_mask) & astrocyte_mask,:],self.r2keep,0)
            S = bipca.math.SVD(backend='torch', use_eig=True,vals_only=True,verbose=False).fit(Y0.T@Y1).S
            ag_results[k] = np.sqrt((np.arccos(S)**2).sum())

        return ag_results

    def runTSNE(self,PCset,astrocyte_mask):

        tsne_embeddings_full = OrderedDict()
        for method,PCs in PCset.items():
            tsne_embeddings_full[method] = np.array(TSNE().fit(PCs))
        
        # only astrocytes
        tsne_embeddings_sub = OrderedDict()
        for method,TSNEs in tsne_embeddings_full.items():
            tsne_embeddings_sub[method] = TSNEs[astrocyte_mask,:]
        
        return tsne_embeddings_full, tsne_embeddings_sub

    def runDE(self,dataset,astrocyte_mask, batch_mask):       
        DE_p_results = OrderedDict()
        logfc_results = OrderedDict()

        for method,data in dataset.items():
            DE_p_results[method] = manwhitneyu_de(data,
                        astrocyte_mask, batch_mask)

            logfc_results[method] = log2fc(data,
                       astrocyte_mask, batch_mask)

        DE_cutoff = 1e-2
        n_DE = {}
        for method in DE_p_results.keys():
            valid_genes = ~np.isnan(DE_p_results[method])
            n_DE[method] = np.sum( DE_p_results[method][valid_genes] < DE_cutoff )


        return n_DE
        
    
        
    @is_subfigure(label=["A","A2","A3","A4","A5","A6","B","B2","B3","B4","B5","B6","C","C2","C3"])
    #@is_subfigure(label="A")
    def _compute_A_B_C(self):

        #results = np.array([0])
        adata = self.load_data()
        clabels = pd.factorize(adata.obs['replicate_id'].values.astype(int))[0]

        
        dataset, PCset = self.runPCA(adata)
        
        astrocyte_mask = adata.obs['cell_types']=='Astrocytes'
        batch_mask = adata.obs['replicate_id'].astype(int) == 1

        # for subplots
        replicate_id_mapper = {0:0,1:5,2:5,3:5}
        clabels_sub = np.array([replicate_id_mapper[int(i)-1] for i in adata[astrocyte_mask,:].obs['replicate_id'].values])

        knn_list = self.runKNNstats(PCset, adata, astrocyte_mask, batch_mask)
        ag_results = self.runAffineGrassman(dataset, astrocyte_mask, batch_mask)
        tsne_embeddings_full, tsne_embeddings_sub = self.runTSNE(PCset,astrocyte_mask)
        n_DE = self.runDE(dataset,astrocyte_mask, batch_mask)
        
        results = {"A":{"embed_df":tsne_embeddings_full["log1p"],"clabels":clabels},
                   "A2":{"embed_df":tsne_embeddings_full["log1p+z"],"clabels":clabels},
                   "A3":{"embed_df":tsne_embeddings_full["Pearson"],"clabels":clabels},
                   "A4":{"embed_df":tsne_embeddings_full["Sanity"],"clabels":clabels},
                   "A5":{"embed_df":tsne_embeddings_full["ALRA"],"clabels":clabels},
                   "A6":{"embed_df":tsne_embeddings_full["BiPCA"],"clabels":clabels},
                   #"B":tsne_embeddings_sub,
                   "B":{"embed_df":tsne_embeddings_sub["log1p"],"clabels":clabels_sub},
                   "B2":{"embed_df":tsne_embeddings_sub["log1p+z"],"clabels":clabels_sub},
                   "B3":{"embed_df":tsne_embeddings_sub["Pearson"],"clabels":clabels_sub},
                   "B4":{"embed_df":tsne_embeddings_sub["Sanity"],"clabels":clabels_sub},
                   "B5":{"embed_df":tsne_embeddings_sub["ALRA"],"clabels":clabels_sub},
                   "B6":{"embed_df":tsne_embeddings_sub["BiPCA"],"clabels":clabels_sub},
                   "C":n_DE,
                   "C2":knn_list,
                   "C3":ag_results}

        return results

    def _tsne_plot(self,embed_df,ax,clabels,POINT_SIZE = 0.2):

        cmap = npg_cmap(alpha=1)
        ax.scatter(x=embed_df[:,0],
            y=embed_df[:,1],
            edgecolors=None,
            marker='.',
            linewidth=POINT_SIZE,
            facecolors= [mpl.colors.to_rgba(cmap(label)) for label in clabels],
            s=POINT_SIZE)
    
    
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        return ax
    
    @is_subfigure(label="A", plots=True)
    @label_me
    def _plot_A(self, axis: mpl.axes.Axes, esults: np.ndarray) -> mpl.axes.Axes:


        embed_df = results[()]["embed_df"]
        clabels = results[()]["clabels"]      
        # insert a new axis for title
        axis2 = axis.inset_axes([0,0,1,1])
        
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

        axis2 = self._tsne_plot(embed_df,axis2,clabels) 
        axis2.set_title("log1p",loc="left")
        
        return axis

    @is_subfigure(label="A2", plots=True)
    def _plot_A2(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        
        embed_df = results[()]["embed_df"]
        clabels = results[()]["clabels"]
        axis = self._tsne_plot(embed_df,axis,clabels)        
        axis.set_title("log1p+z",loc="left")
        
        return axis

    @is_subfigure(label="A3", plots=True)
    def _plot_A3(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        
        embed_df = results[()]["embed_df"]
        clabels = results["clabels"]
        axis = self._tsne_plot(embed_df,axis,clabels)        
        axis.set_title("Pearson",loc="left")
        
        return axis
    
    @is_subfigure(label="A4", plots=True)
    def _plot_A4(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        
        embed_df = results[()]["embed_df"]
        clabels = results[()]["clabels"]
        axis = self._tsne_plot(embed_df,axis,clabels)        
        axis.set_title("Sanity",loc="left")

        # add tsne axis
        axis.spines['left'].set_visible(True)
        axis.spines['bottom'].set_visible(True)
        axis.set_xlim(left=-90)
        axis.set_ylim(bottom=-90)

        axis.spines['left'].set_bounds(-90, -30)
        axis.spines['bottom'].set_bounds(-90, -50)
        axis.set_ylabel('T-SNE 2',fontsize=4)
        axis.set_xlabel('T-SNE 1',fontsize=4)
        axis.xaxis.set_label_coords(0.15, -0.05)
        axis.yaxis.set_label_coords(-0.02, 0.2)
        axis.plot(0.21, -90, ">k", ms=2, transform=axis.get_yaxis_transform(), clip_on=False)
        axis.plot(-90, 0.32, "^k", ms=2, transform=axis.get_xaxis_transform(), clip_on=False)

        
        return axis

    @is_subfigure(label="A5", plots=True)
    def _plot_A5(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:

        embed_df = results[()]["embed_df"]
        clabels = results[()]["clabels"]
        axis = self._tsne_plot(embed_df,axis,clabels)        
        axis.set_title("ALRA",loc="left")


        # add legend
        cmap = npg_cmap(alpha=1)
        tab10map = [(key,mpl.colors.to_rgba(cmap(label))) for key,label in zip([1,2,3,4],[0,1,2,3])]
        label_mapper = {k+1:"Replicate "+str(k+1) for k in range(4)}
        handles_full = [
            mpl.lines.Line2D(
            [],
            [],
            marker='.',
            color=color,
            linewidth=0,
            label=label_mapper[label],
            markersize=10,
            )
            for label, color in tab10map
        ]

        legend = axis.legend(
            handles=handles_full,
            ncol=4,fontsize=MEDIUM_SIZE,bbox_to_anchor=[0.4, -0.2], 
            loc='center',handletextpad=0,columnspacing=0
        )
        
        
        return axis

    @is_subfigure(label="A6", plots=True)
    def _plot_A6(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:

        embed_df = results[()]["embed_df"]
        clabels = results[()]["clabels"]
        
        # insert a new axis and plot tsne inside
        #axis2 = axis.inset_axes([0,0.3,1,0.6])
        axis = self._tsne_plot(embed_df,axis,clabels)
        axis.set_title("BiPCA",loc="left")
        
        
        return axis

    @is_subfigure(label="B", plots=True)
    @label_me
    def _plot_B(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:

        embed_df = results[()]["embed_df"]
        clabels = results[()]["clabels"]      
        # insert a new axis for title
        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,clabels,POINT_SIZE = 0.5) 
        axis2.set_title("log1p",loc="left")
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

        return axis
        

    @is_subfigure(label="B2", plots=True)
    def _plot_B2(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        embed_df = results[()]["embed_df"]
        clabels = results[()]["clabels"]
        
        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,clabels,POINT_SIZE = 0.5) 
        
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        axis.set_title("log1p+z",loc="left")
        
        return axis

    @is_subfigure(label="B3", plots=True)
    def _plot_B3(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:       
        embed_df = results[()]["embed_df"]
        clabels = results[()]["clabels"]

        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,clabels,POINT_SIZE = 0.5) 
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
                
        axis.set_title("Pearson",loc="left")
        return axis

    @is_subfigure(label="B4", plots=True)
    def _plot_B4(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:

        embed_df = results[()]["embed_df"]
        clabels = results[()]["clabels"]


        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,clabels,POINT_SIZE = 0.5) 
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
                
        axis.set_title("Sanity",loc="left")

        # add tsne axis
        axis2.spines['left'].set_visible(True)
        axis2.spines['bottom'].set_visible(True)
        axis2.set_xlim(left=-50)
        axis2.set_ylim(bottom=-20)

        axis2.spines['left'].set_bounds(-20,13)
        axis2.spines['bottom'].set_bounds(-50, -30)
        axis2.set_ylabel('T-SNE 2',fontsize=4)
        axis2.set_xlabel('T-SNE 1',fontsize=4)
        axis2.xaxis.set_label_coords(0.15, -0.05)
        axis2.yaxis.set_label_coords(-0.02, 0.2)
        axis2.plot(0.2, -20, ">k", ms=2, transform=axis2.get_yaxis_transform(), clip_on=False)
        axis2.plot(-50, 0.32, "^k", ms=2, transform=axis2.get_xaxis_transform(), clip_on=False)

        return axis

    @is_subfigure(label="B5", plots=True)
    def _plot_B5(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
     
        embed_df = results[()]["embed_df"]
        clabels = results[()]["clabels"]

        axis2 = axis.inset_axes([-0.2,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,clabels,POINT_SIZE = 0.5) 
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
                
        axis.set_title("ALRA",loc="left")

        # add legend
        cmap = npg_cmap(alpha=1)
        tab10map  = [(key,mpl.colors.to_rgba(cmap(label))) for key,label in zip([1,6],[0,5])]

        label_mapper = {k+1:"Replicate "+str(k+1) for k in range(4)}
        label_mapper[6] = "Others"
        
        handles_sub = [
            mpl.lines.Line2D(
            [],
            [],
            marker='.',
            color=color,
            linewidth=0,
            label=label_mapper[label],
            markersize=10,
            )
            for label, color in tab10map
        ]

        legend = axis.legend(
            handles=handles_sub,
            ncol=2,fontsize=MEDIUM_SIZE,bbox_to_anchor=[0.4, -0.2], 
            loc='center',handletextpad=0,columnspacing=0
        )
        
        
        return axis

    @is_subfigure(label="B6", plots=True)
    def _plot_B6(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        embed_df = results[()]["embed_df"]
        clabels = results[()]["clabels"]

        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,clabels,POINT_SIZE = 0.5) 
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        
               
        axis.set_title("BiPCA",loc="left")
        return axis
    
    @is_subfigure(label="C", plots=True)
    @label_me
    def _plot_C(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:

        n_DE = results[()]
        n_DE_ordered = {k:n_DE[k] for k in self.baseline_methods_order}
        y_pos = np.linspace(0,1.0,6)
        cmap = npg_cmap(1)
        bar_colors=[cmap(algorithm_color_index[method]) for method in self.baseline_methods_order]

        axis.barh(y_pos,
            width=np.array(list(n_DE_ordered.values())), 
            height=np.diff(y_pos)[0],
            color=bar_colors,
        edgecolor='k')
        axis.set_xlabel('\# DE')
        axis.invert_yaxis()
        axis.set_yticks(y_pos, labels=list(n_DE_ordered.keys()))

        
        return axis

    @is_subfigure(label="C2", plots=True)
    def _plot_C2(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:

        y_pos = np.linspace(0,1.0,6)
        cmap = npg_cmap(1)
        bar_colors=[cmap(algorithm_color_index[method]) for method in self.baseline_methods_order]

        knn_list = results[()]
        
        knn_mean = [np.mean(knn_list[method]) for method in self.baseline_methods_order]
        knn_std = [np.std(knn_list[method]) for method in self.baseline_methods_order]
        axis.barh(y_pos,
            width= knn_mean,
            height=np.diff(y_pos)[0],
            xerr = knn_std,
            color=bar_colors,
            edgecolor='k')
        axis.set_xlabel('Local homogeneity')
        axis.invert_yaxis()
        axis.set_xlim(left=0.5)

        axis.set_yticks(y_pos, labels=self.baseline_methods_order)

        
        return axis

    @is_subfigure(label="C3", plots=True)
    def _plot_C3(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:

        ag_results = results[()]
        y_pos = np.linspace(0,1.0,6)
        cmap = npg_cmap(1)
        bar_colors=[cmap(algorithm_color_index[method]) for method in self.baseline_methods_order]

        axis.barh(y_pos,
                  [ag_results[k] for k in self.baseline_methods_order],
                  height=np.diff(y_pos)[0],
                  color=bar_colors,
                  edgecolor='k')
        axis.set_yticks(y_pos, labels=self.baseline_methods_order)
        axis.invert_yaxis()
        axis.set_xlabel('Affine Grassmann distance')

        return axis



class Figure6(Figure):
    _figure_layout = [
        ["A", "B"]
    ]

    def __init__(
        self,
        output_dir = "./",
        num_threads=36,
        seed = 42,
        POINT_SIZE = 20,
        text_S = 10,
        *args,
        **kwargs,
    ):
        self.output_dir = output_dir
        self.num_threads = num_threads
        self.seed = seed
        self.POINT_SIZE = POINT_SIZE
        self.text_S = text_S
        self.results = {}
        super().__init__(*args, **kwargs)

    def load_data(self):

        adata = bipca_datasets.Byrska2022(base_data_directory = self.output_dir).get_filtered_data()['full']
        # run bipca
        torch.set_num_threads(self.num_threads)
        with threadpool_limits(limits=self.num_threads):
            op = bipca.BiPCA(n_components=-1,variance_estimator="binomial",read_counts=2,seed=self.seed)
            Z = op.fit_transform(adata)
            op.write_to_adata(adata)

        self.r2keep = adata.uns['bipca']['rank']
       
        return adata

    def runPCA(self,adata): 
        
        PCset = OrderedDict()

        
        PCset["BiPCA"] = new_svd(StandardScaler(with_std=False).fit_transform(library_normalize(adata.layers['Z_biwhite'])),self.r2keep)
        PCset["noBiPCA"] = new_svd(StandardScaler(with_std=False).fit_transform(library_normalize(adata.X)),self.r2keep)
        
        return PCset


    def runTSNE(self,PCset):

        tsne_embeddings = OrderedDict()
        for method,PCs in PCset.items():
            tsne_embeddings[method] = np.array(TSNE().fit(PCs))       
         
        return tsne_embeddings

    @is_subfigure(label=["A","B"])
    def _compute_A_B(self):

        
        adata = self.load_data()
        population_names = list(Counter(adata.obs['Population'].values).keys())
        clabels = pd.factorize(np.unique(adata.obs['Population'].values))

        PCset = self.runPCA(adata)
        tsne_embeddings = self.runTSNE(PCset)
        
        results = {"A":{"embed_df":tsne_embeddings["BiPCA"],"clabels":clabels,"populations":adata.obs['Population'].values},
                   "B":{"embed_df":tsne_embeddings["noBiPCA"],"clabels":clabels,"populations":adata.obs['Population'].values}
                   }

        return results
        
    def _tsne_plot(self,embed_df,axs,clabels,populations):

        
        
        clrs = sns.color_palette('husl', n_colors=len(clabels[0])) 


        for pix in clabels[0]:
            pname = clabels[1][clabels[0][pix]]
            ix = np.where(populations == pname)[0]
            axs1 = axs.scatter(x=embed_df[ix,0], y=embed_df[ix,1],
                               edgecolors=None,
                               marker='.',
                               facecolors= clrs[pix],label=pname,s=self.POINT_SIZE)
            axs.annotate(pname, 
                 (embed_df[ix,0].mean(),embed_df[ix,1].mean()),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=self.text_S, weight='bold',
                 color="black")

            
    
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.spines['left'].set_visible(False)
        axs.spines['bottom'].set_visible(False)
    
        axs.get_xaxis().set_ticks([])
        axs.get_yaxis().set_ticks([])

        return axs
    
    @is_subfigure(label="A", plots=True)
    @label_me
    def _plot_A(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:


        embed_df = results[()]["embed_df"]
        clabels = results[()]["clabels"] 
        populations = results[()]["populations"] 
        # insert a new axis for title
        axis2 = axis.inset_axes([0,0,1,1])
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

        axis2 = self._tsne_plot(embed_df,axis2,clabels,populations) 
        axis2.set_title("BiPCA",loc="left")
        
        return axis

    @is_subfigure(label="B", plots=True)
    @label_me
    def _plot_B(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        


        embed_df = results[()]["embed_df"]
        clabels = results[()]["clabels"]   
        populations = results[()]["populations"]
        # insert a new axis for title
        axis2 = axis.inset_axes([0,0,1,1])
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

        axis2 = self._tsne_plot(embed_df,axis2,clabels,populations) 
        axis2.set_title("noBiPCA",loc="left")
        
        return axis