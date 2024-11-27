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
from scipy.linalg import norm
import torch
from torch.multiprocessing import Pool

from sklearn.preprocessing import scale as zscore
from sklearn.preprocessing import OneHotEncoder
from openTSNE import TSNE

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
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage,dendrogram
from bipca.experiments import rank_to_sigma
from bipca.experiments import knn_classifier, get_mean_var
from bipca.experiments.utils import uniques

from bipca.experiments import (compute_affine_coordinates_PCA,
                              compute_stiefel_coordinates_from_affine,
                              compute_stiefel_coordinates_from_data,
                              graph_L,
                              Lapalcian_score,
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
    RNA_color_index,
    atac_color_index,
    ST_color_index,
    RNA_fill_color,
    ST_fill_color,
    atac_fill_color,
    tech_label,
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
    data_write_path = "/banach2/jyc/data/BiPCA/"
) -> pd.DataFrame:
    """run_all Apply biPCA to all datasets and save the results to a csv file."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        if not overwrite:
            # extract the already written datasets / samples
            df = pd.read_csv(csv_path)
            written_datasets_samples = (
                (df["Dataset"] + "|" + df["Sample"]).str.split("|").values.tolist()
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
        d = dataset(logger=logger,base_data_directory=data_write_path)
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
        ["a", "a", "b", "b", "c", "c", "d", "d2", "d3"],
        ["e", "e", "e", "f", "f", "f", "g", "g", "g"],
        ["e", "e", "e", "f", "f", "f", "g", "g", "g"]
    ]

    def __init__(
        self,
        seed=42,
        mrows=5000,
        ncols=5000,
        minimum_singular_value=False,
        constant_singular_value=False,
        entrywise_mean=20, 
        libsize_mean=1000,
        ranks=2 ** np.arange(0, 7),
        bs=2.0 ** np.arange(-7, 7),
        cs=2.0 ** np.arange(-7, 7),
        n_iterations=10,
        figure_kwargs: dict = dict(dpi=300, figsize=(8.5, 4.25)),
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
        # params for the plots
        self.dash_linewd = 1
        self.dash_linealp = 0.6
        
        #kwargs['figure_kwargs'] = figure_kwargs
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
        figure_top = 0.85 #0
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
        new_positions["e"].x0 = left
        new_positions["e"].x1 = left + width
        new_positions["f"].x0 = left + width + pad
        new_positions["f"].x1 = left + 2 * width + pad
        #new_positions["H"].x0 = left + width + pad
        #new_positions["H"].x1 = left + 2 * width + pad
        new_positions["g"].x0 = left + 2 * width + 2 * pad
        new_positions["g"].x1 = right
        
        # adjust first row super columns
        # the super columns are [A,A,B,B,C,C] and [D, D2,D3]
        # we need [A - C] to key on E-(F/H), while [D,D1,D2] keys on G
        # we also want each [A-C] to be "square"ish, while [D,D2,D3] is a rectangle
        # start with [A-C]
        left = figure_left
        right = new_positions["g"].x1
        pad = sub_column_pad
        # the minimum y0 of [A-F] to get a reasonable whitespace between the rows is 0.75
        # therefore the maximum height of [A-F] is 0.88-0.75 = 0.13
        width = (right - left - 2 * pad) / 3
        height = 0.13
        square_dimension = np.minimum(width, height)
        # now we have the square dimension, we can compute x0 and x1 for A-C.
        new_positions["a"].x0 = left
        new_positions["a"].x1 = left + square_dimension
        new_positions["b"].x0 = left + square_dimension + pad
        new_positions["b"].x1 = left + 2 * square_dimension + pad
        new_positions["c"].x0 = left + 2 * square_dimension + 2 * pad
        new_positions["c"].x1 = left + 3 * square_dimension + 2 * pad
        # now we can compute the positions of [D,D2,D3]
        # the ticklabels on D take a lot of room, so we need to adjust the left
        left = new_positions["g"].x0  + 0.03# +0.07
        right = figure_right
        pad = 0.01  # this pads between the shared axes
        width = (right - left - 2 * pad) / 3
        # these can be rectangular, but have the same height as A-C
        new_positions["d"].x0 = left
        new_positions["d"].x1 = left + width
        new_positions["d2"].x0 = left + width + pad
        new_positions["d2"].x1 = left + 2 * width + pad
        new_positions["d3"].x0 = left + 2 * width + 2 * pad
        new_positions["d3"].x1 = right
        # finally, set the height of the first row
        for label in ["a", "b", "c"]:
            new_positions[label].y0 = 0.88 - square_dimension
            new_positions[label].y1 = 0.88
        for label in ["d", "d2", "d3"]:  # give a little extra room for the legend
            new_positions[label].y0 = 0.88 - square_dimension
            new_positions[label].y1 = 0.88

        # next, we need to adjust the height of the second row / super row.
        # it will be of height 2 * square_dimension + a row pad
        first_row_offset = figure_top - square_dimension - super_row_pad
        second_row_height = 2 * square_dimension
        pad = 0
        y1 = first_row_offset
        y0 = first_row_offset - second_row_height
        new_positions["e"].y0 = y0
        new_positions["e"].y1 = y1
        new_positions["g"].y0 = y0
        new_positions["g"].y1 = y1
        # [F and H] will split their height and have a pad
        #H_J_height = (second_row_height - sub_row_pad) / 2
        new_positions["f"].y1 = y1
        new_positions["f"].y0 = y0 #y1 - H_J_height
        
        
        

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

    @is_subfigure(label="a")
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

    @is_subfigure(label="a",plots=True)
    @label_me(2.4)
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
            rf"${{{2**i}}}$" if i % 2 == 0 else None
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
        axis.set_xlabel(r"true $r$", wrap=True)
        axis.set_ylabel(r"estimated $\hat{r}$", wrap=True)
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

    @is_subfigure(label="b")
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

    @is_subfigure(label="b", plots=True)
    @label_me(1.5)
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

    @is_subfigure(label="c")
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

    @is_subfigure(label="c", plots=True)
    @label_me(1.5)
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

    @is_subfigure(label=["d", "d2", "d3"])
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
        results = {"d": r, "d2": b, "d3": c}
        return results

    @is_subfigure(label="d", plots=True)
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

    @is_subfigure(label="d2", plots=True)
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
        axis.set_xlim([0.825, 1.925])
        axis.set_xticks([0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,1.7,1.8,1.9], minor=True)
        axis.sharey(self["d"].axis)
        axis.tick_params(axis="y", left=False, labelleft=False)
        return axis

    @is_subfigure(label="d3", plots=True)
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
        axis.sharey(self["d"].axis)
        axis.tick_params(axis="y", left=False, labelleft=False)
        return axis

    @is_subfigure(label=["e", "f", "g"])
    def _compute_E_F_G(self):
        df = run_all(
            csv_path=self.base_plot_directory / "results" / "dataset_parameters.csv",
            logger=self.logger,
        )
        df = df[df["Modality"] != "SingleNucleotidePolymorphism"]
        df = df[df["Modality"] != "GEXATAC_Multiome"]
        df.loc[df["Dataset"] == "SCORCH_PFC","Technology"] = "Multiome_rna" 
        df.loc[df["Dataset"] == "HagemannJensen2022","Technology"] = "SmartSeqV3xpress" 
        
        E = np.ndarray((len(df), 4), dtype=object)
        E[:, 0] = df.loc[:, "Modality"].values
        E[:, 1] = df.loc[:, "Linear coefficient (b)"].values
        E[:, 2] = df.loc[:, "Quadratic coefficient (c)"].values
        E[:, 3] = df.loc[:, "Technology"].values
        F = np.ndarray((len(df), 4), dtype=object)
        F[:, 0] = df.loc[:, "Modality"].values
        F[:, 1] = df.loc[:, "Filtered # observations"].values
        F[:, 2] = df.loc[:, "Rank"].values
        F[:, 3] = df.loc[:, "Technology"].values
        #G = np.ndarray((len(df), 3), dtype=object)
        #G[:, 0] = df.loc[:, "Modality"].values
        #G[:, 1] = df.loc[:, "Rank"].values / df.loc[
        #    :, ["Filtered # observations", "Filtered # features"]
        #].values.min(1)
        #G[:, 2] = df.loc[:, "Kolmogorov-Smirnov distance"].values
        G = np.ndarray((len(df), 4), dtype=object)
        G[:, 0] = df.loc[:, "Modality"].values
        G[:, 1] = df.loc[:, "Filtered # features"].values
        G[:, 2] = df.loc[:, "Rank"].values
        G[:, 3] = df.loc[:, "Technology"].values

     
        results = {"e": E, "f": F, "g": G}
        return results

    @is_subfigure(label="e", plots=True)
    @label_me
    def _plot_E(self, axis: mpl.axes.Axes,  results:np.ndarray) -> mpl.axes.Axes:
        # c plot w/ resampling
        df = pd.DataFrame(results, columns=["Modality", "b", "c","Technology"])
        # remove the atac fragments
        df = df[(df["Technology"] != "Multiome_ATAC_fragment")&(df["Technology"] != "10x_ATAC_fragment")]
        df = df[["Modality", "b", "c"]]
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
        ylim0 = -1e-3
        ylim1 = 30
        axis.axvline(
            x=1,ymin=ylim0,ymax=ylim1,
            alpha=self.dash_linealp,c='k',linestyle="--",linewidth=self.dash_linewd
        )
        axis.annotate(r"$\hat{b} = 1$",(1.1,ylim1-10))
        
        axis.set_yscale("symlog", linthresh=1e-2, linscale=0.5)
        pos_log_ticks = 10.0 ** np.arange(-2, 3)
        neg_log_ticks = 10.0 ** np.arange(
            -2,
            1,
        )
        yticks = np.hstack((-1 * neg_log_ticks, 0, pos_log_ticks))

        axis.set_yticks(
            yticks, labels=compute_latex_ticklabels(yticks, 10, skip=False,include_base=True)
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
        axis.set_ylim(ylim0,ylim1)

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

    @is_subfigure(label="f", plots=True)
    @label_me
    def _plot_F(self, axis: mpl.axes.Axes,  results:np.ndarray) -> mpl.axes.Axes:
        # rank vs number of observations
        df = pd.DataFrame(
           results,
            columns=["Modality", "# observations", "rank","Technology"],
        )
        # remove the atac fragments
        df = df[(df["Technology"] != "Multiome_ATAC_fragment")&(df["Technology"] != "10x_ATAC_fragment")]
        df = df[["Modality", "# observations", "rank"]]
        
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
        xlim0,xlim1 = axis.get_xlim()
        axis.axhline(
            y=50,xmin=xlim0,xmax=xlim1,
            alpha=self.dash_linealp,c='k',linestyle="--",linewidth=self.dash_linewd
        )
        axis.annotate("Seurat/Scanpy\ndefault: 50",(200,55))
        axis.set_yscale("log")
        axis.set_xscale("log")
        axis.set_xticks(
            [10**3, 10**4, 10**5],
            labels=compute_latex_ticklabels([10**3, 10**4, 10**5], 10,include_base = True,skip=False),
        )
        #axis.set_yticks(
        #    [10**1, 10**2],
        #    labels=compute_latex_ticklabels([10**1, 10**2], 10,  include_base = True,skip=False),
        #)
        axis.set_yticks(
            [10, 30, 50,100], labels = ['$10$', '$30$', '$50$', '$100$']
        )
        axis.set_xlabel(r"\# observations ")
        axis.set_ylabel(r"estimated $\hat{r}$")
        set_spine_visibility(axis, which=["top", "right"], status=False)

        return axis


    @is_subfigure(label="g", plots=True)
    @label_me
    def _plot_G(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        df = pd.DataFrame(
            results,
            columns=["Modality", "# features", "rank","Technology"],
        )
        # remove the atac fragments
        df = df[(df["Technology"] != "Multiome_ATAC_fragment")&(df["Technology"] != "10x_ATAC_fragment")]
        df = df[["Modality", "# features", "rank"]]
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
        xlim0,xlim1 = axis.get_xlim()
        axis.axhline(
            y=50,xmin=xlim0,xmax=xlim1,
            alpha=self.dash_linealp,c='k',linestyle="--",linewidth=self.dash_linewd
        )
        axis.annotate("Seurat/Scanpy\ndefault: 50",(600,55))
        axis.set_yscale("log")
        axis.set_xscale("log")
        axis.set_xticks(
            [10**3, 10**4, 10**5],
            labels=compute_latex_ticklabels([10**3, 10**4, 10**5], 10,include_base = True,skip=False),
        )
        axis.set_yticks(
            [10, 30, 50,100], labels = ['$10$', '$30$', '$50$', '$100$']
        )
        axis.set_xlabel(r"\# features")
        axis.set_ylabel(r"estimated $\hat{r}$")
        set_spine_visibility(axis, which=["top", "right"], status=False)

        return axis
    # 

class SuppFigure_qvf(Figure):
    _figure_layout = [
        ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        ["d","d","d","d1","d1","d1","d1","d1","d1"],
        ["d","d","d","d1","d1","d1","d1","d1","d1"]
    ]

    def __init__(
        self,
        seed=42,
        output_dir = "/banach2/jyc/data/BiPCA/",
        figure_kwargs: dict = dict(dpi=300, figsize=(8.5, 4.25)),
        *args,
        **kwargs,
    ):
        self.output_dir = output_dir
        self.seed = seed
        self.results = {}
        
        # params for the plots
        self.dash_linewd = 1
        self.dash_linealp = 0.6
        
        kwargs['figure_kwargs'] = figure_kwargs
        super().__init__(*args, **kwargs)


    def get10xATAC_frag_params(self):
        # a list of bipca data names
        # obtain the adata
        # run bipca and return bs and cs
        csv_path = Path(self.base_plot_directory / "results" / "atac_fragment_parameters.csv")
        if csv_path.exists():
            parameters = pd.read_csv(csv_path,index_col=0)
            
        else:
            tenXatac_dataset_names = np.array(["OpenChallengeMultiomeData_ATAC","TenX2019MouseBrainATAC",
                                               "TenX2019PBMCATAC","TenX2022MouseCortexATAC","TenX2022PBMCATAC"])
            tenXatac_datasets = np.array([bipca_datasets.OpenChallengeMultiomeData_ATAC(base_data_directory = self.output_dir),
                                          bipca_datasets.TenX2019MouseBrainATAC(base_data_directory = self.output_dir),
                                          bipca_datasets.TenX2019PBMCATAC(base_data_directory = self.output_dir),
                                          bipca_datasets.TenX2022MouseCortexATAC(base_data_directory = self.output_dir),
                                          bipca_datasets.TenX2022PBMCATAC(base_data_directory = self.output_dir)])
            #tenXatac_flags = np.array([True if dataset.__name__ in tenXatac_datasets else False for dataset in all_datasets])
            #tenXatac_datasets_class_list = all_datasets[tenXatac_flags]
    
            #adata_list_open = bipca.experiments.datasets.().get_filtered_data()
            #adata_list = {}
            parameters = []
            for i,dataset in enumerate(tenXatac_datasets):
                
                adatas = dataset.get_filtered_data(samples=dataset.samples)
    
                for sample in dataset.samples:
                    if issparse(adatas[sample].X):
                        X = np.ceil(adatas[sample].X.toarray()/2)
                    else:
                        X = np.ceil(adatas[sample].X/2)
                    op = BiPCA(n_components=-1,seed=self.seed,subsample_threshold=100,
                           n_subsamples=5, logger=dataset.logger).fit(X)
                    op.get_plotting_spectrum()
            
                
                
                    parameters.append(
                    {
                        "Dataset": tenXatac_dataset_names[i],
                        "Sample": sample,
                        "Modality": dataset.modality,
                        "Technology": dataset.technology + "_fragment",
                        "Unfiltered # observations": None,
                        "Unfiltered # features": None,
                        "Filtered # observations": None,
                        "Filtered # features": None,
                        "Kolmogorov-Smirnov distance": op.plotting_spectrum["kst"],
                        "Rank": op.mp_rank,
                        "Linear coefficient (b)": op.b,
                        "Quadratic coefficient (c)": op.c,
                    }
                    )
            parameters = pd.DataFrame(parameters)
            parameters.to_csv(csv_path)
        
        return parameters

        

    @is_subfigure(label=["a",  "b","c","d","d1"])
    def _compute_A_B_C_D(self):
        df = run_all(
            csv_path=self.base_plot_directory / "results" / "dataset_parameters.csv",
            logger=self.logger,
        )
        # TODO: add the code to generate b anc c for atac fragments
        #       then add to the table

        # TODO: change the tech of  HagemannJensen2022 to SmartSeqV3xpress in the code
        df.loc[df["Dataset"] == "HagemannJensen2022","Technology"] = "SmartSeqV3xpress" 

        df = df[df["Modality"] != "SingleNucleotidePolymorphism"]

        # add the atac fragment datasets
        df_atac_fragment = self.get10xATAC_frag_params()
        df_atac_fragment["Dataset-Sample"] = df_atac_fragment["Dataset"] + '-' + df_atac_fragment["Sample"]
        df_atac_fragment = df_atac_fragment[df.columns]
        df = pd.concat((df,df_atac_fragment))
        
        A = np.ndarray((len(df), 4), dtype=object)
        A[:, 0] = df.loc[:, "Modality"].values
        A[:, 1] = df.loc[:, "Linear coefficient (b)"].values
        A[:, 2] = df.loc[:, "Quadratic coefficient (c)"].values
        A[:, 3] = df.loc[:, "Technology"].values

        # 
        # extract the atac rank
        df.set_index("Dataset-Sample",inplace=True)
        df_atac = df.loc[(df["Modality"] == "SingleCellATACSeq")&(df["Dataset"] != "Buenrostro2018"),["Dataset","Technology","Rank"]]
        
        reads_rank = df_atac.loc[~df_atac["Technology"].str.contains("_fragment"),"Rank"]
        fragment_rank = df_atac.loc[df_atac["Technology"].str.contains("_fragment"),"Rank"]
        # reorder the index
        fragment_rank = fragment_rank.loc[reads_rank.index]

        
        
        results = {"a":A, "b":A, "c":A,
                   "d":np.concatenate((reads_rank.values.reshape(1,-1),fragment_rank.values.reshape(1,-1)),axis=0),"d1":[]}
        return results

    

    @is_subfigure(label="a", plots=True)
    @label_me
    def _plot_A(self, axis: mpl.axes.Axes,  results:np.ndarray) -> mpl.axes.Axes:
        # c plot w/ resampling
        df = pd.DataFrame(results, columns=["Modality", "b", "c","Technology"])

        # only keep singlecell rna seq
        df = df[df["Modality"] == "SingleCellRNASeq"]
        
        # shuffle the points
        idx = np.random.default_rng(self.seed).permutation(df.shape[0])
        df_shuffled = df.iloc[idx, :]
        x, y = df_shuffled.loc[:, ["b", "c"]].values.T
        
        c = df_shuffled["Technology"].map(RNA_fill_color).apply(pd.Series).values
        
        axis.scatter(
            x,
            y,
            s=10,
            c=c,
            marker="o",
            linewidth=0.1,
            edgecolor="k",
        )
        ylim0,ylim1 = 1e-1, 5
        #axis.axvline(
        #    x=1,ymin=ylim0,ymax=ylim1,
        #    alpha=self.dash_linealp,c='k',linestyle="--",linewidth=self.dash_linewd
        #)
        #axis.annotate(r"$\hat{b} = 1$",(1.1,ylim1-10))
        axis.set_yscale("symlog", linthresh=1e-2, linscale=0.5)
        pos_log_ticks = 10.0 ** np.arange(-2, 3)
        neg_log_ticks = 10.0 ** np.arange(
            -2,
            1,
        )
        yticks = np.hstack((-1 * neg_log_ticks, 0, pos_log_ticks))

        axis.set_yticks(
            yticks, labels=compute_latex_ticklabels(yticks, 10, skip=False,include_base=True)
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
        axis.set_ylim(ylim0,ylim1)

        axis.axvline(
            x=1,ymin=ylim0,ymax=ylim1,
            alpha=self.dash_linealp,c='k',linestyle="--",linewidth=self.dash_linewd
        )
        axis.annotate(r"$\hat{b} = 1$",(1.1,ylim1-0.3))
        
        axis.set_xlabel(r"estimated linear variance $\hat{b}$")
        axis.set_ylabel(r"estimated quadratic variance $\hat{c}$")
        set_spine_visibility(axis, which=["top", "right"], status=False)

        # build the legend handles and labels
        label2color = {
            tech_label[name]: color for name, color in RNA_fill_color.items()
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


    @is_subfigure(label="b", plots=True)
    @label_me
    def _plot_B(self, axis: mpl.axes.Axes,  results:np.ndarray) -> mpl.axes.Axes:
        # c plot w/ resampling
        df = pd.DataFrame(results, columns=["Modality", "b", "c","Technology"])

        # only keep singlecell rna seq
        df = df[df["Modality"] == "SpatialTranscriptomics"]
        
        # shuffle the points
        idx = np.random.default_rng(self.seed).permutation(df.shape[0])
        df_shuffled = df.iloc[idx, :]
        x, y = df_shuffled.loc[:, ["b", "c"]].values.T
        
        c = df_shuffled["Technology"].map(ST_fill_color).apply(pd.Series).values
        
        axis.scatter(
            x,
            y,
            s=10,
            c=c,
            marker="o",
            linewidth=0.1,
            edgecolor="k",
        )
        ylim0,ylim1 = 5e-3, 2
        #axis.axvline(
        #    x=1,ymin=ylim0,ymax=ylim1,
        #    alpha=self.dash_linealp,c='k',linestyle="--",linewidth=self.dash_linewd
        #)
        #axis.annotate(r"$\hat{b} = 1$",(1.1,ylim1-10))
        axis.axvline(
            x=1,ymin=ylim0,ymax=ylim1,
            alpha=self.dash_linealp,c='k',linestyle="--",linewidth=self.dash_linewd
        )
        axis.annotate(r"$\hat{b} = 1$",(1.1,ylim1-0.2))
        
        axis.set_yscale("symlog", linthresh=1e-2, linscale=0.5)
        pos_log_ticks = 10.0 ** np.arange(-2, 3)
        neg_log_ticks = 10.0 ** np.arange(
            -2,
            1,
        )
        yticks = np.hstack((-1 * neg_log_ticks, 0, pos_log_ticks))

        axis.set_yticks(
            yticks, labels=compute_latex_ticklabels(yticks, 10, skip=False,include_base=True)
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
        axis.set_ylim(ylim0,ylim1)

        axis.set_xlabel(r"estimated linear variance $\hat{b}$")
        axis.set_ylabel(r"estimated quadratic variance $\hat{c}$")
        set_spine_visibility(axis, which=["top", "right"], status=False)

        # build the legend handles and labels
        label2color = {
            tech_label[name]: color for name, color in ST_fill_color.items()
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

    @is_subfigure(label="c", plots=True)
    @label_me
    def _plot_C(self, axis: mpl.axes.Axes,  results:np.ndarray) -> mpl.axes.Axes:
        # c plot w/ resampling
        df = pd.DataFrame(results, columns=["Modality", "b", "c","Technology"])

        # only keep singlecell rna seq
        df = df[df["Modality"] == "SingleCellATACSeq"]
        
        # shuffle the points
        idx = np.random.default_rng(self.seed).permutation(df.shape[0])
        df_shuffled = df.iloc[idx, :]
        x, y = df_shuffled.loc[:, ["b", "c"]].values.T
        
        c = df_shuffled["Technology"].map(atac_fill_color).apply(pd.Series).values
        
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
        ylim0 = 1e-2
        ylim1 = 3
        axis.axvline(
            x=1,ymin=ylim0,ymax=ylim1,
            alpha=self.dash_linealp,c='k',linestyle="--",linewidth=self.dash_linewd
        )
        axis.annotate(r"$\hat{b} = 1$",(1.1,ylim1-0.3))
        
        axis.set_yticks(
            yticks, labels=compute_latex_ticklabels(yticks, 10, skip=False,include_base=True)
        )
        xticks = [0.9, 1.0, 1.5, 2.0]
        axis.set_xticks(
            xticks,
            labels=[0.9, 1, 1.5, 2],
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
        axis.set_ylim(ylim0, ylim1)

        axis.set_xlabel(r"estimated linear variance $\hat{b}$")
        axis.set_ylabel(r"estimated quadratic variance $\hat{c}$")
        set_spine_visibility(axis, which=["top", "right"], status=False)

        # build the legend handles and labels
        label2color = {
            tech_label[name]: color for name, color in atac_fill_color.items()
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
            loc="lower center",
            # bbox_to_anchor=(1.65, -0.5),
            # ncol=3,
            columnspacing=0,
            handletextpad=0,
            frameon=False,
        )
        return axis
        
    @is_subfigure(label=["d"], plots=True)
    @label_me
    def _plot_D(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        reads_rank,fragment_rank = self.results['d'][0,:],self.results['d'][1,:]
        max_rank = np.max(self.results['d'])
        axis.scatter(reads_rank,fragment_rank,s=5,c="black")
        axis.plot([0,max_rank],[0,max_rank],c='r',linestyle="--")

        axis.set_xlabel("Estimated rank on scATAC read counts")
        axis.set_ylabel("Estimated rank on scATAC fragment counts")
        return axis

    @is_subfigure(label=["d1"], plots=True)
    def _plot_D1(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        set_spine_visibility(axis,status=False)
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        return axis

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
    normalizations AFTER low rank approximation"""
    """The  panel will show mean-variance after low rank approximation"""
    """they are A: raw, B: BiPCA,
    C: log1p, D: log1p+z, F: Pearson
    G: Sanity, H: ALRA"""

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


class SupplementaryFigure3(Figure):
    _figure_layout = [
        ["A", "A", "B", "B","B","B"],
        ["C","C","D","D","E","E",],
        ['F','F','G','G','G2','G2']
        
    ]
    """This SupplementaryFigure shows the mean-variance relationships for all
    normalizations AFTER truncated SVD"""
    """they are A: raw, B: BiPCA,
    C: log1p, D: log1p+z, E: Pearson
    G: Sanity, H: ALRA"""

    def __init__(self,mean_cdf=False,
                      figure_kwargs: dict = dict(dpi=300, figsize=(8.5, 4.25)),
                      *args,**kwargs):
        self.mean_cdf = mean_cdf
        kwargs['figure_kwargs'] = figure_kwargs
        super().__init__(*args,**kwargs)

    @is_subfigure(label=["A", "B", "C", "D", "E","F","G","G2"])
    def _compute_A(self):
        dataset = bipca_datasets.TenX2016PBMC(
            store_filtered_data=True, logger=self.logger
        )
        adata = dataset.get_filtered_data(samples=["full"])["full"]
        path = dataset.filtered_data_paths["full.h5ad"]
        todo = ["log1p", "log1p+z", "ALRA", "Pearson","Sanity","BiPCA"]
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
            "Pearson",
            "Sanity",
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
                r = 50
            elif layer == 'ALRA':
                r = adata.uns['ALRA']['alra_k']
            else:
                r = adata.uns['bipca']['rank']
            U,S,V = SVD(backend='torch',n_components=r).fit_transform(Y)
            Y = (U*S)@V.T
            results[0, ix + 2] = f'rank {r} approximation of {layer}'
            _, results[1:, ix + 2] = get_mean_var(Y, axis=0)
        results = {
            "A": results[:, [0, 1, 2]],
            "B": results[:, [0, 1, 3]],
            "C": results[:, [0, 1, 4]],
            "D": results[:, [0, 1, 5]],
            "E": results[:, [0, 1, 6]],
            "F": results[:, [0, 1, 7]],
            "G": results[:, [0, 1, 8]],
            "G2": [],
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
    @is_subfigure(label="F", plots=True)
    @label_me
    def _plot_F(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        # Sanity  mean-var
        df = pd.DataFrame(results[1:, :], columns=["gene", "mean", "var"])
        df["var"] = df["var"].astype(float)
        df["mean"] = df["mean"].astype(float)
        axis = mean_var_plot(axis, df,mean_cdf=self.mean_cdf)

        axis.set_title(results[0, 2])
        yticks = np.arange(-3,1,)
        axis.set_yticks(10.0**yticks, labels=[fr"${t}$" if t%2==0 else None for t in yticks])


        return axis
    @is_subfigure(label="G", plots=True)
    @label_me
    def _plot_G(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        # Sanity  mean-var
        df = pd.DataFrame(results[1:, :], columns=["gene", "mean", "var"])
        df["var"] = df["var"].astype(float)
        df["mean"] = df["mean"].astype(float)
        axis = mean_var_plot(axis, df,mean_cdf=self.mean_cdf)

        axis.set_title(results[0, 2])
        yticks = np.arange(-3,1,)
        axis.set_yticks(10.0**yticks, labels=[fr"${t}$" if t%2==0 else None for t in yticks])


        return axis
    @is_subfigure(label="G2", plots=True)
    @label_me
    def _plot_G2(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        set_spine_visibility(axis,status=False)
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        return axis
        
class Figure3(Figure):
    """Marker genes figure"""
    _figure_layout = [
        ["a", "a", "a", "a2", "a2", "a2","a3", "a3", "a3", "a4", "a4", "a4", "a5", "a5", "a5", "a6", "a6", "a6"],
        # ["A", "A", "A", "A2", "A2", "A2","A3", "A3", "A3", "A4", "A4", "A4", "A5", "A5", "A5", "A6", "A6", "A6"],
        ["b","b2", "b3","b4","b5","b6","c","c2","c3","c4","c5","c6","d","d2","d3","d4","d5","d6"],
        ["e","e","e","e2","e2","e2","e3","e3","e3","e4","e4","e4","d","d2","d3","d4","d5","d6"],

        # ["E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E"],
    ]
    _ix_to_layer_mapping = {ix:key for ix,key in enumerate(algorithm_color_index.keys())}
    _subfig_to_celltype = {'b':['CD4+ T cells','CD8+ T cells'],'c':['CD56+ natural killer cells'],'d':['CD19+ B cells']}
    _celltype_ordering = ['CD8+ T cells','CD4+ T cells','CD56+ natural killer cells','CD19+ B cells']
    _default_marker_annotations_url = {"reference_HPA.tsv":(
                                    #"https://www.proteinatlas.org/search?format=tsv&download=yes"
                                    
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
                 output_dir = './',
                 sanity_installation_path = "/Sanity/bin/Sanity",
    npts_kde: int = 1000,
    group_size: int = 6000,
    niter: int = 10,
    seed: Number = 42,
    *args, **kwargs):

        self.output_dir = output_dir
        self.sanity_installation_path = sanity_installation_path
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
        new_positions['a'].x0 = figure_left
        new_positions['a'].x1 = new_positions['a'].x0 + A_space
        for i in range(2,7):
            cur_label = f"a{i}"
            last_label = f"a{i-1}" if i > 2 else "a"
            cur = new_positions[cur_label]
            last = new_positions[last_label]
            cur.x0 = last.x1 + sub_column_pad
            cur.x1 = cur.x0 + A_space
            new_positions[cur_label] = cur
        BCD_space = (figure_right - figure_left-2*super_column_pad)
        # compute the space for each of B, C, and D
        col_space = (BCD_space)/3
        new_positions['b'].x0 = figure_left
        # new_positions['B'].x1 = figure_left + col_space
        new_positions['c'].x0 = new_positions['b'].x0 + col_space + super_column_pad
        # new_positions['C'].x1 = new_positions['C'].x0 + col_space
        new_positions['d'].x0 = new_positions['c'].x0 + col_space + super_column_pad

        # new_positions['D'].x1 = new_positions['D'].x0 + col_space
        #this is the space occupied by a subcolumn
        sub_space = (col_space - 5*sub_column_pad)/6
        # compute the new positions for B,C,D
        cols = ["b","c","d"]
        for ix,label in enumerate(cols):
            # set x0 and x1 for each subcolumn of label
            new_positions[label].x1 = new_positions[label].x0 + sub_space
            new_positions[label].y1 = new_positions['a'].y0 - super_row_pad
            for i in range(2,7):
                cur_label = f"{label}{i}"
                last_label = f"{label}{i-1}" if i > 2 else label
                cur = new_positions[cur_label]
                last = new_positions[last_label]
                cur.x0 = last.x1 + sub_column_pad
                cur.x1 = cur.x0 + sub_space
                cur.y1 = last.y1
                new_positions[f"{label}{i}"] = cur
                
        
        EFGH_space = new_positions['d'].x0-super_column_pad - figure_left - 3*sub_column_pad
        col_space = (EFGH_space)/4
        new_positions['e'].x0 = figure_left
        new_positions['e'].x1 = new_positions['e'].x0 + col_space
        EFGH = ['e','e2','e3','e4']
        for ix in range(1,len(EFGH)):
            cur = EFGH[ix]
            last = EFGH[ix-1]
            new_positions[cur].x0 = new_positions[last].x0 + col_space + sub_column_pad
            new_positions[cur].x1 = new_positions[cur].x0 + col_space
        for label in ['d','d2','d3','d4','d5','d6','e','e2','e3','e4']:
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
        #paths_non_existant = list(filter(lambda path: not path.exists(), marker_annotations_path))
        #if len(paths_non_existant) > 0:
        #    raise FileNotFoundError(f"{paths_non_existant} does not exist")
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
    

    # JYC: hack marker_annotations_path to /banach1/jay/bistochastic_normalization/data/results/Figure3/reference_HPA.tsv 
   
    def _compute_marker_annotations(self) -> pd.DataFrame:
        # compute marker annotations
        self.acquire_marker_annotations()
        marker_annotations = self.extract_marker_annotations_to_df(**self._marker_annotations_to_df_kwargs,
                                                                   marker_annotations_path = [Path("/banach1/jay/bistochastic_normalization/data/results/Figure3/reference_HPA.tsv")])
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
        dataset = bipca_datasets.Zheng2017(store_filtered_data=True, logger=self.logger,base_data_directory = self.output_dir #jyc: need to specify data dir
        )
        if (adata:=getattr(self, 'adata', None)) is None:
            adata = dataset.get_filtered_data(samples=["markers"])["markers"]
        path = dataset.filtered_data_paths["markers.h5ad"]
        todo = ["log1p", "log1p+z", "Pearson", "Sanity", "ALRA", "BiPCA"]
        #bipca_kwargs = dict(n_components=-1,backend='torch', dense_svd=True,use_eig=True)
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        adata = apply_normalizations(adata, write_path = path,
                                    n_threads=64, apply=todo,
                                    #normalization_kwargs={'BiPCA':bipca_kwargs},
                                    sanity_installation_path=self.sanity_installation_path,
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
                    lab = 'a'
                else:
                    lab = f'a{ix+1}'
                results_out[lab] = np.vstack(y,dtype=object)
        return results_out
                    
    @is_subfigure(label=['a','a2','a3','a4','a5','a6'])
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
        # JYC: get rid of the marker names from the ct names
        ct_mapper = {'CD8+\nT cells':"CD8+\nT cells", 
                     'CD4+\nT cells':"CD4+\nT cells", 'CD56+\nNK cells':"NK cells",'CD19+\nB cells':"B cells"}
        ct = [ct_mapper[c] for c in ct]
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
        # JYC: change the coords
        y_tick_coord_adj = np.array([0.1,0.1,0.3,0.3])
        axis.set_yticks(yticks+y_tick_coord_adj, ct, minor=True,
                        fontsize=SMALL_SIZE,
                        verticalalignment='bottom',
                        horizontalalignment='right')
        for text in axis.get_yticklabels(minor=True):
            
            #text_str = text.get_text()
            #text_height = text.get_window_extent().height
            bb = text.get_tightbbox().transformed(axis.transData.inverted())
            #if (text_str == "B cells") | (text_str == "NK cells"):
            #    y0_coord = bb.y0+0.01+text_height
            #else: 
            #    y0_coord = bb.y0+0.01
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
    @is_subfigure(label='a',plots=True)
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
                label = r'$[\mathrm{AUROC}]$'))
        self.figure.legend(handles=handles,
            bbox_to_anchor=[0.85,axis.get_position().y0+0.002],fontsize=SMALL_SIZE,frameon=True,ncols=3,
            loc='upper center',handletextpad=0,columnspacing=0)
        self.figure.text(0.5,axis.get_position().y0-0.002,r'Scaled expression',fontsize=MEDIUM_SIZE,ha='center',va='top')
        return axis
    
    @is_subfigure(label='a2',plots=True)
    def _plot_A2(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        results = self._process_KDE_AUC_results(results)
        return self._plot_ridgeline(axis, results,sharey_label='a')
    
    @is_subfigure(label='a3',plots=True)
    def _plot_A3(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        results = self._process_KDE_AUC_results(results)
        return self._plot_ridgeline(axis, results,sharey_label='a')
    
    @is_subfigure(label='a4',plots=True)
    def _plot_A4(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        results = self._process_KDE_AUC_results(results)
        return self._plot_ridgeline(axis, results,sharey_label='a')
    
    @is_subfigure(label='a5',plots=True)
    def _plot_A5(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        results = self._process_KDE_AUC_results(results)
        return self._plot_ridgeline(axis, results,sharey_label='a')

    @is_subfigure(label='a6',plots=True)
    def _plot_A6(self, axis: mpl.axes.Axes, results: np.ndarray)-> mpl.axes.Axes:
        results = self._process_KDE_AUC_results(results)
        return self._plot_ridgeline(axis, results,sharey_label='a')


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
            label=['b','b2','b3','b4','b5','b6',
                    'c','c2','c3','c4','c5','c6',
                    'd','d2','d3','d4','d5','d6',
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
        for label, target_cluster in zip(['e','e2','e3','e4'],self._celltype_ordering):
            results_out[label] = np.vstack([resultsE[0], resultsE[(resultsE[:,2] == target_cluster) &(resultsE[:,3] == '+')]])
        return results_out
    
    @is_subfigure(label=['b','b2','b3','b4','b5','b6',
                        'c','c2','c3','c4','c5','c6',
                        'd','d2','d3','d4','d5','d6',
                        'e','e2','e3','e4'
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
        # JYC: rename the labels 
        xlabels_mapper = {"CD8+":"CD8+T","CD4+":"CD4+T","CD56+":"NK","CD19+":"B"}
        xlabels = np.array([xlabels_mapper[xla] for xla in xlabels])
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

        
        cbar.set_label(r'AUROC',labelpad=-13)
        cbar.set_ticks(ticks = [0,0.25,0.5,0.75,1.0])
        cbar.set_ticklabels([r'$0$',r'$.25$',r'$.5$',r'$.75$',r'$1$'],horizontalalignment='center')
        cbar.ax.tick_params(axis='x', direction='out',length=2.5,pad=1)
        return cbar
    @is_subfigure(label=['b'], plots=True)
    @label_me(4)
    def _plot_B(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        axis,im = self._process_heatmaps(axis, results,return_im=True)
        # add the colorbar
        cbar = self._add_AUC_colorbar([axis, *[self[f'b{i}'].axis for i in range(2,7)]])
        title = self._subfig_to_celltype['b']
        title = title.replace('natural killer','NK').replace('cells', 'cell') if title != ['CD4+ T cells','CD8+ T cells'] else 'T cell'
        title += ' markers'
        self.figure.text((self['b3'].axis.get_position().x0+self['b4'].axis.get_position().x1)/2,axis.get_position().y1+0.025,title,fontsize=MEDIUM_SIZE,ha='center',va='top')

        return axis

    @is_subfigure(label=['b2'], plots=True)
    def _plot_B2(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="b")

    @is_subfigure(label=['b3'], plots=True)
    def _plot_B3(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="b")

    @is_subfigure(label=['b4'], plots=True)
    def _plot_B4(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="b")

    @is_subfigure(label=['b5'], plots=True)
    def _plot_B5(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="b")

    @is_subfigure(label=['b6'], plots=True)
    def _plot_B6(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="b")

    @is_subfigure(label=['c'], plots=True)
    @label_me(4)
    def _plot_C(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        axis,im = self._process_heatmaps(axis, results,return_im=True)
        label='c'
        #colorbar
        cbar = self._add_AUC_colorbar([axis, *[self[f'c{i}'].axis for i in range(2,7)]])
        title = self._subfig_to_celltype[label]
        title = title[0] if len(title) == 1 else title
        title = title.replace('natural killer','NK').replace('cells', 'cell') if title != ['CD4+ T cells','CD8+ T cells'] else 'T cell'
        title += ' markers'
        title = title.replace('CD56+ ','').replace('CD19+ ','')
        self.figure.text((self[f'{label}3'].axis.get_position().x0+self[f'{label}4'].axis.get_position().x1)/2,axis.get_position().y1+0.025,title,fontsize=MEDIUM_SIZE,ha='center',va='top')
        return axis

    @is_subfigure(label=['c2'], plots=True)
    def _plot_C2(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="c")

    @is_subfigure(label=['c3'], plots=True)
    def _plot_C3(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="c")

    @is_subfigure(label=['c4'], plots=True)
    def _plot_C4(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="c")

    @is_subfigure(label=['c5'], plots=True)
    def _plot_C5(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="c")

    @is_subfigure(label=['c6'], plots=True)
    def _plot_C6(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="c")

    @is_subfigure(label=['d'], plots=True)
    @label_me(4)
    def _plot_D(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        label='d'
        axis,im = self._process_heatmaps(axis, results,return_im=True)
        # add the colorbar
        cbar = self._add_AUC_colorbar([axis, *[self[f'd{i}'].axis for i in range(2,7)]])
        title = self._subfig_to_celltype[label]
        title = title[0] if len(title) == 1 else title
        title = title.replace('natural killer','NK').replace('cells', 'cell') if title != ['CD4+ T cells','CD8+ T cells'] else 'T cell'
        title += ' markers'
        title = title.replace('CD56+ ','').replace('CD19+ ','')
        self.figure.text((self[f'{label}3'].axis.get_position().x0+self[f'{label}4'].axis.get_position().x1)/2,axis.get_position().y1+0.025,title,fontsize=MEDIUM_SIZE,ha='center',va='top')

        return axis

    @is_subfigure(label=['d2'], plots=True)
    def _plot_D2(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="d")

    @is_subfigure(label=['d3'], plots=True)
    def _plot_D3(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="d")

    @is_subfigure(label=['d4'], plots=True)
    def _plot_D4(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="d")

    @is_subfigure(label=['d5'], plots=True)
    def _plot_D5(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="d")

    @is_subfigure(label=['d6'], plots=True)
    def _plot_D6(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._process_heatmaps(axis, results, sharey_label="d")

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
        axis.set_xlabel('AUROC')
        axis.set_xticks([0,0.2,0.4,0.6,0.8,1.0],labels = [0,.2,.4,.6,.8,1])
        axis.set_xticks([0.1,0.3,0.5,0.7,0.9],minor=True)
        # JYC: rename the labels
        ct_mapper = {'CD8+ T cells':"CD8+ T cells", 
                     'CD4+ T cells':"CD4+ T cells", 'CD56+ NK cells':"NK cells",'CD19+ B cells':"B cells"}
        celltype = ct_mapper[celltype]
        axis.set_title(celltype)
        return axis
    @is_subfigure(label='e', plots=True)
    @label_me
    def _plot_E(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._plot_EFGH(axis, results, sharey_label=None)
    @is_subfigure(label='e2', plots=True)
    def _plot_E2(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._plot_EFGH(axis, results, sharey_label='e')
    @is_subfigure(label='e3', plots=True)

    def _plot_E3(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._plot_EFGH(axis, results, sharey_label='e')
    @is_subfigure(label='e4', plots=True)
    def _plot_E4(self, axis: mpl.axes.Axes, results: np.ndarray) -> mpl.axes.Axes:
        return self._plot_EFGH(axis, results, sharey_label='e')


class Figure4(Figure):
    _figure_layout = [
        ["A", "A", "A", "A", "A", "A"],
        ["A", "A", "A", "A", "A", "A"],
        ["B", "B", "B", "B", "B", "B"],
        ["B", "B", "B", "B", "B", "B"],
        ["C", "C", "D", "D", "D2", "D2"],
        ["C","C", "D3","D3","D4", "D4"]
    ]

    def __init__(
        self,
        output_dir = "./",
        sanity_installation_path = "/Sanity/bin/Sanity",
        load_normalized = False,
        normalized_data_path = None,
        seed = 42,
        TSNE_POINT_SIZE_A=0.4,
        TSNE_POINT_SIZE_B =0.8,
        LINE_WIDTH=0.2,
        *args,
        **kwargs,
    ):
        self.output_dir = output_dir
        self.seed = seed
        self.load_normalized = load_normalized
        self.normalized_data_path = normalized_data_path
        self.n_iterations = 10 # number of iterations to compute Silhouette
        self.k_list = [5,10,15]
        self.TSNE_POINT_SIZE_A = TSNE_POINT_SIZE_A
        self.TSNE_POINT_SIZE_B = TSNE_POINT_SIZE_B
        self.LINE_WIDTH = LINE_WIDTH
        self.sanity_installation_path = sanity_installation_path
        self.method_keys = ['log1p', 'log1p+z', 'Pearson', 'Sanity', 'ALRA','Z_biwhite']
        self.method_names = ['log1p', 'log1p+z', 'Pearson', 'Sanity', 'ALRA', 'BiPCA']
        self.open_cite_sample_ids = bipca_datasets.OpenChallengeCITEseqData()._sample_ids
        self.open_multi_sample_ids = bipca_datasets.OpenChallengeMultiomeData()._sample_ids
        
        self.results = {}
        super().__init__(*args, **kwargs)

    def loadClassificationData(self,dataset_name,
                                    sanity_installation_path="/Sanity/bin/Sanity",cells2remove=None,cells2remove_meta=None):
        Path(self.output_dir+"/classification/").mkdir(parents=True, exist_ok=True)
        Path(self.output_dir+"/classification/"+dataset_name).mkdir(parents=True, exist_ok=True)
        adata = getattr(bipca_datasets,dataset_name)(base_data_directory = self.output_dir+"/classification/").get_filtered_data()['full']
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        if (cells2remove is not None) &  (cells2remove_meta is not None):
            adata = adata[[ cp not in cells2remove for cp in adata.obs[cells2remove_meta]],:].copy()
            # filter for sparse genes
            genes2keep = np.sum(adata.X > 0,axis=0) > 100
            adata = adata[:,genes2keep]
        Path(self.output_dir+"/classification_norm/").mkdir(parents=True, exist_ok=True)
        adata = apply_normalizations(adata,write_path=self.output_dir+"/classification_norm/{}.h5ad".format(dataset_name),
                                                                normalization_kwargs={"log1p":{},
                            "log1p+z":{},
                            "Pearson":dict(clip=np.inf),
                            "ALRA":{"seed":42},
                            "Sanity":{}, 
                            "BiPCA":dict(n_components=-1, backend="torch",seed=42)},
                                     sanity_installation_path=sanity_installation_path)
        return adata
        
    def loadClassificationDataAll(self,
                          sanity_installation_path="/Sanity/bin/Sanity"):
        
        
        if Path(self.output_dir+"/classification_norm/Zheng2017.h5ad").exists():
            adata_zh2017 = sc.read_h5ad(self.output_dir+"/classification_norm/Zheng2017.h5ad")
        else:
            adata_zh2017 = self.loadClassificationData(dataset_name="Zheng2017",sanity_installation_path=sanity_installation_path,
                                                       cells2remove=["CD4+ T cells","CD4+CD45RA+CD25- naive T cells",
                                                                     "CD4+CD25+ regulatory T cells","CD4+CD45RO+ memory T cells",
                                                                     "CD8+ cytotoxic T cells"],cells2remove_meta="cluster")
        if Path(self.output_dir+"/classification_norm/Stoeckius2017.h5ad").exists():
            adata_st2017 = sc.read_h5ad(self.output_dir+"/classification_norm/Stoeckius2017.h5ad")
        else:
            adata_st2017 =  self.loadClassificationData(dataset_name="Stoeckius2017",sanity_installation_path=sanity_installation_path)
        if Path(self.output_dir+"/classification_norm/Stuart2019.h5ad").exists():
            adata_st2019 = sc.read_h5ad(self.output_dir+"/classification_norm/Stuart2019.h5ad")
        else:
            adata_st2019 =  self.loadClassificationData(dataset_name="Stuart2019",sanity_installation_path=sanity_installation_path)              
            
            
        return adata_zh2017,adata_st2017,adata_st2019
    
         
    def loadOpenChallengeData(self,
                          load_normalized=False,
                          normalized_data_path = None,
                          overwrite=False,
                          sanity_installation_path="/Sanity/bin/Sanity"):
        if load_normalized:
            adata_cite_list = {sid:sc.read_h5ad(normalized_data_path + "/citeseq_" +sid + ".h5ad") for sid in bipca_datasets.OpenChallengeCITEseqData()._sample_ids}
            adata_multiome_list = {sid:sc.read_h5ad(normalized_data_path + "/atac_" +sid + ".h5ad") for sid in bipca_datasets.OpenChallengeMultiomeData()._sample_ids}
        else:
            if not Path(self.output_dir+"/normalized/").exists() or overwrite:
                Path(self.output_dir+"/citeseq/").mkdir(parents=True, exist_ok=True)
                adata_cite_list = bipca_datasets.OpenChallengeCITEseqData(base_data_directory = self.output_dir + "/citeseq/").get_filtered_data()
                Path(self.output_dir+"/multiome/").mkdir(parents=True, exist_ok=True)
                adata_multiome_list = bipca_datasets.OpenChallengeMultiomeData(base_data_directory = self.output_dir + "/multiome/").get_filtered_data()

                Path(self.output_dir+"/normalized/").mkdir(parents=True, exist_ok=True)
                for sid,adata in adata_cite_list.items():
                    adata_cite_list[sid] = apply_normalizations(adata,write_path=self.output_dir+"/normalized/citeseq_"+sid+".h5ad",
                                                                normalization_kwargs={"log1p":{},
                            "log1p+z":{},
                            "Pearson":dict(clip=np.inf),
                            "ALRA":{"seed":42},
                            "Sanity":{}, 
                            "BiPCA":dict(n_components=-1, backend="torch", seed=42)},
                                     sanity_installation_path=sanity_installation_path)
                for sid,adata in adata_multiome_list.items():
                    adata_multiome_list[sid] = apply_normalizations(adata,write_path=self.output_dir+"/normalized/atac_"+sid+".h5ad",
                                                                    normalization_kwargs={"log1p":{},
                            "log1p+z":{},
                            "Pearson":dict(clip=np.inf),
                            "ALRA":{"seed":42},
                            "Sanity":{}, 
                            "BiPCA":dict(n_components=-1, backend="torch", seed=42)},
                                         sanity_installation_path=sanity_installation_path)
            else:
                adata_cite_list = {sid:sc.read_h5ad(self.output_dir+"/normalized/citeseq_"+sid+".h5ad") for sid in bipca_datasets.OpenChallengeCITEseqData()._sample_ids}
                adata_multiome_list = {sid:sc.read_h5ad(self.output_dir+"/normalized/atac_"+sid+".h5ad") for sid in bipca_datasets.OpenChallengeMultiomeData()._sample_ids}
    
        self.adata_cite_list = adata_cite_list
        self.adata_multiome_list = adata_multiome_list
        return adata_cite_list,adata_multiome_list

    def loadPFCdata(self,
                       adata_path = "/banach1/jyc/bipca/biPCA_copy_Dec8_2023/biPCA/results/um1_data/small_PCs_experiment/bipca_normalized_adata/HCTXJ_CTR_PFC_X.h5ad"):
        if Path(self.output_dir+"/smallPCs/bipca.h5ad").exists():
            adata = sc.read_h5ad(self.output_dir+"/smallPCs/bipca.h5ad")
        else:
            adata = sc.read_h5ad(adata_path)
            torch.set_num_threads(36)
            with threadpool_limits(limits=36):
                op = bipca.BiPCA(n_components=-1,seed=42)
                PCs = op.fit_transform(adata.X,counts=False)
            PCs =  np.asarray(PCs)
            adata.obsm['biPCs'] = PCs
            for i in range(PCs.shape[1]):
                adata.obs['biPCs_'+str(i+1)] = PCs[:,i].reshape(-1)  
            Path(self.output_dir+"/smallPCs/").mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(self.output_dir+"/smallPCs/bipca.h5ad")
        return adata
                           
    def getPCs(self,adata_list,computed_data_path):
        
        if Path(computed_data_path).exists():
            PCs = np.load(computed_data_path,allow_pickle=True)[()]
        else:
            PCs = OrderedDict()
            for sid,adata in adata_list.items():
                PCs[sid] = OrderedDict()
                for method in self.method_keys:
                    if issparse(adata.layers[method]):
                        adata.layers[method] = adata.layers[method].toarray()
                    if method == 'ALRA':
                        PCs[sid][method] = new_svd(adata.layers[method],r=adata.uns['ALRA']["alra_k"])
                    elif method == "Z_biwhite":
                        PCs[sid][method] = new_svd(adata.layers[method],r=adata.uns['bipca']["rank"])
                    else:
                        PCs[sid][method] = new_svd(adata.layers[method],r=50) # 50
            np.save(computed_data_path,PCs)
        
        return PCs
        
    def getYlabels(self,adata_list,label_token_list):
        ylabel_dict = {}
        label_conversion_dict = {}
        for sid in adata_list.keys():
            annotations_all = np.unique(adata_list[sid].obs[label_token_list[sid]])
            label_convertor = {k:i for i,k in enumerate(annotations_all)}
            ylabel_dict[sid] = np.array([label_convertor[label] for label in adata_list[sid].obs[label_token_list[sid]]])
            label_conversion_dict[sid] = label_convertor
        return ylabel_dict,label_conversion_dict
        
    def computeKNNaccuracy(self,PCs,ylabel_dict,k_list):
        knn_acc_mat = np.zeros((len(PCs.keys()),6,len(k_list)))
        for six,sid in enumerate(PCs.keys()):
            for mix,method in enumerate(self.method_keys):
                for k_idx,k in enumerate(k_list):
                    #rng = np.random.default_rng(round)
                    #sample_idx = rng.choice(PCs[sid][method].shape[0],int(PCs[sid][method].shape[0]*ratio),replace=False)
                    X = PCs[sid][method]
                    y = ylabel_dict[sid]
                    #neigh = KNeighborsClassifier(n_neighbors=k,n_jobs=64)
                    #neigh.fit(X,y)
                    #y_pred = neigh.predict(X)
                    
                    neigh = NearestNeighbors(n_neighbors=k,n_jobs=16).fit(X)
                    knn = neigh.kneighbors(return_distance=False)
                    y_pred = mode(y[knn], axis=1).mode.flatten()
                    
                    
                    knn_acc_mat[six,mix,k_idx] = balanced_accuracy_score(y,y_pred)
        return knn_acc_mat
        
    def computeSilhouette(self,PCs,ylabel_dict):
        sil_score_mat = np.zeros((len(PCs.keys()),6,self.n_iterations))
        for six,sid in enumerate(PCs.keys()):
            for mix,method in enumerate(self.method_keys):
        
                for round in range(self.n_iterations):
                    score = silhouette_score(PCs[sid][method],ylabel_dict[sid],
                                     sample_size=int(PCs[sid][method].shape[0] * 0.8),
                                     random_state=round)
                    sil_score_mat[six,mix,round] = score
        return sil_score_mat

    @is_subfigure(label=["A"])
    def _compute_A(self):
        adata_zh2017,adata_st2017,adata_st2019 = self.loadClassificationDataAll(sanity_installation_path = self.sanity_installation_path)
        adata_list = {"Zheng2017":adata_zh2017,"Stoeckius2017":adata_st2017,"Stuart2019":adata_st2019}
        #_,adata_list = self.loadOpenChallengeData(load_normalized=self.load_normalized,
        #                                                                 normalized_data_path = self.normalized_data_path,
        #                                                                 sanity_installation_path = self.sanity_installation_path)
        PCs_list = self.getPCs(adata_list,
                                    computed_data_path=self.output_dir+"/classification_norm/PCs_classification.npy")
        y_list,_  = self.getYlabels(adata_list,
                                 label_token_list={"Stoeckius2017":'protein_annotations',
                 "Stuart2019":'cell_types',
                "Zheng2017":"cluster"})
        #self.y_list = y_list
        #y_list,_ = self.getYlabels(adata_list,label_token_list={sid:"cell_type" for sid in PCs_list.keys()})
        knn_acc_mat = self.computeKNNaccuracy(PCs_list,y_list,k_list=self.k_list)
        results = {"A":knn_acc_mat}
        return results
        
    @is_subfigure(label=["B"])
    def _compute_B(self):
        adata_cite_list,adata_multiome_list = self.loadOpenChallengeData(load_normalized=self.load_normalized,
                                                                         normalized_data_path = self.normalized_data_path,
                                                                         sanity_installation_path = self.sanity_installation_path)
        PCs_cite_list = self.getPCs(adata_cite_list,
                                    computed_data_path=self.output_dir+"/normalized/OpChallengePCs_cite.npy")
        
        #PCs_multiome_list = self.getPCs(adata_multiome_list,
        #                               computed_data_path=self.output_dir+"/normalized/OpChallengePCs_multi.npy")
        y_cite_list,_ = self.getYlabels(adata_cite_list,label_token_list={sid:"cell_type" for sid in PCs_cite_list.keys()})
        #y_multiome_list,_ = self.getYlabels(adata_multiome_list,label_token_list={sid:"cell_type" for sid in PCs_cite_list.keys()})

        silhouette_cite = self.computeSilhouette(PCs_cite_list,y_cite_list)
        #silhouette_multiome = self.computeSilhouette(PCs_multiome_list,y_multiome_list)

        results = {"B":silhouette_cite}
                  #"B":silhouette_multiome}
        return results
    @is_subfigure(label=['C'])
    def _compute_C(self):
        
        
        results = {"C":[]}
        return results

    @is_subfigure(label=['D', 'D2', 'D3', 'D4'])
    def _compute_D(self):
        adata = self.loadPFCdata()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.tl.rank_genes_groups(adata, "cluster_27", method="wilcoxon",groups=['27'],reference="OPC",key_added = "wilcoxon")
        de_df = sc.get.rank_genes_groups_df(adata, group= None,log2fc_min=1,pval_cutoff=0.01,key="wilcoxon")

        
        results = {"D":[de_df['names'][0]],'D2':[de_df['names'][1]], 
                   'D3':[de_df['names'][2]], 'D4':[de_df['names'][3]]}
        return results
        
    @is_subfigure(label=["A"], plots=True)
    @label_me(0.2)
    def _plot_A(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        result1 = self.results["A"][1:3,:,:].transpose(0, 2, 1)

        result1_recast_df = pd.DataFrame(np.zeros((2  * result1.shape[1] * result1.shape[2],4)),columns=["sid","k","methods","accuracy"])
        dataset_names = ["Stoeckius2017","Stuart2019"]
        for d_id in range(result1.shape[0]):
            for i_id in range(result1.shape[1]):
                for m_id in range(result1.shape[2]):
        
                    row_idx = d_id*(result1.shape[1]*result1.shape[2])+i_id*result1.shape[2]+m_id
                    result1_recast_df.loc[row_idx,"sid"] = dataset_names[d_id]
                    result1_recast_df.loc[row_idx,"methods"] = self.method_names[m_id]
                    result1_recast_df.loc[row_idx,"k"] = str(self.k_list[i_id])
                    result1_recast_df.loc[row_idx,"accuracy"] = result1[d_id,i_id,m_id]
        result1_recast_df["modality"] = "CITEseq"           
        result1_recast_df['sid_k_methods'] = np.array([result1_recast_df.iloc[rix,:]["sid"] +
                                               '_x_' + result1_recast_df.iloc[rix,:]["k"]+ 
                                               '_x_' + result1_recast_df.iloc[rix,:]["methods"]
                                                   for rix in range(result1_recast_df.shape[0])])
           
        algorithm_fill_color_palette_keys = []
        algorithm_fill_color_palette_values = []
        for d_id in range(result1.shape[0]):
            for i_id in range(result1.shape[1]):
                for m_id in range(result1.shape[2]):

                    rix = d_id*(result1.shape[1]*result1.shape[2])+i_id*result1.shape[2]+m_id   
                    algorithm_fill_color_palette_keys.append(result1_recast_df.iloc[rix,:]['sid_k_methods'])
                    algorithm_fill_color_palette_values.append(algorithm_fill_color[result1_recast_df.iloc[rix,:]['methods']])
                algorithm_fill_color_palette_keys.append("dummy1_"+str(rix))
                algorithm_fill_color_palette_values.append('k')
                
                
            algorithm_fill_color_palette_keys.append("dummy1_"+str(rix))
            algorithm_fill_color_palette_keys.append("dummy2_"+str(rix))
            algorithm_fill_color_palette_keys.append("dummy3_"+str(rix))
            algorithm_fill_color_palette_keys.append("dummy4_"+str(rix))
            algorithm_fill_color_palette_values.append('k')
            algorithm_fill_color_palette_values.append('k')
            algorithm_fill_color_palette_values.append('k')
            algorithm_fill_color_palette_values.append('k')

        algorithm_fill_color_palette = dict(zip(algorithm_fill_color_palette_keys,algorithm_fill_color_palette_values))
        self.algorithm_fill_color_palette = algorithm_fill_color_palette
        axis = sns.pointplot(result1_recast_df,errorbar="sd",
                    x='sid_k_methods',
                    hue='sid_k_methods',
                    palette = algorithm_fill_color_palette,
                    order=list(algorithm_fill_color_palette.keys()),
                    errwidth=0.5,
                    y="accuracy",
                    scale=0.5,
                    dodge=False,ax=axis)
        # add legend
        handles = [
            mpl.lines.Line2D(
            [],
            [],
            marker='.',
            color=color,
            linewidth=0,
            label=method,
            markersize=10,
            )
            for method, color in algorithm_fill_color.items()
        ]

        legend = axis.legend(
            handles=handles,
            ncol=1,fontsize=8,bbox_to_anchor=[0.95, 0.8], 
            loc='center',handletextpad=0,columnspacing=0
        )
        
        n_ticks = int(len(self.k_list)*2)
        tick_positions=[2.5,9.5,16.5,26.5,33.5,40.5]
        axis.set_xticks(tick_positions,np.hstack([["Stoeckius2017 (k=" + str(k) + ')' for k in self.k_list],
                                                  ["Stuart2019 (k=" + str(k) + ')' for k in self.k_list]]))
        
        axis.tick_params(bottom = False) 
        axis.set_xlabel('')
        
        axis.set_title('')  
        axis.set_yticks([0.8,0.9,1])
        axis.set_xlim(-0.5)
        axis.set_ylim(0.7,top=1)
        for tick_pos in tick_positions:
            axis.hlines(0.7,tick_pos-2,tick_pos+2,colors='k',lw=0.5)
        set_spine_visibility(axis,which=["top", "right", "bottom"],status=False)
        plt.setp(axis.lines, zorder=100,clip_on=False)
        plt.setp(axis.collections, zorder=100, label="",clip_on=False)
        sns.despine(trim=True,bottom=True)
        return axis 
        
    @is_subfigure(label=["B"], plots=True)
    @label_me(0.2)
    def _plot_B(self, axis: mpl.axes.Axes ,results:np.ndarray) -> mpl.axes.Axes:
        result1 = self.results["B"]
        result1_recast_df = pd.DataFrame(np.zeros((result1.shape[0] * result1.shape[1] * result1.shape[2],4)),columns=["sid","methods","iter","Silhouette scores"])
        for d_id in range(result1.shape[0]):
            for m_id in range(result1.shape[1]):
                for i_id in range(result1.shape[2]):
                    row_idx = d_id*(result1.shape[1]*result1.shape[2])+m_id*result1.shape[2]+i_id
                    result1_recast_df.loc[row_idx,"sid"] = self.open_cite_sample_ids[d_id]
                    result1_recast_df.loc[row_idx,"methods"] = self.method_names[m_id]
                    result1_recast_df.loc[row_idx,"iter"] = int(i_id)
                    result1_recast_df.loc[row_idx,"Silhouette scores"] = result1[d_id,m_id,i_id]
        result1_recast_df["modality"] = "CITEseq"           
        result1_recast_df['sid_x_methods'] = np.array([result1_recast_df.iloc[rix,:]["sid"] + '_x_' + result1_recast_df.iloc[rix,:]["methods"]
                                                   for rix in range(result1_recast_df.shape[0])])
        self.result1_recast_df = result1_recast_df
        
        result1_recast_df_short = result1_recast_df[["methods","sid_x_methods"]].copy()
        result1_recast_df_short.drop_duplicates(inplace=True)
        algorithm_fill_color_palette_keys = []
        algorithm_fill_color_palette_values = []
        for rix,dataset_method in enumerate(result1_recast_df_short['sid_x_methods']):
            if (rix != 0) & ((rix % 6) ==0):
                algorithm_fill_color_palette_keys.append("dummy1_"+str(rix))
                algorithm_fill_color_palette_keys.append("dummy2_"+str(rix))
        
                algorithm_fill_color_palette_values.append('k')
                algorithm_fill_color_palette_values.append('k')
        
                algorithm_fill_color_palette_keys.append(result1_recast_df_short.iloc[rix,:]['sid_x_methods'])
                algorithm_fill_color_palette_values.append(algorithm_fill_color[result1_recast_df_short.iloc[rix,:]['methods']])
            else:
                algorithm_fill_color_palette_keys.append(result1_recast_df_short.iloc[rix,:]['sid_x_methods'])
                algorithm_fill_color_palette_values.append(algorithm_fill_color[result1_recast_df_short.iloc[rix,:]['methods']])


        algorithm_fill_color_palette = dict(zip(algorithm_fill_color_palette_keys,algorithm_fill_color_palette_values))


        
        #g = sns.FacetGrid(result1_recast_df, row="modality",sharey=False,sharex=False)
        axis = sns.pointplot(result1_recast_df,errorbar="sd",
                    x='sid_x_methods',
                    hue='sid_x_methods',
                    palette = algorithm_fill_color_palette,
                    order=list(algorithm_fill_color_palette.keys()),
                    errwidth=0.5,
                    y="Silhouette scores",
                    scale=0.5,
                    dodge=False,ax=axis)
        # add legend
        handles = [
            mpl.lines.Line2D(
            [],
            [],
            marker='.',
            color=color,
            linewidth=0,
            label=method,
            markersize=10,
            )
            for method, color in algorithm_fill_color.items()
        ]

        legend = axis.legend(
            handles=handles,
            ncol=1,fontsize=8,bbox_to_anchor=[0.95, 0.8], 
            loc='center',handletextpad=0,columnspacing=0
        )
        
        n_ticks = len(self.open_cite_sample_ids)
        tick_positions=[2.5 + i*8 for i in range(n_ticks)]
        axis.set_xticks(tick_positions, ["batch"+str(i+1) for i in range(n_ticks)])
        
        axis.tick_params(bottom = False,pad=0.5) 
        axis.set_xlabel('')
        
        axis.set_title('')  
        axis.set_yticks([0,0.2,0.4])
        axis.set_xlim(-0.5)
        axis.set_ylim(-0.02,top=0.4)
        for tick_pos in tick_positions:
            axis.hlines(-0.02,tick_pos-2,tick_pos+2,colors='k',lw=0.5)
        set_spine_visibility(axis,which=["top", "right", "bottom"],status=False)
        plt.setp(axis.lines, zorder=100,clip_on=False)
        plt.setp(axis.collections, zorder=100, label="",clip_on=False)
        sns.despine(trim=True,bottom=True)
        
        return axis
 
    @is_subfigure(label=["C"], plots=True)
    @label_me
    def _plot_C(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'pfc_data'):
            self.pfc_data = self.loadPFCdata()
        cluster_labels = self.pfc_data[self.pfc_data.obs['cell_types_predicted_cluster_consensus'] == 'OPC',:].obs["leidenIs27"]
        lut = dict(zip(['True','False'], fill_cmap.colors[:2,:])) 
        row_colors = np.array([lut[clabel] for clabel in cluster_labels])
        row_orders = np.argsort(cluster_labels)

        axis = sns.heatmap(self.pfc_data[self.pfc_data.obs['cell_types_predicted_cluster_consensus'] == 'OPC',:].obs[["biPCs_"+str(i+1) for i in range(50,72)]].iloc[row_orders,:],
                          ax=axis,cmap=heatmap_cmap)
        axis.tick_params(axis='y', which='major', pad=6, length=0,
                         labelright=False,right=False,
                         labeltop=False,top=False,
                        labelbottom=False,bottom=False)
        axis.get_xaxis().set_ticks([])
        axis.set_yticks([750,350],["OPC cluster","Others"])
        axis.set_xlabel("PC 51 - 72",labelpad=5)
        for i, color in enumerate(row_colors[row_orders]):
            axis.add_patch(plt.Rectangle(xy=(-0.05, i), width=0.05, height=1, color=color, lw=0,
                               transform=axis.get_yaxis_transform(), clip_on=False))

        
        return axis  
    
    @is_subfigure(label=["D"], plots=True)
    @label_me
    def _plot_D(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'pfc_data'):
            self.pfc_data = self.loadPFCdata()
        
        de_name = self.results["D"]
        OPC_type = {"27":"OPC cluster","OPC":"Others","others":"OtherCP"}
        self.pfc_data.obs['OPC_annotation'] = self.pfc_data.obs["cluster_27"].map(OPC_type)
        axis = sc.pl.violin(
            self.pfc_data[self.pfc_data.obs['cell_types_predicted_cluster_consensus'] == 'OPC',:],
            de_name,use_raw=False,
            groupby="OPC_annotation",
            stripplot=False, 
            palette={"OPC cluster":fill_cmap.colors[0,:],
                    "Others":fill_cmap.colors[1,:]},
            xlabel=' ',
            inner="box", 
            ax = axis
            )
        return axis  

    @is_subfigure(label=["D2"], plots=True)
    def _plot_D2(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'pfc_data'):
            self.pfc_data = self.loadPFCdata()
        de_name = self.results["D2"]
        OPC_type = {"27":"OPC cluster","OPC":"Others","others":"OtherCP"}
        self.pfc_data.obs['OPC_annotation'] = self.pfc_data.obs["cluster_27"].map(OPC_type)
        
        axis = sc.pl.violin(
            self.pfc_data[self.pfc_data.obs['cell_types_predicted_cluster_consensus'] == 'OPC',:],
            de_name,use_raw=False,
            groupby="OPC_annotation",
            stripplot=False, 
            xlabel=' ',
            palette={"OPC cluster":fill_cmap.colors[0,:],
                    "Others":fill_cmap.colors[1,:]},
            inner="box", 
            ax = axis
            )
        
        return axis 
        
    @is_subfigure(label=["D3"], plots=True)
    def _plot_D3(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'pfc_data'):
            self.pfc_data = self.loadPFCdata()
        de_name = self.results["D3"]
        OPC_type = {"27":"OPC cluster","OPC":"Others","others":"OtherCP"}
        self.pfc_data.obs['OPC_annotation'] = self.pfc_data.obs["cluster_27"].map(OPC_type)
        
        axis = sc.pl.violin(
            self.pfc_data[self.pfc_data.obs['cell_types_predicted_cluster_consensus'] == 'OPC',:],
            de_name,use_raw=False,
            groupby="OPC_annotation",
            stripplot=False, 
            xlabel=' ',
            palette={"OPC cluster":fill_cmap.colors[0,:],
                    "Others":fill_cmap.colors[1,:]},
            inner="box", 
            ax = axis
            )
        
        return axis  
        
    @is_subfigure(label=["D4"], plots=True)
    def _plot_D4(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'pfc_data'):
            self.pfc_data = self.loadPFCdata()
        de_name = self.results["D4"]
        OPC_type = {"27":"OPC cluster","OPC":"Others","others":"OtherCP"}
        self.pfc_data.obs['OPC_annotation'] = self.pfc_data.obs["cluster_27"].map(OPC_type)
        
        axis = sc.pl.violin(
            self.pfc_data[self.pfc_data.obs['cell_types_predicted_cluster_consensus'] == 'OPC',:],
            de_name,use_raw=False,
            groupby="OPC_annotation",
            stripplot=False, 
            xlabel=' ',
            palette={"OPC cluster":fill_cmap.colors[0,:],
                    "Others":fill_cmap.colors[1,:]},
            inner="box", 
            ax = axis
            )
        
        return axis  
        
class Figure5(Figure):
    _figure_layout = [
        ["a", "a",  "a2", "a2","a3","a3"],
        #["a", "a",  "a2", "a2","a3","a3"],
        ["a4","a4", "a5","a5", "a6","a6"],
        #["a4","a4", "a5","a5", "a6","a6"],
        ["b", "b","b2","b2","b3","b3"],
        #["b", "b","b2","b2","b3","b3"],
        ["b4","b4","b5","b5","b6","b6"],
        #["b4","b4","b5","b5","b6","b6"],
        ["c","c",  "c2", "c2","c3","c3"],
        #["c","c",  "c2", "c2","c3","c3"]
    ]

    def __init__(
        self,
        output_dir = "./",
        sanity_installation_path = "/Sanity/bin/Sanity",
        seed = 42,
        n_repeats = 10, # number of repeats for computing knn stats
        TSNE_POINT_SIZE_A=0.6,
        TSNE_POINT_SIZE_B =1,
        LINE_WIDTH=0.2,
        figure_kwargs: dict = dict(dpi=300, figsize=(8.5,  10.625)),
        *args,
        **kwargs
    ):
        

        self.output_dir = output_dir
        self.seed = seed
        self.n_repeats = n_repeats
        self.TSNE_POINT_SIZE_A = TSNE_POINT_SIZE_A
        self.TSNE_POINT_SIZE_B = TSNE_POINT_SIZE_B
        self.LINE_WIDTH = LINE_WIDTH
        self.sanity_installation_path = sanity_installation_path
        self.method_keys = ['log1p', 'log1p+z', 'Pearson', 'Sanity', 'ALRA','Z_biwhite']
        self.method_names = list(algorithm_color_index.keys())
        self.results = {}
        kwargs['figure_kwargs'] = figure_kwargs
        super().__init__(*args, **kwargs)
    
    def load_data(self):
        #adata = sc.read_h5ad(self.output_dir+"SCORCH_INS_OUD_processed.h5ad")
        
        if os.path.isfile(self.output_dir+"fig5_normalized.h5ad"):
            adata = sc.read_h5ad(self.output_dir+"fig5_normalized.h5ad")
        else:
            adata = bipca_datasets.SCORCH_INS(base_data_directory = self.output_dir).get_filtered_data(store_filtered_data=True)['full']
            if issparse(adata.X):
                adata.X = adata.X.toarray()
            adata = apply_normalizations(adata,write_path=self.output_dir+"fig5_normalized.h5ad",
                                         normalization_kwargs={"log1p":{},
                            "log1p+z":{},
                            "Pearson":dict(clip=np.inf),
                            "ALRA":{"seed":42},
                            "Sanity":{}, 
                            "BiPCA":dict(n_components=-1, backend="torch", seed=42)},
                                         sanity_installation_path=self.sanity_installation_path)
        
        adata = self._annotate_adata(adata)

        
        self.adata = adata
        return adata

    def _annotate_adata(self,adata):
        
        adata.obsm['bipca_pcs'] = new_svd(adata.layers["Z_biwhite"],r=adata.uns["bipca"]['rank'])
        adata.obsm['bipca_tsne'] = np.array(TSNE(n_jobs=36).fit(adata.obsm['bipca_pcs']))
        sc.pp.neighbors(adata, n_neighbors=10, 
                use_rep = "bipca_pcs",method="gauss",
                n_pcs= adata.uns['bipca']['rank'])

        sc.tl.leiden(
            adata,
            resolution=0.5,
            random_state=42,
            n_iterations=2,
            directed=False,
        )
        adata.obs['cell_types'] = "others"
        adata.obs['cell_types'][adata.obs['leiden'] == "1"] = "Astrocytes"
        adata.obs['cell_types'][adata.obs['leiden'] == "2"] = "Astrocytes"
        return adata

    def runPCA(self): 
        
        PCset = OrderedDict()

        for method_key in self.method_keys:
            
            if method_key == 'ALRA':
                print(method_key)
                PCset[method_key] = new_svd(self.adata.layers[method_key],r=self.adata.uns['ALRA']["alra_k"])
            elif method_key == "Z_biwhite":
                print(method_key)
                PCset[method_key] = new_svd(self.adata.layers[method_key],r=self.adata.uns['bipca']["rank"])
            else:
                print(method_key)
                PCset[method_key] = new_svd(self.adata.layers[method_key] - np.mean(self.adata.layers[method_key],axis=0),r=50) # 50

            
        self.PCset = PCset
        return PCset


    def runTSNE(self):

        tsne_embeddings_full = OrderedDict()
        for method_key,PCs in self.PCset.items():
            tsne_embeddings_full[method_key] = np.array(TSNE().fit(PCs))
        
        # only astrocytes
        tsne_embeddings_sub = OrderedDict()
        for method_key,TSNEs in tsne_embeddings_full.items():
            tsne_embeddings_sub[method_key] = TSNEs[self.adata.obs['cell_types']=='Astrocytes',:]
        
        return tsne_embeddings_full, tsne_embeddings_sub

    def runLaplacianScore(self):

    
        #enc = OneHotEncoder(sparse_output=False)
        astrocyte_mask = self.adata.obs['cell_types']=='Astrocytes'
        #batch_label_onehot = enc.fit_transform(self.adata[astrocyte_mask,:].obs['replicate_id'].values.astype(int).reshape(-1,1) == 1)
        batch_label_onehot = (self.adata[astrocyte_mask,:].obs['replicate_id'].values.astype(int).reshape(-1,1) == 1)*1       
        batch_label_onehot = batch_label_onehot / norm(batch_label_onehot)

        ls_results = np.zeros((6))
        for ix,k in enumerate(self.method_keys):

            L,_,_ = graph_L(self.PCset[k][astrocyte_mask,:])
            score_vec = Lapalcian_score(batch_label_onehot,L)
            ls_results[ix] = score_vec #np.mean(score_vec)

        return ls_results


    def runAffineGrassman(self):

        ag_results = np.zeros((6))
        astrocyte_mask = self.adata.obs['cell_types']=='Astrocytes'
        batch_mask = self.adata.obs['replicate_id'].astype(int) == 1
        r2keep = self.adata.uns['bipca']['rank']
        for ix,k in enumerate(self.method_keys):
    
            Y0 = compute_stiefel_coordinates_from_data(self.adata.layers[k][(~batch_mask) & astrocyte_mask,:],r2keep,0)
            Y1 = compute_stiefel_coordinates_from_data(self.adata.layers[k][(batch_mask) & astrocyte_mask,:],r2keep,0)
            S = bipca.math.SVD(backend='torch', use_eig=True,vals_only=True,verbose=False).fit(Y0.T@Y1).S
            ag_results[ix] = np.sqrt((np.arccos(S)**2).sum())

        return ag_results

    def runDE(self):       
        DE_p_results = OrderedDict()
        logfc_results = OrderedDict()

        astrocyte_mask = self.adata.obs['cell_types']=='Astrocytes'
        batch_mask = self.adata.obs['replicate_id'].astype(int) == 1

        for k in self.method_keys:
            DE_p_results[k] = mannwhitneyu_de(self.adata.layers[k],
                        astrocyte_mask, batch_mask)

        DE_cutoff = 1e-2
        n_DE = np.zeros((6))
        for ix,method in enumerate(DE_p_results.keys()):
            valid_genes = ~np.isnan(DE_p_results[method])
            n_DE[ix] = np.sum( DE_p_results[method][valid_genes] < DE_cutoff )


        return n_DE
        
    
        
    @is_subfigure(label=["a","a2","a3","a4","a5","a6","b","b2","b3","b4","b5","b6","c","c2","c3"])
    def _compute_A_B_C(self):

        
        adata = self.load_data()
        PCset = self.runPCA()
        

        # for subplots
        tsne_embeddings_full, tsne_embeddings_sub = self.runTSNE()
        
        ls_results = self.runLaplacianScore()
        ag_results = self.runAffineGrassman()
        n_DE = self.runDE()
        
        results = {"a":tsne_embeddings_full["log1p"],
                   "a2":tsne_embeddings_full["log1p+z"],
                   "a3":tsne_embeddings_full["Pearson"],
                   "a4":tsne_embeddings_full["Sanity"],
                   "a5":tsne_embeddings_full["ALRA"],
                   "a6":tsne_embeddings_full["Z_biwhite"],
                   #"B":tsne_embeddings_sub,
                   "b":tsne_embeddings_sub["log1p"],
                   "b2":tsne_embeddings_sub["log1p+z"],
                   "b3":tsne_embeddings_sub["Pearson"],
                   "b4":tsne_embeddings_sub["Sanity"],
                   "b5":tsne_embeddings_sub["ALRA"],
                   "b6":tsne_embeddings_sub["Z_biwhite"],
                   "c":n_DE,
                   "c2":ls_results,
                   "c3":ag_results}

        return results


    def _tsne_plot(self,embed_df,ax,clabels,line_wd,point_s):

        # shuffle the points
        idx = np.random.default_rng(self.seed).permutation(embed_df.shape[0])
        
        cmap = npg_cmap(alpha=1)
        ax.scatter(x=embed_df[idx,0],
            y=embed_df[idx,1],
            facecolors= [mpl.colors.to_rgba(cmap(label)) for label in clabels[idx]],
            edgecolors=None,
            marker='.',
            linewidth=line_wd,
            s=point_s)

        #ax.set_aspect('equal', 'box')
    
        set_spine_visibility(ax,status=False)
    
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        return ax
             
    @is_subfigure(label=["a"], plots=True)
    @label_me(1)
    def _plot_A(self, axis: mpl.axes.Axes ,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()
        embed_df = self.results["a"]

        clabels = pd.factorize(self.adata.obs['replicate_id'].values.astype(int))[0]
        # insert a new axis for title
        axis2 = axis.inset_axes([0,0,1,1])
        
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

        axis2 = self._tsne_plot(embed_df,axis2,clabels=clabels,
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_A) 
        axis2.set_title("log1p",loc="left")
        
        return axis

    @is_subfigure(label=["a2"], plots=True)
    def _plot_A2(self, axis: mpl.axes.Axes ,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()
        
        embed_df = self.results["a2"]
        clabels = pd.factorize(self.adata.obs['replicate_id'].values.astype(int))[0]
        axis = self._tsne_plot(embed_df,axis,clabels=clabels,
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_A)    
        axis.set_title("log1p+z",loc="center")
        
        return axis

    @is_subfigure(label=["a3"], plots=True)
    def _plot_A3(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()
        
        embed_df = self.results["a3"]
        clabels = pd.factorize(self.adata.obs['replicate_id'].values.astype(int))[0]
        axis = self._tsne_plot(embed_df,axis,clabels=clabels,
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_A)         
        axis.set_title("Pearson",loc="center")
        
        return axis
    
    @is_subfigure(label=["a4"], plots=True)
    def _plot_A4(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()
        
        embed_df = self.results["a4"]
        clabels = pd.factorize(self.adata.obs['replicate_id'].values.astype(int))[0]
        axis = self._tsne_plot(embed_df,axis,clabels=clabels,
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_A)        
        axis.set_title("Sanity",loc="center")
        
        return axis

    @is_subfigure(label=["a5"], plots=True)
    def _plot_A5(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()

        embed_df = self.results["a5"]
        clabels = pd.factorize(self.adata.obs['replicate_id'].values.astype(int))[0]
        axis = self._tsne_plot(embed_df,axis,clabels=clabels,
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_A)       
        axis.set_title("ALRA",loc="center")


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
            ncol=4,fontsize=8,bbox_to_anchor=[0.4, -0.2], 
            loc='center',handletextpad=0,columnspacing=0
        )
        
        
        return axis

    @is_subfigure(label=["a6"], plots=True)
    def _plot_A6(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()

        embed_df = self.results["a6"]
        clabels = pd.factorize(self.adata.obs['replicate_id'].values.astype(int))[0]
        
        # insert a new axis and plot tsne inside
        #axis2 = axis.inset_axes([0,0.3,1,0.6])
        clabels = pd.factorize(self.adata.obs['replicate_id'].values.astype(int))[0]
        axis = self._tsne_plot(embed_df,axis,clabels=clabels,
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_A)       
        axis.set_title("BiPCA",loc="center")
        
        
        return axis

    @is_subfigure(label=["b"], plots=True)
    @label_me(1)
    def _plot_B(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()

        embed_df = self.results["b"]
        replicate_id_mapper = {0:0,1:5,2:5,3:5}
        astrocyte_mask = self.adata.obs['cell_types']=='Astrocytes'
        clabels = np.array([replicate_id_mapper[int(i)-1] for i in self.adata[astrocyte_mask,:].obs['replicate_id'].values])      
        # insert a new axis for title
        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,clabels=clabels,
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_B) 
        axis2.set_title("log1p",loc="center")
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

        return axis
        
    @is_subfigure(label=["b2"], plots=True)
    def _plot_B2(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()       
        embed_df = self.results["b2"]
        replicate_id_mapper = {0:0,1:5,2:5,3:5}
        astrocyte_mask = self.adata.obs['cell_types']=='Astrocytes'
        clabels = np.array([replicate_id_mapper[int(i)-1] for i in self.adata[astrocyte_mask,:].obs['replicate_id'].values])      
        
        
        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,clabels=clabels,
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_B) 
        
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        axis.set_title("log1p+z",loc="center")
        
        return axis

    @is_subfigure(label=["b3"], plots=True)
    def _plot_B3(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()       
        embed_df = self.results["b3"]
        replicate_id_mapper = {0:0,1:5,2:5,3:5}
        astrocyte_mask = self.adata.obs['cell_types']=='Astrocytes'
        clabels = np.array([replicate_id_mapper[int(i)-1] for i in self.adata[astrocyte_mask,:].obs['replicate_id'].values])      
        

        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,clabels=clabels,
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_B) 
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
                
        axis.set_title("Pearson",loc="center")
        return axis

    @is_subfigure(label=["b4"], plots=True)
    def _plot_B4(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()  
        embed_df = self.results["b4"]
        replicate_id_mapper = {0:0,1:5,2:5,3:5}
        astrocyte_mask = self.adata.obs['cell_types']=='Astrocytes'
        clabels = np.array([replicate_id_mapper[int(i)-1] for i in self.adata[astrocyte_mask,:].obs['replicate_id'].values])      


        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,clabels=clabels,
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_B) 
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
                
        axis.set_title("Sanity",loc="center")

        return axis

    @is_subfigure(label=["b5"], plots=True)
    def _plot_B5(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()       
        embed_df = self.results["b5"]
        replicate_id_mapper = {0:0,1:5,2:5,3:5}
        astrocyte_mask = self.adata.obs['cell_types']=='Astrocytes'
        clabels = np.array([replicate_id_mapper[int(i)-1] for i in self.adata[astrocyte_mask,:].obs['replicate_id'].values])      


        axis2 = axis.inset_axes([-0.2,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,clabels=clabels,
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_B) 
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
                
        axis.set_title("ALRA",loc="center")

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
            ncol=2,fontsize=8,bbox_to_anchor=[0.4, -0.2], 
            loc='center',handletextpad=0,columnspacing=0
        )
        
        
        return axis

    @is_subfigure(label=["b6"], plots=True)
    def _plot_B6(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()       
        embed_df = self.results["b6"]
        replicate_id_mapper = {0:0,1:5,2:5,3:5}
        astrocyte_mask = self.adata.obs['cell_types']=='Astrocytes'
        clabels = np.array([replicate_id_mapper[int(i)-1] for i in self.adata[astrocyte_mask,:].obs['replicate_id'].values])      


        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,clabels=clabels,
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_B) 
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        
               
        axis.set_title("BiPCA",loc="center")
        return axis
    
    @is_subfigure(label=["c"], plots=True)
    @label_me
    def _plot_C(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()

        n_DE = self.results['c']
        n_DE_ordered = {k:n_DE[ix] for ix,k in enumerate(algorithm_color_index.keys())}
        y_pos = np.linspace(0,1.0,6)
        cmap = npg_cmap(1)
        bar_colors=[cmap(v) for v in algorithm_color_index.values()]

        axis.barh(y_pos,
            width=np.array(list(n_DE_ordered.values())), 
            height=np.diff(y_pos)[0],
            color=bar_colors,
        edgecolor='k')
        axis.set_xlabel('\# DE')
        axis.invert_yaxis()
        axis.set_yticks(y_pos, labels=list(algorithm_color_index.keys()))

        
        return axis

    @is_subfigure(label=["c2"], plots=True)
    def _plot_C2(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:

        y_pos = np.linspace(0,1.0,6)
        cmap = npg_cmap(1)
        bar_colors=[cmap(v) for v in algorithm_color_index.values()]

        ls_results = self.results['c2']
        
        ls2plot_list = [ls_results[ix] for ix,method in enumerate(algorithm_color_index.keys())]
        axis.barh(y_pos,
            width= ls2plot_list,
            height=np.diff(y_pos)[0],
            color=bar_colors,
            edgecolor='k')
        axis.set_xlabel('Laplacian score')
        axis.invert_yaxis()
        axis.set_xlim(left=0.4)

        axis.set_yticks(y_pos, labels=list(algorithm_color_index.keys()))

        
        return axis

    @is_subfigure(label=["c3"], plots=True)
    def _plot_C3(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:

        ag_results = self.results['c3']
        y_pos = np.linspace(0,1.0,6)
        cmap = npg_cmap(1)
        bar_colors=[cmap(v) for v in algorithm_color_index.values()]

        axis.barh(y_pos,
                  [ag_results[ix] for ix,k in enumerate(algorithm_color_index.keys())],
                  height=np.diff(y_pos)[0],
                  color=bar_colors,
                  edgecolor='k')
        axis.set_yticks(y_pos, labels=list(algorithm_color_index.keys()))
        axis.invert_yaxis()
        axis.set_xlabel('Affine Grassmann distance')

        return axis

class SupplementaryFigure5(Figure):
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
        sanity_installation_path = "/Sanity/bin/Sanity",
        seed = 42,
        TSNE_POINT_SIZE_A=0.4,
        TSNE_POINT_SIZE_B =0.8,
        LINE_WIDTH=0.2,
        *args,
        **kwargs,
    ):
        self.output_dir = output_dir
        self.seed = seed
        self.TSNE_POINT_SIZE_A = TSNE_POINT_SIZE_A
        self.TSNE_POINT_SIZE_B = TSNE_POINT_SIZE_B
        self.LINE_WIDTH = LINE_WIDTH
        self.sanity_installation_path = sanity_installation_path
        self.method_keys = ['log1p', 'log1p+z', 'Pearson', 'Sanity', 'ALRA','Z_biwhite']
        self.method_names = ['log1p', 'log1p+z', 'Pearson', 'Sanity', 'ALRA', 'BiPCA']
        self.results = {}
        super().__init__(*args, **kwargs)

    
    
    def load_data(self):
        
        adata = bipca_datasets.SCORCH_INS_OUD(base_data_directory = self.output_dir).get_filtered_data()['full']
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        if os.path.isfile(self.output_dir+"fig5_normalized.h5ad"):
            adata = sc.read_h5ad(self.output_dir+"fig5_normalized.h5ad")
        else:
            adata = apply_normalizations(adata,write_path=self.output_dir+"fig5_normalized.h5ad",
                                         sanity_installation_path=self.sanity_installation_path)
        
        #self.r2keep = adata_others.uns['bipca']['rank']
        self.adata = adata
        return adata

    def runPCA(self): 
        
        PCset = OrderedDict()

        for method_key in self.method_keys:
            PCset[method_key] = new_svd(self.adata.layers[method_key],self.adata.uns['bipca']['rank'])
            
        self.PCset = PCset
        return PCset


    def runTSNE(self):

        tsne_embeddings_full = OrderedDict()
        for method_key,PCs in self.PCset.items():
            tsne_embeddings_full[method_key] = np.array(TSNE().fit(PCs))
        
        # only astrocytes
        tsne_embeddings_sub = OrderedDict()
        for method_key,TSNEs in tsne_embeddings_full.items():
            tsne_embeddings_sub[method_key] = TSNEs[self.adata.obs['cell_types']=='Astrocytes',:]
        
        return tsne_embeddings_full, tsne_embeddings_sub

    
        
    @is_subfigure(label=["A","A2","A3","A4","A5","A6","B","B2","B3","B4","B5","B6","C","C2","C3"])
    def _compute_A_B_C(self):

        
        adata = self.load_data()
        PCset = self.runPCA()
        n_UMIs = adata.obs['total_UMIs'].values
        
        tsne_embeddings_full, tsne_embeddings_sub = self.runTSNE()
        
        results = {"A":tsne_embeddings_full["log1p"],
                   "A2":tsne_embeddings_full["log1p+z"],
                   "A3":tsne_embeddings_full["Pearson"],
                   "A4":tsne_embeddings_full["Sanity"],
                   "A5":tsne_embeddings_full["ALRA"],
                   "A6":tsne_embeddings_full["Z_biwhite"],
                   #"B":tsne_embeddings_sub,
                   "B":tsne_embeddings_sub["log1p"],
                   "B2":tsne_embeddings_sub["log1p+z"],
                   "B3":tsne_embeddings_sub["Pearson"],
                   "B4":tsne_embeddings_sub["Sanity"],
                   "B5":tsne_embeddings_sub["ALRA"],
                   "B6":tsne_embeddings_sub["Z_biwhite"],
                   "C":n_UMIs,
                   "C2":[],
                   "C3":[],
                   }

        return results

    def _tsne_plot(self,embed_df,ax,c,line_wd,point_s):

        cmap = npg_cmap(alpha=1)
        ax.scatter(x=embed_df[:,0],
            y=embed_df[:,1],c=c,
                   cmap=heatmap_cmap,
            edgecolors=None,
            marker='.',
            linewidth=line_wd,
            s=point_s)
    
    
        set_spine_visibility(ax,status=False)
    
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        return ax
    
    @is_subfigure(label=["A"], plots=True)
    @label_me(1)
    def _plot_A(self, axis: mpl.axes.Axes ,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()
        embed_df = self.results["A"]
        
        # insert a new axis for title
        axis2 = axis.inset_axes([0,0,1,1])
        
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

        axis2 = self._tsne_plot(embed_df,axis2,c=np.log(self.adata.obs['total_UMIs'].values),
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_A) 
        axis2.set_title("log1p",loc="left")
        
        return axis

    @is_subfigure(label=["A2"], plots=True)
    def _plot_A2(self, axis: mpl.axes.Axes ,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()
        
        embed_df = self.results["A2"]
        
        axis = self._tsne_plot(embed_df,axis,c=np.log(self.adata.obs['total_UMIs'].values),
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_A)        
        axis.set_title("log1p+z",loc="left")
        
        return axis

    @is_subfigure(label=["A3"], plots=True)
    def _plot_A3(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()
        
        embed_df = self.results["A3"]
        
        axis = self._tsne_plot(embed_df,axis,c=np.log(self.adata.obs['total_UMIs'].values),
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_A)        
        axis.set_title("Pearson",loc="left")
        
        return axis
    
    @is_subfigure(label=["A4"], plots=True)
    def _plot_A4(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()
        
        embed_df = self.results["A4"]
        
        axis = self._tsne_plot(embed_df,axis,c=np.log(self.adata.obs['total_UMIs'].values),
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_A)        
        axis.set_title("Sanity",loc="left")
        
        return axis

    @is_subfigure(label=["A5"], plots=True)
    def _plot_A5(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()

        embed_df = self.results["A5"]
        
        axis2 = self._tsne_plot(embed_df,axis,c=np.log(self.adata.obs['total_UMIs'].values),
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_A)        
        axis2.set_title("ALRA",loc="left")
        
        vmin = np.min(np.log(adata.obs['total_UMIs'].values[adata.obs['cell_types']=='Astrocytes']))
        vmax = np.max(np.log(adata.obs['total_UMIs'].values[adata.obs['cell_types']=='Astrocytes']))
        norm = colors.Normalize(vmin, vmax)
        cb1 = self.figure.colorbar(cm.ScalarMappable( norm = norm,cmap=heatmap_cmap), ax=axis2,location='bottom')
        #cb1.set_label(label='log(n_UMIs)',weight='bold', horizontalalignment='right')
        cb1.ax.set_title('log(n_UMIs)',loc="right")
        
        return axis

    @is_subfigure(label=["A6"], plots=True)
    def _plot_A6(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()

        embed_df = self.results["A6"]
        
        
        axis = self._tsne_plot(embed_df,axis,c=np.log(self.adata.obs['total_UMIs'].values),
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_A)
        axis.set_title("BiPCA",loc="left")
        
        
        return axis

    @is_subfigure(label=["B"], plots=True)
    @label_me(1)
    def _plot_B(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()

        embed_df = self.results["B"]
             
        # insert a new axis for title
        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,c=np.log(self.adata.obs['total_UMIs'].values[self.adata.obs['cell_types']=='Astrocytes']),
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_B) 
        axis2.set_title("log1p",loc="left")
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

        return axis
        

    @is_subfigure(label=["B2"], plots=True)
    def _plot_B2(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()       
        embed_df = self.results["B2"]
        
        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,c=np.log(self.adata.obs['total_UMIs'].values[self.adata.obs['cell_types']=='Astrocytes']),
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_B) 
        
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        axis.set_title("log1p+z",loc="left")
        
        return axis

    @is_subfigure(label=["B3"], plots=True)
    def _plot_B3(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()       
        embed_df = self.results["B3"]

        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,c=np.log(self.adata.obs['total_UMIs'].values[self.adata.obs['cell_types']=='Astrocytes']),
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_B) 
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
                
        axis.set_title("Pearson",loc="left")
        return axis

    @is_subfigure(label=["B4"], plots=True)
    def _plot_B4(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()  
        embed_df = self.results["B4"]


        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,c=np.log(self.adata.obs['total_UMIs'].values[self.adata.obs['cell_types']=='Astrocytes']),
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_B) 
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
                
        axis.set_title("Sanity",loc="left")


        return axis

    @is_subfigure(label=["B5"], plots=True)
    def _plot_B5(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()       
        embed_df = self.results["B5"]
        

        axis2 = axis.inset_axes([-0.2,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,c=np.log(self.adata.obs['total_UMIs'].values[self.adata.obs['cell_types']=='Astrocytes']),
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_B) 
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        
        vmin = np.min(np.log(self.adata.obs['total_UMIs'].values[self.adata.obs['cell_types']=='Astrocytes']))
        vmax = np.max(np.log(self.adata.obs['total_UMIs'].values[self.adata.obs['cell_types']=='Astrocytes']))
        norm = colors.Normalize(vmin, vmax)
        cb2 = self.figure.colorbar(cm.ScalarMappable( norm = norm,cmap=heatmap_cmap), ax=axis,location='bottom')
        #cb2.set_label(label='log(n_UMIs)',weight='bold', horizontalalignment='right')
        cb2.ax.set_title('log(n_UMIs)',loc="right")
        axis.set_title("ALRA",loc="left")

        
        return axis

    @is_subfigure(label=["B6"], plots=True)
    def _plot_B6(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()       
        embed_df = self.results["B6"]
        

        axis2 = axis.inset_axes([0,0,1,1])
        axis2 = self._tsne_plot(embed_df,axis2,c=np.log(self.adata.obs['total_UMIs'].values[self.adata.obs['cell_types']=='Astrocytes']),
                                line_wd=self.LINE_WIDTH,point_s=self.TSNE_POINT_SIZE_B) 
        set_spine_visibility(axis,status=False)
    
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        
               
        axis.set_title("BiPCA",loc="left")
        return axis
    
    @is_subfigure(label=["C"], plots=True)
    @label_me
    def _plot_C(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        if not hasattr(self, 'adata'):
            self.adata = self.load_data()

        n_UMIs = [self.adata.obs['total_UMIs'].values[self.adata.obs['replicate_id'] == i+1] for i in range(4) ]
        cmap = npg_cmap(1)
        tab10map = [(key,mpl.colors.to_rgba(cmap(label))) for key,label in zip([1,2,3,4],[0,1,2,3])]
        label_mapper = {k+1:"Replicate "+str(k+1) for k in range(4)}
        
        
        ax = axis.boxplot(n_UMIs[::-1],patch_artist=True, showfliers=False,vert=False)
        for patch, color in zip(ax['boxes'], [tab10map[i][1] for i in range(4)][::-1]):
            patch.set_facecolor(color)   

        axis.set_yticks(np.arange(4) + 1,list(label_mapper.values())[::-1])
        #axis.invert_yaxis()
        axis.set_xlabel("\# UMIs")
        
        #axis.set_xticks(np.arange(4) + 1,list(label_mapper.values()))
        #axis.set_ylabel("\# UMIs")

        return axis


    @is_subfigure(label=["C2"], plots=True)
    def _plot_C2(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        set_spine_visibility(axis,status=False)
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        return axis

    @is_subfigure(label=["C3"], plots=True)
    def _plot_C3(self, axis: mpl.axes.Axes,results:np.ndarray) -> mpl.axes.Axes:
        set_spine_visibility(axis,status=False)
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
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