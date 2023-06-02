from functools import singledispatch
from typing import Dict, Union, Optional, List, Tuple

import numpy as np
import scipy.stats as stats
from bipca.plotting import set_spine_visibility
from .base import mpl, plt


## generic python  utilties
def replace_from_dict(s: str, d: Dict[str, str]) -> str:
    """replace_from_dict replaces all keys in a dictionary with their values in a string

    Args:
        s (str): string to replace
        d (Dict[str, str]): dictionary of replacements

    Returns:
        str: string with replacements
    """
    for k, v in d.items():
        s = s.replace(k, v)
    return s


## Plotting utilities
def get_alpha_cmap_from_cmap(cmap, alpha=1):
    cmap_arr = cmap(np.arange(cmap.N))
    cmap_arr[:, -1] = alpha
    return mpl.colors.ListedColormap(cmap_arr)


def npg_cmap(alpha=1):
    colors = [
        "#E64B35FF",
        "#4DBBD5FF",
        "#00A087FF",
        "#3C5488FF",
        "#F39B7FFF",
        "#8491B4FF",
        "#91D1C2FF",
        "#DC0000FF",
        "#7E6148FF",
        "#B09C85FF",
    ]
    cmap = mpl.colors.ListedColormap(colors)
    return get_alpha_cmap_from_cmap(cmap, alpha=alpha)


def plot_y_equals_x(
    ax, linewidth=1, linestyle="--", color="k", label=r"$y=x$", **kwargs
):
    """plot_y_equals_x plots the line y=x on a given axis"""
    def_kwargs = ["linewidth", "linestyle", "color", "label"]
    for kwarg in def_kwargs:
        if kwarg not in kwargs:
            kwargs[kwarg] = locals()[kwarg]

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    max_pt = 1e15
    min_pt = np.minimum(xlim[0], ylim[0])
    ax.plot([min_pt, max_pt], [min_pt, max_pt], **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return ax


def logb(x, b):
    return np.log10(x) / np.log10(b)


def compute_axis_limits(x, scale, scale_params):
    if scale == "linear":
        range = x.max() - x.min()
        margin = 0.05  # 5% margin on either side of the range
        xmin = x.min() - margin * range
        xmax = x.max() + margin * range
    else:  # log, symlog, etc..
        # if the base is not specified, use 10
        if "base" not in scale_params:
            base = 10
        else:
            base = scale_params["base"]
        xmin = x.min()
        xmin = xmin - base ** np.floor(logb(xmin / base, base))
        xmax = x.max()
        xmax = xmax + base ** np.floor(logb(xmax / base, base))
    return xmin, xmax


def compute_minor_log_ticks(major_ticks, b):
    minor_ticks = []
    for xx in major_ticks:
        minor_ticks.extend(
            [
                b ** np.floor(logb(xx, b)) + b ** np.floor(logb(xx / b, b)) * i
                for i in range(1, b)
            ]
        )
    return minor_ticks


def correct_log0(x, b):
    newmin = b ** (np.floor(logb(x[np.nonzero(x)].min(), b)) - 1)
    return np.where(x == 0, newmin, x), newmin, np.any(x == 0)


def parameter_estimation_plot(
    axis: mpl.axes.Axes,
    results: Dict[str, np.ndarray],
    mean: bool = True,
    errorbars: bool = True,
    jitter: bool = True,
    errorbar_kwargs: Dict = dict(fmt="none", color="k"),
    scatter_kwargs: Dict = dict(s=5, color="k"),
) -> mpl.axes.Axes:
    if mean:
        x = np.unique(results["x"])
        y = np.zeros(x.shape)
        for ix, val in enumerate(x):
            inds = results["x"] == val
            y[ix] = results["y"][inds].mean()
        if errorbars:
            yerr = np.zeros((2, x.shape[0]))
            for ix, val in enumerate(x):
                inds = results["x"] == val
                yerr[:, ix] = stats.t.interval(
                    alpha=0.95,
                    df=len(results["y"][inds]) - 1,
                    loc=y[ix],
                    scale=stats.sem(results["y"][inds]),
                )
                yerr[:, ix] -= y[ix]
                yerr[:, ix] = np.abs(yerr[:, ix])
            axis.errorbar(
                x,
                y,
                yerr,
                **errorbar_kwargs,
            )
        if (marker := scatter_kwargs.pop("marker", False)) and isinstance(marker, list):
            for x, y, marker in zip(x, y, marker):
                axis.scatter(x, y, marker=marker, **scatter_kwargs)
        else:
            axis.scatter(x, y, **scatter_kwargs)
    else:
        x = results["x"].astype(float)
        y = results["y"].astype(float)
        if jitter:
            jit = np.random.normal(0, 0.001, size=x.shape) * x
            y += jit
            x += jit
        if (marker := scatter_kwargs.pop("marker", False)) and isinstance(marker, list):
            for x, y, marker in zip(x, y, marker):
                axis.scatter(x, y, marker=marker, **scatter_kwargs)
        else:
            axis.scatter(x, y, **scatter_kwargs)

    axis = plot_y_equals_x(axis, zorder=-1000)

    return axis
