from functools import singledispatch
from typing import Dict, Union, Optional, List, Tuple

import numpy as np
import scipy.stats as stats
from bipca.plotting import set_spine_visibility
from .base import mpl, plt

from bipca.experiments.utils import download_url, download_urls, get_files, flatten


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
    b = float(b)
    if b == 2:
        offset = lambda x: b ** (np.floor(logb(x, b)) - 1)
        steps = b
    else:
        offset = lambda x: b ** np.floor(logb(x, b))
        steps = b - 1
    steps = int(steps)
    vmin = np.floor(logb(np.min(major_ticks), b))
    vmax = np.ceil(logb(np.max(major_ticks), b))
    major_ticks = np.arange(vmin, vmax, 1)
    major_ticks = b**major_ticks
    for xx in major_ticks:
        minor_ticks.extend(
            [b ** np.floor(logb(xx, b)) + offset(xx) * i for i in range(1, steps)]
        )
    return np.asarray(minor_ticks)


def compute_latex_ticklabels(
    ticks, b, skip=True, include_base=False, never_skip_zero=True
):
    ticks = np.asarray(ticks)
    is_negative = ticks < 0
    is_zero = ticks == 0
    ticks = np.abs(ticks)
    labels = []
    if include_base:
        base_str = rf"{b}^"
    else:
        base_str = ""
    for ix, t in enumerate(ticks):
        if not skip or (skip and ix % 2 == 0) or (never_skip_zero and is_zero[ix]):
            if is_zero[ix]:
                label = r"$0$"
            else:
                label = rf"{{{int(logb(t, b))}}}$"
                if is_negative[ix]:
                    label = r"$-" + base_str + label
                else:
                    label = r"$" + base_str + label
        else:
            label = None
        labels.append(label)
    return labels


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


def boxplot(axis, dist, colors=None):
    step = 0.1
    start = 1
    num = dist.shape[1]
    ypos = (np.arange(0, num) * step + start)[::-1]
    if colors is None:
        colors = []
    else:
        assert len(colors) == num
    ec = "k"
    lineprops = dict(linewidth=0.5, color=ec)
    bplot = axis.boxplot(
        dist,
        showfliers=False,
        widths=0.075,
        patch_artist=True,
        positions=ypos,
        vert=False,
        medianprops=lineprops,
        boxprops=dict(linewidth=0.5, edgecolor=ec),
        whiskerprops=lineprops,
        capprops=lineprops,
    )
    ypos = np.asarray([ypos for _ in range(dist.shape[0])])
    if len(colors) > 1:
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
    axis.scatter(dist, ypos, s=5, marker="x", linewidth=0.5, color="grey", zorder=2)
    axis.set_ylim(0.95, 1.55)
    return axis

def mean_var_plot(axis, df, mean_cdf = False):
        x = df["mean"]
        if mean_cdf:
            x = x.argsort().argsort() / len(x)
        axis.scatter(x, df["var"], s=0.5, c="k", marker="o")
        axis.set_xlabel(r"mean ($\mathrm{log}_{10}$)")
        axis.set_ylabel(r"variance ($\mathrm{log}_{10}$)")
        axis.set_xscale("log")
        axis.set_yscale("log")
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        xticks = axis.get_xticks()
        yticks = axis.get_yticks()
        axis.set_xticks(
            xticks,
            labels=compute_latex_ticklabels(xticks, 10),
        )
        axis.set_yticks(
            yticks,
            labels=compute_latex_ticklabels(yticks, 10),
        )
        axis.set_yticks(compute_minor_log_ticks(yticks, 10), minor=True)
        axis.set_xticks(compute_minor_log_ticks(xticks, 10), minor=True)

        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        set_spine_visibility(axis, which=["top", "right"], status=False)

        return axis
