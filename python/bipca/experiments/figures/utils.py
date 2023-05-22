from functools import singledispatch
from typing import Dict, Union, Optional, List, Tuple

import matplotlib as mpl
import numpy as np

from bipca.plotting import set_spine_visibility
from .base import algorithm_to_npg_cmap_index


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
    parameter_name: str,
    parameter_var: str,
    xscale: str = "linear",
    xscale_params: Dict = {},
    yscale: str = "linear",
    yscale_params: Dict = {},
) -> mpl.axes.Axes:
    xlim = compute_axis_limits(results["x"], xscale, xscale_params)
    ylim = compute_axis_limits(results["y"], yscale, yscale_params)
    axis.scatter(
        results["x"],
        results["y"],
        s=10,
        color=npg_cmap()(algorithm_to_npg_cmap_index["BiPCA"]),
    )
    axis = plot_y_equals_x(axis)
    axis.set_xlabel(rf"True {parameter_name} ${parameter_var}$")
    axis.set_ylabel(rf"Estimated {parameter_name} $\hat{{{parameter_var}}}$")

    axis.set_xscale(xscale, **xscale_params)
    axis.set_yscale(yscale, **yscale_params)
    set_spine_visibility(axis, which=["top", "right"], status=False)
    # set the axis limits
    axis.set_xlim(xlim)

    axis.set_ylim(ylim)
    return axis
