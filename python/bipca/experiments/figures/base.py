from collections import OrderedDict
from typing import Optional, Dict, Tuple, List, Union
from pathlib import Path
from numbers import Number
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tasklogger

from bipca.utils import flatten
from bipca.experiments.base import (
    ABC,
    abstractclassattribute,
    abstractmethod,
    classproperty,
)

# Params for exporting to illustrator
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
# Params for plotting
SMALL_SIZE = 8
BIGGER_SIZE = 10
plt.rcParams["text.usetex"] = True

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Orders, cmaps, etc used for all figures
algorithm_to_npg_cmap_index = OrderedDict(
    [("log1p", 0), ("log1p_z", 4), ("SCT", 3), ("Sanity", 1), ("ALRA", 8), ("BiPCA", 2)]
)


def is_subfigure(label: str):
    """is_subfigure is a decorator that marks a method as a subfigure.

    Parameters
    ----------
    label : str
        The label of the subfigure.
    """

    def decorator(func):
        func._subfigure_label = label
        return func

    return decorator


def plots(func):
    """plots is a decorator that marks a method as a plot.
    If a method is not marked as a plot, but is labeled as a subfigure, it will be used
    to compute the subfigure.
    Parameters
    ----------
    label : str
        The label of the plot.
    """

    func._plots = True
    return func


class SubFigure(object):
    """Wrapping interface for subfigures in a figure."""

    def __init__(
        self,
        axis: mpl.axes.Axes,
        label: str,
        plot: callable,
        compute: callable,
        *args,
        **kwargs,
    ):
        self.label = label
        self.plot_func = plot
        self.compute_func = compute
        self._axis = axis
        self._parent = plot.__self__

    @property
    def axis(self):
        return self._axis

    def full_extent(self, pad=0.0):
        """Get the full extent of an axes, including axes labels, tick labels, and
        titles.
        https://stackoverflow.com/a/26432947
        """
        # For text objects, we need to draw the figure first, otherwise the extents
        # are undefined.
        ax = self.axis
        ax.figure.canvas.draw()
        items = ax.get_xticklabels() + ax.get_yticklabels()
        items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        # items += [ax, ax.title]
        bbox = mpl.transforms.Bbox.union([item.get_window_extent() for item in items])

        return bbox.expanded(1.0 + pad, 1.0 + pad)

    def compute(self, recompute: bool = False):
        """compute computes the subfigure"""
        if self.label not in self._parent.results or recompute:
            self._parent.results[self.label] = self.compute_func()

    def plot(self, axis: mpl.axes.Axes = None, recompute: bool = False):
        """plot plots a subfigure

        Parameters
        ----------
        recompute
            Whether to recompute the subfigure, by default False
        """
        if axis is None:
            axis = self.axis
        self.compute(recompute=recompute)
        axis = self.plot_func(self.axis)
        return axis


class Figure(ABC):
    _figure_layout: List[
        Union[str, List[Union[str, List[str]]]]
    ] = abstractclassattribute()

    def __init__(
        self,
        base_plot_directory: str = "/bipca_data",
        formatstr: str = "pdf",
        logger: Optional[tasklogger.TaskLogger] = None,
        verbose: int = 1,
        figure_kwargs: dict = {},
        subplot_mosaic_kwargs: dict = {},
    ):
        self.verbose = verbose
        if logger is None:
            self.logger = tasklogger.TaskLogger(
                name=self.__class__.__name__, level=self.verbose, if_exists="increment"
            )

        self._base_plot_directory = Path(base_plot_directory).resolve()
        self.formatstr = formatstr
        self._figure, self._subfigures = self._init_figures(
            figure_kwargs, subplot_mosaic_kwargs
        )

    def _check_subfigures_(self):
        assigned_locations = flatten(self._figure_layout)
        not_assigned = []
        not_computable = []
        not_plottable = []
        for subfig, v in self._subfigures.items():
            if subfig not in assigned_locations:
                not_assigned.append(subfig)
            if "compute" not in v:
                not_computable.append(subfig)
            if "plot" not in v:
                not_plottable.append(subfig)
        if len(not_assigned + not_computable + not_plottable) > 0:
            error = f"Cannot initialize {self.figure_name}."
            for error_msg, bad_subfigs in [
                ("not assigned in self.figure_name._figure_layout", not_assigned),
                ("lack a compute method", not_computable),
                ("lack a plotting method", not_plottable),
            ]:
                if len(bad_subfigs) > 0:
                    error += (
                        f"\n\tSubfigures {*bad_subfigs,} "
                        f"are not valid as they {error_msg}"
                    )
            raise ValueError(error)
        else:
            return True

    def _init_figures(self, figure_kwargs={}, subplot_mosaic_kwargs={}):
        # build the figure and subfigure axes
        self._check_subfigures_()
        if "layout" not in figure_kwargs:
            figure_kwargs["layout"] = "constrained"
        fig = plt.figure(**figure_kwargs)
        subfig_axes = fig.subplot_mosaic(self._figure_layout, **subplot_mosaic_kwargs)
        subfigures = {
            k: SubFigure(
                axis=subfig_axes[k],
                label=k,
                plot=v["plot"].__get__(self, type(self)),
                compute=v["compute"].__get__(self, type(self)),
            )
            for k, v in self._subfigures.items()
            if all(["compute" in v, "plot" in v])
        }
        return fig, subfigures

    def __init_subclass__(cls, **kwargs):
        """__init_subclass__: Initialize subclasses of Figure.

        This function sets `cls._figure=cls.__name__` when cls is a
        first-generation subclass of `Figure`.
        """
        if __class__ in cls.__bases__:
            cls._figure_name = cls.__name__
        cls._subfigures = {}
        for method_name in filter(lambda s: "abstractmethods" not in s, dir(cls)):
            method = getattr(cls, method_name)
            if label := getattr(method, "_subfigure_label", False):
                if label not in cls._subfigures:
                    cls._subfigures[label] = {}
                if getattr(method, "_plots", False):
                    cls._subfigures[label]["plot"] = method
                else:
                    cls._subfigures[label]["compute"] = method
        cls._subfigures = {k: cls._subfigures[k] for k in sorted(cls._subfigures)}

    @classproperty
    def figure_name(cls):
        """figure: The figure associated with a 1st generation subclass of Figure.

        `cls.figure` is set when a class is initialized that immediately bases
        `Figure`. For instance, if `cls` bases `base_cls`, and `base_cls` bases
        `Figure`, then `cls.figure = base_cls.figure`.

        Returns
        -------
        str
            The figure name

        Raises
        ------
        NotImplementedError
            If figure is not implemented, e.g., the base
            class Figure.
        """
        if figure := getattr(cls, "_figure_name", False):
            return figure
        else:
            raise NotImplementedError(
                f"Property `figure_name` not implemented for {cls.__name__}"
            )

    @property
    def figure(self) -> mpl.figure.Figure:
        """figure: The mpl.figure.Figure associated with Figure.

        Returns
        -------
        mpl.figure.Figure
            The figure
        """
        return self._figure

    def compute_subfigure(self, label: str):
        """compute_subfigure computes a specific subfigure"""
        self._subfigures[label]["compute"](self)

    def plot_subfigure(
        self,
        label: str,
        save: bool = True,
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        subfig = self._subfigures[label]
        fig = self.figure
        ax = subfig.plot()
        if save:
            extent = subfig.full_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(
                str(self.plotting_path(subfigure_label=label)),
                bbox_inches=extent,
                transparent=False,
            )
        return ax

    def plot_subfigures(
        self, save: bool = True, show: bool = False
    ) -> Dict[str, Tuple[mpl.figure.Figure, mpl.axes.Axes]]:
        """plot_subfigures plots all subfigures in the order they are defined in the

        Parameters
        ----------
        save
            Save the figure to disk.
        show
            Show the figure.
        """
        return {
            label: self.plot_subfigure(label, save=save) for label in self._subfigures
        }

    @property
    def figure_path(self) -> Path:
        """figure_path returns the base path of a figure."""
        pth = Path(self._base_plot_directory / "results" / self.figure_name)
        pth.mkdir(parents=True, exist_ok=True)
        return pth

    def plotting_path(self, subfigure_label: Optional[str] = None) -> Path:
        """plotting_path returns the path to save a subfigure.

        Parameters
        ----------
        subfigure_label
            Subfigure path to retrieve. If not provided, the path to the figure
            directory is returned.

        Returns
        -------
        Path
            Path to subfigure
        """
        pth = Path(self._base_plot_directory / "results" / self.figure_name)
        pth.mkdir(parents=True, exist_ok=True)
        if subfigure_label is None:
            return self.figure_path / (self.figure_name + "." + self.formatstr)
        else:
            return Path(
                self.figure_path
                / (self.figure_name + subfigure_label + "." + self.formatstr)
            )
