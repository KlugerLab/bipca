from collections import OrderedDict
from typing import Optional, Dict, Tuple, List, Union, Set
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tasklogger

from bipca.utils import flatten
from bipca.experiments.base import (
    ABC,
    abstractclassattribute,
    classproperty,
)

# Params for exporting to illustrator
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
# Params for plotting
SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10
plt.rcParams["text.usetex"] = True
plt.rcParams["axes.labelpad"] = 1.0
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def optional_arg_decorator(fn):
    def wrapped_decorator(*args):
        if len(args) == 1 and callable(args[0]):
            return fn(args[0])

        else:

            def real_decorator(decoratee):
                return fn(decoratee, *args)

            return real_decorator

    return wrapped_decorator


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


@optional_arg_decorator
def label_me(func, amount: int = 1):
    """label_me is a decorator that marks a plotting method for labeling.
    If a method is not marked as a plot, but is labeled as a subfigure, it will be used
    to compute the subfigure.

    """

    func._label_me = amount
    return func


def plots(func):
    """plots is a decorator that marks a method as a plot.
    If a method is not marked as a plot, but is labeled as a subfigure, it will be used
    to compute the subfigure.

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
        label_me: bool = True,
        is_child: bool = False,
        children: Set["SubFigure"] = None,
        *args,
        **kwargs,
    ):
        self.label = label
        self.plot_func = plot
        self.compute_func = compute
        self.label_me = label_me
        self._children = set() if children is None else children
        self._is_child = is_child
        self._axis = axis
        self._parent = compute.__self__

        self.logger = self._parent.logger

    @property
    def is_child(self) -> bool:
        return self._is_child

    @property
    def children(self) -> Set[str]:
        return self._children

    def register_child(self, child: str):
        self._children.add(child)

    # children.append = self._children.append

    @property
    def axis(self) -> mpl.axes.Axes:
        return self._axis

    @axis.setter
    def axis(self, value) -> mpl.axes.Axes:
        if isinstance(value, mpl.axes.Axes):
            self._axis = value
        else:
            raise ValueError("axis must be an mpl.axes.Axes object")

    @property
    def parent(self) -> "Figure":
        return self._parent

    @property
    def plotting_path(self) -> Path:
        return self.parent.plotting_path(self.label)

    @property
    def results_path(self) -> Path:
        return self.plotting_path.parent / (
            self.parent.figure_name + self.label + ".npy"
        )

    def full_extent(self, pad=0.01) -> mpl.transforms.Bbox:
        """Get the full extent of an axes, including axes labels, tick labels, and
        titles.
        https://stackoverflow.com/a/26432947
        """
        # For text objects, we need to draw the figure first, otherwise the extents
        # are undefined.
        ax = self.axis
        ax.figure.canvas.draw()
        items = ax.get_xticklabels() + ax.get_yticklabels()
        items = ax.get_children()
        # items += [ax, ax.title]
        bbox = mpl.transforms.Bbox.union(
            [
                bbox
                for item in items
                if not np.any(np.isinf((bbox := item.get_window_extent()).size))
                and np.all(bbox.size != 0)
            ]
        )
        child_bboxes = []
        for child in self.children:
            child_bboxes.append(child.full_extent(pad=0))
        bbox = mpl.transforms.Bbox.union([bbox] + child_bboxes)

        return bbox.expanded(1.0 + pad, 1.0 + pad)

    def compute(self, save: bool = True, recompute: bool = False) -> None:
        """compute computes the subfigure"""

        # compute the data if it's not in the results and not accessible from the disk,
        # or if we want to recompute
        if recompute or (
            self.label not in self._parent.results and not self.results_path.exists()
        ):
            with self.logger.task(f"{self.plotting_path.stem} results"):
                results = self.compute_func()
            if isinstance(self.compute_func._subfigure_label, list):
                for el in self.compute_func._subfigure_label:
                    # unpack results from multi-subfigure compute functions
                    self._parent.results[el] = results[el]
                    if save:
                        np.save(self.parent[el].results_path, self.parent.results[el])
            else:
                self._parent.results[self.label] = results
                if save:
                    np.save(self.results_path, self._parent.results[self.label])
        else:
            if self.label in self._parent.results:
                pass
            else:
                if self.results_path.exists():
                    self.parent.results[self.label] = np.load(
                        self.results_path, allow_pickle=True
                    )
                else:
                    # don't think we ever get here, but just in case, run the compute
                    self.parent.results[self.label] = self.compute_func()
            # compute the children
            for child in self.children:
                child.compute(save=save, recompute=recompute)

    @property
    def results(self):
        """results returns the results of the subfigure"""
        return self.parent.results[self.label]

    def plot(
        self,
        axis: mpl.axes.Axes = None,
        save: bool = True,
        save_data: bool = True,
        recompute_data: bool = False,
        clear: bool = True,
    ) -> mpl.axes.Axes:
        """plot plots a subfigure

        Parameters
        ----------
        recompute_data
            Whether to recompute the subfigure, by default False
        """
        if axis is None:
            axis = self.axis
        if axis == self.axis or clear:
            axis.clear()
        self.compute(recompute=recompute_data, save=save_data)
        assert self.label in self.parent.results
        axis = self.plot_func(axis)
        # plot the children
        for child in self.children:
            child.plot(
                save=False, recompute_data=False
            )  # false because we already computed by self.compute, and we don't
            # want to save because we're saving the parent
        if self.label_me > 0:
            axis.set_title(
                rf"\textbf{{{self.label}}}",
                fontdict={"fontsize": 12, "weight": "black"},
                ha="right",
                loc="left",
                x=-0.15 * self.label_me,
                y=1.0,
            )
        if save:
            self.save()
        return axis

    def save(self) -> None:
        """save saves the subfigure to disk"""
        if self._is_child:
            # search for the parent
            for label, subfig in self.parent.items():
                if self.label in subfig.children:
                    subfig.save()
                    return
        else:
            extent = self.full_extent().transformed(
                self.parent.figure.dpi_scale_trans.inverted()
            )
            self.parent.figure.savefig(
                str(self.plotting_path),
                bbox_inches=extent,
                transparent=True,
            )


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
        save_figure: bool = True,
        save_subfigures: bool = True,
        save_data: bool = True,
        recompute_data: bool = False,
        figure_kwargs: dict = dict(dpi=300, figsize=(8.5, 8.5)),
        subplot_mosaic_kwargs: dict = {},
    ):
        self.verbose = verbose
        if logger is None:
            self.logger = tasklogger.TaskLogger(
                name=self.__class__.__name__, level=self.verbose, if_exists="increment"
            )
        else:
            self.logger = logger
        self._base_plot_directory = Path(base_plot_directory).resolve()
        self.formatstr = formatstr
        self.recompute_data = recompute_data
        self.save_figure = save_figure
        self.save_subfigures = save_subfigures
        self.save_data = save_data
        self._figure, self._subfigures = self._init_figures(
            figure_kwargs, subplot_mosaic_kwargs
        )
        if hasattr(self, "_layout"):
            self._layout()

    def __getitem__(self, key) -> SubFigure:
        return self._subfigures.get(key)

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
                (f"not assigned in {self.figure_name}._figure_layout", not_assigned),
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
        if "layout" not in figure_kwargs and not hasattr(self, "_layout"):
            print('defaulting to "constrained" layout')
            figure_kwargs["layout"] = "constrained"
        fig = plt.figure(**figure_kwargs)
        subfig_axes = fig.subplot_mosaic(self._figure_layout, **subplot_mosaic_kwargs)
        # add subfigure interfaces for each subfigure.
        # first, build all the top-level subfigures. These don't have any internal
        # references, so we build them first.
        subfigures = {}
        for label, funcs in self._subfigures.items():
            if label[-1].isdigit():
                # this is a child subfigure
                pass
            else:
                if all(["compute" in funcs, "plot" in funcs]):
                    subfigures[label] = SubFigure(
                        axis=subfig_axes[label],
                        label=label,
                        plot=getattr(self, funcs["plot"].__name__),
                        compute=getattr(self, funcs["compute"].__name__),
                        label_me=getattr(funcs["plot"], "_label_me", False),
                    )
                # rebind the parent's compute and plot methods to the interface

        # now, build all the child subfigures. These have internal references to
        # their parents, so we build them second.
        for label, funcs in self._subfigures.items():
            if label[-1].isdigit():
                parent_label = label[:-1]
                assert parent_label in subfigures, f"{parent_label} not in {subfigures}"
                parent_funcs = self._subfigures[parent_label]
                if all(["compute" in funcs, "plot" in parent_funcs]):
                    subfigures[label] = SubFigure(
                        axis=subfig_axes[label],
                        label=label,
                        plot=getattr(self, funcs["plot"].__name__),
                        compute=getattr(self, funcs["compute"].__name__),
                        label_me=False,
                        is_child=True,
                    )
                    subfigures[parent_label].register_child(subfigures[label])
        return fig, dict(sorted(subfigures.items()))

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
                if isinstance(label, list):
                    for el in label:
                        if el not in cls._subfigures:
                            cls._subfigures[el] = {}
                        if getattr(method, "_plots", False):
                            cls._subfigures[el]["plot"] = method
                        else:
                            cls._subfigures[el]["compute"] = method
                else:
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
    def base_plot_directory(self) -> Path:
        return self._base_plot_directory

    @property
    def subfigures(self) -> Dict[str, SubFigure]:
        return {
            label: subfig
            for label, subfig in self._subfigures.items()
            if not subfig.is_child
        }

    @property
    def figure(self) -> mpl.figure.Figure:
        """figure: The mpl.figure.Figure associated with Figure.

        Returns
        -------
        mpl.figure.Figure
            The figure
        """
        return self._figure

    def compute_subfigure(self, label: str, recompute: bool = None, save: bool = None):
        """compute_subfigure computes a specific subfigure"""
        save = self.save_data if save is None else save
        recompute = self.recompute_data if recompute is None else recompute
        self._subfigures[label].compute(recompute=recompute, save=save)

    def plot_subfigure(
        self,
        label: str,
        save: bool = None,
        save_data: bool = None,
        recompute_data: bool = None,
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        save = self.save_subfigures if save is None else save
        save_data = self.save_data if save_data is None else save_data
        recompute_data = (
            self.recompute_data if recompute_data is None else recompute_data
        )
        subfig = self._subfigures[label]
        ax = subfig.plot(save=save, save_data=save_data, recompute_data=recompute_data)

        return ax

    def plot_figure(
        self,
        save: bool = None,
        save_subfigures: bool = None,
        save_data: bool = None,
        recompute_data: bool = None,
        clear: bool = True,
    ) -> Dict[str, Tuple[mpl.figure.Figure, mpl.axes.Axes]]:
        """plot_subfigures plots all subfigures in the order they are defined in the

        Parameters
        ----------
        save
            Save the figure to disk.
        save_subfigures
            Save the subfigure to disk.
        save_data
            Save the data to disk.
        recompute_data
            Recompute the data.
        """
        save = self.save_figure if save is None else save
        save_subfigures = (
            self.save_subfigures if save_subfigures is None else save_subfigures
        )
        save_data = self.save_data if save_data is None else save_data
        recompute_data = (
            self.recompute_data if recompute_data is None else recompute_data
        )
        results = {
            label: self.plot_subfigure(
                label,
                save=save_subfigures,
                save_data=save_data,
                recompute_data=recompute_data,
                clear=clear,
            )
            for label in self.subfigures
        }
        if save:
            self.save()
        return results

    def save(self):
        """save saves the figure to disk."""
        self.figure.savefig(
            str(self.plotting_path()), transparent=False, bbox_inches="tight"
        )

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
