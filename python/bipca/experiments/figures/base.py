from collections import OrderedDict
from typing import Optional, Dict, Tuple
from pathlib import Path
from numbers import Number
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tasklogger

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
plotting_alg_to_npg_cmap_idx = OrderedDict(
    [("log1p", 0), ("log1p_z", 4), ("SCT", 3), ("Sanity", 1), ("ALRA", 8), ("bipca", 2)]
)


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


class Figure(ABC):
    _required_datasets = abstractclassattribute()

    def __init__(
        self,
        base_plot_directory: str = "./figures",
        formatstr: str = "pdf",
        logger: Optional[tasklogger.TaskLogger] = None,
        verbose: int = 1,
    ):
        self.verbose = verbose
        if logger is None:
            self.logger = tasklogger.TaskLogger(
                name=self.__class__.__name__, level=self.verbose, if_exists="increment"
            )

        self._base_plot_directory = Path(base_plot_directory).resolve()
        self.formatstr = formatstr

    def __init_subclass__(cls, **kwargs):
        """__init_subclass__: Initialize subclasses of Figure.

        This function sets `cls._figure=cls.__name__` when cls is a
        first-generation subclass of `Figure`.
        """
        if __class__ in cls.__bases__:
            cls._figure = cls.__name__
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
    def figure(cls):
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
        if figure := getattr(cls, "_figure", False):
            return figure
        else:
            raise NotImplementedError(
                f"Property `figure` not implemented for {cls.__name__}"
            )

    def compute_subfigure(self, label: str):
        """compute_subfigure computes a specific subfigure"""
        self._subfigures[label]["compute"](self)

    def plot_subfigure(
        self, label: str, save: bool = True, show: bool = False
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """plot_subfigure plots a specific subfigure

        Parameters
        ----------
        label
            The label of the subfigure to plot.
        save
            Save the figure to disk.
        """
        fig, ax = self._subfigures[label]["plot"](self)
        if save:
            fig.savefig(
                str(self.plotting_path(subfigure_label=label)),
                bbox_inches="tight",
                transparent=False,
            )
        if show:
            fig.show()
        else:
            plt.close(fig)
        return fig, ax

    def compute_subfigures(self):
        """compute_subfigures computes all subfigures in the order they are defined in
        the class."""

        for label in self._subfigures:
            self.compute_subfigure(label)

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
            label: self.plot_subfigure(label, save=save, show=show)
            for label in self._subfigures
        }

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
        pth = Path(self._base_plot_directory / self.figure)
        pth.mkdir(parents=True, exist_ok=True)
        if subfigure_label is None:
            return pth
        else:
            return Path(str(pth) + subfigure_label + "." + self.formatstr)
