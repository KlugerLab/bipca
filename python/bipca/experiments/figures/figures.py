import itertools
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

from bipca import BiPCA
from bipca.experiments.figures.base import Figure, is_subfigure, plots
from bipca.experiments.datasets import *


class Figure2(Figure):
    _required_datasets = None

    _figure_layout = [
        ["A", "B", "C", "D"],
        ["E", "E", "F", "F"],
        ["G", "H", "M", "M"],
        ["I", "J", "M", "M"],
    ]

    def __init__(
        self,
        seed=42,
        mrows=2000,
        ncols=4000,
        ranks=2 ** np.arange(0, 8),
        bs=2.0 ** np.arange(-7, 0),
        cs=2.0 ** np.arange(-7, 0),
        n_iterations=10,
        *args,
        **kwargs
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
    def generate_A(self):
        """generate_A Generate subfigure 2A, simulating the rank recovery in BiPCA."""
        seeds = [self.seed + i for i in range(self.n_iterations)]
        FixedPoisson = partial(
            RankRPoisson, mean=4, mrows=self.mrows, ncols=self.ncols, verbose=0
        )
        parameters = itertools.product(self.ranks, seeds)
        datasets = map(
            lambda ele: FixedPoisson(rank=ele[0], seed=ele[1]).get_filtered_data(),
            parameters,
        )
        results = map(
            lambda x: (
                x.uns["rank"],
                BiPCA(backend="torch", n_components=-1, seed=42, verbose=0)
                .fit(x.X)
                .mp_rank,
            ),
            datasets,
        )
        self.results["A"] = np.array(list(results))

    @is_subfigure(label="A")
    @plots
    def plot_A(self):
        fig, ax = plt.subplots(1)
        return fig, ax

    @is_subfigure(label="B")
    def generate_B(self):
        """generate_B Generate subfigure 2B, simulating the recovery of
        QVF parameter b in BiPCA."""

        seeds = [self.seed + i for i in range(self.n_iterations)]
        # use partial to make a QVFNegativeBinomial with fixed rank, c, and mean
        FixedNegativeBinomial = partial(
            QVFNegativeBinomial,
            rank=1,
            c=0.001,
            mean=4,
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
            ).get_filtered_data(),
            parameters,
        )

        # run biPCA on the datasets, extract the b parameter, and store the results
        results = map(
            lambda x: (
                x.uns["b"],
                BiPCA(backend="torch", seed=42, verbose=0, n_components=-1).fit(x.X).b,
            ),
            datasets,
        )
        self.results["B"] = np.array(list(results))

    @is_subfigure(label="C")
    def generate_C(self):
        """generate_C generate subfigure 2C, simulating the recovery of QVF
        parameter c"""
        # generate seeds
        seeds = [self.seed + i for i in range(self.n_iterations)]
        # use partial to make a QVFNegativeBinomial with fixed rank, b, and mean
        FixedNegativeBinomial = partial(
            QVFNegativeBinomial,
            rank=1,
            b=1,
            mean=4,
            mrows=self.mrows,
            ncols=self.ncols,
            verbose=0,
        )
        # generate the parameter set as combinations of c and seeds
        parameters = itertools.product(self.cs, seeds)
        # map the experiment over the parameters
        datasets = map(
            lambda ele: FixedNegativeBinomial(c=ele[0], seed=ele[1]), parameters
        )

        # run biPCA on the datasets, extract the c parameter, and store the results
        results = map(
            lambda x: (
                x.uns["c"],
                BiPCA(backend="torch", seed=42, n_components=-1, verbose=0).fit(x.X).c,
            ),
            datasets,
        )

        self.results["C"] = np.array(list(results))
