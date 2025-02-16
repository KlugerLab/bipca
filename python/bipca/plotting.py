from collections.abc import Iterable
from numbers import Number
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import numpy as np
import pandas as pd
from scipy import stats
from .math import emp_pdf_loss, L2, L1, MarcenkoPastur, KDE
from .utils import feature_scale
from matplotlib.offsetbox import AnchoredText
from anndata._core.anndata import AnnData
from pychebfun import Chebfun
from matplotlib.ticker import (
    MaxNLocator,
    SymmetricalLogLocator,
    FuncFormatter,
    MultipleLocator,
)
import warnings
import shutil

mpl.set_loglevel("CRITICAL")
#usetex = mpl.checkdep_usetex(True) # deprecated in mpl >=3.6.0
usetex = True if shutil.which('tex') else False
plt.rcParams["text.usetex"] = usetex
mpl.set_loglevel("NOTSET")


def set_latex(latex=None):
    global usetex
    if latex is None:
        latex = not usetex
    if latex is True:
        mpl.set_loglevel("CRITICAL")
        #usetex = mpl.checkdep_usetex(True)
        usetex = True if shutil.which('tex') else False
        mpl.set_loglevel("NOTSET")

    else:
        usetex = latex
    plt.rcParams["text.usetex"] = usetex


def MP_histogram(
    svs,
    gamma,
    median=True,
    cutoff=None,
    theoretical_median=None,
    linewidth=1,
    hist_color=None,
    pdf_color="r",
    loss_fun=[L1, L2],
    ax=None,
    bins=100,
    histkwargs={},
):
    """
    Histogram of covariance eigenvalues compared to the theoretical Marcenko-Pastur law.

    Compute a density-normalized histogram of the covariance eigenvalues in `svs`
    and plot the histogram alongside the theoretical Marcenko-Pastur law.
    If multiple sets of eigenvalues are provided (by passing a list of arrays to `svs`),
    an average over many histograms is performed.

    Parameters
    ----------
    svs : array or list of arrays
        Covariance eigenvalues. If a list is provided,
        the output histograms will be the average of individual histograms.
    gamma : float
        Aspect ratio of the corresponding wide data matrix (gamma <= 1).
    cutoff : float, optional
        The Marcenko-Pastur rank cutoff. Defaults to (1+np.sqrt(gamma))**2
    theoretical_median : float, optional
        Theoretical median of the Marcenko-Pastur distribution. By default this is computed from the input.
    loss_fun : list of callable or False, optional
        Default L2. Compute and print loss according to `bipca.math.loss_fun`
    evaluate_on_bin : bool, optional
        Default True. Evaluate the theoretical Marcenko-Pastur distribution on the bins computed for the histogram, rather than a tiling.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Matplotlib axis object to plot the histograms in. Defaults to new axis.
    Other Parameters
    ----------------
    histkwargs : dict, optional
        Keyword arguments to np.histogram.
    """

    if cutoff is None:
        cutoff = (1 + np.sqrt(gamma)) ** 2
    if ax is None:
        ax = plt.axes()

    MP = MarcenkoPastur(gamma=gamma)
    if theoretical_median is None:
        theoretical_median = MP.median()
    sv = svs
    n, bins = np.histogram(
        sv[sv < cutoff * 2], bins=bins, range=[0, cutoff * 2], density=True, *histkwargs
    )
    actual_median = np.median(sv[sv < cutoff])
    w = bins[:-1] - bins[1:]
    ax.hist(bins[:-1], bins, weights=n, color=hist_color, zorder=10)
    est_dist = stats.rv_histogram([n, bins])

    xx = np.linspace(MP.a, MP.b, 10000)
    ax.plot(xx, MP.pdf(xx), "--", color=pdf_color, linewidth=linewidth, zorder=10)
    if median:
        ax.axvline(theoretical_median, c="r")
        ax.axvline(actual_median, c="y")
    if loss_fun:
        if isinstance(loss_fun, list):
            est_loss = [
                emp_pdf_loss(lambda x: MP.pdf(x), est_dist.pdf, loss=loss)
                for loss in loss_fun
            ]
        else:
            est_loss = [emp_pdf_loss(lambda x: MP.pdf(x), est_dist.pdf, loss=loss_fun)]
            loss_fun = [loss_fun]
        loss_str = "Error:"
        for val, fun in zip(est_loss, loss_fun):
            loss_str += "\n"
            loss_str += str(fun.__name__) + ": {:.3f}".format(val)
        anchored_text = AnchoredText(loss_str, loc="upper right", frameon=True)
        ax.add_artist(anchored_text)

    return ax


def MP_histograms_from_bipca(
    bipcaobj,
    both=False,
    legend=True,
    median=True,
    subtitle=True,
    full_text=True,
    bins=300,
    linewidth=1,
    fig=None,
    axes=None,
    figsize=(10, 5),
    dpi=300,
    title="",
    output="",
    figkwargs={},
    histkwargs={},
    anchoredtextprops={},
    **kwargs,
):
    """
    Spectral density before and after bipca biscaling and noise variance normalization from a single BiPCA object.

    Plot the spectral density of
    1) the unscaled, non-normalized
    2) scaled, non-normalized, and
    3) scaled, noise-variance normalized
    covariance matrices and the corresponding Marcenko-Pastur law learned by a `BiPCA` object.

    Parameters
    ----------
    bipcaobj : bipca.bipca.BiPCA
        A fit BiPCA estimator that contains `data_covariance_eigenvalues` and `biscaled_normalized_covariance_eigenvalues` attributes.
        These attributes may be set by bipcaobj.get_histogram_data().

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Matplotlib axis containing histogram.

    Other Parameters
    ----------------
    fig : matplotlib.figure.Figure, optional
        Figure object to plot the 3 new histograms in.
    axes : list of 3 or more matplotlib.axes._subplots.AxesSubplot, optional
    figsize : tuple, optional
        Figure size in inches
    """

    fig, axes = get_figure(fig=fig, axes=axes, dpi=dpi, figsize=figsize, **figkwargs)

    (
        plotting_spectrum,
        isquadratic,
        rank,
        M,
        N,
        gamma,
        b,
        c,
        bhat,
        chat,
        bhat_var,
        chat_var,
        kst,
        theoretical_median,
        cutoff,
        presvs,
        postsvs,
    ) = unpack_bipcaobj(bipcaobj)

    if presvs is None:
        if both:
            warnings.warn(
                "Both=True, but no presvs are stored in the object. Setting both to False."
            )
        both = False
    if axes is None:
        if both:
            naxes = 2
        else:
            naxes = 1
        axes = add_rows_to_figure(fig, ncols=naxes)
    else:
        if not isinstance(axes, Iterable):
            axes = [axes]
        naxes = len(axes)

    if len(axes) != naxes:
        raise ValueError("Number of axes must be 2")
    if both:
        ax1 = axes[0]
        ax1 = MP_histogram(
            presvs,
            gamma,
            cutoff=cutoff,
            theoretical_median=theoretical_median,
            median=median,
            linewidth=linewidth,
            loss_fun=False,
            bins=bins,
            ax=ax1,
            histkwargs=histkwargs,
            **kwargs,
        )
        if subtitle:
            ax1.set_title("Unscaled covariance " r"$\frac{1}{N}XX^T$")
        ax1.set_xlabel("Eigenvalue")
        ax1.set_ylabel("Density")
        ax1.grid(True)
        ax2 = axes[1]
    else:
        ax2 = axes[0]
        ax1 = None

    ax2 = MP_histogram(
        postsvs,
        gamma,
        cutoff=cutoff,
        theoretical_median=theoretical_median,
        median=median,
        linewidth=linewidth,
        loss_fun=False,
        bins=bins,
        ax=ax2,
        histkwargs=histkwargs,
        **kwargs,
    )
    if subtitle:
        ax2.set_title("Biwhitened covariance " r"$\frac{{1}}{{N}}YY^T$")
    ax2.set_xlabel("Eigenvalue")
    ax2.set_ylabel("Density")
    if isquadratic:
        if full_text:
            if usetex:
                txt = (
                    r"$ \displaystyle KS = {:.3f},~r = {:n}$"
                    "\n"
                    r"$b = {:.3f},~c = {:.3f}$"
                )
                "\n" r"$\hat{{b}} ={:.3f},~std(\hat{{b}}) ={:.3e}$"
                "\n" r"$\hat{{c}} ={:.3f},~std(\hat{{c}}) ={:.3e}$"
            else:
                txt = r"$ KS = {:.3f},~r = {:n}$" "\n" r"$b = {:.3f},~c = {:.3f}$"
                "\n" r"$\hat{{b}} ={:.3f},~std(\hat{{b}}) ={:.3e}$"
                "\n" r"$\hat{{c}} ={:.3f},~std(\hat{{c}}) ={:.3e}$"
            anchored_text = AnchoredText(
                txt.format(
                    kst, rank, b, c, bhat, np.sqrt(bhat_var), chat, np.sqrt(chat_var)
                ),
                loc="upper right",
                frameon=True,
                prop=anchoredtextprops,
            )
        else:
            if usetex:
                txt = (
                    r"$ \displaystyle KS = {:.3f},~r = {:n}$"
                    "\n"
                    r"$b = {:.3f},~c = {:.3f}$"
                )
            else:
                txt = r"$ KS = {:.3f},~r = {:n}$" "\n" r"$b = {:.3f},~c = {:.3f}$"
            anchored_text = AnchoredText(
                txt.format(kst, rank, b, c),
                loc="upper right",
                frameon=True,
                prop=anchoredtextprops,
            )
        ax2.add_artist(anchored_text)
    else:
        if usetex:
            txt = r"$\displaystyle KS = {:.3f},~r = {:n}$"
        else:
            txt = r"$ KS = {:.3f},~r = {:n}$"
        anchored_text = AnchoredText(
            txt.format(kst, rank),
            loc="upper right",
            frameon=True,
            prop=anchoredtextprops,
        )
        ax2.add_artist(anchored_text)
    ax2.grid(True)
    fig.tight_layout()
    if legend and median:
        fig.legend(
            ["Marcenko-Pastur PDF", "data"],
            loc="center",
            bbox_to_anchor=(0.5, 0),
            ncol=3,
        )
    ax2.text(0.5, 1.25, title, fontsize=16, ha="center", transform=ax2.transAxes)
    # fig.tight_layout()
    if output != "":
        plt.savefig(output, bbox_inches="tight")
    if both:
        return fig, ax1, ax2
    else:
        return fig, ax2


def spectra_from_bipca(
    bipcaobj,
    scale="linear",
    fig=None,
    minus=[10, 10],
    plus=[10, 10],
    axes=None,
    dpi=300,
    figsize=(10, 5),
    title="",
    output="",
    figkwargs={},
):
    fig, axes = get_figure(fig=fig, axes=axes, dpi=dpi, figsize=figsize, **figkwargs)
    if axes is None:
        naxes = 2
        axes = add_rows_to_figure(fig, ncols=naxes)

    (
        plotting_spectrum,
        isquadratic,
        rank,
        M,
        N,
        gamma,
        b,
        c,
        bhat,
        chat,
        bhat_var,
        chat_var,
        kst,
        theoretical_median,
        cutoff,
        presvs,
        postsvs,
    ) = unpack_bipcaobj(bipcaobj)

    svs = [presvs, postsvs]
    pre_rank = (presvs >= cutoff).sum()
    postrank = rank
    ranks = np.array([pre_rank, postrank], dtype=int)
    if isinstance(minus, int):
        minus = [minus] * 2
    if isinstance(plus, int):
        plus = [plus] * 2
    ranges = []
    for ix, rank in enumerate(ranks):
        ranges.append(
            (
                np.clip(rank - minus[ix] - 1, 0, M - 1),
                np.clip(rank + plus[ix], 1, M - 1),
            )
        )
    # needs some code for truncation or axis splitting
    x = []
    for lo, hi in ranges:
        x.append(np.arange(lo, hi))
    for ix, ax in enumerate(axes):
        # the plotting loop
        svs_idx = x[ix]
        the_svs = svs[ix][svs_idx]
        ax.bar(svs_idx + 1, the_svs, width=0.9)
        ax.axvline(
            x=ranks[ix] + 0.5, c="xkcd:light orange", linestyle="--", linewidth=2
        )
        ax.axhline(y=cutoff, c="xkcd:light red", linestyle="--", linewidth=2)
        ax.legend(
            [
                r"$\frac{\lambda_X(k)^2}{N}$",
                "selected rank = " + str(ranks[ix]),
                r"MP threshold $(1 + \sqrt{\gamma})^2$",
            ],
            loc="upper right",
        )
        ax.set_xlabel("Eigenvalue index k")
        ax.set_ylabel("Eigenvalue")
        ax.set_ylim(
            [
                np.min(the_svs) - 0.1 * np.min(the_svs),
                np.max(the_svs) + 0.1 * np.max(the_svs),
            ]
        )
        ax.set_xlim([np.min(svs_idx) + 1 - 0.9, np.max(svs_idx) + 1 + 0.6])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        x_ticks = np.append(ax.get_xticks()[1:-2] - 1, np.min(svs_idx) + 1)
        x_ticks = np.append(x_ticks, np.max(svs_idx + 1))
        ax.set_xticks(x_ticks)
        ax.set_yscale(scale)
        if scale == "symlog":
            ax.yaxis.set_major_locator(MajorSymLogLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(symlogfmt))
        # y_ticks = np.append(ax.get_yticks(), np.min(the_svs))
        # y_ticks = np.append(y_ticks,np.max(the_svs)+0.1*np.max(the_svs))
        ax.set_yticks(ax.get_yticks())
    axes[0].set_title("Unscaled covariance \n" r"$\frac{1}{N}XX^T$")
    axes[1].set_title("Biscaled covariance \n" r"$\frac{1}{N}YY^T$")

    fig.suptitle(title)
    fig.tight_layout()
    if output != "":
        plt.savefig(output, bbox_inches="tight")

    return fig, axes[0], axes[1]


def KS_from_bipca(
    bipcaobj,
    var="all",
    row=True,
    sharey=True,
    fig=None,
    title="",
    axes=None,
    dpi=300,
    figsize=None,
    output="",
    labelfontsize=16,
    figkwargs={},
):
    # parse the input var
    acceptable_var = ["all", "q", "sigma", "b", "c"]
    if isinstance(var, list):
        for vix, v in enumerate(var):
            if isinstance(v, str):
                if v.lower() in acceptable_var:
                    var[vix] = v.lower()
                else:
                    raise ValueError(
                        "var must be a string or a list of "
                        "strings in ['all','q','sigma','b','c']"
                    )
            else:
                raise TypeError("var must be a str ing or list of strings")
    elif isinstance(var, str):
        var = var.lower()
        if var in acceptable_var:
            var = [var]
        else:
            raise ValueError(
                "var must be a string or a list of "
                "strings in ['all','q','sigma','b','c']"
            )
    else:
        raise TypeError("var must be a string or list of strings")
    if var == "all" or any(v == "all" for v in var):
        var = ["q", "sigma", "b", "c"]
    ncols = len(var)

    if figsize is None:
        figsize = (int(ncols * 5), 5)

    fig, axes = get_figure(fig, axes, figsize=figsize, dpi=dpi, **figkwargs)

    if axes is None:
        axes = add_rows_to_figure(fig, ncols=ncols, sharey=True)
    else:
        if not isinstance(axes, Iterable):
            axes = [axes]

    if len(axes) != ncols:
        raise ValueError("Number of axes must be equal to len(var)")

    plotting_spectrum, isquadratic = unpack_bipcaobj(bipcaobj)[:2]
    if not isquadratic:
        raise ValueError("Cannot make KS plots for non-quadratic bipcaobj")
    pts = plotting_spectrum["fits"]
    pts = [pts[k] for k in pts.keys()]

    for v, ax in zip(var, axes):
        for ix, fitted_dict in enumerate(pts):
            x = fitted_dict[str(v)]
            y = fitted_dict["kst"]
            if v == "q":
                # build a chebfun object and get the minima
                domain = [0, 1]
                a, b = domain[0], domain[-1]
                p = Chebfun.from_coeff(fitted_dict["coefficients"], domain=domain)
                x2 = np.linspace(0, 1, 100000)
                pd = p.differentiate()
                pdd = pd.differentiate()
                e = pd.roots()
                mi = e[pdd(e) > 0]
                if len(mi) > 0:
                    mii = np.argmin(p(mi))
                    mii = mi[mii]

                else:
                    mii = x2
                    mi = np.min(p(x2))
                mi_x2 = p(x2)
                if (np.min(mi_x2) < p(mii)).any():
                    mii = np.argmin(mi_x2)
                    mii = x2[mii]
                px = ax.plot(x2, p(x2), zorder=1)
                ax.scatter(
                    mii,
                    p(mii),
                    color=px[-1].get_color(),
                    marker="x",
                    s=300,
                    label="Global minima",
                    zorder=3,
                )

            else:
                pdict = {xx: yy for xx, yy in zip(x, y)}
                p = lambda xx: list(map(lambda k: pdict[k], xx))
                px = ax.plot(x, p(x))
                mii = np.argmin(p(x))
                mi = np.min(p(x))
                ax.scatter(
                    x[mii],
                    mi,
                    color=px[-1].get_color(),
                    marker="x",
                    s=300,
                    label="Global minima",
                    zorder=3,
                )

            ax.scatter(x, p(x), label=f"Sample {ix+1} node", s=20)
            ax.set_xlabel(v, fontsize=labelfontsize)
    for ax in axes:
        ax.grid()
    axes[0].set_ylabel("KS", fontsize=labelfontsize)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    if output != "":
        plt.savefig(output, bbox_inches="tight")


def get_density_with_domain(
    data,
    apply_kde=True,
    jitter=0.025,
    npts=1000,
    X=None,
    xmin=0,
    xmax=1,
    prescaled=False,
    scaling="l1",
    **kwargs,
):
    # expects row-wise inputs!
    nrows = data.shape[0]
    Y = np.where(np.isnan(data), 0, data)
    if apply_kde:
        if npts is None:
            npts = Y.shape[1]

        if X is not None:
            assert npts in X.shape
        else:
            X = np.asarray([np.linspace(np.min(y), np.max(y), npts) for y in Y])
        if X.ndim == Y.ndim:
            if X.shape[0] == 1:
                X = np.asarray([X.squeeze()] * nrows)
            assert X.shape[0] == nrows
        else:
            X = np.asarray([X.squeeze()] * nrows)

        if jitter > 0:
            jit = np.abs(np.random.randn(*Y.shape)) * jitter
            Y += jit
        if Y.shape[1] > 1:
            y_kde = np.apply_along_axis(KDE, 1, Y)
        else:
            y_kde = [KDE(Y)]
        Y = np.asarray(
            list(map(lambda tupl: tupl[1].pdf(X[tupl[0], :]), enumerate(y_kde)))
        )
        Y = feature_scale(Y, axis=1)
    else:
        if X is not None:
            npts = Y.shape[1]
            assert npts in X.shape
        else:
            X = np.linspace(xmin, xmax, npts)

        if X.ndim == Y.ndim:
            if X.shape[0] == 1:
                X = np.asarray([X.squeeze()] * nrows)
            assert X.shape[0] == nrows
        else:
            X = np.asarray([X.squeeze()] * nrows)

    if prescaled:
        Y = Y
    else:
        if scaling.lower() == "l1":
            Y = Y / Y.sum(1)[:, None]
            Y = feature_scale(Y)
        else:
            Y = feature_scale(Y, axis=1)
    return Y, X, nrows, npts


def plot_density(
    data,
    ax,
    apply_kde=True,
    origin=0,
    npts=1000,
    X=None,
    xmin=0,
    xmax=1,
    prescaled=False,
    scaling="l1",
    color=None,
    line_color=None,
    fill_alpha=0.5,
    fill_color=None,
    linewidth=0.5,
    vanish_at=1e-3,
    zorder=0,
    **kwargs,
):
    assert vanish_at is False or isinstance(vanish_at, Number)

    Y, X, nrows, npts = get_density_with_domain(
        data,
        apply_kde=apply_kde,
        npts=npts,
        X=X,
        xmin=xmin,
        xmax=xmax,
        scaling=scaling,
        prescaled=prescaled,
    )

    if color is None:
        if line_color is not None:
            color = line_color
        else:
            color = "k"
    if isinstance(color, Iterable) and not isinstance(color, str) and len(color) != 4:
        assert len(color) == nrows
        line_color = np.asarray([list(mpl.colors.to_rgba(c)) for c in color])
    else:
        line_color = np.asarray([mpl.colors.to_rgba(color)] * nrows)
    if fill_color is None:
        fill_color = line_color.copy()
        fill_color[:, -1] *= fill_alpha
        fill_color = np.asarray([mpl.colors.to_rgba(fill) for fill in fill_color])
    elif (
        isinstance(fill_color, Iterable)
        and not isinstance(fill_color, str)
        and len(fill_color) != 4
    ):
        assert len(fill_color) == nrows
        fill_color = np.asarray([mpl.colors.to_rgba(fill) for fill in fill_color])
    else:
        fill_color = np.asarray([mpl.colors.to_rgba(fill_color)] * nrows)

    for row_index in range(nrows):
        y = Y[row_index, :]
        x = X[row_index, :]

        if vanish_at is not False:
            x = x[y > vanish_at]
            y = y[y > vanish_at]

        y = y + origin
        base = np.ones_like(y) * origin
        ax.plot(
            x,
            y,
            c=line_color[row_index, :],
            linewidth=linewidth,
            zorder=row_index + zorder,
        )
        ax.fill_between(
            x,
            base,
            y,
            color=fill_color[row_index, :],
            zorder=(row_index / nrows) * 0.1 + zorder,
        )
    return ax


def ridgeline(
    x,
    ax,
    f,
    key="group",
    axis=1,
    reverse=False,
    overlap=0.05,
    yticklabels=None,
    color="k",
    fill_alpha=0.5,
    fill_color=None,
    xaxis=True,
    reindexlevel=1,
    order=None,
    axislinewidth=0.7,
    pad=0.02,
    **kwargs,
):
    if isinstance(x, pd.DataFrame):
        if key is not None:
            groups = x.groupby(key, axis=axis)
            numgroups = len(groups)
        else:
            if axis == 1:
                groups = x.iteritems()
            else:
                groups = x.iterrows()
            groups = list(groups)
            numgroups = len(groups)
        if order is not None:
            groups = [
                group[1]
                .dropna(how="all")
                .reindex(level=reindexlevel, columns=order)
                .values
                for group in groups
            ]
        else:
            groups = [group[1].dropna(how="all").values for group in groups]
        if axis == 1:
            groups = [group.T for group in groups]

        numsubgroups = len(groups[0])
    else:
        if axis == 1:
            numgroups = x.shape[0]
            groups = x
        else:
            numgroups = x.shape[1]
            groups = x.T
        numsubgroups = 1

    if isinstance(color, Iterable) and not isinstance(color, str):
        assert len(color) == numgroups or len(color) == numsubgroups
        if len(color) == numgroups:
            line_color = []
            for c in color:
                if isinstance(c, Iterable) and not isinstance(c, str) and len(c) != 4:
                    assert len(c) == numsubgroups
                    line_color.append([list(mpl.colors.to_rgba(cc)) for cc in c])
                else:
                    line_color.append(
                        [list(mpl.colors.to_rgba(c)) for i in range(numsubgroups)]
                    )
            line_color = np.asarray(line_color)
        elif len(color) == numsubgroups:
            line_color = np.asarray(
                [list(mpl.colors.to_rgba(c)) for c in color] * numgroups
            )
    else:
        line_color = np.asarray(
            [[mpl.colors.to_rgba(color)] * numsubgroups] * numgroups
        )
    line_color = line_color.reshape(numgroups, numsubgroups, 4)
    if fill_color is None:
        fill_color = line_color.copy()
        fill_color[:, :, -1] *= fill_alpha
        fill_color = np.asarray(
            [[mpl.colors.to_rgba(c) for c in fill] for fill in fill_color]
        )
    elif isinstance(fill_color, Iterable) and not isinstance(fill_color, str):
        assert len(fill_color) == numgroups or len(fill_color) == numsubgroups
        if len(fill_color) == numgroups:
            fc = []
            for c in fill_color:
                if isinstance(c, Iterable) and not isinstance(c, str) and len(c) != 4:
                    assert len(c) == numsubgroups
                    fc.append(np.asarray([list(mpl.colors.to_rgba(cc)) for cc in c]))
                else:
                    fc.append(
                        np.asarray(
                            [list(mpl.colors.to_rgba(c)) for i in range(numsubgroups)]
                        )
                    )
            fill_color = np.asarray(fc)
        elif len(fill_color) == numsubgroups:
            fill_color = np.asarray(
                [list(mpl.colors.to_rgba(c)) for c in fill_color] * numgroups
            )
    else:
        fill_color = np.asarray(
            [[mpl.colors.to_rgba(fill_color)] * numsubgroups] * numgroups
        )
    fill_color = fill_color.reshape(numgroups, numsubgroups, 4)
    origins = []
    for rix, data in enumerate(groups):
        if data.ndim == 1:
            data = data[None, :]
        if rix == 0:
            origin = 0
            zorder = 0
        else:
            if not reverse:
                if ax.lines[0].get_ydata().size == 0:
                    val = 0
                else:
                    val = np.max(ax.lines[0].get_ydata())

                origin = origin + val * (1 - overlap)
                zorder = -rix
            else:
                Y, X, _, _ = get_density_with_domain(data, **kwargs)
                origin = origin - Y.max() * (1 - overlap)
                zorder = rix
        z = 1 if reverse else -1

        kwargs["fill_color"] = fill_color[rix, :, :].squeeze()
        kwargs["line_color"] = line_color[rix, :, :].squeeze()
        ax = f(data, ax, origin=origin, zorder=zorder, **kwargs)
        if xaxis:
            ax.axhline(
                y=origin, color="k", linewidth=axislinewidth, zorder=zorder + -(z) * 0.5
            )
        origins.append(origin)
    plotymin, plotymax, plotxmin, plotxmax = [], [], [], []
    for l in ax.lines:
        ydata = l.get_ydata()
        xdata = l.get_xdata()
        if len(ydata) != 0:
            plotymin.append(np.min(ydata))
            plotymax.append(np.max(ydata))
            plotxmin.append(np.min(xdata))
            plotxmax.append(np.max(xdata))
    plotymin, plotymax = np.min(plotymin), np.max(plotymax)
    plotxmin, plotxmax = np.min(plotxmin), np.max(plotxmax)
    set_spine_visibility(ax=ax)
    ax.set_xticks([])
    ax.set_xlim([plotxmin - pad, plotxmax + pad])
    ax.set_ylim([plotymin - pad, plotymax + pad])
    if yticklabels is None:
        if ax.get_yticklabels == []:
            ax.tick_params(left=False)
    else:
        assert numgroups == len(yticklabels)
        ax.set_yticks(origins)
        ax.set_yticklabels(yticklabels)

    return ax


def stacked_violin(
    data,
    ax,
    apply_kde=True,
    origin=0,
    npts=1000,
    X=None,
    xmin=0,
    xmax=1,
    prescaled=False,
    scaling="l1",
    yticklabels=None,
    color=None,
    line_color=None,
    fill_alpha=0.5,
    fill_color=None,
    linewidth=0.5,
    zorder=0,
    vanish_at=1e-3,
):
    assert vanish_at is False or isinstance(vanish_at, Number)

    Y, X, nrows, npts = get_density_with_domain(
        data,
        apply_kde=apply_kde,
        npts=npts,
        X=X,
        xmin=xmin,
        xmax=xmax,
        scaling=scaling,
        prescaled=prescaled,
    )

    if color is None:
        if line_color is not None:
            color = line_color
        else:
            color = "k"

    if isinstance(color, Iterable) and not isinstance(color, str):
        assert len(color) == nrows
        line_color = np.asarray([list(mpl.colors.to_rgba(c)) for c in color])
    else:
        line_color = np.asarray([mpl.colors.to_rgba(color)] * nrows)
    if fill_color is None:
        fill_color = line_color.copy()
        fill_color[:, -1] *= fill_alpha
        fill_color = np.asarray([mpl.colors.to_rgba(fill) for fill in fill_color])
    elif isinstance(fill_color, Iterable) and not isinstance(fill_color, str):
        assert len(fill_color) == nrows
        fill_color = np.asarray([mpl.colors.to_rgba(fill) for fill in fill_color])
    else:
        fill_color = np.asarray([mpl.colors.to_rgba(fill_color)] * nrows)

    for row_index in range(nrows):
        y = Y[row_index, :]
        x = X[row_index, :]
        if vanish_at is not False:
            x = x[y > vanish_at]
            y = y[y > vanish_at]
        y = y

        ax.plot(
            x,
            y + origin,
            c=line_color[row_index, :],
            linewidth=linewidth,
            zorder=row_index + zorder,
        )
        ax.plot(
            x,
            -y + origin,
            c=line_color[row_index, :],
            linewidth=linewidth,
            zorder=row_index + zorder,
        )
        ax.fill_between(
            x,
            -y + origin,
            y + origin,
            color=fill_color[row_index, :],
            zorder=row_index + zorder,
        )

    return ax


def set_spine_visibility(
    ax=None, which=["top", "right", "bottom", "left"], status="toggle"
):
    if ax is None:
        ax = plt.gca()
    if not isinstance(which, Iterable) or isinstance(which, str):
        which = [which]
    if not isinstance(status, Iterable) or isinstance(status, str):
        status = [status] * len(which)
    assert all([stat in ["toggle", True, False] for stat in status])

    for spine, stat in zip(which, status):
        if stat == "toggle":
            stat = not (ax.spines[spine].get_visible())
        ax.spines[spine].set_visible(stat)


def get_figure(fig=None, axes=None, **kwargs):
    if fig is None:
        if axes is None:  # neither fig nor axes was supplied.
            fig = plt.figure()
        else:
            if isinstance(axes, Iterable):
                pass
            else:
                axes = [axes]
            fig = axes[0].figure
    try:
        fig.set(**kwargs)
    except AttributeError as e:
        if str(e).endswith("'figsize'"):
            fig.set_size_inches(kwargs["figsize"])
    return fig, axes


def unpack_bipcaobj(bipcaobj):
    if isinstance(bipcaobj, AnnData):
        bipcadict = bipcaobj.uns["bipca"]
        plotting_spectrum = bipcadict["plotting_spectrum"]
        try:  # backwards compatibilty
            variance_estimator = bipcadict["variance_estimator"]
            isquadratic = variance_estimator == "quadratic"
        except:
            isquadratic = True
        rank = bipcadict["rank"]
    else:
        plotting_spectrum = bipcaobj.plotting_spectrum
        isquadratic = bipcaobj.variance_estimator == "quadratic"
        rank = bipcaobj.mp_rank

    M, N = plotting_spectrum["shape"]
    if M > N:
        gamma = N / M
    else:
        gamma = M / N

    # unpack the singular values
    try:
        # raw singular values
        presvs = plotting_spectrum["X"]
        presvs = -np.sort(-np.round(presvs, 4))
    except:
        presvs = None
    try:
        # biwhitened singular values
        postsvs = plotting_spectrum["Y"]
        postsvs = -np.sort(-np.round(postsvs, 4))
    except:
        postsvs = None

    if isquadratic:
        b = plotting_spectrum["b"]
        c = plotting_spectrum["c"]
        bhat = plotting_spectrum["bhat"]
        chat = plotting_spectrum["chat"]
        bhat_var = plotting_spectrum["bhat_var"]
        chat_var = plotting_spectrum["chat_var"]
    else:
        b = None
        c = None
        bhat = None
        chat = None
        bhat_var = None
        chat_var = None

    kst = plotting_spectrum["kst"]
    MP = MarcenkoPastur(gamma=gamma)

    theoretical_median = MP.median()
    cutoff = MP.b

    return (
        plotting_spectrum,
        isquadratic,
        rank,
        M,
        N,
        gamma,
        b,
        c,
        bhat,
        chat,
        bhat_var,
        chat_var,
        kst,
        theoretical_median,
        cutoff,
        presvs,
        postsvs,
    )


def colors_from_clusters(
    labels,
    cmap="tab10",
    color_function=lambda x: x,
    marker_function=lambda x: "s",
    linewidth_function=lambda x: 0,
    markersize_function=lambda x: 8,
):
    """colors_from_clusters
    Generate an array of rgba colors, a list of legend handles, and a label-to-rgba dictionary from a labels vector.

    Parameters
    ----------
    labels : length N array-like of strings
        The label assignments for each point.
    cmap : str, callable, or None, default 'tab10'
        A matplotlib colormap to use. If a string, the colormap is retrieved from `mpl.colormaps`.
        If a callable, the function should return at least len(unique(`labels`)) unique rgba values.
        If None, the outputs of `color_function` are used directly for colors.
    color_function : Callable or dict, optional
        A color mapping function. If used in conjunction with `cmap`, this function takes values in the range
        0, 1, ..., len(unique(`labels`))-1 and maps them into a suitable range for the colormap. When `cmap` is None,
        this function should return rgba values given values in the range 0, 1, ... len(unique(`labels`))-1 .
    marker_function : Callable or dict, optional
        _description_, by default lambdax:'s'
    linewidth_function : _type_, optional
        _description_, by default lambdax:0
    markersize_function : _type_, optional
        _description_, by default lambdax:8

    Returns
    -------
    _type_
        _description_
    """
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    clust2ix = {clust: ix for ix, clust in enumerate(np.unique(labels))}
    if cmap is not None:
        # color function maps into a cmap
        if hasattr(cmap, "resampled"):
            cmap = cmap.resampled(len(clust2ix))
        _color_function_ = lambda x: cmap(color_function(x))
    else:
        # color function (hopefully) returns valid colors
        _color_function_ = lambda x: color_function(x)
    handles, label2colormap = generate_custom_legend_handles(
        clust2ix,
        color_function=_color_function_,
        marker_function=marker_function,
        linewidth_function=linewidth_function,
        markersize_function=markersize_function,
    )
    color_assignments = np.asarray(np.vectorize(label2colormap.get)(labels)).T
    ncolors = len(set(label2colormap.values()))
    if ncolors < len(clust2ix):
        warnings.warn(
            f"Colormap contains {ncolors} colors, but {len(clust2ix)} "
            "clusters were specified. Some colors will be duplicates",
            RuntimeWarning,
        )
    return color_assignments, handles, label2colormap


def generate_custom_legend_handles(
    cluster_color_assignment,
    color_function=lambda x: x,
    marker_function=lambda x: "s",
    linewidth_function=lambda x: 0,
    markersize_function=lambda x: 8,
):
    if isinstance(color_function, dict):
        color_function = color_function.get
    if isinstance(marker_function, dict):
        marker_function = marker_function.get
    if isinstance(linewidth_function, dict):
        linewidth_function = linewidth_function.get
    if isinstance(markersize_function, dict):
        markersize_function = markersize_function.get
    label2colormap = {
        key: mpl.colors.to_rgba(color_function(value))
        for key, value in cluster_color_assignment.items()
    }
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker=marker_function(label),
            color=color,
            linewidth=linewidth_function(label),
            label=label,
            markersize=markersize_function(label),
        )
        for label, color in label2colormap.items()
    ]
    return handles, label2colormap


def add_colored_tick(ax, val, label, dim="x", color="red", **tick_params):
    if not isinstance(val, Iterable):
        val = [val]
        label = [label]
    bgaxis = ax.inset_axes([0, 0, 1, 1], zorder=2)
    bgaxis.set_ylim(ax.get_ylim())
    bgaxis.set_xlim(ax.get_xlim())
    bgaxis.set_xticks([])
    bgaxis.set_yticks([])
    bgaxis.spines["top"].set_visible(False)
    bgaxis.spines["right"].set_visible(False)
    bgaxis.spines["left"].set_visible(False)
    bgaxis.spines["bottom"].set_visible(False)
    bgaxis.tick_params(colors=color)
    if dim == "x":
        bgaxis.set_xscale(ax.xaxis.get_scale())
        bgaxis.xaxis.set_major_formatter(ax.xaxis.get_major_formatter())
        bgaxis.xaxis.set_major_locator(ax.xaxis.get_major_locator())
        bgaxis.set_xticks(val)
        bgaxis.set_xticklabels(label)
    else:
        bgaxis.set_yscale(ax.yaxis.get_scale())
        bgaxis.yaxis.set_major_formatter(ax.yaxis.get_major_formatter())
        bgaxis.yaxis.set_major_locator(ax.yaxis.get_major_locator())
        bgaxis.set_yticks(val)
        bgaxis.set_yticklabels(label)
    bgaxis.patch.set_alpha(0.01)
    bgaxis.tick_params("both", **tick_params)
    return bgaxis


def add_rows_to_figure(
    fig, ncols=None, nrows=1, sharey=False, wspace=0, share_labels=True
):
    """
    Add rows to a figure.
    Returns ncols*nrows axes, which defaults to
    the number of columns in the current figure, i.e. a single row with no spanning elements.


    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add rows to
    ncols : int
        Number of columns in the row, defaults to the current columns in the figure.
    nrows : int
        Number of rows to add

    Returns
    -------
    list(matplotlib.axes._subplots.AxesSubplot)
        The new ncols*nrows axes

    """
    new_axes = []
    if len(fig.axes) == 0:
        if ncols is None:
            ncols = 1
        if sharey:
            wspace = wspace
        else:
            wspace = None
        new_gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=wspace)
        for j in range(0, nrows):
            for i in range(ncols):
                newpos = new_gs[j, i]
                if sharey and i >= 1:
                    new_axes.append(fig.add_subplot(newpos, sharey=new_axes[j * ncols]))
                    if share_labels:
                        for label in new_axes[-1].get_yticklabels():
                            label.set_visible(False)
                else:
                    new_axes.append(fig.add_subplot(newpos))
    else:
        old_gs = fig.axes[0].get_gridspec()  # get the old gridspec
        curnrows = old_gs.nrows
        curncols = old_gs.ncols
        if ncols is None:
            ncols = curncols
        div = ncols
        if curncols % div != 0:  # we need to resize the grid
            tgtncols = curncols * div
        else:
            tgtncols = curncols
        histstride = tgtncols // div
        ogstride = tgtncols // curncols
        if sharey:
            wspace = wspace
        else:
            wspace = None
        new_gs = gridspec.GridSpec(
            nrows=curnrows + nrows, ncols=tgtncols, wspace=wspace
        )
        for ax in fig.axes:
            currentposition = ax.get_subplotspec()
            r0 = currentposition.rowspan
            c0 = currentposition.colspan
            newposition = new_gs[
                r0.start : r0.stop, c0.start * ogstride : c0.stop * ogstride
            ]
            ax.set_position(newposition.get_position(fig))
            ax.set_subplotspec(newposition)
        for j in range(-1 * nrows, 0):
            for i in range(ncols):
                newpos = new_gs[j, i * histstride : (i + 1) * histstride]
                if sharey and i >= 1:
                    new_axes.append(
                        fig.add_subplot(newpos, sharey=new_axes[abs((j + 1) * ncols)])
                    )
                    if share_labels:
                        for label in new_axes[-1].get_yticklabels():
                            label.set_visible(False)
                else:
                    new_axes.append(fig.add_subplot(newpos))

    return new_axes


class MajorSymLogLocator(SymmetricalLogLocator):
    def __init__(self):
        super().__init__(base=10.0, linthresh=1.0)

    @staticmethod
    def orders_magnitude(vmin, vmax):
        max_size = np.log10(max(abs(vmax), 1))
        min_size = np.log10(max(abs(vmin), 1))

        if vmax > 1 and vmin > 1:
            return max_size - min_size
        elif vmax < -1 and vmin < -1:
            return min_size - max_size
        else:
            return max(min_size, max_size)

    def tick_values(self, vmin, vmax):
        if vmax < vmin:
            vmin, vmax = vmax, vmin

        orders_magnitude = self.orders_magnitude(vmin, vmax)

        if orders_magnitude <= 1:
            spread = vmax - vmin
            exp = np.floor(np.log10(spread))
            rest = spread * 10 ** (-exp)

            stride = 10**exp * (
                0.25 if rest < 2.0 else 0.5 if rest < 4 else 1.0 if rest < 6 else 2.0
            )

            vmin = np.floor(vmin / stride) * stride
            return np.arange(vmin, vmax, stride)

        if orders_magnitude <= 2:
            pos_a, pos_b = np.floor(np.log10(max(vmin, 1))), np.ceil(
                np.log10(max(vmax, 1))
            )
            positive_powers = 10 ** np.linspace(pos_a, pos_b, int(pos_b - pos_a) + 1)
            positive = np.ravel(np.outer(positive_powers, [1.0, 5.0]))

            linear = np.array([0.0]) if vmin < 1 and vmax > -1 else np.array([])

            neg_a, neg_b = np.floor(np.log10(-min(vmin, -1))), np.ceil(
                np.log10(-min(vmax, -1))
            )
            negative_powers = -(
                10 ** np.linspace(neg_b, neg_a, int(neg_a - neg_b) + 1)[::-1]
            )
            negative = np.ravel(np.outer(negative_powers, [1.0, 5.0]))

            return np.concatenate([negative, linear, positive])

        else:
            pos_a, pos_b = np.floor(np.log10(max(vmin, 1))), np.ceil(
                np.log10(max(vmax, 1))
            )
            positive = 10 ** np.linspace(pos_a, pos_b, int(pos_b - pos_a) + 1)

            linear = np.array([0.0]) if vmin < 1 and vmax > -1 else np.array([])

            neg_a, neg_b = np.floor(np.log10(-min(vmin, -1))), np.ceil(
                np.log10(-min(vmax, -1))
            )
            negative = -(10 ** np.linspace(neg_b, neg_a, int(neg_a - neg_b) + 1)[::-1])

            return np.concatenate([negative, linear, positive])


def symlogfmt(x, pos):
    return f"{x:.6f}".rstrip("0")


def extract_color_list_from_string(string):
    string = string.replace('"', "")
    string = string.split(" ")
    return string


def get_alpha_cmap_from_cmap(cmap, alpha=1):
    cmap_arr = cmap(np.arange(cmap.N))
    cmap_arr[:, -1] = alpha
    return mpl.colors.ListedColormap(cmap_arr)


def aaas_cmap(alpha=1):
    string = '"#3B4992FF" "#EE0000FF" "#008B45FF" "#631879FF" "#008280FF" "#BB0021FF" "#5F559BFF" "#A20056FF" "#808180FF" "#1B1919FF"'
    output = extract_color_list_from_string(string)
    cmap = mpl.colors.ListedColormap(output)
    return get_alpha_cmap_from_cmap(cmap, alpha=alpha)


def gg_cmap(alpha=1):
    string = '"#F8766D" "#D89000" "#A3A500" "#39B600" "#00BF7D" "#00BFC4" "#00B0F6" "#9590FF" "#E76BF3" "#FF62BC"'
    output = extract_color_list_from_string(string)
    cmap = mpl.colors.ListedColormap(output)
    return get_alpha_cmap_from_cmap(cmap, alpha=alpha)
