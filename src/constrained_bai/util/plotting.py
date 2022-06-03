# see https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
# import matplotlib
#
# matplotlib.use("Agg")

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Source: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap


def set_plot_style(family="serif"):
    import matplotlib
    import seaborn as sns

    linewidth = 3
    markeredgewidth = 3

    sns.set_context(
        "paper",
        font_scale=2.5,
        rc={"lines.linewidth": linewidth, "lines.markeredgewidth": markeredgewidth},
    )
    sns.set_style("white")
    matplotlib.rc(
        "font",
        **{
            "family": family,
            "serif": ["Computer Modern"],
            "sans-serif": ["Latin Modern"],
        },
    )
    matplotlib.rc("text", usetex=True)
    matplotlib.rc("lines", markersize=linewidth)
    matplotlib.rc("lines", markeredgewidth=markeredgewidth)


def plot_result_percentiles(
    xvalues,
    yvalues,
    legend_label,
    color,
    hatch,
    legend_handles,
    legend_labels,
    fix_negative=True,
    markers_every=1,
    plot_mean=False,
    plot_stddev=False,
    plot_stderr=False,
    dont_plot_percentiles=False,
    marker=None,
    alpha=1,
    zorder=1,
    linestyle="-",
):
    """
    Plot percentiles or mean/stdev of a series of datapoints.

    xvalues: list of shape (n)
    yvalues: list of shape (n, m)
    """
    assert not (plot_stddev and plot_stderr)
    n = len(xvalues)
    assert len(yvalues) == n

    xvalues = np.array(xvalues)
    idx = np.argsort(xvalues)
    xvalues = xvalues[idx]
    middle_list, lower_list, upper_list = [], [], []

    for i in idx:
        if plot_mean:
            middle = np.mean(yvalues[i])
        else:
            middle = np.median(yvalues[i])

        if plot_stddev or plot_stderr:
            stddev = np.std(yvalues[i])
            if plot_stderr:
                stddev /= np.sqrt(len(yvalues[i]))
            lower = middle - stddev
            upper = middle + stddev
        else:
            lower = np.percentile(yvalues[i], 25)
            upper = np.percentile(yvalues[i], 75)

        middle_list.append(middle)
        lower_list.append(lower)
        upper_list.append(upper)

    p1 = plt.plot(
        xvalues,
        middle_list,
        color=color,
        marker=marker,
        markevery=markers_every,
        markersize=11,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )[0]

    if not dont_plot_percentiles:
        p2 = plt.fill_between(
            xvalues,
            lower_list,
            upper_list,
            color=color,
            # hatch=hatch_cycle[hatch],
            alpha=0.2,
            zorder=zorder,
        )
        legend_handles.append((p1, p2))
    else:
        legend_handles.append((p1,))

    legend_labels.append(legend_label)
    return legend_handles, legend_labels


def customized_box_plot(ax, steps, lowers, q1s, medians, q3s, uppers, *args, **kwargs):
    """
    Generates a customized boxplot based on the given percentile values.

    Based on:
    https://stackoverflow.com/questions/27214537/
    is-it-possible-to-draw-a-matplotlib-boxplot-given-the-percentile-values-instead
    """
    boxes = [
        {
            "label": step,
            "whislo": lower,  # Bottom whisker position
            "q1": q1,  # First quartile (25th percentile)
            "med": median,  # Median         (50th percentile)
            "q3": q3,  # Third quartile (75th percentile)
            "whishi": upper,  # Top whisker position
            "fliers": [],  # Outliers
        }
        for i, (step, lower, q1, median, q3, upper) in enumerate(
            zip(steps, lowers, q1s, medians, q3s, uppers)
        )
    ]
    ax.bxp(boxes, showfliers=False, positions=steps, *args, **kwargs)
