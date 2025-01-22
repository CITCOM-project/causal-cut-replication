"""
This module provides common code to define a grouped boxplot.
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon


def color(colour, flierprops=None):
    if flierprops is None:
        flierprops = {}
    return {
        "boxprops": {"color": colour},
        "capprops": {"color": colour},
        "whiskerprops": {"color": colour},
        "flierprops": {"color": colour, "markeredgecolor": colour} | flierprops,
        "medianprops": {"color": colour},
    }


def plot_grouped_boxplot(
    groups,
    savepath=None,
    width=0.6,
    labels=None,
    colours=None,
    markers=None,
    title=None,
    xticklabels=None,
    xlabel=None,
    ylabel=None,
    ax=None,
    legend_args={},
):
    if ax is None:
        _, ax = plt.subplots()
    positions = max(len(x) for x in groups)
    plots = len(groups)
    if isinstance(labels, list) and len(labels) != plots:
        raise ValueError("If providing labels, please ensure that you provide as many as you have plots")
    if isinstance(colours, list) and len(colours) != plots:
        raise ValueError("If providing colours, please ensure that you provide as many as you have plots")
    for i, boxes in enumerate(groups):
        marker = markers[i] if isinstance(markers, list) else markers if markers is not None else "o"

        ax.boxplot(
            boxes,
            positions=np.array(range(positions)) * (plots + 1) + i,
            widths=width,
            label=labels[i] if labels is not None else None,
            **color(
                colours[i] if colours is not None else None,
                flierprops={"marker": marker, "markersize": width * 2},
            ),
        )

    if xticklabels is not None:
        ax.set_xticks(
            np.array(range(len(xticklabels))) * (plots + 1) + (((plots + (plots / 2) - 1) * width) / 2),
            xticklabels,
        )
    if labels is not None:
        ax.legend(**legend_args)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if savepath is not None:
        plt.savefig(savepath)
        plt.clf()


def bag_plot(
    x,
    y,
    label=None,
    savepath=None,
    colour=None,
    marker=None,
    title=None,
    xlabel=None,
    ylabel=None,
):
    # Calculate quartiles for both X and Y
    q1_x = np.percentile(x, 25)
    q3_x = np.percentile(x, 75)
    q1_y = np.percentile(y, 25)
    q3_y = np.percentile(y, 75)

    iqr_x = q3_x - q1_x
    iqr_y = q3_y - q1_y

    q1_minus_x = q1_x - (1.5 * iqr_x)
    q3_plus_x = q3_x + (1.5 * iqr_x)
    q1_minus_y = q1_y - (1.5 * iqr_y)
    q3_plus_y = q3_y + (1.5 * iqr_y)

    # Calculate median
    median_x = np.median(x)
    median_y = np.median(y)

    # Create a polygon for the Q1-Q3 region
    polygon = plt.Polygon([(q1_x, q1_y), (q1_x, q3_y), (q3_x, q3_y), (q3_x, q1_y)], alpha=0.2)
    polygon.set_color(colour)
    plt.gca().add_patch(polygon)
    polygon = plt.Polygon(
        [(q1_minus_x, q1_minus_y), (q1_minus_x, q3_plus_y), (q3_plus_x, q3_plus_y), (q3_plus_x, q1_minus_y)], alpha=0.2
    )
    polygon.set_color(colour)
    plt.gca().add_patch(polygon)

    # Plot the median point
    plt.scatter(x, y, color=colour, marker=marker, label=label)
    plt.scatter(median_x, median_y, color="black", marker=marker, s=50)

    # Set plot limits and labels
    plt.xlim(min(x) - 0.1, max(x) + 0.1)
    plt.ylim(min(y) - 0.1, max(y) + 0.1)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if savepath is not None:
        plt.legend()
        plt.savefig(savepath)
        plt.clf()
