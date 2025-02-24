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
    yticklabels=None,
    xlabel=None,
    ylabel=None,
    ax=None,
    legend_args={},
    position_offsets=None,
):
    if ax is None:
        _, ax = plt.subplots()
    positions = max(len(x) for x in groups)
    plots = len(groups)
    if isinstance(labels, list) and len(labels) != plots:
        raise ValueError("If providing labels, please ensure that you provide as many as you have plots")
    if isinstance(colours, list) and len(colours) != plots:
        raise ValueError("If providing colours, please ensure that you provide as many as you have plots")
    if position_offsets is None:
        position_offsets = [0] * positions
    if isinstance(np.array(position_offsets), list) and len(np.array(position_offsets)) != positions:
        raise ValueError("If providing position_offsets, please ensure that you provide as many as you have positions")
    for i, boxes in enumerate(groups):
        marker = markers[i] if isinstance(markers, list) else markers if markers is not None else "o"

        ax.boxplot(
            boxes,
            positions=np.array(range(positions)) * (plots + 1) + i + np.array(position_offsets),
            widths=width,
            label=labels[i] if labels is not None else None,
            **color(
                colours[i] if colours is not None else None,
                flierprops={"marker": marker, "markersize": width * 2},
            ),
        )

    if xticklabels is not None:
        ax.set_xticks(
            np.array(range(len(xticklabels))) * (plots + 1)
            + (((plots + (plots / 2) - 1) * width) / 2)
            + np.array(position_offsets),
            xticklabels,
        )
    if yticklabels is not None:
        ax.set_yticks(yticklabels)
    if labels is not None:
        ax.legend(**legend_args)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0)
        plt.clf()
