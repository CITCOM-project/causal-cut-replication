"""
This module provides common code to define a grouped boxplot.
"""

import numpy as np
import matplotlib.pyplot as plt


def color(color, flierprops={}):
    return dict(
        boxprops=dict(color=color),
        capprops=dict(color=color),
        whiskerprops=dict(color=color),
        flierprops=dict(color=color, markeredgecolor=color) | flierprops,
        medianprops=dict(color=color),
    )


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
):
    positions = max(len(x) for x in groups)
    _, ax = plt.subplots()
    plots = len(groups)
    if isinstance(labels, list) and len(labels) != plots:
        raise ValueError(
            "If providing labels, please ensure that you provide as many as you have plots"
        )
    if isinstance(colours, list) and len(labels) != plots:
        raise ValueError(
            "If providing colours, please ensure that you provide as many as you have plots"
        )
    for i, boxes in enumerate(groups):
        marker = (
            markers[i]
            if isinstance(markers, list)
            else markers if markers is not None else "o"
        )
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
    ax.set_xticks(
        np.array(range(len(xticklabels))) * (plots + 1)
        + (((plots + (plots / 2) - 1) * width) / 2),
        xticklabels,
    )

    if labels is not None:
        ax.legend()
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(xlabel)

    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.clf()
