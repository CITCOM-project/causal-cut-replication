"""
This module provides common code to define a grouped boxplot.
"""

import numpy as np
import matplotlib.pyplot as plt


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
    showfliers=True,
    zigzag=[],
    logscale=False,
):
    if ax is None:
        _, ax = plt.subplots()
    positions = max(len(x) for x in groups)
    plots = len(groups)
    if isinstance(labels, list) and len(labels) != plots:
        raise ValueError("If providing labels, please ensure that you provide as many as you have plots")
    if isinstance(colours, list) and len(colours) != plots:
        raise ValueError(
            f"If providing {len(colours)} colours, please ensure that you provide as many as you have plots ({plots})"
        )
    if position_offsets is None:
        position_offsets = [0] * positions
    if isinstance(np.array(position_offsets), list) and len(np.array(position_offsets)) != positions:
        raise ValueError("If providing position_offsets, please ensure that you provide as many as you have positions")
    boxplots = {}
    for i, boxes in enumerate(groups):
        if logscale:
            boxes = [np.log(box) for box in boxes]
        shift = i + 0.15 if len(groups) == 1 else i
        marker = markers[i] if isinstance(markers, list) else markers if markers is not None else "o"

        boxes = ax.boxplot(
            boxes,
            positions=np.array(range(positions)) * (plots + 1) + shift + np.array(position_offsets),
            widths=width,
            label=labels[i] if labels is not None else None,
            patch_artist=True,  # fill with color
            **color(
                colours[i] if colours is not None else None,
                flierprops={"marker": marker, "markersize": width * 2},
            ),
            showfliers=showfliers,
        )
        for k, v in boxes.items():
            if k not in boxplots:
                boxplots[k] = []
            boxplots[k] += v
        if colours is not None:
            for patch, median in zip(boxes["boxes"], boxes["medians"]):
                patch.set_facecolor(colours[i])
                vertices = patch.get_path().vertices
                box_y_min = vertices[0, 1]
                box_y_max = vertices[2, 1]
                if box_y_min < box_y_max:
                    median.set_color("black")
    for mark in zigzag:
        ax.plot(
            mark,
            [0, 0],
            transform=ax.transAxes,
            marker="^",
            markersize=12,
            linestyle="none",
            color="w",
            mec="w",
            mew=1,
            clip_on=False,
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
    return boxplots
