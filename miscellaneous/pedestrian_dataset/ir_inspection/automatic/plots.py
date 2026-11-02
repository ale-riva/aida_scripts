"""
Pure figure-building functions for the dataset inspection tool.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


def bev_density_figure(annotations: pd.DataFrame) -> Figure:
    """
    2-D histogram of annotation centers in the center-of-gravity frame, plotted
    with y = left on the x-axis (positive/left on the left side of the plot) and
    x = forward on the y-axis. Density is colored from black (0) to red (the
    dataset's actual peak per-cell count), so the scale always matches the data.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    bev_cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])

    bev_cmap.set_under("white")

    if annotations.empty:
        ax.text(0.5, 0.5, "No annotations", ha="center", va="center", transform=ax.transAxes)
        quadrant_counts = {"top-left": 0, "top-right": 0, "bottom-left": 0, "bottom-right": 0}
    else:
        left = annotations["y"].to_numpy()
        forward = annotations["x"].to_numpy()
        margin = 5.0
        left_range = (float(left.min()) - margin, float(left.max()) + margin)
        forward_range = (float(forward.min()) - margin, float(forward.max()) + margin)
        counts, left_edges, forward_edges = np.histogram2d(left, forward, bins=80, range=[left_range, forward_range])
        mesh = ax.pcolormesh(left_edges, forward_edges, counts.T, cmap=bev_cmap, vmin=0, vmax=max(counts.max(), 1))
        fig.colorbar(mesh, ax=ax, label="annotation count")

        forward_positive, left_positive = forward >= 0, left >= 0
        quadrant_counts = {
            "top-left": int((forward_positive & left_positive).sum()),
            "top-right": int((forward_positive & ~left_positive).sum()),
            "bottom-left": int((~forward_positive & left_positive).sum()),
            "bottom-right": int((~forward_positive & ~left_positive).sum()),
        }

    ax.scatter([0], [0], marker="x", color="red", s=60, label="center of gravity")
    ax.set_xlabel("y - left (m)")
    ax.set_ylabel("x - forward (m)")
    ax.set_title("BEV annotation density (CoG frame)")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_xaxis()
    ax.legend(loc="upper right")

    fig.subplots_adjust(bottom=0.15)
    quadrant_text = "   ".join(f"{label}: {count}" for label, count in quadrant_counts.items())
    fig.text(0.5, 0.02, quadrant_text, ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def distance_distribution_figure(distance_hist: pd.DataFrame) -> Figure:
    """
    Grouped bar chart: annotation count and distinct-frame count per distance bin.
    distance_hist: DataFrame indexed by bin label, columns "annotation_count" and
    "frame_count" (see inspect_dataset.distance_histogram).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    positions = np.arange(len(distance_hist))
    width = 0.4
    # ax.bar(positions - width / 2, distance_hist["annotation_count"], width, label="annotations")
    # ax.bar(positions + width / 2, distance_hist["frame_count"], width, label="frames")
    ax.bar(positions, distance_hist["frame_count"], 2 * width, label="frames")
    ax.set_xticks(positions)
    ax.set_xticklabels(distance_hist.index, rotation=45, ha="right")
    ax.set_xlabel("distance from CoG (m)")
    ax.set_ylabel("count")
    ax.set_title("Distance distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def frame_count_figure(frame_count_hist: pd.Series) -> Figure:
    """Frame count per annotations-per-frame bucket (0, 1, 2, 3, 3+)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(frame_count_hist.index.astype(str), frame_count_hist.to_numpy(), label="frames")
    ax.set_title("Annotations per frame")
    ax.set_xlabel("annotations")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    return fig


def point_count_figure(point_count_hist: pd.Series) -> Figure:
    """Label count per exact number of LiDAR points inside its 3D box."""
    fig, ax = plt.subplots(figsize=(6, 5))
    if point_count_hist.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.bar(point_count_hist.index, point_count_hist.to_numpy(), label="labels")
    ax.set_title("Points inside bounding box")
    ax.set_xlabel("number of points")
    ax.set_ylabel("label count")
    ax.legend()
    fig.tight_layout()
    return fig