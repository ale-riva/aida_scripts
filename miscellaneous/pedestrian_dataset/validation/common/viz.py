"""
Shared visualization helpers.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _depth_colors(depth: np.ndarray):
    """Map depths to a perceptually reasonable colormap range (5..60 m typical)."""
    import matplotlib
    d = np.clip(depth, 0, np.percentile(depth, 95) if depth.size else 1.0)
    norm = matplotlib.colors.Normalize(vmin=float(d.min()) if d.size else 0.0,
                                       vmax=float(d.max()) if d.size else 1.0)
    return matplotlib.colormaps["turbo"](norm(d))


# ---------------------------------------------------------------------------
# LiDAR -> image projection overlay
# ---------------------------------------------------------------------------

def lidar_projection_overlay(
    image: Optional[np.ndarray],
    uv: np.ndarray,
    depth: np.ndarray,
    out_path: str,
    title: str,
    point_size: float = 2.0,
) -> Optional[str]:
    """
    Scatter projected LiDAR points (colored by depth) over an image. Points that
    fall outside the image bounds or behind the camera are dropped, and the
    in-bounds fraction is annotated (a strong calibration sanity signal).
    """
    try:
        plt = _plt()
        if image is None:
            return None
        H, W = image.shape[:2]
        valid = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        u, v = uv[:, 0], uv[:, 1]
        inb = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        fig, ax = plt.subplots(figsize=(W / 140, H / 140), dpi=130)
        ax.imshow(image)
        if inb.any():
            ax.scatter(u[inb], v[inb], c=_depth_colors(depth[inb]),
                       s=point_size, edgecolors="none")
        frac = float(inb.sum()) / float(max(valid.sum(), 1))
        ax.set_title(f"{title}\n{inb.sum()} pts in-image "
                     f"({frac*100:.0f}% of {int(valid.sum())} in front)", fontsize=8)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception as exc:
        logger.warning("[viz] projection overlay failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Annotation overlay (2-D boxes + projected 3-D boxes)
# ---------------------------------------------------------------------------

def annotation_overlay(
    image: Optional[np.ndarray],
    boxes2d: Sequence[Tuple[float, float, float, float]],
    boxes3d_uv: Sequence[np.ndarray],
    edges: Sequence[Tuple[int, int]],
    out_path: str,
    title: str,
    labels: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """
    Draw axis-aligned 2-D boxes (green) and projected 3-D box wireframes
    (orange) over an image. boxes3d_uv is a list of (8,2) corner arrays.
    """
    try:
        plt = _plt()
        if image is None:
            return None
        H, W = image.shape[:2]
        fig, ax = plt.subplots(figsize=(W / 140, H / 140), dpi=130)
        ax.imshow(image)

        for i, (x1, y1, x2, y2) in enumerate(boxes2d):
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor="#2ecc71", linewidth=1.4))
            if labels and i < len(labels):
                ax.text(x1, max(0, y1 - 3), labels[i], color="#2ecc71",
                        fontsize=7, va="bottom")

        for corners in boxes3d_uv:
            if corners is None:
                continue
            for a, b in edges:
                ax.plot([corners[a, 0], corners[b, 0]],
                        [corners[a, 1], corners[b, 1]],
                        color="#e67e22", linewidth=1.0)

        ax.set_title(title, fontsize=8)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception as exc:
        logger.warning("[viz] annotation overlay failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Generic scatter / histogram helpers
# ---------------------------------------------------------------------------

def histogram_grid(
    series: dict,
    threshold: Optional[float],
    out_path: str,
    suptitle: str,
    xlabel: str,
) -> Optional[str]:
    """One histogram subplot per non-empty series; optional red threshold line."""
    try:
        plt = _plt()
        cols = [k for k, v in series.items() if len(v)]
        if not cols:
            return None
        ncols = min(len(cols), 3)
        nrows = (len(cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                                 squeeze=False)
        fig.suptitle(suptitle, fontsize=12, y=1.02)
        for i, col in enumerate(cols):
            ax = axes[i // ncols][i % ncols]
            vals = np.asarray(series[col], dtype=float)
            ax.hist(vals, bins=40, color="#4a7fb5", edgecolor="white", linewidth=0.3)
            if threshold is not None:
                ax.axvline(threshold, color="#c0392b", linestyle="--", linewidth=1.2)
            ax.set_title(col, fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
        for j in range(len(cols), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception as exc:
        logger.warning("[viz] histogram grid failed: %s", exc)
        return None


def scatter_timeline(
    y: Sequence[float],
    out_path: str,
    title: str,
    ylabel: str,
    hlines: Optional[Sequence[Tuple[float, str]]] = None,
) -> Optional[str]:
    try:
        plt = _plt()
        if not len(y):
            return None
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(np.arange(len(y)), y, s=8, alpha=0.6, color="#4a7fb5")
        for val, lbl in (hlines or []):
            ax.axhline(val, linestyle="--", linewidth=1.2, label=lbl)
        ax.set_title(title, fontsize=12); ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("index", fontsize=10)
        if hlines:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception as exc:
        logger.warning("[viz] timeline failed: %s", exc)
        return None


def xy_trajectory(
    xs: Sequence[float],
    ys: Sequence[float],
    out_path: str,
    title: str,
) -> Optional[str]:
    try:
        plt = _plt()
        if len(xs) < 2:
            return None
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(xs, ys, "-o", ms=2, lw=0.8, color="#4a7fb5")
        ax.scatter([xs[0]], [ys[0]], c="#27ae60", s=30, label="start", zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], c="#c0392b", s=30, label="end", zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title, fontsize=11); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception as exc:
        logger.warning("[viz] trajectory failed: %s", exc)
        return None