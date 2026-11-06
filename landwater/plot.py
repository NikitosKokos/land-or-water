"""Render prediction grids as a small-multiples figure (and optional per-model PNGs)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .core import LAND, MISSING, UNKNOWN, WATER, Score, fmt_num, truth  # noqa: E402

BG = "#f9f8f5"
INK = "#1d1d1b"
MUTED = "#6b6a66"
UNCLEAR = "#8a8985"
NOT_ASKED = "#d6d4cd"

# Map view: index = cell code (WATER, LAND, UNKNOWN, MISSING).
MAP_COLORS = ["#000000", "#ffffff", UNCLEAR, NOT_ASKED]

# Error view codes and colors. Correct cells are muted so errors stand out.
OK_WATER, OK_LAND, FALSE_LAND, FALSE_WATER, D_UNKNOWN, D_MISSING = range(6)
DIFF_COLORS = ["#2b2b29", "#e6e5e0", "#eb6834", "#2a78d6", UNCLEAR, NOT_ASKED]


@dataclass
class Panel:
    label: str
    pred: np.ndarray  # cell codes
    score: Score | None
    is_truth: bool = False


def diff_grid(pred: np.ndarray, step: float) -> np.ndarray:
    is_land = truth(step)
    out = np.where(is_land, OK_LAND, OK_WATER).astype(np.int8)
    out[(pred == LAND) & ~is_land] = FALSE_LAND
    out[(pred == WATER) & is_land] = FALSE_WATER
    out[pred == UNKNOWN] = D_UNKNOWN
    out[pred == MISSING] = D_MISSING
    return out


def truth_grid(step: float) -> np.ndarray:
    return np.where(truth(step), LAND, WATER).astype(np.int8)


def _panel_title(p: Panel) -> str:
    title = p.label
    if p.score is not None:
        title += f"   {p.score.accuracy:.1%}"
        if p.score.coverage < 0.999:
            title += f"  ({p.score.coverage:.0%} asked)"
    return title


def render(panels: list[Panel], step: float, out: Path, diff: bool = False,
           title: str | None = None, ncols: int = 2) -> Path:
    n = len(panels)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)

    # Layout in inches; axes are placed by hand so 2:1 maps tile without distortion.
    pw = {1: 8.0, 2: 6.0}.get(ncols, 5.0)
    ph = pw / 2
    margin, gap_x, gap_y = 0.4, 0.6, 0.2
    header, row_title, footer = 1.5, 0.42, 0.45
    fig_w = 2 * margin + ncols * pw + (ncols - 1) * gap_x
    fig_h = header + nrows * (row_title + ph) + (nrows - 1) * gap_y + footer

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    colors = DIFF_COLORS if diff else MAP_COLORS
    cmap = ListedColormap(colors)

    for k, p in enumerate(panels):
        r, c = divmod(k, ncols)
        x = margin + c * (pw + gap_x)
        y_top = fig_h - header - r * (row_title + ph + gap_y) - row_title
        ax = fig.add_axes((x / fig_w, (y_top - ph) / fig_h, pw / fig_w, ph / fig_h))
        data = diff_grid(p.pred, step) if diff else p.pred
        ax.imshow(data, cmap=cmap, vmin=-0.5, vmax=len(colors) - 0.5,
                  interpolation="nearest", aspect="auto", extent=(-180, 180, -90, 90))
        ax.set_axis_off()
        ax.text(0, 1.04, _panel_title(p), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=14, color=INK)

    n_models = sum(1 for p in panels if not p.is_truth)
    if title is None:
        title = (f"How {n_models} blind Claude{'s' if n_models != 1 else ''} "
                 f"see{'' if n_models != 1 else 's'} the Earth")
        if diff:
            title = f"Where {n_models} blind Claude{'s' if n_models != 1 else ''} get the Earth wrong"
    pred_any = np.stack([p.pred for p in panels])
    n_points = panels[0].pred.size
    subtitle = (f"Asked “Land or Water?” at every {fmt_num(step)}° of the globe · "
                f"{n_points:,} points · no images, no tools, no thinking")
    has_missing = bool((pred_any == MISSING).any())
    if not diff:
        subtitle += " · white = “Land”"
        if (pred_any == UNKNOWN).any():
            subtitle += " · gray = no clear answer"
        if has_missing:
            subtitle += " · light gray = not asked yet"

    fig.text(margin / fig_w, 1 - 0.3 / fig_h, title, ha="left", va="top",
             fontsize=26 if ncols > 1 else 20, fontweight="bold", color=INK)
    fig.text(margin / fig_w, 1 - 1.0 / fig_h, subtitle, ha="left", va="top", fontsize=12, color=MUTED)

    if diff:
        handles = [
            Patch(color=DIFF_COLORS[FALSE_LAND], label="said Land, is water"),
            Patch(color=DIFF_COLORS[FALSE_WATER], label="said Water, is land"),
            Patch(color=UNCLEAR, label="no clear answer"),
        ]
        if has_missing:
            handles.append(Patch(color=NOT_ASKED, label="not asked yet"))
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1 - margin / fig_w, 1 - 0.3 / fig_h),
                   ncol=1 if ncols == 1 else len(handles), frameon=False, fontsize=11, labelcolor=INK)

    fig.text(1 - margin / fig_w, 0.18 / fig_h, "accuracy vs 1-km land mask, area-weighted",
             ha="right", va="bottom", fontsize=11, color=MUTED)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    return out


def save_pixels(pred: np.ndarray, out: Path, diff: bool = False, step: float = 2.0, width: int = 1440) -> Path:
    """Pixel-exact single map, upscaled with nearest neighbour."""
    data = diff_grid(pred, step) if diff else pred
    colors = DIFF_COLORS if diff else MAP_COLORS
    rgb = np.array([matplotlib.colors.to_rgb(c) for c in colors])[data]
    scale = max(1, width // data.shape[1])
    rgb = rgb.repeat(scale, axis=0).repeat(scale, axis=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out, rgb)
    return out
