"""Matplotlib helper for the CSVs VIDUR writes.

A convenience, not the product -- the CSV is the product and Origin does the
final polish. What this guarantees is that a quick look at a series is honest:
fixed colour order, a legend whenever there is more than one sample, and no
second y-axis.

Palette: Okabe-Ito, reordered so no adjacent pair falls in the colour-vision
floor band. Validated, not eyeballed -- worst adjacent pair deltaE 9.6 (deutan),
20.0 (normal vision); lightness band, chroma floor and normal-vision floor all
pass. Three of the six sit below 3:1 contrast on white, which is why a legend is
always drawn and offset traces are labelled directly.
"""
from __future__ import annotations

import csv
from pathlib import Path

# Fixed order. Never cycled: a seventh sample does not get a reused colour,
# because two samples in one figure sharing a colour is a wrong figure.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]


def _read_csv(path: str | Path) -> tuple[list[str], list[list[float]]]:
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], rows[1:]
    cols = [[float(r[i]) if r[i] not in ("", "nan") else float("nan")
             for r in data] for i in range(len(header))]
    return header, cols


def _style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#888888")
    ax.tick_params(width=0.8, colors="#444444", labelcolor="#222222")


def plot_file(csv_path, y_column="norm_max", ax=None, label=None):
    """Plot one VIDUR CSV. Returns the axes."""
    import matplotlib.pyplot as plt
    header, cols = _read_csv(csv_path)
    if y_column not in header:
        raise ValueError(f"{csv_path} has no column {y_column!r}; has {header}")
    x, y = cols[0], cols[header.index(y_column)]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, linewidth=1.2, color=PALETTE[0],
            label=label or Path(csv_path).stem)
    _style(ax, header[0], y_column)
    return ax


def plot_series(csv_paths, labels=None, y_column="norm_max", offset=0.0,
                xlabel=None, title=None, ax=None):
    """Overlay or stack several CSVs -- the usual growth-series figure.

    `offset` > 0 stacks them (waterfall); each trace is then labelled directly
    at its right-hand end as well as in the legend, since offset plots are read
    top-to-bottom rather than by colour.
    """
    import matplotlib.pyplot as plt
    paths = [Path(p) for p in csv_paths]
    if len(paths) > len(PALETTE):
        raise ValueError(
            f"{len(paths)} samples but the fixed palette has {len(PALETTE)} colours. "
            "Colours are not recycled -- split this into small multiples, or pass a "
            "subset.")
    labels = labels or [p.stem for p in paths]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4 + 0.5 * len(paths) * bool(offset)))

    unit = None
    for i, (p, lab) in enumerate(zip(paths, labels)):
        header, cols = _read_csv(p)
        unit = unit or header[0]
        x, y = cols[0], cols[header.index(y_column)]
        y = [v + offset * i for v in y]
        ax.plot(x, y, linewidth=1.2, color=PALETTE[i], label=lab)
        if offset:
            ax.annotate(lab, (x[-1], y[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color="#222222")

    _style(ax, xlabel or unit, y_column, title)
    if offset:
        ax.set_yticks([])                      # a stacked axis has no meaningful scale
        ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)       # always: identity is never colour alone
    return ax


def save(ax, path, dpi=300):
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return str(path)
