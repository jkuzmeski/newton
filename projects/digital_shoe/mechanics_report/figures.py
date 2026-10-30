# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot helpers shared with the Digital Instron sweep report."""

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
INK = "#243746"


def style() -> None:
    """Set report colors, readable labels, and editable vector text."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": "#788792",
            "grid.color": "#DFE5E9",
            "grid.alpha": 0.8,
            "lines.linewidth": 2.1,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "two-term-footwear-report",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def decorate(ax, xlabel: str, ylabel: str) -> None:
    """Label an axis and add a quiet background grid."""
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, linewidth=0.7)
    ax.set_axisbelow(True)


def cells(ax, xy, spacing, values=None, **kwargs):
    """Draw square column footprints at their geometric spacing."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    patches = [Rectangle((x - spacing / 2, y - spacing / 2), spacing, spacing) for x, y in xy]
    collection = PatchCollection(patches, linewidth=0.22, **kwargs)
    if values is not None:
        collection.set_array(values)
    ax.add_collection(collection)
    return collection
