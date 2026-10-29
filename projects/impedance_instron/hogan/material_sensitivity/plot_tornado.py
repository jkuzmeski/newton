"""Plot the marathon-time tornado chart: ``plot_tornado.py tornado_data.json OUT.png``."""

import json
import sys


def main(data_path, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(data_path) as f:
        d = json.load(f)
    groups = d["groups"]
    names = list(groups)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, ctrl in zip(axes, ("score", "sample"), strict=True):
        m = d["marathon065"][ctrl]
        for y, g in enumerate(names):
            lo, hi, ilo, ihi = groups[g]
            for v, color, h in (
                (lo, "tab:green", 0.6),
                (hi, "tab:red", 0.6),
                (ilo, "#9be29b", 0.3),
                (ihi, "#f2a3a3", 0.3),
            ):
                if v:
                    ax.barh(y, m[v], height=h, color=color)
                    if h > 0.5:
                        ax.text(
                            m[v] + (6 if m[v] >= 0 else -6),
                            y,
                            f"{m[v]:+.0f}",
                            va="center",
                            ha="left" if m[v] >= 0 else "right",
                            fontsize=8,
                        )
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{ctrl} controller (frozen)")
        ax.set_xlabel("marathon time change at coupling 0.65 [s]  (+ = faster)")
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(-560, 480)
    axes[0].set_yticks(range(len(names)), names, fontsize=9)
    fig.suptitle(
        "Slider FR3 shoe: one-at-a-time material sensitivity (held-out windows, original geometry)\ngreen = first-listed setting, red = second; pale = inner settings",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
