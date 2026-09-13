"""
The one reliable way to compare "flatness" (the loss-landscape "rise" metric,
docs/16 section 8) ACROSS different (solver, mu) cells. The 3D terrain plots
(plot_multidirection_comparison.py) are good for seeing one cell's own shape
in detail, and for comparing 3 seeds of the SAME cell against each other with
a shared z-axis -- but eyeballing "is this cell's terrain flatter than that
OTHER cell's terrain" across two separate 3D images is unreliable even with
matched axes, because viewing angle and terrain roughness both distort
perceived steepness. A plain 2D bar chart of the actual number sidesteps that
completely.

Bars = mean rise across the 3 direction-pair seeds. Error bars = min/max
across those seeds (how much the number moved under a different random
direction choice -- see docs/16 section 8's outlier-seed discussion). Bar
color = verdict (green=converged, orange=unstable), so the "trapped cells sit
in flatter basins" claim is directly visible as a color/height pattern, not
just implied by an adjacent table.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "results", "phase3", "stiffness_map")
SOLVERS = ["euler", "midpoint", "rk4", "tsit5"]
MUS = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0]
SOLVER_COLORS = {"euler": "#1f77b4", "midpoint": "#2ca02c", "rk4": "#9467bd", "tsit5": "#17becf"}


def _fmt_mu(mu):
    return f"{int(mu)}.0" if mu == int(mu) else str(mu)


def load_verdicts():
    """verdict per (solver, mu), read from the real probe metrics.json -- so
    the bar color reflects the actual measured outcome, not a guess."""
    out = {}
    for solver in SOLVERS:
        for mu in MUS:
            path = os.path.join(ROOT, "probe", f"{solver}_mu{_fmt_mu(mu)}_dt0.05", "metrics.json")
            with open(path) as f:
                out[(solver, mu)] = json.load(f)["verdict"]
    return out


def main():
    with open(os.path.join(ROOT, "loss_landscape", "rise_multiseed.json")) as f:
        rise_data = json.load(f)
    verdicts = load_verdicts()

    fig, ax = plt.subplots(figsize=(16, 7))
    n_solvers = len(SOLVERS)
    group_width = 0.8
    bar_width = group_width / n_solvers
    x = np.arange(len(MUS))

    verdict_color = {"converged": "#2ca02c", "unstable": "#e67e22", "diverged": "#d62728"}

    for j, solver in enumerate(SOLVERS):
        means, mins, maxs, colors = [], [], [], []
        for mu in MUS:
            entry = rise_data[solver][str(mu)]
            means.append(entry["mean"])
            mins.append(entry["mean"] - entry["min"])
            maxs.append(entry["max"] - entry["mean"])
            colors.append(verdict_color[verdicts[(solver, mu)]])
        offset = (j - (n_solvers - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, width=bar_width * 0.9, color=colors,
                      edgecolor="black", linewidth=0.8, yerr=[mins, maxs], capsize=3,
                      error_kw=dict(linewidth=1.2, ecolor="#444444"))
        for xi, m, s in zip(x + offset, means, [solver] * len(MUS)):
            ax.text(xi, -1.3, s[:3], ha="center", va="top", fontsize=7, rotation=90, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels([f"μ = {m:g}" for m in MUS], fontsize=12, fontweight="bold")
    ax.set_ylabel("rise (orders of magnitude, mean of 3 direction-pair seeds)", fontsize=11)
    ax.set_title("Loss-landscape \"rise\" per (solver, μ) — bar height = mean, "
                 "error bars = seed-to-seed range\n"
                 "green = converged, orange = unstable — this is the reliable way to "
                 "compare flatness across cells (not eyeballing separate 3D plots)",
                 fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="black", label=v) for v, c in verdict_color.items()
               if v != "diverged"]  # never observed in this track -- omit from the legend
    ax.legend(handles=handles, loc="upper right", fontsize=10, title="verdict")

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    out_path = os.path.join(ROOT, "loss_landscape", "rise_comparison_bars.png")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
