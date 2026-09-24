"""Redraw the loss-landscape rise bar chart with report terminology."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMPL = os.path.join(ROOT, "implementation")
SRC = ROOT + r"\implementation\results\phase3\stiffness_map\loss_landscape\rise_multiseed.json"
OUT = ROOT + r"\report\figures\track_d\rise_comparison_bars.png"

data = json.load(open(SRC))
SOLVERS = ["euler", "midpoint", "rk4", "tsit5"]
SHORT = {"euler": "Euler", "midpoint": "Mid", "rk4": "RK4", "tsit5": "Tsit5"}
MUS = ["0.1", "0.5", "1.0", "2.0", "5.0", "8.0"]
# Seed-42 verdicts (Table: original 24-cell sweep)
NOT_CONV = {("0.1", s) for s in SOLVERS} | {("2.0", s) for s in SOLVERS} | {("1.0", "euler")}
C_OK, C_BAD = "#2f6f4e", "#c7792b"

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(8.6, 3.6), constrained_layout=True)
w = 0.2
for i, mu in enumerate(MUS):
    for j, s in enumerate(SOLVERS):
        cell = data[s][mu]
        x = i + (j - 1.5) * w
        mean, lo, hi = cell["mean"], cell["min"], cell["max"]
        color = C_BAD if (mu, s) in NOT_CONV else C_OK
        ax.bar(x, mean, width=w * 0.9, color=color, edgecolor="white", linewidth=0.6)
        ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor="#374151",
                    elinewidth=0.8, capsize=2)
        ax.text(x, -0.9, SHORT[s], ha="center", va="top", fontsize=6.5, rotation=90, color="#555")
ax.set_xticks(range(len(MUS)))
ax.set_xticklabels([rf"$\mu={float(m):g}$" for m in MUS])
ax.tick_params(axis="x", pad=26, length=0)
ax.set_ylabel("rise (decades)")
ax.set_ylim(0, None)
ax.grid(axis="y", color="#e5e7eb", lw=0.6)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_title("Loss-landscape rise per (solver, $\\mu$): mean over three random direction-pairs, "
             "bars show the range across direction-pairs", fontsize=8.5)
ax.legend(handles=[Patch(color=C_OK, label="converged"), Patch(color=C_BAD, label="not converged")],
          title="verdict (seed 42)", fontsize=7.5, title_fontsize=7.5, frameon=False, loc="upper right")
fig.savefig(OUT, dpi=220)
print("saved")
