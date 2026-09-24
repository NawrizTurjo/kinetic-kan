"""Redraw hybrid-basis gate traces with report terminology (no internal run names)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMPL = os.path.join(ROOT, "implementation")
RES = ROOT + r"\implementation\results\phase3\hybrid_basis"
OUT = ROOT + r"\report\figures\track_c"
RUNS = {
    "lv_full": ("lv_full_alpha_beta.png", "Lotka-Volterra, 10,000 epochs"),
    "pendulum_full": ("pendulum_full_alpha_beta.png", "Pendulum, 10,000 epochs (gate learning rate 15x)"),
    "probe_pendulum": ("hybrid_probe_pendulum_alpha_beta.png", "Pendulum, 2,000-epoch probe (gate learning rate 1x)"),
    "pendulum_3k": ("hybrid_pendulum_3k_alpha_beta.png", "Pendulum, 3,000 epochs (gate learning rate 15x)"),
}
plt.rcParams.update({"font.family": "serif", "font.size": 9})
for run, (fname, title) in RUNS.items():
    h = json.load(open(RES + rf"\{run}\training_history.json"))
    a, b = h["alpha"], h["beta"]
    ep = range(1, len(a) + 1)
    fig, ax = plt.subplots(figsize=(4.6, 2.8), constrained_layout=True)
    ax.plot(ep, b, color="#1f4e79", lw=1.8, label=r"$\beta$ (RBF weight)")
    ax.plot(ep, a, color="#9b1a20", lw=1.8, label=r"$\alpha$ (B-spline weight)")
    ax.axhline(0.5, color="#9aa0a6", lw=0.7, ls=":")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(1, len(a))
    ax.set_xlabel("epoch")
    ax.set_ylabel("gate weight")
    ax.set_title(title, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(len(a), b[-1], f"  {b[-1]:.3f}", va="center", fontsize=7.5, color="#1f4e79", clip_on=False)
    ax.text(len(a), a[-1], f"  {a[-1]:.3f}", va="center", fontsize=7.5, color="#9b1a20", clip_on=False)
    ax.legend(fontsize=7.5, frameon=False, loc="center right")
    fig.savefig(OUT + "\\" + fname, dpi=220)
    plt.close(fig)
    print(run, len(a), "final a/b", round(a[-1], 4), round(b[-1], 4))
