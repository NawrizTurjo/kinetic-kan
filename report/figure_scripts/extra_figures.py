"""Two report figures built from saved outputs.

1. closeout/phase_space_kan_vs_mlp.png: the three Lotka-Volterra models' saved
   phase-space plots side by side (KAN-ODE, SiLU MLP, the paper's tanh MLP).
2. phase4/epoch_budget_loss.png: training loss over 50,000 epochs for the KAN-ODE
   and the parameter-matched MLP (plus Euler and the paper's MLP), from the saved
   training histories of the extended-budget runs.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "implementation", "results")
FIG = os.path.join(ROOT, "report", "figures")

# ---- 1. phase-space triptych ------------------------------------------------------
runs = ["kanode_flagship", "mlpode_baseline_silu", "mlpode_baseline"]
ims = [Image.open(os.path.join(RES, "benchmarks", r, "phase_space.png")).convert("RGB") for r in runs]
w, h = ims[0].size
strip = Image.new("RGB", (w * len(ims), h), "white")
for i, im in enumerate(ims):
    strip.paste(im, (i * w, 0))
os.makedirs(os.path.join(FIG, "closeout"), exist_ok=True)
strip.save(os.path.join(FIG, "closeout", "phase_space_kan_vs_mlp.png"))

# ---- 2. 50k-epoch training loss ---------------------------------------------------
series = [
    ("tsit5_rbf_50k", "KAN-ODE (RBF, Tsit5)", "#1f4e79", "-"),
    ("mlp_silu_50k", "MLP-ODE (SiLU, parameter-matched)", "#9b1a20", "-"),
    ("euler_50k", "KAN-ODE (RBF, Euler)", "#5f6b2d", "--"),
    ("mlp_paperspec_50k", "MLP-ODE (tanh, paper specification)", "#6b7280", ":"),
]
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(7.0, 3.3), constrained_layout=True)
for run, label, color, ls in series:
    h = json.load(open(os.path.join(RES, "phase4", "epoch_budget_check", run, "training_history.json")))
    tr = np.array(h["train_losses"])
    best = np.minimum.accumulate(tr)  # running best: the quantity model selection uses
    ax.semilogy(np.arange(1, len(best) + 1), best, color=color, ls=ls, lw=1.6, label=label)
ax.axvline(10000, color="#9aa0a6", lw=0.8, ls=":")
ax.text(10700, 2e-6, "original 10,000-epoch budget", fontsize=7.5, color="#555", rotation=90, va="bottom")
ax.set_xlabel("epoch")
ax.set_ylabel("best training MSE so far (log scale)")
ax.set_xlim(1, 50000)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.6, which="major")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
os.makedirs(os.path.join(FIG, "phase4"), exist_ok=True)
fig.savefig(os.path.join(FIG, "phase4", "epoch_budget_loss.png"), dpi=220)
print("done")
