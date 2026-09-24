"""Re-plot fixed-run trajectories from saved checkpoints with correct state labels."""
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMPL = os.path.join(ROOT, "implementation")
sys.path.insert(0, ROOT + r"\implementation")
from kan import KAN
from ode import NeuralODE, ZeroSumField, VanishingDimField
from evaluate import _rebuild_data
from utils import compute_r2_score

RUNS = {
    "sir_gated": (r"\implementation\results\_fixed\sir_fixed_full\best_model.pt",
                  ["$S$ (susceptible)", "$I$ (infected)", "$R$ (recovered)"], "time (days)"),
    "pendulum_silu_win5": (r"\implementation\results\_fixed\pendulum_control_win5\best_model.pt",
                           [r"$\theta$ (rad)", r"$\omega$ (rad/s)"], "time (s)"),
}
COLORS = ["#9b1a20", "#1f4e79", "#5f6b2d"]
plt.rcParams.update({"font.family": "serif", "font.size": 9})

for name, (ck, labels, xlab) in RUNS.items():
    ckpt = torch.load(ROOT + ck, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    model = KAN(layers_hidden=c["layers_hidden"], grid_len=c["grid_len"],
                grid_lims=tuple(c.get("grid_lims", [-1.0, 1.0])), basis_func=c["basis_func"],
                normalizer=c["normalizer"], base_act=c["base_act"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    field = model
    if c.get("conserve_mode") == "projection":
        field = ZeroSumField(model)
    if c.get("vanish_dim") is not None:
        field = VanishingDimField(field, dim=int(c["vanish_dim"]))
    node = NeuralODE(func=field, method=c["solver"], substeps=c["substeps"])
    data, _ = _rebuild_data(c)
    ts = float(c.get("time_scale") or 1.0)
    with torch.no_grad():
        pred = node(y0=data.y0, t=data.t_full / ts).numpy()
    t = data.t_full.numpy()
    y = data.y_full.numpy()
    n = len(data.t_train)
    print(name, "extrap R2 =", round(compute_r2_score(y[n:], pred[n:]), 5))

    fig, ax = plt.subplots(figsize=(6.4, 2.9), constrained_layout=True)
    ax.axvspan(t[0], data.t_split, color="#f3f4f6", zorder=0)
    ax.text(data.t_split, ax.get_ylim()[1], "", ha="left")
    for k, lab in enumerate(labels):
        ax.plot(t, y[:, k], color=COLORS[k], lw=1.8, label=f"{lab}, true")
        ax.plot(t, pred[:, k], color=COLORS[k], lw=1.4, ls=(0, (4, 2.5)), label=f"{lab}, predicted")
    ax.axvline(data.t_split, color="#374151", lw=0.8, ls=":")
    ax.set_xlabel(xlab)
    ax.set_xlim(t[0], t[-1])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ymax = ax.get_ylim()[1]
    ax.text(data.t_split * 0.5, ymax * 0.97, "training window", ha="center", va="top", fontsize=8, color="#374151")
    ax.text(data.t_split + (t[-1] - data.t_split) * 0.5, ymax * 0.97, "extrapolation", ha="center", va="top",
            fontsize=8, color="#374151")
    ax.legend(fontsize=7, ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.savefig(ROOT + rf"\report\figures\phase2\{name}_trajectory.png", dpi=220)
    plt.close(fig)
