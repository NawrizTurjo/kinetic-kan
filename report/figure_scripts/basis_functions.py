import sys
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMPL = os.path.join(ROOT, "implementation")
sys.path.insert(0, ROOT + r"\implementation")
from kan.basis import get_basis_function

G = 5
grid = torch.linspace(-1, 1, G)
h = 2 / (G - 1)
u = torch.linspace(-1, 1, 401).view(-1, 1)
names = [("rbf", "Gaussian RBF"), ("bspline", "Cubic B-spline"), ("rswaf", "RSWAF"),
         ("iqf", "IQF"), ("chebyshev", "Chebyshev"), ("lagrange", "Lagrange"),
         ("newton", "Newton")]
ink = "#1c2026"
acc = (155 / 255, 26 / 255, 32 / 255)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": "#9aa0a6",
                     "axes.linewidth": 0.6, "xtick.color": "#555", "ytick.color": "#555"})
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.4), constrained_layout=True)
for ax, (k, title) in zip(axes.flat, names):
    B = get_basis_function(k)(u, grid, h).squeeze(1).detach().numpy()
    n = B.shape[1]
    for i in range(n):
        ax.plot(u.squeeze().numpy(), B[:, i], lw=1.4, color=acc, alpha=0.35 + 0.65 * (i + 1) / n)
    for z in grid.numpy():
        ax.axvline(z, color="#d1d5db", lw=0.5, zorder=0)
    ax.set_title(title, fontsize=8.5, color=ink)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.tick_params(length=2, labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
ax = axes.flat[7]
ax.axis("off")
note = ("Normalised edge input\n" + r"$u=\tanh(x)\in[-1,1]$" + ",\n"
        + r"$G=5$" + " grid centres (grey lines).\n\n"
        "Local support: B-spline.\nGlobal support: every other basis.")
ax.text(0.02, 0.5, note, fontsize=7.5, color=ink, va="center")
fig.savefig(ROOT + r"\report\figures\methods\basis_functions.png", dpi=220)
print("saved")
