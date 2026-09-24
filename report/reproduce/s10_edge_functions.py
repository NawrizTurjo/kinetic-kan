"""
The learned first-layer edge functions phi_{i,j} of the RBF and B-spline models.

Each input is swept over the range it takes on the true orbit while the other
input is held at its orbit mean; the ten edge outputs (spline part + SiLU base
part, from KANLayer.get_activations) are plotted, the widest one highlighted.

Report: Figure "edge-functions" (Section 3.5).
Output: figures/analysis/edge_functions.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import C, TW, Y28, load_model, save


def main():
    fig, axes = plt.subplots(1, 4, figsize=(TW, 1.95), sharey=False)
    xr = [np.linspace(Y28[:, i].min(), Y28[:, i].max(), 300) for i in range(2)]
    col = 0
    for key, name in [("basis_rbf", "RBF"), ("basis_bspline", "B-spline")]:
        m, _ = load_model(key, double=True)
        layer = m.layers[0]
        for i, var in enumerate(["u_1", "u_2"]):
            ax = axes[col]
            X = np.zeros((300, 2)); X[:, i] = xr[i]; X[:, 1 - i] = Y28[:, 1 - i].mean()
            sp, ba = layer.get_activations(torch.tensor(X))
            phi = (sp + ba)[:, i, :].numpy()  # [300, 10]
            spread = phi.max(0) - phi.min(0)
            top = int(np.argmax(spread))
            for j in range(10):
                ax.plot(xr[i], phi[:, j], color=C["crimson"] if j == top else C["ink"],
                        lw=1.4 if j == top else 0.7, alpha=1 if j == top else 0.45)
            ax.set_title(f"{name}: $\\phi_{{{i+1},j}}({var})$", pad=3)
            ax.set_xlabel(f"${var}$ (range seen on orbit)")
            col += 1
    axes[0].set_ylabel("edge output")
    fig.tight_layout(w_pad=0.7)
    save(fig, "edge_functions")


if __name__ == "__main__":
    main()
