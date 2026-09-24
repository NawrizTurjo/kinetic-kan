"""
The seven edge bases, their conditioning, and how much of the grid a trained model uses.

  * figure: the G = 5 basis functions of each family on the normalised input
    u in [-1, 1] (one highlighted), with the 2-norm condition number kappa_2 of
    the 400 x G design matrix Psi, and kappa_2 against grid size G = 3..12;
  * knot utilisation: for the RBF and B-spline checkpoints, the range of tanh(x)
    each layer actually sees on the orbit and the share of basis mass per knot.

The basis functions are imported from implementation/kan/basis.py, so the
figure shows the code that was trained, not a re-implementation.

Report: Figure "basis-catalog" and the kappa_2 column of Table "basis-ablation" (Section 3.3).
Output: figures/analysis/basis_catalog.pdf
        numbers.json -> basis_kappa_G5, basis_kappa_vs_G, knot_utilisation
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import BASES, BASIS_LABEL, C, N_14, TW, Y28, load_model, save, save_numbers
from kan.basis import bspline_basis, chebyshev_basis, iqf, lagrange_basis, newton_basis, rbf, rswaf

BASIS_FN = dict(rbf=rbf, bspline=bspline_basis, chebyshev=chebyshev_basis, lagrange=lagrange_basis,
                newton=newton_basis, iqf=iqf, rswaf=rswaf)


def design_matrix(name, G, n=400):
    u = torch.linspace(-1, 1, n, dtype=torch.float64)
    grid = torch.linspace(-1, 1, G, dtype=torch.float64)
    h = 2.0 / (G - 1)
    return u.numpy(), BASIS_FN[name](u[:, None], grid, h)[:, 0, :].numpy()


def fig_basis_catalog():
    fig, axes = plt.subplots(2, 4, figsize=(TW, 3.0))
    kappa = {}
    for ax, name in zip(axes.flat, BASES):
        u, Psi = design_matrix(name, 5)
        kappa[name] = float(np.linalg.cond(Psi))
        for g in range(5):
            hl = g == 3
            ax.plot(u, Psi[:, g], color=C["crimson"] if hl else C["ink"],
                    lw=1.5 if hl else 0.8, alpha=1.0 if hl else 0.55, zorder=3 if hl else 2)
        ax.axvspan(np.tanh(0.30), 1.0, color=C["blue"], alpha=0.07, lw=0)
        ax.set_title(f"{BASIS_LABEL[name]}   $\\kappa_2(\\Psi)={kappa[name]:.1f}$", pad=3)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        for z in np.linspace(-1, 1, 5):
            ax.axvline(z, color=C["rule"], lw=0.5, zorder=0)
    # last panel: kappa versus grid size
    ax = axes.flat[7]
    Gs = np.arange(3, 13)
    styles = dict(bspline=(C["crimson"], "-"), rbf=(C["blue"], "-"), chebyshev=(C["amber"], "-"),
                  lagrange=(C["teal"], "-"), newton=(C["purple"], "-"), iqf=(C["gray"], "--"),
                  rswaf=(C["gray"], ":"))
    kG = {}
    for name in BASES:
        ks = [np.linalg.cond(design_matrix(name, G)[1]) for G in Gs]
        kG[name] = [float(k) for k in ks]
        col, ls = styles[name]
        ax.semilogy(Gs, ks, color=col, ls=ls, lw=1.1, marker="o", ms=2.2)
    for name, dy in [("newton", 1.0), ("lagrange", 0.55), ("rbf", 1.9), ("bspline", 0.62), ("chebyshev", 1.6)]:
        ax.annotate(BASIS_LABEL[name].replace("Gaussian ", ""), (Gs[-1], kG[name][-1] * dy),
                    xytext=(2, 0), textcoords="offset points", fontsize=6, va="center",
                    color=C["ink"])
    ax.set_xlim(3, 15.8)
    ax.set_xticks([3, 6, 9, 12])
    ax.set_title("$\\kappa_2(\\Psi)$ vs. grid size $G$", pad=3)
    ax.grid(True, which="major")
    for ax in list(axes[1, :3]):
        ax.set_xlabel("normalised input $u$")
    axes[1, 3].set_xlabel("grid size $G$")
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    save(fig, "basis_catalog")
    print("  kappa_2 at G=5:", {k: round(v, 1) for k, v in kappa.items()})
    return dict(basis_kappa_G5=kappa, basis_kappa_vs_G={"G": Gs.tolist(), **kG})


def knot_utilisation():
    """How much of each layer's grid the trained network's inputs actually reach."""
    out = {}
    for key in ["basis_rbf", "basis_bspline"]:
        m, _ = load_model(key, double=True)
        x = torch.tensor(Y28[:N_14])
        per_layer = []
        for layer in m.layers:
            un = torch.tanh(x)
            psi = layer.basis_func(un, layer.grid, layer.denominator)  # [B, I, G]
            mass = psi.abs().mean(dim=(0, 1)).numpy()
            per_layer.append(dict(u_min=float(un.min()), u_max=float(un.max()),
                                  mean_abs_basis=(mass / mass.sum()).tolist()))
            x = layer(x)
        out[key] = per_layer
    return dict(knot_utilisation=out)


def main():
    num = fig_basis_catalog()
    num.update(knot_utilisation())
    save_numbers(num)


if __name__ == "__main__":
    main()
