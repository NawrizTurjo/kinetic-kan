"""
Where in phase space is the learned vector field accurate?

The relative field error ||f_theta(u) - f(u)|| / mean ||f|| is evaluated on a
170 x 140 grid covering the orbit, for KAN-ODE and the SiLU MLP-ODE at 10k and
50k epochs. The training arc (thick) and the full orbit (thin) are overlaid.
On-orbit averages are also recorded for the training window and for (3.5, 14].

Report: Figure "field-error" and the field-error row of Table "kan-vs-mlp"
(Sections 2.4, 3.5, 9).
Output: figures/analysis/field_error_maps.pdf
        numbers.json -> field_error, field_scale
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from common import (C, EQ_TRUE, N_14, N_TRAIN, SEQ, TW, Y28, f_true_np, load_model, save,
                    save_numbers)


def main():
    lo = Y28.min(0) - 0.35
    hi = Y28.max(0) + 0.35
    xs = np.linspace(max(lo[0], 0.15), hi[0], 170)
    ys = np.linspace(max(lo[1], 0.15), hi[1], 140)
    X, Yg = np.meshgrid(xs, ys)
    U = np.stack([X, Yg], -1).reshape(-1, 2)
    f = f_true_np(U)
    fscale = float(np.mean(np.linalg.norm(f_true_np(Y28[:N_14]), axis=1)))
    panels = [("basis_rbf", "KAN-ODE, 10k epochs"), ("mlp_silu", "MLP-ODE (SiLU), 10k epochs"),
              ("kan_50k", "KAN-ODE, 50k epochs"), ("mlp_silu_50k", "MLP-ODE (SiLU), 50k epochs")]
    fig, axes = plt.subplots(1, 4, figsize=(TW, 2.05), sharey=True)
    norm = LogNorm(vmin=1e-3, vmax=1.0)
    stats = {}
    for ax, (key, title) in zip(axes, panels):
        m, _ = load_model(key, double=True)
        fl = m(torch.tensor(U)).numpy()
        err = (np.linalg.norm(fl - f, axis=1) / fscale).reshape(X.shape)
        im = ax.pcolormesh(X, Yg, np.clip(err, 1e-3, 1.0), cmap=SEQ, norm=norm, shading="auto", rasterized=True)
        ax.plot(Y28[:, 0], Y28[:, 1], color=C["blue"], lw=0.7)
        ax.plot(Y28[:N_TRAIN, 0], Y28[:N_TRAIN, 1], color=C["ink"], lw=2.0)
        ax.plot(*EQ_TRUE, marker="+", color=C["ink"], ms=6, mew=1.0)
        ax.set_title(title, pad=3, fontsize=7.5)
        ax.set_xlabel("prey $u_1$")
        ax.grid(False)
        on = np.linalg.norm(m(torch.tensor(Y28[:N_14])).numpy() - f_true_np(Y28[:N_14]), axis=1) / fscale
        stats[key] = dict(on_orbit_train=float(on[:N_TRAIN].mean()), on_orbit_extrap=float(on[N_TRAIN:].mean()),
                          box_median=float(np.median(err)), box_frac_below_1pct=float(np.mean(err < 0.01)))
        print(f"  {key:14s} on-orbit field error, (3.5,14]: {100 * stats[key]['on_orbit_extrap']:.2f}%")
    axes[0].set_ylabel("predator $u_2$")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label("$\\|\\mathbf{f}_\\theta-\\mathbf{f}\\|\\,/\\,\\overline{\\|\\mathbf{f}\\|}$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, "field_error_maps")
    save_numbers(dict(field_error=stats, field_scale=fscale))


if __name__ == "__main__":
    main()
