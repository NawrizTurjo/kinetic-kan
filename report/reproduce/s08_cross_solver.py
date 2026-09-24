"""
Cross-solver transfer and the Euler modified equation.

  * transfer matrix: each solver-ablation checkpoint (rows: solver used in
    training) is integrated with every solver (columns) at h = 0.05 and scored
    on (3.5, 14]. The diagonal reproduces the solver-ablation table.
  * backward error analysis: Euler applied to a field g follows
    g - (h/2) g'g + O(h^2), so to match the data an Euler-trained network must
    learn the INVERSE modified field g = f + (h/2) J f. The script measures the
    learned correction f_theta - f against that prediction on the training
    window (cosine similarity and magnitude ratio).

Report: Figure "cross-solver" and the Euler modified-equation paragraph (Section 3.2).
Output: figures/analysis/cross_solver_transfer.pdf
        numbers.json -> cross_solver_extrap_mse, euler_modified_equation
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from common import (C, N_14, N_TRAIN, SEQ, SOLVER_LABEL, SOLVERS, Y28, f_true_np, jac_true,
                    load_model, rollout, save, save_numbers)


def main():
    M = np.zeros((6, 6))
    for i, trained in enumerate(SOLVERS):
        for j, used in enumerate(SOLVERS):
            pr = rollout(f"solver_{trained}", solver=used, t_end=14.0)
            M[i, j] = np.mean((pr[N_TRAIN:] - Y28[N_TRAIN:N_14]) ** 2)

    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    im = ax.imshow(M, cmap=SEQ, norm=LogNorm(vmin=M.min() * 0.8, vmax=M.max() * 1.2))
    lab = [SOLVER_LABEL[s] for s in SOLVERS]
    ax.set_xticks(range(6), lab, rotation=35, ha="right")
    ax.set_yticks(range(6), lab)
    ax.set_xlabel("solver used at evaluation")
    ax.set_ylabel("solver used in training")
    ax.grid(False)
    lnorm = LogNorm(vmin=M.min(), vmax=M.max())
    for i in range(6):
        for j in range(6):
            v = M[i, j]
            e = int(math.floor(math.log10(v)))
            txt = f"{v / 10 ** e:.1f}e{e}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.6,
                    color="white" if lnorm(v) > 0.55 else C["ink"],
                    fontweight="bold" if i == j else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("extrapolation MSE on $(3.5, 14]$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save(fig, "cross_solver_transfer")

    # Euler backward error analysis (see module docstring)
    m, _ = load_model("solver_euler", double=True)
    mt, _ = load_model("solver_tsit5", double=True)
    h = 0.05
    U = Y28[:N_TRAIN]
    fl = m(torch.tensor(U)).numpy()
    ft = mt(torch.tensor(U)).numpy()
    f = f_true_np(U)
    finv = np.stack([f[k] + 0.5 * h * jac_true(U[k]) @ f[k] for k in range(len(U))])
    corr = fl - f
    pred = finv - f
    euler = dict(
        rel_err_to_true=float(np.linalg.norm(fl - f) / np.linalg.norm(f)),
        rel_err_to_inverse_modified=float(np.linalg.norm(fl - finv) / np.linalg.norm(f)),
        predicted_correction_rel_size=float(np.linalg.norm(pred) / np.linalg.norm(f)),
        tsit5_rel_err_to_true=float(np.linalg.norm(ft - f) / np.linalg.norm(f)),
        cosine_learned_vs_predicted_correction=float((corr * pred).sum() / (np.linalg.norm(corr) * np.linalg.norm(pred))),
        correction_ratio=float(np.linalg.norm(corr) / np.linalg.norm(pred)))
    print("  euler modified equation:", {k: round(v, 4) for k, v in euler.items()})
    save_numbers(dict(cross_solver_extrap_mse=dict(rows_trained=SOLVERS, cols_integrated=SOLVERS, M=M.tolist()),
                      euler_modified_equation=euler))


if __name__ == "__main__":
    main()
