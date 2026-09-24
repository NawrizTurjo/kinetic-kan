"""
Pointwise prediction error ||u_hat(t) - u(t)|| over [0, 28].

Each checkpoint is rolled out with its training solver from u(0), with no
retraining, twice as far as the scored horizon. Panel (a) compares solvers and
bases at 10k epochs; panel (b) compares KAN-ODE and MLP-ODE at 10k and 50k.
The slope of a linear fit to the running-max error envelope on (3.5, 28] is
recorded as the error growth rate.

Report: Figure "error-vs-time" (Section 3.5) and the far-horizon numbers in Section 9.
Output: figures/analysis/error_vs_time.pdf
        numbers.json -> error_growth
"""

import matplotlib.pyplot as plt
import numpy as np

from common import C, N_14, N_TRAIN, T28, TW, Y28, panel_label, rollout, save, save_numbers


def main():
    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.3), sharey=True)
    groups = [
        [("solver_euler", "Euler", C["crimson"]), ("solver_heun", "Heun", C["amber"]),
         ("solver_midpoint", "Midpoint", C["blue"]), ("solver_tsit5", "Tsit5", C["purple"]),
         ("basis_chebyshev", "Tsit5 + Chebyshev", C["teal"])],
        [("basis_rbf", "KAN-ODE 10k", C["blue"]), ("kan_50k", "KAN-ODE 50k", C["purple"]),
         ("mlp_silu", "MLP-ODE 10k", C["crimson"]), ("mlp_silu_50k", "MLP-ODE 50k", C["amber"])],
    ]
    growth = {}
    for ax, grp in zip(axes, groups):
        for key, lab, col in grp:
            e = np.linalg.norm(rollout(key) - Y28, axis=1)
            ax.semilogy(T28, e + 1e-8, color=col, lw=0.9, label=lab)
            # envelope growth rate over the extrapolation window (fit to running max)
            env = np.maximum.accumulate(e[N_TRAIN:])
            k = np.polyfit(T28[N_TRAIN:], env, 1)[0]
            growth[key] = dict(err_t14=float(e[N_14 - 1]), err_t28=float(e[-1]),
                               max_err_0_14=float(e[:N_14].max()), max_err_14_28=float(e[N_14:].max()),
                               envelope_slope=float(k))
            print(f"  {key:16s} max error on (14,28]: {growth[key]['max_err_14_28']:.2e}")
        ax.axvspan(0, 3.5, color=C["soft"], lw=0, zorder=0)
        ax.axvline(14, color=C["ink"], lw=0.6, ls=":")
        ax.set_xlim(0, 28)
        ax.set_ylim(1e-4, 30)
        ax.set_xlabel("$t$")
        ax.legend(fontsize=6, loc="upper left", ncol=3, bbox_to_anchor=(0.14, 1.0), columnspacing=0.8, handlelength=1.4)
        ax.grid(True, which="major")
    axes[0].text(1.75, 1.5e-4, "train", ha="center", fontsize=6.5)
    axes[0].text(14.3, 1.5e-4, "end of scored horizon", fontsize=6, ha="left")
    axes[0].set_ylabel("pointwise error $\\|\\hat{\\mathbf{u}}(t)-\\mathbf{u}(t)\\|_2$")
    axes[0].set_title("solver and basis choice (10k epochs)", pad=3)
    axes[1].set_title("architecture and training budget (Tsit5)", pad=3)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.tight_layout(w_pad=0.8)
    save(fig, "error_vs_time")
    save_numbers(dict(error_growth=growth))


if __name__ == "__main__":
    main()
