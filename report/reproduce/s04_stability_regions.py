"""
Linear stability of the integrators against the spectrum of the learned field.

  (a) boundaries |R(z)| = 1 of R(z) = 1 + z b^T (I - zA)^{-1} 1 for each tableau,
      overlaid with h*lambda for the eigenvalues of the true Jacobian and of the
      learned Jacobian (Tsit5/RBF checkpoint) along the orbit on [0, 14], h = 0.05;
  (b) per-step amplitude error | |R(i w h)| - 1 | on the imaginary axis, which
      shows that Heun and Midpoint share one R(z) and so cannot differ in damping.

Report: Figure "stability" (Section 3.1).
Output: figures/analysis/stability_regions.pdf
        numbers.json -> stability, linear_amplitude_change_per_period_h005
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from common import (C, LV, N_14, TW, Y28, jac_true, learned_jacobian, load_model,
                    panel_label, save, save_numbers)
from rk_theory import stability_function


def main():
    x = np.linspace(-4.6, 2.6, 330)
    y = np.linspace(-3.6, 3.6, 300)
    Z = x[None, :] + 1j * y[:, None]
    fig, axes = plt.subplots(1, 2, figsize=(TW, 3.05))
    ax = axes[0]
    specs = [("euler", C["crimson"], "Euler"), ("heun", C["amber"], "RK2 (Heun = Midpoint)"),
             ("rk4", C["teal"], "RK4"), ("dopri5", C["gray"], "DOPRI5"), ("tsit5", C["purple"], "Tsit5")]
    for name, col, lab in specs:
        R = np.abs(stability_function(name, Z))
        ax.contour(x, y, R, levels=[1.0], colors=[col], linewidths=1.2)
        ax.plot([], [], color=col, label=lab)
    ax.axhline(0, color=C["ink"], lw=0.4)
    ax.axvline(0, color=C["ink"], lw=0.4)
    # eigenvalues of h*J along the orbit, learned (Tsit5 model) and true
    m, _ = load_model("solver_tsit5", double=True)
    h = 0.05
    lam_l = np.concatenate([np.linalg.eigvals(learned_jacobian(m, u)) for u in Y28[:N_14]])
    lam_t = np.concatenate([np.linalg.eigvals(jac_true(u)) for u in Y28[:N_14]])
    ax.scatter((h * lam_t).real, (h * lam_t).imag, s=5, color=C["ink"], marker="o", lw=0, label="$h\\lambda$, true $J$")
    ax.scatter((h * lam_l).real, (h * lam_l).imag, s=6, color=C["blue"], marker="x", lw=0.6,
               label="$h\\lambda$, learned $J_\\theta$")
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[0], y[-1]); ax.set_aspect("equal")
    ax.set_xlabel("Re$(z)$"); ax.set_ylabel("Im$(z)$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6, handlelength=1.4, ncol=3, columnspacing=0.8)
    ax.set_title("stability boundaries $|R(z)|=1$", pad=3)
    ins = ax.inset_axes([0.745, 0.33, 0.25, 0.34])
    ins.scatter((h * lam_t).real, (h * lam_t).imag, s=3, color=C["ink"], lw=0)
    ins.scatter((h * lam_l).real, (h * lam_l).imag, s=4, color=C["blue"], marker="x", lw=0.5)
    ins.axhline(0, color=C["ink"], lw=0.3); ins.axvline(0, color=C["ink"], lw=0.3)
    ins.set_xlim(-0.3, 0.3); ins.set_ylim(-0.3, 0.3)
    ins.set_xticks([-0.2, 0, 0.2]); ins.set_yticks([-0.2, 0, 0.2])
    ins.tick_params(labelsize=5, length=1.5, pad=1)
    ins.set_title("zoom: $h\\lambda$ near 0", fontsize=5.5, pad=1)
    ins.grid(False)
    ins.set_facecolor("white")
    panel_label(ax, "a")
    stab = dict(max_abs_hlambda_learned=float(np.abs(h * lam_l).max()),
                max_abs_hlambda_true=float(np.abs(h * lam_t).max()),
                max_re_lambda_learned=float(lam_l.real.max()),
                max_re_lambda_true=float(lam_t.real.max()),
                max_abs_lambda_true=float(np.abs(lam_t).max()),
                max_abs_lambda_learned=float(np.abs(lam_l).max()))
    # amplification per step on the imaginary axis: |R(i w)| - 1
    ax = axes[1]
    w = np.logspace(-2, np.log10(2.5), 200)
    for name, col, lab in specs:
        R = np.abs(stability_function(name, 1j * w))
        dev = R - 1
        ax.loglog(w[dev > 0], dev[dev > 0], color=col, lw=1.2, label=lab + " (growth)")
        ax.loglog(w[dev < 0], -dev[dev < 0], color=col, lw=1.2, ls="--")
    omega = math.sqrt(LV["alpha"] * LV["gamma"])
    ax.axvline(h * omega, color=C["ink"], lw=0.7, ls=":")
    ax.text(h * omega * 1.08, 1e-13, "$h\\omega_0$", fontsize=6.5)
    ax.set_ylim(1e-14, 10)
    ax.set_xlabel("$\\omega h$  (purely oscillatory mode $z=i\\omega h$)")
    ax.set_ylabel("$|\\,|R(i\\omega h)|-1\\,|$")
    ax.set_title("per-step amplitude error (solid: growth, dashed: decay)", pad=3)
    ax.grid(True, which="major")
    panel_label(ax, "b")
    fig.tight_layout(w_pad=1.2)
    save(fig, "stability_regions")
    per_period = {}
    steps = (2 * math.pi / omega) / h
    for name, _, _ in specs:
        R = abs(stability_function(name, np.array([1j * omega * h]))[0])
        per_period[name] = float(R ** steps - 1)
    print("  amplitude change per period at h=0.05:", per_period)
    save_numbers(dict(stability=stab, linear_amplitude_change_per_period_h005=per_period))


if __name__ == "__main__":
    main()
