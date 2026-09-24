"""
Work-precision study of the six fixed-step solvers on the TRUE Lotka-Volterra field.

This checks the solver implementations themselves, with no learning involved:
  (a) global error at t = 3.5 against a DOP853 reference (rtol = atol = 1e-13)
      for h = 0.4 / 2^k, k = 0..7, and the observed order from a log-log fit
      over the asymptotic range (h <= 0.1, 1e-11 < error < 1e-1);
  (b) the same error against the number of function evaluations;
  (c) each solver's own truncation error at the training discretisation
      (h = 0.05), scored exactly like a learned model (the last column of the
      solver-ablation table).

Report: Figure "work-precision" and Table "solver-ablation" (Section 3.1-3.2).
Output: figures/analysis/work_precision.pdf
        numbers.json -> work_precision, solver_truncation_true_field
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import torch

from common import (C, DATA28, N_14, N_TRAIN, SOLVER_COLOR, SOLVER_LABEL, SOLVERS, TW, Y28,
                    f_true_np, f_true_torch, metrics, panel_label, save, save_numbers)
from ode.solvers import odeint
from rk_theory import tableau


def main():
    T = 3.5
    ref = scipy.integrate.solve_ivp(lambda t, u: f_true_np(u), (0, T), [1.0, 1.0], method="DOP853",
                                    rtol=1e-13, atol=1e-13).y[:, -1]
    hs = 0.4 / 2 ** np.arange(0, 8)
    out = {}
    for name in SOLVERS:
        errs, nfes = [], []
        s = len(tableau(name)[1])
        for h in hs:
            n = int(round(T / h))
            t = torch.linspace(0, T, n + 1, dtype=torch.float64)
            u = odeint(f_true_torch, torch.tensor([1.0, 1.0], dtype=torch.float64), t, method=name)[-1].numpy()
            errs.append(float(np.max(np.abs(u - ref))))
            nfes.append(s * n)
        errs = np.array(errs)
        ok = (errs > 1e-11) & (errs < 1e-1) & (hs <= 0.1 + 1e-12)
        slope = float(np.polyfit(np.log(hs[ok]), np.log(errs[ok]), 1)[0])
        out[name] = dict(h=hs.tolist(), err=errs.tolist(), nfe=nfes, observed_order=slope)
        print(f"  {name:8s} observed order {slope:.2f}")

    # the solver's own truncation error at the training discretisation (h = dt/substeps = 0.05)
    trunc = {}
    t = DATA28.t_full[:N_14]
    for name in SOLVERS:
        u = odeint(f_true_torch, DATA28.y0, t, method=name, substeps=2).numpy()
        trunc[name] = dict(extrap_mse=float(np.mean((u[N_TRAIN:] - Y28[N_TRAIN:N_14]) ** 2)),
                           learned_extrap_mse=metrics(f"solver_{name}")["best"]["extrap_mse"])
        print(f"  {name:8s} true-field truncation MSE {trunc[name]['extrap_mse']:.2e}")

    marks = dict(euler="o", heun="s", midpoint="D", rk4="^", dopri5="v", tsit5="P")
    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.45))
    for name in SOLVERS:
        d = out[name]
        lab = f"{SOLVER_LABEL[name]} ($\\hat p={d['observed_order']:.2f}$)"
        axes[0].loglog(d["h"], d["err"], marker=marks[name], ms=3.2, color=SOLVER_COLOR[name], lw=1.0, label=lab)
        axes[1].loglog(d["nfe"], d["err"], marker=marks[name], ms=3.2, color=SOLVER_COLOR[name], lw=1.0,
                       label=SOLVER_LABEL[name])
    for ax in axes:
        ax.axhline(1e-12, color=C["gray"], lw=0.5, ls=":")
        ax.set_ylim(1e-14, 1e6)
        ax.set_ylabel(r"global error $\|\mathbf{u}_h(3.5)-\mathbf{u}(3.5)\|_\infty$")
        ax.grid(True, which="major")
    axes[0].axvline(0.05, color=C["ink"], lw=0.7, ls="--")
    axes[0].text(0.05, 3e-15, " training $h$", fontsize=6.5, va="bottom", ha="left")
    axes[0].set_xlabel("step size $h$")
    axes[0].legend(loc="upper left", fontsize=6.0, ncol=2, handlelength=1.6, columnspacing=0.8)
    axes[1].set_xlabel("function evaluations over $[0, 3.5]$")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "work_precision")
    save_numbers(dict(work_precision=out, solver_truncation_true_field=trunc))


if __name__ == "__main__":
    main()
