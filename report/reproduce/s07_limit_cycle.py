"""
Centre or limit cycle? Integrate learned fields from several initial conditions.

The true Lotka-Volterra system has a centre: every initial condition stays on
its own closed orbit (grey). Each learned field is integrated with Tsit5
(h = 0.05) from four initial conditions for t in [0, 40]; if all of them
converge to one closed curve, the model has learned an attracting limit cycle.
The red cross marks the learned equilibrium found by Newton's method.

Report: Figure "limit-cycle" (Section 3.5).
Output: figures/analysis/limit_cycle_test.pdf
        numbers.json -> limit_cycle_test
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (C, EQ_TRUE, TW, f_true_torch, first_integral, load_model, save, save_numbers)
from ode import NeuralODE
from ode.solvers import odeint
from s06_equilibria_invariant import newton_equilibrium


def main():
    ics = [(1.0, 1.0), (2.0, 1.2), (4.0, 2.2), (0.6, 0.6)]
    ic_col = [C["ink"], C["blue"], C["amber"], C["teal"]]
    t = torch.linspace(0, 40, 401, dtype=torch.float64)
    panels = [("basis_rbf", "KAN-ODE, 10k"), ("kan_50k", "KAN-ODE, 50k"),
              ("mlp_silu", "MLP-ODE (SiLU), 10k"), ("mlp_silu_50k", "MLP-ODE (SiLU), 50k")]
    fig, axes = plt.subplots(1, 4, figsize=(TW, 1.85), sharex=True, sharey=True)
    out = {}
    for ax, (key, title) in zip(axes, panels):
        m, _ = load_model(key, double=True)
        node = NeuralODE(m, method="tsit5", substeps=2)
        Hs = []
        for ic, col in zip(ics, ic_col):
            y0 = torch.tensor(ic, dtype=torch.float64)
            tr = odeint(f_true_torch, y0, t, method="tsit5", substeps=2).numpy()
            pr = node(y0, t).numpy()
            ax.plot(tr[:, 0], tr[:, 1], color=C["gray"], lw=1.8, alpha=0.4)
            ok = np.all(np.isfinite(pr), axis=1) & (np.abs(pr).max(axis=1) < 50)
            pr = pr[ok]
            ax.plot(pr[:, 0], pr[:, 1], color=col, lw=0.7)
            ax.plot(*ic, "o", color=col, ms=2.5)
            late = pr[-100:] if len(pr) > 100 else pr
            Hs.append(dict(ic=list(ic), H_true=float(first_integral(np.array(ic))),
                           H_learned_late_mean=float(first_integral(late).mean()) if np.all(late > 0) else None,
                           escaped=bool(len(pr) < 401)))
        u_star, _ = newton_equilibrium(m, EQ_TRUE)
        ax.plot(*u_star, marker="x", color=C["crimson"], ms=5, mew=1.2)
        ax.plot(*EQ_TRUE, marker="+", color=C["ink"], ms=6, mew=1.0)
        ax.set_title(title, pad=2, fontsize=7.5)
        ax.set_xlabel("$u_1$")
        ax.set_xlim(0, 9); ax.set_ylim(0, 6.5)
        out[key] = Hs
        print(f"  {key:14s} late-time H per initial condition:",
              [None if h["H_learned_late_mean"] is None else round(h["H_learned_late_mean"], 3) for h in Hs])
    axes[0].set_ylabel("$u_2$")
    fig.tight_layout(w_pad=0.3)
    save(fig, "limit_cycle_test")
    save_numbers(dict(limit_cycle_test=out))


if __name__ == "__main__":
    main()
