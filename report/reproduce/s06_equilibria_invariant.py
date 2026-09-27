"""
Equilibria, periods and first-integral drift of every learned vector field.

For each checkpoint:
  * Newton-Raphson on f_theta(u) = 0 started from the true equilibrium (3, 1.5),
    then the eigenvalues of the learned Jacobian there;
  * the orbit period from upward mean-crossings of the predicted prey signal on
    [0, 28], against the true period from a long DOP853 solve;
  * drift of the Lotka-Volterra first integral H along the prediction.
Also the drift of H when the TRUE field is integrated by each solver at h = 0.05,
which separates solver drift from learned-model drift.

Each row also gets a lightweight on-orbit field error (mean
||f_theta - f|| / mean ||f||, on the training window and on (3.5, 14]) so
that a model can be cited in a table without needing the full field-error
map of Figure "field-error" (s09_field_error_maps.py), which is fixed to the
four KAN-vs-MLP-at-two-budgets panels discussed in Section 3.5.

Report: Table "equilibria", the field/period/H rows of Tables "kan-vs-mlp"
and "phase4", and Figure "first-integral-drift" (Sections 2.4, 3.5 and 9).
Output: figures/analysis/first_integral_drift.pdf
        numbers.json -> equilibria, true_period, true_eig_im, H0, H_drift_true_field_solver
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import torch

from common import (BASES, C, DATA28, EQ_TRUE, LV, N_14, N_TRAIN, SOLVER_COLOR, SOLVER_LABEL, SOLVERS,
                    T28, TW, Y28, f_true_np, f_true_torch, first_integral, learned_jacobian, load_model,
                    panel_label, rollout, save, save_numbers)
from ode.solvers import odeint

KEYS = [f"solver_{s}" for s in SOLVERS] + [f"basis_{b}" for b in BASES] + \
       ["mlp_silu", "kan_50k", "euler_50k", "bspline_25k", "mlp_silu_50k",
        "mlp_tanh_exact", "mlp_tanh_exact_50k"]


def newton_equilibrium(model, u0, iters=30):
    u = torch.tensor(u0, dtype=torch.float64)
    trace = []
    for _ in range(iters):
        F = model(u[None])[0]
        trace.append(float(F.norm()))
        if trace[-1] < 1e-13:
            break
        J = torch.autograd.functional.jacobian(lambda v: model(v[None])[0], u)
        step = torch.linalg.lstsq(J, F[:, None]).solution[:, 0]
        u = u - step
        if not torch.isfinite(u).all():
            break
    return u.numpy(), trace


def orbit_period(t, y):
    """Mean period from upward crossings of the prey signal through its mean (linear interpolation)."""
    s = y[:, 0] - EQ_TRUE[0]
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    tc = t[idx] - s[idx] * (t[idx + 1] - t[idx]) / (s[idx + 1] - s[idx])
    return float(np.mean(np.diff(tc))) if len(tc) > 2 else float("nan"), tc


def main():
    # true period from a long high-accuracy solve
    tl = np.linspace(0, 60, 60001)
    yl = scipy.integrate.solve_ivp(lambda t, u: f_true_np(u), (0, 60), [1.0, 1.0], t_eval=tl,
                                   method="DOP853", rtol=1e-12, atol=1e-12).y.T
    T_true, _ = orbit_period(tl, yl)
    H0 = float(first_integral(Y28[0]))
    fscale = float(np.mean(np.linalg.norm(f_true_np(Y28[:N_14]), axis=1)))
    rows = {}
    for key in KEYS:
        m, c = load_model(key, double=True)
        u_star, trace = newton_equilibrium(m, EQ_TRUE)
        lam = np.linalg.eigvals(learned_jacobian(m, u_star))
        pr = rollout(key)
        T_hat, _ = orbit_period(T28, pr)
        H = first_integral(pr)
        on_orbit = np.linalg.norm(m(torch.tensor(Y28[:N_14])).numpy() - f_true_np(Y28[:N_14]), axis=1) / fscale
        rows[key] = dict(u_star=u_star.tolist(), dist=float(np.linalg.norm(u_star - EQ_TRUE)),
                         converged=bool(trace[-1] < 1e-10),
                         newton_residuals=trace, eig_re=float(lam.real.max()), eig_im=float(np.abs(lam.imag).max()),
                         period=T_hat, period_rel_err=float((T_hat - T_true) / T_true),
                         H_drift_max_rel=float(np.max(np.abs(H - H0)) / H0),
                         H_drift_end_rel=float((H[-1] - H0) / H0),
                         on_orbit_field_error_train=float(on_orbit[:N_TRAIN].mean()),
                         on_orbit_field_error_extrap=float(on_orbit[N_TRAIN:].mean()))
        print(f"  {key:16s} u*=({u_star[0]:.3f},{u_star[1]:.3f}) Re(lam)={lam.real.max():+.4f} "
              f"T={T_hat:.4f} dH/H={rows[key]['H_drift_end_rel']:+.2e} "
              f"field_err(extrap)={100 * rows[key]['on_orbit_field_error_extrap']:.2f}%")

    # pure solver drift of H on the TRUE field at the training discretisation
    drift = {}
    for name in SOLVERS:
        u = odeint(f_true_torch, DATA28.y0, DATA28.t_full, method=name, substeps=2).numpy()
        drift[name] = float((first_integral(u)[-1] - H0) / H0)

    # figure: H drift over time, true-field solver drift vs learned-model drift
    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.3), sharey=False)
    for name in SOLVERS:
        u = odeint(f_true_torch, DATA28.y0, DATA28.t_full, method=name, substeps=2).numpy()
        d = np.abs(first_integral(u) - H0) / H0
        axes[0].semilogy(T28[1:], d[1:] + 1e-17, color=SOLVER_COLOR[name], lw=1.0, label=SOLVER_LABEL[name])
        pr = rollout(f"solver_{name}")
        d2 = (first_integral(pr) - H0) / H0
        axes[1].plot(T28, d2, color=SOLVER_COLOR[name], lw=1.0, label=SOLVER_LABEL[name])
    axes[0].set_title("true field $\\mathbf{f}$ integrated by each solver ($h=0.05$)", pad=3)
    axes[0].set_ylabel("$|H(\\mathbf{u}_n)-H_0|/H_0$")
    axes[0].set_ylim(1e-16, 1e-1)
    axes[1].set_title("learned field $\\mathbf{f}_\\theta$, integrated by its training solver", pad=3)
    axes[1].set_ylabel("$(H(\\hat{\\mathbf{u}})-H_0)/H_0$")
    axes[1].axhline(0, color=C["ink"], lw=0.5)
    for ax in axes:
        ax.axvspan(0, 3.5, color=C["soft"], lw=0, zorder=0)
        ax.axvline(14, color=C["ink"], lw=0.6, ls=":")
        ax.set_xlabel("$t$")
        ax.set_xlim(0, 28)
    axes[0].legend(ncol=3, loc="lower right", fontsize=6, handlelength=1.4)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "first_integral_drift")
    save_numbers(dict(equilibria=rows, true_period=T_true, true_eig_im=math.sqrt(LV["alpha"] * LV["gamma"]),
                      H0=H0, H_drift_true_field_solver=drift))


if __name__ == "__main__":
    main()
