"""
Observation-noise and step-size sweeps.

  * noise: extrapolation and training-window MSE (both against the clean truth)
    of the sigma = 0.01, 0.05, 0.1 runs, with power-law fits MSE = c sigma^k;
  * step size: the dt = 0.05, 0.1, 0.2 runs, plus the Tsit5 truncation error of
    the TRUE field at each dt (h = dt/2) against a DOP853 reference, which shows
    the step-size effect is about data density rather than solver accuracy.
    The step-size figure is the team's original
    results/figures/02_error_vs_stepsize_loglog.png, so only numbers are made here.

Report: Figure "noise-dt" (right panel) and Table "noise-dt" (Section 3.4).
Output: figures/analysis/noise_sensitivity.pdf
        numbers.json -> noise_fit, dt_sweep
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import torch

from common import C, f_true_np, f_true_torch, metrics, save, save_numbers
from ode.solvers import odeint


def noise():
    sig = np.array([0.01, 0.05, 0.1])
    ex = np.array([metrics(f"noise_{s}")["best"]["extrap_mse"] for s in ["0.01", "0.05", "0.1"]])
    trn = np.array([metrics(f"noise_{s}")["best"]["train_mse"] for s in ["0.01", "0.05", "0.1"]])
    k, logc = np.polyfit(np.log(sig), np.log(ex), 1)
    k2, logc2 = np.polyfit(np.log(sig), np.log(trn), 1)
    fit = dict(extrap_slope=float(k), extrap_prefactor=float(np.exp(logc)),
               train_slope=float(k2), train_prefactor=float(np.exp(logc2)))
    print(f"  extrapolation MSE ~ {fit['extrap_prefactor']:.1f} sigma^{k:.2f}; "
          f"training MSE ~ {fit['train_prefactor']:.2f} sigma^{k2:.2f}")
    clean = metrics("noise_0")["best"]
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    ss = np.logspace(-2.1, -0.95, 50)
    ax.loglog(sig, ex, "o", color=C["crimson"], ms=4, label="extrapolation MSE")
    ax.loglog(ss, np.exp(logc) * ss ** k, color=C["crimson"], lw=0.8, ls="--",
              label=f"fit $\\propto\\sigma^{{{k:.2f}}}$")
    ax.loglog(sig, trn, "s", color=C["blue"], ms=4, label="training-window MSE (vs. clean truth)")
    ax.loglog(ss, np.exp(logc2) * ss ** k2, color=C["blue"], lw=0.8, ls="--",
              label=f"fit $\\propto\\sigma^{{{k2:.2f}}}$")
    ax.loglog(ss, ss ** 2, color=C["gray"], lw=0.8, ls=":", label="$\\sigma^2$ (noise variance)")
    ax.axhline(clean["extrap_mse"], color=C["crimson"], lw=0.5, ls=":")
    ax.text(0.052, clean["extrap_mse"] * 0.55, "noise-free extrapolation MSE", fontsize=6, color=C["ink"])
    ax.set_xlabel("observation-noise standard deviation $\\sigma$")
    ax.set_ylabel("MSE")
    ax.set_ylim(3e-5, 1.0)
    ax.legend(fontsize=5.6, loc="upper left", handlelength=1.6)
    ax.grid(True, which="major")
    fig.tight_layout()
    save(fig, "noise_sensitivity")
    return fit


def stepsize():
    dts = [0.05, 0.1, 0.2]
    ex_dt = [metrics(f"dt_{d}")["best"]["extrap_mse"] for d in ["0.05", "0.1", "0.2"]]
    tr_dt = [metrics(f"dt_{d}")["best"]["train_mse"] for d in ["0.05", "0.1", "0.2"]]
    npts = [metrics(f"dt_{d}")["config"]["n_train_points"] for d in ["0.05", "0.1", "0.2"]]
    trunc = []
    for d in dts:
        n = int(round(14 / d)) + 1
        t = torch.linspace(0, 14, n, dtype=torch.float64)
        ref = scipy.integrate.solve_ivp(lambda tt, u: f_true_np(u), (0, 14), [1.0, 1.0], t_eval=t.numpy(),
                                        method="DOP853", rtol=1e-13, atol=1e-13).y.T
        u = odeint(f_true_torch, torch.tensor([1.0, 1.0], dtype=torch.float64), t, method="tsit5", substeps=2).numpy()
        ntr = int(round(3.5 / d)) + 1
        trunc.append(float(np.mean((u[ntr:] - ref[ntr:]) ** 2)))
    print("  Tsit5 true-field truncation MSE per dt:", dict(zip(dts, [f"{x:.1e}" for x in trunc])))
    return dict(dt=dts, extrap=ex_dt, train=tr_dt, n_train=npts, tsit5_truncation_true_field=trunc)


def main():
    save_numbers(dict(noise_fit=noise(), dt_sweep=stepsize()))


if __name__ == "__main__":
    main()
