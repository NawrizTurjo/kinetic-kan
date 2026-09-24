"""
Cost-accuracy plane: training wall-clock against extrapolation MSE.

Every point is read from a run's metrics.json (training_time_seconds and
best.extrap_mse): the six solvers (RBF basis), the six non-RBF bases (Tsit5)
and the SiLU MLP baseline, all at 10k epochs on the same machine. The dashed
staircase is the lower-left Pareto front.

Report: Figure "pareto" (Section 3.2).
Output: figures/analysis/pareto_cost_accuracy.pdf
        numbers.json -> pareto_front
"""

import matplotlib.pyplot as plt
import numpy as np

from common import BASES, BASIS_LABEL, C, SOLVER_LABEL, SOLVERS, metrics, save, save_numbers


def main():
    pts = []
    for s in SOLVERS:
        mt = metrics(f"solver_{s}")
        pts.append((SOLVER_LABEL[s], mt["training_time_seconds"], mt["best"]["extrap_mse"], "solver"))
    for b in BASES:
        if b == "rbf":
            continue
        mt = metrics(f"basis_{b}")
        pts.append((BASIS_LABEL[b], mt["training_time_seconds"], mt["best"]["extrap_mse"], "basis"))
    mt = metrics("mlp_silu")
    pts.append(("MLP-ODE (SiLU)", mt["training_time_seconds"], mt["best"]["extrap_mse"], "mlp"))
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    sty = dict(solver=(C["blue"], "o", "solver sweep (RBF)"), basis=(C["crimson"], "s", "basis sweep (Tsit5)"),
               mlp=(C["teal"], "D", "MLP baseline"))
    seen = set()
    for lab, t, e, kind in pts:
        col, mk, leg = sty[kind]
        ax.scatter(t, e, color=col, marker=mk, s=18, zorder=3, edgecolor="white", lw=0.6,
                   label=leg if kind not in seen else None)
        seen.add(kind)
        off = {"Tsit5": (4, -7), "DOPRI5": (4, 3), "RK4": (4, 3), "Midpoint": (4, -8), "Heun": (4, 2),
               "Euler": (4, 2), "B-spline": (-10, 5), "Lagrange": (-36, -3), "Chebyshev": (4, 2),
               "IQF": (4, 2), "RSWAF": (4, 2), "Newton": (4, 2), "MLP-ODE (SiLU)": (-24, 6)}[lab]
        ax.annotate(lab, (t, e), xytext=off, textcoords="offset points", fontsize=6, color=C["ink"])
    # Pareto front (lower-left hull)
    arr = sorted([(t, e) for _, t, e, _ in pts])
    front, best = [], np.inf
    for t, e in arr:
        if e < best:
            front.append((t, e)); best = e
    ft, fe = zip(*front)
    ax.step(ft, fe, where="post", color=C["gray"], lw=0.8, ls="--", zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training wall-clock for 10k epochs (s)")
    ax.set_ylabel("extrapolation MSE")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6, ncol=3, handletextpad=0.2, columnspacing=0.8)
    ax.grid(True, which="major")
    fig.tight_layout()
    save(fig, "pareto_cost_accuracy")
    front_labels = [lab for lab, t, e, _ in pts if (t, e) in front]
    print("  Pareto front:", front_labels)
    save_numbers(dict(pareto_front=front_labels))


if __name__ == "__main__":
    main()
