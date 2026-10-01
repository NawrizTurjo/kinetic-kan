"""
Extrapolation horizon doubled to t = 28 (Figure "kan-vs-mlp-t28", Table "extrap-horizon").

Same experiment as task_extrap28 in implementation/phase2_closeout.py, with the
same loader, integrator, metrics and plot layout, except that the tanh MLP-ODE is
the paper's literal network trained with the paper's own learning rate and
initialization (results/mlp_paperspec_exact_10k_result) instead of the run at this
project's default settings (results/benchmarks/mlpode_baseline).

Nothing is retrained: each saved checkpoint is integrated from u(0) out to t = 28.

Output: report/figures/ablation/07_extrapolation_t28.png
        report/figures/analysis/numbers.json -> extrap28
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
IMPL = REPORT.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.chdir(IMPL)  # phase2_closeout and the run folders use paths relative to implementation/

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(HERE))
from common import save_numbers  # noqa: E402

# imported after common so utils.plotting's rcParams (the original figure's style) win
import phase2_closeout as p2  # noqa: E402
from data import generate_lotka_volterra_data  # noqa: E402

RUNS = {
    "KAN-ODE (RBF)": "results/benchmarks/kanode_flagship",
    "MLP-ODE (SiLU)": "results/benchmarks/mlpode_baseline_silu",
    "MLP-ODE (tanh, paper lr/init)": "results/mlp_paperspec_exact_10k_result",
}
OUT = REPORT / "figures" / "ablation" / "07_extrapolation_t28.png"


def main():
    out, curves = {}, {}
    for label, d in RUNS.items():
        field, cfg = p2.load_field(d)
        # the paper-settings checkpoint (saved by a Kaggle script) omits "dt"; its
        # 141 points on [0, 14] are the same 0.1 grid every other run uses
        data = generate_lotka_volterra_data(
            t_start=cfg.get("t_start", 0.0), t_end=28.0, dt=cfg.get("dt", 0.1),
            t_train_end=cfg["t_train_end"], noise_std=0.0, seed=cfg.get("seed", 42),
            **{k: cfg["data_params"][k] for k in ("alpha", "beta", "gamma", "delta")
               if k in cfg.get("data_params", {})},
        )
        pred = p2.integrate(field, cfg, data.y0, data.t_full)
        y, t = data.y_full.numpy(), data.t_full.numpy()
        n_tr = len(data.t_train)
        n_14 = int(np.searchsorted(t, 14.0, side="right"))
        curves[label] = (t, y, pred)
        w = {"train": p2._metrics(y[:n_tr], pred[:n_tr]),
             "extrap_3.5_14": p2._metrics(y[n_tr:n_14], pred[n_tr:n_14]),
             "far_14_28": p2._metrics(y[n_14:], pred[n_14:])}
        w["degradation"] = w["far_14_28"]["mse"] / w["extrap_3.5_14"]["mse"]
        out[label] = w
        print(f"  {label:30s} train {w['train']['mse']:.3e}  (3.5,14] {w['extrap_3.5_14']['mse']:.3e}"
              f"  (14,28] {w['far_14_28']['mse']:.3e}  degrade {w['degradation']:.1f}x")

    fig, axes = plt.subplots(len(curves), 1, figsize=(11, 3.1 * len(curves)),
                             dpi=150, sharex=True, squeeze=False)
    for ax, (label, (t, y, p)) in zip(axes[:, 0], curves.items()):
        ax.plot(t, y[:, 0], "k-", lw=1.6, label="True prey")
        ax.plot(t, y[:, 1], color="0.45", lw=1.6, label="True predator")
        ax.plot(t, p[:, 0], "--", color="#d62728", lw=1.5, label="Pred prey")
        ax.plot(t, p[:, 1], "--", color="#1f77b4", lw=1.5, label="Pred predator")
        ax.axvline(3.5, color="green", ls=":", lw=1.8)
        ax.axvline(14.0, color="purple", ls=":", lw=1.8)
        ax.set_ylabel("population")
        ax.set_title(f"{label} — train | extrapolation | far-extrapolation",
                     fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_ylim(-1, 9)
    axes[0, 0].legend(ncol=4, fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("time $t$   (green = train cutoff 3.5, purple = old horizon 14.0)")
    fig.suptitle("Extrapolation Horizon Doubled to $t=28$ (8 periods)", fontweight="bold")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT.relative_to(REPORT.parent).as_posix()}")
    save_numbers(dict(extrap28=out))


if __name__ == "__main__":
    main()
