"""
[PHASE2-CLOSEOUT] The three Phase-2 deliverables that need no new training runs.

Every task here re-uses checkpoints and histories that are already on disk, so the
whole script runs in well under a minute. See docs/11_phase2_closeout.md.

Closes, against docs/04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md Phase 2:

  extrap28   Task 2.3 -- "extrapolation ... and beyond to t=28.0". No retraining is
             needed: training ends at t=3.5, so a longer horizon is pure integration
             of an already-trained vector field.
  energy     Task 2.4 -- "verify physical generalizability and energy decay capturing".
             utils.compute_energy_violation and utils.plot_pendulum_phase_and_energy
             were built in Phase 1 and had never been called.
  figures    Task 2.5 -- "multi-panel publication figures (phase portraits with
             streamlines, error vs dt log-log curves)". results/figures/ did not exist.

Usage
-----
    python phase2_closeout.py --task all
    python phase2_closeout.py --task extrap28
    python phase2_closeout.py --task energy
    python phase2_closeout.py --task figures

Outputs
-------
    results/phase2_closeout/extrap28.json     Task 2.3 metrics
    results/phase2_closeout/energy.json       Task 2.4 metrics
    results/figures/*.png                     Task 2.5 figures (300 DPI)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

from kan import KAN, MLP_ODE           # noqa: E402
from ode import NeuralODE, ZeroSumField, VanishingDimField   # noqa: E402
from data import (                     # noqa: E402
    generate_lotka_volterra_data,
    generate_damped_pendulum_data,
    lotka_volterra_deriv,
    compute_pendulum_energy,
)
from utils import (                    # noqa: E402
    compute_mse, compute_rmse, compute_mae,
    compute_r2_score, compute_relative_l2_error,
    compute_energy_violation,
    plot_phase_portrait_with_streamlines,
    plot_pendulum_phase_and_energy,
    plot_model_comparison_curves,
)

FIG_DIR = "results/figures"
OUT_DIR = "results/phase2_closeout"


# ---------------------------------------------------------------------------
# Shared checkpoint loading
# ---------------------------------------------------------------------------

def load_field(run_dir, which="best_model.pt"):
    """
    Rebuild the exact field a checkpoint was trained with.

    Mirrors evaluate.py: grid_lims, conserve_mode and vanish_dim are all part of the
    FIELD rather than the weights, so a rebuild that ignores them silently integrates
    a different ODE than the run actually produced.
    """
    ckpt = torch.load(os.path.join(run_dir, which), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    if cfg.get("model_type", "kan").lower() == "mlp":
        model = MLP_ODE(layers_hidden=cfg["layers_hidden"], activation=cfg.get("mlp_act", "tanh"))
    else:
        model = KAN(
            layers_hidden=cfg["layers_hidden"],
            grid_len=cfg["grid_len"],
            grid_lims=tuple(cfg.get("grid_lims", [-1.0, 1.0])),
            basis_func=cfg["basis_func"],
            normalizer=cfg.get("normalizer", "tanh"),
            base_act=cfg.get("base_act", "silu"),
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    field = model
    if cfg.get("conserve_mode") == "projection":
        field = ZeroSumField(model)
    if cfg.get("vanish_dim") is not None:
        field = VanishingDimField(field, dim=int(cfg["vanish_dim"]))
    return field, cfg


def integrate(field, cfg, y0, t):
    """Integrate on the clock the run was trained on (see --time_scale)."""
    node = NeuralODE(func=field, method=cfg["solver"], substeps=cfg.get("substeps", 2))
    scale = float(cfg.get("time_scale", 1.0) or 1.0)
    with torch.no_grad():
        return node(y0=y0, t=t / scale).numpy()


def _metrics(y, p):
    return {
        "mse": compute_mse(y, p),
        "rmse": compute_rmse(y, p),
        "mae": compute_mae(y, p),
        "r2": compute_r2_score(y, p),
        "rel_l2": compute_relative_l2_error(y, p),
    }


# ---------------------------------------------------------------------------
# Task 2.3 -- extrapolation to t = 28
# ---------------------------------------------------------------------------

def task_extrap28():
    """
    Double the evaluation horizon from t=14 to t=28 (8 Lotka-Volterra periods).

    The plan asks for extrapolation "beyond to t=28.0". Because training ends at
    t=3.5, nothing needs retraining -- the trained vector field is simply integrated
    further. The window (3.5, 14] is therefore directly comparable with every number
    already published in docs/05, and (14, 28] is entirely new territory.
    """
    runs = {
        "KAN-ODE (RBF)": "results/benchmarks/kanode_flagship",
        "MLP-ODE (SiLU)": "results/benchmarks/mlpode_baseline_silu",
        "MLP-ODE (tanh, paper)": "results/benchmarks/mlpode_baseline",
    }

    print("=" * 100)
    print("TASK 2.3  --  extrapolation horizon doubled to t = 28.0  (no retraining)")
    print("=" * 100)

    out, curves = {}, {}
    for label, d in runs.items():
        if not os.path.exists(os.path.join(d, "best_model.pt")):
            print(f"  ! skipping {label}: no checkpoint at {d}")
            continue
        field, cfg = load_field(d)

        # Same training window, same equation parameters, horizon extended 14 -> 28.
        data = generate_lotka_volterra_data(
            t_start=cfg.get("t_start", 0.0), t_end=28.0, dt=cfg["dt"],
            t_train_end=cfg["t_train_end"], noise_std=0.0, seed=cfg.get("seed", 42),
            **{k: cfg["data_params"][k] for k in ("alpha", "beta", "gamma", "delta")
               if k in cfg.get("data_params", {})},
        )
        pred = integrate(field, cfg, data.y0, data.t_full)
        y, t = data.y_full.numpy(), data.t_full.numpy()
        n_tr = len(data.t_train)
        n_14 = int(np.searchsorted(t, 14.0, side="right"))
        curves[label] = (t, y, pred, n_tr)

        out[label] = {
            "run_dir": d,
            "windows": {
                "train [0, 3.5]": _metrics(y[:n_tr], pred[:n_tr]),
                "extrap (3.5, 14]": _metrics(y[n_tr:n_14], pred[n_tr:n_14]),
                "far-extrap (14, 28]": _metrics(y[n_14:], pred[n_14:]),
                "full [0, 28]": _metrics(y, pred),
            },
            "periods_extrapolated": round((28.0 - cfg["t_train_end"]) / 3.5, 1),
        }

    hdr = f"{'model':<24}{'train':>12}{'(3.5,14]':>12}{'(14,28]':>12}{'R2 (14,28]':>13}{'degrade':>10}"
    print(hdr)
    print("-" * len(hdr))
    for label, r in out.items():
        w = r["windows"]
        near, far = w["extrap (3.5, 14]"]["mse"], w["far-extrap (14, 28]"]["mse"]
        print(f"{label:<24}{w['train [0, 3.5]']['mse']:>12.3e}{near:>12.3e}{far:>12.3e}"
              f"{w['far-extrap (14, 28]']['r2']:>13.4f}{far / near:>9.1f}x")
    print("-" * len(hdr))
    print("'degrade' = far-extrapolation MSE divided by near-extrapolation MSE.")
    print("A value near 1 means error does NOT compound as the horizon doubles.")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "extrap28.json"), "w") as f:
        json.dump(out, f, indent=4)

    # Figure: all models over the doubled horizon, with both split lines marked.
    if curves:
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(len(curves), 1, figsize=(11, 3.1 * len(curves)),
                                 dpi=150, sharex=True, squeeze=False)
        for ax, (label, (t, y, p, n_tr)) in zip(axes[:, 0], curves.items()):
            ax.plot(t, y[:, 0], "k-", lw=1.6, label="True prey")
            ax.plot(t, y[:, 1], color="0.45", lw=1.6, label="True predator")
            ax.plot(t, p[:, 0], "--", color="#d62728", lw=1.5, label="Pred prey")
            ax.plot(t, p[:, 1], "--", color="#1f77b4", lw=1.5, label="Pred predator")
            ax.axvline(3.5, color="green", ls=":", lw=1.8)
            ax.axvline(14.0, color="purple", ls=":", lw=1.8)
            ax.set_ylabel("population")
            ax.set_title(f"{label} — train | extrapolation | NEW far-extrapolation",
                         fontsize=10, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.set_ylim(-1, 9)
        axes[0, 0].legend(ncol=4, fontsize=8, loc="upper right")
        axes[-1, 0].set_xlabel("time $t$   (green = train cutoff 3.5, purple = old horizon 14.0)")
        fig.suptitle("Task 2.3: Extrapolation Horizon Doubled to $t=28$ (8 periods)",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "07_extrapolation_t28.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved {FIG_DIR}/07_extrapolation_t28.png and {OUT_DIR}/extrap28.json")
    return out


# ---------------------------------------------------------------------------
# Task 2.4 -- pendulum energy dissipation
# ---------------------------------------------------------------------------

def task_energy(run_dir="results/_fixed/pendulum_control_win5"):
    """
    Verify the pendulum run captures physical energy decay.

    For mu > 0 the true system is strictly dissipative: dE/dt = -m L^2 mu omega^2 <= 0.
    A model can fit the trajectory and still get the ENERGY wrong, so this is a
    genuinely independent physical check rather than a restatement of the MSE.
    """
    print("=" * 100)
    print("TASK 2.4  --  damped-pendulum energy dissipation")
    print("=" * 100)
    if not os.path.exists(os.path.join(run_dir, "best_model.pt")):
        print(f"  ! no checkpoint at {run_dir}")
        return None

    field, cfg = load_field(run_dir)
    data = generate_damped_pendulum_data(
        t_start=cfg.get("t_start", 0.0), t_end=cfg["t_end"], dt=cfg["dt"],
        t_train_end=cfg["t_train_end"], noise_std=0.0, seed=cfg.get("seed", 42),
        **{k: cfg["data_params"][k] for k in ("mu", "g", "length", "mass")
           if k in cfg.get("data_params", {})},
    )
    pred = integrate(field, cfg, data.y0, data.t_full)
    y, t = data.y_full.numpy(), data.t_full.numpy()
    n_tr = len(data.t_train)
    p = cfg["data_params"]

    e_true = compute_pendulum_energy(y, g=p["g"], length=p["length"], mass=p["mass"])
    e_pred = compute_pendulum_energy(pred, g=p["g"], length=p["length"], mass=p["mass"])

    # Monotonic decay is the physical invariant. Count how often each series rises.
    d_true, d_pred = np.diff(e_true), np.diff(e_pred)
    tol = 1e-9
    viol = compute_energy_violation(pred, lambda a: compute_pendulum_energy(
        a, g=p["g"], length=p["length"], mass=p["mass"]))

    res = {
        "run_dir": run_dir,
        "params": {k: p[k] for k in ("mu", "g", "length", "mass")},
        "energy_true": {"E0": float(e_true[0]), "E_final": float(e_true[-1]),
                        "fraction_dissipated": float(1 - e_true[-1] / e_true[0])},
        "energy_pred": {"E0": float(e_pred[0]), "E_final": float(e_pred[-1]),
                        "fraction_dissipated": float(1 - e_pred[-1] / e_pred[0])},
        "monotonic_decay": {
            "true_increasing_steps": int((d_true > tol).sum()),
            "pred_increasing_steps": int((d_pred > tol).sum()),
            "total_steps": int(len(d_true)),
        },
        "energy_rmse": {
            "train": float(np.sqrt(((e_true[:n_tr] - e_pred[:n_tr]) ** 2).mean())),
            "extrap": float(np.sqrt(((e_true[n_tr:] - e_pred[n_tr:]) ** 2).mean())),
        },
        "drift_vs_own_E0": viol,
    }

    print(f"  mu = {p['mu']}  (mu > 0 => strictly dissipative, dE/dt <= 0)")
    print(f"  true   E0 = {e_true[0]:.4f} -> E_final = {e_true[-1]:.4f}   "
          f"({100 * res['energy_true']['fraction_dissipated']:.1f}% dissipated)")
    print(f"  pred   E0 = {e_pred[0]:.4f} -> E_final = {e_pred[-1]:.4f}   "
          f"({100 * res['energy_pred']['fraction_dissipated']:.1f}% dissipated)")
    print(f"  energy RMSE: train {res['energy_rmse']['train']:.4e} | "
          f"extrapolation {res['energy_rmse']['extrap']:.4e}")
    print(f"  monotonic decay violated on {res['monotonic_decay']['pred_increasing_steps']}"
          f"/{res['monotonic_decay']['total_steps']} steps "
          f"(true: {res['monotonic_decay']['true_increasing_steps']})")
    verdict = ("CAPTURED" if res["energy_rmse"]["extrap"] < 0.1 * e_true[0]
               else "NOT captured")
    print(f"  VERDICT: energy decay {verdict}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "energy.json"), "w") as f:
        json.dump(res, f, indent=4)

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_pendulum_phase_and_energy(
        t_full=t, y_true=y, y_pred=pred,
        energy_true=e_true, energy_pred=e_pred, train_len=n_tr,
        title=f"Damped Pendulum ($\\mu={p['mu']}$): Phase Portrait & Energy Dissipation",
        save_path=os.path.join(FIG_DIR, "04_pendulum_energy.png"),
    )
    print(f"\nSaved {FIG_DIR}/04_pendulum_energy.png and {OUT_DIR}/energy.json")
    return res


# ---------------------------------------------------------------------------
# Task 2.5 -- publication figures
# ---------------------------------------------------------------------------

def _fig_streamlines():
    """Phase portrait with the LEARNED vector field drawn as streamlines."""
    d = "results/benchmarks/kanode_flagship"
    if not os.path.exists(os.path.join(d, "best_model.pt")):
        return None
    field, cfg = load_field(d)
    data = generate_lotka_volterra_data(
        t_end=cfg["t_end"], dt=cfg["dt"], t_train_end=cfg["t_train_end"],
        seed=cfg.get("seed", 42))
    pred = integrate(field, cfg, data.y0, data.t_full)

    def vf(pts):
        with torch.no_grad():
            return field(torch.tensor(pts, dtype=torch.float32)).numpy()

    plot_phase_portrait_with_streamlines(
        vector_field_fn=vf,
        y_true=data.y_full.numpy(), y_pred=pred, train_len=len(data.t_train),
        x_range=(0.05, 7.0), y_range=(0.05, 5.0), grid_density=30,
        title="KAN-ODE: Learned Vector Field & Limit Cycle (Lotka-Volterra)",
        save_path=os.path.join(FIG_DIR, "01_phase_portrait_streamlines.png"),
    )
    return "01_phase_portrait_streamlines.png"


def _fig_dt_loglog():
    """Error vs step size on log-log axes, with the known confound annotated."""
    pts = []
    for tag in sorted(os.listdir("results/benchmarks/stepsize")):
        f = f"results/benchmarks/stepsize/{tag}/metrics.json"
        if not os.path.exists(f):
            continue
        m = json.load(open(f))
        pts.append((m["config"]["dt"], m["best"]["train_mse"],
                    m["best"]["extrap_mse"], m["config"]["n_train_points"]))
    if len(pts) < 2:
        return None
    pts.sort()
    dt = np.array([p[0] for p in pts])
    tr = np.array([p[1] for p in pts])
    ex = np.array([p[2] for p in pts])

    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=150)
    ax.loglog(dt, tr, "o-", lw=2, ms=9, color="#1f77b4", label="Train MSE")
    ax.loglog(dt, ex, "s-", lw=2, ms=9, color="#d62728", label="Extrapolation MSE")
    for d_, e_, n_ in zip(dt, ex, [p[3] for p in pts]):
        ax.annotate(f"$\\Delta t$={d_:g}\n{n_} train pts", (d_, e_),
                    textcoords="offset points", xytext=(8, 8), fontsize=8)
    # reference slopes anchored at the finest step
    for order, style in ((1, ":"), (2, "--")):
        ax.loglog(dt, ex[0] * (dt / dt[0]) ** order, style, color="0.6", lw=1.2,
                  label=f"$O(\\Delta t^{order})$ reference")
    ax.set_xlabel("Step size $\\Delta t$")
    ax.set_ylabel("MSE")
    ax.set_title("Error vs. Step Size (RBF + Tsit5, 10,000 epochs)",
                 fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    # Caption below the axes rather than inside them: an in-axes box overlapped
    # the train-MSE line and the dt=0.1 point annotation.
    fig.text(0.5, -0.02,
             "CONFOUNDED: changing $\\Delta t$ also changes the number of training samples "
             "(19 / 36 / 71).\nNot a pure discretisation study — the non-monotonic minimum at "
             "$\\Delta t=0.1$ reflects both effects. See docs/05.",
             ha="center", va="top", fontsize=8.5, color="#8B0000",
             bbox=dict(boxstyle="round", fc="#fff3f3", ec="#d62728", alpha=0.95))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_error_vs_stepsize_loglog.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    return "02_error_vs_stepsize_loglog.png"


def _fig_curves(name, runs, title, thresh):
    """Semilog loss-decay overlay for a set of runs."""
    hist = {}
    for label, d in runs.items():
        f = os.path.join(d, "training_history.json")
        if os.path.exists(f):
            hist[label] = json.load(open(f))
    if not hist:
        return None
    plot_model_comparison_curves(
        histories=hist, metric_key="train_losses", target_threshold=thresh,
        title=title, save_path=os.path.join(FIG_DIR, name))
    return name


def task_figures():
    print("=" * 100)
    print("TASK 2.5  --  multi-panel publication figures (300 DPI)")
    print("=" * 100)
    os.makedirs(FIG_DIR, exist_ok=True)
    made = []

    for fn in (_fig_streamlines, _fig_dt_loglog):
        r = fn()
        if r:
            made.append(r)

    made.append(_fig_curves(
        "03_kan_vs_mlp_convergence.png",
        {"KAN-ODE (RBF, 240p)": "results/benchmarks/kanode_flagship",
         "MLP-ODE (SiLU, 252p)": "results/benchmarks/mlpode_baseline_silu",
         "MLP-ODE (tanh, paper, 252p)": "results/benchmarks/mlpode_baseline"},
        "KAN-ODE vs. Parameter-Matched MLP-ODE: Loss Decay", 1e-4))

    made.append(_fig_curves(
        "05_solver_convergence.png",
        {s.replace("solver_", "").upper(): f"results/benchmarks/ablation_solvers/{s}"
         for s in sorted(os.listdir("results/benchmarks/ablation_solvers"))},
        "Solver Order Ablation: Loss Decay (RBF, $\\Delta t=0.1$)", 1e-4))

    made.append(_fig_curves(
        "06_basis_convergence.png",
        {b.replace("basis_", "").upper(): f"results/benchmarks/ablation_activations/{b}"
         for b in sorted(os.listdir("results/benchmarks/ablation_activations"))},
        "Basis Function Ablation: Loss Decay (Tsit5, $G=5$)", 1e-4))

    made = [m for m in made if m]
    # ASCII only: the Windows console defaults to cp1252 and raises
    # UnicodeEncodeError on characters like U+2713.
    for m in sorted(made):
        print(f"  [ok] {FIG_DIR}/{m}")
    print("\n  NOT generated: 3D Lorenz attractor figure "
          "(plot_3d_lorenz_trajectory) -- the Lorenz sweep has never been run.")
    return made


def main():
    ap = argparse.ArgumentParser(description="[PHASE2-CLOSEOUT] run-free Phase-2 deliverables")
    ap.add_argument("--task", default="all",
                    choices=["all", "extrap28", "energy", "figures"])
    ap.add_argument("--pendulum_run", default="results/_fixed/pendulum_control_win5",
                    help="checkpoint dir used for the Task 2.4 energy check")
    a = ap.parse_args()

    if a.task in ("all", "extrap28"):
        task_extrap28()
        print()
    if a.task in ("all", "energy"):
        task_energy(a.pendulum_run)
        print()
    if a.task in ("all", "figures"):
        task_figures()


if __name__ == "__main__":
    main()
