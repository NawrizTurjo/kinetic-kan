import argparse
import json
import os
import statistics
import time

import torch

from train import train_kan_ode, DATASETS
from utils.plotting import plot_benchmark_comparison


# Metrics lifted out of each run for the summary tables. Everything is reported at the
# best-epoch checkpoint (selected on training loss), so no column is a min-over-epochs
# statistic and no column is selected against the extrapolation window.
REPORTED = ["train_mse", "extrap_mse", "full_mse", "extrap_r2", "extrap_rel_l2"]


def _run_one(save_dir, seed, **kwargs):
    """Train a single configuration and flatten the metrics we tabulate."""
    t0 = time.time()
    res = train_kan_ode(save_dir=save_dir, seed=seed, **kwargs)
    row = {k: float(res["best"][k]) for k in REPORTED}
    row["final_extrap_mse"] = float(res["final"]["extrap_mse"])
    row["best_epoch"] = res["best_epoch"]
    row["time_sec"] = time.time() - t0
    row["nfe_per_epoch_train"] = res["metrics"]["nfe_per_epoch_train"]
    row["parameters"] = res["config"]["parameters"]
    return row


def _aggregate(rows):
    """Collapse N seed repetitions into mean/std per metric."""
    agg = {"n_seeds": len(rows), "seeds": [r["seed"] for r in rows]}
    numeric = [k for k in rows[0] if k != "seed" and isinstance(rows[0][k], (int, float))]
    for k in numeric:
        vals = [r[k] for r in rows]
        agg[f"{k}_mean"] = statistics.mean(vals)
        agg[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    agg["per_seed"] = rows
    return agg


def run_sweep(name, variants, save_dir, seeds, base_kwargs, subdir_fmt):
    """
    Run one ablation axis across a list of variant kwarg-overrides, repeated over seeds.

    Args:
        variants: list of (label, kwargs_override) pairs.
        seeds: list of integer seeds; N>1 produces mean +/- std.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    print("\n" + "=" * 78)
    print(f"SWEEP: {name}")
    print(f"Variants: {[lbl for lbl, _ in variants]}")
    print(f"Seeds: {seeds} | Base: {base_kwargs}")
    print("=" * 78)

    for label, override in variants:
        rows = []
        for seed in seeds:
            tag = subdir_fmt[label] if isinstance(subdir_fmt, dict) else subdir_fmt.format(label=label)
            run_dir = os.path.join(save_dir, tag if len(seeds) == 1 else f"{tag}_seed{seed}")
            print(f"\n---> {name}: {label} (seed {seed}) ...")
            try:
                row = _run_one(run_dir, seed, **{**base_kwargs, **override})
                row["seed"] = seed
                rows.append(row)
            except Exception as e:
                print(f"  ERROR on {label} seed {seed}: {type(e).__name__}: {e}")

        if rows:
            entry = _aggregate(rows)
            entry["status"] = "success"
            entry.update({k: v for k, v in override.items() if isinstance(v, (str, int, float))})
        else:
            entry = {"status": "failed"}
        results[label] = entry

    _save_and_report(results, save_dir, name)
    return results


def _save_and_report(results, save_dir, title):
    """Print the summary table, dump JSON, and render the comparison bar chart."""
    print("\n" + "=" * 108)
    print(f"SUMMARY: {title.upper()}")
    print("=" * 108)
    hdr = (f"{'Configuration':<22} | {'Train MSE':<20} | {'Extrap MSE':<20} | "
           f"{'Extrap R2':<16} | {'Time (s)':<10}")
    print(hdr)
    print("-" * 108)

    valid = {}
    for name, d in results.items():
        if d.get("status") != "success":
            print(f"{name:<22} | {'FAILED':<20} | {'FAILED':<20} | {'--':<16} | {'--':<10}")
            continue
        n = d["n_seeds"]
        if n > 1:
            tr = f"{d['train_mse_mean']:.3e}+-{d['train_mse_std']:.1e}"
            ex = f"{d['extrap_mse_mean']:.3e}+-{d['extrap_mse_std']:.1e}"
            r2 = f"{d['extrap_r2_mean']:.4f}+-{d['extrap_r2_std']:.3f}"
        else:
            tr = f"{d['train_mse_mean']:.4e}"
            ex = f"{d['extrap_mse_mean']:.4e}"
            r2 = f"{d['extrap_r2_mean']:.4f}"
        print(f"{name:<22} | {tr:<20} | {ex:<20} | {r2:<16} | {d['time_sec_mean']:<10.1f}")
        valid[name] = {"test_mse": d["extrap_mse_mean"], **d}

    print("=" * 108)

    with open(os.path.join(save_dir, "benchmark_summary.json"), "w") as f:
        json.dump(results, f, indent=4)

    if valid:
        plot_benchmark_comparison(
            results=valid,
            metric="test_mse",
            title=f"{title} - Extrapolation MSE (best-epoch checkpoint)",
            save_path=os.path.join(save_dir, "benchmark_comparison.png"),
        )
    print(f"Saved summary and comparison plot to '{save_dir}'.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAN-ODE Testing & Benchmarking Facility")
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "activations", "solvers", "stepsize", "models", "noise"],
        help="activations=Table 2 | solvers=Table 1 | stepsize=dt sweep | models=KAN vs MLP | noise=sigma sweep",
    )
    parser.add_argument("--epochs", type=int, default=10000, help="Epochs per experiment")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate (standardised across the project)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Seeds; more than one gives mean +/- std")
    parser.add_argument("--dataset", type=str, default="lotka_volterra", choices=list(DATASETS))
    parser.add_argument("--solver", type=str, default="tsit5", help="Fixed solver for the basis/noise/model sweeps")
    parser.add_argument("--basis", type=str, default="rbf", help="Fixed basis for the solver/stepsize sweeps")
    parser.add_argument("--substeps", type=int, default=2, help="Integration substeps per reporting interval")
    parser.add_argument("--dt", type=float, default=None, help="Observation step (default: dataset-specific)")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Noise sigma for non-noise sweeps")
    parser.add_argument("--dts", type=float, nargs="+", default=[0.20, 0.10, 0.05, 0.01], help="Step sizes for --mode stepsize")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.10], help="Noise levels for --mode noise")
    parser.add_argument("--save_dir", type=str, default="results/benchmarks", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="Run a fast 300-epoch smoke test")

    args = parser.parse_args()
    epochs = 300 if args.quick else args.epochs

    common = dict(
        dataset=args.dataset,
        num_epochs=epochs,
        lr=args.lr,
        substeps=args.substeps,
        dt=args.dt,
        noise_std=args.noise_std,
        device=args.device,
    )

    if args.mode in ("activations", "all"):
        run_sweep(
            "Basis Function Ablation (Table 2)",
            [(b, {"basis_func": b}) for b in
             ["rbf", "bspline", "chebyshev", "lagrange", "newton", "rswaf", "iqf"]],
            os.path.join(args.save_dir, "ablation_activations"),
            args.seeds,
            {**common, "solver": args.solver},
            "basis_{label}",
        )

    if args.mode in ("solvers", "all"):
        run_sweep(
            "ODE Integrator Ablation (Table 1)",
            [(s, {"solver": s}) for s in
             ["tsit5", "rk4", "dopri5", "midpoint", "heun", "euler"]],
            os.path.join(args.save_dir, "ablation_solvers"),
            args.seeds,
            {**common, "basis_func": args.basis},
            "solver_{label}",
        )

    if args.mode in ("stepsize", "all"):
        run_sweep(
            "Step-Size Sweep (Table 1b)",
            [(f"dt{d}", {"dt": d}) for d in args.dts],
            os.path.join(args.save_dir, "stepsize"),
            args.seeds,
            {**common, "basis_func": args.basis, "solver": args.solver},
            "{label}",
        )

    if args.mode in ("models", "all"):
        run_sweep(
            "KAN-ODE vs MLP-ODE (Table 3)",
            [("kan", {"model_type": "kan", "basis_func": args.basis}),
             ("mlp", {"model_type": "mlp"})],
            args.save_dir,
            args.seeds,
            {**common, "solver": args.solver},
            {"kan": "kanode_flagship", "mlp": "mlpode_baseline"},
        )

    if args.mode in ("noise", "all"):
        run_sweep(
            "Noise Robustness Sweep",
            [(f"sigma{s}", {"noise_std": s}) for s in args.sigmas],
            os.path.join(args.save_dir, "noise"),
            args.seeds,
            {**common, "basis_func": args.basis, "solver": args.solver},
            "{label}",
        )
