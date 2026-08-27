"""
[FIX-2026-08] Diagnostic reader for the cross-domain stability runs.

Reads a results tree (results/_probe, results/_fixed, or results/benchmarks) and
reports the quantities that actually distinguish a converged run from a failed one.
Aggregate MSE does not: on the damped pendulum the failure is confined to theta while
omega fits well, and on SIR it is a frozen fixed point that a single MSE number hides.

Sections
--------
summary    one row per run: losses, extrapolation R2, gradient spike ratio, NaN steps
pendulum   per-dimension theta/omega RMSE + whether theta can still swing negative
sir        per-compartment RMSE, S+I+R mass conservation, extrapolation drift
flatness   loss at 20/80/100% of the run -- catches "low but stopped descending"

Usage
-----
    python analyze_fixes.py --root results/_probe
    python analyze_fixes.py --root results/_fixed
    python analyze_fixes.py --root results/benchmarks --only summary

Reference values from the two FAILED pre-fix runs, for comparison:
    pendulum : theta_rmse 0.7555, omega_rmse 0.0477, theta_pred_min -0.211
               (true theta_min is -1.385), train_mse 2.87e-1, extrap_r2 -1.318
    sir      : I_rmse 0.0776, mass [1.0044, 1.0045], extrap drift ~2e-4 (frozen),
               best train_mse 4.39e-4, final train_mse NaN
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kan import KAN, MLP_ODE          # noqa: E402
from ode import NeuralODE, ZeroSumField, VanishingDimField             # noqa: E402
from data import (                    # noqa: E402
    generate_lotka_volterra_data,
    generate_damped_pendulum_data,
    generate_lorenz_data,
    generate_sir_data,
)

_GENERATORS = {
    "lotka_volterra": generate_lotka_volterra_data,
    "damped_pendulum": generate_damped_pendulum_data,
    "lorenz": generate_lorenz_data,
    "sir": generate_sir_data,
}

_STATE_NAMES = {
    "lotka_volterra": ["prey", "pred"],
    "damped_pendulum": ["theta", "omega"],
    "lorenz": ["x", "y", "z"],
    "sir": ["S", "I", "R"],
}


def _find_runs(root):
    """Every directory under `root` holding a metrics.json, sorted by name."""
    return sorted(
        os.path.dirname(f) for f in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True)
    )


def _fmt(v, spec=".3e"):
    """Format a number, tolerating None and NaN (both appear in failed runs)."""
    if v is None:
        return "-"
    if isinstance(v, float) and not math.isfinite(v):
        return "NaN"
    return format(v, spec)


def _rebuild(config):
    """Regenerate the exact ground truth a checkpoint was trained against."""
    dataset = config.get("dataset", "lotka_volterra")
    gen = _GENERATORS[dataset]
    kwargs = dict(
        t_start=config.get("t_start", 0.0),
        t_end=config["t_end"],
        dt=config["dt"],
        t_train_end=config["t_train_end"],
        noise_std=config.get("noise_std", 0.0),
        seed=config.get("seed", 42),
    )
    if dataset == "lotka_volterra":
        for k in ("alpha", "beta", "gamma", "delta"):
            if k in config.get("data_params", {}):
                kwargs[k] = config["data_params"][k]
    return gen(**kwargs), dataset


def _predict(run_dir):
    """Load best_model.pt, rebuild its data, and integrate the full horizon."""
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    if not cfg:
        return None

    data, dataset = _rebuild(cfg)
    if cfg.get("model_type", "kan").lower() == "mlp":
        model = MLP_ODE(layers_hidden=cfg["layers_hidden"], activation=cfg.get("mlp_act", "tanh"))
    else:
        model = KAN(
            layers_hidden=cfg["layers_hidden"],
            grid_len=cfg["grid_len"],
            # older metrics.json predate --grid_lims; fall back to the old hardcoded span
            grid_lims=tuple(cfg.get("grid_lims", [-1.0, 1.0])),
            basis_func=cfg["basis_func"],
            normalizer=cfg.get("normalizer", "tanh"),
            base_act=cfg.get("base_act", "silu"),
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # [FIX-2026-08 / S4] see evaluate.py -- projection lives in the field, not the weights
    field = ZeroSumField(model) if cfg.get("conserve_mode") == "projection" else model
    if cfg.get("vanish_dim") is not None:
        field = VanishingDimField(field, dim=int(cfg["vanish_dim"]))
    node = NeuralODE(func=field, method=cfg["solver"], substeps=cfg.get("substeps", 2))
    # [FIX-2026-08 / S3] runs trained with --time_scale learned g = time_scale * f
    # and must be replayed on that same clock; older runs default to 1.0 (no-op).
    time_scale = float(cfg.get("time_scale", 1.0) or 1.0)
    with torch.no_grad():
        pred = node(y0=data.y0, t=data.t_full / time_scale).numpy()
    return {
        "cfg": cfg,
        "dataset": dataset,
        "y": data.y_full.numpy(),
        "pred": pred,
        "n_train": len(data.t_train),
    }


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_summary(runs):
    print("\n" + "=" * 104)
    print("SUMMARY  (best-epoch checkpoint; spike = max/median pre-clip gradient norm)")
    print("=" * 104)
    hdr = (f"{'run':<20}{'train_mse':>12}{'extrap_mse':>13}{'extrap_R2':>11}"
           f"{'final_train':>13}{'spike':>10}{'nonfin':>8}{'epochs':>8}")
    print(hdr)
    print("-" * len(hdr))
    for d in runs:
        m = json.load(open(os.path.join(d, "metrics.json")))
        s = m.get("stability", {})
        spike = s.get("grad_norm_spike_ratio")
        print(f"{os.path.basename(d):<20}"
              f"{_fmt(m['best']['train_mse']):>12}"
              f"{_fmt(m['best']['extrap_mse']):>13}"
              f"{_fmt(m['best']['extrap_r2'], '.4f'):>11}"
              f"{_fmt(m['final']['train_mse']):>13}"
              f"{(_fmt(spike, '.1f') + 'x') if spike else '-':>10}"
              f"{str(s.get('nonfinite_grad_steps', '-')):>8}"
              f"{str(s.get('epochs_run', '-')):>8}")
    print("=" * 104)
    print("PASS: final_train is a number (not NaN) | nonfin == 0 | epochs == the full budget")
    print("NOTE: a large 'spike' is NOT itself a failure once --grad_clip is on -- it is the")
    print("      PRE-clip norm, and clipping bounds the step. It only signals trouble when")
    print("      nonfin > 0, or when clipping is disabled. Judge fit by train_mse/extrap_mse.")


def section_pendulum(runs):
    rows = [(d, r) for d in runs if (r := _predict(d)) and r["dataset"] == "damped_pendulum"]
    if not rows:
        return
    print("\n" + "=" * 104)
    print("DAMPED PENDULUM  --  per-dimension breakdown")
    print("=" * 104)
    print("The aggregate MSE hides this failure: omega dominates the loss and fits well,")
    print("while theta -- whose derivative IS omega -- does not. Watch theta_rmse and")
    print("whether the prediction can still reach the true minimum of -1.385.")
    hdr = (f"\n{'run':<20}{'theta_rmse':>12}{'omega_rmse':>12}"
           f"{'theta_min':>11}{'theta_max':>11}{'verdict':>16}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    for d, r in rows:
        y, p, n = r["y"], r["pred"], r["n_train"]
        th = float(np.sqrt(((y[:n, 0] - p[:n, 0]) ** 2).mean()))
        om = float(np.sqrt(((y[:n, 1] - p[:n, 1]) ** 2).mean()))
        tmin, tmax = float(p[:n, 0].min()), float(p[:n, 0].max())
        # the pre-fix run stalled at theta_min = -0.211 against a true -1.385
        verdict = "IMPROVED" if th < 0.5 and tmin < -0.8 else ("partial" if th < 0.5 else "still failing")
        print(f"{os.path.basename(d):<20}{th:>12.4f}{om:>12.4f}{tmin:>11.3f}{tmax:>11.3f}{verdict:>16}")
    print(f"\n  true theta range over the training window: [-1.385, 2.000]")
    print(f"  pre-fix reference: theta_rmse 0.7555 | omega_rmse 0.0477 | theta_min -0.211")


def section_sir(runs):
    rows = [(d, r) for d in runs if (r := _predict(d)) and r["dataset"] == "sir"]
    if not rows:
        return
    print("\n" + "=" * 104)
    print("SIR  --  compartments, mass conservation, and the frozen-fixed-point check")
    print("=" * 104)
    print("The pre-fix model parked on a spurious equilibrium: its extrapolation drifted")
    print("by <3e-4 over 30 simulated days while the true I decays 0.057 -> 0.004.")
    print("A near-zero extrap_drift therefore means STILL FROZEN, not 'stable'.")
    hdr = (f"\n{'run':<20}{'S_rmse':>10}{'I_rmse':>10}{'R_rmse':>10}"
           f"{'mass_min':>10}{'mass_max':>10}{'I_drift':>10}{'verdict':>14}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    for d, r in rows:
        y, p, n = r["y"], r["pred"], r["n_train"]
        rmse = [float(np.sqrt(((y[n:, i] - p[n:, i]) ** 2).mean())) for i in range(3)]
        mass = p[n:].sum(axis=1)
        i_drift = float(p[n:, 1].max() - p[n:, 1].min())
        true_i_drift = float(y[n:, 1].max() - y[n:, 1].min())
        frozen = i_drift < 0.1 * true_i_drift
        verdict = "STILL FROZEN" if frozen else ("IMPROVED" if rmse[1] < 0.05 else "tracking")
        print(f"{os.path.basename(d):<20}{rmse[0]:>10.4f}{rmse[1]:>10.4f}{rmse[2]:>10.4f}"
              f"{mass.min():>10.4f}{mass.max():>10.4f}{i_drift:>10.4f}{verdict:>14}")
        true_ref = true_i_drift
    print(f"\n  true I drift over the extrapolation window: {true_ref:.4f}  (mass must be 1.0000)")
    print(f"  pre-fix reference: I_rmse 0.0776 | mass [1.0044, 1.0045] | I_drift 0.0003 (frozen)")
    print(f"  NOTE: report RMSE, not R2, for this window -- its true signal std is 0.002-0.017")


def section_flatness(runs):
    print("\n" + "=" * 104)
    print("FLATNESS  --  a low loss that stopped descending is still a failed fit")
    print("=" * 104)
    hdr = (f"{'run':<20}{'ep20%':>13}{'ep80%':>13}{'ep100%':>13}"
           f"{'ratio 80/100':>14}{'verdict':>15}")
    print(hdr)
    print("-" * len(hdr))
    for d in runs:
        hp = os.path.join(d, "training_history.json")
        if not os.path.exists(hp):
            continue
        t = json.load(open(hp))["train_losses"]
        if len(t) < 10:
            continue
        a, b, c = t[int(len(t) * 0.2) - 1], t[int(len(t) * 0.8) - 1], t[-1]
        ratio = (b / c) if (c and math.isfinite(b) and math.isfinite(c) and c > 0) else float("nan")
        # ratio = loss(80%) / loss(final):
        #   >1.05  the loss was still coming down over the last fifth of the run
        #   ~1.00  flat -- stopped improving (how the pendulum failed, ratio 1.003)
        #   <0.95  the final loss is WORSE than at 80% (late-training instability,
        #          e.g. the dt=0.2 sweep run which regressed 17x in its last 500 epochs)
        #   NaN    the run diverged to NaN (SIR pre-fix)
        if not math.isfinite(ratio):
            verdict = "NaN/diverged"
        elif ratio < 0.95:
            verdict = "REGRESSED"
        elif ratio < 1.05:
            verdict = "FLAT"
        else:
            verdict = "descending"
        print(f"{os.path.basename(d):<20}{_fmt(a):>13}{_fmt(b):>13}{_fmt(c):>13}"
              f"{_fmt(ratio, '.3f'):>14}{verdict:>15}")
    print("=" * 104)
    print("descending = still improving at the end (what you want) | FLAT = stopped (~1.00,")
    print("how the pendulum failed twice) | REGRESSED = final worse than at 80% | NaN = died.")


def main():
    ap = argparse.ArgumentParser(description="[FIX-2026-08] read the stability-fix runs")
    ap.add_argument("--root", default="results/_probe", help="results tree to read")
    ap.add_argument("--only", default="all",
                    choices=["all", "summary", "pendulum", "sir", "flatness"],
                    help="restrict to one section")
    args = ap.parse_args()

    runs = _find_runs(args.root)
    if not runs:
        print(f"No metrics.json found under '{args.root}'.")
        return
    print(f"Reading {len(runs)} run(s) under '{args.root}'")

    if args.only in ("all", "summary"):
        section_summary(runs)
    if args.only in ("all", "pendulum"):
        section_pendulum(runs)
    if args.only in ("all", "sir"):
        section_sir(runs)
    if args.only in ("all", "flatness"):
        section_flatness(runs)


if __name__ == "__main__":
    main()
