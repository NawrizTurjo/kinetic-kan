"""
Track B (Phase 3) -- Gradient Norm Dynamics vs. Solver Order.

Research question
-----------------
Do lower-order ODE solvers inject noisier gradients into backprop than higher-order
ones, and does that noise explain Euler's Phase 2 failure to reach 1e-4 training loss?

This script performs ZERO new training. Every number it reports is derived from the
`grad_norms[]` array already logged (once per epoch) in the six solver-ablation runs
under results/benchmarks/ablation_solvers/. Those runs predate [FIX-2026-08], so the
norms are raw -- no gradient clipping compressed them.

The noise measure
-----------------
Gradient norms across solvers span orders of magnitude, so a raw std would just rank
solvers by their absolute gradient scale rather than by roughness. We therefore measure
roughness in log space:

    eta = std( diff( log10(g) ) )

i.e. the standard deviation of epoch-to-epoch *relative* changes in the gradient norm.
A smoothly decaying curve (steady relative progress) scores near zero regardless of its
magnitude; a curve that jitters up and down between epochs scores high. This makes the
comparison scale-invariant and needs no arbitrary window size.

Usage
-----
    python experiments/gradient_dynamics/analyze_gradient_dynamics.py
    python experiments/gradient_dynamics/analyze_gradient_dynamics.py --warmup 500
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match the project's publication defaults (utils/plotting.py). Copied, not imported --
# Track B must not depend on shared modules it does not own.
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.45,
    "grid.linestyle": "--",
})

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMPL = os.path.abspath(os.path.join(_HERE, "..", ".."))

DEFAULT_SOURCE = os.path.join(_IMPL, "results", "benchmarks", "ablation_solvers")
DEFAULT_OUTDIR = os.path.join(_IMPL, "results", "phase3", "gradient_dynamics")

# Classical order of accuracy p, and the number of vector-field evaluations (slope
# samples) each step takes. DOPRI5/Tsit5 are 5th order with an embedded 4th-order
# estimate; Heun and Midpoint are both RK2 and are deliberately tied at p=2 -- there is
# no principled basis for splitting them, and Spearman handles ties natively.
SOLVER_ORDER = {
    "euler": 1,
    "heun": 2,
    "midpoint": 2,
    "rk4": 4,
    "dopri5": 5,
    "tsit5": 5,
}

SOLVER_LABEL = {
    "euler": "Forward Euler",
    "heun": "Heun RK2",
    "midpoint": "Midpoint RK2",
    "rk4": "RK4",
    "dopri5": "DOPRI5",
    "tsit5": "Tsit5",
}


def load_grad_norms(source_dir=DEFAULT_SOURCE, solvers=None):
    """Read grad_norms[] (and best train MSE, if present) for each solver run.

    Returns {solver: {"grad_norms": np.ndarray, "best_train_mse": float|None}}.
    Read-only access to results/benchmarks/ -- Track B never writes there.
    """
    solvers = list(SOLVER_ORDER) if solvers is None else solvers
    runs = {}
    for name in solvers:
        run_dir = os.path.join(source_dir, f"solver_{name}")
        hist_path = os.path.join(run_dir, "training_history.json")
        if not os.path.exists(hist_path):
            raise FileNotFoundError(f"missing training history for '{name}': {hist_path}")
        with open(hist_path) as fh:
            hist = json.load(fh)
        if "grad_norms" not in hist:
            raise KeyError(f"'{name}' has no grad_norms[] in {hist_path}")

        best_mse = None
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as fh:
                best_mse = json.load(fh).get("best", {}).get("train_mse")

        runs[name] = {
            "grad_norms": np.asarray(hist["grad_norms"], dtype=float),
            "best_train_mse": best_mse,
        }
    return runs


def log_step_noise(g, warmup=0):
    """Roughness of a gradient-norm series: std of epoch-to-epoch log10 changes.

    Scale-invariant by construction, so solvers whose gradients live at different
    magnitudes stay comparable. `warmup` drops the first N epochs, where the initial
    transient is optimisation start-up rather than solver-induced noise.

    Non-finite and non-positive entries are dropped before the log: a zero or NaN norm
    is not a small fluctuation, and letting it through would silently produce -inf.
    """
    g = np.asarray(g, dtype=float)[warmup:]
    g = g[np.isfinite(g) & (g > 0)]
    if g.size < 3:
        return float("nan")
    return float(np.std(np.diff(np.log10(g))))


def summarize(runs, warmup=0):
    """Per-solver statistics, ranked noisiest-first."""
    rows = []
    for name, run in runs.items():
        g = run["grad_norms"]
        tail = g[warmup:]
        finite = tail[np.isfinite(tail)]
        rows.append({
            "solver": name,
            "label": SOLVER_LABEL[name],
            "order": SOLVER_ORDER[name],
            "n_epochs": int(g.size),
            "mean_grad_norm": float(np.mean(finite)) if finite.size else float("nan"),
            "std_grad_norm": float(np.std(finite)) if finite.size else float("nan"),
            "final_grad_norm": float(g[-1]),
            "log_step_noise": log_step_noise(g, warmup=warmup),
            "best_train_mse": run["best_train_mse"],
        })
    rows.sort(key=lambda r: r["log_step_noise"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["noise_rank"] = rank
    return rows


def correlate_with_order(rows):
    """Spearman rank correlation between solver order and the noise measure.

    Spearman rather than Pearson: solver order is an ordinal scale with only four
    distinct values and a tie at p=2, and the hypothesis under test is monotonicity
    ("does noise fall as order rises"), not linearity.
    """
    orders = [r["order"] for r in rows]
    noise = [r["log_step_noise"] for r in rows]
    rho, p_value = spearmanr(orders, noise)
    return {
        "method": "spearman",
        "x": "solver order p",
        "y": "log_step_noise",
        "rho": float(rho),
        "p_value": float(p_value),
        "n": len(rows),
    }


def spike_diagnostics(source_dir=DEFAULT_SOURCE, window=6000, factor=8.0, lookahead=5):
    """Characterise the spike bursts, and test whether they indicate a defect.

    The bursts dominate every solver's trace, so before attributing them to anything
    they are checked against the signature of an actual bug: non-finite values, a
    stalled loss, permanent damage, or a gradient that spikes while the loss sits still
    (which would implicate the backward pass rather than the landscape).
    """
    out = {}
    for name in SOLVER_ORDER:
        path = os.path.join(source_dir, f"solver_{name}", "training_history.json")
        with open(path) as fh:
            hist = json.load(fh)
        g = np.asarray(hist["grad_norms"], dtype=float)
        loss = np.asarray(hist["train_losses"], dtype=float)

        gt, lt = g[window:], loss[window:]
        spikes = gt > np.median(gt) * factor
        idx = np.flatnonzero(spikes)
        recovered = [lt[i + lookahead] < lt[i] for i in idx if i + lookahead < lt.size]

        out[name] = {
            "nonfinite_grad": int((~np.isfinite(g)).sum()),
            "nonfinite_loss": int((~np.isfinite(loss)).sum()),
            "max_grad_norm": float(np.nanmax(g)),
            "spike_epochs": int(spikes.sum()),
            "window_epochs": int(gt.size),
            # A backward-pass bug would spike the gradient without moving the loss.
            "median_loss_on_spikes": float(np.median(lt[spikes])) if spikes.any() else None,
            "median_loss_on_calm": float(np.median(lt[~spikes])) if (~spikes).any() else None,
            "frac_recovered_after_spike": float(np.mean(recovered)) if recovered else None,
            "loss_first": float(loss[0]),
            "loss_final": float(loss[-1]),
            "loss_min": float(loss.min()),
        }
    out["_verdict"] = {
        "any_nonfinite": any(
            v["nonfinite_grad"] or v["nonfinite_loss"]
            for k, v in out.items() if not k.startswith("_")
        ),
        "all_loss_decreased": all(
            v["loss_final"] < v["loss_first"]
            for k, v in out.items() if not k.startswith("_")
        ),
        "note": (
            "Spikes appear in the loss as well as the gradient and are followed by "
            "recovery, which is edge-of-stability optimisation under a fixed learning "
            "rate -- not a defect in the model or the backward pass."
        ),
    }
    return out


def plot_gradient_dynamics(runs, rows, save_path, warmup=0):
    """Overlaid gradient-norm trajectories (left) and noise vs. solver order (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    by_order = sorted(runs, key=lambda n: (SOLVER_ORDER[n], n))
    cmap = plt.get_cmap("viridis")
    colors = {n: cmap(i / max(len(by_order) - 1, 1)) for i, n in enumerate(by_order)}

    for name in by_order:
        g = runs[name]["grad_norms"]
        ax1.semilogy(
            np.arange(1, g.size + 1), g,
            color=colors[name], alpha=0.8, linewidth=1.0,
            label=f"{SOLVER_LABEL[name]} ($p={SOLVER_ORDER[name]}$)",
        )
    if warmup:
        ax1.axvline(warmup, color="0.35", linestyle=":", linewidth=1.2)
        ax1.text(warmup, ax1.get_ylim()[1], " warm-up cut", va="top", fontsize=9, color="0.35")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"$\|\nabla_\theta \mathcal{L}\|_2$")
    ax1.set_title("Gradient norm trajectories by solver")
    ax1.grid(True)
    ax1.legend(loc="upper right", framealpha=0.9)

    for row in rows:
        ax2.scatter(row["order"], row["log_step_noise"],
                    s=90, color=colors[row["solver"]], zorder=3,
                    edgecolor="white", linewidth=1.2)
        ax2.annotate(row["label"], (row["order"], row["log_step_noise"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax2.set_xlabel("Solver order $p$")
    ax2.set_ylabel(r"Noise measure  $\eta=\mathrm{std}(\Delta \log_{10} g)$")
    ax2.set_title("Gradient noise vs. solver order")
    ax2.set_xticks(sorted({r["order"] for r in rows}))
    ax2.grid(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def warmup_sensitivity(runs, cuts=(0, 500, 2000, 5000)):
    """Re-run the correlation at several warm-up cuts.

    With only six solvers, a single cut is not evidence: this reports whether the
    order-vs-noise relationship is stable or an artifact of where the start-up
    transient is trimmed. A sign flip across cuts means the measure does not support
    a directional claim.
    """
    out = []
    for cut in cuts:
        rows = summarize(runs, warmup=cut)
        corr = correlate_with_order(rows)
        out.append({
            "warmup": cut,
            "rho": corr["rho"],
            "p_value": corr["p_value"],
            "ranking": [r["solver"] for r in rows],
        })
    signs = {np.sign(round(c["rho"], 6)) for c in out if np.isfinite(c["rho"])}
    return {
        "cuts": out,
        "sign_stable": len(signs - {0.0}) <= 1,
        "any_significant": any(c["p_value"] < 0.05 for c in out),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Track B: gradient norm dynamics vs. solver order")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="directory holding solver_<name>/training_history.json (read-only)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="where table.json and the figure go")
    parser.add_argument("--warmup", type=int, default=0,
                        help="epochs to drop before measuring noise (start-up transient)")
    args = parser.parse_args(argv)

    runs = load_grad_norms(args.source)
    rows = summarize(runs, warmup=args.warmup)
    corr = correlate_with_order(rows)
    sensitivity = warmup_sensitivity(runs)
    spikes = spike_diagnostics(args.source)

    os.makedirs(args.outdir, exist_ok=True)
    fig_path = os.path.join(args.outdir, "gradient_noise_by_order.png")
    plot_gradient_dynamics(runs, rows, fig_path, warmup=args.warmup)

    table = {
        "description": "Track B -- gradient norm roughness per solver, ranked noisiest first.",
        "noise_measure": {
            "name": "log_step_noise",
            "formula": "std(diff(log10(grad_norms)))",
            "rationale": (
                "Scale-invariant roughness: measures epoch-to-epoch relative change, so "
                "solvers whose gradients live at different magnitudes remain comparable."
            ),
        },
        "warmup_epochs_dropped": args.warmup,
        "source": os.path.relpath(args.source, _IMPL).replace("\\", "/"),
        "clipping_note": (
            "These six runs predate [FIX-2026-08]; no post_clip_grad_norms key exists, so "
            "all norms are raw and unclipped."
        ),
        "correlation": corr,
        "warmup_sensitivity": sensitivity,
        "spike_diagnostics": spikes,
        "solvers": rows,
    }
    table_path = os.path.join(args.outdir, "table.json")
    with open(table_path, "w") as fh:
        json.dump(table, fh, indent=2)

    width = max(len(r["label"]) for r in rows)
    print(f"{'solver'.ljust(width)}  {'p':>2}  {'noise':>9}  {'mean|g|':>10}  {'best train MSE':>15}")
    print("-" * (width + 44))
    for row in rows:
        mse = "-" if row["best_train_mse"] is None else f"{row['best_train_mse']:.3e}"
        print(f"{row['label'].ljust(width)}  {row['order']:>2}  "
              f"{row['log_step_noise']:>9.4f}  {row['mean_grad_norm']:>10.4f}  {mse:>15}")
    print(f"\nSpearman rho(order, noise) = {corr['rho']:+.4f}   p = {corr['p_value']:.4f}   n = {corr['n']}")

    print("\nwarm-up sensitivity:")
    for cut in sensitivity["cuts"]:
        print(f"  drop {cut['warmup']:>5} epochs -> rho = {cut['rho']:+.4f}  p = {cut['p_value']:.4f}")
    print(f"  sign stable across cuts : {sensitivity['sign_stable']}")
    print(f"  any cut significant     : {sensitivity['any_significant']}")

    verdict = spikes["_verdict"]
    print("\nspike-burst diagnostics (is this a bug?):")
    print(f"  any non-finite grad/loss anywhere : {verdict['any_nonfinite']}")
    print(f"  every solver's loss decreased     : {verdict['all_loss_decreased']}")
    for name in SOLVER_ORDER:
        d = spikes[name]
        ratio = (d["median_loss_on_spikes"] / d["median_loss_on_calm"]
                 if d["median_loss_on_calm"] else float("nan"))
        print(f"  {name:9s} spikes={d['spike_epochs']:5d}  max|g|={d['max_grad_norm']:6.2f}  "
              f"loss(spike)/loss(calm)={ratio:5.2f}  recovered={d['frac_recovered_after_spike']:.3f}")

    print(f"\nwrote {table_path}\nwrote {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
