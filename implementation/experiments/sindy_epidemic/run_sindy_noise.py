"""
E1 -- SINDy vs. KAN-ODE under observational noise, plus L1 edge pruning.
[Phase 3 / Track E, Novelty 4]

Research question
-----------------
SINDy recovers dynamics by sparse regression against a fixed symbolic library;
KAN-ODE recovers them by gradient descent through a differentiable solver. They
fail for different reasons. Which degrades more gracefully as observational
noise rises?

Protocol -- what makes this apples-to-apples
--------------------------------------------
Both methods see the SAME data. The Lotka-Volterra generator, horizon, split and
seed are read out of each KAN checkpoint's own `config` rather than restated
here, so the trajectory SINDy fits is bit-identical to the one the Phase-2 run
at that sigma trained on. Both are then scored the same three ways `train.py`
scores everything (train / extrap / full), against the CLEAN ground truth -- the
noise is a corruption of the observations, never of the target.

The KAN numbers are the existing `results/benchmarks/noise/sigma*` checkpoints,
re-integrated, not retrained. Nothing in Phase 2 is re-run.

Two SINDy configurations are fit at every sigma:
  * `fd`       -- plain finite differences. The honest default.
  * `smoothed` -- `SmoothedFiniteDifference`, SINDy's own noise defence.
Reporting only the first would understate SINDy; reporting only the second
would compare a denoised pipeline against a non-denoised one. Both are shown.

A conceptual asymmetry, stated up front (see the write-up for the full argument)
-------------------------------------------------------------------------------
A KAN edge is strictly univariate -- it sees one input scalar. The true LV field
contains a genuine bilinear cross-term beta*x*y, which a single additive layer of
univariate edges cannot express and must approximate by composition across
layers. SINDy's degree-2 polynomial library contains `x y` as a literal
candidate and can recover it exactly. So a term-by-term symbolic comparison is
not a fair contest, and this script does not stage one: the headline metric is
trajectory reconstruction under noise, with coefficient recovery reported for
SINDy alone as a diagnostic of where its sparse regression breaks down.

Usage
-----
    python run_sindy_noise.py            # full E1: SINDy + pruning + figures

Owner: Monjur Hossain Khan (Shovon), 2105043.
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch

import common  # sets up sys.path for kan/ode/data/utils
from common import NOISE_LEVELS, NOISE_SWEEP_DIR, ensure_results_dir, integrate, split_mse
from pruning import prune_edges, surviving_edge_report

from data import generate_lotka_volterra_data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match the project's publication defaults (utils/plotting.py) so Track E figures
# sit alongside the existing ones without a visible style break.
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.45,
    "grid.linestyle": "--",
})

# Validated categorical slots 1-3 (blue / orange / aqua). Every pair clears the
# all-pairs CVD and normal-vision floors; aqua sits under 3:1 on a light surface,
# so every series is also direct-labeled and distinctly marked.
C_SINDY_FD = "#2a78d6"
C_SINDY_SM = "#eb6834"
C_KAN = "#1baf7a"

# Ordinal blue ramp for the pruning levels (magnitude, not identity). Starts at
# step 250, the lightest step that still clears 2:1 on a light surface.
RAMP_BLUE = ["#86b6ef", "#3987e5", "#256abf", "#0d366b"]

# The four terms actually present in the true Lotka-Volterra field, in pysindy's
# PolynomialLibrary(degree=2) feature naming:
#     dx/dt = alpha*x - beta*x*y     ->  x: +alpha,  x y: -beta
#     dy/dt = delta*x*y - gamma*y    ->  y: -gamma,  x y: +delta
TRUE_TERMS = {
    0: {"x": ("alpha", +1.0), "x y": ("beta", -1.0)},
    1: {"y": ("gamma", -1.0), "x y": ("delta", +1.0)},
}

# Each layer holds 20 edges (2->10 and 10->2), so a per-layer percentile of 5
# removes exactly the single weakest edge in each layer. The low end of this
# sweep is what locates the point where pruning first costs anything at all --
# a coarse 25/50/75 grid would only show that it is already catastrophic.
PRUNE_PERCENTILES = [0.0, 5.0, 10.0, 25.0, 50.0, 75.0]


# ==============================================================================
# SINDy
# ==============================================================================

def fit_sindy(t_train, y_train, differentiation="fd", threshold=0.05, degree=2):
    """
    Fit one SINDy model on the training window.

    Args:
        differentiation: "fd" (plain finite differences) or "smoothed"
            (`SmoothedFiniteDifference`, SINDy's built-in noise defence).
        threshold: STLSQ sparsity threshold, held FIXED across noise levels so
            the sweep measures noise sensitivity and not threshold tuning.
    """
    import pysindy as ps

    if differentiation == "smoothed":
        diff = ps.SmoothedFiniteDifference()
    elif differentiation == "fd":
        diff = ps.FiniteDifference()
    else:
        raise ValueError(f"Unknown differentiation '{differentiation}'")

    model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=degree),
        optimizer=ps.STLSQ(threshold=threshold),
        differentiation_method=diff,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(np.asarray(y_train, dtype=float), t=np.asarray(t_train, dtype=float),
                  feature_names=["x", "y"])
    return model


def simulate_sindy(model, y0, t_full):
    """
    Integrate a discovered SINDy model over the full horizon.

    A sparse model fit to noisy derivatives is not guaranteed to be stable --
    at high sigma it can pick up a spurious positive-feedback term and run away.
    That is a genuine result about the method, not an error to suppress, so a
    failed or non-finite integration is caught and reported as such rather than
    crashing the sweep.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = np.asarray(model.simulate(np.asarray(y0, dtype=float),
                                            np.asarray(t_full, dtype=float)))
        if sim.shape[0] != len(t_full):
            # Some pysindy versions return len(t)-1 points; pad so downstream
            # slicing stays aligned with the ground-truth array.
            pad = np.repeat(sim[-1:], len(t_full) - sim.shape[0], axis=0)
            sim = np.concatenate([sim, pad], axis=0)
        return sim, ("ok" if np.isfinite(sim).all() else "non_finite")
    except Exception as exc:                                   # noqa: BLE001
        return np.full((len(t_full), 2), np.nan), f"integration_failed: {type(exc).__name__}"


def coefficient_recovery(model, true_params):
    """
    How close SINDy's recovered coefficients are to the generator's own.

    Reports, per true term, the recovered value and its relative error, plus the
    count of spurious terms (nonzero coefficients on library features that do
    not appear in the true field) -- the second number matters as much as the
    first, since a sparse method that recovers the right terms AND invents five
    others has not really recovered the system.
    """
    names = list(model.get_feature_names())
    coefs = np.asarray(model.coefficients())            # [n_eq, n_features]

    out = {"terms": {}, "spurious_terms": 0, "spurious_detail": []}
    max_rel = 0.0
    for eq, terms in TRUE_TERMS.items():
        for feat, (param, sign) in terms.items():
            true_val = sign * float(true_params[param])
            got = float(coefs[eq, names.index(feat)]) if feat in names else 0.0
            rel = abs(got - true_val) / abs(true_val)
            max_rel = max(max_rel, rel)
            out["terms"][f"eq{eq}:{feat}"] = {
                "true": true_val, "recovered": got, "rel_error": rel,
            }

    for eq in range(coefs.shape[0]):
        expected = set(TRUE_TERMS.get(eq, {}).keys())
        for j, feat in enumerate(names):
            if feat not in expected and abs(coefs[eq, j]) > 1e-8:
                out["spurious_terms"] += 1
                out["spurious_detail"].append(
                    {"eq": eq, "feature": feat, "coef": float(coefs[eq, j])}
                )
    out["max_rel_error"] = max_rel
    return out


# ==============================================================================
# Sweep
# ==============================================================================

def run_sweep(prune_percentiles=None):
    prune_percentiles = prune_percentiles or PRUNE_PERCENTILES
    import pysindy as ps

    results = {
        "track": "E1 -- SINDy comparison under noise + L1 edge pruning",
        "owner": {"name": "Monjur Hossain Khan (Shovon)", "id": "2105043"},
        "pysindy_version": ps.__version__,
        "torch_version": torch.__version__,
        "sindy_config": {
            "library": "PolynomialLibrary(degree=2)",
            "optimizer": "STLSQ(threshold=0.05)",
            "note": "threshold held fixed across sigma so the sweep measures "
                    "noise sensitivity, not per-level tuning",
        },
        "prune_percentiles": prune_percentiles,
        "levels": [],
    }

    for sigma, slug in NOISE_LEVELS:
        ckpt_path = os.path.join(NOISE_SWEEP_DIR, slug, "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Missing Phase-2 checkpoint {ckpt_path}. Track E reuses the "
                f"existing noise sweep and never retrains it."
            )

        model, config = common.load_kan_checkpoint(ckpt_path)

        # Rebuild the data from the CHECKPOINT's config, so SINDy is handed the
        # exact trajectory this KAN was trained on -- same seed, same noise draw.
        p = config.get("data_params", {})
        data = generate_lotka_volterra_data(
            alpha=p.get("alpha", 1.5), beta=p.get("beta", 1.0),
            gamma=p.get("gamma", 3.0), delta=p.get("delta", 1.0),
            t_start=config.get("t_start", 0.0), t_end=config.get("t_end", 14.0),
            dt=config.get("dt", 0.1), t_train_end=config.get("t_train_end", 3.5),
            noise_std=config.get("noise_std", sigma), seed=config.get("seed", 42),
        )
        n_train = len(data.t_train)
        y_full_np = data.y_full.numpy()
        t_full_np = data.t_full.numpy()

        entry = {
            "sigma": sigma,
            "checkpoint": os.path.relpath(ckpt_path, common.IMPL_ROOT).replace("\\", "/"),
            "n_train_points": n_train,
            "n_full_points": len(t_full_np),
            "sindy": {},
            "kan": {},
        }

        # ---- SINDy, both differentiation settings ------------------------
        for diff in ("fd", "smoothed"):
            sm = fit_sindy(data.t_train.numpy(), data.y_train.numpy(), differentiation=diff)
            sim, status = simulate_sindy(sm, data.y0.numpy(), t_full_np)
            eqs = [str(e) for e in sm.equations(precision=4)]
            entry["sindy"][diff] = {
                "equations": eqs,
                "simulation_status": status,
                **split_mse(y_full_np, sim, n_train),
                "coefficients": coefficient_recovery(sm, data.params),
            }
            entry["sindy"][diff]["_traj"] = sim  # stripped before serialising
            print(f"  [sigma={sigma:<4} SINDy/{diff:<8}] {status:<28} "
                  f"full_mse={entry['sindy'][diff]['full_mse']:.4e}  "
                  f"max_coef_rel_err={entry['sindy'][diff]['coefficients']['max_rel_error']:.3f}  "
                  f"spurious={entry['sindy'][diff]['coefficients']['spurious_terms']}")
            for e in eqs:
                print(f"        {e}")

        # ---- KAN: unpruned, then pruned at each level --------------------
        for prune_base in (False, True):
            key = "prune_base" if prune_base else "spline_only"
            entry["kan"][key] = []
            for pct in prune_percentiles:
                # prune_edges deep-copies, so `model` stays the untouched baseline
                # across every (prune_base, percentile) combination below.
                pm, info = prune_edges(model, threshold_percentile=pct,
                                       prune_base=prune_base)
                pred = integrate(pm, config, data.y0, data.t_full)
                rec = {
                    "threshold_percentile": pct,
                    "sparsity": info["sparsity"],
                    "edges_pruned": info["total_edges_pruned"],
                    "total_edges": info["total_edges"],
                    "magnitude_cutoffs": [l["magnitude_cutoff"] for l in info["layers"]],
                    **split_mse(y_full_np, pred, n_train),
                }
                if pct == prune_percentiles[-1]:
                    rec["surviving_edges"] = surviving_edge_report(pm)
                rec["_traj"] = pred
                entry["kan"][key].append(rec)
                print(f"  [sigma={sigma:<4} KAN/{key:<11} prune={pct:>5.1f}%] "
                      f"train={rec['train_mse']:.4e} extrap={rec['extrap_mse']:.4e} "
                      f"full={rec['full_mse']:.4e}")

        entry["_data"] = {"t_full": t_full_np, "y_full": y_full_np,
                          "y_train": data.y_train.numpy(), "n_train": n_train}
        results["levels"].append(entry)

    return results


# ==============================================================================
# Figures
# ==============================================================================

def place_end_labels(ax, entries, x_pad_frac=0.42, min_sep_frac=0.075):
    """
    Direct-label each series at its last finite point, without collisions.

    Direct labels are what discharges the relief obligation on the low-contrast
    aqua slot, but naively annotating the final point overlaps whenever two
    series finish close together -- which, on these log axes, they do.

    Positions are computed and separated in AXES-FRACTION space, not data or
    display space. That is the detail that makes this hold: `tight_layout` and
    `bbox_inches="tight"` both resize the axes box after the labels are placed,
    which rescales any display-space gap and silently re-collides the labels;
    a fractional gap is invariant under exactly that resize.

    Args:
        entries: list of (label, color, x, y) anchor points, in data coordinates.
        x_pad_frac: extra x-range reserved for the labels, as a fraction of the
            current span -- so labels sit inside the axes and cannot collide
            with a neighbouring panel.
        min_sep_frac: minimum vertical gap between two labels, as a fraction of
            the axes height.
    """
    entries = [(lb, c, x, y) for lb, c, x, y in entries if y is not None and np.isfinite(y)]
    if not entries:
        return

    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + (x1 - x0) * x_pad_frac)

    to_axes = ax.transAxes.inverted().transform
    placed = sorted(
        ((lb, x, y, to_axes(ax.transData.transform((x, y)))) for lb, _c, x, y in entries),
        key=lambda e: e[3][1],
    )
    ys = [p[3][1] for p in placed]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_sep_frac)

    # De-collision only ever pushes labels UP, so a tight cluster near the top
    # can walk the stack off the axes. Slide it back down as a unit, then clamp.
    overflow = ys[-1] - (1.0 - min_sep_frac / 2)
    if overflow > 0:
        ys = [y - overflow for y in ys]
    ys = [min(max(y, min_sep_frac / 2), 1.0 - min_sep_frac / 2) for y in ys]

    # Anchor every label to a common x just right of the last data point, so the
    # leader lines read as a single column rather than a ragged edge.
    x_lab = max(p[3][0] for p in placed) + 0.03
    for (lb, x, y, _), y_frac in zip(placed, ys):
        ax.annotate(
            lb, xy=(x, y), xycoords="data",
            xytext=(x_lab, y_frac), textcoords=ax.transAxes,
            va="center", ha="left", fontsize=8.5, color="#52514e",
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color="#c9c8c2", linewidth=0.8,
                            shrinkA=0, shrinkB=3),
        )


def plot_noise_comparison(results, save_path):
    """
    Three panels, one axis each, sharing an x of sigma:
      (a) training-window reconstruction MSE
      (b) extrapolation MSE -- the claim the project actually cares about
      (c) SINDy coefficient recovery: worst relative error on the four true terms
    KAN appears in (a) and (b) as the unpruned checkpoint; panel (c) is
    SINDy-only by construction, since a KAN has no symbolic coefficients to
    recover (see the univariate-edge argument in the module docstring).
    """
    sigmas = [lvl["sigma"] for lvl in results["levels"]]
    x = np.arange(len(sigmas))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=150)

    series = [
        ("SINDy (finite diff.)", C_SINDY_FD, "o", "-",
         lambda lvl, k: lvl["sindy"]["fd"][k]),
        ("SINDy (smoothed FD)", C_SINDY_SM, "s", "-",
         lambda lvl, k: lvl["sindy"]["smoothed"][k]),
        ("KAN-ODE (unpruned)", C_KAN, "D", "--",
         lambda lvl, k: next(r for r in lvl["kan"]["spline_only"]
                             if r["threshold_percentile"] == 0.0)[k]),
    ]

    for ax, metric, title in (
        (axes[0], "train_mse", "(a) Training window  $t \\in [0, 3.5]$"),
        (axes[1], "extrap_mse", "(b) Extrapolation  $t \\in (3.5, 14]$"),
    ):
        labels, diverged = [], []
        for label, color, marker, ls, getter in series:
            vals = [getter(lvl, metric) for lvl in results["levels"]]
            plotted = [v if (v is not None and np.isfinite(v)) else np.nan for v in vals]
            ax.plot(x, plotted, color=color, marker=marker, linestyle=ls,
                    markersize=8, markeredgecolor="white", markeredgewidth=1.2,
                    label=label, zorder=3)
            finite = [i for i, v in enumerate(plotted) if np.isfinite(v)]
            if finite:
                labels.append((label, color, x[finite[-1]], plotted[finite[-1]]))
            diverged += [i for i, v in enumerate(plotted) if not np.isfinite(v)]

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:g}" for s in sigmas])
        ax.set_xlabel("Observation noise $\\sigma$")
        ax.set_ylabel("MSE vs. clean ground truth")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", framealpha=0.92)
        # Flag any level whose discovered model failed to integrate at all.
        for i in sorted(set(diverged)):
            ax.annotate("diverged", (x[i], ax.get_ylim()[1]), ha="center", va="top",
                        fontsize=8, color="#e34948")
        # Direct labels last, so the axis limits they extend are final.
        place_end_labels(ax, labels)

    ax = axes[2]
    labels = []
    for label, color, marker, key in (
        ("SINDy (finite diff.)", C_SINDY_FD, "o", "fd"),
        ("SINDy (smoothed FD)", C_SINDY_SM, "s", "smoothed"),
    ):
        vals = [lvl["sindy"][key]["coefficients"]["max_rel_error"]
                for lvl in results["levels"]]
        ax.plot(x, vals, color=color, marker=marker, markersize=8,
                markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
        labels.append((label, color, x[-1], vals[-1]))
    ax.axhline(0.10, color="#52514e", linestyle=":", linewidth=1.4, zorder=2)
    ax.annotate("10% recovery error", (0.03, 0.105), xycoords=("axes fraction", "data"),
                fontsize=8.5, color="#52514e")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:g}" for s in sigmas])
    ax.set_xlabel("Observation noise $\\sigma$")
    ax.set_ylabel("Worst relative error, 4 true terms")
    ax.set_title("(c) SINDy coefficient recovery")
    ax.grid(True, which="both", alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", framealpha=0.92)
    place_end_labels(ax, labels)

    fig.suptitle("SINDy vs. KAN-ODE robustness to observational noise "
                 "(Lotka-Volterra, identical data)", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {save_path}")


def plot_pruning(results, save_path):
    """
    Pruning level (magnitude, so a one-hue ordinal ramp) against reconstruction
    error, faceted by whether the residual base branch was severed too. One
    y-axis per panel; both panels share it so the two facets are comparable.
    """
    pcts = results["prune_percentiles"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150, sharey=True)

    for ax, key, title in (
        (axes[0], "spline_only",
         "(a) Spline branch only\n$C \\to 0$, residual $W\\,b(x)$ kept"),
        (axes[1], "prune_base",
         "(b) Whole edge\n$C \\to 0$ and $W \\to 0$"),
    ):
        for j, lvl in enumerate(results["levels"]):
            color = RAMP_BLUE[min(j, len(RAMP_BLUE) - 1)]
            vals = [next(r for r in lvl["kan"][key]
                         if r["threshold_percentile"] == p)["full_mse"] for p in pcts]
            ax.plot(pcts, vals, color=color, marker="o", markersize=7,
                    markeredgecolor="white", markeredgewidth=1.1,
                    label=f"$\\sigma = {lvl['sigma']:g}$", zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(pcts)
        ax.set_xlabel("Edges pruned (percentile of $L_1$ edge magnitude)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xlim(-4, pcts[-1] + 4)
        # No direct labels here: sigma is an ORDERED magnitude carried by the
        # one-hue ramp, so the legend reads as a scale. Four labels on curves
        # that all converge to the same plateau would be noise, not relief --
        # and sindy_vs_kan_noise.json is the table view.

    axes[0].set_ylabel("Full-horizon MSE vs. clean ground truth")
    axes[0].legend(loc="upper left", framealpha=0.92, title="Training noise")
    fig.suptitle("Effect of $L_1$ edge pruning on the Phase-2 noise-sweep checkpoints "
                 "(no retraining)", y=1.03)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {save_path}")


def plot_trajectories(results, save_path):
    """
    What the numbers in panel (b) actually look like: true trajectory against
    each method's reconstruction at the cleanest and the noisiest sigma.
    """
    levels = [results["levels"][0], results["levels"][-1]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), dpi=150)

    for ax, lvl in zip(axes, levels):
        d = lvl["_data"]
        t, y = d["t_full"], d["y_full"]
        n_train = d["n_train"]
        ax.plot(t, y[:, 0], color="#52514e", linewidth=2.4, alpha=0.85, label="True $x$ (prey)")
        ax.plot(t, y[:, 1], color="#52514e", linewidth=2.4, alpha=0.45,
                linestyle="-", label="True $y$ (predator)")
        ax.scatter(t[:n_train], d["y_train"][:, 0], s=10, color="#0b0b0b",
                   alpha=0.55, zorder=4, label="Observed $x$ (noisy)")

        # Plain finite differences, i.e. SINDy's BETTER variant at every sigma
        # here -- showing the weaker one would stack the picture against it.
        tr = lvl["sindy"]["fd"]["_traj"]
        ax.plot(t, tr[:, 0], color=C_SINDY_FD, linestyle="--",
                label="SINDy (finite diff.) $x$")
        kan = next(r for r in lvl["kan"]["spline_only"] if r["threshold_percentile"] == 0.0)
        ax.plot(t, kan["_traj"][:, 0], color=C_KAN, linestyle="--", label="KAN-ODE $x$")

        # Headroom above the data so the legend never sits on a peak.
        top = float(np.nanmax(y)) * 1.75
        ax.set_ylim(-0.6, top)
        ax.axvline(t[n_train - 1], color="#0b0b0b", linestyle=":", linewidth=1.6)
        ax.annotate("train | extrapolate", (t[n_train - 1], float(np.nanmax(y)) * 1.06),
                    xytext=(5, 0), textcoords="offset points", fontsize=8.5,
                    color="#52514e", va="center")
        ax.set_xlabel("$t$")
        ax.set_ylabel("state")
        ax.set_title(f"$\\sigma = {lvl['sigma']:g}$")
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95, ncol=2)

    fig.suptitle("Prey-component reconstruction: cleanest vs. noisiest level", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {save_path}")


# ==============================================================================
# Serialisation
# ==============================================================================

def strip_arrays(obj):
    """Drop the `_`-prefixed trajectory/data payloads before writing JSON."""
    if isinstance(obj, dict):
        return {k: strip_arrays(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_arrays(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main():
    ap = argparse.ArgumentParser(description="Track E / E1: SINDy vs KAN under noise")
    ap.add_argument("--out", default=None, help="Output directory (default: this track's)")
    args = ap.parse_args()

    out_dir = args.out or ensure_results_dir()
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("E1: SINDy vs KAN-ODE under observational noise + L1 edge pruning")
    print("=" * 78)

    results = run_sweep()

    plot_noise_comparison(results, os.path.join(out_dir, "sindy_vs_kan_noise.png"))
    plot_pruning(results, os.path.join(out_dir, "pruning_degradation.png"))
    plot_trajectories(results, os.path.join(out_dir, "sindy_vs_kan_trajectories.png"))

    json_path = os.path.join(out_dir, "sindy_vs_kan_noise.json")
    with open(json_path, "w") as f:
        json.dump(strip_arrays(results), f, indent=2)
    print(f"  table  -> {json_path}")
    print("E1 complete.")


if __name__ == "__main__":
    main()
