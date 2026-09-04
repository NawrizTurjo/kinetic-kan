"""
E2 -- Fitting the fixed SIR pipeline to a realistic outbreak curve.
[Phase 3 / Track E, narrowed Novelty 6]

Research question
-----------------
`docs/10_sir_root_cause_and_fix.md` rescued SIR with two changes -- `--time_scale`
and `--conserve_mode projection`. Both were derived from properties of the SIR
*generator*: its natural timescale 1/gamma, and its exact invariant S+I+R=1. Do
they transfer to data whose true equations are NOT the generator of the fit?

`load_empirical_epidemic_data()` is that case. It is an asymmetric log-normal
outbreak wave with observational noise and a 7-day moving average, built in
Phase 1 and never used in any run. Its state is 2D -- [infected, cumulative
recovered], min-max normalised to [0, 1] -- and no compartmental ODE generated
it, so there is no "true" vector field to recover.

Why this is not just "rerun SIR with a different loader"
--------------------------------------------------------
Two of the three SIR fixes turn out to be inapplicable here, and saying so with
a measurement rather than an assumption is most of the work:

  * `conserve_mode projection` pins sum(y) to sum(y0) for all time. SIR has that
    invariant exactly. This dataset does NOT: sum(y) runs from 0.0019 to 1.5400
    (measured, see `--diagnose`), because cumulative recovered is monotonically
    increasing by construction. Applying the projection would conserve the wrong
    constant -- it would hold the total at its t=0 value of 0.0027 forever.

  * `vanish_dim` gates the field by y[dim], making y[dim]=0 a manifold of
    equilibria. Here y0[0] = 0.0027 after normalisation, so the gate would
    multiply the entire field by 0.0027 at t=0 and the trajectory would barely
    move -- the pathology `--time_scale` exists to avoid.

Both are run anyway, as 2,000-epoch CONTROL arms, so those claims rest on
measured losses and not only on the algebra.

`--time_scale`, by contrast, is pure units and transfers directly. The argument
is the same four-number init-time diagnostic that cracked SIR, reproduced by
`--diagnose` before any training happens.

Usage
-----
    python run_epidemic_fit.py --diagnose               # init-time numbers only
    python run_epidemic_fit.py --tag ts24 --time_scale 24 --epochs 2000
    python run_epidemic_fit.py --collect                # figures + summary JSON

Owner: Monjur Hossain Khan (Shovon), 2105043.
"""

import argparse
import copy
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import common
from common import ensure_results_dir, split_mse

from data import load_empirical_epidemic_data
from kan import KAN, count_parameters
from ode import NeuralODE, ZeroSumField, VanishingDimField
from utils import compute_gradient_norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.titlesize": 14, "lines.linewidth": 1.8,
    "grid.alpha": 0.45, "grid.linestyle": "--",
})

C_TRUTH = "#52514e"
C_FIT = "#2a78d6"
C_ALT = "#eb6834"
C_THIRD = "#1baf7a"

RUNS_SUBDIR = "epidemic_runs"

# Held at the project's validated defaults so the only things varying across
# arms are time_scale and the structural field wrapper.
BASE = dict(layers_hidden=[2, 10, 2], grid_len=5, basis_func="rbf",
            normalizer="tanh", base_act="silu", solver="tsit5", substeps=2,
            lr=2e-3, grad_clip=1.0, seed=42)


# ==============================================================================
# Init-time diagnostic -- the four numbers, measured before any training
# ==============================================================================

def diagnose(time_scales=(1.0, 10.0, 20.0, 24.0, 30.0, 40.0)):
    """
    Reproduce docs/10's diagnostic on this dataset.

    The SIR root cause was found by measuring, at INITIALISATION: the network's
    own field magnitude, the data's derivative scale, their ratio, and where an
    untrained trajectory ends up. None of it requires a training run, and all of
    it is what justifies the time_scale choice below.
    """
    data = load_empirical_epidemic_data()
    torch.manual_seed(BASE["seed"])
    np.random.seed(BASE["seed"])
    model = KAN(layers_hidden=BASE["layers_hidden"], grid_len=BASE["grid_len"],
                basis_func=BASE["basis_func"], normalizer=BASE["normalizer"],
                base_act=BASE["base_act"])
    node = NeuralODE(func=model, method=BASE["solver"], substeps=BASE["substeps"])

    dt_train = (data.t_train[1:] - data.t_train[:-1]).unsqueeze(-1)
    true_deriv = ((data.y_train[1:] - data.y_train[:-1]) / dt_train).abs().mean().item()
    with torch.no_grad():
        init_field = model(data.y_train).abs().mean().item()

    sums = data.y_full.sum(-1)
    horizon = float(data.t_full[-1] - data.t_full[0])

    d = {
        "dataset": "load_empirical_epidemic_data()",
        "state_dim": int(data.y_full.shape[-1]),
        "state_meaning": ["infected (normalised)", "cumulative recovered (normalised)"],
        "n_train_points": int(len(data.t_train)),
        "n_full_points": int(len(data.t_full)),
        "horizon_days": horizon,
        "t_split_days": float(data.t_split),

        # --- the four numbers ---
        "init_field_magnitude": init_field,
        "true_field_magnitude": true_deriv,
        "init_over_true_ratio": init_field / true_deriv,
        "untrained_trajectory_range": None,   # filled per time_scale below

        # --- applicability of the other two SIR fixes ---
        "sum_invariant": {
            "sum_y0": float(sums[0]),
            "sum_min": float(sums.min()),
            "sum_max": float(sums.max()),
            "sum_std": float(sums.std()),
            "sum_range": float(sums.max() - sums.min()),
            "is_invariant": bool(float(sums.max() - sums.min()) < 1e-3),
        },
        "vanish_dim_gate": {
            "y0_infected": float(data.y0[0]),
            "min_infected": float(data.y_full[:, 0].min()),
            "argmin_infected_day": float(data.t_full[int(data.y_full[:, 0].argmin())]),
        },
        "time_scale_probe": [],
    }

    for ts in time_scales:
        with torch.no_grad():
            pred = node(y0=data.y0, t=data.t_full / ts)
            loss = F.mse_loss(pred[:len(data.t_train)], data.y_train).item()
        d["time_scale_probe"].append({
            "time_scale": float(ts),
            "rescaled_horizon": horizon / ts,
            "rescaled_true_deriv": true_deriv * ts,
            "untrained_traj_min": float(pred.min()),
            "untrained_traj_max": float(pred.max()),
            "epoch0_train_mse": loss,
        })
    d["untrained_trajectory_range"] = [
        d["time_scale_probe"][0]["untrained_traj_min"],
        d["time_scale_probe"][0]["untrained_traj_max"],
    ]
    return d


# ==============================================================================
# Training -- one arm
# ==============================================================================

def train_arm(tag, time_scale=1.0, conserve_projection=False, vanish_dim=None,
              epochs=2000, print_freq=250, out_root=None):
    """
    A thin training loop over the same five primitives `train_kan_ode` composes:
    KAN, NeuralODE, the data generator, Adam, and `compute_gradient_norm`.

    `train.py` is not reused directly because its dataset registry has no entry
    for `load_empirical_epidemic_data` and its CLI cannot reach the structural
    wrappers for a dataset it does not know. Per roadmap 2.2 that is a reason to
    write a small script in this folder -- not a reason to edit `train.py`.

    Carries `train.py`'s X1 non-finite gradient guard: a control arm that is
    EXPECTED to fail (projection, vanish_dim) is exactly the situation where a
    poisoned parameter set would otherwise burn the whole budget.
    """
    out_dir = os.path.join(out_root or ensure_results_dir(RUNS_SUBDIR), tag)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(BASE["seed"])
    np.random.seed(BASE["seed"])
    data = load_empirical_epidemic_data()
    torch.manual_seed(BASE["seed"])
    np.random.seed(BASE["seed"])

    n_train = len(data.t_train)
    t_train_s = data.t_train / float(time_scale)
    t_full_s = data.t_full / float(time_scale)

    model = KAN(layers_hidden=BASE["layers_hidden"], grid_len=BASE["grid_len"],
                basis_func=BASE["basis_func"], normalizer=BASE["normalizer"],
                base_act=BASE["base_act"])
    total_p, _ = count_parameters(model)

    field = model
    if conserve_projection:
        field = ZeroSumField(field)
    if vanish_dim is not None:
        field = VanishingDimField(field, dim=int(vanish_dim))
    node = NeuralODE(func=field, method=BASE["solver"], substeps=BASE["substeps"])

    config = {
        "tag": tag, "dataset": "empirical_epidemic",
        "time_scale": float(time_scale),
        "conserve_mode": "projection" if conserve_projection else None,
        "vanish_dim": int(vanish_dim) if vanish_dim is not None else None,
        "num_epochs": epochs, "parameters": total_p,
        "state_dim": int(data.y_full.shape[-1]),
        "n_train_points": n_train, "n_full_points": int(len(data.t_full)),
        "t_split": float(data.t_split), "t_end": float(data.t_full[-1]),
        "torch_version": torch.__version__,
        **{k: v for k, v in BASE.items() if k != "layers_hidden"},
        "layers_hidden": BASE["layers_hidden"],
    }

    opt = torch.optim.Adam(model.parameters(), lr=BASE["lr"])
    train_losses, grad_norms, post_clip = [], [], []
    best_loss, best_epoch, best_state = float("inf"), -1, None
    nonfinite_steps, first_nonfinite, streak, aborted = 0, None, 0, None
    epochs_run = epochs
    ABORT_STREAK = 100

    print(f"[{tag}] time_scale={time_scale} projection={conserve_projection} "
          f"vanish_dim={vanish_dim} epochs={epochs} params={total_p}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        pred = node(y0=data.y0, t=t_train_s)
        loss = F.mse_loss(pred, data.y_train)
        loss.backward()

        gnorm = compute_gradient_norm(model)
        grad_norms.append(gnorm)
        lv = loss.item()
        train_losses.append(lv)

        if lv < best_loss:
            best_loss, best_epoch = lv, epoch
            best_state = {"model_state_dict": copy.deepcopy(model.state_dict()),
                          "epoch": epoch, "train_mse": lv, "config": config}

        if math.isfinite(gnorm) and math.isfinite(lv):
            streak = 0
            if BASE["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), BASE["grad_clip"])
            post_clip.append(compute_gradient_norm(model))
            opt.step()
        else:
            nonfinite_steps += 1
            streak += 1
            if first_nonfinite is None:
                first_nonfinite = epoch
                print(f"[{tag}] non-finite gradient at epoch {epoch} "
                      f"(gnorm={gnorm}, loss={lv}); skipping the step.")
            opt.zero_grad(set_to_none=True)
            post_clip.append(0.0)

        if epoch % print_freq == 0 or epoch == 1:
            print(f"[{tag}] epoch {epoch:>6}  train={lv:.4e}  best={best_loss:.4e}  "
                  f"gnorm={gnorm:.2e}")

        if streak >= ABORT_STREAK:
            aborted, epochs_run = epoch, epoch
            print(f"[{tag}] ABORT at epoch {epoch}: {streak} consecutive non-finite steps.")
            break

    elapsed = time.time() - t0

    y_full_np = data.y_full.numpy()

    def score(state):
        if state is not None:
            model.load_state_dict(state)
        with torch.no_grad():
            p = node(y0=data.y0, t=t_full_s).cpu().numpy()
        return p, split_mse(y_full_np, p, n_train)

    final_pred, final_m = score(None)
    best_pred, best_m = score(best_state["model_state_dict"] if best_state else None)

    finite = [g for g in grad_norms if math.isfinite(g)]
    ordered = sorted(finite)
    median = ordered[len(ordered) // 2] if ordered else float("nan")

    metrics = {
        "config": config,
        "selection": {"criterion": "min_train_mse", "best_epoch": best_epoch,
                      "best_train_mse_during_training": best_loss},
        "best": best_m,
        "final": final_m,
        "training_time_seconds": elapsed,
        "seconds_per_epoch": elapsed / max(epochs_run, 1),
        "stability": {
            "grad_clip": BASE["grad_clip"],
            "nonfinite_grad_steps": nonfinite_steps,
            "first_nonfinite_epoch": first_nonfinite,
            "aborted_at_epoch": aborted,
            "epochs_run": epochs_run,
            "grad_norm_median_preclip": median,
            "grad_norm_max_preclip": (ordered[-1] if ordered else float("nan")),
            "grad_norm_spike_ratio": (ordered[-1] / median) if ordered and median > 0 else None,
        },
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump({"train_losses": train_losses, "grad_norms": grad_norms,
                   "post_clip_grad_norms": post_clip}, f)
    np.save(os.path.join(out_dir, "best_prediction.npy"), best_pred)
    if best_state is not None:
        torch.save(best_state, os.path.join(out_dir, "best_model.pt"))

    print(f"[{tag}] done in {elapsed:.1f}s | best train={best_m['train_mse']:.4e} "
          f"extrap={best_m['extrap_mse']:.4e} full={best_m['full_mse']:.4e}")
    return metrics


# ==============================================================================
# Collection & figures
# ==============================================================================

def load_runs(run_root):
    runs = {}
    for path in sorted(glob.glob(os.path.join(run_root, "*", "metrics.json"))):
        tag = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            runs[tag] = json.load(f)
        pred = os.path.join(os.path.dirname(path), "best_prediction.npy")
        if os.path.exists(pred):
            runs[tag]["_pred"] = np.load(pred)
        hist = os.path.join(os.path.dirname(path), "training_history.json")
        if os.path.exists(hist):
            with open(hist) as f:
                runs[tag]["_history"] = json.load(f)
    return runs


def plot_time_scale_sweep(runs, save_path):
    """
    Training loss against time_scale -- the confirmation half of the argument
    whose prediction came from the init-time diagnostic.
    """
    # Select the sweep arms by tag, not by epoch budget: the full run shares
    # both the winning time_scale and the plain-field settings, so a
    # budget-based filter would silently plot it as a second point at that s.
    ts_runs = sorted(
        ((r["config"]["time_scale"], tag, r) for tag, r in runs.items()
         if tag.startswith("ts")),
        key=lambda e: e[0],
    )
    if not ts_runs:
        return False

    xs = [e[0] for e in ts_runs]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for metric, color, marker, label in (
        ("train_mse", C_FIT, "o", "Training window $t \\in [0, 44]$ d"),
        ("extrap_mse", C_ALT, "s", "Extrapolation $t \\in (44, 120]$ d"),
    ):
        ys = [e[2]["best"][metric] for e in ts_runs]
        ax.plot(xs, ys, color=color, marker=marker, markersize=8,
                markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)

    best = min(ts_runs, key=lambda e: e[2]["best"]["train_mse"])
    ax.axvline(best[0], color=C_TRUTH, linestyle=":", linewidth=1.5, zorder=2)
    ax.annotate(f"selected: $s = {best[0]:g}$", (best[0], ax.get_ylim()[1]),
                xytext=(6, -12), textcoords="offset points", fontsize=9,
                color=C_TRUTH, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("Time scale $s$   (integrate on $\\tau = t/s$)")
    ax.set_ylabel("MSE at best epoch")
    ax.set_title("Time-scale sweep on the empirical outbreak (2,000-epoch probes)")
    ax.grid(True, which="both", alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {save_path}")
    return True


def plot_epidemic_fit(runs, full_tag, save_path):
    """
    The headline: the fitted trajectory against the outbreak, with the two
    inapplicable SIR fixes shown as what they are -- failed control arms.
    """
    data = load_empirical_epidemic_data()
    t, y = data.t_full.numpy(), data.y_full.numpy()
    n_train = len(data.t_train)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), dpi=150)

    ax = axes[0]
    ax.plot(t, y[:, 0], color=C_TRUTH, linewidth=2.4, label="Observed infected")
    ax.plot(t, y[:, 1], color=C_TRUTH, linewidth=2.4, alpha=0.42,
            label="Observed cumulative recovered")
    if full_tag in runs and "_pred" in runs[full_tag]:
        p = runs[full_tag]["_pred"]
        ax.plot(t, p[:, 0], color=C_FIT, linestyle="--", label="KAN-ODE infected")
        ax.plot(t, p[:, 1], color=C_THIRD, linestyle="--",
                label="KAN-ODE cumulative recovered")
    ax.axvline(t[n_train - 1], color="#0b0b0b", linestyle=":", linewidth=1.6)
    ax.annotate("train | extrapolate", (t[n_train - 1], 1.12), xytext=(5, 0),
                textcoords="offset points", fontsize=8.5, color=C_TRUTH, va="center")
    ax.set_ylim(-0.08, 1.30)
    ax.set_xlabel("day")
    ax.set_ylabel("normalised state")
    ax.set_title("(a) Fit and extrapolation, selected configuration")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.94, fontsize=8.5)

    ax = axes[1]
    ax.plot(t, y[:, 0], color=C_TRUTH, linewidth=2.4, label="Observed infected")
    arms = [(full_tag, C_FIT, "time scale only"),
            ("proj", C_ALT, "+ conserve projection"),
            ("vanish", C_THIRD, "+ vanish_dim 0")]
    for tag, color, label in arms:
        if tag in runs and "_pred" in runs[tag]:
            mse = runs[tag]["best"]["full_mse"]
            ax.plot(t, runs[tag]["_pred"][:, 0], color=color, linestyle="--",
                    label=f"{label}  (full MSE {mse:.3f})")
    ax.axvline(t[n_train - 1], color="#0b0b0b", linestyle=":", linewidth=1.6)
    ax.set_ylim(-0.08, 1.30)
    ax.set_xlabel("day")
    ax.set_ylabel("normalised infected")
    ax.set_title("(b) Control arms: the two SIR fixes that do not transfer")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.94, fontsize=8.5)

    fig.suptitle("KAN-ODE on a non-mechanistic outbreak curve", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {save_path}")


def collect(out_dir, full_tag="full"):
    run_root = os.path.join(out_dir, RUNS_SUBDIR)
    runs = load_runs(run_root)
    if not runs:
        raise SystemExit(f"No runs found under {run_root}. Train some arms first.")

    diag = diagnose()
    summary = {
        "track": "E2 -- real epidemic fit with the fixed SIR pipeline",
        "owner": {"name": "Monjur Hossain Khan (Shovon)", "id": "2105043"},
        "diagnostic": diag,
        "runs": {tag: {k: v for k, v in r.items() if not k.startswith("_")}
                 for tag, r in runs.items()},
    }

    plot_time_scale_sweep(runs, os.path.join(out_dir, "epidemic_time_scale_sweep.png"))
    if full_tag in runs:
        plot_epidemic_fit(runs, full_tag, os.path.join(out_dir, "real_epidemic_fit.png"))

    path = os.path.join(out_dir, "real_epidemic_metrics.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  table  -> {path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Track E / E2: real epidemic fit")
    ap.add_argument("--diagnose", action="store_true",
                    help="Print (and save) the init-time diagnostic; no training.")
    ap.add_argument("--collect", action="store_true",
                    help="Aggregate finished arms into figures + summary JSON.")
    ap.add_argument("--tag", default=None, help="Name of this training arm.")
    ap.add_argument("--time_scale", type=float, default=1.0)
    ap.add_argument("--conserve_projection", action="store_true")
    ap.add_argument("--vanish_dim", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--full_tag", default="full")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or ensure_results_dir()

    if args.diagnose:
        d = diagnose()
        print(json.dumps(d, indent=2))
        path = os.path.join(out_dir, "init_diagnostic.json")
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"  -> {path}")
        return

    if args.collect:
        collect(out_dir, full_tag=args.full_tag)
        return

    if not args.tag:
        ap.error("--tag is required when training an arm")
    train_arm(args.tag, time_scale=args.time_scale,
              conserve_projection=args.conserve_projection,
              vanish_dim=args.vanish_dim, epochs=args.epochs,
              out_root=os.path.join(out_dir, RUNS_SUBDIR))


if __name__ == "__main__":
    main()
