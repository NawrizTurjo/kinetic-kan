"""
Decisive follow-up experiment: does the paper's LITERAL recipe train?

Every prior run of `[2,50,2]+tanh` (`mlpode_baseline` at 10k, `mlp_paperspec_50k`
at 50k) used this project's DEFAULT learning rate (2e-3) and DEFAULT
initialization (plain Glorot uniform, gain=1.0) for every model, KAN or MLP
alike. Reading the base paper's own official Julia repository turned up two
settings that differ from that default, specifically for its MLP baseline,
and neither has ever been tested:

    Setting              Paper's own code        This project's default
    -------------------  -----------------------  -----------------------
    Learning rate (MLP)  1e-2                      2e-3   (5x smaller)
    Initial weight scale Glorot / 1e5 (near-zero)  Glorot, gain=1.0 (full)

This script changes ONLY those two things -- architecture ([2,50,2]), activation
(tanh), dataset (Lotka-Volterra), solver (fixed-step Tsit5, substeps=2), and
seed (42) are all IDENTICAL to `mlpode_baseline`/`mlp_paperspec_50k`, so any
change in outcome is attributable to LR + init-scale alone, not a confound.

Deliberately NOT included: gradient clipping. The paper's own code does not
clip, and clipping would change the optimization dynamics being tested. The
one safety net kept is the project's standard non-finite-gradient guard
(skip a poisoned step, abort after 100 consecutive ones) -- it is a no-op on
any healthy run and only prevents wasting the whole run on a silent NaN.

`MLP_ODE`'s `init_scale` constructor argument already implements the /1e5
rescale exactly (it's a gain multiplier inside `nn.init.xavier_uniform_`,
mathematically identical to "sample Glorot, then divide by 1e5") -- nothing
in kan/mlp.py needed to change. `train.py` itself does not expose
`init_scale` on its CLI/`train_kan_ode()` signature, so rather than edit that
shared file, this script reuses the same primitives `train_kan_ode` itself
composes (`MLP_ODE`, `NeuralODE`, `generate_lotka_volterra_data`, the metrics
functions) directly, following this project's own established pattern for
one-off experiments that need a capability the CLI doesn't expose (see
`experiments/stiffness_map/run_sweep.py` for precedent). No file outside this
script is touched.

USAGE (Kaggle): upload/mount the repo as a dataset, adjust REPO_ROOT below,
then Run All. Output is zipped at the end exactly like the other
`kaggle_50k_*.py` scripts, for the same download-and-extract workflow.
USAGE (local smoke test): just run `python kaggle_mlp_paperspec_exact_recipe.py`
with REPO_ROOT pointing at your local `kinetic-kan` checkout.
"""
import os

# ==============================================================================
# CONFIGURE THIS before running
# ==============================================================================
REPO_ROOT = "/kaggle/input/datasets/shamshossain/kinetic-kan/kinetic-kan"  # <-- your confirmed working mount path
SAVE_DIR = "/kaggle/working/implementation/results/phase4/paper_recipe_audit/mlp_paperspec_exact_10k"
NUM_EPOCHS = 10000          # you're running 10k first; bump to 50000 later if this trains
LR = 1e-2                   # paper's own Julia code, MLP baseline
INIT_SCALE = 1e-5           # paper's own Julia code: Glorot weights divided by 1e5
DEVICE = "cpu"              # match every other run in this project (CPU-only comparisons)
SEED = 42
PRINT_EVERY = 500

# Reference metrics.json files to diff against, if present locally (both use
# this project's DEFAULT lr=2e-3, init_scale=1.0 -- the only two things this
# script changes). Comparison is best-effort; a missing file is not fatal.
REFERENCE_10K = os.path.join(
    REPO_ROOT, "implementation", "results", "benchmarks", "mlpode_baseline", "metrics.json",
)
REFERENCE_50K = os.path.join(
    REPO_ROOT, "implementation", "results", "phase4", "epoch_budget_check",
    "mlp_paperspec_50k", "metrics.json",
)

import sys
import json
import copy
import math
import shutil
import time

_impl_dir = os.path.join(REPO_ROOT, "implementation")
sys.path.insert(0, _impl_dir)

# Fail loudly and specifically here, instead of a bare "No module named 'kan'"
# several frames deep -- that error means REPO_ROOT doesn't actually point at
# a checkout containing implementation/kan/, almost always a stale/incorrect
# mount path (Kaggle's actual dataset mount path can differ from what's typed
# above -- check the notebook's Input panel on the right for the exact path,
# it usually looks like /kaggle/input/<dataset-slug>/<repo-folder-name>).
if not os.path.isdir(os.path.join(_impl_dir, "kan")):
    print(f"REPO_ROOT does not resolve to a valid checkout.")
    print(f"  REPO_ROOT       = {REPO_ROOT}")
    print(f"  looked for      = {os.path.join(_impl_dir, 'kan')}  (not found)")
    if os.path.isdir("/kaggle/input"):
        print(f"  contents of /kaggle/input: {os.listdir('/kaggle/input')}")
    raise FileNotFoundError(
        f"Fix REPO_ROOT at the top of this script to your dataset's actual "
        f"mount path (see the Input panel in the Kaggle notebook UI), then re-run."
    )

import torch
import numpy as np
import torch.nn.functional as F

from kan import MLP_ODE, count_parameters
from ode import NeuralODE
from data import generate_lotka_volterra_data
from utils import (
    compute_mse, compute_rmse, compute_mae,
    compute_r2_score, compute_relative_l2_error,
    compute_gradient_norm, estimate_lipschitz_bound, track_nfe,
)

NONFINITE_ABORT_STREAK = 100


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device(DEVICE)
    print(f"Detected {os.cpu_count()} CPU core(s).")
    print(f"Testing the PAPER-EXACT recipe: [2,50,2]+tanh, lr={LR}, init_scale={INIT_SCALE}, "
          f"{NUM_EPOCHS} epochs. Only lr and init_scale differ from mlpode_baseline/"
          f"mlp_paperspec_50k -- watch whether train_mse actually descends this time.")

    # Same seeding discipline as train.py: seed once for data, re-seed before
    # model init so model init is governed by SEED alone.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data = generate_lotka_volterra_data(seed=SEED)  # all defaults == kanode_flagship's own config
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t_train = data.t_train.to(device)
    t_full = data.t_full.to(device)
    y_train = data.y_train.to(device)
    y_full = data.y_full.to(device)
    y0_init = data.y0.to(device)
    n_train = len(t_train)

    model = MLP_ODE(layers_hidden=[2, 50, 2], activation="tanh", init_scale=INIT_SCALE).to(device)
    total_p, _ = count_parameters(model)
    print(f"Model: MLP-ODE [2,50,2] tanh, {total_p} parameters, init_scale={INIT_SCALE}")

    node = NeuralODE(func=model, method="tsit5", substeps=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    run_config = {
        "model_type": "mlp", "dataset": "lotka_volterra",
        "layers_hidden": [2, 50, 2], "mlp_act": "tanh",
        "solver": "tsit5", "substeps": 2,
        "lr": LR, "init_scale": INIT_SCALE, "num_epochs": NUM_EPOCHS,
        "seed": SEED, "parameters": total_p,
        "t_start": 0.0, "t_end": float(t_full[-1].item()),
        "t_train_end": float(t_train[-1].item()),
        "n_train_points": n_train, "n_full_points": len(t_full),
        "data_params": {k: float(v) for k, v in data.params.items()},
        "device": str(device), "torch_version": torch.__version__,
        "note": "paper-exact lr + init_scale audit run; everything else matches "
                "mlpode_baseline/mlp_paperspec_50k for direct comparison",
    }

    train_losses, test_losses, grad_norms = [], [], []
    best_train_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    nonfinite_streak = 0
    nonfinite_grad_steps = 0
    first_nonfinite_epoch = None
    aborted_at_epoch = None
    epochs_run = NUM_EPOCHS

    t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()
        pred_train = node(y0=y0_init, t=t_train)
        mse_train = F.mse_loss(pred_train, y_train)
        mse_train.backward()

        gnorm = compute_gradient_norm(model)
        grad_norms.append(gnorm)
        train_loss_val = mse_train.item()
        train_losses.append(train_loss_val)

        if train_loss_val < best_train_loss:
            best_train_loss = train_loss_val
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        step_is_finite = math.isfinite(gnorm) and math.isfinite(train_loss_val)
        if step_is_finite:
            nonfinite_streak = 0
            optimizer.step()  # deliberately NO clipping -- see module docstring
        else:
            nonfinite_grad_steps += 1
            nonfinite_streak += 1
            if first_nonfinite_epoch is None:
                first_nonfinite_epoch = epoch
                print(f"\n[non-finite] epoch {epoch}: gnorm={gnorm}, loss={train_loss_val}. "
                      f"Skipping the step, keeping previous weights.")
            optimizer.zero_grad(set_to_none=True)

        if epoch % 10 == 0 or epoch == 1 or epoch == NUM_EPOCHS:
            with torch.no_grad():
                test_loss_val = F.mse_loss(node(y0=y0_init, t=t_full), y_full).item()
        else:
            test_loss_val = test_losses[-1] if test_losses else float("inf")
        test_losses.append(test_loss_val)

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"epoch {epoch:6d}/{NUM_EPOCHS}  train={train_loss_val:.4e}  "
                  f"monitor={test_loss_val:.4e}  gnorm={gnorm:.3e}  "
                  f"best={best_train_loss:.4e}  ({elapsed:.0f}s elapsed)")

        if nonfinite_streak >= NONFINITE_ABORT_STREAK:
            aborted_at_epoch = epoch
            epochs_run = epoch
            print(f"\nABORTING at epoch {epoch}: {nonfinite_streak} consecutive non-finite "
                  f"steps -- parameters are poisoned. best checkpoint (epoch {best_epoch}, "
                  f"train_mse={best_train_loss:.4e}) is already saved.")
            break

    total_time = time.time() - t0
    print(f"\nFinished in {total_time/60:.1f} min. Best train MSE: {best_train_loss:.4e} "
          f"(epoch {best_epoch})")

    # Score final and best checkpoints, same split convention as train.py's score().
    y_full_np = y_full.cpu().numpy()

    def score(state_dict):
        if state_dict is not None:
            model.load_state_dict(state_dict)
        with torch.no_grad():
            pred = node(y0=y0_init, t=t_full).cpu().numpy()
        return {
            "train_mse": compute_mse(y_full_np[:n_train], pred[:n_train]),
            "extrap_mse": compute_mse(y_full_np[n_train:], pred[n_train:]),
            "full_mse": compute_mse(y_full_np, pred),
            "extrap_rmse": compute_rmse(y_full_np[n_train:], pred[n_train:]),
            "extrap_mae": compute_mae(y_full_np[n_train:], pred[n_train:]),
            "extrap_r2": compute_r2_score(y_full_np[n_train:], pred[n_train:]),
            "extrap_rel_l2": compute_relative_l2_error(y_full_np[n_train:], pred[n_train:]),
            "full_r2": compute_r2_score(y_full_np, pred),
        }

    final_metrics = score(None)
    best_metrics = score(best_state_dict if best_state_dict is not None else None)

    lipschitz_est = estimate_lipschitz_bound(model, x_domain=y_train)
    nfe_per_epoch = track_nfe("tsit5", num_steps=n_train - 1, substeps=2)

    metrics_summary = {
        "config": run_config,
        "selection": {"criterion": "min_train_mse", "best_epoch": best_epoch,
                      "best_train_mse_during_training": best_train_loss},
        "best": best_metrics,
        "final": final_metrics,
        "estimated_lipschitz_bound": lipschitz_est,
        "nfe_per_epoch_train": nfe_per_epoch,
        "training_time_seconds": total_time,
        "seconds_per_epoch": total_time / max(epochs_run, 1),
        "stability": {
            "nonfinite_grad_steps": nonfinite_grad_steps,
            "first_nonfinite_epoch": first_nonfinite_epoch,
            "aborted_at_epoch": aborted_at_epoch,
            "epochs_run": epochs_run,
        },
    }
    with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump({"train_losses": train_losses, "test_losses": test_losses,
                    "grad_norms": grad_norms}, f)
    if best_state_dict is not None:
        torch.save({"model_state_dict": best_state_dict, "epoch": best_epoch,
                    "train_mse": best_train_loss, "config": run_config},
                   os.path.join(SAVE_DIR, "best_model.pt"))

    print(f"\n=== RESULT ===")
    print(f"  best  : train={best_metrics['train_mse']:.4e}  extrap={best_metrics['extrap_mse']:.4e}  "
          f"extrap_r2={best_metrics['extrap_r2']:.4f}  Lipschitz={lipschitz_est:.2f}")
    print(f"  final : train={final_metrics['train_mse']:.4e}  extrap={final_metrics['extrap_mse']:.4e}  "
          f"extrap_r2={final_metrics['extrap_r2']:.4f}")

    for label, ref_path in [("10k, lr=2e-3/init_scale=1.0 (mlpode_baseline)", REFERENCE_10K),
                             ("50k, lr=2e-3/init_scale=1.0 (mlp_paperspec_50k)", REFERENCE_50K)]:
        if os.path.isfile(ref_path):
            ref = json.load(open(ref_path))["best"]
            print(f"\n  Reference [{label}]:")
            print(f"    train={ref['train_mse']:.4e}  extrap={ref['extrap_mse']:.4e}  "
                  f"extrap_r2={ref['extrap_r2']:.4f}")
        else:
            print(f"\n  (Reference not found locally: {ref_path} -- compare manually.)")

    if best_metrics["train_mse"] < 0.01:
        print("\n  -> TRAINS where lr=2e-3/init_scale=1.0 did not: this confirms the "
              "recipe-difference hypothesis (LR + init-scale), not a fundamental "
              "architecture/activation problem.")
    else:
        print("\n  -> STILL DOES NOT TRAIN even with the paper's own lr + init-scale: "
              "the recipe-difference hypothesis is refuted, and the tanh-activation "
              "mechanism itself needs a different explanation.")

    zip_base = "/kaggle/working/mlp_paperspec_exact_10k_result"
    try:
        shutil.make_archive(zip_base, "zip", SAVE_DIR)
        print(f"\nZipped to {zip_base}.zip -- download from the notebook's Output tab.")
    except Exception as e:
        print(f"\n(Zipping skipped/failed: {e} -- fine for a local run.)")


if __name__ == "__main__":
    main()
