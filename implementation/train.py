import argparse
import copy
import json
import math  # [FIX-2026-08 / X1] needed for math.isfinite() in the gradient guard
import os
import platform
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from kan import KAN, MLP_ODE, count_parameters
from ode import NeuralODE
from data import (
    generate_lotka_volterra_data,
    generate_damped_pendulum_data,
    generate_lorenz_data,
    generate_sir_data,
)
from utils import (
    compute_kan_regularization,
    plot_trajectory_comparison,
    plot_phase_space,
    plot_loss_curves,
    plot_gradient_norm_dynamics,
    compute_mse,
    compute_rmse,
    compute_mae,
    compute_r2_score,
    compute_relative_l2_error,
    compute_gradient_norm,
    estimate_lipschitz_bound,
    track_nfe,
)


# ==============================================================================
# Dataset Registry
# ==============================================================================
# Every generator exposes the same core interface:
#   generate_*(t_start, t_end, dt, t_train_end, noise_std, seed, ...)
#     -> obj with .t_train .t_full .y_train .y_full .y0 .params .t_split
# so the training loop is dataset-agnostic. Defaults below are each generator's
# own natural horizon/step, used when the caller does not override them.
DATASETS = {
    "lotka_volterra": {
        "fn": generate_lotka_volterra_data,
        "defaults": {"t_end": 14.0, "dt": 0.1, "t_train_end": 3.5},
        "state_dim": 2,
    },
    "damped_pendulum": {
        "fn": generate_damped_pendulum_data,
        "defaults": {"t_end": 10.0, "dt": 0.05, "t_train_end": 3.0},
        "state_dim": 2,
    },
    "lorenz": {
        "fn": generate_lorenz_data,
        "defaults": {"t_end": 20.0, "dt": 0.01, "t_train_end": 8.0},
        "state_dim": 3,
    },
    "sir": {
        "fn": generate_sir_data,
        "defaults": {"t_end": 80.0, "dt": 0.5, "t_train_end": 30.0},
        "state_dim": 3,
    },
}


def _git_sha() -> str:
    """Short commit hash of the working tree, for result provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _adapt_layers(layers, state_dim):
    """Force a layer spec's input/output width to match the dataset's state dimension."""
    layers = list(layers)
    layers[0] = state_dim
    layers[-1] = state_dim
    return layers


def _stability_summary(grad_norms, grad_clip, nonfinite_grad_steps, first_nonfinite_epoch,
                       aborted_at_epoch=None, epochs_run=None):
    """
    [FIX-2026-08 / X4] Condense the per-epoch gradient history into the handful of
    numbers that actually diagnose a training-stability failure.

    `spike_ratio` (max / median of the finite pre-clip norms) is the statistic that
    identified both Phase-2 cross-domain failures: 7186x for the damped pendulum
    (epoch 4963) and 2.47e6x for SIR (epoch 5018). A healthy run sits in the low
    tens. `clip_engaged_fraction` shows how often clipping actually bound the step,
    which the pre-clip norms alone cannot reveal.
    """
    finite = [g for g in grad_norms if math.isfinite(g)]
    if not finite:
        return {
            "grad_clip": grad_clip,
            "nonfinite_grad_steps": nonfinite_grad_steps,
            "first_nonfinite_epoch": first_nonfinite_epoch,
            "aborted_at_epoch": aborted_at_epoch,
            "epochs_run": epochs_run,
            "note": "no finite gradient norms recorded",
        }

    ordered = sorted(finite)
    median = ordered[len(ordered) // 2]
    maximum = ordered[-1]
    engaged = (
        sum(1 for g in finite if g > grad_clip) / len(finite)
        if grad_clip is not None and grad_clip > 0.0
        else 0.0
    )
    return {
        "grad_clip": grad_clip,
        "nonfinite_grad_steps": nonfinite_grad_steps,
        "first_nonfinite_epoch": first_nonfinite_epoch,
        "aborted_at_epoch": aborted_at_epoch,
        "epochs_run": epochs_run,
        "grad_norm_median_preclip": median,
        "grad_norm_max_preclip": maximum,
        "grad_norm_spike_ratio": (maximum / median) if median > 0 else None,
        "grad_norm_max_epoch": grad_norms.index(maximum) + 1,
        "clip_engaged_fraction": engaged,
    }


def train_kan_ode(
    model_type="kan",
    dataset="lotka_volterra",
    layers_hidden=[2, 10, 2],
    mlp_layers=[2, 50, 2],
    grid_len=5,
    basis_func="rbf",
    normalizer="tanh",
    base_act="silu",
    mlp_act="tanh",
    use_base_act=True,
    solver="tsit5",
    substeps=2,
    lr=2e-3,
    num_epochs=5000,
    act_reg=0.0,
    entropy_reg=0.0,
    alpha=1.5,
    beta=1.0,
    gamma=3.0,
    delta=1.0,
    x0=1.0,
    y0=1.0,
    t_start=0.0,
    t_end=None,
    t_train_end=None,
    dt=None,
    noise_std=0.0,
    grad_clip=1.0,
    # ==========================================================================
    # [FIX-2026-08] Cross-domain stability options. See docs/06_suggested_fixes.md
    # and docs/07_fix_changelog.md. EVERY default below reproduces the exact
    # pre-fix behaviour, so all 26 existing Phase-2 runs remain bit-reproducible.
    # ==========================================================================
    loss_weighting="none",   # [X2] "none" | "std" -- per-dimension loss balancing
    conserve_sum=None,       # [S2] e.g. 1.0 for SIR (S+I+R=1); None disables
    conserve_weight=1.0,     # [S2] penalty weight for the conservation residual
    grid_lims=(-1.0, 1.0),   # [P3] KAN grid span; previously hardcoded to (-1,1)
    seed=42,
    save_dir="results/run_experiment",
    print_freq=200,
    device="cpu",
):
    """
    Train a KAN-ODE or MLP-ODE on a dynamical system with gradient-norm logging.

    Model selection protocol
    ------------------------
    The best checkpoint is selected on **training** loss, never on the evaluation
    trajectory. Selecting on test/extrapolation loss would be selection on the very
    data the extrapolation claim is made against; the metrics reported here are
    therefore genuinely out-of-sample. Both the best-epoch and final-epoch models are
    scored, and both sets of numbers land in `metrics.json`.

    Reported errors are split three ways and never conflated:
      * train  : MSE over the fitted window        [t_start, t_train_end]
      * extrap : MSE strictly after the split      (t_train_end, t_end]
      * full   : MSE over the whole horizon        [t_start, t_end]
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(DATASETS)}")
    spec = DATASETS[dataset]

    # Explicit, up-front seeding. The dataset generators also seed as a side effect,
    # but relying on that made reproducibility depend on data being built before the
    # model -- a coupling that breaks silently the moment call order changes.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Generate Dataset
    horizon = dict(spec["defaults"])
    if t_end is not None:
        horizon["t_end"] = t_end
    if dt is not None:
        horizon["dt"] = dt
    if t_train_end is not None:
        horizon["t_train_end"] = t_train_end

    data_kwargs = dict(
        t_start=t_start,
        noise_std=noise_std,
        seed=seed,
        **horizon,
    )
    if dataset == "lotka_volterra":
        data_kwargs.update(
            alpha=alpha, beta=beta, gamma=gamma, delta=delta, x0=x0, y0=y0
        )
    data = spec["fn"](**data_kwargs)

    # The generators call torch.manual_seed internally; re-seed so that model init is
    # governed by `seed` alone and not by how many random draws the noise used.
    torch.manual_seed(seed)
    np.random.seed(seed)

    t_train = data.t_train.to(device)
    t_full = data.t_full.to(device)
    y_train = data.y_train.to(device)
    y_full = data.y_full.to(device)
    y0_init = data.y0.to(device)

    n_train = len(t_train)
    state_dim = y_full.shape[-1]

    # 2. Build Model & Neural ODE Integrator
    if model_type.lower() == "mlp":
        mlp_layers = _adapt_layers(mlp_layers, state_dim)
        model = MLP_ODE(layers_hidden=mlp_layers, activation=mlp_act).to(device)
        total_p, train_p = count_parameters(model)
        model_desc = f"MLP-ODE {mlp_layers} ({mlp_act}, {total_p} params)"
        arch_layers = mlp_layers
    else:
        layers_hidden = _adapt_layers(layers_hidden, state_dim)
        model = KAN(
            layers_hidden=layers_hidden,
            grid_len=grid_len,
            # [FIX-2026-08 / P3] grid_lims is now plumbed through instead of
            # falling back to KAN's hardcoded (-1, 1). Widening it lets the
            # spline grid cover state dimensions that tanh pushes towards the
            # domain edges. Default (-1.0, 1.0) == previous behaviour.
            grid_lims=tuple(grid_lims),
            basis_func=basis_func,
            normalizer=normalizer,
            base_act=base_act,
            use_base_act=use_base_act,
        ).to(device)
        total_p, train_p = count_parameters(model)
        model_desc = f"KAN-ODE {layers_hidden} (grid={grid_len}, basis={basis_func}, {total_p} params)"
        arch_layers = layers_hidden

    node = NeuralODE(func=model, method=solver, substeps=substeps).to(device)

    # Full resolved configuration -- written verbatim into the checkpoint and into
    # metrics.json so any result can be traced back to the exact run that produced it.
    run_config = {
        "model_type": model_type,
        "dataset": dataset,
        "layers_hidden": arch_layers,
        "grid_len": grid_len,
        "basis_func": basis_func if model_type.lower() == "kan" else "none",
        "normalizer": normalizer,
        "base_act": base_act,
        "mlp_act": mlp_act,
        "use_base_act": use_base_act,
        "solver": solver,
        "substeps": substeps,
        "lr": lr,
        "num_epochs": num_epochs,
        "act_reg": act_reg,
        "entropy_reg": entropy_reg,
        "seed": seed,
        "noise_std": noise_std,
        "grad_clip": grad_clip,
        # [FIX-2026-08] recorded so every result is traceable to the exact
        # stability settings it was produced under.
        "loss_weighting": loss_weighting,
        "conserve_sum": conserve_sum,
        "conserve_weight": conserve_weight if conserve_sum is not None else None,
        "grid_lims": list(grid_lims),
        "t_start": t_start,
        "t_end": horizon["t_end"],
        "t_train_end": horizon["t_train_end"],
        "dt": horizon["dt"],
        "state_dim": state_dim,
        "n_train_points": n_train,
        "n_full_points": len(t_full),
        "data_params": {k: float(v) for k, v in data.params.items()},
        "parameters": total_p,
        "device": str(device),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "git_sha": _git_sha(),
    }

    # 3. Optimizer (constant LR matching the paper -- no scheduler)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    grad_norms = []
    best_train_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    # ==========================================================================
    # [FIX-2026-08 / X2] Per-dimension loss weighting.
    #
    # Plain F.mse_loss averages the squared error over ALL state dimensions
    # equally, so the widest-spread dimension dominates the gradient. Measured
    # training-window std imbalance vs. outcome:
    #     Lotka-Volterra  x=1.986  y=1.367            -> 1.45x   converges
    #     Damped pendulum th=1.010 om=2.534           -> 2.51x   FAILS
    #     SIR             S=0.358  I=0.113  R=0.332   -> 3.19x   FAILS
    # On the pendulum this shows up directly: omega is fitted to RMSE 0.048
    # while theta -- whose derivative IS omega -- only reaches RMSE 0.756.
    #
    # With loss_weighting="std" each dimension is weighted by 1/var, then the
    # weights are renormalised to mean 1.0 so the weighted loss stays on the
    # same order of magnitude as the plain MSE it replaces.
    #
    # NOTE: `mse_train` (plain, unweighted) is still what gets logged and
    # plotted, and every number in metrics.json is recomputed post-hoc by
    # score() from the integrated trajectory. So switching this on changes what
    # is OPTIMISED, never how results are MEASURED -- the new runs stay directly
    # comparable with the existing 26.
    # ==========================================================================
    if loss_weighting == "none":
        loss_weights = None
    elif loss_weighting == "std":
        dim_std = y_train.std(dim=0)                       # [state_dim]
        inv_var = 1.0 / (dim_std ** 2 + 1e-12)
        loss_weights = (inv_var / inv_var.mean()).detach()  # mean(weights) == 1
        print(f"[FIX/X2] loss_weighting='std' | per-dim std: "
              f"{[round(v, 4) for v in dim_std.tolist()]} -> weights: "
              f"{[round(v, 4) for v in loss_weights.tolist()]}")
    else:
        raise ValueError(
            f"Unknown loss_weighting '{loss_weighting}'. Expected 'none' or 'std'."
        )

    if conserve_sum is not None:
        print(f"[FIX/S2] conservation penalty active: sum(u) -> {conserve_sum} "
              f"(weight {conserve_weight})")

    # [FIX-2026-08 / X1+X3+X4] stability instrumentation
    post_clip_grad_norms = []       # [X3] norm AFTER clipping, to show clipping engaged
    nonfinite_grad_steps = 0        # [X1] optimizer steps skipped due to inf/NaN
    first_nonfinite_epoch = None    # [X1] when the first one happened
    nonfinite_streak = 0            # [X1] consecutive skipped steps, for the abort below
    aborted_at_epoch = None         # [X1] set if training stopped early
    epochs_run = num_epochs         # [X1] actual epochs completed (== num_epochs normally)

    # [FIX-2026-08 / X1] If the PARAMETERS themselves have already gone non-finite,
    # every subsequent forward pass returns NaN and the guard below can only keep
    # skipping steps -- the run is unrecoverable and just burns compute. SIR spent
    # its last 1,299 epochs in exactly that state. Abort after this many consecutive
    # skipped steps; best_model.pt is already safely on disk by then.
    NONFINITE_ABORT_STREAK = 100

    print("=" * 70)
    print(f"Training Model: {model_desc}")
    print(f"Dataset: {dataset} | Solver: {solver} (substeps={substeps}) | LR: {lr} | Epochs: {num_epochs}")
    print(f"Seed: {seed} | Noise sigma: {noise_std} | dt: {horizon['dt']}")
    print(f"Horizon: Train [{t_start}, {horizon['t_train_end']}] ({n_train} pts) | "
          f"Full [{t_start}, {horizon['t_end']}] ({len(t_full)} pts)")
    print("=" * 70)

    start_time = time.time()

    pbar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch", ncols=125)
    for epoch in pbar:
        optimizer.zero_grad()

        # Integrate forward over training time interval
        pred_train = node(y0=y0_init, t=t_train)

        # Plain unweighted MSE. ALWAYS computed and always what gets logged, so
        # loss_curves.png and train_losses[] stay comparable across every run in
        # the project regardless of the stability options below.
        mse_train = F.mse_loss(pred_train, y_train)

        # [FIX-2026-08 / X2] the quantity actually optimised: weighted if
        # loss_weighting="std", otherwise literally mse_train (default path,
        # bit-identical to pre-fix behaviour).
        if loss_weights is None:
            data_loss = mse_train
        else:
            data_loss = (loss_weights * (pred_train - y_train) ** 2).mean()

        # Add regularization if specified (for KAN models)
        if isinstance(model, KAN) and (act_reg > 0.0 or entropy_reg > 0.0):
            reg_loss = compute_kan_regularization(model, act_reg=act_reg, entropy_reg=entropy_reg)
        else:
            reg_loss = 0.0

        # ---------------------------------------------------------------------
        # [FIX-2026-08 / S2] Conservation-law penalty.
        # SIR carries the exact invariant S + I + R = 1, which nothing in the
        # loss or architecture previously enforced -- the failed run drifted to
        # sum(u) = 1.0045 in extrapolation and 0.949-1.029 during training.
        # Disabled (conserve_sum=None) for every system that has no such
        # invariant, e.g. Lotka-Volterra and the pendulum.
        # ---------------------------------------------------------------------
        if conserve_sum is not None:
            cons_residual = pred_train.sum(dim=-1) - float(conserve_sum)
            cons_loss = conserve_weight * (cons_residual ** 2).mean()
        else:
            cons_loss = 0.0

        total_loss = data_loss + reg_loss + cons_loss
        total_loss.backward()

        # Continuous Gradient Norm Logging ||nabla_theta L||_2 (PRE-clip)
        gnorm = compute_gradient_norm(model)
        grad_norms.append(gnorm)

        train_loss_val = mse_train.item()
        train_losses.append(train_loss_val)

        # Checkpoint on TRAINING loss only -- see docstring. Two subtleties:
        #  * deepcopy is required: state_dict() returns references to the live
        #    parameter tensors, so storing it directly would keep tracking the
        #    optimizer's later updates and the "best" model would silently become
        #    the final model.
        #  * the snapshot must be taken BEFORE optimizer.step(), because these are
        #    the weights that actually produced train_loss_val. Snapshotting after
        #    the step would label the updated weights with the pre-update loss.
        if train_loss_val < best_train_loss:
            best_train_loss = train_loss_val
            best_epoch = epoch
            best_state_dict = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "train_mse": train_loss_val,
                "grad_norm": gnorm,
                "config": run_config,
            }

        # =====================================================================
        # [FIX-2026-08 / X1] Non-finite gradient guard.
        #
        # torch.nn.utils.clip_grad_norm_ computes
        #       clip_coef = max_norm / (total_norm + 1e-6)
        # so if total_norm has already overflowed float32 to inf, then
        # clip_coef -> 0 and the subsequent grad.mul_(clip_coef) evaluates
        #       inf * 0 = NaN
        # which permanently poisons every parameter. Clipping bounds the
        # optimizer STEP; it cannot undo an overflow in the forward/backward
        # pass. The 10,000-epoch SIR run died exactly this way: gradient norm
        # reached 5.82e18, went non-finite at epoch 8686, loss NaN from 8702,
        # and the last 1,299 epochs plus final_model.pt were lost.
        #
        # Skipping the step keeps the last known-good weights and lets training
        # continue. Arrays are appended to on every branch so all per-epoch
        # histories stay index-aligned with `epoch`.
        # =====================================================================
        step_is_finite = math.isfinite(gnorm) and math.isfinite(train_loss_val)

        if step_is_finite:
            nonfinite_streak = 0
            if grad_clip is not None and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            # [FIX-2026-08 / X3] `grad_norms` above is the PRE-clip norm, so on
            # its own it cannot show whether clipping actually engaged. Record
            # the post-clip norm too, making that readable straight from the
            # history instead of requiring forensic analysis after the fact.
            post_clip_grad_norms.append(compute_gradient_norm(model))
            optimizer.step()
        else:
            nonfinite_grad_steps += 1
            nonfinite_streak += 1
            if first_nonfinite_epoch is None:
                first_nonfinite_epoch = epoch
                print(f"\n[FIX/X1] Non-finite gradient at epoch {epoch} "
                      f"(gnorm={gnorm}, loss={train_loss_val}). Skipping the "
                      f"optimizer step and keeping the previous weights.")
            optimizer.zero_grad(set_to_none=True)
            post_clip_grad_norms.append(0.0)  # step dropped: effective update is zero

        # Monitor the full-horizon loss periodically (diagnostic only -- never used
        # for model selection).
        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            with torch.no_grad():
                pred_full = node(y0=y0_init, t=t_full)
                mse_test = F.mse_loss(pred_full, y_full)
            test_loss_val = mse_test.item()
        else:
            test_loss_val = test_losses[-1] if test_losses else float("inf")
        test_losses.append(test_loss_val)

        pbar.set_postfix({
            "train": f"{train_loss_val:.3e}",
            "monitor": f"{test_loss_val:.3e}",
            "gnorm": f"{gnorm:.2e}",
            "best": f"{best_train_loss:.3e}",
        })

        # [FIX-2026-08 / X1] Early abort, placed at the very END of the loop body
        # so every per-epoch array (train_losses, test_losses, grad_norms,
        # post_clip_grad_norms) is appended to before we leave -- keeping them all
        # the same length and index-aligned for the plots and the history JSON.
        if nonfinite_streak >= NONFINITE_ABORT_STREAK:
            aborted_at_epoch = epoch
            epochs_run = epoch
            print(f"\n[FIX/X1] ABORTING at epoch {epoch}: {nonfinite_streak} consecutive "
                  f"non-finite steps means the parameters themselves are poisoned and "
                  f"the run cannot recover. best_model.pt (epoch {best_epoch}, "
                  f"train_mse={best_train_loss:.4e}) is already saved and is valid.")
            pbar.close()
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s! "
          f"Best train MSE: {best_train_loss:.4e} (epoch {best_epoch})")

    if best_state_dict is not None:
        torch.save(best_state_dict, os.path.join(save_dir, "best_model.pt"))
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": num_epochs, "config": run_config},
        os.path.join(save_dir, "final_model.pt"),
    )

    y_full_np = y_full.cpu().numpy()
    t_full_np = t_full.cpu().numpy()

    def score(state_dict):
        """Integrate one parameter set and split the error into train/extrap/full."""
        if state_dict is not None:
            model.load_state_dict(state_dict)
        with torch.no_grad():
            pred = node(y0=y0_init, t=t_full).cpu().numpy()
        return pred, {
            "train_mse": compute_mse(y_full_np[:n_train], pred[:n_train]),
            "extrap_mse": compute_mse(y_full_np[n_train:], pred[n_train:]),
            "full_mse": compute_mse(y_full_np, pred),
            "extrap_rmse": compute_rmse(y_full_np[n_train:], pred[n_train:]),
            "extrap_mae": compute_mae(y_full_np[n_train:], pred[n_train:]),
            "extrap_r2": compute_r2_score(y_full_np[n_train:], pred[n_train:]),
            "extrap_rel_l2": compute_relative_l2_error(y_full_np[n_train:], pred[n_train:]),
            "full_rmse": compute_rmse(y_full_np, pred),
            "full_mae": compute_mae(y_full_np, pred),
            "full_r2": compute_r2_score(y_full_np, pred),
            "full_rel_l2": compute_relative_l2_error(y_full_np, pred),
        }

    # Score the final-epoch model first, then the best-epoch model (which is loaded
    # last so all downstream plots depict the checkpoint we actually report).
    final_pred, final_metrics = score(None)
    best_pred, best_metrics = score(
        best_state_dict["model_state_dict"] if best_state_dict is not None else None
    )

    lipschitz_est = estimate_lipschitz_bound(model, x_domain=y_train)
    nfe_per_traj = track_nfe(solver, num_steps=len(t_full) - 1, substeps=substeps)

    metrics_summary = {
        "config": run_config,
        "selection": {
            "criterion": "min_train_mse",
            "best_epoch": best_epoch,
            "best_train_mse_during_training": best_train_loss,
        },
        "best": best_metrics,
        "final": final_metrics,
        "estimated_lipschitz_bound": lipschitz_est,
        "nfe_per_trajectory": nfe_per_traj,
        "nfe_per_epoch_train": track_nfe(solver, num_steps=n_train - 1, substeps=substeps),
        "training_time_seconds": total_time,
        # [FIX-2026-08 / X1] divide by epochs ACTUALLY run, not the requested
        # budget, so an early-aborted run does not report a misleadingly small
        # per-epoch cost. Identical to the old value when nothing aborts.
        "seconds_per_epoch": total_time / max(epochs_run, 1),
        # [FIX-2026-08 / X4] Post-hoc stability summary. Previously the only way
        # to tell whether a run had suffered a gradient blowup was to load
        # training_history.json and compute the max/median ratio by hand -- which
        # is how both the pendulum (7186x) and SIR (2.47e6x) spikes were found.
        # Surfacing it here makes every future failure readable straight off
        # metrics.json and collates into results/tables/*.csv.
        "stability": _stability_summary(
            grad_norms=grad_norms,
            grad_clip=grad_clip,
            nonfinite_grad_steps=nonfinite_grad_steps,
            first_nonfinite_epoch=first_nonfinite_epoch,
            aborted_at_epoch=aborted_at_epoch,
            epochs_run=epochs_run,
        ),
    }

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    history = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "grad_norms": grad_norms,
        # [FIX-2026-08 / X3] post-clip norms, same length/indexing as grad_norms.
        # grad_norms[i] > post_clip_grad_norms[i] means clipping engaged at
        # epoch i+1; a 0.0 entry means the step was dropped by the X1 guard.
        "post_clip_grad_norms": post_clip_grad_norms,
    }
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    # Generate Output Plots (from the reported best-epoch model)
    plot_label = f"MLP-ODE ({mlp_act.upper()})" if model_type.lower() == "mlp" else f"KAN-ODE ({basis_func.upper()})"

    plot_trajectory_comparison(
        t_full=t_full_np,
        y_true=y_full_np,
        y_pred=best_pred,
        t_split=horizon["t_train_end"],
        title=f"{plot_label} + {solver.upper()} on {dataset}",
        save_path=os.path.join(save_dir, "trajectory_comparison.png"),
    )

    plot_phase_space(
        y_true=y_full_np,
        y_pred=best_pred,
        train_len=n_train,
        title=f"Phase Space: True vs {plot_label}",
        save_path=os.path.join(save_dir, "phase_space.png"),
    )

    plot_loss_curves(
        train_losses=train_losses,
        test_losses=test_losses,
        title=f"{plot_label} Training Dynamics (+ {solver.upper()})",
        save_path=os.path.join(save_dir, "loss_curves.png"),
    )

    plot_gradient_norm_dynamics(
        grad_norms=grad_norms,
        title=f"{plot_label} Gradient Norm Dynamics (||grad_theta L||_2)",
        save_path=os.path.join(save_dir, "gradient_norm_dynamics.png"),
    )

    print(f"Results, metrics, and plots saved to '{save_dir}'.")
    print(f"  best  : train={best_metrics['train_mse']:.4e}  "
          f"extrap={best_metrics['extrap_mse']:.4e}  full={best_metrics['full_mse']:.4e}")
    print(f"  final : train={final_metrics['train_mse']:.4e}  "
          f"extrap={final_metrics['extrap_mse']:.4e}  full={final_metrics['full_mse']:.4e}")

    return {
        "model": model,
        "node": node,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "grad_norms": grad_norms,
        "best": best_metrics,
        "final": final_metrics,
        "best_epoch": best_epoch,
        "metrics": metrics_summary,
        "config": run_config,
        "pred_trajectory": best_pred,
        "training_time_seconds": total_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KAN-ODE or MLP-ODE on a dynamical system")
    parser.add_argument("--model", type=str, default="kan", choices=["kan", "mlp"], help="Model architecture")
    parser.add_argument("--dataset", type=str, default="lotka_volterra", choices=list(DATASETS), help="Dynamical system")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 10, 2], help="KAN hidden layer dimensions")
    parser.add_argument("--mlp_layers", type=int, nargs="+", default=[2, 50, 2], help="MLP layer dimensions (paper Table I: 2 50 2 = 252 params)")
    parser.add_argument("--grid_len", type=int, default=5, help="Number of grid centers for KAN basis")
    parser.add_argument(
        "--basis", type=str, default="rbf",
        choices=["rbf", "rswaf", "iqf", "bspline", "chebyshev", "lagrange", "newton"],
        help="KAN Basis function",
    )
    parser.add_argument("--solver", type=str, default="tsit5", choices=["tsit5", "rk4", "dopri5", "euler", "midpoint", "heun"], help="ODE Integrator")
    parser.add_argument("--substeps", type=int, default=2, help="Integration substeps per reporting interval")
    parser.add_argument("--act", type=str, default="silu", help="KAN base activation (silu, tanh, relu, gelu)")
    parser.add_argument("--mlp_act", type=str, default="tanh", help="MLP activation (paper Table I uses tanh)")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--act_reg", type=float, default=0.0, help="L1 regularization weight")
    parser.add_argument("--entropy_reg", type=float, default=0.0, help="Entropy regularization weight")
    parser.add_argument("--dt", type=float, default=None, help="Observation step size (default: dataset-specific)")
    parser.add_argument("--t_end", type=float, default=None, help="Full horizon end time (default: dataset-specific)")
    parser.add_argument("--t_train_end", type=float, default=None, help="Train/extrapolation split (default: dataset-specific)")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Gaussian observational noise sigma on the training window")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm clipping threshold (0.0 to disable)")
    # ==========================================================================
    # [FIX-2026-08] Cross-domain stability flags. All defaults == pre-fix
    # behaviour, so omitting every flag below reproduces the existing Phase-2
    # results exactly. See docs/08_how_to_run_fixes.md for the tested recipes.
    # ==========================================================================
    parser.add_argument("--loss_weighting", type=str, default="none", choices=["none", "std"],
                        help="[X2] 'std' weights each state dimension by 1/var so no single "
                             "dimension dominates the gradient (default: none)")
    parser.add_argument("--conserve_sum", type=float, default=None,
                        help="[S2] Penalise deviation of sum(state) from this value, e.g. 1.0 "
                             "for SIR's S+I+R=1 invariant (default: disabled)")
    parser.add_argument("--conserve_weight", type=float, default=1.0,
                        help="[S2] Weight of the --conserve_sum penalty (default: 1.0)")
    parser.add_argument("--normalizer", type=str, default="tanh", choices=["tanh", "sigmoid", "identity"],
                        help="[P3] KAN input normalizer. Previously hardcoded to tanh and not "
                             "reachable from the CLI (default: tanh)")
    parser.add_argument("--grid_lims", type=float, nargs=2, default=[-1.0, 1.0], metavar=("LO", "HI"),
                        help="[P3] KAN spline grid span. Widen past (-1, 1) when tanh pushes a "
                             "state dimension onto the domain edges (default: -1.0 1.0)")
    parser.add_argument("--save_dir", type=str, default="results/run_experiment", help="Directory for checkpoints and plots")
    parser.add_argument("--print_freq", type=int, default=200, help="(unused; retained for CLI compatibility)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")

    args = parser.parse_args()

    train_kan_ode(
        model_type=args.model,
        dataset=args.dataset,
        layers_hidden=args.layers,
        mlp_layers=args.mlp_layers,
        grid_len=args.grid_len,
        basis_func=args.basis,
        solver=args.solver,
        substeps=args.substeps,
        base_act=args.act,
        mlp_act=args.mlp_act,
        lr=args.lr,
        num_epochs=args.epochs,
        act_reg=args.act_reg,
        entropy_reg=args.entropy_reg,
        dt=args.dt,
        t_end=args.t_end,
        t_train_end=args.t_train_end,
        noise_std=args.noise_std,
        grad_clip=args.grad_clip,
        # [FIX-2026-08] stability options -- see the argparse block above
        loss_weighting=args.loss_weighting,
        conserve_sum=args.conserve_sum,
        conserve_weight=args.conserve_weight,
        normalizer=args.normalizer,
        grid_lims=tuple(args.grid_lims),
        save_dir=args.save_dir,
        print_freq=args.print_freq,
        seed=args.seed,
        device=args.device,
    )
