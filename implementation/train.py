import argparse
import copy
import json
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
        mse_train = F.mse_loss(pred_train, y_train)

        # Add regularization if specified (for KAN models)
        if isinstance(model, KAN) and (act_reg > 0.0 or entropy_reg > 0.0):
            reg_loss = compute_kan_regularization(model, act_reg=act_reg, entropy_reg=entropy_reg)
        else:
            reg_loss = 0.0

        total_loss = mse_train + reg_loss
        total_loss.backward()

        # Continuous Gradient Norm Logging ||nabla_theta L||_2
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

        optimizer.step()

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
        "seconds_per_epoch": total_time / max(num_epochs, 1),
    }

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    history = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "grad_norms": grad_norms,
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
        save_dir=args.save_dir,
        print_freq=args.print_freq,
        seed=args.seed,
        device=args.device,
    )
