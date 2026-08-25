import argparse
import os

import numpy as np
import torch

from kan import KAN, MLP_ODE
from ode import NeuralODE
from train import DATASETS
from utils import (
    plot_trajectory_comparison,
    plot_phase_space,
    compute_mse,
    compute_mae,
    compute_r2_score,
    compute_relative_l2_error,
)


def _rebuild_data(config):
    """
    Regenerate the exact ground truth the checkpoint was trained against.

    Reads the data-generation parameters out of the checkpoint config rather than
    calling the generator with hardcoded defaults; otherwise any run with a non-default
    seed, noise level, step size, or equation parameters would be silently scored
    against a different trajectory.
    """
    dataset = config.get("dataset", "lotka_volterra")
    spec = DATASETS[dataset]
    defaults = spec["defaults"]

    kwargs = dict(
        t_start=config.get("t_start", 0.0),
        t_end=config.get("t_end", defaults["t_end"]),
        dt=config.get("dt", defaults["dt"]),
        t_train_end=config.get("t_train_end", defaults["t_train_end"]),
        noise_std=config.get("noise_std", 0.0),
        seed=config.get("seed", 42),
    )
    if dataset == "lotka_volterra":
        p = config.get("data_params", {})
        for key in ("alpha", "beta", "gamma", "delta"):
            if key in p:
                kwargs[key] = p[key]
    return spec["fn"](**kwargs), dataset


def evaluate_checkpoint(checkpoint_path: str, solver: str = None, save_dir: str = "results/eval"):
    """Load a trained checkpoint and report train / extrapolation / full-horizon metrics."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")

    os.makedirs(save_dir, exist_ok=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    model_type = config.get("model_type", "kan")
    layers_hidden = config.get("layers_hidden", [2, 10, 2])
    grid_len = config.get("grid_len", 5)
    basis_func = config.get("basis_func", "rbf")
    normalizer = config.get("normalizer", "tanh")
    base_act = config.get("base_act", "silu")
    mlp_act = config.get("mlp_act", "tanh")
    eval_solver = solver if solver is not None else config.get("solver", "tsit5")
    # Match the training discretization -- evaluating at a different substep count
    # integrates a different trajectory and will not reproduce the run's own metrics.
    substeps = config.get("substeps", 2)

    # 1. Instantiate the architecture the checkpoint was actually trained with
    if model_type.lower() == "mlp":
        model = MLP_ODE(layers_hidden=layers_hidden, activation=mlp_act)
        label = f"MLP-ODE ({mlp_act.upper()})"
    else:
        model = KAN(
            layers_hidden=layers_hidden,
            grid_len=grid_len,
            basis_func=basis_func,
            normalizer=normalizer,
            base_act=base_act,
        )
        label = f"KAN-ODE ({basis_func.upper()})"

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    node = NeuralODE(func=model, method=eval_solver, substeps=substeps)

    # 2. Data (reconstructed from the checkpoint's own config)
    data, dataset = _rebuild_data(config)
    t_train, t_full = data.t_train, data.t_full
    y_full = data.y_full.numpy()

    # 3. Predict
    with torch.no_grad():
        pred_full = node(y0=data.y0, t=t_full).numpy()

    n_train = len(t_train)
    t0, t1 = t_full[0].item(), t_full[-1].item()
    splits = {
        f"Training Horizon [{t0}, {data.t_split}] ({n_train} steps)":
            (y_full[:n_train], pred_full[:n_train]),
        f"Extrapolation Horizon ({data.t_split}, {t1}] ({len(y_full) - n_train} steps)":
            (y_full[n_train:], pred_full[n_train:]),
        f"Overall Full Horizon [{t0}, {t1}] ({len(y_full)} steps)":
            (y_full, pred_full),
    }

    print("=" * 72)
    print(f"Evaluation Report: {checkpoint_path}")
    print(f"Model: {model_type} {layers_hidden} | Basis: {basis_func} | Solver: {eval_solver} "
          f"(substeps={substeps})")
    print(f"Dataset: {dataset} | seed={config.get('seed')} | noise={config.get('noise_std')} "
          f"| dt={config.get('dt')} | epoch={ckpt.get('epoch')}")
    print("=" * 72)

    out = {}
    for name, (yt, yp) in splits.items():
        m = {
            "mse": compute_mse(yt, yp),
            "mae": compute_mae(yt, yp),
            "r2": compute_r2_score(yt, yp),
            "rel_l2": compute_relative_l2_error(yt, yp),
        }
        print(f"{name}:")
        print(f"  - MSE    : {m['mse']:.4e}")
        print(f"  - MAE    : {m['mae']:.4e}")
        print(f"  - R2     : {m['r2']:.4f}")
        print(f"  - Rel L2 : {m['rel_l2']:.4e}")
        print("-" * 72)
        out[name.split(" Horizon")[0].strip().lower()] = m

    plot_trajectory_comparison(
        t_full=t_full.numpy(),
        y_true=y_full,
        y_pred=pred_full,
        t_split=data.t_split,
        title=f"Evaluation: {label} + {eval_solver.upper()}",
        save_path=os.path.join(save_dir, "eval_trajectory.png"),
    )

    plot_phase_space(
        y_true=y_full,
        y_pred=pred_full,
        train_len=n_train,
        title=f"Evaluation Phase Portrait ({label})",
        save_path=os.path.join(save_dir, "eval_phase_space.png"),
    )

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained KAN-ODE / MLP-ODE checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--solver", type=str, default=None, help="Override solver method for inference")
    parser.add_argument("--save_dir", type=str, default="results/eval", help="Directory for evaluation plots")

    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, solver=args.solver, save_dir=args.save_dir)
