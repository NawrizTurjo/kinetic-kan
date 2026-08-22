import argparse
import os
import torch
import numpy as np

from kan import KAN
from ode import NeuralODE
from data import generate_lotka_volterra_data
from utils import plot_trajectory_comparison, plot_phase_space


def compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 determination coefficient."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    return 1.0 - (ss_res / (ss_tot + 1e-10))


def evaluate_checkpoint(
    checkpoint_path: str,
    solver: str = None,
    save_dir: str = "results/eval",
):
    """
    Load a trained KAN-ODE model checkpoint and evaluate performance.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")
        
    os.makedirs(save_dir, exist_ok=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})
    
    layers_hidden = config.get("layers_hidden", [2, 10, 2])
    grid_len = config.get("grid_len", 5)
    basis_func = config.get("basis_func", "rbf")
    normalizer = config.get("normalizer", "tanh")
    base_act = config.get("base_act", "silu")
    eval_solver = solver if solver is not None else config.get("solver", "tsit5")
    
    # 1. Instantiate Model
    kan_model = KAN(
        layers_hidden=layers_hidden,
        grid_len=grid_len,
        basis_func=basis_func,
        normalizer=normalizer,
        base_act=base_act,
    )
    kan_model.load_state_dict(ckpt["model_state_dict"])
    kan_model.eval()
    
    node = NeuralODE(func=kan_model, method=eval_solver)
    
    # 2. Data
    data = generate_lotka_volterra_data()
    t_train = data.t_train
    t_full = data.t_full
    y_train = data.y_train.numpy()
    y_full = data.y_full.numpy()
    y0_init = data.y0
    
    # 3. Predict
    with torch.no_grad():
        pred_full = node(y0=y0_init, t=t_full).numpy()
        
    n_train = len(t_train)
    pred_train = pred_full[:n_train]
    pred_test = pred_full[n_train:]
    y_test = y_full[n_train:]
    
    # 4. Metrics
    train_mse = np.mean((pred_train - y_train) ** 2)
    train_mae = np.mean(np.abs(pred_train - y_train))
    train_r2 = compute_r2_score(y_train, pred_train)
    
    test_mse = np.mean((pred_test - y_test) ** 2)
    test_mae = np.mean(np.abs(pred_test - y_test))
    test_r2 = compute_r2_score(y_test, pred_test)
    
    full_mse = np.mean((pred_full - y_full) ** 2)
    full_mae = np.mean(np.abs(pred_full - y_full))
    full_r2 = compute_r2_score(y_full, pred_full)
    
    print("=" * 65)
    print(f"Evaluation Report: {checkpoint_path}")
    print(f"Configuration: {layers_hidden} | Grid: {grid_len} | Basis: {basis_func} | Solver: {eval_solver}")
    print("=" * 65)
    print(f"Training Horizon [0.0, {data.t_split}] ({n_train} steps):")
    print(f"  - MSE: {train_mse:.4e}")
    print(f"  - MAE: {train_mae:.4e}")
    print(f"  - R² : {train_r2:.4f}")
    print("-" * 65)
    print(f"Extrapolation Horizon [{data.t_split}, 14.0] ({len(y_test)} steps):")
    print(f"  - MSE: {test_mse:.4e}")
    print(f"  - MAE: {test_mae:.4e}")
    print(f"  - R² : {test_r2:.4f}")
    print("-" * 65)
    print(f"Overall Full Horizon [0.0, 14.0] ({len(y_full)} steps):")
    print(f"  - MSE: {full_mse:.4e}")
    print(f"  - MAE: {full_mae:.4e}")
    print(f"  - R² : {full_r2:.4f}")
    print("=" * 65)
    
    # Save plots
    plot_trajectory_comparison(
        t_full=t_full.numpy(),
        y_true=y_full,
        y_pred=pred_full,
        t_split=data.t_split,
        title=f"Evaluation: KAN-ODE ({basis_func.upper()} + {eval_solver.upper()})",
        save_path=os.path.join(save_dir, "eval_trajectory.png"),
    )
    
    plot_phase_space(
        y_true=y_full,
        y_pred=pred_full,
        train_len=n_train,
        title=f"Evaluation Phase Portrait ({basis_func.upper()})",
        save_path=os.path.join(save_dir, "eval_phase_space.png"),
    )
    
    return {
        "train_mse": train_mse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "full_mse": full_mse,
        "full_mae": full_mae,
        "full_r2": full_r2,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained KAN-ODE checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--solver", type=str, default=None, help="Override solver method for inference")
    parser.add_argument("--save_dir", type=str, default="results/eval", help="Directory to save evaluation plots")
    
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, solver=args.solver, save_dir=args.save_dir)
