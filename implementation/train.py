import argparse
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from kan import KAN, MLP_ODE, count_parameters
from ode import NeuralODE
from data import generate_lotka_volterra_data
from utils import compute_kan_regularization, plot_trajectory_comparison, plot_phase_space, plot_loss_curves


def train_kan_ode(
    model_type="kan",
    layers_hidden=[2, 10, 2],
    mlp_layers=[2, 14, 8, 8, 2],
    grid_len=5,
    basis_func="rbf",
    normalizer="tanh",
    base_act="silu",
    use_base_act=True,
    solver="tsit5",
    substeps=2,
    lr=5e-4,
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
    t_end=14.0,
    t_train_end=3.5,
    dt=0.1,
    noise_std=0.0,
    seed=42,
    save_dir="results/run_experiment",
    print_freq=200,
    device="cpu",
):
    """
    Train a KAN-ODE or MLP-ODE on Lotka-Volterra dynamics.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)
    
    # 1. Generate Dataset
    data = generate_lotka_volterra_data(
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        x0=x0, y0=y0, t_start=t_start, t_end=t_end,
        t_train_end=t_train_end, dt=dt, noise_std=noise_std,
        seed=seed,
    )
    
    t_train = data.t_train.to(device)
    t_full = data.t_full.to(device)
    y_train = data.y_train.to(device)
    y_full = data.y_full.to(device)
    y0_init = data.y0.to(device)
    
    # 2. Build Model & Neural ODE Integrator
    if model_type.lower() == "mlp":
        model = MLP_ODE(
            layers_hidden=mlp_layers,
            activation=base_act,
        ).to(device)
        total_p, train_p = count_parameters(model)
        model_desc = f"MLP-ODE {mlp_layers} ({total_p} params)"
    else:
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
    
    node = NeuralODE(
        func=model,
        method=solver,
        substeps=substeps,
    ).to(device)
    
    # 3. Optimizer (constant LR matching the paper — no scheduler)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    best_test_loss = float("inf")
    best_state_dict = None
    
    print("=" * 70)
    print(f"Training Model: {model_desc}")
    print(f"ODE Solver: {solver} (substeps={substeps}) | LR: {lr} | Epochs: {num_epochs}")
    print(f"Time Horizon: Train [0.0, {t_train_end}] ({len(t_train)} pts) | Full [0.0, {t_end}] ({len(t_full)} pts)")
    print("=" * 70)
    
    start_time = time.time()
    
    pbar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch", ncols=120)
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Integrate forward over training time interval
        pred_train = node(y0=y0_init, t=t_train) # [N_train, 2]
        mse_train = F.mse_loss(pred_train, y_train)
        
        # Add regularization if specified (for KAN models)
        if isinstance(model, KAN):
            reg_loss = compute_kan_regularization(model, act_reg=act_reg, entropy_reg=entropy_reg)
        else:
            reg_loss = 0.0
            
        total_loss = mse_train + reg_loss
        
        total_loss.backward()
        optimizer.step()
        
        train_loss_val = mse_train.item()
        train_losses.append(train_loss_val)
        
        # Evaluate test loss periodically (every 10 epochs) to save time
        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            with torch.no_grad():
                pred_full = node(y0=y0_init, t=t_full)
                mse_test = F.mse_loss(pred_full, y_full)
            test_loss_val = mse_test.item()
        else:
            test_loss_val = test_losses[-1] if test_losses else float("inf")
        test_losses.append(test_loss_val)
        
        if test_loss_val < best_test_loss:
            best_test_loss = test_loss_val
            best_state_dict = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "train_mse": train_loss_val,
                "test_mse": test_loss_val,
                "config": {
                    "model_type": model_type,
                    "layers_hidden": layers_hidden if model_type == "kan" else mlp_layers,
                    "grid_len": grid_len,
                    "basis_func": basis_func,
                    "normalizer": normalizer,
                    "base_act": base_act,
                    "solver": solver,
                }
            }
            torch.save(best_state_dict, os.path.join(save_dir, "best_model.pt"))
        
        # Update tqdm progress bar with live loss info
        pbar.set_postfix({
            "train": f"{train_loss_val:.3e}",
            "test": f"{test_loss_val:.3e}",
            "best": f"{best_test_loss:.3e}",
        })
            
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s! Best Test MSE: {best_test_loss:.4e}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    
    # Load best model for final evaluation & plotting
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict["model_state_dict"])
        
    with torch.no_grad():
        final_pred = node(y0=y0_init, t=t_full).cpu().numpy()
        
    y_full_np = y_full.cpu().numpy()
    t_full_np = t_full.cpu().numpy()
    
    # Generate output plots
    plot_label = f"MLP-ODE ({base_act.upper()})" if model_type.lower() == "mlp" else f"KAN-ODE ({basis_func.upper()})"
    plot_trajectory_comparison(
        t_full=t_full_np,
        y_true=y_full_np,
        y_pred=final_pred,
        t_split=t_train_end,
        title=f"{plot_label} + {solver.upper()} on Lotka-Volterra",
        save_path=os.path.join(save_dir, "trajectory_comparison.png"),
    )
    
    plot_phase_space(
        y_true=y_full_np,
        y_pred=final_pred,
        train_len=len(t_train),
        title=f"Phase Space: True vs {plot_label}",
        save_path=os.path.join(save_dir, "phase_space.png"),
    )
    
    plot_loss_curves(
        train_losses=train_losses,
        test_losses=test_losses,
        title=f"{plot_label} Training Dynamics (+ {solver.upper()})",
        save_path=os.path.join(save_dir, "loss_curves.png"),
    )
    
    print(f"Results and plots saved to '{save_dir}'.")
    
    return {
        "model": model,
        "node": node,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "best_test_mse": best_test_loss,
        "final_train_mse": train_losses[-1],
        "final_test_mse": test_losses[-1],
        "pred_trajectory": final_pred,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KAN-ODE or MLP-ODE on Lotka-Volterra Dynamics")
    parser.add_argument("--model", type=str, default="kan", choices=["kan", "mlp"], help="Model architecture ('kan' or 'mlp')")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 10, 2], help="KAN Hidden layer dimensions (e.g. 2 10 2)")
    parser.add_argument("--mlp_layers", type=int, nargs="+", default=[2, 14, 8, 8, 2], help="MLP Layer dimensions (default: 2 14 8 8 2 for 252 params)")
    parser.add_argument("--grid_len", type=int, default=5, help="Number of grid centers for KAN basis")
    parser.add_argument(
        "--basis",
        type=str,
        default="rbf",
        choices=["rbf", "rswaf", "iqf", "bspline", "chebyshev", "lagrange", "newton"],
        help="KAN Basis function",
    )
    parser.add_argument("--solver", type=str, default="tsit5", choices=["tsit5", "rk4", "dopri5", "euler", "midpoint", "heun"], help="ODE Integrator")
    parser.add_argument("--substeps", type=int, default=2, help="Integration substeps per interval")
    parser.add_argument("--act", type=str, default="silu", help="Activation function (silu, tanh, relu)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (paper uses 5e-4)")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--act_reg", type=float, default=0.0, help="L1 regularization weight")
    parser.add_argument("--entropy_reg", type=float, default=0.0, help="Entropy regularization weight")
    parser.add_argument("--save_dir", type=str, default="results/run_experiment", help="Directory to save checkpoints and plots")
    parser.add_argument("--print_freq", type=int, default=200, help="Epoch print frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train_kan_ode(
        model_type=args.model,
        layers_hidden=args.layers,
        mlp_layers=args.mlp_layers,
        grid_len=args.grid_len,
        basis_func=args.basis,
        solver=args.solver,
        substeps=args.substeps,
        base_act=args.act,
        lr=args.lr,
        num_epochs=args.epochs,
        act_reg=args.act_reg,
        entropy_reg=args.entropy_reg,
        save_dir=args.save_dir,
        print_freq=args.print_freq,
        seed=args.seed,
    )
