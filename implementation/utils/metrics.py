"""
Scientific Machine Learning (SciML) Metrics, Diagnostic Tools, and Lipschitz Bounds.

Provides standardized quantitative evaluation routines:
1. Error Metrics: MSE, RMSE, MAE, R^2 Score, Relative L2 Error.
2. Gradient Dynamics: Parameter L2 Gradient Norm tracking (||nabla_theta L||_2).
3. Numerical ODE Cost: Number of Function Evaluations (NFE) tracker.
4. Theoretical & Empirical Lipschitz Bounds: Jacobian spectral norm & Spline-KAN bounds.
5. Physical Invariant Violations: Hamiltonian / Energy drift quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Union, Callable, Optional


def compute_mse(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Mean Squared Error (MSE)."""
    if isinstance(y_true, np.ndarray):
        return float(np.mean((y_true - y_pred) ** 2))
    return float(F.mse_loss(y_pred, y_true).item())


def compute_rmse(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(compute_mse(y_true, y_pred)))


def compute_mae(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Mean Absolute Error (MAE)."""
    if isinstance(y_true, np.ndarray):
        return float(np.mean(np.abs(y_true - y_pred)))
    return float(torch.mean(torch.abs(y_pred - y_true)).item())


def compute_r2_score(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Coefficient of Determination (R^2 Score):
        R^2 = 1 - (sum (y - y_hat)^2) / (sum (y - y_mean)^2)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_relative_l2_error(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Relative L2 norm error:
        ||y - y_hat||_2 / ||y||_2
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    diff_norm = np.linalg.norm(y_true - y_pred)
    true_norm = np.linalg.norm(y_true)
    if true_norm < 1e-12:
        return float(diff_norm)
    return float(diff_norm / true_norm)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a PyTorch module.
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute Euclidean L2 norm of the full parameter gradient vector:
        ||nabla_theta L||_2 = sqrt( sum_{p} ||nabla_p L||_2^2 )
        
    Used to track loss landscape sharpness and vanishing/exploding gradients in Neural ODEs.
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm_sq += float(param_norm.item() ** 2)
    return float(np.sqrt(total_norm_sq))


def estimate_lipschitz_bound(
    model: nn.Module,
    x_domain: Optional[torch.Tensor] = None,
    in_features: int = 2,
    num_samples: int = 200,
    perturb_eps: float = 1e-3,
    domain_bounds: Tuple[float, float] = (-2.0, 2.0),
    device: str = "cpu",
) -> float:
    """
    Estimate the empirical Lipschitz constant of the vector field f_theta:
        L_emp = max_{u != v} ( ||f(u) - f(v)||_2 / ||u - v||_2 )
        
    Measures the maximum rate of vector field divergence to assess Picard-Lindelof stability.
    """
    if hasattr(model, "eval") and callable(model.eval):
        model.eval()
    device = torch.device(device)
    
    if x_domain is None:
        u = torch.empty(num_samples, in_features, device=device).uniform_(domain_bounds[0], domain_bounds[1])
    else:
        u = x_domain.to(device)
        num_samples = u.shape[0]
        in_features = u.shape[-1]
        
    perturbation = torch.randn_like(u)
    perturbation = perturbation / (torch.norm(perturbation, dim=-1, keepdim=True) + 1e-12) * perturb_eps
    v = u + perturbation
    
    with torch.no_grad():
        fu = model(u)
        fv = model(v)
        
    diff_f = torch.norm(fu - fv, dim=-1) # [num_samples]
    diff_x = torch.norm(u - v, dim=-1)   # [num_samples]
    
    ratios = diff_f / (diff_x + 1e-12)
    return float(ratios.max().item())


def compute_energy_violation(
    y_pred: Union[torch.Tensor, np.ndarray],
    energy_fn: Callable[[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]],
) -> Dict[str, float]:
    """
    Quantify physical conservation law violations along predicted trajectory:
        - Max Absolute Energy Drift: max |E(t) - E(0)|
        - Mean Energy Drift: mean |E(t) - E(0)|
        - Relative Energy Error: max |E(t) - E(0)| / E(0)
    """
    energy = energy_fn(y_pred)
    if isinstance(energy, torch.Tensor):
        energy = energy.detach().cpu().numpy()
        
    e0 = energy[0]
    drift = np.abs(energy - e0)
    max_drift = float(np.max(drift))
    mean_drift = float(np.mean(drift))
    rel_error = float(max_drift / (np.abs(e0) + 1e-12))
    
    return {
        "max_energy_drift": max_drift,
        "mean_energy_drift": mean_drift,
        "rel_energy_error": rel_error,
        "e0": float(e0),
        "e_final": float(energy[-1]),
    }


def track_nfe(solver_name: str, num_steps: int, substeps: int = 2) -> int:
    """
    Calculate theoretical Number of Function Evaluations (NFE) for Runge-Kutta ODE solvers:
    - Euler: 1 eval/substep
    - Heun / Midpoint (RK2): 2 evals/substep
    - RK4: 4 evals/substep
    - Tsit5 / Dopri5 (RK5): 6-7 evals/substep (with FSAL = 6 evals)
    """
    solver_eval_per_step = {
        "euler": 1,
        "heun": 2,
        "midpoint": 2,
        "rk4": 4,
        "tsit5": 6,
        "dopri5": 6,
    }
    evals = solver_eval_per_step.get(solver_name.lower(), 4)
    return num_steps * substeps * evals
