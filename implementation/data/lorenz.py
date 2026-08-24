"""
3D Chaotic Lorenz Attractor Dynamical System Dataset Generator.

Models the classical 3D non-linear atmospheric convection system (Lorenz, 1963):
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

Canonical Chaotic Parameter Regime:
    sigma = 10.0  (Prandtl number)
    rho   = 28.0  (Rayleigh number)
    beta  = 8/3   (Geometric factor)

Demonstrates multi-scale non-toy generalization and strange attractor topology.
"""

import numpy as np
import scipy.integrate
import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Dict


@dataclass
class LorenzData:
    """Container for 3D Lorenz Attractor dataset splits and ground truth."""
    t_train: torch.Tensor       # [N_train]
    t_full: torch.Tensor        # [N_full]
    y_train: torch.Tensor       # [N_train, 3] (x, y, z)
    y_full: torch.Tensor        # [N_full, 3]
    y0: torch.Tensor            # [3]
    params: Dict[str, float]    # {sigma, rho, beta}
    t_split: float              # Train cutoff time


def lorenz_deriv(
    t: float,
    y: np.ndarray,
    sigma: float,
    rho: float,
    beta: float,
) -> np.ndarray:
    """
    Continuous vector field for 3D Lorenz attractor:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z
    """
    x, y_coord, z = y[0], y[1], y[2]
    dxdt = sigma * (y_coord - x)
    dydt = x * (rho - z) - y_coord
    dzdt = x * y_coord - beta * z
    return np.array([dxdt, dydt, dzdt])


def generate_lorenz_data(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0: float = 1.0,
    y0: float = 1.0,
    z0: float = 1.0,
    t_start: float = 0.0,
    t_end: float = 20.0,
    dt: float = 0.01,
    t_train_end: float = 8.0,
    noise_std: float = 0.0,
    seed: Optional[int] = 42,
    dtype: torch.dtype = torch.float32,
) -> LorenzData:
    """
    Generate ground-truth 3D Lorenz chaotic trajectory using high-precision DOP853 solver.
    
    Args:
        sigma: Prandtl number (default: 10.0)
        rho: Rayleigh number (default: 28.0)
        beta: Geometric aspect ratio (default: 8/3 = 2.6667)
        x0, y0, z0: Initial 3D coordinates (default: 1.0, 1.0, 1.0)
        t_start, t_end: Full simulation time span (default: [0.0, 20.0])
        dt: Step size for numerical evaluation (default: 0.01 s)
        t_train_end: Training time horizon cutoff (default: 8.0 s)
        noise_std: Standard deviation of additive Gaussian observation noise
        seed: Random seed for deterministic reproducibility
        dtype: PyTorch tensor datatype
        
    Returns:
        LorenzData container.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    num_points = int(np.round((t_end - t_start) / dt)) + 1
    t_eval = np.linspace(t_start, t_end, num_points)
    
    sol = scipy.integrate.solve_ivp(
        fun=lorenz_deriv,
        t_span=(t_start, t_end),
        y0=[x0, y0, z0],
        t_eval=t_eval,
        args=(sigma, rho, beta),
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    
    y_full_np = sol.y.T # [N_full, 3]
    
    # Add noise to training segment if specified
    y_noisy_np = y_full_np.copy()
    if noise_std > 0.0:
        y_noisy_np += np.random.normal(0.0, noise_std, size=y_noisy_np.shape)
        
    train_idx_end = int(np.round((t_train_end - t_start) / dt)) + 1
    t_train_np = t_eval[:train_idx_end]
    y_train_np = y_noisy_np[:train_idx_end]
    
    return LorenzData(
        t_train=torch.tensor(t_train_np, dtype=dtype),
        t_full=torch.tensor(t_eval, dtype=dtype),
        y_train=torch.tensor(y_train_np, dtype=dtype),
        y_full=torch.tensor(y_full_np, dtype=dtype),
        y0=torch.tensor([x0, y0, z0], dtype=dtype),
        params={"sigma": sigma, "rho": rho, "beta": beta},
        t_split=t_train_end,
    )
