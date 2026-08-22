import numpy as np
import scipy.integrate
import torch
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class LotkaVolterraData:
    """Container for Lotka-Volterra dataset splits and ground truth."""
    t_train: torch.Tensor       # [N_train]
    t_full: torch.Tensor        # [N_full]
    y_train: torch.Tensor       # [N_train, 2] (training segment)
    y_full: torch.Tensor        # [N_full, 2] (ground truth full trajectory)
    y0: torch.Tensor            # [2]
    params: dict                # {alpha, beta, gamma, delta}
    t_split: float              # Train cutoff time (e.g. 3.5)


def lotka_volterra_deriv(t: float, y: np.ndarray, alpha: float, beta: float, delta: float, gamma: float) -> np.ndarray:
    """
    Ground-truth continuous dynamics:
        dx/dt = alpha * x - beta * x * y
        dy/dt = delta * x * y - gamma * y
    """
    x, y_prey = y[0], y[1]
    dxdt = alpha * x - beta * x * y_prey
    dydt = delta * x * y_prey - gamma * y_prey
    return np.array([dxdt, dydt])


def generate_lotka_volterra_data(
    alpha: float = 1.5,
    beta: float = 1.0,
    gamma: float = 3.0,
    delta: float = 1.0,
    x0: float = 1.0,
    y0: float = 1.0,
    t_start: float = 0.0,
    t_end: float = 14.0,
    dt: float = 0.1,
    t_train_end: float = 3.5,
    noise_std: float = 0.0,
    seed: Optional[int] = 42,
    dtype: torch.dtype = torch.float32,
) -> LotkaVolterraData:
    """
    Generate synthetic Lotka-Volterra benchmark dataset as used in the KAN-ODE paper.
    
    Args:
        alpha: Prey birth rate (default: 1.5)
        beta: Predation rate (default: 1.0)
        gamma: Predator death rate (default: 3.0)
        delta: Predator reproduction rate (default: 1.0)
        x0, y0: Initial prey and predator concentrations (default: 1.0, 1.0)
        t_start, t_end: Full simulation time span (default: [0.0, 14.0])
        dt: Time step size (default: 0.1)
        t_train_end: Time cutoff for training set (default: 3.5)
        noise_std: Standard deviation of additive Gaussian observation noise
        seed: Random seed for reproducibility
        dtype: PyTorch tensor datatype
        
    Returns:
        LotkaVolterraData object containing training and full trajectories.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    num_points = int(np.round((t_end - t_start) / dt)) + 1
    t_eval = np.linspace(t_start, t_end, num_points)
    
    # Solve ground truth using high-precision Runge-Kutta (DOP853 / RK45)
    sol = scipy.integrate.solve_ivp(
        fun=lotka_volterra_deriv,
        t_span=(t_start, t_end),
        y0=[x0, y0],
        t_eval=t_eval,
        args=(alpha, beta, delta, gamma),
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    
    y_full_np = sol.y.T # Shape: [N_full, 2]
    
    # Add noise to training segment if requested
    y_noisy_np = y_full_np.copy()
    if noise_std > 0.0:
        y_noisy_np += np.random.normal(0.0, noise_std, size=y_noisy_np.shape)
        
    # Split index
    train_idx_end = int(np.round((t_train_end - t_start) / dt)) + 1
    
    t_train_np = t_eval[:train_idx_end]
    y_train_np = y_noisy_np[:train_idx_end]
    
    return LotkaVolterraData(
        t_train=torch.tensor(t_train_np, dtype=dtype),
        t_full=torch.tensor(t_eval, dtype=dtype),
        y_train=torch.tensor(y_train_np, dtype=dtype),
        y_full=torch.tensor(y_full_np, dtype=dtype),
        y0=torch.tensor([x0, y0], dtype=dtype),
        params={"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
        t_split=t_train_end,
    )
