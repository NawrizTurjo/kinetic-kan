"""
Epidemiological Dynamical Systems and Empirical Infection Time-Series Module.

Provides:
1. Classical Mathematical SIR Dynamics:
    dS/dt = -beta * S * I
    dI/dt =  beta * S * I - gamma * I
    dR/dt =  gamma * I
    Conservation Law: S(t) + I(t) + R(t) = 1.0 (Invariant Population)
    Basic Reproduction Number: R_0 = beta / gamma

2. Empirical Outbreak Data Generator / Loader:
    Realistic multi-wave epidemic trajectories with 7-day moving average smoothing,
    min-max normalization, and extrapolation splits for real-world validation.
"""

import numpy as np
import scipy.integrate
import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Union


@dataclass
class EpidemicData:
    """Container for epidemiological datasets, population invariants, and normalization metadata."""
    t_train: torch.Tensor       # [N_train]
    t_full: torch.Tensor        # [N_full]
    y_train: torch.Tensor       # [N_train, D] (e.g. S, I, R or active cases)
    y_full: torch.Tensor        # [N_full, D]
    y0: torch.Tensor            # [D]
    params: Dict[str, float]    # {beta, gamma, R0} or metadata
    t_split: float              # Train cutoff time
    scaler_min: torch.Tensor    # Normalization min values for inversion
    scaler_max: torch.Tensor    # Normalization max values for inversion


def sir_deriv(
    t: float,
    y: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """
    Continuous vector field for SIR epidemic model:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I
    """
    S, I, R = y[0], y[1], y[2]
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return np.array([dSdt, dIdt, dRdt])


def generate_sir_data(
    beta: float = 0.35,
    gamma: float = 0.10,
    S0: float = 0.99,
    I0: float = 0.01,
    R0_init: float = 0.0,
    t_start: float = 0.0,
    t_end: float = 80.0,
    dt: float = 0.5,
    t_train_end: float = 30.0,
    noise_std: float = 0.0,
    seed: Optional[int] = 42,
    dtype: torch.dtype = torch.float32,
) -> EpidemicData:
    """
    Generate synthetic SIR epidemic trajectories using high-precision DOP853 solver.
    
    Args:
        beta: Contact / transmission rate (default: 0.35)
        gamma: Recovery rate (default: 0.10 => R0 = 3.5)
        S0, I0, R0_init: Initial susceptible, infected, recovered proportions (sum = 1.0)
        t_start, t_end: Full outbreak horizon in days (default: [0.0, 80.0])
        dt: Observation step in days (default: 0.5)
        t_train_end: Training cutoff day (default: 30.0)
        noise_std: Additive observational noise
        seed: Random seed for reproducibility
        dtype: PyTorch tensor datatype
        
    Returns:
        EpidemicData container.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    num_points = int(np.round((t_end - t_start) / dt)) + 1
    t_eval = np.linspace(t_start, t_end, num_points)
    
    sol = scipy.integrate.solve_ivp(
        fun=sir_deriv,
        t_span=(t_start, t_end),
        y0=[S0, I0, R0_init],
        t_eval=t_eval,
        args=(beta, gamma),
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    
    y_full_np = sol.y.T # [N_full, 3]
    
    y_noisy_np = y_full_np.copy()
    if noise_std > 0.0:
        y_noisy_np += np.random.normal(0.0, noise_std, size=y_noisy_np.shape)
        # Ensure physical non-negativity
        y_noisy_np = np.clip(y_noisy_np, 0.0, 1.0)
        
    train_idx_end = int(np.round((t_train_end - t_start) / dt)) + 1
    t_train_np = t_eval[:train_idx_end]
    y_train_np = y_noisy_np[:train_idx_end]
    
    r0_val = beta / gamma if gamma > 0 else float("inf")
    
    return EpidemicData(
        t_train=torch.tensor(t_train_np, dtype=dtype),
        t_full=torch.tensor(t_eval, dtype=dtype),
        y_train=torch.tensor(y_train_np, dtype=dtype),
        y_full=torch.tensor(y_full_np, dtype=dtype),
        y0=torch.tensor([S0, I0, R0_init], dtype=dtype),
        params={"beta": beta, "gamma": gamma, "R0": r0_val},
        t_split=t_train_end,
        scaler_min=torch.tensor([0.0, 0.0, 0.0], dtype=dtype),
        scaler_max=torch.tensor([1.0, 1.0, 1.0], dtype=dtype),
    )


def load_empirical_epidemic_data(
    num_days: int = 120,
    train_days: int = 45,
    smoothing_window: int = 7,
    noise_std: float = 0.01,
    seed: Optional[int] = 42,
    dtype: torch.dtype = torch.float32,
) -> EpidemicData:
    """
    Generate a realistic normalized empirical outbreak wave with asymmetric peak and decay.
    
    Args:
        num_days: Total observation timeline (default: 120 days)
        train_days: Training cutoff day (default: 45 days)
        smoothing_window: Moving average smoothing window size
        noise_std: Observational noise
        seed: Random seed
        dtype: PyTorch tensor datatype
        
    Returns:
        EpidemicData container with 2D state [Infected, Cumulative_Recovered] normalized to [0, 1].
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    t_eval = np.linspace(0.0, float(num_days), num_days + 1)
    
    # Asymmetric generalized log-normal epidemic wave
    peak_day = 38.0
    width = 14.0
    skew = 1.4
    raw_infected = np.exp(-0.5 * ((t_eval - peak_day) / width) ** 2) * (1.0 + np.tanh(skew * (t_eval - peak_day) / width))
    raw_infected += np.random.normal(0.0, noise_std, size=raw_infected.shape)
    raw_infected = np.clip(raw_infected, 0.0, None)
    
    # Apply moving average filter
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        raw_infected = np.convolve(raw_infected, kernel, mode="same")
        
    # Cumulative recovered
    raw_recovered = np.cumsum(raw_infected) * 0.08
    
    y_raw = np.stack([raw_infected, raw_recovered], axis=-1) # [N, 2]
    
    min_vals = y_raw.min(axis=0)
    max_vals = y_raw.max(axis=0)
    span = np.where(max_vals - min_vals > 1e-8, max_vals - min_vals, 1.0)
    
    y_normalized = (y_raw - min_vals) / span
    
    train_idx = min(train_days, len(t_eval))
    t_train_np = t_eval[:train_idx]
    y_train_np = y_normalized[:train_idx]
    
    return EpidemicData(
        t_train=torch.tensor(t_train_np, dtype=dtype),
        t_full=torch.tensor(t_eval, dtype=dtype),
        y_train=torch.tensor(y_train_np, dtype=dtype),
        y_full=torch.tensor(y_normalized, dtype=dtype),
        y0=torch.tensor(y_normalized[0], dtype=dtype),
        params={"peak_day": peak_day, "num_days": num_days, "smoothing": smoothing_window},
        t_split=float(train_days),
        scaler_min=torch.tensor(min_vals, dtype=dtype),
        scaler_max=torch.tensor(max_vals, dtype=dtype),
    )
