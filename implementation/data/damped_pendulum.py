"""
Non-Linear Damped Pendulum Dynamical System Dataset Generator.

Models the second-order non-linear physical oscillator:
    d^2(theta)/dt^2 + mu * d(theta)/dt + (g / L) * sin(theta) = 0

In state-space formulation with u = [theta, omega]^T:
    du_1/dt = u_2
    du_2/dt = -mu * u_2 - (g / L) * sin(u_1)

Physical Invariants:
    Total Mechanical Energy: E(theta, omega) = 0.5 * m * L^2 * omega^2 + m * g * L * (1 - cos(theta))
    - When mu = 0: dE/dt = 0 (Conservative Hamiltonian system)
    - When mu > 0: dE/dt = -m * L^2 * mu * omega^2 <= 0 (Strictly dissipative system)
"""

import numpy as np
import scipy.integrate
import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Union


@dataclass
class DampedPendulumData:
    """Container for Damped Pendulum dataset splits, ground truth, and energy diagnostics."""
    t_train: torch.Tensor       # [N_train]
    t_full: torch.Tensor        # [N_full]
    y_train: torch.Tensor       # [N_train, 2] (theta, omega)
    y_full: torch.Tensor        # [N_full, 2]
    y0: torch.Tensor            # [2]
    params: Dict[str, float]    # {mu, g, length, mass}
    t_split: float              # Train cutoff time
    energy_full: torch.Tensor   # [N_full] Mechanical energy over time


def damped_pendulum_deriv(
    t: float,
    y: np.ndarray,
    mu: float,
    g: float,
    length: float,
) -> np.ndarray:
    """
    Continuous vector field for non-linear damped pendulum:
        d(theta)/dt = omega
        d(omega)/dt = -mu * omega - (g / length) * sin(theta)
    """
    theta, omega = y[0], y[1]
    dtheta_dt = omega
    domega_dt = -mu * omega - (g / length) * np.sin(theta)
    return np.array([dtheta_dt, domega_dt])


def compute_pendulum_energy(
    y: Union[np.ndarray, torch.Tensor],
    g: float = 9.81,
    length: float = 1.0,
    mass: float = 1.0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute total mechanical energy (kinetic + potential) of pendulum:
        E = 0.5 * m * (L * omega)^2 + m * g * L * (1 - cos(theta))
    """
    if isinstance(y, torch.Tensor):
        theta = y[..., 0]
        omega = y[..., 1]
        kinetic = 0.5 * mass * (length * omega) ** 2
        potential = mass * g * length * (1.0 - torch.cos(theta))
        return kinetic + potential
    else:
        theta = y[..., 0]
        omega = y[..., 1]
        kinetic = 0.5 * mass * (length * omega) ** 2
        potential = mass * g * length * (1.0 - np.cos(theta))
        return kinetic + potential


def generate_damped_pendulum_data(
    mu: float = 0.5,
    g: float = 9.81,
    length: float = 1.0,
    mass: float = 1.0,
    theta0: float = 2.0,
    omega0: float = 0.0,
    t_start: float = 0.0,
    t_end: float = 10.0,
    dt: float = 0.05,
    t_train_end: float = 3.0,
    noise_std: float = 0.0,
    seed: Optional[int] = 42,
    dtype: torch.dtype = torch.float32,
) -> DampedPendulumData:
    """
    Generate ground-truth non-linear damped pendulum dataset using high-precision DOP853 solver.
    
    Args:
        mu: Damping coefficient (stiffness parameter; 0.0 = conservative, >0.0 = dissipative)
        g: Gravitational acceleration (default: 9.81 m/s^2)
        length: Pendulum arm length (default: 1.0 m)
        mass: Pendulum bob mass (default: 1.0 kg)
        theta0, omega0: Initial angular position and velocity (default: 2.0 rad, 0.0 rad/s)
        t_start, t_end: Full simulation time span (default: [0.0, 10.0])
        dt: Observation time step (default: 0.05 s)
        t_train_end: Training time horizon cutoff (default: 3.0 s)
        noise_std: Standard deviation of Gaussian observation noise
        seed: Random seed for deterministic reproducibility
        dtype: PyTorch tensor datatype
        
    Returns:
        DampedPendulumData container.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    num_points = int(np.round((t_end - t_start) / dt)) + 1
    t_eval = np.linspace(t_start, t_end, num_points)
    
    sol = scipy.integrate.solve_ivp(
        fun=damped_pendulum_deriv,
        t_span=(t_start, t_end),
        y0=[theta0, omega0],
        t_eval=t_eval,
        args=(mu, g, length),
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    
    y_full_np = sol.y.T # [N_full, 2]
    
    # Add noise to training segment if specified
    y_noisy_np = y_full_np.copy()
    if noise_std > 0.0:
        y_noisy_np += np.random.normal(0.0, noise_std, size=y_noisy_np.shape)
        
    train_idx_end = int(np.round((t_train_end - t_start) / dt)) + 1
    t_train_np = t_eval[:train_idx_end]
    y_train_np = y_noisy_np[:train_idx_end]
    
    energy_np = compute_pendulum_energy(y_full_np, g=g, length=length, mass=mass)
    
    return DampedPendulumData(
        t_train=torch.tensor(t_train_np, dtype=dtype),
        t_full=torch.tensor(t_eval, dtype=dtype),
        y_train=torch.tensor(y_train_np, dtype=dtype),
        y_full=torch.tensor(y_full_np, dtype=dtype),
        y0=torch.tensor([theta0, omega0], dtype=dtype),
        params={"mu": mu, "g": g, "length": length, "mass": mass},
        t_split=t_train_end,
        energy_full=torch.tensor(energy_np, dtype=dtype),
    )
