import torch
import torch.nn as nn
from typing import Callable, Tuple, List, Optional, Union


# ==============================================================================
# Tsitouras 5/4 Runge-Kutta Coefficients (Tsit5)
# Tsitouras, Ch. (2011). Computers & Mathematics with Applications, 62(2), 770-775.
# ==============================================================================
TSIT5_C = [
    0.0,
    0.161,
    0.327,
    0.9,
    0.9800255409045097,
    1.0,
    1.0,
]

TSIT5_A = [
    [],
    [0.161],
    [-0.008480655492357, 0.3354806554923570],
    [2.897153057105494, -6.359448489975075, 4.362295432869581],
    [5.32586482843925895, -11.74888356406283, 7.495539342889836, -0.09249506636175525],
    [5.86145544294642038, -12.92096931784711, 8.159367898576159, -0.071584973281401006, -0.02826905039406838],
    [0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774],
]

TSIT5_B = [
    0.09646076681806523,
    0.01,
    0.4798896504144996,
    1.379008574103742,
    -3.290069515436081,
    2.324710524099774,
    0.0,
]

TSIT5_E = [  # Error estimator weights: b_sol - b_embedded (5th order - 4th order)
    -0.00178001105222577714,
    -0.0008164344596567469,
    0.007880878010261995,
    -0.1447110071732629,
    0.5823571654525552,
    -0.45808210592918697,
    1.0 / 66.0,
]


# ==============================================================================
# Dormand-Prince 5(4) Coefficients (DOPRI5)
# ==============================================================================
DOPRI5_C = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
DOPRI5_A = [
    [],
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84],
]
DOPRI5_B = [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0]


# ==============================================================================
# Single-Step Solvers
# ==============================================================================

def step_euler(func: Callable, t: torch.Tensor, y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Explicit Euler 1st order step."""
    return y + dt * func(t, y)


def step_midpoint(func: Callable, t: torch.Tensor, y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Explicit Midpoint 2nd order step."""
    k1 = func(t, y)
    y_mid = y + 0.5 * dt * k1
    k2 = func(t + 0.5 * dt, y_mid)
    return y + dt * k2


def step_heun(func: Callable, t: torch.Tensor, y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Heun's 2nd order step (Modified Euler)."""
    k1 = func(t, y)
    y_pred = y + dt * k1
    k2 = func(t + dt, y_pred)
    return y + 0.5 * dt * (k1 + k2)


def step_rk4(func: Callable, t: torch.Tensor, y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Classical Runge-Kutta 4th order step."""
    k1 = func(t, y)
    k2 = func(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = func(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def step_tsit5(func: Callable, t: torch.Tensor, y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Tsitouras 5/4 Runge-Kutta, FIXED step (Tsit5).

    Implements the 5th-order solution weights (TSIT5_A / TSIT5_B) of Tsitouras (2011).
    This is NOT the adaptive Tsit5() of Julia's DifferentialEquations.jl: there is no
    error-based step-size control here, so the embedded 4th-order estimator (TSIT5_E)
    is unused by this function. Step-size control is exposed instead through the
    `substeps` argument of `odeint`.

    Only 6 stages are evaluated. The 7th stage (the FSAL stage) is skipped because
    TSIT5_B[6] == 0.0, so it contributes nothing to the fixed-step update; evaluating
    it would cost one extra function evaluation per step for no change in the result.
    """
    k = []
    # Stage 1
    k.append(func(t, y))

    # Stages 2 to 6 (stage 7 is FSAL-only and has zero solution weight)
    for stage_idx in range(1, 6):
        t_stage = t + TSIT5_C[stage_idx] * dt
        y_stage = y
        for j in range(stage_idx):
            y_stage = y_stage + dt * TSIT5_A[stage_idx][j] * k[j]
        k.append(func(t_stage, y_stage))

    # Combine with 5th order weights
    y_next = y
    for j in range(len(k)):
        if TSIT5_B[j] != 0.0:
            y_next = y_next + dt * TSIT5_B[j] * k[j]

    return y_next


def step_dopri5(func: Callable, t: torch.Tensor, y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Dormand-Prince 5(4), FIXED step (DOPRI5).

    As with `step_tsit5`, this applies only the 5th-order solution weights with no
    adaptive step-size control. The 7th (FSAL) stage is skipped since
    DOPRI5_B[6] == 0.0, giving 6 function evaluations per step.
    """
    k = []
    k.append(func(t, y))

    for stage_idx in range(1, 6):
        t_stage = t + DOPRI5_C[stage_idx] * dt
        y_stage = y
        for j in range(stage_idx):
            y_stage = y_stage + dt * DOPRI5_A[stage_idx][j] * k[j]
        k.append(func(t_stage, y_stage))

    y_next = y
    for j in range(len(k)):
        if DOPRI5_B[j] != 0.0:
            y_next = y_next + dt * DOPRI5_B[j] * k[j]

    return y_next


# Registry of step functions
STEP_SOLVERS = {
    "tsit5": step_tsit5,
    "rk4": step_rk4,
    "dopri5": step_dopri5,
    "euler": step_euler,
    "midpoint": step_midpoint,
    "heun": step_heun,
}


# ==============================================================================
# Full Trajectory Integrator (odeint)
# ==============================================================================

def odeint(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t: torch.Tensor,
    method: str = "tsit5",
    substeps: int = 1,
) -> torch.Tensor:
    """
    Integrate an ODE system dy/dt = func(t, y) from initial state y0 over time points t.
    
    Args:
        func: Callable f(t, y) returning dy/dt of same shape as y.
        y0: Initial state tensor of shape (..., state_dim).
        t: 1D Tensor of evaluation time points [t_0, t_1, ..., t_N].
        method: ODE solver name ('tsit5', 'rk4', 'dopri5', 'euler', 'midpoint', 'heun').
        substeps: Number of intermediate integration substeps per interval for higher accuracy.
        
    Returns:
        Tensor of shape (len(t), ..., state_dim) containing the trajectory.
    """
    method = method.lower()
    if method not in STEP_SOLVERS:
        raise ValueError(f"Unknown method '{method}'. Available: {list(STEP_SOLVERS.keys())}")
        
    step_fn = STEP_SOLVERS[method]
    
    # Initialize trajectory list
    trajectory = [y0]
    curr_y = y0
    
    for i in range(len(t) - 1):
        t_start = t[i]
        t_end = t[i + 1]
        dt_sub = (t_end - t_start) / substeps
        
        for s in range(substeps):
            curr_t = t_start + s * dt_sub
            curr_y = step_fn(func, curr_t, curr_y, dt_sub)
            
        trajectory.append(curr_y)
        
    return torch.stack(trajectory, dim=0)
