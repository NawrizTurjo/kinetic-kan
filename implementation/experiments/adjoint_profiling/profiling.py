"""
Track A -- core measurement primitives for adjoint vs. autograd profiling.

Design choice, stated up front because it drives every number this module produces:
both the "direct autograd" path (this project's own `NeuralODE`) and the "adjoint"
path (`torchdiffeq.odeint_adjoint`) are run with classical RK4 at a MATCHED step
size. `tests/test_p3_adjoint_profiling.py::test_rk4_implementations_agree` confirms
the two RK4 implementations produce the same forward trajectory to ~1e-7 relative
error. That isolates exactly one variable -- how the gradient is computed -- so any
residual difference in peak memory, wall-clock, or gradient value is attributable to
the differentiation method (backprop through the unrolled solver vs. solving the
continuous adjoint ODE backward in time), not to comparing two different integrators.
"""
import copy
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from kan import KAN
from ode import NeuralODE, ZeroSumField


def build_field(layers_hidden, grid_len, basis_func="rbf", base_act="silu",
                 conserve=False, seed=42) -> nn.Module:
    """Fresh KAN vector field, optionally wrapped for SIR's exact zero-sum invariant."""
    torch.manual_seed(seed)
    model = KAN(layers_hidden=layers_hidden, grid_len=grid_len,
                basis_func=basis_func, base_act=base_act)
    return ZeroSumField(model) if conserve else model


def clone_field(field: nn.Module) -> nn.Module:
    """Deep copy so two runs can start from identical initial weights."""
    return copy.deepcopy(field)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1e6


def train_direct_autograd(field, y0, t, y_train, substeps, num_epochs, lr, device,
                           desc: str = "direct autograd") -> Dict:
    """Profile the project's own NeuralODE: direct backprop through the unrolled RK4."""
    device = torch.device(device)
    field = field.to(device)
    y0, t, y_train = y0.to(device), t.to(device), y_train.to(device)
    node = NeuralODE(func=field, method="rk4", substeps=substeps).to(device)
    optimizer = torch.optim.Adam(field.parameters(), lr=lr)

    _reset_peak_memory(device)
    start = time.perf_counter()
    loss = None
    pbar = tqdm(range(num_epochs), desc=desc, unit="epoch", ncols=110)
    for _ in pbar:
        optimizer.zero_grad()
        pred = node(y0=y0, t=t)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"mse": f"{loss.item():.3e}"})
    elapsed = time.perf_counter() - start

    return {
        "method": "direct_autograd",
        "solver": "rk4", "substeps": substeps,
        "peak_vram_mb": _peak_memory_mb(device),
        "wallclock_s": elapsed,
        "wallclock_s_per_1000_epochs": elapsed / num_epochs * 1000,
        "final_train_mse": loss.item(),
        "num_epochs": num_epochs,
    }


def train_adjoint(field, y0, t, y_train, substeps, num_epochs, lr, device,
                   desc: str = "adjoint") -> Dict:
    """Profile torchdiffeq.odeint_adjoint: continuous adjoint sensitivity, O(1) memory."""
    from torchdiffeq import odeint_adjoint

    device = torch.device(device)
    field = field.to(device)
    y0, t, y_train = y0.to(device), t.to(device), y_train.to(device)
    step_size = float((t[1] - t[0]).item() / substeps)
    optimizer = torch.optim.Adam(field.parameters(), lr=lr)

    _reset_peak_memory(device)
    start = time.perf_counter()
    loss = None
    pbar = tqdm(range(num_epochs), desc=desc, unit="epoch", ncols=110)
    for _ in pbar:
        optimizer.zero_grad()
        pred = odeint_adjoint(field, y0, t, method="rk4", options={"step_size": step_size})
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"mse": f"{loss.item():.3e}"})
    elapsed = time.perf_counter() - start

    return {
        "method": "adjoint",
        "solver": "rk4", "substeps": substeps, "step_size": step_size,
        "peak_vram_mb": _peak_memory_mb(device),
        "wallclock_s": elapsed,
        "wallclock_s_per_1000_epochs": elapsed / num_epochs * 1000,
        "final_train_mse": loss.item(),
        "num_epochs": num_epochs,
    }


def paired_gradient_relative_error(field, y0, t, y_train, substeps, device) -> Dict:
    """
    The blueprint's test_adjoint.py check: compare grad_theta(L) computed by direct
    autograd vs. odeint_adjoint, starting from IDENTICAL weights, on the SAME
    numerical trajectory (matched-step RK4). Returns the relative L2 error between
    the two flattened gradient vectors.
    """
    from torchdiffeq import odeint_adjoint

    device = torch.device(device)
    y0, t, y_train = y0.to(device), t.to(device), y_train.to(device)

    field_direct = clone_field(field).to(device)
    node = NeuralODE(func=field_direct, method="rk4", substeps=substeps).to(device)
    pred_direct = node(y0=y0, t=t)
    F.mse_loss(pred_direct, y_train).backward()
    grad_direct = torch.cat([p.grad.detach().flatten()
                              for p in field_direct.parameters() if p.grad is not None])

    field_adjoint = clone_field(field).to(device)
    step_size = float((t[1] - t[0]).item() / substeps)
    pred_adjoint = odeint_adjoint(field_adjoint, y0, t, method="rk4",
                                   options={"step_size": step_size})
    F.mse_loss(pred_adjoint, y_train).backward()
    grad_adjoint = torch.cat([p.grad.detach().flatten()
                               for p in field_adjoint.parameters() if p.grad is not None])

    diff = (grad_direct - grad_adjoint).norm().item()
    denom = grad_direct.norm().item()
    traj_diff = (pred_direct.detach() - pred_adjoint.detach()).norm()
    traj_denom = pred_direct.detach().norm()

    return {
        "grad_relative_error": diff / denom if denom > 0 else float("nan"),
        "grad_norm_direct": grad_direct.norm().item(),
        "grad_norm_adjoint": grad_adjoint.norm().item(),
        "trajectory_relative_diff": (traj_diff / traj_denom).item() if traj_denom > 0 else float("nan"),
    }


def memory_vs_trajectory_length(
    state_dim: int,
    layers_hidden: List[int],
    grid_len: int,
    n_t_values: List[int],
    device,
    substeps: int = 2,
    seed: int = 42,
) -> Dict[str, List]:
    """
    Single forward+backward pass (no training loop) at each trajectory length N_t,
    isolating the O(1)-vs-O(N_t) memory claim from any training-loop/optimizer
    overhead. This is the headline figure: memory vs. trajectory length, both
    methods overlaid.
    """
    from torchdiffeq import odeint_adjoint

    device = torch.device(device)
    direct_mb, adjoint_mb = [], []
    for n_t in n_t_values:
        torch.manual_seed(seed)
        y0 = torch.randn(state_dim, device=device)
        t = torch.linspace(0.0, 1.0, n_t, device=device)
        target = torch.randn(n_t, state_dim, device=device)
        step_size = float((t[1] - t[0]).item() / substeps)

        field = build_field(layers_hidden, grid_len, seed=seed).to(device)
        node = NeuralODE(func=field, method="rk4", substeps=substeps).to(device)
        _reset_peak_memory(device)
        pred = node(y0=y0, t=t)
        F.mse_loss(pred, target).backward()
        direct_mb.append(_peak_memory_mb(device))
        del field, node, pred

        field2 = build_field(layers_hidden, grid_len, seed=seed).to(device)
        _reset_peak_memory(device)
        pred2 = odeint_adjoint(field2, y0, t, method="rk4", options={"step_size": step_size})
        F.mse_loss(pred2, target).backward()
        adjoint_mb.append(_peak_memory_mb(device))
        del field2, pred2

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"n_t": n_t_values, "direct_autograd_mb": direct_mb, "adjoint_mb": adjoint_mb}
