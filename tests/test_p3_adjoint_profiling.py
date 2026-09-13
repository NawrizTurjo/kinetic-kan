"""
Track A -- Adjoint vs. Autograd Profiling. Owner: Nawriz Ahmed Turjo.

Fast, CPU-only correctness checks for experiments/adjoint_profiling/profiling.py.
These do NOT reproduce Table 4 (that needs GPU + thousands of epochs, see
run_profile.py) -- they verify the measurement primitives themselves are correct
at trivial scale, per the Phase 1 rule of one test file per new module.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "implementation")))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "implementation", "experiments", "adjoint_profiling")))

import pytest
import torch

from ode import NeuralODE, ZeroSumField
from profiling import (
    build_field,
    clone_field,
    train_direct_autograd,
    train_adjoint,
    paired_gradient_relative_error,
    memory_vs_trajectory_length,
)

torchdiffeq = pytest.importorskip("torchdiffeq")


def test_kan_forward_is_torchdiffeq_compatible():
    """Track A task 1: confirm KAN's forward(t, y) needs no wrapper for torchdiffeq."""
    field = build_field(layers_hidden=[2, 4, 2], grid_len=4, seed=0)
    y0 = torch.randn(2)
    t = torch.linspace(0.0, 1.0, 5)
    out = torchdiffeq.odeint(field, y0, t, method="rk4", options={"step_size": 0.1})
    assert out.shape == (5, 2)


def test_rk4_implementations_agree():
    """
    Both profiling paths use RK4 at a matched step size specifically so the
    gradient comparison isolates the differentiation method. This confirms
    that design assumption: the project's own RK4 and torchdiffeq's RK4 must
    produce (near-)identical forward trajectories for the same step size.
    """
    field = build_field(layers_hidden=[2, 4, 2], grid_len=4, seed=0)
    y0 = torch.randn(2)
    t = torch.linspace(0.0, 1.0, 11)
    substeps = 2
    step_size = float((t[1] - t[0]).item() / substeps)

    node = NeuralODE(func=clone_field(field), method="rk4", substeps=substeps)
    out_direct = node(y0=y0, t=t)
    out_td = torchdiffeq.odeint(clone_field(field), y0, t, method="rk4",
                                 options={"step_size": step_size})

    rel_err = (out_direct - out_td).norm() / out_direct.norm()
    assert rel_err.item() < 1e-5


def test_paired_gradient_relative_error_is_small():
    """
    Direct autograd and odeint_adjoint, from identical weights on the same
    matched-step RK4 trajectory, should agree closely -- this is the
    definition-of-done check ($< 5\\times10^{-3}$), run here at toy scale.
    """
    field = build_field(layers_hidden=[2, 4, 2], grid_len=4, seed=0)
    y0 = torch.randn(2)
    t = torch.linspace(0.0, 1.0, 11)
    y_train = torch.randn(11, 2)

    result = paired_gradient_relative_error(field, y0, t, y_train, substeps=2, device="cpu")
    assert result["grad_relative_error"] < 5e-3
    assert result["grad_norm_direct"] > 0.0


def test_paired_gradient_relative_error_handles_zero_sum_field():
    """SIR's track uses ZeroSumField -- confirm the wrapper doesn't break the gradient check."""
    model = build_field(layers_hidden=[3, 4, 3], grid_len=4, conserve=False, seed=0)
    field = ZeroSumField(model)
    y0 = torch.tensor([0.99, 0.01, 0.0])
    t = torch.linspace(0.0, 1.0, 11)
    y_train = torch.randn(11, 3)

    result = paired_gradient_relative_error(field, y0, t, y_train, substeps=2, device="cpu")
    assert result["grad_relative_error"] < 5e-3


def test_train_direct_autograd_runs_and_reports_shape():
    field = build_field(layers_hidden=[2, 4, 2], grid_len=4, seed=0)
    y0 = torch.randn(2)
    t = torch.linspace(0.0, 1.0, 6)
    y_train = torch.randn(6, 2)

    result = train_direct_autograd(field, y0, t, y_train, substeps=2, num_epochs=3, lr=1e-2, device="cpu")
    assert result["num_epochs"] == 3
    assert result["peak_vram_mb"] is None  # CPU: no VRAM to report
    assert result["final_train_mse"] >= 0.0


def test_train_adjoint_runs_and_reports_shape():
    field = build_field(layers_hidden=[2, 4, 2], grid_len=4, seed=0)
    y0 = torch.randn(2)
    t = torch.linspace(0.0, 1.0, 6)
    y_train = torch.randn(6, 2)

    result = train_adjoint(field, y0, t, y_train, substeps=2, num_epochs=3, lr=1e-2, device="cpu")
    assert result["num_epochs"] == 3
    assert result["final_train_mse"] >= 0.0


def test_memory_vs_trajectory_length_sweep_shape():
    sweep = memory_vs_trajectory_length(
        state_dim=2, layers_hidden=[2, 4, 2], grid_len=4,
        n_t_values=[5, 10], device="cpu", substeps=2,
    )
    assert sweep["n_t"] == [5, 10]
    assert len(sweep["direct_autograd_mb"]) == 2
    assert len(sweep["adjoint_mb"]) == 2
    # On CPU there's no VRAM to report; the sweep should still run cleanly.
    assert all(v is None for v in sweep["direct_autograd_mb"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="peak-VRAM measurement requires CUDA")
def test_memory_sweep_shows_adjoint_flatter_than_direct_on_gpu():
    """
    The actual research claim: direct autograd's memory grows with N_t while
    adjoint's stays ~flat. Verified on GPU (VRAM is None/unmeasurable on CPU).
    """
    sweep = memory_vs_trajectory_length(
        state_dim=2, layers_hidden=[2, 4, 2], grid_len=4,
        n_t_values=[10, 200], device="cuda", substeps=2,
    )
    direct_growth = sweep["direct_autograd_mb"][1] - sweep["direct_autograd_mb"][0]
    adjoint_growth = sweep["adjoint_mb"][1] - sweep["adjoint_mb"][0]
    assert direct_growth > adjoint_growth
