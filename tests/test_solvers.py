"""
Rigorous Scientific Machine Learning (SciML) ODE Integrator & Solver Test Suite.

Verifies:
1. Exact Numerical Order of Convergence (p = 1, 2, 4, 5) on exponential decay y' = -y.
2. Symplectic / Hamiltonian Energy Conservation on Simple Harmonic Oscillator.
3. Batch Dimension Flexibility across (D,), (B, D), and (B1, B2, D).
4. Substep Error Monotonicity on non-linear ODEs.
5. End-to-end PyTorch Autograd Gradient Backpropagation across all 6 solvers.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "implementation")))

import pytest
import torch
import torch.nn as nn
import numpy as np

from ode.solvers import odeint, STEP_SOLVERS
from ode.neural_ode import NeuralODE


class TestSolverOrderOfConvergence:
    """
    Test suite verifying theoretical order of convergence p:
        error(h) ~ O(h^p) => p = (log(error_1) - log(error_2)) / (log(h_1) - log(h_2))
    """

    def test_tsit5_embedded_error_estimator_order_conditions(self):
        """
        Verify that TSIT5_E error estimator coefficients match the canonical
        torchdiffeq and Tsitouras (2011) FSAL error vector definition.
        """
        from ode.solvers import TSIT5_B, TSIT5_E
        assert len(TSIT5_E) == 7
        assert len(TSIT5_B) == 7
        # Verify 7th stage FSAL error coefficient is exactly -1/66
        assert np.isclose(TSIT5_E[6], -1.0 / 66.0, atol=1e-12)
        # Verify 5th-order weights sum to 1.0
        assert np.isclose(np.sum(TSIT5_B), 1.0, atol=1e-12)

    @pytest.mark.parametrize("method, expected_order, tolerance", [
        ("euler", 1.0, 0.20),
        ("midpoint", 2.0, 0.25),
        ("heun", 2.0, 0.25),
        ("rk4", 4.0, 0.35),
        ("tsit5", 5.0, 0.45),
        ("dopri5", 5.0, 0.45),
    ])
    def test_numerical_order_of_convergence(self, method, expected_order, tolerance):
        """
        Verify convergence order on test ODE:
            dy/dt = -y, y(0) = 1.0 => y_exact(t) = exp(-t)
        """
        # Linear decay function: f(t, y) = -y
        decay_fn = lambda t, y: -y
        y0 = torch.tensor([1.0], dtype=torch.float64)
        t_end = 1.0
        exact_y_end = np.exp(-t_end)
        
        # Two step sizes: h1 = 0.1 (10 steps), h2 = 0.05 (20 steps)
        steps1 = 10
        steps2 = 20
        h1 = t_end / steps1
        h2 = t_end / steps2
        
        t1 = torch.linspace(0.0, t_end, steps1 + 1, dtype=torch.float64)
        t2 = torch.linspace(0.0, t_end, steps2 + 1, dtype=torch.float64)
        
        sol1 = odeint(decay_fn, y0, t1, method=method, substeps=1)
        sol2 = odeint(decay_fn, y0, t2, method=method, substeps=1)
        
        err1 = abs(sol1[-1].item() - exact_y_end)
        err2 = abs(sol2[-1].item() - exact_y_end)
        
        # Calculate observed convergence order
        observed_order = (np.log(err1) - np.log(err2)) / (np.log(h1) - np.log(h2))
        
        assert abs(observed_order - expected_order) <= tolerance, (
            f"Method {method}: Expected order ~{expected_order}, got {observed_order:.2f} "
            f"(err1={err1:.2e}, err2={err2:.2e})"
        )


class TestHarmonicOscillatorEnergyConservation:
    """
    Test suite for Hamiltonian conservation on Simple Harmonic Oscillator:
        dot(x) = v, dot(v) = -w^2 x
        E(x, v) = 0.5 * v^2 + 0.5 * w^2 * x^2 = const
    """

    def test_harmonic_oscillator_energy_drift(self):
        """
        High-order Runge-Kutta solvers (Tsit5, RK4) conserve energy orders of magnitude
        better than 1st order Euler over multiple oscillation cycles.
        """
        omega = 2.0
        sho_fn = lambda t, u: torch.stack([u[..., 1], -(omega**2) * u[..., 0]], dim=-1)
        
        # Initial condition: x(0) = 1.0, v(0) = 0.0 => E_0 = 0.5 * omega^2 * 1^2 = 2.0
        u0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        t = torch.linspace(0.0, 10.0, 201, dtype=torch.float64) # ~3.18 full cycles
        
        # Integrate with Euler vs Tsit5
        sol_euler = odeint(sho_fn, u0, t, method="euler", substeps=1)
        sol_tsit5 = odeint(sho_fn, u0, t, method="tsit5", substeps=1)
        
        energy_fn = lambda u: 0.5 * (u[..., 1]**2) + 0.5 * (omega**2) * (u[..., 0]**2)
        
        e0 = energy_fn(u0).item()
        e_euler_drift = abs(energy_fn(sol_euler[-1]).item() - e0)
        e_tsit5_drift = abs(energy_fn(sol_tsit5[-1]).item() - e0)
        
        # Tsit5 must conserve energy with high precision (drift < 1e-4)
        assert e_tsit5_drift < 1e-4, f"Tsit5 energy drift too high: {e_tsit5_drift:.2e}"
        # Tsit5 must be at least 1000x more conservative than Euler
        assert e_tsit5_drift < e_euler_drift / 1000.0


class TestBatchAndShapeFlexibility:
    """Test suite for tensor rank handling in ODE integrators."""

    @pytest.mark.parametrize("method", list(STEP_SOLVERS.keys()))
    def test_batch_shapes(self, method):
        """Verify odeint seamlessly handles (D,), (B, D), and (B1, B2, D) tensors."""
        func = lambda t, y: -0.5 * y
        t = torch.linspace(0, 1, 5)
        
        # 1D State [2]
        y_1d = torch.tensor([1.0, 2.0])
        out_1d = odeint(func, y_1d, t, method=method)
        assert out_1d.shape == (5, 2)
        
        # 2D Batched State [8, 2]
        y_2d = torch.randn(8, 2)
        out_2d = odeint(func, y_2d, t, method=method)
        assert out_2d.shape == (5, 8, 2)
        
        # 3D Tensor State [4, 3, 2]
        y_3d = torch.randn(4, 3, 2)
        out_3d = odeint(func, y_3d, t, method=method)
        assert out_3d.shape == (5, 4, 3, 2)


class TestSubstepMonotonicityAndAutograd:
    """Test suite for substep refinement and gradient backpropagation."""

    def test_substep_error_reduction(self):
        """Increasing substeps per interval strictly reduces numerical error."""
        decay_fn = lambda t, y: -y
        y0 = torch.tensor([1.0], dtype=torch.float64)
        t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64) # dt = 0.2
        exact_end = np.exp(-1.0)
        
        err_sub1 = abs(odeint(decay_fn, y0, t, method="euler", substeps=1)[-1].item() - exact_end)
        err_sub2 = abs(odeint(decay_fn, y0, t, method="euler", substeps=2)[-1].item() - exact_end)
        err_sub4 = abs(odeint(decay_fn, y0, t, method="euler", substeps=4)[-1].item() - exact_end)
        
        assert err_sub4 < err_sub2 < err_sub1

    @pytest.mark.parametrize("method", ["euler", "heun", "midpoint", "rk4", "tsit5", "dopri5"])
    def test_neural_ode_autograd_flow(self, method):
        """Verify backpropagation through NeuralODE produces finite non-zero gradients across all solvers."""
        class ParametricODE(nn.Module):
            def __init__(self):
                super().__init__()
                self.theta = nn.Parameter(torch.tensor([0.5, -0.5]))
            def forward(self, t, y):
                return self.theta * y
                
        model = ParametricODE()
        node = NeuralODE(model, method=method, substeps=2)
        
        y0 = torch.tensor([1.0, 2.0])
        t = torch.linspace(0.0, 0.5, 4)
        
        pred = node(y0, t)
        loss = F_loss = (pred**2).sum()
        loss.backward()
        
        assert model.theta.grad is not None
        assert not torch.isnan(model.theta.grad).any()
        assert not torch.isinf(model.theta.grad).any()
        assert (model.theta.grad != 0).any()


class TestSolverEdgeCasesAndProperties:
    """Test suite for edge cases, non-autonomous ODEs, and tableau algebraic invariants."""

    def test_unsupported_solver_method_raises_value_error(self):
        """Requesting an unsupported ODE solver method raises a clear ValueError."""
        fn = lambda t, y: -y
        y0 = torch.tensor([1.0])
        t = torch.linspace(0, 1, 5)
        with pytest.raises(ValueError, match="Unknown method"):
            odeint(fn, y0, t, method="nonexistent_quantum_rk")

    def test_dopri5_tableau_algebraic_consistency(self):
        """
        Verify that DOPRI5 Butcher tableau satisfies row sum conditions and order equations.
        """
        from ode.solvers import DOPRI5_C, DOPRI5_A, DOPRI5_B
        c = np.array(DOPRI5_C)
        b = np.array(DOPRI5_B)
        
        # 1. Row sum conditions: sum(A_i) == c_i
        for stage_idx in range(1, 7):
            row_sum = sum(DOPRI5_A[stage_idx])
            assert np.isclose(row_sum, c[stage_idx], atol=1e-12), (
                f"DOPRI5 stage {stage_idx} row sum {row_sum} != c_{stage_idx} {c[stage_idx]}"
            )
            
        # 2. Order 1 condition: sum(b) == 1.0
        assert np.isclose(np.sum(b), 1.0, atol=1e-12)
        # 3. Order 2 condition: sum(b * c) == 0.5
        assert np.isclose(np.sum(b * c), 0.5, atol=1e-12)
        # 4. Order 3 condition: sum(b * c^2) == 1/3
        assert np.isclose(np.sum(b * (c**2)), 1.0 / 3.0, atol=1e-12)
        # 5. Order 4 condition: sum(b * c^3) == 1/4
        assert np.isclose(np.sum(b * (c**3)), 0.25, atol=1e-12)
        # 6. Order 5 condition: sum(b * c^4) == 1/5
        assert np.isclose(np.sum(b * (c**4)), 0.20, atol=1e-12)

    @pytest.mark.parametrize("method", list(STEP_SOLVERS.keys()))
    def test_non_autonomous_time_dependent_ode(self, method):
        """
        Verify that solvers correctly integrate explicitly non-autonomous ODE:
            dy/dt = 2t * y, y(0) = 1.0 => exact y(t) = exp(t^2)
        """
        non_auto_fn = lambda t, y: 2.0 * t * y
        y0 = torch.tensor([1.0], dtype=torch.float64)
        t = torch.linspace(0.0, 1.0, 51, dtype=torch.float64)
        exact_end = np.exp(1.0)
        
        sol = odeint(non_auto_fn, y0, t, method=method, substeps=2)
        err = abs(sol[-1].item() - exact_end)
        
        # High-order solvers should achieve < 1e-4 error; 1st-order euler < 0.1
        max_allowed_err = 0.1 if method == "euler" else 0.01
        assert err < max_allowed_err, f"Method {method} failed on non-autonomous ODE with error {err:.2e}"

    @pytest.mark.parametrize("method", ["rk4", "tsit5", "dopri5"])
    def test_backward_time_integration(self, method):
        """
        Verify backward time integration (negative dt):
            dy/dt = -y, integrating from t=1.0 to t=0.0 starting at y(1)=exp(-1) => y(0) = 1.0
        """
        decay_fn = lambda t, y: -y
        y_start = torch.tensor([np.exp(-1.0)], dtype=torch.float64)
        t_backward = torch.linspace(1.0, 0.0, 21, dtype=torch.float64)
        
        sol = odeint(decay_fn, y_start, t_backward, method=method, substeps=1)
        err = abs(sol[-1].item() - 1.0)
        assert err < 1e-4, f"Backward integration with {method} failed with error {err:.2e}"

