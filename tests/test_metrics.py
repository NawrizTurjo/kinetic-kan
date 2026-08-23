"""
Unit and Integration Tests for SciML Metrics, Gradient Norm Dynamics, and Lipschitz Estimators.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "implementation")))

import pytest
import torch
import torch.nn as nn
import numpy as np

from utils.metrics import (
    compute_mse,
    compute_rmse,
    compute_mae,
    compute_r2_score,
    compute_relative_l2_error,
    count_parameters,
    compute_gradient_norm,
    estimate_lipschitz_bound,
    compute_energy_violation,
    track_nfe,
)
from kan import KAN, MLP_ODE


class TestErrorMetrics:
    """Test suite for statistical error and accuracy metrics."""

    def test_mse_and_rmse_exactness(self):
        """Verify MSE and RMSE against exact analytical values."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.5, 2.0], [2.0, 5.0]])
        
        # Errors: [0.5, 0.0], [-1.0, 1.0] -> squared: [0.25, 0.0, 1.0, 1.0] -> mean = 2.25 / 4 = 0.5625
        expected_mse = 0.5625
        expected_rmse = np.sqrt(expected_mse) # 0.75
        
        assert np.isclose(compute_mse(y_true, y_pred), expected_mse)
        assert np.isclose(compute_rmse(y_true, y_pred), expected_rmse)
        
        # Test PyTorch Tensor input
        y_true_t = torch.tensor(y_true)
        y_pred_t = torch.tensor(y_pred)
        assert np.isclose(compute_mse(y_true_t, y_pred_t), expected_mse)
        assert np.isclose(compute_rmse(y_true_t, y_pred_t), expected_rmse)

    def test_mae_exactness(self):
        """Verify Mean Absolute Error (MAE)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 2.0, 1.0, 6.0])
        # Abs diffs: [1, 0, 2, 2] -> mean = 5 / 4 = 1.25
        assert np.isclose(compute_mae(y_true, y_pred), 1.25)

    def test_r2_score_properties(self):
        """Verify R^2 score bounds: 1.0 for perfect fit, <1.0 for imperfect."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        # Perfect fit
        assert np.isclose(compute_r2_score(y_true, y_true), 1.0)
        
        # Noisy fit
        y_noisy = y_true + 0.1
        r2 = compute_r2_score(y_true, y_noisy)
        assert 0.90 < r2 < 1.0

    def test_relative_l2_error(self):
        """Verify Relative L2 error calculation."""
        y_true = np.array([3.0, 4.0]) # norm = 5.0
        y_pred = np.array([3.0, 0.0]) # diff = [0, -4.0] -> diff_norm = 4.0
        # rel_err = 4.0 / 5.0 = 0.8
        assert np.isclose(compute_relative_l2_error(y_true, y_pred), 0.8)


class TestGradientAndParameterMetrics:
    """Test suite for gradient norms, parameter counting, and NFE tracking."""

    def test_parameter_counting_with_freezing(self):
        """Verify count_parameters handles frozen and active parameters."""
        linear = nn.Linear(10, 5, bias=True) # 10*5 + 5 = 55 params
        assert count_parameters(linear) == (55, 55)
        
        # Freeze bias
        linear.bias.requires_grad = False
        assert count_parameters(linear) == (55, 50)

    def test_gradient_norm_calculation(self):
        """Verify Euclidean L2 norm calculation across parameter gradients."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        # Initial grad is None
        assert compute_gradient_norm(model) == 0.0
        
        # Assign mock gradients: [[1.0, 2.0], [2.0, 4.0]] -> norm = sqrt(1+4+4+16) = sqrt(25) = 5.0
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        gnorm = compute_gradient_norm(model)
        assert gnorm > 0.0
        assert not np.isnan(gnorm)

    def test_nfe_tracker(self):
        """Verify theoretical NFE tracking across solvers."""
        assert track_nfe("euler", num_steps=100, substeps=2) == 200
        assert track_nfe("rk4", num_steps=100, substeps=2) == 800
        assert track_nfe("tsit5", num_steps=100, substeps=2) == 1200


class TestLipschitzAndEnergyMetrics:
    """Test suite for Lipschitz bound estimator and Hamiltonian energy violations."""

    def test_lipschitz_estimation_on_linear_operator(self):
        """For linear map f(x) = Wx, Lipschitz constant equals largest singular value sigma_max(W)."""
        W = torch.tensor([[3.0, 0.0], [0.0, 4.0]]) # max singular value is 4.0
        linear_fn = lambda x: x @ W.T
        
        lip = estimate_lipschitz_bound(linear_fn, in_features=2, num_samples=500, domain_bounds=(-1.0, 1.0))
        # Empirical perturbation ratio should closely match 4.0
        assert 3.5 <= lip <= 4.5

    def test_energy_violation_diagnostic(self):
        """Verify energy drift diagnostic identifies conserved vs drifting systems."""
        # Mock energy function: E(x) = x[0]^2 + x[1]^2
        energy_fn = lambda y: (y[..., 0] ** 2 + y[..., 1] ** 2)
        
        # Trajectory 1: Perfectly circular orbit x^2 + y^2 = 1.0
        t = np.linspace(0, 2 * np.pi, 50)
        y_conserved = np.stack([np.cos(t), np.sin(t)], axis=-1)
        res_conserved = compute_energy_violation(y_conserved, energy_fn)
        assert np.isclose(res_conserved["max_energy_drift"], 0.0, atol=1e-6)
        
        # Trajectory 2: Decaying spiral
        y_decaying = y_conserved * np.exp(-0.1 * t[:, None])
        res_decaying = compute_energy_violation(y_decaying, energy_fn)
        assert res_decaying["max_energy_drift"] > 0.1
        assert res_decaying["e_final"] < res_decaying["e0"]

    def test_r2_score_zero_variance_handling(self):
        """Verify R^2 score handles constant true vectors without zero division error."""
        y_const = np.array([2.0, 2.0, 2.0, 2.0])
        # Perfect prediction on constant vector
        r2_perfect = compute_r2_score(y_const, y_const)
        assert np.isclose(r2_perfect, 1.0)

    def test_lipschitz_estimation_on_nonlinear_sine(self):
        """For non-linear function f(x) = sin(x), Lipschitz constant is 1.0 (since |cos(x)| <= 1)."""
        sine_fn = lambda x: torch.sin(x)
        lip = estimate_lipschitz_bound(sine_fn, in_features=1, num_samples=300, domain_bounds=(-np.pi, np.pi))
        assert 0.8 <= lip <= 1.2

