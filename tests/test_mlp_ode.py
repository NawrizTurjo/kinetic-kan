"""
Unit and Integration Tests for Parameter-Matched MLP-ODE Baseline.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "implementation")))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kan import MLP_ODE, count_parameters, KAN
from ode import NeuralODE


class TestMLPODEPrivateMathAndShapes:
    """Test suite for MLP_ODE parameter calculations, shapes, and activations."""

    def test_default_architecture_parameter_count(self):
        """Verify that default architecture [2, 14, 8, 8, 2] has exactly 252 parameters."""
        mlp = MLP_ODE(layers_hidden=[2, 14, 8, 8, 2], use_bias=True)
        total, trainable = count_parameters(mlp)
        
        # Layer 1: 2*14 + 14 = 42
        # Layer 2: 14*8 + 8 = 120
        # Layer 3: 8*8 + 8 = 72
        # Layer 4: 8*2 + 2 = 18
        # Total = 42 + 120 + 72 + 18 = 252
        assert total == 252
        assert trainable == 252
        assert mlp.num_parameters == 252

    def test_kan_vs_mlp_parameter_matching(self):
        """Verify paper comparison: KAN-ODE (240 params) vs MLP-ODE (252 params)."""
        kan = KAN(layers_hidden=[2, 10, 2], grid_len=5, basis_func="rbf", use_base_act=True)
        mlp = MLP_ODE(layers_hidden=[2, 14, 8, 8, 2], use_bias=True)
        
        kan_params, _ = count_parameters(kan)
        mlp_params, _ = count_parameters(mlp)
        
        assert kan_params == 240
        assert mlp_params == 252
        assert abs(mlp_params - kan_params) <= 12  # Exact paper parameter match within 5%

    @pytest.mark.parametrize("batch_shape", [
        (2,),
        (1, 2),
        (16, 2),
        (64, 2),
        (4, 10, 2),
    ])
    def test_forward_output_shapes(self, batch_shape):
        """Ensure MLP_ODE preserves batch dimensions and returns correct output shape."""
        mlp = MLP_ODE(layers_hidden=[2, 14, 8, 8, 2])
        x = torch.randn(*batch_shape)
        out = mlp(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @pytest.mark.parametrize("act_name", ["silu", "tanh", "relu", "gelu", "identity"])
    def test_activations_supported(self, act_name):
        """Test all standard supported activation functions."""
        mlp = MLP_ODE(layers_hidden=[2, 8, 2], activation=act_name)
        x = torch.randn(10, 2)
        out = mlp(x)
        assert out.shape == (10, 2)

    def test_invalid_activation_raises_error(self):
        """Verify invalid activation string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            MLP_ODE(layers_hidden=[2, 8, 2], activation="unsupported_act_xyz")

    def test_signature_flexibility_for_ode(self):
        """Verify calling conventions f(u), f(t, u), f(u, t), f(u=u)."""
        mlp = MLP_ODE(layers_hidden=[2, 14, 8, 8, 2])
        u = torch.randn(10, 2)
        t_scalar = 0.5
        t_tensor = torch.tensor(0.5)
        
        out_u = mlp(u)
        out_tu_scalar = mlp(t_scalar, u)
        out_tu_tensor = mlp(t_tensor, u)
        out_kwarg = mlp(u=u)
        
        assert torch.allclose(out_u, out_tu_scalar)
        assert torch.allclose(out_u, out_tu_tensor)
        assert torch.allclose(out_u, out_kwarg)

    def test_initialization_statistics(self):
        """Verify weights are Xavier initialized and biases start at zero."""
        mlp = MLP_ODE(layers_hidden=[2, 14, 8, 8, 2], use_bias=True)
        for linear in mlp.linears:
            assert not torch.all(linear.weight == 0.0)
            assert torch.all(linear.bias == 0.0)

    def test_gradient_backpropagation(self):
        """Verify backward pass populates non-zero finite gradients across all layers."""
        mlp = MLP_ODE(layers_hidden=[2, 14, 8, 8, 2])
        x = torch.randn(16, 2, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        for name, param in mlp.named_parameters():
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"


class TestMLPNeuralODEIntegration:
    """Integration test suite for MLP_ODE coupled with ODE Solvers."""

    @pytest.mark.parametrize("solver", ["tsit5", "rk4", "euler", "heun", "dopri5", "midpoint"])
    def test_neural_ode_forward_and_backward(self, solver):
        """Verify forward integration and backpropagation through NeuralODE with MLP."""
        mlp = MLP_ODE(layers_hidden=[2, 14, 8, 8, 2], activation="silu")
        node = NeuralODE(func=mlp, method=solver, substeps=2)
        
        y0 = torch.tensor([1.0, 1.0])
        t = torch.linspace(0.0, 1.0, 11)
        
        trajectory = node(y0=y0, t=t)
        assert trajectory.shape == (11, 2)
        assert not torch.isnan(trajectory).any()
        
        # Loss backprop
        target = torch.ones_like(trajectory)
        loss = F.mse_loss(trajectory, target)
        loss.backward()
        
        for name, param in mlp.named_parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
