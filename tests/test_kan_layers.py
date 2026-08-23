"""
Unit and Integration Tests for KAN Basis Functions, KDense Layers, and KAN Models.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "implementation")))

import pytest
import torch
import torch.nn as nn
from kan import (
    rbf,
    rswaf,
    iqf,
    bspline_basis,
    chebyshev_basis,
    lagrange_basis,
    newton_basis,
    get_basis_function,
    BASIS_FUNCTIONS,
    KDense,
    KAN,
)

ALL_BASES = ["rbf", "rswaf", "iqf", "bspline", "chebyshev", "lagrange", "newton"]


class TestBasisFunctions:
    """Test suite for individual mathematical basis functions."""

    @pytest.mark.parametrize("basis_name", ALL_BASES)
    @pytest.mark.parametrize("batch_size,in_features,grid_len", [
        (1, 2, 5),
        (16, 2, 5),
        (32, 4, 7),
        (5, 1, 3),
    ])
    def test_basis_shapes_and_validity(self, basis_name, batch_size, in_features, grid_len):
        """Verify tensor output shapes and absence of NaNs / Infs."""
        fn = get_basis_function(basis_name)
        grid = torch.linspace(-1.0, 1.0, grid_len)
        h = (grid[1] - grid[0]).item() if grid_len > 1 else 1.0
        x = torch.linspace(-1.0, 1.0, batch_size * in_features).reshape(batch_size, in_features)
        
        out = fn(x, grid, h)
        assert out.shape == (batch_size, in_features, grid_len)
        assert not torch.isnan(out).any(), f"{basis_name} produced NaNs"
        assert not torch.isinf(out).any(), f"{basis_name} produced Infs"

    def test_basis_registry_aliases(self):
        """Verify case-insensitivity and alias resolution."""
        assert get_basis_function("RBF") == rbf
        assert get_basis_function("gaussian") == rbf
        assert get_basis_function("b_spline") == bspline_basis
        assert get_basis_function("spline") == bspline_basis
        assert get_basis_function("cheby") == chebyshev_basis
        assert get_basis_function("lagrangian") == lagrange_basis
        assert get_basis_function("divided_differences") == newton_basis

        with pytest.raises(ValueError):
            get_basis_function("unknown_nonexistent_basis")

    def test_lagrange_cardinal_property(self):
        """Verify Lagrange cardinal property L_i(z_j) = delta_{ij}."""
        grid = torch.linspace(-1.0, 1.0, 5)
        h = (grid[1] - grid[0]).item()
        z_nodes = grid.unsqueeze(1) # [5, 1]
        
        l_eval = lagrange_basis(z_nodes, grid, h)[:, 0, :] # [5, 5]
        identity = torch.eye(5)
        assert torch.allclose(l_eval, identity, atol=1e-5)

    def test_lagrange_partition_of_unity(self):
        """Verify Lagrange partition of unity sum_i L_i(x) = 1."""
        grid = torch.linspace(-1.0, 1.0, 5)
        h = (grid[1] - grid[0]).item()
        x = torch.linspace(-1.0, 1.0, 50).unsqueeze(1)
        
        l_eval = lagrange_basis(x, grid, h)
        l_sum = l_eval.sum(dim=-1)
        assert torch.allclose(l_sum, torch.ones_like(l_sum), atol=1e-5)

    def test_chebyshev_exact_values(self):
        """Verify Chebyshev polynomial evaluation against exact formula at x=0.5."""
        grid = torch.linspace(-1.0, 1.0, 5)
        h = (grid[1] - grid[0]).item()
        x = torch.tensor([[0.5]])
        
        # T_0=1, T_1=0.5, T_2=-0.5, T_3=-1.0, T_4=-0.5
        expected = torch.tensor([1.0, 0.5, -0.5, -1.0, -0.5])
        actual = chebyshev_basis(x, grid, h)[0, 0]
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_newton_exact_values(self):
        """Verify Newton basis against analytical formula at x=0.0."""
        grid = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        h = 0.5
        x = torch.tensor([[0.0]])
        
        # N_0=1, N_1=1, N_2=0.5, N_3=0.0, N_4=0.0
        expected = torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0])
        actual = newton_basis(x, grid, h)[0, 0]
        assert torch.allclose(actual, expected, atol=1e-5)

    @pytest.mark.parametrize("basis_name", ALL_BASES)
    def test_basis_autograd_flow(self, basis_name):
        """Verify gradient backpropagation through every basis function."""
        fn = get_basis_function(basis_name)
        grid = torch.linspace(-1.0, 1.0, 5)
        h = (grid[1] - grid[0]).item()
        
        x = torch.linspace(-0.8, 0.8, 10).unsqueeze(1).requires_grad_(True)
        out = fn(x, grid, h)
        loss = (out ** 2).sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert torch.count_nonzero(x.grad) > 0


class TestKDenseAndKANLayer:
    """Test suite for KDense layer and multi-layer KAN model."""

    @pytest.mark.parametrize("basis_name", ALL_BASES)
    def test_kdense_forward_backward(self, basis_name):
        """Verify KDense forward pass and parameter gradient updates."""
        layer = KDense(
            in_features=2,
            out_features=3,
            grid_len=5,
            basis_func=basis_name,
            use_base_act=True,
        )
        x = torch.randn(8, 2, requires_grad=True)
        out = layer(x)
        assert out.shape == (8, 3)
        
        loss = out.sum()
        loss.backward()
        
        assert layer.C.grad is not None
        assert not torch.isnan(layer.C.grad).any()
        if layer.W is not None:
            assert layer.W.grad is not None
            assert not torch.isnan(layer.W.grad).any()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.parametrize("basis_name", ALL_BASES)
    def test_kan_model_forward_backward(self, basis_name):
        """Verify multi-layer KAN ODE vector field forward/backward."""
        model = KAN(
            layers_hidden=[2, 10, 2],
            grid_len=5,
            basis_func=basis_name,
            normalizer="tanh",
            base_act="silu",
        )
        x = torch.randn(16, 2)
        # Vector field forward: f_theta(t, u) -> dudt
        dudt = model(t=0.0, x=x)
        assert dudt.shape == (16, 2)
        
        loss = dudt.sum()
        loss.backward()
        
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_kdense_get_activations(self):
        """Verify extraction of individual edge curves."""
        layer = KDense(in_features=2, out_features=3, grid_len=5, basis_func="rbf")
        x = torch.randn(4, 2)
        spline_acts, base_acts = layer.get_activations(x)
        assert spline_acts.shape == (4, 2, 3)
        assert base_acts.shape == (4, 2, 3)
