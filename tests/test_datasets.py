"""
Unit and Physical Invariant Tests for Dynamical System Datasets:
1. Lotka-Volterra Predator-Prey System
2. Non-linear Damped Pendulum & Hamiltonian Energy Dissipation
3. 3D Chaotic Lorenz Attractor
4. SIR Epidemiological Dynamics & Population Conservation
5. Empirical Outbreak Time-Series
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "implementation")))

import pytest
import torch
import numpy as np

from data import (
    generate_lotka_volterra_data,
    LotkaVolterraData,
    generate_damped_pendulum_data,
    DampedPendulumData,
    compute_pendulum_energy,
    generate_lorenz_data,
    LorenzData,
    generate_sir_data,
    load_empirical_epidemic_data,
    EpidemicData,
)
from kan import KAN, MLP_ODE
from ode import NeuralODE


class TestLotkaVolterraDataset:
    """Test suite for Lotka-Volterra dataset generator."""

    def test_shapes_and_types(self):
        """Verify tensor shapes, types, and split lengths."""
        data = generate_lotka_volterra_data(t_start=0.0, t_end=14.0, dt=0.1, t_train_end=3.5)
        assert isinstance(data, LotkaVolterraData)
        assert data.t_train.shape == (36,)
        assert data.t_full.shape == (141,)
        assert data.y_train.shape == (36, 2)
        assert data.y_full.shape == (141, 2)
        assert data.y0.shape == (2,)
        assert not torch.isnan(data.y_full).any()

    def test_reproducibility(self):
        """Verify deterministic reproducibility with random seeds."""
        d1 = generate_lotka_volterra_data(noise_std=0.05, seed=123)
        d2 = generate_lotka_volterra_data(noise_std=0.05, seed=123)
        d3 = generate_lotka_volterra_data(noise_std=0.05, seed=999)
        
        assert torch.allclose(d1.y_train, d2.y_train)
        assert not torch.allclose(d1.y_train, d3.y_train)


class TestDampedPendulumDataset:
    """Test suite for Non-linear Damped Pendulum and Energy Conservation."""

    def test_shapes_and_types(self):
        """Verify tensor shapes and container structure."""
        data = generate_damped_pendulum_data(mu=0.5, t_start=0.0, t_end=10.0, dt=0.05, t_train_end=3.0)
        assert isinstance(data, DampedPendulumData)
        assert data.t_train.shape == (61,)
        assert data.t_full.shape == (201,)
        assert data.y_train.shape == (61, 2)
        assert data.y_full.shape == (201, 2)
        assert data.energy_full.shape == (201,)
        assert not torch.isnan(data.y_full).any()

    def test_undamped_energy_conservation(self):
        """When mu=0 (conservative oscillator), total mechanical energy is conserved: dE/dt = 0."""
        data = generate_damped_pendulum_data(mu=0.0, t_start=0.0, t_end=10.0, dt=0.02, theta0=1.5, omega0=0.0)
        energy = data.energy_full
        
        # Energy should remain constant throughout simulation
        max_energy_deviation = (energy.max() - energy.min()).item()
        assert max_energy_deviation < 1e-4, f"Energy drift {max_energy_deviation} too large for mu=0"

    def test_damped_energy_dissipation(self):
        """When mu > 0 (dissipative system), total mechanical energy strictly decreases over time."""
        data = generate_damped_pendulum_data(mu=0.8, t_start=0.0, t_end=10.0, dt=0.05, theta0=2.0, omega0=0.0)
        energy = data.energy_full.numpy()
        
        # Energy at the end must be significantly lower than initial energy
        assert energy[-1] < energy[0] * 0.1
        # Energy diff must be non-positive (monotonic decrease within small tolerance)
        energy_diffs = np.diff(energy)
        assert (energy_diffs <= 1e-6).all(), "Damped pendulum energy did not monotonically dissipate"


class TestLorenzDataset:
    """Test suite for 3D Chaotic Lorenz Attractor."""

    def test_shapes_and_types(self):
        """Verify 3D coordinates and full trajectory bounds."""
        data = generate_lorenz_data(t_start=0.0, t_end=10.0, dt=0.01, t_train_end=4.0)
        assert isinstance(data, LorenzData)
        assert data.t_train.shape == (401,)
        assert data.t_full.shape == (1001,)
        assert data.y_train.shape == (401, 3)
        assert data.y_full.shape == (1001, 3)
        assert data.y0.shape == (3,)
        assert not torch.isnan(data.y_full).any()

    def test_lorenz_attractor_bounding_box(self):
        """Verify that 3D trajectory stays inside known Lorenz attractor geometric bounds."""
        data = generate_lorenz_data(t_start=0.0, t_end=20.0, dt=0.01, x0=1.0, y0=1.0, z0=1.0)
        x = data.y_full[:, 0].numpy()
        y = data.y_full[:, 1].numpy()
        z = data.y_full[:, 2].numpy()
        
        assert np.abs(x).max() < 35.0
        assert np.abs(y).max() < 40.0
        assert z.min() >= 0.0
        assert z.max() < 60.0

    def test_3d_neural_ode_integration_with_lorenz(self):
        """Verify that 3D KAN-ODE model can integrate and backprop over 3D Lorenz trajectory."""
        kan_3d = KAN(layers_hidden=[3, 8, 3], grid_len=3, basis_func="rbf")
        node = NeuralODE(func=kan_3d, method="tsit5", substeps=1)
        
        y0 = torch.tensor([1.0, 1.0, 1.0])
        t = torch.linspace(0.0, 0.5, 6)
        pred = node(y0=y0, t=t)
        
        assert pred.shape == (6, 3)
        loss = pred.sum()
        loss.backward()
        
        for p in kan_3d.parameters():
            assert p.grad is not None


class TestEpidemicDatasets:
    """Test suite for SIR dynamics and empirical outbreak data."""

    def test_sir_population_conservation(self):
        """In classical SIR, total population S(t) + I(t) + R(t) = 1.0 is strictly conserved."""
        data = generate_sir_data(beta=0.35, gamma=0.10, S0=0.99, I0=0.01, R0_init=0.0, t_end=60.0, dt=0.5)
        assert isinstance(data, EpidemicData)
        assert data.y_full.shape[-1] == 3
        
        # Check conservation law
        pop_sum = data.y_full.sum(dim=-1)
        deviation = torch.abs(pop_sum - 1.0).max().item()
        assert deviation < 1e-5, f"Population conservation violated: max deviation = {deviation}"

    def test_empirical_outbreak_normalization(self):
        """Verify empirical outbreak loader scales outputs to [0, 1] range without NaNs."""
        data = load_empirical_epidemic_data(num_days=100, train_days=40, smoothing_window=7)
        assert isinstance(data, EpidemicData)
        assert data.y_full.shape == (101, 2)
        assert data.y_train.shape == (40, 2)
        
        assert data.y_full.min().item() >= 0.0
        assert data.y_full.max().item() <= 1.0 + 1e-5
        assert not torch.isnan(data.y_full).any()
