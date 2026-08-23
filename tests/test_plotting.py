"""
Unit and Integration Tests for Publication-Grade Plotting Utilities.
"""

import sys
import os
import shutil
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "implementation")))

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")

from utils.plotting import (
    plot_trajectory_comparison,
    plot_phase_space,
    plot_phase_portrait_with_streamlines,
    plot_3d_lorenz_trajectory,
    plot_pendulum_phase_and_energy,
    plot_loss_curves,
    plot_model_comparison_curves,
    plot_gradient_norm_dynamics,
    plot_benchmark_comparison,
)
from data import (
    generate_lotka_volterra_data,
    generate_damped_pendulum_data,
    generate_lorenz_data,
    compute_pendulum_energy,
)


@pytest.fixture
def temp_plot_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestPlottingFunctions:
    """Test suite ensuring all publication plotting renderers execute and produce valid image files."""

    def test_plot_trajectory_comparison(self, temp_plot_dir):
        """Verify 1D trajectory comparison output file."""
        t = np.linspace(0, 10, 50)
        y_true = np.stack([np.sin(t), np.cos(t)], axis=-1)
        y_pred = y_true + 0.05
        
        save_path = os.path.join(temp_plot_dir, "test_trajectory.png")
        plot_trajectory_comparison(t, y_true, y_pred, t_split=3.5, save_path=save_path)
        
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 1000

    def test_plot_phase_space(self, temp_plot_dir):
        """Verify 2D phase portrait output file."""
        t = np.linspace(0, 2 * np.pi, 50)
        y_true = np.stack([np.sin(t), np.cos(t)], axis=-1)
        y_pred = y_true * 1.02
        
        save_path = os.path.join(temp_plot_dir, "test_phase_space.png")
        plot_phase_space(y_true, y_pred, train_len=25, save_path=save_path)
        
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 1000

    def test_plot_phase_portrait_with_streamlines(self, temp_plot_dir):
        """Verify 2D streamline vector field overlay."""
        # Define mock linear vector field: dot(x) = y, dot(y) = -x
        vector_field_fn = lambda u: np.stack([u[:, 1], -u[:, 0]], axis=-1)
        
        t = np.linspace(0, 2 * np.pi, 50)
        y_true = np.stack([np.cos(t), -np.sin(t)], axis=-1)
        y_pred = y_true * 0.98
        
        save_path = os.path.join(temp_plot_dir, "test_streamlines.png")
        plot_phase_portrait_with_streamlines(
            vector_field_fn=vector_field_fn,
            y_true=y_true,
            y_pred=y_pred,
            train_len=25,
            x_range=(-2.0, 2.0),
            y_range=(-2.0, 2.0),
            grid_density=15,
            save_path=save_path,
        )
        
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 1000

    def test_plot_3d_lorenz_trajectory(self, temp_plot_dir):
        """Verify 3D chaotic attractor projection."""
        data = generate_lorenz_data(t_start=0.0, t_end=2.0, dt=0.02)
        y_true = data.y_full.numpy()
        y_pred = y_true + np.random.normal(0, 0.1, y_true.shape)
        
        save_path = os.path.join(temp_plot_dir, "test_lorenz_3d.png")
        plot_3d_lorenz_trajectory(y_true, y_pred, train_len=50, save_path=save_path)
        
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 1000

    def test_plot_pendulum_phase_and_energy(self, temp_plot_dir):
        """Verify 2-panel damped pendulum phase & energy dissipation figure."""
        data = generate_damped_pendulum_data(mu=0.5, t_end=5.0, dt=0.05)
        y_true = data.y_full.numpy()
        y_pred = y_true * 0.99
        e_true = data.energy_full.numpy()
        e_pred = compute_pendulum_energy(data.y_full * 0.99).numpy()
        
        save_path = os.path.join(temp_plot_dir, "test_pendulum.png")
        plot_pendulum_phase_and_energy(
            t_full=data.t_full.numpy(),
            y_true=y_true,
            y_pred=y_pred,
            energy_true=e_true,
            energy_pred=e_pred,
            train_len=40,
            save_path=save_path,
        )
        
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 1000

    def test_plot_model_comparison_curves(self, temp_plot_dir):
        """Verify multi-model comparison loss decay plot."""
        histories = {
            "KAN-ODE (RBF)": {"test_losses": [1.0, 0.1, 0.01, 1e-4, 1e-5]},
            "MLP-ODE (SiLU)": {"test_losses": [1.0, 0.5, 0.2, 0.05, 0.01]},
        }
        
        save_path = os.path.join(temp_plot_dir, "test_comparison.png")
        plot_model_comparison_curves(
            histories=histories,
            metric_key="test_losses",
            target_threshold=3e-5,
            save_path=save_path,
        )
        
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 1000

    def test_plot_gradient_norm_and_benchmark(self, temp_plot_dir):
        """Verify gradient norm dynamics and benchmark bar chart."""
        grad_norms = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
        p1 = os.path.join(temp_plot_dir, "test_grad_norm.png")
        plot_gradient_norm_dynamics(grad_norms, save_path=p1)
        assert os.path.exists(p1)
        
        benchmark_data = {
            "Euler": {"test_mse": 1e-2},
            "Heun": {"test_mse": 5e-4},
            "RK4": {"test_mse": 2e-5},
            "Tsit5": {"test_mse": 1e-5},
        }
        p2 = os.path.join(temp_plot_dir, "test_benchmarks.png")
        plot_benchmark_comparison(benchmark_data, metric="test_mse", save_path=p2)
        assert os.path.exists(p2)
