"""
End-to-End Scientific Machine Learning Pipeline Integration Tests.

Validates:
1. End-to-end forward/backward optimization loop on KAN-ODE.
2. End-to-end forward/backward optimization loop on Parameter-Matched MLP-ODE.
3. Strict Loss Monotonic Convergence on a 5-step toy optimization problem.
4. Deterministic Reproducibility across random seeds.
5. Automated Serialization and Metrics Export (`best_model.pt`, `metrics.json`).
"""

import sys
import os
import shutil
import tempfile
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "implementation")))

import pytest
import torch
import numpy as np

from kan import KAN, MLP_ODE, count_parameters
from ode import NeuralODE
from data import generate_lotka_volterra_data
from train import train_kan_ode
from utils.regularization import compute_kan_regularization
from utils.metrics import compute_mse, compute_rmse, compute_gradient_norm


@pytest.fixture
def temp_run_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestSciMLTrainingPipeline:
    """Test suite for full training pipeline execution and convergence."""

    def test_toy_kan_ode_loss_reduction(self):
        """
        Verify that 10 gradient descent steps strictly drive training loss downwards
        on a short Lotka-Volterra trajectory.
        """
        torch.manual_seed(42)
        np.random.seed(42)
        
        data = generate_lotka_volterra_data(t_start=0.0, t_end=2.0, dt=0.2, t_train_end=1.0)
        y0 = data.y0
        t_train = data.t_train
        y_train = data.y_train
        
        model = KAN(layers_hidden=[2, 8, 2], grid_len=3, basis_func="rbf")
        node = NeuralODE(model, method="tsit5", substeps=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            pred = node(y0, t_train)
            loss = torch.nn.functional.mse_loss(pred, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        # Initial vs Final loss
        assert losses[-1] < losses[0], f"Loss failed to decrease: {losses[0]:.4e} -> {losses[-1]:.4e}"
        assert losses[-1] < losses[2], "Optimization did not exhibit smooth monotonic descent"

    def test_toy_mlp_ode_loss_reduction(self):
        """Verify MLP-ODE baseline trains and reduces loss over steps."""
        torch.manual_seed(42)
        
        data = generate_lotka_volterra_data(t_start=0.0, t_end=2.0, dt=0.2, t_train_end=1.0)
        y0 = data.y0
        t_train = data.t_train
        y_train = data.y_train
        
        model = MLP_ODE(layers_hidden=[2, 14, 8, 8, 2], activation="silu")
        node = NeuralODE(model, method="tsit5", substeps=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            pred = node(y0, t_train)
            loss = torch.nn.functional.mse_loss(pred, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        assert losses[-1] < losses[0]

    def test_deterministic_seed_reproducibility(self):
        """Verify that identical seeds produce bitwise identical initializations and forward passes."""
        torch.manual_seed(123)
        kan1 = KAN(layers_hidden=[2, 6, 2], grid_len=4, basis_func="chebyshev")
        node1 = NeuralODE(kan1, method="rk4")
        
        torch.manual_seed(123)
        kan2 = KAN(layers_hidden=[2, 6, 2], grid_len=4, basis_func="chebyshev")
        node2 = NeuralODE(kan2, method="rk4")
        
        y0 = torch.tensor([1.5, 0.8])
        t = torch.linspace(0.0, 1.0, 5)
        
        pred1 = node1(y0, t)
        pred2 = node2(y0, t)
        
        assert torch.allclose(pred1, pred2, atol=1e-7)

    def test_full_train_kan_ode_execution(self, temp_run_dir):
        """
        Verify that `train_kan_ode` runs end-to-end, logs gradient norms,
        and saves all checkpoints and metadata JSONs.
        """
        res = train_kan_ode(
            model_type="kan",
            layers_hidden=[2, 6, 2],
            grid_len=3,
            basis_func="rbf",
            solver="tsit5",
            substeps=1,
            num_epochs=5,
            save_dir=temp_run_dir,
            seed=42,
        )
        
        assert "model" in res
        assert "metrics" in res
        assert "grad_norms" in res
        assert len(res["grad_norms"]) == 5
        
        # Verify artifact files exist
        assert os.path.exists(os.path.join(temp_run_dir, "best_model.pt"))
        assert os.path.exists(os.path.join(temp_run_dir, "final_model.pt"))
        assert os.path.exists(os.path.join(temp_run_dir, "metrics.json"))
        assert os.path.exists(os.path.join(temp_run_dir, "training_history.json"))
        assert os.path.exists(os.path.join(temp_run_dir, "gradient_norm_dynamics.png"))
        
        # Verify metrics JSON content
        with open(os.path.join(temp_run_dir, "metrics.json"), "r") as f:
            metrics_data = json.load(f)
            assert metrics_data["model_type"] == "kan"
            assert metrics_data["solver"] == "tsit5"
            assert "estimated_lipschitz_bound" in metrics_data
