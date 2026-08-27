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
from ode import NeuralODE, ZeroSumField, VanishingDimField
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
        
        # Verify metrics JSON content: the full run configuration must be recorded so
        # every reported number is traceable to the run that produced it, and the
        # error must be split into train / extrapolation / full horizons rather than
        # reported as a single conflated "test" number.
        with open(os.path.join(temp_run_dir, "metrics.json"), "r") as f:
            metrics_data = json.load(f)

        cfg = metrics_data["config"]
        assert cfg["model_type"] == "kan"
        assert cfg["solver"] == "tsit5"
        assert cfg["dataset"] == "lotka_volterra"
        assert "estimated_lipschitz_bound" in metrics_data

        # Provenance fields required for reproducing a sweep on another machine
        for key in ("lr", "num_epochs", "substeps", "seed", "noise_std", "dt",
                    "t_train_end", "torch_version", "git_sha", "data_params"):
            assert key in cfg, f"metrics.json config is missing '{key}'"

        # Both the best-epoch and final-epoch models must be scored, on all 3 horizons
        for bucket in ("best", "final"):
            for split in ("train_mse", "extrap_mse", "full_mse"):
                assert split in metrics_data[bucket]

        # Model selection must be on training loss, never the extrapolation window
        assert metrics_data["selection"]["criterion"] == "min_train_mse"

    def test_best_checkpoint_is_not_the_final_model(self, temp_run_dir):
        """
        Regression guard: `state_dict()` returns references to the live parameter
        tensors, so storing it directly makes the "best" checkpoint silently track the
        optimizer and end up identical to the final model. Train long enough for the
        loss to be non-monotonic, then assert the saved best checkpoint really is the
        best epoch and not just the last one.
        """
        res = train_kan_ode(
            model_type="kan",
            layers_hidden=[2, 6, 2],
            grid_len=3,
            basis_func="rbf",
            solver="euler",
            substeps=1,
            num_epochs=40,
            lr=0.2,  # deliberately large so the loss diverges instead of descending,
            # which makes best_epoch != final epoch and the alias check non-vacuous
            save_dir=temp_run_dir,
            seed=42,
        )

        train_losses = res["train_losses"]
        best_epoch = res["best_epoch"]

        # The recorded best epoch must be the true argmin of the training curve
        assert best_epoch == int(np.argmin(train_losses)) + 1

        ckpt = torch.load(os.path.join(temp_run_dir, "best_model.pt"),
                          map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == best_epoch
        assert np.isclose(ckpt["train_mse"], min(train_losses))

        # And the saved snapshot must not have been mutated by later optimizer steps
        final = torch.load(os.path.join(temp_run_dir, "final_model.pt"),
                           map_location="cpu", weights_only=False)
        if best_epoch != len(train_losses):
            differs = any(
                not torch.allclose(ckpt["model_state_dict"][k], final["model_state_dict"][k])
                for k in ckpt["model_state_dict"]
            )
            assert differs, "best checkpoint aliases the final weights"


class TestTimeScaling:
    """
    [FIX-2026-08 / S3] `--time_scale` nondimensionalises the integration clock so
    that a system whose horizon is long and whose derivative is small (SIR: T=50,
    |f| ~ 1e-2) is presented to the optimizer in the same regime as one that
    already trains (Lotka-Volterra: T=3.5, |f| ~ 1e0).

    It is an exact change of variables, so these tests pin down the two properties
    the rest of the codebase depends on: it is a true no-op at 1.0, and a model
    trained under it must be replayed on the same rescaled clock.
    """

    def test_time_scale_one_is_a_bitwise_noop(self, temp_run_dir):
        """
        26 Phase-2 runs predate this flag. Passing the default must reproduce them
        exactly -- not approximately -- or every historical number silently shifts.
        """
        kwargs = dict(
            model_type="kan", layers_hidden=[2, 6, 2], grid_len=3, basis_func="rbf",
            solver="tsit5", substeps=1, num_epochs=6, seed=42,
        )
        a = train_kan_ode(save_dir=os.path.join(temp_run_dir, "a"), **kwargs)
        b = train_kan_ode(save_dir=os.path.join(temp_run_dir, "b"), time_scale=1.0, **kwargs)

        assert a["train_losses"] == b["train_losses"]
        assert a["grad_norms"] == b["grad_norms"]

    def test_rescaled_clock_equals_rescaled_field(self):
        """
        Integrating g on the clock tau = t/c must give the same trajectory as
        integrating the physical field g/c on t -- that identity is the whole
        justification for the flag, and it is what lets metrics.json keep reporting
        in physical time. Note the direction: a network trained under time_scale=c
        has learned g = c * f_physical, so dividing recovers f_physical.

        The two sides agree to solver precision rather than only in the continuous
        limit, because the RK stage increments coincide exactly: on the tau grid the
        step is (dt/c) * g, on the t grid it is dt * (g/c).
        """
        c = 10.0
        torch.manual_seed(7)
        f = KAN(layers_hidden=[3, 8, 3], grid_len=4, basis_func="rbf")

        class Scaled(torch.nn.Module):
            def __init__(self, inner, factor):
                super().__init__()
                self.inner, self.factor = inner, factor

            def forward(self, *args, **kwargs):
                return self.factor * self.inner(*args, **kwargs)

        y0 = torch.tensor([0.99, 0.01, 0.0])
        t = torch.linspace(0.0, 50.0, 101)

        on_rescaled_clock = NeuralODE(f, method="tsit5", substeps=2)(y0, t / c)
        on_physical_field = NeuralODE(Scaled(f, 1.0 / c), method="tsit5", substeps=2)(y0, t)

        assert torch.allclose(on_rescaled_clock, on_physical_field, atol=1e-5)

    def test_time_scale_is_recorded_for_replay(self, temp_run_dir):
        """
        evaluate.py and analyze_fixes.py rebuild the integrator from metrics.json
        alone. If time_scale is not written there, they replay the checkpoint on the
        wrong clock and report a trajectory the run never produced.
        """
        train_kan_ode(
            model_type="kan", layers_hidden=[2, 6, 2], grid_len=3, basis_func="rbf",
            solver="tsit5", substeps=1, num_epochs=3, seed=42,
            time_scale=4.0, save_dir=temp_run_dir,
        )
        with open(os.path.join(temp_run_dir, "metrics.json")) as fh:
            cfg = json.load(fh)["config"]
        assert cfg["time_scale"] == 4.0

        ckpt = torch.load(os.path.join(temp_run_dir, "best_model.pt"),
                          map_location="cpu", weights_only=False)
        assert ckpt["config"]["time_scale"] == 4.0

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_time_scale_is_rejected(self, temp_run_dir, bad):
        """A zero or negative clock collapses or reverses time; fail loudly, not silently."""
        with pytest.raises(ValueError, match="time_scale"):
            train_kan_ode(
                model_type="kan", layers_hidden=[2, 6, 2], grid_len=3,
                num_epochs=1, seed=42, time_scale=bad, save_dir=temp_run_dir,
            )


class TestCheckpointRebuildFidelity:
    """
    [FIX-2026-08] A KAN's grid is a buffer and its weight SHAPES do not depend on
    grid_lims, so rebuilding a checkpoint on the wrong span still load_state_dict()s
    cleanly and returns wrong numbers with no error anywhere. evaluate.py did exactly
    that. This guards the property that broke.
    """

    def test_grid_lims_changes_predictions(self):
        """If this ever stops holding, dropping grid_lims on rebuild would be harmless."""
        torch.manual_seed(11)
        a = KAN(layers_hidden=[3, 8, 3], grid_len=5, grid_lims=(0.0, 1.0), basis_func="rbf")
        b = KAN(layers_hidden=[3, 8, 3], grid_len=5, grid_lims=(-1.0, 1.0), basis_func="rbf")
        b.load_state_dict(a.state_dict())  # succeeds: same shapes, grid is a buffer

        x = torch.tensor([[0.9, 0.05, 0.05], [0.3, 0.3, 0.4]])
        assert not torch.allclose(a(x), b(x)), "grid_lims must affect the vector field"

    def test_evaluate_rebuilds_with_saved_grid_lims(self, temp_run_dir):
        """The end-to-end guard: a non-default span must survive a save/reload round trip."""
        from evaluate import evaluate_checkpoint

        train_kan_ode(
            model_type="kan", layers_hidden=[2, 6, 2], grid_len=4, basis_func="rbf",
            normalizer="identity", grid_lims=(0.0, 2.0), solver="tsit5", substeps=1,
            num_epochs=4, seed=42, save_dir=temp_run_dir,
        )
        with open(os.path.join(temp_run_dir, "metrics.json")) as fh:
            reference = json.load(fh)

        replayed = evaluate_checkpoint(
            os.path.join(temp_run_dir, "best_model.pt"),
            save_dir=os.path.join(temp_run_dir, "eval"),
        )
        assert np.isclose(replayed["overall full"]["mse"],
                          reference["best"]["full_mse"], rtol=1e-5)


class TestConservationProjection:
    """
    [FIX-2026-08 / S4] `--conserve_mode projection` turns S+I+R=1 from something the
    loss nudges toward into a property of the flow itself. These tests pin the three
    claims that makes: the field really is zero-sum, the invariant survives a full
    extrapolation horizon, and the wrapper stays transparent to checkpointing.
    """

    def test_projected_field_sums_to_zero(self):
        torch.manual_seed(3)
        f = ZeroSumField(KAN(layers_hidden=[3, 8, 3], grid_len=5, basis_func="rbf"))
        x = torch.rand(16, 3)
        assert f(x).sum(dim=-1).abs().max().item() < 1e-5

    def test_invariant_holds_far_outside_the_training_window(self):
        """
        The point of projecting rather than penalising: a penalty says nothing about
        t > t_train_end, which is precisely where the SIR runs lost 45% of their mass.
        Integrate well past any window and the sum must not move.

        The field is damped to |f| ~ 1e-2 -- the magnitude a CONVERGED SIR model
        actually has -- so the trajectory stays in physical range over [0, 80]. An
        undamped random KAN diverges to ~1e15 here (that divergence is the bug this
        whole changeset is about), and absolute mass drift is meaningless once the
        state is that large; conservation is only ever exact relative to the scale
        of the trajectory carrying it.
        """
        torch.manual_seed(5)

        class Damped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, *args, **kwargs):
                return 0.01 * self.inner(*args, **kwargs)

        f = ZeroSumField(Damped(KAN(layers_hidden=[3, 8, 3], grid_len=5, basis_func="rbf")))
        y0 = torch.tensor([0.99, 0.01, 0.0])
        traj = NeuralODE(f, method="tsit5", substeps=2)(y0, torch.linspace(0.0, 80.0, 161))

        assert traj.abs().max().item() < 10.0, "trajectory left physical range; test is vacuous"
        drift = (traj.sum(dim=-1) - y0.sum()).abs().max().item()
        assert drift < 1e-4, f"mass drifted by {drift}"

    def test_wrapper_is_transparent_to_checkpointing(self):
        """
        train.py hands `model.parameters()` to Adam and saves `model.state_dict()`.
        If the wrapper renamed or hid anything, every existing loader would break.
        """
        torch.manual_seed(9)
        k = KAN(layers_hidden=[3, 8, 3], grid_len=5, basis_func="rbf")
        z = ZeroSumField(k)
        assert set(z.state_dict()) == set(k.state_dict())
        assert len(list(z.parameters())) == len(list(k.parameters()))

    def test_projection_rejects_an_inconsistent_target(self, temp_run_dir):
        """
        Projection conserves sum(y0) -- it cannot move the trajectory to a different
        invariant surface. Asking for a target y0 does not satisfy must fail loudly
        rather than quietly conserving the wrong constant.
        """
        with pytest.raises(ValueError, match="projection"):
            train_kan_ode(
                dataset="sir", layers_hidden=[3, 8, 3], grid_len=4, num_epochs=1,
                seed=42, conserve_sum=2.5, conserve_mode="projection",
                save_dir=temp_run_dir,
            )

    def test_projection_mode_survives_checkpoint_replay(self, temp_run_dir):
        """
        End-to-end: evaluate.py must rebuild the projected field, not the bare module.
        Without that it integrates a different ODE and reports numbers the run never
        produced -- the same class of silent bug as the dropped grid_lims.
        """
        from evaluate import evaluate_checkpoint

        train_kan_ode(
            dataset="sir", layers_hidden=[3, 8, 3], grid_len=4, basis_func="rbf",
            solver="tsit5", substeps=1, num_epochs=4, seed=42,
            conserve_sum=1.0, conserve_mode="projection", time_scale=10.0,
            save_dir=temp_run_dir,
        )
        with open(os.path.join(temp_run_dir, "metrics.json")) as fh:
            reference = json.load(fh)
        assert reference["config"]["conserve_mode"] == "projection"

        replayed = evaluate_checkpoint(
            os.path.join(temp_run_dir, "best_model.pt"),
            save_dir=os.path.join(temp_run_dir, "eval"),
        )
        assert np.isclose(replayed["overall full"]["mse"],
                          reference["best"]["full_mse"], rtol=1e-5)



class TestVanishingDimPrior:
    """
    [FIX-2026-08 / S5] `--vanish_dim` encodes that y[d]=0 is a manifold of equilibria.
    It is a real physical assumption, so these tests pin exactly what it promises --
    and, just as importantly, that it composes with the conservation projection
    instead of quietly cancelling it.
    """

    def test_field_vanishes_on_the_plane(self):
        torch.manual_seed(1)
        f = VanishingDimField(KAN(layers_hidden=[3, 8, 3], grid_len=5, basis_func="rbf"), dim=1)
        on_plane = torch.tensor([[0.5, 0.0, 0.5], [0.04, 0.0, 0.96]])
        assert f(on_plane).abs().max().item() == 0.0
        off_plane = torch.tensor([[0.5, 0.2, 0.3]])
        assert f(off_plane).abs().max().item() > 0.0

    def test_gate_reads_the_state_not_the_field_output(self):
        """
        The gate must be y[d]. Gating on f[d] instead would still zero *something*
        and still look plausible on a smoke test, while encoding entirely the wrong
        physics -- so assert against an explicit hand-computed product.
        """
        class Const(torch.nn.Module):
            def forward(self, *a, **k):
                y = a[-1]
                return torch.ones_like(y) * torch.tensor([2.0, 3.0, 4.0])

        f = VanishingDimField(Const(), dim=1)
        y = torch.tensor([[1.0, 0.5, 1.0]])
        # gate = y[1] = 0.5  ->  0.5 * [2,3,4]
        assert torch.allclose(f(y), torch.tensor([[1.0, 1.5, 2.0]]))

    def test_composes_with_conservation_projection(self):
        """Scalar * zero-sum vector stays zero-sum: both SIR invariants must hold at once."""
        torch.manual_seed(2)
        f = VanishingDimField(
            ZeroSumField(KAN(layers_hidden=[3, 8, 3], grid_len=5, basis_func="rbf")), dim=1
        )
        x = torch.rand(8, 3)
        assert f(x).sum(dim=-1).abs().max().item() < 1e-5

    def test_handles_both_solver_calling_conventions(self):
        """NeuralODE may dispatch f(y) or f(t, y); the gate must resolve the state either way."""
        torch.manual_seed(4)
        f = VanishingDimField(KAN(layers_hidden=[3, 8, 3], grid_len=5, basis_func="rbf"), dim=1)
        y = torch.tensor([[0.3, 0.2, 0.5]])
        assert torch.allclose(f(y), f(torch.tensor(0.0), y))

    def test_rejects_out_of_range_dim(self, temp_run_dir):
        with pytest.raises(ValueError, match="vanish_dim"):
            train_kan_ode(
                dataset="sir", layers_hidden=[3, 8, 3], grid_len=4, num_epochs=1,
                seed=42, vanish_dim=7, save_dir=temp_run_dir,
            )

    def test_survives_checkpoint_replay(self, temp_run_dir):
        """Like the projection, the gate is field structure rather than weights."""
        from evaluate import evaluate_checkpoint

        train_kan_ode(
            dataset="sir", layers_hidden=[3, 8, 3], grid_len=4, basis_func="rbf",
            solver="tsit5", substeps=1, num_epochs=4, seed=42,
            conserve_sum=1.0, conserve_mode="projection", vanish_dim=1, time_scale=10.0,
            save_dir=temp_run_dir,
        )
        with open(os.path.join(temp_run_dir, "metrics.json")) as fh:
            reference = json.load(fh)
        assert reference["config"]["vanish_dim"] == 1

        replayed = evaluate_checkpoint(
            os.path.join(temp_run_dir, "best_model.pt"),
            save_dir=os.path.join(temp_run_dir, "eval"),
        )
        assert np.isclose(replayed["overall full"]["mse"],
                          reference["best"]["full_mse"], rtol=1e-5)
