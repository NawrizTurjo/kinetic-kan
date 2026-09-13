"""
Unit Tests for Phase 3 / Track E: SINDy Comparison and Real Epidemic Fit.

Covers:
1. `prune_edges` semantics -- no mutation, exact no-op at 0%, group-wise
   zeroing of whole edges, and the base-branch flag.
2. Checkpoint rebuilding -- that Track E re-integrates the Phase-2 noise-sweep
   checkpoints to the numbers those runs actually published.
3. The empirical-epidemic facts the E2 argument is built on: that the zero-sum
   invariant does NOT hold, and that the init-time field/derivative mismatch is
   the one docs/10 describes.

Owner: Monjur Hossain Khan (Shovon), 2105043.
"""

import sys
import os
import copy
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "implementation")))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "implementation", "experiments", "sindy_epidemic")))

import pytest
import torch
import numpy as np

from kan import KAN
from data import load_empirical_epidemic_data, generate_lotka_volterra_data

from pruning import prune_edges, edge_strengths, surviving_edge_report
import common


IMPL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "implementation"))
NOISE_DIR = os.path.join(IMPL_ROOT, "results", "benchmarks", "noise")


def make_model(seed=0):
    torch.manual_seed(seed)
    return KAN(layers_hidden=[2, 10, 2], grid_len=5, basis_func="rbf")


# ==============================================================================
# 1. Pruning
# ==============================================================================

class TestEdgeStrengths:
    """Per-edge magnitude reduction over the spline coefficient group."""

    def test_shape_is_per_edge(self):
        model = make_model()
        for layer in model.layers:
            s = edge_strengths(layer)
            assert s.shape == (layer.out_features, layer.in_features)

    def test_is_mean_absolute_spline_weight(self):
        """Strength must reduce the G coefficients of an edge, not sample one."""
        layer = make_model().layers[0]
        C = layer.C.detach().view(layer.out_features, layer.in_features, layer.grid_len)
        expected = C.abs().mean(dim=-1)
        assert torch.allclose(edge_strengths(layer), expected)

    def test_include_base_adds_base_weight(self):
        layer = make_model().layers[0]
        without = edge_strengths(layer, include_base=False)
        with_base = edge_strengths(layer, include_base=True)
        assert torch.allclose(with_base, without + layer.W.detach().abs())


class TestPruneEdges:
    """`prune_edges` is a pure function returning a masked copy."""

    def test_does_not_mutate_input(self):
        model = make_model()
        before = copy.deepcopy(model.state_dict())
        prune_edges(model, threshold_percentile=90.0, prune_base=True)
        for k, v in model.state_dict().items():
            assert torch.equal(v, before[k]), f"prune_edges mutated {k}"

    def test_zero_percentile_is_exact_no_op(self):
        """Ties at the minimum must not be swept away by an off-by-one `<=`."""
        model = make_model()
        pruned, info = prune_edges(model, threshold_percentile=0.0)
        assert info["total_edges_pruned"] == 0
        assert info["sparsity"] == 0.0
        for k, v in model.state_dict().items():
            assert torch.equal(v, pruned.state_dict()[k])

    @pytest.mark.parametrize("pct", [25.0, 50.0, 75.0])
    def test_requested_sparsity_is_achieved(self, pct):
        model = make_model()
        _, info = prune_edges(model, threshold_percentile=pct)
        assert info["sparsity"] == pytest.approx(pct / 100.0, abs=0.05)

    def test_prunes_whole_edges_not_scattered_coefficients(self):
        """
        A pruned edge must lose ALL grid_len of its spline weights. Zeroing a
        subset would leave a lumpier curve rather than removing the edge.
        """
        model = make_model()
        pruned, _ = prune_edges(model, threshold_percentile=50.0)
        for layer in pruned.layers:
            C = layer.C.detach().view(layer.out_features, layer.in_features, layer.grid_len)
            zeros_per_edge = (C == 0).sum(dim=-1)
            assert set(zeros_per_edge.unique().tolist()) <= {0, layer.grid_len}

    def test_prune_base_flag_controls_residual_branch(self):
        model = make_model()
        spline_only, _ = prune_edges(model, threshold_percentile=50.0, prune_base=False)
        whole_edge, _ = prune_edges(model, threshold_percentile=50.0, prune_base=True)
        # The residual branch survives in one case and not the other.
        assert (spline_only.layers[0].W == 0).sum() == 0
        assert (whole_edge.layers[0].W == 0).sum() > 0

    def test_per_layer_keeps_every_layer_populated(self):
        model = make_model()
        pruned, info = prune_edges(model, threshold_percentile=75.0, per_layer=True)
        for layer_info in info["layers"]:
            assert layer_info["edges_pruned"] < layer_info["edges"]

    @pytest.mark.parametrize("pct", [-1.0, 101.0])
    def test_rejects_out_of_range_percentile(self, pct):
        with pytest.raises(ValueError):
            prune_edges(make_model(), threshold_percentile=pct)

    def test_pruning_changes_the_forward_pass(self):
        model = make_model()
        pruned, _ = prune_edges(model, threshold_percentile=50.0)
        x = torch.randn(16, 2)
        with torch.no_grad():
            assert not torch.allclose(model(x), pruned(x))

    def test_surviving_edge_report_counts_match(self):
        model = make_model()
        pruned, info = prune_edges(model, threshold_percentile=50.0)
        report = surviving_edge_report(pruned)
        alive = sum(r["alive_edges"] for r in report)
        assert alive == info["total_edges"] - info["total_edges_pruned"]


# ==============================================================================
# 2. Checkpoint rebuilding against the published Phase-2 numbers
# ==============================================================================

NOISE_CASES = [("sigma0", 0.0), ("sigma0.01", 0.01), ("sigma0.05", 0.05), ("sigma0.1", 0.1)]


class TestCheckpointReintegration:
    """
    Track E's whole E1 baseline rests on re-integrating checkpoints it did not
    train. If the rebuild silently differs -- the failure mode catalogued in
    docs/10 Part 5 -- every comparison against SINDy is measured against the
    wrong curve. So pin it to the numbers those runs published.
    """

    @pytest.mark.parametrize("slug,sigma", NOISE_CASES)
    def test_matches_published_metrics(self, slug, sigma):
        ckpt = os.path.join(NOISE_DIR, slug, "best_model.pt")
        metrics_path = os.path.join(NOISE_DIR, slug, "metrics.json")
        if not (os.path.exists(ckpt) and os.path.exists(metrics_path)):
            pytest.skip(f"Phase-2 noise sweep artifact missing for {slug}")

        with open(metrics_path) as f:
            published = json.load(f)["best"]

        model, config = common.load_kan_checkpoint(ckpt)
        p = config.get("data_params", {})
        data = generate_lotka_volterra_data(
            alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"], delta=p["delta"],
            t_start=config["t_start"], t_end=config["t_end"], dt=config["dt"],
            t_train_end=config["t_train_end"], noise_std=config["noise_std"],
            seed=config["seed"],
        )
        pred = common.integrate(model, config, data.y0, data.t_full)
        got = common.split_mse(data.y_full.numpy(), pred, len(data.t_train))

        # rel=1e-3, not tighter: these MSEs come out of ~1450 sequential float32
        # solver stages, and CI runs three OSes against different BLAS builds. A
        # genuine rebuild bug (wrong grid_lims, a missing field wrapper, the wrong
        # clock) moves these by orders of magnitude, not by parts in a thousand --
        # so the looser bound still catches everything this test exists to catch.
        assert got["full_mse"] == pytest.approx(published["full_mse"], rel=1e-3)
        assert got["train_mse"] == pytest.approx(published["train_mse"], rel=1e-3)
        assert got["extrap_mse"] == pytest.approx(published["extrap_mse"], rel=1e-3)

    def test_config_noise_std_matches_directory(self):
        for slug, sigma in NOISE_CASES:
            ckpt = os.path.join(NOISE_DIR, slug, "best_model.pt")
            if not os.path.exists(ckpt):
                pytest.skip(f"Phase-2 noise sweep artifact missing for {slug}")
            _, config = common.load_kan_checkpoint(ckpt)
            assert config["noise_std"] == pytest.approx(sigma)


class TestSplitMse:
    def test_splits_partition_the_horizon(self):
        y = np.random.RandomState(0).randn(20, 2)
        p = y + 0.1
        m = common.split_mse(y, p, n_train=8)
        assert m["train_mse"] == pytest.approx(0.01, rel=1e-6)
        assert m["full_mse"] == pytest.approx(0.01, rel=1e-6)
        assert m["finite"]

    def test_non_finite_prediction_is_reported_not_raised(self):
        y = np.zeros((10, 2))
        p = np.full((10, 2), np.inf)
        m = common.split_mse(y, p, n_train=4)
        assert m["finite"] is False
        assert np.isnan(m["full_mse"])


# ==============================================================================
# 3. The empirical-epidemic facts E2's argument depends on
# ==============================================================================

E2_RUNS = os.path.join(IMPL_ROOT, "results", "phase3", "sindy_epidemic", "epidemic_runs")


class TestEpidemicArmReplay:
    """
    `best_prediction.npy` is gitignored, so a fresh clone can only redraw E2's
    figures by re-integrating each arm from its committed `best_model.pt`. That
    replay path has to honour time_scale, train_days and the structural wrappers
    -- if it does not, `--collect` draws a trajectory the run never produced.
    """

    @pytest.mark.parametrize("tag", ["ts24", "proj", "vanish", "split60", "full_vanish"])
    def test_replay_reproduces_published_metrics(self, tag):
        import run_epidemic_fit as ref

        metrics_path = os.path.join(E2_RUNS, tag, "metrics.json")
        ckpt = os.path.join(E2_RUNS, tag, "best_model.pt")
        if not (os.path.exists(metrics_path) and os.path.exists(ckpt)):
            pytest.skip(f"E2 arm '{tag}' has not been run in this working tree")

        with open(metrics_path) as f:
            m = json.load(f)
        config = m["config"]

        pred = ref.replay_checkpoint(ckpt, config)
        data = load_empirical_epidemic_data(train_days=config.get("train_days", 45))
        got = common.split_mse(data.y_full.numpy(), pred, len(data.t_train))

        assert got["train_mse"] == pytest.approx(m["best"]["train_mse"], rel=1e-3)
        assert got["extrap_mse"] == pytest.approx(m["best"]["extrap_mse"], rel=1e-3)

    def test_replay_honours_the_structural_wrapper(self):
        """
        The projection arm conserves sum(y) exactly. Replaying it without
        re-applying `ZeroSumField` would integrate a different ODE, and this is
        the cheapest observable that would catch it.
        """
        import run_epidemic_fit as ref

        metrics_path = os.path.join(E2_RUNS, "proj", "metrics.json")
        ckpt = os.path.join(E2_RUNS, "proj", "best_model.pt")
        if not (os.path.exists(metrics_path) and os.path.exists(ckpt)):
            pytest.skip("E2 'proj' arm has not been run in this working tree")

        with open(metrics_path) as f:
            config = json.load(f)["config"]
        sums = ref.replay_checkpoint(ckpt, config).sum(-1)
        assert np.ptp(sums) < 1e-5, "ZeroSumField was not re-applied on replay"


class TestEmpiricalEpidemicPreconditions:
    """
    E2 claims two of SIR's three fixes do not transfer. Both claims are claims
    about this dataset, so they are pinned here -- if the generator ever changes,
    the write-up's reasoning should fail loudly rather than quietly go stale.
    """

    def test_state_is_two_dimensional(self):
        data = load_empirical_epidemic_data()
        assert data.y_full.shape[-1] == 2, "E2 reasons about a 2D infected/recovered pair"

    def test_zero_sum_invariant_does_not_hold(self):
        """
        `conserve_mode projection` pins sum(y) to sum(y0). SIR satisfies that
        exactly; this dataset does not, because cumulative recovered only grows.
        """
        data = load_empirical_epidemic_data()
        sums = data.y_full.sum(-1)
        assert (sums.max() - sums.min()).item() > 1.0
        assert sums[-1].item() > 10 * sums[0].item()

    def test_vanish_dim_gate_would_be_near_zero_at_t0(self):
        """
        `--vanish_dim 0` multiplies the field by y[0]. Normalisation puts the
        infected fraction at ~0.003 on day 0, so the gate would all but freeze
        the trajectory at the initial condition.
        """
        data = load_empirical_epidemic_data()
        assert data.y0[0].item() < 0.01

    def test_init_field_overshoots_the_data_derivative(self):
        """
        The four-number diagnostic from docs/10, on this dataset: a Glorot KAN
        starts an order of magnitude too fast, which is what motivates
        `--time_scale`.
        """
        data = load_empirical_epidemic_data()
        torch.manual_seed(42)
        model = KAN(layers_hidden=[2, 10, 2], grid_len=5, basis_func="rbf")
        with torch.no_grad():
            init_field = model(data.y_train).abs().mean().item()
        dt = (data.t_train[1:] - data.t_train[:-1]).unsqueeze(-1)
        true_deriv = ((data.y_train[1:] - data.y_train[:-1]) / dt).abs().mean().item()
        assert init_field / true_deriv > 5.0

    def test_selected_time_scale_lands_in_the_trainable_regime(self):
        """
        docs/10 characterises the regime that trains as horizon ~5 with a
        derivative ~1e-1. s = 24 is chosen because it puts this dataset there.
        """
        data = load_empirical_epidemic_data()
        horizon = float(data.t_full[-1] - data.t_full[0])
        dt = (data.t_train[1:] - data.t_train[:-1]).unsqueeze(-1)
        true_deriv = ((data.y_train[1:] - data.y_train[:-1]) / dt).abs().mean().item()
        s = 24.0
        assert horizon / s == pytest.approx(5.0, abs=0.1)
        assert 0.1 <= true_deriv * s <= 1.0
