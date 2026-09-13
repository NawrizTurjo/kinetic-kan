"""
Track B (Phase 3) -- tests for the gradient norm dynamics analysis.

There is no new model or training code in this track, so these target the analysis
functions themselves: the noise measure, the loader, and the correlation. They run on
the already-saved JSON files, so the whole file finishes in well under a second.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "implementation")))

import json
import numpy as np
import pytest

from experiments.gradient_dynamics.analyze_gradient_dynamics import (
    SOLVER_ORDER,
    load_grad_norms,
    log_step_noise,
    summarize,
    correlate_with_order,
    warmup_sensitivity,
    spike_diagnostics,
    plot_gradient_dynamics,
)


class TestNoiseMeasure:
    """The noise measure is where a silent off-by-one or a missing log would go unnoticed."""

    def test_smooth_series_scores_near_zero(self):
        # Pure geometric decay: constant relative step, so zero roughness by construction.
        g = 10.0 * (0.999 ** np.arange(2000))
        assert log_step_noise(g) < 1e-9

    def test_ranks_noisier_series_higher(self):
        base = 10.0 * (0.999 ** np.arange(2000))
        rng = np.random.default_rng(0)
        jittery = base * rng.lognormal(mean=0.0, sigma=0.3, size=base.size)
        assert log_step_noise(jittery) > log_step_noise(base)

    def test_scale_invariance(self):
        """A solver with 1000x larger gradients must not score as 1000x noisier."""
        rng = np.random.default_rng(1)
        g = 10.0 * (0.999 ** np.arange(2000)) * rng.lognormal(0.0, 0.2, 2000)
        assert log_step_noise(g) == pytest.approx(log_step_noise(g * 1000.0), rel=1e-12)

    def test_warmup_drops_leading_epochs(self):
        # A violent transient confined to the first 100 epochs, calm thereafter.
        calm = 1.0 * (0.999 ** np.arange(1000))
        spiky = np.concatenate([np.array([1e3, 1e-3] * 50), calm])
        assert log_step_noise(spiky, warmup=100) < log_step_noise(spiky, warmup=0)

    def test_ignores_nonfinite_and_nonpositive(self):
        """NaN/zero norms are failures, not small fluctuations -- they must not reach the log."""
        g = 10.0 * (0.999 ** np.arange(500))
        polluted = g.copy()
        polluted[10] = np.nan
        polluted[20] = 0.0
        assert np.isfinite(log_step_noise(polluted))

    def test_too_short_series_returns_nan(self):
        assert np.isnan(log_step_noise([1.0, 2.0]))


class TestLoader:
    """Guards the read-only contract against results/benchmarks/ablation_solvers/."""

    def test_loads_all_six_solvers(self):
        runs = load_grad_norms()
        assert set(runs) == set(SOLVER_ORDER)
        for name, run in runs.items():
            assert run["grad_norms"].size == 10000, f"{name} has {run['grad_norms'].size} epochs"
            assert np.all(np.isfinite(run["grad_norms"])), f"{name} has non-finite grad norms"

    def test_missing_run_raises(self):
        with pytest.raises(FileNotFoundError):
            load_grad_norms(source_dir=os.path.join(os.path.dirname(__file__), "_nope"))


class TestSummaryAndCorrelation:

    def test_rows_ranked_noisiest_first(self):
        rows = summarize(load_grad_norms())
        noise = [r["log_step_noise"] for r in rows]
        assert noise == sorted(noise, reverse=True)
        assert [r["noise_rank"] for r in rows] == list(range(1, len(rows) + 1))

    def test_correlation_bounded_and_deterministic(self):
        rows = summarize(load_grad_norms())
        first = correlate_with_order(rows)
        second = correlate_with_order(summarize(load_grad_norms()))
        assert -1.0 <= first["rho"] <= 1.0
        assert first["rho"] == pytest.approx(second["rho"], rel=1e-12)
        assert first["n"] == 6

    def test_correlation_recovers_a_planted_monotone_trend(self):
        """Sanity-check the statistic and its sign convention on known inputs.

        Noise falling as order rises -- the Track B hypothesis -- must read as rho = -1.
        """
        falling = [{"order": p, "log_step_noise": -float(p)} for p in (1, 2, 3, 4, 5, 6)]
        assert correlate_with_order(falling)["rho"] == pytest.approx(-1.0)

        rising = [{"order": p, "log_step_noise": float(p)} for p in (1, 2, 3, 4, 5, 6)]
        assert correlate_with_order(rising)["rho"] == pytest.approx(1.0)

    def test_warmup_sensitivity_reports_instability(self):
        """The headline Track B result: the trend is not stable across warm-up cuts."""
        sens = warmup_sensitivity(load_grad_norms())
        assert len(sens["cuts"]) == 4
        assert sens["sign_stable"] is False
        assert sens["any_significant"] is False


class TestSpikeDiagnostics:
    """Backs the write-up's claim that the spike bursts are not an implementation defect."""

    def test_no_numerical_failures_anywhere(self):
        diag = spike_diagnostics()
        assert diag["_verdict"]["any_nonfinite"] is False
        assert diag["_verdict"]["all_loss_decreased"] is True

    def test_spikes_move_the_loss_not_just_the_gradient(self):
        """A backward-pass bug would spike |g| while the loss sat still; it does not."""
        diag = spike_diagnostics()
        for name in SOLVER_ORDER:
            d = diag[name]
            assert d["median_loss_on_spikes"] > d["median_loss_on_calm"], name

    def test_spikes_are_transient(self):
        diag = spike_diagnostics()
        for name in SOLVER_ORDER:
            assert diag[name]["frac_recovered_after_spike"] > 0.5, name

    def test_bursts_are_present_in_every_solver(self):
        """Shared across p=1 and p=5 alike -- which is why they are not truncation error."""
        diag = spike_diagnostics()
        for name in SOLVER_ORDER:
            assert diag[name]["spike_epochs"] > 500, name


class TestArtifacts:

    def test_plot_writes_a_file(self, tmp_path):
        runs = load_grad_norms()
        out = tmp_path / "fig.png"
        plot_gradient_dynamics(runs, summarize(runs), str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_table_json_schema(self):
        """Downstream Phase 3 synthesis reads this file; a missing key breaks it silently."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "implementation",
            "results", "phase3", "gradient_dynamics", "table.json",
        )
        if not os.path.exists(path):
            pytest.skip("table.json not generated yet -- run the analysis script first")
        with open(path) as fh:
            table = json.load(fh)
        assert {r["solver"] for r in table["solvers"]} == set(SOLVER_ORDER)
        assert isinstance(table["correlation"]["rho"], float)
        assert isinstance(table["correlation"]["p_value"], float)
        assert "warmup_sensitivity" in table
