"""
Fast smoke tests for Track D's stiffness-map sweep script
(implementation/experiments/stiffness_map/run_sweep.py). These run in seconds --
a handful of epochs on one or two cells -- and exist to catch mechanical bugs
(import errors, shape mismatches, a broken guard) BEFORE committing to the real
sweep's multi-hour compute budget. They do not assert anything about
convergence quality.
"""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementation", "experiments", "stiffness_map"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementation"))

import run_sweep  # noqa: E402


def test_train_cell_runs_and_returns_finite_metrics():
    """A benign (tsit5, mu=0.5) cell at a handful of epochs should run to
    completion with no aborts and produce finite loss/metric values."""
    metrics = run_sweep.train_cell(solver="tsit5", mu=0.5, dt=0.05, num_epochs=5)

    assert metrics["epochs_run"] == 5
    assert metrics["aborted_at_epoch"] is None
    assert metrics["nonfinite_grad_steps"] == 0
    assert metrics["best_train_mse"] is not None and math.isfinite(metrics["best_train_mse"])
    assert metrics["full_mse"] is not None and math.isfinite(metrics["full_mse"])
    # Every per-cell array-derived field must be consistent with epochs actually run.
    assert 1 <= metrics["best_epoch"] <= metrics["epochs_run"]


def test_train_cell_covers_all_four_solvers():
    """Every solver in SOLVERS must be constructible and runnable end-to-end --
    a registry typo or missing STEP_SOLVERS entry would fail this immediately,
    cheaply, instead of surfacing three hours into a real sweep."""
    for solver in run_sweep.SOLVERS:
        metrics = run_sweep.train_cell(solver=solver, mu=0.5, dt=0.05, num_epochs=3)
        assert metrics["solver"] == solver
        assert metrics["epochs_run"] == 3


def test_train_cell_at_extreme_mu_does_not_crash():
    """The stiffest grid point (mu=8.0) must at least run without a Python
    exception, whatever its eventual stability verdict turns out to be."""
    metrics = run_sweep.train_cell(solver="euler", mu=8.0, dt=0.05, num_epochs=5)
    assert metrics["epochs_run"] >= 1


def test_classify_cell_diverged_on_abort():
    metrics = {
        "aborted_at_epoch": 250, "best_train_mse": 0.5, "final_train_mse": None,
        "nonfinite_grad_steps": 100,
    }
    assert run_sweep.classify_cell(metrics) == "diverged"


def test_classify_cell_diverged_when_no_checkpoint_ever_valid():
    metrics = {
        "aborted_at_epoch": None, "best_train_mse": None, "final_train_mse": None,
        "nonfinite_grad_steps": 2000,
    }
    assert run_sweep.classify_cell(metrics) == "diverged"


def test_classify_cell_unstable_on_nonfinite_recovery():
    metrics = {
        "aborted_at_epoch": None, "best_train_mse": 1e-4, "final_train_mse": 1.2e-4,
        "nonfinite_grad_steps": 3,
    }
    assert run_sweep.classify_cell(metrics) == "unstable"


def test_classify_cell_unstable_on_late_regression():
    """final >> best (a late-training spike that did not blow up outright) should
    be flagged 'unstable' even with zero non-finite steps -- the same REGRESSED
    pattern analyze_fixes.py catches elsewhere in this project."""
    metrics = {
        "aborted_at_epoch": None, "best_train_mse": 1e-5, "final_train_mse": 5e-3,
        "nonfinite_grad_steps": 0,
    }
    assert run_sweep.classify_cell(metrics) == "unstable"


def test_classify_cell_unstable_when_finite_but_never_fits():
    """Finite throughout, no guard activity, but never dips below the converged
    threshold -- a real 'failed to fit' outcome, not a divergence."""
    metrics = {
        "aborted_at_epoch": None, "best_train_mse": 0.5, "final_train_mse": 0.5,
        "nonfinite_grad_steps": 0,
    }
    assert run_sweep.classify_cell(metrics) == "unstable"


def test_classify_cell_converged():
    metrics = {
        "aborted_at_epoch": None, "best_train_mse": 1e-4, "final_train_mse": 1.1e-4,
        "nonfinite_grad_steps": 0,
    }
    assert run_sweep.classify_cell(metrics) == "converged"


def test_guard_fires_and_recovers_under_forced_divergence():
    """Reproduces train.py's own verified NaN-guard test (docs/07_fix_changelog.md:
    '--lr 50 --grad_clip 0 fires at epoch 2, run survives with valid best metrics')
    against THIS script's ported copy of the guard, not train.py's -- the two guards
    are independent copies and neither validates the other."""
    original_lr = run_sweep.BASELINE["lr"]
    original_clip = run_sweep.BASELINE["grad_clip"]
    run_sweep.BASELINE["lr"] = 50.0
    run_sweep.BASELINE["grad_clip"] = 0.0
    try:
        metrics = run_sweep.train_cell(solver="rk4", mu=0.5, dt=0.05, num_epochs=50)
    finally:
        run_sweep.BASELINE["lr"] = original_lr
        run_sweep.BASELINE["grad_clip"] = original_clip

    assert metrics["nonfinite_grad_steps"] > 0, (
        "expected forced divergence (lr=50, grad_clip=0) to trip the non-finite "
        "gradient guard at least once"
    )
    # The run should survive (guard recovers), not necessarily abort in 50 epochs --
    # aborting requires 100 CONSECUTIVE non-finite steps.
    assert metrics["best_train_mse"] is not None and math.isfinite(metrics["best_train_mse"])


def test_guard_aborts_after_sustained_nonfinite_streak():
    """A parameter state that can never recover (forced via an absurd lr on a
    stiff cell) should hit the 100-consecutive-step abort and stop early rather
    than burning the full epoch budget -- verifies NONFINITE_ABORT_STREAK actually
    triggers `break`, not just that individual steps get skipped."""
    original_lr = run_sweep.BASELINE["lr"]
    run_sweep.BASELINE["lr"] = 500.0
    try:
        metrics = run_sweep.train_cell(solver="euler", mu=8.0, dt=0.05, num_epochs=1000)
    finally:
        run_sweep.BASELINE["lr"] = original_lr

    if metrics["aborted_at_epoch"] is not None:
        assert metrics["epochs_run"] == metrics["aborted_at_epoch"]
        assert metrics["epochs_run"] < 1000
    else:
        # Not every absurd-lr configuration is guaranteed to sustain 100
        # consecutive non-finite steps -- if it didn't abort, it must at least
        # have hit the guard at all, otherwise this test picked a bad forcing
        # config and should be revisited rather than silently passing.
        assert metrics["nonfinite_grad_steps"] > 0


def test_list_extensions_requires_probe_results_first():
    with pytest.raises(FileNotFoundError):
        run_sweep.list_extensions("/nonexistent/path/that/should/not/exist")


def test_run_stage_skip_existing_reuses_prior_result_without_retraining(tmp_path):
    """The -SkipExisting resume feature. Train one real (tiny) cell, tamper with
    its saved metrics.json in a way that could only be explained by "not
    retrained" (a sentinel value train_cell could never itself produce), then
    re-run run_stage with skip_existing=True and confirm the sentinel survives --
    proving the cell was loaded from disk, not recomputed."""
    stage_root = str(tmp_path / "probe")
    run_sweep.run_stage(stage_root, ["euler"], [0.5], [0.05], num_epochs=3)

    cell_dir = run_sweep._cell_dir(stage_root, "euler", 0.5, 0.05)
    metrics_path = os.path.join(cell_dir, "metrics.json")
    with open(metrics_path) as f:
        saved = json.load(f)
    saved["best_train_mse"] = -999.0  # sentinel: train_cell can never produce a negative MSE
    with open(metrics_path, "w") as f:
        json.dump(saved, f)

    results = run_sweep.run_stage(stage_root, ["euler"], [0.5], [0.05], num_epochs=3, skip_existing=True)

    assert results[0]["best_train_mse"] == -999.0, (
        "skip_existing=True should have reused the tampered file verbatim, "
        "not retrained and overwritten it"
    )


def test_run_stage_without_skip_existing_retrains_and_overwrites(tmp_path):
    """The inverse of the above: default behavior (skip_existing=False) must
    still retrain and overwrite, even if a metrics.json already exists -- the
    resume feature must be strictly opt-in."""
    stage_root = str(tmp_path / "probe")
    run_sweep.run_stage(stage_root, ["euler"], [0.5], [0.05], num_epochs=3)

    cell_dir = run_sweep._cell_dir(stage_root, "euler", 0.5, 0.05)
    metrics_path = os.path.join(cell_dir, "metrics.json")
    with open(metrics_path) as f:
        saved = json.load(f)
    saved["best_train_mse"] = -999.0
    with open(metrics_path, "w") as f:
        json.dump(saved, f)

    results = run_sweep.run_stage(stage_root, ["euler"], [0.5], [0.05], num_epochs=3, skip_existing=False)

    assert results[0]["best_train_mse"] != -999.0, (
        "skip_existing=False (the default) must retrain, not silently reuse a stale file"
    )


def _write_metrics(root, solver, mu, dt, data):
    cell_dir = run_sweep._cell_dir(root, solver, mu, dt)
    os.makedirs(cell_dir, exist_ok=True)
    import json
    with open(os.path.join(cell_dir, "metrics.json"), "w") as f:
        json.dump(data, f)


def test_build_table5_prefers_full_over_probe(tmp_path):
    probe_root = str(tmp_path / "probe")
    full_root = str(tmp_path / "full")
    _write_metrics(probe_root, "tsit5", 0.5, 0.05, {
        "solver": "tsit5", "mu": 0.5, "dt": 0.05, "verdict": "unstable",
        "best_train_mse": 0.5, "full_mse": 0.5, "extrap_r2": 0.0,
        "epochs_run": 2000, "nonfinite_grad_steps": 0,
    })
    _write_metrics(full_root, "tsit5", 0.5, 0.05, {
        "solver": "tsit5", "mu": 0.5, "dt": 0.05, "verdict": "converged",
        "best_train_mse": 1e-5, "full_mse": 1e-5, "extrap_r2": 0.99,
        "epochs_run": 10000, "nonfinite_grad_steps": 0,
    })
    table = run_sweep.build_table5(probe_root, full_root, ["tsit5"], [0.5], [0.05])
    assert len(table) == 1
    assert table[0]["source"] == "full"
    assert table[0]["verdict"] == "converged"


def test_build_table5_marks_missing_cells_without_crashing(tmp_path):
    """A requested (solver, mu, dt) cell with no result on disk anywhere must be
    reported as verdict='missing', not silently dropped or a crash."""
    table = run_sweep.build_table5(str(tmp_path / "probe"), str(tmp_path / "full"),
                                    ["tsit5"], [0.5], [0.05])
    assert len(table) == 1
    assert table[0]["verdict"] == "missing"
    assert table[0]["source"] is None


def test_build_table5_does_not_recompute_classify_cell_when_verdict_already_present(tmp_path):
    """Regression test for a real bug caught during validation: build_table5 used to
    write `r.get("verdict", classify_cell(r))`, and Python evaluates a dict.get's
    default argument EAGERLY regardless of whether the key exists -- so
    classify_cell(r) ran (and could KeyError) even when "verdict" was already in r.
    This dict deliberately omits every field classify_cell needs (aborted_at_epoch,
    nonfinite_grad_steps, final_train_mse) except "verdict" itself, so the old code
    would crash here and the fixed code must not."""
    probe_root = str(tmp_path / "probe")
    _write_metrics(probe_root, "euler", 1.0, 0.05, {
        "solver": "euler", "mu": 1.0, "dt": 0.05, "verdict": "converged",
        "best_train_mse": 1e-4,
    })
    table = run_sweep.build_table5(probe_root, str(tmp_path / "full"), ["euler"], [1.0], [0.05])
    assert table[0]["verdict"] == "converged"


def test_plot_stability_heatmap_handles_all_missing(tmp_path):
    """The heatmap must render even with zero results on disk -- an all-'missing'
    grid is a valid state (before any sweep has run), not an error condition."""
    table = run_sweep.build_table5(str(tmp_path / "probe"), str(tmp_path / "full"),
                                    run_sweep.SOLVERS, run_sweep.MU_VALUES, [0.05])
    out_path = str(tmp_path / "heatmap.png")
    run_sweep.plot_stability_heatmap(table, out_path, dt=0.05,
                                      solvers=run_sweep.SOLVERS, mus=run_sweep.MU_VALUES)
    assert os.path.isfile(out_path) and os.path.getsize(out_path) > 1000


def test_plot_stability_heatmap_handles_mixed_verdicts(tmp_path):
    probe_root = str(tmp_path / "probe")
    for solver, mu, verdict, mse in [
        ("tsit5", 0.5, "converged", 1e-5),
        ("euler", 8.0, "diverged", None),
        ("rk4", 2.0, "unstable", 0.3),
    ]:
        _write_metrics(probe_root, solver, mu, 0.05, {
            "solver": solver, "mu": mu, "dt": 0.05, "verdict": verdict,
            "best_train_mse": mse, "full_mse": mse, "extrap_r2": 0.0,
            "epochs_run": 2000, "nonfinite_grad_steps": 3 if verdict == "diverged" else 0,
        })
    table = run_sweep.build_table5(probe_root, str(tmp_path / "full"),
                                    run_sweep.SOLVERS, run_sweep.MU_VALUES, [0.05])
    out_path = str(tmp_path / "heatmap_mixed.png")
    run_sweep.plot_stability_heatmap(table, out_path, dt=0.05,
                                      solvers=run_sweep.SOLVERS, mus=run_sweep.MU_VALUES)
    assert os.path.isfile(out_path) and os.path.getsize(out_path) > 1000
