# Track D — Stiffness–Solver Stability Phase Map

**Owner:** Abrar Jahin (2105055) · **Branch:** `feat/p3-stiffness-map`
**Chain:** `docs/09_stability_fix_results.md` (the fixed pendulum recipe this builds
on) → `docs/12_phase3_roadmap.md` Part 4, Track D (scope & task breakdown) → this doc.

## Research question

Sweep the damped pendulum's damping ratio μ ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 8.0} across
four solvers {Euler, Midpoint, RK4, Tsit5} at a fixed step size (Δt=0.05): which
(solver, μ) combinations converge, and which go unstable or diverge outright?

## Status: complete — core sweep, multi-seed extension, and both landscape analyses all finished and cross-verified

`run_sweep.py` passes 19 smoke tests (`tests/test_p3_stiffness_map.py`). The real
24-cell probe sweep (4 solvers × 6 μ, Δt=0.05, 2,000 epochs) has been run and
aggregated — `implementation/results/phase3/stiffness_map/table5.json` and
`stability_heatmap_dt0.05.png` both exist, built via `--stage aggregate`. The full
write-up is `docs/16_p3_stiffness_map_findings.md` — read that for the actual
findings; this file only tracks what's been built and where it lives.

**Everything that grew out of the original μ=2.0 anomaly, now finished:**
- **Multi-seed check, all 4 solvers × 6 μ × 3 seeds (72 cells)**, uniformly
  retrained with checkpoints saved for every cell (`kaggle_full_retrain.py`),
  merged into `implementation/results/phase3/stiffness_map/probe/` and `_seed_test/`.
  Aggregated by `plot_seed_heatmap.py` into median-MSE and converged-fraction
  heatmaps (`implementation/results/phase3/stiffness_map/figures/`).
- **Vector-field "landscape" plots** (`plot_landscape.py`) for all 6 μ, each
  solver rendered from its own checkpoint (`plot_landscape.plot_landscape_grid`,
  wired into `plot_trajectories.py`'s phase-portrait panel) — shows the learned
  dynamics' equilibrium structure directly, including the μ=2.0 spurious-basin
  finding and the μ=1.0 euler-specific trap, both confirmed per-solver.
- **`plot_trajectories.py`** — 4-panel-per-μ trajectory visualization (all panels
  3D), fully populated for all 6 μ values.
- **Training loss landscape** (weight-space, not phase-space — see docs/16 §8
  for the distinction), `kaggle_loss_landscape.py`, run for all 24 cells, plus a
  robustness follow-up (`kaggle_multi_direction_loss_landscape.py`) checking the
  result against 2 more random direction choices per cell. Comparison figures in
  `implementation/results/phase3/stiffness_map/loss_landscape/`.

**Sanity check** ((Tsit5, μ=0.5, Δt=0.05) vs. `pendulum_control_win5`, `--stage
sanity`) — run on Kaggle; did **not** cleanly reproduce (see docs/16 §7) —
training fit was actually slightly better but extrapolation quality dropped
substantially, traced to platform-level floating-point non-determinism, not a
bug. Reported as its own finding rather than treated as a failed check.

## Why this needs its own script

`train_kan_ode()` (`implementation/train.py`) only threads Lotka-Volterra kwargs
(`alpha, beta, gamma, delta`) through to whichever dataset generator is selected —
confirmed by reading the function directly. There is no way to pass `mu` to
`generate_damped_pendulum_data()` via `train.py`'s CLI, so a μ-sweep cannot be done
with `python train.py --dataset damped_pendulum` unchanged. `run_sweep.py` reuses
the same primitives `train_kan_ode()` itself composes — `KAN`, `NeuralODE`, `Adam`,
and a ported copy of the X1 non-finite-gradient guard from `train.py` — in a
standalone ~150-line loop. It does not import from or edit `train.py`, and it does
not touch `kan/`, `ode/`, `data/`, or `utils/`.

## The baseline every cell starts from

The `pendulum_control_win5` recipe from `docs/09_stability_fix_results.md`,
confirmed against that run's own `results/_fixed/pendulum_control_win5/metrics.json`
config block (not retyped from the doc's prose):

```
KAN([2, 10, 2]), grid_len=8, basis=rbf, normalizer=tanh, base_act=silu,
substeps=2, lr=0.003, grad_clip=1.0, t_train_end=5.0, t_end=10.0, dt=0.05, seed=42
```

Using the pre-fix `t_train_end=3.0` window here would just re-measure the
pendulum's already-diagnosed optimization failure at every μ, not stiffness —
`docs/09` and the roadmap's Track D section both warn about this explicitly.

## How to run

### Sequential (single process, simplest)

```powershell
cd implementation/experiments/stiffness_map

# 1. Probe every cell at 2,000 epochs (~cheap — the OFAT discipline from docs/09)
python run_sweep.py --stage probe

# 2. See which cells are worth the full budget, WITHOUT running anything
python run_sweep.py --stage list-extensions

# 3a. Extend exactly those cells to 10,000 epochs
python run_sweep.py --stage full --from-extensions

# 3b. Or run a specific subset manually
python run_sweep.py --stage full --solvers tsit5 rk4 --mus 0.5 2.0 8.0

# Built-in correctness check: reproduce pendulum_control_win5 exactly
python run_sweep.py --stage sanity

# Build table5.json + the heatmap PNG from whatever landed (safe any time, even
# with partial or zero results)
python run_sweep.py --stage aggregate
```

Measured per-epoch cost on the dev machine (NOT the roadmap doc's estimate, which
turned out to be roughly half the real figure): euler 0.73s, midpoint 0.91s,
rk4 1.82s, tsit5 3.02s. Full sequential 4×6 probe grid (24 cells × 2,000 epochs)
≈ **21.6h**.

### Parallel (recommended — `run_parallel.ps1`)

```powershell
cd implementation/experiments/stiffness_map

# Default: one SEPARATE console window per solver (4-way parallel), no
# Windows Terminal needed. Auto-sizes threads-per-job to the machine's core
# count. Each window stays open ("press Enter to close") when its solver's
# whole mu-list finishes.
.\run_parallel.ps1 -Stage probe

# Always dry-run a new invocation first -- prints exactly what would launch,
# launches nothing:
.\run_parallel.ps1 -Stage probe -DryRun

# Tiled single window (needs Windows Terminal's wt.exe on PATH; falls back to
# separate windows with a warning if it isn't found):
.\run_parallel.ps1 -Stage probe -Mode grid

# Hidden background + log files instead (e.g. no interactive console at all):
.\run_parallel.ps1 -Stage probe -Mode headless
```

With 4-way parallel (`-Mode windows`, the default), wall time is bounded by the
**slowest** window (tsit5, ~10.1h for the full 6-mu probe list), not the ~21.6h
sequential sum, since each window handles one solver's full mu-list on its own.

**Why `-Mode windows` is the default, not headless-with-`-NoNewWindow`:** an
earlier version used `-NoNewWindow`, and in production all 4 jobs died
simultaneously with `KeyboardInterrupt` seconds after launch (mid `import torch`)
when the parent console received an interrupt. `-NoNewWindow` children share the
*parent's* console signal — a Ctrl+C or closed window kills every child at once.
A genuinely separate console window (the current default) does not have that
failure mode.

## Stopping and resuming

- **Stop one solver only:** Ctrl+C or close that solver's window. Because each
  runs in its own console (see above), this does **not** affect the other three.
- **Stop all four at once:**
  ```powershell
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*run_sweep.py*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  ```
  (Scoped to processes actually running `run_sweep.py`, unlike a blanket
  `Get-Process python | Stop-Process`.)
- **What survives an interrupt:** each cell only writes its `metrics.json` at the
  very end of that cell's training, so every μ value a window already *finished*
  is safe on disk. Only the μ value it was mid-training on when killed is lost —
  no partial/corrupt file, it just won't exist yet.
- **Resuming without redoing finished work:** pass `-SkipExisting` (PowerShell
  launcher) or `--skip-existing` (`run_sweep.py` directly). Any cell whose
  `metrics.json` already exists is loaded and reused instead of retrained. This
  is strictly opt-in — the default still retrains and overwrites, so a
  deliberate re-run of a bad/stale cell isn't silently skipped.
  ```powershell
  .\run_parallel.ps1 -Stage probe -SkipExisting
  ```
- **Closing the launcher script's own window does NOT stop the spawned
  windows** (`-Mode windows`) — they are independent processes by design. Only
  closing/interrupting *their* windows stops them.

## Final status

Track D is done as originally scoped, and every self-initiated extension has
been completed and cross-checked against real data. Nothing is currently
outstanding. The one thing worth remembering if this track is revisited later:
the cross-platform reproducibility gap (docs/16 §7) means any future re-run on
a different machine should expect numbers to shift somewhat (verdicts have
stayed stable everywhere this was checked, exact values have not) — that's a
property of this project's numerics, not a bug to chase.
