# Reproducing the report's figures, tables and numbers

This folder regenerates everything in `report/` that is **not** a raw training
output: the analysis figures, the results tables, every derived number quoted
in the text, the TikZ diagrams, and the copies of the team's original result
plots. **Nothing here retrains a model.** Every input is a saved run under
`implementation/results/`, written there by `implementation/train.py`.

## Quick start

From the repository root:

```bash
python report/reproduce/run_all.py              # all steps, then builds report/report.pdf
python report/reproduce/run_all.py --no-report  # all steps, no LaTeX build
python report/reproduce/run_all.py s04 s09      # only some steps (prefix match)
python report/reproduce/s04_stability_regions.py  # any step also runs on its own
```

A full run takes about 2.5 minutes on a laptop CPU, plus about a minute for the report build.

Requirements: the implementation's own Python environment (`requirements.txt`
at the repository root; the versions used are in `requirements.txt` here) and,
for the diagrams, `pdflatex` on `PATH` or the bundled `report/tools/tectonic.exe`.

## How it guarantees the numbers are the trained models' numbers

`s00_verify_checkpoints.py` runs first. It reloads every noise-free `dt = 0.1`
checkpoint (`best_model.pt`), integrates it from `u(0)` with its training
solver, and checks that the extrapolation MSE matches the value training wrote
to `metrics.json`. The worst relative deviation is 6.2e-3, from cloud-trained
checkpoints re-scored on a different CPU. The script fails at 1e-2. Every later
step recomputes quantities from those same verified checkpoints.

## Files

| File | Role |
|---|---|
| `common.py` | paths, the registry of analysed runs (`RUNS`), loaders, Lotka-Volterra ground truth, cached rollouts, figure style, `numbers.json` I/O |
| `rk_theory.py` | Butcher tableaux as implemented, rooted trees, elementary weights, stability function R(z) |
| `s00` … `s14` | one script per figure or result group (table below) |
| `collect_result_figures.py` | copies the team's original plots into `report/figures/` and checks they are byte-identical (`--check` only verifies) |
| `build_tikz.py` | compiles `report/figures/tikz/*.tex` to PDF |
| `run_all.py` | runs every step in order, stops at the first failure, then builds the report |

## Outputs

* `report/figures/analysis/*.pdf`: the analysis figures
* `report/figures/analysis/numbers.json`: every derived number quoted in the text. Each script merges its own keys, so steps can be re-run individually.
* `report/figures/analysis/tables.md`: the results tables, formatted as in the report

## Which script makes what

Section numbers refer to the compiled report.

| Report item | Label | Script | Inputs |
|---|---|---|---|
| Table: solver ablation | `tab:solver-ablation` | `s01_tables.py` (+ truncation column from `s03`) | `benchmarks/ablation_solvers/solver_*/metrics.json` |
| Table: basis ablation | `tab:basis-ablation` | `s01_tables.py` (+ κ₂ column from `s05`) | `benchmarks/ablation_activations/basis_*/metrics.json` |
| Table: step size and noise | `tab:noise-dt` | `s01_tables.py` | `benchmarks/stepsize/dt*/`, `benchmarks/noise/sigma*/` |
| Table: KAN vs MLP, top block | `tab:kan-vs-mlp` | `s01_tables.py` | `benchmarks/kanode_flagship/`, `benchmarks/mlpode_baseline*/` |
| Table: KAN vs MLP, bottom block (field error, period, H drift) | `tab:kan-vs-mlp` | `s09`, `s06` | checkpoints |
| Table: per-epoch cost | `tab:cost` | `s01_tables.py` | `seconds_per_epoch` of local and `phase4/` runs |
| Table: Phase 4 | `tab:phase4` | `s01_tables.py` | `phase4/epoch_budget_check/*/metrics.json` |
| Table: order conditions (Sec. 3.1) | `tab:order-conditions` | `s02_order_conditions.py` | tableaux in `implementation/ode/solvers.py` |
| Fig: work-precision (Sec. 3.1) | `fig:work-precision` | `s03_work_precision.py` | true field, DOP853 reference |
| Fig: stability regions (Sec. 3.1) | `fig:stability` | `s04_stability_regions.py` | tableaux, `solver_tsit5` checkpoint |
| Fig: basis catalogue and κ₂ (Sec. 3.3) | `fig:basis-catalog` | `s05_basis_catalog.py` | `implementation/kan/basis.py` |
| Table: learned equilibria (Sec. 3.5) | `tab:equilibria` | `s06_equilibria_invariant.py` | 18 checkpoints |
| Fig: first-integral drift (Sec. 3.5) | `fig:first-integral` | `s06_equilibria_invariant.py` | solver checkpoints, true field |
| Fig: limit-cycle test (Sec. 3.5) | `fig:limit-cycle` | `s07_limit_cycle.py` | KAN / MLP checkpoints, 10k and 50k |
| Fig: cross-solver transfer; Euler modified equation (Sec. 3.2) | `fig:cross-solver` | `s08_cross_solver.py` | solver checkpoints |
| Fig: field-error maps (Sec. 3.5) | `fig:field-error` | `s09_field_error_maps.py` | KAN / MLP checkpoints, 10k and 50k |
| Fig: learned edge functions (Sec. 3.5) | `fig:edges` | `s10_edge_functions.py` | RBF and B-spline checkpoints |
| Table: training milestones (Sec. 3.6); Phase 4 thresholds, paper targets, crossover (Sec. 9) | `tab:milestones` | `s11_training_dynamics.py` | `training_history.json` |
| Fig: cost-accuracy Pareto (Sec. 3.2) | `fig:pareto` | `s12_pareto.py` | `metrics.json` |
| Fig: noise sensitivity, right panel (Sec. 3.4) | `fig:noise-dt` | `s13_noise_stepsize.py` | noise-sweep `metrics.json` |
| Fig: error against time (Sec. 3.5) | `fig:error-vs-time` | `s14_error_vs_time.py` | checkpoints |

### Original result plots (copied unchanged)

These were made by the project code during training and collation, not by
this folder. `collect_result_figures.py` copies them and checks each copy
byte for byte.

| Report file (`report/figures/…`) | Source (`implementation/results/…`) | Made by |
|---|---|---|
| `ablation/03_kan_vs_mlp_convergence.png`, `07_extrapolation_t28.png` | `figures/` | `collate_results.py` |
| `ablation/02_error_vs_stepsize_loglog.png`, `05_solver_convergence.png`, `06_basis_convergence.png` | `figures/` | `collate_results.py` |
| `ablation/solver_{euler,midpoint,tsit5}_phase.png` | `benchmarks/ablation_solvers/solver_*/phase_space.png` | `train.py` |
| `ablation/basis_{bspline,rbf,chebyshev,newton}_phase.png` | `benchmarks/ablation_activations/basis_*/phase_space.png` | `train.py` |
| `phase4/*_loss_curves.png` | `phase4/epoch_budget_check/*/loss_curves.png` | `train.py` |

### Diagrams

`build_tikz.py` compiles `report/figures/tikz/`: `kan_ode_system` (training
loop), `kdense_layer` (KDense layer), `data_split` (time windows), `rk_stages`
(RK stage structure) and `local_vs_global_basis` (local and global support).
They are drawings, not data. The one with curves evaluates the basis formulas
analytically in pgfplots.

## Numbers that are not in `numbers.json`

* The Tracks A–E sections (report Sections 4–8) and the cross-domain
  fixes (pendulum, SIR) are unchanged from the team's report. Their figures
  are in `report/figures/phase2/` and `track_*/`, from the experiment scripts
  under `implementation/experiments/`.
* Table `tab:mlp-breakdown` (the 2000-epoch MLP factorial) is also from the
  team's report.
