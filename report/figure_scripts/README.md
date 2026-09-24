# Figure scripts

Scripts that draw or redraw report figures from saved results. None of them trains a
model. Each writes directly into `report/figures/`. Where a script reuses one of the
project's own plotting functions, it loads a copy with only the labels changed, so
the code under `implementation/` is never modified.

| Script | Figures |
| :--- | :--- |
| `basis_functions.py` | `methods/basis_functions.png`: the seven bases, evaluated by `kan/basis.py` |
| `phase2_trajectories.py` | `phase2/sir_gated_trajectory.png`, `phase2/pendulum_silu_win5_trajectory.png`, rebuilt from checkpoints with correct state labels |
| `hybrid_gate_traces.py` | `track_c/*_alpha_beta.png`: gate traces from the saved training histories |
| `damping_figures.py` | `track_d/` verdict map, seed heatmaps, per-solver field landscapes and the two trajectory grids. It rebuilds the seed-42 trajectories from the saved checkpoints (the original trajectory files were never kept locally) and checks each against its recorded training MSE |
| `rise_bars.py` | `track_d/rise_comparison_bars.png`, from `rise_multiseed.json` |
| `track_e_figures.py` | `track_e/` time-scale sweep, epidemic fit and pruning figures |
| `retile_collages.py` | Superseded by `damping_figures.py`; re-tiles the old 1x4 collages into 2x2 grids |

Run from anywhere, e.g. `python report/figure_scripts/damping_figures.py`.
