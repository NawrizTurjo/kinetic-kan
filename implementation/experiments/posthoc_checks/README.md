# Post-hoc checks

Analyses run on saved checkpoints and training records after the main experiments,
to answer review questions on the final report. None of them trains a model, and
none is covered by the automated test suite in `tests/`.

| Script | Report location | What it computes |
| :--- | :--- | :--- |
| `review_analyses.py` | Pendulum fix table; hybrid-basis contribution table; epoch-budget section | Pendulum models rescored on the common window (5, 10]; each hybrid basis's actual RMS contribution; RBF vs. B-spline training loss at a matched 25,000 epochs |
| `damping_equilibria.py` | Damping study: stability analysis and learned-equilibria table | Newton's method on the learned field's equilibria, Jacobian eigenvalues there, and the largest Jacobian eigenvalue along the trajectory (h\|λ\|) |
| `far_horizon_50k.py` | Extended-extrapolation table at 50,000 epochs | Near (3.5, 14] and far (14, 28] extrapolation MSE for the 10k and 50k checkpoints |
| `sindy_exact_derivatives.py` | Sparse-regression derivative table | Noise-free SINDy with finite-difference vs. exact derivatives |

Run from anywhere, e.g. `python implementation/experiments/posthoc_checks/review_analyses.py`.
`review_analyses.py` extracts the original failing pendulum checkpoint from git
history (commit `6f37a66`) into `_cache/` on first run, so it needs `git` on the path.

The scripts use Windows-style path fragments, like the rest of the project's tooling.
The figure scripts that redraw report figures from saved results live in
`report/figure_scripts/`.
