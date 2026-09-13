# Solver-Aware Neural Dynamics: KAN-ODEs Project State

Implementation and ablation study based on:
> **"KAN-ODEs: Kolmogorov–Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics"**  
> *Benjamin C. Koenig, Suyong Kim, Sili Deng (arXiv:2407.04192, 2024, MIT)*

---

## 📌 Project Overview & Scope
The base paper fixed **Tsit5 (Tsitouras 5/4)** integration with **Gaussian RBF** activations.  
Our project systematically stress-tests both design choices through comprehensive ablation studies across:
1. **Basis / Activation functions** (RBF, B-spline, Polynomials)
2. **ODE Integrators** (Euler, Heun RK2, RK4, DOPRI5, Tsit5)
3. **Dynamical Systems & Robustness** (Lotka-Volterra, Damped Pendulum, SIR Epidemic, Noise sweep)

---

## ⚙️ Technical Setup & Constraints
* **Optimization Method:** Direct backpropagation through the computational graph (PyTorch Autograd). Adjoint sensitivity optimization is not required here because training trajectories are short ($N = 36$ points), making direct backprop faster, exact, and numerically stable without memory bottlenecks.
* **Epoch Budget:** 10,000 epochs per run (the base paper used 100,000, but 10,000 takes ~40 minutes and is sufficient for comparative ablation).

---

## ✅ What is Done
- [x] **Base Paper Pipeline:**
  - [x] Gaussian RBF basis (`kan/basis.py`, `kan/layer.py`, `kan/model.py`)
  - [x] Native PyTorch Tsitouras 5/4 Runge-Kutta integrator (`ode/solvers.py`, `ode/neural_ode.py`) — fixed-step only, not yet adaptive (see TODOs)
  - [x] Lotka-Volterra synthetic data generator ($t_{\text{train}} \in [0, 3.5]$, $t_{\text{full}} \in [0, 14.0]$)
  - [x] Full training, evaluation, and plotting pipeline (`train.py`, `evaluate.py`)
  - [x] Baseline tested & validated (RBF + Tsit5 @ 10,000 epochs)
- [x] **All Basis Functions Implemented & Verified:**
  - [x] Gaussian RBF, RSWAF, IQF
  - [x] Cox-de Boor B-spline ($k=3$), including a clamped/open-uniform knot vector fix so partition-of-unity holds exactly at the domain boundaries (previously decayed to ~17% coverage at the edges), plus a NaN-gradient fix in the Cox-de Boor recursion's division guards
  - [x] Chebyshev polynomials (first kind), Lagrange cardinal polynomials, Newton's divided differences
- [x] **All Solvers Implemented & Verified:** `euler`, `midpoint`, `heun`, `rk4`, `dopri5`, `tsit5` — empirically confirmed to match their theoretical convergence order ($p=1,2,2,4,5,5$) against analytical test ODEs; Tsit5/DOPRI5 Butcher tableaus (`TSIT5_A/B/C`, `DOPRI5_A/B/C`) confirmed correct against the canonical SciML/Tsitouras (2011) coefficients; the previously-unused `TSIT5_E` error-estimator array was also corrected to the canonical values
- [x] **Parameter-Matched MLP-ODE Baseline** (`kan/mlp.py`) for KAN-vs-MLP comparison
- [x] **Extended Dynamical Systems:** Damped Pendulum (`data/damped_pendulum.py`), SIR Epidemic (`data/real_epidemic.py`), 3D chaotic Lorenz attractor (`data/lorenz.py`, bonus system beyond original scope)
- [x] **SciML Metrics Suite** (`utils/metrics.py`): MSE/RMSE/MAE/R²/Relative-L2, NFE tracker, gradient-norm logging, Lipschitz bound estimator, energy-drift diagnostics
- [x] **Benchmarking Infrastructure:**
  - [x] Automated test runner (`test_facility.py`) with metric logging and comparison bar charts
  - [x] Full 10,000-epoch Phase 2 production sweeps across all solvers and all basis functions (`results/benchmarks/`, written up in `docs/05_phase2_benchmark_analysis.md`)
- [x] **Test Suite:** 143/143 tests passing (`tests/`) covering solvers, KAN layers, MLP baseline, datasets, metrics, plotting, and end-to-end pipeline

---

## 📋 What Needs to Be Done (Team TODOs)

> **This section was significantly out of date** (it still listed several items as
> pending that `docs/07_fix_changelog.md`–`docs/11_phase2_closeout.md` had already
> resolved) and has been reconciled against those docs. See `docs/09_stability_fix_results.md`
> §Repo hygiene for the audit that caught this.

### ✅ Resolved since this list was last accurate
- **B-spline ablation re-run** — done; the boundary/NaN-gradient fixes described in
  "What is Done" above landed, and `results/benchmarks/kanode_bspline/` (and the
  ablation-sweep entry) reflect the corrected basis, not the old R² = −2.30 result.
- **Flagship KAN-vs-MLP LR mismatch** — resolved. Both `train.py` and
  `test_facility.py` now default to `lr=2e-3` ("standardised across the project," per
  `test_facility.py`'s own comment), and `results/benchmarks/kanode_flagship/metrics.json`
  now matches `ablation_solvers/solver_tsit5/metrics.json` to the same `train_mse`
  (`8.832586172502488e-05`) — the five-way determinism check documented in
  `docs/05_phase2_benchmark_analysis.md`.
- **Noise robustness sweep** — done (`results/benchmarks/noise/sigma*/`).
- **`--dataset` CLI wiring for damped pendulum, Lorenz, SIR** — done for the *dataset
  choice itself*; `train.py`'s `DATASETS` registry now drives all four via
  `--dataset {lotka_volterra,damped_pendulum,lorenz,sir}`. **Caveat, still relevant for
  Phase 3 Track D:** only Lotka-Volterra's own kwargs (`alpha, beta, gamma, delta`) are
  threaded through to the generator — there is still no `--mu` flag for the damped
  pendulum, so a μ-sweep cannot be done through this CLI (confirmed by reading
  `train_kan_ode`). Track D needs its own thin script, per
  `docs/12_phase3_roadmap.md`.
- **`evaluate.py` silent-data-mismatch bug** — fixed; it now rebuilds the field
  honouring `grid_lims`, `time_scale`, `conserve_mode`, and `vanish_dim` from the saved
  config rather than regenerating data with hardcoded defaults (`docs/10_sir_root_cause_and_fix.md`
  §Part 5 catalogues the three related bugs found and fixed here).
- **RBF basis formula** — fixed; `kan/basis.py::rbf` now implements the paper's Eq. 5
  exactly (`exp(-r²/(2h²))`), with the old bug documented in the function's own
  docstring for the record.

### 🔓 Still genuinely open
- **Adaptive Step-Size Integration.** `step_tsit5`/`step_dopri5` are still fixed-step
  (a fixed number of `substeps` per interval); the embedded error estimator (`TSIT5_E`,
  now corrected to canonical values) still isn't used for real step accept/reject
  logic. The solver docstrings now say this honestly rather than overclaiming
  adaptivity. Implementing true adaptivity remains open if exact reproduction of the
  paper's adaptive-solver behavior is ever needed.
- **No adjoint sensitivity method (paper's Eq. 10)** — still a deliberate, documented
  choice (see Technical Setup above), not a bug. Phase 3 Track A is where this
  actually gets tested empirically (see below).
- **$L_1$ edge-pruning step** — still doesn't exist anywhere in the code.
  `kan/model.py::regularization_loss` only *penalizes* parameter magnitude during
  training; nothing zeros out edges afterward. Needed by Phase 3 Track E.

### 🌟 Phase 3 — Novel Research Contributions (not yet started, no `experiments/` dir exists)
The 6-novelty list once here has been **superseded** by
[`docs/12_phase3_roadmap.md`](../docs/12_phase3_roadmap.md), which is now authoritative
for Phase 3 scope and ownership. Key differences from the original plan: the Lorenz
track has been **dropped entirely** (not deferred), and the SINDy/real-epidemic work
is merged into a single track. Current tracks:

| Track | Contribution | Owner |
| :-: | :--- | :--- |
| A | Adjoint vs. autograd memory/speed profiling (Table 4) | Nawriz Ahmed Turjo |
| B | Gradient norm dynamics vs. solver order | Abhishek Roy |
| C | Learnable softmax hybrid basis (spline + RBF) | Shams Hossain Simanto |
| D | Stiffness–solver stability phase map (Table 5) | Abrar Jahin |
| E | SINDy comparison under noise + real epidemic fit (Table 3) | Monjur Hossain Khan |

Each track is new files only, in its own `experiments/<slug>/` and
`results/phase3/<slug>/` folder — see the roadmap's §Part 2 for the zero-conflict
folder architecture and exactly what existing code each track reuses.

---

## 📁 Repository Structure
```
Implementation/
├── kan/
│   ├── basis.py           # RBF, RSWAF, IQF, B-spline, Chebyshev, Lagrange, Newton
│   ├── layer.py           # KDense layer
│   ├── model.py           # Multi-layer KAN
│   └── mlp.py             # Parameter-matched MLP-ODE baseline
├── ode/
│   ├── solvers.py         # Tsit5, RK4, DOPRI5, Euler, Heun, Midpoint (all fixed-step)
│   └── neural_ode.py      # NeuralODE wrapper
├── data/
│   ├── lotka_volterra.py  # Lotka-Volterra generator
│   ├── damped_pendulum.py # Non-linear damped pendulum generator
│   ├── lorenz.py          # 3D chaotic Lorenz attractor generator
│   └── real_epidemic.py   # SIR epidemic generator
├── utils/
│   ├── regularization.py  # L1 & Entropy regularization
│   ├── metrics.py         # MSE/RMSE/MAE/R²/Rel-L2, NFE, gradient norm, Lipschitz bound
│   └── plotting.py        # Trajectory, Phase Portrait, Loss & Benchmark plots
├── train.py               # Main training script
├── evaluate.py            # Checkpoint evaluation & metrics
├── test_facility.py       # Automated ablation benchmark suite
└── README.md              # Project status and guide

tests/                     # Project-wide PyTest suite (143/143 passing)
├── test_solvers.py        # Convergence order, canonical tableaus
├── test_kan_layers.py     # Basis shapes, partition of unity, autograd flow
├── test_mlp_ode.py        # MLP-ODE parameter matching
├── test_datasets.py       # Dataset generators & physical invariants
├── test_metrics.py        # SciML metrics correctness
├── test_pipeline.py       # End-to-end forward/backward pipeline
└── test_plotting.py       # Plotting utilities

docs/06_suggested_fixes.md    # Cross-domain stability diagnosis (pendulum, SIR)
docs/07_fix_changelog.md      # What changed in train.py / run_phase2.ps1 to fix it
docs/09_stability_fix_results.md  # Pendulum fix verdict + repo hygiene audit
docs/10_sir_root_cause_and_fix.md # SIR root cause, fix, and 3 collateral bugs found
docs/11_phase2_closeout.md    # Extrapolation-to-t=28, energy diagnostics, figures
docs/12_phase3_roadmap.md     # Authoritative Phase 3 scope, tracks, and ownership
```

---

## 🚀 Commands Quick Reference

### Baseline Training (RBF + Tsit5)
```bash
python train.py --basis rbf --solver tsit5 --epochs 10000 --lr 5e-4
```

### Checkpoint Evaluation
```bash
python evaluate.py --checkpoint results/kanode_rbf_tsit5/best_model.pt
```

### Running Ablation Benchmarks
```bash
# Basis ablation
python test_facility.py --mode activations --epochs 1500

# Solver ablation
python test_facility.py --mode solvers --epochs 1500
```

---

## 🧩 Modularity: How to Add New Basis Functions & Solvers

### 1. Adding a New Basis Function
Add your custom basis function in `kan/basis.py` and register it in `BASIS_FUNCTIONS`:
```python
def my_custom_basis(x: torch.Tensor, grid: torch.Tensor, h: float) -> torch.Tensor:
    # x: normalized input [batch, in_features]
    # grid: basis centers [in_features, grid_len]
    # h: grid width (spacing)
    # returns: basis evaluations [batch, in_features, grid_len]
    return torch.sin((x.unsqueeze(-1) - grid) / h)

BASIS_FUNCTIONS["my_basis"] = my_custom_basis
```
Then train with:
```bash
python train.py --basis my_basis
```

### 2. Adding a New ODE Solver
Add your single-step solver function in `ode/solvers.py` and register it in `STEP_SOLVERS`:
```python
def step_my_solver(func, t: torch.Tensor, y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    # func: callable f(t, y) returning dy/dt
    # y: state tensor at time t
    # dt: step size
    # returns: state tensor at time t + dt
    return y + dt * func(t, y)

STEP_SOLVERS["my_solver"] = step_my_solver
```
Then train with:
```bash
python train.py --solver my_solver
```


