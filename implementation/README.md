# Solver-Aware Neural Dynamics: KAN-ODEs Project State

Implementation and ablation study based on:
> **"KAN-ODEs: Kolmogorov–Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics"**  
> *Z. Koenig, J. Kim, Y. Deng (CMAME / arXiv:2407.04192, 2024, MIT)*

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

### 1. Re-run benchmarks affected by recent bug fixes
- [ ] **B-spline ablation re-run (priority):** The catastrophic B-spline result in `docs/05_phase2_benchmark_analysis.md` (R² = −2.30) was produced with a boundary bug that has since been fixed — that basis needs to be re-benchmarked before drawing conclusions about its extrapolation behavior.
- [ ] **Flagship KAN-vs-MLP comparison re-run:** `results/benchmarks/kanode_flagship/` and the ablation-sweep RBF+Tsit5 run report a ~1000x MSE discrepancy despite an identical stated config. Root cause identified as a learning-rate mismatch (`train.py` defaults to `lr=5e-4`, `test_facility.py` defaults to `lr=2e-3`) — the team needs to decide on one LR and re-run Table 3 of the benchmark doc on equal footing with Tables 1 & 2.

### 2. Adaptive Step-Size Integration
- [ ] `step_tsit5`/`step_dopri5` are currently **fixed-step** methods (run a fixed number of `substeps` per reporting interval); they don't yet use the embedded error estimator (`TSIT5_E`, now corrected) for actual adaptive step-size control. Implementing true adaptivity (step accept/reject, error-based step resizing) is still open if the paper's adaptive-solver behavior needs to be reproduced exactly.

### 3. Remaining Ablations & Robustness
- [ ] **Noise Robustness Sweep:** Evaluate sensitivity and generalization under measurement noise ($\sigma \le 0.10$) — not yet run.
- [ ] Wire up CLI/`train.py` support for the damped pendulum, Lorenz, and SIR datasets (currently only Lotka-Volterra is exposed via `--dataset`-style flags; the other generators exist in `data/` but aren't yet driveable from `train.py`/`test_facility.py`).
- [ ] `evaluate.py` regenerates ground-truth data with hardcoded defaults (`generate_lotka_volterra_data()` with no args) rather than reading the data-generation parameters from the saved checkpoint — fine for the default run, but will silently score against the wrong trajectory if a checkpoint was ever trained with a non-default seed/noise/equation parameters.

### 4. Fidelity to the Paper (optional, larger scope)
- [ ] The RBF basis formula (`kan/basis.py::rbf`) is `exp(-((x-z)/h)^2)`, which differs from the paper's Eq. 5 `exp(-r²/(2h²))` by a factor of 2 in the exponent denominator. The learned amplitude weights can partially compensate, but it's not a literal match if exact paper reproduction matters.
- [ ] No adjoint sensitivity method (paper's Eq. 10) — currently using direct autograd backprop through the unrolled solver, which is a deliberate, documented choice for this problem's short trajectories (see Technical Setup above), not a bug.

### 5. Novel Research Contributions (Phase 3 — not yet started, no `experiments/` dir exists)
These 6 novelties are planned in `docs/04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md` (Phase 3) as the project's original-contribution layer beyond reproducing the paper, each targeting its own results table/figure:
- [ ] **[Novelty 1] Gradient Norm Dynamics vs. Solver Order:** Log $\|\nabla_\theta \mathcal{L}\|_2$ every epoch across Euler/RK2/RK4/Tsit5 and show that lower-order solvers inject higher-frequency gradient noise into backprop. (`utils/metrics.py`'s `compute_gradient_norm` already exists and is usable as-is.)
- [ ] **[Novelty 2] Learnable Softmax Hybrid Basis Layer:** Add a new basis in `kan/basis.py` that blends B-spline and RBF via learnable softmax gate weights ($\alpha \cdot \text{Spline} + \beta \cdot \text{RBF}$); track how $\alpha(t)$/$\beta(t)$ evolve during training and whether the hybrid converges faster than either pure basis.
- [ ] **[Novelty 3] Stiffness–Solver Stability Phase Map:** Sweep the Damped Pendulum's damping ratio $\mu \in \{0.1, 0.5, 1.0, 2.0, 5.0, 8.0\}$ across all solvers and step sizes; produce a 2D stable-vs-exploded stability heatmap (→ Table 5).
- [ ] **[Novelty 4] SINDy Baseline Comparison under Noise:** Fit `pysindy.SINDy` on noisy Lotka-Volterra trajectories ($\sigma \in \{0.00, 0.01, 0.05, 0.10\}$) and compare its recovered symbolic equations against KAN-ODE's $L_1$-pruned edge formulas (→ Table 3). Note: this depends on an $L_1$-pruning step for KAN edges that doesn't exist yet either (see below).
- [ ] **[Novelty 5] Adjoint vs. Autograd Memory/Speed Profiling:** Benchmark `torchdiffeq.odeint_adjoint` ($O(1)$ memory) against the current unrolled `loss.backward()` ($O(N_t)$ memory) — peak VRAM, wall-clock per 1000 epochs, and gradient accuracy across state dimensions $d \in \{2, 8\}$ (→ Table 4). This is the natural place to finally test whether adjoint sensitivity (see Technical Setup / Fidelity to the Paper above) matters at larger scale.
- [ ] **[Novelty 6] 3D Lorenz Chaos Evaluation & Real-Data Fit:** `data/lorenz.py` and `data/real_epidemic.py` already exist — still needed: actually train a 3D KAN-ODE on the Lorenz attractor, render the true-vs-predicted strange-attractor geometry, and fit the real epidemic data end-to-end (currently only the data generators exist, not a trained model/evaluation for either).
- [ ] **Supporting building block:** none of the above have a real `experiments/` directory yet — the blueprint names specific scripts (`experiments/04_gradient_dynamics.py` through `experiments/09_lorenz_chaos_eval.py`) that don't exist in the repo yet.
- [ ] **Supporting building block for Novelty 4:** an $L_1$-pruning step over KAN edges (node-magnitude-based, per the original paper's Sec. III A 2) doesn't exist anywhere in the code yet — regularization currently only penalizes parameter magnitude (`kan/model.py::regularization_loss`), it doesn't prune nodes/edges afterward.

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

docs/06_codebase_audit.md  # Detailed bug audit: what was found, root causes, fixes applied
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


