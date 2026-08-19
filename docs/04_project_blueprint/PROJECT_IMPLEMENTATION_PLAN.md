# ⚡ KINETIC-KAN: Complete Project Implementation Plan & Technical Execution Guide

## Solver-Aware Neural Dynamics: Interrogating Numerical Integration and Basis Representations in Kolmogorov-Arnold Network ODEs

**Course:** CSE 402 (Numerical Analysis, Simulation & Modeling) | **Institution:** BUET CSE 4-1
**Team Size:** 5 Members | **Project Codename:** `kinetic-kan`
**Base Repository Reference:** [DENG-MIT/KAN-ODEs](https://github.com/DENG-MIT/KAN-ODEs) (_CMAME_, 2024)

---

# 1. Project Overview & Base Repository Audit

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   BASE CODEBASE AUDIT & TECHNICAL REALITY                              │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Official Base Repo: https://github.com/DENG-MIT/KAN-ODEs (MIT Deng Lab)                              │
│ • Base Language Split: Core paper experiments in Julia (DifferentialEquations.jl); minimal PyTorch demo │
│ • Base PyTorch State: Single monolithic script using Gaussian RBF & default autograd ODE solver        │
│ • Critical Gaps in Official Repo:                                                                      │
│   1. No standalone hand-coded numerical ODE solvers (Euler, RK2, RK4, Dopri5).                         │
│   2. No gradient norm trajectory tracking (||∇L||_2) during backpropagation.                           │
│   3. No B-spline (Cox-de Boor), Lagrange, Chebyshev, or Hybrid Basis implementations.                 │
│   4. No stiffness analysis (damped pendulum μ-sweep) or chaotic attractor benchmarks (3D Lorenz).       │
│   5. No symbolic comparison against SINDy or Adjoint vs. Autograd memory/speed profiling.             │
│ • Our Strategy: Reference official repo for ground-truth data format, but build our entire research    │
│   suite from scratch in a modular, publication-ready Python/PyTorch repository.                        │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# 2. Production Directory & File Structure

Create the project repository with the following clean, professional layout:

```text
kinetic-kan/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Multi-OS (Ubuntu, Win, Mac) & Multi-Python (3.10-3.12) CI
│       ├── sciml_benchmarks.yml    # Automated SciML benchmark suite & artifact generator
│       └── code_quality.yml        # Ruff linting, Mypy type-checking & Black formatting
│
├── configs/
│   ├── base_config.yaml            # Default training, optimizer, and seed configs
│   ├── solver_ablation.yaml        # Hyperparameters for solver order sweeps
│   └── stiffness_sweep.yaml        # Damping ratios and solver tolerances
│
├── datasets/
│   ├── __init__.py
│   ├── lotka_volterra.py           # 2D Predator-Prey ground truth generator (RK4)
│   ├── damped_pendulum.py          # Non-linear pendulum with variable damping (μ-stiffness)
│   ├── lorenz.py                   # 3D Chaotic Lorenz attractor dataset
│   └── real_epidemic.py            # Empirical COVID-19 / Dengue time-series loader
│
├── docs/                           # Master project documentation, proposals & blueprints
│   ├── 00_course_guidelines/       # Official syllabus & course notices
│   ├── 01_literature_and_ideas/    # Literature survey, candidate paper audits & notes
│   ├── 02_proposal/                # Proposal decks & written guides
│   ├── 03_presentation/            # Compiled LaTeX Beamer presentation decks & PDFs
│   ├── 04_project_blueprint/       # Master technical deep dive & execution blueprints
│   └── README.md                   # Documentation index & team charter
│
├── models/
│   ├── __init__.py
│   ├── ode_solvers.py              # Standalone Euler, RK2, RK4, and Adaptive integrators
│   ├── kan_layers.py               # Gaussian RBF, B-spline, Lagrange, Chebyshev & Hybrid Basis
│   ├── kan_ode.py                  # Continuous-time KAN-ODE Vector Field & Pipeline
│   └── mlp_ode.py                  # Baseline MLP-Neural ODE for 10x speedup comparison
│
├── experiments/
│   ├── 01_reproduce_baseline.py    # Verify 10x convergence on Lotka-Volterra (KAN vs MLP)
│   ├── 02_solver_ablation.py       # Part 1: Solver order (p=1..5) & step-size (Δt) sweep
│   ├── 03_basis_ablation.py        # Part 1: RBF vs B-spline vs Lagrange vs Chebyshev
│   ├── 04_gradient_dynamics.py     # Part 2: Gradient norm (||∇L||_2) stability logging
│   ├── 05_hybrid_basis_eval.py     # Part 2: Learnable Softmax Hybrid Basis validation
│   ├── 06_stiffness_phase_map.py   # Part 2: Damped pendulum μ-stiffness stability heatmap
│   ├── 07_sindy_benchmark.py       # Part 2: PySINDy vs. KAN-ODE equation discovery under noise
│   ├── 08_adjoint_profiling.py     # Part 2: Continuous Adjoint vs. Direct Autograd profiling
│   └── 09_lorenz_chaos_eval.py     # Part 2: 3D Lorenz attractor geometry & Lyapunov testing
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                  # MSE, NFE tracker, parameter counter, Lipschitz bound
│   ├── symbolic_prune.py           # L1 edge pruning & symbolic equation extractor
│   └── plotting.py                 # Publication-quality phase portraits, streamplots & heatmaps
│
├── tests/
│   ├── __init__.py
│   ├── test_solvers.py             # Numerical order verification O(h^p) & energy conservation
│   ├── test_kan_layers.py          # Autograd gradcheck, partition of unity & NaN immunity
│   ├── test_pipeline.py            # End-to-end forward/backward & overfit sanity check
│   ├── test_adjoint.py             # Continuous Adjoint vs Autograd parity & O(1) memory test
│   └── test_datasets.py            # Deterministic seed reproducibility & physical invariants
│
├── pyproject.toml                  # Tool configs (pytest, ruff, mypy, coverage)
├── requirements.txt                # Pinned dependency manifest
├── LICENSE                         # MIT License
└── README.md                       # Posh GitHub README with badges and quickstart
```

---

# 3. Environment Setup & Dependency Manifest

### 3.1 Pinned `requirements.txt`

```text
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
torchdiffeq>=0.2.3
pysindy>=1.7.5
sympy>=1.12
pyyaml>=6.0
tqdm>=4.65.0
pytest>=7.3.0
```

### 3.2 Installation Commands

```bash
# 1. Create a clean virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# 2. Upgrade pip and install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 4. Phase-by-Phase Technical Implementation Roadmap

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             FOUR-PHASE SPRINT EXECUTION TIMELINE                                               │
├───────────────┬──────────────────────────────────┬─────────────────────────────┬────────────────────────────────┬──────────────┤
│ Phase         │ Milestone Objective              │ Key Deliverables            │ Assigned Team Leads            │ Duration     │
├───────────────┼──────────────────────────────────┼─────────────────────────────┼────────────────────────────────┼──────────────┤
│ **Phase 1**   │ Core Engine & Working Baseline   │ Solvers, KAN Layers, LV,    │ **ALL 5 MEMBERS IN PARALLEL**  │ **Days 1–3** │
│               │                                  │ MLP Baseline, Plotting Core │ M1, M2, M3, M4, M5             │              │
├───────────────┼──────────────────────────────────┼─────────────────────────────┼────────────────────────────────┼──────────────┤
│ **Phase 2**   │ Part 1: Systematic Benchmarking  │ Tables 1 & 2 (Solvers/Bases)│ **ALL 5 MEMBERS IN PARALLEL**  │ **Days 4–6** │
│               │                                  │ Extrapolation, Noise, Plots │ M1, M2, M3, M4, M5             │              │
├───────────────┼──────────────────────────────────┼─────────────────────────────┼────────────────────────────────┼──────────────┤
│ **Phase 3**   │ Part 2: Novel Research Suite     │ Tables 3, 4, 5 & Heatmaps   │ M1 (Hybrid), M2 (Adjoint),     │ **Days 7–9** │
│               │                                  │ (Novelties 1 to 6)          │ M3 (Gradients), M4 (Stiff/SINDY│              │
│               │                                  │                             │ M5 (3D Lorenz Chaos)           │              │
├───────────────┼──────────────────────────────────┼─────────────────────────────┼────────────────────────────────┼──────────────┤
│ **Phase 4**   │ Statistical Rigor & Report/Paper │ LaTeX Draft, N=5 Bars, Repo │ Abrar (M5 Lead) & All Members  │ **Days 10–12 │
└───────────────┴──────────────────────────────────┴─────────────────────────────┴────────────────────────────────┴──────────────┘
```

---

### 🚀 Phase 1: Core Engine & Working Baseline (Days 1–3)

#### Goal: Every team member builds their isolated module on Days 1–2; integrate and verify the 10x speedup baseline on Day 3.

- **Task 1.1: Standalone Numerical ODE Solvers (`models/ode_solvers.py`)** — 👤 **Member 2: Abhishek Roy (2105033)**
  - Implement `ExplicitEulerIntegrator` ($p=1$), `RK2Integrator` (Heun, $p=2$), `RK4Integrator` (Classical, $p=4$), and adaptive Dopri5 wrapper.
  - Implement `tests/test_solvers.py` to verify analytical convergence slope $\log(\text{error}) / \log(h) \approx p$ on $y' = -y$.

- **Task 1.2: KAN Edge Activation Layer (`models/kan_layers.py`)** — 👤 **Member 1: Nawriz Ahmed Turjo (2105032)**
  - Implement `KANEdgeLayer` with: (a) SiLU residual base, (b) Gaussian RBF basis, (c) Cubic B-spline basis (Cox-de Boor), and (d) Chebyshev basis.
  - Implement `tests/test_kan_layers.py` to verify forward pass shapes `[B, Out]` and backward autograd gradient flow without NaNs.

- **Task 1.3: Ground Truth Generator & KAN-ODE Vector Field (`datasets/lotka_volterra.py`, `models/kan_ode.py`)** — 👤 **Member 3: Monjur Hossain Khan (2105043)**
  - High-precision RK4 simulation ($\Delta t = 0.001$, sub-sampled to $N_t = 100$ on $t \in [0, 5]$).
  - Connect $2$-layer continuous-time KAN Vector Field with solver wrapper and loss function ($\mathcal{L} = \text{MSE} + \lambda_1 \|\mathbf{w}\|_1$).

- **Task 1.4: Parameter-Matched MLP Baseline & Metrics Tracker (`models/mlp_ode.py`, `utils/metrics.py`, `datasets/damped_pendulum.py`)** — 👤 **Member 4: Shams Hossain Simanto (2105048)**
  - Build `MLP_ODE` baseline with equal parameter count ($252$ params) for fair $10\times$ speedup comparison.
  - Implement `utils/metrics.py` (MSE, NFE counter, parameter counter, Lipschitz bound calculator).
  - Generate baseline non-linear Damped Pendulum dataset.

- **Task 1.5: Visualization Engine & Extended Dataset Loaders (`utils/plotting.py`, `datasets/lorenz.py`, `datasets/real_epidemic.py`)** — 👤 **Member 5: Abrar Jahin (2105055)**
  - Build `utils/plotting.py` (publication-quality phase portrait renderers, loss trajectory plotters, vector field streamline plots).
  - Build 3D Chaotic Lorenz attractor dataset generator and real epidemiological infection data loader.

- **Day 3 Milestone Check (All Hands):** — 👤 **Lead: Member 3 (Monjur) & All Members**
  - Run `python experiments/01_reproduce_baseline.py` $\to$ verify KAN-ODE reaches $\text{MSE} < 10^{-4}$ in $< 2000$ epochs ($< 1$ min on CPU), outperforming MLP-ODE by $10\times$. Phase portraits exported by M5, metrics verified by M4.

---

### 🔬 Phase 2: Part 1 Systematic Benchmarking (Days 4–6)

#### Goal: Execute all 5 core benchmarking workstreams in parallel to cover the entire CSE 402 syllabus.

- **Task 2.1: Solver Order Ablation & Discretization Step-Size Sweep (`experiments/02_solver_ablation.py`)** — 👤 **Member 2: Abhishek Roy (2105033)**
  - Sweep Forward Euler ($p=1$), Heun RK2 ($p=2$), Classical RK4 ($p=4$), and Adaptive Dopri5 across $\Delta t \in \{0.20, 0.10, 0.05, 0.01\}$.
  - Measure: Final Train MSE, Extrapolation MSE, NFE per step, and wall-clock speed $\to$ Generate **Table 1**.

- **Task 2.2: Basis Function Representation Ablation (`experiments/03_basis_ablation.py`)** — 👤 **Member 1: Nawriz Ahmed Turjo (2105032)**
  - Train identical KAN-ODE models with: (1) Gaussian RBF ($G=5$), (2) Cubic B-Splines ($G=5, k=3$), (3) Lagrange ($N=4$), and (4) Chebyshev ($N=4$).
  - Measure: Parameter count, convergence epochs to $10^{-4}$ MSE, and boundary stability $\to$ Generate **Table 2**.

- **Task 2.3: Extrapolation Horizon & Convergence Rate Analysis** — 👤 **Member 3: Monjur Hossain Khan (2105043)**
  - Evaluate long-term trajectory extrapolation ($t \in [5, 15]$) on Lotka-Volterra across all models.
  - Benchmark KAN-ODE vs. MLP-ODE loss decay slopes under strictly identical epoch budgets.

- **Task 2.4: Noise Robustness Sweep & Damped Pendulum Benchmark** — 👤 **Member 4: Shams Hossain Simanto (2105048)**
  - Sweep Gaussian observational noise $\sigma \in \{0.00, 0.01, 0.05, 0.10\}$ on training trajectories.
  - Run initial baseline KAN-ODE training on the non-linear Damped Pendulum dataset to verify physical generalizability.

- **Task 2.5: Automated Benchmark Collation, Multi-Panel Figures & Report Draft** — 👤 **Member 5: Abrar Jahin (2105055)**
  - Build automated result collation pipeline: save `results/tables/01_baseline.csv`, `02_solvers.csv`, `03_bases.csv`.
  - Render multi-panel publication figures (Phase portraits with true vs predicted trajectories, error vs $\Delta t$ log-log curves).
  - Author initial Phase 1–2 draft section for course report.

---

### 🌟 Phase 3: Part 2 Novel Research Contributions (Days 7–9)

#### Goal: Implement the 6 high-impact novelties that elevate the work to a publishable standard.

- **Task 3.1: Gradient Norm Dynamics (`experiments/04_gradient_dynamics.py`) [Novelty 1]** — 👤 **Member 3: Monjur Hossain Khan (2105043)**
  - In the training loop, log $\|\nabla_\theta \mathcal{L}\|_2 = \sqrt{\sum_i \|\nabla_{\theta_i} \mathcal{L}\|_2^2}$ at every single epoch.
  - Plot Gradient Norm Trajectory across Euler, RK2, and RK4. Prove lower-order solvers inject high-frequency gradient noise.

- **Task 3.2: Learnable Softmax Hybrid Basis Layer (`experiments/05_hybrid_basis_eval.py`) [Novelty 2]** — 👤 **Member 1: Nawriz Ahmed Turjo (2105032)**
  - Train the learnable Hybrid Basis Layer ($\alpha \text{Spline} + \beta \text{RBF}$) on Lotka-Volterra and Damped Pendulum.
  - Track evolution of blend weights $\alpha(t)$ and $\beta(t)$ during training; prove faster convergence over pure bases.

- **Task 3.3: Stiffness-Solver Stability Phase Map (`experiments/06_stiffness_phase_map.py`) [Novelty 3]** — 👤 **Member 4: Shams Hossain Simanto (2105048)**
  - Sweep damping ratio $\mu \in \{0.1, 0.5, 1.0, 2.0, 5.0, 8.0\}$ on Damped Pendulum across Euler, RK2, RK4 and $\Delta t$.
  - Output **Table 5** and a 2D colored stability matrix heatmap (Stable vs. Exploded).

- **Task 3.4: SINDy Baseline Comparison under Noise (`experiments/07_sindy_benchmark.py`) [Novelty 4]** — 👤 **Member 4: Shams Hossain Simanto (2105048)**
  - Fit `pysindy.SINDy` with polynomial feature library on Lotka-Volterra under noise $\sigma \in \{0.00, 0.01, 0.05, 0.10\}$.
  - Compare SINDy's recovered equations against KAN-ODE $L_1$ pruned edge formulas $\to$ **Table 3**.

- **Task 3.5: Adjoint vs. Autograd Memory & Speed Profiling (`experiments/08_adjoint_profiling.py`) [Novelty 5]** — 👤 **Member 2: Abhishek Roy (2105033)**
  - Benchmark `torchdiffeq.odeint_adjoint` ($O(1)$ memory) vs. unrolled `loss.backward()` ($O(N_t)$ memory).
  - Profile peak GPU VRAM (MB), wall-clock speed per 1000 epochs, and gradient accuracy across $d \in \{2, 8\}$ $\to$ **Table 4**.

- **Task 3.6: 3D Chaotic Lorenz Attractor & Empirical Fit (`experiments/09_lorenz_chaos_eval.py`) [Novelty 6]** — 👤 **Member 5: Abrar Jahin (2105055)**
  - Train 3D KAN-ODE ($\mathbf{u} = [x, y, z]^T$) on the Lorenz attractor ($\sigma=10, \rho=28, \beta=8/3$).
  - Render 3D butterfly phase-space trajectory comparing true vs. predicted strange attractor geometry; fit real epidemiological data.

---

### 📊 Phase 4: Statistical Verification & Final Synthesis (Days 10–12)

- **Task 4.1: Multi-Seed Statistical Profiling ($N=5$)** — 👤 **Member 3 (Monjur Lead) & All Team Members**
  - Run all core benchmark scripts over $N = 5$ random seeds (`[42, 1337, 2024, 7, 999]`). Calculate $\text{Mean} \pm \text{Std Dev}$ for all tables.

- **Task 4.2: Publication Figure Generation (300+ DPI)** — 👤 **Member 5: Abrar Jahin (2105055)**
  - Export 300+ DPI publication plots: Phase portraits with streamlines, Gradient Norm curves, Stiffness heatmap, SINDy error bars, and 3D Lorenz attractor.

- **Task 4.3: LaTeX Paper & Report Compilation** — 👤 **Member 5: Abrar Jahin (2105055 Lead) & All Team Members**
  - Compile the complete findings, multi-seed tables, and figures into the final project report / conference paper draft.

---

# 5. Sequential File-by-File Build Roadmap & Member Responsibilities

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SEQUENTIAL DEVELOPMENT DEPENDENCY GRAPH                              │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                        │
│  STAGE 0: PARALLEL FOUNDATIONS (Days 1–2) — ALL 5 MEMBERS WORK SIMULTANEOUSLY                          │
│  ├── Member 1: models/kan_layers.py & tests/test_kan_layers.py ────────┐                               │
│  ├── Member 2: models/ode_solvers.py & tests/test_solvers.py ──────────┼────────┐                      │
│  ├── Member 3: datasets/lotka_volterra.py & models/kan_ode.py ─────────┼────────┼────────┐             │
│  ├── Member 4: models/mlp_ode.py, utils/metrics.py, pendulum.py ───────┼────────┼────────┼──────┐      │
│  └── Member 5: utils/plotting.py, datasets/lorenz.py, epidemic.py ─────┼────────┼────────┼──────┼────┐ │
│                                                                        │        │        │      │    │ │
│  STAGE 1: INTEGRATION & BASELINE MILESTONE (Day 3)                     │        │        │      │    │ │
│  └── ALL: experiments/01_reproduce_baseline.py (10x speedup verified) ◄┴────────┴────────┴──────┴────┘ │
│                                                                                                        │
│  STAGE 2: PARALLEL BENCHMARKS (Days 4–6)                                                               │
│  ├── Member 1: experiments/03_basis_ablation.py (Table 2: Basis Functions)                             │
│  ├── Member 2: experiments/02_solver_ablation.py (Table 1: Solvers & Step-Sizes)                       │
│  ├── Member 3: Extrapolation Horizon & NFE Efficiency Benchmarking                                     │
│  ├── Member 4: Observational Noise Sweeps (σ ≤ 0.10) & Damped Pendulum Baseline                        │
│  └── Member 5: Automated Table Collation (CSV), Multi-Panel Figures & Report Draft                     │
│                                                                                                        │
│  STAGE 3: NOVEL RESEARCH CONTRIBUTIONS (Days 7–9)                                                      │
│  ├── Member 1: experiments/05_hybrid_basis_eval.py (Learnable Hybrid Basis)                            │
│  ├── Member 2: experiments/08_adjoint_profiling.py (Table 4: Adjoint vs Autograd)                      │
│  ├── Member 3: experiments/04_gradient_dynamics.py (Gradient Norm Tracking)                            │
│  ├── Member 4: experiments/06_stiffness_phase_map.py (Table 5) & exp/07_sindy_benchmark.py (Table 3)   │
│  └── Member 5: experiments/09_lorenz_chaos_eval.py (3D Chaotic Attractor & Real Data Fit)              │
│                                                                                                        │
│  STAGE 4: SYNTHESIS & PAPER ASSEMBLY (Days 10–12)                                                      │
│  └── ALL MEMBERS: Multi-Seed Data Collation (N=5), 300 DPI Figures & Final LaTeX Paper Assembly        │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 👤 Member 1 (2105032 - Nawriz Ahmed Turjo | KAN Architecture Lead)

**Core Responsibility:** KAN Edge Layers, Basis Interpolants & Novel Learnable Hybrid Basis

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MEMBER 1 SEQUENTIAL BUILD PIPELINE (2105032 - NAWRIZ AHMED TURJO)                                     │
├───────┬──────────────────────────────────┬─────────────────────────────────────────────────────────────┤
│ Step  │ File to Build                    │ Technical Purpose & Implementation Details                  │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1.1   │ `models/kan_layers.py`           │ Implement `KANEdgeLayer` with: (a) base SiLU residual,      │
│       │                                  │ (b) Gaussian RBF, (c) Cubic B-spline, (d) Chebyshev basis. │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1.2   │ `tests/test_kan_layers.py`       │ Verify forward pass shapes [B, Out] and backward autograd   │
│       │                                  │ gradient flow across all basis types without NaN/Inf.       │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1.3   │ `experiments/03_basis_ablation.py`│ Phase 2: Compare RBF vs B-spline vs Lagrange vs Chebyshev on│
│       │                                  │ Lotka-Volterra; log epochs to 10⁻⁴ MSE (Table 2).          │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1.4   │ `experiments/05_hybrid_basis_eval.py`│ Phase 3: Train Learnable Hybrid Basis; track blend weights  │
│       │                                  │ α(t), β(t) and verify convergence speedup over pure bases.  │
└───────┴──────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

### 👤 Member 2 (2105033 - Abhishek Roy | ODE Solver Lead)

**Core Responsibility:** Numerical ODE Solvers, Discretization Stability & Adjoint Sensitivity

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MEMBER 2 SEQUENTIAL BUILD PIPELINE (2105033 - ABHISHEK ROY)                                           │
├───────┬──────────────────────────────────┬─────────────────────────────────────────────────────────────┤
│ Step  │ File to Build                    │ Technical Purpose & Implementation Details                  │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2.1   │ `models/ode_solvers.py`          │ Code `ExplicitEulerIntegrator` (p=1), `RK2Integrator` (p=2),│
│       │                                  │ `RK4Integrator` (p=4), and adaptive Dormand-Prince wrapper. │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2.2   │ `tests/test_solvers.py`          │ Unit test solvers on analytical ODE (y'=-y) to verify that  │
│       │                                  │ log(error)/log(h) convergence slope matches order p (1, 2, 4)│
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2.3   │ `experiments/02_solver_ablation.py`│ Phase 2: Solver order (p=1..5) & step-size (Δt) sweep;      │
│       │                                  │ generate Table 1 metrics (Train/Extrap MSE, NFE, Speed).   │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2.4   │ `experiments/08_adjoint_profiling.py`│ Phase 3: Benchmark `odeint_adjoint` (O(1) memory) vs direct │
│       │                                  │ autograd backprop (O(Nt) memory) across d∈{2,8} (Table 4). │
└───────┴──────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

### 👤 Member 3 (2105043 - Monjur Hossain Khan | Optimization & Gradients Lead)

**Core Responsibility:** SciML Integration, Training Pipeline & Gradient Norm Dynamics

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MEMBER 3 SEQUENTIAL BUILD PIPELINE (2105043 - MONJUR HOSSAIN KHAN)                                    │
├───────┬──────────────────────────────────┬─────────────────────────────────────────────────────────────┤
│ Step  │ File to Build                    │ Technical Purpose & Implementation Details                  │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3.1   │ `datasets/lotka_volterra.py` &   │ Phase 1: Ground truth generator (high-precision RK4, Nt=100)│
│       │ `models/kan_ode.py`              │ and continuous-time KAN-ODE Vector Field wrapper.           │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3.2   │ `experiments/01_reproduce_baseline.py`│ Phase 1: Verify base paper claim: KAN-ODE reaches          │
│       │                                  │ MSE ≤ 10⁻⁴ in 10⁴ epochs vs MLP-ODE (10x speedup).          │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3.3   │ Extrapolation & NFE Analysis     │ Phase 2: Evaluate extrapolation stability (t ∈ [5, 15]) and │
│       │                                  │ loss decay slopes under equal compute budgets.              │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3.4   │ `experiments/04_gradient_dynamics.py`│ Phase 3: Log ||∇L||_2 at every epoch across Euler, RK2, RK4;│
│       │                                  │ prove low-order solvers inject high-frequency gradient noise│
└───────┴──────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

### 👤 Member 4 (2105048 - Shams Hossain Simanto | Stability & SINDy Lead)

**Core Responsibility:** Baseline Models, Metrics, Non-linear Stability & SINDy Benchmark

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MEMBER 4 SEQUENTIAL BUILD PIPELINE (2105048 - SHAMS HOSSAIN SIMANTO)                                  │
├───────┬──────────────────────────────────┬─────────────────────────────────────────────────────────────┤
│ Step  │ File to Build                    │ Technical Purpose & Implementation Details                  │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 4.1   │ `models/mlp_ode.py`,             │ Phase 1: Parameter-matched MLP-ODE baseline (252 params),   │
│       │ `utils/metrics.py`, `pendulum.py`│ metrics module (MSE, NFE, Lipschitz), and Damped Pendulum.  │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 4.2   │ Noise Sweeps & Pendulum Baseline │ Phase 2: Evaluate observational noise (σ ≤ 0.10) on LV and  │
│       │                                  │ run baseline KAN-ODE on Damped Pendulum dataset.            │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 4.3   │ `experiments/06_stiffness_phase_map.py`│ Phase 3: Sweep damping μ ∈ [0.1, 8.0] across Euler/RK2/RK4; │
│       │                                  │ generate 2D Stiffness vs Δt stability heatmap (Table 5).    │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 4.4   │ `experiments/07_sindy_benchmark.py`│ Phase 3: Fit PySINDy on noisy trajectories (σ∈{0, 0.05, 0.1});│
│       │                                  │ benchmark recovery accuracy against KAN-ODE (Table 3).      │
└───────┴──────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

### 👤 Member 5 (2105055 - Abrar Jahin | Chaotic Dynamics, Visualization & Paper Lead)

**Core Responsibility:** Visualization Engine, 3D Chaotic Attractor, Real Data & LaTeX Synthesis

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MEMBER 5 SEQUENTIAL BUILD PIPELINE (2105055 - ABRAR JAHIN)                                            │
├───────┬──────────────────────────────────┬─────────────────────────────────────────────────────────────┤
│ Step  │ File to Build                    │ Technical Purpose & Implementation Details                  │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 5.1   │ `utils/plotting.py`,             │ Phase 1: Publication plot engine (phase portraits, streams),│
│       │ `datasets/lorenz.py`, `epidemic.py`│ 3D Chaotic Lorenz attractor and real epidemic data loader.  │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 5.2   │ Automated Collation & Figures    │ Phase 2: Build CSV results table exporter, render multi-    │
│       │                                  │ panel benchmark figures, and draft Phase 1–2 report section.│
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 5.3   │ `experiments/09_lorenz_chaos_eval.py`│ Phase 3: Train 3D KAN-ODE on Lorenz attractor; evaluate     │
│       │                                  │ strange attractor butterfly geometry & real data fitting.   │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 5.4   │ `paper_draft/` & LaTeX Synthesis │ Phase 4: Aggregate multi-seed (N=5) tables & 300 DPI figures│
│       │                                  │ into final academic paper / course project report.          │
└───────┴──────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

# 6. Parallel Execution Architecture & Zero-Conflict Modularity Protocol

To enable all 5 team members to work simultaneously with **zero waiting time** and **zero Git merge conflicts**, the repository enforces a strict decoupling and modularity architecture.

````
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               PARALLEL EXECUTION & DECOUPLING ARCHITECTURE                             │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                        │
│    DAY 1 (INTERFACE HANDSHAKE): Freeze function signatures on 'main' branch                            │
│    ├── ode_solvers.py   ->  def integrator(func, y0, t_span) -> Tensor [B, Nt, D]                      │
│    ├── kan_layers.py    ->  class KANEdgeLayer(in_features, out_features, basis_type)                  │
│    ├── lotka_volterra.py->  def generate_lotka_volterra(steps=100, noise=0.0) -> (t_span, traj)        │
│    ├── mlp_ode.py       ->  class MLP_ODE(in_features, hidden_features, out_features)                  │
│    └── plotting.py      ->  def plot_phase_portrait(true_traj, pred_traj, title, save_path)            │
│                                                                                                        │
│                                           │                                                            │
│         ┌────────────────────┬────────────┼────────────────────┬───────────────────┐                   │
│         ▼                    ▼            ▼                    ▼                   ▼                   │
│  [feat/m1-kan]        [feat/m2-solvers]  [feat/m3-pipeline]   [feat/m4-stability] [feat/m5-chaos]      │
│  Member 1 (Turjo)     Member 2 (Abhishek)Member 3 (Monjur)    Member 4 (Simanto)  Member 5 (Abrar)     │
│  kan_layers.py        ode_solvers.py     kan_ode.py           mlp_ode.py          plotting.py          │
│  test_kan_layers.py   test_solvers.py    test_pipeline.py     metrics.py          datasets/lorenz.py   │
│  exp/03, exp/05       exp/02, exp/08     exp/01, exp/04       pendulum.py, exp/06 datasets/epidemic.py │
│                                                               exp/07              exp/09, LaTeX Draft  │
│                                                                                                        │
│    DAY 10 (SYNTHESIS): Automated CI merges all PRs cleanly into 'main' with zero collision             │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
---

### 🌿 6.1 GitHub Branching Strategy & Workflow Rules

Every member develops exclusively within their isolated feature branch:

| Member | Student ID & Name | Dedicated Git Branch | Primary Files Owned & Developed |
| :---: | :--- | :--- | :--- |
| **M1** | **2105032** — Nawriz Ahmed Turjo | `feat/m1-kan-architecture` | `models/kan_layers.py`, `tests/test_kan_layers.py`, `experiments/03_*.py`, `experiments/05_*.py` |
| **M2** | **2105033** — Abhishek Roy | `feat/m2-numerical-solvers` | `models/ode_solvers.py`, `tests/test_solvers.py`, `experiments/02_*.py`, `experiments/08_*.py` |
| **M3** | **2105043** — Monjur Hossain Khan | `feat/m3-sciml-pipeline` | `datasets/lotka_volterra.py`, `models/kan_ode.py`, `tests/test_pipeline.py`, `experiments/01_*.py`, `experiments/04_*.py` |
| **M4** | **2105048** — Shams Hossain Simanto | `feat/m4-stability-sindy` | `models/mlp_ode.py`, `utils/metrics.py`, `datasets/damped_pendulum.py`, `experiments/06_*.py`, `experiments/07_*.py` |
| **M5** | **2105055** — Abrar Jahin | `feat/m5-chaos-visuals` | `utils/plotting.py`, `datasets/lorenz.py`, `datasets/real_epidemic.py`, `experiments/09_*.py`, `paper_draft/` |

#### Git Execution Protocol Cheatsheet:

```bash
# 1. Clone repository and create personal feature branch
git clone https://github.com/NawrizTurjo/kinetic-kan.git
cd kinetic-kan
git checkout -b feat/m1-kan-architecture   # (Use your respective branch name)

# 2. Daily development cycle (Atomic commits)
git add models/kan_layers.py tests/test_kan_layers.py
git commit -m "feat(kan): implement cubic B-spline Cox-de Boor recursion"

# 3. Keep branch updated with main without conflicts
git checkout main && git pull origin main
git checkout feat/m1-kan-architecture
git rebase main

# 4. Push and open Pull Request for automated CI verification
git push origin feat/m1-kan-architecture
````

---

### 🛡️ 6.2 Zero-Conflict Modularity for Potentially Shared Files

When multiple developers collaborate on a machine learning project, shared files often cause merge conflicts. We prevent this using 5 architectural isolation rules:

#### Rule 1: Decentralized PyTest Modules (`tests/`)

- **Problem:** If everyone writes tests in a single `test_all.py`, simultaneous edits cause constant merge conflicts.
- **Solution:** Tests are 100% decentralized. Each member owns their dedicated test file (`test_kan_layers.py`, `test_solvers.py`, etc.).
- **Shared Fixtures:** Common fixtures (e.g. `device`, `seed_everything`) are placed in `tests/conftest.py` on Day 1 and kept strictly read-only.

#### Rule 2: Frozen CI/CD & Configuration Files (`.github/` & `pyproject.toml`)

- **Problem:** Multiple members editing `ci.yml` or `requirements.txt` simultaneously.
- **Solution:** CI workflows and `pyproject.toml` are committed once on `main` at project kickoff. CI dynamically discovers all test files via globbing (`pytest tests/test_*.py`), so members never touch `.github/` workflows.

#### Rule 3: Isolated Experiment Configurations (`configs/`)

- Instead of one monolithic `config.yaml`, hyperparameters are partitioned into member-specific config files:
  - `configs/m1_basis_config.yaml` (Spline/RBF knot grids)
  - `configs/m2_solver_config.yaml` (Step-sizes $\Delta t$ and tolerance thresholds)
  - `configs/m4_stiffness_config.yaml` (Damping ratios $\mu$ and SINDy polynomial libraries)
  - `configs/m5_lorenz_config.yaml` (Chaotic parameter regimes)

#### Rule 4: Decoupled Utility Modules (`utils/`)

- Utility functions are segregated by domain to prevent collisions:
  - `utils/metrics.py` (Owned by M4: MSE, NFE, Lipschitz calculation)
  - `utils/symbolic_prune.py` (Owned by M4: $L_1$ sparsity pruning & LaTeX equation printer)
  - `utils/plotting.py` (Owned by M5: Matplotlib/Seaborn visualization engine)

#### Rule 5: Non-Overlapping Result Artifacts (`results/`)

- All script outputs are saved with standardized prefix naming:
  - Tables: `results/tables/01_baseline_rep.csv`, `results/tables/02_solver_ablation.csv`, etc.
  - Plots: `results/figures/01_loss_curve.png`, `results/figures/06_stiffness_heatmap.png`, etc.
- Zero file overwrites or name collisions occur when branches are merged.

---

# 7. Production-Grade PyTest Test Suite Architecture (`tests/`)

To ensure absolute numerical correctness, gradient stability, and research reproducibility, the repository includes a multi-layered PyTest suite under `tests/`.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PRODUCTION TEST SUITE ARCHITECTURE                                   │
├───────────────────┬──────────────────────────────────┬─────────────────────────────────────────────────┤
│ Test Module       │ Test Scope & Owner               │ Key Assertions & Tolerances                     │
├───────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────┤
│ `test_solvers.py` │ Integrators (Member 2)           │ Convergence slope error < 5%; Energy drift < 1% │
│ `test_kan_layers` │ Spline/RBF/Hybrid (Member 1)     │ `gradcheck` eps=1e-6; Partition of unity = 1.0  │
│ `test_pipeline.py`│ Vector Field & Flow (Member 3)   │ 100% gradient existence; Overfit MSE < 1e-5     │
│ `test_adjoint.py` │ Adjoint Sensitivity (Member 2)   │ Adjoint vs Autograd grad relative error < 1e-3  │
│ `test_datasets.py`│ Chaotic & Physical Systems (M4/M5)│ Bit-exact seed reproducibility; Invariants hold │
└───────────────────┴──────────────────────────────────┴─────────────────────────────────────────────────┘
```

---

### 🧪 Test Module 1: Numerical Solver Verification (`tests/test_solvers.py`)

```python
import pytest
import torch
import numpy as np
from models.ode_solvers import ExplicitEulerIntegrator, RK2Integrator, RK4Integrator

class LinearDecayODE:
    """Analytical benchmark: dy/dt = -y => y(t) = y0 * exp(-t)"""
    def __call__(self, t, y):
        return -y

def test_solver_convergence_orders():
    """Verify empirical convergence order matches theoretical order O(h^p)."""
    ode = LinearDecayODE()
    y0 = torch.tensor([[1.0]], dtype=torch.float64)
    t_end = 1.0
    step_sizes = [0.1, 0.05, 0.025, 0.0125]

    solvers = {
        'euler': (ExplicitEulerIntegrator(), 1.0),
        'rk2': (RK2Integrator(), 2.0),
        'rk4': (RK4Integrator(), 4.0)
    }

    for name, (solver, expected_p) in solvers.items():
        errors = []
        for h in step_sizes:
            t_span = torch.arange(0, t_end + h, h, dtype=torch.float64)
            pred = solver(ode, y0, t_span)[0, -1, 0].item()
            true_val = np.exp(-t_end)
            errors.append(abs(pred - true_val))

        # Linear regression on log-log curve: log(error) vs log(h)
        slope, _ = np.polyfit(np.log(step_sizes), np.log(errors), 1)
        assert abs(slope - expected_p) < 0.15, f"{name} expected slope ~{expected_p}, got {slope:.3f}"

def test_solver_batch_and_dim_invariance():
    """Verify arbitrary batch sizes [B] and state dimensions [D] are preserved."""
    solver = RK4Integrator()
    ode = lambda t, y: -0.5 * y

    for batch_size in [1, 8, 32]:
        for dim in [2, 4, 8]:
            y0 = torch.randn(batch_size, dim)
            t_span = torch.linspace(0, 1.0, 50)
            traj = solver(ode, y0, t_span)
            assert traj.shape == (batch_size, 50, dim), f"Failed for B={batch_size}, D={dim}"
```

---

### 🧪 Test Module 2: KAN Layer & Hybrid Basis Verification (`tests/test_kan_layers.py`)

```python
import pytest
import torch
from torch.autograd import gradcheck
from models.kan_layers import KANEdgeLayer

def test_kan_autograd_gradcheck():
    """Verify analytical autograd matches numerical finite-difference gradients."""
    layer = KANEdgeLayer(in_features=2, out_features=3, num_knots=5, basis_type='hybrid').to(torch.float64)
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)

    # Run PyTorch gradcheck
    assert gradcheck(layer, x, eps=1e-6, atol=1e-4), "Autograd gradcheck failed for KANEdgeLayer"

def test_hybrid_basis_blend_constraints():
    """Verify softmax gating blend weights strictly sum to 1.0."""
    layer = KANEdgeLayer(in_features=4, out_features=4, basis_type='hybrid')
    weights = torch.softmax(layer.blend_w, dim=0)

    assert torch.allclose(weights.sum(), torch.tensor(1.0)), "Blend weights must sum to 1.0"
    assert (weights >= 0.0).all(), "Blend weights must be non-negative"

def test_extreme_input_nan_immunity():
    """Verify KAN activations do not produce NaN or Inf under extreme domain inputs."""
    layer = KANEdgeLayer(in_features=2, out_features=2, basis_type='hybrid')
    extreme_x = torch.tensor([[-50.0, 50.0], [-100.0, 100.0]], dtype=torch.float32)
    out = layer(extreme_x)

    assert not torch.isnan(out).any(), "KAN output contains NaN under extreme inputs"
    assert not torch.isinf(out).any(), "KAN output contains Inf under extreme inputs"
```

---

### 🧪 Test Module 3: Adjoint Sensitivity & Memory Parity (`tests/test_adjoint.py`)

```python
import pytest
import torch
from models.kan_layers import KANEdgeLayer
from models.kan_ode import KAN_ODE_VectorField, KAN_ODE_Pipeline
from torchdiffeq import odeint, odeint_adjoint

def test_adjoint_vs_autograd_gradient_parity():
    """Verify that continuous adjoint gradients match unrolled discrete autograd gradients."""
    vfield = KAN_ODE_VectorField(state_dim=2, hidden_dim=4, num_knots=5, basis_type='rbf')
    u0 = torch.tensor([[1.0, 1.0]], requires_grad=True)
    t_span = torch.linspace(0, 1.0, 20)

    # 1. Discrete Autograd Gradient
    traj_auto = odeint(vfield, u0, t_span, method='rk4')
    loss_auto = traj_auto.sum()
    grads_auto = torch.autograd.grad(loss_auto, vfield.parameters(), retain_graph=True)

    # 2. Continuous Adjoint Gradient
    traj_adj = odeint_adjoint(vfield, u0, t_span, method='rk4')
    loss_adj = traj_adj.sum()
    grads_adj = torch.autograd.grad(loss_adj, vfield.parameters())

    # Relative gradient error check
    for g_auto, g_adj in zip(grads_auto, grads_adj):
        rel_error = (g_auto - g_adj).norm() / (g_auto.norm() + 1e-8)
        assert rel_error < 5e-3, f"Adjoint gradient error too high: {rel_error:.5f}"
```

---

# 8. GitHub Actions CI/CD Automation Matrix & Workflows

Automated Continuous Integration ensures that any push or pull request is verified across multiple operating systems and Python versions.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    GITHUB ACTIONS CI WORKFLOW MATRIX                                   │
├───────────────────┬──────────────────────────────────┬─────────────────────────────────────────────────┤
│ Workflow File     │ Trigger Events                   │ Operating Systems & Environments                │
├───────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────┤
│ `ci.yml`          │ `push`, `pull_request` on `main` │ Ubuntu 22.04, Windows 2022, macOS 13            │
│                   │                                  │ Python 3.10, 3.11, 3.12                         │
├───────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────┤
│ `sciml_bench.yml` │ `workflow_dispatch`, `release`   │ Ubuntu 22.04 (Runs full SciML benchmark suite)  │
├───────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────┤
│ `code_quality.yml`│ `push`, `pull_request`           │ Ruff Linter, Black Formatter, Mypy Typecheck   │
└───────────────────┴──────────────────────────────────┴─────────────────────────────────────────────────┘
```

---

### ⚙️ Workflow 1: Multi-OS Multi-Python CI (`.github/workflows/ci.yml`)

```yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    name: PyTest (${{ matrix.os }} - Py${{ matrix.python-version }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: "pip"

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov ruff mypy

      - name: Lint with Ruff
        run: ruff check .

      - name: Static Type Check with Mypy
        run: mypy models/ datasets/ utils/ --ignore-missing-imports

      - name: Execute PyTest Suite with Coverage
        run: |
          pytest tests/ -v --cov=models --cov=datasets --cov=utils --cov-report=xml --cov-fail-under=85

      - name: Upload Code Coverage Artifact
        uses: actions/upload-artifact@v4
        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.10'
        with:
          name: coverage-report
          path: coverage.xml
```

---

### ⚙️ Workflow 2: Automated SciML Benchmark Regression (`.github/workflows/sciml_benchmarks.yml`)

```yaml
name: SciML Benchmarks & Artifact Generation

on:
  workflow_dispatch:
  release:
    types: [published]

jobs:
  benchmark:
    name: Run Benchmark Suite & Export Figures
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install Dependencies
        run: |
          pip install -r requirements.txt

      - name: Run Baseline Reproduction
        run: python experiments/01_reproduce_baseline.py

      - name: Run Solver & Basis Ablations
        run: |
          python experiments/02_solver_ablation.py
          python experiments/03_basis_ablation.py
          python experiments/04_gradient_dynamics.py
          python experiments/05_hybrid_basis_eval.py
          python experiments/06_stiffness_phase_map.py
          python experiments/07_sindy_benchmark.py

      - name: Upload Benchmark Tables & Plots
        uses: actions/upload-artifact@v4
        with:
          name: sciml-benchmark-results
          path: |
            results/tables/*.csv
            results/figures/*.png
```

---

### ⚙️ Configuration Manifest (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-ra -q --strict-markers"

[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "F", "I", "W"]
ignore = ["E501"]

[tool.coverage.run]
source = ["models", "datasets", "utils"]
omit = ["tests/*", "experiments/*"]
```

---

# 9. Verification & Validation Protocol

Before declaring any experiment complete, verify against these quantitative acceptance criteria:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    QUANTITATIVE ACCEPTANCE CRITERIA                                    │
├────────────────────────────────────────────────┬───────────────────────────────────────────────────────┤
│ Test Milestone                                 │ Expected Acceptance Metric / Pass Threshold           │
├────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ 1. ODE Solvers Analytical Test (`y' = -y`)     │ Euler slope ≈ 1.0, RK2 slope ≈ 2.0, RK4 slope ≈ 4.0   │
│ 2. Lotka-Volterra Baseline Training            │ Train MSE ≤ 5.0 × 10⁻⁵ in ≤ 2000 epochs (Adam lr=0.01)│
│ 3. 10x MLP-ODE Comparison                      │ KAN-ODE reaches 10⁻⁴ MSE in ≤ 1000 epochs (MLP > 5000)│
│ 4. Solver Truncation Scaling                   │ RK4 MSE at Δt=0.05 is ≥ 10x lower than Euler at Δt=0.01│
│ 5. Noise Robustness                            │ Recovers correct Lotka-Volterra law at σ = 0.05       │
│ 6. Multi-Seed Stability                        │ Relative standard deviation (Std/Mean) < 15% (N=5)    │
│ 7. Memory Footprint                            │ Total VRAM consumption < 500 MB across all trials     │
│ 8. CI Code Coverage                            │ PyTest test coverage ≥ 85% on all core modules        │
└────────────────────────────────────────────────┴───────────────────────────────────────────────────────┘
```

---

# 10. GitHub Repository & Publication Readiness Checklist

- [ ] Repository initialized as **`kinetic-kan`** (or **`kronos-ode`**) with MIT License.
- [ ] Multi-OS GitHub Actions CI workflow (`.github/workflows/ci.yml`) active and green.
- [ ] Professional `README.md` containing:
  - High-impact header badge matrix (CI Status, Python, PyTorch, SciML, License).
  - 1-minute quickstart snippet with CLI commands.
  - Interactive ASCII / Markdown results comparison tables.
  - Embedded high-resolution phase portraits and streamplots.
- [ ] `pyproject.toml` and `requirements.txt` configured with pinned versions.
- [ ] PyTest suite (`pytest tests/`) passing with 100% green status and $\ge 85\%$ coverage.
- [ ] Random seeds fixed for 100% deterministic reproducibility across all experiments.
- [ ] Clean, atomic git commit history mapping to individual WBS milestones.

---

_Ready to begin execution. Proceed to Phase 1 by setting up the repository and testing Module 1 & Module 2._
