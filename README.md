# ⚡ KINETIC-KAN: Solver-Aware Neural Dynamics

> **Interrogating Numerical Integration, Basis Representations, and Adjoint Sensitivities in Kolmogorov-Arnold Network ODEs**
> *Course Project for CSE 402: Numerical Analysis, Simulation & Modeling | Department of CSE, BUET (4-1)*

[![CI Pipeline](https://img.shields.io/badge/CI-Passing-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-143%20passed-brightgreen.svg)]()
[![Python 3.10+](<https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg>)]()
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Phase 2 Status](https://img.shields.io/badge/Phase%202-Complete%20(Lorenz%20dropped)-success.svg)]()
[![Target Venue](<https://img.shields.io/badge/Target-CMAME%20%2F%20NeurIPS%20SciML-purple.svg>)]()

---

## 📌 Executive Summary

**KINETIC-KAN** investigates the interplay between classical numerical analysis subroutines and modern Scientific Machine Learning (SciML) architectures. Standard Neural Ordinary Differential Equations (Neural ODEs) parameterize continuous-time dynamics $\dot{\mathbf{u}} = \mathbf{f}_\theta(\mathbf{u})$ using Multi-Layer Perceptrons (MLPs) as uninterpretable black boxes. The recent MIT landmark (*Koenig, Kim & Deng, CMAME 2024*) introduced **KAN-ODEs**, substituting fixed MLP activations with learnable univariate basis functions on network edges via the Kolmogorov-Arnold representation theorem.

While KAN-ODEs achieved a $10\times$ convergence speedup on the Lotka-Volterra predator-prey system with equal parameters (240 vs. 252), the base implementation kept its numerical time-stepper (`Tsit5`) and basis representation (`Gaussian RBF`) strictly fixed.

This project performs a comprehensive, two-part investigation:

1. **Systematic Benchmarking & Ablation (Part 1):** Rigorously evaluates forward ODE solver orders ($p=1$ to $5$), step-size sensitivity ($\Delta t$), and alternative interpolation bases (Cubic B-splines, Lagrange, Chebyshev orthogonal polynomials).
2. **Novel Scientific Contributions (Part 2):**
   - Analyzes **Gradient Norm Dynamics ($\|\nabla_\theta \mathcal{L}\|_2$)** to show how solver truncation errors corrupt backpropagation.
   - Proposes a **Learnable Softmax Hybrid Basis Layer** blending compact B-splines and smooth RBFs.
   - Evaluates a **Stiffness-Solver Stability Map** on the non-linear Damped Pendulum ($\mu \in [0.1, 8.0]$).
   - Benchmarks symbolic equation discovery against **SINDy** under Gaussian observational noise ($\sigma \le 0.10$), plus a fit to real epidemic data.
   - Profiles memory and runtime of **Continuous Adjoint Sensitivity ($O(1)$ memory)** vs. **Direct Autograd ($O(N_t)$ memory)**.

> The 3D Lorenz attractor system was explored in early scoping (`data/lorenz.py` still
> exists, unused) but has been **deliberately dropped** from the final project scope —
> see [`docs/12_phase3_roadmap.md`](./docs/12_phase3_roadmap.md) §1.2.

---

## 👥 Project Team (Group 05)

Phase 1/2 (base reproduction and ablation studies) were completed collaboratively.
Phase 3 (novel contributions) is organized into five independent, folder-isolated
tracks per [`docs/12_phase3_roadmap.md`](./docs/12_phase3_roadmap.md) — the
authoritative source for current ownership and scope; the table below mirrors it so
this file does not drift out of sync again.

|    Student ID    | Full Name                       | Phase 3 Track                                                    | Git Feature Branch            |
| :---------------: | :------------------------------ | :---------------------------------------------------------------- | :---------------------------- |
| **2105032** | **Nawriz Ahmed Turjo**    | **A** — Adjoint vs. Autograd Memory/Speed Profiling                       | `feat/p3-adjoint-profiling` |
| **2105033** | **Abhishek Roy**          | **B** — Gradient Norm Dynamics vs. Solver Order                          | `feat/p3-gradient-dynamics` |
| **2105048** | **Shams Hossain Simanto** | **C** — Learnable Softmax Hybrid Basis (Spline + RBF)                    | `feat/p3-hybrid-basis`      |
| **2105055** | **Abrar Jahin**           | **D** — Stiffness–Solver Stability Phase Map (Damped Pendulum)           | `feat/p3-stiffness-map`     |
| **2105043** | **Monjur Hossain Khan**   | **E** — SINDy Comparison under Noise + Real Epidemic Fit                 | `feat/p3-sindy-epidemic`    |

> Note: the 3D Lorenz attractor track from the original proposal has been
> **deliberately dropped** from project scope (not deferred) — see
> [`docs/12_phase3_roadmap.md`](./docs/12_phase3_roadmap.md) §1.2. `data/lorenz.py`
> remains in the repository as unused, harmless code; no team member or Phase 3 track
> depends on it.

---

## 🗂️ Project & Repository Architecture

```text
kinetic-kan/
├── docs/                           # Master project documentation, proposals & blueprints
│   ├── 00_course_guidelines/       # Official syllabus & course notices
│   ├── 01_literature_and_ideas/    # Literature survey, candidate paper audits & notes
│   ├── 02_proposal/                # Proposal decks (.pptx) & written guides
│   ├── 03_presentation/            # Compiled LaTeX Beamer proposal decks (.tex, .pdf)
│   ├── 04_project_blueprint/       # Master technical deep dive & execution blueprints
│   └── README.md                   # Documentation index & team charter
│
├── implementation/                 # Active, modular PyTorch KAN-ODE implementation
│   ├── kan/                        # Kolmogorov-Arnold Network Core Layers
│   │   ├── basis.py                # RBF, B-spline, Chebyshev, Lagrange, IQF, RSWAF & Hybrid
│   │   ├── layer.py                # KDense layer (residual + base linear + basis activations)
│   │   └── model.py                # Multi-layer continuous vector field KAN
│   │
│   ├── ode/                        # Numerical ODE Solvers & Continuous Integrators
│   │   ├── solvers.py              # Tsit5, RK4, DOPRI5, Euler, Heun, Midpoint
│   │   └── neural_ode.py           # Continuous-time NeuralODE trajectory integrator wrapper
│   │
│   ├── data/                       # Dynamic System Generators & Empirical Loaders
│   │   ├── lotka_volterra.py       # 2D Predator-Prey ground truth generator (RK4)
│   │   ├── damped_pendulum.py      # Non-linear pendulum with variable damping (μ-stiffness)
│   │   ├── lorenz.py               # 3D Chaotic Lorenz attractor dataset
│   │   └── real_epidemic.py        # Empirical COVID-19 / Dengue time-series loader
│   │
│   ├── utils/                      # Metrics, Regularizers & Publication Plotting
│   │   ├── regularization.py       # L1 sparsity & entropy penalty regularizers
│   │   ├── metrics.py              # MSE, NFE tracker, parameter counter, Lipschitz bounds
│   │   └── plotting.py             # Phase portraits, streamplots, loss curves & 3D renders
│   │
│   ├── experiments/                # Phase 3 tracks land here, one folder per track (not yet created —
│   │                                #   see docs/12_phase3_roadmap.md §Part 2 for the planned layout)
│   │
│   ├── results/                    # Experimental artifacts & checkpoint stores
│   │   ├── kanode_rbf_tsit5/       # Original 10,000-epoch baseline run artifacts
│   │   ├── benchmarks/             # Phase 2 production sweeps: solvers, bases, noise, step-size, etc.
│   │   ├── _probe/, _fixed/        # Cross-domain stability fix probes & full runs (pendulum, SIR)
│   │   ├── phase2_closeout/        # Extrapolation-to-t=28, energy-dissipation JSON
│   │   ├── figures/                # 300 DPI publication figures (Task 2.5)
│   │   ├── tables/                 # Collated per-run and summary CSVs
│   │   └── phase3/                 # Phase 3 track outputs land here (not yet created)
│   │
│   ├── train.py                    # Main KAN-ODE / MLP-ODE training loop with stability guards & gradient-norm logging
│   ├── evaluate.py                 # Checkpoint evaluation, metric extraction & trajectory plotting
│   ├── test_facility.py            # Automated ablation benchmark suite (--mode solvers/activations)
│   ├── analyze_fixes.py            # Read-only diagnostic reader for any results tree
│   ├── phase2_closeout.py          # Extrapolation/energy/figure generation, no new training
│   └── README.md                   # Implementation quickstart guide & modularity documentation
│
├── tests/                          # Production-grade PyTest validation suite (143/143 passing)
│   ├── test_solvers.py             # Numerical order verification O(h^p) & energy conservation
│   ├── test_kan_layers.py          # Autograd gradcheck, partition of unity & NaN immunity
│   ├── test_pipeline.py            # End-to-end forward/backward & overfit sanity check
│   ├── test_mlp_ode.py             # MLP-ODE parameter-matching to paper Table I
│   ├── test_metrics.py             # SciML metrics correctness
│   ├── test_plotting.py            # Plotting utilities
│   └── test_datasets.py            # Deterministic seed reproducibility & physical invariants
│   # test_p3_<slug>.py files land here as each Phase 3 track (below) is implemented —
│   # e.g. Track A's test_adjoint.py does not exist yet; see docs/12_phase3_roadmap.md
│
├── pyproject.toml                  # Tool configs (pytest, ruff, mypy, coverage)
├── requirements.txt                # Pinned dependency manifest
├── LICENSE                         # MIT License
└── README.md                       # Master GitHub README with badges, team roster and quickstart
```

---

## ⚡ Quickstart & Commands Reference

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/NawrizTurjo/kinetic-kan.git
cd kinetic-kan

# Create and activate a virtual environment
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Baseline Model Training (10,000 Epochs)

```bash
cd implementation
python train.py --basis rbf --solver tsit5 --epochs 10000 --lr 2e-3
```

(`--lr 2e-3` is the project-standardized learning rate — both `train.py` and
`test_facility.py` default to it, resolving an earlier LR mismatch between the two
that produced a spurious ~1000x MSE discrepancy; the flag above is shown explicitly
but may be omitted.)

### 3. Checkpoint Evaluation & Metrics

```bash
cd implementation
python evaluate.py --checkpoint results/kanode_rbf_tsit5/best_model.pt
```

### 4. Running Automated Ablation Benchmarks

```bash
cd implementation
# Solver order ablation sweep
python test_facility.py --mode solvers --epochs 10000

# Basis function ablation sweep
python test_facility.py --mode activations --epochs 10000
```

---

## 📚 Quick Links to Documentation

All comprehensive documentation, research plans, and slides are located in the [`docs/`](./docs) directory:

* 🌟 **Phase 3 Roadmap (current, authoritative for Phase 3 scope & ownership):** [`docs/12_phase3_roadmap.md`](./docs/12_phase3_roadmap.md)
* 📖 **Original Project Implementation Plan** (Phases 1–2 authoritative; Phase 3 section superseded by the roadmap above): [`docs/04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md`](./docs/04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md)
* 🔬 **Mathematical Deep Dive & Research Blueprint:** [`docs/04_project_blueprint/KAN_ODE_Project_Deep_Dive_FINAL.md`](./docs/04_project_blueprint/KAN_ODE_Project_Deep_Dive_FINAL.md)
* 📊 **Presentation Scope vs. Final Plan Comparison:** [`docs/04_project_blueprint/Presentation_vs_Final_Plan_Comparison.md`](./docs/04_project_blueprint/Presentation_vs_Final_Plan_Comparison.md)
* 📽️ **Proposal Presentation Slide Deck:** [`docs/03_presentation/KAN-ODE.pdf`](./docs/03_presentation/KAN-ODE.pdf) ([Source `.tex`](./docs/03_presentation/KAN-ODE.tex))
* 📑 **Candidate Paper Audits & DOIs:** [`docs/01_literature_and_ideas/base_paper_links.md`](./docs/01_literature_and_ideas/base_paper_links.md)

---

## 📜 Base Paper Citation & References

### Primary Base Paper

* **Title:** *KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics*
* **Authors:** Benjamin C. Koenig, Suyong Kim, and Sili Deng
* **Affiliation:** Massachusetts Institute of Technology (MIT)
* **Venue:** *Computer Methods in Applied Mechanics and Engineering* (Elsevier), Volume 432, Part A, Article 117397, December 2024.
* **DOI:** [`10.1016/j.cma.2024.117397`](https://doi.org/10.1016/j.cma.2024.117397)
* **arXiv Preprint:** [arXiv:2407.04192 [cs.LG]](https://arxiv.org/abs/2407.04192) | [Direct PDF](https://arxiv.org/pdf/2407.04192.pdf)
* **Official Codebase:** [https://github.com/DENG-MIT/KAN-ODEs](https://github.com/DENG-MIT/KAN-ODEs)

```bibtex
@article{koenig2024kanodes,
  title     = {KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics},
  author    = {Koenig, Benjamin C. and Kim, Suyong and Deng, Sili},
  journal   = {Computer Methods in Applied Mechanics and Engineering},
  volume    = {432},
  pages     = {117397},
  year      = {2024},
  publisher = {Elsevier},
  doi       = {10.1016/j.cma.2024.117397},
  eprint    = {2407.04192},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

### Foundational Precursor & Benchmark References

* **Kolmogorov-Arnold Networks (KAN):**
  > Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark (2024). *KAN: Kolmogorov-Arnold Networks*. arXiv preprint [arXiv:2404.19756](https://arxiv.org/abs/2404.19756).
  >
* **Neural Ordinary Differential Equations:**
  > Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud (2018). *Neural Ordinary Differential Equations*. Advances in Neural Information Processing Systems (NeurIPS 2018), 31. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366).
  >
* **Data-Driven Dynamical System Discovery (SINDy):**
  > Steven L. Brunton, Joshua L. Proctor, J. Nathan Kutz (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*. Proceedings of the National Academy of Sciences (PNAS), 113(15), 3932–3937. [doi:10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113).
  >

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
