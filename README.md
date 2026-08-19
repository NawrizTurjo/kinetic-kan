# ⚡ KINETIC-KAN: Solver-Aware Neural Dynamics

> **Interrogating Numerical Integration, Basis Representations, and Adjoint Sensitivities in Kolmogorov-Arnold Network ODEs**
> *Course Project for CSE 402: Numerical Analysis, Simulation & Modeling | Department of CSE, BUET (4-1)*

[![CI Pipeline](https://img.shields.io/badge/CI-Passing-brightgreen.svg)]()
[![Python 3.10+](<https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg>)]()
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
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
   - Benchmarks symbolic equation discovery against **SINDy** under Gaussian observational noise ($\sigma \le 0.10$).
   - Profiles memory and runtime of **Continuous Adjoint Sensitivity ($O(1)$ memory)** vs. **Direct Autograd ($O(N_t)$ memory)**.
   - Evaluates multi-scale chaotic attractor dynamics on the **3D Lorenz System**.

---

## 👥 Project Team (Group 05)

|    Student ID    | Full Name                       | Primary Research Role                                                             | Git Feature Branch            |
| :---------------: | :------------------------------ | :-------------------------------------------------------------------------------- | :---------------------------- |
| **2105032** | **Nawriz Ahmed Turjo**    | **Lead:** KAN Architecture, B-Spline Layers & Learnable Hybrid Basis        | `feat/m1-kan-architecture`  |
| **2105033** | **Abhishek Roy**          | **Lead:** Numerical ODE Solvers, Step-Size Sweeps & Adjoint Profiling       | `feat/m2-numerical-solvers` |
| **2105043** | **Monjur Hossain Khan**   | **Lead:** SciML Optimization, Loss Landscapes & Gradient Norm Dynamics      | `feat/m3-sciml-pipeline`    |
| **2105048** | **Shams Hossain Simanto** | **Lead:** Non-linear Stability, Damped Pendulum Stiffness & SINDy Benchmark | `feat/m4-stability-sindy`   |
| **2105055** | **Abrar Jahin**           | **Lead:** 3D Chaotic Lorenz Dynamics, Real Epidemiological Fit & Synthesis  | `feat/m5-chaos-visuals`     |

---

## 🗂️ Project & Repository Architecture

```text
kinetic-kan/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Multi-OS & Multi-Python CI pipeline
│       ├── sciml_benchmarks.yml    # Automated SciML benchmark suite
│       └── code_quality.yml        # Ruff linting, Mypy type-checking & Black formatting
│
├── configs/                        # Hyperparameter manifests & experiment configs
│   ├── base_config.yaml            # Default training, optimizer & seed settings
│   ├── solver_ablation.yaml        # Solver order sweeps (p=1..5) & tolerance settings
│   └── stiffness_sweep.yaml        # Damping ratios μ & Jacobian stiffness configurations
│
├── datasets/                       # Dynamic system trajectory generators & loaders
│   ├── lotka_volterra.py           # 2D Predator-Prey ground truth generator (RK4)
│   ├── damped_pendulum.py          # Variable damping non-linear pendulum generator
│   ├── lorenz.py                   # 3D Chaotic Lorenz attractor dataset
│   └── real_epidemic.py            # Empirical COVID-19 / Regional Dengue time-series loader
│
├── docs/                           # Master project documentation, proposals & blueprints
│   ├── 00_course_guidelines/       # Official syllabus & course notices
│   ├── 01_literature_and_ideas/    # Literature survey, candidate paper audits & notes
│   ├── 02_proposal/                # Proposal decks (.pptx) & written guides
│   ├── 03_presentation/            # Compiled LaTeX Beamer proposal decks (.tex, .pdf)
│   ├── 04_project_blueprint/       # Master technical deep dive & execution blueprints
│   └── README.md                   # Documentation index & team charter
│
├── models/                         # Core neural & numerical architectures
│   ├── ode_solvers.py              # Standalone Euler, RK2, RK4, and Adaptive integrators
│   ├── kan_layers.py               # Gaussian RBF, B-spline, Lagrange, Chebyshev & Hybrid Basis
│   ├── kan_ode.py                  # Continuous-time KAN-ODE Vector Field & Pipeline
│   └── mlp_ode.py                  # Baseline MLP-Neural ODE for 10x speedup comparison
│
├── experiments/                    # Reproducible benchmark & novelty experiment scripts
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
├── utils/                          # Metrics, symbolic extractors & plotting tools
│   ├── metrics.py                  # MSE, NFE tracker, parameter counter, Lipschitz bound
│   ├── symbolic_prune.py           # L1 edge pruning & symbolic equation extractor
│   └── plotting.py                 # Publication-quality phase portraits, streamplots & heatmaps
│
├── tests/                          # Production-grade PyTest suite
│   ├── test_solvers.py             # Numerical order verification O(h^p) & energy conservation
│   ├── test_kan_layers.py          # Autograd gradcheck, partition of unity & NaN immunity
│   ├── test_pipeline.py            # End-to-end forward/backward & overfit sanity check
│   ├── test_adjoint.py             # Continuous Adjoint vs Autograd parity & O(1) memory test
│   └── test_datasets.py            # Deterministic seed reproducibility & physical invariants
│
├── pyproject.toml                  # Tool configs (pytest, ruff, mypy, coverage)
├── requirements.txt                # Pinned dependency manifest
├── LICENSE                         # MIT License
└── README.md                       # Project Root README (This file)
```

---

## ⚡ Quickstart & Installation

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

### 2. Run Test Suite

```bash
pytest tests/ -v
```

### 3. Run Baseline Reproduction (Lotka-Volterra 10x Speedup)

```bash
python experiments/01_reproduce_baseline.py
```

---

## 📚 Quick Links to Documentation

All comprehensive documentation, research plans, and slides are located in the [`docs/`](./docs) directory:

* 📖 **Master Project Implementation Plan:** [`docs/04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md`](./docs/04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md)
* 🔬 **Mathematical Deep Dive & Research Blueprint:** [`docs/04_project_blueprint/KAN_ODE_Project_Deep_Dive_FINAL.md`](./docs/04_project_blueprint/KAN_ODE_Project_Deep_Dive_FINAL.md)
* 📊 **Presentation Scope vs. Final Plan Comparison:** [`docs/04_project_blueprint/Presentation_vs_Final_Plan_Comparison.md`](./docs/04_project_blueprint/Presentation_vs_Final_Plan_Comparison.md)
* 📽️ **Proposal Presentation Slide Deck:** [`docs/03_presentation/slide.pdf`](./docs/03_presentation/slide.pdf) ([Source `.tex`](./docs/03_presentation/slide.tex))
* 📑 **Candidate Paper Audits & DOIs:** [`docs/01_literature_and_ideas/base_paper_links.md`](./docs/01_literature_and_ideas/base_paper_links.md)

---

## 📜 Base Paper Citation & References

### Primary Base Paper

* **Title:** *KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics*
* **Authors:** Zachary Koenig, Jihoon Kim, and Yuntian Deng
* **Affiliation:** Massachusetts Institute of Technology (MIT)
* **Venue:** *Computer Methods in Applied Mechanics and Engineering* (Elsevier), Volume 432, Part A, Article 117397, December 2024.
* **DOI:** [`10.1016/j.cma.2024.117397`](https://doi.org/10.1016/j.cma.2024.117397)
* **arXiv Preprint:** [arXiv:2407.04192 [cs.LG]](https://arxiv.org/abs/2407.04192) | [Direct PDF](https://arxiv.org/pdf/2407.04192.pdf)
* **Official Codebase:** [https://github.com/DENG-MIT/KAN-ODEs](https://github.com/DENG-MIT/KAN-ODEs)

```bibtex
@article{koenig2024kanodes,
  title     = {KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics},
  author    = {Koenig, Zachary and Kim, Jihoon and Deng, Yuntian},
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
