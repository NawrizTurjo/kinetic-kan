# ⚡ KINETIC-KAN: Solver-Aware Neural Dynamics

> **Interrogating Numerical Integration, Basis Representations, and Adjoint Sensitivities in Kolmogorov-Arnold Network ODEs**
> *Course Project for CSE 402: Numerical Analysis, Simulation & Modeling | Department of CSE, BUET | Group A_08, Section A*

[![CI Pipeline](https://github.com/NawrizTurjo/kinetic-kan/actions/workflows/ci.yml/badge.svg)](https://github.com/NawrizTurjo/kinetic-kan/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-242%20passed-brightgreen.svg)](./tests)
[![Python 3.10+](<https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg>)](./pyproject.toml)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](./requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

📄 **Final report (ACM format):** [`report/A_08.pdf`](./report/A_08.pdf)

---

## 📌 Summary

Neural ODEs learn an unknown dynamical system $\dot{\mathbf{u}} = \mathbf{f}_\theta(\mathbf{u})$ by fitting the vector field with a neural network and integrating it numerically. **KAN-ODEs** (*Koenig, Kim & Deng, CMAME 2024*) replace the usual multilayer perceptron with a Kolmogorov-Arnold Network, whose every edge is a learnable univariate function. On the Lotka-Volterra predator-prey system the base paper reports that a 240-parameter KAN-ODE extrapolates accurately from a short training window and trains about ten times faster than a parameter-matched (252-parameter) MLP-ODE.

The base paper fixes three design choices: the integrator (Tsit5), the basis (Gaussian RBF) and the gradient method (adjoint sensitivity). We reproduce its Lotka-Volterra benchmark in PyTorch, implementing **six explicit Runge-Kutta integrators** and **seven basis families** from their defining formulas, and then treat those three choices as variables:

1. **Reproduction and ablation:** integrator order (Euler to Tsit5), step size, observation noise, and seven bases (RBF, cubic B-spline, Chebyshev, Lagrange, Newton, IQF, RSWAF), plus KAN-ODE vs. MLP-ODE.
2. **Cross-domain transfer:** a damped pendulum, an SIR epidemic model and a synthetic outbreak curve, with fixes where the recipe fails.
3. **Follow-on studies (Tracks A-E):** adjoint vs. direct-autograd cost, gradient-norm dynamics vs. solver order, a learnable softmax blend of B-spline and RBF bases, a stiffness-solver stability map on the damped pendulum, and sparse symbolic regression (SINDy) under noise.
4. **Budget validation:** whether the conclusions survive a five times larger training budget ($5\times10^4$ epochs).

### Key findings

- **Above order two, the solver does not change accuracy.** Midpoint matches Tsit5 within 0.3% at 29% of the cost.
- **Networks learn the field their solver needs.** The Euler-trained model matches the inverse modified equation, and every trained model, KAN or MLP, is an attracting limit cycle rather than the true neutral centre.
- **The KAN advantage depends on the budget.** At $10^4$ epochs KAN-ODE extrapolates 5.4× better than a parameter-matched MLP; at $5\times10^4$ epochs the MLP is 6.8× better.
- **Local bases win.** B-spline and RBF are tied, while global polynomial bases extrapolate 4-5× worse.
- **SIR's training failure is a scaling problem.** It traces to a 19× unit-scale mismatch at initialization; time rescaling, a conservation projection and a vanishing-field gate move extrapolation $R^2$ from −20.97 to +0.998.

All results are single-seed (seed 42); see the reports for the full discussion and limitations.

---

## 👥 Project Team (Group A_08, Section A)

The base reproduction and ablation studies were done together; each follow-on track had one owner.

|  Student ID  | Name                             | Track                                                        |
| :----------: | :------------------------------- | :----------------------------------------------------------- |
| **2105032** | **Nawriz Ahmed Turjo**           | **A** — Adjoint vs. Autograd Memory/Speed Profiling          |
| **2105033** | **Abhishek Roy**                 | **B** — Gradient Norm Dynamics vs. Solver Order              |
| **2105048** | **Shams Hossain Simanto**        | **C** — Learnable Softmax Hybrid Basis (Spline + RBF)        |
| **2105055** | **Md. Abrar Jahin**              | **D** — Stiffness–Solver Stability Map (Damped Pendulum)     |
| **2105043** | **Monjur Hossain Khan (Shovon)** | **E** — SINDy Comparison under Noise + Outbreak Curve Fit    |

Supervisor: Niaz Rahman, Lecturer, Department of CSE, BUET.

---

## 🗂️ Repository Layout

```text
kinetic-kan/
├── docs/
│   ├── base_paper/                 # The KAN-ODE base paper (arXiv:2407.04192)
│   ├── course_guidelines/          # Syllabus and course notices
│   └── presentation/               # Proposal presentation (PDF + LaTeX source)
│
├── implementation/                 # PyTorch KAN-ODE implementation
│   ├── kan/
│   │   ├── basis.py                # RBF, B-spline, Chebyshev, Lagrange, Newton, IQF, RSWAF (+ hybrid)
│   │   ├── layer.py                # KDense layer (basis expansion + SiLU residual branch)
│   │   ├── model.py                # Multi-layer KAN vector field
│   │   └── mlp.py                  # MLP-ODE baseline
│   ├── ode/
│   │   ├── solvers.py              # Euler, Heun, Midpoint, RK4, DOPRI5, Tsit5 (fixed-step)
│   │   └── neural_ode.py           # NeuralODE trajectory wrapper
│   ├── data/
│   │   ├── lotka_volterra.py       # Predator-prey ground truth (SciPy DOP853, tol 1e-12)
│   │   ├── damped_pendulum.py      # Damped pendulum with variable damping μ
│   │   ├── lorenz.py               # Lorenz attractor
│   │   └── real_epidemic.py        # SIR model and a synthetic outbreak curve
│   ├── utils/                      # Metrics, regularizers, plotting
│   ├── experiments/                # Track A-E scripts and the extended-budget runs
│   ├── results/                    # Every run: metrics.json, training_history.json, checkpoints
│   ├── train.py                    # Single training entry point (KAN-ODE or MLP-ODE)
│   ├── evaluate.py                 # Re-score a checkpoint and plot its trajectory
│   ├── test_facility.py            # Ablation sweeps (solvers, bases, step size, noise, models)
│   ├── run_phase2.ps1              # Launches the ablation runs in parallel
│   ├── collate_results.py          # Collects run metrics into CSV tables
│   ├── analyze_fixes.py            # Summarizes the cross-domain fix runs
│   └── phase2_closeout.py          # Extrapolation and energy analysis
│
├── report/                         # Final course report, ACM sigconf format (A_08.pdf)
│   ├── A_08.tex, kk-acm.sty        # LaTeX sources (acmart.cls, ACM bib style bundled)
│   ├── sections/                   # Report sections
│   ├── figures/                    # Report figures
│   ├── reproduce/                  # Regenerates every derived figure, table and number
│   └── tools/                      # Tectonic LaTeX engine
│
├── tests/                          # PyTest suite (242 passed, 1 skipped)
├── .github/workflows/ci.yml        # Tests on Python 3.10-3.12
├── pyproject.toml                  # pytest, ruff, mypy and coverage settings
├── requirements.txt                # Dependencies (minimum versions)
└── LICENSE                         # MIT
```

---

## ⚡ Quickstart

### 1. Environment setup

```bash
git clone https://github.com/NawrizTurjo/kinetic-kan.git
cd kinetic-kan

python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Everything runs on CPU. Python 3.10 or newer is required.

### 2. Run the tests

```bash
python -m pytest tests -q
```

### 3. Train the baseline KAN-ODE (10,000 epochs)

```bash
cd implementation
python train.py --basis rbf --solver tsit5 --epochs 10000 --save_dir results/my_run
```

The defaults reproduce the base paper's setup: a [2, 10, 2] KAN with 5 RBF basis functions, Tsit5 with 2 sub-steps, Adam at learning rate 2e-3, seed 42. Use `--model mlp` for the MLP-ODE baseline, `--dataset` for the other systems, and `python train.py --help` for every option. Each run writes its metrics, training history and checkpoints to `--save_dir`.

### 4. Evaluate a checkpoint

```bash
cd implementation
python evaluate.py --checkpoint results/benchmarks/kanode_flagship/best_model.pt
```

### 5. Run the ablation sweeps

```bash
cd implementation
python test_facility.py --mode solvers       # integrator ablation
python test_facility.py --mode activations   # basis ablation
python test_facility.py --mode stepsize      # Δt sweep
python test_facility.py --mode noise         # observation-noise sweep
python test_facility.py --mode models        # KAN-ODE vs. MLP-ODE
```

Or run the whole suite with `.\run_phase2.ps1 -Only all` (PowerShell, from `implementation/`).

### 6. Reproduce the report's figures and tables

The trained runs are committed under `implementation/results/`, so no retraining is needed. From the repository root:

```bash
python report/reproduce/run_all.py               # all steps, then build report/A_08.pdf
python report/reproduce/run_all.py --no-report   # skip the LaTeX build
```

This takes a few minutes on a laptop CPU. To build only the report:

```bash
cd report
tools/tectonic.exe -k A_08.tex
```

See [Reproducibility](#-reproducibility) below for the full guide.

---

## 🔁 Reproducibility

Every figure, table and derived number in the report can be regenerated from the committed runs without retraining. The full guide is [`report/reproduce/README.md`](./report/reproduce/README.md). It covers:

* **How to run it:** the one command that reruns the whole pipeline, how to run single steps, and how to check that the copied result plots are unchanged.
* **What each step produces:** a table mapping every script to the report figure or table it makes, what it reads and what it writes.
* **Where each result came from:** the run folder under `implementation/results/` behind each result, and the script or command that trained it.
* **Caveats:** which tanh MLP run the report uses, why some checkpoints have no stored `dt`, and how closely cloud-trained checkpoints reproduce on a different CPU (within 0.62%).

The run settings (software versions, seeding, data generation and checkpoint selection) are summarized in Appendix A of [`A_08.pdf`](./report/A_08.pdf).

---

## 📚 Documentation

* 📄 **Final report (ACM):** [`report/A_08.pdf`](./report/A_08.pdf)
* 🔁 **Reproduction guide:** [`report/reproduce/README.md`](./report/reproduce/README.md)
* 📽️ **Proposal presentation:** [`docs/presentation/KAN-ODE.pdf`](./docs/presentation/KAN-ODE.pdf) ([LaTeX source](./docs/presentation/KAN-ODE.tex))
* 📑 **Base paper:** [`docs/base_paper/2407.04192v3.pdf`](./docs/base_paper/2407.04192v3.pdf)
* 📝 **Report terminology glossary:** [`CONTEXT.md`](./CONTEXT.md)

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
@article{Koenig_2024,
   title={KAN-ODEs: Kolmogorov–Arnold network ordinary differential equations for learning dynamical systems and hidden physics},
   volume={432},
   ISSN={0045-7825},
   url={http://dx.doi.org/10.1016/j.cma.2024.117397},
   DOI={10.1016/j.cma.2024.117397},
   journal={Computer Methods in Applied Mechanics and Engineering},
   publisher={Elsevier BV},
   author={Koenig, Benjamin C. and Kim, Suyong and Deng, Sili},
   year={2024},
   month=Dec, pages={117397} }
```

### Foundational References

* **Kolmogorov-Arnold Networks (KAN):**
  > Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark (2024). *KAN: Kolmogorov-Arnold Networks*. arXiv preprint [arXiv:2404.19756](https://arxiv.org/abs/2404.19756).
* **Neural Ordinary Differential Equations:**
  > Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud (2018). *Neural Ordinary Differential Equations*. Advances in Neural Information Processing Systems (NeurIPS 2018), 31. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366).
* **Data-Driven Dynamical System Discovery (SINDy):**
  > Steven L. Brunton, Joshua L. Proctor, J. Nathan Kutz (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*. Proceedings of the National Academy of Sciences (PNAS), 113(15), 3932–3937. [doi:10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113).

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
