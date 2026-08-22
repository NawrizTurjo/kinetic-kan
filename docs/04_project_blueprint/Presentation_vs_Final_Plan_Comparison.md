# Presentation vs. Final Plan Comparison: Scope & Roadmap

## CSE 402: Numerical Analysis, Simulation & Modeling

**Project:** Solver-Aware Neural Dynamics (KAN-ODEs) | **Team Size:** 5 Students | **Date:** August 2026

---

## 1. Executive Summary & At-a-Glance Comparison

The **Presentation** served as our proposal pitch to the course instructors (highlighting core syllabus alignment and foundational experiments). The **Final Plan** ([`KAN_ODE_Project_Deep_Dive_FINAL.md`](./KAN_ODE_Project_Deep_Dive_FINAL.md)) is a **direct superset** of the presentation — retaining 100% of the original proposal while adding advanced scientific machine learning (SciML) and publication-grade enhancements.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       AT-A-GLANCE SCOPE COMPARISON                                       │
├───────────────────────────────┬──────────────────────────────────┬───────────────────────────────────────┤
│ Dimension                     │ In Proposal Presentation (Slides)│ In Extended Final Plan (Deep Dive)    │
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────┤
│ **Primary Focus**             │ Course grading & syllabus fit    │ Course grading + Publication in SciML │
│ **Architecture Structure**    │ Single unified pipeline          │ Formal Two-Part Division (Part 1 & 2) │
│ **ODE Solvers**               │ Euler, RK2, RK4, Adaptive        │ Euler, RK2, RK4, Dopri5 + Adjoint ODE │
│ **Basis Functions**           │ B-splines, RBF, Lagrange, Poly   │ Spline, RBF, Poly + **Hybrid Basis**  │
│ **Dynamic Systems**           │ Lotka-Volterra, Pendulum, SIR    │ LV, Pendulum, SIR + **3D Lorenz Chaos**│
│ **Equation Discovery**        │ Symbolic KAN pruning             │ KAN Pruning vs. **SINDy Benchmark**   │
│ **Optimization Mechanics**    │ Fixed Adam + Adjoint (stated)    │ **Adjoint vs. Autograd Profiling**    │
│ **Gradient Analysis**         │ Not mentioned                    │ **Gradient Norm Dynamics (||∇L||_2)**  │
│ **Stiffness Analysis**        │ Mentioned generally              │ **Stiffness vs. Solver Stability Map**│
│ **Data Types**                │ Synthetic RK4 generated data     │ Synthetic + **Real Epidemiological**  │
│ **Theoretical Rigor**         │ Qualitative syllabus alignment   │ **Formal Lipschitz Continuity Bound** │
│ **Statistical Evaluation**    │ Single representative runs       │ **Multi-Seed (N=5, Mean ± Std Dev)**  │
└───────────────────────────────┴──────────────────────────────────┴───────────────────────────────────────┘
```

---

## 2. What We Pitched in the Presentation (The Core Proposal)

The presentation ([`KAN-ODE.pdf`](../03_presentation/KAN-ODE.pdf)) focused on **three core design decisions** of the base MIT paper and proposed stress-testing them:

1. **Base Paper Foundations (MIT 2024):**
   - Concept of KAN-ODEs: learning continuous vector fields $\dot{\mathbf{u}} = \mathbf{f}_\theta(\mathbf{u})$ with learnable 1D edge curves.
   - The $10\times$ convergence speedup over MLP-Neural ODEs on Lotka-Volterra with equal parameters ($240$ vs $252$).
2. **Three Key Design Choices in Base Paper:**
   - *Integrator:* Fixed `Tsit5` adaptive Runge-Kutta solver.
   - *Activation:* Fixed `Gaussian RBF` basis.
   - *Optimization:* Fixed `Adjoint Sensitivity` with Adam optimizer and $L_1$ sparsity pruning.
3. **Our Proposed Extensions (The 3 Slides of Deliverables):**
   - *Extension 1 (Solver Ablation):* Forward Euler ($p=1$), Heun RK2 ($p=2$), Classical RK4 ($p=4$), and Adaptive Dormand-Prince ($p=5$).
   - *Extension 2 (Basis Swap):* Cubic B-splines, Gaussian RBF, Lagrange, and Newton polynomials.
   - *Extension 3 (Robustness & Cross-Domain):* Non-linear Damped Pendulum, SIR epidemic dynamics, and Gaussian noise sweep ($\sigma \le 0.10$).
4. **Course Syllabus Convergence:**
   - Demonstrating ODE integration, interpolation techniques, truncation errors, and gradient optimization.

---

## 3. What Was Added in the Final Plan (The Publication Upgrades)

In the final formalized plan, everything from the presentation becomes **Part 1 (Systematic Benchmarking)**, and we introduce **Part 2 (Novel Research Contributions)** to make the project publication-ready for top SciML venues (*NeurIPS/ICLR AI for Science Workshops* or *CMAME/Neural Networks* journals):

```
                                  FINAL PROJECT ROADMAP
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
       ┌───────────────────────────────┐             ┌───────────────────────────────┐
       │   PART 1: FROM PRESENTATION   │             │  PART 2: NOVEL ADDITIONS      │
       │   (Systematic Benchmarking)   │             │  (Publication Upgrades)       │
       ├───────────────────────────────┤             ├───────────────────────────────┤
       │ 1. Exact Base Reproduction    │             │ 1. Gradient Norm Dynamics     │
       │ 2. Solver Order Sweep (p=1..5)│             │ 2. Learnable Hybrid Basis     │
       │ 3. Step-size Sensitivity (Δt) │             │ 3. Stiffness Stability Map    │
       │ 4. Basis Functions (Spline/RBF│             │ 4. SINDy Baseline Benchmark   │
       │    /Lagrange/Chebyshev)       │             │ 5. Adjoint vs Autograd Profile│
       │ 5. Noise Sweep & Extrapol.    │             │ 6. 3D Chaotic Lorenz Benchmark│
       │ 6. Pendulum & SIR Dynamics    │             │ 7. Real Epidemiological Data  │
       │                               │             │ 8. Lipschitz Stability Proofs │
       └───────────────────────────────┘             └───────────────────────────────┘
```

### Detailed Breakdown of the 8 New Additions:

1. **Gradient Norm Dynamics ($\|\nabla_\theta \mathcal{L}\|_2$ Logging) [Novelty 1]:**
   - *What it is:* Tracking the $L_2$ norm of backpropagated gradients at every epoch across different solver orders.
   - *Why it matters:* Proves that coarse solvers (Euler/RK2) inject numerical gradient noise that destabilizes training, independent of trajectory MSE.
2. **Learnable Hybrid Basis Architecture [Novelty 2]:**
   - *What it is:* A new KAN edge activation blending B-splines (local compact support) and Gaussian RBFs (global smoothness) with learnable softmax gating weights: $\phi(x) = w_b \text{silu}(x) + \alpha \text{Spline}(x) + \beta \text{RBF}(x)$.
3. **Stiffness Ratio vs. Solver Stability Phase Map [Novelty 3]:**
   - *What it is:* Systematically sweeping damping coefficient $\mu \in [0.1, 8.0]$ on the Damped Pendulum to create a 2D heatmap diagnosing exactly when fixed-step solvers explode.
4. **SINDy (Sparse Identification of Nonlinear Dynamics) Benchmark [Novelty 4]:**
   - *What it is:* Comparing KAN-ODE symbolic distillation against SINDy (the SciML gold standard by Brunton et al., 2016) on clean and noisy data ($\sigma \in \{0.01, 0.05, 0.10\}$).
5. **Continuous Adjoint vs. Direct Autograd Profiling [Novelty 5]:**
   - *What it is:* Profiling memory consumption ($O(1)$ vs $O(N_t)$), wall-clock speed, and gradient fidelity between Pontryagin continuous adjoints and discrete unrolled autograd.
6. **Multi-Scale Chaotic Dynamics (3D Lorenz Attractor) [Novelty 6]:**
   - *What it is:* Benchmarking KAN-ODEs on the non-linear chaotic Lorenz system to test strange attractor geometry, Lyapunov exponents, and phase-space topology.
7. **Real-World Epidemiological Time-Series Validation [Novelty 7]:**
   - *What it is:* Applying continuous-time SIR/SEIR models to real empirical Dengue / COVID-19 infection records instead of solely synthetic trajectories.
8. **Theoretical Lipschitz Continuity & Stability Bounds [Novelty 8]:**
   - *What it is:* Formal mathematical proof showing that B-spline/RBF edge activations yield strictly bounded Lipschitz constants ($L \le |w_b| L_{\text{silu}} + |w_s| \frac{k}{h} \sum |c_i|$), explaining why KAN-ODEs are intrinsically more stable than deep MLPs.

---

## 4. Why This Two-Part Strategy is Optimal

1. **Guaranteed Course Excellence (Grade Protection):**
   - Part 1 covers 100% of what was promised in the presentation and satisfies every pillar of the CSE 402 syllabus (ODEs, Splines, Interpolation, Errors, Optimization).
2. **Zero Compute Overhead:**
   - Total model parameter count remains under $300$. All experiments run in minutes on a free Kaggle T4 GPU or even a standard laptop CPU (<500 MB VRAM).
3. **Publication Quality (High ROI):**
   - Part 2 elevates the project from a course assignment into a genuine, peer-reviewed research paper for top SciML workshops or computational mechanics journals.

---

## 5. Summary Mapping Table

| Topic / Component                                                | In Presentation? |      In Final Plan?      | Role / Purpose                                           |
| :--------------------------------------------------------------- | :--------------: | :----------------------: | :------------------------------------------------------- |
| **Base Reproduction (Lotka-Volterra)**                     |      ✅ Yes      |          ✅ Yes          | Establishes experimental baseline ($10\times$ speedup) |
| **ODE Solver Ablation (Euler/RK2/RK4/Dopri5)**             |      ✅ Yes      |          ✅ Yes          | Core numerical syllabus requirement                      |
| **Basis Ablation (Spline/RBF/Lagrange/Poly)**              |      ✅ Yes      |          ✅ Yes          | Core interpolation syllabus requirement                  |
| **Noise Robustness ($\sigma \le 0.10$)**                 |      ✅ Yes      |          ✅ Yes          | Evaluates trajectory fitting stability                   |
| **Damped Pendulum & SIR Dynamics**                         |      ✅ Yes      |          ✅ Yes          | Cross-domain physical/biological validation              |
| **Gradient Norm Trajectory ($\|\nabla\mathcal{L}\|_2$)** |      ❌ No      | ✅**Yes (Part 2)** | Novel optimization insight on solver error coupling      |
| **Learnable Hybrid Basis Layer**                           |      ❌ No      | ✅**Yes (Part 2)** | Novel architectural contribution                         |
| **Stiffness vs. Solver Stability Heatmap**                 |      ❌ No      | ✅**Yes (Part 2)** | Practical diagnostic for stiff dynamic regimes           |
| **SINDy Comparison Benchmark**                             |      ❌ No      | ✅**Yes (Part 2)** | Validates symbolic equation discovery claim              |
| **Continuous Adjoint vs. Autograd Profiling**              |      ❌ No      | ✅**Yes (Part 2)** | Memory & speed trade-off profiling                       |
| **3D Lorenz Chaotic Attractor**                            |      ❌ No      | ✅**Yes (Part 2)** | Demonstrates multi-scale non-toy generalization          |
| **Real-World Epidemiological Dataset**                     |      ❌ No      | ✅**Yes (Part 2)** | Empirical real-world case study                          |
| **Lipschitz Bound Derivation**                             |      ❌ No      | ✅**Yes (Part 2)** | Theoretical foundation for journal review                |
| **Multi-Seed ($N=5$) Statistical Error Bars**            |      ❌ No      | ✅**Yes (Part 2)** | Scientific rigor standard for peer review                |

---

*Reference Files:*

- Proposal Presentation Deck: [`KAN-ODE.tex`](../03_presentation/KAN-ODE.tex)
- Complete Final Project Blueprint: [`KAN_ODE_Project_Deep_Dive_FINAL.md`](./KAN_ODE_Project_Deep_Dive_FINAL.md)
