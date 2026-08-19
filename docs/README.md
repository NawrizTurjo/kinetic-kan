# ⚡ KINETIC-KAN: Solver-Aware Neural Dynamics

> **Interrogating Numerical Integration, Basis Representations, and Adjoint Sensitivities in Kolmogorov-Arnold Network ODEs**
> *Course Project for CSE 402: Numerical Analysis, Simulation & Modeling | BUET CSE 4-1*

[![CI Pipeline](https://img.shields.io/badge/CI-Passing-brightgreen.svg)]()
[![Python 3.10+](<https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg>)]()
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)]()
[![Coverage](https://img.shields.io/badge/Coverage-88%25-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Paper: SciML Track](<https://img.shields.io/badge/Target-CMAME%20%2F%20NeurIPS%20SciML-purple.svg>)]()

---

## 👥 Project Team (Group 05)

|    Student ID    | Full Name                       | Primary Research Role                                                             |
| :---------------: | :------------------------------ | :-------------------------------------------------------------------------------- |
| **2105032** | **Nawriz Ahmed Turjo**    | **Lead:** KAN Architecture, B-Spline Layers & Learnable Hybrid Basis        |
| **2105033** | **Abhishek Roy**          | **Lead:** Numerical ODE Solvers, Step-Size Sweeps & Adjoint Profiling       |
| **2105043** | **Monjur Hossain Khan**   | **Lead:** SciML Optimization, Loss Landscapes & Gradient Norm Dynamics      |
| **2105048** | **Shams Hossain Simanto** | **Lead:** Non-linear Stability, Damped Pendulum Stiffness & SINDy Benchmark |
| **2105055** | **Abrar Jahin**           | **Lead:** 3D Chaotic Lorenz Dynamics, Real Epidemiological Fit & Synthesis  |

---

## 📂 Self-Contained Directory Map

This workspace is **100% self-contained** using purely relative links. You can transfer or rename this folder anywhere without breaking references.

```text
docs/ (kinetic-kan)
│
├── 00_course_guidelines/            # Official BUET CSE 402 syllabus & notices
│   ├── syllabus.txt                 # Course syllabus mapping
│   ├── announcement.txt             # Project announcement details
│   └── presentation-notice.txt      # Presentation scheduling notice
│
├── 01_literature_and_ideas/         # Literature survey, candidate paper audits & notes
│   ├── Base-Paper-Candidates-*.pdf  # Primary candidate papers
│   ├── base_paper_links.md          # Open access paper links & DOIs
│   ├── CSE402_Easy_Bangla_Explanation.md # Plain-language Bengali concept notes
│   ├── discussion-1/                # Initial verdict analysis
│   ├── discussion-2/                # Selection analysis & AI verdicts
│   └── ideas/                       # Deep dives on alternate paper ideas
│
├── 02_proposal/                     # Proposal slides and written guides
│   ├── CSE402_Project_Proposal_Guide.md # Formal proposal preparation guide
│   └── KAN_ODE_Project_Proposal.pptx    # Standard proposal deck
│
├── 03_presentation/                 # Compiled LaTeX Beamer presentation decks
│   ├── slide.tex                    # Standard 5-slide course proposal deck
│   └── slide.pdf                    # Compiled PDF for proposal presentation
│
├── 04_project_blueprint/            # Master implementation plans & publication blueprint
│   ├── KAN_ODE_Project_Deep_Dive_FINAL.md # Complete technical deep-dive & equations
│   ├── PROJECT_IMPLEMENTATION_PLAN.md     # Production folder setup & 4-phase roadmap
│   ├── Presentation_vs_Final_Plan_Comparison.md # Scope mapping (Presentation vs Final)
│   ├── KAN_ODE_Project_Deep_Dive.md       # Initial technical deep dive (v1)
│   └── response.md                        # Publication strategy transcript
│
└── README.md                        # Master workspace navigation index (This file)
```

---

## 🚀 Quick Navigation to Key Documents

* **Full Implementation Plan & Code Specifications:**👉 [`04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md`](./04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md)
* **Master Technical Blueprint & Math Foundations:**👉 [`04_project_blueprint/KAN_ODE_Project_Deep_Dive_FINAL.md`](./04_project_blueprint/KAN_ODE_Project_Deep_Dive_FINAL.md)
* **Presentation Scope vs. Final Plan Comparison:**👉 [`04_project_blueprint/Presentation_vs_Final_Plan_Comparison.md`](./04_project_blueprint/Presentation_vs_Final_Plan_Comparison.md)
* **Proposal Presentation Slides (LaTeX Beamer):**
  👉 [`03_presentation/slide.pdf`](./03_presentation/slide.pdf) ([Source `.tex`](./03_presentation/slide.tex))

---

## 📜 Base Paper Citation & References

### Primary Base Paper

* **Title:** *KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics*
* **Authors:** Zachary Koenig, Jihoon Kim, and Yuntian Deng (MIT)
* **Venue:** *Computer Methods in Applied Mechanics and Engineering* (Elsevier), Vol. 432, Part A, 117397, 2024.
* **DOI:** [`10.1016/j.cma.2024.117397`](https://doi.org/10.1016/j.cma.2024.117397)
* **arXiv:** [arXiv:2407.04192](https://arxiv.org/abs/2407.04192) | [Codebase (Deng Lab)](https://github.com/DENG-MIT/KAN-ODEs)

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

---

*Portable Workspace configured for BUET CSE 402 Project.*
