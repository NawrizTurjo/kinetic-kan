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

Roles below reflect current Phase 3 assignments — see
[`12_phase3_roadmap.md`](../docs/12_phase3_roadmap.md) for full task breakdowns
(authoritative; supersedes the Phase 3 ownership in `PROJECT_IMPLEMENTATION_PLAN.md`).

|    Student ID    | Full Name                       | Phase 1–2 Role | Phase 3 Track |
| :---------------: | :------------------------------ | :-------------------------------------------------------------------------------- | :--- |
| **2105032** | **Nawriz Ahmed Turjo**    | Phase 1 Foundation Lead — KAN architecture, B-spline/Chebyshev/Lagrange bases, solver tests, CI | **A** — Adjoint vs. Autograd Profiling |
| **2105033** | **Abhishek Roy**          | Initial engine scaffold — core `ode/`, `kan/`, `train.py`, first Lotka-Volterra baseline | **B** — Gradient Norm Dynamics |
| **2105043** | **Monjur Hossain Khan** ("Shovon") | SIR root-cause diagnosis & fix (`--time_scale`, `--conserve_mode projection`) | **E** — SINDy Comparison + Real Epidemic Fit |
| **2105048** | **Shams Hossain Simanto** | Cross-domain stability fixes (pendulum + SIR), Phase 2 closeout | **C** — Learnable Hybrid Basis |
| **2105055** | **Abrar Jahin**           | Table collation, publication figures | **D** — Stiffness–Solver Stability Map |

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
│   ├── KAN-ODE.tex                  # Standard course proposal deck
│   └── KAN-ODE.pdf                  # Compiled PDF for proposal presentation
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

* **Full Implementation Plan & Code Specifications:**👉 [`04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md`](./04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md) *(original plan — Phase 3 sections superseded, see below)*
* **Master Technical Blueprint & Math Foundations:**👉 [`04_project_blueprint/KAN_ODE_Project_Deep_Dive_FINAL.md`](./04_project_blueprint/KAN_ODE_Project_Deep_Dive_FINAL.md)
* **Presentation Scope vs. Final Plan Comparison:**👉 [`04_project_blueprint/Presentation_vs_Final_Plan_Comparison.md`](./04_project_blueprint/Presentation_vs_Final_Plan_Comparison.md)
* **Proposal Presentation Slides (LaTeX Beamer):**
  👉 [`03_presentation/KAN-ODE.pdf`](./03_presentation/KAN-ODE.pdf) ([Source `.tex`](./03_presentation/KAN-ODE.tex))

---

## 📖 Chronological Project Log (Phase 2 onward) — authoritative for current status

The blueprint above describes the *plan*. What actually happened, in order, is recorded
here — each doc states its own status and links the next:

| # | Document | What it covers |
| :-: | :--- | :--- |
| 05 | [`05_phase2_benchmark_analysis.md`](./05_phase2_benchmark_analysis.md) | Phase 2 Lotka-Volterra results: solver ablation (Table 1), basis ablation (Table 2), KAN vs. MLP (Table 3), step-size sweep, noise sweep |
| 06 | [`06_suggested_fixes.md`](./06_suggested_fixes.md) | Diagnosis of the pendulum and SIR cross-domain failures |
| 07 | [`07_fix_changelog.md`](./07_fix_changelog.md) | Code changes implementing the fixes (`train.py`, `run_phase2.ps1`) |
| 08 | [`08_how_to_run_fixes.md`](./08_how_to_run_fixes.md) | How to reproduce the fix runs |
| 09 | [`09_stability_fix_results.md`](./09_stability_fix_results.md) | Pendulum fix results and verdict (`--t_train_end 5.0`) |
| 10 | [`10_sir_root_cause_and_fix.md`](./10_sir_root_cause_and_fix.md) | SIR root cause and fix (`--time_scale`, `--conserve_mode projection`) |
| 11 | [`11_phase2_closeout.md`](./11_phase2_closeout.md) | Extrapolation to $t{=}28$, pendulum energy check, publication figures — Phase 2 closed except Lorenz (removed) |
| 12 | [`12_phase3_roadmap.md`](./12_phase3_roadmap.md) | **Phase 3 plan — current, authoritative** for scope and task ownership |
| 13–17 | *(reserved, one per Phase 3 track — see `12` §5.1)* | Each track's findings write-up, filed as it completes |

---

## 📜 Base Paper Citation & References

### Primary Base Paper

* **Title:** *KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics*
* **Authors:** Benjamin C. Koenig, Suyong Kim, and Sili Deng (MIT)
* **Venue:** *Computer Methods in Applied Mechanics and Engineering* (Elsevier), Vol. 432, Part A, 117397, 2024.
* **DOI:** [`10.1016/j.cma.2024.117397`](https://doi.org/10.1016/j.cma.2024.117397)
* **arXiv:** [arXiv:2407.04192](https://arxiv.org/abs/2407.04192) | [Codebase (Deng Lab)](https://github.com/DENG-MIT/KAN-ODEs)

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

---

*Portable Workspace configured for BUET CSE 402 Project.*
