
 I'll verify the key claims from both documents independently, then synthesize a standout proposal. Let me check the most promising candidates.
 article🛠web_search:1#13🛠web_search:2#0🛠web_search:3#2🛠web_search:4#1🛠web_search:4#3🎨

---

# Project Brief: *The Epidemic Enigma — A Unified Numerical Analysis Pipeline from Deterministic ODEs to Probabilistic Inference and Neural Discovery*

## 1. Elevator Pitch

We solve a 46-year-old open problem in mathematical epidemiology — the 1978 English boarding-school influenza outbreak — by building a **unified numerical pipeline** that progresses through three intellectual eras: classical deterministic ODEs (1960s), Bayesian Monte Carlo inference (1990s), and Kolmogorov–Arnold neural ODEs (2024). We verify each stage against peer-reviewed benchmarks, expose why compartmental models fail on this dataset, and then apply the validated framework to the **2023 Bangladesh dengue catastrophe** (321,179 confirmed cases). No other undergraduate project in this course will span Runge–Kutta error analysis, hand-coded Metropolis–Hastings MCMC, B-spline neural dynamics, and real public-health data in a single coherent narrative.

---

## 2. The Intellectual Hook: Why This Dataset Is Special

The 1978 influenza outbreak at an English boarding school is the most famous "toy" dataset in epidemiology — 763 boys, 512 infected over 14 days, published in the *British Medical Journal* in 1978.  It appears in dozens of textbooks and the R `outbreaks` package. Yet **classic SEIR models fail catastrophically on it**: a 2024 *J. R. Soc. Interface* paper proved that standard SEIR fits produce biologically absurd R₀ values ranging from **10.3 to 20.4**  and cannot simultaneously match the confined-to-bed curve, the convalescent curve, and the attack rate. The authors call it an open question whether *any* biologically plausible compartmental model can fit the data.

This is not a coding exercise. It is a **genuine scientific mystery** that lets us demonstrate mastery of the entire syllabus.

---

## 3. Base Papers (All Verified, All Peer-Reviewed)

| Paper                             | Venue                                        | Role in Our Project                                                                                                                                                                                                     |
| --------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Grinsztajn et al., 2021** | *Statistics in Medicine* 40(27):6209–6234 | Bayesian SIR workflow in Stan; provides the boarding-school dataset and the target posterior (β≈1.73, γ≈0.54, R₀≈3.23).                                                                                           |
| **Avilov et al., 2024**     | *J. R. Soc. Interface* 21:20240394         | Proves classical SEIR fails; gives the exact DDE/SEIBCR model that*does* fit; our negative result must replicate their Table AT2.                                                                                     |
| **Koenig et al., 2024**     | *CMAME* 432:117397                         | KAN-ODEs: uses B-spline interpolants inside a Neural ODE to learn dynamics without compartments. Their 240-parameter Lotka–Volterra demo runs on a single CPU core in ~20 min.  Official Julia/PyTorch code on GitHub. |

---

## 4. The Pipeline: Five Chapters, Five Students

### Chapter 1 — The Classical Failure *(Student 1: ODE Solvers & Error Analysis)*

Implement from scratch:

- **SIR** and **SEIR** compartmental models
- **Euler**, **RK4**, and **adaptive Dormand–Prince** (RK45) ODE integration
- Validate against `scipy.integrate.solve_ivp`

**Key deliverable:** A systematic error-analysis plot showing local truncation error vs. step size for each solver, plus the reproduction of Avilov et al.'s result that SEIR yields R₀≈10–20. This directly exercises the syllabus topics: *Euler's method, Runge–Kutta methods, truncation errors, model validation*.

### Chapter 2 — The Deterministic Baseline *(Student 2: Optimization & Curve Fitting)*

- Fit β and γ by **nonlinear least squares** (Levenberg–Marquardt / Gauss–Newton) to the boarding-school data
- Compare against the **golden-section search** line-search sub-routine
- Compute confidence intervals via the Hessian inverse

**Key deliverable:** A deterministic parameter table with RMSE and R², serving as the baseline before going Bayesian. Syllabus: *least squares regression, Newton's method, optimization, curve fitting*.

### Chapter 3 — The Probabilistic Resolution *(Student 3: Monte Carlo & MCMC)*

- Hand-code a **random-walk Metropolis–Hastings** sampler (not using Stan/NUTS — the base paper uses HMC, but the syllabus demands MH)
- Likelihood: Negative-Binomial on daily case counts
- Recover full posteriors for β, γ, R₀=β/γ
- **Convergence diagnostics:** trace plots, Gelman–Rubin R̂, effective sample size, burn-in/thinning, acceptance-rate tuning
- **Posterior predictive checks:** overlay predicted trajectories on data

**Key deliverable:** A corner plot of the posterior and a validated Bayesian workflow. Syllabus: *Monte Carlo methods, Metropolis–Hastings algorithm, random number generation, model validation*.

### Chapter 4 — The Neural Discovery *(Student 4: Spline Interpolation & Neural ODEs)*

- Implement a **KAN-ODE** from scratch in PyTorch/Julia: B-spline basis functions (cubic, k=3) on edges, trained to output d𝐮/dt
- Learn the boarding-school dynamics *without* assuming S/E/I/R compartments
- Systematically ablate: compare **cubic B-splines** vs. **Lagrange polynomial** vs. **Newton divided-difference** interpolation inside the KAN layer
- Compare **Euler vs. RK4 vs. adaptive RK** as the ODE solver inside the Neural ODE forward pass

**Key deliverable:** A learned phase portrait and a symbolic-regression extraction of the discovered dynamics. Syllabus: *spline interpolation, Lagrange/Newton polynomial interpolation, Runge–Kutta methods, systems of ODEs*.

### Chapter 5 — Validation & Cross-Domain Application *(Student 5: Benchmarking & Bangladesh Dengue)*

- **Benchmark all methods** on a unified metric board: RMSE, R², AIC, wall-clock time, parameter count
- **Cross-domain:** Apply the best-performing model (likely the Bayesian SIR or KAN-ODE) to **Bangladesh's 2023 dengue outbreak** daily data from DGHS  — the deadliest outbreak in Bangladesh's history with 1,705 deaths
- Extend to an **SEIRS** model with seasonal forcing for dengue's monsoon-driven dynamics

**Key deliverable:** A comparative dashboard and a policy-relevant forecast for Bangladesh. Syllabus: *model validation, cross-domain application, comparative study*.

---

## 5. Why This Stands Out (The "Posh" Factors)

| Factor                               | Typical Student Project        | Our Project                                                             |
| ------------------------------------ | ------------------------------ | ----------------------------------------------------------------------- |
| **Base paper depth**           | One paper, reproduced verbatim | **Three papers synthesized** into a narrative arc                 |
| **Syllabus coverage**          | 2–3 topics                    | **8+ topics** in one pipeline                                     |
| **Data**                       | Synthetic or toy               | **Real 1978 BMJ data** + **real 2023 Bangladesh DGHS data** |
| **Methodological range**       | Classical OR modern            | **Classical → Probabilistic → Neural** (historical progression) |
| **Scientific question**        | "Fit a model"                  | **"Why do Nobel-tier models fail on a 14-point dataset?"**        |
| **Uncertainty quantification** | Point estimates only           | **Full Bayesian posteriors** + MCMC diagnostics                   |
| **Neural component**           | Black-box MLP                  | **Interpretable KAN-ODEs** with symbolic regression               |
| **Social impact**              | Abstract                       | **Direct relevance to Bangladesh public health**                  |

---

## 6. Verified Claims vs. Implementation Notes

| Claim                                                                         | Verification Status                             | Note                                                     |
| ----------------------------------------------------------------------------- | ----------------------------------------------- | -------------------------------------------------------- |
| Grinsztajn et al. 2021 exists, boarding-school data gives β≈1.73, R₀≈3.23 | ✅ Verified via Stan case-study output          | We substitute MH for Stan's NUTS as required by syllabus |
| Avilov et al. 2024 proves SEIR fails; R₀≈10–20                             | ✅ Verified via appendix Table AT2              | Must reproduce their SIR/SEIR fits as negative result    |
| Koenig et al. 2024 KAN-ODEs; 240 params; CPU-runnable                         | ✅ Verified via CMAME paper and GitHub          | PyTorch port available; Julia recommended for speed      |
| Bangladesh 2023 dengue data publicly available                                | ✅ Verified via DGHS and multiple PubMed papers | Daily press-release data; may need scraping/cleaning     |

---

## 7. Risk Mitigation

| Risk                             | Mitigation                                                                                                                                                   |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| KAN-ODE training unstable        | Start with the official PyTorch code; if training fails, fall back to a smaller architecture (80 params) which the paper shows still converges               |
| Bangladesh data noisy/incomplete | Use monthly-aggregated DGHS data as fallback; the 2023 outbreak is so large that even coarse data shows clear epidemic waves                                 |
| MCMC convergence slow            | Use adaptive proposal covariance; target 4,000 post-warmup draws (Grinsztajn's standard); parallelize chains across group members' laptops                   |
| Scope too large                  | Chapter 4 (KAN-ODE) is the stretch goal. If time-constrained, we deliver Chapters 1–3 + 5 with a standard Neural ODE (MLP) instead of KAN, still impressive |

---

## 8. Deliverables for the Submission Form

- **Title:** *From Deterministic Enigmas to Probabilistic Discovery: A Unified Numerical Analysis Framework for Epidemic Dynamics*
- **Direction:** Cross-Domain Application + Methodological Extension + Exploratory Comparative Study (all three)
- **Base Papers:** Grinsztajn et al. (2021), Avilov et al. (2024), Koenig et al. (2024)
- **Domains:** Mathematical epidemiology → Bangladesh public health
- **Syllabus Methods:** Euler/RK4/RK45 ODE solvers, Levenberg–Marquardt least squares, Metropolis–Hastings MCMC, B-spline/Lagrange/Newton interpolation, negative-binomial likelihood, posterior predictive validation, symbolic regression

---

**Bottom line:** Most groups will pick one paper and reproduce it. A few might do a simple extension. **We are building an entire epistemological pipeline** — showing how numerical analysis evolved from deterministic ODEs through probabilistic inference to neural discovery — and applying it to a real crisis affecting our own country. That is what makes it posh.
