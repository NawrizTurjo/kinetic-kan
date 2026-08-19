# Summary of AI Project Recommendations

We analyzed multiple AI-generated proposals and deep dives to evaluate the best base paper and project direction for the CSE-402 Numerical Analysis and Simulation course project. Here is a summary of the verdicts from each AI analysis:

## 1. cg-dive.md (The "Numerical Robustness" Perspective)
* **Top Recommendation:** **Bundle Adjustment in the Eager Mode** (Zhan et al., 2026).
* **Verdict:** Ranks Bundle Adjustment as #1 because it allows students to deeply investigate the numerical robustness, conditioning, and linear solvers (Gauss-Newton vs. Levenberg-Marquardt) in 3D reconstruction. It argues that KAN-ODEs is good (Rank 2) but actually uses Gaussian RBFs, not B-splines. Ranks SIR + Bayesian inference as Rank 3.
* **Key Insight:** Don't just implement a numerical method; make the numerical method's stability and conditioning the actual research question.

## 2. ds-dive.md (The "AI/ML Hybrid" Perspective)
* **Top Recommendation:** **KAN-ODE Hybrid for Dynamical System Discovery** (Koenig et al., 2024).
* **Verdict:** Strongly advocates for KAN-ODEs because it stacks two syllabus topics (Spline Interpolation and ODE Solvers). Proposes an extension where students swap the default B-splines for Lagrange polynomials and the default adaptive ODE solver for Euler/RK4.
* **Runner-ups:** SIR + Bayesian MCMC, Bundle Adjustment, STORK Diffusion.

## 3. km-dive.md (The "Unified Pipeline" Perspective)
* **Top Recommendation:** **The Epidemic Enigma (SIR + Bayesian MCMC + KAN-ODEs)**.
* **Verdict:** Proposes a massive, 5-person unified pipeline bridging three papers (Grinsztajn 2021, Avilov 2024, Koenig 2024). Students would prove classical SEIR fails on the 1978 English boarding-school dataset, use MCMC for probabilistic inference, and then use Neural ODEs (KAN-ODEs) for discovery.
* **Key Insight:** Highly ambitious, combining classical deterministic models, Bayesian inference, and neural discovery into a single narrative.

## 4. qn-dive.md (The "Uncertainty Quantification" Perspective)
* **Top Recommendation:** **Opinion Dynamics / Sociophysics** (Oestereich et al., 2023).
* **Verdict:** Focuses on Uncertainty Quantification (UQ) applied to social dynamics and opinion models. Recommends using ODE solvers (Runge-Kutta), Metropolis-Hastings MCMC, and splines to model how opinions evolve. 
* **Key Insight:** Shifts the focus from pure prediction to investigating the theoretical limits of data-driven learning (adversarial systems).

## 5. z-dive.md (The "Syllabus Purist" Perspective)
* **Top Recommendation:** **SIR + Bayesian Epidemic Inference** (Grinsztajn et al., 2021).
* **Verdict:** Explicitly rejects KAN-ODEs and Bundle Adjustment as "trendy but shallow," arguing they are ML/CV projects masquerading as numerical analysis and will lead to debugging PyTorch instead of doing math. Recommends hand-coding RK4 and Metropolis-Hastings for the SIR model.
* **Key Insight:** Proving that numerical truncation errors from poor ODE solvers (like Euler) destroy the likelihood landscape for MCMC convergence is a "god-tier" insight that directly validates the syllabus.

## 6. gm-dive.md (The "Comprehensive Matrix" Perspective)
* **Top Recommendation:** Provides domain-specific recommendations:
  * **For AI/ML Focus:** KAN-ODEs.
  * **For Applied Simulation/MCMC:** Bayesian SIR-MCMC.
  * **For CV/Robotics:** Eager-Mode Bundle Adjustment.
* **Verdict:** Identifies the strengths of each paper and provides a structured risk analysis and 6-week execution roadmap for the team.
