# Final Project Selection Analysis (ML-Focused Edition)

Given that your 5-member team has already completed a Machine Learning course and possesses ML experience, the constraints and risk factors change significantly. You don't need to shy away from PyTorch or ML-heavy architectures. 

Instead, you can leverage your ML background to tackle a "Posh", cutting-edge project that bridges modern AI with classical Numerical Analysis. 

We also evaluated these options against hardware constraints: **Everything recommended below is 100% runnable on Kaggle's Free GPUs (P100, T4x2), and some even run on CPUs in minutes.**

---

## 🏆 The Ultimate ML/Numerical Choice: **KAN-ODEs for Dynamical System Discovery**
**Base Paper:** *KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics* (Koenig, Kim & Deng, CMAME 2024).

This is widely considered by the AI deep-dives as the single best scaffold for a team with ML experience. It fuses two distinct syllabus topics into one state-of-the-art AI architecture.

### Why this is the absolute best choice for your team:
1. **Syllabus Double-Hit:** KANs (Kolmogorov-Arnold Networks) replace traditional Neural Network weights with learnable activation functions on the edges. These functions are typically **B-splines** (Syllabus Topic: *Spline Interpolation*). The network predicts the derivative $du/dt$, which is then integrated forward in time using an **ODE Solver like RK4** (Syllabus Topic: *Solution of ODEs, Runge-Kutta*). 
2. **Kaggle GPU Feasibility (Extremely High):** The paper explicitly states that their flagship benchmark (the Lotka-Volterra predator-prey model) uses a tiny model with only **240 parameters** and trains to convergence in $10^4$ epochs in **~20 minutes on a single CPU core**. A free Kaggle T4 or P100 GPU will train this in seconds/minutes. It is highly lightweight.
3. **The "Wow" Factor (Methodological Extension):** The base paper uses standard adaptive ODE solvers and standard spline basis functions. Your project's unique contribution will be systematically benchmarking alternative syllabus methods. You will swap the B-splines for Lagrange polynomials or Newton's polynomial interpolation, and you will swap the ODE solver for Euler, RK2, and RK4, measuring how numerical truncation error affects the Neural Network's training loss.

### Pros and Cons
**Pros:**
* Immensely impressive to professors; combining KANs with Numerical Analysis is very 2024/2025 cutting-edge.
* Excellent open-source code available (in Python/PyTorch and Julia) to use as a starting scaffold.
* Very lightweight compute requirements.

**Cons:**
* You will need to carefully frame the report so the professor sees the *Numerical Analysis* (the splines and ODE solvers) rather than just the *Machine Learning* (the training loop and loss).

---

## 🥈 Runner Up: **Diffusion ODE Sampling (STORK)**
**Base Paper:** *STORK: Faster Diffusion and Flow Matching Sampling...* (Tan et al., 2025).

If you want to do Generative AI (Diffusion Models), this is a brilliant numerical project.

### Why it's good:
Diffusion models generate images by solving a "Probability Flow ODE" backwards in time. The process of generating an image is literally just running a numerical ODE solver (like Euler or Runge-Kutta). The paper STORK proposes a stabilized RK method for diffusion. 

**Kaggle Feasibility:** You **do not** need to train a diffusion model from scratch (which takes clusters of GPUs). You just download a small pre-trained diffusion model (like a CIFAR-10 diffusion model) which easily fits in Kaggle's 16GB VRAM. Your code will just be modifying the *inference/sampling* step.

**The Project:** Compare Euler, Heun (RK2), standard RK4, and the paper's STORK sampler. Measure the Frechet Inception Distance (FID) of the generated images vs. the Number of Function Evaluations (NFE) (which is the step-size). 

**Pros:** Highly visual, GenAI is very popular, direct application of Runge-Kutta.
**Cons:** Diffusion codebases can be heavy and annoying to refactor just to inject custom ODE solvers.

---

## 🥉 Honorable Mention: **SIR + Bayesian MCMC**
**Base Paper:** *Bayesian workflow for disease transmission modeling in Stan* (Grinsztajn et al., 2021).

If you want to lean heavily into Statistical Machine Learning / Probabilistic Inference, this remains a God-tier project. You use **Metropolis-Hastings MCMC** (Syllabus: Monte Carlo methods) to infer epidemiological parameters of an ODE model (Syllabus: ODEs) from real-world data (1978 Boarding School Flu). 

**Kaggle Feasibility:** Extremely high. MCMC on a 14-data-point ODE model runs in seconds on any CPU/GPU.

---

## Next Steps for Submission
If you choose the **KAN-ODEs** project, here is how you should fill out your Google Form (Deadline: August 15):

* **Base Paper:** Koenig et al. (2024), "KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics", *Computer Methods in Applied Mechanics and Engineering*.
* **Project Scope / Direction:** Methodological Extension & Cross-Domain Application. 
* **Description:** We will build a Kolmogorov-Arnold Network ODE (KAN-ODE) to model a non-linear dynamical system. KANs utilize spline interpolation for learnable activations, while Neural ODEs require numerical integration for the forward pass. We will extend the base paper by methodologically interrogating its default numerical subroutines: we will replace the default B-splines with other syllabus interpolants (e.g., Lagrange, Newton's polynomials) and replace the adaptive ODE solver with custom-implemented Euler and RK4 solvers. We will analyze how numerical truncation error and interpolation accuracy impact the convergence and training stability of the Neural ODE.
