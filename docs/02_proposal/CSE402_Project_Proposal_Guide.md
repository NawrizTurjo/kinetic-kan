# CSE-402: Numerical Analysis, Simulation & Modeling
## Team Project Proposal & Base Paper Selection Guide (Comprehensive Edition)

> **Course Context:** 5-Member Team Project | **Submission Deadline:** 15 August (2:00 PM)  
> **Key Objective:** Select a published base paper, understand its baseline implementation, and execute a meaningful methodological extension or cross-domain application that directly demonstrates mastery of the course syllabus.

---

## 1. Executive Summary & Team Decision Overview

Since our team already completed the 3rd-year Machine Learning course, we have the mathematical and programming background to tackle **Scientific ML (SciML), LLM Inference Optimization, Graph Neural Networks, 3D Vision, or Generative AI** without being restricted to purely toy simulations. All proposed options below are computationally lightweight (runnable on free Kaggle GPUs or standard laptop CPUs in minutes).

We have synthesized the top **7 flagship project tracks** across 4 major domains:

| Track | Focus Domain | Candidate Project | Key Reference | Core Numerical Fusion |
| :---: | :--- | :--- | :--- | :--- |
| **A** | **Scientific ML** | KAN-ODEs for Dynamical System Discovery | CMAME 2024, MIT | Spline/RBF Interpolation + Runge-Kutta ODE Solvers inside Neural ODEs |
| **B** | **Probabilistic ML** | Bayesian Epidemic Inference & Numerical Dynamics | Wiley 2021 | Runge-Kutta ODEs + Metropolis-Hastings MCMC + Least Squares on Real Data |
| **C** | **LLMs & Language Systems** | Speculative Decoding via Metropolis-Hastings | ICML 2023 | Monte Carlo Rejection Sampling + Metropolis Acceptance + Discrete RNG |
| **D** | **Geometric Deep Learning** | Continuous Graph Neural Diffusion (GRAND) | ICML 2021 | Graph Laplacian Eigenvalue Decomposition (QR) + Runge-Kutta Graph ODEs |
| **E** | **3D Gaussian Splatting** | Second-Order Levenberg-Marquardt Optimization | 2024 / 2026 | Non-linear Least Squares (LM vs Adam) + Linear Solvers in 3DGS |
| **F** | **Spatial AI / Robotics** | Differentiable Bundle Adjustment | IEEE T-RO 2026 | Non-linear Least Squares (Gauss-Newton/LM) + Direct Linear Solvers (LU/GJ) |
| **G** | **Generative AI** | Diffusion Probability Flow ODE Sampling | STORK 2025 | High-Order Runge-Kutta Integrators + Image Quality vs NFE Tradeoffs |

### 📊 At-A-Glance: Strategic Comparison of All 7 Project Tracks

| Track / Rank | Candidate & Base Paper | Venue & Status | Syllabus Coverage | Implementation Risk | Research & "Posh" Ceiling | Best Suited For |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **🥇 Track B** | **Bayesian Epidemic Dynamics (SIR/SEIR + MCMC)**<br>*Grinsztajn et al. (2021) / Avilov et al. (2024)* | *Statistics in Medicine* & *J. R. Soc. Interface* | **95%** (5 major pillars) | **Very Low** (100% CPU, pure NumPy) | **Exceptional** (Scientific discovery + UQ) | **Top Recommendation for Full Team** |
| **🥈 Track C** | **LLM Speculative Decoding via Metropolis-Hastings**<br>*Leviathan et al. (Google Research, ICML 2023)* | *ICML (PMLR)* / DeepMind | **80%** (MCMC + RNG + Optimization) | **Low-Moderate** (Kaggle GPU / Open Weights) | **Exceptional** (Modern LLM acceleration) | Groups focused on GenAI, LLMs & Systems |
| **🥉 Track A** | **KAN-ODEs for Dynamical Systems**<br>*Koenig, Kim & Deng (MIT, CMAME 2024)* | *CMAME (Elsevier)* | **65%** (Splines/RBF + ODEs + Gradients) | **Moderate** (PyTorch Neural ODEs) | **Very High** (Cutting-edge SciML) | Groups with strong PyTorch / ML background |
| **Track D** | **Continuous Graph Neural Diffusion (GRAND)**<br>*Chamberlain et al. (ICML 2021)* | *ICML (PMLR)* | **75%** (Laplacian QR Eigen + RK4 ODEs) | **Low-Moderate** (Cora/Citeseer benchmarks) | **Very High** (Geometric DL / Graph Theory) | Groups interested in Graph ML & Spectral Theory |
| **Track E** | **3D Gaussian Splatting with Levenberg-Marquardt**<br>*Höllein et al. (TUM 2024 / 3DV 2026)* | *3DV / arXiv* | **70%** (Nonlinear LS + LU Solvers) | **Moderate** (3DGS rasterization pipeline) | **Exceptional** (Photorealistic 3D Vision) | Groups passionate about 3D Vision & NeRFs |
| **Track F** | **Differentiable Bundle Adjustment in Eager Mode**<br>*Zhan et al. (IEEE T-RO 2026)* | *IEEE Trans. Robotics (T-RO)* | **65%** (Gauss-Newton/LM + LU Solvers) | **Low-Moderate** (Synthetic 3D Scenes) | **Very High** (Robotics / SLAM focus) | Groups with Computer Vision & SLAM focus |
| **Track G** | **Diffusion Probability-Flow ODE Sampling (STORK)**<br>*Tan et al. (2025)* | *arXiv / Open Code* | **55%** (Stiff Runge-Kutta ODE Solvers) | **Moderate** (Pretrained CIFAR diffusion) | **Very High** (Image Generative AI) | Groups passionate about Image Generation |

---

## 2. Track A: KAN-ODEs for Dynamical System Discovery (Scientific ML)

### 📌 Base Paper & Publication Background
* **Paper Title:** *KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics*
* **Authors:** Benjamin C. Koenig, Suyong Kim, Sili Deng (MIT)
* **Venue & Date:** *Computer Methods in Applied Mechanics and Engineering (CMAME)*, Vol. 432, Article 117397, **2024 (Peer-Reviewed Journal)**.
* **Recency:** ~2 years recent; represents the cutting edge of Scientific Machine Learning (SciML) combining Kolmogorov-Arnold Networks with continuous-depth Neural ODEs.

### 📄 What the Original Paper Actually Did (Baseline)
* **Core Problem:** Standard Neural ODEs use traditional MLPs (fixed activations on nodes, weights on edges) which act as black boxes and struggle with interpretability.
* **Paper's Method:** Placed learnable univariate activation functions directly on the network edges. The network predicts dynamic rates of change ($du/dt$), integrated through a standard differential equation solver (`Tsit5` / Dormand-Prince).
* **Baseline Benchmark:** Demonstrated on the **Lotka-Volterra predator-prey system**. A tiny 240-parameter KAN-ODE reached a training loss of $2.6 \times 10^{-5}$ in $10^4$ epochs, significantly outperforming an MLP Neural ODE (which took $10^5$ epochs for $3.0 \times 10^{-5}$ loss) on a single CPU core.

### 🚀 What Extra Work Our Group Will Do (Our Scope of Extension)
1. **ODE Solver Ablation Study:** The original paper adopted an off-the-shelf adaptive solver without testing numerical solver order. We will decouple the time-integrator and systematically benchmark **Forward Euler vs. Heun (RK2) vs. Classical RK4 vs. Adaptive Dormand-Prince**, tabulating training loss, stability under noisy data, and Number of Function Evaluations (NFE).
2. **Interpolation Basis Function Swap:** Replace the default basis functions with alternative syllabus interpolants (**Cubic B-splines vs. Lagrange Polynomials vs. Newton's Divided Differences vs. Chebyshev Orthogonal Polynomials**) and evaluate gradient stability.
3. **Cross-Domain Dynamics:** Deploy our hybrid KAN-ODE to discover equations from a secondary system (e.g., Damped Pendulum or SIR epidemic curve).

### 👥 5-Member Work Breakdown
* **Member 1 (ODE Solver Specialist):** Builds custom standalone Euler, RK2, and RK4 integrators; tracks forward time-stepping and local truncation errors.
* **Member 2 (KAN Architecture Specialist):** Implements the KAN layer with B-spline and alternative polynomial basis functions.
* **Member 3 (Training & SciML Pipeline Lead):** Builds the Neural ODE training pipeline in PyTorch on the Lotka-Volterra benchmark.
* **Member 4 (Numerical Error & Stability Analyst):** Evaluates loss convergence curves, NFE efficiency, and stability under noise.
* **Member 5 (Cross-Domain & Visualization Lead):** Deploys the trained KAN-ODE on a secondary dynamic dataset and plots phase portrait comparisons.

---

## 3. Track B: Bayesian Epidemic Inference & Numerical Dynamics (Probabilistic ML)

### 📌 Base Paper & Publication Background
* **Primary Paper:** *Bayesian workflow for disease transmission modeling in Stan* (Léo Grinsztajn, Elizaveta Semenova, Charles C. Margossian, Julien Riou)
  * **Venue & Date:** *Statistics in Medicine*, 40(27):6209–6234, **2021 (Wiley, Peer-Reviewed Journal)**.
* **Companion Paper:** *The 1978 English boarding school influenza outbreak: where the classic SEIR model fails* (K. Avilov et al.)
  * **Venue & Date:** *Journal of the Royal Society Interface*, 21(220):20240394, **2024 (Royal Society, Peer-Reviewed)**.
* **Recency:** Highly mature 2021 Bayesian baseline paired with a landmark 2024 Royal Society paper exposing structural failure modes.

### 📄 What the Original Paper Actually Did (Baseline)
* **Core Problem:** Inferring epidemiological parameters ($\beta$ transmission rate, $\gamma$ recovery rate, and $R_0 = \beta/\gamma$) from sparse, noisy daily outbreak data with full Bayesian credible intervals.
* **Paper's Method:** Coupled deterministic ODE integration of the classic SIR system with probabilistic Markov Chain Monte Carlo (using Stan's HMC/NUTS engine) on the famous **1978 British Medical Journal (BMJ) English boarding school influenza outbreak** ($N=763$ boys, 14 daily data points).
* **Baseline Benchmark:** Recovered posterior values $\beta \approx 1.73/\text{day}$, $\gamma \approx 0.54/\text{day}$, giving basic reproduction number $R_0 \approx 3.23$.

### 🚀 What Extra Work Our Group Will Do (Our Scope of Extension)
1. **The Solver-Inference Numerical Proof (Flagship Insight):** Instead of using high-level Stan black-box engines, we will hand-code the **Random-Walk Metropolis-Hastings MCMC sampler** and the ODE solvers from scratch in pure NumPy. We will run MCMC using a coarse Euler solver vs. a precision RK4 solver to mathematically prove that **low-order numerical truncation errors distort the likelihood landscape and destroy MCMC convergence**, whereas RK4 restores smooth mixing.
2. **SEIR Structural Failure Reproduction:** Reproduce the 2024 Royal Society paper's finding that extending the model to SEIR causes the inferred reproduction number to jump to an exaggerated, biologically absurd $R_0 \approx 10-15$.
3. **Local High-Impact Application:** Deploy our validated Bayesian MCMC framework to the **2023 Bangladesh Dengue Outbreak** dataset (321,179 cases from DGHS) to estimate transmission dynamics and forecast peak waves.

### 👥 5-Member Work Breakdown
* **Member 1 (Deterministic ODE Lead):** Hand-codes Euler, RK2, and RK4 solvers; conducts $\Delta t$ step-size convergence and truncation error analysis.
* **Member 2 (Optimization & Fitting Lead):** Implements non-linear least squares (Gauss-Newton) to find baseline deterministic estimates of $\beta$ and $\gamma$.
* **Member 3 (Stochastic MCMC Lead):** Hand-codes the Random-Walk Metropolis-Hastings sampler from scratch with Poisson/Negative-Binomial likelihoods.
* **Member 4 (UQ & Coupling Analyst):** Evaluates posterior distributions, trace plots, and proves how ODE solver order affects MCMC convergence.
* **Member 5 (SEIR & Bangladesh Dengue Lead):** Extends model to SEIR/SEIRS and fits the pipeline to the 2023 Bangladesh Dengue dataset.

---

## 4. Track C: Speculative Decoding via Metropolis-Hastings (LLM Systems)

### 📌 Base Paper & Publication Background
* **Primary Paper:** *Fast Inference from Transformers via Speculative Decoding* (Yaniv Leviathan, Matan Kalman, Yossi Matias - Google Research)
  * **Venue & Date:** *International Conference on Machine Learning (ICML)*, **2023 (Top-Tier AI Conference)**.
* **Companion Paper:** *Accelerating Large Language Model Decoding with Speculative Sampling* (Charlie Chen et al. - Google DeepMind, **2023**).
* **Recency:** ~3 years recent; the foundational, industry-standard paradigm used today across all major LLM serving engines (vLLM, Ollama, TensorRT-LLM).

### 📄 What the Original Paper Actually Did (Baseline)
* **Core Problem:** Autoregressive LLM token generation is memory-bandwidth bound (reading billions of parameters per single token generated).
* **Paper's Method:** Used a small, fast "draft model" (e.g. 0.5B) to generate a sequence of $K$ candidate tokens in parallel, then used a large "target model" (e.g. 3B/7B) to verify all $K$ tokens in a single forward pass.
* **Baseline Benchmark:** Proved that using a modified **Metropolis-Hastings rejection sampling rule** ($\alpha = \min(1, \frac{P(x)}{Q(x)})$), the speculative process achieves a **2x–3x lossless speedup** with mathematical guarantee that the output probability distribution is 100% identical to the target model alone.

### 🚀 What Extra Work Our Group Will Do (Our Scope of Extension)
1. **From-Scratch Numerical Implementation:** Hand-code the exact Metropolis-Hastings acceptance/rejection loop and probability recovery adjustment from scratch on open-weights models (e.g. Qwen2.5-0.5B draft + Qwen2.5-3B target on Kaggle GPU).
2. **Entropy-Adaptive MCMC Thresholding:** Standard speculative decoding uses a fixed draft length $K$. We will design an **entropy-adaptive draft acceptance mechanism** that dynamically scales $K$ based on token Shannon entropy $-\sum p \log p$.
3. **Draft Length Optimization:** Formulate the latency cost function as a 1D optimization problem to solve for optimal draft length $K^*$ across generation temperatures.

### 👥 5-Member Work Breakdown
* **Member 1 (LLM Serving Lead):** Sets up HuggingFace inference pipelines for draft and target models on Kaggle.
* **Member 2 (Metropolis-Hastings Verification Lead):** Implements the modified Metropolis-Hastings accept/reject verification algorithm.
* **Member 3 (RNG & Distribution Lead):** Hand-codes categorical sampling (inverse CDF) and adjusted residual distribution generators.
* **Member 4 (Optimization & Step-Size Lead):** Builds the 1D optimization module to find the optimal draft sequence length $K^*$ vs temperature.
* **Member 5 (Verification & Latency Analyst):** Proves distribution fidelity (KL-divergence = 0) and plots speedup latency graphs.

---

## 5. Track D: Continuous Graph Neural Diffusion - GRAND (Geometric Deep Learning)

### 📌 Base Paper & Publication Background
* **Paper Title:** *GRAND: Graph Neural Diffusion*
* **Authors:** B. Chamberlain, J. Rowbottom, M. Gorinova, S. Webb, E. Rossi, M. Bronstein (Twitter / Oxford / Imperial)
* **Venue & Date:** *International Conference on Machine Learning (ICML)*, **2021 (Top-Tier AI Conference)**.
* **Recency:** Landmark paper that founded continuous graph neural PDEs, bridging differential geometry with graph deep learning.

### 📄 What the Original Paper Actually Did (Baseline)
* **Core Problem:** Deep Graph Neural Networks suffer from "over-smoothing" (node representations collapse and become indistinguishable after 4–8 layers).
* **Paper's Method:** Formulated graph deep learning as a continuous heat diffusion PDE: $\frac{\partial X(t)}{\partial t} = \text{div}(G(X(t)) \nabla X(t))$, governed by the Graph Laplacian and integrated using continuous numerical ODE solvers.
* **Baseline Benchmark:** Outperformed standard GCN and GAT on benchmark citation graphs (Cora, Citeseer, Pubmed) while enabling arbitrarily deep continuous feature propagation without over-smoothing.

### 🚀 What Extra Work Our Group Will Do (Our Scope of Extension)
1. **Laplacian Spectral Analysis:** Implement custom **Power Method and QR Algorithm** from scratch to extract Graph Laplacian eigenvalues and prove that Dirichlet energy decay is mathematically bounded by the smallest non-zero eigenvalue ($\lambda_2$).
2. **ODE Solver Comparison on Graphs:** Systematically benchmark **Explicit Euler vs. Classical RK4 vs. Implicit Linear Solves** $(I + \Delta t L) X_{t+1} = X_t$ (solved with custom LU decomposition) on classification accuracy vs. integration step size.

### 👥 5-Member Work Breakdown
* **Member 1 (Laplacian & Spectral Lead):** Computes Graph Laplacian and custom QR / Power Method eigenvalue solvers.
* **Member 2 (Explicit ODE Integrator Lead):** Implements forward Euler and RK4 graph diffusion integrators.
* **Member 3 (Implicit Linear Solver Lead):** Implements implicit Euler time-stepping using custom LU / Gauss-Jordan solvers.
* **Member 4 (GNN Pipeline Lead):** Builds PyTorch Geometric training pipeline on node classification benchmarks.
* **Member 5 (Over-Smoothing & Stability Analyst):** Evaluates Dirichlet energy decay and classification accuracy across integration depth $t$.

---

## 6. Track E: Second-Order Optimization in 3D Gaussian Splatting (3D Vision)

### 📌 Base Paper & Publication Background
* **Primary Paper:** *3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt* (Lukas Höllein et al. - Technical University of Munich, **2024**).
* **Companion Paper:** *Matrix-free Second-order Optimization of Gaussian Splats with Residual Sampling* (3DV **2026**).
* **Recency:** Extremely fresh (2024–2026); focuses on accelerating 3D Gaussian Splatting, the dominant paradigm in photorealistic 3D vision.

### 📄 What the Original Paper Actually Did (Baseline)
* **Core Problem:** Standard 3D Gaussian Splatting uses 1st-order Adam optimization, requiring 30,000+ iterations and slow convergence on complex geometries.
* **Paper's Method:** Formulated 3D scene reconstruction as a large-scale non-linear least squares problem, replacing Adam with a customized **Levenberg-Marquardt (LM)** optimizer solving the normal equations $(J^T J + \lambda I) \delta = -J^T r$.
* **Baseline Benchmark:** Achieved 20%–30% faster training time to reach target PSNR reconstruction quality compared to vanilla Adam.

### 🚀 What Extra Work Our Group Will Do (Our Scope of Extension)
1. **Optimization Engine Comparison:** On a small synthetic 3D multi-view benchmark scene, compare **Gradient Descent vs. Gauss-Newton vs. Levenberg-Marquardt** under varying initialization noise.
2. **Damping Parameter ($\lambda$) Sensitivity Analysis:** Analyze how Levenberg-Marquardt damping prevents matrix rank-deficiency and singular collapses in sparse/ill-conditioned camera views.

### 👥 5-Member Work Breakdown
* **Member 1 (3D Scene Lead):** Sets up synthetic multi-view benchmark dataset and 3DGS rasterization pipeline.
* **Member 2 (Jacobian & Residual Lead):** Computes projection residual vectors and Jacobian matrices $J$.
* **Member 3 (Levenberg-Marquardt Optimizer Lead):** Implements custom LM optimizer with adaptive $\lambda$ damping.
* **Member 4 (Linear Solvers Lead):** Implements custom LU decomposition / CG solvers for the normal equations.
* **Member 5 (Rendering & Quality Analyst):** Renders novel view test angles, computes PSNR/SSIM metrics, and plots convergence curves.

---

## 7. Track F: Differentiable Bundle Adjustment in Eager Mode (Robotics & SLAM)

### 📌 Base Paper & Publication Background
* **Paper Title:** *Bundle Adjustment in the Eager Mode*
* **Authors:** J. Zhan, Y. Su, G. Agam, B. Wen
* **Venue & Date:** *IEEE Transactions on Robotics (T-RO)*, **2026 (Flagship IEEE Robotics Journal)**.
* **Recency:** Published 2026; state-of-the-art differentiable robotics framework from Spatial AI & Robotics Lab.

### 📄 What the Original Paper Actually Did (Baseline)
* **Core Problem:** Real-time visual SLAM systems (like DROID-SLAM) use custom CUDA kernels with fixed Gauss-Newton optimization, lacking generalizability and robustness to poor geometric initialization.
* **Paper's Method:** Built a PyTorch-native, differentiable Bundle Adjustment framework using Levenberg-Marquardt and second-order sparse solvers directly in eager execution mode.
* **Baseline Benchmark:** Solved large-scale camera pose and 3D landmark optimization across robotics benchmark datasets (BAL, synthetic scenes).

### 🚀 What Extra Work Our Group Will Do (Our Scope of Extension)
1. **Conditioning & Geometric Degeneracy Study:** Systematically vary camera baseline distance to induce severe ill-conditioning ($\kappa(J^T J) > 10^6$) and measure the exact breakdown point of Gauss-Newton vs. Levenberg-Marquardt.
2. **Linear System Engine Swap:** Compare custom **Gauss Elimination vs. LU Decomposition (with/without partial pivoting)** for solving the normal equations under noisy image observations.

### 👥 5-Member Work Breakdown
* **Member 1 (Synthetic Geometry Lead):** Generates multi-camera synthetic 3D scenes, projection models, and controlled noise perturbations.
* **Member 2 (Optimizer Lead - Gauss-Newton):** Implements Jacobian derivation and Gauss-Newton normal equation solver.
* **Member 3 (Optimizer Lead - Levenberg-Marquardt):** Implements adaptive LM damping and line-search strategies.
* **Member 4 (Linear Solvers & Numerical Analyst):** Builds custom LU and Gauss-Jordan solvers; benchmarks condition numbers and round-off error.
* **Member 5 (Visualization & Benchmarking Lead):** Renders 3D camera/point reconstructions and evaluates convergence speed across noise levels.

---

## 8. Track G: Diffusion ODE Sampling - STORK (Generative AI)

### 📌 Base Paper & Publication Background
* **Paper Title:** *STORK: Faster Diffusion and Flow Matching Sampling by Resolving both Stiffness and Structure-Dependence*
* **Authors:** Z. Tan, Y. Wang, A. Bertozzi, E. Ryu (UCLA / Seoul National University)
* **Venue & Date:** arXiv:2505.24210, **2025 (Open Code on GitHub)**.
* **Recency:** ~1 year recent; specifically addresses numerical analysis stiffness inside modern diffusion models (SD 3.5, FLUX).

### 📄 What the Original Paper Actually Did (Baseline)
* **Core Problem:** Generating images with continuous diffusion models requires integrating a stiff Probability-Flow ODE backwards in time. Standard solvers require 50–100 Number of Function Evaluations (NFE), making sampling slow.
* **Paper's Method:** Formulated diffusion sampling through classical numerical analysis stiffness theory (Burden & Faires) and introduced a class of Stabilized Taylor Orthogonal Runge-Kutta (STORK) integrators.
* **Baseline Benchmark:** Achieved high-fidelity image generation in the 20–50 NFE budget without retraining the base diffusion model.

### 🚀 What Extra Work Our Group Will Do (Our Scope of Extension)
1. **Training-Free Kaggle Evaluation:** Load a pre-trained CIFAR-10 diffusion model and purely modify the reverse sampling loop (zero expensive retraining needed).
2. **ODE Integrator Benchmark:** Benchmark **Euler vs. Heun (RK2) vs. Classical RK4 vs. Stabilized RK** across NFE budgets (10, 20, 50 steps), measuring image quality (FID score) vs. wall-clock sampling latency.

### 👥 5-Member Work Breakdown
* **Member 1 (Euler & Heun Sampler Lead):** Implements low-order ODE sampling routines on the reverse diffusion trajectory.
* **Member 2 (High-Order RK4 Sampler Lead):** Implements classic 4th-order Runge-Kutta diffusion samplers.
* **Member 3 (Stabilized RK Lead):** Implements stiff-ODE stabilized Runge-Kutta solvers.
* **Member 4 (Metric & Image Quality Lead):** Builds the FID / image quality evaluation pipeline on Kaggle GPU.
* **Member 5 (NFE Tradeoff & Report Lead):** Runs step-size sensitivity sweeps and analyzes compute cost vs. perceptual fidelity.

---

## 9. Ready-to-Submit Google Form Drafts

### If Choosing Track A (KAN-ODEs):
* **Project Title:** *Solver-Aware Neural Dynamics: Evaluating Spline Interpolants and Runge-Kutta Integrators in Kolmogorov-Arnold Network ODEs*
* **Base Paper Title:** *KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics (CMAME, 2024)*
* **Initial Idea:** *We will build a KAN-ODE pipeline to learn nonlinear dynamical systems (Lotka-Volterra benchmark). We will extend the paper by systematically investigating its numerical subroutines: swapping default spline basis functions with alternative interpolants (Lagrange, Newton polynomials) and benchmarking forward ODE integration across Euler, RK2, and RK4 solvers to quantify the impact of truncation error on Neural ODE learning stability.*

### If Choosing Track B (Bayesian Epidemic Inference - RECOMMENDED):
* **Project Title:** *Stochastic-Deterministic Synthesis: From Runge-Kutta to Metropolis-Hastings in Bayesian Epidemic Dynamics and Structural Model Validation*
* **Base Paper Title:** *Bayesian workflow for disease transmission modeling in Stan (Statistics in Medicine, 2021)*
* **Initial Idea:** *We will model disease transmission dynamics by solving compartmental ODE systems (SIR/SEIR) using custom-built Euler and RK4 solvers. We will establish deterministic baseline parameters via non-linear least squares, then implement a from-scratch Metropolis-Hastings MCMC algorithm (Poisson likelihood) on the 1978 BMJ influenza dataset. We will study how forward ODE solver truncation errors propagate into MCMC posterior distortion, examine structural model limits (SIR vs SEIR), and deploy the pipeline to forecast real-world Bangladesh 2023 Dengue outbreak dynamics.*

### If Choosing Track C (Speculative Decoding via Metropolis-Hastings in LLMs):
* **Project Title:** *Accelerating Large Language Model Inference via Metropolis-Hastings Rejection Sampling in Speculative Decoding*
* **Base Paper Title:** *Fast Inference from Transformers via Speculative Decoding (ICML, 2023)*
* **Initial Idea:** *We will study speculative sampling in Large Language Models as a discrete Monte Carlo verification process. We will implement the modified Metropolis-Hastings acceptance/rejection rule to verify draft model token sequences against a target model in a single forward pass without altering the target output distribution. We will analyze the acceptance probability under varying temperature and token entropy, and implement an optimization algorithm to determine the optimal draft length K.*

### If Choosing Track D (Continuous Graph Neural Diffusion - GRAND):
* **Project Title:** *Continuous-Time Graph Neural Diffusion: Analyzing Laplacian Eigenvalue Decays and Numerical ODE Solvers in Deep GNNs*
* **Base Paper Title:** *GRAND: Graph Neural Diffusion (ICML, 2021)*
* **Initial Idea:** *We will model deep graph representation learning as a continuous diffusion PDE. We will implement custom QR / Power Method algorithms to compute the Graph Laplacian eigenvalue spectrum and investigate how spectral decay controls over-smoothing. We will compare Explicit Euler, RK4, and Implicit linear solves for integrating the continuous graph ODE forward in time across node classification benchmarks.*

### If Choosing Track E (3D Gaussian Splatting with Levenberg-Marquardt):
* **Project Title:** *Second-Order Levenberg-Marquardt Optimization for Accelerated Convergence in 3D Gaussian Splatting*
* **Base Paper Title:** *3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt (2024)*
* **Initial Idea:** *We will investigate the numerical performance of second-order optimization in 3D Gaussian Splatting novel view synthesis. We will implement a custom Levenberg-Marquardt optimizer solving the normal equations (J^T J + lambda I) delta = -J^T r and compare convergence speed and reconstruction PSNR against first-order Adam on a multi-view benchmark scene.*

### If Choosing Track F (Bundle Adjustment in Eager Mode):
* **Project Title:** *Numerically Robust Differentiable Bundle Adjustment: Optimization, Linear Solvers, and Conditioning in 3D Reconstruction*
* **Base Paper Title:** *Bundle Adjustment in the Eager Mode (IEEE Transactions on Robotics, 2026)*
* **Initial Idea:** *We will investigate the numerical stability of second-order optimizers in 3D bundle adjustment. We will build a synthetic multi-view reconstruction pipeline and compare Gauss-Newton, Levenberg-Marquardt, and gradient descent under varying geometric noise and ill-conditioned camera baselines. We will also benchmark direct linear solvers (LU decomposition vs Gauss-Jordan with pivoting) for solving the normal equations.*

### If Choosing Track G (Diffusion ODEs - STORK):
* **Project Title:** *Stiff ODE Solvers in Generative Diffusion: Accuracy, Stability, and NFE Tradeoffs in Reverse Probability-Flow Sampling*
* **Base Paper Title:** *STORK: Faster Diffusion and Flow Matching Sampling by Resolving both Stiffness and Structure-Dependence (2025)*
* **Initial Idea:** *We will study numerical ODE integration within the reverse probability-flow sampling pass of continuous-time diffusion models. Using a pre-trained image diffusion model, we will benchmark Euler, Heun (RK2), standard RK4, and stabilized Runge-Kutta samplers to measure image quality (FID) vs. Number of Function Evaluations (NFE) across varying step sizes.*

---
*Ready for team discussion and vote.*
