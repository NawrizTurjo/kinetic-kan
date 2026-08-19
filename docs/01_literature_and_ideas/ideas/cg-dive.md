
I read both PDFs fully, and I also checked the important paper-level claims against the actual publications rather than inheriting the PDFs’ rankings. The result is noticeably different from the “KAN-ODEs is automatically #1” conclusion.

## My overall verdict

For a project that needs to look **research-grade, numerically serious, visually impressive, and clearly beyond a routine course implementation**, I would rank the strongest directions:

| Rank        | Base paper / direction                                                               | Project ceiling | Numerical-analysis depth | Implementation risk |      “Posh” factor |
| ----------- | ------------------------------------------------------------------------------------ | --------------: | -----------------------: | ------------------: | -------------------: |
| **1** | **Bundle Adjustment in the Eager Mode — numerical robustness of BA**          |      ★★★★★ |               ★★★★★ |          ★★★★☆ | **★★★★★** |
| **2** | **KAN-ODEs — solver-aware scientific ML**                                     |      ★★★★★ |               ★★★★☆ |          ★★★★☆ | **★★★★★** |
| **3** | **SIR + Bayesian inference — deterministic vs stochastic epidemic inference** |      ★★★★★ |               ★★★★☆ |          ★★★☆☆ | **★★★★★** |
| **4** | **STORK / diffusion ODE sampling**                                             |      ★★★★★ |               ★★★★★ |          ★★★★★ | **★★★★★** |
| **5** | **Gillespie vs deterministic ODE modeling**                                    |      ★★★★☆ |               ★★★★★ |          ★★☆☆☆ |           ★★★★☆ |
| **6** | **MPPT + numerical optimization/root finding**                                 |      ★★★★☆ |               ★★★★★ |          ★★☆☆☆ |           ★★★★☆ |
| **7** | **Fast Otsu**                                                                  |      ★★★☆☆ |               ★★★★☆ |          ★☆☆☆☆ |           ★★★★☆ |
| **8** | **NaSch traffic CA**                                                           |      ★★★★☆ |               ★★★☆☆ |          ★☆☆☆☆ |           ★★★★☆ |
| **9** | **Ising / Metropolis**                                                         |      ★★★★☆ |               ★★★★☆ |          ★☆☆☆☆ |           ★★★★☆ |

And there is one **very important correction** to the research in the PDF:

> The KAN-ODE paper does **not** use B-spline interpolation in its actual KAN implementation.

The paper explicitly says that, although the original KAN work used B-splines, **their KAN-ODE implementation uses Gaussian radial basis functions (RBFs)**. ([NSF Public Access Repository][1])

So the first PDF's statement that KAN-ODEs itself gives you “B-spline interpolation + RK ODE integration” is not correct.

That substantially changes how I would evaluate it.

---

# 1. My strongest recommendation: Bundle Adjustment

### Base paper

**Zhan et al., “Bundle Adjustment in the Eager Mode”**

The paper is now a **2026 IEEE Transactions on Robotics paper**, not merely the 2024 arXiv preprint the PDF treated it as. The authors also provide code. ([Spatial AI & Robotics Lab][2])

This is an exceptionally good base paper for your course because the underlying problem is:

[
\min_\theta \sum_{i,j}|r_{ij}(\theta)|^2
]

which immediately gives you **nonlinear least squares**, followed by an iterative numerical optimizer.

The paper explicitly formulates bundle adjustment as nonlinear least squares and uses **Levenberg–Marquardt**, with the update requiring a linear system:

[
(J^TJ+\lambda,\mathrm{diag}(J^TJ))\Delta\theta=-J^Tr
]

and discusses its relationship to Gauss–Newton and gradient descent. ([ResearchGate][3])

That is almost tailor-made for your syllabus.

### Why this one is unusually good

Your syllabus contains:

* least-squares fitting
* Gauss elimination
* LU decomposition
* Newton's method / optimization
* gradient methods
* systems of equations
* round-off error
* approximation
* model validation
* visualization

Bundle adjustment naturally touches **several of these simultaneously**.

Instead of making a project where the numerical method is merely hidden inside some modern AI system, you can make the numerical method itself the protagonist.

---

# The project I would actually build

## **Numerically Robust Bundle Adjustment: A Study of Optimization, Linear Solvers, Conditioning, and Noise in 3D Reconstruction**

The base paper provides the BA framework.

Your contribution is not:

> “We implemented bundle adjustment.”

That's too ordinary.

Your contribution becomes:

> **How does the numerical solver affect the stability and accuracy of differentiable bundle adjustment under increasingly difficult geometric and numerical conditions?**

That is a real numerical-analysis question.

### Your experiment becomes something like this

Generate a synthetic 3D scene:

```text
        • • • • •
      • • • • • • •
    • • • • • • • • •
        3D points
          ↓
     ┌───────────┐
     │ Camera 1  │
     └───────────┘
          ↓
     ┌───────────┐
     │ Camera 2  │
     └───────────┘
```

Project the points into multiple cameras.

Add controlled noise.

Start with bad camera/point estimates.

Then solve the same BA problem using:

**A. Gauss–Newton**

[
(J^TJ)\Delta=-J^Tr
]

**B. Levenberg–Marquardt**

[
(J^TJ+\lambda D)\Delta=-J^Tr
]

**C. Gradient descent**

[
\theta_{k+1}=\theta_k-\alpha\nabla L
]

And then, crucially, investigate **how the inner linear system is solved**:

* Gaussian elimination
* LU decomposition
* possibly different pivoting strategies

Now you've suddenly connected **three separate chunks of the syllabus** into one modern robotics system.

---

# And then comes the part that makes it stand out

Don't merely ask:

> Which method is fastest?

Ask:

### **When does each method fail?**

Systematically vary:

### 1. Measurement noise

[
\sigma = 0,;0.1,;0.5,;1,;2,\ldots
]

### 2. Initial-guess quality

Excellent initialization → mediocre → terrible.

### 3. Camera baseline

Very small baseline makes depth poorly constrained.

That creates a genuinely interesting **conditioning** problem.

### 4. Number of observations

10 points → 50 → 100 → 500 → ...

### 5. Degenerate geometry

Nearly collinear / poorly distributed points.

### 6. Linear-system conditioning

Measure:

[
\kappa(J^TJ)
]

and correlate it with solver instability.

This is where the project stops looking like a normal undergraduate “we ran three algorithms” project.

You are studying:

> **optimization behavior as a function of numerical conditioning.**

That's very numerical-analysis-ish.

---

# An even better extension

Create a **condition-aware hybrid optimizer**.

For example:

```text
           compute J
              ↓
       condition estimate
              ↓
     ┌────────┴────────┐
     ↓                 ↓
 well-conditioned    ill-conditioned
     ↓                 ↓
 Gauss-Newton        LM damping
     └────────┬────────┘
              ↓
          update θ
```

The hypothesis becomes:

> Gauss–Newton is efficient near well-conditioned solutions, while LM becomes more reliable under poor conditioning or poor initialization.

Then you empirically determine whether your hypothesis is actually true.

That is much more research-flavored than simply reproducing the original paper.

The original BA paper itself emphasizes second-order optimization and sparse optimization infrastructure, and its framework goes beyond BA toward other sparse optimization problems such as pose-graph optimization. ([Spatial AI & Robotics Lab][4])

---

# Why I rank this above KAN-ODE

The PDFs' strongest KAN-ODE argument was:

> spline interpolation + RK ODE = two syllabus topics.

But that premise is wrong for this paper.

The actual paper says its implementation uses **Gaussian RBF basis functions**, not B-splines. ([NSF Public Access Repository][1])

The paper absolutely does use an ODE solver, and it is experimentally interesting: the Lotka–Volterra experiment uses Tsit5, an adaptive ODE integrator, and compares a 240-parameter KAN-ODE to a 252-parameter Neural ODE. ([NSF Public Access Repository][1])

The paper reports the KAN-ODE reaching lower training loss than the comparable Neural ODE in its experiment, and it also explores sparsification and symbolic regression. ([NSF Public Access Repository][1])

So KAN-ODE is still excellent.

But your numerical-analysis project would probably become:

> compare Euler/RK2/RK4/adaptive RK inside KAN-ODE.

That's good.

The BA project lets you construct an entire numerical investigation around:

> optimization + nonlinear least squares + linear systems + conditioning + stability + round-off + convergence.

That is a much richer match to your syllabus.

---

# 2. KAN-ODE is still a fantastic second choice

Actually, I would transform it significantly from what the PDF proposed.

The paper is a legitimate peer-reviewed **Computer Methods in Applied Mechanics and Engineering** publication, volume 432, article 117397, published in 2024. ([NSF Public Access Repository][1])

And the core experiment is very reproducible:

* Lotka–Volterra
* tiny model
* 240 parameters
* synthetic data
* adaptive ODE integration
* training vs extrapolation

The paper explicitly used Tsit5 for its ground-truth integration and KAN-ODE solution. ([NSF Public Access Repository][1])

### Your improved project

## **Solver-Aware Neural Dynamics: How Numerical Integration Controls the Reliability of Neural ODEs**

Rather than pretending the paper is about spline interpolation, make the **ODE solver** the numerical centerpiece.

Compare:

[
Euler
]

[
RK2
]

[
RK4
]

[
\text{adaptive RK}
]

under:

* step size
* noisy observations
* sparse observations
* long-term extrapolation
* stiffened dynamics
* computational cost

Then measure:

[
\text{global error}
]

[
\text{training loss}
]

[
\text{forecast error}
]

[
\text{NFE}
]

[
\text{runtime}
]

[
\text{stability}
]

That gives you a beautiful central question:

> **Does a more accurate numerical integrator actually produce a more reliable learned dynamical model?**

That's a genuinely interesting scientific-ML question.

---

# 3. SIR + Bayesian inference is deceptively powerful

The second PDF's SIR recommendation is genuinely strong. The underlying paper is peer-reviewed **Statistics in Medicine (2021)** and explicitly presents a Bayesian workflow for disease-transmission models. ([Wiley Online Library][5])

And the project naturally connects:

[
\text{ODE}
+
\text{numerical integration}
+
\text{parameter fitting}
+
\text{MCMC}
+
\text{uncertainty}
+
\text{model validation}
]

That's fantastic.

But there is another important correction.

The base paper uses **Stan's Hamiltonian Monte Carlo / NUTS**, not literal Metropolis-Hastings. The PDF itself admits this.

So the project would partly be:

> reproduce the model, then replace/augment the inference mechanism with your own Metropolis-Hastings implementation.

That's completely reasonable, but it makes the relation to the base paper slightly less direct.

### Still, this could become an absolutely beautiful project

Instead of:

> “We fitted an SIR model.”

Do:

## **When Deterministic Epidemic Models Lie: Numerical Solver and Bayesian Uncertainty Analysis of Compartmental Epidemic Models**

Compare:

**Deterministic parameter fitting**

versus

**Bayesian parameter inference**

and then:

[
Euler \leftrightarrow RK4
]

and perhaps:

[
SIR \leftrightarrow SEIR
]

while reporting posterior uncertainty in:

[
\beta,\gamma,R_0
]

The really interesting part is **model inadequacy**.

The PDF correctly points out that the famous 1978 English boarding-school influenza data are not perfectly fitted by simple compartmental models, and recent work explicitly discusses the failure of the classic SEIR model on those data.

That gives you something much more sophisticated than:

> "our graph matches the paper."

You can investigate:

> **Is the numerical solver wrong, is the parameter estimate wrong, or is the model itself wrong?**

That is excellent scientific modeling.

---

# 4. STORK is probably the flashiest, but I'd be careful

This is the **“holy shit, these students did diffusion models”** option.

STORK is explicitly built around Runge–Kutta methods for diffusion/flow-matching ODE sampling, exactly in the numerical-analysis territory your syllabus covers. The PDF describes comparing Euler/Heun/RK4/stabilized RK at different NFE budgets.

The concept is excellent.

You could make:

## **Numerical Solver Choice in Diffusion Sampling: Accuracy–NFE–Stability Tradeoffs**

Compare:

[
Euler
]

[
Heun
]

[
RK4
]

[
STORK
]

against:

* NFE
* image quality
* FID / another quality measure
* runtime
* stability

This would look extremely modern.

### Why it isn't #1

Because the **full generative pipeline introduces a lot of experimental baggage**.

The numerical analysis can easily become secondary to:

> “we made a diffusion model work.”

That's exactly what I would avoid.

It is a fantastic project **only if you deliberately keep the pretrained model fixed and make the ODE solver the star**.

---

# 5. Gillespie is an underrated monster choice

The Gillespie paper is genuine classic literature: Daniel Gillespie's 1977 paper in *The Journal of Physical Chemistry*. ([DOI][6])

And it has an extremely elegant numerical-analysis story:

### Deterministic world

[
\frac{dx}{dt}=f(x)
]

solve with RK4.

### Stochastic world

simulate individual events:

[
\tau \sim \mathrm{Exponential}(a_0)
]

and select the next reaction based on propensity.

Then compare:

[
\text{Gillespie}
\quad\text{vs}\quad
\text{Euler/RK4}
]

on the same system.

That's powerful because you're not merely comparing algorithms.

You're asking:

> **When does the deterministic approximation stop being a faithful representation of the underlying stochastic process?**

That's a proper modeling question.

For example:

## **When Do ODE Models Break? A Numerical and Stochastic Study of Epidemic Dynamics**

Run stochastic SIR against deterministic SIR.

Then investigate:

* population size
* stochastic variance
* extinction probability
* mean trajectory
* variance around mean trajectory
* runtime
* number of realizations

This is highly aligned with the “simulation & modeling” half of the course.

And it is much less likely to collapse under implementation problems.

---

# 6. MPPT is one of the safest genuinely numerical projects

The PV paper is real and peer-reviewed in *Results in Engineering*. Its actual contribution is a predictor-corrector MPPT algorithm under changing irradiation and temperature. ([Directory of Open Access Journals][7])

The broader literature confirms that numerical methods such as Newton–Raphson, bisection, false position and related methods are genuinely used for PV MPP determination. ([ScienceDirect][8])

This could make a very clean project:

## **Numerical Optimization of Maximum Power Point Tracking under Rapidly Changing Solar Conditions**

Compare:

* Bisection
* False Position
* Newton-Raphson
* Golden-section
* Predictor-corrector

under:

* changing irradiation
* changing temperature
* noisy measurements
* rapidly changing operating conditions

Measure:

[
\text{iterations}
]

[
\text{tracking error}
]

[
\text{oscillation}
]

[
\text{convergence}
]

[
\text{energy captured}
]

This is probably one of the best projects if the professor is **very traditional about numerical analysis**.

But it doesn't have quite the same “modern research” aura as differentiable BA / scientific ML.

---

# 7. Fast Otsu is clever, but I'd reject it as your final project

The paper really exists and reports exactly the kinds of results cited in the PDF: it proposes replacing exhaustive Otsu threshold search with a bisection-like procedure and reports substantial reductions in evaluations/iterations on 48 images. ([arXiv][9])

But there is an important conceptual issue that the PDF itself noticed:

> calling this “bisection” is mathematically questionable.

True bisection is fundamentally a **root-finding algorithm relying on a sign-changing bracket**.

Otsu's problem here is a **unimodal maximization problem**.

So you are going to spend part of your project explaining why something called “bisection” is not really standard bisection.

That's interesting academically, but I wouldn't build your flagship project around a terminology controversy.

---

# 8. NaSch and Ising are safe, but too easy for your ambitions

These are excellent simulation projects.

Nagel–Schreckenberg is a genuine foundational paper in stochastic cellular-automaton traffic modeling. ([ResearchGate][10])

Metropolis's 1953 paper is genuinely the seminal Monte Carlo paper cited in the PDF, describing a modified Monte Carlo integration method for interacting particles. ([DOI][11])

They're scientifically beautiful.

But here's the problem:

A competent student can produce the core simulation in a weekend.

So unless you go very deep into:

* finite-size scaling
* critical phenomena
* autocorrelation
* phase transitions
* parameter sensitivity
* uncertainty quantification

the project presentation risks becoming:

> “Here is our simulation. Look, the curve looks like the paper.”

Given your stated objective, I'd avoid that.

---

# What I would **not** choose

### Pure Schelling

Too easy unless you add serious quantitative analysis. Even the PDF acknowledges this risk.

### Pure Monte Carlo option pricing

Very reproducible, but likely many groups will arrive at something similar.

### Pure Ising

Too classic.

### Pure traffic CA

Too straightforward.

### Pure KAN

Too ML-heavy and numerically shallow unless the numerical-method question is deliberately designed.

### Pure “compare five numerical methods”

Dangerous.

That becomes exactly what your project announcement calls an exploratory/comparative project, but without a meaningful research question it may look like:

> “We implemented everything from Chapter 3.”

The **comparison needs to explain a phenomenon**, not merely produce a leaderboard.

---

# The project design I think could genuinely beat most groups

I would structure your project around this philosophy:

## **Don't make the numerical method a tool inside the project. Make the numerical method the research question.**

That's the difference.

For example, weak:

> “We use Levenberg–Marquardt to reconstruct a 3D scene.”

Strong:

> **“How does numerical conditioning determine the stability of second-order optimization in 3D bundle adjustment?”**

Weak:

> “We compare RK4 and Euler in a Neural ODE.”

Strong:

> **“When does numerical integration accuracy translate into better learned dynamical-system generalization?”**

Weak:

> “We use Metropolis-Hastings to estimate SIR parameters.”

Strong:

> **“Can stochastic inference distinguish numerical error, parameter uncertainty, and structural model error in epidemic dynamics?”**

That framing is what makes the project feel like a research project rather than a course assignment.

---

# My final ranking for **your specific goal**

### 🥇 **Bundle Adjustment — Numerical Robustness & Conditioning**

**Base:** *Bundle Adjustment in the Eager Mode*, IEEE T-RO 2026. ([Spatial AI & Robotics Lab][2])

**Why:** Maximum syllabus coverage + modern robotics + mathematically serious + excellent visualization + huge extension space.

**Core numerical story:**

[
\text{Nonlinear Least Squares}
\rightarrow
\text{GN / LM / Gradient}
\rightarrow
\text{Linear Systems}
\rightarrow
\text{Conditioning}
\rightarrow
\text{Stability}
]

This is my **#1**.

---

### 🥈 **KAN-ODE — Solver-Aware Scientific Machine Learning**

**Base:** *KAN-ODEs*, CMAME 2024. ([NSF Public Access Repository][1])

**Why:** Extremely modern, visually attractive, connects numerical analysis with scientific ML.

But critically, **do not sell it as “B-spline + RK”**. The paper's implementation uses RBFs. ([NSF Public Access Repository][1])

---

### 🥉 **SIR + Bayesian inference**

**Base:** Grinsztajn et al., *Statistics in Medicine* 2021. ([Wiley Online Library][5])

**Why:** Probably the best combination of modeling, uncertainty, ODE solving, stochastic simulation and real data.

It has an unusually good research narrative because model inadequacy itself becomes part of the investigation.

---

### 4. **Gillespie + RK4**

Probably the best “pure simulation & modeling” project.

### 5. **STORK / diffusion solvers**

Highest visual wow factor, highest execution risk.

### 6. **PV MPPT numerical optimization**

Safest high-quality traditional numerical-analysis project.

### 7. **NaSch / Ising**

Excellent but harder to make exceptional.

### 8. **Fast Otsu**

Clever and very easy, but not enough intellectual surface area for the project you're describing.

---

# And there is one more thing I would do

Your final project should ideally have **three layers**:

### Layer 1 — Reproduction

Reproduce the base paper's numerical core.

### Layer 2 — Controlled numerical investigation

Change one numerical property at a time and measure:

[
\text{accuracy, convergence, stability, conditioning, runtime}
]

### Layer 3 — Your own numerical contribution

Introduce a principled modification and test whether it actually improves something.

That gives the report a natural structure:

> **Base paper → numerical weakness/question → controlled experiments → proposed improvement → rigorous evaluation**

That's much stronger than:

> **Paper → implementation → graphs**

---

## The strongest concrete title I'd put on the submission form

**“Numerically Robust Differentiable Bundle Adjustment: A Study of Optimization, Linear Solvers, Conditioning, and Stability”**

Base paper: **Zhan et al., “Bundle Adjustment in the Eager Mode,” IEEE Transactions on Robotics, 2026.** ([Spatial AI & Robotics Lab][2])

And I'd keep **KAN-ODE** as your backup, not because it is weaker, but because the original research notes overstated its direct syllabus overlap by incorrectly treating the paper's RBF basis as spline interpolation.

That correction alone is exactly why I would **not blindly follow the AI-generated ranking in the PDFs**.

[1]: https://par.nsf.gov/servlets/purl/10580154
[2]: https://sairlab.org/bae/?utm_source=chatgpt.com
[3]: https://www.researchgate.net/publication/384116490_Bundle_Adjustment_in_the_Eager_Mode?utm_source=chatgpt.com
[4]: https://sairlab.org/bae/
[5]: https://onlinelibrary.wiley.com/doi/10.1002/sim.9164?utm_source=chatgpt.com
[6]: https://doi.org/doi%3A10.1021/j100540a008?utm_source=chatgpt.com
[7]: https://doaj.org/article/2b4ad50738e0402faf50464fd784f4cf?utm_source=chatgpt.com
[8]: https://www.sciencedirect.com/science/article/abs/pii/S1755008420300685?utm_source=chatgpt.com
[9]: https://arxiv.org/abs/2509.16179?utm_source=chatgpt.com
[10]: https://www.researchgate.net/publication/263504490_A_cellular_automaton_model_for_freeway_traffic
[11]: https://doi.org/10.1063/1.1699114?utm_source=chatgpt.com
