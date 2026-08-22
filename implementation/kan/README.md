# Kolmogorov-Arnold Network (KAN) Architecture & Implementation

This directory contains the PyTorch implementation of the **Kolmogorov-Arnold Network (KAN)** modules used as the core parameterized vector field in the KAN-ODE framework, mirroring the formulation described in the paper:
> **"KAN-ODEs: Kolmogorov–Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics"**  
> *Benjamin C. Koenig, Suyong Kim, Sili Deng (CMAME / arXiv:2407.04192)*

---

## Table of Contents
1. [Mathematical Foundations & MLP vs. KAN](#1-mathematical-foundations--mlp-vs-kan)
2. [Why Are Grids Necessary in KAN?](#2-why-are-grids-necessary-in-kan)
3. [Network Topology: Why `[2, 10, 2]` & Parameter Breakdown](#3-network-topology-why-2-10-2--parameter-breakdown)
4. [How KAN-ODE Learns Dynamics (Lotka-Volterra Case Study)](#4-how-kan-ode-learns-dynamics-lotka-volterra-case-study)
5. [Backpropagation in Neural ODEs: Direct vs. Adjoint Sensitivity Method](#5-backpropagation-in-neural-odes-direct-vs-adjoint-sensitivity-method)
6. [Architecture Overview & Workflow](#6-architecture-overview--workflow)
7. [Module Breakdown](#7-module-breakdown)
   - [basis.py (Basis Functions)](#basispy-basis-functions)
   - [layer.py (KDense Layer)](#layerpy-kdense-layer)
   - [model.py (KAN Multi-Layer Network)](#modelpy-kan-multi-layer-network)
8. [Model Parameters & Hyperparameters Reference](#8-model-parameters--hyperparameters-reference)
9. [Tensor Shapes & Step-by-Step Data Flow](#9-tensor-shapes--step-by-step-data-flow)
10. [Interpretability & Activation Extraction](#10-interpretability--activation-extraction)
11. [Regularization & Sparsification](#11-regularization--sparsification)
12. [Code Usage Examples](#12-code-usage-examples)

---

## 1. Mathematical Foundations & MLP vs. KAN

### Standard MLP vs. KAN
| Concept | Standard MLP / Linear Layer | Kolmogorov-Arnold Network (KAN) |
|---|---|---|
| **Edge Function** | Linear multiplication: $y = w \cdot x$ | **Learnable 1D continuous curve $\phi(x)$** |
| **Node (Neuron)** | Fixed non-linear activation $\sigma(\sum \text{edges})$ | **Simple sum $\sum \text{edges}$** |
| **Non-linearity Source** | Global node activations (ReLU, GELU, Tanh) | **Localized 1D basis functions along every edge** |
| **Interpretability** | Black box (entangled weights) | **White box (each edge is an explicit 1D curve)** |

---

### The Kolmogorov-Arnold Representation Theorem (KAT)

In 1957, mathematician Andrey Kolmogorov (alongside Vladimir Arnold solving Hilbert's 13th problem) proved that any multivariate continuous function $f(\mathbf{x})$ defined on a bounded domain $[0, 1]^{d_{\text{in}}}$ can be written as a finite composition of univariate (1D) continuous functions and addition:

$$
f(\mathbf{x}) = \sum_{q=1}^{2d_{\text{in}}+1} \Phi_q \left( \sum_{p=1}^{d_{\text{in}}} \phi_{q,p}(x_p) \right)
$$

#### Step-by-Step Breakdown of the Equation:
1. **Input Vector $\mathbf{x}$**: $\mathbf{x} = [x_1, x_2, \dots, x_{d_{\text{in}}}]^T$ with $d_{\text{in}}$ scalar inputs.
2. **Inner 1D Functions ($\phi_{q,p}$)**: Each scalar component $x_p$ is transformed independently by a continuous 1D function $\phi_{q,p}: \mathbb{R} \to \mathbb{R}$.
3. **Inner Summation**: The $d_{\text{in}}$ transformed 1D variables are added together to create an intermediate scalar:
   $$
   u_q = \sum_{p=1}^{d_{\text{in}}} \phi_{q,p}(x_p)
   $$
4. **Outer 1D Functions ($\Phi_q$)**: Each intermediate scalar $u_q$ is transformed by another continuous 1D function $\Phi_q: \mathbb{R} \to \mathbb{R}$.
5. **Outer Summation**: All $(2d_{\text{in}} + 1)$ outer branches are summed together to produce the scalar output $f(\mathbf{x})$.

---

### Why a Summation Over $(2d_{\text{in}} + 1)$ is Required?

The number $(2d_{\text{in}} + 1)$ is not arbitrary—it is a **fundamental result from topological dimension theory**:

1. **Topological Embedding Dimension (Menger-Nöbeling & Ostrand Theorems)**:
   - In topology, a compact $n$-dimensional metric space cannot always be embedded without self-intersections in $\mathbb{R}^{2n}$, but it is **guaranteed to be embeddable in $\mathbb{R}^{2n+1}$**.
   - To project points from an $n$-dimensional cube $[0, 1]^n$ down to 1D lines using continuous coordinates while ensuring that **no two distinct points collide ($\mathbf{x}_1 \neq \mathbf{x}_2 \implies \mathbf{u}_1 \neq \mathbf{u}_2$)**, you mathematically need at least $2n + 1$ coordinate projections.
2. **Separating Points Without Ambiguity**:
   - If you had fewer than $2d_{\text{in}} + 1$ terms (e.g. $\le 2d_{\text{in}}$), there would always exist pairs of distinct multi-dimensional points that produce identical inner sums $\sum_p \phi_{q,p}(x_p)$ across all $q$, making it impossible for the outer function $\Phi_q$ to assign them different output values.
   - $(2d_{\text{in}} + 1)$ is the **exact minimum number of 1D branches** needed to guarantee that every unique input $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$ is mapped uniquely without information loss.

---

### How Many Layers Are in the KAT Equation?

The original 1957 theorem represents a strictly **2-layer (depth-2) architecture**:

```
Input Layer (d_in nodes) ──[ Inner 1D functions ϕ_{q,p} ]──► Hidden Layer (2*d_in + 1 nodes)
Hidden Layer (2*d_in + 1 nodes) ──[ Outer 1D functions Φ_q ]──► Output Layer (1 node)
```

- **Layer 1 (Inner Layer)**: Connects $d_{\text{in}}$ input nodes to $(2d_{\text{in}} + 1)$ hidden nodes using 1D edge functions $\phi_{q,p}(x_p)$.
- **Layer 2 (Outer Layer)**: Connects $(2d_{\text{in}} + 1)$ hidden nodes to $1$ output node using 1D edge functions $\Phi_q(u_q)$.

#### Why Classical KAT Failed in Practice & How Modern KANs Solved It:
- **The Classical Bottleneck**: In the 2-layer theorem, forcing an arbitrary function into just 2 layers requires the 1D functions $\phi_{q,p}$ and $\Phi_q$ to be **extremely non-smooth, fractal-like, and pathological**, making them impossible to fit with gradient descent.
- **The Modern KAN Breakthrough (Liu et al., 2024; Koenig et al., 2024)**:  
  Instead of restricting ourselves to 2 layers with pathological 1D functions, modern KANs generalize the Kolmogorov-Arnold principle to **arbitrary depth $L$ and arbitrary layer widths $[d_0, d_1, d_2, \dots, d_L]$** with **smooth, differentiable spline / RBF basis functions**:
  $$
  \mathbf{y} = \left(\mathbf{\Phi}_{L} \circ \mathbf{\Phi}_{L-1} \circ \cdots \circ \mathbf{\Phi}_{1}\right)(\mathbf{x})
  $$
  Stacking multiple smooth KAN layers allows the network to represent complex physical interactions (like Lotka-Volterra dynamics) smoothly, stably, and with high interpretability!

---

### Discrete Formulation in `KDense` Layers

In this codebase, each individual KAN layer (`KDense`) implements:

$$
y_j = \sum_{i=1}^{d_{\text{in}}} \phi_{i,j}(x_i)
$$

where each 1D edge function $\phi_{i,j}(x)$ is parameterized as:

$$
\phi_{i,j}(x_i) = \underbrace{\sum_{g=1}^{G} c_{j, i, g} \cdot \psi_g(\text{normalizer}(x_i))}_{\text{Spline / Basis Path (Localized Curves)}} + \underbrace{w_{j, i} \cdot \text{base\_act}(x_i)}_{\text{Residual Base Path (Global Stability)}}
$$

---

## 2. Why Are Grids Necessary in KAN?

### The Core Problem
In a standard MLP, an edge only learns **a single scalar weight** $w$. But in a KAN, an edge needs to learn an **entire continuous 1D function** $\phi(x)$ (which could be $\sin(x)$, $x^2$, $e^{-x}$, or any arbitrary non-linear curve).

How do you give a neural network the freedom to learn *any arbitrary 1D shape* without hardcoding mathematical formulas?

### The Solution: A Discrete 1D Grid with Localized Basis Functions
To synthesize arbitrary curves, KAN divides the 1D input domain into a sequence of anchor points called a **Grid**:

```
        Basis Bell Curves centered at each Grid knot (z_g)
               ψ₁(x)       ψ₂(x)       ψ₃(x)       ψ₄(x)       ψ₅(x)
              /     \     /     \     /     \     /     \     /     \
             /   •   \   /   •   \   /   •   \   /   •   \   /   •   \
       ─────┴─────────┴─────────┴─────────┴─────────┴─────
         z₁ = -1.0  z₂ = -0.5  z₃ = 0.0   z₄ = 0.5   z₅ = 1.0  <-- (Grid centers)
       |◄─────────────────── grid_lims = (-1.0, 1.0) ───────────────────►|
```

1. **Anchor Points ($\mathbf{z}$)**:
   A fixed 1D array of $G$ knot centers (e.g. $[-1.0, -0.5, 0.0, 0.5, 1.0]$ for `grid_len=5` and `grid_lims=(-1.0, 1.0)`).
2. **Localized Basis Kernels ($\psi_g$)**:
   At each anchor point $z_g$, we place a localized bell curve (e.g. Gaussian Radial Basis Function $\psi_g(u) = \exp(-((u - z_g)/h)^2)$).
3. **Learnable Coefficients ($c_g$)**:
   The network assigns a trainable weight $c_g$ to each bell curve. By adjusting these amplitudes during gradient descent, the sum $\sum c_g \psi_g(u)$ can smoothly morph into **any 1D mathematical function**.

### Why Not Just Use Global Polynomials (like $x, x^2, x^3$)?
- **Local Control vs. Global Interference**: Localized basis functions (RBF/Splines) only activate when $x$ is near knot $z_g$. Updating $c_g$ modifies the curve *locally* without distorting the function in distant regions (avoiding catastrophic forgetting / Runge's phenomenon).
- **Analogy to Numerical Methods**: This is identical to **Finite Element Methods (FEM)** or **Spline Interpolation** in numerical analysis, where complex curves are represented by piecewise local basis elements.

---

## 3. Network Topology: Why `[2, 10, 2]` & Parameter Breakdown

### 1. Why `layers_hidden = [2, 10, 2]`?

- **Input Dimension ($d_{\text{in}} = 2$)**:  
  Represents the 2 instantaneous physical states $[x(t), y(t)]$ of the dynamical system (prey $x$, predator $y$).
- **Output Dimension ($d_{\text{out}} = 2$)**:  
  Represents the 2 time derivatives $\left[\frac{dx}{dt}, \frac{dy}{dt}\right]$.
- **Hidden Dimension ($h_1 = 10$) — Why a hidden layer is essential**:
  - The true Lotka-Volterra dynamics contains **multiplicative interaction terms** ($x \cdot y$):
    $$\frac{dx}{dt} = \alpha x - \beta x y, \quad \frac{dy}{dt} = \delta x y - \gamma y$$
  - A single-layer KAN $[2 \to 2]$ can only compute additive 1D operations: $\phi_1(x) + \phi_2(y)$. It **cannot compute multiplications like $x \cdot y$** directly!
  - By adding a hidden layer $[2 \to 10 \to 2]$, the first layer computes 10 intermediate non-linear feature combinations (analogous to $\ln(x), \ln(y)$ or $(x+y)^2, (x-y)^2$), and the second layer recombines them:
    $$x \cdot y = \frac{(x+y)^2 - (x-y)^2}{4}$$
  - A hidden width of $10$ provides sufficient expressive capacity for symbolic basis transformations without overfitting.

---

### 2. Exact Trainable Parameter Count Calculation

For a 2-layer network `[2, 10, 2]` with `grid_len = 5` and `use_base_act = True`:

```
Layer 0: KDense(in_features=2, out_features=10, grid_len=5)
├── Spline Weights C_0: shape [10, 2 * 5] = [10, 10] ──► 100 parameters
└── Base Weights W_0:   shape [10, 2]                ──►  20 parameters
                                             Subtotal ──► 120 parameters

Layer 1: KDense(in_features=10, out_features=2, grid_len=5)
├── Spline Weights C_1: shape [2, 10 * 5] = [2, 50]  ──► 100 parameters
└── Base Weights W_1:   shape [2, 10]                ──►  20 parameters
                                             Subtotal ──► 120 parameters
────────────────────────────────────────────────────────────────────────
GRAND TOTAL TRAINABLE PARAMETERS                      ──► 240 parameters
```

$$\text{Total Parameters} = \sum_{l=0}^{L-1} \left( d_{l+1} \cdot d_l \cdot G + d_{l+1} \cdot d_l \right) = (10 \cdot 2 \cdot 5 + 10 \cdot 2) + (2 \cdot 10 \cdot 5 + 2 \cdot 10) = 120 + 120 = \mathbf{240}$$

---

## 4. How KAN-ODE Learns Dynamics (Lotka-Volterra Case Study)

A common point of confusion for beginners is: **"How does the model train on $t \in [0.0, 3.5]\text{s}$ and predict all the way to $t = 14.0\text{s}$?"**

### 1. What KAN-ODE Actually Learns
KAN-ODE does **NOT** do direct time-series curve fitting (i.e. it does not map $t \to \mathbf{y}$).  
Instead, KAN learns the **governing differential equation (velocity vector field)**:

$$\frac{d\mathbf{y}}{dt} = f_{\theta}(\mathbf{y}(t)), \quad \text{where } \mathbf{y}(t) = \begin{bmatrix} x(t) \\ y(t) \end{bmatrix} = \begin{bmatrix} \text{Prey} \\ \text{Predator} \end{bmatrix}$$

- **Input to KAN**: The current state of the system $\mathbf{y} = [x, y]$ (populations).
- **Output from KAN**: The instantaneous rates of change $\left[\frac{dx}{dt}, \frac{dy}{dt}\right]$.

```
               ┌───────────────────────────┐
  State [x, y] │   KAN Model (f_θ)         │   Derivatives [dx/dt, dy/dt]
──────────────►│ Learnable 1D edge curves  ├──────────────────────────────►
               └───────────────────────────┘
```

### 2. The Training Phase ($t \in [0.0, 3.5]\text{s}$)
In the benchmark dataset:
- We observe the predator-prey system from $t = 0.0$ to $t = 3.5$ (roughly **the first 1/4 of a single periodic cycle**).
- **Integration Step**:
  The numerical ODE solver (e.g. `Tsit5` or `RK4`) starts at initial condition $\mathbf{y}(0) = [1.0, 1.0]^T$.  
  At each sub-step, the solver calls KAN $f_\theta(\mathbf{y})$ to obtain velocities and computes the trajectory:
  $$\hat{\mathbf{y}}(t) = \mathbf{y}(0) + \int_0^t f_{\theta}(\mathbf{y}(\tau)) \, d\tau$$
- **Loss & Backpropagation**:
  The Mean Squared Error (MSE) between simulated trajectory $\hat{\mathbf{y}}(t)$ and true data $\mathbf{y}(t)$ is computed over $t \in [0.0, 3.5]$:
  $$\mathcal{L}_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{k=1}^{N_{\text{train}}} \|\hat{\mathbf{y}}(t_k) - \mathbf{y}(t_k)\|^2$$
  Autograd backpropagates through the ODE integrator to adjust the KAN spline weights $\mathbf{C}$ and base weights $\mathbf{W}$.

### 3. The Testing / Extrapolation Phase ($t \in [0.0, 14.0]\text{s}$)
- Once training completes, KAN has learned the *true continuous vector field* $f_\theta(\mathbf{y}) \approx \begin{bmatrix} \alpha x - \beta x y \\ \delta x y - \gamma y \end{bmatrix}$.
- To evaluate extrapolation, we pass the learned KAN into the ODE solver and integrate forward from $t = 0.0$ all the way to $t = 14.0$.
- Because KAN learned the **underlying law of physics** rather than memorizing time coordinates, the solver seamlessly generates **multiple future periodic oscillations** and closed phase-space loops without ever having seen data past $t = 3.5\text{s}$!

---

## 5. Backpropagation in Neural ODEs: Direct vs. Adjoint Sensitivity Method

When optimizing a Neural ODE $\frac{d\mathbf{y}}{dt} = f_{\theta}(\mathbf{y})$, gradients of the loss $\mathcal{L}$ with respect to network parameters $\theta$ can be calculated in two main ways:

### 1. Direct Backpropagation Through the Solver (Implemented in `ode/solvers.py`)
- **How it works**: The numerical integrator (`odeint`) chains every Runge-Kutta arithmetic step directly in PyTorch. PyTorch Autograd tracks the forward graph across all $N$ time steps and backpropagates through each RK substep.
- **Where it is implemented**:
  - [`ode/solvers.py`](file:///e:/4-1/Numerical%20Lab/Project/Implementation/ode/solvers.py): `odeint()`, `step_tsit5()`, `step_rk4()`.
  - [`ode/neural_ode.py`](file:///e:/4-1/Numerical%20Lab/Project/Implementation/ode/neural_ode.py): `NeuralODE` forward module.
- **Pros**: Fast, exact gradients, straightforward to implement and debug.
- **Memory Footprint**: $\mathcal{O}(N)$ where $N$ is the number of integration steps (requires keeping intermediate solver activations in GPU/CPU memory).

### 2. The Adjoint Sensitivity Method (Memory-Efficient $\mathcal{O}(1)$ Backpropagation)
- **Concept** (Chen et al., 2018; Pontryagin's Maximum Principle):  
  Instead of storing every intermediate step during the forward pass, we define the **adjoint state**:
  $$\mathbf{a}(t) = \frac{\partial \mathcal{L}}{\partial \mathbf{y}(t)}$$
  The adjoint satisfies its own continuous backward differential equation:
  $$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^T \frac{\partial f_{\theta}(\mathbf{y}(t))}{\partial \mathbf{y}}$$
  The total parameter gradient is computed by integrating backward in time from $t_N$ to $t_0$:
  $$\frac{d\mathcal{L}}{d\theta} = -\int_{t_N}^{t_0} \mathbf{a}(t)^T \frac{\partial f_{\theta}(\mathbf{y}(t))}{\partial \theta} \, dt$$
- **Memory Footprint**: $\mathcal{O}(1)$ constant memory! We only store the initial and final states; the backward pass reconstructs the trajectory in reverse.
- **When to use**: Highly beneficial for very long integration trajectories (thousands of steps) or deep systems where GPU memory is constrained.

---

## 6. Architecture Overview & Workflow

```
                     Input x ∈ ℝ^{d_in}
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    [ Normalizer σ(x) ]               [ Base Activation b(x) ]
    e.g. tanh → [-1, 1]               e.g. SiLU / Swish
            │                                   │
            ▼                                   ▼
[ Basis Evaluation ψ_g(σ(x)) ]                  │
  Gaussian RBF / RSWAF / IQF                    │
  Shape: [Batch, d_in, G]                       │
            │                                   │
            ▼                                   ▼
[ Spline Linear Transform C ]         [ Base Linear Transform W ]
  C @ basis_flat^T                      W @ b(x)^T
  Shape: [Batch, d_out]                 Shape: [Batch, d_out]
            │                                   │
            └─────────────────┬─────────────────┘
                              ▼
                        Sum ( + )
                              │
                              ▼
                    Output y ∈ ℝ^{d_out}
```

---

## 7. Module Breakdown

### `basis.py`: Basis Functions

This module provides localized non-linear kernel functions $\psi_g(u)$ evaluated over uniform grid centers $\mathbf{z} = \{z_1, z_2, \dots, z_G\}$ with step width $h = \frac{z_G - z_1}{G - 1}$.

#### 1. Gaussian Radial Basis Function (`rbf`)
The primary basis function used in the paper:

$$\psi_g(u) = \exp\left( -\left(\frac{u - z_g}{h}\right)^2 \right)$$

- **Properties**: Smooth, infinitely differentiable, strictly localized support.
- **Best For**: Smooth continuous vector fields (e.g. Lotka-Volterra, planetary orbits).

#### 2. Reflectional SWitch Activation Function (`rswaf`)
A hyperbolic-secant squared basis:

$$\psi_g(u) = \text{sech}^2\left(\frac{u - z_g}{h}\right) = 1 - \tanh^2\left(\frac{u - z_g}{h}\right)$$

- **Properties**: Heavier tails than Gaussian RBF; avoids sharp exponential cutoffs.

#### 3. Inverse Quadratic Function (`iqf`)
Cauchy/Lorentzian-type kernel:

$$\psi_g(u) = \frac{1}{1 + \left(\frac{u - z_g}{h}\right)^2}$$

- **Properties**: Polynomial decay with broader receptive field.

#### 4. B-Spline Basis (`bspline_basis`)
Piecewise polynomial splines computed via Cox-de Boor recursion:

$$B_{i,0}(u) = \begin{cases} 1 & \text{if } z_i \le u < z_{i+1} \\ 0 & \text{otherwise} \end{cases}$$
$$B_{i,k}(u) = \frac{u - z_i}{z_{i+k} - z_i} B_{i,k-1}(u) + \frac{z_{i+k+1} - u}{z_{i+k+1} - z_{i+1}} B_{i+1,k-1}(u)$$

---

### `layer.py`: `KDense` Layer

The `KDense` class implements the single Kolmogorov-Arnold dense layer:

```python
class KDense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_len: int = 5,
        grid_lims: Tuple[float, float] = (-1.0, 1.0),
        basis_func: Union[str, Callable] = "rbf",
        normalizer: Union[str, Callable] = "tanh",
        base_act: Union[str, Callable] = "silu",
        use_base_act: bool = True,
        init_scale: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    )
```

#### Forward Pass Computation:
1. **Input Flattening**: Ensures input matches $[B, d_{\text{in}}]$.
2. **Normalization**: $\mathbf{x}_{\text{norm}} = \text{normalizer}(\mathbf{x}) \in [-1, 1]$.
3. **Basis Evaluation**: Evaluates basis over grid buffer $\mathbf{z}$:
   $$\mathbf{\Psi} \in \mathbb{R}^{B \times d_{\text{in}} \times G}$$
4. **Flattening**: Reshapes $\mathbf{\Psi}$ to $\mathbb{R}^{B \times (d_{\text{in}} \cdot G)}$.
5. **Spline Projection**: $\mathbf{y}_{\text{spline}} = \mathbf{\Psi}_{\text{flat}} \mathbf{C}^T \in \mathbb{R}^{B \times d_{\text{out}}}$.
6. **Base Residual Projection**: $\mathbf{y}_{\text{base}} = \text{base\_act}(\mathbf{x}) \mathbf{W}^T \in \mathbb{R}^{B \times d_{\text{out}}}$.
7. **Combination**: $\mathbf{y} = \mathbf{y}_{\text{spline}} + \mathbf{y}_{\text{base}}$.

#### Initialization:
Parameters $\mathbf{C}$ and $\mathbf{W}$ are initialized using Glorot / Xavier Uniform scaled by `init_scale`:

$$\text{bound}_C = \sqrt{\frac{6}{(d_{\text{in}} \cdot G) + d_{\text{out}}}} \times \text{init\_scale}$$

$$\text{bound}_W = \sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}} \times \text{init\_scale}$$

---

### `model.py`: `KAN` Multi-Layer Network

The `KAN` class sequentially chains multiple `KDense` layers:

```python
class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int] = [2, 10, 2],
        grid_len: int = 5,
        grid_lims: Tuple[float, float] = (-1.0, 1.0),
        basis_func: Union[str, Callable] = "rbf",
        normalizer: Union[str, Callable] = "tanh",
        base_act: Union[str, Callable] = "silu",
        use_base_act: bool = True,
        init_scale: float = 1.0,
    )
```

#### Key Capabilities:
- **`forward(x)`**: Propagates input through all sequential `KDense` layers.
- **`regularization_loss(act_reg, entropy_reg)`**: Evaluates parameter $L_1$ and entropy penalties.
- **`get_layer_activations(x)`**: Extracts layer-wise intermediate outputs and individual edge contributions for symbolic analysis.

---

## 8. Model Parameters & Hyperparameters Reference

| Parameter | Type | Default | Description & Significance |
|---|---|---|---|
| `layers_hidden` | `List[int]` | `[2, 10, 2]` | Defines the layer architecture (e.g. $[d_{\text{in}}, h_1, \dots, d_{\text{out}}]$). For 2D dynamical systems like Lotka-Volterra, `in_features=2` and `out_features=2`. |
| `grid_len` | `int` | `5` | Number of basis knots $G$ per edge. Higher values increase representation capacity / resolution at the cost of more parameters. |
| `grid_lims` | `Tuple[float, float]` | `(-1.0, 1.0)` | The bounding interval $[z_{\min}, z_{\max}]$ across which the $G$ grid knots are linearly distributed. Matches the $\tanh$ normalizer output range. |
| `basis_func` | `str` / `Callable` | `"rbf"` | Non-linear basis kernel. Options: `'rbf'` (Gaussian RBF), `'rswaf'`, `'iqf'`, `'bspline'`. |
| `normalizer` | `str` / `Callable` | `"tanh"` | Pre-basis squashing function that maps unbounded inputs into `grid_lims`. Options: `'tanh'`, `'sigmoid'`, `'identity'`. |
| `base_act` | `str` / `Callable` | `"silu"` | Residual activation function $b(x)$ providing a linear base path. Options: `'silu'`, `'relu'`, `'tanh'`, `'gelu'`, `'identity'`. |
| `use_base_act` | `bool` | `True` | Whether to include the linear residual branch $\mathbf{W} b(\mathbf{x})$. If `False`, only the basis branch is used. |
| `init_scale` | `float` | `1.0` (model) / `1e-5` (layer) | Scaling factor applied to Xavier uniform initialization to control initial gradient scale. |
| `act_reg` | `float` | `0.0` | Weight $\lambda_{\text{act}}$ of the $L_1$ parameter regularization loss for network sparsification. |
| `entropy_reg` | `float` | `0.0` | Weight $\lambda_{\text{ent}}$ of the entropy loss to encourage edge specialization. |

---

## 9. Tensor Shapes & Step-by-Step Data Flow

For a batch of input states $\mathbf{X} \in \mathbb{R}^{B \times d_{\text{in}}}$ passing through a single layer `KDense(in_features, out_features, grid_len=G)`:

| Step | Operation | Output Tensor Shape |
|---|---|---|
| 1. Input | Raw input tensor $\mathbf{X}$ | `(B, in_features)` |
| 2. Normalization | $\mathbf{X}_{\text{norm}} = \tanh(\mathbf{X})$ | `(B, in_features)` |
| 3. Grid Expansion | $(\mathbf{X}_{\text{norm}} - \mathbf{z}) / h$ | `(B, in_features, G)` |
| 4. Basis Output | $\mathbf{\Psi} = \exp\left( - ((\mathbf{X}_{\text{norm}} - \mathbf{z}) / h)^2 \right)$ | `(B, in_features, G)` |
| 5. Reshape | $\mathbf{\Psi}_{\text{flat}} = \text{reshape}(\mathbf{\Psi})$ | `(B, in_features * G)` |
| 6. Spline Matmul | $\mathbf{\Psi}_{\text{flat}} \mathbf{C}^T$ with $\mathbf{C} \in \mathbb{R}^{\text{out} \times (\text{in} \cdot G)}$ | `(B, out_features)` |
| 7. Base Matmul | $\text{base\_act}(\mathbf{X}) \mathbf{W}^T$ with $\mathbf{W} \in \mathbb{R}^{\text{out} \times \text{in}}$ | `(B, out_features)` |
| 8. Output | $\mathbf{Y} = \mathbf{Y}_{\text{spline}} + \mathbf{Y}_{\text{base}}$ | `(B, out_features)` |

---

## 10. Interpretability & Activation Extraction

Unlike black-box MLPs, every edge $(i, j)$ in a KAN corresponds to a distinct 1D activation function $\phi_{i,j}(x_i)$.

The `get_activations()` method in `KDense` separates these contributions using tensor contractions (`torch.einsum`):

```python
# C_reshaped: [out_features, in_features, grid_len]
# basis_vals: [Batch, in_features, grid_len]
spline_acts = torch.einsum("big,oig->bio", basis_vals, C_reshaped)  # Shape: [Batch, in_features, out_features]
base_acts   = torch.einsum("bi,oi->bio", base_val, self.W)          # Shape: [Batch, in_features, out_features]
```

This decomposes the total prediction into explicit edge-wise curves, enabling:
1. **Symbolic Regression**: Fitting mathematical expressions (e.g. $\alpha x$, $-\beta x y$) to individual edges.
2. **Pruning**: Removing edges where $\sum |\phi_{i,j}| \approx 0$.
3. **Phase-space Attribution**: Visualizing which variable drives the dynamics at different points of the trajectory.

---

## 11. Regularization & Sparsification

To promote sparsity and simplify symbolic recovery, `KAN.regularization_loss()` implements the regularization penalty defined in Equation (12) of the paper:

$$\mathcal{L}_{\text{reg}} = \lambda_{\text{act}} \sum_{p \in \Theta} |p| + \lambda_{\text{ent}} \left( - \sum_{p \in \Theta} \bar{p} \log \bar{p} \right)$$

where:
- $\Theta$ is the set of all trainable parameters ($\mathbf{C}$ and $\mathbf{W}$).
- $\bar{p} = \frac{|p|}{\sum_{p' \in \Theta} |p'| + \epsilon}$ is the normalized parameter importance distribution.
- **$L_1$ Regularization ($\lambda_{\text{act}}$)** drives inactive edge weights towards exact zero.
- **Entropy Regularization ($\lambda_{\text{ent}}$)** minimizes the entropy of $\bar{p}$, forcing the network to concentrate capacity into a minimal subset of dominant edges.

---

## 12. Code Usage Examples

### Instantiating a KAN Model
```python
import torch
from kan import KAN

# Create a 2-layer KAN: 2 inputs -> 10 hidden -> 2 outputs
model = KAN(
    layers_hidden=[2, 10, 2],
    grid_len=5,
    grid_lims=(-1.0, 1.0),
    basis_func="rbf",
    normalizer="tanh",
    base_act="silu",
    use_base_act=True,
    init_scale=1.0,
)

# Forward pass on a batch of 32 2D states
x = torch.randn(32, 2)
y = model(x)
print("Output shape:", y.shape)  # torch.Size([32, 2])
```

### Computing Regularization Loss
```python
# Compute L1 + Entropy penalty during training step
reg_loss = model.regularization_loss(act_reg=1e-4, entropy_reg=1e-4)
total_loss = mse_loss + reg_loss
total_loss.backward()
```

### Inspecting Edge Activations
```python
activations = model.get_layer_activations(x)
for layer_idx, act in enumerate(activations):
    print(f"Layer {layer_idx} Spline Activations Shape:", act["spline"].shape)
    # e.g., torch.Size([32, 2, 10]) for Layer 0
```
