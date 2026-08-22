# Solver-Aware Neural Dynamics: KAN-ODEs Project State

Implementation and ablation study based on:
> **"KAN-ODEs: Kolmogorov–Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics"**  
> *Z. Koenig, J. Kim, Y. Deng (CMAME / arXiv:2407.04192, 2024, MIT)*

---

## 📌 Project Overview & Scope
The base paper fixed **Tsit5 (Tsitouras 5/4)** integration with **Gaussian RBF** activations.  
Our project systematically stress-tests both design choices through comprehensive ablation studies across:
1. **Basis / Activation functions** (RBF, B-spline, Polynomials)
2. **ODE Integrators** (Euler, Heun RK2, RK4, DOPRI5, Tsit5)
3. **Dynamical Systems & Robustness** (Lotka-Volterra, Damped Pendulum, SIR Epidemic, Noise sweep)

---

## ⚙️ Technical Setup & Constraints
* **Optimization Method:** Direct backpropagation through the computational graph (PyTorch Autograd). Adjoint sensitivity optimization is not required here because training trajectories are short ($N = 36$ points), making direct backprop faster, exact, and numerically stable without memory bottlenecks.
* **Epoch Budget:** 10,000 epochs per run (the base paper used 100,000, but 10,000 takes ~40 minutes and is sufficient for comparative ablation).

---

## ✅ What is Done
- [x] **Base Paper Pipeline:**
  - [x] Gaussian RBF basis (`kan/basis.py`, `kan/layer.py`, `kan/model.py`)
  - [x] Native PyTorch Tsitouras 5/4 Runge-Kutta integrator (`ode/solvers.py`, `ode/neural_ode.py`)
  - [x] Lotka-Volterra synthetic data generator ($t_{\text{train}} \in [0, 3.5]$, $t_{\text{full}} \in [0, 14.0]$)
  - [x] Full training, evaluation, and plotting pipeline (`train.py`, `evaluate.py`)
  - [x] Baseline tested & validated (RBF + Tsit5 @ 10,000 epochs)
- [x] **Benchmarking Infrastructure:**
  - [x] Automated test runner (`test_facility.py`) with metric logging and comparison bar charts
- [x] **Skeleton / Partial Solvers & Bases (Implemented in code, pending testing):**
  - [x] Solvers drafted: `rk4`, `dopri5`, `euler`, `midpoint`, `heun`
  - [x] Bases drafted: `rswaf`, `iqf`, basic `bspline`

---

## 📋 What Needs to Be Done (Team TODOs)

### 1. Basis Functions Implementation & Verification
- [ ] **B-Splines (Priority):** Implement and test full Cox-de Boor B-spline basis ($k=3$) within KAN layers.
- [ ] **Polynomial Bases:** Implement Newton Polynomial and Lagrange Polynomial basis options in `kan/basis.py`.
- [ ] Run activation ablation benchmarks across all basis functions on Lotka-Volterra.

### 2. Integrator Ablation Study
- [ ] Run full 10,000-epoch benchmark sweeps across:
  - Forward Euler ($p=1$)
  - Heun RK2 ($p=2$)
  - Classical RK4 ($p=4$)
  - Adaptive Dormand-Prince (DOPRI5)
- [ ] Compare metrics: Train/Test MSE, Number of Function Evaluations (NFE), Trajectory/Phase-Space Drift, and Wall-clock training time.

### 3. Extended Dynamical Systems & Robustness
- [ ] **Damped Pendulum Dynamics:** Add non-linear damped pendulum benchmark dataset and test KAN-ODE convergence.
- [ ] **SIR Epidemic Dynamics:** Implement SIR compartment model dataset and evaluate extrapolation.
- [ ] **Noise Robustness Sweep:** Evaluate sensitivity and generalization under measurement noise ($\sigma \le 0.10$).

---

## 📁 Repository Structure
```
Implementation/
├── kan/
│   ├── basis.py           # Gaussian RBF, B-spline, Polynomial bases
│   ├── layer.py           # KDense layer
│   └── model.py           # Multi-layer KAN
├── ode/
│   ├── solvers.py         # Tsit5, RK4, DOPRI5, Euler, Heun, Midpoint
│   └── neural_ode.py      # NeuralODE wrapper
├── data/
│   ├── lotka_volterra.py  # Lotka-Volterra generator
│   └── ...                # (Pendulum, SIR to be added)
├── utils/
│   ├── regularization.py  # L1 & Entropy regularization
│   └── plotting.py        # Trajectory, Phase Portrait, Loss & Benchmark plots
├── train.py               # Main training script
├── evaluate.py            # Checkpoint evaluation & metrics
├── test_facility.py       # Automated ablation benchmark suite
└── README.md              # Project status and guide
```

---

## 🚀 Commands Quick Reference

### Baseline Training (RBF + Tsit5)
```bash
python train.py --basis rbf --solver tsit5 --epochs 10000 --lr 5e-4
```

### Checkpoint Evaluation
```bash
python evaluate.py --checkpoint results/kanode_rbf_tsit5/best_model.pt
```

### Running Ablation Benchmarks
```bash
# Basis ablation
python test_facility.py --mode activations --epochs 1500

# Solver ablation
python test_facility.py --mode solvers --epochs 1500
```

---

## 🧩 Modularity: How to Add New Basis Functions & Solvers

### 1. Adding a New Basis Function
Add your custom basis function in `kan/basis.py` and register it in `BASIS_FUNCTIONS`:
```python
def my_custom_basis(x: torch.Tensor, grid: torch.Tensor, h: float) -> torch.Tensor:
    # x: normalized input [batch, in_features]
    # grid: basis centers [in_features, grid_len]
    # h: grid width (spacing)
    # returns: basis evaluations [batch, in_features, grid_len]
    return torch.sin((x.unsqueeze(-1) - grid) / h)

BASIS_FUNCTIONS["my_basis"] = my_custom_basis
```
Then train with:
```bash
python train.py --basis my_basis
```

### 2. Adding a New ODE Solver
Add your single-step solver function in `ode/solvers.py` and register it in `STEP_SOLVERS`:
```python
def step_my_solver(func, t: torch.Tensor, y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    # func: callable f(t, y) returning dy/dt
    # y: state tensor at time t
    # dt: step size
    # returns: state tensor at time t + dt
    return y + dt * func(t, y)

STEP_SOLVERS["my_solver"] = step_my_solver
```
Then train with:
```bash
python train.py --solver my_solver
```


