# Track A -- Adjoint vs. Autograd Profiling: Findings

> **Auto-generated draft from `run_profile.py`'s actual output** (`results/phase3/adjoint_profiling/table4.json`). Review the prose before treating this as final; the numbers are real, the wording is a first pass.

## Research question

`implementation/README.md` documents direct autograd through the unrolled solver as a deliberate choice for this project's short trajectories ($N=36$-$101$ points), on the grounds that adjoint sensitivity's $O(1)$-memory advantage doesn't matter until trajectories get long. This track measures whether that holds, on real hardware (NVIDIA GeForce RTX 3060), and where it stops holding.

## Method

Both paths run classical RK4 at a **matched step size** -- this project's own `NeuralODE(method='rk4')` for direct autograd, `torchdiffeq.odeint_adjoint(method='rk4', options={'step_size':...})` for the adjoint path. A forward-trajectory check (`tests/test_p3_adjoint_profiling.py`) confirms the two RK4 implementations agree to ~1e-7 relative error, so this isolates the differentiation method as the only variable -- any difference below is attributable to backprop-through-unrolled-solver vs. continuous-adjoint-ODE, not to comparing two different integrators.

## Table 4 -- peak VRAM, wall-clock, gradient accuracy

| System | $N_t$ | Method | Peak VRAM (MB) | s / 1000 epochs | Grad rel. error |
| :--- | :---: | :--- | :---: | :---: | :---: |
| Lotka-Volterra | 36 | Direct autograd | 18.61 | 317.52 | 2.85e-07 |
| Lotka-Volterra | 36 | Adjoint | 17.08 | 672.85 | (same row) |
| SIR | 101 | Direct autograd | 21.52 | 984.75 | 4.88e-07 |
| SIR | 101 | Adjoint | 17.12 | 1871.64 | (same row) |

## Memory vs. trajectory length

![memory vs trajectory length](../../implementation/results/phase3/adjoint_profiling/memory_vs_trajectory_length.png)

Sweeping $N_t \in \{36, 101, 201, 401, 801, 1601, 3201\}$ with a single forward+backward pass per point (isolating the memory claim from training-loop/optimizer overhead), the crossover where adjoint's $O(1)$ memory starts winning is: **$N_t \approx 36$**.

## Finding

At this project's actual trajectory lengths ($N=36$ for Lotka-Volterra, $N=101$ for SIR), the gradient computed via the continuous adjoint matches direct autograd to within tolerance ($< 5\times10^{-3}$ relative error) on both systems, so `implementation/README.md`'s claim -- adjoint sensitivity is unnecessary here -- is evaluated on both memory and wall-clock in the table above. State explicitly, after reviewing the real numbers: does direct autograd remain cheaper in both VRAM and wall-clock at these lengths? At what $N_t$ (see crossover above) would that stop being true?

*(TODO: replace this paragraph with one written after reading the actual table above -- the auto-generated text states the setup and the crossover mechanically; the judgment call about what it MEANS for the project's short-trajectory regime is not something to leave auto-generated.)*
