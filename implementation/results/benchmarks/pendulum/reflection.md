# Reflection: Damped Pendulum Benchmark

**Run:** `results/benchmarks/pendulum/` — `layers=[2,10,2]`, RBF, $G=5$, Tsit5,
`substeps=2`, lr $=2\times10^{-3}$, 10,000 epochs, seed 42, `git_sha=9bcbfaf`.

**Result:** did not converge. `best.train_mse = 0.3107` (best epoch = 10000, i.e. the
run never found a better checkpoint than its own endpoint). `extrap_mse = 1.029`,
`extrap_r2 = -1.046`, `L = 116.79` (vs `L = 6.91` for the converged Lotka-Volterra
flagship). See `training_history.json`, `loss_curves.png`, `gradient_norm_dynamics.png`.

## What happened

State coverage is not the problem: 0.0% of extrapolation states fall outside the
training box. The model simply never fits. Training loss descends $7.34 \to 0.44$ by
epoch 1000, then hits a single catastrophic event:

| Epoch | Grad norm | Note |
| :--- | :--- | :--- |
| ~4900 (median across run) | 0.106 | typical |
| **4963** | **759.86** | **7186× the run's median** |
| 10000 | — | loss plateaued at 0.3107, flat from ~epoch 8000 |

Loss rises to $1.21$ immediately after the spike and only partially recovers to
$0.311$ over the remaining ~5000 epochs — worse than the pre-spike trajectory was
heading toward. Loss at epoch 8000 is $1.03\times$ loss at epoch 10000: the run is
flat, not still descending, when the epoch budget runs out.

## Root cause: cross-checked against SIR, same signature

The SIR run in `../sir/reflection.md` shows the identical pattern — a single-epoch
gradient-norm explosion partway through training, from which the model never fully
recovers within the remaining budget. The Lotka-Volterra flagship run also spikes
(50.2 at epoch 1016, 1422× its median of 0.0353) but survives, for two likely reasons:
the spike is ~15× smaller in absolute magnitude, and it happens at epoch 1016 —
leaving 9000 epochs of recovery room, versus pendulum's ~5000 and SIR's ~5000.

`train.py` calls no gradient clipping anywhere (confirmed by inspection — grad norms
are computed and logged, never bounded before `optimizer.step()`). This is standard
exposure for unrolled backprop through an ODE solver: an occasional stiff/high-curvature
region in the learned vector field produces one enormous gradient, and plain Adam has
no defense against it.

A previously-floated hypothesis — that `tanh` normalization saturates the
$\omega \in [-4.60, 3.52]$ dimension onto $[-1.000, +0.998]$, starving the spline basis
of resolution — was tested and **not confirmed nor ruled out**: Lotka-Volterra's prey
dimension is also saturated ($[0.739, 1.000]$) and trains fine, so saturation alone
does not explain the difference. It may still be a contributing factor once the
gradient-blowup confound is removed, but cannot be evaluated in isolation from this run.

## What is NOT yet known

Because the run never stabilizes, this artifact cannot answer:
- Whether `[2,10,2]` + RBF + $G=5$ is an adequate capacity for the pendulum's dynamics.
- Whether the `tanh` saturation on $\omega$ actually matters once training is stable.
- Whether pendulum extrapolation would be strong or weak from a converged fit — the
  current $R^2 = -1.046$ reflects a broken optimization, not a capacity or
  generalization ceiling.

## What needs to happen (deferred to Phase 3 — not run today, no time remaining)

1. **Add gradient clipping to `train.py`** (`torch.nn.utils.clip_grad_norm_` before
   `optimizer.step()`) — the one code change most likely to fix both this and the SIR
   failure, since it addresses the shared mechanism rather than either system's
   particulars. This is a pipeline fix, not a per-system tune.
2. **Re-run pendulum after the fix**, same config, and check whether it converges
   before touching any other hyperparameter (grid width, `grid_lims`, LR).
3. **Only if it still fails after clipping**, revisit the `tanh`-saturation hypothesis:
   widen `grid_lims` or rescale $\omega$ before normalizing, and/or increase `grid_len`
   from 5.
4. **This blocks Phase 3 Novelty 3** (the stiffness phase map, which sweeps pendulum
   damping $\mu$ across solvers) — that task should not be assigned until this run
   converges, since it is built directly on this system.

## Suggested command once clipping is added

```
python train.py --dataset damped_pendulum --layers 2 10 2 --grid_len 5 --lr 0.002 \
  --epochs 10000 --save_dir results/benchmarks/pendulum
```

Same config as this run — the point is to isolate whether clipping alone fixes it
before changing anything else.
