# Track A -- Adjoint vs. Autograd Profiling

**Owner:** Nawriz Ahmed Turjo (2105032) · **Branch:** `feat/p3-adjoint-profiling`
**Novelty 5 -> Table 4.** See `docs/12_phase3_roadmap.md` Part 4 for the full spec.

## Research question

The base paper uses continuous adjoint sensitivity ($O(1)$ memory in trajectory
length) instead of this project's direct autograd through the unrolled solver
($O(N_t)$ memory). `implementation/README.md` documents this as a *deliberate,
justified* choice for short trajectories ($N=36$-$101$ points) -- is that
justification actually true, and where does it stop being true?

## Method

**Design choice, stated up front because it drives every number this track
produces:** both the "direct autograd" path (this project's own `NeuralODE`)
and the "adjoint" path (`torchdiffeq.odeint_adjoint`) are run with **classical
RK4 at a matched step size**. `tests/test_p3_adjoint_profiling.py::
test_rk4_implementations_agree` confirms the two RK4 implementations produce
the same forward trajectory to ~1e-7 relative error. That isolates exactly one
variable -- how the gradient is computed -- so any residual difference in peak
memory, wall-clock, or gradient value is attributable to the differentiation
method (backprop through the unrolled solver vs. solving the continuous
adjoint ODE backward in time), not to comparing two different integrators.

`KAN.forward` needs **no wrapper** to work with `torchdiffeq` -- its
`forward(t, y)` dispatch (see `kan/model.py`) already matches what
`torchdiffeq.odeint[_adjoint]` calls. Confirmed directly:
`tests/test_p3_adjoint_profiling.py::test_kan_forward_is_torchdiffeq_compatible`.

Three measurements, per system (Lotka-Volterra $d=2$, $N=36$; SIR $d=3$,
$N=101$, using the **fixed** `--time_scale 10.0` / `--conserve_mode projection`
recipe from `docs/10_sir_root_cause_and_fix.md` so the run actually converges):

1. **Gradient relative error** -- `paired_gradient_relative_error()` builds one
   model, deep-copies it once per method, and compares
   $\|\nabla_\text{auto} - \nabla_\text{adj}\| / \|\nabla_\text{auto}\|$ from
   **identical initial weights** on the same matched-step RK4 trajectory. This
   is the check the original blueprint sketched as `test_adjoint.py`.
2. **Peak VRAM + wall-clock per 1000 epochs** -- `train_direct_autograd()` and
   `train_adjoint()` each run their own full training loop (`Adam`, matched
   step size) for `--epochs` epochs on a fresh `torch.cuda.reset_peak_memory_stats`
   window.
3. **Memory vs. trajectory length** (the headline figure) --
   `memory_vs_trajectory_length()` does a single forward+backward pass (no
   training loop, so optimizer-state overhead doesn't confound the reading) at
   a sweep of synthetic trajectory lengths $N_t$, isolating the $O(1)$-vs-$O(N_t)$
   claim directly and finding the crossover point where adjoint's constant
   memory starts winning -- this subsumes the roadmap's "optional stretch"
   (a longer pendulum trajectory to find the crossover) without needing an
   actual pendulum training run.

## Files

- `profiling.py` -- the four measurement primitives above, reusable and unit-tested.
- `run_profile.py` -- CLI that runs the full profiling suite and writes
  `results/phase3/adjoint_profiling/table4.json`,
  `results/phase3/adjoint_profiling/memory_vs_trajectory_length.png`, and a
  **draft** `docs/13_p3_adjoint_profiling_findings.md` auto-filled with the real
  numbers (the prose's factual claims are generated from the JSON; the
  interpretive paragraph is left as a `TODO` for a human read of the actual
  table -- see the file for exactly what's auto vs. manual).

## How to run

```powershell
cd implementation/experiments/adjoint_profiling

# Fast correctness smoke test (no GPU needed, seconds):
python run_profile.py --epochs 5 --sweep_lengths 10 20 40 --device cpu

# The real deliverable run (~1 hour per the roadmap's compute budget; needs
# CUDA to make the O(1)-vs-O(N_t) memory claim visible -- VRAM is None on CPU):
python run_profile.py --device cuda
```

`--epochs` defaults to 2000 (roadmap: "1,000-2,000 epochs is plenty" for
profiling, since this measures resource usage per step, not final accuracy).
`--sweep_lengths` defaults to `36 101 201 401 801 1601 3201`, chosen to bracket
both systems' actual $N$ (36, 101) and extend well past them to find the
crossover.

## Dependency

`torchdiffeq` (already in `requirements.txt`, already installed in this repo's
`venv` at version 0.2.5 -- confirmed via `pip show torchdiffeq`, no install
step needed here). No edit to `requirements.txt` from this branch (see
`docs/12_phase3_roadmap.md` §2.4 -- that's a single one-line-per-package PR
after all five Phase 3 branches merge).

## Definition of done (from the roadmap)

| Check | Target | Status |
| :--- | :--- | :--- |
| Gradient relative error, both systems | $< 5\times10^{-3}$ | confirmed at toy scale in tests (~1e-7); real-scale numbers land in `table4.json` after `run_profile.py --device cuda` |
| Both systems profiled | direct autograd **and** adjoint | implemented, delegated to run |
| Written finding | states explicitly whether "adjoint isn't needed here" still holds, and at what trajectory length it would stop holding | auto-drafted in `docs/13_p3_adjoint_profiling_findings.md`, needs a human read of the real table |
