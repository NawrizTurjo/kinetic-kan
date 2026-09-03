# 🌟 Phase 3 Roadmap: Novel Research Contributions

> **Status:** ready to start — no code written yet, no branches created
> **Prerequisite:** ✅ Phase 2 merged into `main` (`8471f49`), CI green
> **Chain:** [`09`](./09_stability_fix_results.md) pendulum fix · [`10`](./10_sir_root_cause_and_fix.md) SIR fix · [`11`](./11_phase2_closeout.md) Phase 2 closeout → **this doc**
> **Supersedes:** `04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md` §Phase 3 (Task 3.1–3.6, the per-member pipeline tables, and the Lorenz-inclusive Novelty 6) — that document is the original plan, kept for the record; **this document is authoritative for what Phase 3 actually is and who owns what.**

---

# Part 1 — Scope

## 1.1 What Phase 3 is

Phase 2 answered *"what happens when you unfix the paper's two frozen choices (solver,
basis) and give it harder dynamics?"* — six systematic ablations, all now closed.

Phase 3 answers a different kind of question: **five focused, independent research
questions that go beyond reproducing or stress-testing the paper**, each contributing
one original table or figure toward the final report. This is the "Part 2" the proposal
promised and the presentation was pitched against — the material that turns a course
replication project into something with its own findings.

The original blueprint framed this as six novelties. With Lorenz removed (§1.2), Novelty
6 narrows to real-epidemic fitting only, and is merged with the SINDy track since both
compare the learned model against an independent form of ground truth — five tracks,
five people:

| Track | Novelty | Table/Figure |
| :-: | :--- | :--- |
| A | Adjoint vs. autograd memory/speed profiling | Table 4 |
| B | Gradient norm dynamics vs. solver order | — |
| C | Learnable softmax hybrid basis (spline + RBF) | — |
| D | Stiffness–solver stability phase map | Table 5 |
| E | SINDy comparison under noise + real epidemic fit | Table 3* |

*(Table 3 in the original numbering was reassigned to the KAN-vs-MLP comparison during
Phase 2; SINDy's table gets a fresh number at synthesis time.)*

## 1.2 Scope changes from the original blueprint — final

Two items are **fully removed**, not deferred:

- **❌ Lorenz.** Dropped entirely from project scope. `data/lorenz.py` and its supporting
  code remain in the repository (harmless, unused) but **no Phase 3 task depends on it**
  and no team member should spend time on it. Track E absorbs the real-epidemic half of
  the original Novelty 6; the Lorenz half is simply gone.
- **❌ Δt = 0.01.** Already deferred with reasoning in [`11`](./11_phase2_closeout.md)
  §Deferred (10× the cost of Δt=0.1 for a fourth point on an already-confounded sweep).
  Confirmed as a permanent drop. No Phase 3 task depends on it.

Everything else in the original six survives, with one **necessitated correction**:
the stiffness phase map (Track D) must be built on the **fixed** pendulum config from
[`09`](./09_stability_fix_results.md) — `--t_train_end 5.0`, default SiLU — not the
pre-fix recipe the blueprint was written against. Building it on the failing config
would just be measuring the pendulum's optimisation failure, not its stiffness.

## 1.3 What "novel" means here, concretely

Each track should produce:
1. **A quantitative table or figure** — not a demo, a result with numbers.
2. **A written finding** — one paragraph stating what was learned, in the same
   evidence-first style as [`06`](./06_suggested_fixes.md)–[`11`](./11_phase2_closeout.md):
   claim, number, caveat.
3. **A reproducibility artifact** — a script anyone can re-run to get the same table.

## 1.4 Explicitly out of scope for Phase 3

- Lorenz (removed, §1.2)
- Δt = 0.01 (removed, §1.2)
- Multi-seed replication (N=5) — that is Phase 4 Task 4.1, deliberately deferred until
  every track's *methodology* is settled; re-running with 5 seeds before that is wasted
  compute if the recipe still changes
- Editing any file under `kan/`, `ode/`, `data/`, `utils/`, `train.py`,
  `evaluate.py`, `test_facility.py`, `collate_results.py`, `run_phase2.ps1`,
  `analyze_fixes.py`, `phase2_closeout.py`, or any file under `results/` other than a
  track's own output folder — see Part 3

---

# Part 2 — The Zero-Conflict Folder Architecture

## 2.1 The rule

> **Every Phase 3 deliverable is new files, in one person's own folder. No existing
> file is edited by anyone.**

This is stronger than "try not to conflict" — if followed, merge conflicts are not just
unlikely, they are **structurally impossible**, because git only conflicts when two
branches edit the *same lines of the same file*. Five branches that each only *add* new
files under five disjoint directories cannot conflict with each other, and can be merged
into `main` in any order.

## 2.2 Why this is actually achievable in this codebase

This is not a hopeful aspiration — it is a direct consequence of how the codebase is
already built, verified against the real source:

| Extension point | Mechanism | Verified in |
| :--- | :--- | :--- |
| New basis function | `KDense(basis_func=...)` accepts a raw **callable**, not just a string — `Union[str, Callable]` | [`kan/layer.py`](../implementation/kan/layer.py) |
| New solver | `STEP_SOLVERS` dict is directly importable and assignable: `STEP_SOLVERS["myname"] = fn` | [`ode/solvers.py`](../implementation/ode/solvers.py); documented pattern in `implementation/README.md` §Modularity |
| New "physics-aware" field wrapper | `ZeroSumField` / `VanishingDimField` show the exact template: wrap a trained `nn.Module`, add structure, feed to `NeuralODE(func=...)` | [`ode/neural_ode.py`](../implementation/ode/neural_ode.py), built for [`10`](./10_sir_root_cause_and_fix.md) |
| Any dataset with a custom parameter (e.g. pendulum μ) | `generate_damped_pendulum_data(mu=..., ...)` is a plain importable function — **not** gated behind `train.py`'s CLI, which only threads Lotka-Volterra kwargs through | [`data/damped_pendulum.py`](../implementation/data/damped_pendulum.py) |
| Reproducing a training loop | `train_kan_ode()` is directly importable, but so are its five components (`KAN`, `NeuralODE`, a `generate_*` function, an optimizer, `compute_gradient_norm`) — a track needing a variant loop (e.g. sweeping μ, which `train.py`'s CLI cannot do) writes ~40 lines reusing those primitives, not a `train.py` edit | confirmed by reading `train.py` §`train_kan_ode` |
| Loading a checkpoint correctly | `evaluate.py` and `phase2_closeout.py` both show the **required** pattern — rebuild the field honouring `grid_lims`, `time_scale`, `conserve_mode`, `vanish_dim` from the saved config, not just the weights | [`10`](./10_sir_root_cause_and_fix.md) §Part 5 catalogues what breaks if you don't |

**Conclusion: nothing in Phase 3, as scoped below, requires editing a shared file.**
Where a track needs "new capability," it is new capability *added from outside* via
these existing hooks — never a change to the file that defines the hook.

## 2.3 Standard folder layout — every track follows this shape

Folders are named by **task**, not by team member — so a future reassignment (like the
one that already happened once, see Part 3) never requires renaming anything:

```
implementation/
├── experiments/
│   └── <slug>/                   ← that track's ENTIRE codebase
│       ├── README.md             ← research question, method, how to run, findings
│       ├── run_*.py              ← the experiment script(s)
│       └── *.py                  ← any new basis/field/utility, private to this folder
│
├── results/
│   └── phase3/
│       └── <slug>/               ← that track's ENTIRE output — plots, JSON, checkpoints
│
tests/
└── test_p3_<slug>.py             ← that track's own test file (Phase 1 Rule 1 pattern)

docs/
└── 1<N>_p3_<slug>_findings.md    ← that track's write-up, number pre-assigned below
```

The five slugs, fixed for the rest of this document:
`adjoint_profiling` (A) · `gradient_dynamics` (B) · `hybrid_basis` (C) ·
`stiffness_map` (D) · `sindy_epidemic` (E).

Nobody reads from another track's `experiments/` or `results/phase3/` folder except to
*cite* a finished number in their own write-up (read-only, via `git show` or a copy of
the JSON, never an import).

## 2.4 The one shared touchpoint, and how it's handled without conflict

Two tracks need a package not currently installed (`torchdiffeq`, `pysindy`, both
already in `requirements.txt` but absent from every checked environment). Adding them
is a one-line edit to a shared file.

**Resolution: do not touch `requirements.txt` during individual work.** Install locally
(`pip install torchdiffeq` / `pip install pysindy`) and note the exact pinned version in
your own `README.md`. At final integration (Part 5), one person opens a single
one-line-per-package PR against `requirements.txt` after all five branches have merged.
This is the only place in the whole plan where two people's changes could theoretically
land on the same file, and it is delayed until there is nothing left to conflict with.

## 2.5 Git workflow

```powershell
git checkout main && git pull origin main
git checkout -b feat/p3-<slug>                # e.g. feat/p3-stiffness-map
# ... work only inside your three folders (2.3) ...
git add experiments/<slug>/ results/phase3/<slug>/ tests/test_p3_<slug>.py docs/1<N>_*.md
git commit -m "feat(p3-<slug>): <what>"
git push origin feat/p3-<slug>
# open PR into main
```

**Branch names start with `feat/`, matching `ci.yml`'s trigger
(`branches: [main, develop, "feat/**"]`)** — unlike `fix/phase1-revisit`, whose commits
were never CI-verified because `fix/**` isn't matched. Every Phase 3 branch gets CI on
every push, for free.

**Merge order is unconstrained** — folder isolation means any order is safe. Merge as
each track finishes; do not block on the slowest one.

---

# Part 3 — Track Ownership

| Track | Owner | ID |
| :--- | :--- | :---: |
| **A** — Adjoint vs. Autograd Profiling | Nawriz Ahmed Turjo | **2105032** |
| **B** — Gradient Norm Dynamics | Abhishek Roy | **2105033** |
| **C** — Learnable Hybrid Basis | Shams Hossain Simanto | **2105048** |
| **D** — Stiffness–Solver Stability Map | Abrar Jahin | **2105055** |
| **E** — SINDy Comparison + Real Epidemic Fit | Monjur Hossain Khan ("Shovon") | **2105043** |

Track A uses GPU profiling (peak VRAM comparison) — Nawriz has GPU access, which is why
that track is his.

Compute estimates throughout this document are **grounded in measured per-epoch costs
from this project's own runs**, not guesses — useful for planning your own track's
runtime regardless of which one it is:

| System | Measured cost | Source |
| :--- | :---: | :--- |
| Lotka-Volterra (RBF, Tsit5) | $0.17$ s/epoch | `kanode_flagship/metrics.json` |
| Lotka-Volterra (B-spline) | $0.43$ s/epoch | `kanode_bspline/metrics.json` |
| Pendulum (win 5.0, $G{=}8$) | $0.77$ s/epoch | `pendulum_control_win5/metrics.json` |
| SIR (time-scaled) | $1.1$–$1.4$ s/epoch | `sir_fixed*/metrics.json` |

---

# Part 4 — The Five Tracks

---

## 🅰️ Track A — Adjoint vs. Autograd Profiling  *(Novelty 5 → Table 4)*

**Owner:** Nawriz Ahmed Turjo (2105032) · **Branch:** `feat/p3-adjoint-profiling`
**Folder:** `experiments/adjoint_profiling/` · **Results:** `results/phase3/adjoint_profiling/`
**Write-up:** `docs/13_p3_adjoint_profiling_findings.md`

### Research question

The base paper uses continuous adjoint sensitivity ($O(1)$ memory in trajectory length)
instead of this project's direct autograd through the unrolled solver ($O(N_t)$ memory).
`implementation/README.md` documents this as a *deliberate, justified* choice for short
trajectories ($N{=}36$–$101$ points) — **is that justification actually true**, and where
does it stop being true?

### Why this is the light + GPU track

Profiling wall-clock and peak memory needs only **short runs at a handful of
configurations** — the measurement is resource usage per step, not final accuracy, so
1,000–2,000 epochs is plenty. GPU is what makes the $O(1)$-vs-$O(N_t)$ memory claim
*visible* (VRAM headroom on CPU is usually too large to show the effect) — hence the
GPU assignment.

### What already exists to reuse

- `KAN`, `NeuralODE` — the model and integrator, unchanged
- `torchdiffeq.odeint_adjoint` — new dependency, `pip install torchdiffeq`
- `data.generate_lotka_volterra_data`, `data.generate_sir_data` — for $d{=}2$ and $d{=}3$
  state dimensions (the plan's original axis)
- `utils.compute_gradient_norm`, `torch.cuda.max_memory_allocated()` — instrumentation

### Task breakdown

1. Wrap the existing `KAN` vector field for `torchdiffeq` compatibility (its
  `forward(t, y)` signature already matches — confirm with a 5-line smoke test).
2. Run **direct autograd** (current `NeuralODE`) vs. **`odeint_adjoint`** on:
  - Lotka-Volterra ($d{=}2$, $N{=}36$ train points — the paper's own regime)
  - SIR ($d{=}3$, $N{=}101$ train points, using the fixed `--time_scale` recipe from
    [`10`](./10_sir_root_cause_and_fix.md) so the run actually converges)
3. For each: peak VRAM (`torch.cuda.max_memory_allocated()`, reset between runs),
  wall-clock per 1,000 epochs, and **gradient relative error**
  $\|\nabla_{\text{auto}} - \nabla_{\text{adj}}\| / \|\nabla_{\text{auto}}\|$ — this is
  exactly the check sketched in the original blueprint's `test_adjoint.py` draft (which
  does not exist yet in the repo — you are the one creating it).
4. Optional stretch: repeat at one longer synthetic trajectory (e.g. pendulum at
  `--t_end 40`) to find the crossover point where adjoint's constant memory actually
  starts winning.

### Compute budget

$\lesssim 1$ hour total — 4–6 short runs (2,000 epochs) at $0.17$–$1.4$ s/epoch.

### Deliverables

- `results/phase3/adjoint_profiling/table4.json` — VRAM, wall-clock, grad error per
  (method × system)
- One figure: memory vs. trajectory length, both methods overlaid
- `docs/13_p3_adjoint_profiling_findings.md`

### Definition of done

| Check | Target |
| :--- | :--- |
| Gradient relative error, both systems | $< 5\times10^{-3}$ |
| Both systems profiled | direct autograd **and** adjoint |
| Written finding | states explicitly whether "adjoint isn't needed here" still holds, and at what trajectory length it would stop holding |

### Files you must not touch

Everything outside `experiments/adjoint_profiling/`, `results/phase3/adjoint_profiling/`,
`tests/test_p3_adjoint_profiling.py`, `docs/13_p3_adjoint_profiling_findings.md`.

---

## 🅱️ Track B — Gradient Norm Dynamics vs. Solver Order  *(Novelty 1)*

**Owner:** Abhishek Roy (2105033) · **Branch:** `feat/p3-gradient-dynamics`
**Folder:** `experiments/gradient_dynamics/` · **Results:** `results/phase3/gradient_dynamics/`
**Write-up:** `docs/14_p3_gradient_dynamics_findings.md`

### Research question

Do lower-order solvers (Euler, Heun/Midpoint) inject higher-frequency, noisier gradients
into backprop than higher-order ones (RK4, Tsit5)? The plan's own note: *"log
$\|\nabla_\theta\mathcal{L}\|_2$ every epoch across solvers — `compute_gradient_norm`
already exists and is usable as-is."*

### Why this is the lightest track — genuinely zero new training

**Every run in the project already has this data.** `training_history.json` for all 6
solver-ablation runs (`ablation_solvers/solver_{euler,heun,midpoint,rk4,dopri5,tsit5}`)
carries a per-epoch `grad_norms[]` array, logged since Phase 1. Post-`FIX-2026-08` runs
additionally carry `post_clip_grad_norms[]`. **This track is pure analysis and plotting
of files that already exist on disk.**

### What already exists to reuse

- `results/benchmarks/ablation_solvers/solver_*/training_history.json` — 6 files, one
  per solver, each a plain `{"train_losses": [...], "grad_norms": [...], ...}` JSON
- `implementation/analyze_fixes.py` — read as a **template** for how to load and process
  many `metrics.json`/`training_history.json` files cleanly (do not import from it or
  edit it — copy the pattern into your own script)

### Task breakdown

1. Load all 6 solvers' `grad_norms[]`. Compute, per solver: mean, std, and a
  **high-frequency noise measure** — e.g. the std of the first-difference
  $\Delta g_i = g_{i+1} - g_i$, or a rolling-window coefficient of variation. Pick one,
  justify it in the write-up.
2. Plot all 6 gradient-norm trajectories overlaid (semilog-y, since norms span orders
  of magnitude) — solver order colour-coded (Euler=1 → Tsit5=5).
3. Test the specific claim: does noise (your chosen measure) **decrease monotonically**
  with solver order $p$? Report the correlation, not just a plot — a number the reader
  can check.
4. Cross-check: does gradient noisiness track the [`06`](./06_suggested_fixes.md) §Table 1
  finding that Euler is the *only* solver failing to reach $10^{-4}$ training loss? If
  Euler's gradients are measurably noisier, that is a plausible **mechanism** for that
  Phase 2 finding, not just a correlation — worth stating explicitly.

### Compute budget

**Zero new training.** A few hours of analysis and plotting.

### Deliverables

- `results/phase3/gradient_dynamics/gradient_noise_by_order.png`
- `results/phase3/gradient_dynamics/table.json` — noise measure per solver, ranked
- `docs/14_p3_gradient_dynamics_findings.md`

### Definition of done

| Check | Target |
| :--- | :--- |
| All 6 solvers covered | Euler, Heun, Midpoint, RK4, DOPRI5, Tsit5 |
| Noise measure explicitly defined and justified | one paragraph, in the write-up |
| Correlation with solver order reported as a number | not just "the plot shows..." |
| Link drawn (or explicitly refuted) to Euler's Phase-2 training failure | stated either way |

### Files you must not touch

Everything outside `experiments/gradient_dynamics/`, `results/phase3/gradient_dynamics/`,
`tests/test_p3_gradient_dynamics.py`, `docs/14_p3_gradient_dynamics_findings.md`.
**Read-only** access to `results/benchmarks/ablation_solvers/`.

---

## 🅲️ Track C — Learnable Softmax Hybrid Basis  *(Novelty 2)*

**Owner:** Shams Hossain Simanto (2105048) · **Branch:** `feat/p3-hybrid-basis`
**Folder:** `experiments/hybrid_basis/` · **Results:** `results/phase3/hybrid_basis/`
**Write-up:** `docs/15_p3_hybrid_basis_findings.md`

### Research question

Phase 2 found B-spline and RBF are statistically tied on accuracy but differ sharply on
cost (B-spline $3.9\times$ RBF's wall-clock) and on *why* they work (local compact
support vs. smooth global kernel). Does a **learnable blend**
$\phi(x) = \alpha \cdot \text{Spline}(x) + \beta \cdot \text{RBF}(x)$, with $\alpha,\beta$
softmax-gated and trained end-to-end, converge faster than either pure basis, or
inherit the worse of both (B-spline's cost, no accuracy gain)?

### What already exists to reuse — the key enabling fact

**`KDense(basis_func=...)` accepts a raw callable.** You do **not** need to register
anything in `kan.basis.BASIS_FUNCTIONS` or edit `kan/basis.py` at all:

```python
# entirely inside experiments/hybrid_basis/hybrid_basis.py — zero shared-file edits
import torch, torch.nn as nn
from kan.basis import bspline_basis, rbf

class HybridBasis(nn.Module):
    def __init__(self, grid_len):
        super().__init__()
        self.blend_logits = nn.Parameter(torch.zeros(2))   # -> softmax(alpha, beta)

    def forward(self, x, grid, h):
        w = torch.softmax(self.blend_logits, dim=0)
        return w[0] * bspline_basis(x, grid, h) + w[1] * rbf(x, grid, h)

# usage: KAN(basis_func=HybridBasis(grid_len=5), ...)
```

`bspline_basis` and `rbf` are plain importable functions
([`kan/basis.py`](../implementation/kan/basis.py)) — reused, not modified.

### Task breakdown

1. Implement `HybridBasis` as above (or your own variant — this sketch is a starting
  point, not a spec).
2. Train on **Lotka-Volterra** (the clean, well-understood system) and the **fixed
  pendulum** (`--t_train_end 5.0` config from [`09`](./09_stability_fix_results.md) —
  do not use the pre-fix pendulum recipe).
3. Log $\alpha(\text{epoch})$, $\beta(\text{epoch})$ — does the gate converge toward one
  basis, split evenly, or oscillate? This is the headline plot.
4. Compare against the pure-RBF and pure-B-spline numbers already published in
  [`05`](./05_phase2_benchmark_analysis.md) §Table 2 (**read-only citation, not a
  re-run**) on: epochs to $10^{-3}$/$10^{-4}$ training loss, extrapolation MSE,
  wall-clock per epoch.

### Compute budget

$\sim 2$–$4$ hours — a handful of 10,000-epoch runs at $0.17$–$0.77$ s/epoch across 2
systems, plus a short probe phase (2,000 epochs) to sanity-check the gate is learning
anything before committing to full runs — follow the OFAT discipline established in
[`09`](./09_stability_fix_results.md) §Methodological lessons: **probe first, one
variable at a time, then commit to the full budget.**

### Deliverables

- `results/phase3/hybrid_basis/alpha_beta_evolution.png` — the gate trajectory
- `results/phase3/hybrid_basis/table.json` — hybrid vs. pure-RBF vs. pure-B-spline
- `docs/15_p3_hybrid_basis_findings.md`

### Definition of done

| Check | Target |
| :--- | :--- |
| Gate weights logged every epoch, both systems | not just final values |
| Compared against **existing** Table 2 numbers | citation, not a wasted re-run |
| Explicit verdict | faster / same / worse than the better pure basis, stated numerically |

### Files you must not touch

Everything outside `experiments/hybrid_basis/`, `results/phase3/hybrid_basis/`,
`tests/test_p3_hybrid_basis.py`, `docs/15_p3_hybrid_basis_findings.md`. In particular:
**do not edit `kan/basis.py`** — the callable-injection pattern above makes this
unnecessary.

---

## 🅳️ Track D — Stiffness–Solver Stability Phase Map  *(Novelty 3 → Table 5)*

**Owner:** Abrar Jahin (2105055) · **Branch:** `feat/p3-stiffness-map`
**Folder:** `experiments/stiffness_map/` · **Results:** `results/phase3/stiffness_map/`
**Write-up:** `docs/16_p3_stiffness_map_findings.md`

### Research question

Sweep the damped pendulum's damping ratio $\mu \in \{0.1, 0.5, 1.0, 2.0, 5.0, 8.0\}$
across solvers and step sizes: which (solver, μ, Δt) combinations converge, and which
explode or fail to fit? Produces the plan's Table 5 and a 2D stability heatmap.

### ⚠️ This must build on the fixed pendulum recipe — not the blueprint's original one

The blueprint was written before [`09`](./09_stability_fix_results.md) existed. The
pendulum's own default $\mu{=}0.5$ **did not converge** on `--t_train_end 3.0` with SiLU
— that failure is root-caused and fixed. Building the stiffness map on the *unfixed*
recipe would just re-measure that same optimisation failure at every μ, producing a
map of "did the known bug fire" rather than "is the system stiff." **Use
`--t_train_end 5.0`, default SiLU** (the `pendulum_control_win5` config) as the baseline
every cell of the sweep starts from.

### The gap that makes this a real coding task, not just a CLI sweep

`generate_damped_pendulum_data(mu=...)` is a plain function, but **`train.py`'s CLI has
no way to pass `mu` through** — it only threads Lotka-Volterra kwargs (`alpha, beta,
gamma, delta`) into `train_kan_ode()`. Sweeping μ therefore **cannot** be done through
`run_phase2.ps1` or `python train.py --dataset damped_pendulum` unchanged.

**Resolution (folder-isolated, no shared-file edit):** write your own ~40-line training
loop inside `experiments/stiffness_map/run_sweep.py` that directly reuses `KAN`,
`NeuralODE`, `generate_damped_pendulum_data(mu=mu_val, ...)`, and
`torch.optim.Adam` — the same components `train_kan_ode()` itself composes, just wired
by you instead of through `train.py`'s CLI. This is a small, self-contained script, not
a reimplementation of the whole pipeline. Include the X1 non-finite gradient guard
(`math.isfinite(gnorm)` before `optimizer.step()`, `train.py`'s pattern) — a μ sweep
covering stiff regimes is exactly the situation that guard was built for.

### Task breakdown

1. Write the standalone sweep script (above). Fix $G{=}8$, RBF basis, default SiLU,
  `--t_train_end 5.0`, `--grad_clip 1.0` — every axis except (solver, μ, Δt) held at
  the [`09`](./09_stability_fix_results.md)-validated defaults.
2. Grid: **4 solvers** {Euler, Midpoint, RK4, Tsit5} × **6 μ values** × **1–2 Δt**
  (start with just the default Δt=0.05; add a second value only if time allows —
  see budget below).
3. **Short budget per cell first.** Run the full grid at 2,000 epochs (probe-style,
  matching the discipline that found the pendulum's actual fix). Only extend the
  *promising or borderline* cells to 10,000 — don't spend the full budget on cells that
  visibly diverge in the first 2,000 epochs.
4. Classify each cell: **converged** (train MSE < some threshold, e.g. $10^{-2}$,
  finite final loss) / **unstable** (gradient guard fired, or final $\gg$ best) /
  **diverged** (NaN). Render as a 2D heatmap: μ (rows) × solver (columns), Δt as a
  facet if you ran a second value.
5. Write the finding: is there a μ threshold past which even the "safe" solvers
  (RK4, Tsit5) fail? Does Euler fail earlier than the others, consistent with Track B's
  gradient-noise finding if that lands first (cross-reference if useful, not required)?

### Compute budget

$4 \text{ solvers} \times 6\ \mu \times 2{,}000\text{ epochs} \times 0.77\text{ s/epoch}
\approx 24$ runs $\times \sim 26$ min $\approx 10.3$ h **sequential**. Solver cost
actually varies with NFE (Euler is cheaper per epoch, Tsit5 costlier — the $0.77$ figure
is Tsit5's measured cost, so treat this as an upper bound), and running 3-way parallel
(the pattern established in `run_phase2.ps1`, using `.WaitForExit()` — **not** the old
`Wait-Process` PID-reuse pattern documented as buggy in
[`07`](./07_fix_changelog.md)) brings it to **roughly 3.5–4 hours**.

### Deliverables

- `results/phase3/stiffness_map/stability_heatmap.png` — Table 5's figure
- `results/phase3/stiffness_map/table5.json` — every (solver, μ, Δt) cell's outcome
- `docs/16_p3_stiffness_map_findings.md`

### Definition of done

| Check | Target |
| :--- | :--- |
| Every cell classified | converged / unstable / diverged, no blanks |
| Non-finite gradient guard present in the sweep script | protects against burning compute on unrecoverable runs, per [`07`](./07_fix_changelog.md) §X1 |
| Baseline μ=0.5 cell matches known result | should reproduce `pendulum_control_win5`'s convergence at (Tsit5, μ=0.5, Δt=0.05) — a built-in sanity check |
| Written finding states a threshold or trend | not just "here is the heatmap" |

### Files you must not touch

Everything outside `experiments/stiffness_map/`, `results/phase3/stiffness_map/`,
`tests/test_p3_stiffness_map.py`, `docs/16_p3_stiffness_map_findings.md`.

### Optional stretch (only if the core sweep finishes early)

[`09`](./09_stability_fix_results.md) §Recommended next step identifies that
$\dot\theta = \omega$ is a **known, exact** relation the model currently gets 8.85%
wrong on the training manifold. A `SecondOrderField` wrapper — same pattern as
`ZeroSumField`/`VanishingDimField` in `ode/neural_ode.py`, but defined as a **new class
inside your own folder** (`experiments/stiffness_map/structural_field.py`), never
editing `ode/neural_ode.py` — hardcoding $f_1 = y_2$ and only learning $f_2$. Testing
whether this measurably shrinks the stable region of the phase map (fewer parameters to
mis-learn) would be a strong bonus finding, not required for the track to be complete.

---

## 🅴️ Track E — SINDy Comparison + Real Epidemic Fit  *(Novelty 4 + narrowed Novelty 6)*

**Owner:** Monjur Hossain Khan ("Shovon") (2105043) · **Branch:** `feat/p3-sindy-epidemic`
**Folder:** `experiments/sindy_epidemic/` · **Results:** `results/phase3/sindy_epidemic/`
**Write-up:** `docs/17_p3_sindy_epidemic_findings.md`

Two sub-tasks, bundled because both are "compare the learned model against an
independent form of ground truth" in flavour. In the original blueprint these were two
separate members' work (SINDy under one Novelty, epidemic fitting bundled with the
now-removed Lorenz under another) — merged here into a single track now that Lorenz is
gone.

### E1 — SINDy comparison under noise

**Research question.** Fit `pysindy.SINDy` (a sparse-regression symbolic-discovery
method — fundamentally different from KAN's gradient-based fitting) on noisy
Lotka-Volterra trajectories at $\sigma \in \{0.00, 0.01, 0.05, 0.10\}$ — the exact noise
levels already swept for KAN in [`05`](./05_phase2_benchmark_analysis.md) §Table 5 — and
compare robustness.

**What already exists to reuse.**
- `data.generate_lotka_volterra_data(noise_std=...)` — identical data generator KAN's
  own noise sweep used, so the comparison is apples-to-apples
- `results/benchmarks/noise/sigma*/metrics.json` — KAN's own noise-robustness numbers,
  **read-only citation**, do not re-run
- `pip install pysindy` — new dependency (§2.4)

**The building block you need to build yourself: L1 edge pruning.** The blueprint calls
for comparing SINDy's recovered symbolic terms against **KAN's $L_1$-pruned edges** —
this pruning step does not exist anywhere in the codebase (`utils/regularization.py`
only *penalises* parameter magnitude during training; it never zeros anything out
afterward). Build it as a small self-contained function in your own folder:

```python
# experiments/sindy_epidemic/pruning.py — new file, no edits to utils/
import copy
def prune_edges(model, threshold_percentile=50):
    """Zero the weakest-magnitude spline weights (KDense.C, per edge) below a
    percentile threshold. Returns a NEW model via a state_dict copy + mask —
    does not mutate the input model."""
    pruned = copy.deepcopy(model)
    for layer in pruned.layers:
        # C is [out_features, in_features * grid_len]; reduce per (in,out) edge
        ...  # your implementation
    return pruned
```

**⚠️ Conceptual gotcha, worth a paragraph in the write-up rather than something to
"fix."** A KAN edge is strictly **univariate** — each edge sees exactly one input
scalar. Lotka-Volterra's true dynamics contain a genuine **bilinear cross-term**
$\beta x y$. A single KAN layer's additive edge sum cannot represent that cross-term
directly (it requires composition across layers to approximate); SINDy's polynomial
library, by contrast, includes $xy$ as a literal candidate term and can recover it
exactly. This is an architectural difference worth reporting, not a bug in either
method — it directly explains why a term-by-term symbolic comparison is harder than it
sounds, and should shape how you frame the comparison (trajectory-reconstruction
accuracy under noise, rather than claiming literal formula equivalence).

**Task breakdown.**
1. Fit SINDy at each of the 4 noise levels (polynomial feature library, degree 2 is
  enough to include the true cross-terms).
2. Build `prune_edges()`, apply to the existing noise-sweep KAN checkpoints
  (`results/benchmarks/noise/sigma*/best_model.pt`) at a couple of pruning thresholds.
3. Compare: trajectory reconstruction MSE (SINDy vs. pruned-KAN vs. unpruned-KAN) at
  each $\sigma$; whether SINDy recovers something recognisably close to the true
  $(\alpha,\beta,\gamma,\delta)$ coefficients at each noise level (report where it
  breaks down).

**Compute budget.** SINDy fitting is fast (closed-form sparse regression, not
gradient-based) — minutes, not hours. Pruning is a forward-pass evaluation of existing
checkpoints, also fast. **Under 30 minutes of actual compute**; the work here is code
and analysis, not waiting for runs.

### E2 — Real epidemic data fit

**Research question.** With Lorenz removed, this is what Novelty 6 becomes: fit the
now-**fixed** SIR pipeline to a realistic (synthetic-but-structured) outbreak curve and
see whether the [`10`](./10_sir_root_cause_and_fix.md) fixes — `--time_scale`,
`--conserve_mode projection`, optionally `--vanish_dim` — generalise to a case where the
"true" equations aren't the exact generator of the data.

**What already exists to reuse.**
- `data.load_empirical_epidemic_data()` — Phase-1-built, asymmetric log-normal outbreak
  wave with realistic noise and smoothing, **never used in any run yet**
- The exact `--time_scale`/`--conserve_mode projection` recipe from
  [`10`](./10_sir_root_cause_and_fix.md), reusable via your own thin script (same
  reuse pattern as Track D — call `KAN`, `NeuralODE`, `ZeroSumField` directly)

**Task breakdown.**
1. Load `load_empirical_epidemic_data()`; note it returns a **2D** state
  (infected + cumulative recovered, normalised $[0,1]$), not SIR's 3D — check whether
  `--conserve_mode projection` even applies here (it assumes a zero-sum invariant that
  a 2D infected/recovered pair does not literally have) and **document that reasoning**
  rather than blindly reapplying it.
2. Determine an appropriate `--time_scale` the same way [`10`](./10_sir_root_cause_and_fix.md)
  did — measure $|f_{\text{init}}|$ vs. the data's own derivative scale *before*
  training (the diagnostic that cracked SIR was four numbers measured at
  initialisation, not a training run).
3. Train, evaluate train/extrapolation split, report whether the fixed SIR machinery
  transfers to non-mechanistic data.

**Compute budget.** A handful of 5,000–10,000-epoch runs at roughly SIR's measured cost
($1.1$–$1.4$ s/epoch) — $\sim 2$–$3$ hours.

### Deliverables (both sub-tasks)

- `results/phase3/sindy_epidemic/sindy_vs_kan_noise.json` + comparison figure
- `results/phase3/sindy_epidemic/pruning.py` output on the noise-sweep checkpoints
- `results/phase3/sindy_epidemic/real_epidemic_fit.png` + metrics JSON
- `docs/17_p3_sindy_epidemic_findings.md` covering both

### Definition of done

| Check | Target |
| :--- | :--- |
| SINDy fit at all 4 noise levels | including where it visibly breaks down |
| Pruning utility applied to existing checkpoints | not retrained from scratch |
| Univariate-edge limitation discussed explicitly | in the write-up, not silently glossed over |
| Real epidemic run's `time_scale` choice justified by an init-time measurement | not guessed |
| Whether `conserve_mode projection` was applicable is explicitly reasoned about | stated even if the answer is "not directly, because..." |

### Files you must not touch

Everything outside `experiments/sindy_epidemic/`, `results/phase3/sindy_epidemic/`,
`tests/test_p3_sindy_epidemic.py`, `docs/17_p3_sindy_epidemic_findings.md`.
**Read-only** access to `results/benchmarks/noise/`.

---

# Part 5 — Synthesis, Sequencing, and What Comes After

## 5.1 Doc numbers pre-assigned (so parallel branches never collide on a filename)

| Track | Owner | Write-up |
| :--- | :--- | :--- |
| A | Nawriz | `docs/13_p3_adjoint_profiling_findings.md` |
| B | Abhishek | `docs/14_p3_gradient_dynamics_findings.md` |
| C | Shams | `docs/15_p3_hybrid_basis_findings.md` |
| D | Abrar | `docs/16_p3_stiffness_map_findings.md` |
| E | Monjur | `docs/17_p3_sindy_epidemic_findings.md` |

## 5.2 Suggested start order (not a hard dependency — all five can start immediately)

If sequencing matters for your own scheduling: **B starts first** (zero compute, could
be done same-day) and can informally inform D's write-up if it finishes first (optional
cross-reference, not a blocker). **D should start early** given its 3.5–4 h compute
tail. A, C, and E can start whenever convenient.

## 5.3 Installing what you need, without touching `requirements.txt`

```powershell
# Track A only
pip install torchdiffeq

# Track E only
pip install pysindy
```

Note the exact version installed (`pip show torchdiffeq` / `pip show pysindy`) in your
own `README.md` — needed for the one-line `requirements.txt` PR at final integration
(§2.4).

## 5.4 After all five merge — brief Phase 4 preview

Not part of this plan's scope, but worth naming so nobody is surprised: Phase 4
(`PROJECT_IMPLEMENTATION_PLAN.md` §Phase 4) is multi-seed replication ($N{=}5$) of the
findings that matter most for the final report, publication-grade figure regeneration,
and LaTeX synthesis — Abrar's existing figures/paper-lead role (separate from, and
unaffected by, the Track D assignment above) continues there. Nothing in Phase 3 should
be blocked on Phase 4, and no Phase 3 track needs to pre-emptively add seed loops —
that discipline is Phase 4's job, applied selectively to whichever Phase 3 findings turn
out to matter for the report.

---

*Previous: [`11_phase2_closeout.md`](./11_phase2_closeout.md) — Phase 2 is closed except
for the deliberately-dropped Lorenz and Δt=0.01 items, both finalised as removed in
§1.2 above.*
