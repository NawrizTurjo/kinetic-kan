# Track D: Stiffness–Solver Stability Phase Map — Findings

> **Owner:** Abrar Jahin (2105055) · **Branch:** `feat/p3-stiffness-map`
> **Chain:** [`12_phase3_roadmap.md`](./12_phase3_roadmap.md) Part 4, Track D (original scope) → this doc
> **Status:** Complete. Core sweep re-verified across all 4 solvers × 6 μ ×
> 3 seeds (72 cells total, uniformly retrained with checkpoints saved for
> every cell). The μ=2.0 mechanistic trap (§3) and the seed-sensitivity
> pattern (§4) are confirmed for all 4 solvers independently, not just
> euler. A cross-platform reproducibility gap (§7) was found along the
> way and is reported honestly rather than smoothed over. The training
> *loss*-landscape (weight-space, as opposed to this doc's phase-space
> vector-field landscape) has also been run for all 24 cells and checked
> against 2 more random measurement choices per cell — see §8.

---

## 0. Summary

The original goal was a straightforward stability heatmap: sweep damping ratio
μ × solver, see where things blow up. **Nothing blew up anywhere** — zero cells
diverged or hit the non-finite-gradient guard across all 24 (solver, μ) cells,
consistent with the classical stability-theory prediction that Δt=0.05 is
comfortably inside every tested solver's stability region even at the stiffest
μ tried (8.0). That prediction, and why it holds, is in §1.

Instead, the sweep surfaced something more interesting: **at several μ values,
all four solvers failed identically** — not from instability, but by converging
to the same wrong answer. Investigating *why* (§2–§3) led to a discovery that
changed how the rest of this track was run: the failures are not properties of
μ at all, but of **which random seed the run happened to start from** (§4).
That finding forced a methodology change, described and applied in §5.

**This track's real contribution is a mechanism, and section 3 is where it
lives.** §3 shows directly, from every solver's own trained weights, that a bad seed at
μ=2.0 produces a genuine second equilibrium in the learned dynamics — a
spurious fixed point ~0.05 units from the true one — verified independently
for all 4 solvers via the vector-field "landscape" plots
(`results/phase3/stiffness_map/figures/mu*/phase_portrait_3d_mu*.png`). The
converged-fraction numbers in §4 are real but statistically weak (n=3 seeds,
§6) — the mechanistic confirmation in §3 is what this track can actually
defend without needing a larger seed count. §7 reports an incidental but
real cross-platform reproducibility gap found while scaling this up on
Kaggle; §8 previews a still-pending training-loss-landscape analysis.

---

## 1. The stability question: settled quickly, in favor of "nothing here"

**Prediction (derived before running anything):** linearizing the damped
pendulum at equilibrium, the eigenvalues are `λ = (-μ ± √(μ²-4g/L))/2`. For
underdamped μ (< 2√(g/L) ≈ 6.26, i.e. μ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}), the
modulus `|λ| = √(g/L) ≈ 3.13` is *constant*, independent of μ. At Δt=0.05, that
gives `h|λ| ≈ 0.157` for every underdamped cell — far inside even Euler's
smallest stability region (`h|λ| ≤ 2` on the real axis). Only μ=8.0
(overdamped) introduces a fast eigenvalue (`λ_fast ≈ -7.7`), and even there
`h|λ_fast| ≈ 0.385` stays well inside every solver's region.

**Result:** confirmed exactly. `nonfinite_grad_steps = 0` and
`aborted_at_epoch = null` for all 24 cells. Whatever varies across this grid,
it isn't classical forward-integration stability — this project's own earlier
diagnosis of *training*-instability (docs 09/10, the `e^{L·T}` backprop
sensitivity mechanism) doesn't manifest here either, likely because μ=8.0's
`T=5` horizon is short enough that it never approaches the regime where that
mechanism dominated for SIR.

**Scoping note: all 4 solvers tested are explicit, single-step Runge-Kutta
methods.** `implementation/ode/solvers.py`'s full registry is
`{tsit5, rk4, dopri5, euler, midpoint, heun}` — every one of them has a
*bounded* absolute-stability region (Dahlquist theory, §1's own basis), which
is exactly why a hard step-size limit exists to check against in the first
place. No implicit / Backward-Differentiation-Formula (BDF) solver is
implemented anywhere in this codebase. BDF methods are specifically built for
stiff systems because they're typically A-stable or L-stable — stable at
*any* step size for a linear test system like this one, with no bounded
region to compute at all. This track's title says "stiffness," but its scope
is really "how do explicit RK methods of different orders behave as μ
increases" — it cannot say anything about whether an implicit method would
have behaved differently, because none was ever in scope to compare against.
Given §1's own prediction (Δt=0.05 stays deep inside even Euler's stability
region at every tested μ), this project's pendulum setup may never actually
reach a regime "stiff enough" to separate explicit methods from an implicit
one in the first place — a genuinely stiffer configuration (larger μ, or a
much larger Δt) would be needed to make that comparison meaningful.

---

## 2. The original 24-cell sweep (single seed = 42)

*(Numbers below are from the uniform 72-cell retrain described in §7 — all
four solvers retrained together, same script, same session, with
checkpoints saved for every cell. Verdicts are identical to the very first
run of this sweep; the exact MSE values shifted slightly, which is itself
the finding reported in §7, not noise to ignore.)*

| μ | euler | midpoint | rk4 | tsit5 |
|---:|---:|---:|---:|---:|
| 0.1 | unstable 1.7e-2 | unstable 3.0e-2 | unstable 2.2e-2 | unstable 2.4e-2 |
| 0.5 | **converged** 4.6e-3 | **converged** 7.9e-3 | **converged** 4.7e-3 | **converged** 9.2e-3 |
| 1.0 | unstable 1.0e-1 | converged 5.4e-4 | converged 5.3e-4 | converged 4.0e-4 |
| 2.0 | unstable 1.8e-2 | unstable 1.8e-2 | unstable 1.8e-2 | unstable 1.8e-2 |
| 5.0 | converged 6.1e-5 | converged 7.0e-5 | converged 6.9e-5 | converged 6.8e-5 |
| 8.0 | converged 2.1e-5 | converged 2.3e-5 | converged 2.6e-5 | converged 2.6e-5 |

Two things stood out immediately:
- **μ=2.0 fails identically across all four solvers**, agreeing to 2
  significant figures (1.8e-2) despite completely different integration
  schemes.
- **μ=1.0 is euler-specific**: euler fails while the other three converge
  cleanly, and euler's μ=1.0 result (1.0e-1) is even worse than its own μ=0.1
  result.

At the time, this looked like it might be a genuine solver-order effect (the
whole point of a "stability phase map"). §3–§4 below found otherwise.

---

## 3. Investigating μ=2.0: ruling out the obvious explanations

Two natural hypotheses were tested and refuted before finding the real cause:

**Hypothesis: not enough training data near the equilibrium.** Refuted.
μ=2.0 actually has *more* near-origin training samples (37, closest approach
0.022) than μ=0.5 or μ=1.0 (0 samples within 0.1/0.2, closest approach
0.60/0.21) — both of which converge fine regardless.

**Hypothesis: the small residual dynamics near equilibrium are drowned out by
the large early-transient loss.** Refuted. μ=5.0 and μ=8.0 have even *smaller*
late-window energy fractions than μ=2.0 (essentially zero) and still converge
to near-perfect precision.

**What actually explains it, confirmed directly from the trained model's
weights:** a checkpoint was saved for euler's μ=2.0 run (`--save-checkpoints`,
added to `run_sweep.py` specifically for this investigation) and evaluated at
two points:

| Point | `f_θ` | `\|f_θ\|` |
|---|---|---:|
| True equilibrium (0, 0) | (0.088, 2.735) | **2.74** |
| Nearby grid-search minimum (-0.05, -0.02) | (0.008, 0.004) | **0.0086** |

The model's learned vector field has a *large* nonzero derivative at the true
equilibrium — (0,0) is nowhere close to a fixed point of what it learned — and
instead has a genuine, stable, spurious equilibrium about 0.05–0.07 units away.
Verified further: the predicted trajectory stays pinned within ±0.003 of that
spurious point for the entire extrapolation window (t=5 to t=10, five seconds
never seen in training), for all four solvers independently. This is a real,
solver-independent local-minimum trap in the *learned dynamics*, not a
numerical integration artifact — all four solvers were correctly integrating
the same wrong field.

One architectural fact ruled out a specific mechanism for free, no training
needed: `KDense`'s residual pathway is `W·SiLU(x)`, and `SiLU(0) = 0` exactly,
for any `W`. So the residual/skip-connection term cannot be the source of a
bias at the origin — only the spline/RBF term can, and does.

**Update.** Once every solver had its own saved checkpoint (§7), we could
finally check whether this holds up beyond euler. It does: `f_θ(0,0)` was
evaluated directly from each solver's own model at μ=2.0:

| Solver | `f_θ(0,0)` | `\|f_θ(0,0)\|` |
|---|---|---:|
| euler | (0.062, 2.772) | 2.772 |
| midpoint | (0.048, 2.876) | 2.876 |
| rk4 | (0.038, 2.654) | 2.654 |
| tsit5 | (0.074, 2.854) | 2.855 |

All four independently-trained models have essentially the same large
nonzero derivative at the true equilibrium (magnitude 2.65–2.88) — this was
previously only checked for euler and *assumed* to generalize because all
four solvers integrated the same wrong trajectory under a shared seed; it is
now a directly verified result, not an assumption.

**Visual confirmation — the vector-field "landscape."** `plot_landscape.py`
renders `|f_θ(θ, ω)|` as an actual 3D terrain (a genuine analogue of a
textbook phase-portrait, not a training loss surface — see §8 for that
distinction) with the real trajectory drawn as a path across it. At μ=2.0,
every solver's own terrain shows the true equilibrium (marked with a star)
sitting on a slope, not in the bottom of a valley, while the trajectory
visibly settles into a nearby, separate low point instead
(`results/phase3/stiffness_map/figures/mu2.0/phase_portrait_3d_mu2.0.png`).
The same tool applied at μ=1.0 shows the more interesting mixed case
directly: euler's own terrain traps its trajectory away from the true
equilibrium while midpoint/rk4/tsit5's own terrains each show a clean dive
into the correct valley
(`.../mu1.0/phase_portrait_3d_mu1.0.png`) — the euler-specific gap from §2 is
visible as a structural difference in the *learned dynamics themselves*, not
just a difference in a summary MSE number.

These plots surfaced one more small thing worth stating plainly. Even a
model we call "converged" doesn't necessarily put its own lowest point
exactly at (0, 0). Checking rk4's μ=1.0 model directly — a solver with a
clean converged verdict — its field at the true equilibrium has magnitude
0.175, but a fine grid search finds an even lower point nearby, at
(θ≈0.03, ω≈-0.83), with magnitude 0.078. The gap between these two points is
small enough that the training-MSE-based "converged" label still holds up —
the trajectory this model produces does track the real pendulum closely. But
it shows that "converged" measures how well the trajectory matches the data;
whether the model's own idea of equilibrium sits exactly on the physical one
is a separate question the training loss doesn't directly enforce. The μ=2.0
trap earlier in this section is the same gap grown large enough to produce a
genuinely wrong answer instead of a harmless rounding difference.

---

## 4. The discovery that changed the methodology: blame the seed, not μ

The obvious next question — is this trap a deterministic property of μ=2.0,
or does it depend on the random initialization? — was tested directly: euler
was retrained at μ=2.0 with two more seeds (1, 7), all else identical.

| Seed | Verdict | best_train_mse |
|---|---|---:|
| 42 (original) | unstable | 1.82e-2 |
| 1 | unstable | 1.68e-2 |
| 7 | **converged** | **7.09e-5** |

**Seed 7 escapes the trap entirely**, reaching essentially the same quality as
μ=5.0/8.0's best results. The trap is seed-dependent, not structural.

This raised an obvious concern: if μ=2.0 is seed-sensitive, are the *other*
μ values too — and is the original single-seed sweep (§2) actually telling us
anything reliable about μ at all, or just about the luck of sharing seed=42
across every solver? This was checked directly: euler was retrained at every
remaining μ value (0.1, 0.5, 1.0, 5.0, 8.0) with the same two extra seeds.

| μ | seed=42 | seed=1 | seed=7 |
|---:|---|---|---|
| 0.1 | unstable | unstable | unstable |
| 0.5 | converged | **unstable** | converged |
| 1.0 | unstable | unstable | converged |
| 2.0 | unstable | unstable | converged |
| 5.0 | converged | converged | converged |
| 8.0 | converged | converged | converged |

**Only the two extremes are seed-robust.** μ=0.1 (trajectory never settles
within the training window) is uniformly hard regardless of seed. μ=5.0 and
8.0 (trajectory settles quickly, well inside the window) are uniformly easy
regardless of seed. **Every transitional μ value — 0.5, 1.0, 2.0, where the
trajectory partially but not fully settles — has at least one of three tested
seeds land in a bad local minimum.**

### Update — extended to all 4 solvers (§7's uniform retrain)

The table above was euler-only when first written. `kaggle_full_retrain.py`
retrained **every** (solver, μ, seed) combination — 4 solvers × 6 μ × 3 seeds
(42, 1, 7) = 72 cells, checkpoints included — closing that gap directly
instead of leaving it as an assumption. Converged-fraction (over 3 seeds) for
every solver:

| μ | euler | midpoint | rk4 | tsit5 |
|---:|---:|---:|---:|---:|
| 0.1 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.5 | 0.67 | 0.67 | 0.67 | 0.67 |
| 1.0 | **0.33** | **0.67** | **0.67** | **0.67** |
| 2.0 | 0.33 | 0.33 | 0.33 | 0.33 |
| 5.0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8.0 | 1.00 | 1.00 | 1.00 | 1.00 |

μ=0.1/5.0/8.0 (the seed-robust extremes) and μ=0.5/2.0 generalize exactly as
the euler-only table predicted — identical converged-fraction across all 4
solvers. **μ=1.0 is the one genuine exception**: euler is worse (0.33) than
the other three (0.67). The per-seed breakdown explains why, and it's more
specific than "euler is worse at μ=1.0" — it's that **two different seeds are
bad for two different reasons**:

| Solver | seed=42 | seed=1 | seed=7 |
|---|---|---|---|
| euler | **unstable** (1.03e-1) | **unstable** (9.77e-2) | converged (6.26e-4) |
| midpoint | converged (5.40e-4) | **unstable** (9.68e-2) | converged (5.91e-4) |
| rk4 | converged (5.29e-4) | **unstable** (9.81e-2) | converged (5.14e-4) |
| tsit5 | converged (4.01e-4) | **unstable** (9.71e-2) | converged (5.90e-4) |

**Seed=1 is a universally bad initialization at μ=1.0** — all 4 solvers land
in the same failure mode (MSE ≈ 9.7–9.8e-2, essentially identical across
solvers, the same signature as μ=2.0's shared-trap pattern in §3). **Seed=42
is additionally, uniquely bad for euler only** — the other three converge
cleanly under seed=42. So μ=1.0's euler-specific gap in the original §2 table
was real, but it is the *sum* of a seed-1 failure every solver shares plus a
second, euler-only failure under seed=42 — not evidence that euler is simply
"worse at μ=1.0" in general.

median best_train_mse (over 3 seeds) tells the same story with the actual
error magnitudes:

| μ | euler | midpoint | rk4 | tsit5 |
|---:|---:|---:|---:|---:|
| 0.1 | 8.3e-2 | 3.0e-2 | 1.9e-2 | 2.4e-2 |
| 0.5 | 4.6e-3 | 7.9e-3 | 4.7e-3 | 9.2e-3 |
| 1.0 | 9.8e-2 | 5.9e-4 | 5.3e-4 | 5.9e-4 |
| 2.0 | 1.7e-2 | 1.7e-2 | 1.7e-2 | 1.7e-2 |
| 5.0 | 6.1e-5 | 5.6e-5 | 5.7e-5 | 5.7e-5 |
| 8.0 | 2.1e-5 | 2.3e-5 | 2.6e-5 | 2.6e-5 |

Euler's μ=1.0 median (9.8e-2) is two orders of magnitude worse than the other
three (~5.5e-4) — the euler-specific gap is not a rounding difference, it's a
qualitatively different outcome, and it is now a *measured* fact across all 4
solvers rather than an inference from euler alone.

### Why this matters more than the original heatmap

The 24-cell sweep in §2 used **one shared seed for all four solvers**. Given
the table above, "all four solvers fail identically at μ=2.0" was never
evidence that solvers agree on stiffness — it was four solvers inheriting the
exact same coin flip from one shared initialization. **At this Δt, solver
choice does not appear to determine outcome; the interaction between random
seed and μ does.** A stability map built from N=1 seed cannot distinguish "this
μ is genuinely hard" from "this μ got an unlucky draw" — and half of the six μ
values tested here would have looked deceptively clean or deceptively broken
depending purely on which single seed had been used.

This is the same gap the project's own roadmap already named in the abstract
(Phase 4: *"every number in the project is N=1; no ordering claim in Table 1
is publishable without [multi-seed replication]"*) — this track just produced
a concrete, mechanistic demonstration of exactly why that caveat is load-bearing,
rather than a boilerplate limitation.

---

## 5. Methodology change made in response

**Because of this finding, per-cell verdicts from a single seed are no longer
treated as reliable on their own.** Going forward for this track, a cell's
reported outcome is the **aggregate over multiple seeds**, not one run.

Two aggregation choices were compared on the data already in hand (seeds
42, 1, 7 for euler across all six μ):

| μ | mean MSE | median MSE | converged fraction |
|---:|---:|---:|---:|
| 0.1 | 1.064e+0 | 7.88e-2 | 0.00 |
| 0.5 | 9.122e-2 | 4.67e-3 | 0.67 |
| 1.0 | 6.725e-2 | 9.79e-2 | 0.33 |
| 2.0 | 1.171e-2 | 1.68e-2 | 0.33 |
| 5.0 | 5.694e-5 | 6.03e-5 | 1.00 |
| 8.0 | 2.344e-5 | 2.09e-5 | 1.00 |

**The mean is not a good summary statistic here and is reported only for
comparison.** μ=0.1's mean (1.064) is dragged an order of magnitude above its
own median (0.0788) by a single outlier seed (3.08) — the underlying
distribution is bimodal (a cell either lands in a good basin or a bad one),
not a spread of similar values a mean would meaningfully describe.
**`median` and `converged_fraction` are adopted as the reported statistics**
for any future aggregation in this track: median is robust to the outlier
runs a trapped seed produces, and converged-fraction directly answers the
question a stability map is supposed to answer ("how often does this
configuration actually work?") without pretending a single number can capture
a bimodal outcome.

---

## 6. Honest limitations of this investigation

- **Only 3 seeds per μ.** A converged-fraction of 0.33 from 3 seeds has wide
  uncertainty; distinguishing "genuinely ~1/3 of seeds fail here" from
  "~1/2 or ~1/5 fail here" needs more seeds than were run for this
  investigation. This is now confirmed across all 4 solvers (§4), not just
  euler, but the seed *count* itself hasn't grown — 4× more cells at the same
  n=3 doesn't resolve the underlying statistical weakness of n=3.
- **No formal criterion was set in advance for "how many seeds is enough."**
  The choice to stop at 3 was pragmatic (compute budget), not principled.
- **What this track can actually defend is the mechanism, not a failure
  rate.** Given the n=3 limitation above cannot be fixed
  without materially more compute, this track leans on §3's mechanistic
  result (a spurious equilibrium confirmed directly from trained weights,
  independent of any seed-count argument) as its headline finding, rather
  than the converged-fraction numbers in §4 — those numbers are still
  reported because they motivated and are illustrated by the mechanistic
  story, but they are not claimed as a statistically powered result.
- **The two originally-resolved gaps** (only euler tested across seeds; only
  euler's checkpoint saved) **are now closed** — §4 and §3 both report
  results for all 4 solvers independently, via the uniform 72-cell retrain in
  §7.

These are exactly the kind of gaps Phase 4 (multi-seed replication) is meant
to close project-wide — this track's honest position is that its headline
numbers (§2's heatmap) are suggestive, not confirmed by a statistically
powered seed count, even though the *mechanism* behind them (§3) is directly
verified.

---

## 7. Cross-platform reproducibility gap (found incidentally, reported honestly)

While extending this track's infrastructure to run on Kaggle (for compute the
local machine couldn't provide in reasonable time), the same (solver=tsit5,
μ=0.5, Δt=0.05) sanity cell already validated locally against
`pendulum_control_win5` (see `experiments/stiffness_map/README.md`) was run
again on Kaggle, same seed, same code, same epoch budget:

| Metric | Reference (`pendulum_control_win5`) | Kaggle re-run | Ratio |
|---|---:|---:|---:|
| `best_train_mse` | 9.28e-5 | 6.17e-5 | 0.66× |
| `full_mse` | 4.48e-2 | 9.91e-2 | 2.21× |
| `extrap_r2` | 0.6472 | 0.2188 | 0.34× |

Training fit (`best_train_mse`) is actually *better* on Kaggle, but
extrapolation quality is substantially worse — `extrap_r2` drops by roughly
two-thirds. Same finding, independently, at smaller scale: the uniform
72-cell retrain's numbers (§2's table) shifted from the very first local run
of the same 24 (solver, μ) cells (e.g. euler at μ=0.1: 2.9e-2 locally →
1.7e-2 on Kaggle) while every verdict (converged/unstable) stayed identical.

**Interpretation:** with the same seed and code, floating-point summation
order differs across platforms/BLAS backends (local Windows vs. Kaggle's
Linux CPU environment), and this project's models are sensitive enough to
that difference — compounded over thousands of nonlinear optimization steps
— to produce meaningfully different final numbers, especially in
extrapolation behavior. This is **not** the same phenomenon as §4's
seed-sensitivity (a different random initialization landing in a different
basin) — this is nominally the *same* run landing at a measurably different
point purely from platform-level floating-point non-determinism. It's
reported here because it's a real, verified fact about this project's
reproducibility, not because it changes any of this track's qualitative
conclusions (verdicts were unaffected everywhere it was checked).

---

## 8. The shape of the training loss around each solution

Sections 3 and 4 look at what a trained model actually does — where its
equilibrium sits, whether it lands there reliably across seeds. This section
asks a different question: once training finished, how forgiving was the
spot it settled into? If you nudged the weights slightly in some direction,
would the fit barely change, or would it fall apart? A model sitting in a
broad, gentle dip is more robust to small perturbations than one sitting at
the bottom of a narrow, steep well, even if both reach the same final error.

To measure this without trying to visualize a space with hundreds of
dimensions (this model has 360 weights in total — small by deep-learning
standards, but still far too many to plot directly), we picked each trained
model's actual final weights, chose two
random directions to step away from them, and re-measured the real training
error at a grid of small steps in those two directions. That produces an
ordinary three-dimensional surface — the two step sizes on the flat plane,
the resulting error as height — for something that is normally impossible to
draw. This is a standard technique from the deep learning literature (Li et
al., *Visualizing the Loss Landscape of Neural Nets*), adapted here to KAN-ODE
training. Every plot shows the same shape near its center: the trained point
sits at the bottom of a dip, and error rises as you move away from it. That
part is expected everywhere and is not itself informative — a trained model
is, by definition, sitting at a local low point, so a dip has to be there. What
differs, and what we actually measured, is how quickly that dip rises: some
climb steeply within a small step, others climb much more gradually over the
same distance.

We summarize each cell with one number, the "rise": how many orders of
magnitude the error climbs by the time you've stepped away from the trained
point, compared to the error you started at. A small rise means the dip is
wide and gentle; a large rise means it's narrow and steep.

| μ | euler | midpoint | rk4 | tsit5 |
|---:|---:|---:|---:|---:|
| 0.1 | 5.36 | 5.55 | 8.30 | 6.02 |
| 0.5 | 10.92 | 9.55 | 10.81 | 9.52 |
| 1.0 | **5.53** | **9.03** | **8.99** | **9.12** |
| 2.0 | **4.23** | **4.24** | **4.22** | **4.20** |
| 5.0 | 8.75 | 8.75 | 8.76 | 8.76 |
| 8.0 | 9.14 | 9.15 | 9.12 | 9.12 |

The pattern lines up exactly with what the rest of this document already
found, and it points the opposite way from a common rule of thumb in the
machine learning literature (that a wide, gentle minimum tends to be the
better-behaved one):

- **Every solver's μ=2.0 model — the one that fell into the wrong,
  spurious equilibrium described in §3 — sits in the widest, gentlest dip
  of any cell we measured**, roughly half the rise of every other μ. The
  four solvers didn't just agree on the wrong answer; they landed in the
  same easy-to-reach, low-effort kind of minimum to get there.
- **At μ=1.0, the split falls exactly on the line we already knew about.**
  Euler, the one solver that failed here, shows the same shallow rise as
  the trapped μ=2.0 models. Midpoint, RK4, and Tsit5, which all converged
  correctly, show a steep rise matching the other successful μ values. This
  is a second, independent measurement — coming from the shape of the
  training loss, nothing to do with the pendulum's physics directly — and it
  draws the same euler-versus-the-rest line that the seed experiments in §4
  already drew from a completely different angle.
- **Reading this carefully**: a model that got stuck in the wrong answer
  seems to have landed somewhere generic and easy to fall into — many nearby
  weight settings give roughly the same mediocre result. A model that
  actually learned the pendulum's real dynamics sits somewhere much more
  exacting — getting a real trajectory right in detail seems to require
  landing close to one specific, narrow setting of the weights, not just
  somewhere in a broad, forgiving region. That runs against the usual
  intuition that flatter is safer; here, flat is where training gave up, and
  steep is where it actually worked.

A few honest caveats about the table above. The model itself is small — 360
parameters in total (two KAN layers, `[2, 10, 2]` with an 8-point grid) — so
this only looks along two directions out of 360 available in the model's
full weight space. That's still a small slice, but a far larger fraction of
the whole space than it would be for a typical deep network with thousands
or millions of parameters, so a two-direction view is more representative
here than the same technique would be elsewhere. Even so, it is a slice, not
the whole space, and we didn't just note that as a limitation — we went and
checked it directly.

### Checking it: does the rise number survive a different choice of direction?

We reran the same measurement two more times per cell, each time with a
completely different pair of random directions, giving three independent
"rise" values per (solver, μ) combination instead of one. The honest answer
is: **the exact number is not stable, but the comparison between cells is.**

One of the three direction-pairs turned out to be an outlier almost
everywhere. Because every model here shares the exact same architecture, the
same fixed random draw produces essentially the same raw direction (before
it gets rescaled to each model's own weights) for every single cell — and
that particular draw happened to run toward a much steeper wall than the
other two, inflating its rise number by roughly two to three times, in every
one of the 24 cells, not just a few. That is not noise; it is one specific
unlucky direction doing the same thing everywhere.

| μ | euler | midpoint | rk4 | tsit5 |
|---:|---:|---:|---:|---:|
| 0.1 | 5.4 / 5.1 / 13.2 | 5.6 / 9.1 / 13.3 | 8.3 / 5.6 / 17.9 | 6.0 / 5.2 / 15.8 |
| 0.5 | 10.9 / 5.3 / 17.9 | 9.6 / 5.2 / 20.7 | 10.8 / 5.4 / 22.1 | 9.5 / 5.5 / 20.7 |
| 1.0 | **5.5 / 3.9 / 11.7** | 9.0 / 6.8 / 17.2 | 9.0 / 6.6 / 16.2 | 9.1 / 6.7 / 16.6 |
| 2.0 | **4.2 / 4.7 / 13.1** | **4.2 / 4.7 / 13.7** | **4.2 / 4.7 / 13.8** | **4.2 / 4.7 / 13.6** |
| 5.0 | 8.8 / 7.1 / 13.9 | 8.8 / 7.0 / 14.6 | 8.8 / 7.0 / 14.6 | 8.8 / 7.0 / 14.6 |
| 8.0 | 9.1 / 7.3 / 13.7 | 9.2 / 7.5 / 13.1 | 9.1 / 7.4 / 13.6 | 9.1 / 7.4 / 13.7 |

(each cell: the three individual rise values, in the order the direction
pairs were tried)

Despite that outlier inflating every absolute number, **the pattern this
section is actually built on survives untouched**: averaging all three
values together, μ=2.0 is still the lowest, flattest row for midpoint, RK4,
and Tsit5, and euler's own two trapped μ values (1.0 and 2.0) still sit
clearly below its own μ=0.5/5.0/8.0 results. The ranking — trapped models
sit in flatter basins than converged ones — held up under a direction that,
by accident, was considerably more aggressive than the other two. Two figures
make this visible directly rather than asking you to trust the averages:
`results/phase3/stiffness_map/loss_landscape/multidirection_comparison_euler_mu2.0.png`
shows euler's μ=2.0 model under all three direction-pairs side by side — the
third panel's outlier spike is obvious, but the shallow valley near the
trained point looks the same in all three. The companion pair,
`multidirection_comparison_euler_mu1.0.png` and
`multidirection_comparison_tsit5_mu1.0.png`, shows the same test at μ=1.0 for
the one solver that failed there versus one that succeeded: euler's dip
never goes below roughly zero on the log scale in any of the three views,
while tsit5's drops to around -3 or -4 in every one of them, outlier
direction included.

So: the exact "rise" figure for any single cell should not be read as a
precise measurement — a different random direction can move it by an order
of magnitude or more. What should be trusted is the comparison between
cells, which is where every claim in this section is actually anchored, and
which three independent direction choices all agreed on.

The rise numbers are also affected by how small each model's starting error
already was — μ=8.0's errors are tiny to begin with (around 0.00002), so even
a small absolute increase looks like a large jump on this scale, which is why
its rise number isn't directly comparable in an absolute sense to, say,
μ=0.1's. The comparisons this section actually relies on — trapped versus
converged models at the *same* μ, where the starting error is close to the
same size to begin with — don't run into that problem.
