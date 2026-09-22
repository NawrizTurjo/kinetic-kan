# Phase 4: Epoch-Budget Sensitivity Check — Findings

> **Owners:** Monjur & Shams (2105048) · **Chain:** [`18_phase4_roadmap.md`](./18_phase4_roadmap.md)
> §2 (the question and the plan) → [`19_kaggle_execution_guide.md`](./19_kaggle_execution_guide.md)
> (how it was run) → **this doc** (what came back).
> **Status:** All 5 representative configs trained and downloaded from Kaggle,
> checkpoints/metrics verified byte-for-byte against `training_history.json`'s own
> epoch counts, results placed under `implementation/results/phase4/epoch_budget_check/`.
> Phase 2's original `results/benchmarks/` is untouched. This closes out Phase 4's
> compute-running part of §2; §3 (baseline-vs-paper audit) is a separate, still-open,
> reading-only task.

---

## 0. Summary

**The ranking is *not* stable — and the single biggest change is the most consequential
one in the whole project.** At 10,000 epochs, [`05`](./05_phase2_benchmark_analysis.md)
Table 3 reports KAN-ODE generalizing $5.35\times$ better than the parameter-matched
SiLU-MLP baseline — reproducing the paper's central qualitative claim. At 50,000 epochs,
**that reverses**: the SiLU-MLP's extrapolation MSE drops to $5.90\times10^{-6}$, becoming
$6.82\times$ *better* than KAN-ODE's $4.02\times10^{-5}$. The "KAN-ODE generalises better"
finding, as measured on this specific problem under this specific protocol, is an
artifact of the 10k-epoch budget, not a property that holds with more training.

Three other, smaller results:

- **RBF vs. B-spline's "statistically tied" basis result survives** — $6.1\%$ apart at
  10k, $5.6\%$ apart at 25k/50k, well inside the margin either finding calls a tie. This
  part of Table 2 is not what changes.
- **Euler's gap to Tsit5 narrows sharply but does not close** — from $28.1\times$ worse
  extrapolation at 10k to $4.8\times$ worse at 50k. Euler is still visibly the weakest
  solver, but a meaningful fraction of Table 1's Euler-vs-everyone-else gap was indeed
  just Euler needing more time, exactly as §2.2 predicted it might.
- **The paper-spec MLP (`[2,50,2]`+tanh) still does not train at 50k** — train MSE
  barely moves ($1.083 \to 1.011$) and extrapolation $R^2$ actually gets *worse*
  ($0.4215 \to -0.484$). This is a direct, already-answered data point for §3's audit:
  this baseline's failure is an implementation/architecture story, not an epoch-budget
  one.

Also recorded along the way: real Kaggle CPU throughput ran **~2.3–2.5$\times$ slower
per epoch** than the local machine for every KAN/solver config (not just B-spline), and
the earlier 5-epoch smoke test's $2.52\text{s/epoch}$ reading for B-spline overestimated
the true 25k-epoch steady-state rate ($1.544\text{s/epoch}$) by $1.63\times$ — both are
recorded in §5 as a calibration lesson for any future Kaggle time budgeting.

---

## 1. What was actually run

| # | Config | Epochs run | Notes |
| :-: | :--- | :---: | :--- |
| 1 | Euler (`ablation_solvers/solver_euler`) | 50,000 | full 5× budget |
| 2 | Tsit5 / RBF (`kanode_flagship`) | 50,000 | full 5× budget, the reference point |
| 3 | B-spline (`ablation_activations/basis_bspline`) | **25,000** | reduced from the planned 50,000 — see below |
| 4 | MLP-SiLU (`mlpode_baseline_silu`) | 50,000 | full 5× budget |
| 5 | MLP paper-spec (`mlpode_baseline`) | 50,000 | full 5× budget |

**Why B-spline stopped at 25k, not 50k:** its smoke test on real Kaggle hardware came
back at $2.52\text{s/epoch}$ — naively extrapolated to 50k epochs, that implied
~35 hours, well past what a single Kaggle session should be pushed to. The budget was
cut to 25,000 epochs (still $2.5\times$ the original 10k, and still inside the
supervisor's requested 25k–50k range — just at the bottom of it instead of the top) as
a scope reduction rather than risk losing an unfinished multi-hour run. In hindsight
(§5) the smoke test itself overestimated the true steady-state rate — the real run
averaged $1.544\text{s/epoch}$ over its full 25,000 epochs, not $2.52\text{s/epoch}$ —
so 50k might well have fit in a single session after all, but the 25k decision was the
right call to make *at the time*, with the information available then.

**Consequence for this doc:** every B-spline comparison below is 10k-vs-**25k**, not
10k-vs-50k like the other four. This is flagged explicitly everywhere it matters (it
does not change the RBF/B-spline tie conclusion, §2.2, but it does mean B-spline's
train-MSE ranking result is on a shorter runway than the other four and a full 50k
B-spline run would be worth doing if the team revisits this).

**Integrity check performed before anything was moved into the repo:** all 5
`metrics.json` files parsed cleanly; `training_history.json` entry counts matched
`metrics.json`'s own `stability.epochs_run` and `config.num_epochs` exactly (25,000 for
B-spline, 50,000 for the other four — no truncated or duplicated runs); all 10
checkpoint files (`best_model.pt` + `final_model.pt` × 5 configs) loaded successfully
via `torch.load`. `git status` before and after the move shows only the new
`implementation/results/phase4/` tree as untracked — nothing under Phase 2's
`results/benchmarks/` was touched.

**Where the results live now:**
```
implementation/results/phase4/epoch_budget_check/
├── euler_50k/           (best_model.pt, final_model.pt, metrics.json, training_history.json, 4 plots)
├── tsit5_rbf_50k/
├── bspline_50k/          (25,000 epochs — see above)
├── mlp_silu_50k/
└── mlp_paperspec_50k/
```

---

## 2. Results, 10k vs. new budget, per config

All numbers below are read directly from each run's own `metrics.json["best"]` (the
`min_train_mse` checkpoint, same selection criterion Phase 2 used) — nothing recomputed
or eyeballed from plots.

| Config | Epochs | Train MSE | Extrap. MSE | Extrap. $R^2$ | Lipschitz $L$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Euler | 10k | $1.377\times10^{-4}$ | $2.507\times10^{-3}$ | $0.99914$ | $8.55$ |
| Euler | **50k** | $\mathbf{8.851\times10^{-6}}$ | $\mathbf{1.929\times10^{-4}}$ | $\mathbf{0.99993}$ | $7.72$ |
| Tsit5/RBF | 10k | $8.833\times10^{-5}$ | $8.923\times10^{-5}$ | $0.99997$ | $6.91$ |
| Tsit5/RBF | **50k** | $\mathbf{6.587\times10^{-6}}$ | $\mathbf{4.025\times10^{-5}}$ | $\mathbf{0.999986}$ | $7.75$ |
| B-spline | 10k | $7.202\times10^{-5}$ | $8.379\times10^{-5}$ | $0.99997$ | $8.44$ |
| B-spline | **25k** | $\mathbf{9.654\times10^{-6}}$ | $\mathbf{4.263\times10^{-5}}$ | $\mathbf{0.999985}$ | $7.05$ |
| MLP-SiLU | 10k | $9.505\times10^{-5}$ | $4.773\times10^{-4}$ | $0.99984$ | $5.07$ |
| MLP-SiLU | **50k** | $\mathbf{1.816\times10^{-6}}$ | $\mathbf{5.905\times10^{-6}}$ | $\mathbf{0.999998}$ | $5.65$ |
| MLP-paperspec | 10k | $1.0827$ | $1.6855$ | $0.4215$ | $38.22$ |
| MLP-paperspec | **50k** | $1.0114$ | $\mathbf{4.3237}$ | $\mathbf{-0.4841}$ | $\mathbf{104.75}$ |

Every config improves its **train** MSE by a large factor with more epochs (all still
descending at 10k, as expected — $7.5\times$–$52.3\times$ improvement, MLP-paperspec's
$1.07\times$ aside). The interesting part is what happens to each one's **rank relative
to the others**, not whether each one individually got better.

---

## 3. Ranking stability — the actual question this check was for

### 3.1 By train MSE (the `min_train_mse` selection criterion)

| Rank @ 10k | Rank @ 50k/25k |
| :--- | :--- |
| 1. B-spline ($7.20\times10^{-5}$) | 1. MLP-SiLU ($1.82\times10^{-6}$) |
| 2. Tsit5/RBF ($8.83\times10^{-5}$) | 2. Tsit5/RBF ($6.59\times10^{-6}$) |
| 3. MLP-SiLU ($9.50\times10^{-5}$) | 3. Euler ($8.85\times10^{-6}$) |
| 4. Euler ($1.38\times10^{-4}$) | 4. B-spline ($9.65\times10^{-6}$) |
| *(MLP-paperspec, off-scale both times)* | |

Completely reshuffled apart from Tsit5/RBF holding the #2 spot both times. MLP-SiLU
jumps from 3rd to 1st; B-spline falls from 1st to last; Euler moves from last (among
the converging four) to 3rd.

### 3.2 By extrapolation MSE (the metric Tables 1–3 bold as the headline number)

| Rank @ 10k | Rank @ 50k/25k |
| :--- | :--- |
| 1. B-spline ($8.38\times10^{-5}$) | 1. MLP-SiLU ($5.90\times10^{-6}$) |
| 2. Tsit5/RBF ($8.92\times10^{-5}$) | 2. Tsit5/RBF ($4.02\times10^{-5}$) |
| 3. MLP-SiLU ($4.77\times10^{-4}$) | 3. B-spline ($4.26\times10^{-5}$) |
| 4. Euler ($2.51\times10^{-3}$) | 4. Euler ($1.93\times10^{-4}$) |

Also not stable, but differently: **Euler stays in last place both times** (its
absolute gap narrows a lot — $28.1\times$ worse than Tsit5 at 10k, $4.8\times$ worse at
50k — but its rank doesn't move). **RBF and B-spline swap 1st/2nd but stay within ~6%
of each other both times** — consistent with them being a genuine statistical tie
rather than a real ordering, at both budgets. **MLP-SiLU is the one whose rank
actually flips** — from 3rd (worse than both KAN variants) to a clear 1st (better than
both).

### 3.3 The headline finding: KAN-vs-MLP generalisation reverses

This is the one result from this whole check that changes a claim actually printed in
[`05`](./05_phase2_benchmark_analysis.md) Table 3:

| | 10k epochs | 50k epochs |
| :--- | :---: | :---: |
| KAN-ODE (RBF) extrap. MSE | $8.923\times10^{-5}$ | $4.025\times10^{-5}$ |
| MLP-ODE (SiLU) extrap. MSE | $4.773\times10^{-4}$ | $5.905\times10^{-6}$ |
| **Ratio, and who's ahead** | KAN **$5.35\times$** better | MLP **$6.82\times$** better |

At 10k epochs, from statistically-identical training loss, KAN-ODE's extrapolation is
$5.35\times$ better than the parameter-matched MLP — this is the exact number
[`05`](./05_phase2_benchmark_analysis.md) reports as reproducing "the paper's central
qualitative claim." At 50k epochs, the SiLU-MLP has not only closed that gap, it has
opened a **$6.82\times$** gap in the *other* direction, and its Lipschitz bound is now
even lower than at 10k ($5.65$ vs. $5.07$ — it stayed the smoother model of the two, it
just kept getting smoother *faster*). **On this problem, under this protocol, the
"KAN-ODE generalises better" finding is a 10k-epoch-budget artifact, not a stable
property.** This is a single-seed result, same caveat as everything else this project
reports at $N=1$ — but it is a real, verified reversal, not noise near a tie (a
$6.8\times$ gap is not close).

---

## 4. The paper-spec MLP: confirmed implementation/architecture story, not epoch-budget

`mlpode_baseline` (`[2,50,2]` + tanh, the paper's literal architecture) was the one
config in this batch built specifically to answer §3's audit question for itself: does
it eventually converge given more time, or is it stuck regardless of budget?

| | 10k epochs | 50k epochs |
| :--- | :---: | :---: |
| Train MSE | $1.0827$ | $1.0114$ ($1.07\times$, essentially flat) |
| Extrap. MSE | $1.6855$ | $4.3237$ (**$2.57\times$ worse**) |
| Extrap. $R^2$ | $0.4215$ | $-0.4841$ (**worse than predicting the mean**) |
| Lipschitz $L$ | $38.22$ | $104.75$ |
| Grad-clip engaged | — | $99.3\%$ of steps |

Given 5× the training budget, this architecture does not converge — it does not even
hold steady, its extrapolation gets measurably *worse*, and its estimated Lipschitz
bound nearly triples to $104.75$ (for comparison, every other config in this batch sits
between $5.6$ and $8.6$), with gradient clipping engaging on almost every single step.
**This is not an epoch-budget problem.** More training does not fix it and arguably
makes the extrapolation behavior worse, consistent with [`05`](./05_phase2_benchmark_analysis.md)'s
existing finding that the *activation function* (tanh vs. SiLU), not training time, is
what determines whether this baseline fits at all. This result is now ready to drop
directly into [`18`](./18_phase4_roadmap.md) §3's audit table as an already-answered
row.

---

## 5. A Kaggle-vs-local cost lesson, recorded for future planning

Real per-epoch throughput, measured over the *entire* run (not a short smoke test),
compared to the local-machine numbers [`18`](./18_phase4_roadmap.md) §2.3 was built on:

| Config | Local s/epoch | Real Kaggle s/epoch | Ratio |
| :--- | :---: | :---: | :---: |
| Euler | $0.0254$ | $0.0627$ | $2.47\times$ |
| Tsit5/RBF | $0.1705$ | $0.4033$ | $2.37\times$ |
| B-spline | $0.6680$ | $1.5438$ | $2.31\times$ |
| MLP-SiLU | $0.2856$ | $0.2722$ | $0.95\times$ (slightly *faster* on Kaggle) |
| MLP-paperspec | $0.0736$ | $0.1595$ | $2.17\times$ |

Four of the five configs ran **~2.3–2.5× slower per epoch on Kaggle** than on the local
dev machine used for Phase 2's original calibration — this is not a B-spline-specific
issue, it's a general Kaggle-CPU-vs-local-CPU gap, consistent with the cross-platform
reproducibility/performance gap Track D independently documented in
[`16`](./16_p3_stiffness_map_findings.md) §7. MLP-SiLU is the one outlier (Kaggle
slightly *faster*) — no clear mechanism identified for that, noted here rather than
guessed at.

**The smoke-test-overestimate lesson:** B-spline's 5-epoch smoke test read
$2.52\text{s/epoch}$; the real 25,000-epoch run averaged $1.544\text{s/epoch}$ — the
smoke test overestimated the true rate by $1.63\times$, almost certainly because a
5-epoch sample is dominated by one-time startup cost (cold BLAS/import overhead) rather
than steady state. **For any future Kaggle time budgeting on this project, calibrate
from at least ~200–500 epochs, not 5** — a cheap enough sample to still be fast, but
long enough to amortize startup cost out of the estimate.

Actual total compute: $23.19$h serial-equivalent CPU time (vs. [`18`](./18_phase4_roadmap.md)
§2.3's $17.0$h estimate — about $1.36\times$ over plan, matching the ~2.3× per-epoch
slowdown partially offset by B-spline's 25k-not-50k cut). Actual wall-clock, bounded by
the slowest job (B-spline, $10.72$h), landed close to the $9.3$h planned wall-clock —
the 25k cut absorbed almost exactly the per-epoch slowdown that would otherwise have
blown that estimate out.

---

## 6. What this means for the writeup

Per [`18`](./18_phase4_roadmap.md) §2.5, this is the **"ranking does not hold"**
branch, not the reassuring one — and the team should report it as such rather than
only the parts that confirm Phase 2. Concretely, for [`05`](./05_phase2_benchmark_analysis.md)
and the final report:

1. **Table 3's "KAN-ODE generalises better" claim needs a qualifier, not a retraction.**
   Suggested wording: *"This result holds at the project's fixed 10,000-epoch training
   budget. A follow-up check (`docs/20`, single seed) training both models to 50,000
   epochs found this reverses — the parameter-matched MLP baseline ultimately
   generalises $6.8\times$ better than KAN-ODE on this problem. The 10k-epoch KAN-ODE
   advantage should be read as a training-speed/budget effect, not a stable structural
   property of the two architectures on Lotka-Volterra."* This is a more interesting and
   more honest finding than a simple "10k was fine" would have been — it directly
   contradicts the base paper's central claim in a specific, well-characterized way,
   which is exactly the kind of result the supervisor's original question was trying to
   surface.
2. **Table 1's Euler entry can keep its 10k number** with an added note that the gap to
   Tsit5 narrows substantially (to $4.8\times$) by 50k — Euler is still the weakest
   solver, just less dramatically so than a 10k-only comparison suggests.
2. **Table 2's RBF/B-spline tie stands as reported** — no change needed there, though a
   footnote that B-spline was only extended to 25k (not 50k, for cost reasons) would be
   honest disclosure if this doc's numbers get cited directly.
3. **Whether the full 24-config ablation should move to a higher epoch budget** is a
   bigger decision than this check can make alone, per [`18`](./18_phase4_roadmap.md)
   §2.5 — this check's job was to determine *whether* the rankings are budget-sensitive
   (they are, for the KAN-vs-MLP comparison specifically), not to re-run the whole
   ablation. Flag to the team/supervisor as a decision point: either (a) keep all 24
   configs at 10k with the qualifier above added to Table 3 specifically, or (b) commit
   to re-running Table 3's two columns (and possibly the rest of Table 1/2) at a higher
   budget before the final report — a full-ablation-at-25k cost estimate would need to
   be built fresh from this run's real Kaggle rates (§5), not the local-machine numbers
   [`18`](./18_phase4_roadmap.md) originally used.
4. **§3's audit table** can now cite §4 above as a direct answer for the paper-spec MLP
   row: confirmed implementation/architecture mismatch, not an epoch-budget effect.

---

## 7. Definition of done — this doc's scope

| Check | Status |
| :--- | :---: |
| All 5 configs' results downloaded, integrity-verified, placed under `results/phase4/epoch_budget_check/` | ✅ |
| Phase 2's original `results/benchmarks/` left untouched | ✅ (`git status` confirms) |
| 10k-vs-new-budget comparison reported for all 5 configs | ✅ (§2–§4) |
| Ranking-stability verdict stated explicitly, both directions | ✅ (§3 — unstable, reported honestly per [`18`](./18_phase4_roadmap.md) §2.5) |
| Real-vs-planned cost discrepancy explained, not just flagged | ✅ (§5) |
| Exact caveat sentence(s) proposed for `docs/05`/final report | ✅ (§6) |
| §3's baseline-vs-paper audit table | ⬜ still open — a reading/writing task, no compute; §4 above is a ready-made input for its MLP row |

**This closes out Phase 4's epoch-budget compute-running task.** The remaining open
item from [`18`](./18_phase4_roadmap.md) is §3's field-by-field paper-vs-implementation
comparison (reading the original paper/its repo and writing up the comparison table) —
unstarted, no dependency on anything in this doc except §4's MLP finding.

---

*Previous: [`19_kaggle_execution_guide.md`](./19_kaggle_execution_guide.md) — how these
5 runs were executed. [`18_phase4_roadmap.md`](./18_phase4_roadmap.md) — why these 5
configs and what the outcomes mean.*
