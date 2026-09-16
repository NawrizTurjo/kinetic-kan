# 🌟 Phase 4 Roadmap: Statistical Rigor & Final Synthesis

> **Status:** ready to start — Phase 3 fully merged (`main`), 249/250 tests passing
> (1 correctly CUDA-gated skip), all 5 tracks independently verified.
> **Prerequisite:** ✅ Phase 3 closed (all 5 `feat/p3-*` branches + the
> `fix/missing-baseline-checkpoints` branch merged into `main`).
> **Chain:** [`11`](./11_phase2_closeout.md) Phase 2 closeout → [`12`](./12_phase3_roadmap.md)
> Phase 3 roadmap → **this doc**
> **Supersedes:** the original plan's Task 4.1 ("multi-seed replication, $N=5$") is
> **out of scope as of this revision**, replaced by the supervisor-requested epoch-budget
> sensitivity check below (§2) — a more fundamental question than seed variance: is the
> project's fixed 10,000-epoch ablation budget even long enough to trust the rankings it
> produced, before worrying about how much those rankings wobble seed-to-seed.
> **Execution platform:** Kaggle notebooks (CPU) — same infrastructure as before, just
> pointed at a different, smaller job list (§2.4).

---

# Part 1 — What Phase 4 actually is now

Two validation tasks, plus the synthesis work that was always part of Phase 4:

1. **Epoch-budget sensitivity check (§2).** Per the supervisor's request: take a
   handful of representative configurations from the 24-config Phase 2 Lotka-Volterra
   ablation, train them 2.5–5$\times$ longer (**50,000 epochs**, Lotka-Volterra only —
   no other dataset), and check whether the *relative ranking* among configurations
   holds. If it does, the project keeps reporting all 24 configs at 10,000 epochs, with
   an explicit note that the ablation runs under a fixed budget and absolute numbers may
   not match the original paper.
2. **Baseline-vs-paper implementation audit (§3).** Also per the supervisor: confirm
   that wherever this project's numbers differ from the original paper's, the cause is
   the fixed epoch budget — not an undocumented difference in architecture, optimizer,
   or training settings.
3. **Synthesis** — 300 DPI publication figures, the LaTeX report/paper draft, a final
   codebase audit, and presentation polish. Owners for this half were already fixed in
   [`12`](./12_phase3_roadmap.md) §5.4 and the original plan; §4 below restates them.

**What is explicitly out of scope now:** multi-seed replication (the original Task
4.1), and re-opening any Phase 3 track's own findings/conclusions. This doc's two
validation tasks are about the **Phase 2 ablation's epoch budget**, not about Phase 3's
novel-contribution tracks.

---

# Part 2 — Epoch-budget sensitivity check

## 2.1 The question, precisely

Phase 2's full ablation — 24 configurations, all Lotka-Volterra, all trained for a
fixed 10,000 epochs — produced Tables 1–3 (solver order, basis function, model type)
plus the noise and step-size sweeps. The supervisor's concern: **10,000 epochs was
never validated as "enough"** — it's possible some configurations are still improving
at epoch 10,000 and would overtake others given more budget, which would mean the
tables' *rankings*, not just their absolute numbers, are an artifact of stopping early.

**The check:** pick a few representative configs, train them to 50,000 epochs (5$\times$
the original budget — within the supervisor's suggested 25k–50k range, at the top end
so one run settles the question instead of needing a second round at a higher budget
later), and compare each one's **rank relative to the others** at epoch 10,000 vs.
epoch 50,000. Stable ranking → 10k stands, with the fixed-budget caveat stated
explicitly wherever the ablation is reported. Unstable ranking → the team needs to
decide whether the full 24-config ablation needs to move to a higher budget (§2.5
covers what that would cost, without committing to it now).

## 2.2 Picking "a few representative" configs — which ones, and why

Five configs, chosen to cover the specific rankings a supervisor would actually
question, not an arbitrary sample:

| # | Config | Table it's from | Why this one |
| :-: | :--- | :--- | :--- |
| 1 | **Euler** (solver ablation) | Table 1 | The one solver that *doesn't* reach $10^{-4}$ within 10k epochs (`results/benchmarks/ablation_solvers/solver_euler`) — the single config most likely to change the picture if it just needed more time. |
| 2 | **Tsit5 / RBF** (`kanode_flagship`) | Tables 1 & 2 | The project's reference point — every other config in both tables is compared against this one. If *this* moves, every ranking built on it is suspect. |
| 3 | **B-spline** (`ablation_activations/basis_bspline`, same solver as #2) | Table 2 | Phase 2 called RBF and B-spline "statistically tied" at 10k — directly checks whether that tie holds or one pulls ahead with more budget. |
| 4 | **MLP-ODE, parameter-matched** (`mlpode_baseline_silu`) | Table 3 | The converged, fairly-sized MLP baseline KAN-ODE's convergence-speed claim is measured against. |
| 5 | **MLP-ODE, paper-spec** (`mlpode_baseline`, $[2,50,2]$+tanh) | Table 3 | **Currently fails to train at all** (best train MSE $1.08$ — worse than predicting the mean). This is the literal architecture the base paper describes. 50k epochs answers §3's question directly for this one config: does it eventually converge (an epoch-budget story) or stay stuck (an implementation/architecture story)? |

**Deliberately not included:** the other 5 basis functions (Chebyshev, Lagrange,
Newton, RSWAF, IQF — all clearly behind RBF/B-spline at 10k already, less central to
"did the *leaders'* ranking change") and the noise/step-size sweeps (those characterize
one recipe's robustness to a varying condition, not a ranking *among* configs, so the
sensitivity question doesn't apply the same way). If the team wants broader coverage,
`basis_chebyshev` (next-best basis, cheap — §2.3) is the natural 6th addition.

**A real limitation, stated up front:** checking only the best (Tsit5) and worst
(Euler) solvers is a proxy for "did Table 1's ranking change," not proof that the
*middle* order (Heun/Midpoint/RK4/Dopri5) is equally stable — a genuinely exhaustive
check would re-run all 6. Five configs is what "a few representative architectures"
supports on a reasonable budget; if the two extremes hold stable, that's suggestive
but not conclusive for the middle of the table, and the writeup should say so rather
than implying full re-verification.

## 2.3 Real cost, per config (measured, not estimated)

Extrapolated directly from each config's own already-recorded `seconds_per_epoch`
(`results/benchmarks/.../metrics.json`) — not a guess:

| Config | Measured s/epoch | 10k epochs (already done) | **50k epochs (new)** |
| :--- | :---: | :---: | :---: |
| Euler | $0.0254$ | $4.2$ min | $\approx 21$ min |
| Tsit5 / RBF (`kanode_flagship`) | $0.1705$ | $28.4$ min | $\approx 2.37$ h |
| B-spline | $0.668$ | $1.86$ h | $\approx 9.28$ h |
| MLP, parameter-matched (`_silu`) | $0.2856$ | $47.6$ min | $\approx 3.97$ h |
| MLP, paper-spec (fails to train) | $0.0736$ | $12.3$ min | $\approx 1.02$ h |
| **Total (serial, one core)** | | | **$\approx 17.0$ h** |

## 2.4 Execution plan

Same Kaggle infrastructure as the rest of this project — one notebook per config (each
is a single training run with nothing to internally parallelize, so there's no benefit
to bundling multiple configs into one notebook the way a multi-cell sweep would).
**5 notebooks, run simultaneously.** Reuse `train_kan_ode()`
(`implementation/train.py`) directly — the 5 ready-to-paste scripts already exist under
`implementation/experiments/epoch_budget_check/kaggle_50k_*.py`, one per config, with
the full click-by-click guide in
[`19_kaggle_execution_guide.md`](./19_kaggle_execution_guide.md). Unlike a multi-job
sweep, each of these is a **single** training run per notebook, so there's no
`ProcessPoolExecutor`/thread-pinning needed — the one job gets the notebook's full 4
cores via PyTorch's own multi-threaded BLAS:

```python
from train import train_kan_ode

# Example: config #5 above (paper-spec MLP, the one currently failing to train)
metrics = train_kan_ode(
    model_type="mlp", dataset="lotka_volterra",
    mlp_layers=[2, 50, 2], mlp_act="tanh",
    lr=2e-3, num_epochs=50000, seed=42,
    save_dir="results/phase4/epoch_budget_check/mlpode_baseline_50k",
    device="cpu",
)
```

**Wall-clock: $\approx 9.3$ h**, bounded by the B-spline notebook (the slowest), since
all 5 run at once — comfortably inside a single Kaggle session regardless of whether
the account's real concurrent-notebook limit turns out to be 5 or 10 (only 5 notebooks
are needed here either way).

Output goes to `results/phase4/epoch_budget_check/<config>_50k/` — kept separate from
Phase 2's original `results/benchmarks/` (read-only, unmodified) for the same reason
every other Phase 4 output uses its own `results/phase4/` tree: nothing here should
touch or risk corrupting an already-reviewed Phase 2 result.

## 2.5 What the two possible outcomes mean for the writeup

**If ranking holds** (each config's 50k-epoch position matches its 10k-epoch
position relative to the others): report all 24 configs at 10,000 epochs, as already
done, and add one explicit sentence to the ablation write-up (`docs/05` and wherever
the final report cites Table 1/2/3):
> *"This ablation uses a fixed 10,000-epoch training budget for all configurations,
> validated by an epoch-sensitivity check (5 representative configs to 50,000 epochs,
> `docs/18` §2) showing stable relative rankings; absolute error values may not
> exactly match the original paper's own budget."*

**If ranking does not hold** for one or more of the 5 configs: that specific config's
Table 1/2/3 placement is unreliable at 10k and needs its own longer run before being
reported at all. Whether that means bumping the *entire* 24-config ablation to a
higher budget is a bigger decision than this check can make alone — full-ablation cost
at, say, 25k epochs would be roughly $2.5\times$ Phase 2's own already-spent compute
(a rough multiple of whatever that total was, not computed here since it depends on
which budget the team would actually pick) — flag it to the team and decide then,
rather than pre-committing to a number now.

**Either way, this is a factual, verifiable outcome** — resist the temptation to only
report the reassuring half. If Euler is still climbing at 50k while Tsit5 has plateaued,
that is itself a finding (order-dependent convergence rate, not just order-dependent
convergence *quality*) worth a sentence in the report, not something to omit because it
complicates the "10k is fine" story.

---

# Part 3 — Baseline-vs-paper implementation audit

## 3.1 The question

The supervisor's second ask: confirm that wherever this project's numbers diverge from
the original paper's, it's because of the fixed epoch budget (§2) — not because
something in the implementation quietly differs from what the paper actually
specifies (a different optimizer setting, a different normalization, a different grid
resolution, etc.).

## 3.2 This project's actual recipe (the known half of the comparison)

Already fully determined from `kanode_flagship`'s own recorded config — nothing to
re-derive:

```
KAN([2, 10, 2]), grid_len=5, basis=rbf, normalizer=tanh, base_act=silu,
solver=tsit5, substeps=2, lr=0.002, epochs=10000,
t_start=0.0, t_train_end=3.5, t_end=14.0, dt=0.1
```

The MLP side has **two** recorded variants, not one — itself part of the answer to
this section's question:

| | Architecture | Activation | Result at 10k |
| :--- | :--- | :--- | :---: |
| Paper-spec (`mlpode_baseline`) | $[2, 50, 2]$ | tanh | fails to train (train MSE $1.08$) |
| Parameter-matched fix (`mlpode_baseline_silu`) | $[2, 14, 8, 8, 2]$ | SiLU | converges (train MSE $9.5\times10^{-5}$) |

That the paper's literal architecture doesn't train at all under this project's
setup, while a re-sized SiLU version does, is *already* evidence pointing at an
architecture/activation mismatch for the MLP baseline specifically — not purely an
epoch-budget question. §2.2's config #5 (50k epochs on the paper-spec MLP) is what
turns "already evidence" into a confirmed answer.

## 3.3 What's still needed — the actual audit task

**Nobody has yet sat down and compared this project's recipe field-by-field against
the paper's own stated methodology.** This is a reading/research task, not a training
run:

1. Pull the paper's own training section (arXiv:2407.04192 / CMAME Vol. 432,
   Article 117397) and, if accessible, the official codebase
   (`github.com/DENG-MIT/KAN-ODEs`) for its actual default config on the
   Lotka-Volterra experiment specifically.
2. Compare, field by field, against §3.2's table above: epoch count, learning rate,
   optimizer, KAN grid size, basis function, normalization scheme, base activation,
   ODE solver and step size, and (for the MLP baseline) architecture and activation.
3. For every field that differs, note it explicitly — don't silently absorb it into
   "the paper used more epochs." Some differences (e.g. epoch count) are expected and
   already the subject of §2; others (e.g. a different learning rate or grid
   resolution) would be a genuine implementation delta worth its own sentence in the
   report, separate from the epoch-budget story.
4. Write the findings into a short section of the final report (or a small addendum
   to `docs/05_phase2_benchmark_analysis.md`) — a table of "paper setting vs. this
   project's setting" per field, with a one-line verdict per row (matches / explained
   difference / unexplained difference needing further work).

**No compute needed for this task** — it's reading and documentation, doable in
parallel with §2's Kaggle runs.

---

# Part 4 — Time budget & ownership

| Task | Owner | Compute | Wall-clock |
| :--- | :--- | :---: | :---: |
| §2 Epoch-budget check (5 configs, Kaggle, parallel) | Monjur & Shams (adapted from the original plan's Task 4.1 co-lead role) | $\approx 17.0$ h CPU-time | $\approx 9.3$ h wall-clock, 1 Kaggle wave |
| §3 Baseline-vs-paper audit | Monjur & Shams (same role — no new compute, can run alongside §2) | — | reading/writing, not compute-bound |
| Publication figures & report (Task 4.2/4.3, [`12`](./12_phase3_roadmap.md) §5.4) | Abrar | — | — |
| Codebase audit & presentation polish (Task 4.4) | Nawriz & Abhishek | — | — |

Monjur and Shams keep the same *role* the original plan gave them for Task 4.1
(statistical/methodological rigor) — just pointed at this revised, supervisor-directed
task instead of seed replication. Abrar's and Nawriz/Abhishek's roles are unchanged
from [`12`](./12_phase3_roadmap.md) §5.4 and the original plan's Task 4.2–4.4.

---

# Part 5 — Definition of done

| Check | Target |
| :--- | :--- |
| All 5 representative configs trained to 50,000 epochs | results under `results/phase4/epoch_budget_check/`, Phase 2's original `results/benchmarks/` untouched |
| Ranking comparison (10k vs. 50k) reported explicitly | for each of the 5 configs, not just a pass/fail summary |
| Fixed-budget caveat added to the ablation write-up | if ranking holds — the exact sentence in §2.5, or equivalent, actually landing in `docs/05`/the final report |
| Paper-vs-implementation comparison table written | §3.3's field-by-field table, with a verdict per row |
| Publication figures at 300+ DPI | for every table/finding going into the final report |
| Final report / paper draft compiled | LaTeX, all tables and figures integrated |
| Final codebase audit complete | no stray artifacts, all 5 Phase 3 tracks' folder-isolation still intact |

---

*Previous: [`12_phase3_roadmap.md`](./12_phase3_roadmap.md) — Phase 3 is closed, all
5 tracks merged into `main`, 249/250 tests passing.*
