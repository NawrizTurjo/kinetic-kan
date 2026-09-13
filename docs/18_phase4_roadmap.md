# 🌟 Phase 4 Roadmap: Statistical Rigor & Final Synthesis

> **Status:** ready to start — Phase 3 fully merged (`main`), 249/250 tests passing
> (1 correctly CUDA-gated skip), all 5 tracks independently verified.
> **Prerequisite:** ✅ Phase 3 closed (all 5 `feat/p3-*` branches + the
> `fix/missing-baseline-checkpoints` branch merged into `main`).
> **Chain:** [`11`](./11_phase2_closeout.md) Phase 2 closeout → [`12`](./12_phase3_roadmap.md)
> Phase 3 roadmap → **this doc**
> **Supersedes:** `04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md` Task 4.1's literal
> $N=5$/seed-list ("`[42, 1337, 2024, 7, 999]`", "run all core benchmark scripts") — kept
> for the record, but **not followed literally**; §2 below explains why, with the actual
> measured compute cost of doing so.
> **Execution platform:** Kaggle notebooks (CPU), not local machines — §2.5 has the
> concrete parallel plan (4 cores/notebook × up to 5 notebooks at once).

---

# Part 1 — What Phase 4 actually is

Two things, in order:

1. **Multi-seed replication** of the findings that matter most for the final report —
   not everything Phase 1–3 produced, and not literally $N=5$ everywhere. §2 works out
   why from real, measured per-epoch costs already sitting in this repo, not guesses.
2. **Synthesis** — 300 DPI publication figures, the LaTeX report/paper draft, a final
   codebase audit, and presentation polish. Owners for this half were already fixed in
   [`12`](./12_phase3_roadmap.md) §5.4 and the original plan; §4 below just restates and
   slightly adapts them to the track structure Phase 3 actually used.

**What is explicitly not in scope:** re-opening any Phase 3 track's own findings or
conclusions. Phase 4 adds error bars to numbers that already exist; it does not
re-litigate what those numbers mean. If a finding's *sign* flips under more seeds, that
is reported as a genuine new result (Track B's own §3.3 warm-up-cut sign-flip is exactly
this kind of honest reporting, done well) — but the *investigation* stays inside the
owning track's doc, not this one.

---

# Part 2 — The seed-count question, answered from measured data

## 2.1 Why $N=5$ "run everything again" does not fit in the time available

The original plan's Task 4.1 says "run all core benchmark scripts over $N=5$ random
seeds." Taken completely literally — every Phase 2 table *and* every Phase 3 track, all
five re-run four more times each — the actual measured per-epoch costs already recorded
in this repo's own `metrics.json` files put that at:

| Component | One extra seed pass costs | For 4 more seeds (→ $N=5$) |
| :--- | :---: | :---: |
| Phase 2 solver ablation (6 solvers × 10k ep, LV) | $1.58$ h | $6.3$ h |
| Phase 2 noise sweep (4 σ × 10k ep, LV) | $1.83$ h | $7.3$ h |
| Track C — LV hybrid (10k ep) | $2.4$ h | $9.6$ h |
| Track C — pendulum hybrid (10k ep) | $7.1$ h | $28.4$ h |
| Track D — 24-cell sweep (4 solver × 6 μ, 2k ep probe) | $12.7$ h | $50.6$ h |
| Track E — all 12 epidemic arms | $7.6$ h | $30.6$ h |
| Track A | not seed-sensitive (§2.3) | — |
| **Total** | $\approx 33$ h / seed | **$\approx 133$ h** |

$133$ hours of serial CPU compute, on top of the $\approx 30$ h already spent getting
Phase 3 to $N{=}1$ (Track C's pendulum alone took the better part of a day). Spread
across 5 people that is still $\approx 27$ h of *background* compute each, achievable in
principle — but the real bottleneck is **Track D's $50.6$ h**: at $\sim 32$ minutes per
cell for Tsit5 alone (the adaptive solver dominates the sweep's cost — see §2.4), a
naive 4-more-seeds pass there is a multi-day undertaking by itself, run by one person.

**This is exactly the scenario [`12`](./12_phase3_roadmap.md) §5.4 pre-warned about:**
*"no Phase 3 track needs to pre-emptively add seed loops — that discipline is Phase 4's
job, applied **selectively** to whichever Phase 3 findings turn out to matter."*
Selectively is the operative word — not "all five, four more times each."

## 2.1b Correction: Track D is already at $N=3$

Track D ran seeds $\{42, 1, 7\}$ as part of its *own* investigation
([`16`](./16_p3_stiffness_map_findings.md) §4), before Phase 4 existed. **Under the
$N=3$ target below, Track D needs zero additional runs.** Every other track (A, B, C, E)
is still at $N=1$ (seed 42 only) and needs 2 more seeds to reach $N=3$. This matters a
lot for the Kaggle plan in Part 2.5 — Track D only re-enters the picture if the team
goes for the $N=5$ stretch goal (§2.5).

## 2.2 The actual recommendation: $N=3$ total (2 more seeds), scoped per-track

**Two more seeds — not four — for everything, and a trimmed scope for the two most
expensive tracks.** This is not a compromise invented for this document: **Track D
already did exactly this** (seeds $42, 1, 7$, $N=3$) as part of its own investigation,
and that choice held up — it was enough to find and confirm the seed-sensitivity
mechanism (§4 of [`16`](./16_p3_stiffness_map_findings.md)) with a real, load-bearing
result. $N=3$ is the project's own proven-sufficient standard, not a downgrade.

| | $N=5$, everything, serial | **$N=3$, scoped (recommended)** |
| :--- | :---: | :---: |
| Total added compute (CPU-time) | $\approx 133$ h | $\approx 18.4$ h (§3's table — Track D contributes $0$, already at $N{=}3$) |
| Per-person, one core each, no Kaggle parallelism | $\approx 27$ h | $\approx 0.2$–$8.3$ h |
| **On Kaggle, parallel (§2.5)** | not modeled here — see §2.5's $N=5$ analysis | **$\approx 4.5$ h wall-clock, 2 notebook-waves** |
| Achievable in | the rest of the term (serial) / a couple of Kaggle sessions (parallel, §2.5) | a single day, even without Kaggle |

*(This table predates §2.5's Kaggle plan and originally assumed one core per job with
no notebook-level parallelism — kept for the reasoning trail, but §2.5's wall-clock
numbers are what to actually plan around.)*

**If $N=5$ is still wanted for the report's headline numbers specifically** (not
everything), the cheapest path is: do the $N=3$ pass first, look at which
findings show real seed-sensitivity (probably Track D's transitional $\mu$ values and
Track C's pendulum result — both already flagged as seed-suspect), and extend *only
those* to 2 further seeds. That is a much smaller, targeted addition once the $N=3$ data
exists — decide after seeing it, not before.

## 2.3 Why Track A is exempt from seed replication

Track A measures **peak VRAM and wall-clock** — properties of the hardware and the
$O(N_t)$-vs-$O(1)$ algorithmic structure, not of the random weight initialization. A
different seed changes the specific trained weights, not the *shape* of the memory
curve or the *ratio* of two integration methods' wall-clock cost. Multi-seed replication
answers "is this pattern real or a lucky draw" — a question that doesn't apply to a
claim like "adjoint uses $O(1)$ memory," which is true by construction of the method,
already demonstrated across 7 trajectory lengths (§Memory vs. trajectory length,
[`13`](./13_p3_adjoint_profiling_findings.md)), not by chance.
**One optional, cheap check** is still worth doing (§3, Track A row): confirm the
gradient-agreement number ($\sim 10^{-7}$ relative error) stays tiny at 1–2 more seeds —
a 10-minute sanity check, not a re-run of the profiling sweep.

## 2.4 Why Track D's cost is so lopsided

Tsit5 alone is $6.3$ h of the $12.7$ h one-seed total (Dopri5-family adaptive solvers
evaluate the field far more times per step than Euler does) — the four solvers are
**not** interchangeable in cost. Two further optimizations, on top of dropping to
$N=3$:

- **Skip $\mu \in \{0.1, 5.0, 8.0\}$ for the extra seeds.** [`16`](./16_p3_stiffness_map_findings.md)
  §4 already established these three are seed-*robust* extremes (converged-fraction
  identical across all tested seeds) — re-running them again mostly re-confirms
  something already settled. Keep the seed budget on $\mu \in \{0.5, 1.0, 2.0\}$, the
  three transitional values that actually showed seed-sensitivity.
- This halves the per-seed cost to $\approx 6.3$ h (half the $\mu$ values), which is
  where §3's Track D estimate comes from.

---

## 2.5 Running this on Kaggle, in parallel — the actual execution plan

> **For the full click-by-click walkthrough (account setup, uploading the repo,
> creating notebooks, running, downloading results, merging them back), see
> [`19_kaggle_execution_guide.md`](./19_kaggle_execution_guide.md).** This section
> covers the reasoning and the numbers; that doc covers the actual steps.

**Why Kaggle at all:** local machines have already been running near-continuously for
Phase 3 (Track C's pendulum alone took most of a day); Phase 4 moves everything to
Kaggle notebooks so nobody's own PC has to run for hours unattended.

**The platform constraint driving this whole section:** each Kaggle CPU notebook gets
**4 cores**, and the team can have **up to 5 notebooks committed and running
simultaneously**. That's **20-way parallelism**, not 4 — the trick is treating "5
notebooks" as 5 independent 4-core sandboxes, not 5 processes fighting over one shared
box.

**Track D already has the template — reuse it, don't reinvent it.**
`kaggle_full_retrain.py` is the pattern every other track's Kaggle script should copy:

```python
# The reusable shape (see kaggle_full_retrain.py for the real version):
os.environ["OMP_NUM_THREADS"] = "1"       # BEFORE importing torch
os.environ["MKL_NUM_THREADS"] = "1"       # -- 4 workers x 1 thread each uses all
os.environ["OPENBLAS_NUM_THREADS"] = "1"  #    4 cores with zero oversubscription

with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as ex:
    futures = {ex.submit(run_one_job, *job): job for job in job_list}
    # ... collect results as they complete, write metrics.json per job ...

shutil.make_archive("/kaggle/working/results", "zip", OUT_ROOT)  # download from Output tab
```

**Only Track D has this today.** Tracks B, C, and E need a small new
`kaggle_<slug>_seeds.py` written before any of this plan is actually runnable on Kaggle —
each one is a shallow wrapper: import the track's own training entry point directly
(`train_kan_ode` for Track B/Phase-2 dependency, `run_hybrid.run` for Track C,
`run_epidemic_fit`'s training function for Track E) in place of `run_sweep.train_cell`,
and reuse the exact `ProcessPoolExecutor` + thread-pinning + skip-existing + zip-on-exit
shape above. Track A doesn't need one — its Phase 4 task is a 2-seed sanity check small
enough to just run inline.

### N=3 plan — concrete Kaggle job list

| Track | Notebooks needed | Job list per notebook | Wall-clock (parallel) |
| :--- | :---: | :--- | :---: |
| A | 1 | Gradient-agreement check, 2 seeds, inline (no sweep needed) | $\approx 20$ min |
| B | 2 (one per new seed) | 6 solvers, 4-way parallel inside the notebook | $\approx 0.4$ h each, simultaneous |
| C | 3 | LV seed A; LV seed B; pendulum 1 seed to $\geq5{,}000$ ep — one job per notebook, no internal parallelism needed (each is a single training run) | $2.4$h / $2.4$h / $3.5$h — the pendulum one is the long pole |
| D | 0 | **already done** (§2.1b) | — |
| E | 2 (one per new seed) | noise sweep (4σ) + 3 headline epidemic arms = 7 jobs, 4-way parallel inside the notebook | $\approx 1$ h each, simultaneous |

**Total notebook-slots needed: 8, but only 5 run at once** — so this goes in **2 waves**:

- **Wave 1** (5 notebooks): Track C's 3 + Track B's 2 → bounded by Track C's pendulum
  notebook, $\approx 3.5$ h.
- **Wave 2** (3 notebooks): Track E's 2 + Track A's 1 → bounded by Track E,
  $\approx 1$ h.
- **Total wall-clock: $\approx 4.5$ h**, comfortably inside one Kaggle session's length
  limit either wave, and easily finished in a single day — a large improvement over the
  $\approx 18.4$ CPU-hour figure in §2.2/Part 3, which is core-time, not wall-clock, and
  assumed no Kaggle-style notebook-level parallelism.

*(Session-length and weekly-quota limits are a Kaggle platform policy that changes over
time — worth the team double-checking current limits before relying on the $4.5$h
number above; every individual notebook here is well under any commonly-cited CPU
session cap regardless.)*

### N=5 stretch goal — is it possible?

**Yes, and Track D — the one part of this already run — is now the cheap part, not the
bottleneck**, precisely because Kaggle's notebook-level parallelism turns its
"$12.55$h of compute" (§2.4) into a couple of hours of wall-clock:

| Track | Extra seeds needed | Notebooks | Wall-clock (parallel) |
| :--- | :---: | :---: | :---: |
| A | 2 more (4 total) | 1 | $\approx 40$ min |
| B | 2 more (4 total) | 2 more (4 total) | $\approx 0.4$ h, simultaneous with the N=3 pair |
| C | 2 more (4 total, LV); pendulum — team call, see below | 2 more LV notebooks; 1–3 more pendulum notebooks | $2.4$h (LV); $3.5$–$7.1$h each (pendulum) |
| D | 2 more (reaches $N=5$) | 2 (one per seed, 4-way parallel each, scoped to $\mu\in\{0.5,1.0,2.0\}$ — §2.4) | $\approx 1.6$ h each, simultaneous |
| E | 2 more (4 total) | 2 more (4 total) | $\approx 1$ h, simultaneous with the N=3 pair |

**Track C's pendulum is the one genuine judgment call left.** The N=3 pass answers
"does the overfit-onset recur at a second seed, yes/no" with 1 extra seed
($3.5$h). Going to $N=5$ pendulum means 3 *more* seeds beyond that — worth deciding
**after** the N=3 answer is in, not before: if seed 2 confirms the pattern, 1–2 more
seeds at the cheaper $5{,}000$-epoch check is probably enough evidence; a full
second $10$k confirmation run ($7.1$h) is only worth it for whichever seed's number
actually goes in the final report table.

**Bottom line: with everything else running, total added wall-clock for the full N=5
stretch is on the order of a second $\approx 4$–$5$ hour Kaggle session** (batched
similarly to N=3, 2 waves of ≤5 notebooks) **on top of** the N=3 plan above — very
achievable across a couple of days, not a blocker. The team does not need to choose
between N=3 and N=5 up front: run N=3 first, look at the results, then decide which
specific findings (if any) justify the 2 extra seeds for N=5, per §2.2's original point.

---

# Part 3 — Time budget, per track (the $N=3$, scoped plan)

| Track | Owner | What gets re-run | Extra seeds | Total added compute (CPU-time) | Kaggle wall-clock (§2.5) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| A | Nawriz | Gradient-agreement check only (no VRAM/wall-clock re-sweep — §2.3) | 2 | $\approx 0.2$ h | $\approx 20$ min |
| B | Abhishek | Phase 2's 6-solver LV ablation (also strengthens Table 1 directly) | 2 | $\approx 3.2$ h | $\approx 0.4$ h |
| C | Shams | LV hybrid full 10k (both extra seeds); pendulum hybrid to $\geq 5{,}000$ epochs, **1** extra seed only, far enough past the epoch-3,000 overfit onset to see if it recurs | 2 (LV) / 1 (pendulum) | $\approx 8.3$ h | $\approx 3.5$ h |
| D | Abrar | **Already at $N=3$ (§2.1b) — nothing to run** | 0 | $0$ h | $0$ h |
| E | Monjur | Phase 2 noise sweep (4σ) + 3 headline epidemic arms (`full_plain`, `full_vanish`, `ts24`) only, not all 12 | 2 | $\approx 6.7$ h | $\approx 1$ h |
| | | | | **$\approx 18.4$ h total compute** | **$\approx 4.5$ h wall-clock, 2 Kaggle waves (§2.5)** |

**Two numbers per row, and they mean different things.** "Total added compute" is CPU
core-time — what it would cost on one core, one job at a time. "Kaggle wall-clock" is
what the team actually experiences, once each track's job list is split across the
4-core-per-notebook / 5-notebooks-at-once model in §2.5. The CPU-time column is what
determines *whether a plan is efficient*; the wall-clock column is what determines
*whether it fits in a weekend*. Both matter, but only the wall-clock column is what to
plan a schedule around.

Every row reuses the exact script each track already has, or a small new
`kaggle_<slug>_seeds.py` modeled on Track D's existing `kaggle_full_retrain.py` (§2.5) —
**no changes to `run_hybrid.py`, `run_sweep.py`, or any other track's core logic are
needed**, only new `--seed` values, a smaller job list where noted, and (for B/C/E) a
thin Kaggle wrapper script that doesn't exist yet.

**Track C's own asymmetry, explained:** LV gets both extra seeds at full budget because
it's cheap ($2.4$h each) and its own finding ("ties, doesn't win") is a clean
accuracy-vs-cost comparison that benefits from real error bars. The pendulum gets only
**one** extra seed, because the question that actually matters there — *does the
epoch-3,000 overfitting onset happen again, or was seed 42 unlucky?* — only needs one
more data point to answer "yes, general" or "inconclusive, needs more"; a second full
$10$k pendulum re-run ($7.1$h) is deferred until that first answer is in.

---

# Part 4 — Ownership beyond the seed re-runs

These roles were already fixed by [`12`](./12_phase3_roadmap.md) §5.4 and the original
plan's Task 4.2–4.4; restated here so this doc is self-contained.

- **Monjur & Shams — statistical synthesis (adapted from the original plan's Task 4.1
  co-lead role).** Once every track's $N=3$ pass lands, they compute Mean ± Std across
  every extended table and write the report's statistical-rigor section. This is a
  coordination/aggregation role, not "re-run everyone else's experiments" — each track
  owner runs their own extra seeds (Part 3), matching the zero-conflict, folder-isolated
  model that worked cleanly for all of Phase 3.
- **Abrar — figures & report lead (Task 4.2/4.3, unchanged from [`12`](./12_phase3_roadmap.md) §5.4).**
  300+ DPI publication figures, LaTeX paper/report compilation. **No longer double-booked
  with a heavy re-run** — Track D already sits at $N=3$ (§2.1b), so unlike the earlier
  draft of this plan, his Phase 4 load is just figures + report, freeing him to start
  those immediately rather than waiting on a multi-hour sweep. His only optional extra
  task is the $N=5$ stretch (§2.5), and only if the team decides to go there.
- **Nawriz & Abhishek — codebase audit & presentation polish (Task 4.4, unchanged).**
  Final peer code review across all 5 merged tracks, clean up any stray artifacts,
  assemble the presentation deck. Abhishek also owns Track B's own re-run (Part 3); both
  are light enough to combine with this role.

---

# Part 5 — Definition of done

| Check | Target |
| :--- | :--- |
| Every Part 3 row's re-runs complete | metrics saved under each track's existing `results/phase3/<slug>/` folders, `_seedN` or equivalent suffix, no overwriting the original $N{=}1$ run |
| Mean ± Std reported | for every table/finding that got re-seeded, not just the headline number |
| Sign-flips or contradicted findings reported honestly | if a Phase 3 conclusion doesn't hold at $N=3$, that goes in the report as a finding, not quietly dropped (matching Track B's and Track D's own precedent of reporting inconvenient results) |
| Publication figures at 300+ DPI | for every table/finding going into the final report |
| Final report / paper draft compiled | LaTeX, all tables and figures integrated |
| Final codebase audit complete | no stray artifacts, all 5 tracks' folder-isolation still intact |

---

*Previous: [`12_phase3_roadmap.md`](./12_phase3_roadmap.md) — Phase 3 is closed, all
5 tracks merged into `main`, 249/250 tests passing.*
