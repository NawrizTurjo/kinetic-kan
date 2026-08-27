# 📈 Cross-Domain Stability Fixes: Results & Verdict

> **Branch:** `fix/phase1-revisit`
> **Status:** ✅ **complete** — 11 probes @ 2,000 epochs (`results/_probe/`) + 4 full runs @ 10,000 epochs (`results/_fixed/`)
> **Chain:** [`06`](./06_suggested_fixes.md) diagnosis → [`07`](./07_fix_changelog.md) code → [`08`](./08_how_to_run_fixes.md) how to run → **this doc: pendulum results & verdict** → [`10`](./10_sir_root_cause_and_fix.md) SIR root cause
>
> 📌 **Scope note.** This document is now the **pendulum** record. SIR was later
> root-caused and fixed by a different mechanism entirely — see
> [`10_sir_root_cause_and_fix.md`](./10_sir_root_cause_and_fix.md). The SIR section below
> is retained only as the historical record of the attempt that failed, and is clearly
> marked as superseded.

---

## 🏁 Verdict

| System | Before | After | Verdict |
| :--- | :---: | :---: | :--- |
| **Damped pendulum** | $R^2 = -1.318$, full MSE $8.99\times10^{-1}$ | $R^2 = \mathbf{+0.647}$, full MSE $\mathbf{4.48\times10^{-2}}$ | ✅ **fixed — 20× better** |
| **SIR** | died at epoch 8686 (NaN) | see [`10`](./10_sir_root_cause_and_fix.md) | ✅ **fixed, by `--time_scale` + projection** |

**The pendulum fix is one flag: `--t_train_end 5.0`.** The activation change we predicted
would be necessary turned out not to be — and the control run added to test that
prediction is what refuted it.

---

## 🏆 Which pendulum configuration to use

### ✅ Ship `pendulum_control_win5`

```powershell
python train.py --dataset damped_pendulum --basis rbf --solver tsit5 `
  --grid_len 8 --lr 0.003 --grad_clip 1.0 --t_train_end 5.0 --epochs 10000
```

**No `--act` flag** — it uses the default SiLU. The entire fix is the training window.

### Why this one and not `pendulum_fixed`

Both ran 10,000 epochs with window 5.0; they differ only in activation.

| | **`pendulum_control_win5`** (SiLU) | `pendulum_fixed` (identity) |
| :--- | :---: | :---: |
| Train MSE | $9.28\times10^{-5}$ | $\mathbf{2.55\times10^{-5}}$ (3.6× better) |
| $\theta$ RMSE | $0.0114$ | $\mathbf{0.0048}$ (2.4× better) |
| Extrap MSE | $\mathbf{8.99\times10^{-2}}$ | $1.34\times10^{-1}$ |
| **Full MSE $[0,10]$** | $\mathbf{4.48\times10^{-2}}$ | $6.68\times10^{-2}$ |
| **Extrap $R^2$** | $\mathbf{+0.647}$ | $+0.473$ |
| **Final ÷ best train MSE** | $\mathbf{1.1\times}$ | $94.3\times$ ⚠️ |
| Max pre-clip gradient | $\mathbf{147}$ | $6{,}799$ |
| Flatness verdict | ✅ `descending` | ⚠️ `REGRESSED` |

Three reasons, in order of importance:

**1. It generalises better.** Full-horizon MSE $4.48\times10^{-2}$ vs $6.68\times10^{-2}$
— a **49% advantage** — and it is the only one of the two whose extrapolation $R^2$
clears $0.6$. Since extrapolation *is* the deliverable, this decides it.

**2. It is stable.** `pendulum_fixed` ends 94× worse than its own best checkpoint (a
late-training spike; max pre-clip gradient $6{,}799$ against $147$). Selection-on-training-loss
means its reported checkpoint is still sound, but the run is one unlucky seed away from
losing it. `pendulum_control_win5` finishes essentially where it peaked ($1.1\times$).

**3. It is simpler.** Default activation, one changed flag. Nothing to justify in the
report beyond the window.

> **The trade is explicit:** `identity` wins on *fit* (3.6× better train MSE, 2.4× better
> $\theta$ RMSE) and loses on *generalisation and stability*. This is the third time in
> this project that **the better fit was the worse model** — see §Methodological lessons.

### One caveat on the epoch budget

| identity + window 5.0 | Full MSE | $R^2$ |
| :--- | :---: | :---: |
| @ 2,000 epochs (`pend_id_win5`) | $\mathbf{4.32\times10^{-2}}$ | $\mathbf{+0.662}$ |
| @ 10,000 epochs (`pendulum_fixed`) | $6.68\times10^{-2}$ | $+0.473$ |

5× the compute produced a **55% worse** trajectory. The best pendulum number in the whole
project is still a 2,000-epoch probe run — and it beats `pendulum_control_win5` by 4%,
which at $N=1$ is not a distinguishable difference.

**Implication:** the 10,000-epoch budget was tuned on Lotka-Volterra and does not
automatically transfer. A short budget sweep on the pendulum would likely find the knee
well below 10,000.

---

## 📊 Every pendulum run, ranked

Ranked by **full-horizon MSE over $[0,10]$** — the only window-independent metric, since
changing `--t_train_end` moves both the training set *and* the extrapolation window.

| Run | act | window | epochs | Train MSE | **Full MSE** | Extrap $R^2$ | final ÷ best |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pend_id_win5` | identity | **5.0** | 2,000 | $5.00\times10^{-4}$ | $\mathbf{4.32\times10^{-2}}$ | $\mathbf{+0.662}$ | $1.0\times$ |
| **`pendulum_control_win5`** ⭐ | **silu** | **5.0** | 10,000 | $9.28\times10^{-5}$ | $\mathbf{4.48\times10^{-2}}$ | $\mathbf{+0.647}$ | $1.1\times$ |
| `pendulum_fixed` | identity | **5.0** | 10,000 | $\mathbf{2.55\times10^{-5}}$ | $6.68\times10^{-2}$ | $+0.473$ | $94.3\times$ |
| `pend_tanh_win5` | tanh | **5.0** | 2,000 | $1.28\times10^{-2}$ | $1.37\times10^{-1}$ | $-0.026$ | $1.0\times$ |
| `pend_id_lossw` | identity | 3.0 | 2,000 | $5.79\times10^{-4}$ | $6.39\times10^{-1}$ | $-0.823$ | $1.0\times$ |
| `pend_tanhact` | tanh | 3.0 | 2,000 | $8.20\times10^{-3}$ | $7.53\times10^{-1}$ | $-1.143$ | $1.2\times$ |
| `pend_control` | silu | 3.0 | 2,000 | $3.29\times10^{-1}$ | $8.94\times10^{-1}$ | $-1.269$ | $1.0\times$ |
| `pendulum` *(original failure)* | silu | 3.0 | 10,000 | $2.87\times10^{-1}$ | $8.99\times10^{-1}$ | $-1.318$ | $1.0\times$ |
| `pend_gridlims` | silu | 3.0 | 2,000 | $4.18\times10^{-1}$ | $1.04$ | $-1.615$ | $1.0\times$ |
| `pend_identity` | identity | 3.0 | 2,000 | $2.62\times10^{-4}$ | $1.11$ | $-2.170$ | $1.0\times$ |
| `pend_lossw` | silu | 3.0 | 2,000 | $3.43\times10^{-3}$ | $1.48$ | $-3.234$ | $1.0\times$ |

**The table splits cleanly on one variable.** Every window-5.0 run occupies the top four
places with a positive or near-zero $R^2$; every window-3.0 run is below them with a
strongly negative $R^2$ — *regardless of activation, loss weighting, or grid span.*

**The window is the fix. Everything else is second-order.**

---

## 🔬 The 2×2: our explanation was wrong

`pendulum_control_win5` existed for exactly one purpose — to fill the missing cell of the
activation × window factorial at matched budget:

| | **window 3.0** | **window 5.0** |
| :--- | :---: | :---: |
| **SiLU** | $8.99\times10^{-1}$, $R^2 = -1.318$ ❌ | $\mathbf{4.48\times10^{-2}}$, $R^2 = \mathbf{+0.647}$ ✅ |
| **identity** | $1.11$, $R^2 = -2.170$ ❌ *(2,000 ep)* | $6.68\times10^{-2}$, $R^2 = +0.473$ ✅ |

### ❌ Hypothesis C2 is refuted

[`06`](./06_suggested_fixes.md) §Part 2 argued that the `KDense` residual branch
$W\!\cdot\!\mathrm{silu}(x)$ **cannot represent** $\dot\theta = \omega$ across both signs,
because $\mathrm{silu}(-4) = -0.072$. The supporting evidence was a probe of the learned
$f_1(0,\omega)$ that came back roughly constant at $\approx -3$, sloping the wrong way.

**That probe was measured on a model that had failed to converge, and it does not
generalise.** With window 5.0, SiLU reaches $\theta$ RMSE $0.0114$ and $\theta_{\min} =
-1.380$ against a true $-1.385$ — it learns the relation perfectly well.

What survives is a weaker, data-efficiency claim, still worth reporting:

> At the short window, `identity` reaches train MSE $2.62\times10^{-4}$ in **2,000
> epochs** while SiLU is still at $2.87\times10^{-1}$ after **10,000** — a $\sim$1,000×
> gap at one-fifth the compute. SiLU is not incapable; it is dramatically harder to
> optimise when data coverage is thin.

**The strong form — *"SiLU cannot represent sign-changing dynamics"* — must not go in the
report.** Only the conditional form is supported. A ⛔ banner marking this appears at the
head of C2 in [`06`](./06_suggested_fixes.md).

### Why the control run mattered

Without it we would have shipped a confident architectural claim that the very next
experiment would have demolished. It cost one extra job riding alongside two already
running, and it changed the finding.

**This is the single most valuable methodological lesson of the whole effort.**

### Findings that survived

**Finding A — a good fit is not a good model.** `pend_identity` fits **1,256×** better
than the control and produces a *worse* full-horizon trajectory (1.110 vs 0.894).
`pend_lossw`: 96× better fit, 1.7× worse overall.

**Finding B — the tanh-saturation hypothesis is dead.** `pend_gridlims` (widening the
grid to $(-3,3)$, the fix saturation would imply) finished **second-to-last**, worse than
the untouched control. This matches the direct measurement in [`06`](./06_suggested_fixes.md):
Lotka-Volterra is *more* saturated (41.7%, 6 of 7 grid cells empty) than pendulum $\omega$
(39.3%, 1 empty) and converges fine. **Closed.**

**Finding C — loss weighting hurts the pendulum.** `pend_id_lossw` gives the best
$\theta$ RMSE in the table ($0.0065$) and a full MSE **15× worse** than `pend_id_win5`.
Dropped from the pendulum recipe. (It is a different story for SIR — see
[`10`](./10_sir_root_cause_and_fix.md).)

---

## 🦠 SIR — superseded by doc 10

> ## ⛔ THIS SECTION IS HISTORICAL
>
> The attempt recorded here (`--conserve_sum` + `--loss_weighting std`) **failed**, and
> its stated conclusions — including *"the frozen fixed point is a task-design problem"* —
> were **wrong**. SIR was subsequently root-caused and fixed. **See
> [`10_sir_root_cause_and_fix.md`](./10_sir_root_cause_and_fix.md).** Retained only to
> document what was tried and why it did not work.

`sir_fixed` (first attempt: `--conserve_sum 1.0 --loss_weighting std`) diverged at epoch
3,055 and auto-aborted at 3,154. Mass $1.0147$; `I_drift` $0.0000$ (still frozen).

**What the actual root cause turned out to be:** SIR is posed over a horizon of $50$ with
a derivative of $\sim10^{-2}$, so a Glorot-initialised KAN starts **19× too fast** and
epoch-0 loss is $3.16\times10^{2}$. From there $\mathbf{f}\to\mathbf{0}$ is simply the
steepest descent direction available — the "frozen fixed point" was rational optimizer
behaviour, not a scale-sensitivity defect of the MSE. Rescaling time
(`--time_scale 10`) removes the cause. Mass drift was fixed by projecting the field onto
the zero-sum subspace (`--conserve_mode projection`) rather than penalising it.

Final SIR results: mass $\mathbf{1.0000}$, $I$ drift $\mathbf{0.0524}$ against a true
$0.0525$, extrapolation $R^2 = \mathbf{+0.9981}$. Full detail in [`10`](./10_sir_root_cause_and_fix.md).

**The one conclusion from this attempt that did hold:** the X1 non-finite gradient guard
worked exactly as designed — where the pre-fix run silently wrote a NaN `final_model.pt`
and burned 1,299 wasted epochs, this run kept a valid checkpoint, stopped at 3,154, and
reported precisely what happened.

---

## ✅ Fix scorecard (final)

| ID | Fix | Verdict | Evidence |
| :-: | :--- | :--- | :--- |
| **P4** | Longer training window | ✅ **the pendulum fix** | $R^2$ $-1.318 \to +0.647$; works for **both** activations |
| **X1** | Non-finite gradient guard + abort | ✅ **works, proved in production** | caught SIR's divergence at 3,055, preserved a valid checkpoint |
| **S3** | `--time_scale` *(doc 10)* | ✅ **the SIR fix** | removes divergence **and** the frozen fixed point |
| **S4** | `--conserve_mode projection` *(doc 10)* | ✅ **works** | mass error $5.9\times10^{-2} \to 7\times10^{-7}$, on all horizons |
| **S5** | `--vanish_dim` *(doc 10)* | ✅ **optional bonus** | 25× extrap RMSE, but not required to pass |
| **P1** | `--act identity` | ⚠️ **speed, not quality** | ~1,000× faster to fit at window 3.0; worse extrapolation *and* stability at window 5.0 |
| **X2** | `--loss_weighting std` | ⚠️ **system-dependent** | worst pendulum option; superseded for SIR by S3/S4 |
| **S2** | `--conserve_sum` penalty | ❌ **wrong instrument** | scoped to the training window, where mass was already fine |
| **P3** | `--grid_lims` | ❌ **rejected** | worse than the untouched control |
| **X3/X4** | post-clip logging + `stability` block | ✅ **works** | made both the SIR abort and the pendulum late spike diagnosable in one command |

### Three hypotheses published and then refuted

Recording all three, because the refutations are as informative as the fixes:

1. **"Gradient clipping is the shared root cause of both failures"**
   ([`05`](./05_phase2_benchmark_analysis.md), both `reflection.md` files). It was
   *necessary* for SIR and *irrelevant* for the pendulum — and per [`10`](./10_sir_root_cause_and_fix.md)
   even for SIR it treated a symptom: the gradient spike is at epoch **11**, not 8,686.
2. **"SiLU cannot represent sign-changing dynamics"** ([`06`](./06_suggested_fixes.md) §C2).
   Refuted by `pendulum_control_win5`.
3. **"Tanh saturation starves the spline basis"** (`pendulum/reflection.md`). Rejected
   twice — by direct measurement and by `pend_gridlims`.

---

## 🛠️ Methodological lessons

**1. Always run the control that could falsify your claim.** `pendulum_control_win5` cost
one parallel job and overturned the headline finding.

**2. One factor at a time.** The Phase-2 re-runs changed clipping, `grid_len` and `lr`
together and could attribute nothing. Every run here moved one variable.

**3. Aggregate MSE hid both failures.** $\omega$ dominates the pendulum loss while
$\theta$ fails; SIR's frozen fixed point yields a respectable MSE. Per-dimension
diagnostics (`analyze_fixes.py`) were necessary, not decorative.

**4. Choose a metric that survives a changed experiment.** `extrap_mse` is not comparable
across `--t_train_end` values. Ranking by `train_mse` would have selected `identity`
twice — the worse model both times.

**5. A better fit is not a better model.** Three separate occurrences: `pend_identity`,
`pendulum_fixed` vs `pendulum_control_win5`, and identity at 2,000 vs 10,000 epochs.

**6. More epochs is not automatically better.** The best pendulum result came from a
2,000-epoch run.

**7. Verify pass criteria before trusting them.** The "spike ratio in the low tens" rule
was wrong — it measures the *pre-clip* norm. Doc [`10`](./10_sir_root_cause_and_fix.md)
adds the mirror-image caveat: on a converged run the ratio inflates because the *median*
collapses, so read `grad_norm_max_preclip` instead.

**8. `Wait-Process` is unsafe on long sweeps.** PID reuse killed the first 8-job probe
after every job had succeeded. Fixed with `.WaitForExit()`; see [`07`](./07_fix_changelog.md).

**9. Fix the cause, not the symptom.** The strongest result in this whole effort
([`10`](./10_sir_root_cause_and_fix.md)) came from measuring the field magnitude *at
initialisation* rather than tuning the optimizer. Four numbers taken before the first
step explained two of three failures.

---

## 📋 Where the project stands

**Resolved**

* ✅ Pendulum — **fixed**, $R^2 = +0.647$, 20× better full-horizon MSE, via `--t_train_end 5.0`
* ✅ SIR — **fixed**, $R^2 = +0.9981$, mass exact, via `--time_scale` + projection ([`10`](./10_sir_root_cause_and_fix.md))
* ✅ Three hypotheses tested and closed (clipping as shared cause, tanh saturation, SiLU representational limit)
* ✅ Three silent checkpoint-replay bugs found and fixed ([`10`](./10_sir_root_cause_and_fix.md) §Part 5)

**Open**

* ⚠️ **Late-training instability persists**, mildly, in three runs — `pendulum_fixed`
  (final 94× its best) and both SIR runs are flagged `REGRESSED`. Selection-on-training-loss
  makes the reported checkpoints sound, but this is not claimed to be fixed.
* ⚠️ **Pendulum epoch budget** — 2,000 beats 10,000; worth a short sweep for the knee.
* ⚠️ **Pendulum accuracy still trails SIR by orders of magnitude** ($R^2$ $0.647$ vs
  $0.998$). See the recommendation below.
* ❌ **Lorenz never run** — last outstanding Phase-2 deliverable
  (`.\run_phase2.ps1 -Only lorenz`, ~2.6 h). Given [`10`](./10_sir_root_cause_and_fix.md),
  check its $\lvert f\rvert \cdot T$ product first — Lorenz has $\lvert f\rvert\sim10^{1}$,
  so it may need `--time_scale` in the *opposite* direction.
* ❌ **Multi-seed replication** (Phase 4 Task 4.1) — every number here is $N=1$.

### 💡 Recommended next step for the pendulum

Doc [`10`](./10_sir_root_cause_and_fix.md)'s central insight — *supply the missing
structure analytically instead of asking the model to infer it* — has a direct pendulum
analogue that has **not** been tried:

$$\dot\theta = \omega \qquad\qquad \dot\omega = -\mu\omega - \tfrac{g}{L}\sin\theta$$

**The first component is exactly known.** A `SecondOrderField` wrapper — same pattern as
`ZeroSumField` / `VanishingDimField` — hardcoding $f_1 = y_2$ would halve what must be
learned and make the relation exact rather than approximated.

This also explains retrospectively why `--act identity` helped at the short window: it
let $W\!\cdot\!x$ represent $f_1 = \omega$ linearly. The structural version makes it
exact instead of merely learnable. Most likely route to closing the gap with SIR.

**Phase 3 readiness**

* **Novelty 3** (stiffness phase map) — ✅ **unblocked.** Use `--t_train_end 5.0` and the
  default SiLU; the activation choice is second-order.
* **Novelty 1** (gradient-norm dynamics) — ✅ ready, **no new runs needed**; every
  completed run carries per-epoch `grad_norms`, post-fix runs add `post_clip_grad_norms`.
* **Novelties 4 & 5** — blocked on `pysindy` / `torchdiffeq` (not installed) and a
  missing $L_1$ edge-pruning step.

---

## 🚨 Repo hygiene — fix before the report is submitted

Found while auditing the codebase; unrelated to the stability work, recorded so they are
not lost.

### 1. 🔴 The base paper's authors are wrong in four places

The KAN-ODEs authors are **Benjamin C. Koenig, Suyong Kim, Sili Deng**. Only
`implementation/README.md` has this right. Every other citation — **including the
presentation deck already delivered** — says *Zachary Koenig, Jihoon Kim, Yuntian Deng*:

| File | Line | Contains |
| :--- | :---: | :--- |
| `README.md` | 170, 180 | prose citation **and** the BibTeX `author` field |
| `docs/README.md` | 82, 90 | prose citation **and** BibTeX |
| `docs/03_presentation/KAN-ODE.tex` | 130 | `Z. Koenig, J. Kim, Y. Deng` — **needs recompiling** |
| `docs/02_proposal/CSE402_Project_Proposal_Guide.md` | 43 | `Z. Koenig, J. Kim, Y. Deng` |

### 2. 🟠 `implementation/README.md` is badly out of date

Its "What Needs to Be Done" list still marks as *pending* things that are **done**: the
noise sweep, `--dataset` CLI wiring, the RBF factor-of-2 correction, the B-spline re-run,
the flagship re-run, the `evaluate.py` config bug. A teammate reading it would redo
finished work.

### 3. 🟠 The root `README.md` describes a repo that does not exist

Lists `tests/test_adjoint.py` (never written) and per-member branches
`feat/m1-kan-architecture` … `feat/m5-chaos-visuals` (none exist). Still advertises
`--lr 5e-4`, the abandoned learning rate.

### 4. 🟡 CI never ran on this branch

`ci.yml` triggers on `main`, `develop`, `feat/**`. This branch is `fix/phase1-revisit` —
`fix/**` is not matched, so no commit here was CI-verified.

### 5. 🟡 Two Phase-3 dependencies are not installed

```powershell
pip install torchdiffeq pysindy
```
