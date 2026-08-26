# 📈 Cross-Domain Stability Fixes: Probe Results & Full-Run Plan

> **Branch:** `fix/phase1-revisit`
> **Status:** probe **complete** (11 runs @ 2,000 epochs, `results/_probe/`) · full runs **IN PROGRESS** (3 jobs @ 10,000 epochs → `results/_fixed/`, ETA ~3 h)
> **Chain:** [`06`](./06_suggested_fixes.md) diagnosis → [`07`](./07_fix_changelog.md) code changes → [`08`](./08_how_to_run_fixes.md) how to run → **this doc: what the runs actually showed**

---

## 🏁 Executive summary

The damped pendulum, which failed **three times** across Phase 2, now converges and
extrapolates. The decisive result:

| | Best pre-fix run | **`pend_id_win5`** | Change |
| :--- | :---: | :---: | :---: |
| Train MSE | $2.87\times10^{-1}$ | $\mathbf{5.00\times10^{-4}}$ | **574× better** |
| $\theta$ RMSE | $0.7555$ | $\mathbf{0.0211}$ | **36× better** |
| Full-horizon MSE $[0,10]$ | $8.94\times10^{-1}$ | $\mathbf{4.32\times10^{-2}}$ | **20.7× better** |
| **Extrapolation $R^2$** | $-1.318$ | $\mathbf{+0.662}$ | **first positive value ever** |

It took **two independent fixes**, and neither works alone:

1. **`--act identity`** — the SiLU residual branch cannot represent a linear function
   across both signs, so it could not learn $\dot\theta = \omega$. Fixes the *fit*.
2. **`--t_train_end 5.0`** — a 1.5-period training window does not constrain a
   5-period horizon. Fixes the *extrapolation*.

Applied alone, `identity` produces a **worse** full-horizon trajectory than doing
nothing (1.110 vs 0.894) — a textbook overfit. Only together do they work.

SIR improved substantially on every metric but **remains frozen on a spurious fixed
point**; that is a genuine open limitation, documented below.

> ⚠️ **These are 2,000-epoch probe numbers.** The 10,000-epoch confirmation run is still
> executing. Treat every figure here as provisional until `results/_fixed/` lands.

---

## 🔬 What was run

| Batch | Runs | Epochs | Output | Status |
| :--- | :---: | :---: | :--- | :--- |
| Probe round 1 | 8 (5 pendulum + 3 SIR) | 2,000 | `results/_probe/` | ✅ complete |
| Probe round 2 | 3 pendulum | 2,000 | `results/_probe/` | ✅ complete |
| **Probe total** | **11** | | `results/_probe/tables/` | ✅ |
| Full run | 3 (2 pendulum + 1 SIR) | 10,000 | `results/_fixed/` | ⏳ **in progress** |

**Everything below the executive summary is from the 11 completed probe runs.** The
full-budget numbers are not in yet; every conclusion here is stated at the 2,000-epoch
budget and is pending confirmation at 10,000.

Each run changes **exactly one thing** from the failing baseline (one-factor-at-a-time),
which is what makes the attribution below possible. The earlier Phase-2 re-runs changed
clipping, `grid_len` and `lr` simultaneously and consequently could not attribute
anything.

**Baseline for both systems:** RBF basis, Tsit5, `substeps=2`, $G=8$,
lr $3\times10^{-3}$, `--grad_clip 1.0`, seed 42.

---

## 🎯 Pendulum results

Ranked by **full-horizon MSE over $[0,10]$**. This is the only metric comparable across
runs, because changing `--t_train_end` changes *both* the training set and the
extrapolation window — so `extrap_mse` for the `win5` runs is measured over $(5,10]$
while the others use $(3,10]$.

| Run | What changed | Train MSE | $\theta$ RMSE | **Full MSE** | Extrap $R^2$ | vs. control |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`pend_id_win5`** | `identity` + window 5.0 | $5.00\times10^{-4}$ | $\mathbf{0.0211}$ | $\mathbf{4.32\times10^{-2}}$ | $\mathbf{+0.662}$ | **20.7× better** ✅ |
| `pend_tanh_win5` | `tanh` + window 5.0 | $1.28\times10^{-2}$ | $0.1343$ | $1.37\times10^{-1}$ | $-0.026$ | 6.5× better ✅ |
| `pend_id_lossw` | `identity` + `loss_weighting` | $5.79\times10^{-4}$ | $\mathbf{0.0065}$ | $6.39\times10^{-1}$ | $-0.823$ | 1.4× better |
| `pend_tanhact` | `tanh` | $8.20\times10^{-3}$ | $0.1149$ | $7.53\times10^{-1}$ | $-1.143$ | 1.2× better |
| `pend_control` | *(baseline)* | $3.29\times10^{-1}$ | $0.8015$ | $8.94\times10^{-1}$ | $-1.269$ | — |
| `pend_gridlims` | `grid_lims -3 3` | $4.18\times10^{-1}$ | $0.8932$ | $1.04$ | $-1.615$ | 1.2× **worse** ❌ |
| `pend_identity` | `identity` | $\mathbf{2.62\times10^{-4}}$ | $0.0132$ | $1.11$ | $-2.170$ | 1.2× **worse** ❌ |
| `pend_lossw` | `loss_weighting std` | $3.43\times10^{-3}$ | $0.0403$ | $1.48$ | $-3.234$ | 1.7× **worse** ❌ |

### Finding 1 — hypothesis C2 confirmed: SiLU cannot span both signs

Every activation change fixes the fit dramatically. `theta_min` is the tell:

| Run | $\theta_{\min}$ predicted | True | Can it swing negative? |
| :--- | :---: | :---: | :--- |
| `pend_control` (SiLU) | $-0.279$ | $-1.385$ | ❌ no |
| `pend_gridlims` (SiLU) | $-0.461$ | $-1.385$ | ❌ no |
| `pend_identity` | $\mathbf{-1.396}$ | $-1.385$ | ✅ yes |
| `pend_id_lossw` | $-1.397$ | $-1.385$ | ✅ yes |
| `pend_tanhact` | $-1.503$ | $-1.385$ | ✅ yes |

The mechanism, established in [`06`](./06_suggested_fixes.md) §Part 2: the `KDense`
residual branch is $W\!\cdot\!\mathrm{silu}(x)$, and $\mathrm{silu}(-4) = -0.072$ — it
collapses to $\approx 0$ for negative inputs. Learning $\dot\theta = \omega$ requires a
linear function across **both** signs. The spline branch cannot supply it either
(bounded by `tanh`). **The pendulum is the only system in this project with
sign-changing states** — Lotka-Volterra and SIR are strictly positive, where SiLU is
near-linear. That is why the recipe transferred to one family and not the other.

### Finding 2 — a good fit is not a good model

`pend_identity` fits the training window **1,256×** better than the control and produces
a **worse** full-horizon trajectory (1.110 vs 0.894). Same for `pend_lossw`: 96× better
fit, 1.7× worse overall.

The training window $[0,3]$ is only **1.5 oscillation periods** of a 5-period horizon,
and the trajectory traces a 1-D curve through 2-D state space. Fitting that curve
perfectly does not constrain the vector field off it. Extending to $[0,5]$ turns
$R^2 = -2.170$ into $\mathbf{+0.662}$.

### Finding 3 — the tanh-saturation hypothesis is dead

`pend_gridlims` (widening the spline grid to $(-3,3)$, the fix that saturation would
imply) is **the second-worst run in the table** — worse than the untouched control. This
matches the direct measurement in [`06`](./06_suggested_fixes.md): Lotka-Volterra is
*more* saturated (41.7%, 6 of 7 grid cells empty) than pendulum $\omega$ (39.3%, 1 empty)
and converges fine. Closed.

### Finding 4 — loss weighting helps the fit, hurts extrapolation

`pend_id_lossw` gives the best $\theta$ RMSE in the whole table ($0.0065$) but a
full-horizon MSE of $0.639$ — **15× worse** than `pend_id_win5`. It is therefore
**dropped** from the pendulum's full-run configuration. (It is retained for SIR, where
it is the single best-performing option — see below.)

---

## 🦠 SIR results

All three at 2,000 epochs, so compare **probe-to-probe only** — the 10k pre-fix
reference had `I_rmse` $0.0776$, which these do not beat simply because they are
undertrained by 5×.

| Run | Train MSE | Extrap MSE | **Full MSE** | `I_rmse` | **Mass $S{+}I{+}R$** | `I_drift` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sir_control` | $1.74\times10^{-2}$ | $7.21\times10^{-2}$ | $3.78\times10^{-2}$ | $0.2840$ | $1.1347$ | $0.0000$ |
| `sir_conserve` | $1.19\times10^{-2}$ | $4.42\times10^{-2}$ | $2.39\times10^{-2}$ | $0.1939$ | $\mathbf{1.0086}$ | $0.0000$ |
| **`sir_lossw`** | $\mathbf{7.31\times10^{-3}}$ | $\mathbf{3.29\times10^{-2}}$ | $\mathbf{1.68\times10^{-2}}$ | $\mathbf{0.1771}$ | $\mathbf{0.9977}$ | $0.0000$ |

**`--conserve_sum` works:** mass error drops from **13.5% → 0.86%** (15× closer to the
invariant) and full MSE improves 1.6×.

**`--loss_weighting std` wins on every metric** — and reaches $0.9977$ mass *without*
being told the invariant exists, because balancing the compartments implicitly keeps
the sum honest.

They address different failure modes, so the full run **combines them**.
`sir_conserve`'s flatness ratio of $4.334$ shows it was still descending steeply at
epoch 2,000, so the 5× longer budget should help substantially.

### ⚠️ Open limitation: SIR is still frozen

All three runs show `I_drift = 0.0000` over a window where the true I decays
$0.0568 \to 0.0043$ (true drift $0.0525$). The model learns $\mathbf{f}\approx\mathbf{0}$
near the endemic equilibrium and parks there. The pre-fix 10k run was also frozen
($0.0003$), so **this is structural, not a short-budget artifact.** Expect it to persist
at 10,000 epochs.

Root cause is task design, not optimisation: the extrapolation window has standard
deviation $0.002$–$0.017$ against $0.11$–$0.36$ in training, and SIR derivatives are
~270× smaller than Lotka-Volterra's, so an unweighted MSE supplies almost no gradient
signal there. Candidate next steps (untested): per-compartment *relative* loss, or
training I on a log scale.

> 📐 **Report RMSE/MAE for the SIR extrapolation window, not $R^2$.** With that
> near-zero variance the $R^2$ denominator collapses, which is why the values look
> catastrophic ($-190$ to $-419$) while RMSE is a reasonable $0.03$–$0.07$.

---

## ✅ Fix scorecard

| ID | Fix | Verdict | Evidence |
| :-: | :--- | :--- | :--- |
| **X1** | Non-finite gradient guard | ✅ **works** | `nonfinite = 0` on all 11 runs; the pre-fix 10k SIR died at epoch 8686 |
| **X2** | `--loss_weighting std` | ⚠️ **system-dependent** | best option for SIR; hurts pendulum extrapolation → dropped there |
| **S2** | `--conserve_sum` | ✅ **works** | SIR mass error 13.5% → 0.86% |
| **P3** | `--grid_lims` | ❌ **rejected** | `pend_gridlims` is worse than the control |
| **P1** | `--act identity` | ✅ **works** | 1,256× better fit; only fix that lets $\theta$ swing negative |
| **P4** | longer window | ✅ **works** | $R^2$: $-2.170 \to +0.662$ |
| **X3/X4** | post-clip logging + `stability` block | ✅ **works** | corrected a wrong pass criterion — see below |

### A criterion we got wrong

The original pass rule in [`08`](./08_how_to_run_fixes.md) required a gradient
`spike_ratio` "in the low tens". The probe disproved it: `pend_identity` hit
**25,879×** and `sir_lossw` **10,356,094×**, both completing cleanly with
`nonfinite = 0`. That figure is the **pre-clip** norm and `--grad_clip` bounds the step
regardless. It only signals trouble when `nonfinite > 0` or clipping is disabled. Both
the doc and `analyze_fixes.py` have been corrected.

---

## ▶️ Full-run plan

`run_phase2.ps1 -Only fixes` has been updated from the probe results (was
`identity + loss_weighting`, now `identity + window`):

| Job | Configuration | Purpose |
| :--- | :--- | :--- |
| `pendulum_fixed` | `--act identity --t_train_end 5.0` | the winner at full budget |
| `sir_fixed` | `--conserve_sum 1.0 --loss_weighting std` | both SIR fixes combined |
| `pendulum_control_win5` | `--act silu --t_train_end 5.0` | **control** — completes the 2×2 |

The third job closes a real gap. We have three cells of the activation × window
factorial and are missing the fourth:

| | window 3.0 | window 5.0 |
| :--- | :---: | :---: |
| **SiLU** | $0.894$ (control) | ❓ **this run** |
| **identity** | $1.110$ | $\mathbf{0.0432}$ |

Without it, "both fixes were necessary" is an inference. With it, it is measured — and
if SiLU + window 5.0 also lands near $0.04$, the activation claim collapses and we would
need to know that before writing it up.

### Command

```powershell
cd d:\level4\Term1\NUM_project\kinetic-kan\implementation
.\run_phase2.ps1 -Only fixes -Epochs 10000 -MaxParallel 3
```

3 jobs × 4 threads = 12 logical cores. Output: `results/_fixed/`.

> ⏱️ **Budget ~3 hours.** The `win5` runs use 101 training points instead of 61, so they
> are ~1.7× more expensive per epoch than the earlier pendulum runs. Measured in flight:
> ~1.0–1.3 s/epoch under 3-way contention. *(Ignore the ETA in the first ~2 minutes —
> tqdm's early estimate reads 5–6 h before the rate stabilises.)*

### 🔎 Early signal from the in-flight run (≈6%, epoch ~640 of 10,000)

**Preliminary — not a result.** Recorded because it already bears on the 2×2:

| Job | Train MSE @ ~640 | Full-horizon monitor |
| :--- | :---: | :---: |
| `pendulum_fixed` (identity + win5) | $\mathbf{2.98\times10^{-3}}$ | $\mathbf{4.66\times10^{-2}}$ |
| `pendulum_control_win5` (SiLU + win5) | $1.53\times10^{-1}$ | $4.01\times10^{-1}$ |

At matched epochs the SiLU control is **~51× worse on the fit** and **~9× worse on the
full horizon**. If that holds, the activation fix is confirmed as necessary — the longer
window alone does **not** rescue SiLU. `pendulum_fixed` has also already reached
$4.66\times10^{-2}$ on the monitor at 6% of budget, essentially matching the probe's
*final* $4.32\times10^{-2}$.

### Verify afterwards

```powershell
python analyze_fixes.py --root results/_fixed
python collate_results.py --root results/_fixed --out results/_fixed/tables --bucket best
```

**Success criteria:**

| Check | Target |
| :--- | :--- |
| `pendulum_fixed` extrap $R^2$ | $> +0.662$ (the probe value at 1/5 the budget) |
| `pendulum_fixed` full MSE | $< 4.32\times10^{-2}$ |
| `pendulum_control_win5` full MSE | $\gg 4.32\times10^{-2}$ — confirms the activation fix was needed |
| `sir_fixed` mass | $\to 1.0000$ |
| `sir_fixed` `I_rmse` | $< 0.0776$ (the 10k pre-fix reference) |
| all | `final_train` not NaN · `nonfinite = 0` · `aborted = null` |

---

## 🛠️ Methodological lessons (worth carrying into Phase 3)

**1. One factor at a time, or you learn nothing.** The Phase-2 re-runs changed clipping,
`grid_len` and `lr` together, so "did clipping help?" needed forensic analysis of the
training histories. This round changed exactly one thing per run, and the attribution
fell straight out of the summary table.

**2. Aggregate MSE hid both failures.** On the pendulum, $\omega$ dominates the loss and
fits well while $\theta$ — the dimension that actually fails — is invisible in the
average. On SIR, a frozen fixed point produces a perfectly respectable MSE. Both needed
per-dimension diagnostics (`analyze_fixes.py`), not a single number.

**3. Pick a metric that survives a changed experiment.** `extrap_mse` is not comparable
across different `--t_train_end` values, because the window itself moves. **`full_mse`
over the whole horizon is window-independent** and is what the pendulum ranking is
built on. Ranking by `train_mse` instead would have picked `pend_identity` — the run
with a *worse-than-baseline* trajectory.

**4. A short smoke test cannot predict mid-training stability.** Established in
`sir/reflection.md` (the 14-config, 100-epoch sweep could not have foreseen a blowup at
epoch 5018) and re-confirmed here: gradient spikes of $2.6\times10^{4}$ and
$1.0\times10^{7}$ appeared only well into full-length runs.

**5. Verify a pass criterion before trusting it.** The "spike ratio in the low tens"
rule was wrong and would have failed three perfectly healthy runs. See §A criterion we
got wrong.

**6. Operational: `Wait-Process` is unsafe on long sweeps.** The first 8-job probe
aborted with `Access is denied` on an `svchost` — Windows had recycled the PID of a
finished job, and `Wait-Process` resolves by PID. **No training was affected**; 6 of 8
had already written `metrics.json` and the other 2 kept running. Fixed by using
`.WaitForExit()` on the captured `Process` handle. Full write-up in
[`07`](./07_fix_changelog.md) §Bug fix: PID-reuse race.

> **If a sweep ever dies again:** nothing is lost. The training processes are
> independent. Wait for `Get-Process python` to return empty, then run
> `collate_results.py` manually.

---

## 📋 Where the project stands

**Resolved**

* ✅ Pendulum non-convergence — root-caused to two independent mechanisms, both fixed
* ✅ SIR NaN death at epoch 8686 — X1 guard; `nonfinite = 0` across all 11 probes
* ✅ SIR mass-conservation violation — `--conserve_sum`
* ✅ tanh-saturation hypothesis — measured and **rejected**
* ✅ "gradient clipping is the shared root cause" — **disproved**; it was necessary for
  SIR and irrelevant for the pendulum

**Open**

* ⚠️ SIR frozen fixed point — structural; needs relative or log-scale loss
* ⚠️ Full-budget confirmation of everything above — the run described here
* ❌ **Lorenz never run** — last outstanding Phase-2 deliverable
  (`.\run_phase2.ps1 -Only lorenz`, ~2.6 h). Expect it to need X1 and the activation fix,
  since Lorenz is also sign-changing.
* ❌ Multi-seed replication (Phase 4 Task 4.1) — no ordering claim in Table 1 is
  publishable at $N=1$

**Phase 3 readiness**

* **Novelty 3** (stiffness phase map) — **unblocked once `pendulum_fixed` confirms**,
  and it must use `--act identity`
* **Novelty 1** (gradient-norm dynamics) — ready now, **needs no new runs**; all 37
  completed runs carry per-epoch `grad_norms`, and post-fix runs add
  `post_clip_grad_norms`
* **Novelties 4 & 5** — still blocked on `pysindy` / `torchdiffeq` (not installed) and a
  missing $L_1$ edge-pruning step

**Documents to update after the full run:** [`05`](./05_phase2_benchmark_analysis.md)
§Phase 2 Completion Status, and both `reflection.md` files — all three still state that
gradient clipping is the shared root cause, which these runs disproved.

---

## 🚨 Repo hygiene — fix before the report is submitted

These are unrelated to the stability work but were found while auditing the codebase,
and are recorded here so they are not lost.

### 1. 🔴 The base paper's authors are wrong in four places

The KAN-ODEs authors are **Benjamin C. Koenig, Suyong Kim, Sili Deng**. Only
`implementation/README.md` has this right. Every other citation — **including the
presentation deck that was already delivered** — says *Zachary Koenig, Jihoon Kim,
Yuntian Deng*:

| File | Line | Contains |
| :--- | :---: | :--- |
| `README.md` | 170, 180 | prose citation **and** the BibTeX `author` field |
| `docs/README.md` | 82, 90 | prose citation **and** BibTeX |
| `docs/03_presentation/KAN-ODE.tex` | 130 | `Z. Koenig, J. Kim, Y. Deng` — **needs recompiling** |
| `docs/02_proposal/CSE402_Project_Proposal_Guide.md` | 43 | `Z. Koenig, J. Kim, Y. Deng` |

A wrong citation in a submitted report is a cheap, avoidable mark against the work.
Correct all four and rebuild `KAN-ODE.pdf`.

### 2. 🟠 `implementation/README.md` is badly out of date

Its "What Needs to Be Done" list still marks as *pending* several things that are
**done**: the noise sweep, the `--dataset` CLI wiring, the RBF factor-of-2 correction,
the B-spline re-run, the flagship re-run, and the `evaluate.py` config bug. A teammate
reading it would redo finished work.
**[`05`](./05_phase2_benchmark_analysis.md) is the accurate document**; the
implementation README needs a rewrite.

### 3. 🟠 The root `README.md` describes a repo that does not exist

* Lists `tests/test_adjoint.py` — never written.
* Lists per-member branches `feat/m1-kan-architecture` … `feat/m5-chaos-visuals` — none
  exist. The real branches are `feat/phase1-foundation-completion`,
  `feat/phase2-benchmarks`, `fix/phase1-revisit`.
* Still advertises `--lr 5e-4`, the abandoned learning rate (the project standardised on
  $2\times10^{-3}$).

### 4. 🟡 CI never ran on this branch

`.github/workflows/ci.yml` triggers on `main`, `develop`, `feat/**`. This branch is
`fix/phase1-revisit` — **`fix/**` is not matched**, so none of these commits were
CI-verified. Either add `fix/**` to the trigger list or rely on the PR to `main`
(PRs to `main` do trigger it).

### 5. 🟡 Two Phase-3 dependencies are not installed

`torchdiffeq` (Novelty 5) and `pysindy` (Novelty 4) are in `requirements.txt` but
missing from the environment. Everything else imports fine.

```powershell
pip install torchdiffeq pysindy
```
