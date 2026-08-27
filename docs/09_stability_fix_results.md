# 📈 Cross-Domain Stability Fixes: Results & Verdict

> **Branch:** `fix/phase1-revisit`
> **Status:** probe **complete** (11 runs @ 2,000 epochs, `results/_probe/`) · full runs **complete** (3 runs @ 10,000 epochs, `results/_fixed/`)
> **Chain:** [`06`](./06_suggested_fixes.md) diagnosis → [`07`](./07_fix_changelog.md) code changes → [`08`](./08_how_to_run_fixes.md) how to run → **this doc: results and verdict**

---

## 🏁 Verdict in one line

**Damped pendulum: FIXED.** **SIR: NOT FIXED.** And the mechanism behind the pendulum
fix is **not** what we predicted — the control run added to test that prediction
refuted it.

| System | Before | After | Verdict |
| :--- | :---: | :---: | :--- |
| **Damped pendulum** | $R^2 = -1.318$, full MSE $8.99\times10^{-1}$ | $R^2 = \mathbf{+0.647}$, full MSE $\mathbf{4.48\times10^{-2}}$ | ✅ **20× better, first positive $R^2$** |
| **SIR** | died at epoch 8686 (NaN) | diverged at epoch 3055, auto-aborted 3154 | ❌ **still unresolved** |

---

## 🔬 The pendulum: fixed, but our explanation was wrong

### The 2×2, now complete at 10,000 epochs

This is the table the `pendulum_control_win5` run existed to fill:

| | **window 3.0** | **window 5.0** |
| :--- | :---: | :---: |
| **SiLU** (original) | $8.99\times10^{-1}$, $R^2 = -1.318$ ❌ | $\mathbf{4.48\times10^{-2}}$, $R^2 = \mathbf{+0.647}$ ✅ |
| **identity** | $1.11$, $R^2 = -2.170$ ❌ *(2,000 ep)* | $6.68\times10^{-2}$, $R^2 = +0.473$ ✅ |

*(full-horizon MSE over $[0,10]$ — the only window-independent metric)*

**Reading across the bottom-left to top-right: the window is the decisive variable, not
the activation.** SiLU — the activation we blamed — works perfectly well once the
training window is long enough. It reaches $\theta$ RMSE $0.0114$ and $\theta_{\min} =
-1.380$ against a true $-1.385$.

### ❌ Hypothesis C2 is refuted

[`06`](./06_suggested_fixes.md) §Part 2 argued that the `KDense` residual branch
$W\!\cdot\!\mathrm{silu}(x)$ **cannot represent** $\dot\theta = \omega$ across both
signs, because $\mathrm{silu}(-4) = -0.072$. The evidence looked strong: a probe of the
learned $f_1(0,\omega)$ came back roughly constant at $\approx -3$ and sloping the wrong
way.

**That was measured on a model that had failed to converge**, and it does not
generalise. Given a longer window, SiLU learns the relation fine.

What survives is a weaker, data-efficiency claim, and it is still worth reporting:

> At the short window (3.0), `identity` reaches train MSE $2.62\times10^{-4}$ in **2,000
> epochs** while SiLU is still at $2.87\times10^{-1}$ after **10,000** — a $\sim$1,000×
> gap at one-fifth the compute. SiLU is not incapable; it is dramatically harder to
> optimise when data coverage is thin.

The strong form — *"SiLU cannot represent sign-changing dynamics"* — **must not go in
the report.** Only the conditional form is supported.

### Why the control run mattered

Without `pendulum_control_win5` we would have shipped a confident architectural claim
that the very next experiment would have demolished. The 2×2 cost one extra job riding
alongside two already running; it changed the paper's central finding.

**This is the single most valuable methodological lesson of the whole effort.**

### The best configuration is SiLU + window 5.0

| Run | Train MSE | $\theta$ RMSE | **Full MSE** | $R^2$ | Final vs best | Stability |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`pendulum_control_win5`** | $9.28\times10^{-5}$ | $0.0114$ | $\mathbf{4.48\times10^{-2}}$ | $\mathbf{+0.647}$ | $1.07\times$ | ✅ `descending` |
| `pendulum_fixed` (identity) | $\mathbf{2.55\times10^{-5}}$ | $\mathbf{0.0048}$ | $6.68\times10^{-2}$ | $+0.473$ | $94\times$ worse | ⚠️ `REGRESSED` |

`identity` wins on **fit** (3.6× better train MSE, 2.4× better $\theta$ RMSE) and loses
on **generalisation and stability** — its final-epoch loss is 94× its best, a late-training
spike (pre-clip gradient peak $1.95\times10^{5}$, clipping engaged on only 5.4% of steps).
`pendulum_control_win5` ends essentially where it peaked.

Same pattern as the probe, one level up: **the better fit is again the worse model.**

### More epochs made the pendulum worse

| identity + window 5.0 | Full MSE | $R^2$ |
| :--- | :---: | :---: |
| @ 2,000 epochs (`pend_id_win5`) | $\mathbf{4.32\times10^{-2}}$ | $\mathbf{+0.662}$ |
| @ 10,000 epochs (`pendulum_fixed`) | $6.68\times10^{-2}$ | $+0.473$ |

5× the compute produced a **55% worse** trajectory. The extra epochs bought training-set
fit at the cost of generalisation. The best pendulum result in this entire project is
still the 2,000-epoch probe run.

**Implication:** the 10,000-epoch budget is not automatically right for every system. It
was tuned on Lotka-Volterra.

---

## 🦠 SIR: not fixed

`sir_fixed` (`--conserve_sum 1.0 --loss_weighting std`) **diverged**:

| | |
| :--- | :--- |
| First non-finite gradient | epoch **3055** |
| Auto-aborted | epoch **3154** (100 consecutive skipped steps) |
| Best checkpoint | epoch 2877, train MSE $1.96\times10^{-2}$ |
| Final | train MSE $1.05\times10^{-1}$ — 5.3× worse than best |
| Mass $S{+}I{+}R$ | $1.0147$ |
| `I_drift` | $0.0000$ — **still frozen** |

Compared with the pre-fix 10,000-epoch run: `I_rmse` $0.0776 \to 0.1848$ and mass
$1.0044 \to 1.0147$ — **worse on both**. Partly confounded (3,154 epochs vs 10,000), but
there is no reading of this as an improvement.

### What we learned anyway

**1. The X1 guard did exactly its job.** Where the pre-fix run silently produced a NaN
`final_model.pt` and burned 1,299 wasted epochs, this run kept a valid best checkpoint,
stopped at epoch 3,154, and reported precisely what happened. **The failure is now
visible and cheap instead of silent and expensive.** That is the guard working as
designed, on a real failure, in production.

**2. The combination is less stable than either part.** At 2,000 epochs `sir_conserve`
and `sir_lossw` were both healthy, and `sir_lossw` was the best SIR run in the project
(train $7.31\times10^{-3}$). Combined at 10,000 they diverged by epoch 3,055. The
conservation penalty and the reweighting appear to interact badly — plausibly because
reweighting already inflates the I-compartment gradient and the penalty adds another
term on top.

**3. The frozen fixed point is untouched by any of this.** Every SIR run — pre-fix,
probe, and full — shows `I_drift ≈ 0` where the truth decays $0.0568 \to 0.0043$. This
is a task-design problem, not an optimisation one: the extrapolation window sits in the
endemic equilibrium where the true signal std is $0.002$–$0.017$ against $0.11$–$0.36$ in
training, so MSE supplies almost no gradient there.

### Recommended next step for SIR

Run **`--loss_weighting std` alone** at 10,000 epochs — the best single option measured,
and it avoids the interaction that broke the combination:

```powershell
python train.py --dataset sir --basis rbf --solver tsit5 --t_train_end 50.0 `
  --layers 3 16 3 --grid_len 8 --lr 0.003 --grad_clip 1.0 --loss_weighting std `
  --epochs 10000 --save_dir results/_fixed/sir_lossw_only
```

If it also diverges, drop the learning rate to $1\times10^{-3}$ before changing anything
structural. For the frozen fixed point specifically, the untested lever is a
**relative/log-scale loss on I**, not more optimisation tuning.

---

## ✅ Fix scorecard (final)

| ID | Fix | Verdict | Evidence |
| :-: | :--- | :--- | :--- |
| **X1** | Non-finite gradient guard + abort | ✅ **works, proved in production** | caught SIR's divergence at 3055, preserved a valid checkpoint, stopped 6,846 wasted epochs |
| **P4** | Longer training window | ✅ **the decisive pendulum fix** | $R^2$ $-1.318 \to +0.647$; works for **both** activations |
| **P1** | `--act identity` | ⚠️ **helps convergence speed, not final quality** | ~1,000× faster to fit at window 3.0; *worse* extrapolation and stability at window 5.0 |
| **X2** | `--loss_weighting std` | ⚠️ **system-dependent** | best single SIR option; worst pendulum option; unstable when combined with S2 |
| **S2** | `--conserve_sum` | ⚠️ **helps alone, unstable combined** | mass 13.5% → 0.86% error at 2,000 ep; diverged at 10,000 with X2 |
| **P3** | `--grid_lims` | ❌ **rejected** | worse than the untouched control |
| **X3/X4** | post-clip logging + `stability` block | ✅ **works** | made the SIR abort diagnosable in one command |

### Two hypotheses we published and then refuted

Recording both, because the refutations are as informative as the fixes:

1. **"Gradient clipping is the shared root cause of both failures"**
   ([`05`](./05_phase2_benchmark_analysis.md), both `reflection.md` files). Clipping was
   *necessary* for SIR and *irrelevant* for the pendulum, which failed identically with
   it.
2. **"SiLU cannot represent sign-changing dynamics"** ([`06`](./06_suggested_fixes.md)
   §C2). Refuted by `pendulum_control_win5`. Only the data-efficiency form survives.

A third, **tanh saturation**, was proposed in `pendulum/reflection.md` and rejected twice
— once by direct measurement (Lotka-Volterra is *more* saturated and converges) and once
by `pend_gridlims` finishing worse than the control.

---

## 🛠️ Methodological lessons

**1. Always run the control that could falsify your claim.** `pendulum_control_win5` cost
one extra parallel job and overturned the headline finding. Without it we would have
published a wrong architectural claim.

**2. One factor at a time.** The Phase-2 re-runs changed clipping, `grid_len` and `lr`
together and could attribute nothing. Every run here moved one variable.

**3. Aggregate MSE hid both failures.** $\omega$ dominates the pendulum loss while
$\theta$ fails; SIR's frozen fixed point yields a respectable MSE. Per-dimension
diagnostics (`analyze_fixes.py`) were necessary, not decorative.

**4. Choose a metric that survives a changed experiment.** `extrap_mse` is not comparable
across `--t_train_end` values. Ranking by `train_mse` would have selected `identity`
twice — the worse model both times.

**5. A better fit is not a better model.** This appeared three separate times:
`pend_identity` (1,256× better fit, worse trajectory), `pendulum_fixed` vs
`pendulum_control_win5`, and identity at 2,000 vs 10,000 epochs.

**6. More epochs is not automatically better.** The best pendulum result came from a
2,000-epoch run. The 10,000-epoch budget was tuned on Lotka-Volterra and does not
transfer.

**7. Verify pass criteria before trusting them.** The "spike ratio in the low tens" rule
was wrong and would have failed three healthy runs — it measures the *pre-clip* norm.

**8. `Wait-Process` is unsafe on long sweeps.** PID reuse killed the first 8-job probe
after every job had succeeded. Fixed with `.WaitForExit()`; full write-up in
[`07`](./07_fix_changelog.md).

---

## 📋 Where the project stands

**Resolved**

* ✅ Pendulum non-convergence — **fixed**, $R^2 = +0.647$, 20× better full-horizon MSE
* ✅ SIR silent NaN death — the X1 guard converts it into a visible, cheap, diagnosed abort
* ✅ Three hypotheses tested and closed: gradient clipping as shared cause (refuted),
  tanh saturation (rejected twice), SiLU representational limit (refuted)

**Open**

* ❌ **SIR does not converge.** Next: `--loss_weighting std` alone at 10,000 epochs
* ❌ **SIR frozen fixed point** — structural; needs a relative or log-scale loss on I
* ⚠️ **Pendulum epoch budget** — 2,000 beats 10,000; worth a short sweep to find the knee
* ⚠️ **Best pendulum config is the control**, so `run_phase2.ps1 -Only fixes` should be
  updated to `--act silu --t_train_end 5.0` before anyone re-runs it
* ❌ **Lorenz never run** — last outstanding Phase-2 deliverable
  (`.\run_phase2.ps1 -Only lorenz`, ~2.6 h)
* ❌ **Multi-seed replication** (Phase 4 Task 4.1) — every number here is $N=1$

**Phase 3 readiness**

* **Novelty 3** (stiffness phase map) — ✅ **unblocked.** Use `--t_train_end 5.0`; the
  activation choice is second-order, so keep the default SiLU
* **Novelty 1** (gradient-norm dynamics) — ✅ ready, **no new runs needed**; 40 completed
  runs carry per-epoch `grad_norms`, post-fix runs add `post_clip_grad_norms`
* **Novelties 4 & 5** — blocked on `pysindy` / `torchdiffeq` (not installed) and a
  missing $L_1$ edge-pruning step

**Documents that are now stale and must be corrected**

* [`05`](./05_phase2_benchmark_analysis.md) §Phase 2 Completion Status — still says
  gradient clipping is the shared root cause
* `pendulum/reflection.md` and `sir/reflection.md` — same
* [`06`](./06_suggested_fixes.md) §Part 2 hypothesis C2 — **refuted**; the strong claim
  must not be carried into the report

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
