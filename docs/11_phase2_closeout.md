# 🏁 Phase 2 Closeout: The Three Run-Free Deliverables

> **Branch:** `fix/phase1-revisit`
> **Status:** ✅ Tasks 2.3, 2.4 and 2.5 closed — **no new training runs**, total runtime under one minute
> **Script:** `implementation/phase2_closeout.py`
> **Chain:** [`09`](./09_stability_fix_results.md) pendulum · [`10`](./10_sir_root_cause_and_fix.md) SIR · **this doc: Phase 2 closeout**

---

## 🏁 Why these three could be closed for free

An audit against
[`PROJECT_IMPLEMENTATION_PLAN.md`](./04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md)
§Phase 2 found five outstanding items. Three of them needed **no training at all** —
only analysis and plotting over checkpoints and histories already on disk:

| Plan task | What the plan asked for | Why it was free |
| :--- | :--- | :--- |
| **2.3** | *"extrapolation … and beyond to $t=28.0$"* | Training ends at $t=3.5$; a longer horizon is pure integration of an already-trained field |
| **2.4** | *"verify … energy decay capturing"* | `compute_energy_violation` and `plot_pendulum_phase_and_energy` were built in Phase 1 and had **never been called** |
| **2.5** | *"multi-panel publication figures (phase portraits with streamlines, error vs $\Delta t$ log-log)"* | `results/figures/` did not exist; four Phase-1 plotting functions were dead code |

The other two — **Lorenz** and **$\Delta t = 0.01$** — need hours of compute and are
deliberately deferred (see §Deferred).

```powershell
cd implementation
python phase2_closeout.py --task all      # ~40 s, no training
```

Outputs land in `results/phase2_closeout/*.json` and `results/figures/*.png` (300 DPI).

---

## 📈 Task 2.3 — Extrapolation horizon doubled to $t = 28$

Training window unchanged at $[0, 3.5]$. The horizon is extended from $t=14$ (4 periods)
to $t=28$ (**8 periods**), so the window $(3.5, 14]$ stays directly comparable with every
number in [`05`](./05_phase2_benchmark_analysis.md) and $(14, 28]$ is entirely new.

| Model | Train $[0,3.5]$ | Extrap $(3.5,14]$ | **Far-extrap $(14,28]$** | $R^2$ far | **Degradation** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KAN-ODE (RBF, 240p)** | $8.83\times10^{-5}$ | $8.92\times10^{-5}$ | $\mathbf{9.72\times10^{-5}}$ | $\mathbf{1.0000}$ | $\mathbf{1.1\times}$ |
| MLP-ODE (SiLU, 252p) | $9.50\times10^{-5}$ | $4.78\times10^{-4}$ | $3.95\times10^{-3}$ | $0.9986$ | $8.3\times$ |
| MLP-ODE (tanh, paper) | $1.08\times10^{0}$ | $1.69\times10^{0}$ | $4.03\times10^{0}$ | $-0.394$ | $2.4\times$ |

*Degradation = far-extrapolation MSE ÷ near-extrapolation MSE. A value near 1 means error
does **not** compound as the horizon doubles.*

### 🔑 The new result: the KAN advantage widens with horizon

**KAN-ODE error is essentially flat across 8 periods** ($8.83 \to 8.92 \to 9.72
\times10^{-5}$, a 10% rise from training window to double horizon). The MLP degrades
$8.3\times$ over the same extension.

The consequence for Table 3:

| Horizon | KAN vs MLP-SiLU advantage |
| :--- | :---: |
| $(3.5, 14]$ — as published in [`05`](./05_phase2_benchmark_analysis.md) | $5.4\times$ |
| $(14, 28]$ — **this run** | $\mathbf{40.6\times}$ |

This is a materially stronger statement of the paper's central claim than the project had
before. It is not that the KAN is merely more accurate at a fixed horizon — its error
**does not accumulate**, so the gap widens with every additional period. Relative $L_2$
stays at $0.33\%$ out to $t=28$, against the MLP's $2.16\%$.

The paper's literal `[2,50,2]`+tanh baseline degrades from $R^2 = 0.42$ to
$R^2 = \mathbf{-0.394}$ — it becomes worse than predicting the mean.

📊 `results/figures/07_extrapolation_t28.png` · 📄 `results/phase2_closeout/extrap28.json`

---

## ⚡ Task 2.4 — Damped-pendulum energy dissipation

Run: `results/_fixed/pendulum_control_win5` (the config selected in
[`09`](./09_stability_fix_results.md)). For $\mu > 0$ the true system is **strictly
dissipative**:

$$\frac{dE}{dt} = -m L^2 \mu\, \omega^2 \le 0$$

This is a genuinely independent physical check — a model can fit the trajectory and still
get the energy wrong.

| | True | Predicted |
| :--- | :---: | :---: |
| $E_0$ | $13.892$ | $13.892$ |
| $E_{\text{final}}$ | $0.110$ | $0.801$ |
| Fraction dissipated | $99.2\%$ | $\mathbf{94.2\%}$ |
| Energy RMSE (train) | — | $8.19\times10^{-2}$ |
| Energy RMSE (extrapolation) | — | $3.99\times10^{-1}$ |
| **Steps where $E$ increases** | $\mathbf{0 / 200}$ | $\mathbf{40 / 200}$ |

### ✅ Verdict: captured in aggregate — ⚠️ but the invariant is violated

The model dissipates **94.2%** of the initial energy against a true 99.2%, and energy
RMSE inside the training window is $0.082$ J against $E_0 = 13.9$ J (0.6%). At that level
the gross dissipation physics is clearly learned.

**But the strict invariant fails.** The predicted energy **rises on 40 of 200 steps**,
where the true energy never rises once. That is physically impossible for a damped
pendulum, and `04_pendulum_energy.png` shows it plainly: after the $t=5$ split the
predicted energy curve plateaus around $0.8$ J and visibly steps upward several times
while the truth continues decaying toward zero.

This is the **same under-damping** identified in [`09`](./09_stability_fix_results.md):
the true θ decays to $0.004$ by $t=9.1$ while the model keeps oscillating at $0.237$. The
energy diagnostic is an independent confirmation of it, expressed as a violated
conservation law rather than an RMSE — a considerably stronger way to report the
limitation.

> **Reporting guidance.** State it as: *"the model captures 94.2% of the energy
> dissipation and tracks $E(t)$ to 0.6% inside the training window, but violates strict
> monotonic decay on 20% of extrapolation steps — it is under-damped."* Do **not** claim
> the energy physics is fully captured.

📊 `results/figures/04_pendulum_energy.png` · 📄 `results/phase2_closeout/energy.json`

---

## 🖼️ Task 2.5 — Publication figures (300 DPI)

`results/figures/` created; **6 figures** generated from existing artifacts.

| File | Content | Plan item |
| :--- | :--- | :--- |
| `01_phase_portrait_streamlines.png` | KAN's **learned vector field** as streamlines, with the true and predicted limit cycle overlaid | *"phase portraits with streamlines"* |
| `02_error_vs_stepsize_loglog.png` | Error vs $\Delta t$ on log-log axes with $O(\Delta t)$ / $O(\Delta t^2)$ reference slopes | *"error vs $\Delta t$ log-log curves"* |
| `03_kan_vs_mlp_convergence.png` | Semilog loss decay: KAN vs MLP-SiLU vs MLP-tanh | Table 3 support |
| `04_pendulum_energy.png` | 2-panel: inward spiral + energy dissipation | Task 2.4 |
| `05_solver_convergence.png` | Loss decay across all 6 integrators | Table 1 support |
| `06_basis_convergence.png` | Loss decay across all 7 bases | Table 2 support |
| `07_extrapolation_t28.png` | 3-panel trajectories over the doubled horizon | Task 2.3 |

**Figure 1** finally uses `plot_phase_portrait_with_streamlines`, built in Phase 1 and
never called. It is the most publication-ready artifact in the project: the streamlines
are the *model's own* $\mathbf{f}_\theta$, sampled on a $30\times30$ grid, showing the
learned spiral centre and the predicted orbit lying exactly on the truth.

**Figure 2** carries an explicit red caption recording that the sweep is **confounded** —
changing $\Delta t$ also changes the training-set size (19 / 36 / 71 samples), so the
non-monotonic minimum at $\Delta t = 0.1$ reflects both effects. Publishing the figure
without that caveat would misrepresent it.

### ❌ Not generated

`plot_3d_lorenz_trajectory` — still dead code, because **the Lorenz sweep has never been
run**. The script reports this explicitly rather than failing silently.

---

## ⏸️ Deferred (with reasons)

| Item | Plan task | Cost | Decision |
| :--- | :--- | :--- | :--- |
| **Lorenz sweep** | 2.5 | ~2.6 h + likely fix iterations | **Deferred.** The only genuinely missing *system*. Before running, apply [`10`](./10_sir_root_cause_and_fix.md)'s initialisation diagnostic: Lorenz has $\lvert f\rvert \sim 10^1$, so it may need `--time_scale` in the **opposite** direction to SIR. A 30-second check that could save the whole run. |
| **$\Delta t = 0.01$** | 2.1 | ~4.7 h | **Dropped, deliberately.** 10× the $\Delta t=0.1$ cost, and [`05`](./05_phase2_benchmark_analysis.md) already establishes the whole sweep is confounded. A fourth confounded point buys little. **Recommended replacement:** re-cast as a `substeps` sweep at fixed $\Delta t$, which isolates discretisation from data density. |

Both are recorded as **explicit, reasoned deferrals** rather than silent gaps.

---

## ✅ Phase 2 status after this closeout

| Task | Owner | Status |
| :--- | :--- | :--- |
| **2.1** Solver ablation → Table 1 | M3 | ✅ Complete ($\Delta t = 0.01$ deliberately dropped, reason recorded) |
| **2.2** Basis ablation, 7 bases → Table 2 | M4 | ✅ Complete |
| **2.3** Extrapolation horizon + KAN vs MLP | M3 | ✅ **Complete** — $t\to28$ closed here |
| **2.4** Noise sweep + pendulum baseline | M4 | ✅ **Complete** — energy decay verified here |
| **2.5** Multi-system + CSV + figures | M5 | ⚠️ SIR ✅, CSV ✅, figures ✅ — **Lorenz deferred** |

**Phase 2 is closed except for Lorenz.** Every deliverable that does not require the
Lorenz system is now done, and the two deferrals are documented with cost and rationale.

### Still outstanding, project-wide

* ❌ **Lorenz** — the one remaining Phase-2 run
* ❌ **Multi-seed replication** ($N=5$, Phase 4 Task 4.1) — every number in the project is
  $N=1$; no ordering claim in Table 1 is publishable without it
* ⚠️ **CI has never run on this branch** — `ci.yml` matches `feat/**`, not `fix/**`
* ⚠️ **Pendulum `SecondOrderField`** — [`09`](./09_stability_fix_results.md) §Recommended
  next step; $\dot\theta = \omega$ is exactly known and the model currently gets it 8.85%
  wrong

---

## 🔁 Reproducing

```powershell
cd d:\level4\Term1\NUM_project\kinetic-kan\implementation

python phase2_closeout.py --task all        # all three, ~40 s
python phase2_closeout.py --task extrap28   # Task 2.3 only
python phase2_closeout.py --task energy     # Task 2.4 only
python phase2_closeout.py --task figures    # Task 2.5 only

# point the energy check at a different pendulum run
python phase2_closeout.py --task energy --pendulum_run results/_fixed/pendulum_fixed
```

The script rebuilds each field exactly as `evaluate.py` does — honouring `grid_lims`,
`conserve_mode`, `vanish_dim` and `time_scale` — so it cannot fall into the silent
replay-mismatch class of bugs catalogued in [`10`](./10_sir_root_cause_and_fix.md) §Part 5.
It **reads only**: no existing result is modified or overwritten.
