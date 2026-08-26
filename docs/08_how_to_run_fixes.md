# ▶️ How to Run the Cross-Domain Stability Fixes

> **Companion docs:** [`06_suggested_fixes.md`](./06_suggested_fixes.md) (why) ·
> [`07_fix_changelog.md`](./07_fix_changelog.md) (what changed)
> **All commands are PowerShell, run from `implementation/`.**
> **Nothing here writes to `results/benchmarks/`** — the probe and fix runs use
> `results/_probe/` and `results/_fixed/`.

---

## 🧭 The workflow at a glance

```
Step 0  smoke test          ~2 min    verify the plumbing before committing hours
Step 1  probe (8 runs)     ~75 min    WHICH fix matters? one variable per run
Step 2  read the table      ~5 min    pick the winner
Step 3  fixes (2 runs)    ~110 min    full 10,000-epoch runs with the winner
Step 4  verify              ~5 min    confirm convergence + no blowup
```

Do **not** skip Step 2. The `fixes` target stacks fixes the probe has not yet
validated — if the probe disagrees, edit the two `Add-Job` lines in `run_phase2.ps1`
before running Step 3.

---

## ⚙️ Prerequisites

```powershell
cd d:\level4\Term1\NUM_project\kinetic-kan\implementation
```

* Nothing else heavy should be running — these saturate the CPU.
* No new packages needed. Verified working on Python 3.12.5 / torch 2.6.0+cpu / 12 logical cores.
* Confirm you are on the right branch:

```powershell
git branch --show-current      # expect: fix/phase1-revisit
git status --short             # expect: only train.py + run_phase2.ps1 modified
```

---

## 🚦 Step 0 — Smoke test (~2 min)

Verifies every new flag is wired before spending hours. `-DryRun` prints the plan and
launches nothing; the 20-epoch pass actually exercises the code.

```powershell
# 1. Print the plan without running anything
.\run_phase2.ps1 -Only probe -DryRun

# 2. Actually run all 8 jobs for 20 epochs into a throwaway directory
.\run_phase2.ps1 -Only probe -Epochs 20 -MaxParallel 2 -ProbeDir results/_smoke
```

✅ **Expected:** `All 8 runs produced metrics.json.` and a collated table.
❌ **If any run fails**, read `results/_smoke/_logs/<name>.err.log` and stop — do not
proceed to Step 1.

Then discard the smoke output:

```powershell
Remove-Item -Recurse -Force results/_smoke
```

---

## 🔬 Step 1 — The probe (~75 min)

Eight runs at 2,000 epochs. **Each changes exactly one thing** from the failing
baseline, so the summary table directly identifies the responsible fix.

```powershell
.\run_phase2.ps1 -Only probe -Epochs 2000 -MaxParallel 2
```

| Job | Changed from baseline | Question it answers |
| :--- | :--- | :--- |
| `pend_control` | *(nothing)* | does the failure reproduce at 2,000 epochs? |
| `pend_identity` | `--act identity` | can a linear residual branch learn $\dot\theta=\omega$? |
| `pend_tanhact` | `--act tanh` | does any odd-symmetric activation work? |
| `pend_lossw` | `--loss_weighting std` | is it the $\omega$-dominated loss? |
| `pend_gridlims` | `--grid_lims -3 3` | is it grid coverage after `tanh`? |
| `sir_control` | *(nothing)* | baseline for comparison |
| `sir_conserve` | `--conserve_sum 1.0` | does enforcing $S{+}I{+}R{=}1$ help? |
| `sir_lossw` | `--loss_weighting std` | does balancing the I compartment help? |

Output: `results/_probe/<job>/` plus `results/_probe/tables/summary_best.csv`.
Live progress: `results/_probe/_logs/<job>.err.log` (tqdm writes to stderr).

**Check progress at any time:**

```powershell
Get-ChildItem results/_probe/_logs/*.err.log | ForEach-Object {
    $last = ((Get-Content $_ -Raw) -split "`r" | Where-Object { $_.Trim() } | Select-Object -Last 1)
    "{0,-16} {1}" -f $_.BaseName.Replace('.err',''), $last.Trim()
}
```

Sample output (from the completed pre-fix runs — note SIR's `train=nan`, the failure
this work fixes):

```text
pendulum   Training: 100%|#| 10000/10000 [1:09:33<00:00, 2.40epoch/s, train=2.882e-01, ... best=2.865e-01]
sir        Training: 100%|#| 10000/10000 [1:47:43<00:00, 1.55epoch/s, train=nan, monitor=nan, best=4.389e-04]
```

---

## 📊 Step 2 — Read the results (~5 min)

Everything is in one command. `analyze_fixes.py` reports the quantities that actually
distinguish a converged run from a failed one — **aggregate MSE does not**, which is
precisely why both failures were missed the first time round.

```powershell
python analyze_fixes.py --root results/_probe
```

Add `--only summary|pendulum|sir|flatness` to isolate one section. The standard
CSV collation still works too:

```powershell
python collate_results.py --root results/_probe --out results/_probe/tables --bucket best
```

### What the four sections tell you

**`summary`** — one row per run: train/extrap MSE, extrapolation $R^2$,
**`final_train`** (a `NaN` here means the run died), **`spike`** (max/median pre-clip
gradient norm) and **`nonfin`** (steps the X1 guard dropped).

**`pendulum`** — the decisive section. Reports $\theta$ and $\omega$ RMSE separately
plus `theta_min`, and prints a verdict.

> ✅ **Success:** `theta_rmse` well below the control's **0.7555**, and `theta_min`
> approaching the true **−1.385** instead of stalling at **−0.211**. The verdict column
> says `IMPROVED` when both hold, `still failing` otherwise.

**`sir`** — per-compartment RMSE, `mass_min`/`mass_max` (the $S{+}I{+}R{=}1$ invariant),
and `I_drift`.

> ✅ **Success:** `mass` closes on **1.0000** (was 1.0044) and `I_rmse` drops below
> **0.0776**.
> ⚠️ **The trap:** a near-zero `I_drift` is **not** stability — it means the model is
> still frozen on a spurious fixed point. True drift over that window is **0.0525**;
> the failed run produced **0.0003**. The verdict column flags this as `STILL FROZEN`.

**`flatness`** — loss at 20% / 80% / 100% of the run and the 80:100 ratio.

> ✅ `descending` (ratio > 1.05) is what you want.
> ❌ `FLAT` (≈1.00) means it stopped improving — the pendulum's exact failure (1.003).
> ❌ `REGRESSED` (< 0.95) means the final loss is worse than at 80%.
> ❌ `NaN/diverged` means the run died.

*(Sanity check: run it against `results/benchmarks` and it correctly reports the
pendulum as `FLAT` / `still failing`, SIR as `NaN/diverged` / `STILL FROZEN`, and
independently rediscovers the `dt0.2` late-training regression documented in
[`05`](./05_phase2_benchmark_analysis.md) §Table 4.)*

---

## 🏁 Step 3 — Full runs (~110 min)

**Only after Step 2.** If the probe favoured something other than the defaults, edit
the `pendulum_fixed` / `sir_fixed` `Add-Job` lines in `run_phase2.ps1` first (they are
under the `[FIX-2026-08] fixes` block).

```powershell
.\run_phase2.ps1 -Only fixes -Epochs 10000 -MaxParallel 2
```

As shipped this runs:

```
pendulum_fixed : --act identity --loss_weighting std --grid_len 8 --lr 0.003 --grad_clip 1.0
sir_fixed      : --conserve_sum 1.0 --loss_weighting std --t_train_end 50 --layers 3 16 3
                 --grid_len 8 --lr 0.003 --grad_clip 1.0
```

Output: `results/_fixed/` + `results/_fixed/tables/`.

> ⏱️ SIR is the long pole (~108 min at 10,000 epochs); the pendulum finishes around
> 70 min. Budget ~2 hours wall-clock.

---

## ✅ Step 4 — Verify (~5 min)

```powershell
python analyze_fixes.py --root results/_fixed
```

### Pass / fail criteria

| Check | Pass | Reference (the failed runs) |
| :--- | :--- | :--- |
| `final train_mse` is a number, not `NaN` | ✅ required | SIR previously **NaN** |
| `nonfinite_grad_steps` | `0` ideally; small is fine | SIR previously 1,303 |
| `aborted_at_epoch` | `null` | — |
| `spike_ratio` | **low tens** | pendulum 7186× · SIR 2.47e6× |
| pendulum `train_mse` | $\ll 2.87\times10^{-1}$ | old best $2.87\times10^{-1}$ |
| pendulum `extrap_r2` | $> 0$ | old $-1.318$ |
| SIR `train_mse` | $\lesssim 4.4\times10^{-4}$ | old best $4.39\times10^{-4}$ |
| SIR extrap **RMSE** | $< 6.1\times10^{-2}$ | old $6.14\times10^{-2}$ |

> 📐 **Report RMSE, not $R^2$, for the SIR extrapolation window.** Its true signal has
> std $0.002$–$0.017$ versus $0.11$–$0.36$ in training, so $R^2$'s denominator nearly
> vanishes and the value is not interpretable. See `06_suggested_fixes.md` §Part 1.

### Also confirm the loss is not merely flat

A low final loss that stopped descending is still a failed fit — exactly how the
pendulum failed twice. The `flatness` section covers this; the verdict must read
`descending`, not `FLAT`:

```powershell
python analyze_fixes.py --root results/_fixed --only flatness
```

---

## 🧯 Troubleshooting

| Symptom | Cause / action |
| :--- | :--- |
| `RUN(S) FAILED -- no metrics.json` | Read `results/_probe/_logs/<name>.err.log`. Re-run just the failures with `-SkipExisting`. |
| Dry run shows `--lr 0.002` **and** `--lr 0.003` | Expected. `argparse` takes the last, so **0.003 wins**. Pre-existing launcher behaviour, confirmed in `metrics.json`. |
| `[FIX/X1] ABORTING at epoch N` | The run diverged unrecoverably. `best_model.pt` is still valid. Lower `--lr` or raise `--grad_clip`. |
| Everything is very slow | Check `-MaxParallel`; on 12 cores use 2. Timings under parallel execution are **not** reportable — use `-Serial` for that. |
| Want to discard a probe and restart | `Remove-Item -Recurse -Force results/_probe` |
| Need to reproduce an old Phase-2 number | Omit every new flag — defaults are bit-identical to the pre-fix code (verified). |

---

## 📋 Copy-paste block

```powershell
cd d:\level4\Term1\NUM_project\kinetic-kan\implementation

# Step 0 - smoke (~2 min)
.\run_phase2.ps1 -Only probe -Epochs 20 -MaxParallel 2 -ProbeDir results/_smoke
Remove-Item -Recurse -Force results/_smoke

# Step 1 - probe (~75 min)
.\run_phase2.ps1 -Only probe -Epochs 2000 -MaxParallel 2

# Step 2 - read, then DECIDE before continuing
python analyze_fixes.py --root results/_probe

# Step 3 - full runs (~110 min).  Edit the fixes Add-Job lines first if the
#          probe disagreed with the shipped defaults.
.\run_phase2.ps1 -Only fixes -Epochs 10000 -MaxParallel 2

# Step 4 - verify
python analyze_fixes.py --root results/_fixed
```

---

## 📌 After the runs

1. **Update the stale docs.** [`05_phase2_benchmark_analysis.md`](./05_phase2_benchmark_analysis.md)
   §Phase 2 Completion Status and both `reflection.md` files still say gradient
   clipping is the shared root cause. The clipped re-runs disproved that for the
   pendulum.
2. **Decide what to commit.** `results/_probe/` is diagnostic; `results/_fixed/` is the
   reportable artifact. Checkpoints under `results/**` are un-ignored by
   `.gitignore`, so they will be tracked if added.
3. **Lorenz is still never run** — the last outstanding Phase-2 deliverable
   (`.\run_phase2.ps1 -Only lorenz`, ~2.6 h). Expect it to need these same fixes.
4. **Novelty 3** (stiffness phase map) unblocks only once the pendulum converges.
5. **Novelty 1** (gradient-norm dynamics) needs **no new runs at all** — every
   completed run already carries per-epoch `grad_norms`, and now `post_clip_grad_norms`
   too. Cheapest available Phase-3 win; can start in parallel with everything above.
