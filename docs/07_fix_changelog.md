# 🧾 Change Log: Cross-Domain Stability Fixes (`FIX-2026-08`)

> **Branch:** `fix/phase1-revisit`
> **Scope:** implements the proposals in [`06_suggested_fixes.md`](./06_suggested_fixes.md)
> **Files modified:** `implementation/train.py`, `implementation/run_phase2.ps1`
> **File added:** `implementation/analyze_fixes.py` — **nothing else touched**
> **How to run the result:** [`08_how_to_run_fixes.md`](./08_how_to_run_fixes.md)

---

## 🔒 The safety guarantee

**Every new option defaults to the exact pre-fix behaviour.** Running any existing
command with no new flags produces bit-identical output to before.

This was verified, not assumed. A 60-epoch Lotka-Volterra run with default arguments
was executed against the pre-fix `train.py` (extracted from git at `aece84b`) and
against the patched one, then compared field by field:

| Compared | Result |
| :--- | :--- |
| All 11 `best` metrics (`train_mse`, `extrap_mse`, `full_mse`, RMSE, MAE, $R^2$, rel-$L_2$ …) | ✅ **bit-identical** |
| All 11 `final` metrics | ✅ **bit-identical** |
| `train_losses[]` — all 60 epochs | ✅ **bit-identical** |
| `test_losses[]` — all 60 epochs | ✅ **bit-identical** |
| `grad_norms[]` — all 60 epochs | ✅ **bit-identical** |
| `selection.best_epoch` | ✅ identical |
| `estimated_lipschitz_bound` | ✅ identical |

So the **26 committed Phase-2 runs remain reproducible**, and the five-way determinism
check (`train_mse = 8.832586172502488e-05`) documented in
[`05`](./05_phase2_benchmark_analysis.md) still holds.

**Every edit is tagged `[FIX-2026-08]`** and carries a sub-tag (`X1`, `X2`, `X3`, `X4`,
`S2`, `P3`) matching the IDs in `06_suggested_fixes.md`. To review the whole change set:

```powershell
git diff -- implementation/train.py implementation/run_phase2.ps1
grep -n "FIX-2026-08" implementation/train.py implementation/run_phase2.ps1
```

There are **22** tagged comment blocks in `train.py` and **7** in `run_phase2.ps1`.

---

## 📦 What changed in `implementation/train.py`

### [X1] Non-finite gradient guard — *always on*

**Problem.** `clip_grad_norm_` computes `clip_coef = max_norm / (total_norm + 1e-6)`.
Once `total_norm` overflows float32 to `inf`, `clip_coef → 0` and the following
`grad.mul_(clip_coef)` evaluates `inf × 0 = NaN`, permanently poisoning every weight.
Clipping bounds the optimizer *step*; it cannot undo an overflow in the
forward/backward pass. The 10,000-epoch SIR run died exactly this way — gradient norm
$5.82\times10^{18}$, non-finite from epoch 8686, loss NaN from 8702, losing its last
1,299 epochs and `final_model.pt`.

**Change.** Before clipping and stepping, the step is now checked:

```python
step_is_finite = math.isfinite(gnorm) and math.isfinite(train_loss_val)
if step_is_finite:
    ...clip...; optimizer.step()
else:
    nonfinite_grad_steps += 1
    optimizer.zero_grad(set_to_none=True)   # drop the step, keep last good weights
```

**Why it is safe.** On a healthy run `step_is_finite` is always `True`, so the code
path is byte-for-byte the old one. Both branches append to every per-epoch array, so
`train_losses`, `test_losses`, `grad_norms` and `post_clip_grad_norms` stay the same
length and index-aligned.

**Plus an early abort.** If the *parameters themselves* are already non-finite, every
later forward pass returns NaN and the guard can only keep skipping — the run is
unrecoverable and just burns compute (SIR spent 1,299 epochs like this). After
`NONFINITE_ABORT_STREAK = 100` consecutive skipped steps, training stops early. The
`break` sits at the very **end** of the loop body so all arrays are appended to first.

> ⚠️ **Known limitation, stated honestly.** The guard fires on a non-finite
> *gradient*. If a *finite but enormous* gradient produces a step that overflows the
> weights, the guard is one epoch too late and the abort path takes over instead. For
> SIR's actual failure this is fine — the gradient went non-finite at 8686, **16
> epochs before** the loss did, so the weights were still good when the guard would
> have fired.

### [X2] Per-dimension loss weighting — `--loss_weighting` (default `none`)

**Problem.** `F.mse_loss` averages squared error equally over all state dimensions, so
the widest-spread dimension dominates the gradient. Measured training-window std
imbalance versus outcome:

| System | Per-dimension std | Imbalance | Outcome |
| :--- | :--- | :---: | :---: |
| Lotka-Volterra | $x{=}1.986$, $y{=}1.367$ | $1.45\times$ | ✅ converges |
| Damped pendulum | $\theta{=}1.019$, $\omega{=}2.556$ | $2.51\times$ | ❌ fails |
| SIR ($t_{\text{train}}{=}50$) | $S{=}0.360$, $I{=}0.113$, $R{=}0.333$ | $3.19\times$ | ❌ fails |

On the pendulum this is directly visible: $\omega$ reaches RMSE $0.048$ while
$\theta$ — whose derivative *is* $\omega$ — only reaches $0.756$.

**Change.** With `--loss_weighting std`, each dimension is weighted by $1/\sigma_d^2$
and the weights are renormalised to mean $1.0$ so the loss keeps the same order of
magnitude. Actual weights emitted at startup:

```
pendulum : std [1.0186, 2.5555]         -> weights [1.7258, 0.2742]
sir      : std [0.3603, 0.1131, 0.3333] -> weights [0.2436, 2.4719, 0.2846]
```

θ now carries $6.3\times$ the weight of ω; I carries $\approx10\times$ that of S.

> **Critical design decision.** `mse_train` (plain, unweighted) is still what gets
> logged, plotted, and used for checkpoint selection, and every number in
> `metrics.json` is recomputed post-hoc by `score()` from the integrated trajectory.
> This flag changes **what is optimised, never how results are measured** — so runs
> using it stay directly comparable with the existing 26.

### [S2] Conservation-law penalty — `--conserve_sum` (default off)

**Problem.** SIR carries the exact invariant $S+I+R=1$, which nothing enforced. The
failed run drifted to $\sum u = 1.0045$ in extrapolation and $0.949$–$1.029$ during
training.

**Change.** When `--conserve_sum V` is given, adds
$\lambda \cdot \overline{(\sum_i u_i - V)^2}$ to the optimised loss
(`--conserve_weight`, default 1.0). Disabled by default, and meaningless for systems
without such an invariant (Lotka-Volterra, pendulum) — so leave it off there.

### [P3] `grid_lims` and `normalizer` exposed

`grid_lims` was never plumbed through `train_kan_ode` at all — `KAN` silently defaulted
it to $(-1,1)$. `normalizer` was a function keyword but had no CLI flag. Both are now
reachable as `--grid_lims LO HI` and `--normalizer {tanh,sigmoid,identity}`, with
defaults `-1.0 1.0` and `tanh` — i.e. unchanged.

### [X3] Post-clip gradient logging

`grad_norms[]` records the **pre**-clip norm, so on its own it cannot show whether
clipping engaged. `training_history.json` now also carries `post_clip_grad_norms[]`,
same length and indexing. `grad_norms[i] > post_clip_grad_norms[i]` means clipping
bound the step at epoch `i+1`; a `0.0` entry means the X1 guard dropped it.

### [X4] Stability summary in `metrics.json`

New `stability` block, so a blowup is readable straight off `metrics.json` instead of
requiring the forensic history analysis that found the original 7186× and 2.47e6×
spikes:

```json
"stability": {
    "grad_clip": 1.0,
    "nonfinite_grad_steps": 0,
    "first_nonfinite_epoch": null,
    "aborted_at_epoch": null,
    "epochs_run": 10000,
    "grad_norm_median_preclip": 2.707,
    "grad_norm_max_preclip": 53.08,
    "grad_norm_spike_ratio": 19.61,
    "grad_norm_max_epoch": 1,
    "clip_engaged_fraction": 0.967
}
```

`grad_norm_spike_ratio` is the statistic that diagnosed both Phase-2 failures. **A
healthy run sits in the low tens.** `clip_engaged_fraction` shows how often clipping
actually bound the step.

Also: `seconds_per_epoch` now divides by epochs **actually run** rather than the
requested budget, so an aborted run does not report a misleadingly small per-epoch
cost. Identical whenever nothing aborts.

---

## 📦 What changed in `implementation/run_phase2.ps1`

Two new `-Only` targets, **neither included in `all`**, both writing to their own
output roots so `results/benchmarks/` can never be overwritten by a diagnostic run:

| Target | Jobs | Output root |
| :--- | :---: | :--- |
| `probe` | 8 short one-factor-at-a-time diagnostics | `results/_probe/` |
| `fixes` | 2 full-length runs with fixes stacked | `results/_fixed/` |

New parameters `-ProbeDir` / `-FixedDir` hold those paths. An `$ActiveRoot` variable
routes **logs and CSV collation** to the matching root too, so
`results/benchmarks/_logs/` and the committed `results/tables/*.csv` are never touched.

**The `probe` set is deliberately OFAT** — each job changes exactly *one* thing from
the failing baseline:

| Job | Changed from baseline | Tests |
| :--- | :--- | :--- |
| `pend_control` | *(nothing — the control)* | reproduces the failure |
| `pend_identity` | `--act identity` | hypothesis C2 |
| `pend_tanhact` | `--act tanh` | hypothesis C2 |
| `pend_lossw` | `--loss_weighting std` | hypothesis C1 |
| `pend_gridlims` | `--grid_lims -3 3` | fix P3 |
| `sir_control` | *(nothing — the control)* | reproduces the failure |
| `sir_conserve` | `--conserve_sum 1.0` | fix S2 |
| `sir_lossw` | `--loss_weighting std` | fix X2 |

This is the discipline the last round lacked: those re-runs changed clipping,
`grid_len` and `lr` simultaneously, which is why "did clipping help?" needed forensic
log analysis instead of being readable off the table.

**Unchanged:** `solvers`, `activations`, `models`, `bspline`, `mlpfix`, `systems`,
`lorenz`, `stepsize`, `noise` and `all` all behave exactly as before. Verified by dry
run — `-Only systems` still targets `results/benchmarks/`.

---

## 📦 New file: `implementation/analyze_fixes.py`

A read-only diagnostic reader for any results tree. It exists because **aggregate MSE
hides both failures** — which is exactly why they were misdiagnosed the first time.

```powershell
python analyze_fixes.py --root results/_probe
python analyze_fixes.py --root results/_fixed --only flatness
```

| Section | Reports |
| :--- | :--- |
| `summary` | per-run losses, extrap $R^2$, `final_train` (NaN ⇒ died), gradient spike ratio, dropped steps |
| `pendulum` | $\theta$ vs $\omega$ RMSE separately, `theta_min`, verdict |
| `sir` | per-compartment RMSE, $S{+}I{+}R$ mass conservation, `I_drift`, frozen-fixed-point detection |
| `flatness` | loss at 20/80/100%, verdict: `descending` / `FLAT` / `REGRESSED` / `NaN/diverged` |

It reads nothing but `metrics.json`, `training_history.json` and `best_model.pt`, and
tolerates the old schema (pre-fix runs simply show `-` in the new columns).

**Validated against the known-bad runs**: it correctly reports the pendulum as
`FLAT` / `still failing` (θ RMSE 0.7555, `theta_min` −0.211), SIR as
`NaN/diverged` / `STILL FROZEN` (I RMSE 0.0776, mass 1.0044, drift 0.0003 against a
true 0.0525) — and independently rediscovered the `dt0.2` late-training regression
already documented in [`05`](./05_phase2_benchmark_analysis.md) §Table 4.

---

## ✅ Verification performed

| Test | Result |
| :--- | :--- |
| `train.py` parses; `--help` lists all 5 new flags | ✅ |
| **Regression:** 60-epoch LV defaults vs. pre-fix code | ✅ **bit-identical** (22 metrics + 3 arrays) |
| `--loss_weighting std` on pendulum and SIR | ✅ weights computed and printed correctly |
| `--conserve_sum 1.0` on SIR | ✅ penalty activates, run completes |
| `--act identity` / `--act tanh` on pendulum | ✅ both train |
| `--grid_lims -3 3` + `--normalizer` | ✅ accepted, run completes |
| `--model mlp` still builds `[2,50,2]` tanh, 252 params | ✅ unregressed |
| **NaN guard:** forced divergence (`--lr 50 --grad_clip 0`) | ✅ fires at epoch 2, run survives with **valid** best metrics instead of NaN |
| **Early abort:** same, 5000-epoch budget | ✅ stops at epoch 101; all 4 history arrays length 101; best metrics finite |
| `stability` block populated in `metrics.json` | ✅ |
| `-Only probe` / `-Only fixes` dry runs | ✅ 8 and 2 jobs, correct roots |
| `-Only systems` dry run | ✅ still `results/benchmarks/`, args unchanged |

---

## ⚠️ Things to be aware of

1. **The `fixes` target stacks fixes that the `probe` has not yet validated.** It
   assumes `--act identity` + `--loss_weighting std` for the pendulum and
   `--conserve_sum` + `--loss_weighting std` for SIR. **Read the probe results first**
   and edit those two `Add-Job` lines if the probe disagrees. Do not run `fixes` blind.
2. **Duplicate `--lr` on the command line.** The launcher emits `--lr 0.002` from the
   global parameter and then `--lr 0.003` from the job-specific args. `argparse` takes
   the last, so **0.003 wins**. This is pre-existing behaviour inherited from the
   `systems` target and is confirmed by `metrics.json` (`"lr": 0.003`) — noted here so
   nobody reads the dry-run output and panics.
3. **Nothing was changed in `kan/`, `ode/`, `data/`, `utils/`, `tests/`, or any
   results file.** The fixes are entirely in the training loop and the launcher.
4. **Existing `metrics.json` files have no `stability` block.** `collate_results.py`
   reads it with `.get()` semantics on the fields it needs, so old runs still collate —
   they simply have no stability columns.
5. **`results/_probe/` and `results/_fixed/` are new directories.** Their `*.pt` files
   are covered by the existing `!implementation/results/**/best_model.pt` un-ignore
   rule, so checkpoints there will be tracked if added. Decide deliberately whether to
   commit probe artifacts or keep them local.

---

*Next: [`08_how_to_run_fixes.md`](./08_how_to_run_fixes.md) for the exact commands and
how to read the output.*
