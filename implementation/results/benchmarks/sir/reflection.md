# Reflection: SIR Epidemic Benchmark

**Run:** `results/benchmarks/sir/` — `layers=[3,16,3]`, RBF, $G=8$, Tsit5,
`substeps=2`, lr $=3\times10^{-3}$, 10,000 epochs, seed 42, `t_train_end=50` (extended
from the original 30, per the earlier diagnosis in `docs/05`), `git_sha=f150626`.

**Result:** did not converge. `best.train_mse = 0.0566` at epoch 3400;
`final.train_mse = 0.1184` at epoch 10000 — **the run ends roughly 2× worse than its
own best checkpoint**, saved separately as `best_model.pt`. `extrap_r2 = -607.6`
(best) / `-392.6` (final). See `training_history.json`, `loss_curves.png`,
`gradient_norm_dynamics.png`.

This run followed a 2+ hour, 14-config diagnostic sweep (`temp/sir_*`,
`temp/sir_champ_*`) at 100 epochs each, used to select `[3,16,3]` + $G{=}8$ + lr
$3\times10^{-3}$. **That sweep could not have caught this failure** — see below.

## What happened: a single gradient-norm explosion, not a bad hyperparameter choice

| Epoch | Loss | Grad norm |
| :--- | :--- | :--- |
| 5001 | 0.0735 | 0.26 |
| 5011 | 0.0706 | 0.45 |
| **5018** | **12,226** | **1,229,894** |
| 5021 | 87.6 | 6,817 |
| 5501 | 0.818 | 1.17 |
| 10000 | 0.1185 | — |

At epoch 5018 the gradient norm reaches **2.47 million times** the run's median
(0.498). Loss jumps from 0.07 to 12,226 in one optimizer step. The next ~5000 epochs
are spent crawling back down — recovering only to 0.118, not the pre-spike 0.057.
The best-checkpoint selection mechanism (which snapshots on training loss, before the
optimizer step, per the Phase-1 aliasing fix) preserved the epoch-3400 weights in
`best_model.pt`, so that artifact is usable; `final_model.pt` is not.

This is the same failure signature documented in `../pendulum/reflection.md` (grad
spike 7186× median at epoch 4963, also unrecovered) and present in smaller form in the
Lotka-Volterra flagship run (50.2, 1422× median, at epoch 1016 — survives because it
is ~15× smaller and happens early enough to leave 9000 recovery epochs). `train.py`
has no gradient clipping. **Root cause is almost certainly the same across all three:
no gradient clipping, combined with occasional stiff steps through the learned vector
field during unrolled ODE-solver backprop.**

**Why the diagnostic sweep didn't warn about this:** all 14 sweep configs ran for 100
epochs. The blowup in the full run happens at epoch 5018 — invisible to any 100-epoch
smoke test. The sweep correctly ranks early-training fit quality; it cannot predict
mid-training stability. Future architecture searches need either full-length runs or
explicit spike monitoring (e.g. abort/flag if `grad_norm` exceeds some multiple of its
running median), not just short smoke tests.

## Second, separate issue: is the extended training window (`t_train_end=50`) even sufficient?

The original SIR diagnosis (see `docs/05_phase2_benchmark_analysis.md`, Table 3b) found
73% of extrapolation states outside the training box at `t_train_end=30`, and
recommended extending the window. Re-checking coverage at `t_train_end=50` (this run):

| Compartment | Train range | Extrap range (t=50–80) | Extrap variance | Train variance |
| :--- | :--- | :--- | :--- | :--- |
| S | $[0.043, 0.990]$ | $[0.034, 0.042]$ | $5.0\times10^{-6}$ | $0.129$ |
| I | $[0.010, 0.359]$ | $[0.004, 0.057]$ | $2.2\times10^{-4}$ | $0.013$ |
| R | $[0.000, 0.898]$ | $[0.901, 0.961]$ | $2.9\times10^{-4}$ | $0.110$ |

Two findings, and they cut in different directions:

1. **The severe coverage gap is gone.** S and R now only miss the training box by a
   tiny margin (R exceeds the training max by 0.003) — the epidemic is asymptotically
   settling into its endemic equilibrium just past the training cutoff, not jumping to
   an unseen region. This is a much milder gap than the original 73%/huge-margin
   finding.
2. **But the extrapolation window's true signal is now almost flat** — variance
   4–5 orders of magnitude below the training window's. This makes $R^2$ structurally
   unreliable there: its denominator collapses, so even a small absolute error
   produces a large negative $R^2$. $R^2 = -607$ should not be read as "607× worse
   than a naive mean predictor" in any intuitive sense — MSE/RMSE are the metrics to
   trust for this window, not $R^2$.

However: the observed `extrap_rmse = 0.323` is *larger than the entire dynamic range*
of the true signal in that window (~0.01–0.06 per compartment). That is not explainable
by the R²-metric artifact alone — it means the extrapolated trajectory is genuinely
wrong in absolute terms, consistent with running on blowup-damaged weights.

**These two problems are confounded in this run and cannot be separated from it.**
Whether `t_train_end=50` actually fixes SIR extrapolation is still an open question —
this run cannot answer it, because the gradient blowup corrupted the model before the
window-extension fix could be fairly evaluated.

## What needs to happen (deferred to Phase 3 — not run today, no time remaining)

1. **Add gradient clipping to `train.py`** (shared fix with pendulum — see its
   `reflection.md`). Do this once, for both systems.
2. **Re-run SIR after the fix, same config** (`t_train_end=50`, `[3,16,3]`, $G{=}8$,
   lr $3\times10^{-3}$), to isolate whether clipping alone resolves the blowup.
3. **Only then re-evaluate extrapolation quality** using RMSE/MAE primarily, with $R^2$
   reported but explicitly caveated for this window's near-zero variance.
4. **If extrapolation is still poor after a stable run**, treat it as a genuine
   task-design finding (endemic-equilibrium windows are inherently low-signal for
   MSE-based training) rather than an optimization bug, and consider a per-compartment
   normalized loss.

## Suggested command once clipping is added

```
python train.py --dataset sir --t_train_end 50.0 --layers 3 16 3 --grid_len 8 \
  --lr 0.003 --epochs 10000 --save_dir results/benchmarks/sir
```

Same config as this run — the point is to isolate whether clipping alone fixes the
blowup before changing anything else (window, architecture, or loss weighting).
