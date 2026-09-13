# Track B — Gradient Norm Dynamics vs. Solver Order

**Owner:** Abhishek Roy (2105033) · **Branch:** `feat/p3-gradient-dynamics`
**Write-up:** [`docs/14_p3_gradient_dynamics_findings.md`](../../../docs/14_p3_gradient_dynamics_findings.md)

## Research question

Do lower-order solvers (Euler, Heun) inject noisier gradients into backprop than
higher-order ones (RK4, Tsit5)? And if so, is that a plausible *mechanism* for the
Phase 2 finding that Euler is the only solver failing to reach $10^{-4}$ training loss?

## Method

Zero new training. Every number comes from the per-epoch `grad_norms[]` array already
logged in the six solver-ablation runs under
`results/benchmarks/ablation_solvers/solver_*/training_history.json` (10,000 epochs
each, accessed read-only).

**Noise measure.** Gradient norms span orders of magnitude across solvers, so a raw
standard deviation would rank solvers by absolute gradient scale rather than by
roughness. We measure roughness in log space instead:

$$\eta = \mathrm{std}\big(\Delta \log_{10} g\big)$$

the standard deviation of epoch-to-epoch *relative* change. A smoothly decaying curve
scores near zero regardless of magnitude; a curve that jitters between epochs scores
high. Scale-invariant, and needs no arbitrary window size.

**Correlation.** Spearman rank correlation against solver order $p$, not Pearson —
order is an ordinal scale with four distinct values and a tie at $p=2$, and the
hypothesis under test is monotonicity, not linearity.

**Robustness.** Because $n=6$ is small, the script re-runs the correlation at four
warm-up cuts (0 / 500 / 2,000 / 5,000 epochs dropped) and reports whether the sign is
stable and whether any cut reaches significance. This is what turns a single number
into an honest result.

## How to run

```powershell
python experiments/gradient_dynamics/analyze_gradient_dynamics.py
python experiments/gradient_dynamics/analyze_gradient_dynamics.py --warmup 2000   # optional
```

Writes `results/phase3/gradient_dynamics/table.json` and
`gradient_noise_by_order.png`. No GPU, no new dependencies (`numpy`, `scipy`,
`matplotlib` only), runs in a few seconds.

Tests: `python -m pytest tests/test_p3_gradient_dynamics.py -q` (14 tests, <3 s).

## Findings — summary

**The hypothesis is not supported.** Euler *is* nominally the noisiest solver
($\eta = 0.2805$) and does have the worst training loss, but:

- the spread across all six solvers is tiny (0.2621–0.2805, a 7% range);
- the ranking is not monotone in order — RK4 ($p{=}4$) is second-noisiest, ahead of
  both RK2 variants;
- $\rho = -0.088$, $p = 0.87$ — no correlation with solver order;
- the sign **flips** across warm-up cuts ($-0.088 \to -0.441 \to +0.706$) and no cut is
  significant.

The figure shows why: all six gradient-norm traces are dominated by large, sustained
spike bursts that appear in Euler and Tsit5 alike. Whatever drives that structure is
shared across solvers — it is not solver truncation error, and it swamps any
order-dependent signal the measure might otherwise resolve.

Track B therefore reports a **negative result**: at this sample size and on this
system, gradient-norm roughness does not explain Euler's convergence failure. See the
write-up for the full argument and its caveats.

## Environment

No new packages installed; `requirements.txt` untouched (per roadmap §2.4).
