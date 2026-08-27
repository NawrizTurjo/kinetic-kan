# 🔧 Suggested Fixes: Cross-Domain Failures & Open Diagnostic Threads

> **Project:** KINETIC-KAN (Solver-Aware Neural Dynamics) — BUET CSE 402, Group 05
> **Branch:** `fix/phase1-revisit`
> **Written after:** the clipped-gradient re-runs of `pendulum` and `sir` (`run_phase2.ps1 -Only systems -MaxParallel 2`)
> **Status:** proposals only — **none of the changes below are implemented in the codebase.**

---

## 📌 Why this document exists

Phase 2 closed with two systems failing: the damped pendulum and SIR. Both failures
were root-caused in
[`pendulum/reflection.md`](../implementation/results/benchmarks/pendulum/reflection.md)
and [`sir/reflection.md`](../implementation/results/benchmarks/sir/reflection.md) to a
**single shared mechanism** — a mid-training gradient-norm explosion that `train.py`
had no clipping to defend against. Gradient clipping was added, both systems were
re-run, and this document reports what that revealed.

**Headline: clipping worked exactly as designed, and it was not sufficient for either
system.** The shared-root-cause conclusion was half right. One system (SIR) improved
enormously and then failed a *different* way; the other (pendulum) did not improve at
all, which means its real cause was never the gradient explosion.

Everything below is evidence first, then diagnosis, then a prioritised fix list with
runnable commands.

---

## 📊 Part 0: What we believed vs. what the runs now show

The team carried four working hypotheses into this round. Two are now refuted, one is
confirmed, one is untested. Recording them explicitly so nobody re-derives a dead end.

| # | Hypothesis (as held by the team) | Verdict now | Evidence |
| :-: | :--- | :--- | :--- |
| **H1** | The paper's `[2,50,2]`+tanh MLP fails only because **tanh needs far more epochs** — the paper used 100,000, we run 10,000 | ❌ **Refuted** | Four controlled 2,000-epoch runs in [`05`](./05_phase2_benchmark_analysis.md) §Table 3. Halving the LR made tanh *worse* (A vs B), and swapping tanh→SiLU at fixed architecture gave a **335×** improvement (A vs D). An epoch-budget problem does not respond to activation swaps that way. |
| **H2** | The MLP-ODE with **multiple hidden layers + SiLU** converges and lands somewhat worse than KAN — defensible as the honest baseline | ✅ **Confirmed** | Table 3: MLP-SiLU extrap MSE $4.77\times10^{-4}$ vs KAN $8.92\times10^{-5}$ — KAN is $5.4\times$ better from statistically identical training loss. **But see §5 — the MLP converges 4.1× *faster*, which contradicts the paper's claim and must be reported.** |
| **H3** | Both cross-domain runs fail because of **gradient explosion, from a small divisor somewhere** | ⚠️ **Half right** | The explosions were real and clipping eliminated them (pendulum 7186× → 11.8×). But **the pendulum did not improve**, so explosion was not its cause. And no small divisor was ever located — the RBF basis has no division at all. The actual SIR mechanism is float32 **overflow**, described in §1. |
| **H4** | PyTorch is slower than the paper's Julia stack, hence the wall-clock mismatch | ✅ **Confirmed, expected** | ~15–22× slower per epoch. Structural, not a bug. See §6. |

---

## 🧪 Part 1: SIR — training is fixed, the run now dies of NaN

### What the re-run produced

| Metric | Old run (no clipping) | **New run (clipping)** |
| :--- | :---: | :---: |
| Best train MSE | $5.66\times10^{-2}$ | $\mathbf{4.39\times10^{-4}}$ — **129× better** |
| Best epoch | 3400 | 6925 |
| Extrap MSE | — | $3.77\times10^{-3}$ |
| Extrap RMSE | $0.323$ | $\mathbf{0.0614}$ |
| Extrap $R^2$ | $-607.6$ | $-20.97$ |
| Lipschitz $L$ | — | $12.18$ |
| **Final-epoch model** | 2× worse than best | ❌ **NaN** |

Config: `[3,16,3]`, RBF, $G{=}8$, Tsit5, `substeps=2`, lr $3\times10^{-3}$,
`--t_train_end 50`, `--grad_clip 1.0`, 864 params, 10,000 epochs, seed 42, 6464 s.

**The training fix genuinely worked.** Clipping held for 8,685 epochs and the model
reached a training loss two orders of magnitude better than before.

### Then it overflowed

| Epoch | Event |
| :--- | :--- |
| 6925 | best checkpoint — `train_mse` $=4.39\times10^{-4}$ |
| — | finite gradient norms: median $4.44$, **max $5.82\times10^{18}$** |
| **8686** | gradient norm first **non-finite** (1303 non-finite entries follow) |
| **8702** | training loss first **NaN** (1299 NaN entries follow) |
| 10000 | run ends; `final_model.pt` contains NaN weights |

### 🔬 Root cause: clipping cannot survive an `inf`

`torch.nn.utils.clip_grad_norm_` computes

$$\texttt{clip\_coef} = \frac{\texttt{max\_norm}}{\texttt{total\_norm} + 10^{-6}}$$

If `total_norm` overflows float32 to `inf`, then `clip_coef → 0`, and the subsequent
`grad.mul_(clip_coef)` evaluates `inf × 0 = NaN`. **Clipping converts an overflow into
a NaN rather than preventing it.** It bounds the *optimizer step*; it does nothing
about the *forward pass* producing overflowing values in the first place.

`best_model.pt` survived only because of a Phase-1 design decision: the snapshot is
taken on **training loss, before `optimizer.step()`**
([`train.py`](../implementation/train.py) — see the checkpoint block). That one choice
is the reason this run is salvageable at all.

> ⚠️ `implementation/results/benchmarks/sir/final_model.pt` now contains **NaN
> weights**. It is committed as documentation of the failure. **Do not load it.** Use
> `best_model.pt` (epoch 6925).

### 🔬 Second finding: the model parks on a spurious fixed point

Scoring `best_model.pt` per compartment over the extrapolation window $t \in (50, 80]$:

| Compartment | True range | **Predicted range** | RMSE |
| :--- | :---: | :---: | :---: |
| S | $[0.0342,\ 0.0423]$ | $[0.0354,\ \mathbf{0.0356}]$ | $0.0025$ |
| I | $[0.0043,\ \mathbf{0.0568}]$ | $[0.0969,\ \mathbf{0.0972}]$ | $\mathbf{0.0777}$ |
| R | $[0.9009,\ \mathbf{0.9614}]$ | $[0.8718,\ \mathbf{0.8721}]$ | $\mathbf{0.0726}$ |

Training window RMSE for comparison: S $0.0288$, I $0.0157$, R $0.0156$ — the fit
inside the observed window is fine.

**The predicted trajectory drifts by less than $3\times10^{-4}$ across 30 simulated
days.** It has learned $\mathbf{f} \approx \mathbf{0}$ near equilibrium and frozen at
the *wrong* equilibrium — roughly **22× too high on I** (0.097 vs a true 0.004 at
$t=80$), while reality continues to evolve.

This is a much more actionable diagnosis than "cannot extrapolate." The model is not
diverging, oscillating, or unstable. It stops too early, at the wrong place.

### 🔬 Third finding: mass conservation is violated

SIR carries the exact invariant $S + I + R = 1$. The model does not know this:

| Window | Predicted $S+I+R$ | True |
| :--- | :---: | :---: |
| Train $[0,50]$ | $[0.9490,\ 1.0287]$ | $1.0$ |
| Extrap $(50,80]$ | $[1.0044,\ 1.0045]$ | $1.0$ |

Nothing in the loss, architecture, or regularisation enforces it.

### 📐 Note on reporting SIR

Extrapolation-window standard deviations are $0.002$–$0.017$, versus $0.11$–$0.36$ in
the training window. $R^2$'s denominator nearly vanishes there, so $R^2 = -20.97$ does
**not** mean "21× worse than a mean predictor" in any intuitive sense. **Report RMSE
and MAE for this window; cite $R^2$ only with the low-variance caveat.**

### ✅ Proposed fixes — SIR

**S1 — NaN/inf guard (highest priority, ~5 lines, protects every future run).**
In [`train.py`](../implementation/train.py), before clipping and stepping:

```python
if not math.isfinite(gnorm):
    optimizer.zero_grad(set_to_none=True)
    nonfinite_steps += 1
    continue                      # log it, drop the step, keep the weights
if grad_clip is not None and grad_clip > 0.0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
optimizer.step()
```

Record `nonfinite_steps` in `metrics.json`. Without this, any run on any system can
silently burn hours and finish with a NaN `final_model.pt`. This is a **pipeline fix,
not a per-system tune** — the same class of change as gradient clipping was.

**S2 — Enforce mass conservation.** Two options:

* *Structural (preferred):* predict two compartments and derive the third,
  $R = 1 - S - I$. Reduces the effective state dimension and makes the invariant exact
  by construction.
* *Soft:* add $\lambda\,(\textstyle\sum_i u_i - 1)^2$ to the loss.

The structural version is a genuine **structure-preserving integration** contribution,
which maps directly onto the CSE 402 syllabus and is worth more than a penalty term in
the write-up.

**S3 — Per-compartment loss normalisation.** Training-window standard deviations are
S $0.358$, I $0.113$, R $0.332$ — a **3.19×** imbalance. I is both the
epidemiologically interesting compartment *and* the one receiving the least gradient,
and it is exactly where the $0.0777$ extrapolation RMSE lands. Divide each dimension
by its training std before the MSE.

**S4 — Reporting protocol.** RMSE/MAE primary; $R^2$ caveated (see above).

> **Assessment:** SIR is already at $4.39\times10^{-4}$ training loss. **S1 + S2 alone
> have a good chance of turning this into a publishable positive result.**

---

## 🎯 Part 2: Damped pendulum — clipping worked, the model still does not fit

### What the re-run produced

| Metric | Old run (no clipping) | **New run (clipping)** |
| :--- | :---: | :---: |
| Best train MSE | $3.107\times10^{-1}$ | $2.865\times10^{-1}$ (−8%) |
| Extrap MSE | $1.029$ | $1.165$ (worse) |
| Extrap $R^2$ | $-1.046$ | $\mathbf{-1.318}$ (worse) |
| Lipschitz $L$ | $116.79$ | $\mathbf{145.8}$ (stiffer) |
| Grad max / median | $\mathbf{7186\times}$ @ ep 4963 | $\mathbf{11.8\times}$ @ ep 23 |
| Loss ratio 8000→10000 | $1.03$ | $\mathbf{1.003}$ (dead flat) |
| Parameters | 240 | 360 |

Config: `[2,10,2]`, RBF, $G{=}8$, Tsit5, `substeps=2`, lr $3\times10^{-3}$,
`--grad_clip 1.0`, 10,000 epochs, seed 42, 4173 s.

**Clipping did exactly what it was designed to do** — no mid-training spike, and the
$11.8\times$ excursion sits at epoch 23, a normal early-training transient. The model
also had **50% more parameters** than the failed run. And it is no better: flat from
epoch 8000, worse extrapolation, a stiffer learned field.

> **Conclusion: the gradient explosion was a real bug worth fixing, but it was never
> the pendulum's root cause.**

### 🔬 What is actually broken — the per-dimension split

| Window | $\theta$ RMSE | $\omega$ RMSE |
| :--- | :---: | :---: |
| Train $[0,3]$ | $\mathbf{0.7555}$ | $0.0477$ |
| Extrap $(3,10]$ | $0.3632$ | $1.4829$ |

$\omega$ is fitted almost perfectly. $\theta$ is not. Predicted $\theta$ spans only
$[-0.211,\ 2.000]$ against a true $[-1.385,\ 2.000]$ — it starts at the correct initial
condition, decays, and then **cannot swing negative**.

That matters because $\dot\theta = \omega$ is the *trivial* half of the system — a pure
identity with no nonlinearity. Probing the learned field $f_1(\theta{=}0, \omega)$
against the truth $f_1 = \omega$:

| $\omega$ | True $f_1$ | Learned $f_1$ |
| :---: | :---: | :---: |
| $-4.00$ | $-4.00$ | $-2.925$ |
| $-1.00$ | $-1.00$ | $-3.347$ |
| $0.00$ | $0.00$ | $-3.179$ |
| $+1.00$ | $+1.00$ | $-4.371$ |
| $+3.00$ | $+3.00$ | $-4.981$ |

The truth is a line of slope $+1$. The model learned something roughly **constant at
$\approx -3$, sloping the wrong way.**

> ⚠️ *Caveat:* that probe holds $\theta$ fixed at 0, which is off the training data
> manifold — read it as suggestive. The on-manifold evidence ($\theta$ RMSE $0.755$,
> never reaching $-1.385$) is the solid part.

### ❌ The tanh-saturation hypothesis is refuted by measurement

`reflection.md` floated, and declined to confirm, the idea that
$\omega \in [-4.60, 3.52]$ saturating under `tanh` starves the spline basis. Measured
grid occupancy on the actual training data ($G{=}8$, so 7 cells):

| System | Dim | Saturated ($\lvert\tanh\rvert>0.99$) | Empty cells | Outcome |
| :--- | :--- | :---: | :---: | :---: |
| Lotka-Volterra | prey $x$ | $\mathbf{41.7\%}$ | $\mathbf{6/7}$ | ✅ $R^2=0.99997$ |
| Lotka-Volterra | predator $y$ | $22.2\%$ | $4/7$ | ✅ |
| Pendulum | $\theta$ | $\mathbf{0.0\%}$ | $\mathbf{0/7}$ | ❌ $R^2=-1.32$ |
| Pendulum | $\omega$ | $39.3\%$ | $1/7$ | ❌ |

**Lotka-Volterra is more saturated than the pendulum** — all 36 prey points collapse
into a single grid cell — and it converges to $R^2 = 0.99997$. Meanwhile pendulum
$\theta$ is perfectly spread across all 7 cells and is *the dimension that fails*.

Grid occupancy does not explain the difference. `reflection.md` was right to withhold
this one.

### 🔬 Two better candidates (both untested)

**C1 — Unweighted-MSE scale imbalance.** Training-window standard deviations:

| System | Per-dimension std | Imbalance | Outcome |
| :--- | :--- | :---: | :---: |
| Lotka-Volterra | $x{=}1.986$, $y{=}1.367$ | $\mathbf{1.45\times}$ | ✅ converges |
| Pendulum | $\theta{=}1.010$, $\omega{=}2.534$ | $\mathbf{2.51\times}$ | ❌ |
| SIR | $S{=}0.358$, $I{=}0.113$, $R{=}0.332$ | $\mathbf{3.19\times}$ | ❌ |

$\omega$ contributes roughly $6\times$ more to the squared loss than $\theta$, so the
optimizer fits $\omega$ and starves $\theta$ — precisely the observed split. The
correlation across all three systems is perfect, but that is **three data points: a
hypothesis, not proof.**

> ## ⛔ C2 BELOW WAS REFUTED — see [`09`](./09_stability_fix_results.md)
>
> The `pendulum_control_win5` run (SiLU + `--t_train_end 5.0`, 10,000 epochs) reaches
> $\theta$ RMSE $0.0114$ and $\theta_{\min} = -1.380$ against a true $-1.385$. **SiLU
> learns $\dot\theta = \omega$ perfectly well once the training window is long enough**,
> so the claim below — that it *cannot represent* the relation — is wrong. The probe of
> $f_1(0,\omega)$ that motivated it was measured on a model that had failed to converge.
>
> **What survives:** at the short window, `identity` reaches train MSE
> $2.62\times10^{-4}$ in 2,000 epochs while SiLU is still at $2.87\times10^{-1}$ after
> 10,000 — a data-efficiency gap, not a representational limit. **Do not carry the
> strong claim into the report.**

**C2 — The SiLU residual branch cannot span both signs.** The `KDense` base path is
$W \cdot \mathrm{silu}(x)$, and $\mathrm{silu}(w) = w\,\sigma(w)$:

| $w$ | $\mathrm{silu}(w)$ | identity would be |
| :---: | :---: | :---: |
| $-4.0$ | $\mathbf{-0.072}$ | $-4.0$ |
| $-2.0$ | $\mathbf{-0.238}$ | $-2.0$ |
| $-1.0$ | $-0.269$ | $-1.0$ |
| $+1.0$ | $0.731$ | $+1.0$ |
| $+4.0$ | $3.928$ | $+4.0$ |

SiLU is near-linear for $w > 0$ and **collapses to $\approx 0$ for $w < 0$**.
Representing $\dot\theta = \omega$ requires a linear function across **both** signs.
The spline path cannot supply it either — it passes through `tanh`, which is bounded.

**The pendulum is the only system in this project with sign-changing states.**
Lotka-Volterra ($x, y > 0$) and SIR ($S, I, R \in [0,1]$) are strictly positive, where
SiLU behaves near-linearly. This is a clean structural explanation for why the recipe
transfers to one family of systems and not the other.

### ✅ Proposed fixes — pendulum

**P1 — Test the activation hypothesis. One flag, no code change, runnable today.**

`--act` maps to the KAN's `base_act`, and [`layer.py`](../implementation/kan/layer.py)
already supports `identity` and `tanh` (the argparse has no `choices=` restriction).
With `identity`, the residual branch becomes a true linear map $W\!\cdot\!x$ that can
represent $\dot\theta = \omega$ **exactly**. `tanh` is odd-symmetric and also handles
both signs.

A three-way 2,000-epoch probe (~15 min each; the current run is flat from ~epoch 1000,
so 2,000 is enough to see whether $\theta$ is fitted at all):

```powershell
cd d:\level4\Term1\NUM_project\kinetic-kan\implementation

python train.py --dataset damped_pendulum --act identity --grid_len 8 --lr 0.003 `
  --epochs 2000 --save_dir results/_probe/pend_identity

python train.py --dataset damped_pendulum --act tanh --grid_len 8 --lr 0.003 `
  --epochs 2000 --save_dir results/_probe/pend_tanh

python train.py --dataset damped_pendulum --act silu --grid_len 8 --lr 0.003 `
  --epochs 2000 --save_dir results/_probe/pend_silu      # control
```

Compare **$\theta$ specifically** — aggregate MSE hides it, because $\omega$ dominates:

```powershell
python evaluate.py --checkpoint results/_probe/pend_identity/best_model.pt `
  --save_dir results/_probe/pend_identity/eval
```

If `identity` fixes $\theta$, that is a clean, well-evidenced finding worth a paragraph
in the report: *the KAN's residual-branch activation determines which sign regimes the
model can represent, and SiLU silently fails on sign-changing states.* That generalises
beyond this project and is a legitimate contribution.

**P2 — Per-dimension normalised loss.** Same change as **S3**. Independent of P1 and
likely additive.

**P3 — Rescale the state before the normaliser.** Dividing $\omega$ by $\approx 3$
gives 0% saturation and 0 empty cells (verified: $\omega/3 \to \tanh \in [-0.911,
0.825]$, cells $[15,14,7,4,5,7,9]$). **Ranked below P1/P2** — Lotka-Volterra converges
while *more* saturated, so this is unlikely to be the primary cause.

**P4 — Longer training window.** $[0,3]$ is only **1.5 oscillation periods**
($T \approx 2.006$ s) while extrapolation covers **4.99**. Try `--t_train_end 5.0`.
Cheap, but it changes the *task* rather than fixing the *model* — run it **after**
P1/P2 or the result is confounded.

> ⚠️ **CLI limitation.** Neither `--normalizer` nor `grid_lims` is exposed on the
> command line. `normalizer` is a `train_kan_ode()` keyword (reachable from Python);
> `grid_lims` is not plumbed through `train_kan_ode` at all — `KAN` defaults it to
> $(-1,1)$. So **P3 requires a code change, whereas P1 requires none.**

---

## 🛡️ Part 3: Cross-cutting fixes (apply once, benefit every system)

| ID | Fix | Cost | Rationale |
| :-: | :--- | :--- | :--- |
| **X1** | NaN/inf guard before `optimizer.step()` (**= S1**) | ~5 lines | SIR lost 1,300 epochs and its final model to this. Any system can hit it. |
| **X2** | Per-dimension loss normalisation (**= S3 = P2**) | ~5 lines | Imbalance tracks failure across all 3 systems (1.45× ✅ / 2.51× ❌ / 3.19× ❌). |
| **X3** | Log `grad_norm` **post**-clip alongside pre-clip | 2 lines | Currently only the pre-clip norm is recorded, so "did clipping engage?" needs forensic work on the history arrays. |
| **X4** | Spike monitor: flag/abort if `grad_norm` exceeds $k\times$ its running median | ~10 lines | `sir/reflection.md` established that 100-epoch smoke tests cannot predict mid-training stability. This can. |
| **X5** | Add `fix/**` to the `ci.yml` trigger list | 1 line | All four commits on this branch were pushed **without CI verification**. |

**X1 and X2 are the two highest-value changes in this entire document.** Both are
small, both are pipeline-level rather than per-system tunes, and both address
mechanisms that are *measured*, not guessed.

---

## 🧵 Part 4: The MLP-tanh question (H1) — settled, do not re-litigate

The team's working explanation was that `[2,50,2]`+tanh fails only because tanh
converges slowly and 10,000 epochs is not the paper's 100,000. **The data does not
support this.** Four controlled 2,000-epoch runs (from
[`05`](./05_phase2_benchmark_analysis.md) §Table 3):

| Run | Architecture | Act. | lr | Train MSE | Extrap MSE | Extrap $R^2$ |
| :-- | :--- | :---: | :---: | :---: | :---: | :---: |
| A | `[2,50,2]` | tanh | $2\times10^{-3}$ | $1.21\times10^{0}$ | $1.98\times10^{0}$ | $0.319$ |
| B | `[2,50,2]` | tanh | $5\times10^{-4}$ | $1.85\times10^{0}$ | $9.23\times10^{0}$ | $-2.169$ |
| C | `[2,14,8,8,2]` | SiLU | $2\times10^{-3}$ | $\mathbf{4.28\times10^{-4}}$ | $\mathbf{6.97\times10^{-4}}$ | $\mathbf{0.99976}$ |
| D | `[2,50,2]` | SiLU | $2\times10^{-3}$ | $3.62\times10^{-3}$ | $2.68\times10^{2}$ | $-91.06$ |

* **A vs D** — architecture fixed, activation varied: a **335×** gap. *The activation
  governs whether the model fits at all.*
* **C vs D** — activation fixed, architecture varied: extrapolation
  $7.0\times10^{-4}$ vs $2.7\times10^{2}$. *Depth governs extrapolation.*
* **A vs B** — halving the learning rate makes it **worse**. *Not an LR problem.*
* Init-time tanh saturation was measured and **ruled out**: only 3.2% of first-layer
  pre-activations exceed $\lvert z\rvert > 2$ for `[2,50,2]`, versus **15.9%** for the
  deep SiLU network that trains perfectly.

A pure epoch-budget deficit does not respond to an activation swap with a 335× jump at
a *fixed* budget. The mechanism remains uncharacterised, but "it just needs more
epochs" is excluded.

**Reporting guidance (unchanged):** use the **SiLU** column for every KAN-vs-MLP claim
— beating a crippled baseline proves nothing. Cite the tanh column separately as a
reproducibility note on the paper's stated architecture.

### ⚠️ The finding that must not be omitted

| | KAN-ODE | MLP-ODE (SiLU) |
| :--- | :---: | :---: |
| Extrap MSE | $\mathbf{8.92\times10^{-5}}$ | $4.77\times10^{-4}$ |
| Epochs → $10^{-3}$ | $5282$ | $\mathbf{1297}$ |
| Epochs → $10^{-4}$ | $9617$ | $\mathbf{9145}$ |

KAN extrapolates $5.4\times$ better — the paper's central claim, reproduced. **But the
MLP converges $4.1\times$ faster**, directly contradicting the paper's reported
$\sim\!10\times$ KAN speed advantage. Report both. The honest framing is: *the KAN's
advantage is in generalisation, not in fitting speed.*

---

## ⏱️ Part 5: Julia vs. PyTorch wall-clock (H4) — structural, not a bug

Reference points: the paper's Julia stack runs ~19 min per 100,000 epochs
($\approx 0.011$ s/epoch). Our Lotka-Volterra RBF+Tsit5 run takes 1712 s per 10,000
epochs ($\approx 0.171$ s/epoch); heavier configs run slower still.

That is a **~15–22× per-epoch gap**, and it is expected:

1. **Compilation.** `Lux.jl` + `DifferentialEquations.jl` compile the unrolled solver
   to native code. PyTorch pays Python interpreter overhead on **every one of the 6
   Tsit5 stages, per substep, per interval, per epoch.**
2. **Tiny tensors.** The state is 2–3 dimensional. At this size, per-op dispatch
   overhead dominates and completely swamps any BLAS advantage.
3. **No batching.** A single trajectory is integrated sequentially — nothing amortises
   the dispatch cost.

**Evidence it is uniform overhead and not a hidden bug:** Table 1b shows wall-clock
tracking NFE almost perfectly — $254 : 499 : 498 : 1019 : 1699 : 1712$ s against NFE
ratios $1 : 2 : 2 : 4 : 6 : 6$. A pathological slowdown would break that
proportionality. It does not.

**Recommendation:** state the gap explicitly in the report as a
framework-implementation difference, and **compare per-epoch cost ratios between our
own configurations, never absolute seconds against the paper.** Table 1b's within-batch
timings are trustworthy; Table 3's cross-batch timings are not. *(Abhishek has already
raised this with NZR by email — no code action required.)*

---

## 🚦 Part 6: Priority order

| Rank | Fix | Effort | Blocks | Expected payoff |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **X1** NaN/inf guard | ~5 lines | every future run | prevents silent total loss of a run |
| **2** | **P1** activation probe (`--act identity/tanh`) | **0 lines** — 3 commands | pendulum, Novelty 3 | decisive test of the leading hypothesis |
| **3** | **X2** per-dimension loss normalisation | ~5 lines | pendulum + SIR | the one mechanism correlating with all 3 outcomes |
| **4** | **S2** SIR mass conservation | ~10 lines | SIR | structure preservation — a syllabus-aligned contribution |
| **5** | **X3/X4** post-clip logging + spike monitor | ~12 lines | diagnostics | makes the next failure readable off the table |
| **6** | **P4** longer pendulum window | 1 flag | pendulum | run only *after* 2 and 3 |
| **7** | **X5** CI trigger for `fix/**` | 1 line | repo hygiene | — |

**Start with #2.** It costs nothing, needs no code change, and its outcome determines
whether the pendulum work is an activation problem or a loss-weighting problem — which
in turn decides how much of #3 is needed.

### Discipline note

Both re-runs in this round changed **three variables at once** (clipping + `grid_len`
+ `lr`), which is why "did clipping help?" required forensic analysis of the training
histories instead of being readable straight off the results table. **Change one thing
at a time** — the same OFAT discipline the Lotka-Volterra sweeps already follow. The
P1 probe above is deliberately constructed that way: one flag, one control, one
question.

---

## ❓ Part 7: What is still unknown

1. **Why `[2,50,2]`+tanh fails to train.** Epoch budget, LR, and init-time saturation
   are all excluded. Mechanism uncharacterised.
2. **Whether the pendulum failure is activation (C2) or loss weighting (C1)** — or
   both. P1 separates them.
3. **Whether SIR extrapolation is fixable at all**, or whether an endemic-equilibrium
   window is intrinsically low-signal for MSE-based training. Answerable only after a
   run that neither explodes nor NaNs.
4. **Whether Lorenz needs any of this.** ❌ **Never run** —
   `.\run_phase2.ps1 -Only lorenz`, ~2.6 h coarsened. Given the pattern, expect it to
   need X1 and X2 too. This is the last outstanding Phase-2 deliverable.
5. **Multi-seed replication.** Formally Phase 4 Task 4.1, but **no ordering claim in
   Table 1 is publishable without it** — the Midpoint/RK4/DOPRI5/Tsit5 spread is 3.8%
   at $N=1$.

---

## 📎 Impact on Phase 3

* **Novelty 3 (stiffness phase map) remains blocked.** It sweeps pendulum damping
  $\mu$ across solvers and is built directly on a system that does not converge. Do not
  assign it until P1/X2 land and the pendulum fits.
* **Novelty 1 (gradient-norm dynamics) is unblocked and needs no new runs.** All 26
  completed runs already carry per-epoch `grad_norms` in `training_history.json`. This
  is pure analysis of existing artifacts — the cheapest available Phase-3 win, and it
  can proceed in parallel with everything above.
* **Novelties 4 and 5 have missing dependencies.** `pysindy` and `torchdiffeq` are
  listed in `requirements.txt` but are **not installed** in the current environment.
  Novelty 4 additionally needs an $L_1$ edge-pruning step that does not exist anywhere
  in the code — `regularization.py` penalises parameter magnitude but never prunes.

```powershell
pip install torchdiffeq pysindy
```

---

*Companion documents:
[`05_phase2_benchmark_analysis.md`](./05_phase2_benchmark_analysis.md) (empirical
report),
[`pendulum/reflection.md`](../implementation/results/benchmarks/pendulum/reflection.md)
and [`sir/reflection.md`](../implementation/results/benchmarks/sir/reflection.md)
(pre-clipping failure diagnoses — note both now overstate gradient clipping as the
shared root cause and should be updated).*
