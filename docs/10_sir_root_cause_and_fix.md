# 🦠 SIR: Root Cause and Fix

> **Branch:** `fix/phase1-revisit`
> **Status:** ✅ **complete** — 3 probes @ 2,000 epochs (`results/_probe/`) + 2 full runs @ 10,000 epochs (`results/_fixed/`). All acceptance criteria pass.
> **Chain:** [`06`](./06_suggested_fixes.md) diagnosis → [`07`](./07_fix_changelog.md) code → [`08`](./08_how_to_run_fixes.md) how to run → [`09`](./09_stability_fix_results.md) probe results → **this doc: SIR root cause**

---

## 🏁 Executive summary

SIR was the last failing system. It failed **three different ways** — NaN divergence, a
frozen fixed point, and mass drift — and every earlier attempt treated those as three
problems to be patched separately. They were not. Two of them share a single cause, and
the third was never a fitting problem at all.

**SIR is not harder dynamics than Lotka-Volterra. It is the same kind of dynamics
written in the wrong units.**

All figures below are **10,000-epoch runs**, compared against the 10,000-epoch pre-fix
baseline from [`06`](./06_suggested_fixes.md) — like for like.

| | Pre-fix 10k run | **`sir_fixed`** | **`sir_fixed_full`** |
| :--- | :---: | :---: | :---: |
| | (clipping only) | time_scale + projection | + `--vanish_dim 1` |
| Train MSE | $4.39\times10^{-4}$ | $1.08\times10^{-5}$ | $\mathbf{2.28\times10^{-8}}$ |
| Extrap RMSE | $0.0614$ | $\mathbf{0.0140}$ | $\mathbf{5.70\times10^{-4}}$ |
| $I$ RMSE (extrap) | $0.0776$ | $0.0194$ | $\mathbf{0.0003}$ |
| Mass error | $0.0045$ | $\mathbf{0.0000}$ | $\mathbf{0.0000}$ |
| $I$ drift (true $0.0525$) | $0.0003$ (frozen) | $0.0837$ | $\mathbf{0.0524}$ |
| Extrapolation $R^2$ | $-20.97$ | $-0.145$ | $\mathbf{+0.9981}$ |
| Max gradient norm | $5.82\times10^{18}$ | $92.3$ | $\mathbf{1.19}$ |
| Final-epoch model | ❌ **NaN** | finite | finite |

### 🔑 The headline result: the physics prior is *optional*

**`sir_fixed` passes every acceptance criterion using only `--time_scale` and
`--conserve_mode projection`** — two changes that assume nothing about SIR beyond what
its own equations already state. One is a change of units, the other is algebra.

`--vanish_dim` (which *does* make a physical claim, see Part 3) then buys a further
**25× on extrapolation RMSE** and drives $I$ drift to $0.0524$ against a true $0.0525$.
It is a genuine improvement, but it is not what makes SIR work.

> ⚠️ **Caveat, both runs.** The flatness check reports both as **REGRESSED**: the
> final-epoch model is worse than the best (`sir_fixed` $2.43\times10^{-4}$ vs
> $1.08\times10^{-5}$ @ ep 8807; `sir_fixed_full` $1.64\times10^{-5}$ vs
> $2.28\times10^{-8}$ @ ep 7857). Selection-on-training-loss means the reported
> checkpoints are sound, but a mild late-training instability remains and is **not**
> claimed to be fixed.

---

## 🔬 Part 1: The measurement that explains two of the three failures

Everything below follows from four numbers taken at **initialisation**, before a single
optimizer step:

| Quantity | Value |
| :--- | ---: |
| KAN's initial field $\lvert f_\theta(y)\rvert$ | $0.203$ |
| True SIR field $\lvert f\rvert$ | $0.011$ |
| Trajectory integrated over $[0, 50]$ | runs to $\mathbf{-51.8}$ |
| Epoch-0 loss / gradient norm | $3.16\times10^{2}$ / $4.89\times10^{3}$ |

A Glorot-initialised KAN starts **19× too fast**. Integrated over a horizon of 50, that
sends the state to $-51.8$ when the physical range is $[0,1]$.

### Why the gradient "explosion" was never a late-training instability

Backprop through the solver differentiates ~1,200 nested nonlinear steps, and its
sensitivity grows like $e^{LT}$. That product is what separates the systems that train
from the ones that do not:

| System | $T$ | $\lvert f\rvert$ | Outcome |
| :--- | :---: | :---: | :--- |
| Lotka-Volterra | $3.5$ | $\sim10^{0}$ | converges |
| Damped pendulum | $5.0$ | $\sim10^{0}$ | converges (after the window fix) |
| **SIR** | $\mathbf{50.0}$ | $\mathbf{\sim10^{-2}}$ | **explodes** |

The runs confirm the timing directly. `grad_norm_max_epoch` is **11** for `sir_control`
and **12** for `sir_lossw` — the spike is at epoch *eleven*, not epoch 8,686. The
$5.82\times10^{18}$ overflow reported in [`06`](./06_suggested_fixes.md) was the *end* of
a process that began immediately. Gradient clipping was therefore always treating a
symptom, which is exactly why adding it moved the failure rather than removing it.

### Why the "frozen fixed point" was rational optimizer behaviour

From an epoch-0 loss of $3.16\times10^{2}$, the steepest available descent direction is
$\mathbf{f} \to \mathbf{0}$: a constant trajectory scores $\approx10^{-1}$, a **~3,000×**
improvement, and it is reachable immediately. The optimizer takes it and parks there.
The true $10^{-2}$ signal never competes.

So the frozen fixed point was never a defect of the MSE's scale sensitivity to be fixed
by reweighting. It was the correct answer to the question the optimizer was actually
being asked.

### The fix: nondimensionalise time

SIR's natural timescale is the recovery time $1/\gamma = 10$ days. Substituting
$\tau = t/10$ gives the exact change of variables

$$\frac{dy}{d\tau} = 10\,f(y), \qquad \tau \in [0, 5]$$

which lands SIR in precisely the regime the other two systems already train in — horizon
$5$, derivative $\sim10^{-1}$. Implemented as `--time_scale`.

Nothing downstream is rescaled: predictions come out at the same physical sample times,
so every metric, plot and checkpoint stays directly comparable with the existing 26 runs.
`--time_scale 1.0` (the default) is a **bitwise** no-op, verified by test.

| | baseline | **`--time_scale 10`** |
| :--- | :---: | :---: |
| Train MSE @ 1,500 ep | $2.37\times10^{-2}$ | $\mathbf{2.4\times10^{-5}}$ |
| Max gradient norm | $1.97\times10^{6}$ | $\mathbf{92}$ |
| $I$ drift | $5\times10^{-7}$ (frozen) | $0.385$ (moving) |

---

## ⚖️ Part 2: Mass conservation — a penalty is the wrong instrument

With rescaling in place, mass error was still $0.35$–$0.45$. Worse, `--conserve_sum` at
weight $1.0$ made it **worse**, not better.

That is not a tuning failure, it is a scoping failure. `--conserve_sum` penalises the
residual **over the training window only**. Mass inside $[0,50]$ was already fine; all
the drift happens in extrapolation, where the penalty has no term at all.

$S+I+R=1$ is an *exact linear invariant*:

$$\frac{dS}{dt}+\frac{dI}{dt}+\frac{dR}{dt} = (-\beta SI) + (\beta SI - \gamma I) + (\gamma I) = 0$$

so it should be a property of the **flow**, not of the fit. Subtracting the componentwise
mean projects the field onto the zero-sum subspace:

$$f_{\text{proj}}(y) = f(y) - \frac{1}{n}\sum_i f_i(y) \quad\Longrightarrow\quad \sum_i f_{\text{proj},i}(y) = 0$$

Implemented as `--conserve_mode projection` (see `ZeroSumField`). It costs one mean and
one subtraction, and holds on every horizon.

| | `--conserve_sum` penalty | **`--conserve_mode projection`** |
| :--- | :---: | :---: |
| Max mass error, full horizon | $5.9\times10^{-2}$ | $\mathbf{7\times10^{-7}}$ |
| Holds outside the training window | ✗ | ✅ by construction |
| Weight to tune | yes | none |

---

## 🎯 Part 3: The extrapolation failure was never an optimization problem

After Parts 1 and 2 the fit on $[0,50]$ is essentially perfect (S/I/R tracked to
$\sim0.005$) and extrapolation still failed, at extrap RMSE $0.095$. The trajectory dump
shows what actually happens after the training window ends:

| $t$ | $S_{\text{true}}$ | $S_{\text{pred}}$ | $I_{\text{true}}$ | $I_{\text{pred}}$ |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 0.0427 | 0.0479 | 0.0593 | 0.0562 |
| 60 | 0.0372 | 0.0790 | 0.0250 | $-0.0052$ |
| 70 | 0.0351 | 0.1390 | 0.0105 | $-0.0766$ |
| 80 | 0.0342 | 0.2680 | 0.0043 | $\mathbf{-0.3320}$ |

Predicted $I$ goes **negative**, and $S$ *rises* from 0.048 to 0.268 — susceptibles
spontaneously reappearing. Both are physically impossible. Probing the learned field at
the disease-free equilibrium gives the reason:

$$y = [0.034,\; \mathbf{0.000},\; 0.966]: \quad f_{\text{true}} = [0,0,0], \qquad f_{\text{learned}} = [+0.00859,\, -0.00630,\, -0.00229]$$

**The model learned a non-zero flow at an equilibrium.** Nothing stops the trajectory
there, so $I$ is driven straight through zero.

### Why no loss function could have fixed this

The extrapolation window lives at $S \in [0.034, 0.042]$ with small $I$. Of the 101
training samples, **exactly one** is in that corner of state space — the other 16 low-$I$
samples sit at $S \in [0.92, 0.99]$, the *pre-epidemic growth phase*, which is a
completely different dynamical regime.

The model is being asked to extrapolate a vector field into a region represented by
$1/101$ of the data. Relative loss on $I$, log-space loss, per-dimension reweighting —
none of these create information that is not in the training set. This is why
`--loss_weighting std` **hurts** here (extrap RMSE $0.163$ vs $0.095$) and `lr 1e-3`
hurts more ($0.430$).

### The fix: supply the missing structure analytically

Every term in the SIR field carries a factor of $I$:

$$\frac{dS}{dt} = I\cdot(-\beta S), \qquad \frac{dI}{dt} = I\cdot(\beta S - \gamma), \qquad \frac{dR}{dt} = I\cdot(\gamma)$$

So $I=0$ is a **line of equilibria** (a disease-free population stays disease-free), and
the remaining cofactor $h = [-\beta S,\; \beta S - \gamma,\; \gamma]$ is merely **affine
in $S$**. Gating the field by $y_I$ hands the model the tail behaviour exactly, instead
of asking it to infer it from one sample, and reduces what must be learned to something
that generalises trivially. Implemented as `--vanish_dim` (see `VanishingDimField`).

> ### ⚠️ This one is a physical claim, not a numerical fix
>
> `--time_scale` and `--conserve_mode projection` are units and algebra — they assume
> nothing about the system beyond what its own equations state. **`--vanish_dim` asserts
> a fact about the model being fitted**: that the plane $y_d = 0$ consists of equilibria.
>
> That is exact for compartmental epidemic models and false in general. It is therefore
> **opt-in, off by default**, and `run_phase2.ps1` runs `sir_fixed_noprior` as an explicit
> control so its contribution is reported rather than absorbed into the headline number.
> It composes with the projection because a scalar times a zero-sum vector is still
> zero-sum, so both invariants hold at once.

---

## 📊 Part 4: Probe results (2,000 epochs)

Recipe: `--time_scale 10.0 --conserve_sum 1.0 --conserve_mode projection --vanish_dim 1`
on the standard SIR baseline (`[3,16,3]`, RBF, $G=8$, Tsit5, `substeps=2`, lr $3\times10^{-3}$,
`--grad_clip 1.0`, seed 42).

| run | train_mse | extrap_mse | extrap RMSE | extrap $R^2$ | mass err | $I$ drift |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sir_control` (baseline) | $1.74\times10^{-2}$ | $7.21\times10^{-2}$ | $0.2685$ | $-419.2$ | $0.1347$ | $0.0000$ |
| `sir_lossw` | $7.31\times10^{-3}$ | $3.29\times10^{-2}$ | $0.1814$ | $-190.7$ | $0.0023$ | $0.0000$ |
| `sir_conserve` | $1.19\times10^{-2}$ | $4.42\times10^{-2}$ | $0.2102$ | $-256.6$ | $0.0086$ | $0.0000$ |
| `sir_van_only` | $7.90\times10^{-7}$ | $7.77\times10^{-6}$ | $0.0028$ | $+0.955$ | $1.2\times10^{-3}$ | $0.0524$ |
| **`sir_van_proj`** | $1.71\times10^{-6}$ | $1.54\times10^{-5}$ | $\mathbf{0.0039}$ | $\mathbf{+0.910}$ | $\mathbf{0.0000}$ | $\mathbf{0.0522}$ |
| `sir_van_proj_g5` | $7.53\times10^{-7}$ | $7.29\times10^{-6}$ | $0.0027$ | $+0.958$ | $0.0000$ | $0.0527$ |

`sir_van_proj` is the shipped recipe: `van_only` is marginally more accurate but does not
conserve mass exactly, and `g5` matches it with fewer parameters (a promising ablation,
not yet run at full budget).

> ⚠️ **These probe numbers overstate how much the physics prior matters.** At 2,000
> epochs the gated runs beat the ungated ones by ~25× on extrap RMSE ($0.0039$ vs the
> $0.095$ measured for `time_scale` + projection at that budget). At the full 10,000-epoch
> budget the ungated recipe reaches $0.0140$ and **passes on its own** — see Part 4b. The
> prior converges faster; it is not what makes SIR work.

---

## 🏆 Part 4b: Full runs (10,000 epochs) and acceptance criteria

| Metric | Target | `sir_fixed` | | `sir_fixed_full` | |
| :--- | :--- | :--- | :-: | :--- | :-: |
| Status / exit | 10,000 epochs, `aborted: null` | 10,000, `None` | ✅ | 10,000, `None` | ✅ |
| `final_train` | finite (not NaN) | $2.43\times10^{-4}$ | ✅ | $1.64\times10^{-5}$ | ✅ |
| `nonfinite_grad_steps` | $0$ | $\mathbf{0}$ | ✅ | $\mathbf{0}$ | ✅ |
| Mass conservation | err $<1\%$ | $\mathbf{1.0000}$ | ✅ | $\mathbf{1.0000}$ | ✅ |
| $I$ drift | $>0.03$ (true $0.0525$) | $0.0837$ | ✅ | $\mathbf{0.0524}$ | ✅ |
| Extrap RMSE | $<0.05$ | $\mathbf{0.0140}$ | ✅ | $\mathbf{0.00057}$ | ✅ |

**Both configurations pass all six.** The previous run died of NaN at epoch 8,686 and
lost its final 1,299 epochs; both of these complete all 10,000 with **zero** non-finite
gradient steps and a finite final model.

Full-horizon $R^2$ is $0.99905$ (`sir_fixed`) and $1.00000$ (`sir_fixed_full`).

Note that $I$ drift distinguishes the two in a way RMSE alone does not: `sir_fixed`
overshoots the true decay by $1.6\times$ ($0.0837$ vs $0.0525$) while still landing inside
the RMSE bar, whereas the gated run reproduces it to $0.0524$. Both are far from the
frozen $0.0003$ of the pre-fix run.

> 📊 **Ignore `spike_ratio` on these runs.** It reads $619.8\times$ and $48{,}813\times$,
> which looks alarming next to the $2.47\times10^{6}$ that originally flagged SIR. It is
> max/median, and the *median* gradient is now $\sim10^{-5}$ — the ratio is large because
> the denominator collapsed, not because the numerator grew. The absolute maxima are
> $92.3$ and $\mathbf{1.19}$, against $5.82\times10^{18}$ before. Read
> `grad_norm_max_preclip` for these runs, not the ratio.
>
> 📐 **And on $R^2$ for this window.** [`06`](./06_suggested_fixes.md) correctly warned
> that extrapolation-window $R^2$ is near-meaningless because the true signal std there is
> only $0.002$–$0.017$. That caveat cuts both ways: it made $-20.97$ look worse than it
> was, and it makes $+0.9981$ *harder* to achieve than a normal $R^2$. **RMSE remains the
> headline metric**; $R^2$ is reported only because its sign flipped for the first time.

---

## 🛠️ Part 5: Collateral bugs found

Three latent bugs in the checkpoint-replay path, all of the same class: the field that
gets rebuilt is not the field that was trained, and **nothing raises an error**.

| Bug | Why it is silent | Fixed in |
| :--- | :--- | :--- |
| `evaluate.py` dropped `grid_lims` | `grid` is a buffer and weight *shapes* do not depend on it, so `load_state_dict` succeeds on the wrong grid | `evaluate.py` |
| `evaluate.py` / `analyze_fixes.py` ignored `time_scale` | replaying on the physical clock integrates $10\times$ the intended field | both |
| Neither knew about `conserve_mode` / `vanish_dim` | projection and gating are field *structure*, not weights, so they vanish on rebuild | both |

Any run using a non-default grid span was being **mis-scored**, silently, before this.
Regression tests now assert that a non-default `grid_lims` actually changes predictions
(otherwise the guard would be vacuous) and that each option survives a save/reload round
trip.

---

## ✅ What changed

| Flag | Fixes | Default | Nature | Needed to pass? |
| :--- | :--- | :--- | :--- | :---: |
| `--time_scale` | divergence **and** frozen fixed point | `1.0` (bitwise no-op) | units | ✅ yes |
| `--conserve_mode projection` | mass drift, on all horizons | `penalty` (old behaviour) | algebra | ✅ yes |
| `--vanish_dim` | tail extrapolation accuracy | `None` (off) | **physical assumption** | ❌ optional |

The two fixes required to pass every acceptance criterion are both assumption-free. The
one that encodes physics is a $25\times$ accuracy improvement on top, not a dependency —
which is the result worth reporting, because it means **SIR's failure was a numerical
problem and it has a numerical fix**.

All 26 existing Phase-2 runs are unaffected: every default reproduces prior behaviour, and
`--time_scale 1.0` is verified bitwise-identical on both `train_losses` and `grad_norms`.

### Reproducing

```powershell
# the fix (no physical assumptions)
python train.py --dataset sir --basis rbf --solver tsit5 --t_train_end 50.0 `
  --layers 3 16 3 --grid_len 8 --lr 0.003 --grad_clip 1.0 `
  --time_scale 10.0 --conserve_sum 1.0 --conserve_mode projection `
  --epochs 10000 --save_dir results/_fixed/sir_fixed

# + the compartmental-model prior
python train.py --dataset sir --basis rbf --solver tsit5 --t_train_end 50.0 `
  --layers 3 16 3 --grid_len 8 --lr 0.003 --grad_clip 1.0 `
  --time_scale 10.0 --conserve_sum 1.0 --conserve_mode projection --vanish_dim 1 `
  --epochs 10000 --save_dir results/_fixed/sir_fixed_full

python analyze_fixes.py --root results/_fixed
```
