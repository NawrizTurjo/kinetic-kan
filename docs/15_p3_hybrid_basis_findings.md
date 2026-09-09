# 🧬 Track C — Learnable Softmax Hybrid Basis: Implementation Guideline & Findings

> **Owner:** Shams Hossain Simanto (2105048) · **Branch:** `feat/p3-hybrid-basis`
> **Folder:** `experiments/hybrid_basis/` · **Results:** `results/phase3/hybrid_basis/`
> **Status:** 📋 guideline complete, implementation not yet started
> **Parent plan:** [`12_phase3_roadmap.md`](./12_phase3_roadmap.md) §Track C
>
> This file is meant to be **self-contained** — everything needed to execute Track C
> end-to-end is here, without flipping back to `12`. It starts as a plan; the
> **§Findings** section at the bottom is a placeholder to fill in as runs complete, so
> this same file becomes the final write-up.

---

## 1. Research question

Phase 2 found B-spline and RBF statistically tied on accuracy
([`05`](./05_phase2_benchmark_analysis.md) §Table 2) but very different in character:
B-spline is $3.9\times$ RBF's wall-clock and has **compact local support** (each edge
function is zero outside its knot span); RBF is cheap and has **smooth global support**
(every edge function has infinite tails).

Does a **learnable blend**
$$\phi(x) = \alpha \cdot \text{Spline}(x) + \beta \cdot \text{RBF}(x), \qquad (\alpha,\beta) = \text{softmax}(\text{logits})$$
trained end-to-end, converge faster than either pure basis — or does it just inherit
the worse of both (B-spline's cost, no accuracy gain)?

---

## 2. Environment

**No new dependency.** Unlike Track A (`torchdiffeq`) or Track E (`pysindy`), this track
only reuses `torch` and the existing `kan` package. Nothing to `pip install`.

---

## 3. ⚠️ Verified gotcha — read this before writing any training code

**`train_kan_ode()` cannot be used directly with a raw `HybridBasis` module.** This was
tested, not assumed:

```python
# reproduction — do not re-run, this is recorded for reference
from train import train_kan_ode
res = train_kan_ode(basis_func=HybridBasis(grid_len=5), num_epochs=3, save_dir=...)
```

**Result:** training runs to completion correctly (loss descends, gradients flow into
`blend_logits` as expected) — and then, at the very last step, crashes:

```
TypeError: Object of type HybridBasis is not JSON serializable
```

**Root cause:** `train_kan_ode()`'s `run_config` dict stores `basis_func` verbatim
(`"basis_func": basis_func if model_type.lower()=="kan" else "none"`) and later
`json.dump()`s that whole dict into `metrics.json`. This works fine for the string names
(`"rbf"`, `"bspline"`, …) every existing run has used, and was never exercised against a
raw callable — Track C is the first thing in the project to try that.

**Impact if ignored:** at the *full* 10,000-epoch budget, this means training completes,
every epoch's compute is spent, and the crash only happens at the final write —
discovering it there would be the single most expensive way to learn about it.

**Resolution (folder-isolated, no edit to `train.py`):** do **not** call
`train_kan_ode()` for this track. Write your own training loop in
`experiments/hybrid_basis/` that mirrors its structure — same seeding, same
best-on-train-loss checkpointing, same non-finite gradient guard — but records
`basis_func` as a **descriptive string** in its own config dict, keeping the actual
`HybridBasis` object out of anything that gets JSON-serialized. §5 below is exactly
that loop, already written.

This also means `evaluate.py` cannot load a Track C checkpoint later (it reconstructs
`KAN(basis_func=<string>)` via `get_basis_function()`, which has no `"hybrid"` entry and
never will, per the no-shared-edit rule). §7 below includes a small standalone loader
for your own use — same reasoning as Track D's standalone sweep script.

---

## 3b. ⚠️ Second verified gotcha — the pendulum's learning rate does not auto-correct

**Found the hard way:** the first pendulum probe (2,000 epochs, ~1h44m of real compute)
came back with train MSE **0.294** — the same order of magnitude as the *pre-fix*
pendulum failures in [`09`](./09_stability_fix_results.md), not anywhere close to
`pendulum_control_win5`'s converged $9.28\times10^{-5}$. A $263\times$ gradient spike
(median $5.52$, max $1453.4$ at epoch 278) confirmed something was genuinely wrong, not
just slow.

**Root cause:** `run_hybrid.py`'s `run()` function auto-corrected `grid_len` to $8$ for
`damped_pendulum` (unconditionally, inside the `elif` branch), but did **not** apply the
same correction to `lr`. Every command in this doc's §8 omits `--lr`, so the pendulum
probe silently ran at the argparse default `lr=2e-3` — Lotka-Volterra's rate, not
[`09`](./09_stability_fix_results.md)'s validated `lr=3e-3` for the pendulum. `grid_len`
was right; `lr` was wrong; nothing printed at launch said so — the only trace was inside
`metrics.json`'s `config` block afterwards, findable only by explicitly comparing it
against `pendulum_control_win5`'s own recorded config.

**Fixed properly, not just patched:** `lr` and `grid_len` both now default to `None` in
`run()` and in the CLI, and are resolved to the validated per-dataset value (LV:
$\text{lr}=2\times10^{-3}$, $G=5$; pendulum: $\text{lr}=3\times10^{-3}$, $G=8$) **only
when not explicitly supplied** — an explicit `--lr` still overrides, so this is not a
second silent-override bug in the other direction. The resolved values are now also
printed at launch:

```
[damped_pendulum] resolved config: lr=0.003  grid_len=8  epochs=2000  grad_clip=1.0  seed=42
```

**Verified**, not assumed: re-ran LV, pendulum, and pendulum-with-an-explicit-override
at 3 epochs each — all three resolved exactly as intended, then the full `tests/`
suite (172 tests) re-confirmed clean.

**Consequence: the first pendulum probe result is invalid and does not represent
whether the hybrid basis works on the pendulum.** It represents the pendulum trained at
the wrong learning rate, which is already known from Phase 2 to fail regardless of
basis. It is not carried into §11 Findings below; the pendulum probe was re-run with
the fix and only that result is reported.

---

## 4. Design decision: one shared gate, or one gate per layer?

`KAN(basis_func=an_instance, ...)` passes the **same object** to every `KDense` layer
it builds. For the Lotka-Volterra architecture `[2, 10, 2]` (2 layers) or the pendulum
architecture `[2, 10, 2]` (also 2 layers), this means:

- **Shared gate (recommended default):** one `HybridBasis()` instance, reused for every
  layer. Both layers' edges are blended by the *same* $(\alpha, \beta)$, which is
  learned jointly. Simple, matches the original framing in `12` ("track the evolution of
  blend weights $\alpha(t), \beta(t)$" — singular), and PyTorch handles the parameter
  sharing correctly (`Module.parameters()` deduplicates by identity, so `blend_logits`
  is optimized once, not double-counted — verified: gradient norm and loss both moved
  sensibly across 3 test epochs above).
- **Per-layer gate (optional extension, only if time allows):** instantiate one
  `HybridBasis()` per layer and assign it *after* construction:
  ```python
  model = KAN(layers_hidden=[2, 10, 2], grid_len=5, basis_func=HybridBasis(grid_len=5))
  for layer in model.layers:
      layer.basis_func = HybridBasis(grid_len=5)   # replaces the shared one, per layer
  ```
  This works because `nn.Module.__setattr__` re-registers a submodule on *any*
  assignment, not just inside `__init__`. Interesting if you want to ask "does the
  first layer prefer a different basis than the second?" — not required for the core
  finding.

**Start with the shared gate.** Only build the per-layer variant if the probe stage
finishes early and you want a stretch result.

---

## 5. `HybridBasis` — the module itself

```python
# experiments/hybrid_basis/hybrid_basis.py
"""
[TRACK C] Learnable softmax-gated blend of Cubic B-spline and Gaussian RBF bases.
Zero edits to kan/basis.py -- bspline_basis and rbf are imported, not modified.
"""
import torch
import torch.nn as nn
from kan.basis import bspline_basis, rbf


class HybridBasis(nn.Module):
    """
    phi(x) = alpha * Spline(x) + beta * RBF(x),  (alpha, beta) = softmax(logits)

    KDense calls its basis_func as basis_func(x_norm, grid, h) -- a plain 3-arg call.
    Because this is an nn.Module, that call goes through nn.Module.__call__, which
    dispatches to .forward(x, grid, h) below. Assigning an instance of this class to
    KDense.basis_func (or passing it into KAN(basis_func=...)) auto-registers it as a
    submodule, so blend_logits shows up in model.parameters() and receives gradients
    from the ordinary training loop -- no special-casing needed in the optimizer.
    """
    def __init__(self, grid_len: int, init_logits=(0.0, 0.0)):
        super().__init__()
        # (0.0, 0.0) -> softmax gives exactly (0.5, 0.5): unbiased 50/50 start.
        # grid_len is accepted for symmetry with the other basis_fn signatures /
        # future validation, though bspline_basis and rbf both infer it from `grid`.
        self.grid_len = grid_len
        self.blend_logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))

    def forward(self, x: torch.Tensor, grid: torch.Tensor, h: float) -> torch.Tensor:
        w = torch.softmax(self.blend_logits, dim=0)
        return w[0] * bspline_basis(x, grid, h) + w[1] * rbf(x, grid, h)

    def blend_weights(self) -> tuple:
        """Read-only (alpha, beta) as plain floats, for logging."""
        w = torch.softmax(self.blend_logits.detach(), dim=0)
        return float(w[0]), float(w[1])
```

**Note on parameter count.** This adds `blend_logits` (2 scalars) on top of the usual
240-parameter `[2,10,2]`/$G{=}5$ KAN, so a hybrid-basis run has **242 parameters**,
not 240 — verified in the reproduction above (`"242 params"` in its log line). Trivial
in absolute terms, but state it explicitly wherever you cite Table 2's 240-parameter
numbers, rather than silently comparing 242 against 240.

---

## 6. Standalone training loop

Mirrors `train_kan_ode()`'s structure exactly (seeding order, best-on-train-loss
checkpoint taken via `deepcopy` *before* `optimizer.step()`, the X1 non-finite gradient
guard) — just with `basis_func` kept out of anything JSON-serialized, per §3.

```python
# experiments/hybrid_basis/run_hybrid.py
"""
[TRACK C] Standalone training loop for the learnable hybrid basis.
Reuses KAN, NeuralODE, the dataset generators, and utils -- does not call
train_kan_ode() (see docs/15 SS3 for why) and does not edit any shared file.
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
# This file lives at implementation/experiments/hybrid_basis/run_hybrid.py, so
# "implementation/" is exactly two levels up -- no extra suffix needed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from kan import KAN, count_parameters
from ode import NeuralODE
from data import generate_lotka_volterra_data, generate_damped_pendulum_data
from utils import (
    compute_mse, compute_rmse, compute_mae,
    compute_r2_score, compute_relative_l2_error, compute_gradient_norm,
)
from hybrid_basis import HybridBasis


def run(dataset, epochs, lr, grid_len, grad_clip, save_dir, seed=42, device="cpu", log_every=500):
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset == "lotka_volterra":
        data = generate_lotka_volterra_data(seed=seed)           # defaults match Table 2
        layers = [2, 10, 2]
    elif dataset == "damped_pendulum":
        # [09]-fixed recipe: t_train_end=5.0, default SiLU base_act, G=8.
        data = generate_damped_pendulum_data(t_train_end=5.0, seed=seed)
        layers = [2, 10, 2]
        grid_len = 8
    else:
        raise ValueError(dataset)

    torch.manual_seed(seed)  # re-seed so model init isn't coupled to data-gen draws
    np.random.seed(seed)

    hybrid = HybridBasis(grid_len=grid_len)
    model = KAN(layers_hidden=layers, grid_len=grid_len, basis_func=hybrid).to(device)
    node = NeuralODE(func=model, method="tsit5", substeps=2).to(device)
    total_p, _ = count_parameters(model)

    t_train, t_full = data.t_train.to(device), data.t_full.to(device)
    y_train, y_full = data.y_train.to(device), data.y_full.to(device)
    y0 = data.y0.to(device)
    n_train = len(t_train)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, alpha_hist, beta_hist, grad_norms = [], [], [], []
    best_loss, best_epoch, best_state = float("inf"), -1, None
    nonfinite_steps = 0

    pbar = tqdm(range(1, epochs + 1), desc=f"hybrid/{dataset}", ncols=110)
    for epoch in pbar:
        optimizer.zero_grad()
        pred = node(y0=y0, t=t_train)
        loss = F.mse_loss(pred, y_train)
        loss.backward()

        gnorm = compute_gradient_norm(model)
        grad_norms.append(gnorm)
        loss_val = loss.item()
        train_losses.append(loss_val)
        a, b = hybrid.blend_weights()
        alpha_hist.append(a)
        beta_hist.append(b)

        if loss_val < best_loss:
            best_loss, best_epoch = loss_val, epoch
            best_state = copy.deepcopy(model.state_dict())   # BEFORE optimizer.step()

        # [FIX-2026-08 / X1] pattern from train.py -- skip the step on a non-finite
        # gradient instead of poisoning the weights.
        if math.isfinite(gnorm) and math.isfinite(loss_val):
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        else:
            nonfinite_steps += 1
            optimizer.zero_grad(set_to_none=True)

        pbar.set_postfix({"loss": f"{loss_val:.3e}", "a": f"{a:.3f}", "b": f"{b:.3f}"})

        # [LOG] tqdm's \r-updated bar is nearly unreadable once redirected to a file
        # (every update lands on the same visual line). tqdm.write() emits a normal
        # newline-terminated line instead -- readable both live and in a redirected
        # log file, and safe to interleave with the bar.
        if epoch % log_every == 0 or epoch == epochs:
            tqdm.write(
                f"[{dataset}] epoch {epoch}/{epochs}  loss={loss_val:.4e}  "
                f"best={best_loss:.4e}@{best_epoch}  alpha={a:.4f}  beta={b:.4f}  "
                f"gnorm={gnorm:.3e}  nonfinite_total={nonfinite_steps}"
            )

    # Score both checkpoints, train/extrap/full split -- same convention as train.py
    def score(state_dict):
        model.load_state_dict(state_dict)
        with torch.no_grad():
            pred = node(y0=y0, t=t_full).cpu().numpy()
        y = y_full.cpu().numpy()
        return {
            "train_mse": compute_mse(y[:n_train], pred[:n_train]),
            "extrap_mse": compute_mse(y[n_train:], pred[n_train:]),
            "extrap_r2": compute_r2_score(y[n_train:], pred[n_train:]),
            "extrap_rel_l2": compute_relative_l2_error(y[n_train:], pred[n_train:]),
            "full_mse": compute_mse(y, pred),
        }

    final_metrics = score(model.state_dict())
    best_metrics = score(best_state) if best_state is not None else final_metrics

    # basis_func recorded as a STRING here -- never the object itself (see SS3).
    config = {
        "dataset": dataset, "basis_func": "hybrid_softmax_bspline_rbf",
        "layers_hidden": layers, "grid_len": grid_len, "lr": lr, "epochs": epochs,
        "grad_clip": grad_clip, "seed": seed, "parameters": total_p,
    }
    metrics = {
        "config": config,
        "selection": {"criterion": "min_train_mse", "best_epoch": best_epoch},
        "best": best_metrics, "final": final_metrics,
        "nonfinite_grad_steps": nonfinite_steps,
        "final_blend_weights": {"alpha": alpha_hist[-1], "beta": beta_hist[-1]},
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump({"train_losses": train_losses, "grad_norms": grad_norms,
                   "alpha": alpha_hist, "beta": beta_hist}, f)
    torch.save({"model_state_dict": best_state, "config": config},
              os.path.join(save_dir, "best_model.pt"))

    print(f"\n{dataset}: best train={best_metrics['train_mse']:.4e} "
          f"full={best_metrics['full_mse']:.4e} "
          f"final blend alpha={alpha_hist[-1]:.3f} beta={beta_hist[-1]:.3f}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["lotka_volterra", "damped_pendulum"], required=True)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--grid_len", type=int, default=5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=500,
                    help="print a clean, file-log-friendly status line every N epochs")
    args = ap.parse_args()
    run(args.dataset, args.epochs, args.lr, args.grid_len, args.grad_clip,
        args.save_dir, args.seed, log_every=args.log_every)
```

---

## 7. Standalone checkpoint loader (since `evaluate.py` can't be used — §3)

```python
# experiments/hybrid_basis/load_hybrid.py
"""[TRACK C] Reconstruct a HybridBasis checkpoint for inspection/plotting."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                            # this dir, so `hybrid_basis` is importable
sys.path.insert(0, os.path.join(_HERE, "..", ".."))   # implementation/, so `kan` is importable

import torch
from kan import KAN
from hybrid_basis import HybridBasis

def load(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    hybrid = HybridBasis(grid_len=cfg["grid_len"])
    model = KAN(layers_hidden=cfg["layers_hidden"], grid_len=cfg["grid_len"], basis_func=hybrid)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
```

---

## 8. Execution plan

### Stage 1 — Probe (short budget, sanity-check the gate is learning anything)

```powershell
cd D:\level4\Term1\NUM_project\kinetic-kan\implementation
python experiments\hybrid_basis\run_hybrid.py --dataset lotka_volterra --epochs 2000 --save_dir results\phase3\hybrid_basis\probe_lv
python experiments\hybrid_basis\run_hybrid.py --dataset damped_pendulum --epochs 2000 --save_dir results\phase3\hybrid_basis\probe_pendulum
```

> `experiments/` and `results/` are both direct children of `implementation/` (per
> [`12`](./12_phase3_roadmap.md) §2.3's layout), so every command in this section is
> written for `cwd = implementation/` with **no** `..\` prefix. This was a real bug in
> an earlier draft — a scratch-mirror test accidentally validated a sibling layout
> instead of this nested one; caught by re-testing at the real repo path before running
> anything expensive. See `run_hybrid.py`'s own `sys.path` comment for the fix.

Check before continuing: does `training_history.json`'s `alpha`/`beta` arrays actually
*move* from the 0.5/0.5 start? If they sit frozen at 0.5/0.5 the whole run, something
is wrong (e.g. gradients not reaching `blend_logits` — re-check that `hybrid` is the
literal object passed into `KAN(basis_func=hybrid)`, not a copy).

> ⚠️ **Revised timing — the §Load-Distribution table's per-epoch costs (0.17s LV,
> 0.77s pendulum) are for a *single* basis. Hybrid computes B-spline *and* RBF every
> forward pass, so it costs more.** Calibrated directly (100-epoch runs, this machine):
>
> | System | Measured hybrid cost | Probe (2,000 ep) | Full (10,000 ep) |
> | :--- | :---: | :---: | :---: |
> | Lotka-Volterra | $0.87$ s/epoch | $\approx 29$ min | $\approx 2.4$ h |
> | Damped pendulum | $2.54$ s/epoch | $\approx 85$ min | $\approx 7.1$ h |
>
> Pendulum alone is a **multi-hour** commitment at full budget — plan accordingly
> (background/overnight), and treat the probe stage as a real checkpoint, not a
> formality, before committing to it.

### Stage 2 — Full budget (only after Stage 1 looks sane)

Both systems run the full 10,000-epoch budget, matching Table 2's methodology exactly
(a deliberate choice — see the discussion above: `docs/09` found more epochs made the
*plain* pendulum worse, so the full run for the hybrid pendulum is not a foregone
"more is better" default, but running it anyway keeps this result directly comparable
to Table 2's epoch-matched numbers).

Each command redirects to a log file with `*> file.log` (PowerShell's all-streams
redirect — `tqdm`'s bar goes to stderr, so a plain `>` alone would miss it):

```powershell
python experiments\hybrid_basis\run_hybrid.py --dataset lotka_volterra --epochs 10000 --log_every 500 --save_dir results\phase3\hybrid_basis\lv_full *> results\phase3\hybrid_basis\lv_full.log
python experiments\hybrid_basis\run_hybrid.py --dataset damped_pendulum --epochs 10000 --log_every 500 --save_dir results\phase3\hybrid_basis\pendulum_full *> results\phase3\hybrid_basis\pendulum_full.log
```

Run them in **separate PowerShell windows** (or with `Start-Process` — see below) so
both progress simultaneously rather than the second waiting for the first to finish.

**Checking progress while they run.** The log file will contain both `tqdm`'s raw
`\r`-updated bar (unreadable as a normal file — every update overwrites the same
visual line) *and* the clean `[dataset] epoch N/epochs ...` lines from `--log_every`
(a real one-line-per-checkpoint status, added specifically because the raw bar output
is not useful once redirected). Filter to just the clean lines:

```powershell
Get-Content results\phase3\hybrid_basis\lv_full.log -Raw |
    Select-String -Pattern '\[lotka_volterra\] epoch.*' -AllMatches |
    ForEach-Object { $_.Matches.Value }
```

> Verified — `-Pattern '\[lotka_volterra\] epoch'` alone (no trailing `.*`) only
> returns the matched substring itself (`"[lotka_volterra] epoch"`, nothing after),
> since `.Matches.Value` is exactly what the pattern matched. The `.*` is required to
> capture the rest of the line; `.` doesn't match past the line's own `\n`, so it stops
> correctly and doesn't swallow the next raw `tqdm` bar chunk.

Or, to watch it update live as the run progresses (same idea as `tail -f`):

```powershell
Get-Content results\phase3\hybrid_basis\lv_full.log -Wait -Tail 20 |
    Select-String -Pattern '\[lotka_volterra\] epoch.*'
```

**Running both in the background from one window**, if you'd rather not open two:

```powershell
Start-Process python -ArgumentList "experiments\hybrid_basis\run_hybrid.py --dataset lotka_volterra --epochs 10000 --log_every 500 --save_dir results\phase3\hybrid_basis\lv_full" -RedirectStandardOutput results\phase3\hybrid_basis\lv_full.log -RedirectStandardError results\phase3\hybrid_basis\lv_full.err.log -NoNewWindow
Start-Process python -ArgumentList "experiments\hybrid_basis\run_hybrid.py --dataset damped_pendulum --epochs 10000 --log_every 500 --save_dir results\phase3\hybrid_basis\pendulum_full" -RedirectStandardOutput results\phase3\hybrid_basis\pendulum_full.log -RedirectStandardError results\phase3\hybrid_basis\pendulum_full.err.log -NoNewWindow
```

`Start-Process` splits stdout/stderr into two files (`tqdm`'s bar and the clean
`log_every` lines both go to stderr, so `pendulum_full.err.log` is the one to watch);
the `*>` form above is simpler for a single foreground run in its own window. Note:
`Start-Process` here is the *actual* command being run (not something I'm running for
you) — you're launching it yourself, from your own PowerShell window.

### Stage 3 — The headline plot: $\alpha(t)$, $\beta(t)$

```powershell
python experiments\hybrid_basis\plot_blend.py
```

Run with `cwd = implementation/` (same as everything else). It walks all four possible
tags (`probe_lv`, `probe_pendulum`, `lv_full`, `pendulum_full`), skipping any not run
yet, and writes `results/phase3/hybrid_basis/<tag>_alpha_beta.png` for each that exists.

### Stage 4 — Compare against Table 2 (read-only citation, do not re-run)

Run with `cwd = implementation/`, same convention as every other command in this doc:

```python
import json
rbf = json.load(open("results/benchmarks/ablation_activations/basis_rbf/metrics.json"))
bsp = json.load(open("results/benchmarks/ablation_activations/basis_bspline/metrics.json"))
hyb = json.load(open("results/phase3/hybrid_basis/lv_full/metrics.json"))
for name, m in [("RBF", rbf), ("B-spline", bsp), ("Hybrid", hyb)]:
    b = m["best"]
    print(f"{name:10s} train={b['train_mse']:.3e}  extrap={b['extrap_mse']:.3e}  R2={b.get('extrap_r2', float('nan')):.4f}")
```

Read the numbers **live** from these files rather than copying figures out of
[`05`](./05_phase2_benchmark_analysis.md) by hand — avoids transcription drift.

---

## 9. Definition of done

| Check | Target |
| :--- | :--- |
| Gate weights logged every epoch, both systems | not just final values |
| Compared against **existing** Table 2 numbers | citation via Stage 4's script, not a wasted re-run |
| Explicit verdict | faster / same / worse than the better pure basis, stated numerically |
| 242 vs. 240 parameter count noted wherever compared | not silently glossed over |

## 10. Files you must not touch

Everything outside `experiments/hybrid_basis/`, `results/phase3/hybrid_basis/`,
`tests/test_p3_hybrid_basis.py`, this file. In particular: **do not edit
`kan/basis.py`** or `train.py` — §3's workaround exists specifically so neither is
necessary.

---

## 11. Findings *(fill in after running)*

### Stage 1 probe result

*(alpha/beta at epoch 1 vs. epoch 2000, both systems — did the gate move?)*

### Stage 2 full-budget result

*(table: Hybrid vs. RBF vs. B-spline — train MSE, extrap MSE, R², wall-clock)*

### Verdict

*(faster / same / worse than the better pure basis — with numbers)*

### Anything unexpected

*(e.g. did alpha/beta converge to one basis, split evenly, or oscillate?)*
