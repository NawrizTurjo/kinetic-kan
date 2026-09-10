# 🧬 Track C — Learnable Softmax Hybrid Basis: Implementation Guideline & Findings

> **Owner:** Shams Hossain Simanto (2105048) · **Branch:** `feat/p3-hybrid-basis`
> **Folder:** `experiments/hybrid_basis/` · **Results:** `results/phase3/hybrid_basis/`
> **Status:** ✅ **complete**, including a follow-up (§11a). Both 10,000-epoch
> full-budget runs finished (LV in 3h25m, pendulum in 12h31m, run concurrently).
> LV: clean, near-parity with pure bases. Pendulum: overfits badly if trained the
> full 10k (extrap R² −2.17 vs. pure RBF's +0.647) — but a follow-up run capped at
> the identified pre-overfit sweet spot (3,000 epochs, §11a) shows hybrid is
> actually competitive with pure RBF at a matched epoch count. The result is
> budget-dependent, not a flat "hybrid is worse."
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

**This section is the final, as-shipped version** — it has been through two rounds of
fixes since the first draft (§3b's `lr`/`grid_len` bug, §7b's missing plots, §10b's
`blend_lr_mult` gate-speed fix) plus one dead-code cleanup (an unused `import time`,
found during the pre-10k re-audit in §10c and removed). The full `tests/` suite (172
tests) and a post-cleanup 5-epoch smoke test on both datasets were both re-run clean
against this exact version before it was trusted for the 10,000-epoch runs.

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

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
# This file lives at implementation/experiments/hybrid_basis/run_hybrid.py, so
# "implementation/" (the package root holding kan/, ode/, data/, utils/) is exactly
# two levels up -- no extra "implementation" suffix needed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from kan import KAN, count_parameters
from ode import NeuralODE
from data import generate_lotka_volterra_data, generate_damped_pendulum_data
from utils import (
    compute_mse, compute_r2_score, compute_relative_l2_error, compute_gradient_norm,
    plot_trajectory_comparison, plot_phase_space, plot_loss_curves, plot_gradient_norm_dynamics,
)
from hybrid_basis import HybridBasis


def run(dataset, epochs, lr=None, grid_len=None, grad_clip=1.0, save_dir=None,
        seed=42, device="cpu", log_every=500, blend_lr_mult=1.0):
    """
    lr and grid_len default to None so a dataset-specific validated value can be
    supplied automatically (below) WITHOUT silently overriding an explicit CLI value
    -- the bug this replaced. `grid_len` previously did override unconditionally
    (`grid_len = 8` regardless of what was passed for damped_pendulum); `lr` did not
    override at all, so the pendulum branch silently ran at LV's lr=2e-3 instead of
    the [09]-validated 3e-3 whenever --lr was omitted. Both are now resolved the same
    way: use the CLI value if one was given, else the validated per-dataset default.
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset == "lotka_volterra":
        data = generate_lotka_volterra_data(seed=seed)           # defaults match Table 2
        layers = [2, 10, 2]
        if lr is None:
            lr = 2e-3        # Table 2's own lr
        if grid_len is None:
            grid_len = 5      # Table 2's own G
    elif dataset == "damped_pendulum":
        # [09]-fixed recipe: t_train_end=5.0, default SiLU base_act, G=8, lr=3e-3.
        data = generate_damped_pendulum_data(t_train_end=5.0, seed=seed)
        layers = [2, 10, 2]
        if lr is None:
            lr = 3e-3         # [09] pendulum_control_win5's validated lr -- NOT 2e-3
        if grid_len is None:
            grid_len = 8      # [09] pendulum_control_win5's validated G
    else:
        raise ValueError(dataset)

    # Print the RESOLVED config immediately -- this is what a silent lr/grid_len
    # mismatch (the actual bug this replaced) looked like: nothing wrong printed at
    # launch, and the only trace was inside metrics.json's config block afterwards,
    # findable only by comparing it against docs/09's own recipe after the fact.
    print(f"[{dataset}] resolved config: lr={lr}  grid_len={grid_len}  epochs={epochs}  "
          f"grad_clip={grad_clip}  seed={seed}  blend_lr_mult={blend_lr_mult}")

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

    # [GATE-FIX] blend_lr_mult=1.0 (default) is a no-op: a single param group at
    # `lr`, identical to plain Adam(model.parameters(), lr=lr). >1.0 gives
    # blend_logits its own, faster-moving group -- targets a specific evidenced
    # bottleneck (see §10b): the gate was moving in the right direction
    # (toward RBF) but slowly, still 12.6% B-spline at epoch 2000, exactly the
    # window where the pure-RBF pendulum recipe was already near-converged
    # (train MSE 0.0023 at the same epoch, vs 0.271 here). B-spline was never
    # validated on the pendulum anywhere in this project, so prolonged exposure
    # to it during that critical window is the leading suspect. This does not
    # bias the gate's DIRECTION -- it still starts neutral (0.5/0.5) and is
    # fully gradient-driven -- only how fast it can move.
    if blend_lr_mult != 1.0:
        blend_id = id(hybrid.blend_logits)
        other_params = [p for p in model.parameters() if id(p) != blend_id]
        optimizer = torch.optim.Adam([
            {"params": other_params, "lr": lr},
            {"params": [hybrid.blend_logits], "lr": lr * blend_lr_mult},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses, alpha_hist, beta_hist, grad_norms = [], [], [], [], []
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

        # Periodic full-horizon monitor loss, same cadence as train.py
        # (epoch%10==0 or first/last epoch) -- NOT every epoch, since a full
        # extra t_full integration every step would roughly double training cost
        # on top of the training-window pass already done above.
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            with torch.no_grad():
                test_loss_val = F.mse_loss(node(y0=y0, t=t_full), y_full).item()
        else:
            test_loss_val = test_losses[-1] if test_losses else float("inf")
        test_losses.append(test_loss_val)

        pbar.set_postfix({"loss": f"{loss_val:.3e}", "a": f"{a:.3f}", "b": f"{b:.3f}"})

        # [LOG] tqdm's \r-updated bar is nearly unreadable once redirected to a file
        # (every update lands on the same visual line, so a plain `> file.log` capture
        # is one giant carriage-return-separated blob). tqdm.write() emits a normal
        # newline-terminated line instead -- readable both live and in a redirected
        # log file, and safe to interleave with the bar (that's what it's for).
        if epoch % log_every == 0 or epoch == epochs:
            tqdm.write(
                f"[{dataset}] epoch {epoch}/{epochs}  loss={loss_val:.4e}  "
                f"best={best_loss:.4e}@{best_epoch}  alpha={a:.4f}  beta={b:.4f}  "
                f"gnorm={gnorm:.3e}  nonfinite_total={nonfinite_steps}"
            )

    # Score both checkpoints, train/extrap/full split -- same convention as train.py.
    # Returns (metrics_dict, pred) -- pred is reused below for the plots so scoring
    # the same checkpoint twice (once for numbers, once for plots) is avoided.
    def score(state_dict):
        model.load_state_dict(state_dict)
        with torch.no_grad():
            pred = node(y0=y0, t=t_full).cpu().numpy()
        y = y_full.cpu().numpy()
        metrics_dict = {
            "train_mse": compute_mse(y[:n_train], pred[:n_train]),
            "extrap_mse": compute_mse(y[n_train:], pred[n_train:]),
            "extrap_r2": compute_r2_score(y[n_train:], pred[n_train:]),
            "extrap_rel_l2": compute_relative_l2_error(y[n_train:], pred[n_train:]),
            "full_mse": compute_mse(y, pred),
        }
        return metrics_dict, pred

    final_metrics, final_pred = score(model.state_dict())
    best_metrics, best_pred = score(best_state) if best_state is not None else (final_metrics, final_pred)

    # basis_func recorded as a STRING here -- never the object itself (see SS3).
    config = {
        "dataset": dataset, "basis_func": "hybrid_softmax_bspline_rbf",
        "layers_hidden": layers, "grid_len": grid_len, "lr": lr, "epochs": epochs,
        "grad_clip": grad_clip, "seed": seed, "parameters": total_p,
        "blend_lr_mult": blend_lr_mult,
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
        json.dump({"train_losses": train_losses, "test_losses": test_losses,
                   "grad_norms": grad_norms, "alpha": alpha_hist, "beta": beta_hist}, f)
    torch.save({"model_state_dict": best_state, "config": config},
              os.path.join(save_dir, "best_model.pt"))

    # [PLOTS] previously missing entirely -- run_hybrid.py never called any of the
    # utils.plotting functions train.py's train_kan_ode() calls automatically, so a
    # completed run produced only best_model.pt/metrics.json/training_history.json
    # and nothing visual. Reusing the SAME plotting utilities every other run in the
    # project uses (not reimplementing them) keeps hybrid-basis figures directly
    # comparable to Table 2's existing per-basis plots.
    labels = (("Prey ($x$)", "Predator ($y$)") if dataset == "lotka_volterra"
              else (r"Angle $\theta$", r"Angular velocity $\omega$"))
    plot_trajectory_comparison(
        t_full=t_full.cpu().numpy(), y_true=y_full.cpu().numpy(), y_pred=best_pred,
        t_split=data.t_split, labels=labels,
        title=f"Hybrid Basis: Trajectory Comparison ({dataset})",
        save_path=os.path.join(save_dir, "trajectory_comparison.png"))
    plot_phase_space(
        y_true=y_full.cpu().numpy(), y_pred=best_pred, train_len=n_train, labels=labels,
        title=f"Hybrid Basis: Phase Portrait ({dataset})",
        save_path=os.path.join(save_dir, "phase_space.png"))
    plot_loss_curves(
        train_losses=train_losses, test_losses=test_losses,
        title=f"Hybrid Basis: Training/Monitor Loss ({dataset})",
        save_path=os.path.join(save_dir, "loss_curves.png"))
    plot_gradient_norm_dynamics(
        grad_norms=grad_norms,
        title=f"Hybrid Basis: Gradient Norm Dynamics ({dataset})",
        save_path=os.path.join(save_dir, "gradient_norm_dynamics.png"))
    # alpha/beta gate evolution -- the one plot specific to this track, not part of
    # utils.plotting since no other track has a blend gate to visualize.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(alpha_hist, label=r"$\alpha$ (B-spline weight)", color="#1f77b4")
    plt.plot(beta_hist, label=r"$\beta$ (RBF weight)", color="#d62728")
    plt.xlabel("epoch"); plt.ylabel("softmax weight"); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"Hybrid Basis: Gate Evolution ({dataset})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "alpha_beta_evolution.png"), dpi=200)
    plt.close()

    print(f"\n{dataset}: best train={best_metrics['train_mse']:.4e} "
          f"full={best_metrics['full_mse']:.4e} "
          f"final blend alpha={alpha_hist[-1]:.3f} beta={beta_hist[-1]:.3f}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["lotka_volterra", "damped_pendulum"], required=True)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=None,
                    help="omit to use the validated per-dataset default (2e-3 LV, 3e-3 pendulum)")
    ap.add_argument("--grid_len", type=int, default=None,
                    help="omit to use the validated per-dataset default (G=5 LV, G=8 pendulum)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=500,
                    help="print a clean, file-log-friendly status line every N epochs")
    ap.add_argument("--blend_lr_mult", type=float, default=1.0,
                    help="1.0 (default) = no-op, single lr for all params. >1.0 gives "
                         "blend_logits its own faster lr = base_lr * blend_lr_mult "
                         "(see §10b for why this exists)")
    args = ap.parse_args()
    run(args.dataset, args.epochs, args.lr, args.grid_len, args.grad_clip,
        args.save_dir, args.seed, log_every=args.log_every,
        blend_lr_mult=args.blend_lr_mult)
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

## 7b. Plots — added after the probe stage, backfilled onto the probes themselves

`run_hybrid.py` originally wrote only `best_model.pt` / `metrics.json` /
`training_history.json` — unlike `train_kan_ode()`, it never called any of
`utils.plotting`'s functions, so a completed run produced no figures at all. Fixed:
every run now also writes (reusing the exact same plotting utilities every other run
in the project uses, not reimplementing them):

| File | From |
| :--- | :--- |
| `trajectory_comparison.png` | `utils.plot_trajectory_comparison` |
| `phase_space.png` | `utils.plot_phase_space` |
| `loss_curves.png` | `utils.plot_loss_curves` (needed adding `test_losses` tracking — a periodic full-horizon monitor loss, same `epoch%10==0` cadence as `train.py`, **not** every epoch, since that would roughly double compute) |
| `gradient_norm_dynamics.png` | `utils.plot_gradient_norm_dynamics` |
| `alpha_beta_evolution.png` | track-specific, inlined (no other track has a blend gate) |

**The two already-completed probes predate this fix** and were backfilled without
retraining via `experiments/hybrid_basis/regenerate_plots.py` (loads the saved
checkpoint + history, re-integrates, calls the same 5 plot functions):

```powershell
python experiments\hybrid_basis\regenerate_plots.py --run_dir results\phase3\hybrid_basis\probe_lv
python experiments\hybrid_basis\regenerate_plots.py --run_dir results\phase3\hybrid_basis\probe_pendulum
```

**What the phase portraits actually show, beyond the MSE numbers:** LV's predicted
orbit (train + 4 extrapolated periods) visually overlaps the true orbit almost
exactly. The pendulum's does not — the training-window prediction collapses into a
narrow, wrong-shaped oscillation rather than tracking the true decaying spiral, and
the extrapolation stays tangled near the origin rather than resolving to a clean
inward spiral. This is consistent with, and more vivid than, the "still descending,
not yet converged" read from the loss curve alone (§11 below).

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

**Pendulum uses `--blend_lr_mult 15`** — the §10b gate-speed fix, confirmed at the full
2,000-epoch probe scale (24× train-MSE improvement, phase portrait visually corrected
from a broken shape to a clean matching spiral) before being carried into this full run.
**LV does not** — its probe never showed the pendulum's slow-gate problem (the opposite,
if anything: hybrid beat pure RBF 14× at epoch 2,000 there), so this run is a
straight scale-up of `probe_lv` with nothing changed.

Each command redirects to a log file with `*> file.log` (PowerShell's all-streams
redirect — `tqdm`'s bar goes to stderr, so a plain `>` alone would miss it):

```powershell
python experiments\hybrid_basis\run_hybrid.py --dataset lotka_volterra --epochs 10000 --log_every 500 --save_dir results\phase3\hybrid_basis\lv_full *> results\phase3\hybrid_basis\lv_full.log
python experiments\hybrid_basis\run_hybrid.py --dataset damped_pendulum --epochs 10000 --log_every 500 --blend_lr_mult 15 --save_dir results\phase3\hybrid_basis\pendulum_full *> results\phase3\hybrid_basis\pendulum_full.log
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

**Running both in parallel from one window — same pattern as `run_phase2.ps1`**
(thread-pinning + `Start-Process -PassThru` + `.WaitForExit()` on the captured
`Process` object, not a bare PID — see that script's own `[FIX-2026-08]` comment
on why PID-based `Wait-Process` is unsafe once a job exits and Windows recycles
the PID):

```powershell
$root = "D:\level4\Term1\NUM_project\kinetic-kan\implementation"
cd $root

$logDir = "results\phase3\hybrid_basis\_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Thread pinning, same reasoning as run_phase2.ps1: this machine has 12 logical
# cores; PyTorch grabs every core by default, so 2 unpinned parallel jobs would
# oversubscribe the CPU and inflate both wall-clock numbers. 12/2 = 6 each.
$env:OMP_NUM_THREADS      = 6
$env:MKL_NUM_THREADS      = 6
$env:OPENBLAS_NUM_THREADS = 6

$lv = Start-Process python -ArgumentList "experiments\hybrid_basis\run_hybrid.py --dataset lotka_volterra --epochs 10000 --log_every 500 --save_dir results\phase3\hybrid_basis\lv_full" -WorkingDirectory $root -NoNewWindow -PassThru -RedirectStandardOutput "$logDir\lv_full.log" -RedirectStandardError "$logDir\lv_full.err.log"

$pend = Start-Process python -ArgumentList "experiments\hybrid_basis\run_hybrid.py --dataset damped_pendulum --epochs 10000 --log_every 500 --blend_lr_mult 15 --save_dir results\phase3\hybrid_basis\pendulum_full" -WorkingDirectory $root -NoNewWindow -PassThru -RedirectStandardOutput "$logDir\pendulum_full.log" -RedirectStandardError "$logDir\pendulum_full.err.log"

Write-Host "Launched lv_full (pid $($lv.Id)) and pendulum_full (pid $($pend.Id))"

$lv.WaitForExit()
$pend.WaitForExit()
Write-Host "Both runs finished."
```

**Verified, not assumed** (a redirect-split smoke test with `tqdm.write()` +
a real bar): `tqdm.write()` — the clean `[dataset] epoch N/epochs ...` `log_every`
lines — goes to **stdout**; the raw `\r`-updated bar goes to **stderr**. So
`lv_full.log`/`pendulum_full.log` (stdout) contain *only* the clean
one-line-per-checkpoint status — no bar noise to filter out at all — while the
`.err.log` files catch the raw bar, useful only if you want to watch it move
live. `-WorkingDirectory` is required since `Start-Process` does not reliably
inherit the calling shell's current directory otherwise. Note: this command is
the *actual* thing being run (not something run on your behalf) — you launch it
yourself, from your own PowerShell window, and it returns your prompt only after
both jobs finish (because of the two `.WaitForExit()` calls) — open a second
window if you want to keep using this one meanwhile.

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

| Check | Target | Status |
| :--- | :--- | :---: |
| Gate weights logged every epoch, both systems | not just final values | ✅ §11 |
| Compared against **existing** Table 2 numbers | citation via Stage 4's script, not a wasted re-run | ✅ §11 |
| Explicit verdict | faster / same / worse than the better pure basis, stated numerically | ✅ §11 Verdict |
| 242 vs. 240 parameter count noted wherever compared | not silently glossed over | ✅ §11, §5 |

## 10. Files you must not touch

Everything outside `experiments/hybrid_basis/`, `results/phase3/hybrid_basis/`,
`tests/test_p3_hybrid_basis.py`, this file. In particular: **do not edit
`kan/basis.py`** or `train.py` — §3's workaround exists specifically so neither is
necessary.

---

## 10b. ⚠️ Pendulum-specific finding — the probe result was misleading, and a targeted fix

The Stage 1 sanity checks (§8: gate moved, no NaN, loss descending) **passed** for the
pendulum probe, but passing those checks turned out not to mean the result was healthy.
Direct comparison against `pendulum_control_win5` (pure RBF, same recipe, known to
converge to $R^2=0.647$ by epoch 10,000) at the **same epoch count**:

| Epoch | Pure RBF pendulum (`pendulum_control_win5`) | Hybrid pendulum (probe) |
| :---: | :---: | :---: |
| 1,000 | $0.0847$ | $0.328$ |
| 2,000 | $\mathbf{0.0023}$ | $\mathbf{0.271}$ — **117× worse** |

Pure RBF was essentially converged by epoch 2,000; the hybrid probe was nowhere close.
**This rules out "just needs the same 10,000 epochs pure RBF took"** — pure RBF didn't
need that long. The same comparison on Lotka-Volterra shows the *opposite* pattern
(hybrid beats pure RBF $14\times$ at epoch 2,000), so this is not a general
"blending costs early speed" effect — it's pendulum-specific.

**Leading hypothesis:** B-spline was never validated on the pendulum anywhere in this
project (Phase 2's basis ablation only ran on Lotka-Volterra). The gate starts neutral
and only slowly shifts toward RBF — still 12.6% B-spline at epoch 2,000 in the probe —
so if B-spline genuinely doesn't suit the pendulum's dynamics, that contamination during
exactly the window where pure RBF was racing to convergence is the most likely drag.

**Fix — `--blend_lr_mult`:** give `blend_logits` its own Adam parameter group at
`lr × blend_lr_mult`, leaving every other parameter (and the validated `lr=0.003`,
`G=8`, `t_train_end=5.0`, everything else) untouched:

```python
if blend_lr_mult != 1.0:
    blend_id = id(hybrid.blend_logits)
    other_params = [p for p in model.parameters() if id(p) != blend_id]
    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": lr},
        {"params": [hybrid.blend_logits], "lr": lr * blend_lr_mult},
    ])
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)   # default: unchanged
```

This targets *speed*, not *direction* — the gate is already moving the right way
(toward RBF), just slowly. It still starts neutral (0.5/0.5) and is fully
gradient-driven; nothing is hand-biased toward an assumed answer.

**Verified before the real run:**
- `blend_lr_mult=1.0` (default) reproduces the original 3-epoch smoke test's `train_mse`
  **exactly** — zero regression.
- Parameter-group split covers all 362 parameters with no overlap (360 + 2), confirmed
  by direct inspection of `optimizer.param_groups`.
- A 50-epoch functional test at `blend_lr_mult=15` reached $\alpha=0.170$ (83% RBF) —
  **more RBF-dominant than the original probe reached by epoch 200** ($\alpha=0.415$).
  No instability introduced, `nonfinite_total=0` throughout.
- Full `tests/` suite (172 tests) re-confirmed clean after the change.

**Re-run confirmed the fix works, at the real 2,000-epoch probe scale (not just the
50-epoch functional test above):** repeated with `--blend_lr_mult 15`, saved to
`results/phase3/hybrid_basis/probe_pendulum_gatefix/`, separate from the original
(unfixed) `probe_pendulum` so both remain available for direct comparison.

| | Unfixed `probe_pendulum` | Fixed `probe_pendulum_gatefix` | Pure RBF (`pendulum_control_win5`, for reference) |
| :--- | :---: | :---: | :---: |
| Best train MSE (2,000 ep) | $0.2715$ | $\mathbf{0.01108}$ — **24× better** | $0.0023$ |
| Final blend weights | $\alpha{=}0.126,\ \beta{=}0.874$ | $\alpha{=}0.002,\ \beta{=}0.998$ | — |
| `nonfinite_grad_steps` | $0$ | $0$ | — |

The fixed run's train MSE closed most of the gap to pure RBF (from $117\times$ worse to
$\approx 5\times$ worse at matched epochs), and the gate essentially fully committed to
RBF ($99.8\%$). **Visually confirmed too, not just numerically:** the fixed run's
`phase_space.png` shows a clean, correctly-decaying spiral tracking the true orbit
closely through the training window — a completely different shape from the original
probe's collapsed, wrong-shaped oscillation (§7b). The extrapolation segment still shows
a phase lag from the true orbit (consistent with `extrap_r2` still being slightly
negative, $-0.02$, at the *min-train-mse* checkpoint) — but note this same
negative-`extrap_r2`-at-the-best-checkpoint pattern is present in the **unfixed** probe
too ($-0.05$), and that probe's own *final*-epoch checkpoint (not the min-train-mse one)
scores $+0.44$ — so this looks like an artifact of the min-train-mse selection criterion
picking an epoch that overfits the training window at some cost to extrapolation,
present in both runs, not something the gate fix introduced.

**Decision:** fix adopted for the full pendulum run (`--blend_lr_mult 15`, §8 Stage 2).
Not yet known whether it holds up at 10,000 epochs (e.g. whether the late-run gradient
norm upticks seen in both probes near epoch 2,000 grow into a real problem over 5× more
epochs) — that is exactly what the full run will determine.

---

## 10c. Final pre-10k recheck (everything double-checked before committing real compute)

Immediately before queuing the 10,000-epoch runs, did one more full pass end-to-end:

- **Full `tests/` suite re-run clean: 172/172 passed**, including all 10
  `test_p3_hybrid_basis.py` tests — no regression from any change made in §3b/§7b/§10b.
- **Line-by-line re-audit of all three Track C source files**
  (`hybrid_basis.py`, `run_hybrid.py`, `load_hybrid.py`) against the actual current
  files on disk (not from memory) — lr/grid_len resolution, the `blend_lr_mult`
  optimizer-group split, non-finite-gradient handling, best-checkpoint timing
  (`deepcopy` before `optimizer.step()`), JSON-safety of the saved `config` (never the
  raw `HybridBasis` object), and all 5 plot calls — no further problems found.
- **One dead-code cleanup:** an unused `import time` in `run_hybrid.py` (never called
  anywhere in the file) — removed. Not a bug, just noise found during the re-audit.
- **Post-cleanup smoke test, both datasets, 5 epochs each** (pendulum with
  `--blend_lr_mult 15`, matching what the real 10k pendulum run will use): both
  resolved configs printed correctly, gate moved on both, and all 8 expected output
  files (`metrics.json`, `training_history.json`, `best_model.pt`, and the 5 plots)
  were generated with no errors, then deleted (scratch-only, not committed).

No outstanding bugs identified. §8 Stage 2's commands are considered final as of this
recheck.

---

## 11. Findings — complete

### Stage 1 probe result — complete

Both systems' gates start at exactly $(\alpha,\beta)=(0.5,0.5)$ (unbiased, per
`HybridBasis`'s `init_logits=(0.0,0.0)`) and move measurably by epoch 2,000:

| System | Run | $\alpha$ (epoch 1) | $\alpha$ (epoch 2000, final) | Best train MSE |
| :--- | :--- | :---: | :---: | :---: |
| Lotka-Volterra | `probe_lv` | $0.500$ | $0.134$ | $0.001989$ |
| Damped pendulum | `probe_pendulum` (unfixed) | $0.500$ | $0.126$ | $0.2715$ |
| Damped pendulum | `probe_pendulum_gatefix` (`blend_lr_mult=15`) | $0.500$ | $0.002$ | $0.01108$ |

Both systems' gates move the **same direction** (toward RBF, $\beta \to 1$), but at very
different speeds and with very different consequences — LV converges well either way
(hybrid actually *beats* pure RBF there at matched epochs, §10b), while the pendulum's
un-accelerated gate speed measurably hurt convergence (§10b) until corrected. See §10b
for the full analysis and the fix; `probe_pendulum_gatefix` is the version carried
forward into the full 10,000-epoch pendulum run.

### Stage 2 full-budget result

Both 10,000-epoch runs completed cleanly (`nonfinite_grad_steps=0` in both),
launched concurrently per §8's parallel-launch pattern. Real wall-clock:
LV **3h25m**, pendulum **12h31m** (both inflated somewhat above their solo-calibrated
estimates — §8's revised timing table — by CPU contention from running together;
see the live discussion during the run for the arithmetic).

**Lotka-Volterra — Hybrid vs. the two pure bases (Table 2, read live via Stage 4's script):**

| Basis | Train MSE | Extrap MSE | Extrap R² | Params | Cost |
| :--- | :---: | :---: | :---: | :---: | :---: |
| RBF | $8.83\times10^{-5}$ | $8.92\times10^{-5}$ | $1.0000$ | $240$ | $0.171$ s/epoch (solo) |
| B-spline | $7.20\times10^{-5}$ | $8.38\times10^{-5}$ | $1.0000$ | $240$ | $0.668$ s/epoch (solo) |
| **Hybrid** | $1.47\times10^{-4}$ | $1.78\times10^{-4}$ | $0.9999$ | $242$ | $\approx 0.87$ s/epoch (solo) — roughly the **sum** of both pure bases' costs |

At the full 10,000-epoch budget, hybrid is essentially **tied with, but not better
than**, either pure basis on LV — both pure bases actually edge it out slightly once
fully converged (R² rounds to $1.0000$ for both vs. hybrid's $0.9999$). This is a
different picture than the 2,000-epoch probe, where hybrid appeared to *beat* pure
RBF (§10b) — at 2,000 epochs pure RBF hadn't yet finished converging, so hybrid's
apparent lead there was a mid-training snapshot, not a full-budget result. Cost is the
real story: hybrid pays for both bases every forward pass, so its per-epoch cost is
close to the *sum* of the two pure bases, not a saving.

**Damped pendulum — Hybrid vs. pure RBF (no B-spline reference exists for the pendulum
— Phase 2's basis ablation only ran on Lotka-Volterra):**

| | Hybrid (`pendulum_full`, `blend_lr_mult=15`) | Pure RBF (`pendulum_control_win5`, same recipe) |
| :--- | :---: | :---: |
| Best train MSE | $9.44\times10^{-5}$ (matches pure RBF) | $9.28\times10^{-5}$ |
| Extrap MSE | $\mathbf{0.809}$ | $0.090$ — **9× better** |
| Extrap R² | $\mathbf{-2.17}$ | $+0.647$ |
| Final gate | $\alpha{=}0.021,\ \beta{=}0.979$ | — |
| Params | $362$ | $360$ |

Train-set fit is essentially identical between the two — the hybrid network matches
pure RBF's training accuracy almost exactly. **Extrapolation is where it falls apart,**
and the *reason* is visible directly in the saved per-epoch history, not inferred:

- The full-horizon monitor loss (`training_history.json`'s `test_losses`, evaluated on
  train+extrap together every 10 epochs) reaches its own best value around
  **epoch ≈3,000** ($\approx 0.055$ — actually *better* than pure RBF's eventual number)
  — then **rises and plateaus around $0.4$** for the remaining $7{,}000$ epochs, while
  train loss keeps monotonically improving all the way to $9.4\times10^{-5}$. This is
  textbook overfitting to the training window, and it is *not* subtle — see
  `pendulum_full/loss_curves.png`.
- The `min_train_mse` checkpoint-selection criterion (the same convention used
  everywhere else in this project, `train.py` included) picks epoch $9{,}987$ — deep
  inside the overfit region — which is exactly why the headline "best" number above is
  so bad. **No earlier checkpoint was retained** (`run_hybrid.py` only keeps a
  `deepcopy` at the epoch with the lowest train loss seen *so far*, which by definition
  is a late epoch once training has run this long) — so the actual weights at the
  epoch-3,000 sweet spot are not recoverable from this run without re-running.
- **The gate itself is not the cause.** `pendulum_full/alpha_beta_evolution.png` shows
  $\beta$ reaching $\approx 0.995$ by epoch $\approx 250$ and staying there (drifting
  only slightly back to $0.979$ after epoch $6{,}000$) — the gate had long since settled
  into a near-pure-RBF configuration well before the overfitting onset at epoch 3,000,
  so this is not a gate-oscillation artifact.
- **Pure RBF, on the identical recipe, does not show this pattern.** Its own
  `test_losses` history stays bounded between $0.02$ and $0.09$ across the *entire*
  $10{,}000$ epochs — no comparable blow-up. So a network that is $\approx 98\%$ RBF by
  final weight, having arrived there via a hybrid gate, generalizes measurably worse
  over a long run than a network that was pure RBF from initialization, despite
  matching its training fit almost exactly.
- Visually confirmed in `phase_space.png`: the training-window prediction overlaps the
  true orbit almost perfectly; the extrapolated trajectory collapses into a small,
  incorrect loop instead of continuing the true decaying spiral in toward the origin.
- Confirmed **not** an LV-style pattern — `lv_full/loss_curves.png` shows train and
  extrapolation loss decreasing together in lockstep for the full 10,000 epochs, no
  divergence at any point.

### Verdict

- **Lotka-Volterra: no benefit, no harm, real cost.** Hybrid ties pure RBF/B-spline on
  final accuracy (R² 0.9999 vs. 1.0000) but costs roughly the **sum** of both bases'
  per-epoch compute ($\approx 0.87$ vs. $0.17$–$0.67$ s/epoch solo) for **242
  parameters instead of 240**. The gate lands at $\alpha{=}0.116/\beta{=}0.884$ — a
  genuine partial blend, not a collapse to one basis — but that blend does not
  translate into an accuracy edge at full budget. **Not worth its cost on this system.**
- **Damped pendulum: negative result at the full 10k budget, but budget-dependent —
  see the §11a follow-up below.** At the `min_train_mse` selection convention used
  throughout this project, the full 10,000-epoch hybrid run is **9× worse** than pure
  RBF on extrapolation MSE and swings from a healthy positive R² ($+0.647$, pure RBF)
  to strongly negative ($-2.17$, hybrid) — despite matching pure RBF's training fit
  almost exactly and despite the `blend_lr_mult` gate-speed fix working exactly as
  designed (confirmed at 2,000-epoch probe scale, §10b). The failure mode is a
  late-training overfitting collapse in extrapolation quality that the equivalent
  pure-RBF network does not exhibit on the same recipe — **not** the
  originally-hypothesized "slow gate" problem, which the fix did correctly resolve.
  **However**, a follow-up run capped at the pre-overfit sweet spot (3,000 epochs, §11a)
  shows hybrid is actually **competitive with, and briefly ahead of, pure RBF at a
  matched epoch count** — so the full-budget number above is not the whole story; it
  is specifically what happens if this recipe is trained *past* its optimum.
- **Overall: the learnable hybrid basis does not outperform the better pure basis on
  either system tested, and on the pendulum specifically it is measurably worse and in
  a way not yet fully explained.** The 2,000-epoch probe stage's optimistic read (§10b:
  "hybrid catches up to pure RBF") was a correct description of *training* dynamics at
  that budget, but did not anticipate the full-budget overfitting divergence — a second
  instance in this track (after §3b's `lr` bug) of a short-budget signal not
  generalizing to the full run, this time for a genuine dynamical reason rather than a
  configuration bug.

### Anything unexpected

- **The gate does converge, decisively, on both systems** — LV to a genuine partial
  blend ($\alpha{=}0.116$), pendulum to near-total RBF ($\alpha{=}0.021$) — confirming
  the softmax gate is learnable and gradient-driven exactly as designed, on both
  datasets, at full budget. This part of the mechanism works correctly.
- **The most unexpected result: matching pure RBF's training accuracy did not mean
  matching its extrapolation quality**, on the one system (pendulum) where the two
  bases' behavior differs most. A network that is $\approx 98\%$ RBF by final weight
  behaves measurably differently, over a long training run, than one that was $100\%$
  RBF from initialization — despite an essentially identical training loss trajectory
  for most of the run. Plausible contributing factors, **not verified, listed as open
  questions for future work**: (a) the residual $\sim 2\%$ B-spline contribution,
  though tiny, is still receiving gradients throughout training and may act as a slow
  destabilizing perturbation rather than a neutral no-op; (b) the extra `blend_logits`
  parameters change the loss landscape's local geometry near this solution even after
  they've stopped moving much; (c) this specific run may simply be an unlucky
  seed/trajectory — untested here, since only `seed=42` was run for either system (no
  multi-seed error bars for Track C, unlike Phase 2's `-Seeds` sweeps).
- **A practical lesson about checkpointing:** saving only the `min_train_mse` snapshot
  (mirroring `train.py`'s own convention) means that when a run overfits like this
  pendulum one did, the actually-good intermediate solution is unrecoverable after the
  fact. A useful extension for any follow-up work on this track would be to also
  checkpoint on `min` full-horizon monitor loss, not just train loss — this would have
  let this run report both numbers instead of only the overfit one. Not implemented
  here since it would change `run_hybrid.py`'s saved-checkpoint semantics without a
  clear signal beforehand that this system specifically would need it (LV never showed
  the pattern that would have motivated this).

## 11a. Follow-up — pendulum at the pre-overfit sweet spot (3,000 epochs)

§11's "anything unexpected" section flagged that the `pendulum_full` run's own
full-horizon monitor loss actually bottomed out around epoch 2,950–3,000 (value
$\approx 0.054$, briefly *better* than pure RBF's number), but that no checkpoint was
saved there — only the `min_train_mse` snapshot from deep in the overfit region (epoch
9,987) survives from that run.

**Confirmed the exact minimum, from the saved per-epoch history, before re-running
anything:** `argmin` of `pendulum_full/training_history.json`'s `test_losses` is epoch
**2,950**, value $0.0542$ — climbing sharply again by epoch 3,200 ($0.129$). A fresh run
was launched, same recipe as `pendulum_full` (`lr=0.003`, `grid_len=8`,
`blend_lr_mult=15`) but capped at **3,000 epochs**, saved separately to
`results/phase3/hybrid_basis/pendulum_3k/` — `pendulum_full`, `probe_pendulum`, and
`probe_pendulum_gatefix` are all untouched.

| | `pendulum_3k` (3,000 ep, hybrid) | Pure RBF **at the same epoch** (`pendulum_control_win5`'s own history, epoch 3,000) | Pure RBF at full budget (10,000 ep, for reference) |
| :--- | :---: | :---: | :---: |
| Full-horizon MSE | $\mathbf{0.0553}$ | $0.0808$ | $0.0448$ |
| Extrap MSE | $0.107$ | — | $0.090$ |
| Extrap R² | $\mathbf{+0.580}$ | — | $+0.647$ |
| Final gate | $\alpha{=}0.004,\ \beta{=}0.996$ | — | — |

**At a matched epoch count, hybrid actually beats pure RBF's own epoch-3,000 number**
($0.0553$ vs. $0.0808$ full-horizon MSE), and its extrapolation R² ($+0.580$) is close
to pure RBF's fully-converged, full-10k-budget number ($+0.647$) — a completely
different picture from the catastrophic $-2.17$ the same recipe produces at 10,000
epochs. Visually confirmed too: `pendulum_3k/phase_space.png` shows the training-window
prediction tracking the true orbit closely, and — unlike `pendulum_full`'s collapsed
extrapolation loop — the extrapolated trajectory follows the *correct* inward-decaying
shape, even though it doesn't overlap the true orbit as tightly as pure RBF's
full-budget result does.

**What this changes about the verdict:** the pendulum result is not simply "hybrid is
worse" — it is **budget-dependent**. Hybrid is competitive with pure RBF up to roughly
epoch 3,000 on this recipe, then overfits severely if training continues to 10,000
epochs, in a way pure RBF itself does not. Pure RBF's own advantage at full budget is
therefore not really about basis quality — it's about which system tolerates the full
10,000-epoch budget without overfitting its extrapolation behavior. This refines, rather
than reverses, §11's overall verdict: hybrid still does not *beat* the better pure basis
on this system at any budget tested, but the size and cause of the gap depends heavily
on when training is stopped, and the fix from §10b (`blend_lr_mult`) demonstrably works
correctly at both the 2,000-epoch and 3,000-epoch scales — the divergence is specific to
prolonged training, not to the fix or the gate.

---

### Deliverable files (per `docs/12` §Track C)

Generated after both 10k runs completed, no new training required (`pendulum_3k` is the
one exception — a genuinely new, cheap 3,000-epoch run added after the 10k runs to
capture the pre-overfit checkpoint documented in §11a):

- `results/phase3/hybrid_basis/table.json` — the full comparison table above, in
  machine-readable form (hybrid vs. RBF vs. B-spline where it exists, both systems, plus
  the §11a epoch-matched pendulum comparison), built directly from each run's own
  `metrics.json` (Stage 4's script, written to disk instead of just printed).
- `results/phase3/hybrid_basis/{probe_lv,probe_pendulum,lv_full,pendulum_full,pendulum_3k}_alpha_beta.png`
  — the gate-trajectory plot for every completed run, via `plot_blend.py` (§8 Stage 3).
  Per-run copies (`alpha_beta_evolution.png`) also exist inside each run's own
  subdirectory, generated automatically by `run_hybrid.py` itself (§7b) — these
  top-level copies are the same data, just collected in one place per `docs/12`'s
  named deliverable path.
