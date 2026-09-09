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
# "implementation/" (the package root holding kan/, ode/, data/, utils/) is exactly
# two levels up -- no extra "implementation" suffix needed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from kan import KAN, count_parameters
from ode import NeuralODE
from data import generate_lotka_volterra_data, generate_damped_pendulum_data
from utils import (
    compute_mse, compute_rmse, compute_mae,
    compute_r2_score, compute_relative_l2_error, compute_gradient_norm,
)
from hybrid_basis import HybridBasis


def run(dataset, epochs, lr=None, grid_len=None, grad_clip=1.0, save_dir=None,
        seed=42, device="cpu", log_every=500):
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
          f"grad_clip={grad_clip}  seed={seed}")

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
    ap.add_argument("--lr", type=float, default=None,
                    help="omit to use the validated per-dataset default (2e-3 LV, 3e-3 pendulum)")
    ap.add_argument("--grid_len", type=int, default=None,
                    help="omit to use the validated per-dataset default (G=5 LV, G=8 pendulum)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=500,
                    help="print a clean, file-log-friendly status line every N epochs")
    args = ap.parse_args()
    run(args.dataset, args.epochs, args.lr, args.grid_len, args.grad_clip,
        args.save_dir, args.seed, log_every=args.log_every)
