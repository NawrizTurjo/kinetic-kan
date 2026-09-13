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
    # bottleneck (see docs/15 SS11a): the gate was moving in the right direction
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
                         "(see docs/15 SS11a for why this exists)")
    args = ap.parse_args()
    run(args.dataset, args.epochs, args.lr, args.grid_len, args.grad_clip,
        args.save_dir, args.seed, log_every=args.log_every,
        blend_lr_mult=args.blend_lr_mult)
