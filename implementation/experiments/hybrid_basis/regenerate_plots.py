"""
[TRACK C] Regenerate the 5 standard plots for an ALREADY-COMPLETED run, from its
saved best_model.pt + training_history.json -- no retraining.

Exists because run_hybrid.py did not call any plotting functions until it was fixed
(see docs/15 SS3b); this backfills plots for runs completed before that fix (the two
probes) without spending the compute to retrain them.

Usage (run from implementation/):
    python experiments/hybrid_basis/regenerate_plots.py --run_dir results/phase3/hybrid_basis/probe_lv
    python experiments/hybrid_basis/regenerate_plots.py --run_dir results/phase3/hybrid_basis/probe_pendulum
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt

from load_hybrid import load
from data import generate_lotka_volterra_data, generate_damped_pendulum_data
from ode import NeuralODE
from utils import plot_trajectory_comparison, plot_phase_space, plot_loss_curves, plot_gradient_norm_dynamics


def main(run_dir):
    model, cfg = load(os.path.join(run_dir, "best_model.pt"))
    dataset = cfg["dataset"]

    if dataset == "lotka_volterra":
        data = generate_lotka_volterra_data(seed=cfg["seed"])
        labels = ("Prey ($x$)", "Predator ($y$)")
    elif dataset == "damped_pendulum":
        data = generate_damped_pendulum_data(t_train_end=5.0, seed=cfg["seed"])
        labels = (r"Angle $\theta$", r"Angular velocity $\omega$")
    else:
        raise ValueError(dataset)

    node = NeuralODE(func=model, method="tsit5", substeps=2)
    with torch.no_grad():
        pred = node(y0=data.y0, t=data.t_full).numpy()
    y_full = data.y_full.numpy()
    n_train = len(data.t_train)

    hist = json.load(open(os.path.join(run_dir, "training_history.json")))
    # Older training_history.json (from before the SS3b/plotting fix) has no
    # test_losses; fall back to train_losses so plot_loss_curves still has two
    # equal-length series rather than crashing on a missing key.
    test_losses = hist.get("test_losses", hist["train_losses"])

    plot_trajectory_comparison(
        t_full=data.t_full.numpy(), y_true=y_full, y_pred=pred,
        t_split=data.t_split, labels=labels,
        title=f"Hybrid Basis: Trajectory Comparison ({dataset})",
        save_path=os.path.join(run_dir, "trajectory_comparison.png"))
    plot_phase_space(
        y_true=y_full, y_pred=pred, train_len=n_train, labels=labels,
        title=f"Hybrid Basis: Phase Portrait ({dataset})",
        save_path=os.path.join(run_dir, "phase_space.png"))
    plot_loss_curves(
        train_losses=hist["train_losses"], test_losses=test_losses,
        title=f"Hybrid Basis: Training/Monitor Loss ({dataset})",
        save_path=os.path.join(run_dir, "loss_curves.png"))
    plot_gradient_norm_dynamics(
        grad_norms=hist["grad_norms"],
        title=f"Hybrid Basis: Gradient Norm Dynamics ({dataset})",
        save_path=os.path.join(run_dir, "gradient_norm_dynamics.png"))

    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(hist["alpha"], label=r"$\alpha$ (B-spline weight)", color="#1f77b4")
    plt.plot(hist["beta"], label=r"$\beta$ (RBF weight)", color="#d62728")
    plt.xlabel("epoch"); plt.ylabel("softmax weight"); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"Hybrid Basis: Gate Evolution ({dataset})")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "alpha_beta_evolution.png"), dpi=200)
    plt.close()

    print(f"Regenerated 5 plots in {run_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. results/phase3/hybrid_basis/probe_lv")
    args = ap.parse_args()
    main(args.run_dir)
