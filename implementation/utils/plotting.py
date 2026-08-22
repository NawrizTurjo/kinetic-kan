import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from typing import Optional, List, Dict


def plot_trajectory_comparison(
    t_full: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t_split: float = 3.5,
    title: str = "KAN-ODE Predator-Prey Trajectory",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot ground truth vs KAN-ODE predicted trajectories over time.
    """
    plt.figure(figsize=(10, 5), dpi=150)
    
    # Ground truth
    plt.plot(t_full, y_true[:, 0], color="#2ca02c", linewidth=2.2, label="Prey (True $x$)")
    plt.plot(t_full, y_true[:, 1], color="#1f77b4", linewidth=2.2, label="Predator (True $y$)")
    
    # Prediction
    plt.plot(t_full, y_pred[:, 0], color="#2ca02c", linestyle="--", linewidth=2.0, label="Prey (KAN-ODE $\hat{x}$)")
    plt.plot(t_full, y_pred[:, 1], color="#1f77b4", linestyle="--", linewidth=2.0, label="Predator (KAN-ODE $\hat{y}$)")
    
    # Train / Test split marker
    plt.axvline(x=t_split, color="black", linestyle=":", linewidth=1.8, label=f"Train/Test Split ($t={t_split}$)")
    
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Time $t$", fontsize=12)
    plt.ylabel("Population Density", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper right", fontsize=10)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_phase_space(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_len: int,
    title: str = "Phase Portrait ($x$ vs $y$)",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot Phase Space (Prey vs Predator).
    """
    plt.figure(figsize=(7, 6), dpi=150)
    
    # True orbit
    plt.plot(y_true[:, 0], y_true[:, 1], "k-", linewidth=2.2, label="True Orbit")
    
    # KAN-ODE Training portion
    plt.plot(y_pred[:train_len, 0], y_pred[:train_len, 1], color="#ff7f0e", linewidth=2.0, label="KAN-ODE (Train segment)")
    # KAN-ODE Extrapolation portion
    plt.plot(y_pred[train_len-1:, 0], y_pred[train_len-1:, 1], color="#d62728", linestyle="--", linewidth=2.0, label="KAN-ODE (Extrapolation)")
    
    # Start point
    plt.scatter([y_true[0, 0]], [y_true[0, 1]], color="blue", s=80, zorder=5, label="Initial State $(x_0, y_0)$")
    
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Prey Population ($x$)", fontsize=12)
    plt.ylabel("Predator Population ($y$)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_loss_curves(
    train_losses: List[float],
    test_losses: List[float],
    title: str = "Training and Test Loss",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot loss curves on semilogarithmic scale.
    """
    plt.figure(figsize=(8, 4.5), dpi=150)
    epochs = np.arange(1, len(train_losses) + 1)
    
    plt.semilogy(epochs, train_losses, color="#1f77b4", linewidth=2.0, label="Train Loss (MSE)")
    plt.semilogy(epochs, test_losses, color="#d62728", linewidth=2.0, linestyle="--", label="Test / Extrapolation Loss (MSE)")
    
    plt.title(title, fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Squared Error (Log Scale)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_benchmark_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "test_mse",
    title: str = "Benchmark Comparison",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot benchmark results across different configurations as a bar chart.
    """
    plt.figure(figsize=(9, 5), dpi=150)
    
    names = list(results.keys())
    values = [results[k][metric] for k in names]
    
    bars = plt.bar(names, values, color="#4c72b0", edgecolor="black", alpha=0.85)
    plt.yscale("log")
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.15,
            f"{height:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )
        
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.ylabel(f"{metric.replace('_', ' ').upper()} (Log Scale)", fontsize=11)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
