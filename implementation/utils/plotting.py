"""
Publication-Grade Scientific Machine Learning (SciML) Plotting Suite.

Provides publication-quality visualizations for dynamical systems, Neural ODEs, and KANs:
1. 1D Time-Series Trajectory Fits & Extrapolation splits (`plot_trajectory_comparison`).
2. 2D Phase Portraits with continuous vector field streamlines (`plot_phase_portrait_with_streamlines`).
3. 2D Classical Phase Space orbits (`plot_phase_space`).
4. 3D Chaotic Strange Attractor projections (`plot_3d_lorenz_trajectory`).
5. Non-linear Pendulum Phase & Hamiltonian Energy dissipation (`plot_pendulum_phase_and_energy`).
6. Multi-Model Loss Decay Comparisons with convergence thresholds (`plot_model_comparison_curves`).
7. Continuous Parameter Gradient Norm dynamics (`plot_gradient_norm_dynamics`).
8. Quantitative Benchmark ablation bar charts (`plot_benchmark_comparison`).
"""

import os
from typing import Optional, List, Dict, Tuple, Union, Callable
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# Configure Matplotlib publication defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.45,
    "grid.linestyle": "--",
})


def plot_trajectory_comparison(
    t_full: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t_split: float = 3.5,
    labels: Tuple[str, str] = ("Prey ($x$)", "Predator ($y$)"),
    title: str = "Trajectory Comparison (Train vs. Extrapolation)",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot ground truth vs predicted multi-variable trajectories over time with train/test split.
    """
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    
    # Ground truth lines
    ax.plot(t_full, y_true[:, 0], color="#2ca02c", linewidth=2.2, label=f"True {labels[0]}")
    ax.plot(t_full, y_true[:, 1], color="#1f77b4", linewidth=2.2, label=f"True {labels[1]}")
    
    # Predicted lines
    ax.plot(t_full, y_pred[:, 0], color="#2ca02c", linestyle="--", linewidth=2.0, label=f"Predicted {labels[0]}")
    ax.plot(t_full, y_pred[:, 1], color="#1f77b4", linestyle="--", linewidth=2.0, label=f"Predicted {labels[1]}")
    
    # Train / Test split vertical line
    ax.axvline(x=t_split, color="black", linestyle=":", linewidth=1.8, label=f"Split ($t={t_split}$)")
    
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("State Magnitude")
    ax.grid(True)
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_phase_space(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_len: int,
    labels: Tuple[str, str] = ("Prey ($x$)", "Predator ($y$)"),
    title: str = "Phase Portrait ($x$ vs $y$)",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot 2D Phase Space orbit showing train and extrapolation segments.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=150)
    
    ax.plot(y_true[:, 0], y_true[:, 1], "k-", linewidth=2.2, label="True Orbit")
    ax.plot(y_pred[:train_len, 0], y_pred[:train_len, 1], color="#ff7f0e", linewidth=2.0, label="Predicted (Train)")
    ax.plot(y_pred[train_len - 1:, 0], y_pred[train_len - 1:, 1], color="#d62728", linestyle="--", linewidth=2.0, label="Predicted (Extrapolation)")
    
    ax.scatter([y_true[0, 0]], [y_true[0, 1]], color="blue", s=70, zorder=5, label="Initial State $\\mathbf{u}_0$")
    
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(True)
    ax.legend(frameon=True)
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_phase_portrait_with_streamlines(
    vector_field_fn: Callable[[np.ndarray], np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_len: int,
    x_range: Tuple[float, float] = (0.0, 4.0),
    y_range: Tuple[float, float] = (0.0, 4.0),
    grid_density: int = 25,
    labels: Tuple[str, str] = ("Prey ($x$)", "Predator ($y$)"),
    title: str = "Phase Portrait with Vector Field Streamlines",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot 2D Phase Portrait with continuous vector field streamlines in the background.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    
    # 1. Create 2D Evaluation Grid
    x = np.linspace(x_range[0], x_range[1], grid_density)
    y = np.linspace(y_range[0], y_range[1], grid_density)
    X, Y = np.meshgrid(x, y)
    
    # Stack into points [N*N, 2]
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    derivs = vector_field_fn(grid_points)
    
    U = derivs[:, 0].reshape(X.shape)
    V = derivs[:, 1].reshape(Y.shape)
    speed = np.sqrt(U**2 + V**2)
    
    # 2. Render Streamlines
    strm = ax.streamplot(
        X, Y, U, V,
        color=speed,
        cmap="Blues",
        density=1.1,
        linewidth=0.9,
        arrowsize=1.0,
    )
    fig.colorbar(strm.lines, ax=ax, label="Vector Field Velocity $\\|\\mathbf{f}(\\mathbf{u})\\|_2$")
    
    # 3. Overlay Trajectories
    ax.plot(y_true[:, 0], y_true[:, 1], "k-", linewidth=2.2, label="True Orbit")
    ax.plot(y_pred[:train_len, 0], y_pred[:train_len, 1], color="#ff7f0e", linewidth=2.2, label="Predicted (Train)")
    ax.plot(y_pred[train_len - 1:, 0], y_pred[train_len - 1:, 1], color="#d62728", linestyle="--", linewidth=2.2, label="Predicted (Extrapolation)")
    
    # 4. Mark Initial State
    ax.scatter([y_true[0, 0]], [y_true[0, 1]], color="blue", s=80, zorder=5, label="Initial State $\\mathbf{u}_0$")
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_3d_lorenz_trajectory(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_len: int,
    title: str = "3D Chaotic Lorenz Attractor Trajectory",
    elev: float = 25.0,
    azim: float = 45.0,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot 3D multi-angle chaotic attractor geometry showing true vs predicted strange attractor.
    """
    fig = plt.figure(figsize=(9, 7.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    
    # True trajectory
    ax.plot(y_true[:, 0], y_true[:, 1], y_true[:, 2], color="black", alpha=0.5, linewidth=1.2, label="True Strange Attractor")
    
    # Predicted Train
    ax.plot(y_pred[:train_len, 0], y_pred[:train_len, 1], y_pred[:train_len, 2], color="#ff7f0e", linewidth=1.8, label="Predicted (Train)")
    
    # Predicted Extrapolation
    ax.plot(y_pred[train_len - 1:, 0], y_pred[train_len - 1:, 1], y_pred[train_len - 1:, 2], color="#d62728", linestyle="--", linewidth=1.6, label="Predicted (Extrapolation)")
    
    # Initial Condition
    ax.scatter([y_true[0, 0]], [y_true[0, 1]], [y_true[0, 2]], color="blue", s=80, label="Initial State $\\mathbf{u}_0$")
    
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel("State $x(t)$")
    ax.set_ylabel("State $y(t)$")
    ax.set_zlabel("State $z(t)$")
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_pendulum_phase_and_energy(
    t_full: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    energy_true: np.ndarray,
    energy_pred: np.ndarray,
    train_len: int,
    title: str = "Damped Pendulum: Phase Portrait & Energy Dissipation",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    2-panel publication figure for Damped Pendulum:
    - Left: Phase portrait (Angle theta vs Angular Velocity omega).
    - Right: Mechanical Energy E(t) dissipation over time.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    
    # Left Panel: Phase Space
    ax1.plot(y_true[:, 0], y_true[:, 1], "k-", linewidth=2.0, label="True Spiral")
    ax1.plot(y_pred[:train_len, 0], y_pred[:train_len, 1], color="#ff7f0e", linewidth=1.8, label="Predicted (Train)")
    ax1.plot(y_pred[train_len - 1:, 0], y_pred[train_len - 1:, 1], color="#d62728", linestyle="--", linewidth=1.8, label="Predicted (Extrapolation)")
    ax1.scatter([y_true[0, 0]], [y_true[0, 1]], color="blue", s=60, label="$\\mathbf{u}_0$")
    ax1.set_title("Phase Portrait $(\\theta \\text{ vs } \\omega)$", fontweight="bold")
    ax1.set_xlabel("Angle $\\theta$ (rad)")
    ax1.set_ylabel("Angular Velocity $\\omega$ (rad/s)")
    ax1.grid(True)
    ax1.legend(frameon=True)
    
    # Right Panel: Energy Dissipation
    ax2.plot(t_full, energy_true, "k-", linewidth=2.0, label="True Energy $E(t)$")
    ax2.plot(t_full, energy_pred, color="#d62728", linestyle="--", linewidth=1.8, label="Predicted Energy $\\hat{E}(t)$")
    ax2.axvline(x=t_full[train_len - 1], color="gray", linestyle=":", linewidth=1.5, label="Train/Test Split")
    ax2.set_title("Total Mechanical Energy $E(\\theta, \\omega)$", fontweight="bold")
    ax2.set_xlabel("Time $t$ (s)")
    ax2.set_ylabel("Energy (Joules)")
    ax2.grid(True)
    ax2.legend(frameon=True)
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_loss_curves(
    train_losses: List[float],
    test_losses: List[float],
    title: str = "Training and Test Loss",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot single model training and test loss curves on semilog scale.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    epochs = np.arange(1, len(train_losses) + 1)
    
    ax.semilogy(epochs, train_losses, color="#1f77b4", linewidth=2.0, label="Train Loss (MSE)")
    ax.semilogy(epochs, test_losses, color="#d62728", linewidth=2.0, linestyle="--", label="Test Extrapolation Loss (MSE)")
    
    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (Log Scale)")
    ax.grid(True, which="both")
    ax.legend(frameon=True)
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_model_comparison_curves(
    histories: Dict[str, Dict[str, List[float]]],
    metric_key: str = "test_losses",
    target_threshold: Optional[float] = 3e-5,
    title: str = "Loss Decay Comparison: KAN-ODE vs. MLP-ODE Baseline",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Overlay multi-line semilog loss curves comparing KAN-ODE configurations vs MLP-ODE baselines.
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-."]
    
    for idx, (model_name, data) in enumerate(histories.items()):
        losses = data[metric_key]
        epochs = np.arange(1, len(losses) + 1)
        color = colors[idx % len(colors)]
        style = linestyles[idx % len(linestyles)]
        ax.semilogy(epochs, losses, label=model_name, color=color, linestyle=style, linewidth=2.0)
        
    if target_threshold is not None:
        ax.axhline(
            y=target_threshold,
            color="black",
            linestyle=":",
            linewidth=1.8,
            label=f"Target $\\text{{MSE}} = {target_threshold:.0e}$",
        )
        
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Extrapolation Test Loss (MSE, Log Scale)")
    ax.grid(True, which="both")
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_gradient_norm_dynamics(
    grad_norms: List[float],
    title: str = "Gradient Norm Dynamics ($\\Vert \\nabla_\\theta \\mathcal{L} \\Vert_2$)",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot parameter gradient L2 norm dynamics over training epochs on log-linear scale.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    epochs = np.arange(1, len(grad_norms) + 1)
    
    ax.semilogy(epochs, grad_norms, color="#9467bd", linewidth=1.8, label=r"$\Vert \nabla_\theta \mathcal{L} \Vert_2$")
    
    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Gradient Norm $\Vert \nabla_\theta \mathcal{L} \Vert_2$ (Log Scale)")
    ax.grid(True, which="both")
    ax.legend(frameon=True)
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


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
    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=150)
    
    names = list(results.keys())
    values = [results[k][metric] for k in names]
    
    bars = ax.bar(names, values, color="#4c72b0", edgecolor="black", alpha=0.85)
    ax.set_yscale("log")
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.15,
            f"{height:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_ylabel(f"{metric.replace('_', ' ').upper()} (Log Scale)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.grid(True, which="both", axis="y")
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
