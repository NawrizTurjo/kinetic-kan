"""
Track E shared plumbing: import paths, checkpoint rebuilding, integration.

Nothing here is novel; it exists so that `run_sindy_noise.py`, `run_epidemic_fit.py`
and `pruning.py` do not each re-derive the same three chores. In particular
`load_kan_checkpoint` reproduces the rebuild contract that `evaluate.py` and
`phase2_closeout.py` both implement -- honouring `grid_lims`, `conserve_mode`,
`vanish_dim` and `time_scale` from the SAVED config rather than the current
defaults. `docs/10_sir_root_cause_and_fix.md` Part 5 catalogues what silently
breaks when a loader skips that step, which is why it is centralised here.

Owner: Monjur Hossain Khan (Shovon), 2105043 -- Phase 3 Track E.
"""

import os
import sys

# implementation/ is the import root for `kan`, `ode`, `data`, `utils`.
IMPL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if IMPL_ROOT not in sys.path:
    sys.path.insert(0, IMPL_ROOT)

import numpy as np
import torch

from kan import KAN
from ode import NeuralODE, ZeroSumField, VanishingDimField

# Every artifact this track produces lands here and nowhere else (roadmap 2.3).
RESULTS_DIR = os.path.join(IMPL_ROOT, "results", "phase3", "sindy_epidemic")

# Read-only inputs owned by earlier phases.
NOISE_SWEEP_DIR = os.path.join(IMPL_ROOT, "results", "benchmarks", "noise")

# The four sigma levels Phase 2 swept for KAN, in the exact directory spelling
# used on disk. Track E reuses these so the comparison is apples-to-apples.
NOISE_LEVELS = [
    (0.00, "sigma0"),
    (0.01, "sigma0.01"),
    (0.05, "sigma0.05"),
    (0.10, "sigma0.1"),
]


def ensure_results_dir(subdir: str = "") -> str:
    """Create (if needed) and return this track's output directory."""
    path = os.path.join(RESULTS_DIR, subdir) if subdir else RESULTS_DIR
    os.makedirs(path, exist_ok=True)
    return path


def build_field(model: torch.nn.Module, config: dict) -> torch.nn.Module:
    """
    Re-apply the structural field wrappers a checkpoint was TRAINED under.

    Projection and gating are properties of the vector field, not of the weights,
    so `load_state_dict` alone rebuilds a different ODE than the run integrated.
    """
    field = model
    if config.get("conserve_mode") == "projection":
        field = ZeroSumField(field)
    if config.get("vanish_dim") is not None:
        field = VanishingDimField(field, dim=int(config["vanish_dim"]))
    return field


def load_kan_checkpoint(checkpoint_path: str):
    """
    Rebuild the exact KAN a checkpoint was trained as.

    Returns:
        (model, config) -- the bare KAN module (so pruning can reach `.layers`)
        and the run config verbatim. Wrap with `build_field` before integrating.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    if config.get("model_type", "kan").lower() != "kan":
        raise ValueError(f"{checkpoint_path} is not a KAN checkpoint.")

    model = KAN(
        layers_hidden=config.get("layers_hidden", [2, 10, 2]),
        grid_len=config.get("grid_len", 5),
        # Pre-FIX-2026-08 checkpoints (the noise sweep among them) carry no
        # `grid_lims` key at all; (-1, 1) is what they were actually trained on.
        grid_lims=tuple(config.get("grid_lims", [-1.0, 1.0])),
        basis_func=config.get("basis_func", "rbf"),
        normalizer=config.get("normalizer", "tanh"),
        base_act=config.get("base_act", "silu"),
        use_base_act=config.get("use_base_act", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def integrate(model: torch.nn.Module, config: dict, y0: torch.Tensor,
              t: torch.Tensor, solver: str = None) -> np.ndarray:
    """
    Integrate a rebuilt checkpoint on the clock it was trained on.

    The learned field is g = time_scale * f, so it must be integrated over
    tau = t / time_scale. Predictions still come back at the PHYSICAL sample
    times in `t`, which is what keeps every metric comparable with the runs
    already published in docs/05.
    """
    field = build_field(model, config)
    node = NeuralODE(
        func=field,
        method=solver or config.get("solver", "tsit5"),
        substeps=config.get("substeps", 2),
    )
    time_scale = float(config.get("time_scale", 1.0) or 1.0)
    with torch.no_grad():
        return node(y0=y0, t=t / time_scale).cpu().numpy()


def split_mse(y_true: np.ndarray, y_pred: np.ndarray, n_train: int) -> dict:
    """
    Split the error the same three ways `train.py` does, so Track E's numbers
    sit directly alongside the project's existing tables without translation.
    """
    finite = np.isfinite(y_pred).all()
    if not finite:
        return {"train_mse": float("nan"), "extrap_mse": float("nan"),
                "full_mse": float("nan"), "finite": False}
    err = (y_true - y_pred) ** 2
    return {
        "train_mse": float(err[:n_train].mean()),
        "extrap_mse": float(err[n_train:].mean()),
        "full_mse": float(err.mean()),
        "finite": True,
    }
