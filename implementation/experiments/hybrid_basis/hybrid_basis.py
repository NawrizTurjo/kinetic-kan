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
