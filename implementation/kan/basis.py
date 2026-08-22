import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def rbf(x: torch.Tensor, grid: torch.Tensor, h: float) -> torch.Tensor:
    """
    Gaussian Radial Basis Function (RBF) as used in the paper:
    phi(x) = exp(-((x - z) / h)^2)
    
    Args:
        x: Normalized input tensor of shape (*, in_features)
        grid: Grid centers of shape (grid_len,) or (in_features, grid_len)
        h: Grid step / denominator (float)
        
    Returns:
        Tensor of shape (*, in_features, grid_len)
    """
    # x: [..., I] -> [..., I, 1]
    x_expanded = x.unsqueeze(-1)
    # y = (x - grid) / h
    y = (x_expanded - grid) / h
    return torch.exp(-(y ** 2))


def rswaf(x: torch.Tensor, grid: torch.Tensor, h: float) -> torch.Tensor:
    """
    Reflectional SWitch Activation Function (RSWAF):
    phi(x) = sech((x - z) / h)^2 = 1 - tanh((x - z) / h)^2
    
    Args:
        x: Normalized input tensor of shape (*, in_features)
        grid: Grid centers
        h: Grid denominator
    """
    x_expanded = x.unsqueeze(-1)
    y = (x_expanded - grid) / h
    return 1.0 - torch.tanh(y) ** 2


def iqf(x: torch.Tensor, grid: torch.Tensor, h: float) -> torch.Tensor:
    """
    Inverse Quadratic Function (IQF):
    phi(x) = 1 / (1 + ((x - z) / h)^2)
    
    Args:
        x: Normalized input tensor of shape (*, in_features)
        grid: Grid centers
        h: Grid denominator
    """
    x_expanded = x.unsqueeze(-1)
    y = (x_expanded - grid) / h
    return 1.0 / (1.0 + y ** 2)


def bspline_basis(x: torch.Tensor, grid: torch.Tensor, h: float = None, k: int = 3) -> torch.Tensor:
    """
    B-Spline basis functions for comparison.
    
    Args:
        x: Input tensor of shape (*, in_features)
        grid: Base grid centers of shape (grid_len,)
        h: Grid step
        k: Spline order (default 3 = cubic)
    """
    if h is None:
        h = (grid[-1] - grid[0]).item() / (len(grid) - 1) if len(grid) > 1 else 1.0
        
    # Extend grid on both ends for degree k splines
    left_padding = grid[0] - torch.arange(k, 0, -1, device=grid.device, dtype=grid.dtype) * h
    right_padding = grid[-1] + torch.arange(1, k + 1, device=grid.device, dtype=grid.dtype) * h
    full_grid = torch.cat([left_padding, grid, right_padding])
    
    x_expanded = x.unsqueeze(-1)
    # 0th order basis (step functions)
    bases = ((x_expanded >= full_grid[:-1]) & (x_expanded < full_grid[1:])).to(x.dtype)
    
    # Cox-de Boor recursion
    for deg in range(1, k + 1):
        left_denom = full_grid[deg:-1] - full_grid[:-deg-1]
        right_denom = full_grid[deg+1:] - full_grid[1:-deg]
        
        left_term = (x_expanded - full_grid[:-deg-1]) / (left_denom + 1e-8) * bases[..., :-1]
        right_term = (full_grid[deg+1:] - x_expanded) / (right_denom + 1e-8) * bases[..., 1:]
        bases = left_term + right_term
        
    # Slice to match the expected grid_len dimensions
    return bases[..., :len(grid)]


BASIS_FUNCTIONS = {
    "rbf": rbf,
    "rswaf": rswaf,
    "iqf": iqf,
    "bspline": bspline_basis,
}


def get_basis_function(name: str):
    """Retrieve basis function by name."""
    name = name.lower()
    if name not in BASIS_FUNCTIONS:
        raise ValueError(f"Unknown basis function: '{name}'. Available: {list(BASIS_FUNCTIONS.keys())}")
    return BASIS_FUNCTIONS[name]
