"""
Kolmogorov-Arnold Network (KAN) Basis Functions Module.

This module provides localized, polynomial, and orthogonal basis function families
to parameterize the learnable 1D edge activation functions phi(x) in KDense layers:

1. Gaussian Radial Basis Functions (RBF)
2. Reflectional SWitch Activation Functions (RSWAF)
3. Inverse Quadratic Functions (IQF)
4. Cox-de Boor B-Splines (Degree k=3 / Cubic)
5. Chebyshev Polynomials of the First Kind (T_n(x))
6. Lagrange Cardinal Polynomials (L_i(x))
7. Newton's Divided Differences Polynomials (N_k(x))

All functions adhere to the unified signature:
    basis_fn(x: torch.Tensor, grid: torch.Tensor, h: float) -> torch.Tensor
returning shape: (*, in_features, grid_len).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable, Dict


# ==============================================================================
# 1. Localized Radial & Rational Kernel Bases
# ==============================================================================

def rbf(x: torch.Tensor, grid: torch.Tensor, h: float) -> torch.Tensor:
    """
    Gaussian Radial Basis Function (RBF):
        phi_i(x) = exp(-((x - z_i) / h)^2)
    
    Args:
        x: Normalized input tensor of shape (*, in_features)
        grid: Grid centers of shape (grid_len,)
        h: Grid step denominator (float)
        
    Returns:
        Tensor of shape (*, in_features, grid_len)
    """
    x_expanded = x.unsqueeze(-1)
    y = (x_expanded - grid) / h
    return torch.exp(-(y ** 2))


def rswaf(x: torch.Tensor, grid: torch.Tensor, h: float) -> torch.Tensor:
    """
    Reflectional SWitch Activation Function (RSWAF):
        phi_i(x) = sech((x - z_i) / h)^2 = 1 - tanh((x - z_i) / h)^2
    
    Args:
        x: Normalized input tensor of shape (*, in_features)
        grid: Grid centers of shape (grid_len,)
        h: Grid step denominator (float)
        
    Returns:
        Tensor of shape (*, in_features, grid_len)
    """
    x_expanded = x.unsqueeze(-1)
    y = (x_expanded - grid) / h
    return 1.0 - torch.tanh(y) ** 2


def iqf(x: torch.Tensor, grid: torch.Tensor, h: float) -> torch.Tensor:
    """
    Inverse Quadratic Function (IQF):
        phi_i(x) = 1 / (1 + ((x - z_i) / h)^2)
    
    Args:
        x: Normalized input tensor of shape (*, in_features)
        grid: Grid centers of shape (grid_len,)
        h: Grid step denominator (float)
        
    Returns:
        Tensor of shape (*, in_features, grid_len)
    """
    x_expanded = x.unsqueeze(-1)
    y = (x_expanded - grid) / h
    return 1.0 / (1.0 + y ** 2)


# ==============================================================================
# 2. Cox-de Boor B-Spline Basis
# ==============================================================================

def bspline_basis(x: torch.Tensor, grid: torch.Tensor, h: float = None, k: int = 3) -> torch.Tensor:
    """
    Cox-de Boor B-Spline basis functions of degree k (default k=3 cubic splines).
    
    Mathematically:
        B_{i,0}(x) = 1 if t_i <= x < t_{i+1}, else 0
        B_{i,d}(x) = ((x - t_i) / (t_{i+d} - t_i)) * B_{i,d-1}(x)
                   + ((t_{i+d+1} - x) / (t_{i+d+1} - t_{i+1})) * B_{i+1,d-1}(x)
                   
    Args:
        x: Input tensor of shape (*, in_features)
        grid: Base grid knot centers of shape (grid_len,)
        h: Knot step spacing (float, optional)
        k: Spline polynomial degree (default: 3 = cubic)
        
    Returns:
        Tensor of shape (*, in_features, grid_len)
    """
    grid_len = grid.shape[0]
    if h is None or h <= 0:
        h = (grid[-1] - grid[0]).item() / max(grid_len - 1, 1)
        
    # Symmetrically pad k knots on the left and k knots on the right
    # Total knots = grid_len + 2*k
    left_padding = grid[0] - torch.arange(k, 0, -1, device=grid.device, dtype=grid.dtype) * h
    right_padding = grid[-1] + torch.arange(1, k + 1, device=grid.device, dtype=grid.dtype) * h
    knots = torch.cat([left_padding, grid, right_padding])
    
    x_expanded = x.unsqueeze(-1)  # [..., in_features, 1]
    
    # 0th-degree B-splines (piecewise constant intervals)
    is_in_interval = (x_expanded >= knots[:-1]) & (x_expanded < knots[1:])
    # Include right boundary endpoint
    is_in_interval[..., -1] = is_in_interval[..., -1] | (x_expanded[..., 0] >= knots[-1])
    bases = is_in_interval.to(x.dtype)
    
    # Cox-de Boor recursion for degrees 1 to k
    for deg in range(1, k + 1):
        num_bases = knots.shape[0] - deg - 1
        t_left = knots[:num_bases]
        t_right = knots[deg : deg + num_bases]
        t_next_left = knots[1 : num_bases + 1]
        t_next_right = knots[deg + 1 : deg + num_bases + 1]
        
        denom_left = t_right - t_left
        denom_right = t_next_right - t_next_left
        
        left_term = torch.where(
            denom_left > 1e-8,
            (x_expanded - t_left) / denom_left * bases[..., :-1],
            torch.zeros_like(bases[..., :-1])
        )
        right_term = torch.where(
            denom_right > 1e-8,
            (t_next_right - x_expanded) / denom_right * bases[..., 1:],
            torch.zeros_like(bases[..., 1:])
        )
        bases = left_term + right_term

    # Return exactly grid_len basis slices
    return bases[..., :grid_len]


# ==============================================================================
# 3. Orthogonal & Polynomial Bases (Chebyshev, Lagrange, Newton)
# ==============================================================================

def chebyshev_basis(x: torch.Tensor, grid: torch.Tensor, h: float = None) -> torch.Tensor:
    """
    Chebyshev Polynomial Basis of the First Kind: T_n(x).
    
    Recurrence Relation:
        T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2x * T_n(x) - T_{n-1}(x)
        
    Properties:
        - Orthogonal on [-1, 1] with weight w(x) = 1 / sqrt(1 - x^2)
        - Minimax optimal approximation minimizing Runge's boundary oscillation
        
    Args:
        x: Input tensor of shape (*, in_features) (mapped to [-1, 1])
        grid: Grid tensor of shape (grid_len,)
        h: Spacing parameter (unused, maintained for uniform API signature)
        
    Returns:
        Tensor of shape (*, in_features, grid_len)
    """
    grid_len = grid.shape[0]
    x_expanded = x.unsqueeze(-1)  # [..., in_features, 1]
    
    if grid_len == 1:
        return torch.ones_like(x_expanded)
    
    t_0 = torch.ones_like(x_expanded)
    t_1 = x_expanded
    
    cheb_list = [t_0, t_1]
    for _ in range(2, grid_len):
        t_next = 2.0 * x_expanded * cheb_list[-1] - cheb_list[-2]
        cheb_list.append(t_next)
        
    return torch.cat(cheb_list[:grid_len], dim=-1)


def lagrange_basis(x: torch.Tensor, grid: torch.Tensor, h: float = None) -> torch.Tensor:
    """
    Lagrange Cardinal Polynomial Basis:
        L_i(x) = prod_{j != i} (x - z_j) / (z_i - z_j)
        
    Properties:
        - Cardinal Property: L_i(z_j) = delta_{ij} (Kronecker delta)
        - Partition of Unity: sum_{i=0}^{G-1} L_i(x) = 1 for all x
        
    Args:
        x: Input tensor of shape (*, in_features)
        grid: Grid interpolation nodes of shape (grid_len,)
        h: Spacing parameter (unused, maintained for uniform API signature)
        
    Returns:
        Tensor of shape (*, in_features, grid_len)
    """
    grid_len = grid.shape[0]
    x_expanded = x.unsqueeze(-1)  # [..., in_features, 1]
    
    if grid_len == 1:
        return torch.ones_like(x_expanded)
    
    lagrange_list = []
    for i in range(grid_len):
        z_i = grid[i]
        diffs = []
        for j in range(grid_len):
            if i != j:
                z_j = grid[j]
                denom = z_i - z_j
                diffs.append((x_expanded - z_j) / denom)
        l_i = torch.prod(torch.stack(diffs, dim=-1), dim=-1)
        lagrange_list.append(l_i)
        
    return torch.cat(lagrange_list, dim=-1)


def newton_basis(x: torch.Tensor, grid: torch.Tensor, h: float = None) -> torch.Tensor:
    """
    Newton's Divided Differences Polynomial Basis:
        N_0(x) = 1
        N_k(x) = prod_{j=0}^{k-1} (x - z_j) for k = 1, ..., grid_len - 1
        
    Properties:
        - Nested hierarchy: higher-order terms add incremental polynomial degrees
        - Evaluates numerical stability and conditioning under gradient backpropagation
        
    Args:
        x: Input tensor of shape (*, in_features)
        grid: Grid nodes of shape (grid_len,)
        h: Spacing parameter (unused, maintained for uniform API signature)
        
    Returns:
        Tensor of shape (*, in_features, grid_len)
    """
    grid_len = grid.shape[0]
    x_expanded = x.unsqueeze(-1)  # [..., in_features, 1]
    
    if grid_len == 1:
        return torch.ones_like(x_expanded)
        
    newton_list = [torch.ones_like(x_expanded)]
    curr_prod = torch.ones_like(x_expanded)
    
    for k in range(1, grid_len):
        z_prev = grid[k - 1]
        curr_prod = curr_prod * (x_expanded - z_prev)
        newton_list.append(curr_prod)
        
    return torch.cat(newton_list, dim=-1)


# ==============================================================================
# 4. Basis Registry & Discovery Factory
# ==============================================================================

BASIS_FUNCTIONS: Dict[str, Callable] = {
    "rbf": rbf,
    "gaussian": rbf,
    "rswaf": rswaf,
    "iqf": iqf,
    "bspline": bspline_basis,
    "b_spline": bspline_basis,
    "spline": bspline_basis,
    "chebyshev": chebyshev_basis,
    "cheby": chebyshev_basis,
    "lagrange": lagrange_basis,
    "lagrangian": lagrange_basis,
    "newton": newton_basis,
    "divided_differences": newton_basis,
}


def get_basis_function(name: str) -> Callable:
    """
    Retrieve basis function by canonical name or alias.
    
    Supported names & aliases:
        - 'rbf' / 'gaussian'
        - 'rswaf'
        - 'iqf'
        - 'bspline' / 'b_spline' / 'spline'
        - 'chebyshev' / 'cheby'
        - 'lagrange' / 'lagrangian'
        - 'newton' / 'divided_differences'
    """
    cleaned_name = name.lower().strip().replace("-", "_")
    if cleaned_name not in BASIS_FUNCTIONS:
        available = sorted(list(set(BASIS_FUNCTIONS.keys())))
        raise ValueError(
            f"Unknown basis function: '{name}'. Available bases and aliases: {available}"
        )
    return BASIS_FUNCTIONS[cleaned_name]
