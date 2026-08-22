import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Callable, Tuple
from .basis import get_basis_function, rbf


class KDense(nn.Module):
    """
    Kolmogorov-Arnold Dense Layer matching the original paper formulation (kdense.jl).
    
    ========================================================================================
    CONCEPTUAL INTUITION (For those familiar with MLPs / CNNs):
    ========================================================================================
    - In standard MLPs:
        Edge: y = W * x  (linear scalar multiplication)
        Node: Act(sum(edges))  (fixed non-linearity like ReLU/GELU at neuron)
    
    - In KANs:
        Node: Simple summation  (sum(edges))
        Edge: Learnable 1D non-linear function phi(x)
        
    How do we make an edge function phi(x) learnable and flexible?
    We place several "anchor points" (called a GRID) along the input range and assign a 
    localized basis bell-curve (like Gaussian RBF) to each anchor point. By learning the 
    amplitudes (spline weights C) of these bell-curves, the network can shape phi(x) into 
    ANY arbitrary 1D function (sin, exp, polynomials, x^2, etc.)!
    
    ========================================================================================
    MATHEMATICAL FORMULA:
    ========================================================================================
        y = C * basis(normalizer(x)) + W * base_act(x)
        
    where:
        - normalizer: Maps unconstrained input x to a known bounded range [-1, 1] (e.g. tanh).
        - grid: A fixed 1D array of G anchor centers evenly spaced across grid_lims [-1, 1].
        - basis: Evaluates localized basis kernels (Gaussian RBF) centered at each grid point.
        - C: Trainable spline weights [out_features, in_features * grid_len] scaling the basis curves.
        - base_act & W: A residual linear pathway (like a ResNet skip-connection) for training stability.

    ========================================================================================
    CLASS ATTRIBUTES & PARAMETERS EXPLAINED:
    ========================================================================================
    Args:
        in_features (int):
            Number of input dimensions / channels (e.g., 2 for Lotka-Volterra [x, y]).
        
        out_features (int):
            Number of output dimensions / channels (e.g., 2 for output derivatives [dx/dt, dy/dt]).
        
        grid_len (int, default=5):
            Number of grid knot points (anchor centers) G per edge.
            - Analogy: Like resolution / sampling rate.
            - Higher grid_len (e.g. 8, 10): Higher capacity to fit wiggly, high-frequency curves,
              but increases parameter count.
            - Lower grid_len (e.g. 3, 5): Smoother, simpler curves that avoid overfitting.
            
        grid_lims (Tuple[float, float], default=(-1.0, 1.0)):
            The interval [z_min, z_max] over which the grid points are placed.
            Because inputs are passed through normalizer (tanh), the normalized input is always 
            guaranteed to land inside [-1.0, 1.0]. Setting grid_lims=(-1.0, 1.0) ensures that 
            the grid covers exactly the entire active domain of the inputs.
            
        basis_func (Union[str, Callable], default="rbf"):
            The localized kernel function placed at each grid point.
            - "rbf": Gaussian Radial Basis Function exp(-((x - z)/h)^2). Smooth and localized.
            - "rswaf": Reflectional Switch Activation Function (1 - tanh^2). Heavier tails.
            - "iqf": Inverse Quadratic Function 1 / (1 + ((x - z)/h)^2).
            - "bspline": Piecewise polynomial B-splines.
            
        normalizer (Union[str, Callable], default="tanh"):
            Activation used to squash/bound raw input variables into the grid interval.
            - "tanh": Squashes (-inf, +inf) -> [-1, 1] to perfectly match grid_lims.
            - "identity": No normalization (raw input must already lie in grid_lims).
            
        base_act (Union[str, Callable], default="silu"):
            The smooth residual activation function b(x) for the linear base branch.
            Similar to SiLU / Swish in standard deep networks.
            
        use_base_act (bool, default=True):
            If True, adds the residual linear branch W * base_act(x).
            This acts like a continuous skip-connection that guarantees smooth non-zero gradients 
            even before the localized RBF grid weights have converged.
            
        init_scale (float, default=1e-5):
            Multiplicative scaling factor applied to the Xavier/Glorot uniform parameter initialization.
            Helps prevent exploding gradients during early ODE integration steps.
            
        dtype (torch.dtype, default=torch.float32):
            Data type for tensors and learnable weights.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_len: int = 5,
        grid_lims: Tuple[float, float] = (-1.0, 1.0),
        basis_func: Union[str, Callable] = "rbf",
        normalizer: Union[str, Callable] = "tanh",
        base_act: Union[str, Callable] = "silu",
        use_base_act: bool = True,
        init_scale: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    ):
        super(KDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_len = grid_len
        self.grid_lims = grid_lims
        self.use_base_act = use_base_act
        self.init_scale = init_scale
        
        # -----------------------------------------------------------------------------
        # 1. Grid Setup:
        # Step width h (denominator) = (z_max - z_min) / (grid_len - 1)
        # e.g., for span [-1, 1] with grid_len=5: grid = [-1.0, -0.5, 0.0, 0.5, 1.0], h = 0.5
        # -----------------------------------------------------------------------------
        span = grid_lims[1] - grid_lims[0]
        if span <= 0:
            raise ValueError("grid_lims[1] must be greater than grid_lims[0]")
        self.denominator = span / (grid_len - 1) if grid_len > 1 else 1.0
        
        # Persistent non-trainable grid buffer (moves with model to CPU/CUDA automatically)
        grid = torch.linspace(grid_lims[0], grid_lims[1], grid_len, dtype=dtype)
        self.register_buffer("grid", grid)
        
        # -----------------------------------------------------------------------------
        # 2. Basis Kernel Function Setup
        # -----------------------------------------------------------------------------
        if isinstance(basis_func, str):
            self.basis_name = basis_func
            self.basis_func = get_basis_function(basis_func)
        else:
            self.basis_name = getattr(basis_func, "__name__", "custom")
            self.basis_func = basis_func
            
        # -----------------------------------------------------------------------------
        # 3. Input Normalizer Setup (tanh maps unbounded values into [-1, 1])
        # -----------------------------------------------------------------------------
        if isinstance(normalizer, str):
            self.normalizer_name = normalizer
            if normalizer == "tanh":
                self.normalizer = torch.tanh
            elif normalizer == "sigmoid":
                self.normalizer = torch.sigmoid
            elif normalizer == "identity" or normalizer == "none":
                self.normalizer = nn.Identity()
            else:
                raise ValueError(f"Unknown normalizer: {normalizer}")
        else:
            self.normalizer_name = getattr(normalizer, "__name__", "custom")
            self.normalizer = normalizer
            
        # -----------------------------------------------------------------------------
        # 4. Residual Base Activation Setup
        # -----------------------------------------------------------------------------
        if isinstance(base_act, str):
            self.base_act_name = base_act
            if base_act in ["silu", "swish"]:
                self.base_act = F.silu
            elif base_act == "relu":
                self.base_act = F.relu
            elif base_act == "tanh":
                self.base_act = torch.tanh
            elif base_act == "gelu":
                self.base_act = F.gelu
            elif base_act == "identity":
                self.base_act = nn.Identity()
            else:
                raise ValueError(f"Unknown base activation: {base_act}")
        else:
            self.base_act_name = getattr(base_act, "__name__", "custom")
            self.base_act = base_act

        # -----------------------------------------------------------------------------
        # 5. Trainable Parameters:
        # - Spline Weights C: shape [out_features, in_features * grid_len]
        # - Base Linear Weights W: shape [out_features, in_features]
        # -----------------------------------------------------------------------------
        self.C = nn.Parameter(torch.empty(out_features, in_features * grid_len, dtype=dtype))
        
        if self.use_base_act:
            self.W = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        else:
            self.register_parameter("W", None)
            
        self.reset_parameters()

    def reset_parameters(self):
        """Glorot / Xavier Uniform initialization matching Julia implementation."""
        # Spline weight initialization
        fan_in_C = self.in_features * self.grid_len
        fan_out_C = self.out_features
        bound_C = math.sqrt(6.0 / (fan_in_C + fan_out_C)) * self.init_scale
        nn.init.uniform_(self.C, -bound_C, bound_C)
        
        if self.use_base_act and self.W is not None:
            fan_in_W = self.in_features
            fan_out_W = self.out_features
            bound_W = math.sqrt(6.0 / (fan_in_W + fan_out_W)) * self.init_scale
            nn.init.uniform_(self.W, -bound_W, bound_W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (..., in_features)
        Returns:
            Output tensor of shape (..., out_features)
        """
        orig_shape = x.shape
        in_dim = orig_shape[-1]
        assert in_dim == self.in_features, f"Expected input dim {self.in_features}, got {in_dim}"
        
        x_flat = x.view(-1, self.in_features) # [Batch, in_features]
        
        # 1. Normalize input to [-1, 1]
        x_norm = self.normalizer(x_flat) # [Batch, in_features]
        
        # 2. Compute basis values
        # basis: [Batch, in_features, grid_len]
        basis_vals = self.basis_func(x_norm, self.grid, self.denominator)
        
        # 3. Flatten basis to [Batch, in_features * grid_len]
        basis_flat = basis_vals.reshape(basis_vals.size(0), self.in_features * self.grid_len)
        
        # 4. Spline contribution: C @ basis^T -> [Batch, out_features]
        spline_out = F.linear(basis_flat, self.C)
        
        # 5. Base linear activation contribution: W @ base_act(x)^T
        if self.use_base_act and self.W is not None:
            base_val = self.base_act(x_flat)
            base_out = F.linear(base_val, self.W)
            out = spline_out + base_out
        else:
            out = spline_out
            
        return out.view(*orig_shape[:-1], self.out_features)

    def get_activations(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract individual spline and base activations for visualization and symbolic regression.
        
        Returns:
            spline_activations: [Batch, in_features, out_features]
            base_activations: [Batch, in_features, out_features] (or None)
        """
        x_flat = x.view(-1, self.in_features)
        x_norm = self.normalizer(x_flat)
        basis_vals = self.basis_func(x_norm, self.grid, self.denominator) # [B, I, G]
        
        # Reshape C into [O, I, G]
        C_reshaped = self.C.view(self.out_features, self.in_features, self.grid_len)
        # spline_act[b, i, o] = sum_g C[o, i, g] * basis[b, i, g]
        # einsum: 'big,oig->bio'
        spline_acts = torch.einsum("big,oig->bio", basis_vals, C_reshaped)
        
        base_acts = None
        if self.use_base_act and self.W is not None:
            base_val = self.base_act(x_flat) # [B, I]
            # einsum: 'bi,oi->bio'
            base_acts = torch.einsum("bi,oi->bio", base_val, self.W)
            
        return spline_acts, base_acts
