import torch
import torch.nn as nn
from typing import List, Union, Callable, Tuple, Optional
from .layer import KDense


class KAN(nn.Module):
    """
    Multi-layer Kolmogorov-Arnold Network (KAN) for learning continuous dynamical systems.
    
    A KAN is constructed by stacking multiple `KDense` layers sequentially.
    
    Args:
        layers_hidden (List[int], default=[2, 10, 2]):
            Layer dimensions / channel counts from input to output.
            e.g., [2, 10, 2] creates:
              - Layer 0: KDense(2 -> 10)
              - Layer 1: KDense(10 -> 2)
              
        grid_len (int, default=5):
            Number of uniform grid knot points G per edge in each layer.
            Higher values increase the resolution/capacity of the learnable 1D edge curves.
            
        grid_lims (Tuple[float, float], default=(-1.0, 1.0)):
            The interval [z_min, z_max] across which grid knot points are placed.
            Matches the [-1.0, 1.0] output range of the normalizer (tanh).
            
        basis_func (Union[str, Callable], default="rbf"):
            Basis function kernel placed at each grid point ('rbf', 'rswaf', 'iqf', 'bspline').
            
        normalizer (Union[str, Callable], default="tanh"):
            Input squashing function mapping unbounded state variables into grid_lims [-1.0, 1.0].
            
        base_act (Union[str, Callable], default="silu"):
            Smooth residual linear activation function (SiLU/Swish) for stable gradient flow.
            
        use_base_act (bool, default=True):
            Whether to include the residual linear branch W * base_act(x).
            
        init_scale (float, default=1.0):
            Scaling multiplier for Xavier/Glorot initialization.
    """
    def __init__(
        self,
        layers_hidden: List[int] = [2, 10, 2],
        grid_len: int = 5,
        grid_lims: Tuple[float, float] = (-1.0, 1.0),
        basis_func: Union[str, Callable] = "rbf",
        normalizer: Union[str, Callable] = "tanh",
        base_act: Union[str, Callable] = "silu",
        use_base_act: bool = True,
        init_scale: float = 1.0,
    ):
        super(KAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.grid_len = grid_len
        self.grid_lims = grid_lims
        self.basis_name = basis_func if isinstance(basis_func, str) else getattr(basis_func, "__name__", "custom")
        self.normalizer_name = normalizer if isinstance(normalizer, str) else getattr(normalizer, "__name__", "custom")
        self.base_act_name = base_act if isinstance(base_act, str) else getattr(base_act, "__name__", "custom")
        
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                KDense(
                    in_features=layers_hidden[i],
                    out_features=layers_hidden[i + 1],
                    grid_len=grid_len,
                    grid_lims=grid_lims,
                    basis_func=basis_func,
                    normalizer=normalizer,
                    base_act=base_act,
                    use_base_act=use_base_act,
                    init_scale=init_scale,
                )
            )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Evaluate KAN on input state tensor.
        Handles both autonomous f(x) and non-autonomous f(t, x) / f(x, t) signatures.
        """
        if "x" in kwargs:
            x = kwargs["x"]
        elif "u" in kwargs:
            x = kwargs["u"]
        elif "y" in kwargs:
            x = kwargs["y"]
        elif len(args) == 1:
            x = args[0]
        elif len(args) >= 2:
            # If called as func(t, y) as common in ODE solvers:
            # check which argument matches the input dimension
            if isinstance(args[0], (int, float)) or (isinstance(args[0], torch.Tensor) and args[0].numel() == 1 and args[1].shape[-1] == self.layers_hidden[0]):
                x = args[1]
            else:
                x = args[0]
        else:
            raise ValueError("KAN forward requires an input tensor.")

        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, act_reg: float = 1.0, entropy_reg: float = 1.0) -> torch.Tensor:
        """
        Compute L1 and Entropy regularization over network parameters (Eq. 12 in manuscript).
        
        L_reg = act_reg * sum(|p|) + entropy_reg * (-sum(p_bar * log(p_bar)))
        where p_bar = |p| / sum(|p|)
        """
        total_p = []
        for p in self.parameters():
            if p.requires_grad:
                total_p.append(p.view(-1))
                
        if not total_p:
            return torch.tensor(0.0)
            
        all_params = torch.cat(total_p)
        l1 = torch.abs(all_params)
        act_loss = torch.sum(l1)
        
        if entropy_reg > 0:
            p_bar = l1 / (act_loss + 1e-10)
            entropy_loss = -torch.sum(p_bar * torch.log(p_bar + 1e-10))
            return act_reg * act_loss + entropy_reg * entropy_loss
        else:
            return act_reg * act_loss

    def get_layer_activations(self, x: torch.Tensor):
        """
        Extract activations for all layers given an input trajectory.
        """
        activations = []
        current_x = x
        for layer in self.layers:
            spline_acts, base_acts = layer.get_activations(current_x)
            current_x = layer(current_x)
            activations.append({
                "spline": spline_acts,
                "base": base_acts,
                "output": current_x,
            })
        return activations
