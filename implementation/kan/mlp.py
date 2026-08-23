"""
MLP-ODE Baseline Architecture Module.

Provides a parameter-matched Multi-Layer Perceptron Ordinary Differential Equation
(MLP-ODE) vector field network for direct comparative benchmarking against KAN-ODEs
as established in the base MIT paper (Koenig et al., CMAME 2024).

Base Paper Parameter Comparison on 2D Lotka-Volterra:
- KAN-ODE (2 -> 10 -> 2, grid_len=5): 240 parameters
- MLP-ODE (2 -> 14 -> 8 -> 8 -> 2):  252 parameters (Parameter-Matched Baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Callable, Tuple, Optional


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a PyTorch module.
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class MLP_ODE(nn.Module):
    """
    Multi-Layer Perceptron (MLP) vector field for Neural ODEs:
        du/dt = f_theta(t, u)
        
    Default architecture [2, 14, 8, 8, 2] with bias yields exactly 252 parameters,
    matching the 240-parameter KAN-ODE architecture [2, 10, 2] from the base paper.
    
    Args:
        layers_hidden (List[int], default=[2, 14, 8, 8, 2]):
            Layer dimensions from input to output.
        activation (Union[str, Callable], default="silu"):
            Non-linear activation applied at intermediate hidden layers.
            Options: 'silu', 'tanh', 'relu', 'gelu', 'identity'.
        use_bias (bool, default=True):
            Whether linear layers include additive bias terms.
        init_scale (float, default=1.0):
            Multiplicative scaling factor for Xavier/Glorot weight initialization.
    """
    def __init__(
        self,
        layers_hidden: List[int] = [2, 14, 8, 8, 2],
        activation: Union[str, Callable] = "silu",
        use_bias: bool = True,
        init_scale: float = 1.0,
    ):
        super(MLP_ODE, self).__init__()
        self.layers_hidden = layers_hidden
        self.use_bias = use_bias
        self.init_scale = init_scale
        
        # 1. Activation setup
        if isinstance(activation, str):
            self.activation_name = activation.lower()
            if self.activation_name in ["silu", "swish"]:
                self.activation = F.silu
            elif self.activation_name == "tanh":
                self.activation = torch.tanh
            elif self.activation_name == "relu":
                self.activation = F.relu
            elif self.activation_name == "gelu":
                self.activation = F.gelu
            elif self.activation_name in ["identity", "none"]:
                self.activation = nn.Identity()
            else:
                raise ValueError(f"Unknown activation function: '{activation}'")
        else:
            self.activation_name = getattr(activation, "__name__", "custom")
            self.activation = activation
            
        # 2. Layer construction
        self.linears = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.linears.append(
                nn.Linear(
                    in_features=layers_hidden[i],
                    out_features=layers_hidden[i + 1],
                    bias=use_bias,
                )
            )
            
        self.reset_parameters()

    def reset_parameters(self):
        """Xavier/Glorot uniform parameter initialization."""
        for linear in self.linears:
            nn.init.xavier_uniform_(linear.weight, gain=self.init_scale)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    @property
    def num_parameters(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Evaluate continuous vector field.
        Handles both autonomous f(u) and non-autonomous f(t, u) / f(u, t) signatures.
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
            # Handle ODE solver calling signature f(t, u)
            if isinstance(args[0], (int, float)) or (isinstance(args[0], torch.Tensor) and args[0].numel() == 1 and args[1].shape[-1] == self.layers_hidden[0]):
                x = args[1]
            else:
                x = args[0]
        else:
            raise ValueError("MLP_ODE forward requires an input state tensor.")

        for i, linear in enumerate(self.linears):
            x = linear(x)
            # Apply activation to all layers except the final linear output layer
            if i < len(self.linears) - 1:
                x = self.activation(x)
                
        return x
