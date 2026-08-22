import torch
import torch.nn as nn
from typing import Callable, Optional
from .solvers import odeint, STEP_SOLVERS


class NeuralODE(nn.Module):
    """
    Neural Ordinary Differential Equation (Neural ODE) wrapper.
    
    Dynamical system:
        dy/dt = f_theta(t, y)
        y(t_0) = y0
        
    Args:
        func: PyTorch module or callable f(x) or f(t, x) defining vector field.
        method: Integration method ('tsit5', 'rk4', 'dopri5', 'euler', 'midpoint', 'heun').
        substeps: Number of integration substeps per interval (default 1).
    """
    def __init__(
        self,
        func: nn.Module,
        method: str = "tsit5",
        substeps: int = 1,
    ):
        super(NeuralODE, self).__init__()
        self.func = func
        self.method = method
        self.substeps = substeps

    def _ode_func(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Wrapper handling both f(y) and f(t, y) signature."""
        try:
            return self.func(t, y)
        except TypeError:
            return self.func(y)

    def forward(
        self,
        y0: torch.Tensor,
        t: torch.Tensor,
        method: Optional[str] = None,
        substeps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Integrate ODE trajectory.
        
        Args:
            y0: Initial state [batch_size, dim] or [dim]
            t: Time points [T]
            method: Override default integration method
            substeps: Override default substeps
            
        Returns:
            Trajectory tensor [T, batch_size, dim] or [T, dim]
        """
        solver_method = method if method is not None else self.method
        solver_substeps = substeps if substeps is not None else self.substeps
        
        return odeint(
            func=self._ode_func,
            y0=y0,
            t=t,
            method=solver_method,
            substeps=solver_substeps,
        )
