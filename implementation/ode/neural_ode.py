import inspect

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
        self._accepts_time = self._detect_time_argument(func)

    @staticmethod
    def _detect_time_argument(func) -> bool:
        """
        Decide once, at construction time, whether `func` can be called as f(t, y).

        Previously this was decided per call with a try/except TypeError inside the
        integration hot loop, which silently reinterpreted any genuine TypeError
        raised inside the vector field as a calling-convention mismatch. Resolving
        the signature up front removes that failure-masking path entirely.
        """
        call_target = func.forward if isinstance(func, nn.Module) else func
        try:
            sig = inspect.signature(call_target)
        except (TypeError, ValueError):
            return True  # C-implemented / unintrospectable: assume f(t, y)

        positional = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values()):
            return True  # *args forward (KAN, MLP_ODE) handles both conventions
        return len(positional) >= 2

    def _ode_func(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Dispatch to f(t, y) or f(y) using the signature resolved in __init__."""
        if self._accepts_time:
            return self.func(t, y)
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
