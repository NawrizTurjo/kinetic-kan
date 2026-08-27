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


class ZeroSumField(nn.Module):
    """
    [FIX-2026-08 / S4] Wrap a vector field so that sum(y) is an exact invariant.

    Some systems carry a linear conservation law. SIR is the canonical case:

        dS/dt + dI/dt + dR/dt = (-bSI) + (bSI - gI) + (gI) = 0   for all S, I, R

    so S + I + R never leaves its initial value. An unconstrained network does not
    know this. Penalising the residual (`--conserve_sum`) only pins it down on the
    sampled training window and says nothing about extrapolation, which is exactly
    where the SIR runs drifted: the penalty run held mass to 0.86% error inside
    [0, 50] and still left the full-horizon error at 0.45.

    Subtracting the componentwise mean projects the field onto the zero-sum
    subspace,

        f_proj(y) = f(y) - mean_i f_i(y),        so   sum_i f_proj,i(y) = 0

    which makes the conservation law a property of the FLOW rather than of the
    fit. It then holds for every state, at every time, on any horizon, to solver
    precision -- and it costs one mean and one subtraction.

    Two things worth being explicit about:

      * The conserved quantity is sum(y0), whatever that is. It equals the intended
        target only if the initial condition already satisfies the invariant, so
        `train.py` checks that and refuses mismatches rather than silently
        conserving the wrong constant.
      * `parameters()` and `state_dict()` delegate to the wrapped module, so the
        optimizer, the checkpoints and every existing loader keep seeing the bare
        model. The projection is a property of the field, not learned weights --
        which is why anything that REBUILDS the field from a checkpoint (evaluate.py,
        analyze_fixes.py) has to re-apply it, and reads `conserve_mode` to do so.
    """

    def __init__(self, field: nn.Module):
        super().__init__()
        self.field = field

    def forward(self, *args, **kwargs) -> torch.Tensor:
        f = self.field(*args, **kwargs)
        return f - f.mean(dim=-1, keepdim=True)

    # Delegate so the wrapper is transparent to optimizers and checkpointing.
    def parameters(self, recurse: bool = True):
        return self.field.parameters(recurse=recurse)

    def named_parameters(self, *args, **kwargs):
        return self.field.named_parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.field.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.field.load_state_dict(*args, **kwargs)


class VanishingDimField(nn.Module):
    """
    [FIX-2026-08 / S5] Wrap a vector field so that it vanishes identically on the
    hyperplane y[dim] == 0, making that plane an invariant manifold of equilibria.

        f_gated(y) = y[dim] * f(y)

    This is an OPT-IN structural prior, not a general stability fix. It is exact for
    compartmental epidemic models, where every transition term is proportional to the
    infected fraction:

        dS/dt = I * (-beta*S)
        dI/dt = I * ( beta*S - gamma)
        dR/dt = I * ( gamma)

    so I = 0 is a line of equilibria (a disease-free population stays disease-free),
    and the remaining cofactor is merely AFFINE in S.

    Why it matters here specifically. SIR's extrapolation window (t in (50, 80]) sits
    at S in [0.034, 0.042] with small I, and the training window visits that corner of
    state space in exactly ONE of its 101 samples -- the other 16 low-I samples are at
    S in [0.92, 0.99], the pre-epidemic growth phase. The network is therefore asked
    to extrapolate the field into a region it has essentially never seen, which no
    amount of loss reweighting can supply. Factoring out y[dim] hands it the tail
    behaviour analytically instead of asking it to infer it from one point, and
    reduces what must actually be learned to an affine function.

    Compose with ZeroSumField to get both SIR invariants at once: sum(h) = 0 holds for
    the true cofactor, so gating and projection do not fight each other.
    """

    def __init__(self, field: nn.Module, dim: int):
        super().__init__()
        self.field = field
        self.dim = dim

    @staticmethod
    def _state(args, kwargs):
        """
        Recover y from an f(y) / f(t, y) / f(x=...) call. The gate is a function of
        the STATE, not of the field's own output, so this has to resolve the same
        calling convention NeuralODE._ode_func dispatches on.
        """
        for key in ("x", "u", "y"):
            if key in kwargs:
                return kwargs[key]
        if len(args) == 1:
            return args[0]
        if len(args) >= 2:
            # f(t, y): t is a scalar, y carries the state dimension
            first = args[0]
            if isinstance(first, (int, float)) or (
                isinstance(first, torch.Tensor) and first.numel() == 1
            ):
                return args[1]
            return first
        raise ValueError("VanishingDimField could not resolve the state argument.")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        y = self._state(args, kwargs)
        f = self.field(*args, **kwargs)
        gate = y.select(-1, self.dim).unsqueeze(-1)
        return gate * f

    def parameters(self, recurse: bool = True):
        return self.field.parameters(recurse=recurse)

    def named_parameters(self, *args, **kwargs):
        return self.field.named_parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.field.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.field.load_state_dict(*args, **kwargs)
