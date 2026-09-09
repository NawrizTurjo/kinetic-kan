"""
Phase 3 Track C — Learnable Softmax Hybrid Basis: unit and integration tests.

Decentralized per Phase 1 Rule 1 (docs/04 SS6.2): this track owns its own test file
here rather than editing a shared tests/test_*.py. `pythonpath = ["implementation"]`
in pyproject.toml puts `kan`/`ode`/`data`/`utils` on sys.path automatically for every
test file; only `experiments/hybrid_basis/` (this track's own private module) needs an
explicit path insert below.

These also make good on three tests PROJECT_IMPLEMENTATION_PLAN.md SS7 sketched for a
basis_type='hybrid' that was never actually built: blend-weight sum-to-one, gradcheck-
style gradient flow, and NaN immunity under extreme inputs.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "implementation", "experiments", "hybrid_basis")),
)

import pytest
import torch

from kan import KAN, count_parameters
from hybrid_basis import HybridBasis


# ==============================================================================
# HybridBasis in isolation
# ==============================================================================

def test_hybrid_basis_output_shape():
    """Matches the unified basis_fn signature: (*, in_features, grid_len)."""
    hb = HybridBasis(grid_len=5)
    grid = torch.linspace(-1, 1, 5)
    x = torch.randn(8, 2)  # [batch=8, in_features=2]
    out = hb(x, grid, h=0.5)
    assert out.shape == (8, 2, 5)


def test_hybrid_basis_blend_weights_sum_to_one():
    """softmax(logits) must be a valid convex combination: sums to 1, both >= 0."""
    hb = HybridBasis(grid_len=5)
    a, b = hb.blend_weights()
    assert a + b == pytest.approx(1.0, abs=1e-6)
    assert a >= 0.0 and b >= 0.0


def test_hybrid_basis_initial_blend_is_uniform():
    """Default init_logits=(0,0) -> softmax gives exactly (0.5, 0.5)."""
    hb = HybridBasis(grid_len=5)
    a, b = hb.blend_weights()
    assert a == pytest.approx(0.5, abs=1e-6)
    assert b == pytest.approx(0.5, abs=1e-6)


def test_hybrid_basis_output_is_convex_combination():
    """phi(x) must lie between pure-spline and pure-RBF outputs at every start point,
    since it is alpha*Spline + beta*RBF with alpha,beta >= 0 summing to 1."""
    from kan.basis import bspline_basis, rbf
    hb = HybridBasis(grid_len=5)
    grid = torch.linspace(-1, 1, 5)
    x = torch.randn(4, 2)
    h = 0.5
    out = hb(x, grid, h)
    spline_out = bspline_basis(x, grid, h)
    rbf_out = rbf(x, grid, h)
    lo = torch.minimum(spline_out, rbf_out)
    hi = torch.maximum(spline_out, rbf_out)
    assert torch.all(out >= lo - 1e-5) and torch.all(out <= hi + 1e-5)


def test_hybrid_basis_gradient_flows_to_blend_logits():
    """The whole point of the module: blend_logits must receive a nonzero gradient
    from an ordinary backward pass, with no special-casing in the optimizer."""
    hb = HybridBasis(grid_len=5)
    grid = torch.linspace(-1, 1, 5)
    x = torch.randn(4, 2, requires_grad=True)
    out = hb(x, grid, h=0.5)
    out.sum().backward()
    assert hb.blend_logits.grad is not None
    assert torch.any(hb.blend_logits.grad != 0)


def test_hybrid_basis_nan_immunity_under_extreme_inputs():
    """Matches PROJECT_IMPLEMENTATION_PLAN.md SS7's sketched
    test_extreme_input_nan_immunity, now run against the real module."""
    hb = HybridBasis(grid_len=5)
    grid = torch.linspace(-1, 1, 5)
    extreme_x = torch.tensor([[-50.0, 50.0], [-100.0, 100.0]])
    out = hb(extreme_x, grid, h=0.5)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ==============================================================================
# HybridBasis wired into a full KAN model
# ==============================================================================

def test_hybrid_basis_registers_as_kan_submodule():
    """Assigning a HybridBasis instance as KAN's basis_func must auto-register it
    as a submodule so blend_logits appears in model.parameters() and is optimized
    by the ordinary training loop -- this is what makes SS3's standalone loop work
    without any special parameter-group handling."""
    hb = HybridBasis(grid_len=5)
    model = KAN(layers_hidden=[2, 10, 2], grid_len=5, basis_func=hb)
    names = [n for n, _ in model.named_parameters()]
    assert any("blend_logits" in n for n in names)


def test_hybrid_basis_shared_gate_adds_exactly_two_params():
    """[2,10,2] + G=5 RBF-only KAN is 240 params (docs/05 Table 2). A shared
    HybridBasis instance across both KDense layers should add exactly 2 (one
    blend_logits pair, deduplicated by nn.Module.parameters() since it is the
    SAME object passed to every layer -- not one pair per layer)."""
    hb = HybridBasis(grid_len=5)
    model = KAN(layers_hidden=[2, 10, 2], grid_len=5, basis_func=hb)
    total, _ = count_parameters(model)
    assert total == 240 + 2, f"expected 242 (240 base + shared blend_logits), got {total}"


def test_hybrid_basis_forward_through_kan_is_finite():
    """A full forward pass through a 2-layer KAN with the hybrid basis must not
    produce NaN/Inf on ordinary input."""
    hb = HybridBasis(grid_len=5)
    model = KAN(layers_hidden=[2, 10, 2], grid_len=5, basis_func=hb)
    x = torch.randn(6, 2)
    out = model(x)
    assert out.shape == (6, 2)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_hybrid_basis_backward_through_kan_updates_blend_logits():
    """End-to-end: one optimizer step through a full KAN model must move
    blend_logits away from its (0,0) init -- the actual mechanism the probe-stage
    sanity check in docs/15 SS8 relies on."""
    hb = HybridBasis(grid_len=5)
    model = KAN(layers_hidden=[2, 10, 2], grid_len=5, basis_func=hb)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = hb.blend_logits.detach().clone()
    x = torch.randn(6, 2)
    target = torch.randn(6, 2)
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        opt.step()
    after = hb.blend_logits.detach().clone()

    assert not torch.allclose(before, after), "blend_logits did not move after 5 optimizer steps"
