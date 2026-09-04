"""
L1 edge pruning for trained KAN models.  [Phase 3 / Track E]

Why this file exists
--------------------
The blueprint's SINDy comparison calls for holding SINDy's recovered symbolic
terms up against "KAN's L1-pruned edges". `utils/regularization.py` only ever
*penalises* parameter magnitude during training (Eq. 12); nothing anywhere in
the codebase ever zeroes a coefficient out afterwards. So sparsity in this
project has, until now, only ever been a pressure -- never a decision. This
module turns it into a decision.

What a KAN "edge" is here
-------------------------
`KDense` computes, for input i and output o,

    phi_{o,i}(x_i) = sum_g C[o, i*G + g] * basis_g(norm(x_i))  +  W[o, i] * b(x_i)
                     \\_________ spline branch _________/         \\_ base branch _/

so one edge is one (in, out) pair and owns G spline coefficients plus one base
weight. Pruning must therefore act on a whole GROUP of G coefficients at once --
zeroing individual entries of `C` would not remove an edge, it would only make
its learned curve lumpier. `edge_strengths` reduces the G coefficients to a
single per-edge magnitude; `prune_edges` masks the weakest edges entirely.

The base branch is a separate pathway
-------------------------------------
Zeroing `C` alone leaves `W[o, i] * b(x_i)` intact, so the edge still conducts
signal -- its curve is merely reduced to a scaled SiLU. Whether that counts as
"pruned" is a real modelling choice, not an implementation detail, so it is an
explicit flag (`prune_base`) and BOTH settings are measured and reported rather
than one being silently assumed.

Owner: Monjur Hossain Khan (Shovon), 2105043.
"""

import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def edge_strengths(layer: nn.Module, include_base: bool = False) -> torch.Tensor:
    """
    Per-edge L1 magnitude of a `KDense` layer.

    Args:
        layer: a `KDense` instance (must expose `C`, `in_features`, `grid_len`).
        include_base: fold |W[o, i]| into the strength as well. Off by default so
            the ranking measures the spline branch -- the part carrying the
            learned nonlinearity -- rather than being diluted by one linear
            coefficient.

    Returns:
        Tensor [out_features, in_features]; entry (o, i) is the mean absolute
        spline coefficient of edge i -> o. Mean rather than sum, so the numbers
        stay comparable across layers with different `grid_len`.
    """
    C = layer.C.detach()                                   # [O, I*G]
    C_edges = C.view(layer.out_features, layer.in_features, layer.grid_len)
    strength = C_edges.abs().mean(dim=-1)                  # [O, I]

    if include_base and getattr(layer, "W", None) is not None:
        strength = strength + layer.W.detach().abs()
    return strength


def prune_edges(
    model: nn.Module,
    threshold_percentile: float = 50.0,
    prune_base: bool = False,
    per_layer: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Zero the weakest-magnitude edges of a trained KAN.

    Args:
        model: a trained `KAN`. **Not mutated** -- a deep copy is pruned and
            returned, so the caller keeps an untouched baseline to compare with.
        threshold_percentile: percentage of edges to remove, ranked by
            `edge_strengths`. 0.0 is an exact no-op; 50.0 drops the weaker half.
        prune_base: also zero `W[o, i]` for every pruned edge, severing the
            residual branch too. With False the edge keeps `W * silu(x)`.
        per_layer: rank edges within each layer separately. True holds the
            requested sparsity in every layer; False ranks globally, which can
            empty one layer completely and disconnect the network.

    Returns:
        (pruned_model, info) where `info` records, per layer and overall, how
        many edges were removed and the magnitude cutoff applied -- so a pruning
        level is reproducible from the reported numbers alone.
    """
    if not 0.0 <= threshold_percentile <= 100.0:
        raise ValueError(
            f"threshold_percentile must be in [0, 100], got {threshold_percentile}"
        )

    pruned = copy.deepcopy(model)
    layers: List[nn.Module] = list(pruned.layers)

    strengths = [edge_strengths(l, include_base=prune_base) for l in layers]

    if per_layer:
        cutoffs = [
            float(np.percentile(s.cpu().numpy(), threshold_percentile)) for s in strengths
        ]
    else:
        pooled = torch.cat([s.reshape(-1) for s in strengths]).cpu().numpy()
        cutoffs = [float(np.percentile(pooled, threshold_percentile))] * len(layers)

    info = {
        "threshold_percentile": threshold_percentile,
        "prune_base": prune_base,
        "per_layer": per_layer,
        "layers": [],
    }
    total_edges = 0
    total_pruned = 0

    with torch.no_grad():
        for idx, (layer, strength, cutoff) in enumerate(zip(layers, strengths, cutoffs)):
            if threshold_percentile > 0.0:
                # `<` rather than `<=`: at percentile 0 the cutoff IS the minimum,
                # and `<=` would zero every edge tied at it. `<` keeps 0.0 an
                # exact no-op by construction, which the test suite pins.
                mask = strength < cutoff
            else:
                mask = torch.zeros_like(strength, dtype=torch.bool)

            # Expand [O, I] -> [O, I*G] so a masked edge loses all G of its spline
            # coefficients together, never a scattered subset of them.
            C_mask = mask.unsqueeze(-1).expand(
                layer.out_features, layer.in_features, layer.grid_len
            ).reshape(layer.out_features, layer.in_features * layer.grid_len)
            layer.C.masked_fill_(C_mask, 0.0)

            if prune_base and getattr(layer, "W", None) is not None:
                layer.W.masked_fill_(mask, 0.0)

            n_edges = int(mask.numel())
            n_pruned = int(mask.sum().item())
            total_edges += n_edges
            total_pruned += n_pruned
            info["layers"].append({
                "layer": idx,
                "shape": [layer.in_features, layer.out_features],
                "edges": n_edges,
                "edges_pruned": n_pruned,
                "magnitude_cutoff": cutoff,
            })

    info["total_edges"] = total_edges
    info["total_edges_pruned"] = total_pruned
    info["sparsity"] = total_pruned / total_edges if total_edges else 0.0
    return pruned, info


def surviving_edge_report(model: nn.Module) -> List[Dict]:
    """
    Which edges are still alive, and how strong -- the KAN-side analogue of
    SINDy's term list. Reported per layer so the sparsity PATTERN, rather than a
    single scalar, can be held up against SINDy's recovered terms.
    """
    report = []
    for idx, layer in enumerate(model.layers):
        strength = edge_strengths(layer).cpu().numpy()
        alive = strength > 0.0
        report.append({
            "layer": idx,
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "alive_edges": int(alive.sum()),
            "total_edges": int(alive.size),
            "strength_matrix": np.round(strength, 6).tolist(),
        })
    return report
