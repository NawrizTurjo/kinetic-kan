"""[TRACK C] Reconstruct a HybridBasis checkpoint for inspection/plotting."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                          # this dir, so `hybrid_basis` is importable
sys.path.insert(0, os.path.join(_HERE, "..", ".."))  # implementation/, so `kan` is importable

import torch
from kan import KAN
from hybrid_basis import HybridBasis

def load(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    hybrid = HybridBasis(grid_len=cfg["grid_len"])
    model = KAN(layers_hidden=cfg["layers_hidden"], grid_len=cfg["grid_len"], basis_func=hybrid)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
