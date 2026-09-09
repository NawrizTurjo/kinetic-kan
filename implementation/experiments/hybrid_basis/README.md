# Track C — Learnable Softmax Hybrid Basis

Full guideline, verified gotchas, and findings: [`docs/15_p3_hybrid_basis_findings.md`](../../../docs/15_p3_hybrid_basis_findings.md).

## Files

| File | Purpose |
| :--- | :--- |
| `hybrid_basis.py` | `HybridBasis(nn.Module)` — softmax-gated blend of `bspline_basis` and `rbf` |
| `run_hybrid.py` | Standalone training loop (does **not** call `train_kan_ode()` — see docs/15 §3 for why) |
| `load_hybrid.py` | Reconstructs a saved checkpoint for inspection (since `evaluate.py` can't — same reason) |
| `plot_blend.py` | Plots α(epoch)/β(epoch) for every completed run |

## Quick start (run from `implementation/`)

```powershell
python experiments\hybrid_basis\run_hybrid.py --dataset lotka_volterra --epochs 2000 --save_dir results\phase3\hybrid_basis\probe_lv
python experiments\hybrid_basis\plot_blend.py
```

## Tests

`tests/test_p3_hybrid_basis.py` (repo root) — 10 tests covering the basis module in
isolation and wired into a full `KAN` model. Run from repo root: `pytest tests/test_p3_hybrid_basis.py -v`.
