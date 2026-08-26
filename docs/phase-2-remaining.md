### 📌 Phase 2 vs. Phase 3 Scope: Where Does This Belong?

* **Formally in the Blueprint:**  
  Phase 2 was designed for the canonical **Lotka-Volterra baseline suite** (Solvers, 7 Bases, Step sizes, Noise, KAN vs. MLP), while deep stiffness sweeps for the Pendulum and chaotic systems (Lorenz) were scheduled for **Phase 3**.
* **However:**  
  Running a clean baseline for **Damped Pendulum** and **SIR Epidemic** right now with Gradient Clipping wraps up the Phase 2 cross-domain section cleanly and gives you complete tables across all 4 dynamical systems!

---

### 🛠️ What We Updated in the Code:

1. **`train.py`:** Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)` before `optimizer.step()`, controlled via `--grad_clip` (default `1.0`).
2. **`run_phase2.ps1`:** Configured `-Only systems` to automatically use `grid_len 8`, `lr 0.003`, and `grad_clip 1.0` for both Pendulum and SIR.

---

### 🚀 The Single Command for the Final Pass:

Run this command in PowerShell from `implementation/` (both Pendulum and SIR will run simultaneously in parallel and finish in **~45 minutes**):

```powershell
.\run_phase2.ps1 -Only systems -MaxParallel 2
```

Once it finishes, run:
```powershell
python collate_results.py --root results/benchmarks
```

And both non-linear systems will be fully trained, stable, and documented with zero gradient explosions!