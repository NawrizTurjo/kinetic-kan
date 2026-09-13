"""
Ad hoc diagnostic: does a given (solver, mu) cell's pass/fail verdict depend on
the random seed, or is it deterministic for that mu? Grew out of the mu=2.0
investigation in docs/16_p3_stiffness_map_findings.md -- kept as a reusable
script rather than a throwaway, since it's the actual evidence behind that
finding and may be needed again for other mu values.

Usage: python seed_test.py <seed> <mu> [solver]   (solver defaults to euler,
the cheapest, since this is purely about verdict/optimization-landscape
sensitivity, not solver comparison)

Writes results/phase3/stiffness_map/_seed_test/seed<seed>_mu<mu>[_<solver>]/metrics.json
-- a dedicated side directory, never the original probe/ or probe_traj/ roots.

Naming: euler (the first solver tested here, before this was extended to the
other three) keeps its original "seed<seed>_mu<mu>" folder name with no solver
suffix, so its already-completed results stay valid and aren't silently
orphaned by a naming change. Every other solver gets an explicit
"seed<seed>_mu<mu>_<solver>" suffix to avoid collisions now that more than one
solver lands in this directory.
"""
import sys, json, os
sys.path.insert(0, '.')
sys.path.insert(0, '../../implementation')
import run_sweep as rs

seed = int(sys.argv[1])
mu = float(sys.argv[2])
solver = sys.argv[3] if len(sys.argv) > 3 else "euler"

suffix = "" if solver == "euler" else f"_{solver}"
out_dir = f"../../results/phase3/stiffness_map/_seed_test/seed{seed}_mu{mu}{suffix}"
os.makedirs(out_dir, exist_ok=True)
metrics = rs.train_cell(
    solver=solver, mu=mu, dt=0.05, num_epochs=2000, seed=seed,
    save_trajectory_to=os.path.join(out_dir, "trajectory.npz"),
    save_checkpoint_to=os.path.join(out_dir, "checkpoint.pt"),
)
metrics["verdict"] = rs.classify_cell(metrics)
with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"seed={seed} mu={mu} solver={solver}: verdict={metrics['verdict']} best_train_mse={metrics['best_train_mse']}")
