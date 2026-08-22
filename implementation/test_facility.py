import argparse
import os
import time
import json
import torch
import numpy as np

from train import train_kan_ode
from utils.plotting import plot_benchmark_comparison


def run_activation_benchmark(
    activations=["rbf", "rswaf", "iqf", "bspline"],
    solver="tsit5",
    epochs=1500,
    lr=2e-3,
    save_dir="results/benchmark_activations",
):
    """
    Benchmark different KAN basis / activation functions while keeping ODE solver fixed.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    print("\n" + "=" * 70)
    print("RUNNING ACTIVATION FUNCTION BENCHMARK ON LOTKA-VOLTERRA")
    print(f"Activations to test: {activations}")
    print(f"Fixed Solver: {solver.upper()} | Epochs: {epochs} | LR: {lr}")
    print("=" * 70)
    
    for act in activations:
        print(f"\n---> Testing Basis Function: {act.upper()} ...")
        act_save_dir = os.path.join(save_dir, f"basis_{act}")
        t0 = time.time()
        
        try:
            res = train_kan_ode(
                basis_func=act,
                solver=solver,
                num_epochs=epochs,
                lr=lr,
                save_dir=act_save_dir,
                print_freq=max(1, epochs // 5),
            )
            elapsed = time.time() - t0
            
            results[f"Basis: {act.upper()}"] = {
                "basis": act,
                "solver": solver,
                "train_mse": float(res["final_train_mse"]),
                "test_mse": float(res["best_test_mse"]),
                "time_sec": float(elapsed),
                "status": "success",
            }
        except Exception as e:
            print(f"Error testing basis '{act}': {e}")
            results[f"Basis: {act.upper()}"] = {"status": "failed", "error": str(e)}
            
    _save_and_report_benchmark(results, save_dir, "Activation Benchmark Results")
    return results


def run_solver_benchmark(
    solvers=["tsit5", "rk4", "dopri5", "euler"],
    basis_func="rbf",
    epochs=1500,
    lr=2e-3,
    save_dir="results/benchmark_solvers",
):
    """
    Benchmark different ODE integrators while keeping KAN basis fixed to Gaussian RBF.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    print("\n" + "=" * 70)
    print("RUNNING ODE INTEGRATOR BENCHMARK ON LOTKA-VOLTERRA")
    print(f"Solvers to test: {solvers}")
    print(f"Fixed Basis: {basis_func.upper()} (Gaussian RBF) | Epochs: {epochs} | LR: {lr}")
    print("=" * 70)
    
    for s in solvers:
        print(f"\n---> Testing ODE Solver: {s.upper()} ...")
        solver_save_dir = os.path.join(save_dir, f"solver_{s}")
        t0 = time.time()
        
        try:
            res = train_kan_ode(
                basis_func=basis_func,
                solver=s,
                num_epochs=epochs,
                lr=lr,
                save_dir=solver_save_dir,
                print_freq=max(1, epochs // 5),
            )
            elapsed = time.time() - t0
            
            results[f"Solver: {s.upper()}"] = {
                "basis": basis_func,
                "solver": s,
                "train_mse": float(res["final_train_mse"]),
                "test_mse": float(res["best_test_mse"]),
                "time_sec": float(elapsed),
                "status": "success",
            }
        except Exception as e:
            print(f"Error testing solver '{s}': {e}")
            results[f"Solver: {s.upper()}"] = {"status": "failed", "error": str(e)}
            
    _save_and_report_benchmark(results, save_dir, "ODE Integrator Benchmark Results")
    return results


def _save_and_report_benchmark(results: dict, save_dir: str, title: str):
    """Format and print benchmark summary table and save plots."""
    print("\n" + "=" * 75)
    print(f"SUMMARY: {title.upper()}")
    print("=" * 75)
    print(f"{'Configuration':<25} | {'Train MSE':<14} | {'Test (Extrap) MSE':<18} | {'Time (s)':<10}")
    print("-" * 75)
    
    valid_results = {}
    for name, data in results.items():
        if data.get("status") == "success":
            train_m = f"{data['train_mse']:.4e}"
            test_m = f"{data['test_mse']:.4e}"
            t_sec = f"{data['time_sec']:.2f}"
            valid_results[name] = data
        else:
            train_m = "FAILED"
            test_m = "FAILED"
            t_sec = "N/A"
        print(f"{name:<25} | {train_m:<14} | {test_m:<18} | {t_sec:<10}")
        
    print("=" * 75)
    
    # Save JSON summary
    json_path = os.path.join(save_dir, "benchmark_summary.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate bar plot if we have valid results
    if valid_results:
        plot_benchmark_comparison(
            results=valid_results,
            metric="test_mse",
            title=f"{title} - Test Extrapolation MSE",
            save_path=os.path.join(save_dir, "benchmark_comparison.png"),
        )
    print(f"Saved benchmark summary and comparison plot to '{save_dir}'.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAN-ODE Testing & Benchmarking Facility")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "activations", "solvers"],
        help="Benchmark mode: 'activations', 'solvers', or 'all'",
    )
    parser.add_argument("--epochs", type=int, default=1500, help="Epochs per experiment")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="results/benchmarks", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run a fast 300-epoch test")
    
    args = parser.parse_args()
    
    epochs = 300 if args.quick else args.epochs
    
    if args.mode in ["activations", "all"]:
        run_activation_benchmark(
            activations=["rbf", "rswaf", "iqf", "bspline"],
            solver="tsit5",
            epochs=epochs,
            lr=args.lr,
            save_dir=os.path.join(args.save_dir, "activations"),
        )
        
    if args.mode in ["solvers", "all"]:
        run_solver_benchmark(
            solvers=["tsit5", "rk4", "dopri5", "euler"],
            basis_func="rbf",
            epochs=epochs,
            lr=args.lr,
            save_dir=os.path.join(args.save_dir, "solvers"),
        )
