"""TOMLCloud benchmark: XGBoost GPU (gradient-boosted trees)"""

import sys
import time
import json
import gc
import argparse
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.power_monitor import GPUThermalMonitor
from shared.metrics import compute_to_per_batch


def run_xgboost_benchmark(
    duration_s: float = 60.0,
    load_pct: int = 100,
    output: str = None,
    cooldown: bool = True,
    cooldown_target_c: int = 35,
    window_stats_s: float = 0.0,
):
    """
    XGBoost GPU benchmark.

    XGBoost uses a different execution model from PyTorch:
    - Train once, then loop inference calls for the full duration.
    - model.predict() replaces model(x).

    The training phase is excluded from energy measurement.
    Only inference is measured, matching how XGBoost would run
    as a serving container in production.
    """
    N_SAMPLES = 100000
    N_FEATURES = 100
    N_CLASSES = 10

    monitor = GPUThermalMonitor(sample_interval_ms=50)

    if cooldown:
        torch.cuda.empty_cache()
        gc.collect()
        monitor.wait_cooldown(target_c=cooldown_target_c)

    # Measure idle
    idle_stats = monitor.measure_idle(duration_s=10.0, warmup_s=2.0)

    # Generate dataset
    print(f"Generating dataset ({N_SAMPLES} samples, {N_FEATURES} features)...")
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=50,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        random_state=42,
    )
    split = int(0.8 * N_SAMPLES)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model (not measured)
    print("Training XGBoost model (not included in energy measurement)...")
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=N_CLASSES,
        max_depth=6,
        n_estimators=100,
        tree_method="hist",
        device="cuda",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    print("Training complete.")

    # Warmup inference
    print("Warmup...", end=" ", flush=True)
    for _ in range(5):
        _ = model.predict(X_test)
    print("done")

    # Estimate time per inference call
    t0 = time.time()
    for _ in range(5):
        _ = model.predict(X_test)
    time_per_call = (time.time() - t0) / 5.0

    # Duty cycle for load control
    if load_pct >= 100:
        sleep_per_call = 0.0
    else:
        ratio = (100.0 / max(load_pct, 1)) - 1.0
        sleep_per_call = time_per_call * ratio

    print(f"Time per inference call: {time_per_call * 1000:.1f}ms")
    print(f"Load target: {load_pct}% (sleep {sleep_per_call * 1000:.1f}ms between calls)")
    print(f"Running for {duration_s}s...")

    gc.collect()
    time.sleep(0.5)

    # Benchmark loop (inference only)
    monitor.start()
    calls_completed = 0
    run_start = time.time()

    while time.time() - run_start < duration_s:
        _ = model.predict(X_test)
        monitor.sample()
        calls_completed += 1
        if sleep_per_call > 0:
            time.sleep(sleep_per_call)

    summary = monitor.stop()

    # TO counts: each call processes N_TEST_SAMPLES
    to_per_call = compute_to_per_batch("xgboost", batch_size=1)
    # xgboost TO is per-call (processes all test samples)
    to_total = to_per_call["to_total"] * calls_completed

    summary.update({
        "arch": "xgboost",
        "batch_size": len(X_test),
        "batches_completed": calls_completed,
        "load_pct": load_pct,
        "time_per_batch_ms": time_per_call * 1000,
        "to_per_batch": to_per_call["to_total"],
        "to_total": to_total,
        "to_compute": to_per_call["to_compute"] * calls_completed,
        "to_memory": to_per_call["to_memory"] * calls_completed,
        "energy_per_to": summary["delta_energy_j"] / to_total if to_total > 0 else 0.0,
        "n_test_samples": len(X_test),
        "n_features": N_FEATURES,
        "n_estimators": 100,
        "max_depth": 6,
    })

    print(f"\n--- xgboost results ---")
    print(f"  Duration:     {summary['duration_s']:.2f}s")
    print(f"  Infer calls:  {calls_completed}")
    print(f"  Delta power:  {summary['delta_power_w']:.1f}W")
    print(f"  Delta energy: {summary['delta_energy_j']:.1f}J")
    print(f"  Temp range:   {summary['min_temp_c']}-{summary['max_temp_c']}C")
    print(f"  Clock range:  {summary['min_clock_mhz']}-{summary['max_clock_mhz']} MHz")
    print(f"  TO total:     {to_total:.2e}")
    print(f"  E/TO:         {summary['energy_per_to']:.4e} J/TO")

    if output:
        monitor.save_readings(output)
        if window_stats_s > 0:
            win_path = output.replace(".csv", "_windows.csv")
            monitor.save_windows(win_path, window_s=window_stats_s)

        summary_path = output.replace(".csv", "_summary.json")
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary saved to {summary_path}")

    monitor.shutdown()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOMLCloud benchmark: xgboost")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--load-pct", type=int, default=100, choices=[10, 25, 50, 75, 100])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-cooldown", action="store_true")
    parser.add_argument("--cooldown-target", type=int, default=35)
    parser.add_argument("--window-stats", type=float, default=0.0)
    args = parser.parse_args()

    run_xgboost_benchmark(
        duration_s=args.duration,
        load_pct=args.load_pct,
        output=args.output,
        cooldown=not args.no_cooldown,
        cooldown_target_c=args.cooldown_target,
        window_stats_s=args.window_stats,
    )
