"""
TOMLCloud - Common Benchmark Runner
====================================
Shared inference loop used by all 7 architecture benchmark scripts.
Handles duration-based runs, load-level control (Phase 4), and
telemetry via GPUThermalMonitor.

Author: Muntaser Syed
Date: May 2026
"""

import sys
import os
import time
import json
import argparse
import gc
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

# Add parent directory to path so shared module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.power_monitor import GPUThermalMonitor
from shared.metrics import compute_to_per_batch


def run_benchmark(
    arch: str,
    create_model_fn: Callable,
    create_input_fn: Callable,
    duration_s: float = 60.0,
    load_pct: int = 100,
    output: Optional[str] = None,
    cooldown: bool = True,
    cooldown_target_c: int = 35,
    batch_size: int = 32,
    sample_interval_ms: int = 50,
    window_stats_s: float = 0.0,
) -> Dict:
    """
    Run a timed inference benchmark with full telemetry.

    Args:
        arch: Architecture key (e.g., 'resnet18', 'lstm').
        create_model_fn: Callable() -> (model, device). Returns the model on GPU.
        create_input_fn: Callable(device, batch_size) -> input_tensor.
        duration_s: Total run duration in seconds.
        load_pct: Target load percentage (10, 25, 50, 75, 100). Controls duty cycle.
        output: Path to save raw telemetry CSV. None = don't save.
        cooldown: Whether to wait for GPU to cool before starting.
        cooldown_target_c: Target temperature for cooldown.
        batch_size: Inference batch size.
        sample_interval_ms: Telemetry sampling interval.
        window_stats_s: If > 0, also save windowed stats CSV (for Phase 3).

    Returns:
        Dict with summary statistics and batch count.
    """
    device = torch.device("cuda")

    # Initialize monitor
    monitor = GPUThermalMonitor(sample_interval_ms=sample_interval_ms)

    # Optional cooldown
    if cooldown:
        torch.cuda.empty_cache()
        gc.collect()
        monitor.wait_cooldown(target_c=cooldown_target_c)

    # Measure idle baseline
    idle_stats = monitor.measure_idle(duration_s=10.0, warmup_s=2.0)

    # Create model and input
    print(f"Creating {arch} model...")
    model, device = create_model_fn()
    x = create_input_fn(device, batch_size)

    # Warmup (10 batches, not measured)
    print("Warmup...", end=" ", flush=True)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    torch.cuda.synchronize()
    print("done")

    # Estimate time per batch (for duty cycle control)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    torch.cuda.synchronize()
    time_per_batch = (time.time() - t0) / 5.0

    # Compute duty cycle sleep for load_pct
    if load_pct >= 100:
        sleep_per_batch = 0.0
    else:
        # At load_pct=50: run 1 batch, sleep 1 batch duration
        # At load_pct=25: run 1 batch, sleep 3 batch durations
        # At load_pct=10: run 1 batch, sleep 9 batch durations
        ratio = (100.0 / max(load_pct, 1)) - 1.0
        sleep_per_batch = time_per_batch * ratio

    print(f"Time per batch: {time_per_batch * 1000:.1f}ms")
    print(f"Load target: {load_pct}% (sleep {sleep_per_batch * 1000:.1f}ms between batches)")
    print(f"Running for {duration_s}s...")

    # Clear GPU state before benchmark
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(0.5)

    # Main benchmark loop
    monitor.start()
    batches_completed = 0
    run_start = time.time()

    with torch.no_grad():
        while time.time() - run_start < duration_s:
            _ = model(x)
            torch.cuda.synchronize()
            monitor.sample()
            batches_completed += 1

            if sleep_per_batch > 0:
                time.sleep(sleep_per_batch)

    summary = monitor.stop()

    # Compute TO counts
    to_per_batch = compute_to_per_batch(arch, batch_size)
    to_total = to_per_batch["to_total"] * batches_completed

    summary.update({
        "arch": arch,
        "batch_size": batch_size,
        "batches_completed": batches_completed,
        "load_pct": load_pct,
        "time_per_batch_ms": time_per_batch * 1000,
        "to_per_batch": to_per_batch["to_total"],
        "to_total": to_total,
        "to_compute": to_per_batch["to_compute"] * batches_completed,
        "to_memory": to_per_batch["to_memory"] * batches_completed,
        "energy_per_to": summary["delta_energy_j"] / to_total if to_total > 0 else 0.0,
    })

    # Print summary
    print(f"\n--- {arch} results ---")
    print(f"  Duration:     {summary['duration_s']:.2f}s")
    print(f"  Batches:      {batches_completed}")
    print(f"  Delta power:  {summary['delta_power_w']:.1f}W")
    print(f"  Delta energy: {summary['delta_energy_j']:.1f}J")
    print(f"  Temp range:   {summary['min_temp_c']}-{summary['max_temp_c']}C")
    print(f"  Clock range:  {summary['min_clock_mhz']}-{summary['max_clock_mhz']} MHz")
    print(f"  TO total:     {to_total:.2e}")
    print(f"  E/TO:         {summary['energy_per_to']:.4e} J/TO")

    # Save telemetry
    if output:
        monitor.save_readings(output)
        if window_stats_s > 0:
            win_path = output.replace(".csv", "_windows.csv")
            monitor.save_windows(win_path, window_s=window_stats_s)

    # Save summary as JSON alongside CSV
    if output:
        summary_path = output.replace(".csv", "_summary.json")
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary saved to {summary_path}")

    # Cleanup
    del model, x
    torch.cuda.empty_cache()
    gc.collect()
    monitor.shutdown()

    return summary


def make_parser(arch: str) -> argparse.ArgumentParser:
    """Create standard CLI parser for benchmark scripts."""
    parser = argparse.ArgumentParser(
        description=f"TOMLCloud benchmark: {arch}"
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Run duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--load-pct", type=int, default=100, choices=[10, 25, 50, 75, 100],
        help="Target GPU load percentage (default: 100)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path for raw telemetry"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Inference batch size (default: 32)"
    )
    parser.add_argument(
        "--no-cooldown", action="store_true",
        help="Skip GPU cooldown before run"
    )
    parser.add_argument(
        "--cooldown-target", type=int, default=35,
        help="Cooldown target temperature in C (default: 35)"
    )
    parser.add_argument(
        "--window-stats", type=float, default=0.0,
        help="If > 0, save windowed stats with this window size in seconds"
    )
    return parser
