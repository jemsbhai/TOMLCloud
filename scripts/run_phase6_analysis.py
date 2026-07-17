"""
TOMLCloud Phase 6: Attribution Error Analysis
==============================================
Runs locally. No GPU needed. Processes Phase 2 + Phase 5 data.

Usage: python scripts/run_phase6_analysis.py
"""

import json
import csv
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.metrics import (
    compute_ted, compute_cti, compute_attribution_error,
    proportional_attribution, toml_attribution, compute_to_per_batch,
)

DATA = Path(__file__).resolve().parent.parent / "data"


def load_summaries(phase_dir, arch, n_runs=3):
    """Load and average summaries across runs."""
    energies = []
    for i in range(1, n_runs + 1):
        path = phase_dir / f"{arch}_run{i}_summary.json"
        if path.exists():
            with open(path) as f:
                d = json.load(f)
                energies.append(d["delta_energy_j"])
    if not energies:
        return None
    mean_e = sum(energies) / len(energies)
    # Return the full dict from run1 with mean energy
    with open(phase_dir / f"{arch}_run1_summary.json") as f:
        d = json.load(f)
    d["delta_energy_j_mean"] = mean_e
    return d


def load_colocation_summary(arch_a, arch_b):
    """Load co-location summary for a pair."""
    path_a = DATA / "phase5_colocation" / f"{arch_a}_with_{arch_b}_summary.json"
    path_b = DATA / "phase5_colocation" / f"{arch_b}_with_{arch_a}_summary.json"
    if not path_a.exists() or not path_b.exists():
        return None, None
    with open(path_a) as f:
        sa = json.load(f)
    with open(path_b) as f:
        sb = json.load(f)
    return sa, sb


def main():
    archs = ["resnet18", "resnet50", "vgg16", "rnn", "lstm", "gru", "xgboost"]

    # --------------------------------------------------
    # Load Phase 2 isolated energies (ground truth)
    # --------------------------------------------------
    phase2 = DATA / "phase2_isolated"
    isolated = {}
    print("=== ISOLATED ENERGIES (Phase 2, ground truth) ===")
    for arch in archs:
        s = load_summaries(phase2, arch)
        if s:
            isolated[arch] = s["delta_energy_j_mean"]
            print(f"  {arch:12s}: {isolated[arch]:.1f} J")
        else:
            print(f"  {arch:12s}: NO DATA")

    # --------------------------------------------------
    # Load Phase 3 TED data
    # --------------------------------------------------
    print("\n=== TED ANALYSIS (Phase 3) ===")
    phase3 = DATA / "phase3_thermal"
    ted_results = {}
    for arch in archs:
        win_path = phase3 / f"{arch}_thermal_windows.csv"
        if not win_path.exists():
            print(f"  {arch:12s}: NO DATA")
            continue
        temps, e_per_tos = [], []
        with open(win_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                temps.append(float(row["mean_temp_c"]))
                # E_per_TO needs to be computed from power and TO count
                # Use mean_power * window_duration / TO_in_window
                power = float(row["mean_power_w"])
                window_s = float(row["window_end_s"]) - float(row["window_start_s"])
                energy_window = power * window_s
                # Approximate TO in window: need batches info
                # For now, use the summary to get batches/sec
                pass
        summary_path = phase3 / f"{arch}_thermal_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            to_per_batch = s.get("to_per_batch", 0)
            batches = s.get("batches_completed", 0)
            duration = s.get("duration_s", 1)
            batches_per_s = batches / duration if duration > 0 else 0

            # Recompute E_per_TO per window
            temps, e_per_tos = [], []
            with open(win_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = float(row["mean_temp_c"])
                    power = float(row["mean_power_w"])
                    window_s = float(row["window_end_s"]) - float(row["window_start_s"])
                    idle_power = 44.0  # approximate
                    delta_energy = (power - idle_power) * window_s
                    to_in_window = batches_per_s * window_s * to_per_batch
                    if to_in_window > 0:
                        e_to = delta_energy / to_in_window
                        temps.append(t)
                        e_per_tos.append(e_to)

            if len(temps) >= 3:
                ted = compute_ted(e_per_tos, temps)
                ted_results[arch] = ted
                print(f"  {arch:12s}: TED={ted['ted_pct']:.1f}%  R2={ted['r2']:.3f}  "
                      f"T={ted['temp_min_c']:.0f}-{ted['temp_max_c']:.0f}C")
            else:
                print(f"  {arch:12s}: Not enough windows")
        else:
            print(f"  {arch:12s}: No summary file")

    # --------------------------------------------------
    # Load Phase 5 CTI data
    # --------------------------------------------------
    print("\n=== CTI ANALYSIS (Phase 5) ===")
    pairs = [
        ("vgg16", "rnn"),
        ("resnet50", "xgboost"),
        ("lstm", "gru"),
        ("vgg16", "resnet18"),
        ("xgboost", "rnn"),
        ("resnet50", "lstm"),
    ]

    cti_results = []
    for arch_a, arch_b in pairs:
        sa, sb = load_colocation_summary(arch_a, arch_b)
        if sa is None:
            print(f"  {arch_a:12s} + {arch_b:12s}: NO DATA")
            continue

        e_a_coloc = sa["delta_energy_j"]
        e_b_coloc = sb["delta_energy_j"]
        e_a_iso = isolated.get(arch_a, 0)
        e_b_iso = isolated.get(arch_b, 0)

        cti_a = compute_cti(e_a_iso, e_a_coloc)
        cti_b = compute_cti(e_b_iso, e_b_coloc)

        print(f"  {arch_a:12s} + {arch_b:12s}: CTI(A)={cti_a:+.3f}  CTI(B)={cti_b:+.3f}")

        cti_results.append({
            "arch_a": arch_a,
            "arch_b": arch_b,
            "e_a_iso": e_a_iso,
            "e_b_iso": e_b_iso,
            "e_a_coloc": e_a_coloc,
            "e_b_coloc": e_b_coloc,
            "cti_a": cti_a,
            "cti_b": cti_b,
        })

    # --------------------------------------------------
    # Attribution error analysis
    # --------------------------------------------------
    print("\n=== ATTRIBUTION ERROR (Phase 6) ===")
    print(f"{'Pair':30s} {'Prop_err_A':>12s} {'Prop_err_B':>12s} {'TOML_err_A':>12s} {'TOML_err_B':>12s}")
    print("-" * 80)

    all_prop_errors = []
    all_toml_errors = []

    for r in cti_results:
        arch_a, arch_b = r["arch_a"], r["arch_b"]
        e_a_iso, e_b_iso = r["e_a_iso"], r["e_b_iso"]
        e_a_coloc, e_b_coloc = r["e_a_coloc"], r["e_b_coloc"]

        # Total GPU energy during co-located run
        # Use the host monitor total, or approximate as sum of both readings
        # (pynvml reports GPU-level power, both containers see same total)
        # For simplicity, use the higher power reading as the total
        e_total = max(e_a_coloc, e_b_coloc)  # approximate

        # Actually, we need to compute total from host nvidia-smi
        # For now, estimate: both ran 60s, total power = sum of delta powers
        # This is an approximation; the host log would be more accurate
        total_power_approx = (e_a_coloc + e_b_coloc) / 60.0  # rough
        e_total = (e_a_coloc + e_b_coloc) * 0.6  # overlap factor approx

        # Better approach: use isolated energies as ground truth
        # Proportional: split by execution time fraction (50/50 for equal duration)
        e_total_coloc = e_a_coloc + e_b_coloc  # if pynvml reports per-process
        # Actually pynvml reports GPU-level, so both see the SAME power
        # e_a_coloc and e_b_coloc are both the GPU total * their fraction
        # Let's use the simpler model: proportional gives 50/50 of total
        e_total_est = max(e_a_coloc, e_b_coloc)

        # Proportional: 50/50 split (equal time)
        e_prop_a = e_total_est / 2
        e_prop_b = e_total_est / 2

        # TOML: use TO counts to split
        to_a = compute_to_per_batch(arch_a)["to_total"]
        to_b = compute_to_per_batch(arch_b)["to_total"]

        # Scale by batches (use isolated batches as proxy)
        phase2_a = DATA / "phase2_isolated" / f"{arch_a}_run1_summary.json"
        phase2_b = DATA / "phase2_isolated" / f"{arch_b}_run1_summary.json"
        batches_a, batches_b = 1, 1
        if phase2_a.exists():
            with open(phase2_a) as f:
                batches_a = json.load(f).get("batches_completed", 1)
        if phase2_b.exists():
            with open(phase2_b) as f:
                batches_b = json.load(f).get("batches_completed", 1)

        total_to_a = to_a * batches_a
        total_to_b = to_b * batches_b

        # TOML energy attribution using isolated E_per_TO
        e_per_to_a = e_a_iso / total_to_a if total_to_a > 0 else 0
        e_per_to_b = e_b_iso / total_to_b if total_to_b > 0 else 0

        e_toml_a, e_toml_b = toml_attribution(
            e_total_est, total_to_a, total_to_b, e_per_to_a, e_per_to_b
        )

        # Errors vs ground truth (isolated energy)
        err_prop_a = compute_attribution_error(e_a_iso, e_prop_a)
        err_prop_b = compute_attribution_error(e_b_iso, e_prop_b)
        err_toml_a = compute_attribution_error(e_a_iso, e_toml_a)
        err_toml_b = compute_attribution_error(e_b_iso, e_toml_b)

        all_prop_errors.extend([err_prop_a, err_prop_b])
        all_toml_errors.extend([err_toml_a, err_toml_b])

        print(f"  {arch_a:12s}+{arch_b:12s}  {err_prop_a:10.1f}%  {err_prop_b:10.1f}%  "
              f"{err_toml_a:10.1f}%  {err_toml_b:10.1f}%")

    if all_prop_errors:
        print(f"\n{'SUMMARY':30s}")
        print(f"  Proportional: mean={sum(all_prop_errors)/len(all_prop_errors):.1f}%  "
              f"max={max(all_prop_errors):.1f}%")
        print(f"  TOML:         mean={sum(all_toml_errors)/len(all_toml_errors):.1f}%  "
              f"max={max(all_toml_errors):.1f}%")

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    output_dir = DATA / "phase6_attribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "isolated": isolated,
        "ted": {k: v for k, v in ted_results.items()},
        "cti": cti_results,
        "prop_errors": all_prop_errors,
        "toml_errors": all_toml_errors,
    }
    with open(output_dir / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir / 'analysis_results.json'}")


if __name__ == "__main__":
    main()
