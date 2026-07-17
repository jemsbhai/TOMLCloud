"""
TOMLCloud Phase 6 v1: Corrected Attribution Pipeline + Ablation Ladder
=======================================================================
EXP-001 (see LOGBOOK.md). Supersedes run_phase6_analysis.py (v0), which is
preserved untouched for provenance.

Corrections over v0:
  1. E_total integrated from merged container-side 50 ms power samples over
     the union window (v0 used max of the two container delta energies).
     Independent cross-check against the host nvidia-smi dmon trace.
  2. TED-adjusted coefficients: level from Phase 2 calibration, thermal
     sensitivity (J/TO per degC) from Phase 3 regression, applied at the
     co-located temperature recorded in Phase 5. (v0: no adjustment.)
  3. Actual co-located TO counts from Phase 5 summaries (v0 proxied from
     isolated batch counts).
  4. Measured per-run idle baselines everywhere (v0 hardcoded 44.0 W in the
     Phase 3 path).

Ablation ladder (identical inputs, only the attribution rule varies):
  prop        - proportional split by active duration (cAdvisor-equivalent)
  toml_raw    - TO_count x E_per_TO(calibrated); no residual distribution
  toml_static - Eq. 7, static coefficients, overhead by TO share
  toml_ted    - Eq. 7, TED-adjusted coefficients, overhead by TO share
  (secondary, recorded in CSV only: toml_ted_eqsplit - overhead split equally)

Ground truth: mean Phase 2 isolated delta energy (3 runs). Idle power is
excluded everywhere (delta convention) and treated as platform fixed cost.

Runs locally. No GPU needed.
Usage (from repo root): python scripts/run_phase6_analysis_v1.py
"""

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.metrics import (  # noqa: E402
    compute_attribution_error,
    compute_cti,
    compute_ted,
    toml_attribution,
)
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PHASE2 = DATA / "phase2_isolated"
PHASE3 = DATA / "phase3_thermal"
PHASE4 = DATA / "phase4_spp"
PHASE5 = DATA / "phase5_colocation"
OUT = DATA / "phase6_attribution"

ARCHS = ["resnet18", "resnet50", "vgg16", "rnn", "lstm", "gru", "xgboost"]
PAIRS = [
    ("vgg16", "rnn"),
    ("resnet50", "xgboost"),
    ("lstm", "gru"),
    ("vgg16", "resnet18"),
    ("xgboost", "rnn"),
    ("resnet50", "lstm"),
]
LOADS = [10, 25, 50, 75, 100]

BIN_MS = 100.0          # merge bin width for E_total integration
HOST_ACTIVE_MARGIN = 5  # W above host idle to count a row as active
CLAMP_FRACTION = 0.1    # floor for TED-adjusted coefficient, x calibrated

INPUT_HASHES = {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    INPUT_HASHES[str(path.relative_to(ROOT))] = digest
    return digest


def load_json(path: Path) -> dict:
    sha256_file(path)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phase 2: calibration aggregates (ground truth + coefficient level + T_cal)
# ---------------------------------------------------------------------------

def phase2_aggregate() -> dict:
    agg = {}
    for arch in ARCHS:
        de, temps, epto, tos = [], [], [], []
        for i in (1, 2, 3):
            p = PHASE2 / f"{arch}_run{i}_summary.json"
            if not p.exists():
                continue
            s = load_json(p)
            de.append(s["delta_energy_j"])
            temps.append(s["mean_temp_c"])
            epto.append(s["energy_per_to"])
            tos.append(s["to_total"])
        if not de:
            raise FileNotFoundError(f"No Phase 2 summaries for {arch}")
        mean_de = float(np.mean(de))
        agg[arch] = {
            "e_iso_mean_j": mean_de,
            "e_iso_cv_pct": float(np.std(de) / mean_de * 100.0) if mean_de else 0.0,
            "t_cal_c": float(np.mean(temps)),
            "e_per_to_cal": float(np.mean(epto)),
            "to_iso_mean": float(np.mean(tos)),
            "n_runs": len(de),
        }
    return agg


# ---------------------------------------------------------------------------
# Phase 3: TED thermal sensitivity (slope in J/TO per degC) per architecture
# ---------------------------------------------------------------------------

def phase3_ted() -> dict:
    out = {}
    for arch in ARCHS:
        spath = PHASE3 / f"{arch}_thermal_summary.json"
        wpath = PHASE3 / f"{arch}_thermal_windows.csv"
        if not spath.exists() or not wpath.exists():
            print(f"  WARNING: Phase 3 files missing for {arch}; slope=0")
            out[arch] = {"slope_j_per_to_per_c": 0.0, "ted_pct": 0.0,
                         "r2": 0.0, "n_windows": 0,
                         "temp_min_c": 0.0, "temp_max_c": 0.0}
            continue
        s = load_json(spath)
        idle_w = float(s["idle_power_w"])            # measured, not hardcoded
        to_per_batch = float(s["to_per_batch"])
        batches_per_s = float(s["batches_completed"]) / float(s["duration_s"])
        # Run-average throughput assumption (per-window batch counts were not
        # recorded); documented in LOGBOOK EXP-001 protocol.
        to_rate = batches_per_s * to_per_batch       # TO per second

        sha256_file(wpath)
        temps, etos = [], []
        with open(wpath) as f:
            for row in csv.DictReader(f):
                p = float(row["mean_power_w"])
                t = float(row["mean_temp_c"])
                delta_p = p - idle_w
                if delta_p <= 0 or to_rate <= 0:
                    continue
                etos.append(delta_p / to_rate)        # J/TO in this window
                temps.append(t)
        ted = compute_ted(etos, temps)
        out[arch] = {
            "slope_j_per_to_per_c": ted["ted_slope"],
            "ted_pct": ted["ted_pct"],
            "r2": ted["r2"],
            "n_windows": len(etos),
            "temp_min_c": ted["temp_min_c"],
            "temp_max_c": ted["temp_max_c"],
        }
    return out


# ---------------------------------------------------------------------------
# Phase 5: merged-stream E_total per pair + window diagnostics
# ---------------------------------------------------------------------------

def read_container_stream(path: Path):
    sha256_file(path)
    ts, pw = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts.append(float(row["timestamp_ms"]))
            pw.append(float(row["power_w"]))
    return np.array(ts), np.array(pw)


def merged_e_total(arch_a: str, arch_b: str, idle_a: float, idle_b: float) -> dict:
    ts_a, pw_a = read_container_stream(PHASE5 / f"{arch_a}_with_{arch_b}.csv")
    ts_b, pw_b = read_container_stream(PHASE5 / f"{arch_b}_with_{arch_a}.csv")

    idle_mean = (idle_a + idle_b) / 2.0

    bins = {}
    for ts, pw in ((ts_a, pw_a), (ts_b, pw_b)):
        idx = (ts // BIN_MS).astype(np.int64)
        for i, p in zip(idx, pw):
            acc = bins.setdefault(int(i), [0.0, 0])
            acc[0] += p
            acc[1] += 1

    keys = sorted(bins)
    bin_means = np.array([bins[k][0] / bins[k][1] for k in keys])
    e_union = float(np.sum(np.maximum(bin_means - idle_mean, 0.0)) * (BIN_MS / 1000.0))

    a0, a1 = float(ts_a.min()), float(ts_a.max())
    b0, b1 = float(ts_b.min()), float(ts_b.max())
    conc0, conc1 = max(a0, b0), min(a1, b1)
    union0, union1 = min(a0, b0), max(a1, b1)
    union_s = (union1 - union0) / 1000.0
    conc_s = max(conc1 - conc0, 0.0) / 1000.0

    conc_keys = [k for k in keys if conc0 <= k * BIN_MS <= conc1]
    conc_means = np.array([bins[k][0] / bins[k][1] for k in conc_keys]) if conc_keys else np.array([])
    e_conc = float(np.sum(np.maximum(conc_means - idle_mean, 0.0)) * (BIN_MS / 1000.0)) if conc_keys else 0.0

    return {
        "e_total_union_delta_j": e_union,
        "e_total_concurrent_delta_j": e_conc,
        "union_s": union_s,
        "concurrent_s": conc_s,
        "concurrent_fraction": (conc_s / union_s) if union_s > 0 else 0.0,
        "stagger_start_s": abs(a0 - b0) / 1000.0,
        "idle_mean_w": idle_mean,
    }


# ---------------------------------------------------------------------------
# Host dmon cross-check (independent 1 Hz telemetry, previously unused)
# ---------------------------------------------------------------------------

def host_crosscheck(arch_a: str, arch_b: str, e_union: float) -> dict:
    path = PHASE5 / f"host_{arch_a}_{arch_b}.csv"
    if not path.exists():
        return {"host_file": None, "pct_diff_vs_union": None}
    sha256_file(path)
    pwr, sm = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                pwr.append(float(parts[1]))
                sm.append(float(parts[4]))
            except (ValueError, IndexError):
                continue
    pwr = np.array(pwr)
    sm = np.array(sm)
    if len(pwr) == 0:
        return {"host_file": str(path.name), "pct_diff_vs_union": None}

    idle_rows = pwr[sm == 0]
    host_idle = float(np.median(idle_rows)) if len(idle_rows) else float(np.min(pwr))
    active = pwr > (host_idle + HOST_ACTIVE_MARGIN)
    e_host = float(np.sum(pwr[active] - host_idle) * 1.0)  # 1 Hz rows
    pct = (e_host - e_union) / e_union * 100.0 if e_union > 0 else None
    return {
        "host_file": str(path.name),
        "host_idle_w": host_idle,
        "host_active_s": int(np.sum(active)),
        "host_delta_e_j": e_host,
        "pct_diff_vs_union": pct,
    }


# ---------------------------------------------------------------------------
# Attribution methods
# ---------------------------------------------------------------------------

def adjusted_coefficient(cal: float, slope: float, t_cal: float, t_col: float):
    adj = cal + slope * (t_col - t_cal)
    clamped = False
    floor = CLAMP_FRACTION * cal
    if adj < floor:
        adj, clamped = floor, True
    return adj, clamped


def attribute_pair(pair_meta: dict) -> list:
    """Return one result row per (method, container)."""
    e_total = pair_meta["e_total_union_delta_j"]
    rows = []
    a, b = pair_meta["arch_a"], pair_meta["arch_b"]
    dur_a, dur_b = pair_meta["dur_a"], pair_meta["dur_b"]
    to_a, to_b = pair_meta["to_a"], pair_meta["to_b"]
    cal_a, cal_b = pair_meta["cal_a"], pair_meta["cal_b"]
    adj_a, adj_b = pair_meta["adj_a"], pair_meta["adj_b"]
    true_a, true_b = pair_meta["true_a"], pair_meta["true_b"]

    # prop: split by active duration
    share_a = dur_a / (dur_a + dur_b)
    prop_a, prop_b = e_total * share_a, e_total * (1.0 - share_a)

    # toml_raw: physics estimate only, no residual
    raw_a, raw_b = to_a * cal_a, to_b * cal_b

    # toml_static / toml_ted: Eq. 7 (overhead by TO share)
    st_a, st_b = toml_attribution(e_total, to_a, to_b, cal_a, cal_b)
    td_a, td_b = toml_attribution(e_total, to_a, to_b, adj_a, adj_b)

    # secondary: TED coefficients, overhead split equally
    ov = e_total - (to_a * adj_a + to_b * adj_b)
    eq_a = to_a * adj_a + ov / 2.0
    eq_b = to_b * adj_b + ov / 2.0

    for method, ea, eb in (
        ("prop", prop_a, prop_b),
        ("toml_raw", raw_a, raw_b),
        ("toml_static", st_a, st_b),
        ("toml_ted", td_a, td_b),
        ("toml_ted_eqsplit", eq_a, eq_b),
    ):
        for arch, e_attr, e_true in ((a, ea, true_a), (b, eb, true_b)):
            rows.append({
                "pair": f"{a}+{b}",
                "container": arch,
                "method": method,
                "e_attributed_j": e_attr,
                "e_true_j": e_true,
                "abs_err_pct": compute_attribution_error(e_true, e_attr),
                "signed_err_pct": (e_attr - e_true) / e_true * 100.0 if e_true > 0 else 0.0,
            })
    return rows


# ---------------------------------------------------------------------------
# Phase 4: SPP consistency recompute
# ---------------------------------------------------------------------------

def spp_recompute() -> dict:
    out = {}
    for arch in ARCHS:
        xs, ys = [], []
        for load in LOADS:
            p = PHASE4 / f"{arch}_load{load}_summary.json"
            if not p.exists():
                continue
            s = load_json(p)
            xs.append(float(s["to_total"]) / float(s["duration_s"]))
            ys.append(float(s["delta_power_w"]))
        if len(xs) < 3:
            out[arch] = {"spp_r2": 0.0, "slope": 0.0, "intercept": 0.0, "n": len(xs)}
            continue
        slope, intercept, r, _, _ = stats.linregress(xs, ys)
        out[arch] = {"spp_r2": float(r ** 2), "slope": float(slope),
                     "intercept": float(intercept), "n": len(xs)}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TOMLCloud Phase 6 v1  (EXP-001)  corrected attribution + ablation")
    print("=" * 78)

    # ---- Phase 2 calibration -------------------------------------------------
    cal = phase2_aggregate()
    print("\n=== PHASE 2 CALIBRATION (ground truth + coefficient level) ===")
    print(f"{'arch':10s} {'E_iso mean (J)':>14s} {'CV%':>6s} {'T_cal (C)':>10s} {'E/TO cal':>12s}")
    for arch in ARCHS:
        c = cal[arch]
        print(f"{arch:10s} {c['e_iso_mean_j']:14.1f} {c['e_iso_cv_pct']:6.2f} "
              f"{c['t_cal_c']:10.1f} {c['e_per_to_cal']:12.3e}")

    # ---- Phase 3 TED sensitivity --------------------------------------------
    print("\n=== PHASE 3 TED SENSITIVITY (slope for adjustment) ===")
    ted = phase3_ted()
    print(f"{'arch':10s} {'slope J/TO/C':>14s} {'TED%':>7s} {'r2':>6s} {'T range':>12s} {'wins':>5s}")
    for arch in ARCHS:
        t = ted[arch]
        print(f"{arch:10s} {t['slope_j_per_to_per_c']:14.3e} {t['ted_pct']:7.2f} "
              f"{t['r2']:6.3f} {t['temp_min_c']:5.0f}-{t['temp_max_c']:<5.0f} {t['n_windows']:5d}")

    # ---- Phase 5 pair processing --------------------------------------------
    print("\n=== PHASE 5 E_TOTAL (merged container streams) + HOST CROSS-CHECK ===")
    pair_metas, crosschecks, cti_rows = [], [], []
    clamp_warnings = []
    for a, b in PAIRS:
        sa = load_json(PHASE5 / f"{a}_with_{b}_summary.json")
        sb = load_json(PHASE5 / f"{b}_with_{a}_summary.json")
        et = merged_e_total(a, b, sa["idle_power_w"], sb["idle_power_w"])
        hc = host_crosscheck(a, b, et["e_total_union_delta_j"])

        adj_a, cl_a = adjusted_coefficient(
            cal[a]["e_per_to_cal"], ted[a]["slope_j_per_to_per_c"],
            cal[a]["t_cal_c"], sa["mean_temp_c"])
        adj_b, cl_b = adjusted_coefficient(
            cal[b]["e_per_to_cal"], ted[b]["slope_j_per_to_per_c"],
            cal[b]["t_cal_c"], sb["mean_temp_c"])
        if cl_a:
            clamp_warnings.append(f"{a} in {a}+{b}")
        if cl_b:
            clamp_warnings.append(f"{b} in {a}+{b}")

        meta = {
            "arch_a": a, "arch_b": b,
            "dur_a": sa["duration_s"], "dur_b": sb["duration_s"],
            "to_a": sa["to_total"], "to_b": sb["to_total"],
            "cal_a": cal[a]["e_per_to_cal"], "cal_b": cal[b]["e_per_to_cal"],
            "adj_a": adj_a, "adj_b": adj_b,
            "true_a": cal[a]["e_iso_mean_j"], "true_b": cal[b]["e_iso_mean_j"],
            "t_col_a": sa["mean_temp_c"], "t_col_b": sb["mean_temp_c"],
            **et,
        }
        pair_metas.append(meta)
        crosschecks.append({"pair": f"{a}+{b}", **hc,
                            "container_union_delta_e_j": et["e_total_union_delta_j"]})

        diff = hc.get("pct_diff_vs_union")
        diff_s = f"{diff:+6.1f}%" if diff is not None else "   n/a"
        print(f"{a:9s}+{b:9s}  E_union={et['e_total_union_delta_j']:9.0f} J  "
              f"conc_frac={et['concurrent_fraction']:5.2f}  "
              f"stagger={et['stagger_start_s']:4.1f}s  host vs union: {diff_s}")

        # CTI decomposition (per container)
        for arch, s_self, adj, cl in ((a, sa, adj_a, cl_a), (b, sb, adj_b, cl_b)):
            iso = cal[arch]["e_iso_mean_j"]
            cti_rows.append({
                "pair": f"{a}+{b}",
                "container": arch,
                "raw_cti": compute_cti(iso, s_self["delta_energy_j"]),
                "reported_delta_e_j": s_self["delta_energy_j"],
                "to_coloc": s_self["to_total"],
                "to_iso_mean": cal[arch]["to_iso_mean"],
                "throughput_ratio_pct": s_self["to_total"] / cal[arch]["to_iso_mean"] * 100.0,
                "t_cal_c": cal[arch]["t_cal_c"],
                "t_coloc_c": s_self["mean_temp_c"],
                "temp_delta_c": s_self["mean_temp_c"] - cal[arch]["t_cal_c"],
                "e_per_to_cal": cal[arch]["e_per_to_cal"],
                "e_per_to_adj": adj,
                "coeff_adjustment_pct": (adj / cal[arch]["e_per_to_cal"] - 1.0) * 100.0,
                "clamped": cl,
            })

    # ---- Ablation ------------------------------------------------------------
    all_rows = []
    for meta in pair_metas:
        all_rows.extend(attribute_pair(meta))

    # attach TED-attributed energy + contamination factor to CTI rows
    ted_attr = {(r["pair"], r["container"]): r["e_attributed_j"]
                for r in all_rows if r["method"] == "toml_ted"}
    for r in cti_rows:
        ea = ted_attr.get((r["pair"], r["container"]), 0.0)
        r["e_attr_ted_j"] = ea
        r["contamination_factor"] = (r["reported_delta_e_j"] / ea) if ea > 0 else None

    print("\n=== ABLATION: ABSOLUTE ATTRIBUTION ERROR PER CONTAINER (%) ===")
    headline = ["prop", "toml_raw", "toml_static", "toml_ted"]
    by_key = {}
    for r in all_rows:
        by_key.setdefault((r["pair"], r["container"]), {})[r["method"]] = r
    print(f"{'pair':22s} {'container':10s}" + "".join(f"{m:>13s}" for m in headline))
    for (pair, cont), methods in by_key.items():
        vals = "".join(f"{methods[m]['abs_err_pct']:13.1f}" for m in headline)
        print(f"{pair:22s} {cont:10s}{vals}")

    print("\n=== ABLATION SUMMARY (over 12 container observations) ===")
    summary = {}
    print(f"{'method':16s} {'mean%':>8s} {'median%':>9s} {'max%':>8s}")
    for m in headline + ["toml_ted_eqsplit"]:
        errs = [r["abs_err_pct"] for r in all_rows if r["method"] == m]
        summary[m] = {"mean_err_pct": float(np.mean(errs)),
                      "median_err_pct": float(median(errs)),
                      "max_err_pct": float(np.max(errs)),
                      "n": len(errs)}
        tag = "" if m in headline else "  (secondary)"
        print(f"{m:16s} {summary[m]['mean_err_pct']:8.1f} "
              f"{summary[m]['median_err_pct']:9.1f} {summary[m]['max_err_pct']:8.1f}{tag}")

    if clamp_warnings:
        print("\nWARNING: TED coefficient clamped for: " + ", ".join(clamp_warnings))

    # ---- SPP consistency -----------------------------------------------------
    print("\n=== PHASE 4 SPP RECOMPUTE (consistency vs Table IV) ===")
    spp = spp_recompute()
    for arch in ARCHS:
        s = spp[arch]
        print(f"{arch:10s} r2={s['spp_r2']:.3f}  slope={s['slope']:.3e}  "
              f"intercept={s['intercept']:+.1f} W  (n={s['n']})")

    # ---- Archive -------------------------------------------------------------
    def write_csv(path, rows, fields):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write_csv(OUT / "attribution_comparison.csv", all_rows,
              ["pair", "container", "method", "e_attributed_j", "e_true_j",
               "abs_err_pct", "signed_err_pct"])
    write_csv(OUT / "ablation_summary.csv",
              [{"method": m, **v} for m, v in summary.items()],
              ["method", "mean_err_pct", "median_err_pct", "max_err_pct", "n"])
    write_csv(OUT / "cti_decomposition.csv", cti_rows, list(cti_rows[0].keys()))
    write_csv(OUT / "host_crosscheck.csv", crosschecks, list(crosschecks[0].keys()))
    write_csv(OUT / "ted_recomputed.csv",
              [{"arch": a, **ted[a]} for a in ARCHS],
              ["arch"] + list(ted[ARCHS[0]].keys()))
    write_csv(OUT / "spp_recomputed.csv",
              [{"arch": a, **spp[a]} for a in ARCHS],
              ["arch"] + list(spp[ARCHS[0]].keys()))

    results = {
        "experiment": "EXP-001",
        "calibration_phase2": cal,
        "ted_phase3": ted,
        "pair_metas": pair_metas,
        "ablation_rows": all_rows,
        "ablation_summary": summary,
        "cti_decomposition": cti_rows,
        "host_crosscheck": crosschecks,
        "spp_phase4": spp,
    }
    with open(OUT / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    metadata = {
        "experiment": "EXP-001",
        "script": "scripts/run_phase6_analysis_v1.py",
        "run_at": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "bin_ms": BIN_MS,
        "host_active_margin_w": HOST_ACTIVE_MARGIN,
        "clamp_fraction": CLAMP_FRACTION,
        "input_sha256": INPUT_HASHES,
    }
    with open(OUT / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArchived {len(INPUT_HASHES)} input hashes and 8 artifacts to {OUT}")
    print("Done. Update LOGBOOK.md EXP-001 with Results/Observations/Interpretation.")


if __name__ == "__main__":
    main()
