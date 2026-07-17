"""
TOMLCloud - Figure Generation v1
=================================
Regenerates all five paper figures for the CloudCom 2026 revision.
Supersedes generate_figures.py (v0), preserved untouched for provenance.

Changes over v0:
  1. All values computed as means of 3 runs where repetitions exist
     (v0 used run 1 for Fig. 1 and the cross-platform figure).
  2. Measured per-run idle baselines everywhere (v0 hardcoded 48.0 W in the
     thermal figures).
  3. No redundant axes titles duplicating LaTeX captions (IC2E R3).
  4. Larger fonts and thicker lines in the thermal trajectory figure (R3).
  5. Legend placements chosen to avoid overlapping bars or text (R3).
  6. New Fig. 4: TED regression scatter (ResNet-50 and LSTM) replacing the
     cold-start vs equilibrium bars, so the figure shows the exact quantity
     reported in Table III.
  7. Cross-platform figure renamed fig5_cross_platform.pdf to match its
     rendered figure number.

Outputs (PDF + PNG) to figures/:
  fig1_power_range, fig2_container_overhead, fig3_thermal_trajectory,
  fig4_ted_regression, fig5_cross_platform

Run locally: python scripts/generate_figures_v1.py
Requires: matplotlib, numpy, scipy
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FIGS = BASE / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

PHASE1 = DATA / "phase1_baremetal" / "phase1_baremetal"
PHASE2 = DATA / "phase2_isolated"
PHASE3 = DATA / "phase3_thermal"

ARCH_LABELS = ["ResNet-18", "ResNet-50", "VGG-16", "RNN", "LSTM", "GRU", "XGBoost"]
ARCH_KEYS = ["resnet18", "resnet50", "vgg16", "rnn", "lstm", "gru", "xgboost"]

COL_W = 3.5
COL_H = 2.4

C_DARK = '#2166ac'
C_LIGHT = '#67a9cf'
C_RED = '#b2182b'


def load_json(path):
    with open(path) as f:
        return json.load(f)


def phase1_mean(key, field):
    vals = [load_json(PHASE1 / f"{key}_run{i}_summary.json")[field] for i in (1, 2, 3)]
    return float(np.mean(vals))


def phase2_mean(key, field):
    vals = [load_json(PHASE2 / f"{key}_run{i}_summary.json")[field] for i in (1, 2, 3)]
    return float(np.mean(vals))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)
    print(f"  {name} saved.")


# =========================================================================
# FIGURE 1: Power draw across architectures (means of 3 bare-metal runs)
# =========================================================================
def fig1_power_range():
    powers = [phase1_mean(k, "delta_power_w") for k in ARCH_KEYS]

    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    colors = [C_DARK if p > 100 else C_LIGHT for p in powers]
    bars = ax.bar(ARCH_LABELS, powers, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Delta Power (W)')
    ax.set_xlabel('Architecture')
    ax.tick_params(axis='x', rotation=35)

    ratio = max(powers) / min(powers)
    ax.annotate(f'{ratio:.1f}x range',
                xy=(0.5, 0.92), xycoords='axes fraction',
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    for bar, p in zip(bars, powers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{p:.0f}W', ha='center', va='bottom', fontsize=7)

    ax.set_ylim(0, max(powers) * 1.18)
    save(fig, "fig1_power_range")
    print(f"    power range: {min(powers):.1f}W to {max(powers):.1f}W, ratio {ratio:.2f}x")


# =========================================================================
# FIGURE 2: Bare-metal vs container energy (means of 3)
# =========================================================================
def fig2_container_overhead():
    bm = [phase1_mean(k, "delta_energy_j") for k in ARCH_KEYS]
    ct = [phase2_mean(k, "delta_energy_j") for k in ARCH_KEYS]
    overheads = [(c - b) / b * 100 for b, c in zip(bm, ct)]

    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    x = np.arange(len(ARCH_LABELS))
    width = 0.35
    ax.bar(x - width / 2, bm, width, label='Bare-metal',
           color=C_DARK, edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, ct, width, label='Container',
           color=C_RED, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Delta Energy (J)')
    ax.set_xlabel('Architecture')
    ax.set_xticks(x)
    ax.set_xticklabels(ARCH_LABELS, rotation=35, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)

    for i, oh in enumerate(overheads):
        if abs(oh) > 5:
            ax.annotate(f'{oh:+.0f}%\n(thermal)',
                        xy=(x[i] + width / 2, ct[i]),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=6, color=C_RED, fontstyle='italic')

    ax.set_ylim(0, max(max(bm), max(ct)) * 1.22)
    save(fig, "fig2_container_overhead")


# =========================================================================
# FIGURE 3: Thermal trajectory for ResNet-18 and ResNet-50 (measured idle)
# =========================================================================
def _phase3_windows(arch):
    """Per-window (time_min, temp_c, e_to) using measured idle and
    run-average throughput (per-window batch counts were not recorded)."""
    s = load_json(PHASE3 / f"{arch}_thermal_summary.json")
    idle = float(s["idle_power_w"])
    to_rate = (s["batches_completed"] / s["duration_s"]) * s["to_per_batch"]

    times, temps, etos = [], [], []
    with open(PHASE3 / f"{arch}_thermal_windows.csv") as f:
        for row in csv.DictReader(f):
            p = float(row["mean_power_w"])
            t = float(row["mean_temp_c"])
            t0 = float(row["window_start_s"])
            delta = p - idle
            if delta <= 0 or to_rate <= 0:
                continue
            times.append(t0 / 60.0)
            temps.append(t)
            etos.append(delta / to_rate)
    return times, temps, etos


def fig3_thermal_trajectory():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2 + 0.3, COL_H + 0.3))

    for ax, arch, label in [(axes[0], "resnet18", "ResNet-18"),
                            (axes[1], "resnet50", "ResNet-50")]:
        times, temps, etos = _phase3_windows(arch)

        ln1 = ax.plot(times, [e * 1e15 for e in etos], color=C_DARK,
                      linewidth=1.4, label='E/TO')
        ax.set_xlabel('Time (min)', fontsize=11)
        ax.set_ylabel('E/TO (x10$^{-15}$ J)', color=C_DARK, fontsize=11)
        ax.tick_params(axis='y', labelcolor=C_DARK, labelsize=9)
        ax.tick_params(axis='x', labelsize=9)
        ax.set_title(label, fontsize=11)

        ax_t = ax.twinx()
        ln2 = ax_t.plot(times, temps, color=C_RED, linewidth=1.4,
                        linestyle='--', label='Temp')
        ax_t.set_ylabel('Temperature (C)', color=C_RED, fontsize=11)
        ax_t.tick_params(axis='y', labelcolor=C_RED, labelsize=9)

        lns = ln1 + ln2
        ax.legend(lns, [l.get_label() for l in lns], loc='lower right', fontsize=9)

    save(fig, "fig3_thermal_trajectory")


# =========================================================================
# FIGURE 4 (new): TED regressions for ResNet-50 and LSTM
# =========================================================================
def fig4_ted_regression():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2 + 0.3, COL_H + 0.3))

    for ax, arch, label in [(axes[0], "resnet50", "ResNet-50"),
                            (axes[1], "lstm", "LSTM")]:
        _, temps, etos = _phase3_windows(arch)
        t = np.array(temps)
        e = np.array(etos) * 1e15

        slope, intercept, r, _, _ = stats.linregress(t, e)
        tt = np.linspace(t.min(), t.max(), 50)

        ax.scatter(t, e, s=8, color=C_DARK, alpha=0.55, edgecolors='none',
                   label='10s windows')
        ax.plot(tt, intercept + slope * tt, color=C_RED, linewidth=1.6,
                label='Linear fit')

        e_min = intercept + slope * t.min()
        e_max = intercept + slope * t.max()
        ted_pct = (e_max - e_min) / e_min * 100.0

        ax.set_xlabel('Die temperature (C)', fontsize=11)
        ax.set_ylabel('E/TO (x10$^{-15}$ J)', fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.annotate(f'TED = {ted_pct:.1f}%\n$r^2$ = {r**2:.2f}',
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.85))
        ax.legend(loc='lower right', fontsize=8)
        print(f"    {label}: TED {ted_pct:.2f}%  r2 {r**2:.3f}  "
              f"slope {slope*1e-15:.3e} J/TO/C  (check vs Table III)")

    save(fig, "fig4_ted_regression")


# =========================================================================
# FIGURE 5: Cross-platform E/TO (A100 means of 3 vs published 4090 values)
# =========================================================================
def fig5_cross_platform():
    labels = ARCH_LABELS[:6]
    keys = ARCH_KEYS[:6]

    # RTX 4090 Laptop E/TO from the published FLAIRS-39 TOML data
    eto_4090 = [4.389e-15, 7.748e-15, 4.555e-15, 4.887e-15, 5.531e-15, 6.746e-15]
    eto_a100 = [phase1_mean(k, "energy_per_to") for k in keys]

    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, [e * 1e15 for e in eto_4090], width,
           label='RTX 4090 (150W)', color=C_LIGHT, edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, [e * 1e15 for e in eto_a100], width,
           label='A100 (400W)', color=C_DARK, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('E/TO (x10$^{-15}$ J)')
    ax.set_xlabel('Architecture')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)

    print("    Table VII check (A100 means, ratios):")
    for i, lab in enumerate(labels):
        ratio = eto_a100[i] / eto_4090[i]
        y = max(eto_4090[i], eto_a100[i]) * 1e15
        ax.text(x[i], y + 0.25, f'{ratio:.2f}x', ha='center', fontsize=6, color='gray')
        print(f"      {lab:10s}  4090 {eto_4090[i]*1e15:.2f}  "
              f"A100 {eto_a100[i]*1e15:.2f}  ratio {ratio:.2f}")

    ax.set_ylim(0, max(max(eto_4090), max(eto_a100)) * 1e15 * 1.28)
    save(fig, "fig5_cross_platform")


if __name__ == "__main__":
    print("Generating figures (v1)...")
    print(f"Data dir: {DATA}")
    print(f"Output dir: {FIGS}\n")
    fig1_power_range()
    fig2_container_overhead()
    fig3_thermal_trajectory()
    fig4_ted_regression()
    fig5_cross_platform()
    print("\nDone. Upload the five PDFs to Overleaf; fig4_cross_platform.pdf and")
    print("fig5_ted_cold_hot.pdf are superseded and can be removed there.")
