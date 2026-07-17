"""
TOMLCloud - Figure Generation
=============================
Generates all paper figures from Phase 1-3 data.
Run locally: python scripts/generate_figures.py

Requires: matplotlib, numpy, json
"""

import json
import csv
import os
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
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

# Column width for IEEE double-column: ~3.5 inches
COL_W = 3.5
COL_H = 2.4


def load_json(path):
    with open(path) as f:
        return json.load(f)


# =========================================================================
# FIGURE 1: Power draw across architectures (bar chart)
# =========================================================================
def fig1_power_range():
    archs = ["ResNet-18", "ResNet-50", "VGG-16", "RNN", "LSTM", "GRU", "XGBoost"]
    keys = ["resnet18", "resnet50", "vgg16", "rnn", "lstm", "gru", "xgboost"]
    colors_compute = '#2166ac'
    colors_light = '#67a9cf'

    powers = []
    energies = []
    for k in keys:
        s = load_json(DATA / "phase1_baremetal" / "phase1_baremetal" / f"{k}_run1_summary.json")
        powers.append(s["delta_power_w"])
        energies.append(s["delta_energy_j"])

    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    colors = [colors_compute if p > 100 else colors_light for p in powers]
    bars = ax.bar(archs, powers, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Delta Power (W)')
    ax.set_xlabel('Architecture')
    ax.set_title('GPU Power Draw Across ML Architectures (A100 40GB)')
    ax.tick_params(axis='x', rotation=35)

    # Annotate the range
    ax.annotate(f'7.7x range',
                xy=(0.5, 0.92), xycoords='axes fraction',
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    # Add value labels on bars
    for bar, p in zip(bars, powers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{p:.0f}W', ha='center', va='bottom', fontsize=7)

    ax.set_ylim(0, max(powers) * 1.15)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_power_range.pdf")
    fig.savefig(FIGS / "fig1_power_range.png")
    plt.close(fig)
    print("Figure 1: Power range saved.")


# =========================================================================
# FIGURE 2: Container overhead with thermal annotation
# =========================================================================
def fig2_container_overhead():
    archs = ["ResNet-18", "ResNet-50", "VGG-16", "RNN", "LSTM", "GRU", "XGBoost"]
    keys = ["resnet18", "resnet50", "vgg16", "rnn", "lstm", "gru", "xgboost"]

    bm_means = []
    ct_means = []
    for k in keys:
        bm_e = []
        ct_e = []
        for i in range(1, 4):
            bm = load_json(DATA / "phase1_baremetal" / "phase1_baremetal" / f"{k}_run{i}_summary.json")
            ct = load_json(DATA / "phase2_isolated" / f"{k}_run{i}_summary.json")
            bm_e.append(bm["delta_energy_j"])
            ct_e.append(ct["delta_energy_j"])
        bm_means.append(np.mean(bm_e))
        ct_means.append(np.mean(ct_e))

    overheads = [(c - b) / b * 100 for b, c in zip(bm_means, ct_means)]

    fig, ax = plt.subplots(figsize=(COL_W, COL_H))

    x = np.arange(len(archs))
    width = 0.35
    bars_bm = ax.bar(x - width/2, bm_means, width, label='Bare-metal', color='#2166ac', edgecolor='black', linewidth=0.5)
    bars_ct = ax.bar(x + width/2, ct_means, width, label='Container', color='#b2182b', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Delta Energy (J)')
    ax.set_xlabel('Architecture')
    ax.set_title('Bare-Metal vs. Container Energy (60s runs, mean of 3)')
    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=35, ha='right')
    ax.legend(loc='upper left', framealpha=0.9)

    # Annotate thermal artifacts
    for i, (arch, oh) in enumerate(zip(archs, overheads)):
        if abs(oh) > 5:
            ax.annotate(f'{oh:+.0f}%\n(thermal)',
                        xy=(x[i] + width/2, ct_means[i]),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=6, color='#b2182b',
                        fontstyle='italic')

    fig.tight_layout()
    fig.savefig(FIGS / "fig2_container_overhead.pdf")
    fig.savefig(FIGS / "fig2_container_overhead.png")
    plt.close(fig)
    print("Figure 2: Container overhead saved.")


# =========================================================================
# FIGURE 3: Thermal trajectory (E/TO vs time) for ResNet-18 and ResNet-50
# =========================================================================
def fig3_thermal_trajectory():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_W * 2 + 0.3, COL_H))

    for ax, arch, label in [(ax1, "resnet18", "ResNet-18"), (ax2, "resnet50", "ResNet-50")]:
        win_path = DATA / "phase3_thermal" / f"{arch}_thermal_windows.csv"
        sum_path = DATA / "phase3_thermal" / f"{arch}_thermal_summary.json"
        if not win_path.exists():
            continue

        s = load_json(sum_path)
        bps = s["batches_completed"] / s["duration_s"]
        to_pb = s["to_per_batch"]
        idle = 48.0

        times, temps, etos = [], [], []
        with open(win_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_s = float(row["window_start_s"])
                temp = float(row["mean_temp_c"])
                power = float(row["mean_power_w"])
                win_s = float(row["window_end_s"]) - t_s
                de = (power - idle) * win_s
                to_w = bps * win_s * to_pb
                if to_w > 0:
                    times.append(t_s / 60.0)  # minutes
                    temps.append(temp)
                    etos.append(de / to_w)

        color_eto = '#2166ac'
        color_temp = '#b2182b'

        ln1 = ax.plot(times, [e * 1e15 for e in etos], color=color_eto, linewidth=1, label='E/TO')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('E/TO (x10$^{-15}$ J)', color=color_eto)
        ax.tick_params(axis='y', labelcolor=color_eto)
        ax.set_title(label)

        ax_t = ax.twinx()
        ln2 = ax_t.plot(times, temps, color=color_temp, linewidth=1, linestyle='--', label='Temp')
        ax_t.set_ylabel('Temperature (C)', color=color_temp)
        ax_t.tick_params(axis='y', labelcolor=color_temp)

        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='lower right', fontsize=7)

    fig.tight_layout()
    fig.savefig(FIGS / "fig3_thermal_trajectory.pdf")
    fig.savefig(FIGS / "fig3_thermal_trajectory.png")
    plt.close(fig)
    print("Figure 3: Thermal trajectory saved.")


# =========================================================================
# FIGURE 4: Cross-platform E/TO comparison
# =========================================================================
def fig4_cross_platform():
    archs = ["ResNet-18", "ResNet-50", "VGG-16", "RNN", "LSTM", "GRU"]
    keys = ["resnet18", "resnet50", "vgg16", "rnn", "lstm", "gru"]

    # RTX 4090 E/TO (from FLAIRS data)
    eto_4090 = [4.389e-15, 7.748e-15, 4.555e-15, 4.887e-15, 5.531e-15, 6.746e-15]

    # A100 E/TO (from Phase 1)
    eto_a100 = []
    for k in keys:
        s = load_json(DATA / "phase1_baremetal" / "phase1_baremetal" / f"{k}_run1_summary.json")
        eto_a100.append(s["energy_per_to"])

    fig, ax = plt.subplots(figsize=(COL_W, COL_H))

    x = np.arange(len(archs))
    width = 0.35
    ax.bar(x - width/2, [e * 1e15 for e in eto_4090], width,
           label='RTX 4090 (150W)', color='#67a9cf', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, [e * 1e15 for e in eto_a100], width,
           label='A100 (400W)', color='#2166ac', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('E/TO (x10$^{-15}$ J)')
    ax.set_xlabel('Architecture')
    ax.set_title('Energy per Transistor Operation: Cross-Platform')
    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=35, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)

    # Add ratio labels
    for i in range(len(archs)):
        ratio = eto_a100[i] / eto_4090[i]
        y_pos = max(eto_4090[i], eto_a100[i]) * 1e15
        ax.text(x[i], y_pos + 0.3, f'{ratio:.1f}x', ha='center', fontsize=6, color='gray')

    fig.tight_layout()
    fig.savefig(FIGS / "fig4_cross_platform.pdf")
    fig.savefig(FIGS / "fig4_cross_platform.png")
    plt.close(fig)
    print("Figure 4: Cross-platform comparison saved.")


# =========================================================================
# FIGURE 5: TED cold vs hot comparison
# =========================================================================
def fig5_ted_cold_hot():
    archs_available = []
    cold_etos = []
    hot_etos = []
    ted_pcts = []

    for arch, label in [("resnet18", "ResNet-18"), ("resnet50", "ResNet-50")]:
        win_path = DATA / "phase3_thermal" / f"{arch}_thermal_windows.csv"
        sum_path = DATA / "phase3_thermal" / f"{arch}_thermal_summary.json"
        if not win_path.exists():
            continue

        s = load_json(sum_path)
        bps = s["batches_completed"] / s["duration_s"]
        to_pb = s["to_per_batch"]
        idle = 48.0

        rows = []
        with open(win_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                power = float(row["mean_power_w"])
                win_s = float(row["window_end_s"]) - float(row["window_start_s"])
                de = (power - idle) * win_s
                to_w = bps * win_s * to_pb
                if to_w > 0:
                    rows.append(de / to_w)

        if len(rows) < 10:
            continue

        cold = np.mean(rows[:3])
        hot = np.mean(rows[-10:])
        ted = (hot - cold) / cold * 100

        archs_available.append(label)
        cold_etos.append(cold * 1e15)
        hot_etos.append(hot * 1e15)
        ted_pcts.append(ted)

    if not archs_available:
        print("Figure 5: No TED data available yet.")
        return

    fig, ax = plt.subplots(figsize=(COL_W, COL_H))

    x = np.arange(len(archs_available))
    width = 0.35
    bars_cold = ax.bar(x - width/2, cold_etos, width,
                       label='Cold start (first 30s)', color='#67a9cf', edgecolor='black', linewidth=0.5)
    bars_hot = ax.bar(x + width/2, hot_etos, width,
                      label='Thermal equilibrium', color='#b2182b', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('E/TO (x10$^{-15}$ J)')
    ax.set_xlabel('Architecture')
    ax.set_title('Thermal Efficiency Decay: Cold Start vs. Equilibrium')
    ax.set_xticks(x)
    ax.set_xticklabels(archs_available)
    ax.legend(loc='upper left', framealpha=0.9)

    for i, ted in enumerate(ted_pcts):
        y_pos = max(cold_etos[i], hot_etos[i])
        ax.annotate(f'+{ted:.1f}%',
                    xy=(x[i], y_pos + 0.1),
                    ha='center', fontsize=8, fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIGS / "fig5_ted_cold_hot.pdf")
    fig.savefig(FIGS / "fig5_ted_cold_hot.png")
    plt.close(fig)
    print("Figure 5: TED cold vs hot saved.")


if __name__ == "__main__":
    print("Generating figures...")
    print(f"Data dir: {DATA}")
    print(f"Output dir: {FIGS}")
    print()
    fig1_power_range()
    fig2_container_overhead()
    fig3_thermal_trajectory()
    fig4_cross_platform()
    fig5_ted_cold_hot()
    print("\nDone. Figures saved to:", FIGS)
