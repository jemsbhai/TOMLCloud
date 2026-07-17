"""
TOMLCloud - Extended TOML Metrics Suite
========================================
Original 6 metrics from FLAIRS-39 plus 3 new container-aware metrics:
  7. TED  - Thermal Efficiency Decay
  8. SPP  - Sustained Power Proportionality
  9. CTI  - Co-location Thermal Interference

Author: Muntaser Syed
Date: May 2026
IEEE IC2E 2026
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional


# =========================================================================
# ORIGINAL 6 TOML METRICS (from FLAIRS-39)
# =========================================================================

def compute_saft(to_switching: float, num_outputs: float) -> float:
    """
    SAF-T: Switching Activity Factor per Token/Output.
    Measures average transistor state transitions per output unit.
    Lower is more energy-efficient per output.
    """
    if num_outputs == 0:
        return 0.0
    return to_switching / num_outputs


def compute_lsrt(time_high: float, time_low: float, time_total: float) -> float:
    """
    LSRT: Logic State Residence Time.
    Ratio of time transistors spend in stable states vs switching.
    Higher means more static power, lower means more dynamic (switching) power.
    """
    if time_total == 0:
        return 0.0
    return (time_high + time_low) / time_total


def compute_ecu(energy_joules: float, capability_score: float) -> float:
    """
    ECU: Energy per Capability Unit.
    Relates energy consumption to model capability (e.g., accuracy).
    Lower means better energy-capability tradeoff.
    """
    if capability_score == 0:
        return float("inf")
    return energy_joules / capability_score


def compute_mcer(energy_memory: float, energy_compute: float) -> float:
    """
    MCER: Memory-Compute Energy Ratio.
    >1 means memory-bound, <1 means compute-bound.
    """
    if energy_compute == 0:
        return float("inf")
    return energy_memory / energy_compute


def compute_ddev(energy_samples: List[float]) -> float:
    """
    DDEV: Data-Dependent Energy Variation.
    Coefficient of variation of energy across different inputs.
    """
    if len(energy_samples) < 2:
        return 0.0
    mean_e = np.mean(energy_samples)
    if mean_e == 0:
        return 0.0
    return float(np.std(energy_samples) / mean_e)


def compute_cpto(capability_score: float, total_saft: float) -> float:
    """
    CpTO: Capability per Transistor Operation.
    Higher is better: more capability per unit of switching activity.
    """
    if total_saft == 0:
        return 0.0
    return capability_score / total_saft


# =========================================================================
# NEW CONTAINER-AWARE METRICS (IC2E 2026)
# =========================================================================

def compute_ted(
    energy_per_to_series: List[float],
    temperature_series: List[float],
) -> Dict:
    """
    TED: Thermal Efficiency Decay.

    Quantifies how energy per transistor operation degrades as die temperature
    rises under sustained load.

    Computed as the slope of linear regression: E_per_TO ~ beta * temperature.

    Args:
        energy_per_to_series: E_per_TO values per time window (e.g., 10s windows).
        temperature_series: Mean die temperature per window (same length).

    Returns:
        Dict with:
            ted_slope: Joules/TO per degree C (the raw TED value).
            ted_pct: Percentage increase in E_per_TO from T_min to T_max.
            r2: R-squared of the linear fit.
            temp_min_c: Minimum temperature observed.
            temp_max_c: Maximum temperature observed.
    """
    e = np.array(energy_per_to_series, dtype=np.float64)
    t = np.array(temperature_series, dtype=np.float64)

    if len(e) < 3 or len(e) != len(t):
        return {
            "ted_slope": 0.0,
            "ted_pct": 0.0,
            "r2": 0.0,
            "temp_min_c": 0.0,
            "temp_max_c": 0.0,
        }

    slope, intercept, r_value, p_value, std_err = stats.linregress(t, e)

    # Percentage form: how much does E_per_TO increase from T_min to T_max?
    t_min, t_max = float(np.min(t)), float(np.max(t))
    e_at_min = intercept + slope * t_min
    e_at_max = intercept + slope * t_max
    ted_pct = ((e_at_max - e_at_min) / e_at_min * 100) if e_at_min != 0 else 0.0

    return {
        "ted_slope": float(slope),
        "ted_pct": float(ted_pct),
        "r2": float(r_value ** 2),
        "p_value": float(p_value),
        "temp_min_c": t_min,
        "temp_max_c": t_max,
    }


def compute_spp(
    delta_power_series: List[float],
    to_throughput_series: List[float],
) -> Dict:
    """
    SPP: Sustained Power Proportionality.

    Measures how linearly power draw scales with transistor-operation throughput
    at different load levels.

    SPP = R-squared of linear fit: P_delta ~ beta * TO_throughput.

    SPP close to 1.0 means power tracks useful work.
    SPP << 1.0 means disproportionate power at some load levels.

    Args:
        delta_power_series: Delta power (load - idle) at each load level.
        to_throughput_series: TO/s at each load level.

    Returns:
        Dict with:
            spp_r2: The SPP value (R-squared).
            slope: Watts per TO/s.
            intercept: Residual power at zero throughput.
    """
    p = np.array(delta_power_series, dtype=np.float64)
    to = np.array(to_throughput_series, dtype=np.float64)

    if len(p) < 3 or len(p) != len(to):
        return {"spp_r2": 0.0, "slope": 0.0, "intercept": 0.0}

    slope, intercept, r_value, p_value, std_err = stats.linregress(to, p)

    return {
        "spp_r2": float(r_value ** 2),
        "slope": float(slope),
        "intercept": float(intercept),
        "p_value": float(p_value),
    }


def compute_cti(
    energy_isolated: float,
    energy_colocated: float,
) -> float:
    """
    CTI: Co-location Thermal Interference.

    The fractional energy overhead imposed on a container when another
    container shares the same GPU.

    CTI = E_colocated / E_isolated - 1

    CTI = 0 means no interference.
    CTI = 0.3 means 30% more energy when co-located.
    CTI is asymmetric: CTI(A, B) != CTI(B, A) in general.

    Args:
        energy_isolated: Energy consumed when running alone (from Phase 2).
        energy_colocated: Energy consumed when co-located with another workload.

    Returns:
        CTI value (float). Positive means co-location increases energy.
    """
    if energy_isolated <= 0:
        return 0.0
    return (energy_colocated / energy_isolated) - 1.0


def compute_cti_matrix(
    isolated_energies: Dict[str, float],
    colocated_energies: Dict[Tuple[str, str], float],
) -> Dict[Tuple[str, str], float]:
    """
    Compute the full asymmetric CTI matrix for all measured pairs.

    Args:
        isolated_energies: {arch_name: energy_joules} from Phase 2.
        colocated_energies: {(arch_a, arch_b): energy_of_a_when_colocated_with_b}.

    Returns:
        {(arch_a, arch_b): cti_value} for all measured pairs.
    """
    matrix = {}
    for (arch_a, arch_b), e_coloc in colocated_energies.items():
        e_iso = isolated_energies.get(arch_a, 0.0)
        matrix[(arch_a, arch_b)] = compute_cti(e_iso, e_coloc)
    return matrix


# =========================================================================
# ATTRIBUTION ERROR ANALYSIS (Phase 6)
# =========================================================================

def compute_attribution_error(
    e_true: float,
    e_attributed: float,
) -> float:
    """
    Compute percentage attribution error.

    Args:
        e_true: Ground truth energy (isolated run, Phase 2).
        e_attributed: Energy attributed by a method (proportional or TOML).

    Returns:
        Absolute percentage error.
    """
    if e_true <= 0:
        return 0.0
    return abs(e_attributed - e_true) / e_true * 100.0


def proportional_attribution(
    total_energy: float,
    util_a: float,
    util_b: float,
) -> Tuple[float, float]:
    """
    Proportional energy attribution (what cAdvisor does).

    Distributes total GPU energy proportionally to utilization fractions.

    Args:
        total_energy: Total GPU energy during co-located run.
        util_a: Utilization fraction for container A (0-1).
        util_b: Utilization fraction for container B (0-1).

    Returns:
        (energy_a, energy_b) attributed to each container.
    """
    total_util = util_a + util_b
    if total_util == 0:
        return total_energy / 2, total_energy / 2
    return (
        total_energy * util_a / total_util,
        total_energy * util_b / total_util,
    )


def toml_attribution(
    total_energy: float,
    to_count_a: float,
    to_count_b: float,
    e_per_to_a: float,
    e_per_to_b: float,
) -> Tuple[float, float]:
    """
    TOML-aware energy attribution.

    Uses calibrated energy-per-TO to attribute energy, then distributes
    residual overhead proportionally to TO count.

    Args:
        total_energy: Total GPU energy during co-located run.
        to_count_a: Transistor operations completed by container A.
        to_count_b: Transistor operations completed by container B.
        e_per_to_a: Calibrated energy per TO for A (from Phase 2, TED-adjusted).
        e_per_to_b: Calibrated energy per TO for B.

    Returns:
        (energy_a, energy_b) attributed to each container.
    """
    e_raw_a = to_count_a * e_per_to_a
    e_raw_b = to_count_b * e_per_to_b
    e_overhead = total_energy - e_raw_a - e_raw_b

    # Distribute overhead proportionally to TO count
    total_to = to_count_a + to_count_b
    if total_to == 0:
        return total_energy / 2, total_energy / 2

    e_final_a = e_raw_a + e_overhead * (to_count_a / total_to)
    e_final_b = e_raw_b + e_overhead * (to_count_b / total_to)

    return float(e_final_a), float(e_final_b)


# =========================================================================
# TRANSISTOR OPERATION COUNTS (from FLAIRS-39, reused directly)
# =========================================================================

TO_PER_OP = {
    "mac": 5000,
    "sigmoid": 18000,
    "tanh": 15000,
    "relu": 100,
    "comparison": 50,
    "mem_word": 192,
}

ARCH_SPECS = {
    "resnet18": {
        "params": 11.7e6,
        "flops_per_image": 1.8e9,
        "activation_mem": 0.5e6,
        "type": "cnn",
    },
    "resnet50": {
        "params": 25.6e6,
        "flops_per_image": 4.1e9,
        "activation_mem": 2.5e6,
        "type": "cnn",
    },
    "vgg16": {
        "params": 138e6,
        "flops_per_image": 15.5e9,
        "activation_mem": 12e6,
        "type": "cnn",
    },
    "rnn": {
        "params": 0.4e6,
        "ops_per_step": 0.33e6,
        "seq_length": 100,
        "hidden_size": 256,
        "num_layers": 2,
        "type": "rnn",
    },
    "lstm": {
        "params": 1.6e6,
        "ops_per_step": 1.3e6,
        "seq_length": 100,
        "hidden_size": 256,
        "num_layers": 2,
        "type": "lstm",
    },
    "gru": {
        "params": 1.2e6,
        "ops_per_step": 1.0e6,
        "seq_length": 100,
        "hidden_size": 256,
        "num_layers": 2,
        "type": "gru",
    },
    "xgboost": {
        "n_trees": 100,
        "avg_nodes": 63,
        "n_test_samples": 20000,
        "n_infer_calls": 20,
        "type": "gbdt",
    },
}


def compute_to_per_batch(arch: str, batch_size: int = 32) -> Dict[str, float]:
    """
    Compute transistor operations for one batch of inference.

    This is deterministic per architecture and batch size. Used to convert
    batch counts into TO counts during experiments.

    Args:
        arch: Architecture key from ARCH_SPECS.
        batch_size: Batch size.

    Returns:
        Dict with to_compute, to_memory, to_total.
    """
    spec = ARCH_SPECS[arch]

    if spec["type"] == "cnn":
        macs = (spec["flops_per_image"] / 2) * batch_size
        relus = spec["activation_mem"] * batch_size
        mem_weight = spec["params"]  # loaded once per batch
        mem_act = spec["activation_mem"] * batch_size * 2  # read + write

        to_compute = macs * TO_PER_OP["mac"] + relus * TO_PER_OP["relu"]
        to_memory = (mem_weight + mem_act) * TO_PER_OP["mem_word"]

    elif spec["type"] in ("rnn", "lstm", "gru"):
        total_steps = batch_size * spec["seq_length"]
        h = spec["hidden_size"]

        macs = spec["ops_per_step"] * total_steps

        if spec["type"] == "lstm":
            sigmoids = 3 * h * total_steps
            tanhs = 2 * h * total_steps
            act_tos = sigmoids * TO_PER_OP["sigmoid"] + tanhs * TO_PER_OP["tanh"]
        elif spec["type"] == "gru":
            sigmoids = 2 * h * total_steps
            tanhs = 1 * h * total_steps
            act_tos = sigmoids * TO_PER_OP["sigmoid"] + tanhs * TO_PER_OP["tanh"]
        else:
            tanhs = h * total_steps
            act_tos = tanhs * TO_PER_OP["tanh"]

        mem_weight = spec["params"] * total_steps
        mem_hidden = h * total_steps * 2 * spec["num_layers"]

        to_compute = macs * TO_PER_OP["mac"] + act_tos
        to_memory = (mem_weight + mem_hidden) * TO_PER_OP["mem_word"]

    elif spec["type"] == "gbdt":
        n_samples = spec["n_test_samples"]  # per call
        avg_depth = np.log2(spec["avg_nodes"] + 1)
        comparisons = n_samples * spec["n_trees"] * avg_depth

        to_compute = comparisons * TO_PER_OP["comparison"]
        to_memory = comparisons * TO_PER_OP["mem_word"]

    else:
        to_compute = 0.0
        to_memory = 0.0

    return {
        "to_compute": float(to_compute),
        "to_memory": float(to_memory),
        "to_total": float(to_compute + to_memory),
    }
