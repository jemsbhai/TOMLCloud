"""
TOMLCloud - Extended GPU Power & Thermal Monitor
=================================================
Extends the FLAIRS GPUPowerMonitor with temperature, clock frequency,
utilization, and throttle state tracking for container energy attribution.

Author: Muntaser Syed
Date: May 2026
IEEE IC2E 2026
"""

import time
import threading
import csv
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pynvml


@dataclass
class ThermalReading:
    """Single telemetry sample with power, thermal, and clock data."""
    timestamp_ms: float
    elapsed_ms: float
    power_w: float
    temp_c: int
    sm_clock_mhz: int
    mem_clock_mhz: int
    gpu_util_pct: int
    mem_util_pct: int


class GPUThermalMonitor:
    """
    GPU power and thermal monitoring for TOMLCloud experiments.

    Extends the FLAIRS GPUPowerMonitor with:
    - Die temperature tracking
    - SM and memory clock frequency tracking
    - GPU and memory utilization tracking
    - Throttle detection (clock drop from max boost)
    - Windowed statistics for long-duration runs (Phase 3)
    - CSV export per run
    """

    def __init__(self, gpu_index: int = 0, sample_interval_ms: int = 50):
        """
        Args:
            gpu_index: GPU device index (0 for single-GPU Lambda instance).
            sample_interval_ms: Sampling interval in milliseconds.
                50ms = 20 Hz, fast enough for power dynamics.
        """
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.sample_interval_s = sample_interval_ms / 1000.0
        self.readings: List[ThermalReading] = []
        self.idle_power: Optional[float] = None
        self.idle_temp: Optional[int] = None
        self.idle_clock: Optional[int] = None

        # GPU metadata
        self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(self.gpu_name, bytes):
            self.gpu_name = self.gpu_name.decode("utf-8")
        self.power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0

        # Background sampling state
        self._sampling = False
        self._sample_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

    # -----------------------------------------------------------------
    # Low-level reads
    # -----------------------------------------------------------------

    def _read_power_w(self) -> float:
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def _read_temp_c(self) -> int:
        return pynvml.nvmlDeviceGetTemperature(
            self.handle, pynvml.NVML_TEMPERATURE_GPU
        )

    def _read_sm_clock(self) -> int:
        return pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)

    def _read_mem_clock(self) -> int:
        return pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)

    def _read_utilization(self):
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        return util.gpu, util.memory

    def _take_reading(self) -> ThermalReading:
        now = time.time()
        gpu_util, mem_util = self._read_utilization()
        return ThermalReading(
            timestamp_ms=now * 1000.0,
            elapsed_ms=(now - self._start_time) * 1000.0 if self._start_time else 0.0,
            power_w=self._read_power_w(),
            temp_c=self._read_temp_c(),
            sm_clock_mhz=self._read_sm_clock(),
            mem_clock_mhz=self._read_mem_clock(),
            gpu_util_pct=gpu_util,
            mem_util_pct=mem_util,
        )

    # -----------------------------------------------------------------
    # Idle baseline
    # -----------------------------------------------------------------

    def measure_idle(self, duration_s: float = 10.0, warmup_s: float = 2.0) -> Dict:
        """
        Measure GPU idle baseline: power, temperature, clock frequency.

        Args:
            duration_s: Sampling duration after warmup.
            warmup_s: Wait time to let GPU settle.

        Returns:
            Dict with idle_power_w, idle_temp_c, idle_clock_mhz, and statistics.
        """
        print(f"Measuring idle baseline ({warmup_s}s warmup + {duration_s}s sampling)...")
        time.sleep(warmup_s)

        powers, temps, clocks = [], [], []
        t0 = time.time()
        while time.time() - t0 < duration_s:
            powers.append(self._read_power_w())
            temps.append(self._read_temp_c())
            clocks.append(self._read_sm_clock())
            time.sleep(self.sample_interval_s)

        self.idle_power = float(np.mean(powers))
        self.idle_temp = int(np.mean(temps))
        self.idle_clock = int(np.mean(clocks))

        result = {
            "idle_power_w": self.idle_power,
            "idle_power_std_w": float(np.std(powers)),
            "idle_temp_c": self.idle_temp,
            "idle_clock_mhz": self.idle_clock,
            "n_samples": len(powers),
        }
        print(
            f"  Idle: {self.idle_power:.1f}W, {self.idle_temp}C, {self.idle_clock} MHz "
            f"({len(powers)} samples)"
        )
        return result

    # -----------------------------------------------------------------
    # Sampling (foreground, called from benchmark loop)
    # -----------------------------------------------------------------

    def start(self):
        """Begin a measurement run. Call sample() inside your benchmark loop."""
        if self.idle_power is None:
            raise RuntimeError("Call measure_idle() before start().")
        self.readings = []
        self._start_time = time.time()

    def sample(self):
        """Record one telemetry reading. Call after each batch or at fixed intervals."""
        self.readings.append(self._take_reading())

    def stop(self) -> Dict:
        """
        End the run and return summary statistics.

        Returns:
            Dict with duration, power, energy, temperature, and clock stats.
        """
        if not self.readings:
            raise RuntimeError("No readings recorded.")

        duration_s = (self.readings[-1].timestamp_ms - self.readings[0].timestamp_ms) / 1000.0
        powers = np.array([r.power_w for r in self.readings])
        temps = np.array([r.temp_c for r in self.readings])
        clocks = np.array([r.sm_clock_mhz for r in self.readings])

        mean_power = float(np.mean(powers))
        delta_power = mean_power - self.idle_power

        return {
            "duration_s": duration_s,
            "n_samples": len(self.readings),
            "idle_power_w": self.idle_power,
            "mean_power_w": mean_power,
            "delta_power_w": delta_power,
            "power_std_w": float(np.std(powers)),
            "total_energy_j": mean_power * duration_s,
            "delta_energy_j": delta_power * duration_s,
            "mean_temp_c": float(np.mean(temps)),
            "max_temp_c": int(np.max(temps)),
            "min_temp_c": int(np.min(temps)),
            "temp_rise_c": int(np.max(temps)) - int(np.min(temps)),
            "mean_clock_mhz": int(np.mean(clocks)),
            "max_clock_mhz": int(np.max(clocks)),
            "min_clock_mhz": int(np.min(clocks)),
            "clock_drop_pct": (1.0 - int(np.min(clocks)) / max(int(np.max(clocks)), 1)) * 100,
        }

    # -----------------------------------------------------------------
    # Background sampling (for non-instrumented workloads)
    # -----------------------------------------------------------------

    def start_background(self):
        """Start sampling in a background thread at the configured interval."""
        if self.idle_power is None:
            raise RuntimeError("Call measure_idle() before start_background().")
        self.readings = []
        self._start_time = time.time()
        self._sampling = True
        self._sample_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._sample_thread.start()

    def _bg_loop(self):
        while self._sampling:
            self.readings.append(self._take_reading())
            time.sleep(self.sample_interval_s)

    def stop_background(self) -> Dict:
        """Stop background sampling and return summary."""
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=2.0)
        return self.stop()

    # -----------------------------------------------------------------
    # Windowed statistics (Phase 3: thermal profiling)
    # -----------------------------------------------------------------

    def compute_windows(self, window_s: float = 10.0) -> List[Dict]:
        """
        Compute per-window statistics from recorded readings.

        Used in Phase 3 to track E_per_TO vs temperature over a 30-minute run.

        Args:
            window_s: Window duration in seconds.

        Returns:
            List of dicts, one per window.
        """
        if not self.readings:
            return []

        windows = []
        t0 = self.readings[0].elapsed_ms
        win_start = t0

        while True:
            win_end = win_start + window_s * 1000.0
            win_readings = [
                r for r in self.readings
                if win_start <= r.elapsed_ms < win_end
            ]
            if not win_readings:
                break

            powers = np.array([r.power_w for r in win_readings])
            temps = np.array([r.temp_c for r in win_readings])
            clocks = np.array([r.sm_clock_mhz for r in win_readings])
            gpu_utils = np.array([r.gpu_util_pct for r in win_readings])
            mem_utils = np.array([r.mem_util_pct for r in win_readings])

            windows.append({
                "window_start_s": (win_start - t0) / 1000.0,
                "window_end_s": (win_end - t0) / 1000.0,
                "n_samples": len(win_readings),
                "mean_power_w": float(np.mean(powers)),
                "mean_temp_c": float(np.mean(temps)),
                "max_temp_c": int(np.max(temps)),
                "mean_clock_mhz": int(np.mean(clocks)),
                "min_clock_mhz": int(np.min(clocks)),
                "gpu_util_pct": float(np.mean(gpu_utils)),
                "mem_util_pct": float(np.mean(mem_utils)),
            })

            win_start = win_end
            if win_start > self.readings[-1].elapsed_ms:
                break

        return windows

    # -----------------------------------------------------------------
    # CSV export
    # -----------------------------------------------------------------

    def save_readings(self, path: str):
        """Write all raw readings to a CSV file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_ms", "elapsed_ms", "power_w", "temp_c",
                "sm_clock_mhz", "mem_clock_mhz", "gpu_util_pct", "mem_util_pct",
            ])
            for r in self.readings:
                writer.writerow([
                    f"{r.timestamp_ms:.1f}",
                    f"{r.elapsed_ms:.1f}",
                    f"{r.power_w:.2f}",
                    r.temp_c,
                    r.sm_clock_mhz,
                    r.mem_clock_mhz,
                    r.gpu_util_pct,
                    r.mem_util_pct,
                ])
        print(f"Saved {len(self.readings)} readings to {p}")

    def save_windows(self, path: str, window_s: float = 10.0):
        """Compute windowed stats and write to CSV."""
        windows = self.compute_windows(window_s)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if not windows:
            print("No windows to save.")
            return

        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=windows[0].keys())
            writer.writeheader()
            writer.writerows(windows)
        print(f"Saved {len(windows)} windows to {p}")

    # -----------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------

    def wait_cooldown(self, target_c: int = 35, timeout_s: float = 300.0):
        """
        Block until GPU temperature drops below target.

        Args:
            target_c: Target temperature in Celsius.
            timeout_s: Maximum wait time.
        """
        t0 = time.time()
        while True:
            temp = self._read_temp_c()
            if temp <= target_c:
                print(f"GPU cooled to {temp}C (target: {target_c}C)")
                return
            elapsed = time.time() - t0
            if elapsed > timeout_s:
                print(f"WARNING: Timeout after {timeout_s}s. Current temp: {temp}C")
                return
            print(f"  Cooling: {temp}C (target: {target_c}C, {elapsed:.0f}s elapsed)", end="\r")
            time.sleep(5.0)

    def get_metadata(self) -> Dict:
        """Return GPU metadata for the paper's experimental setup section."""
        return {
            "gpu_name": self.gpu_name,
            "power_limit_w": self.power_limit_w,
            "driver_version": pynvml.nvmlSystemGetDriverVersion(),
        }

    def shutdown(self):
        """Release NVML resources."""
        pynvml.nvmlShutdown()
