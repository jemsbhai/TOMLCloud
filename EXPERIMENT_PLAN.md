# TOMLCloud Experiment Plan

**Paper:** Beyond Proportional Allocation: Transistor-Level Energy Attribution for Containerized ML Workloads
**Venue:** IEEE IC2E 2026 (Full paper, 9 pages + refs)
**Cloud:** Lambda Labs, 1x A100 80GB SXM on-demand (~$2.49/hr, full VM with sudo)
**Budget:** $50 max. Estimated usage: 13-15 GPU hours = ~$32-37. Leaves headroom for reruns.
**Local backup:** RTX 4090 Laptop GPU (64GB RAM, Windows) for cross-hardware validation if time permits.
**Baseline reference:** TOML FLAIRS-39 codebase at `E:\data\code\claudecode\toml`

---

## Overview

Seven phases, ordered by dependency. Each phase produces a CSV dataset and at minimum one figure or table for the paper. Phases 1 and 2 can run in a single day. Phase 3 is the longest (thermal profiling requires sustained runs). Phases 4 and 5 are the core contribution. Phase 6 is analysis. Phase 7 is architecture design (no experiments).

**Architectures (carried over from TOML):**

| ID | Architecture | Type | Notes |
|----|-------------|------|-------|
| A1 | ResNet18 | CNN | Known GPU underutilization outlier |
| A2 | ResNet50 | CNN | Compute-heavy |
| A3 | VGG16 | CNN | Memory-heavy, large model |
| A4 | Vanilla RNN 2L-256H | RNN | Lightweight |
| A5 | LSTM 2L-256H | RNN | Gated, moderate |
| A6 | GRU 2L-256H | RNN | Gated, slightly lighter than LSTM |
| A7 | XGBoost GPU | GBDT | Tree-based, different execution pattern |

---

## Phase 0: Lambda Labs Instance Setup

**Goal:** Provision the VM, install Docker and monitoring tools, upload benchmark code, verify power telemetry works.

### 0.1 Instance Provisioning

Launch a single-GPU A100 instance from the Lambda Labs dashboard (cloud.lambdalabs.com). Select:
- GPU: 1x A100 SXM 80GB
- OS: Ubuntu 22.04 (default)
- Region: any US region (closest to you for lower SSH latency)

SSH in:
```bash
ssh -i ~/.ssh/lambda_key ubuntu@<instance-ip>
```

### 0.2 First Thing: Verify Power Telemetry

Before installing anything, confirm nvidia-smi exposes power readings. This is the single hard requirement. If this fails, the instance is useless for this experiment.

```bash
# Check GPU is visible
nvidia-smi

# Check power reading is a real number (not N/A)
nvidia-smi --query-gpu=power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv
# Expected: something like "65.42 W, 400.00 W, 32, 1410 MHz, 1593 MHz"

# Verify continuous monitoring works
nvidia-smi dmon -s pucvmet -d 1 -c 5
# Should print 5 rows of power/utilization/clock/temp data
```

If `power.draw` returns `[N/A]`, stop here and try a different instance or contact Lambda support. This has not been reported as an issue on Lambda A100 instances, but verify before committing hours of work.

### 0.3 System Setup

```bash
# Update and install basics
sudo apt-get update && sudo apt-get install -y \
    docker.io \
    docker-compose \
    htop \
    tmux \
    git \
    python3-pip \
    python3-venv

# Enable Docker without sudo
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit (for GPU-enabled Docker containers)
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 0.4 Clone Project and Upload Benchmarks

From your Windows machine, push your code to the instance. Easiest method:

```powershell
# From your local machine
scp -i ~/.ssh/lambda_key -r E:\data\code\claudecode\TOMLCloud ubuntu@<instance-ip>:~/TOMLCloud
```

Or clone from a git repo if you set one up.

### 0.5 Build Docker Image

On the Lambda instance:

```bash
cd ~/TOMLCloud
docker build -t tomlcloud:latest -f docker/Dockerfile .
```

Dockerfile (to be created in Phase 0 prep):

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY benchmarks/ /app/benchmarks/
COPY shared/ /app/shared/

ENTRYPOINT ["python3"]
```

requirements.txt:
```
torch==2.5.1
torchvision==0.20.1
xgboost==2.1.3
scikit-learn==1.6.1
pynvml==12.560.30
pandas==2.2.3
numpy==1.26.4
```

Pin these versions to match what you currently use in tomlenv. Check with:
```powershell
# On your local machine, in tomlenv
pip freeze | findstr "torch torchvision xgboost scikit-learn pynvml pandas numpy"
```

### 0.6 Install cAdvisor (Conventional Monitoring Baseline)

```bash
docker run -d --name cadvisor \
  --volume=/:/rootfs:ro \
  --volume=/var/run:/var/run:ro \
  --volume=/sys:/sys:ro \
  --volume=/var/lib/docker/:/var/lib/docker:ro \
  --publish=8080:8080 \
  gcr.io/cadvisor/cadvisor:latest
```

Verify: `curl http://localhost:8080/api/v2.1/machine` should return machine info.

### 0.7 Enable NVIDIA MPS (for co-location experiments)

MPS is fully supported on A100. Enable it before Phase 5:

```bash
# Stop any existing MPS daemon
echo quit | sudo nvidia-cuda-mps-control 2>/dev/null

# Set exclusive process mode, then start MPS
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
sudo nvidia-cuda-mps-control -d

# Verify
echo "get_server_list" | sudo nvidia-cuda-mps-control
```

To disable after experiments:
```bash
echo quit | sudo nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c DEFAULT
```

### 0.8 Record Instance Metadata

Save hardware details for the paper's experimental setup section:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem --format=csv > ~/TOMLCloud/data/instance_metadata.csv
cat /proc/cpuinfo | head -30 >> ~/TOMLCloud/data/instance_metadata.csv
free -h >> ~/TOMLCloud/data/instance_metadata.csv
uname -a >> ~/TOMLCloud/data/instance_metadata.csv
```

**Estimated time:** 30-45 minutes for the full setup.

---

## Phase 1: Bare-Metal Baseline (No Container)

**Goal:** Run all 7 benchmarks directly on the VM (no Docker) with extended telemetry. This is the ground truth. Also establishes the A100's thermal and power characteristics.

### Protocol

For each architecture Ai:

1. Cool-down: wait until GPU temp drops below 35C. On the A100 in a datacenter this should be fast (good cooling). Check with:
   ```bash
   nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
   ```
   If above 35C, sleep 120 seconds and recheck.

2. Measure idle baseline: 10 seconds of sampling at 50ms intervals. Record power, temp, clocks.

3. Run sustained inference for 60 seconds. The benchmark script loops inference continuously for the full duration.

4. Sample at 50ms intervals via pynvml from within the benchmark script:
   - power_w (nvmlDeviceGetPowerUsage)
   - temp_c (nvmlDeviceGetTemperature)
   - sm_clock_mhz (nvmlDeviceGetClockInfo CLOCK_SM)
   - mem_clock_mhz (nvmlDeviceGetClockInfo CLOCK_MEM)
   - gpu_util_pct (nvmlDeviceGetUtilizationRates .gpu)
   - mem_util_pct (nvmlDeviceGetUtilizationRates .memory)

5. Repeat 3 times per architecture.

### Runner command

```bash
cd ~/TOMLCloud
python3 scripts/run_phase1_baremetal.py --arch all --duration 60 --runs 3 --output data/phase1_baremetal/
```

### Output

Per-run timeseries: `data/phase1_baremetal/baremetal_{arch}_run{N}.csv`
```
timestamp_ms, elapsed_ms, power_w, temp_c, sm_clock_mhz, mem_clock_mhz, gpu_util_pct, mem_util_pct
```

Summary: `data/phase1_baremetal/summary.csv`
```
arch, run, idle_power_w, mean_power_w, delta_power_w, total_energy_j, mean_temp_c, max_temp_c, mean_clock_mhz, min_clock_mhz, clock_drop_pct
```

### Paper output

Table 1 (partial): bare-metal baseline energy and thermal profile per architecture on A100.

**Expected A100 behavior vs your 4090 Laptop:** Idle power will be higher (~70-80W vs 25W). Load power will also be higher (200-300W vs 100-130W). Thermal throttling will be less aggressive because datacenter cooling keeps the die cooler. This is itself an interesting comparison if you also run on your 4090.

**Estimated time:** ~45 minutes.

---

## Phase 2: Isolated Container Measurements

**Goal:** Run each architecture inside its own Docker container with exclusive GPU access. Compare against bare-metal baseline to quantify container overhead. Simultaneously collect cAdvisor data.

### Protocol

For each architecture Ai:

1. Ensure cAdvisor is running.
2. Start nvidia-smi dmon on host in background:
   ```bash
   nvidia-smi dmon -s pucvmet -d 1 -f data/phase2_isolated/host_{arch}_run{N}.csv &
   DMON_PID=$!
   ```
3. Cool-down to < 35C.
4. Launch container:
   ```bash
   docker run --rm --gpus all \
     --name toml_${arch} \
     -v ~/TOMLCloud/data/phase2_isolated:/output \
     tomlcloud:latest \
     benchmarks/bench_${arch}.py --duration 60 --output /output/${arch}_run${N}_internal.csv
   ```
5. While container runs, scrape cAdvisor every second:
   ```bash
   for i in $(seq 1 65); do
     curl -s http://localhost:8080/api/v2.1/stats/toml_${arch}?type=docker >> data/phase2_isolated/${arch}_run${N}_cadvisor.jsonl
     sleep 1
   done
   ```
6. After container exits, kill dmon:
   ```bash
   kill $DMON_PID
   ```
7. Repeat 3 times.

### Key comparison

```
overhead_pct = (container_energy - baremetal_energy) / baremetal_energy * 100
```

Expected: < 3% for GPU workloads on Lambda. Docker GPU passthrough is essentially zero overhead for compute; the container only virtualizes the filesystem and network.

### Output

`data/phase2_isolated/{arch}_run{N}_internal.csv` (pynvml from inside container)
`data/phase2_isolated/{arch}_run{N}_cadvisor.jsonl` (cAdvisor from host)
`data/phase2_isolated/host_{arch}_run{N}.csv` (nvidia-smi from host)
`data/phase2_isolated/summary.csv` (computed)

### Paper output

Table 1 (completed): bare-metal vs container energy, overhead percentage.

**Estimated time:** ~45 minutes.

---

## Phase 3: Thermal Efficiency Decay (TED)

**Goal:** Quantify how energy per transistor operation degrades as the GPU heats up under sustained load.

### Definition

```
TED(Ai) = d(E_per_TO) / dT
```

Where E_per_TO is measured energy per transistor operation and T is die temperature. Computed as the slope of linear regression of E_per_TO vs temperature across a sustained run.

Normalized percentage form:
```
TED_pct(Ai) = (E_per_TO at T_max - E_per_TO at T_idle) / E_per_TO at T_idle * 100
```

### Why this matters for containers

Two containers running the same workload consume different energy depending on when they start relative to the GPU's thermal state. A container that starts on a cold GPU gets higher clocks and lower energy per operation. A container that starts after another has warmed the die pays a thermal tax. Proportional allocation cannot distinguish these cases.

### Protocol

For each architecture Ai:

1. Cold start: ensure no GPU workloads running. Wait until temp < 30C. On Lambda's A100 with datacenter cooling this should settle quickly.

2. Run sustained inference for 30 minutes continuously inside a container:
   ```bash
   docker run --rm --gpus all \
     -v ~/TOMLCloud/data/phase3_thermal:/output \
     tomlcloud:latest \
     benchmarks/bench_${arch}.py --duration 1800 --output /output/${arch}_thermal.csv --window-stats 10
   ```

3. The benchmark script computes per-window statistics every 10 seconds:
   - Count batches completed in that window
   - Compute TO count = batches * TO_per_batch (deterministic, from FLAIRS data)
   - Compute average power in the window
   - Compute energy = avg_power * window_duration
   - Compute E_per_TO = energy / TO_count
   - Record average temperature and clock frequency for the window

4. Full cool-down between architectures. Wait until temp < 30C.

5. Single run per architecture (the thermal trajectory is highly reproducible on datacenter hardware with consistent cooling).

### Expected A100 behavior

The A100 SXM has a 400W TDP and aggressive datacenter cooling. Temperature will likely rise from ~30C to ~55-65C (less dramatic than the 4090 Laptop which hits 75-85C). Clock throttling will be minimal since the A100 boost algorithm is less aggressive than consumer GPUs. TED values will probably be smaller than on a laptop.

This is actually a useful finding: "even on datacenter hardware with enterprise cooling, thermal effects introduce measurable attribution error." If the effect is small, that isolates the MCER mismatch as the dominant source of attribution error, which is a clean result.

### Analysis

1. Plot E_per_TO vs temperature for each architecture.
2. Fit linear regression. Slope = TED.
3. Compute TED_pct.
4. Compare TED across architectures. Test whether memory-bound workloads (high MCER) show different thermal sensitivity than compute-bound ones.

### Output

`data/phase3_thermal/{arch}_thermal.csv`
Columns:
```
window_idx, window_start_s, window_end_s, batches, to_count, mean_power_w, energy_j, energy_per_to, mean_temp_c, max_temp_c, mean_clock_mhz, min_clock_mhz, gpu_util_pct, mem_util_pct
```

`data/phase3_thermal/ted_summary.csv`
Columns:
```
arch, ted_slope, ted_pct, r2_fit, temp_min_c, temp_max_c, temp_range_c, clock_start_mhz, clock_end_mhz, clock_drop_pct
```

### Paper output

Figure 1: E_per_TO vs temperature for all 7 architectures (multi-line plot).
Table 2: TED values per architecture.

**Estimated time:** ~4 hours (7 x 30min + cool-down).

---

## Phase 4: Sustained Power Proportionality (SPP)

**Goal:** Measure how linearly power draw scales with transistor-operation throughput at different load levels.

### Definition

```
SPP(Ai) = r^2 of linear fit: P_delta ~ beta * TO_throughput
```

Where P_delta = measured power minus idle power, and TO_throughput = transistor operations completed per second.

SPP close to 1.0 = power tracks work. SPP well below 1.0 = disproportionate power at certain load levels.

### Why this matters

Proportional allocation implicitly assumes SPP = 1.0 for every workload. If container A has SPP = 0.95 and container B has SPP = 0.70, proportional allocation overcharges A and undercharges B.

### Protocol

For each architecture Ai, run at 5 load levels by controlling the duty cycle:

| Target load | Method |
|-------------|--------|
| ~10% | 1 batch, sleep 9x batch duration |
| ~25% | 1 batch, sleep 3x batch duration |
| ~50% | 1 batch, sleep 1x batch duration |
| ~75% | 3 batches, sleep 1x batch duration |
| ~100% | Continuous, no sleep |

At each load level:
1. Run for 60 seconds inside a container.
2. Sample power at 50ms.
3. Count completed batches. TO_count = batches * TO_per_batch.
4. TO_throughput = TO_count / 60.
5. Compute mean delta power (mean load power minus idle from Phase 1).

Fit linear regression across the 5 load points. r^2 = SPP.

### Runner

```bash
docker run --rm --gpus all \
  -v ~/TOMLCloud/data/phase4_spp:/output \
  tomlcloud:latest \
  benchmarks/bench_${arch}.py --duration 60 --load-pct ${LOAD} --output /output/${arch}_load${LOAD}.csv
```

### Output

`data/phase4_spp/{arch}_load{N}.csv` (per-run telemetry)

`data/phase4_spp/spp_summary.csv`
Columns:
```
arch, load_10_power_w, load_25_power_w, load_50_power_w, load_75_power_w, load_100_power_w, load_10_to_throughput, load_25_to_throughput, load_50_to_throughput, load_75_to_throughput, load_100_to_throughput, spp_r2, spp_slope, spp_intercept
```

### Paper output

Figure 2: Power vs TO throughput scatter for all architectures with linear fits.
Table 3: SPP values.

**Estimated time:** ~1 hour (7 archs x 5 loads x 60s = 35 min + overhead).

---

## Phase 5: Co-location Thermal Interference (CTI)

**Goal:** Measure the energy overhead one container imposes on another when sharing the A100 via MPS.

### Definition

```
CTI(Ai, Aj) = E_colocated(Ai | Aj running) / E_isolated(Ai) - 1
```

CTI = 0 means no interference. CTI = 0.3 means Ai uses 30% more energy when Aj is co-located.

CTI is asymmetric: CTI(Ai, Aj) != CTI(Aj, Ai) in general.

### MPS setup

Before starting Phase 5, enable MPS on the A100:

```bash
echo quit | sudo nvidia-cuda-mps-control 2>/dev/null
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
sudo nvidia-cuda-mps-control -d
echo "get_server_list" | sudo nvidia-cuda-mps-control
```

MPS is fully supported on A100 and provides true spatial GPU sharing. Both processes execute concurrently on different SMs rather than time-slicing.

### Pair selection

Full matrix: all 21 pairwise combinations of 7 architectures. This is feasible (21 x 60s x 3 reps = ~63 min of measurement). Run all 21.

Priority pairs for deeper analysis (highlight in the paper):

| Pair | Rationale |
|------|-----------|
| VGG16 + RNN | Highest MCER contrast (9.0 vs 1.5) |
| ResNet50 + XGBoost | CNN + GBDT, different execution models |
| LSTM + GRU | Near-identical architectures, sensitivity test |
| VGG16 + ResNet18 | Both CNN, very different GPU utilization |
| XGBoost + RNN | Both lightweight, different paradigms |
| ResNet50 + LSTM | Moderate contrast calibration pair |

### Protocol

For each pair (Ai, Aj):

1. Cool-down to < 35C.
2. Start nvidia-smi dmon on host:
   ```bash
   nvidia-smi dmon -s pucvmet -d 1 -f data/phase5_colocation/host_${Ai}_${Aj}_run${N}.csv &
   ```
3. Launch both containers simultaneously using tmux or background processes:
   ```bash
   # Container A
   docker run --rm --gpus all \
     --name toml_A \
     -v ~/TOMLCloud/data/phase5_colocation:/output \
     tomlcloud:latest \
     benchmarks/bench_${Ai}.py --duration 60 --output /output/${Ai}_with_${Aj}_run${N}.csv &

   # Container B (immediately)
   docker run --rm --gpus all \
     --name toml_B \
     -v ~/TOMLCloud/data/phase5_colocation:/output \
     tomlcloud:latest \
     benchmarks/bench_${Aj}.py --duration 60 --output /output/${Aj}_with_${Ai}_run${N}.csv &

   wait  # Wait for both to finish
   ```
4. Both containers sample power/temp/clock via pynvml internally.
5. Kill nvidia-smi dmon.
6. Repeat 3 times.

### Energy attribution in co-located runs

pynvml reports GPU-level power, not per-process. Both containers see the same total GPU power. To attribute energy:

**Method 1 - Proportional (what cAdvisor does):**
```
E_proportional(Ai) = E_total * (time_active_Ai / (time_active_Ai + time_active_Aj))
```

**Method 2 - TOML-aware:**
```
E_toml(Ai) = TO_count_Ai * E_per_TO_Ai_calibrated
E_toml(Aj) = TO_count_Aj * E_per_TO_Aj_calibrated
# E_per_TO calibrated from Phase 2 isolated runs, adjusted by TED for current temperature
# Remaining energy (thermal overhead, interference) distributed proportionally to TO count
```

**Ground truth:**
```
E_true(Ai) = E_isolated(Ai) from Phase 2
```

The gap between Method 1 and ground truth is the proportional allocation error.
The gap between Method 2 and ground truth is the TOML attribution error.
The difference between these gaps is the paper's core result.

### Output

Per-run: `data/phase5_colocation/{Ai}_with_{Aj}_run{N}.csv`
Host: `data/phase5_colocation/host_{Ai}_{Aj}_run{N}.csv`

`data/phase5_colocation/cti_matrix.csv` (7x7 matrix)
`data/phase5_colocation/cti_summary.csv`
Columns:
```
arch_a, arch_b, run, energy_a_isolated_j, energy_a_colocated_j, cti_a, energy_b_isolated_j, energy_b_colocated_j, cti_b, total_isolated_j, total_colocated_j, overhead_pct, temp_peak_c
```

### Paper output

Figure 3: CTI heatmap (7x7, asymmetric, diagonal = 0).
Table 4: CTI values for priority pairs with discussion.

**Estimated time:** ~2 hours for all 21 pairs with 3 reps each.

---

## Phase 6: Attribution Error Analysis

**Goal:** Central claim of the paper. Quantify misattribution under proportional allocation vs TOML.

### No new experiments. Pure analysis on Phase 2 + Phase 5 data.

### Method

For each co-location pair (Ai, Aj):

**Ground truth:**
```
E_true(Ai) = E_isolated(Ai) from Phase 2
E_true(Aj) = E_isolated(Aj) from Phase 2
```

**Proportional allocation:**
```
E_total_colocated = total GPU energy during co-located run (from host nvidia-smi dmon)
frac_Ai = execution_time_Ai / (execution_time_Ai + execution_time_Aj)
E_proportional(Ai) = E_total_colocated * frac_Ai
```

**TOML-aware attribution:**
```
TO_Ai = transistor operations completed by Ai
TO_Aj = transistor operations completed by Aj
E_per_TO_Ai = calibrated from Phase 2, adjusted by TED(Ai) for measured temperature
E_per_TO_Aj = same

E_toml_raw(Ai) = TO_Ai * E_per_TO_Ai
E_toml_raw(Aj) = TO_Aj * E_per_TO_Aj
E_overhead = E_total_colocated - E_toml_raw(Ai) - E_toml_raw(Aj)
# Distribute overhead proportionally to TO count:
E_toml_final(Ai) = E_toml_raw(Ai) + E_overhead * TO_Ai / (TO_Ai + TO_Aj)
```

**Error metrics:**
```
err_proportional(Ai) = |E_proportional(Ai) - E_true(Ai)| / E_true(Ai) * 100
err_toml(Ai) = |E_toml_final(Ai) - E_true(Ai)| / E_true(Ai) * 100
```

### Key analyses

1. **Mean attribution error:** average err_proportional vs err_toml across all pairs and architectures.

2. **Worst-case pair:** which co-location has the largest proportional allocation error? Hypothesis: VGG16 + RNN (highest MCER contrast, 9.0 vs 1.5).

3. **Error vs MCER contrast:** scatter plot of err_proportional vs |MCER_Ai - MCER_Aj|. Expect positive correlation. This shows the attribution error is predictable from workload characteristics.

4. **Thermal contribution:** for each pair, decompose total attribution error into:
   - MCER-driven error (difference in memory vs compute energy profiles)
   - Thermal-state error (TED-adjusted vs non-adjusted attribution)
   This shows how much of the error is from workload physics vs thermal dynamics.

5. **SPP contribution:** test whether pairs with low-SPP workloads show larger attribution error than pairs where both workloads have high SPP.

### Output

`data/phase6_attribution/attribution_comparison.csv`
Columns:
```
arch_a, arch_b, e_true_a_j, e_true_b_j, e_prop_a_j, e_prop_b_j, e_toml_a_j, e_toml_b_j, err_prop_a_pct, err_prop_b_pct, err_toml_a_pct, err_toml_b_pct, mcer_a, mcer_b, mcer_contrast, ted_a, ted_b, spp_a, spp_b
```

`data/phase6_attribution/summary_stats.csv`
```
method, mean_err_pct, median_err_pct, max_err_pct, min_err_pct, std_err_pct
```

### Paper output

Figure 4: Grouped bar chart comparing attribution error (proportional vs TOML) across all 21 pairs.
Figure 5: Attribution error vs MCER contrast scatter with regression line.
Table 5: Summary statistics for both methods.

**Estimated time:** ~1-2 hours of scripting and analysis, no GPU time.

---

## Phase 7: Container-Aware Attribution Architecture

**No experiments.** Design section for the paper (Section 6).

### Components

1. **TOML Metric Exporter:** A sidecar container or DaemonSet that:
   - Detects running GPU containers via Docker API or Kubernetes pod API
   - Identifies workload type per container (user-declared label or auto-detected via TO profiling)
   - Computes all 9 TOML metrics per container
   - Exposes Prometheus-format metrics on :9400/metrics

2. **Prometheus metrics exposed:**
   ```
   toml_energy_per_to{container, arch}
   toml_mcer{container, arch}
   toml_saft{container, arch}
   toml_ecu{container, arch}
   toml_ted{container, arch}
   toml_spp{container, arch}
   toml_cti{container_a, container_b}
   toml_attributed_energy_joules{container, method="toml"}
   toml_attributed_energy_joules{container, method="proportional"}
   toml_attribution_error_pct{container}
   ```

3. **Grafana dashboard:** Real-time visualization of per-container attributed energy, thermal state, attribution error vs proportional baseline.

4. **Integration points:**
   - Kubernetes: DaemonSet on GPU nodes, ServiceMonitor for Prometheus Operator
   - Cloud billing: export attributed energy to cloud cost management APIs
   - Carbon accounting: feed into Cloud Carbon Footprint, Climatiq, or custom carbon intensity pipelines

### Paper output

Figure 6: System architecture block diagram (TOML Exporter, Prometheus, Grafana, Kubernetes).
One page of text describing the design, integration, and deployment model.

---

## Cost Tracking

Track your Lambda Labs spend in real time. Keep a running log:

```bash
# Add to your ~/.bashrc on the Lambda instance
echo "Instance started at: $(date)" >> ~/TOMLCloud/data/cost_log.txt
```

At ~$2.49/hr, here is the budget breakdown:

| Activity | Estimated hours | Cost |
|----------|----------------|------|
| Phase 0 setup | 0.75 | $1.87 |
| Phase 1 bare-metal | 0.75 | $1.87 |
| Phase 2 isolated | 0.75 | $1.87 |
| Phase 3 thermal | 4.0 | $9.96 |
| Phase 4 SPP | 1.0 | $2.49 |
| Phase 5 co-location | 2.0 | $4.98 |
| Phase 6 analysis | 1.0 | $2.49 |
| Buffer/debugging | 3.0 | $7.47 |
| **Total** | **13.25** | **$33.00** |

That is well within the $50 budget. You could run the entire suite twice and still be under budget.

**Cost-saving tips:**
- Stop the instance between phases if you take a break. Lambda bills by the second.
- Download all data to your local machine after each phase. If the instance dies, you do not lose work.
- Phase 6 and 7 do not need GPU. Do them locally.

---

## Data Backup Protocol

After each phase, pull data to your local machine:

```powershell
scp -i ~/.ssh/lambda_key -r ubuntu@<instance-ip>:~/TOMLCloud/data/ E:\data\code\claudecode\TOMLCloud\data\
```

This protects against instance termination and ensures you can do Phase 6 analysis locally without burning GPU hours.

---

## Directory Structure (Final)

```
TOMLCloud/
├── abstract.md
├── EXPERIMENT_PLAN.md
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
├── benchmarks/
│   ├── bench_resnet18.py
│   ├── bench_resnet50.py
│   ├── bench_vgg16.py
│   ├── bench_rnn.py
│   ├── bench_lstm.py
│   ├── bench_gru.py
│   └── bench_xgboost.py
├── shared/
│   ├── power_monitor.py
│   ├── thermal_monitor.py
│   └── metrics.py
├── scripts/
│   ├── run_phase1_baremetal.py
│   ├── run_phase2_isolated.py
│   ├── run_phase3_thermal.py
│   ├── run_phase4_spp.py
│   ├── run_phase5_colocation.py
│   ├── run_phase6_analysis.py
│   └── setup_instance.sh
├── data/
│   ├── instance_metadata.csv
│   ├── cost_log.txt
│   ├── phase1_baremetal/
│   ├── phase2_isolated/
│   ├── phase3_thermal/
│   ├── phase4_spp/
│   ├── phase5_colocation/
│   └── phase6_attribution/
├── figures/
└── paper/
```

---

## Reuse from TOML FLAIRS Codebase

Copy or adapt from `E:\data\code\claudecode\toml`:

| Component | Source file | What to change |
|-----------|-----------|----------------|
| GPUPowerMonitor class | benchmark_gpu_rigorous.py | Add temp, clock, throttle fields to sampling loop |
| Model definitions | benchmark.py | Split into per-architecture scripts in benchmarks/ |
| TOML metric functions (6 original) | toml_metrics.py | Add TED, SPP, CTI as new functions |
| TO counts per architecture | toml_modelv3.py | Reuse directly |
| DDEV measurement protocol | measure_ddev.py | Reference for methodology |
| Alpha/beta coefficients | toml_coefficients.csv | Reuse for E_per_TO calibration |

---

## Validation Checklist (before writing)

- [ ] nvidia-smi power telemetry confirmed working on Lambda A100
- [ ] Container overhead < 5% for all architectures (or documented)
- [ ] TED shows measurable degradation (expect smaller on A100 than laptop, 5-15%)
- [ ] SPP varies across architectures (not all near 1.0)
- [ ] CTI is asymmetric for high-contrast pairs
- [ ] Proportional allocation error exceeds 2x for at least one pair
- [ ] TOML attribution error is lower than proportional for all tested pairs
- [ ] All measurements have 3 repetitions with acceptable variance (CV < 15%)
- [ ] Host nvidia-smi logs corroborate container pynvml readings
- [ ] All raw data backed up to local machine
- [ ] Total Lambda spend logged and under $50

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Lambda power readings return N/A | Blocks all experiments | Test immediately in Phase 0. If broken, switch to RunPod or use local 4090. |
| A100 datacenter cooling suppresses thermal effects | TED values are small | Report as finding: "datacenter cooling reduces thermal attribution error to X%, isolating MCER mismatch as dominant factor." |
| MPS causes unexpected behavior with XGBoost kernels | Co-location results for XGBoost pairs are noisy | Run XGBoost pairs with both MPS and time-slicing. Report both. |
| Instance preemption or crash mid-experiment | Data loss for current phase | Backup after every phase. Use tmux to survive SSH disconnects. |
| Container startup overhead skews short runs | Phase 2 energy includes startup cost | Exclude first 5 seconds of each run (warm-up period). Document in methodology. |
| Total cost exceeds $50 | Budget overrun | Track spend after each phase. Phase 6+7 can run locally. Worst case: skip reruns. |
