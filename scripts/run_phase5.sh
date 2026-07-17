#!/bin/bash
# Phase 5: CTI (Co-location Thermal Interference)
# 6 priority pairs x 1 rep x 60 seconds = ~15 min total
# Uses MPS for concurrent GPU sharing

set -e
echo "=== PHASE 5: CTI ==="
echo "Started: $(date)"

# Enable MPS
echo "Enabling MPS..."
echo quit | sudo nvidia-cuda-mps-control 2>/dev/null || true
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
sudo nvidia-cuda-mps-control -d
sleep 2
echo "MPS enabled."

# Priority pairs: (arch_a, arch_b)
declare -a PAIRS=(
  "vgg16 rnn"
  "resnet50 xgboost"
  "lstm gru"
  "vgg16 resnet18"
  "xgboost rnn"
  "resnet50 lstm"
)

for pair in "${PAIRS[@]}"; do
  read -r arch_a arch_b <<< "$pair"
  echo "--- ${arch_a} + ${arch_b} ---"

  # Start host nvidia-smi monitoring
  nvidia-smi dmon -s pucvmet -d 1 -f data/phase5_colocation/host_${arch_a}_${arch_b}.csv &
  DMON_PID=$!

  # Launch both workloads concurrently
  if [ "$arch_a" = "xgboost" ]; then
    python3 benchmarks/bench_xgboost.py --duration 60 --no-cooldown \
      --output data/phase5_colocation/${arch_a}_with_${arch_b}.csv &
    PID_A=$!
  else
    python3 benchmarks/bench_${arch_a}.py --duration 60 --no-cooldown \
      --output data/phase5_colocation/${arch_a}_with_${arch_b}.csv &
    PID_A=$!
  fi

  if [ "$arch_b" = "xgboost" ]; then
    python3 benchmarks/bench_xgboost.py --duration 60 --no-cooldown \
      --output data/phase5_colocation/${arch_b}_with_${arch_a}.csv &
    PID_B=$!
  else
    python3 benchmarks/bench_${arch_b}.py --duration 60 --no-cooldown \
      --output data/phase5_colocation/${arch_b}_with_${arch_a}.csv &
    PID_B=$!
  fi

  # Wait for both to finish
  wait $PID_A
  wait $PID_B

  # Stop host monitoring
  kill $DMON_PID 2>/dev/null || true
  sleep 2

  echo "  Pair ${arch_a}+${arch_b} complete."
done

# Disable MPS
echo "Disabling MPS..."
echo quit | sudo nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c DEFAULT
echo "MPS disabled."

echo "Phase 5 complete: $(date)"
