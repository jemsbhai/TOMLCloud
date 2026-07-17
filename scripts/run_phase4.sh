#!/bin/bash
# Phase 4: SPP (Sustained Power Proportionality)
# 7 architectures x 5 load levels x 30 seconds = ~25 min total

set -e
echo "=== PHASE 4: SPP ==="
echo "Started: $(date)"

for arch in resnet18 resnet50 vgg16 rnn lstm gru; do
  for load in 10 25 50 75 100; do
    echo "--- ${arch} @ ${load}% ---"
    python3 benchmarks/bench_${arch}.py \
      --duration 30 \
      --load-pct ${load} \
      --no-cooldown \
      --output data/phase4_spp/${arch}_load${load}.csv
  done
done

for load in 10 25 50 75 100; do
  echo "--- xgboost @ ${load}% ---"
  python3 benchmarks/bench_xgboost.py \
    --duration 30 \
    --load-pct ${load} \
    --no-cooldown \
    --output data/phase4_spp/xgboost_load${load}.csv
done

echo "Phase 4 complete: $(date)"
