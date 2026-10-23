#!/bin/bash
# Stage 5: Real / Synth / Mixed classifier grid at 1%, 10%, 100%, seeds 0-2 (resumable: finished runs are skipped).
# Waits for filtering.py so the two don't share the GPU.
cd "$(dirname "$0")/../.." && source env.sh
while pgrep -f "^python -m roentgen.filtering" >/dev/null; do sleep 60; done
for frac in 0.01 0.1 1.0; do
  for seed in 0 1 2; do
    for data in real synth mixed; do
      d=artifacts/cls/${data}_f${frac}_s${seed}
      if [ -f "$d/best.pt" ] && [ -f "$d/result.json" ]; then echo "skip $d"; continue; fi
      echo "== $data frac=$frac seed=$seed $(date)"
      python -m roentgen.train_classifier --data $data --frac $frac --seed $seed || echo "FAILED $d"
    done
  done
done
echo STAGE5_DONE
