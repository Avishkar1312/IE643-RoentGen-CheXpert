#!/bin/bash
# 2026-10-25: final Stage 2 -> 5 chain on C@15k (replaces the tail of queue_recover.sh):
#   - run C stops at 15k (lr is constant after warmup, so ckpt-15000 is a true 15k model; old run: AUROC 0.847, FID_XRV 0.69)
#   - twins only for the union of the Stage 5 1% + 10% subsets (first 44,603 rows in priority order)
#   - Stage 5 grid at 1% and 10%; 100% has just the judge (real_f1.0_s0)
# Resumable: re-run to continue. Old 22.5k-corpus synth/mixed results stay in git history.
cd "$(dirname "$0")/../.." && source env.sh
set -ex

wait_gpu() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "$1" ]; do sleep 120; done; }
STEP=15000
SYNTH=artifacts/synth/C_step${STEP}_g4.0_dpm25

# Stop run C once ckpt-15000 is fully written (state.pt is saved last; logging past 15000 means it is done).
last_step() { tail -1 artifacts/runs/C/log.jsonl 2>/dev/null | python -c "
import json, sys
try: print(json.loads(sys.stdin.read())['step'])
except Exception: print(0)"; }
if [ ! -f artifacts/runs/C/ckpt-$STEP/state.pt ] || pgrep -f "^python -m roentgen.train_lora --run C" >/dev/null; then
  while [ "$(last_step)" -le "$STEP" ]; do sleep 60; done
  pkill -f "queue_recover.sh" || true
  pkill -f "^python -m roentgen.train_lora --run C" || true
  while pgrep -f "^python -m roentgen.train_lora --run C" >/dev/null; do sleep 10; done
fi
test -f artifacts/runs/C/ckpt-$STEP/trainable.safetensors
echo TRAIN_DONE

# Sanity vs the old C@15k (git show 7140a95:results/classify_C_step15000_256px.json).
if [ ! results/classify_C_step${STEP}_256px.json -nt artifacts/runs/C/ckpt-$STEP ]; then
  wait_gpu 6000
  python -m roentgen.eval_fidelity --run C --step $STEP --size 256 --bs 8 && python -m roentgen.eval_classify --tag C_step$STEP || echo "EVAL_FAILED (sanity only; continuing to generation)"
fi
echo EVAL_DONE

# Stage 3: twins for the 1% + 10% subsets. bs 32 to amortise the autocast weight casts; falls back to 8 on OOM.
wait_gpu 6000
python -m roentgen.generate --step $STEP --limit 44603 --bs 32 >> artifacts/logs/generate.log 2>&1 \
  || python -m roentgen.generate --step $STEP --limit 44603 --bs 8 >> artifacts/logs/generate.log 2>&1
echo DPM_DONE
[ -f $SYNTH/filter.parquet ] || python -m roentgen.filtering --synth $SYNTH
echo STAGE3_DONE

# Stage 5 at 1% / 10%. Real runs don't depend on the twins; synth/mixed must come from this corpus.
for frac in 0.01 0.1; do
  for seed in 0 1 2; do
    for data in real synth mixed; do
      r=artifacts/cls/${data}_f${frac}_s${seed}/result.json
      if [ -f $r ] && { [ $data = real ] || grep -q "C_step${STEP}_g4.0_dpm25" $r; }; then echo "skip $r"; continue; fi
      python -m roentgen.train_classifier --data $data --frac $frac --seed $seed --synth $SYNTH || echo "FAILED $r"
    done
  done
done
echo STAGE5_DONE
python -m roentgen.stage5_table
echo Q15K_DONE
