#!/bin/bash
# 2026-10-24: rebuild everything gitignored after the project folder was deleted, up to and including Stage 5
# (resumable: every step is skipped once its output exists; re-run the queue to continue).
# Stage 2-4 sweeps, runs A/B, the PNDM check and CLIP/qualitative evals are not redone: their results are in git.
# Old values to compare against stay in git history (manifest_meta.json, runs/C/log.jsonl, results/*_C_step22500_*).
cd "$(dirname "$0")/../.." && source env.sh
set -ex

# Wait until the GPU has $1 MiB free (other users' jobs share this card).
wait_gpu() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "$1" ]; do sleep 120; done; }

# Stage 1: manifest. Must reproduce the committed splits exactly, or none of the old results carry over.
if [ ! -f artifacts/manifest.parquet ]; then
  python -m roentgen.chexpert --build > artifacts/logs/manifest.log
  python - <<'EOF'
import json, subprocess
old = json.loads(subprocess.check_output(["git", "show", "HEAD:artifacts/manifest_meta.json"]))
new = json.load(open("artifacts/manifest_meta.json"))
for m in (old, new): m.pop("chexpert_dir")
assert old == new, "manifest differs from the committed one"
print("MANIFEST_MATCH")
EOF
fi
[ -f artifacts/cache/cls224_test.npy ] || python -m roentgen.train_classifier --cache
[ -f artifacts/cache/latents_256.npy ] || { wait_gpu 4000; python -m roentgen.cache_latents --size 256; }
echo STAGE1_DONE

# Judge (Real-only 100%, seed 0): used by filtering and eval_classify.
if [ ! -f artifacts/cls/real_f1.0_s0/best.pt ]; then
  wait_gpu 6000; python -m roentgen.train_classifier --frac 1.0 --seed 0
fi
echo JUDGE_DONE

# Stage 2: run C straight to 22.5k (draws are seeded by micro-step, lr is constant after warmup,
# so this follows the original 0-15k + 15k-25k trajectory). Milestones keep 7.5k/15k/22.5k.
if [ ! -d artifacts/runs/C/ckpt-22500 ]; then
  if ls -d artifacts/runs/C/ckpt-* >/dev/null 2>&1; then res="--resume_from latest"; else res=""; : > artifacts/runs/C/log.jsonl; fi
  wait_gpu 10500
  python -m roentgen.train_lora --run C --steps 22500 --ckpt_every 2500 --milestone_every 7500 $res \
    >> artifacts/runs/C/train.log 2>&1
fi
echo TRAIN_DONE

# Sanity: the retrained C@22.5k should land near the original (FID_XRV 0.59, own AUROC 0.875).
if [ ! results/classify_C_step22500_256px.json -nt artifacts/runs/C/ckpt-22500 ]; then
  wait_gpu 6000
  python -m roentgen.eval_fidelity --run C --step 22500 --size 256 --bs 8
  python -m roentgen.eval_classify --tag C_step22500
fi
echo EVAL_DONE

# Stage 3: DPM++ 25 twin corpus (priority order: Stage 5 1% subsets first), then filtering.
wait_gpu 6000
python -m roentgen.generate >> artifacts/logs/generate.log 2>&1
echo DPM_DONE
[ -f artifacts/synth/C_step22500_g4.0_dpm25/filter.parquet ] || python -m roentgen.filtering
echo STAGE3_DONE

# Stage 5 grid, then the summary table.
bash artifacts/logs/queue_stage5.sh >> artifacts/logs/stage5.log 2>&1
python -m roentgen.stage5_table
echo RECOVER_DONE
