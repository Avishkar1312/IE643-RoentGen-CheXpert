#!/bin/bash
# 2026-10-08: extend C 15k -> 25k (same hparams, constant lr after warmup; ckpt only at 2.5k milestones for disk),
# then eval 17.5k-25k at g=4 and g=5.5. bs 4 for the shared GPU.
cd "$(dirname "$0")/../.." && source env.sh
set -ex
python -m roentgen.train_lora --run C --steps 25000 --resume_from artifacts/runs/C/ckpt-15000 \
  --ckpt_every 2500 --milestone_every 2500 > artifacts/runs/C/train_15k-25k.log 2>&1
echo TRAIN_DONE
for s in 17500 20000 22500 25000; do
  python -m roentgen.eval_fidelity --run C --step $s --size 256 --bs 4
  python -m roentgen.eval_classify --tag C_step$s
  python -m roentgen.eval_fidelity --run C --step $s --size 256 --guidance 5.5 --bs 4
  python -m roentgen.eval_classify --tag C_step${s}_g5.5
done
echo EVAL_DONE
