#!/bin/bash
# 2026-10-07: Stage 4c BiomedCLIP on CPU (GPU busy with queue_resume_ckpt.sh). real already done.
cd "$(dirname "$0")/../.." && source env.sh
set -ex
for t in C_step15000 C_step15000_g7.5 C_step2500 A_step15000 B_step15000 sd; do
  python -m roentgen.eval_clip --tag $t --device cpu
done
echo CLIP_DONE
