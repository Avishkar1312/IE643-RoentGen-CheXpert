#!/bin/bash
# 2026-10-07: (1) finish the guidance sweep (A@g7.5 OOMed at bs 8 on 2026-10-06; existing PNGs are reused);
# (2) checkpoint sweep for C at g=4 (2500 and 15000 already exist). bs 4 for the 512px runs, as a margin for the shared GPU.
cd "$(dirname "$0")/../.." && source env.sh
set -ex
for r in A B; do
  python -m roentgen.eval_fidelity --run $r --step 15000 --size 256 --guidance 7.5 --bs 4
  python -m roentgen.eval_classify --tag ${r}_step15000_g7.5
done
echo GUIDANCE_DONE
for s in 5000 7500 10000 12500; do
  python -m roentgen.eval_fidelity --run C --step $s --size 256 --bs 8
  python -m roentgen.eval_classify --tag C_step$s
done
echo CKPT_DONE
