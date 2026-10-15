#!/bin/bash
# 2026-10-07: C checkpoint sweep at g=4 (retry; bs 8 OOMed when another process held 4.8 GB). bs 4 for margin.
cd "$(dirname "$0")/../.." && source env.sh
set -ex
for s in 5000 7500 10000 12500; do
  python -m roentgen.eval_fidelity --run C --step $s --size 256 --bs 4
  python -m roentgen.eval_classify --tag C_step$s
done
echo CKPT_DONE
