#!/bin/bash
# Guidance-scale sweep (2026-10-06): fidelity + label AUROC vs CFG scale. g=4.0 already exists for all runs.
# Batch sizes halved: the Unreal Engine editor holds ~2.5 GB of GPU memory; default bs OOMed.
cd "$(dirname "$0")/../.." && source env.sh
set -ex
for g in 1.0 2.0 3.0 5.5 7.5; do
  python -m roentgen.eval_fidelity --run C --step 15000 --size 256 --guidance $g --bs 8
  python -m roentgen.eval_classify --tag C_step15000_g${g%.0}
done
# Do A/B condition better under strong guidance?
for r in A B; do
  python -m roentgen.eval_fidelity --run $r --step 15000 --size 256 --guidance 7.5 --bs $([ $r = B ] && echo 4 || echo 8)
  python -m roentgen.eval_classify --tag ${r}_step15000_g7.5
done
echo GUIDANCE_DONE
