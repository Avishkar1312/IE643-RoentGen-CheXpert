#!/bin/bash
# Stage 4a eval queue 1: finish B, then A, both at step 15000 on the common 256px grid.
cd "$(dirname "$0")/../.." && source env.sh
set -x
python -m roentgen.eval_fidelity --run B --step 15000 --size 256
python -m roentgen.eval_fidelity --run A --step 15000 --size 256
echo QUEUE1_DONE
