#!/bin/bash
# Stage 4 queue 2 (starts after queue 1 + classifier cache): judge classifier, baselines, label-faithfulness.
cd "$(dirname "$0")/../.." && source env.sh
until grep -q QUEUE1_DONE artifacts/logs/eval_queue_1.log && [ -f artifacts/cache/cls224_test.npy ] \
      && ! pgrep -f "train_classifier --cache" >/dev/null; do sleep 60; done
set -x
# 1. Judge = Stage 5 Real-only 100% baseline, seed 0
python -m roentgen.train_classifier --frac 1.0 --seed 0
# 2. Baselines: untrained SD 1.4, and the earliest checkpoint of each run
python -m roentgen.eval_fidelity --run sd --size 256
python -m roentgen.eval_fidelity --run A --step 1000 --size 256
python -m roentgen.eval_fidelity --run B --step 1000 --size 256
python -m roentgen.eval_fidelity --run C --step 2500 --size 256
# 3. Label faithfulness for every set
for t in real sd A_step15000 B_step15000 C_step15000 A_step1000 B_step1000 C_step2500; do
  python -m roentgen.eval_classify --tag $t
done
echo QUEUE2_DONE
