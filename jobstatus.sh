#!/bin/bash
# Quick check on background GPU jobs: ./jobstatus.sh   (live view: watch -n 30 ./jobstatus.sh)
cd "$(dirname "$0")"
echo "== $(date '+%a %H:%M') =="
echo "-- running jobs:"
pgrep -af "roentgen\." | grep -v pgrep || echo "   none (finished or crashed — see log below)"
echo "-- Stage 3 generate log (PNDM_DONE / DPM_DONE mark stages, 'Error' = crash):"
grep -v -i warn artifacts/logs/generate.log | tail -4
echo "-- PNGs so far:"
for d in artifacts/synth/*/png; do echo "   $d: $(ls $d | wc -l)"; done
echo "-- GPU:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
