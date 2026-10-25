#!/bin/bash
# Quick check on background GPU jobs: ./jobstatus.sh   (live view: watch -n 30 ./jobstatus.sh)
cd "$(dirname "$0")"
echo "== $(date '+%a %H:%M') =="
echo "-- running jobs:"
pgrep -af "roentgen\." | grep -v pgrep || echo "   none (finished or crashed — see log below)"
echo "-- recovery queue (artifacts/logs/queue_recover.sh, *_DONE marks stages):"
grep -E "_DONE|MATCH|MISSING|Error|Traceback" artifacts/logs/recover.log | tail -3
echo "-- 15k queue (artifacts/logs/queue_15k.sh; TRAIN/EVAL/DPM/STAGE3/STAGE5/Q15K_DONE):"
grep -E "_DONE|FAILED|skip|Error|Traceback" artifacts/logs/q15k.log | tail -3
echo "-- run C step: $(tail -1 artifacts/runs/C/log.jsonl | grep -o '"step": [0-9]*')"
echo "-- Stage 3 generate log (PNDM_DONE / DPM_DONE mark stages, 'Error' = crash):"
grep -v -i warn artifacts/logs/generate.log | tail -4
echo "-- PNGs so far:"
for d in artifacts/synth/*/png; do echo "   $d: $(ls $d | wc -l)"; done
echo "-- GPU:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
