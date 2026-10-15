#!/bin/bash
# Stage 3: paper-sampler check subset, then the full DPM++ twin corpus (resumable: re-run to continue).
cd "$(dirname "$0")/../.." && source env.sh
python -m roentgen.generate --sampler pndm --steps 75 --limit 2000 && echo PNDM_DONE
python -m roentgen.generate && echo DPM_DONE
