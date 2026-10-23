"""Stage 5 summary: test macro AUROC of the real / synth / mixed classifiers at 1%, 10%, 100%.

    python -m roentgen.stage5_table

Reads artifacts/cls/<data>_f<frac>_s<seed>/result.json (written by train_classifier) and reports
mean ± std over seeds per (data, frac), per-target means, and mixed − real / synth − real deltas.
Missing runs are listed rather than skipped silently. Writes results/stage5.json and prints a
markdown table for docs/STATUS.md.
"""
import json

import numpy as np

from roentgen.config import ARTIFACTS, TARGETS
from roentgen.eval_fidelity import RESULTS

DATA = ["real", "synth", "mixed"]
FRACS = ["0.01", "0.1", "1.0"]
SEEDS = [0, 1, 2]


def main():
    cells, missing = {}, []
    for frac in FRACS:
        for data in DATA:
            runs = []
            for seed in SEEDS:
                f = ARTIFACTS / "cls" / f"{data}_f{frac}_s{seed}" / "result.json"
                if f.exists():
                    runs.append(json.loads(f.read_text()))
                else:
                    missing.append(f.parent.name)
            if not runs:
                continue
            macro = np.array([r["test_macro"] for r in runs])
            cells[f"{data}_f{frac}"] = {
                "n_seeds": len(runs), "n_train": runs[0]["n_train"],
                "macro_mean": macro.mean(), "macro_std": macro.std(ddof=1) if len(runs) > 1 else 0.0,
                "per_target": {t: float(np.mean([r["test"][t] for r in runs])) for t in TARGETS},
            }
    for frac in FRACS:
        real = cells.get(f"real_f{frac}")
        for data in ("synth", "mixed"):
            c = cells.get(f"{data}_f{frac}")
            if real and c:
                c["delta_vs_real"] = c["macro_mean"] - real["macro_mean"]

    out = {"cells": cells, "missing": missing}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "stage5.json").write_text(json.dumps(out, indent=2, default=float))

    print("| frac | " + " | ".join(DATA) + " | synth − real | mixed − real |")
    print("|" + "---|" * (len(DATA) + 3))
    for frac in FRACS:
        row = [f"{float(frac):.0%}"]
        for data in DATA:
            c = cells.get(f"{data}_f{frac}")
            row.append(f"{c['macro_mean']:.3f} ± {c['macro_std']:.3f} (n={c['n_seeds']})" if c else "–")
        for data in ("synth", "mixed"):
            d = cells.get(f"{data}_f{frac}", {}).get("delta_vs_real")
            row.append(f"{d:+.3f}" if d is not None else "–")
        print("| " + " | ".join(row) + " |")
    if missing:
        print(f"\nmissing ({len(missing)}): {', '.join(missing)}")
    print(f"wrote {RESULTS / 'stage5.json'}")


if __name__ == "__main__":
    main()
