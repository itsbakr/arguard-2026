# Recovered verbatim from the development session logs (inline run, 2026-08-03 07:53 UTC).
# Run from the repository root: PYTHONPATH=. python taskB_prompts/b1_final_build.py
# Same B1 rule, last build of the B1 file before the final upload (inferred to be run 881203, 0.7552).
# Restore B1 to the proven 9-model recalibrated version (0.7552).
import csv, json, numpy as np
from pathlib import Path
from src import submit
members = [json.loads(p.read_text()) for p in sorted(Path("predictions/taskb_test").glob("*.json"))]
p = np.mean([m["b1_p_unsafe"] for m in members], axis=0)
rows = list(csv.DictReader(open("data/taskB/Arguard_binary_test_no_label.csv", encoding="utf-8")))
ids = [r["id"] for r in rows]
thr = float(np.quantile(p, 0.51))
submit.build_b1({i: ("unsafe" if x >= thr else "safe") for i, x in zip(ids, p)},
                Path("submissions/b1_prediction.zip"), expected_ids=ids)
print("B1 restored to the 0.7552 recipe")
