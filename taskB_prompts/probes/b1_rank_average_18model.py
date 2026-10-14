# Recovered verbatim from the development session logs (inline run, 2026-08-02 20:09 UTC).
# Run from the repository root: PYTHONPATH=. python taskB_prompts/probes/b1_rank_average_18model.py
# 18-model rank average (9 original + 9 balanced), 49% quantile. Test B1 0.7381.
import csv, json, numpy as np
from pathlib import Path
from scipy.stats import rankdata
from src import submit

bal = [json.loads(p.read_text()) for p in sorted(Path("predictions/taskb_test_bal").glob("*.json"))]
old = [json.loads(p.read_text()) for p in sorted(Path("predictions/taskb_test").glob("*.json"))]
rows = list(csv.DictReader(open("data/taskB/Arguard_binary_test_no_label.csv", encoding="utf-8")))
ids = [r["id"] for r in rows]

# Rank-average all 18 members: rank space is immune to the two ensembles'
# different calibration (one trained 80/20, one 50/50).
ranks = np.mean([rankdata(m["b1_p_unsafe"]) for m in bal + old], axis=0)
thr = np.quantile(ranks, 0.51)
pred = {i: ("unsafe" if r >= thr else "safe") for i, r in zip(ids, ranks)}
submit.build_b1(pred, Path("submissions/b1_prediction.zip"), expected_ids=ids)
n = sum(v == "unsafe" for v in pred.values())
print(f"18-model rank-averaged B1: {n}/{len(ids)} unsafe ({n/len(ids):.1%})")
print(submit.describe(Path("submissions/b1_prediction.zip")))
