# Recovered verbatim from the development session logs (inline run, 2026-08-02 20:09 UTC).
# Run from the repository root: PYTHONPATH=. python taskB_prompts/probes/b1_balanced_ensemble_probe.py
# Needs predictions from modal run src/modal_taskb.py::sweep_balanced.
# Spearman 0.946 against the original ensemble. Its B1 file was overwritten and never scored.
import csv, json, numpy as np
from pathlib import Path
from src import submit

bal = [json.loads(p.read_text()) for p in sorted(Path("predictions/taskb_test_bal").glob("*.json"))]
old = [json.loads(p.read_text()) for p in sorted(Path("predictions/taskb_test").glob("*.json"))]
p_bal = np.mean([m["b1_p_unsafe"] for m in bal], axis=0)
p_old = np.mean([m["b1_p_unsafe"] for m in old], axis=0)
rows = list(csv.DictReader(open("data/taskB/Arguard_binary_test_no_label.csv", encoding="utf-8")))
ids = [r["id"] for r in rows]

print("balanced-trained ensemble on test:")
print(f"  p_unsafe: p10 {np.percentile(p_bal,10):.3f} median {np.median(p_bal):.3f} p90 {np.percentile(p_bal,90):.3f}")
print(f"  unsafe-rate at argmax(0.5): {100*(p_bal>=0.5).mean():.1f}%   [old model at 0.5: {100*(p_old>=0.5).mean():.1f}%]")
from scipy.stats import spearmanr
print(f"  rank agreement with old ensemble: spearman {spearmanr(p_bal, p_old).statistic:.3f}")

# Balanced training moves the boundary itself; trust its native 0.5 argmax, but also
# pin the rate to ~49% as a safety net. Compare the two decision sets:
pred_05 = p_bal >= 0.5
thr49 = float(np.quantile(p_bal, 0.51))
pred_49 = p_bal >= thr49
print(f"  0.5-argmax vs 49%-quantile: disagree on {(pred_05 != pred_49).sum()} rows")

# Ship the 49%-pinned version: same operating point that scored 0.7552, better model.
pred = {i: ("unsafe" if x else "safe") for i, x in zip(ids, pred_49)}
submit.build_b1(pred, Path("submissions/b1_prediction.zip"), expected_ids=ids)
print(f"\nwrote balanced-model B1 probe: {sum(pred_49)}/{len(ids)} unsafe")
