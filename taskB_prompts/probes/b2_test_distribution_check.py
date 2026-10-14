# Recovered verbatim from the development session logs (inline run, 2026-08-01 18:12 UTC).
# Run from the repository root: PYTHONPATH=. python taskB_prompts/probes/b2_test_distribution_check.py
# B2 predicted class distribution on test against the train prior.
# B2 has no threshold (argmax), but we can check whether char-vs-encoder blend weight
# matters and whether the predicted class distribution is sane vs train prior.
import csv, json, numpy as np
from pathlib import Path
from src.taskb_baseline import B2_LABELS, load

TEST = Path("predictions/taskb_test")
members = [json.loads(p.read_text()) for p in sorted(TEST.glob("*.json"))]
p_enc = np.mean([m["b2_p_gran"] for m in members], axis=0)
pred_enc = p_enc.argmax(1)
import collections
c = collections.Counter(B2_LABELS[i] for i in pred_enc)
tr = [r for r in csv.DictReader(open("data/taskB/train.csv")) if r['granular_label']]
prior = collections.Counter(r['granular_label'] for r in tr)
tot = len(tr)
print("B2 test predicted vs train prior:")
for lab in B2_LABELS:
    print(f"  {lab:<18} pred {100*c[lab]/len(pred_enc):5.1f}%   train {100*prior[lab]/tot:5.1f}%")
# entropy of predictions - is the model confident?
conf = p_enc.max(1)
print(f"\nmean top-class prob: {conf.mean():.3f}  (low = uncertain/harder test)")
