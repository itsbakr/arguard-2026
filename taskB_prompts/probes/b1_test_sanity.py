# Recovered verbatim from the development session logs (inline run, 2026-08-01 03:43 UTC).
# Run from the repository root: PYTHONPATH=. python taskB_prompts/probes/b1_test_sanity.py
# B1 test score distribution: 84.4% unsafe at threshold 0.35 against a 79.8% train prior.
# Sanity: B1 test unsafe-rate should sit near the train prior (79.8%). 84.4% is a bit
# high because thr 0.35 favours recall; verify the distribution is not degenerate.
import json, csv, numpy as np
from pathlib import Path
TEST = Path("predictions/taskb_test")
members = [json.loads(p.read_text()) for p in sorted(TEST.glob("*.json"))]
p = np.mean([m["b1_p_unsafe"] for m in members], axis=0)
print(f"B1 test p_unsafe: min {p.min():.3f} p10 {np.percentile(p,10):.3f} median {np.median(p):.3f} max {p.max():.3f}")
for t in (0.35, 0.45, 0.5):
    print(f"  thr {t}: {100*(p>=t).mean():.1f}% unsafe")
# train prior
import collections
tr = [r for r in csv.DictReader(open("data/taskB/train.csv")) if r['label']]
print(f"train unsafe rate: {100*sum(r['label']=='unsafe' for r in tr)/len(tr):.1f}%")
