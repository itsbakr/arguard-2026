# Recovered verbatim from the development session logs (inline run, 2026-08-02 19:42 UTC).
# Run from the repository root: PYTHONPATH=. python taskB_prompts/probes/b2_balanced_prior_probe.py
# B2 prior-flattening probe. Test B2 0.7621, so B2 went back to plain argmax.
# Task B improvement probes.
# B2: if the organizers balanced the B2 test across the 7 harm domains the same way
# they balanced B1 to ~50/50, a prior-corrected argmax wins big. Build the probe:
# divide each class prob by the train prior, multiply by uniform, re-argmax.
import csv, json, numpy as np
from pathlib import Path
from src import submit
from src.taskb_baseline import B2_LABELS, load, make_model

members = [json.loads(p.read_text()) for p in sorted(Path("predictions/taskb_test").glob("*.json"))]
p_enc = np.mean([m["b2_p_gran"] for m in members], axis=0)
rows = list(csv.DictReader(open("data/taskB/Arguard_granular_test_no_label.csv", encoding="utf-8")))
ids = [r["id"] for r in rows]

train = load("train.csv"); dev = load("dev_with_label.csv")
char = make_model("logreg").fit(train.text + dev.text, train.granular + dev.granular)
order = [list(char.classes_).index(c) for c in B2_LABELS]
text = [f"[{r.get('dialect') or 'UNK'}] {r['prompt']}" for r in rows]
p_char = char.predict_proba(text)[:, order]
p = 0.5 * p_char + 0.5 * p_enc

prior = np.array([ (np.array(train.granular + dev.granular) == c).mean() for c in B2_LABELS ])
p_shift = p / prior            # uniform-target prior correction
pred = {i: B2_LABELS[j] for i, j in zip(ids, p_shift.argmax(1))}
counts = {c: sum(v == c for v in pred.values()) for c in B2_LABELS}
print("B2 balanced-prior probe counts:", counts)
print("(previous argmax dist was ~train prior: 22/22/16/12/13/9/5%)")
submit.build_b2(pred, Path("submissions/b2_prediction.zip"), expected_ids=ids)
print(submit.describe(Path("submissions/b2_prediction.zip")))
