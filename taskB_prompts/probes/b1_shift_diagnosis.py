# Recovered verbatim from the development session logs (inline run, 2026-08-01 17:59 UTC).
# Run from the repository root: PYTHONPATH=. python taskB_prompts/probes/b1_shift_diagnosis.py
# Prints B1 test score distributions and the reported score. The ~49% unsafe
# estimate was worked out by hand from these numbers; this script does not compute it.
import csv, json, numpy as np
from pathlib import Path
from src.taskb_baseline import B2_LABELS, load, make_model

TEST = Path("predictions/taskb_test")
members = [json.loads(p.read_text()) for p in sorted(TEST.glob("*.json"))]
p_enc = np.mean([m["b1_p_unsafe"] for m in members], axis=0)

print("=== B1 encoder-ensemble p_unsafe on TEST ===")
print(f"  saturation: p10 {np.percentile(p_enc,10):.3f}  median {np.median(p_enc):.3f}  p90 {np.percentile(p_enc,90):.3f}")
print("  unsafe-rate vs threshold:")
for t in (0.35, 0.5, 0.7, 0.9, 0.95, 0.99):
    print(f"    thr {t:.2f} -> {100*(p_enc>=t).mean():.1f}% unsafe")

# What does the char-ngram alone predict on test (less saturated model)?
tr = load("train.csv"); dv = load("dev_with_label.csv")
char = make_model("logreg")
X = tr.text + dv.text; y = tr.label + dv.label
char.fit(X, y)
b1_rows = list(csv.DictReader(open("data/taskB/Arguard_binary_test_no_label.csv", encoding="utf-8")))
b1_text = [f"[{r.get('dialect') or 'UNK'}] {r['prompt']}" for r in b1_rows]
col = list(char.classes_).index("unsafe")
p_char = char.predict_proba(b1_text)[:, col]
print("\n=== B1 char-ngram p_unsafe on TEST ===")
print(f"  p10 {np.percentile(p_char,10):.3f}  median {np.median(p_char):.3f}  p90 {np.percentile(p_char,90):.3f}")
for t in (0.35, 0.5, 0.7):
    print(f"    thr {t:.2f} -> {100*(p_char>=t).mean():.1f}% unsafe")

# Back out the implied true unsafe rate from the reported B1 score.
# accuracy 0.6417, we predicted 84.4% unsafe.
print("\n=== implied test balance ===")
print("  reported B1: acc 0.6417, macro-F1 0.5959, we predicted 84.4% unsafe")
print("  train unsafe rate: 79.8%")
