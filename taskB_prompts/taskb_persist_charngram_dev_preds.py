# Recovered verbatim from the development session logs (inline run, 2026-07-30 03:58 UTC).
# Run from the repository root: PYTHONPATH=. python taskB_prompts/taskb_persist_charngram_dev_preds.py
# Dev phase. Saves the char-ngram B1/B2 dev predictions for src.build_submissions (dev B1 0.9203, B2 0.8620).
# Persist the Task B dev predictions from the baseline so build_submissions can read them.
import json, sys
sys.path.insert(0, ".")
from src.taskb_baseline import run
res = run(report=False)
p = res["_predictions"]
json.dump(p["b1"], open("predictions/b1_dev.json", "w"), ensure_ascii=False)
json.dump(p["b2"], open("predictions/b2_dev.json", "w"), ensure_ascii=False)
json.dump({"b1_threshold": res["_models"]["b1_threshold"],
           "b1_dev_macro_f1": res["b1"]["dev_macro_f1"],
           "b2_dev_macro_f1": res["b2"]["dev_macro_f1"]},
          open("predictions/b_meta.json", "w"), indent=2)
print("saved B predictions")
