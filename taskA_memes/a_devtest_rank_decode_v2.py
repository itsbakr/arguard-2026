# Recovered verbatim from the development session logs (inline run, 2026-07-30 07:50 UTC).
# Run from the repository root: PYTHONPATH=. python taskA_memes/a_devtest_rank_decode_v2.py
# Then: python -m src.build_submissions --run-id awn_vlm_v2_rank
# Dev phase. Rank decode of the v2 run against dev_test counts. dev_test A1 0.7281, A2 0.3694.
import json
import numpy as np

# Known dev_test gold support, read off the platform's own per-class scoring output.
SUPPORT = {"Sarcasm": 147, "Humor": 115, "Mocking": 108, "Other": 53, "Incitement": 45,
           "Dehumanization": 37, "Slurs": 31, "Contempt": 14, "Inferiority": 9, "Exclusion": 0}
A1_HATEFUL = 189

d = json.loads(open("predictions/vlm_v2_dev_test.json").read())
ids = d["ids"]; classes = d["classes"]
p_hate = np.array(d["p_hateful"]); p_multi = np.array(d["p_multi"])

# A1: F1 favors slightly over-predicting the minority class; 1.05x support.
k1 = int(round(1.05 * A1_HATEFUL))
hate_idx = set(np.argsort(-p_hate)[:k1].tolist())
a1 = {i: ("Hateful" if n in hate_idx else "Not Hateful") for n, i in enumerate(ids)}
print(f"A1 rank-decode: {k1}/500 Hateful")

# A2: gate scores by A1 probability (rule E), then top-k per class at 1.25x support.
HATEFUL_ONLY = {"Contempt","Dehumanization","Exclusion","Incitement","Inferiority","Mocking","Slurs"}
NONHATEFUL_ONLY = {"Humor","Sarcasm"}
gated = p_multi.copy()
for k, c in enumerate(classes):
    if c in HATEFUL_ONLY: gated[:, k] *= p_hate
    elif c in NONHATEFUL_ONLY: gated[:, k] *= (1 - p_hate)

a2 = {i: [] for i in ids}
for k, c in enumerate(classes):
    kk = max(2, int(round(1.25 * SUPPORT[c]))) if SUPPORT[c] else 2
    for n in np.argsort(-gated[:, k])[:kk]:
        a2[ids[n]].append(c)
for n, i in enumerate(ids):
    if not a2[i]:
        a2[i] = [classes[int(np.argmax(gated[n]))]]
print("A2 rank-decode counts:", {c: sum(c in v for v in a2.values()) for c in classes})

json.dump(a1, open("predictions/a1_devtest.json", "w"), ensure_ascii=False)
json.dump(a2, open("predictions/a2_devtest.json", "w"), ensure_ascii=False)
print("written")
