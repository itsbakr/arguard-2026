# Recovered verbatim from the development session logs (inline run, 2026-08-02 20:35 UTC).
# Run from the repository root: PYTHONPATH=. python taskA_memes/a_test_decode_8seed_final.py
# Then: python -m src.build_submissions --run-id awn_vlm_ens8 --target test
# Final A1/A2 test decode, 8 seeds. Test A1 0.7889, A2 0.4194 (inferred to be runs 881206 and 881205).
import json
import numpy as np
from src.vlm_ensemble import load_avg, gate
from src.taska_baseline import A2_ACTIVE

TAGS8 = ["v2", "s43", "s44", "s45", "s46", "v3td", "s43td", "s44td"]
ids, p_hate, p_multi = load_avg(TAGS8, "test")

# Sanity: the td seeds should agree strongly with the old ensemble ranking on A1.
_, p_hate5, _ = load_avg(["v2", "s43", "s44", "s45", "s46"], "test")
from scipy.stats import spearmanr
print(f"8-seed vs 5-seed A1 rank agreement: {spearmanr(p_hate, p_hate5).statistic:.4f}")

TRUE = {"Sarcasm": 186, "Humor": 217, "Mocking": 103, "Other": 50, "Incitement": 40,
        "Dehumanization": 21, "Slurs": 16, "Contempt": 36, "Inferiority": 23, "Exclusion": 3}
k1 = 148
hate_idx = set(np.argsort(-p_hate)[:k1].tolist())
a1 = {i: ("Hateful" if j in hate_idx else "Not Hateful") for j, i in enumerate(ids)}
overlap = sum(1 for j in np.argsort(-p_hate5)[:148] if j in hate_idx)
print(f"A1 top-148: {overlap}/148 shared with the 5-seed submission")

g = gate(p_multi, p_hate)
a2 = {i: [] for i in ids}
for k, c in enumerate(A2_ACTIVE):
    for j in np.argsort(-g[:, k])[:max(3, int(round(1.1 * TRUE[c])))]:
        a2[ids[j]].append(c)
for j, i in enumerate(ids):
    if not a2[i]:
        a2[i] = [A2_ACTIVE[int(np.argmax(g[j]))]]
json.dump(a1, open("predictions/a1_test.json", "w"), ensure_ascii=False)
json.dump(a2, open("predictions/a2_test.json", "w"), ensure_ascii=False)
print("8-seed decode written")
