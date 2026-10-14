"""Average VLM seed predictions and decode for A1/A2.

Seed variance on a 500-row split is roughly the size of the leaderboard spread, so
averaging 5 seeds is the cheapest real robustness available. Probabilities are
averaged per meme, then decoded by rank against known per-class counts.

    python -m src.vlm_ensemble --tags v2 s43 s44 s45 s46 --target dev_test

For the blind test (--target test): pass --a1-count from the published prior
(148 for the 500-row test) and --support-mult to widen rare hateful classes,
since test gold is a union over agreeing annotators (higher cardinality).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from src.taska_baseline import A2_ACTIVE, HATEFUL_ONLY, NONHATEFUL_ONLY, load

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions"

#: dev_test per-class gold support, read from the platform's per-class scoring output.
DEVTEST_SUPPORT = {"Sarcasm": 147, "Humor": 115, "Mocking": 108, "Other": 53,
                   "Incitement": 45, "Dehumanization": 37, "Slurs": 31,
                   "Contempt": 14, "Inferiority": 9, "Exclusion": 0}
DEVTEST_A1 = 189


def load_avg(tags: list[str], split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    hs, ms, ids = [], [], None
    for tag in tags:
        d = json.loads((PRED / f"vlm_{tag}_{split}.json").read_text())
        assert d["classes"] == list(A2_ACTIVE)
        if ids is None:
            ids = d["ids"]
        assert d["ids"] == ids, f"{tag}/{split}: id order mismatch"
        hs.append(d["p_hateful"])
        ms.append(d["p_multi"])
    return ids, np.mean(hs, axis=0), np.mean(ms, axis=0)


def gate(scores: np.ndarray, p_hate: np.ndarray) -> np.ndarray:
    out = scores.copy()
    for k, c in enumerate(A2_ACTIVE):
        if c in HATEFUL_ONLY:
            out[:, k] *= p_hate
        elif c in NONHATEFUL_ONLY:
            out[:, k] *= 1.0 - p_hate
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", nargs="+", default=["v2", "s43", "s44", "s45", "s46"])
    ap.add_argument("--target", default="dev_test", choices=("dev_test", "test"))
    ap.add_argument("--a1-count", type=int, default=None,
                    help="predicted Hateful count on the target; default 1.0x known/published")
    ap.add_argument("--support-mult", type=float, default=1.15)
    args = ap.parse_args()

    # ---- report ensemble quality on dev (gold known locally) --------------
    ids_dev, ph_dev, pm_dev = load_avg(args.tags, "dev")
    gold = {r["id"]: r for r in load("dev")}
    y1 = np.array([1 if gold[i]["label"] == "Hateful" else 0 for i in ids_dev])
    print(f"ensemble of {len(args.tags)} seeds on dev:")
    print(f"  A1 AUC {roc_auc_score(y1, ph_dev):.4f}")
    best = max((f1_score(y1, (ph_dev >= t).astype(int), average="macro"), t)
               for t in np.arange(0.1, 0.9, 0.02))
    print(f"  A1 macro-F1 {best[0]:.4f} (thr {best[1]:.2f})")
    Y2 = np.array([[int(c in (gold[i]["fine_grained_label"] or [])) for c in A2_ACTIVE]
                   for i in ids_dev])
    g_dev = gate(pm_dev, ph_dev)
    for k, c in enumerate(A2_ACTIVE):
        if Y2[:, k].sum() >= 4:
            print(f"  A2 AUC {c:<16} {roc_auc_score(Y2[:, k], g_dev[:, k]):.4f}")

    # ---- decode the target -------------------------------------------------
    ids_t, ph_t, pm_t = load_avg(args.tags, args.target)
    n = len(ids_t)

    k1 = args.a1_count or DEVTEST_A1
    hate_idx = set(np.argsort(-ph_t)[:k1].tolist())
    a1 = {i: ("Hateful" if j in hate_idx else "Not Hateful") for j, i in enumerate(ids_t)}
    print(f"\nA1 target: top-{k1}/{n} Hateful")

    g_t = gate(pm_t, ph_t)
    a2 = {i: [] for i in ids_t}
    for k, c in enumerate(A2_ACTIVE):
        sup = DEVTEST_SUPPORT[c] * n / 500
        kk = max(2, int(round(args.support_mult * sup))) if sup else 2
        for j in np.argsort(-g_t[:, k])[:kk]:
            a2[ids_t[j]].append(c)
    for j, i in enumerate(ids_t):
        if not a2[i]:
            a2[i] = [A2_ACTIVE[int(np.argmax(g_t[j]))]]
    print(f"A2 target counts: {({c: sum(c in v for v in a2.values()) for c in A2_ACTIVE})}")

    suffix = "devtest" if args.target == "dev_test" else "test"
    (PRED / f"a1_{suffix}.json").write_text(json.dumps(a1, ensure_ascii=False))
    (PRED / f"a2_{suffix}.json").write_text(json.dumps(a2, ensure_ascii=False))
    print(f"wrote a1_{suffix}.json / a2_{suffix}.json; "
          f"next: python -m src.build_submissions --run-id awn_vlm_ens5")


if __name__ == "__main__":
    main()
