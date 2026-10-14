"""Turn raw VLM probabilities into A1/A2 submissions using decode rule F.

The VLM's `_predict` returns raw sigmoid probabilities on purpose: the decode rule
(per-class thresholds fitted on gated scores, plus a rare-class floor) is fitted
locally against labelled data and applied to the unlabelled target split. Keeping
decoding local means re-fitting costs seconds, not a GPU run.

Inputs, produced by ``modal run src/modal_vlm.py::predict``:
    predictions/vlm_<tag>_dev.json        calibration split (has gold locally)
    predictions/vlm_<tag>_dev_test.json   submission target (dev phase)

    python -m src.vlm_decode --tag v1 --target dev_test

Outputs ``predictions/a1_devtest.json`` and ``predictions/a2_devtest.json`` in the
same shape the char-ngram baseline wrote, so ``src.build_submissions`` needs no
changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src.taska_baseline import A2_ACTIVE, HATEFUL_ONLY, NONHATEFUL_ONLY, best_threshold, load

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions"


def gate(scores: np.ndarray, p_hate: np.ndarray) -> np.ndarray:
    out = scores.copy()
    for k, cls in enumerate(A2_ACTIVE):
        if cls in HATEFUL_ONLY:
            out[:, k] *= p_hate
        elif cls in NONHATEFUL_ONLY:
            out[:, k] *= 1.0 - p_hate
    return out


def load_vlm(tag: str, split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    d = json.loads((PRED / f"vlm_{tag}_{split}.json").read_text())
    assert d["classes"] == list(A2_ACTIVE), f"class order drift: {d['classes']}"
    return d["ids"], np.array(d["p_hateful"]), np.array(d["p_multi"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--target", default="dev_test", choices=("dev_test", "test"))
    ap.add_argument("--floor-mult", type=float, default=1.5,
                    help="rare-class floor as a multiple of the train prior")
    ap.add_argument("--a1-rate", type=float, default=None,
                    help="override the fraction of Hateful predictions on the target "
                         "(the blind test prior is 0.296 vs dev_test's 0.378)")
    args = ap.parse_args()

    # ---- calibration on dev (gold known locally) -------------------------
    dev_gold = load("dev")
    ids_dev, p_hate_dev, p_multi_dev = load_vlm(args.tag, "dev")
    gold_by_id = {r["id"]: r for r in dev_gold}
    y1 = np.array([1 if gold_by_id[i]["label"] == "Hateful" else 0 for i in ids_dev])
    Y2 = np.array([[int(c in (gold_by_id[i]["fine_grained_label"] or [])) for c in A2_ACTIVE]
                   for i in ids_dev])

    thr1, best1 = 0.5, -1.0
    for thr in np.arange(0.10, 0.91, 0.01):
        f = f1_score(y1, (p_hate_dev >= thr).astype(int), average="macro", zero_division=0)
        if f > best1:
            thr1, best1 = float(thr), float(f)
    print(f"A1 dev: thr={thr1:.2f} macro-F1={best1:.4f}")

    gated_dev = gate(p_multi_dev, p_hate_dev)
    thresholds = np.array([best_threshold(Y2[:, k], gated_dev[:, k])[0]
                           for k in range(len(A2_ACTIVE))])
    pred_dev = (gated_dev >= thresholds).astype(int)
    print(f"A2 dev: macro-F1={f1_score(Y2, pred_dev, average='macro', zero_division=0):.4f}")
    per = f1_score(Y2, pred_dev, average=None, zero_division=0)
    for k, cls in enumerate(A2_ACTIVE):
        print(f"   {cls:<16} thr={thresholds[k]:.2f} F1={per[k]:.3f} "
              f"(dev support {int(Y2[:, k].sum())})")

    # ---- decode the target ----------------------------------------------
    ids_t, p_hate_t, p_multi_t = load_vlm(args.tag, args.target)

    a1_hate = p_hate_t >= thr1
    if args.a1_rate is not None:
        # Rank-based override: keep the model's ordering, move the operating point.
        k = int(round(args.a1_rate * len(ids_t)))
        a1_hate = np.zeros(len(ids_t), dtype=bool)
        a1_hate[np.argsort(-p_hate_t)[:k]] = True
        print(f"A1 target: rank-thresholded to {k}/{len(ids_t)} Hateful")
    else:
        print(f"A1 target: {int(a1_hate.sum())}/{len(ids_t)} Hateful at thr={thr1:.2f}")
    a1_pred = {i: ("Hateful" if h else "Not Hateful") for i, h in zip(ids_t, a1_hate)}

    gated_t = gate(p_multi_t, p_hate_t)
    train_prior = np.array(
        [sum(1 for r in load("train") if c in (r["fine_grained_label"] or [])) / 3500
         for c in A2_ACTIVE])
    a2_pred: dict[str, list[str]] = {i: [] for i in ids_t}
    for k, cls in enumerate(A2_ACTIVE):
        chosen = set(np.flatnonzero(gated_t[:, k] >= thresholds[k]).tolist())
        floor = max(2, int(np.ceil(args.floor_mult * train_prior[k] * len(ids_t))))
        if len(chosen) < floor:
            chosen |= set(np.argsort(-gated_t[:, k])[:floor].tolist())
        for i in chosen:
            a2_pred[ids_t[i]].append(cls)
    for idx, rid in enumerate(ids_t):
        if not a2_pred[rid]:
            a2_pred[rid] = [A2_ACTIVE[int(np.argmax(gated_t[idx]))]]

    counts = {c: sum(c in v for v in a2_pred.values()) for c in A2_ACTIVE}
    print(f"A2 target counts: {counts}")

    suffix = "devtest" if args.target == "dev_test" else "test"
    (PRED / f"a1_{suffix}.json").write_text(json.dumps(a1_pred, ensure_ascii=False))
    (PRED / f"a2_{suffix}.json").write_text(json.dumps(a2_pred, ensure_ascii=False))
    (PRED / f"a_thresholds_{args.tag}.json").write_text(json.dumps(
        {"a1": thr1, "a2": dict(zip(A2_ACTIVE, thresholds.tolist()))}, indent=2))
    print(f"wrote a1_{suffix}.json / a2_{suffix}.json; "
          f"next: python -m src.build_submissions --run-id awn_vlm_{args.tag}")


if __name__ == "__main__":
    main()
