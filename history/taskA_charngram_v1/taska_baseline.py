"""Task A text-only baseline over the OCR field, for A1 and A2.

This is *deliberately* the weak system. On this exact meme lineage the image carries
far more signal than the OCR text (image-only ~70 vs text-only ~57 macro-F1 in
NYUAD's MAHED 2025 results), and the ArGuard OCR field is visibly garbled. The real
system is a LoRA fine-tune of a VLM; this exists to

  1. produce a *valid* submission on every subtask while the dev phase is still open,
     so a format bug surfaces now rather than on the final day, and
  2. measure the offset between our offline dev estimate and the hidden dev_test
     leaderboard score.

Two pieces of structure here do carry over to the real system:

*   **A1-gated masking.** The A2 label space is deterministically partitioned by the
    A1 label -- Hateful memes only ever carry {Mocking, Incitement, Slurs,
    Dehumanization, Contempt, Inferiority, Exclusion, Other}, Not-Hateful only
    {Sarcasm, Humor, Other}. We gate on the A1 *probability* rather than the hard
    label so A1 errors do not propagate as hard zeros.
*   **Per-class thresholds.** A2 macro-F1 averages 10 independent binary columns, so
    a false positive on Exclusion costs nothing on Mocking. A single global 0.5
    threshold abandons the rare classes -- which is very likely why the public
    leaderboard tops out at 0.3668.

    python -m src.taska_baseline
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "taskA" / "splits"

A2_ACTIVE: tuple[str, ...] = (
    "Contempt", "Dehumanization", "Exclusion", "Humor", "Incitement",
    "Inferiority", "Mocking", "Other", "Sarcasm", "Slurs",
)
#: Fine-grained classes that only ever appear on a Hateful meme.
HATEFUL_ONLY = frozenset({"Contempt", "Dehumanization", "Exclusion", "Incitement",
                          "Inferiority", "Mocking", "Slurs"})
#: ...and on a Not-Hateful meme. ``Other`` is the only label shared by both.
NONHATEFUL_ONLY = frozenset({"Humor", "Sarcasm"})


def load(split: str) -> list[dict]:
    with (SPLITS / f"{split}.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prep(record: dict) -> str:
    return re.sub(r"\s+", " ", (record.get("text") or "").strip()) or "[EMPTY]"


def make_model():
    """Char n-grams survive OCR garbling better than word tokens do."""
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                        sublinear_tf=True, max_features=300_000),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
    )


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Per-class F1-optimal threshold. Never returns a threshold that predicts nothing.

    For a class with a handful of positives the F1-optimal operating point sits far
    below 0.5, because in multi-label macro-F1 the extra false positives are free --
    they cannot damage any other class's column.
    """
    best = (0.5, -1.0)
    for thr in np.arange(0.02, 0.96, 0.01):
        pred = (scores >= thr).astype(int)
        if pred.sum() == 0:
            continue
        score = f1_score(y_true, pred, zero_division=0)
        if score > best[1]:
            best = (float(thr), float(score))
    return best


def main() -> None:
    train, dev, dev_test = load("train"), load("dev"), load("dev_test")
    # Pool train+dev for threshold fitting: Exclusion has 10 positives in train
    # and 4 in dev, so every positive counts.
    pool = train + dev
    X_pool = [prep(r) for r in pool]
    X_devtest = [prep(r) for r in dev_test]
    print(f"train {len(train)} · dev {len(dev)} · dev_test {len(dev_test)}")

    # ---- A1 -------------------------------------------------------------
    y1 = np.array([r["label"] for r in pool])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof1 = cross_val_predict(make_model(), X_pool, y1, cv=cv,
                             method="predict_proba", n_jobs=-1)
    hateful_col = list(sorted(set(y1))).index("Hateful")
    p_hateful_oof = oof1[:, hateful_col]

    thr1, f1_tuned = 0.5, -1.0
    for thr in np.arange(0.15, 0.86, 0.01):
        pred = np.where(p_hateful_oof >= thr, "Hateful", "Not Hateful")
        score = f1_score(y1, pred, average="macro", zero_division=0)
        if score > f1_tuned:
            thr1, f1_tuned = float(thr), float(score)
    f1_argmax = f1_score(y1, np.where(p_hateful_oof >= 0.5, "Hateful", "Not Hateful"),
                         average="macro", zero_division=0)
    print(f"\nA1  out-of-fold (train+dev): argmax {f1_argmax:.4f}  ->  "
          f"thr={thr1:.2f} {f1_tuned:.4f}")

    model1 = make_model().fit(X_pool, y1)
    p_hateful = model1.predict_proba(X_devtest)[:, hateful_col]
    a1_pred = np.where(p_hateful >= thr1, "Hateful", "Not Hateful")
    n_hateful = int((a1_pred == "Hateful").sum())
    print(f"A1  dev_test: predicting {n_hateful}/500 Hateful ({n_hateful / 5:.1f}%)"
          f"   [dev_test gold prior is 37.8%; the blind test prior is 29.6%]")

    # ---- A2 -------------------------------------------------------------
    Y2 = np.array([[int(c in (r["fine_grained_label"] or [])) for c in A2_ACTIVE] for r in pool])
    thresholds: dict[str, float] = {}
    oof_scores = np.zeros((len(pool), len(A2_ACTIVE)))
    print("\nA2  per-class out-of-fold:")
    for k, cls in enumerate(A2_ACTIVE):
        col = Y2[:, k]
        if col.sum() < 5:      # too few positives to cross-validate honestly
            oof = cross_val_predict(make_model(), X_pool, col,
                                    cv=StratifiedKFold(3, shuffle=True, random_state=42),
                                    method="predict_proba", n_jobs=-1)[:, 1]
        else:
            oof = cross_val_predict(make_model(), X_pool, col, cv=cv,
                                    method="predict_proba", n_jobs=-1)[:, 1]
        oof_scores[:, k] = oof
        thr, f1c = best_threshold(col, oof)
        thresholds[cls] = thr
        f1_at_half = f1_score(col, (oof >= 0.5).astype(int), zero_division=0)
        print(f"   {cls:<16} n={col.sum():<5} thr={thr:.2f}  F1 {f1_at_half:.3f} -> {f1c:.3f}")

    macro_half = f1_score(Y2, (oof_scores >= 0.5).astype(int), average="macro", zero_division=0)
    tuned = np.zeros_like(Y2)
    for k, cls in enumerate(A2_ACTIVE):
        tuned[:, k] = (oof_scores[:, k] >= thresholds[cls]).astype(int)
    macro_tuned = f1_score(Y2, tuned, average="macro", zero_division=0)
    print(f"\nA2  out-of-fold macro-F1: global 0.5 = {macro_half:.4f}  ->  "
          f"per-class thresholds = {macro_tuned:.4f}   (+{macro_tuned - macro_half:.4f})")

    models2 = [make_model().fit(X_pool, Y2[:, k]) for k in range(len(A2_ACTIVE))]
    scores2 = np.column_stack([m.predict_proba(X_devtest)[:, 1] for m in models2])

    # A1-gated masking, soft: scale each class by the probability of the half it lives in.
    gate = np.ones_like(scores2)
    for k, cls in enumerate(A2_ACTIVE):
        if cls in HATEFUL_ONLY:
            gate[:, k] = p_hateful
        elif cls in NONHATEFUL_ONLY:
            gate[:, k] = 1.0 - p_hateful
    gated = scores2 * gate

    a2_pred: dict[str, list[str]] = {}
    for i, record in enumerate(dev_test):
        labels = [c for k, c in enumerate(A2_ACTIVE) if gated[i, k] >= thresholds[c]]
        if not labels:  # never emit an empty set -- fall back to the argmax class
            labels = [A2_ACTIVE[int(np.argmax(gated[i]))]]
        a2_pred[record["id"]] = labels
    counts = {c: sum(c in v for v in a2_pred.values()) for c in A2_ACTIVE}
    print(f"A2  dev_test label counts: {counts}")

    out = ROOT / "predictions"
    out.mkdir(exist_ok=True)
    (out / "a1_devtest.json").write_text(json.dumps(
        {r["id"]: a1_pred[i] for i, r in enumerate(dev_test)}, ensure_ascii=False))
    (out / "a2_devtest.json").write_text(json.dumps(a2_pred, ensure_ascii=False))
    (out / "a_thresholds.json").write_text(json.dumps(
        {"a1": thr1, "a2": thresholds}, indent=2))
    print(f"\nwrote predictions to {out}")


if __name__ == "__main__":
    main()
