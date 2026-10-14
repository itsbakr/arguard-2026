"""Offline scorer for ArGuard Task B -- the organizers shipped only a stub README.

Mirrors the metric semantics of the official Task A scorer (``taskA/baselines/metrics.py``):
macro-F1 as the ranking metric, computed with sklearn's default ``zero_division=0``,
plus accuracy and macro/weighted precision-recall as diagnostics.

Because no official Task B scorer exists, treat every number this produces as *our*
estimate and calibrate it against the hidden dev_test leaderboard before trusting it.

    python -m src.scorer_b --subtask b1 \
        --gold data/taskB/dev_with_label.csv --predictions predictions/b1_dev.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

GOLD_FIELD = {"b1": "label", "b2": "granular_label"}


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_gold(path: Path, subtask: str) -> dict[str, str]:
    """Read the labelled dev file. Rows with an empty gold label are skipped.

    Two train rows (ids 13250, 22656) ship with an empty ``label``; the same
    defensive skip is applied here in case the test release repeats it.
    """
    field = GOLD_FIELD[subtask]
    return {r["id"]: r[field] for r in read_csv(path) if r.get(field)}


def load_predictions(path: Path) -> dict[str, str]:
    """Read a submission-shape CSV with columns ``id,prediction``."""
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"{path}: empty prediction file")
    missing = {"id", "prediction"} - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}; got {sorted(rows[0])}")
    return {r["id"]: r["prediction"] for r in rows}


def score(gold: dict[str, str], pred: dict[str, str]) -> dict[str, float]:
    """Macro-F1 (official) plus diagnostics, over the intersection of ids.

    The id sets must match; a mismatch is an automatic rejection server side, so
    we surface it here rather than quietly scoring a subset.
    """
    if set(pred) != set(gold):
        miss, extra = set(gold) - set(pred), set(pred) - set(gold)
        raise ValueError(
            f"id set mismatch: missing={len(miss)} (e.g. {sorted(miss)[:3]}), "
            f"extra={len(extra)} (e.g. {sorted(extra)[:3]})"
        )
    ids = sorted(gold)
    y_true = [gold[i] for i in ids]
    y_pred = [pred[i] for i in ids]
    labels = sorted(set(y_true) | set(y_pred))

    metrics = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n": len(ids),
    }
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    support = Counter(y_true)
    for name, value in zip(labels, per_class):
        metrics[f"f1__{name}"] = float(value)
        metrics[f"support__{name}"] = support[name]
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subtask", choices=("b1", "b2"), required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    metrics = score(load_gold(args.gold, args.subtask), load_predictions(args.predictions))

    width = max(len(k) for k in metrics)
    print(f"── subtask {args.subtask} ── {args.predictions.name}")
    for key, value in metrics.items():
        print(f"   {key:<{width}}  {value:.4f}" if isinstance(value, float) else
              f"   {key:<{width}}  {value}")

    out = args.out or args.predictions.with_suffix(args.predictions.suffix + ".metrics.json")
    out.write_text(json.dumps({"subtask": args.subtask, "metrics": metrics}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
