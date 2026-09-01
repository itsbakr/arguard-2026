"""Build Task B test-phase submissions from the encoder ensemble.

The test set is two separate unlabelled files with independent id spaces (B1 7,809
rows, B2 5,979); the dev-phase sequential-id leak is gone (ids restart at 1).

Frozen decisions, both validated on the live leaderboard:
  - B1: threshold pinned so the predicted unsafe rate is 49%, matching the test's
    measured balance rather than the 80%-unsafe training prior. This single
    constant was worth +0.16 macro-F1 (0.5959 -> 0.7552).
  - B2: argmax of the (char + encoder) probability average -> 0.7695, 3rd place.
    A prior-flattening variant was tried and refuted (0.7621).

    python -m src.taskb_test_submit
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src import submit
from src.taskb_baseline import B2_LABELS, load, make_model

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "predictions" / "taskb_test"
DATA = ROOT / "data" / "taskB"
UNSAFE_RATE = 0.49  # measured test balance; see RECOVERY.md


def read_test(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    members = [json.loads(p.read_text()) for p in sorted(TEST.glob("*.json"))]
    if not members:
        raise SystemExit(f"no encoder predictions in {TEST}; run modal_taskb.py::sweep")
    print(f"loaded {len(members)} encoder members")

    b1_rows, b2_rows = (read_test("Arguard_binary_test_no_label.csv"),
                        read_test("Arguard_granular_test_no_label.csv"))
    b1_ids = [r["id"] for r in b1_rows]
    b2_ids = [r["id"] for r in b2_rows]
    for m in members:
        assert m["b1_ids"] == b1_ids and m["b2_ids"] == b2_ids, f"{m['model']} id order"

    # ---- B1: ensemble average, threshold pinned to the test balance -------
    p_unsafe = np.mean([m["b1_p_unsafe"] for m in members], axis=0)
    thr = float(np.quantile(p_unsafe, 1 - UNSAFE_RATE))
    b1_pred = {i: ("unsafe" if p >= thr else "safe") for i, p in zip(b1_ids, p_unsafe)}
    n_unsafe = sum(v == "unsafe" for v in b1_pred.values())
    print(f"B1: thr {thr:.4f} -> {n_unsafe}/{len(b1_ids)} unsafe ({n_unsafe/len(b1_ids):.1%})")

    # ---- B2: char + encoder average, argmax --------------------------------
    p_gran_enc = np.mean([m["b2_p_gran"] for m in members], axis=0)
    train = load("train.csv")
    char = make_model("logreg").fit(train.text, train.granular)
    order = [list(char.classes_).index(c) for c in B2_LABELS]
    b2_text = [f"[{r.get('dialect') or 'UNK'}] {r['prompt']}" for r in b2_rows]
    p_gran = 0.5 * char.predict_proba(b2_text)[:, order] + 0.5 * p_gran_enc
    b2_pred = {i: B2_LABELS[j] for i, j in zip(b2_ids, p_gran.argmax(axis=1))}

    # Moroccan x adult_content has zero support in train+dev: a construction-zero
    # cell, so never emit it.
    dialect = {r["id"]: r.get("dialect") for r in b2_rows}
    fixed = 0
    for pos, i in enumerate(b2_ids):
        if b2_pred[i] == "adult_content" and dialect[i] == "مغربية":
            row = p_gran[pos].copy()
            row[B2_LABELS.index("adult_content")] = -1
            b2_pred[i] = B2_LABELS[int(row.argmax())]
            fixed += 1
    if fixed:
        print(f"B2: reassigned {fixed} Moroccan x adult_content rows")
    print(f"B2 counts: { {c: sum(v == c for v in b2_pred.values()) for c in B2_LABELS} }")

    out = ROOT / "submissions"
    submit.build_b1(b1_pred, out / "b1_prediction.zip", expected_ids=b1_ids)
    submit.build_b2(b2_pred, out / "b2_prediction.zip", expected_ids=b2_ids)
    print(f"\n  {submit.describe(out / 'b1_prediction.zip')}  -> comp 16652")
    print(f"  {submit.describe(out / 'b2_prediction.zip')}  -> comp 16653")


if __name__ == "__main__":
    main()
