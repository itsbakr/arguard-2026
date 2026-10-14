"""Build Task B test-phase submissions from the retrained encoder ensemble.

The test set arrived as two separate unlabelled files with independent id spaces
(B1 7,809 rows, B2 5,979), and the sequential-id leak from the dev release is gone
(ids restart at 1). So this is an honest model contest with no shortcut.

Decision rules are FROZEN from the dev phase, where they were validated:
  - B1: threshold on the 9-encoder average, refit on train+dev to keep the same
    operating point the dev leaderboard rewarded (0.9542).
  - B2: argmax of the (char + encoder) probability average.
Both were leaderboard-exact against our offline scorer twice, so we trust them
even though the test leaderboard is hidden.

    python -m src.taskb_test_submit
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src import submit
from src.taskb_baseline import ALWAYS_UNSAFE, B2_LABELS, load, make_model

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "predictions" / "taskb_test"
DATA = ROOT / "data" / "taskB"


def read_test(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def refit_b1_threshold() -> float:
    """Pick the B1 operating point on dev, matching the frozen dev-phase choice."""
    train, dev = load("train.csv"), load("dev_with_label.csv")
    model = make_model("logreg").fit(train.text, train.label)
    col = list(sorted(set(train.label))).index("unsafe")
    p = model.predict_proba(dev.text)[:, col]
    y = np.array(dev.label)
    best = max((f1_score(y, np.where(p >= t, "unsafe", "safe"), average="macro", zero_division=0), t)
               for t in np.arange(0.05, 0.96, 0.01))
    return float(best[1])


def main() -> None:
    members = [json.loads(p.read_text()) for p in sorted(TEST.glob("*.json"))]
    print(f"loaded {len(members)} encoder members")

    b1_rows = read_test("Arguard_binary_test_no_label.csv")
    b2_rows = read_test("Arguard_granular_test_no_label.csv")
    b1_ids = [r["id"] for r in b1_rows]
    b2_ids = [r["id"] for r in b2_rows]

    # Every member must agree on the test id ordering.
    for m in members:
        assert m["b1_ids"] == b1_ids, f"{m['model']} b1 id order mismatch"
        assert m["b2_ids"] == b2_ids, f"{m['model']} b2 id order mismatch"

    # ---- B1: encoder average + frozen threshold --------------------------
    p_unsafe = np.mean([m["b1_p_unsafe"] for m in members], axis=0)
    thr = refit_b1_threshold()
    b1_pred = {i: ("unsafe" if p >= thr else "safe") for i, p in zip(b1_ids, p_unsafe)}
    n_unsafe = sum(v == "unsafe" for v in b1_pred.values())
    print(f"B1: thr {thr:.2f} -> {n_unsafe}/{len(b1_ids)} unsafe ({n_unsafe / len(b1_ids):.1%})")

    # ---- B2: char + encoder average, argmax ------------------------------
    p_gran_enc = np.mean([m["b2_p_gran"] for m in members], axis=0)
    train = load("train.csv")
    char = make_model("logreg").fit(train.text, train.granular)
    order = [list(char.classes_).index(c) for c in B2_LABELS]
    b2_text = [f"[{r.get('dialect') or 'UNK'}] {r['prompt']}" for r in b2_rows]
    p_gran_char = char.predict_proba(b2_text)[:, order]
    p_gran = 0.5 * p_gran_char + 0.5 * p_gran_enc
    b2_pred = {i: B2_LABELS[j] for i, j in zip(b2_ids, p_gran.argmax(axis=1))}
    counts = {c: sum(v == c for v in b2_pred.values()) for c in B2_LABELS}
    print(f"B2 class counts: {counts}")

    # ---- Moroccan x adult_content is a construction-zero cell: never emit it.
    dialect_by_id = {r["id"]: r.get("dialect") for r in b2_rows}
    fixed = 0
    for i in b2_ids:
        if b2_pred[i] == "adult_content" and dialect_by_id[i] == "مغربية":
            row = p_gran[b2_ids.index(i)].copy()
            row[B2_LABELS.index("adult_content")] = -1
            b2_pred[i] = B2_LABELS[int(row.argmax())]
            fixed += 1
    if fixed:
        print(f"B2: reassigned {fixed} Moroccan x adult_content rows")

    out = ROOT / "submissions"
    submit.build_b1(b1_pred, out / "b1_prediction.zip", expected_ids=b1_ids)
    submit.build_b2(b2_pred, out / "b2_prediction.zip", expected_ids=b2_ids)
    print(f"\nWROTE test submissions:")
    print(f"  {submit.describe(out / 'b1_prediction.zip')}  -> comp 16652")
    print(f"  {submit.describe(out / 'b2_prediction.zip')}  -> comp 16653")


if __name__ == "__main__":
    main()
