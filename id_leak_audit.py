"""ArGuard Task B sequential-id label-leak audit (disclosed 29 Jul 2026, fixed by
the organizers before the test release; see the paper's Section 6).

Reproduces, from the public dev-phase CSVs alone, with zero characters of prompt
text read:
  1. the contiguous per-label id runs that fingerprint per-domain concatenation,
  2. a nearest-known-id lookup fitted on train only scoring macro-F1 1.0000 on dev
     for both the binary and granular tasks,
  3. the randomization control: shuffling ids collapses the same lookup to chance.

    python id_leak_audit.py path/to/train.csv path/to/dev_with_label.csv

Note: the granular lookup's exact score is tie-break-sensitive at the handful of
dev ids equidistant between two known neighbors (0.99929 with the symmetric
tie-break below; 1.00000 with the run-aware tie-break used in the original
audit). The binary lookup is 1.00000 either way.
"""

from __future__ import annotations

import csv
import random
import sys
from bisect import bisect_left
from collections import defaultdict


def read(path: str) -> list[tuple[int, str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return [(int(r["id"]), r["label"], r["granular_label"])
                for r in csv.DictReader(fh) if r["label"].strip()]


def runs(rows: list[tuple[int, str, str]], col: int) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for i, lab, gran in sorted(rows):
        v = (lab, gran)[col - 1]
        if out and out[-1][0] == v:
            out[-1] = (v, out[-1][1], i)
        else:
            out.append((v, i, i))
    return out


def macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    tp: dict = defaultdict(int); fp: dict = defaultdict(int); fn: dict = defaultdict(int)
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1; fn[g] += 1
    f1s = []
    for c in labels:
        pr = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1s) / len(f1s)


def nearest_lookup(train: list[tuple[int, str, str]], ids: list[int], col: int) -> list[str]:
    keys = sorted(i for i, *_ in train)
    val = {i: (lab, gran)[col - 1] for i, lab, gran in train}
    out = []
    for q in ids:
        j = bisect_left(keys, q)
        cands = [keys[k] for k in (j - 1, j) if 0 <= k < len(keys)]
        out.append(val[min(cands, key=lambda k: abs(k - q))])
    return out


def main() -> None:
    train, dev = read(sys.argv[1]), read(sys.argv[2])
    for col, name in ((1, "binary"), (2, "granular")):
        r = runs(train + dev, col)
        print(f"{name}: {len(r)} contiguous id runs over the released rows")
        for v, lo, hi in r[:8]:
            print(f"   {v:<20} ids {lo}..{hi}")
        gold = [(lab, gran)[col - 1] for _, lab, gran in sorted(dev)]
        ids = [i for i, *_ in sorted(dev)]
        pred = nearest_lookup(train, ids, col)
        print(f"   nearest-id lookup (train-only fit) dev macro-F1: {macro_f1(gold, pred):.5f}")
        rng = random.Random(0)
        shuf_ids = [i for i, *_ in train]
        rng.shuffle(shuf_ids)
        shuf_train = [(si, lab, gran) for si, (_, lab, gran) in zip(shuf_ids, train)]
        pred_s = nearest_lookup(shuf_train, ids, col)
        print(f"   shuffled-id control (seed 0)  dev macro-F1: {macro_f1(gold, pred_s):.5f}")


if __name__ == "__main__":
    main()
