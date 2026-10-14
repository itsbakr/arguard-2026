"""Minimal reproduction: ArGuard 2026 Task B labels are recoverable from the row `id`.

Self-contained -- Python 3 stdlib only, no dependencies. Reads only the public
files in araieval/ArGuard-2026-tasks. Runs in ~10 seconds.

    python3 repro_id_leak.py

Summary of what it shows:
  1. Sorting the 19,238 released rows by `id` yields 7 perfectly contiguous,
     non-interleaved runs of `granular_label` (and 7 of `label`).
  2. A nearest-known-id lookup fitted on train.csv ONLY, which never reads the
     prompt text, scores macro-F1 1.00000 on dev for both B1 and B2.
  3. Randomly reassigning the ids among the same rows collapses that to ~0.51
     (B1) and ~0.15 (B2) -- confirming the signal is the id ordering itself,
     not anything learnable from the rows.
"""

import bisect
import csv
import io
import random
import urllib.request

BASE = "https://raw.githubusercontent.com/araieval/ArGuard-2026-tasks/main/taskB/data/"


def load(name):
    raw = urllib.request.urlopen(BASE + name).read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(raw)))


def macro_f1(gold, pred):
    classes = sorted(set(gold) | set(pred))
    total = 0.0
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall:
            total += 2 * precision * recall / (precision + recall)
    return total / len(classes)


def contiguous_runs(rows, field):
    """Collapse rows (sorted by id) into maximal runs sharing the same value."""
    runs, prev = [], object()
    for r in sorted(rows, key=lambda r: int(r["id"])):
        value = r[field]
        if value != prev:
            runs.append([value, int(r["id"]), int(r["id"])])
            prev = value
        else:
            runs[-1][2] = int(r["id"])
    return runs


def fit_nearest_id_lookup(rows):
    """The entire 'model': remember (id -> labels) for the training rows."""
    table = sorted((int(r["id"]), r["label"], r["granular_label"]) for r in rows)
    keys = [t[0] for t in table]

    def predict(i):
        pos = bisect.bisect_left(keys, i)
        candidates = ([table[pos]] if pos < len(keys) else []) + ([table[pos - 1]] if pos else [])
        return min(candidates, key=lambda t: abs(t[0] - i))

    return predict


def main():
    train = [r for r in load("train.csv") if r["label"]]  # 2 rows have an empty label
    dev = load("dev_with_label.csv")
    released = train + dev
    print(f"released rows: {len(train)} train (2 empty-label rows dropped) + {len(dev)} dev\n")

    # -- 1. Contiguity ------------------------------------------------------
    for field in ("granular_label", "label"):
        runs = contiguous_runs(released, field)
        print(f"{field}: {len(runs)} contiguous id runs")
        for value, lo, hi in runs:
            print(f"    {value:<18} {lo:>6} .. {hi:>6}   ({hi - lo + 1:>5} ids)")
        print()

    # -- 2. Nearest-known-id lookup, fitted on train only -------------------
    predict = fit_nearest_id_lookup(train)
    print("Nearest-known-id lookup, fitted on train.csv only, scored on dev.")
    print("This model never reads the prompt text.")
    for name, idx, field in (("B1  safe/unsafe", 1, "label"),
                             ("B2  granular_label", 2, "granular_label")):
        gold = [r[field] for r in dev]
        pred = [predict(int(r["id"]))[idx] for r in dev]
        hits = sum(g == p for g, p in zip(gold, pred))
        print(f"    {name:<20} macro-F1 = {macro_f1(gold, pred):.5f}   exact = {hits}/{len(dev)}")

    # -- 3. Randomization control -------------------------------------------
    print("\nControl: reassign the ids randomly among the same rows, refit, rescore.")
    for seed in range(3):
        rng = random.Random(seed)
        shuffled = [int(r["id"]) for r in released]
        rng.shuffle(shuffled)
        remap = {id(r): new for r, new in zip(released, shuffled)}
        table = sorted((remap[id(r)], r["label"], r["granular_label"]) for r in train)
        keys = [t[0] for t in table]

        def predict_shuffled(i, table=table, keys=keys):
            pos = bisect.bisect_left(keys, i)
            cand = ([table[pos]] if pos < len(keys) else []) + ([table[pos - 1]] if pos else [])
            return min(cand, key=lambda t: abs(t[0] - i))

        b1 = macro_f1([r["label"] for r in dev],
                      [predict_shuffled(remap[id(r)])[1] for r in dev])
        b2 = macro_f1([r["granular_label"] for r in dev],
                      [predict_shuffled(remap[id(r)])[2] for r in dev])
        print(f"    seed {seed}:  B1 macro-F1 = {b1:.4f}   B2 macro-F1 = {b2:.4f}")

    # -- 4. Implication for the unreleased test split ------------------------
    known = {int(r["id"]) for r in released}
    unseen = [i for i in range(0, 24051) if i not in known]
    boundaries = [b for _, lo, hi in contiguous_runs(released, "granular_label")
                  for b in (lo, hi)]
    boundaries += [b for _, lo, hi in contiguous_runs(released, "label") for b in (lo, hi)]
    ambiguous = [i for i in unseen if any(abs(i - b) <= 1 for b in boundaries)]
    resolved = len(unseen) - len(ambiguous)
    print(f"\nUnseen ids in [0, 24050]: {len(unseen)}. "
          f"Adjacent to a run boundary: {len(ambiguous)}.")
    print(f"    The same rule resolves {resolved}/{len(unseen)} "
          f"= {100 * resolved / len(unseen):.2f}% of them unambiguously.")
    print("\nNote: taskB/README.md cites 25,071 prompts overall, but only "
          f"{len(known)} released + {len(unseen)} in-range gaps = {len(known) + len(unseen)} "
          "ids are accounted for,\nso roughly 1,020 test ids may lie beyond id 24050 and are "
          "outside the range this rule covers.")


if __name__ == "__main__":
    main()
