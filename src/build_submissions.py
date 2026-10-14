"""Build the four Codabench submission archives and validate them before upload.

Reads the prediction files written by the baselines, packages each one in the exact
shape its competition expects, and re-opens every archive to confirm it holds a
single correctly-named member at the root.

    python -m src.build_submissions --run-id awn_charngram_v1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src import submit

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions"
OUT = ROOT / "submissions"
TASKA_SPLITS = ROOT / "data" / "taskA" / "splits"
TASKB = ROOT / "data" / "taskB"

COMPETITIONS = {
    "a1": ("16909", "prediction.tsv"),
    "a2": ("16910", "prediction.jsonl"),
    "b1": ("16652", "prediction.csv"),
    "b2": ("16653", "prediction.csv"),
}


def devtest_ids(split: str = "dev_test") -> list[str]:
    path = TASKA_SPLITS / f"{split}.jsonl"
    if not path.exists():
        from datasets import load_dataset
        return list(load_dataset("QCRI/ArGuard-Task1", split=split.replace("dev_test", "dev_test"))["id"])
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line)["id"] for line in fh if line.strip()]


def taskb_target_ids() -> list[str]:
    """The dev-phase submission target for Task B is dev_without_label.csv."""
    with (TASKB / "dev_without_label.csv").open(encoding="utf-8", newline="") as fh:
        return [r["id"] for r in csv.DictReader(fh)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="awn_charngram_v1",
                    help="A1 run_id column; must be whitespace-free")
    ap.add_argument("--target", default="devtest", choices=("devtest", "test"),
                    help="which Task A prediction files and id set to package")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    a_split = "dev_test" if args.target == "devtest" else "test"
    a_ids, b_ids = devtest_ids(a_split), taskb_target_ids()
    built: list[tuple[str, Path]] = []

    a1 = json.loads((PRED / f"a1_{args.target}.json").read_text())
    built.append(("a1", submit.build_a1(
        {i: a1[i] for i in a_ids}, OUT / "a1_prediction.zip",
        run_id=args.run_id, expected_ids=a_ids)))

    a2 = json.loads((PRED / f"a2_{args.target}.json").read_text())
    built.append(("a2", submit.build_a2(
        {i: a2[i] for i in a_ids}, OUT / "a2_prediction.zip", expected_ids=a_ids)))

    if args.target == "test":
        # Task B test zips are built by src.taskb_test_submit against the test id
        # spaces; the dev-target files here would silently clobber them.
        print("target=test: skipping B1/B2 (owned by taskb_test_submit)")
        _report(built)
        return

    b1 = json.loads((PRED / "b1_dev.json").read_text())
    built.append(("b1", submit.build_b1(
        {i: b1[i] for i in b_ids}, OUT / "b1_prediction.zip", expected_ids=b_ids)))

    b2 = json.loads((PRED / "b2_dev.json").read_text())
    built.append(("b2", submit.build_b2(
        {i: b2[i] for i in b_ids}, OUT / "b2_prediction.zip", expected_ids=b_ids)))

    _report(built)


def _report(built):
    print("Validated and ready to upload:\n")
    for subtask, path in built:
        comp, member = COMPETITIONS[subtask]
        print(f"  {subtask.upper()}  comp {comp}  {submit.describe(path)}")
        print(f"        {path}")


if __name__ == "__main__":
    main()
