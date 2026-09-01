"""Build and validate Codabench submission archives for all four ArGuard subtasks.

Every subtask wants a zip named ``prediction.zip`` with exactly one file **at the
archive root**. The three formats differ, and for Task B the Codabench *Terms* tab
is authoritative over the repo README (which describes a TSV the platform rejects):

    A1  prediction.tsv    header ``id<TAB>label<TAB>run_id``
    A2  prediction.jsonl  one ``{"id": ..., "labels": [...]}`` per line
    B1  prediction.csv    columns ``id,prediction``  in {safe, unsafe}
    B2  prediction.csv    columns ``id,prediction``  in the 7 granular labels

We write archives with :mod:`zipfile` rather than shelling out to ``zip``. On macOS
``zip -r`` injects ``__MACOSX/`` entries and sweeps up ``.DS_Store``, and both break
Codabench ingestion -- the most common submission failure for Mac users.

Each ``build_*`` validates before writing and raises :class:`FormatError`, so a
malformed submission fails here rather than silently scoring zero.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

A1_LABELS: frozenset[str] = frozenset({"Hateful", "Not Hateful"})

#: The 10 classes the A2 scorer actually averages over.
A2_ACTIVE: tuple[str, ...] = (
    "Contempt", "Dehumanization", "Exclusion", "Humor", "Incitement",
    "Inferiority", "Mocking", "Other", "Sarcasm", "Slurs",
)
#: Accepted by the format checker but ignored by the scorer (zero support).
A2_ZERO_SUPPORT: tuple[str, ...] = ("Extremism", "Historical", "Insults",
                                    "Stereotyping", "Threat")
A2_TAXONOMY: frozenset[str] = frozenset(A2_ACTIVE) | frozenset(A2_ZERO_SUPPORT)

B1_LABELS: frozenset[str] = frozenset({"safe", "unsafe"})
B2_LABELS: frozenset[str] = frozenset({
    "adult_content", "harm-to-others", "self-harm", "harassment",
    "FraudandDeception", "Bully", "hate_speech",
})


class FormatError(ValueError):
    """Raised when a submission would be rejected by the official checker."""


def _check_ids(pred_ids: Sequence[str], expected_ids: Iterable[str] | None) -> None:
    if len(pred_ids) != len(set(pred_ids)):
        dupes = {i for i in pred_ids if pred_ids.count(i) > 1}
        raise FormatError(f"duplicate ids: {sorted(dupes)[:5]}")
    if any(not str(i).strip() for i in pred_ids):
        raise FormatError("empty id present")
    if expected_ids is None:
        return
    expected = {str(i) for i in expected_ids}
    got = {str(i) for i in pred_ids}
    if got != expected:
        missing, extra = expected - got, got - expected
        raise FormatError(
            f"id set mismatch: missing={len(missing)} (e.g. {sorted(missing)[:3]}), "
            f"extra={len(extra)} (e.g. {sorted(extra)[:3]})"
        )


def _write_zip(out_zip: Path, member_name: str, payload: str) -> Path:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member_name, payload.encode("utf-8"))
    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
    if names != [member_name]:
        raise FormatError(
            f"{out_zip.name} must contain exactly ['{member_name}'] at the root, got {names}")
    return out_zip


def build_a1(predictions: Mapping[str, str], out_zip: Path, run_id: str,
             expected_ids: Iterable[str] | None = None) -> Path:
    """Subtask A1 -- TSV with header ``id<TAB>label<TAB>run_id``."""
    if not run_id or any(c.isspace() for c in run_id):
        raise FormatError(f"run_id must be a whitespace-free token, got {run_id!r}")
    bad = {v for v in predictions.values() if v not in A1_LABELS}
    if bad:
        raise FormatError(f"invalid A1 labels {bad}; expected {sorted(A1_LABELS)}")
    _check_ids(list(predictions), expected_ids)

    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n")
    w.writerow(("id", "label", "run_id"))
    for rid, label in predictions.items():
        w.writerow((rid, label, run_id))
    return _write_zip(out_zip, "prediction.tsv", buf.getvalue())


def build_a2(predictions: Mapping[str, Sequence[str]], out_zip: Path,
             expected_ids: Iterable[str] | None = None) -> Path:
    """Subtask A2 -- JSONL, one ``{"id": ..., "labels": [...]}`` per line.

    An empty ``labels`` list is legal, but the scorer averages over all 10 active
    classes with ``zero_division=0``: a class never predicted scores 0.
    """
    bad = {lab for labs in predictions.values() for lab in labs if lab not in A2_TAXONOMY}
    if bad:
        raise FormatError(f"labels outside the 15-class taxonomy: {sorted(bad)}")
    _check_ids(list(predictions), expected_ids)

    lines = [json.dumps({"id": rid, "labels": list(labels)}, ensure_ascii=False)
             for rid, labels in predictions.items()]
    return _write_zip(out_zip, "prediction.jsonl", "\n".join(lines) + "\n")


def _build_task_b(predictions: Mapping[str, str], out_zip: Path,
                  allowed: frozenset[str], expected_ids: Iterable[str] | None) -> Path:
    bad = {v for v in predictions.values() if v not in allowed}
    if bad:
        raise FormatError(f"invalid labels {sorted(bad)}; expected {sorted(allowed)}")
    _check_ids(list(predictions), expected_ids)

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(("id", "prediction"))
    for rid, label in predictions.items():
        w.writerow((rid, label))
    return _write_zip(out_zip, "prediction.csv", buf.getvalue())


def build_b1(predictions, out_zip: Path, expected_ids=None) -> Path:
    """Subtask B1 -- CSV ``id,prediction`` in {safe, unsafe}."""
    return _build_task_b(predictions, out_zip, B1_LABELS, expected_ids)


def build_b2(predictions, out_zip: Path, expected_ids=None) -> Path:
    """Subtask B2 -- CSV ``id,prediction`` over the 7 granular labels.

    Predict for **every** row, including ones believed safe: the released data
    carries a ``granular_label`` on safe rows too, and the submission must cover
    the full id set regardless.
    """
    return _build_task_b(predictions, out_zip, B2_LABELS, expected_ids)


def describe(path: Path) -> str:
    """One-line summary of a built archive, for logging before upload."""
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name).decode("utf-8")
    return f"{path.name}: {name}, {raw.count(chr(10))} lines, {path.stat().st_size:,} bytes"
