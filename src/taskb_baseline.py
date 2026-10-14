"""Task B baseline: character n-gram TF-IDF + linear classifier, for B1 and B2.

Deliberately the cheapest thing that works. Character n-grams sidestep Arabic
morphology and dialectal orthography entirely, train in seconds on CPU, and give a
strong, artifact-robust ensemble member for later. This is the floor we measure
everything else against, and it produces the first real leaderboard submission.

Two decisions that matter more than the model:

*   **B2 covers every row.** The released data carries a ``granular_label`` on safe
    rows too, so B2 is a 7-way harm-*domain* problem over all prompts, not a
    conditional one over unsafe prompts. The task website says otherwise; the data
    and the submission id set both say all rows.
*   **B1 does not use argmax.** Under the 80/20 unsafe/safe skew, argmax collapses
    the minority ``safe`` class and macro-F1 suffers. We sweep the decision
    threshold on out-of-fold predictions and pick the macro-F1 optimum.

    python -m src.taskb_baseline --fit --report
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "taskB"

B2_LABELS = (
    "adult_content", "harm-to-others", "self-harm", "harassment",
    "FraudandDeception", "Bully", "hate_speech",
)
#: Granular categories that never occur on a safe row -- used as a consistency check.
ALWAYS_UNSAFE = frozenset({"Bully", "FraudandDeception", "hate_speech"})


@dataclass
class Split:
    ids: list[str]
    text: list[str]
    dialect: list[str]
    label: list[str]
    granular: list[str]


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load(name: str, labelled: bool = True) -> Split:
    """Load a Task B CSV, dropping rows whose gold label is blank.

    Two train rows (ids 13250, 22656) have an empty ``label``, and one of those has
    an empty ``prompt`` as well. Both sit exactly on a label-run boundary.
    """
    rows = _read(DATA / name)
    if labelled:
        rows = [r for r in rows if r.get("label") and r.get("granular_label")]
    return Split(
        ids=[r["id"] for r in rows],
        text=[_prep(r["prompt"], r.get("dialect", "")) for r in rows],
        dialect=[r.get("dialect") or "UNK" for r in rows],
        label=[r.get("label", "") for r in rows],
        granular=[r.get("granular_label", "") for r in rows],
    )


def _prep(prompt: str, dialect: str) -> str:
    """Normalise whitespace and prepend the dialect as a literal token.

    ``dialect`` is supplied at test time, so it is a legitimate input feature. It is
    a calibrated prior rather than a shortcut -- a dialect-only argmax classifier
    scores at the majority baseline -- but the model can still condition on it.
    """
    prompt = re.sub(r"\s+", " ", (prompt or "").strip())
    return f"[{dialect or 'UNK'}] {prompt}"


def make_model(kind: str):
    """Char 3-5 gram TF-IDF into a linear classifier.

    ``svc`` is the stronger point predictor; ``logreg`` gives calibrated
    probabilities, which is what threshold tuning and later ensembling need.
    """
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        sublinear_tf=True, max_features=400_000,
    )
    if kind == "svc":
        clf = CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)
    else:
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    return make_pipeline(vec, clf)


def tune_binary_threshold(y_true: list[str], proba_unsafe: np.ndarray) -> tuple[float, float]:
    """Sweep the ``unsafe`` decision threshold for the macro-F1 optimum.

    Returns ``(threshold, macro_f1)``. Under an 80/20 skew this lands well away
    from 0.5, and the gain over argmax is typically larger than the gain from a
    better model.
    """
    best = (0.5, -1.0)
    for thr in np.arange(0.05, 0.96, 0.01):
        pred = np.where(proba_unsafe >= thr, "unsafe", "safe")
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        if score > best[1]:
            best = (float(thr), float(score))
    return best


def run(report: bool = True) -> dict:
    train, dev = load("train.csv"), load("dev_with_label.csv")
    print(f"train {len(train.ids)} · dev {len(dev.ids)}")
    results: dict[str, object] = {}

    # ---- B1: probabilities out-of-fold, then a tuned threshold ----------
    model = make_model("logreg")
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, train.text, train.label, cv=cv, method="predict_proba", n_jobs=-1)
    unsafe_col = list(sorted(set(train.label))).index("unsafe")
    thr, oof_f1 = tune_binary_threshold(train.label, oof[:, unsafe_col])
    print(f"\nB1  out-of-fold: argmax macro-F1 = "
          f"{f1_score(train.label, np.where(oof[:, unsafe_col] >= 0.5, 'unsafe', 'safe'), average='macro', zero_division=0):.4f}"
          f"  ->  tuned thr={thr:.2f} macro-F1 = {oof_f1:.4f}")

    model.fit(train.text, train.label)
    dev_proba = model.predict_proba(dev.text)[:, unsafe_col]
    dev_pred_b1 = np.where(dev_proba >= thr, "unsafe", "safe")
    b1_dev = f1_score(dev.label, dev_pred_b1, average="macro", zero_division=0)
    b1_argmax = f1_score(dev.label, np.where(dev_proba >= 0.5, "unsafe", "safe"),
                         average="macro", zero_division=0)
    print(f"B1  dev: argmax {b1_argmax:.4f}  ->  tuned {b1_dev:.4f}   (+{b1_dev - b1_argmax:.4f})")
    results["b1"] = {"threshold": thr, "dev_macro_f1": b1_dev, "argmax_macro_f1": b1_argmax}

    # ---- B2: plain 7-way over every row ---------------------------------
    model_b2 = make_model("svc")
    model_b2.fit(train.text, train.granular)
    dev_pred_b2 = model_b2.predict(dev.text)
    b2_dev = f1_score(dev.granular, dev_pred_b2, average="macro", zero_division=0)
    print(f"B2  dev: macro-F1 = {b2_dev:.4f}")
    results["b2"] = {"dev_macro_f1": b2_dev}

    if report:
        print("\nB2 per-class F1 on dev:")
        per = f1_score(dev.granular, dev_pred_b2, average=None, labels=list(B2_LABELS), zero_division=0)
        for name, value in sorted(zip(B2_LABELS, per), key=lambda t: -t[1]):
            support = sum(g == name for g in dev.granular)
            print(f"   {name:<20} {value:.4f}   (support {support})")

        # Consistency: three granular classes never co-occur with `safe` in train.
        clash = sum(1 for b1, b2 in zip(dev_pred_b1, dev_pred_b2)
                    if b1 == "safe" and b2 in ALWAYS_UNSAFE)
        print(f"\nconsistency: {clash} dev rows predicted safe but assigned an always-unsafe "
              f"category ({clash / len(dev.ids):.1%})")

    results["_predictions"] = {
        "b1": dict(zip(dev.ids, dev_pred_b1.tolist())),
        "b2": dict(zip(dev.ids, dev_pred_b2.tolist())),
    }
    results["_models"] = {"b1": model, "b2": model_b2, "b1_threshold": thr,
                          "b1_unsafe_col": unsafe_col}
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    run(report=args.report)
