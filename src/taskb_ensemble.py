"""Ensemble the 9 fine-tuned encoders with the char-ngram baseline for Task B.

All members were trained on train.csv only, so their dev probabilities are honest
held-out scores. The dev-phase leaderboard scores dev itself, which makes
tune-on-dev the correct move for this phase; for the test phase everything gets
refit on train+dev.

    python -m src.taskb_ensemble
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src.taskb_baseline import ALWAYS_UNSAFE, B2_LABELS, load, make_model

ROOT = Path(__file__).resolve().parents[1]
ENC = ROOT / "predictions" / "taskb_encoders"


def main() -> None:
    train, dev = load("train.csv"), load("dev_with_label.csv")
    dev_ids = dev.ids

    # ---- encoder members --------------------------------------------------
    members = []
    for path in sorted(ENC.glob("*.json")):
        d = json.loads(path.read_text())
        assert d["ids"] == dev_ids, f"{path.name}: id order mismatch"
        assert d["classes"] == list(B2_LABELS), f"{path.name}: class order mismatch"
        members.append(d)
        print(f"loaded {path.name}")

    # ---- char-ngram member (refit, probabilities on dev) ------------------
    char_b1 = make_model("logreg").fit(train.text, train.label)
    unsafe_col = list(sorted(set(train.label))).index("unsafe")
    p_unsafe_char = char_b1.predict_proba(dev.text)[:, unsafe_col]

    char_b2 = make_model("logreg").fit(train.text, train.granular)
    classes_char = list(char_b2.classes_)
    order = [classes_char.index(c) for c in B2_LABELS]
    p_gran_char = char_b2.predict_proba(dev.text)[:, order]

    # ---- B1 ----------------------------------------------------------------
    p_unsafe_enc = np.mean([m["p_unsafe"] for m in members], axis=0)
    variants = {
        "char only": p_unsafe_char,
        "encoders only (9-avg)": p_unsafe_enc,
        "char + encoders": 0.5 * p_unsafe_char + 0.5 * p_unsafe_enc,
    }
    y1 = np.array(dev.label)
    print("\nB1 dev macro-F1 (threshold swept per variant):")
    best_name, best_pred, best_f1, best_thr = None, None, -1.0, 0.5
    for name, p in variants.items():
        thr_b, f_b = 0.5, -1.0
        for thr in np.arange(0.05, 0.96, 0.01):
            f = f1_score(y1, np.where(p >= thr, "unsafe", "safe"),
                         average="macro", zero_division=0)
            if f > f_b:
                thr_b, f_b = float(thr), float(f)
        print(f"   {name:<24} {f_b:.4f}  (thr {thr_b:.2f})")
        if f_b > best_f1:
            best_name, best_f1, best_thr = name, f_b, thr_b
            best_pred = np.where(p >= thr_b, "unsafe", "safe")

    # ---- B2 ----------------------------------------------------------------
    p_gran_enc = np.mean([m["p_gran"] for m in members], axis=0)
    y2 = np.array(dev.granular)
    print("\nB2 dev macro-F1:")
    b2_variants = {
        "char only": p_gran_char,
        "encoders only (9-avg)": p_gran_enc,
        "char + encoders": 0.5 * p_gran_char + 0.5 * p_gran_enc,
    }
    best2_name, best2_pred, best2_f1 = None, None, -1.0
    for name, p in b2_variants.items():
        pred = np.array([B2_LABELS[i] for i in p.argmax(axis=1)])
        f = f1_score(y2, pred, average="macro", zero_division=0)
        print(f"   {name:<24} {f:.4f}")
        if f > best2_f1:
            best2_name, best2_f1, best2_pred = name, float(f), pred

    # Consistency layer: an always-unsafe granular prediction forces unsafe.
    fixed = best_pred.copy()
    flips = 0
    for i, (b1, b2) in enumerate(zip(fixed, best2_pred)):
        if b1 == "safe" and b2 in ALWAYS_UNSAFE:
            fixed[i] = "unsafe"
            flips += 1
    f_after = f1_score(y1, fixed, average="macro", zero_division=0)
    print(f"\nconsistency layer: flipped {flips} safe->unsafe, "
          f"B1 {best_f1:.4f} -> {f_after:.4f}")
    if f_after > best_f1:
        best_pred, best_f1 = fixed, f_after

    print(f"\nFINAL  B1 {best_f1:.4f} ({best_name})  |  B2 {best2_f1:.4f} ({best2_name})")
    print("baseline was B1 0.9203 / B2 0.8620")

    out = ROOT / "predictions"
    json.dump(dict(zip(dev_ids, best_pred.tolist())), open(out / "b1_dev.json", "w"))
    json.dump(dict(zip(dev_ids, best2_pred.tolist())), open(out / "b2_dev.json", "w"))
    print("wrote b1_dev.json / b2_dev.json; "
          "next: python -m src.build_submissions --run-id awn_ens_v1")


if __name__ == "__main__":
    main()
