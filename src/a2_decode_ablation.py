"""Which A2 decode rule actually wins? Measured on held-out dev, not on OOF.

The A2 strategy rests on a claim: macro-F1 averages ten *independent* binary columns,
so a false positive on a rare class cannot damage any other class, and abstaining on
a class guarantees 0. That argues for low per-class thresholds and a floor on rare
classes. dev_test confirmed half of it directly (Exclusion had zero support there, and
predicting it scored the same 0.0000 as abstaining) but did not tell us whether the
aggressive rule *helps* on classes that do have support.

This settles it without spending a leaderboard slot. Train on `train` only, fit every
decode rule on train out-of-fold scores, then score each on `dev`, which the model
never saw. The winner is what we apply to the VLM and to the blind test.

    python -m src.a2_decode_ablation
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.taska_baseline import (
    A2_ACTIVE,
    HATEFUL_ONLY,
    NONHATEFUL_ONLY,
    best_threshold,
    load,
    make_model,
    prep,
)


def gate(scores: np.ndarray, p_hate: np.ndarray) -> np.ndarray:
    out = scores.copy()
    for k, cls in enumerate(A2_ACTIVE):
        if cls in HATEFUL_ONLY:
            out[:, k] *= p_hate
        elif cls in NONHATEFUL_ONLY:
            out[:, k] *= 1.0 - p_hate
    return out


def decode(scores: np.ndarray, thresholds: np.ndarray, floors: np.ndarray | None) -> np.ndarray:
    """Threshold each column, then optionally force a minimum positive count."""
    pred = (scores >= thresholds).astype(int)
    if floors is not None:
        for k in range(scores.shape[1]):
            deficit = int(floors[k]) - int(pred[:, k].sum())
            if deficit > 0:
                for i in np.argsort(-scores[:, k]):
                    if not pred[i, k]:
                        pred[i, k] = 1
                        deficit -= 1
                        if deficit == 0:
                            break
    return pred


def main() -> None:
    train, dev = load("train"), load("dev")
    X_tr, X_dev = [prep(r) for r in train], [prep(r) for r in dev]
    Y_tr = np.array([[int(c in (r["fine_grained_label"] or [])) for c in A2_ACTIVE] for r in train])
    Y_dev = np.array([[int(c in (r["fine_grained_label"] or [])) for c in A2_ACTIVE] for r in dev])
    print(f"fit on train {len(train)}, evaluate on held-out dev {len(dev)}")
    print(f"dev support: {dict(zip(A2_ACTIVE, Y_dev.sum(axis=0).tolist()))}\n")

    # A1 probabilities, needed for the gate.
    y1 = np.array([r["label"] for r in train])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    hate_col = list(sorted(set(y1))).index("Hateful")
    p_hate_oof = cross_val_predict(make_model(), X_tr, y1, cv=cv,
                                   method="predict_proba", n_jobs=-1)[:, hate_col]
    p_hate_dev = make_model().fit(X_tr, y1).predict_proba(X_dev)[:, hate_col]

    # Per-class scores, out-of-fold on train and fitted on dev.
    oof = np.zeros_like(Y_tr, dtype=float)
    devs = np.zeros_like(Y_dev, dtype=float)
    for k in range(len(A2_ACTIVE)):
        col = Y_tr[:, k]
        folds = cv if col.sum() >= 5 else StratifiedKFold(3, shuffle=True, random_state=42)
        oof[:, k] = cross_val_predict(make_model(), X_tr, col, cv=folds,
                                      method="predict_proba", n_jobs=-1)[:, 1]
        devs[:, k] = make_model().fit(X_tr, col).predict_proba(X_dev)[:, 1]

    oof_g, dev_g = gate(oof, p_hate_oof), gate(devs, p_hate_dev)
    prevalence = Y_tr.mean(axis=0)
    n_dev = len(dev)

    def thresholds_from(scores: np.ndarray) -> np.ndarray:
        return np.array([best_threshold(Y_tr[:, k], scores[:, k])[0]
                         for k in range(len(A2_ACTIVE))])

    thr_ungated, thr_gated = thresholds_from(oof), thresholds_from(oof_g)
    half = np.full(len(A2_ACTIVE), 0.5)

    rules = {
        "A  global 0.5, ungated": (devs, half, None),
        "B  global 0.5, gated": (dev_g, half, None),
        "C  per-class thr, ungated": (devs, thr_ungated, None),
        "D  per-class thr on ungated, applied gated  (what we submitted)":
            (dev_g, thr_ungated, None),
        "E  per-class thr on gated, applied gated": (dev_g, thr_gated, None),
    }
    for mult in (1.0, 1.5, 2.0):
        floors = np.maximum(2, np.ceil(mult * prevalence * n_dev))
        rules[f"F  E + rare-class floor at {mult:g}x prior"] = (dev_g, thr_gated, floors)

    print(f"{'rule':<62}{'macro-F1':>10}{'Inferiority':>13}{'Contempt':>10}{'#pred':>8}")
    results = {}
    for name, (scores, thr, floors) in rules.items():
        pred = decode(scores, thr, floors)
        macro = f1_score(Y_dev, pred, average="macro", zero_division=0)
        per = f1_score(Y_dev, pred, average=None, zero_division=0)
        results[name] = float(macro)
        inf_i, con_i = A2_ACTIVE.index("Inferiority"), A2_ACTIVE.index("Contempt")
        print(f"{name:<62}{macro:>10.4f}{per[inf_i]:>13.4f}{per[con_i]:>10.4f}{pred.sum():>8}")

    best = max(results, key=results.get)
    print(f"\nbest: {best}  ({results[best]:.4f})")
    json.dump(results, open("predictions/a2_decode_ablation.json", "w"), indent=2)


if __name__ == "__main__":
    main()
