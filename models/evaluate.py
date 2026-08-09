"""
Evaluate the anomaly detector against the ground-truth labels.

Protocol, fixed before any numbers were looked at
-------------------------------------------------
Temporal split. The first 70% of days train, the last 30% test. A random
split would be wrong here: transactions are ordered in time and a user's
features depend on their own history, so random assignment would let the
model learn from a user's future.

Features are computed once over the whole dataset and then split by time.
That is not leakage: every feature for a given row is built only from
transactions at or before that row (see tests/test_features.py), so a test
row using history from the training period is using its own past, which is
exactly what it would have in production.

The model is unsupervised and never sees a label. Labels are used for two
things only: choosing the decision threshold on the training split, and
measuring on the test split.

Threshold rule, chosen in advance: the point on the training-split
precision/recall curve that maximises F1. Fixed before running so the
threshold is not selected to flatter the result. The old contamination-based
cutoff is reported alongside it for comparison, not as a candidate.

Two feature sets are evaluated under this identical protocol so the
before/after comparison is like for like.

Baselines
---------
A model has to beat something. Two trivial rankings:
  amount    rank transactions by amount, largest first. This is the rule a
            person would write without any machine learning, and if the
            model cannot beat it the model is not earning its place.
  random    shuffled scores. Its PR-AUC lands at the fraud prevalence and
            is the floor any ranking must clear.

Run:
    python models/evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import GROUND_TRUTH_PATH, PROCESSED_DIR, SCORES_DIR, ensure_dirs  # noqa: E402
from models.features import ABSOLUTE_FEATURES, FEATURE_COLUMNS, compute_features  # noqa: E402

TRAIN_FRACTION = 0.70
CONTAMINATION = 0.02
SEED = 42

FEATURE_SETS = {
    "absolute only (original 4)": ABSOLUTE_FEATURES,
    "absolute + user-relative (7)": FEATURE_COLUMNS,
}


def load_data():
    files = sorted(PROCESSED_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No processed data in {PROCESSED_DIR}. Run run_pipeline.py first.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", format="mixed")
    df = df.dropna(subset=["event_time"]).sort_values("event_time").reset_index(drop=True)

    features = compute_features(df)
    features = features.merge(df[["transaction_id", "amount", "label"]],
                              on="transaction_id", how="left")

    if GROUND_TRUTH_PATH.exists():
        truth = pd.read_csv(GROUND_TRUTH_PATH)
        features = features.merge(truth, on="transaction_id", how="left")
    else:
        features["fraud_type"] = None
    features["fraud_type"] = features["fraud_type"].fillna("legitimate")

    return features.sort_values("event_time").reset_index(drop=True)


def temporal_split(features):
    days = features["event_time"].dt.normalize()
    unique_days = np.sort(days.unique())
    cutoff = unique_days[int(len(unique_days) * TRAIN_FRACTION)]
    train = features[days < cutoff].copy()
    test = features[days >= cutoff].copy()
    return train, test, pd.Timestamp(cutoff)


def fit_and_score(train, test, columns):
    """Fit on train only, return anomaly scores for both splits."""
    model = IsolationForest(n_estimators=100, contamination=CONTAMINATION,
                            random_state=SEED, n_jobs=-1)
    model.fit(train[columns].fillna(0))

    # Negated so that larger means more anomalous.
    train_scores = -model.decision_function(train[columns].fillna(0))
    test_scores = -model.decision_function(test[columns].fillna(0))
    return model, train_scores, test_scores


def best_f1_threshold(labels, scores):
    """The threshold maximising F1 on the given split."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # precision_recall_curve returns one more precision/recall than threshold.
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(precision[best]), float(recall[best]), float(f1[best])


def metrics_at(labels, scores, threshold):
    predicted = scores >= threshold
    true_positive = int((predicted & (labels == 1)).sum())
    flagged = int(predicted.sum())
    positives = int((labels == 1).sum())

    precision = true_positive / flagged if flagged else 0.0
    recall = true_positive / positives if positives else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "flagged": flagged, "true_positives": true_positive,
            "alert_rate": flagged / len(labels)}


def typology_recall(test, predicted):
    rows = []
    fraud = test["fraud_type"] != "legitimate"
    for typology in sorted(test.loc[fraud, "fraud_type"].unique()):
        mask = test["fraud_type"] == typology
        n = int(mask.sum())
        caught = int((mask & predicted).sum())
        rows.append({"typology": typology, "n": n, "caught": caught,
                     "recall": caught / n if n else 0.0})
    return rows


def pr_table(labels, scores, points=9):
    """A readable slice of the precision/recall curve."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall, thresholds = precision[:-1], recall[:-1], thresholds
    rows = []
    for target in np.linspace(0.05, 0.85, points):
        idx = int(np.argmin(np.abs(recall - target)))
        rows.append((recall[idx], precision[idx], thresholds[idx]))
    # De-duplicate while preserving order.
    seen, unique = set(), []
    for r, p, t in rows:
        key = round(r, 4)
        if key not in seen:
            seen.add(key)
            unique.append((r, p, t))
    return unique


def main():
    ensure_dirs()
    features = load_data()
    train, test, cutoff = temporal_split(features)

    print("=" * 74)
    print("EVALUATION")
    print("=" * 74)
    print(f"total transactions      {len(features):,}")
    print(f"temporal split at       {cutoff:%Y-%m-%d}  "
          f"(first {TRAIN_FRACTION:.0%} of days train)")
    print(f"train                   {len(train):,} rows, "
          f"{int(train['label'].sum())} fraud ({train['label'].mean():.3%})")
    print(f"test                    {len(test):,} rows, "
          f"{int(test['label'].sum())} fraud ({test['label'].mean():.3%})")
    print(f"\nOnly {int(test['label'].sum())} fraud rows in the test split, so all "
          f"test metrics below\ncarry meaningful sampling noise. Treat differences of a "
          f"few points as noise.")

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    prevalence = float(y_test.mean())

    results = {"split_date": str(cutoff), "test_rows": len(test),
               "test_fraud": int(y_test.sum()), "test_prevalence": prevalence,
               "feature_sets": {}, "baselines": {}}

    # ---------------------------------------------------------------- models
    for name, columns in FEATURE_SETS.items():
        model, train_scores, test_scores = fit_and_score(train, test, columns)

        threshold, tr_p, tr_r, tr_f1 = best_f1_threshold(y_train, train_scores)
        chosen = metrics_at(y_test, test_scores, threshold)

        # What the old contamination cutoff would have given, for comparison.
        contamination_cut = float(np.quantile(test_scores, 1 - CONTAMINATION))
        old_way = metrics_at(y_test, test_scores, contamination_cut)

        pr_auc = float(average_precision_score(y_test, test_scores))
        roc_auc = float(roc_auc_score(y_test, test_scores))

        print("\n" + "=" * 74)
        print(f"{name}   ({len(columns)} features)")
        print("=" * 74)
        print(f"  PR-AUC                  {pr_auc:.4f}"
              f"     (random baseline = {prevalence:.4f})")
        print(f"  ROC-AUC                 {roc_auc:.4f}")
        print(f"\n  threshold chosen on train by max F1: {threshold:.5f}")
        print(f"    train  precision {tr_p:.3f}  recall {tr_r:.3f}  F1 {tr_f1:.3f}")
        print(f"    TEST   precision {chosen['precision']:.3f}  "
              f"recall {chosen['recall']:.3f}  F1 {chosen['f1']:.3f}")
        print(f"           flagged {chosen['flagged']:,} of {len(test):,} "
              f"({chosen['alert_rate']:.2%}), caught {chosen['true_positives']} "
              f"of {int(y_test.sum())}")
        print(f"\n  for comparison, the old contamination={CONTAMINATION} cutoff:")
        print(f"    TEST   precision {old_way['precision']:.3f}  "
              f"recall {old_way['recall']:.3f}  F1 {old_way['f1']:.3f}"
              f"   (flags {old_way['alert_rate']:.2%} by construction)")

        predicted = pd.Series(test_scores >= threshold, index=test.index)
        print("\n  recall by fraud typology, at the chosen threshold")
        print(f"    {'typology':<20}{'n':>5}{'caught':>8}{'recall':>9}")
        typologies = typology_recall(test, predicted)
        for row in typologies:
            print(f"    {row['typology']:<20}{row['n']:>5}{row['caught']:>8}"
                  f"{row['recall']:>8.1%}")

        print("\n  precision/recall tradeoff (test split)")
        print(f"    {'recall':>8}{'precision':>11}{'threshold':>12}")
        for recall_value, precision_value, threshold_value in pr_table(y_test, test_scores):
            print(f"    {recall_value:>8.1%}{precision_value:>11.3f}{threshold_value:>12.5f}")

        results["feature_sets"][name] = {
            "n_features": len(columns), "features": columns,
            "pr_auc": pr_auc, "roc_auc": roc_auc,
            "threshold": threshold,
            "test": chosen, "contamination_cutoff": old_way,
            "typology_recall": typologies,
        }

    # ------------------------------------------------------------- baselines
    print("\n" + "=" * 74)
    print("BASELINES (test split)")
    print("=" * 74)

    amount_scores = test["amount"].to_numpy()
    amount_pr = float(average_precision_score(y_test, amount_scores))
    rng = np.random.default_rng(SEED)
    random_pr = float(average_precision_score(y_test, rng.random(len(y_test))))

    print(f"  rank by amount, largest first   PR-AUC {amount_pr:.4f}")
    print(f"  random ordering                 PR-AUC {random_pr:.4f}")
    print(f"  fraud prevalence                       {prevalence:.4f}")
    results["baselines"] = {"amount_ranking_pr_auc": amount_pr,
                            "random_pr_auc": random_pr}

    # ---------------------------------------------------------------- output
    print("\n" + "=" * 74)
    print("SUMMARY: PR-AUC on the test split")
    print("=" * 74)
    ranked = [("random ordering", random_pr), ("rank by amount", amount_pr)]
    ranked += [(name, r["pr_auc"]) for name, r in results["feature_sets"].items()]
    for name, value in sorted(ranked, key=lambda kv: kv[1]):
        bar = "#" * int(60 * value / max(v for _, v in ranked))
        print(f"  {name:<30} {value:.4f}  {bar}")

    out_path = SCORES_DIR / "evaluation.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
