"""
Evaluate the fraud detectors against ground-truth labels.

Protocol, fixed before any numbers were looked at
-------------------------------------------------
Temporal split. The first 70% of days train, the last 30% test. A random
split would be wrong here: transactions are ordered in time and a user's
features depend on their own history, so random assignment would let a model
learn from a user's future.

Features are computed once over the whole dataset and then split by time.
That is not leakage: every feature for a given row is built only from
transactions at or before that row (asserted in tests/test_features.py), so a
test row using history from the training period is using its own past, which
is what it would have in production.

Threshold rule, chosen in advance: the point on the precision/recall curve
that maximises F1.

Where that curve comes from differs by model type, and the difference
matters. The Isolation Forest never sees a label, so its training-split
scores are not optimistically biased and the threshold can be read off them
directly. A supervised model fitted on the training split will score its own
training rows far too well - a random forest largely memorises them - so a
threshold read off those scores would be far too high and would flag almost
nothing at test time. The supervised thresholds therefore come from
out-of-fold predictions produced by walk-forward TimeSeriesSplit folds within
the training period. Same rule, applied to honest scores.

No hyperparameter search was run on any model. Defaults plus
class_weight="balanced", decided in advance and not revised after seeing
results.

Baselines
---------
  amount    rank transactions by amount, largest first. The rule a person
            would write with no machine learning at all.
  random    shuffled scores. Its PR-AUC lands at the fraud prevalence and is
            the floor any ranking must clear.

Run:
    python models/evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import GROUND_TRUTH_PATH, PROCESSED_DIR, SCORES_DIR, ensure_dirs  # noqa: E402
from models.features import ABSOLUTE_FEATURES, FEATURE_COLUMNS, compute_features  # noqa: E402
from models.thresholds import best_f1_threshold  # noqa: E402

TRAIN_FRACTION = 0.70
CONTAMINATION = 0.02
SEED = 42
N_FOLDS = 5

# Alert rates to report metrics at. A fraud team's real constraint is how
# many alerts an analyst can work through in a day, which fixes this number
# regardless of what the curve looks like.
ALERT_RATES = [0.001, 0.0028, 0.005, 0.01, 0.02]


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
        features = features.merge(pd.read_csv(GROUND_TRUTH_PATH),
                                  on="transaction_id", how="left")
    else:
        features["fraud_type"] = None
    features["fraud_type"] = features["fraud_type"].fillna("legitimate")

    return features.sort_values("event_time").reset_index(drop=True)


def temporal_split(features):
    days = features["event_time"].dt.normalize()
    unique_days = np.sort(days.unique())
    cutoff = unique_days[int(len(unique_days) * TRAIN_FRACTION)]
    return (features[days < cutoff].copy(),
            features[days >= cutoff].copy(),
            pd.Timestamp(cutoff))


def unsupervised_scores(train, test, columns):
    """Isolation Forest. Never sees a label."""
    model = IsolationForest(n_estimators=100, contamination=CONTAMINATION,
                            random_state=SEED, n_jobs=-1)
    model.fit(train[columns].fillna(0))
    # Negated so larger means more anomalous.
    return (-model.decision_function(train[columns].fillna(0)),
            -model.decision_function(test[columns].fillna(0)))


def supervised_scores(estimator, train, test, columns):
    """
    Fit on the training split and return (out_of_fold_train, test) scores.

    The out-of-fold scores come from walk-forward folds so the threshold is
    chosen on predictions the model did not train on. TimeSeriesSplit does
    not assign every row to a test fold, so the earliest rows have no
    out-of-fold score and are excluded from threshold selection.
    """
    X_train = train[columns].fillna(0)
    y_train = train["label"].to_numpy()

    out_of_fold = np.full(len(y_train), np.nan)
    for fold_train, fold_test in TimeSeriesSplit(n_splits=N_FOLDS).split(X_train):
        fitted = clone(estimator)
        fitted.fit(X_train.iloc[fold_train], y_train[fold_train])
        out_of_fold[fold_test] = fitted.predict_proba(X_train.iloc[fold_test])[:, 1]

    final = clone(estimator)
    final.fit(X_train, y_train)
    test_scores = final.predict_proba(test[columns].fillna(0))[:, 1]

    covered = ~np.isnan(out_of_fold)
    return out_of_fold, covered, test_scores


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


def report(name, detail, test, y_test, test_scores, threshold):
    chosen = metrics_at(y_test, test_scores, threshold)
    pr_auc = float(average_precision_score(y_test, test_scores))
    roc_auc = float(roc_auc_score(y_test, test_scores))

    print("\n" + "=" * 78)
    print(f"{name}")
    print(f"  {detail}")
    print("=" * 78)
    print(f"  PR-AUC   {pr_auc:.4f}        ROC-AUC  {roc_auc:.4f}")
    print(f"\n  at the chosen operating point (threshold {threshold:.5f}):")
    print(f"    alert rate {chosen['alert_rate']:.2%}  "
          f"= {chosen['flagged']} alerts per {len(test):,} transactions")
    print(f"    precision  {chosen['precision']:.3f}   "
          f"recall {chosen['recall']:.3f}   F1 {chosen['f1']:.3f}")
    print(f"    caught {chosen['true_positives']} of {int(y_test.sum())} fraud")

    predicted = pd.Series(test_scores >= threshold, index=test.index)
    typologies = typology_recall(test, predicted)
    print(f"\n  recall by typology at that point")
    print(f"    {'typology':<20}{'n':>5}{'caught':>8}{'recall':>9}")
    for row in typologies:
        print(f"    {row['typology']:<20}{row['n']:>5}{row['caught']:>8}"
              f"{row['recall']:>8.1%}")

    print(f"\n  if the analyst team can review a fixed number of alerts")
    print(f"    {'alert rate':>11}{'alerts':>9}{'precision':>11}"
          f"{'recall':>9}{'caught':>9}")
    for rate in ALERT_RATES:
        cut = float(np.quantile(test_scores, 1.0 - rate))
        point = metrics_at(y_test, test_scores, cut)
        marker = "  <- chosen" if abs(rate - chosen["alert_rate"]) < 0.0006 else ""
        print(f"    {rate:>10.2%}{point['flagged']:>9}{point['precision']:>11.3f}"
              f"{point['recall']:>9.3f}{point['true_positives']:>9}{marker}")

    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "threshold": float(threshold),
            "operating_point": chosen, "typology_recall": typologies}


def pr_curve_points(y_test, scores, points=9):
    precision, recall, thresholds = precision_recall_curve(y_test, scores)
    precision, recall = precision[:-1], recall[:-1]
    rows, seen = [], set()
    for target in np.linspace(0.05, 0.85, points):
        idx = int(np.argmin(np.abs(recall - target)))
        key = round(float(recall[idx]), 4)
        if key not in seen:
            seen.add(key)
            rows.append((float(recall[idx]), float(precision[idx]), float(thresholds[idx])))
    return rows


def main():
    ensure_dirs()
    features = load_data()
    train, test, cutoff = temporal_split(features)

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    prevalence = float(y_test.mean())

    print("=" * 78)
    print("EVALUATION")
    print("=" * 78)
    print(f"total transactions   {len(features):,}")
    print(f"temporal split at    {cutoff:%Y-%m-%d}  (first {TRAIN_FRACTION:.0%} of days train)")
    print(f"train                {len(train):,} rows, {int(y_train.sum())} fraud "
          f"({y_train.mean():.3%})")
    print(f"test                 {len(test):,} rows, {int(y_test.sum())} fraud "
          f"({prevalence:.3%})")
    print(f"\nOnly {int(y_test.sum())} fraud rows in the test split. Every figure below "
          f"carries real\nsampling noise; treat differences of a few points as noise.")

    results = {"split_date": str(cutoff), "test_rows": len(test),
               "test_fraud": int(y_test.sum()), "test_prevalence": prevalence,
               "models": {}, "baselines": {}}
    summary = []

    # ------------------------------------------------------- unsupervised
    for name, columns in [
        ("Isolation Forest, 4 absolute features", ABSOLUTE_FEATURES),
        ("Isolation Forest, 7 features", FEATURE_COLUMNS),
    ]:
        train_scores, test_scores = unsupervised_scores(train, test, columns)
        threshold, *_ = best_f1_threshold(y_train, train_scores)
        detail = (f"unsupervised, {len(columns)} features, "
                  f"threshold from training-split scores")
        results["models"][name] = report(name, detail, test, y_test,
                                         test_scores, threshold)
        results["models"][name]["features"] = columns
        summary.append((name, results["models"][name]["pr_auc"]))

        if columns is FEATURE_COLUMNS:
            print("\n  precision/recall tradeoff")
            print(f"    {'recall':>8}{'precision':>11}{'threshold':>12}")
            for recall_value, precision_value, cut in pr_curve_points(y_test, test_scores):
                print(f"    {recall_value:>8.1%}{precision_value:>11.3f}{cut:>12.5f}")

    # --------------------------------------------------------- supervised
    supervised = {
        "Logistic regression, 7 features": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=SEED)),
        "Random forest, 7 features": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
    }

    for name, estimator in supervised.items():
        oof, covered, test_scores = supervised_scores(
            estimator, train, test, FEATURE_COLUMNS)
        chosen = best_f1_threshold(y_train[covered], oof[covered])
        threshold = chosen[0] if chosen else 0.5
        detail = (f"supervised, 7 features, class_weight=balanced, "
                  f"threshold from {N_FOLDS}-fold walk-forward out-of-fold scores")
        results["models"][name] = report(name, detail, test, y_test,
                                         test_scores, threshold)
        results["models"][name]["features"] = FEATURE_COLUMNS
        summary.append((name, results["models"][name]["pr_auc"]))

    # ---------------------------------------------------------- baselines
    print("\n" + "=" * 78)
    print("TRIVIAL BASELINES (test split)")
    print("=" * 78)

    amount_pr = float(average_precision_score(y_test, test["amount"].to_numpy()))
    rng = np.random.default_rng(SEED)
    random_pr = float(average_precision_score(y_test, rng.random(len(y_test))))

    print(f"  rank by amount, largest first   PR-AUC {amount_pr:.4f}")
    print(f"  random ordering                 PR-AUC {random_pr:.4f}")
    print(f"  fraud prevalence                       {prevalence:.4f}")
    results["baselines"] = {"amount_ranking_pr_auc": amount_pr,
                            "random_pr_auc": random_pr}
    summary += [("rank by amount, no model", amount_pr),
                ("random ordering", random_pr)]

    # ------------------------------------------------------------ summary
    print("\n" + "=" * 78)
    print("SUMMARY: PR-AUC on the test split")
    print("=" * 78)
    best = max(value for _, value in summary)
    for name, value in sorted(summary, key=lambda kv: kv[1]):
        print(f"  {name:<40} {value:.4f}  {'#' * int(34 * value / best)}")

    out_path = SCORES_DIR / "evaluation.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
