"""
Train an Isolation Forest on local processed CSVs and score them.

Runs entirely from the filesystem with no warehouse connection, so the model
and its evaluation stay reproducible after a Snowflake trial expires. The
Snowflake path reads the same data from RAW_TRANSACTIONS and produces the
same features, because both use models/features.compute_features.

Reads   data/processed/*.csv
Writes  models/artifacts/<version>.joblib
        models/artifacts/<version>.json   training metadata
        data/scores/scores.csv

Threshold
---------
FLAGGED comes from the F1-maximising point on the precision/recall curve,
computed by models/thresholds.py from the labelled training data. This is the
same rule models/evaluate.py uses to report its operating point, so the flags
written here correspond to a measured precision and recall rather than to an
arbitrary cutoff.

It previously used the model's contamination setting of 0.02, which flagged
2% of whatever was scored regardless of content. On the held-out test split
that cutoff gave precision 0.079 at recall 0.280; the F1-chosen threshold
gives precision 0.320 at recall 0.160 while flagging seven times fewer
transactions.

If no labels are available the job falls back to the contamination cutoff and
says so, rather than silently pretending the threshold means something.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    ARCHIVE_DIR, MODEL_DIR, PROCESSED_DIR, SCORES_DIR, ensure_dirs,
)

ARCHIVE_PROCESSED_DIR = ARCHIVE_DIR / "processed"
from models.features import FEATURE_COLUMNS, compute_features  # noqa: E402
from models.thresholds import best_f1_threshold, threshold_for_alert_rate  # noqa: E402

# Used only when no labels exist to choose a threshold from.
CONTAMINATION_FALLBACK = 0.02


def load_processed(days=None):
    """
    Concatenate processed CSVs, optionally keeping only the last N days.

    Reads the archive as well as the live directory. etl/ingest_to_snowflake.py
    archives a file once it has been loaded to the warehouse, so in the daily
    DAG's order the loader runs before training and would otherwise leave
    nothing to train on. Training wants full history regardless: the model is
    learning what normal behaviour looks like, and throwing away every file
    that happens to have reached Snowflake already would shrink the training
    set on every run.
    """
    files = sorted(PROCESSED_DIR.glob("*.csv")) + \
        sorted(ARCHIVE_PROCESSED_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(
            f"No processed data in {PROCESSED_DIR} or {ARCHIVE_PROCESSED_DIR}.\n"
            "Run the pipeline first:\n"
            "  python etl/generate_transactions.py\n"
            "  python etl/validate_data.py\n"
            "  python etl/ingest_local.py")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", format="mixed")
    df = df.dropna(subset=["event_time"])

    if days:
        cutoff = df["event_time"].max() - pd.Timedelta(days=days)
        df = df[df["event_time"] > cutoff]

    return df.sort_values("event_time").reset_index(drop=True)


def train(features, contamination, seed):
    """Fit an Isolation Forest on the four rolling features."""
    X = features[FEATURE_COLUMNS].fillna(0)
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def score(features, model, version, labels=None):
    """
    Attach an anomaly score to every transaction and decide the flag.

    decision_function returns higher values for more normal points, so it is
    negated to make higher mean more anomalous. The result is min-max scaled
    to [0, 1] purely for readability in the dashboard; the flag is applied to
    the raw score, because a min-max scaling depends on the batch and would
    make the threshold mean something different on every run.
    """
    X = features[FEATURE_COLUMNS].fillna(0)
    raw = -model.decision_function(X)

    low, high = float(raw.min()), float(raw.max())
    normalised = (raw - low) / (high - low) if high != low else raw * 0.0

    out = features[["transaction_id", "user_id", "event_time"]].copy()
    out["score"] = normalised
    out["raw_score"] = raw

    chosen = best_f1_threshold(labels, raw) if labels is not None else None
    if chosen is None:
        threshold = threshold_for_alert_rate(raw, CONTAMINATION_FALLBACK)
        rule = f"contamination fallback ({CONTAMINATION_FALLBACK})"
        print("  no labels available; falling back to the contamination "
              "cutoff. This threshold has no measured precision behind it.")
    else:
        threshold, precision, recall, f1 = chosen
        rule = "max F1 on the precision/recall curve"
        print(f"  threshold {threshold:.5f} by max F1 "
              f"(precision {precision:.3f}, recall {recall:.3f}, F1 {f1:.3f})")

    out["flagged"] = raw >= threshold
    out["model_version"] = version
    out["scored_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    return out, {"threshold": float(threshold), "rule": rule}


def main():
    parser = argparse.ArgumentParser(description="Train and score locally.")
    parser.add_argument("--contamination", type=float, default=0.02,
                        help="Isolation Forest contamination. Currently "
                             "unjustified; see the module docstring.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=None,
                        help="only train on the last N days of data")
    args = parser.parse_args()

    ensure_dirs()

    df = load_processed(days=args.days)
    print(f"loaded {len(df):,} transactions across {df['user_id'].nunique():,} users")
    print(f"  span {df['event_time'].min()} to {df['event_time'].max()}")

    features = compute_features(df)
    print(f"computed features for {len(features):,} transactions")

    version = f"isoforest_{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    model = train(features, args.contamination, args.seed)
    print(f"trained {version} on {len(features):,} rows")

    model_path = MODEL_DIR / f"{version}.joblib"
    joblib.dump(model, model_path)

    # Metadata alongside the artifact, so a scored row can always be traced
    # back to what produced it.
    # Labels choose the threshold only. They are never an input to the model.
    labelled = features.merge(df[["transaction_id", "label"]],
                              on="transaction_id", how="left")
    labels = labelled["label"] if labelled["label"].notna().all() else None

    scored, threshold_info = score(features, model, version, labels)

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_COLUMNS,
        "n_training_rows": int(len(features)),
        "n_users": int(df["user_id"].nunique()),
        "contamination": args.contamination,
        "seed": args.seed,
        "threshold": threshold_info["threshold"],
        "threshold_rule": threshold_info["rule"],
        "sklearn_params": model.get_params(),
        "data_span": [str(df["event_time"].min()), str(df["event_time"].max())],
    }
    (MODEL_DIR / f"{version}.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"saved {model_path.name} and {version}.json")

    scored = scored.merge(df[["transaction_id", "label"]], on="transaction_id", how="left")

    scores_path = SCORES_DIR / "scores.csv"
    scored.to_csv(scores_path, index=False)

    print(f"\nscored {len(scored):,} transactions -> {scores_path}")
    print(f"  flagged {int(scored['flagged'].sum()):,} "
          f"({scored['flagged'].mean():.2%} alert rate) by {threshold_info['rule']}")
    print(f"  score range {scored['score'].min():.4f} to {scored['score'].max():.4f}")

    if scored["label"].notna().any():
        actual_fraud = int(scored["label"].sum())
        caught = int(((scored["label"] == 1) & scored["flagged"]).sum())
        print(f"  ground truth fraud: {actual_fraud}, of which flagged: {caught}")
        print("  These are in-sample figures. For held-out metrics run "
              "models/evaluate.py.")


if __name__ == "__main__":
    main()
