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

Provisional threshold
---------------------
FLAGGED currently comes from the model's own contamination setting, which is
0.02 and has no justification behind it. That is worse than it sounds: it
means roughly 2% of whatever is scored gets flagged regardless of content.
Choosing a threshold on evidence needs the precision/recall work that has not
been done yet, so the flag is written but should not be trusted. The scores
themselves are meaningful; the cutoff is not.
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

from config import MODEL_DIR, PROCESSED_DIR, SCORES_DIR, ensure_dirs  # noqa: E402
from models.features import FEATURE_COLUMNS, compute_features  # noqa: E402


def load_processed(days=None):
    """Concatenate processed CSVs, optionally keeping only the last N days."""
    files = sorted(PROCESSED_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(
            f"No processed data in {PROCESSED_DIR}.\n"
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


def score(features, model, version):
    """
    Attach an anomaly score to every transaction.

    decision_function returns higher values for more normal points, so it is
    negated to make higher mean more anomalous. The result is min-max scaled
    to [0, 1] purely for readability in the dashboard.
    """
    X = features[FEATURE_COLUMNS].fillna(0)
    raw = -model.decision_function(X)

    low, high = float(raw.min()), float(raw.max())
    normalised = (raw - low) / (high - low) if high != low else raw * 0.0

    out = features[["transaction_id", "user_id", "event_time"]].copy()
    out["score"] = normalised
    out["raw_score"] = raw
    # predict() returns -1 for outliers, using the model's own offset derived
    # from contamination. Provisional: see the module docstring.
    out["flagged"] = model.predict(X) == -1
    out["model_version"] = version
    out["scored_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    return out


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
    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_COLUMNS,
        "n_training_rows": int(len(features)),
        "n_users": int(df["user_id"].nunique()),
        "contamination": args.contamination,
        "seed": args.seed,
        "sklearn_params": model.get_params(),
        "data_span": [str(df["event_time"].min()), str(df["event_time"].max())],
    }
    (MODEL_DIR / f"{version}.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"saved {model_path.name} and {version}.json")

    scored = score(features, model, version)

    # Carry the ground-truth label through so the evaluation step has it.
    # The label is never used as an input to the model.
    scored = scored.merge(df[["transaction_id", "label"]], on="transaction_id", how="left")

    scores_path = SCORES_DIR / "scores.csv"
    scored.to_csv(scores_path, index=False)

    print(f"\nscored {len(scored):,} transactions -> {scores_path}")
    print(f"  flagged {int(scored['flagged'].sum()):,} "
          f"({scored['flagged'].mean():.2%}) at the provisional threshold")
    print(f"  score range {scored['score'].min():.4f} to {scored['score'].max():.4f}")

    if "label" in scored and scored["label"].notna().any():
        actual_fraud = int(scored["label"].sum())
        caught = int(((scored["label"] == 1) & scored["flagged"]).sum())
        print(f"  ground truth fraud: {actual_fraud}, of which flagged: {caught}")
        print("  (not an evaluation; precision/recall/PR-AUC are the next task)")


if __name__ == "__main__":
    main()
