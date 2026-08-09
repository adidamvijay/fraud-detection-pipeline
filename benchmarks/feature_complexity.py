"""
Measure the old nested-loop feature computation against the vectorised one.

This is not part of the pipeline. It exists so the claim "the rewrite is
asymptotically faster" is backed by numbers taken on this machine against
real generated data, and so the two implementations can be checked for
agreement rather than assumed equivalent.

Run:
    python benchmarks/feature_complexity.py
"""

import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROCESSED_DIR  # noqa: E402
from models.features import compute_features  # noqa: E402


def compute_features_loop(df):
    """
    The original implementation, reproduced here for comparison only.

    For each user it iterates every transaction and rebuilds a boolean mask
    over that user's entire history to find the trailing 24-hour window. A
    user with k transactions costs O(k^2).
    """
    records = []
    for user, group in df.groupby("user_id"):
        group = group.sort_values("event_time").reset_index(drop=True)
        for i, row in group.iterrows():
            ts = row["event_time"]
            window = group[(group["event_time"] >= ts - timedelta(hours=24))
                           & (group["event_time"] <= ts)]
            total = window["amount"].sum()
            count = len(window)
            records.append({
                "transaction_id": row["transaction_id"],
                "user_id": user,
                "event_time": ts,
                "total_amount_24h": float(total),
                "txn_count_24h": int(count),
                "avg_amount_24h": float(window["amount"].mean()) if count else 0.0,
                "hours_since_last_txn": (
                    (ts - group.loc[i - 1, "event_time"]).total_seconds() / 3600.0
                    if i > 0 else -1.0),
            })
    return pd.DataFrame(records)


def load_transactions():
    files = sorted(PROCESSED_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(
            f"No processed data in {PROCESSED_DIR}. Run the pipeline first:\n"
            "  python etl/generate_transactions.py\n"
            "  python etl/validate_data.py\n"
            "  python etl/ingest_local.py")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def timed(fn, df):
    start = time.perf_counter()
    result = fn(df)
    return time.perf_counter() - start, result


def main():
    df = load_transactions()
    users = sorted(df["user_id"].unique())
    print(f"loaded {len(df):,} transactions across {len(users):,} users\n")

    # Axis 1: more users, history depth per user held constant.
    # Both implementations are linear in total rows here, because the cost
    # driver for the loop is transactions per user, which is not changing.
    # The gap is a large constant factor, not a difference in growth rate.
    print("more users, history per user held constant")
    print(f"{'users':>8} {'rows':>9} {'txns/user':>10} {'loop (s)':>10} "
          f"{'vectorised (s)':>15} {'speedup':>9}")
    print("-" * 68)

    for n_users in (25, 50, 100, 200, len(users)):
        if n_users > len(users):
            continue
        subset = df[df["user_id"].isin(users[:n_users])].copy()

        loop_seconds, _ = timed(compute_features_loop, subset)
        vec_seconds, _ = timed(compute_features, subset)

        speedup = loop_seconds / vec_seconds if vec_seconds else float("inf")
        print(f"{n_users:>8,} {len(subset):>9,} {len(subset)/n_users:>10.1f} "
              f"{loop_seconds:>10.3f} {vec_seconds:>15.3f} {speedup:>8.1f}x")

    # Axis 2: user count fixed, history per user deepened by widening the
    # date range. This is the axis the quadratic term actually lives on.
    print("\ndeeper history per user, user count held constant at 200")
    print(f"{'days':>8} {'rows':>9} {'txns/user':>10} {'loop (s)':>10} "
          f"{'vectorised (s)':>15} {'speedup':>9}")
    print("-" * 68)

    fixed_users = df[df["user_id"].isin(users[:200])]
    latest = fixed_users["event_time"].max()

    for days in (4, 8, 16, 30):
        subset = fixed_users[
            fixed_users["event_time"] > latest - pd.Timedelta(days=days)].copy()
        if subset.empty:
            continue
        n_users_here = subset["user_id"].nunique()

        loop_seconds, _ = timed(compute_features_loop, subset)
        vec_seconds, _ = timed(compute_features, subset)

        speedup = loop_seconds / vec_seconds if vec_seconds else float("inf")
        print(f"{days:>8} {len(subset):>9,} {len(subset)/n_users_here:>10.1f} "
              f"{loop_seconds:>10.3f} {vec_seconds:>15.3f} {speedup:>8.1f}x")

    # Axis 3: isolate the quadratic term.
    # Axes 1 and 2 both come out roughly linear, because at realistic history
    # depths the loop's cost is dominated by the fixed overhead of doing any
    # pandas slice at all, not by the length of the array being sliced. The
    # O(k) mask build only overtakes that constant at much deeper history.
    # Synthetic users with very long histories, spaced so the number of
    # transactions inside the 24-hour window stays constant.
    print("\nisolating the quadratic term: 10 users, increasing history depth")
    print(f"{'txns/user':>10} {'rows':>9} {'loop (s)':>10} {'per row (ms)':>13} "
          f"{'vectorised (s)':>15}")
    print("-" * 62)

    for k in (250, 500, 1000, 2000):
        n_users = 10
        synthetic = pd.DataFrame({
            "transaction_id": [f"t{i}" for i in range(n_users * k)],
            "user_id": [f"U{i // k}" for i in range(n_users * k)],
            # Six hours apart, so a 24-hour window always holds about 4 rows
            # however long the history gets.
            "event_time": pd.Timestamp("2026-01-01") + pd.to_timedelta(
                [(i % k) * 6 for i in range(n_users * k)], unit="h"),
            "amount": 100.0,
        })
        loop_seconds, _ = timed(compute_features_loop, synthetic)
        vec_seconds, _ = timed(compute_features, synthetic)
        per_row_ms = loop_seconds / len(synthetic) * 1000
        print(f"{k:>10,} {len(synthetic):>9,} {loop_seconds:>10.3f} "
              f"{per_row_ms:>13.3f} {vec_seconds:>15.3f}")

    print("  If per-row cost is flat, the constant factor dominates at this")
    print("  depth. If it climbs with history, the O(k) mask build is winning.")

    print("\nchecking the two implementations agree on the full dataset")
    _, loop_result = timed(compute_features_loop, df)
    _, vec_result = timed(compute_features, df)

    key = ["transaction_id"]
    left = loop_result.sort_values(key).reset_index(drop=True)
    right = vec_result.sort_values(key).reset_index(drop=True)

    print(f"  row counts: loop {len(left):,} vs vectorised {len(right):,}")
    for col in ("total_amount_24h", "txn_count_24h", "avg_amount_24h",
                "hours_since_last_txn"):
        max_diff = (left[col].astype(float) - right[col].astype(float)).abs().max()
        status = "match" if max_diff < 1e-6 else f"DIFFER by {max_diff}"
        print(f"  {col:<24} {status}")


if __name__ == "__main__":
    main()
