"""
Per-user rolling 24-hour features, computed once for both training and scoring.

Why one module
--------------
This calculation previously existed in four near-identical copies across
feature_and_train_local.py, feature_and_train.py, score_realtime.py and
update_feature_store.py. Copies drift, and when the training copy and the
scoring copy drift you get training/serving skew: the model is scored on
features that were not computed the way it learned them. Having one
implementation makes that class of bug impossible.

Why vectorised
--------------
Every copy used the same nested loop: group by user, then iterrows(), and
inside that build a boolean mask over the whole group to select the trailing
24-hour window. For a user with k transactions that is O(k) work repeated k
times, so O(k^2) per user.

That asymptotic story is true but it is not the reason the loop is slow, and
it is worth being precise because the measured numbers do not show quadratic
growth. Benchmarks in benchmarks/feature_complexity.py, run on this machine:

  - Widening the user base 20x: loop time grows 20x. Linear.
  - Deepening history per user 7x: loop time grows 7x. Linear.
  - Deepening history per user 8x, from 250 to 2,000 transactions each:
    cost per row moves only from 0.654 ms to 0.699 ms, a 7% rise.

The O(k) mask build is therefore contributing almost nothing at any depth
this project will encounter. What actually costs 0.67 ms per row is the fixed
overhead of doing anything per-row in Python and pandas at all: the iterrows
call, constructing a boolean Series, and materialising a slice. The quadratic
term would only overtake that constant at roughly a hundred thousand
transactions per single user.

So the accurate claim is that this rewrite removes a large constant factor,
not that it changes the growth rate at realistic scale. Measured speedup on
the full 29,178-row dataset is 344x, and the quadratic term is removed as
well, which matters only for pathologically deep single-user history.

This version sorts once and uses pandas' time-based rolling window, which
advances two pointers over sorted data: O(n log n) for the sort, O(n) for the
windows, with the per-row work happening in C rather than Python.

Both implementations are checked for exact agreement on all four features
across the full dataset by the benchmark script.
"""

import pandas as pd

# The original four. Absolute quantities: they describe the transaction and
# its 24-hour window without reference to what is normal for that user.
ABSOLUTE_FEATURES = [
    "total_amount_24h", "txn_count_24h", "avg_amount_24h", "hours_since_last_txn",
]

# Added after measuring the four above. Each is one of those quantities
# divided by that user's own trailing baseline, because absolute features
# cannot distinguish "large for this user" from "large". Measured on the
# absolute set: 51% recall on high-value takeover but 2.5% on card-testing
# bursts, and the highest-scoring legitimate rows were simply heavy spenders.
RELATIVE_FEATURES = [
    "amount_vs_user_median",   # targets account takeover
    "count_vs_user_typical",   # targets card-testing bursts
    "gap_vs_user_typical",     # targets burst velocity
]

FEATURE_COLUMNS = ABSOLUTE_FEATURES + RELATIVE_FEATURES

# Ratio used when a user has no prior history to compare against. 1.0 means
# "indistinguishable from this user's baseline", so a user's first
# transactions are treated as unremarkable rather than anomalous. The
# alternative would flag every new user by construction.
NO_BASELINE_RATIO = 1.0

# Value used for a user's first ever transaction, which has no predecessor.
# Chosen to sit outside the range of any real elapsed time so the model can
# separate it. Known weakness: it is numerically close to zero, which means
# "seconds since last transaction", while semantically it means the opposite.
# It affects one row per user (about 1.7% of rows on the default dataset).
# Documented as a limitation in DESIGN.md rather than hidden.
NO_PREVIOUS_TXN = -1.0


def compute_features(df):
    """
    Given transactions with user_id, event_time and amount, return one row per
    transaction with the four model features.

    The 24-hour window is trailing and inclusive of the current transaction,
    matching the original implementation.
    """
    required = {"transaction_id", "user_id", "event_time", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_features is missing columns: {sorted(missing)}")

    out = df.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)

    # A row without a timestamp cannot be placed in a window at all.
    out = out.dropna(subset=["event_time"])
    if out.empty:
        return pd.DataFrame(columns=[
            "transaction_id", "user_id", "event_time", *FEATURE_COLUMNS])

    # Rolling on a time index requires the index sorted within each group.
    out = out.sort_values(["user_id", "event_time"], kind="mergesort").reset_index(drop=True)

    # closed="both" makes the window [t-24h, t], including a transaction
    # sitting exactly 24 hours earlier. pandas defaults to (t-24h, t], which
    # excludes it. The original loop used >= and <=, so it included that
    # boundary point; matching it keeps this rewrite a pure performance
    # change rather than a silent change of results. On the default dataset
    # the difference showed up on exactly one row in 29,178.
    rolling = (
        out.set_index("event_time")
        .groupby("user_id")["amount"]
        .rolling("24h", closed="both")
    )
    total = rolling.sum().reset_index(drop=True)
    count = rolling.count().reset_index(drop=True)

    out["total_amount_24h"] = total.to_numpy()
    out["txn_count_24h"] = count.to_numpy().astype(int)
    out["avg_amount_24h"] = out["total_amount_24h"] / out["txn_count_24h"]

    gap = out.groupby("user_id")["event_time"].diff().dt.total_seconds() / 3600.0
    out["hours_since_last_txn"] = gap.fillna(NO_PREVIOUS_TXN)

    _add_relative_features(out, gap)

    return out[["transaction_id", "user_id", "event_time", *FEATURE_COLUMNS]]


def _trailing(series, group, how):
    """
    That user's baseline from their earlier transactions only.

    shift(1) happens before expanding(), so the current row's own value can
    never enter its own baseline. This is the whole causality guarantee and
    it is one call: if the shift is removed, every ratio becomes partly a
    comparison of a value against itself. tests/test_features.py fails if
    this property breaks.

    expanding() rather than a fixed window because a user's spending norm is
    a long-run property; using only the last few transactions would let a
    fraud burst redefine the baseline it is being measured against.
    """
    shifted = series.groupby(group).shift(1)
    return getattr(shifted.groupby(group).expanding(), how)().reset_index(level=0, drop=True)


def _safe_ratio(value, baseline):
    """Divide, treating an absent or non-positive baseline as 'no baseline'."""
    usable = baseline.notna() & (baseline > 0)
    return (value / baseline.where(usable)).fillna(NO_BASELINE_RATIO)


def _add_relative_features(out, gap):
    """
    Each feature is an absolute quantity over that user's own trailing
    baseline. Baselines use strictly earlier transactions only.
    """
    users = out["user_id"]

    # How large is this amount against what this user normally spends.
    # Median rather than mean so one prior fraud does not inflate the
    # baseline and hide the next one.
    baseline_amount = _trailing(out["amount"], users, "median")
    out["amount_vs_user_median"] = _safe_ratio(out["amount"], baseline_amount)

    # How busy is this 24-hour window against this user's usual level.
    baseline_count = _trailing(out["txn_count_24h"].astype(float), users, "mean")
    out["count_vs_user_typical"] = _safe_ratio(
        out["txn_count_24h"].astype(float), baseline_count)

    # How fast did this transaction follow the last one, against this user's
    # usual rhythm. Only defined where a previous transaction exists.
    baseline_gap = _trailing(gap, users, "median")
    ratio = _safe_ratio(gap.fillna(0.0), baseline_gap)
    out["gap_vs_user_typical"] = ratio.where(gap.notna(), NO_BASELINE_RATIO)
