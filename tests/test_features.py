"""
Tests for models/features.py, focused on causality.

The relative features divide a transaction by a baseline built from that
user's earlier transactions. Two ways that can silently break:

  1. The baseline includes transactions from the future.
  2. The baseline includes the current transaction itself.

Either one inflates measured performance and would not survive contact with
production, where the future does not exist yet. Both are tested here with
assertions that fail if the shift(1) in _trailing is removed.

Run:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.features import (  # noqa: E402
    ABSOLUTE_FEATURES, FEATURE_COLUMNS, RELATIVE_FEATURES, compute_features,
)


def make_txns(user, times, amounts):
    """Build a minimal transaction frame."""
    return pd.DataFrame({
        "transaction_id": [f"{user}-{i}" for i in range(len(times))],
        "user_id": user,
        "event_time": pd.to_datetime(times),
        "amount": amounts,
    })


class TestNoFutureInformation(unittest.TestCase):
    """A transaction's features must not depend on transactions after it."""

    def setUp(self):
        # One user, hourly transactions, with a large one at the end that
        # would badly distort any baseline that could see it.
        self.df = make_txns(
            "U1",
            [f"2026-01-0{d} 12:00:00" for d in range(1, 9)],
            [100.0, 110.0, 90.0, 105.0, 95.0, 100.0, 100.0, 99999.0],
        )

    def test_prefix_features_are_stable(self):
        full = compute_features(self.df).set_index("transaction_id")

        # Recompute using only the first k transactions. Everything the first
        # k rows are allowed to know is present in both, so their features
        # must be byte-for-byte the same.
        for k in range(1, len(self.df)):
            prefix = compute_features(self.df.head(k)).set_index("transaction_id")
            for txn_id in prefix.index:
                for col in FEATURE_COLUMNS:
                    self.assertAlmostEqual(
                        full.loc[txn_id, col], prefix.loc[txn_id, col], places=9,
                        msg=(f"{col} for {txn_id} changed when transactions "
                             f"after it were removed (prefix length {k}). "
                             "A feature is reading the future."))

    def test_appending_a_transaction_does_not_rewrite_history(self):
        before = compute_features(self.df).set_index("transaction_id")

        later = make_txns("U1", ["2026-01-09 12:00:00"], [50000.0])
        later["transaction_id"] = "U1-later"  # must not collide with U1-0
        extended = pd.concat([self.df, later], ignore_index=True)
        after = compute_features(extended).set_index("transaction_id")

        for txn_id in before.index:
            for col in FEATURE_COLUMNS:
                self.assertAlmostEqual(
                    before.loc[txn_id, col], after.loc[txn_id, col], places=9,
                    msg=f"{col} for {txn_id} changed when a later transaction arrived")


class TestNoSelfReference(unittest.TestCase):
    """A transaction's baseline must not contain the transaction itself."""

    def test_amount_baseline_excludes_current_row(self):
        # Prior amounts 100 and 200, then a large one.
        #   causal baseline = median(100, 200)        = 150 -> ratio 66.666...
        #   leaky  baseline = median(100, 200, 10000) = 200 -> ratio 50.0
        # The two differ, so this assertion detects a missing shift(1).
        df = make_txns("U1",
                       ["2026-01-01 12:00:00", "2026-01-02 12:00:00",
                        "2026-01-03 12:00:00"],
                       [100.0, 200.0, 10000.0])

        result = compute_features(df).set_index("transaction_id")
        ratio = result.loc["U1-2", "amount_vs_user_median"]

        self.assertAlmostEqual(
            ratio, 10000.0 / 150.0, places=6,
            msg="amount_vs_user_median used a baseline that includes the "
                "current transaction's own amount")
        self.assertNotAlmostEqual(
            ratio, 10000.0 / 200.0, places=6,
            msg="baseline matches the leaky calculation exactly")

    def test_count_baseline_excludes_current_row(self):
        # Four transactions one day apart, so every 24h window holds two
        # except the first. txn_count_24h = [1, 2, 2, 2].
        #   at row 3, causal baseline = mean(1, 2, 2) = 1.666...
        #   leaky  baseline = mean(1, 2, 2, 2)        = 1.75
        df = make_txns("U1",
                       ["2026-01-01 12:00:00", "2026-01-02 12:00:00",
                        "2026-01-03 12:00:00", "2026-01-04 12:00:00"],
                       [100.0, 100.0, 100.0, 100.0])

        result = compute_features(df).set_index("transaction_id")
        self.assertAlmostEqual(result.loc["U1-3", "count_vs_user_typical"],
                               2.0 / (5.0 / 3.0), places=6,
                               msg="count_vs_user_typical baseline includes its own row")

    def test_perturbing_a_row_does_not_change_earlier_rows(self):
        base = make_txns("U1",
                         [f"2026-01-0{d} 12:00:00" for d in range(1, 7)],
                         [100.0, 110.0, 90.0, 105.0, 95.0, 100.0])
        perturbed = base.copy()
        perturbed.loc[4, "amount"] = 250000.0

        before = compute_features(base).set_index("transaction_id")
        after = compute_features(perturbed).set_index("transaction_id")

        for txn_id in ["U1-0", "U1-1", "U1-2", "U1-3"]:
            for col in FEATURE_COLUMNS:
                self.assertAlmostEqual(
                    before.loc[txn_id, col], after.loc[txn_id, col], places=9,
                    msg=f"changing a later transaction altered {col} for {txn_id}")

        # And the change must actually propagate forward, otherwise this test
        # would pass against a feature set that ignores amount entirely.
        self.assertNotAlmostEqual(before.loc["U1-5", "amount_vs_user_median"],
                                  after.loc["U1-5", "amount_vs_user_median"],
                                  places=6,
                                  msg="a later row's baseline ignored the change")


class TestBaselineBehaviour(unittest.TestCase):

    def test_first_transaction_is_neutral_not_anomalous(self):
        df = make_txns("U1", ["2026-01-01 12:00:00"], [5000.0])
        result = compute_features(df).iloc[0]
        for col in RELATIVE_FEATURES:
            self.assertEqual(result[col], 1.0,
                             f"{col} should be neutral with no history, got {result[col]}")

    def test_users_do_not_contaminate_each_other(self):
        df = pd.concat([
            make_txns("U1", ["2026-01-01 12:00:00", "2026-01-02 12:00:00"],
                      [100.0, 100.0]),
            make_txns("U2", ["2026-01-01 13:00:00", "2026-01-02 13:00:00"],
                      [50000.0, 50000.0]),
        ], ignore_index=True)

        alone = compute_features(
            df[df.user_id == "U1"]).set_index("transaction_id")
        together = compute_features(df).set_index("transaction_id")

        for txn_id in alone.index:
            for col in FEATURE_COLUMNS:
                self.assertAlmostEqual(
                    alone.loc[txn_id, col], together.loc[txn_id, col], places=9,
                    msg=f"{col} for {txn_id} depends on another user's data")


class TestAbsoluteFeatures(unittest.TestCase):

    def test_rolling_window_is_inclusive_at_both_ends(self):
        # A transaction exactly 24 hours earlier must be inside the window.
        # pandas rolling defaults to excluding it; features.py passes
        # closed="both" to match the original implementation.
        df = make_txns("U1", ["2026-01-01 12:00:00", "2026-01-02 12:00:00"],
                       [100.0, 100.0])
        result = compute_features(df).set_index("transaction_id")
        self.assertEqual(result.loc["U1-1", "txn_count_24h"], 2,
                         "a transaction exactly 24h earlier fell outside the window")

    def test_window_excludes_older_transactions(self):
        df = make_txns("U1", ["2026-01-01 12:00:00", "2026-01-03 12:00:00"],
                       [100.0, 100.0])
        result = compute_features(df).set_index("transaction_id")
        self.assertEqual(result.loc["U1-1", "txn_count_24h"], 1)

    def test_feature_columns_are_all_present(self):
        df = make_txns("U1", ["2026-01-01 12:00:00"], [100.0])
        result = compute_features(df)
        for col in ABSOLUTE_FEATURES + RELATIVE_FEATURES:
            self.assertIn(col, result.columns)

    def test_missing_input_column_raises(self):
        df = make_txns("U1", ["2026-01-01 12:00:00"], [100.0]).drop(columns=["amount"])
        with self.assertRaises(ValueError):
            compute_features(df)


if __name__ == "__main__":
    unittest.main()
