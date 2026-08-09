"""
Tests for the scoring path: threshold selection and models/train_local.

The scoring path is what writes FRAUD_SCORES, so its output shape is a
contract with the warehouse schema and with the dashboard. These tests cover
that contract, the threshold rule, and determinism.

Run:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.features import FEATURE_COLUMNS, compute_features  # noqa: E402
from models.thresholds import best_f1_threshold, threshold_for_alert_rate  # noqa: E402
from models.train_local import score, train  # noqa: E402


def synthetic_transactions(n_users=40, per_user=12, seed=0):
    """A small dataset with a handful of obvious outliers."""
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(n_users):
        base = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=int(rng.integers(0, 48)))
        for i in range(per_user):
            rows.append({
                "transaction_id": f"U{user}-{i}",
                "user_id": f"U{user:03d}",
                "event_time": base + pd.Timedelta(hours=6 * i),
                "amount": float(rng.lognormal(np.log(50), 0.4)),
                "label": 0,
            })
    df = pd.DataFrame(rows)

    # Plant a few unmistakable outliers so the model has something to find.
    fraud_idx = rng.choice(len(df), size=8, replace=False)
    df.loc[fraud_idx, "amount"] *= 60.0
    df.loc[fraud_idx, "label"] = 1
    return df


class TestBestF1Threshold(unittest.TestCase):

    def test_returns_none_without_positives(self):
        self.assertIsNone(best_f1_threshold(np.zeros(50), np.random.rand(50)))

    def test_separates_perfectly_separable_data(self):
        labels = np.array([0] * 90 + [1] * 10)
        scores = np.array([0.1] * 90 + [0.9] * 10)

        threshold, precision, recall, f1 = best_f1_threshold(labels, scores)

        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 1.0)
        self.assertTrue(0.1 < threshold <= 0.9)

    def test_reported_metrics_match_applying_the_threshold(self):
        rng = np.random.default_rng(1)
        labels = (rng.random(400) < 0.05).astype(int)
        scores = rng.random(400) + labels * 0.4

        threshold, precision, recall, _ = best_f1_threshold(labels, scores)

        predicted = scores >= threshold
        actual_precision = (predicted & (labels == 1)).sum() / predicted.sum()
        actual_recall = (predicted & (labels == 1)).sum() / labels.sum()

        self.assertAlmostEqual(precision, actual_precision, places=9,
                               msg="reported precision does not match the threshold")
        self.assertAlmostEqual(recall, actual_recall, places=9,
                               msg="reported recall does not match the threshold")


class TestAlertRateThreshold(unittest.TestCase):

    def test_flags_approximately_the_requested_share(self):
        scores = np.random.default_rng(2).random(10_000)
        for rate in (0.001, 0.01, 0.05):
            with self.subTest(rate=rate):
                cut = threshold_for_alert_rate(scores, rate)
                actual = (scores >= cut).mean()
                self.assertAlmostEqual(actual, rate, delta=0.002)


class TestScoringOutput(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_transactions()
        cls.features = compute_features(cls.df)
        cls.labels = cls.features.merge(
            cls.df[["transaction_id", "label"]], on="transaction_id")["label"]
        cls.model = train(cls.features, contamination=0.02, seed=42)

    def test_one_row_per_transaction(self):
        scored, _ = score(self.features, self.model, "v-test", self.labels)
        self.assertEqual(len(scored), len(self.features))
        self.assertEqual(set(scored["transaction_id"]),
                         set(self.features["transaction_id"]))

    def test_output_columns_match_the_fraud_scores_contract(self):
        scored, _ = score(self.features, self.model, "v-test", self.labels)
        # These are the columns sql/schema.sql expects, lowercased.
        for column in ("transaction_id", "user_id", "event_time", "score",
                       "flagged", "model_version", "scored_at"):
            self.assertIn(column, scored.columns)

    def test_score_is_normalised_to_unit_interval(self):
        scored, _ = score(self.features, self.model, "v-test", self.labels)
        self.assertGreaterEqual(scored["score"].min(), 0.0)
        self.assertLessEqual(scored["score"].max(), 1.0)

    def test_flag_is_boolean(self):
        scored, _ = score(self.features, self.model, "v-test", self.labels)
        self.assertEqual(scored["flagged"].dtype, bool)

    def test_model_version_is_recorded_on_every_row(self):
        scored, _ = score(self.features, self.model, "v-test", self.labels)
        self.assertTrue((scored["model_version"] == "v-test").all(),
                        "a scored row must be traceable to the model that made it")

    def test_flag_is_applied_to_the_raw_score_not_the_scaled_one(self):
        """
        The min-max scaling depends on the batch, so a threshold applied to
        the scaled score would mean something different on every run.
        """
        scored, info = score(self.features, self.model, "v-test", self.labels)
        expected = scored["raw_score"] >= info["threshold"]
        pd.testing.assert_series_equal(scored["flagged"], expected,
                                       check_names=False)

    def test_threshold_rule_is_reported(self):
        _, info = score(self.features, self.model, "v-test", self.labels)
        self.assertIn("F1", info["rule"])

    def test_falls_back_when_no_labels_are_available(self):
        _, info = score(self.features, self.model, "v-test", labels=None)
        self.assertIn("contamination", info["rule"],
                      "without labels the job must say the threshold is a fallback")

    def test_scoring_is_deterministic(self):
        first, _ = score(self.features, self.model, "v-test", self.labels)
        second, _ = score(self.features, self.model, "v-test", self.labels)
        pd.testing.assert_series_equal(first["score"], second["score"])

    def test_training_is_deterministic_for_a_fixed_seed(self):
        a = train(self.features, contamination=0.02, seed=7)
        b = train(self.features, contamination=0.02, seed=7)
        X = self.features[FEATURE_COLUMNS].fillna(0)
        np.testing.assert_allclose(a.decision_function(X), b.decision_function(X))

    def test_the_model_ranks_planted_outliers_above_the_median(self):
        """
        Not a performance claim, a sanity check. If obvious outliers do not
        rank above the median, the scoring path is wired up wrong.
        """
        scored, _ = score(self.features, self.model, "v-test", self.labels)
        merged = scored.merge(self.df[["transaction_id", "label"]],
                              on="transaction_id")
        fraud_mean = merged.loc[merged["label"] == 1, "score"].mean()
        legit_mean = merged.loc[merged["label"] == 0, "score"].mean()
        self.assertGreater(fraud_mean, legit_mean)


class TestFeatureContract(unittest.TestCase):
    """The scoring path depends on features having a stable shape."""

    def test_every_declared_feature_is_produced(self):
        features = compute_features(synthetic_transactions(n_users=5, per_user=4))
        for column in FEATURE_COLUMNS:
            self.assertIn(column, features.columns)

    def test_no_nulls_reach_the_model(self):
        features = compute_features(synthetic_transactions(n_users=5, per_user=4))
        self.assertFalse(features[FEATURE_COLUMNS].isna().any().any(),
                         "a null feature would become 0 silently at scoring time")


if __name__ == "__main__":
    unittest.main()
