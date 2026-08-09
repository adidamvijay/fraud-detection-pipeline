"""
Tests for etl/validate_data.py.

Every rejection rule gets a test that it fires, and a test that it names
itself in the failure_reason column. A validator that silently drops bad rows
is worse than no validator, because the pipeline then reports a clean run
while losing data.

Also covers the file-level behaviour: outputs written before the input is
archived, and a second run finding nothing left to do.

Run:
    python -m unittest discover -s tests -v
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import etl.validate_data as validate_module  # noqa: E402
from config import TRANSACTION_COLUMNS  # noqa: E402
from etl.validate_data import validate_dataframe  # noqa: E402

GOOD_ROW = {
    "transaction_id": "t-good",
    "user_id": "U00001",
    "event_time": "2026-08-01 10:00:00",
    "amount": "125.50",
    "merchant_id": "Amazon",
    "merchant_country": "IN",
    "txn_type": "purchase",
    "device_id": "mobile",
    "ip_address": "10.0.0.1",
    "label": 0,
}


def frame(*overrides):
    """A frame of otherwise-valid rows, each with the given fields replaced."""
    rows = []
    for i, override in enumerate(overrides):
        row = dict(GOOD_ROW)
        row["transaction_id"] = f"t-{i}"
        row.update(override)
        rows.append(row)
    return pd.DataFrame(rows)


class TestRulesFire(unittest.TestCase):
    """Each rule rejects the row it is supposed to reject."""

    def assert_rejected(self, override, expected_reason):
        df = frame({}, override)
        valid, invalid, stats = validate_dataframe(df)

        self.assertEqual(len(valid), 1, "the good row should have survived")
        self.assertEqual(len(invalid), 1, f"{expected_reason} did not reject the row")
        self.assertEqual(stats[expected_reason], 1,
                         f"{expected_reason} did not count the row")
        self.assertIn(expected_reason, invalid.iloc[0]["failure_reason"],
                      "the rejected row does not say which rule it broke")

    def test_unparseable_timestamp(self):
        self.assert_rejected({"event_time": "not-a-date"}, "invalid_event_time")

    def test_empty_timestamp(self):
        self.assert_rejected({"event_time": ""}, "invalid_event_time")

    def test_non_numeric_amount(self):
        self.assert_rejected({"amount": "twelve pounds"}, "invalid_amount")

    def test_negative_amount(self):
        self.assert_rejected({"amount": "-0.01"}, "negative_amount")

    def test_missing_transaction_id(self):
        self.assert_rejected({"transaction_id": None}, "missing_transaction_id")

    def test_malformed_ip(self):
        self.assert_rejected({"ip_address": "not.an.ip.address"}, "invalid_ip")

    def test_ip_octet_above_255(self):
        # The original pattern was \d{1,3} four times, which accepted this.
        self.assert_rejected({"ip_address": "999.999.999.999"}, "invalid_ip")

    def test_ip_with_too_few_octets(self):
        self.assert_rejected({"ip_address": "10.0.1"}, "invalid_ip")

    def test_ip_with_too_many_octets(self):
        self.assert_rejected({"ip_address": "10.0.0.1.5"}, "invalid_ip")

    def test_duplicate_transaction_id(self):
        df = frame({}, {})
        df.loc[1, "transaction_id"] = df.loc[0, "transaction_id"]
        valid, invalid, stats = validate_dataframe(df)

        self.assertEqual(len(valid), 1, "the first occurrence should be kept")
        self.assertEqual(len(invalid), 1, "the second occurrence should be rejected")
        self.assertEqual(stats["duplicate_transaction_id"], 1)
        self.assertIn("duplicate_transaction_id", invalid.iloc[0]["failure_reason"])


class TestValidRowsPass(unittest.TestCase):

    def test_a_clean_row_survives(self):
        valid, invalid, _ = validate_dataframe(frame({}))
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(invalid), 0)

    def test_boundary_ip_addresses_are_accepted(self):
        for ip in ("0.0.0.0", "255.255.255.255", "192.168.1.1", "8.8.8.8"):
            with self.subTest(ip=ip):
                valid, invalid, _ = validate_dataframe(frame({"ip_address": ip}))
                self.assertEqual(len(valid), 1, f"{ip} should be accepted")

    def test_zero_amount_is_accepted(self):
        # Zero is not negative. A refund-to-zero is odd but not malformed, and
        # inventing a rule the data does not need would reject real rows.
        valid, _, _ = validate_dataframe(frame({"amount": "0.00"}))
        self.assertEqual(len(valid), 1)

    def test_output_columns_are_exactly_the_schema(self):
        valid, _, _ = validate_dataframe(frame({}))
        self.assertEqual(list(valid.columns), TRANSACTION_COLUMNS,
                         "validated output must not carry helper columns")

    def test_helper_column_does_not_leak(self):
        # The previous version left an ip_valid column in the output, which
        # then flowed into the archived data.
        valid, _, _ = validate_dataframe(frame({}))
        self.assertNotIn("ip_valid", valid.columns)


class TestMultipleFailures(unittest.TestCase):

    def test_a_row_reports_every_rule_it_broke(self):
        valid, invalid, _ = validate_dataframe(
            frame({}, {"amount": "-5", "ip_address": "bad"}))

        self.assertEqual(len(invalid), 1)
        reason = invalid.iloc[0]["failure_reason"]
        self.assertIn("negative_amount", reason)
        self.assertIn("invalid_ip", reason)

    def test_no_valid_row_is_given_a_reason(self):
        valid, _, _ = validate_dataframe(frame({}, {"amount": "-5"}))
        self.assertNotIn("failure_reason", valid.columns)


class TestMissingColumns(unittest.TestCase):

    def test_missing_column_is_reported_not_crashed_on(self):
        df = frame({}).drop(columns=["ip_address"])
        valid, invalid, stats = validate_dataframe(df)

        self.assertEqual(stats["missing_columns"], ["ip_address"])
        # The column is created as null, so every row fails the IP rule rather
        # than the whole file failing to parse.
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(invalid), 1)

    def test_the_other_rules_still_run(self):
        df = frame({"amount": "-5"}).drop(columns=["ip_address"])
        _, invalid, stats = validate_dataframe(df)
        self.assertEqual(stats["negative_amount"], 1)


class TestFileHandling(unittest.TestCase):
    """Directory-level behaviour, run against temporary directories."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.saved = {name: getattr(validate_module, name) for name in
                      ("OUTBOX_DIR", "VALIDATED_DIR", "BAD_RECORDS_DIR",
                       "ARCHIVE_OUTBOX_DIR")}
        for name, sub in (("OUTBOX_DIR", "outbox"),
                          ("VALIDATED_DIR", "validated"),
                          ("BAD_RECORDS_DIR", "bad_records"),
                          ("ARCHIVE_OUTBOX_DIR", "archive")):
            path = self.tmp / sub
            path.mkdir(parents=True)
            setattr(validate_module, name, path)
        # ensure_dirs would recreate the real project directories.
        validate_module.ensure_dirs = lambda: None

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(validate_module, name, value)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_input(self, name, df):
        df.to_csv(validate_module.OUTBOX_DIR / name, index=False)

    def test_good_and_bad_rows_are_split_into_separate_files(self):
        self.write_input("day1.csv", frame({}, {"amount": "-5"}))
        validate_module.process_files()

        good = pd.read_csv(validate_module.VALIDATED_DIR / "valid_day1.csv")
        bad = pd.read_csv(validate_module.BAD_RECORDS_DIR / "invalid_day1.csv")
        self.assertEqual(len(good), 1)
        self.assertEqual(len(bad), 1)
        self.assertIn("negative_amount", bad.iloc[0]["failure_reason"])

    def test_input_is_archived_after_processing(self):
        self.write_input("day1.csv", frame({}))
        validate_module.process_files()

        self.assertFalse((validate_module.OUTBOX_DIR / "day1.csv").exists(),
                         "input should have moved out of the outbox")
        self.assertTrue((validate_module.ARCHIVE_OUTBOX_DIR / "day1.csv").exists(),
                        "input should be in the archive")

    def test_rerunning_does_nothing(self):
        """Idempotency: a second run finds an empty outbox."""
        self.write_input("day1.csv", frame({}))
        self.assertEqual(validate_module.process_files(), 1)
        self.assertEqual(validate_module.process_files(), 0,
                         "a second run should find nothing to process")

    def test_no_bad_records_file_when_everything_is_clean(self):
        self.write_input("day1.csv", frame({}))
        validate_module.process_files()
        self.assertEqual(list(validate_module.BAD_RECORDS_DIR.glob("*.csv")), [],
                         "a clean file should not produce an empty bad_records file")


if __name__ == "__main__":
    unittest.main()
