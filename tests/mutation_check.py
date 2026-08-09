"""
Check that the test suite can actually fail.

A passing test suite proves nothing on its own. If a test would still pass
after the behaviour it claims to check is deleted, it is decoration. This
script breaks one specific behaviour at a time and asserts that a named test
notices.

It never edits the working tree. The source files are copied to a temporary
directory, the mutation is applied to the copy, and the tests are run there.

Run:
    python tests/mutation_check.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COPIED = ["config.py", "etl", "models", "tests"]

# (description, file, find, replace, test that must start failing)
MUTATIONS = [
    ("validation: stop rejecting negative amounts",
     "etl/validate_data.py",
     '"negative_amount": parsed_amount.notna() & (parsed_amount < 0),',
     '"negative_amount": parsed_amount.notna() & (parsed_amount < -1e18),',
     "tests.test_validation.TestRulesFire.test_negative_amount"),

    ("validation: weaken the IP pattern back to the original",
     "etl/validate_data.py",
     'IP_REGEX = (\n    r"^((25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)\\.){3}"\n'
     '    r"(25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)$"\n)',
     'IP_REGEX = r"^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$"',
     "tests.test_validation.TestRulesFire.test_ip_octet_above_255"),

    ("validation: stop detecting duplicate transaction ids",
     "etl/validate_data.py",
     '"duplicate_transaction_id": df["transaction_id"].duplicated(keep="first"),',
     '"duplicate_transaction_id": df["transaction_id"].isna() & False,',
     "tests.test_validation.TestRulesFire.test_duplicate_transaction_id"),

    ("validation: let helper columns leak into the validated output",
     "etl/validate_data.py",
     'valid = df.loc[~is_bad, TRANSACTION_COLUMNS].copy()',
     'valid = df.loc[~is_bad].copy()\n    valid["ip_valid"] = True',
     "tests.test_validation.TestValidRowsPass.test_output_columns_are_exactly_the_schema"),

    ("validation: stop naming which rule a row broke",
     "etl/validate_data.py",
     'invalid["failure_reason"] = reasons[is_bad]',
     'invalid["failure_reason"] = ""',
     "tests.test_validation.TestRulesFire.test_malformed_ip"),

    ("validation: stop archiving the input file",
     "etl/validate_data.py",
     'path.replace(ARCHIVE_OUTBOX_DIR / path.name)',
     'pass',
     "tests.test_validation.TestFileHandling.test_rerunning_does_nothing"),

    ("features: remove the shift that keeps a row out of its own baseline",
     "models/features.py",
     'shifted = series.groupby(group).shift(1)',
     'shifted = series',
     "tests.test_features.TestNoSelfReference.test_amount_baseline_excludes_current_row"),

    # Note this one is not a causality mutation, though it looks like one.
    # An expanding window still only looks backward, so the no-future-
    # information tests correctly keep passing. What it breaks is the window
    # width: the feature stops meaning "last 24 hours" and starts meaning
    # "all history". The boundary test is what catches it.
    ("features: widen the 24-hour window to unbounded history",
     "models/features.py",
     '.rolling("24h", closed="both")',
     '.expanding()',
     "tests.test_features.TestAbsoluteFeatures.test_window_excludes_older_transactions"),

    ("features: drop closed=both, excluding the exact 24-hour boundary",
     "models/features.py",
     '.rolling("24h", closed="both")',
     '.rolling("24h")',
     "tests.test_features.TestAbsoluteFeatures.test_rolling_window_is_inclusive_at_both_ends"),

    ("scoring: apply the flag to the batch-scaled score instead of the raw one",
     "models/train_local.py",
     'out["flagged"] = raw >= threshold',
     'out["flagged"] = normalised >= threshold',
     "tests.test_scoring.TestScoringOutput.test_flag_is_applied_to_the_raw_score_not_the_scaled_one"),

    ("scoring: silently use the F1 rule even when no labels exist",
     "models/train_local.py",
     'rule = f"contamination fallback ({CONTAMINATION_FALLBACK})"',
     'rule = "max F1 on the precision/recall curve"',
     "tests.test_scoring.TestScoringOutput.test_falls_back_when_no_labels_are_available"),
]


def build_copy(destination):
    for item in COPIED:
        source = PROJECT_ROOT / item
        target = destination / item
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def run_test(workdir, test_id):
    result = subprocess.run(
        [sys.executable, "-m", "unittest", test_id],
        cwd=workdir, capture_output=True, text=True)
    return result.returncode == 0


def main():
    print(f"{'mutation':<62}{'result':>16}")
    print("-" * 78)

    survived = []

    for description, relative_path, find, replace, test_id in MUTATIONS:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            build_copy(workdir)

            target = workdir / relative_path
            original = target.read_text(encoding="utf-8")

            if find not in original:
                print(f"{description:<62}{'PATTERN NOT FOUND':>16}")
                survived.append((description, "the mutation no longer applies"))
                continue

            target.write_text(original.replace(find, replace, 1), encoding="utf-8")

            if run_test(workdir, test_id):
                print(f"{description:<62}{'SURVIVED':>16}")
                survived.append((description, f"{test_id} still passed"))
            else:
                print(f"{description:<62}{'caught':>16}")

    print("-" * 78)
    caught = len(MUTATIONS) - len(survived)
    print(f"{caught} of {len(MUTATIONS)} mutations caught")

    if survived:
        print("\nMutations the suite did not notice:")
        for description, why in survived:
            print(f"  {description}\n    {why}")
        return 1

    print("\nEvery mutation was caught. The tests can fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
