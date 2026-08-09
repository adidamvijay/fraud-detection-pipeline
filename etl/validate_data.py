"""
Data quality validation.

Reads CSVs from data/outbox/, splits each file into rows that pass validation
and rows that do not, and writes them to data/validated/ and
data/bad_records/. The source file is moved to data/archive/outbox/ once it
has been processed, so re-running does not redo completed work.

Rules applied
-------------
1. Every expected column is present. A missing column is created as null so
   the rest of the file can still be checked rather than failing outright.
2. event_time parses as a timestamp.
3. amount parses as a number and is not negative.
4. ip_address matches a dotted-quad shape with each octet in 0-255.
5. transaction_id is present and unique within the file.

Rows failing any rule go to bad_records with a reason column, rather than
being dropped silently. A pipeline that discards bad data without recording
it cannot be audited.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    ARCHIVE_OUTBOX_DIR, BAD_RECORDS_DIR, OUTBOX_DIR,
    TRANSACTION_COLUMNS, VALIDATED_DIR, ensure_dirs,
)

# Each octet is 0-255. The previous pattern was \d{1,3} four times, which
# accepted 999.999.999.999.
IP_REGEX = (
    r"^((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$"
)


def validate_dataframe(df):
    """
    Return (valid_rows, invalid_rows, stats).

    invalid_rows carries a failure_reason column. Helper columns used during
    checking are removed from both outputs.
    """
    df = df.copy()

    missing_columns = [c for c in TRANSACTION_COLUMNS if c not in df.columns]
    for col in missing_columns:
        df[col] = None

    parsed_time = pd.to_datetime(df["event_time"], errors="coerce", format="mixed")
    parsed_amount = pd.to_numeric(df["amount"], errors="coerce")

    failures = {
        "invalid_event_time": parsed_time.isna(),
        "invalid_amount": parsed_amount.isna(),
        "negative_amount": parsed_amount.notna() & (parsed_amount < 0),
        "invalid_ip": ~df["ip_address"].astype(str).str.match(IP_REGEX, na=False),
        "missing_transaction_id": df["transaction_id"].isna(),
        "duplicate_transaction_id": df["transaction_id"].duplicated(keep="first"),
    }

    df["event_time"] = parsed_time
    df["amount"] = parsed_amount

    is_bad = pd.concat(failures.values(), axis=1).any(axis=1)

    # Name every rule a row broke, so bad_records explains itself.
    reasons = pd.Series([""] * len(df), index=df.index)
    for name, mask in failures.items():
        reasons = reasons.where(~mask, reasons.str.cat([name] * len(df), sep=";"))
    reasons = reasons.str.strip(";")

    valid = df.loc[~is_bad, TRANSACTION_COLUMNS].copy()
    invalid = df.loc[is_bad, TRANSACTION_COLUMNS].copy()
    invalid["failure_reason"] = reasons[is_bad]

    stats = {name: int(mask.sum()) for name, mask in failures.items()}
    stats["missing_columns"] = missing_columns
    return valid, invalid, stats


def process_files():
    """Validate every CSV in the outbox. Returns the number of files handled."""
    ensure_dirs()

    files = sorted(OUTBOX_DIR.glob("*.csv"))
    if not files:
        print(f"No files to validate in {OUTBOX_DIR}")
        return 0

    total_valid = total_invalid = 0

    for path in files:
        df = pd.read_csv(path)
        valid, invalid, stats = validate_dataframe(df)

        valid.to_csv(VALIDATED_DIR / f"valid_{path.name}", index=False)
        if not invalid.empty:
            invalid.to_csv(BAD_RECORDS_DIR / f"invalid_{path.name}", index=False)

        # Move the source only after both outputs are safely written.
        path.replace(ARCHIVE_OUTBOX_DIR / path.name)

        total_valid += len(valid)
        total_invalid += len(invalid)

        failed_rules = {k: v for k, v in stats.items() if k != "missing_columns" and v}
        detail = f"  rules triggered: {failed_rules}" if failed_rules else ""
        if stats["missing_columns"]:
            detail += f"  missing columns: {stats['missing_columns']}"
        print(f"{path.name}: {len(valid)} valid, {len(invalid)} invalid{detail}")

    print(f"\nvalidated {len(files)} files: {total_valid} valid rows, "
          f"{total_invalid} invalid rows")
    return len(files)


if __name__ == "__main__":
    process_files()
