import os
import re
import pandas as pd
from datetime import datetime

RAW_DIR = "/project/data/outbox"
BAD_DIR = "/project/data/bad_records"
GOOD_DIR = "/project/data/validated"

os.makedirs(BAD_DIR, exist_ok=True)
os.makedirs(GOOD_DIR, exist_ok=True)

EXPECTED_COLUMNS = [
    "transaction_id", "user_id", "event_time", "amount",
    "merchant_id", "merchant_country", "txn_type",
    "device_id", "ip_address", "label"
]

IP_REGEX = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"


def validate_dataframe(df):
    errors = []

    # 1. Check missing columns
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
            df[col] = None

    # 2. Validate timestamp format
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    invalid_ts = df["event_time"].isna().sum()

    # 3. Validate numeric amount
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    invalid_amount = df["amount"].isna().sum()

    # 4. Validate IP format
    df["ip_valid"] = df["ip_address"].apply(lambda x: bool(re.match(IP_REGEX, str(x))))
    invalid_ip = (~df["ip_valid"]).sum()

    # Rows that failed validation
    df_invalid = df[
        df["event_time"].isna() |
        df["amount"].isna() |
        (~df["ip_valid"])
    ].copy()

    # Valid rows
    df_valid = df[
        ~(df["event_time"].isna() |
        df["amount"].isna() |
        (~df["ip_valid"]))
    ].copy()

    return df_valid, df_invalid, {
        "invalid_timestamps": invalid_ts,
        "invalid_amounts": invalid_amount,
        "invalid_ips": invalid_ip,
    }


def process_files():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if not files:
        print("No raw files to validate.")
        return

    for file in files:
        path = os.path.join(RAW_DIR, file)
        df = pd.read_csv(path)

        print(f"Validating: {file}")
        df_valid, df_invalid, stats = validate_dataframe(df)

        # Save valid rows
        good_path = os.path.join(GOOD_DIR, f"valid_{file}")
        df_valid.to_csv(good_path, index=False)

        # Save invalid rows separately
        if not df_invalid.empty:
            bad_path = os.path.join(BAD_DIR, f"invalid_{file}")
            df_invalid.to_csv(bad_path, index=False)
            print(f"Saved invalid rows → {bad_path}")

        print(f"Valid rows: {len(df_valid)}, Invalid rows: {len(df_invalid)}")
        print("Stats:", stats)


if __name__ == "__main__":
    process_files()
