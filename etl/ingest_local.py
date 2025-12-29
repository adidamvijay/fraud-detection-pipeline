# etl/ingest_local.py
import os
import glob
import pandas as pd
import shutil
from datetime import datetime

OUTBOX = "/project/data/validated"
PROCESSED = "data/processed"
ARCHIVE = os.path.join(OUTBOX, "archived")
os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(ARCHIVE, exist_ok=True)

EXPECTED = ["transaction_id","user_id","event_time","amount",
            "merchant_id","merchant_country","txn_type","device_id","ip_address","label"]

def normalize_df(df):
    # normalize column names to lowercase and strip
    df.columns = [str(c).strip().lower() for c in df.columns]

    # ensure expected columns exist
    for c in EXPECTED:
        if c not in df.columns:
            df[c] = None

    # parse event_time robustly
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')

    # reorder columns to stable expected ordering
    df = df[EXPECTED]
    return df

def process_outbox():
    files = sorted(glob.glob(os.path.join(OUTBOX, "*.csv")))
    if not files:
        print("No files in outbox.")
        return []

    processed_paths = []
    for fpath in files:
        base = os.path.basename(fpath)
        try:
            print("Reading", fpath)
            df = pd.read_csv(fpath)
        except Exception as e:
            print("Failed reading", fpath, " — skipping. Err:", e)
            # move bad file to archive so it doesn't block
            shutil.move(fpath, os.path.join(ARCHIVE, base))
            continue

        df = normalize_df(df)
        dest = os.path.join(PROCESSED, base)
        df.to_csv(dest, index=False)
        # move original to archive
        shutil.move(fpath, os.path.join(ARCHIVE, base))
        processed_paths.append(dest)
        print(f"Processed {fpath} -> {dest} (rows: {len(df)})")
    return processed_paths

if __name__ == "__main__":
    moved = process_outbox()
    print("Done. Processed files:", moved)
