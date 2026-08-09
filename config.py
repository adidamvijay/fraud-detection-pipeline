"""
Every filesystem path and shared constant the pipeline uses.

Why this file exists
--------------------
Paths used to be declared independently in each script, and they disagreed.
etl/ingest_local.py wrote its output to a relative "data/processed", while
etl/ingest_to_snowflake.py read from an absolute "/project/data/processed".
Those are different directories, so the validated data never reached the
warehouse loader. The pipeline was severed by a path mismatch that no single
file could reveal.

Absolute POSIX paths like /project/data/outbox also made the pipeline
unrunnable on Windows: they resolve to C:\\project\\data\\outbox, a directory
that only existed inside the Docker mount the project was developed in.

Everything is now derived from the location of this file, so the pipeline
runs from any working directory, on any OS, with or without Docker.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTBOX_DIR = DATA_DIR / "outbox"            # generated, not yet validated
VALIDATED_DIR = DATA_DIR / "validated"      # passed validation
BAD_RECORDS_DIR = DATA_DIR / "bad_records"  # failed validation, kept for inspection
PROCESSED_DIR = DATA_DIR / "processed"      # normalised, ready to load
SCORES_DIR = DATA_DIR / "scores"            # local scoring output
ARCHIVE_DIR = DATA_DIR / "archive"          # inputs consumed by a completed stage

MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"

# Which fraud typology each injected fraud row belongs to. Evaluation
# metadata, written by the generator and read only by models/evaluate.py.
# Deliberately not part of the transaction schema: in a real system no such
# column would exist, and letting it into the pipeline would invite using it.
GROUND_TRUTH_PATH = DATA_DIR / "fraud_ground_truth.csv"

# Stage inputs are moved here after that stage succeeds. One subdirectory per
# stage, so it stays clear which step consumed which file.
ARCHIVE_OUTBOX_DIR = ARCHIVE_DIR / "outbox"
ARCHIVE_VALIDATED_DIR = ARCHIVE_DIR / "validated"

ALL_DIRS = [
    OUTBOX_DIR, VALIDATED_DIR, BAD_RECORDS_DIR, PROCESSED_DIR,
    SCORES_DIR, ARCHIVE_OUTBOX_DIR, ARCHIVE_VALIDATED_DIR, MODEL_DIR,
]

# The transaction schema, in the order columns are written to disk. Shared so
# the generator, the validator and the loader cannot drift apart.
TRANSACTION_COLUMNS = [
    "transaction_id", "user_id", "event_time", "amount",
    "merchant_id", "merchant_country", "txn_type",
    "device_id", "ip_address", "label",
]

# The four features the model is trained and scored on.
FEATURE_COLUMNS = [
    "total_amount_24h", "txn_count_24h", "avg_amount_24h", "hours_since_last_txn",
]


def ensure_dirs():
    """Create every pipeline directory. Safe to call repeatedly."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def load_env():
    """
    Load .env from the repository root.

    Previously two scripts called load_dotenv("/opt/airflow/.env"), a path
    that only exists inside the Airflow container, so running them anywhere
    else silently produced a connection with every credential set to None.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        return True
    return False


def snowflake_args():
    """
    Connection arguments from the environment.

    Returns None for the whole set if the three required variables are
    missing, so callers can fall back to a local path rather than failing
    with an unhelpful authentication error.
    """
    load_env()
    required = {
        "user": os.getenv("SNOW_USER"),
        "password": os.getenv("SNOW_PWD"),
        "account": os.getenv("SNOW_ACCOUNT"),
    }
    if not all(required.values()):
        return None

    args = dict(required)
    args["database"] = os.getenv("SNOW_DATABASE", "FRAUD_DB")
    args["schema"] = os.getenv("SNOW_SCHEMA", "PUBLIC")
    args["warehouse"] = os.getenv("SNOW_WAREHOUSE", "COMPUTE_WH")
    role = os.getenv("SNOW_ROLE")
    if role:
        args["role"] = role
    return args


def missing_snowflake_vars():
    """Names of the required Snowflake variables that are not set."""
    load_env()
    return [n for n in ("SNOW_USER", "SNOW_PWD", "SNOW_ACCOUNT") if not os.getenv(n)]
