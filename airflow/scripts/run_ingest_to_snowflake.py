import os
import sys

sys.path.append(os.path.abspath("."))

from etl.ingest_to_snowflake import process_outbox

if __name__ == "__main__":
    print("Running Ingest to Snowflake...")
    process_outbox()
    print("Ingestion Completed.")
