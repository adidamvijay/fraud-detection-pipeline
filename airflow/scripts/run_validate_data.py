import os
import sys

sys.path.append(os.path.abspath("."))

from etl.validate_data import validate_outbox

if __name__ == "__main__":
    print("Running Data Validation...")
    validate_outbox()
    print("Validation Completed.")
