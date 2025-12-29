import os
import sys

sys.path.append(os.path.abspath("."))

from models.feature_and_train import run_scoring_pipeline

if __name__ == "__main__":
    print("Running Incremental Scoring Pipeline...")
    run_scoring_pipeline()
    print("Scoring Completed.")
