import os
import sys

sys.path.append(os.path.abspath("."))

from models.feature_and_train_local import run_training_pipeline

if __name__ == "__main__":
    print("Running Model Retraining...")
    run_training_pipeline()
    print("Model Retraining Completed.")
