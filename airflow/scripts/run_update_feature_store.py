import os
import sys

sys.path.append(os.path.abspath("."))

from models.update_feature_store import update_feature_store

if __name__ == "__main__":
    print("Updating Feature Store...")
    update_feature_store()
    print("Feature Store Update Completed.")
