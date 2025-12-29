import csv
import os
import uuid
import random
from datetime import datetime

OUT_DIR = "/project/data/outbox"
os.makedirs(OUT_DIR, exist_ok=True)

MERCHANTS = ["Amazon", "Flipkart", "Myntra", "Uber", "Swiggy"]
COUNTRIES = ["IN", "US", "GB", "CA"]
TXN_TYPES = ["purchase", "withdrawal", "transfer"]
DEVICES = ["mobile", "web", "pos"]

def generate_txn():
    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": f"U{random.randint(1,9999):04d}",
        "event_time": datetime.utcnow().replace(tzinfo=None).isoformat() + "Z",
        "amount": round(random.uniform(10, 5000), 2),
        "merchant_id": random.choice(MERCHANTS),
        "merchant_country": random.choice(COUNTRIES),
        "txn_type": random.choice(TXN_TYPES),
        "device_id": random.choice(DEVICES),
        "ip_address": f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
        "label": 1 if random.random() < 0.02 else 0  # 2% fraud
    }

def write_batch(filename="txns.csv", n=200):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(generate_txn().keys()))
        writer.writeheader()
        for _ in range(n):
            writer.writerow(generate_txn())

    print(f"Wrote {n} transactions → {path}")

if __name__ == "__main__":
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    write_batch(f"transactions_{timestamp}.csv", n=200)
