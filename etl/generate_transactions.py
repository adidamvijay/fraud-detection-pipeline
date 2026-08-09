"""
Synthetic transaction generator.

Writes one CSV per calendar day into data/outbox/, which is where
etl/validate_data.py picks files up.

Why this file was rewritten
---------------------------
The previous version called datetime.utcnow() inside a tight loop, so all 200
transactions it produced landed within 3 milliseconds of each other, and it
assigned user IDs at random from a pool of 9,999 so 198 of 200 users had
exactly one transaction. That made the four 24-hour rolling features
degenerate: txn_count_24h was always 1, hours_since_last_txn was always -1,
and total_amount_24h and avg_amount_24h were both just the transaction
amount. The model was a univariate outlier detector on amount wearing a
costume, and there was no time spread for the dashboard to plot.

What this version does differently
----------------------------------
1. Transactions are spread across a configurable number of days using a
   realistic hour-of-day pattern, so event times have genuine spread.
2. Users are drawn from a fixed population and each has a persistent
   behavioural profile: a daily transaction rate and a spending scale. Users
   therefore accumulate several transactions inside any 24-hour window, which
   is what makes the rolling features carry information.
3. Fraud is injected as episodes matching three known typologies, at a
   configurable rate below 1%.

Honesty note on the fraud patterns
----------------------------------
Two of the three fraud typologies below are, by construction, visible in the
feature space the model uses. That is a deliberate choice and it limits what
the evaluation proves: it measures whether Isolation Forest recovers a signal
that was deliberately planted, not whether it detects real fraud. The third
typology sits inside the legitimate amount distribution specifically so that
recall is not trivially 1.0 and the precision/recall curve has real shape.
See DESIGN.md for the full limitation.

Usage
-----
    python etl/generate_transactions.py
    python etl/generate_transactions.py --users 500 --days 30 --seed 42
"""

import argparse
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import GROUND_TRUTH_PATH, OUTBOX_DIR  # noqa: E402

DEFAULT_OUT_DIR = OUTBOX_DIR

COLUMNS = [
    "transaction_id", "user_id", "event_time", "amount",
    "merchant_id", "merchant_country", "txn_type",
    "device_id", "ip_address", "label",
]

MERCHANTS = ["Amazon", "Flipkart", "Myntra", "Uber", "Swiggy",
             "BigBasket", "Zomato", "IRCTC", "Croma", "Nykaa"]
COUNTRIES = ["IN", "US", "GB", "CA", "SG", "AE"]
TXN_TYPES = ["purchase", "withdrawal", "transfer"]
DEVICES = ["mobile", "web", "pos"]

# Relative transaction volume by hour of day (index 0 = midnight UTC).
# Quiet overnight, a morning ramp, a lunchtime peak and a larger evening peak.
# These are plausible retail-shaped weights, not measured from real data.
HOURLY_WEIGHTS = np.array([
    0.3, 0.2, 0.15, 0.15, 0.2, 0.4,   # 00-05
    0.8, 1.4, 2.0, 2.3, 2.4, 2.6,     # 06-11
    3.0, 2.8, 2.4, 2.3, 2.5, 2.9,     # 12-17
    3.4, 3.5, 3.1, 2.4, 1.5, 0.7,     # 18-23
])

# Monday..Sunday. Slight weekend uplift in consumer spending.
DOW_WEIGHTS = np.array([0.95, 0.95, 1.0, 1.05, 1.15, 1.25, 1.10])

# Share of fraudulent rows contributed by each typology.
ARCHETYPE_MIX = {"card_testing": 0.45, "account_takeover": 0.35, "subtle": 0.20}


def build_user_profiles(rng, n_users):
    """
    Give every user a persistent profile.

    daily_rate   mean transactions per day, Gamma-distributed so most users
                 are light and a few are heavy
    spend_scale  the user's typical transaction amount, lognormal across the
                 population so spending power varies by an order of magnitude
    home_country the country most of their transactions occur in
    ip_prefix    first two octets, stable per user
    """
    user_ids = np.array([f"U{i:05d}" for i in range(n_users)])
    return {
        "user_id": user_ids,
        "daily_rate": rng.gamma(shape=2.0, scale=0.9, size=n_users),
        "spend_scale": rng.lognormal(mean=np.log(60.0), sigma=0.8, size=n_users),
        "home_country": rng.choice(COUNTRIES, size=n_users, p=[0.55, 0.15, 0.10, 0.08, 0.07, 0.05]),
        "ip_a": rng.integers(1, 224, size=n_users),
        "ip_b": rng.integers(0, 256, size=n_users),
    }


def make_ips(rng, profiles, user_idx):
    """IP addresses are sticky per user in the first two octets."""
    n = len(user_idx)
    return np.char.add(
        np.char.add(
            np.char.add(profiles["ip_a"][user_idx].astype(str), "."),
            np.char.add(profiles["ip_b"][user_idx].astype(str), "."),
        ),
        np.char.add(
            np.char.add(rng.integers(0, 256, size=n).astype(str), "."),
            rng.integers(1, 255, size=n).astype(str),
        ),
    )


def make_countries(rng, profiles, user_idx):
    """Most transactions happen in the user's home country."""
    n = len(user_idx)
    home = profiles["home_country"][user_idx]
    away = rng.choice(COUNTRIES, size=n)
    return np.where(rng.random(n) < 0.90, home, away)


def generate_legitimate(rng, profiles, start_date, days):
    """
    Draw a per-user, per-day transaction count from a Poisson distribution
    whose rate is the user's own daily rate scaled by the day of week, then
    place each transaction inside its day using the hourly weights.
    """
    n_users = len(profiles["user_id"])
    day_offsets = np.arange(days)
    dow = np.array([(start_date + timedelta(days=int(d))).weekday() for d in day_offsets])

    rates = profiles["daily_rate"][:, None] * DOW_WEIGHTS[dow][None, :]
    counts = rng.poisson(rates)

    users_grid, days_grid = np.meshgrid(np.arange(n_users), day_offsets, indexing="ij")
    user_idx = np.repeat(users_grid.ravel(), counts.ravel())
    day_idx = np.repeat(days_grid.ravel(), counts.ravel())
    n = len(user_idx)

    hour_p = HOURLY_WEIGHTS / HOURLY_WEIGHTS.sum()
    hours = rng.choice(24, size=n, p=hour_p)
    seconds_within_hour = rng.integers(0, 3600, size=n)

    event_time = (
        pd.Timestamp(start_date)
        + pd.to_timedelta(day_idx, unit="D")
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(seconds_within_hour, unit="s")
    )

    amounts = rng.lognormal(mean=np.log(profiles["spend_scale"][user_idx]), sigma=0.6)

    return pd.DataFrame({
        "user_id": profiles["user_id"][user_idx],
        "event_time": event_time,
        "amount": np.round(amounts, 2),
        "merchant_id": rng.choice(MERCHANTS, size=n),
        "merchant_country": make_countries(rng, profiles, user_idx),
        "txn_type": rng.choice(TXN_TYPES, size=n, p=[0.75, 0.15, 0.10]),
        "device_id": rng.choice(DEVICES, size=n, p=[0.6, 0.3, 0.1]),
        "ip_address": make_ips(rng, profiles, user_idx),
        "label": 0,
    })


def _episode_frame(rng, profiles, user_idx, times, amounts, country_mode, fraud_type):
    """Assemble the rows for one fraud episode."""
    n = len(times)
    idx = np.full(n, user_idx)
    if country_mode == "foreign":
        home = profiles["home_country"][user_idx]
        pool = [c for c in COUNTRIES if c != home]
        countries = rng.choice(pool, size=n)
    else:
        countries = make_countries(rng, profiles, idx)

    return pd.DataFrame({
        "user_id": profiles["user_id"][idx],
        "event_time": times,
        "amount": np.round(amounts, 2),
        "merchant_id": rng.choice(MERCHANTS, size=n),
        "merchant_country": countries,
        "txn_type": rng.choice(TXN_TYPES, size=n),
        "device_id": rng.choice(DEVICES, size=n),
        "ip_address": make_ips(rng, profiles, idx),
        "label": 1,
        "fraud_type": fraud_type,
    })


def generate_fraud(rng, profiles, start_date, days, n_fraud_rows):
    """
    Inject fraud as episodes rather than isolated rows, because that is how
    fraud actually arrives and because isolated rows would not move the
    24-hour window features at all.

    card_testing      a burst of small transactions over minutes, testing
                      whether a stolen card is live. Moves txn_count_24h up
                      and hours_since_last_txn to near zero.
    account_takeover  one to three transactions far above the user's own
                      baseline, usually at an unusual hour and abroad. Moves
                      total_amount_24h and avg_amount_24h up.
    subtle            amounts only two to three times the user's baseline at
                      an ordinary hour. Sits inside the legitimate lognormal
                      tail on purpose, so the model cannot catch everything.
    """
    n_users = len(profiles["user_id"])
    window_seconds = days * 24 * 3600
    episodes = []
    produced = 0

    targets = {k: int(round(v * n_fraud_rows)) for k, v in ARCHETYPE_MIX.items()}
    targets["card_testing"] += n_fraud_rows - sum(targets.values())

    while produced < targets["card_testing"]:
        user_idx = int(rng.integers(0, n_users))
        burst = int(rng.integers(6, 16))
        burst = min(burst, targets["card_testing"] - produced)
        start = pd.Timestamp(start_date) + pd.to_timedelta(
            int(rng.integers(0, window_seconds)), unit="s")
        gaps = np.cumsum(rng.integers(20, 300, size=burst))
        times = start + pd.to_timedelta(gaps, unit="s")
        amounts = rng.uniform(1.0, 15.0, size=burst)
        episodes.append(_episode_frame(rng, profiles, user_idx, times, amounts,
                                       "any", "card_testing"))
        produced += burst

    produced = 0
    while produced < targets["account_takeover"]:
        user_idx = int(rng.integers(0, n_users))
        n = min(int(rng.integers(1, 4)), targets["account_takeover"] - produced)
        day = int(rng.integers(0, days))
        # Unusual hours: overnight, when the account holder is unlikely to notice.
        hours = rng.integers(0, 5, size=n)
        times = (pd.Timestamp(start_date)
                 + pd.to_timedelta(day, unit="D")
                 + pd.to_timedelta(hours, unit="h")
                 + pd.to_timedelta(rng.integers(0, 3600, size=n), unit="s"))
        amounts = profiles["spend_scale"][user_idx] * rng.uniform(8.0, 25.0, size=n)
        episodes.append(_episode_frame(rng, profiles, user_idx, times, amounts,
                                       "foreign", "account_takeover"))
        produced += n

    produced = 0
    while produced < targets["subtle"]:
        user_idx = int(rng.integers(0, n_users))
        n = min(int(rng.integers(1, 3)), targets["subtle"] - produced)
        hour_p = HOURLY_WEIGHTS / HOURLY_WEIGHTS.sum()
        times = (pd.Timestamp(start_date)
                 + pd.to_timedelta(int(rng.integers(0, days)), unit="D")
                 + pd.to_timedelta(rng.choice(24, size=n, p=hour_p), unit="h")
                 + pd.to_timedelta(rng.integers(0, 3600, size=n), unit="s"))
        amounts = profiles["spend_scale"][user_idx] * rng.uniform(2.0, 3.5, size=n)
        episodes.append(_episode_frame(rng, profiles, user_idx, times, amounts,
                                       "any", "subtle"))
        produced += n

    return pd.concat(episodes, ignore_index=True)


def summarise(df):
    """Print the distribution facts that make this data fit for the model."""
    per_user = df.groupby("user_id").size()
    day = df["event_time"].dt.floor("D")

    # Transactions in each user's own preceding 24 hours, which is what the
    # model's rolling features actually see.
    ordered = df.sort_values(["user_id", "event_time"])
    window_counts = (
        ordered.set_index("event_time")
        .groupby("user_id")["amount"]
        .rolling("24h").count()
    )

    legit = df.loc[df["label"] == 0, "amount"]
    fraud = df.loc[df["label"] == 1, "amount"]

    print("\n--- generated data summary ---")
    print(f"rows                  {len(df):,}")
    print(f"users                 {df['user_id'].nunique():,}")
    print(f"date span             {df['event_time'].min()}  ->  {df['event_time'].max()}")
    print(f"distinct event_time   {df['event_time'].nunique():,}"
          f"  ({df['event_time'].nunique() / len(df):.1%} of rows)")
    print(f"distinct days         {day.nunique()}")
    print()
    print(f"fraud rows            {int(df['label'].sum()):,}")
    print(f"fraud rate            {df['label'].mean():.4%}")
    print()
    print("transactions per user      "
          f"min {per_user.min()}  median {per_user.median():.0f}  "
          f"mean {per_user.mean():.1f}  max {per_user.max()}")
    print("txns in preceding 24h      "
          f"median {window_counts.median():.0f}  mean {window_counts.mean():.2f}  "
          f"max {window_counts.max():.0f}")
    print(f"  share of rows where that count is 1: {(window_counts == 1).mean():.1%}")
    print()
    print("amount percentiles         p50      p90      p99      max")
    print(f"  legitimate            {legit.quantile(.5):8.2f} {legit.quantile(.9):8.2f} "
          f"{legit.quantile(.99):8.2f} {legit.max():8.2f}")
    print(f"  fraud                 {fraud.quantile(.5):8.2f} {fraud.quantile(.9):8.2f} "
          f"{fraud.quantile(.99):8.2f} {fraud.max():8.2f}")
    overlap = ((fraud >= legit.quantile(.05)) & (fraud <= legit.quantile(.95))).mean()
    print(f"  fraud rows inside the legitimate 5th-95th percentile band: {overlap:.1%}")
    print()
    print("transactions by hour of day (UTC)")
    by_hour = df["event_time"].dt.hour.value_counts().sort_index()
    peak = by_hour.max()
    for hour, count in by_hour.items():
        print(f"  {hour:02d}  {'#' * int(40 * count / peak):<40} {count:,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--fraud-rate", type=float, default=0.006,
                        help="target share of rows labelled fraud (default 0.6%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--end-date", type=str, default=None,
                        help="last day to generate, YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    if not 0 <= args.fraud_rate < 0.5:
        raise SystemExit("--fraud-rate must be between 0 and 0.5")

    rng = np.random.default_rng(args.seed)

    end_date = (datetime.strptime(args.end_date, "%Y-%m-%d").date()
                if args.end_date else datetime.now(timezone.utc).date())
    start_date = end_date - timedelta(days=args.days - 1)

    profiles = build_user_profiles(rng, args.users)
    legit = generate_legitimate(rng, profiles, start_date, args.days)

    # Solve for the fraud count that lands on the requested overall rate.
    n_fraud = int(round(args.fraud_rate * len(legit) / (1 - args.fraud_rate)))
    fraud = generate_fraud(rng, profiles, start_date, args.days, n_fraud)

    df = pd.concat([legit, fraud], ignore_index=True)
    # Drawn from the seeded generator rather than uuid.uuid4(), which uses the
    # OS entropy source and would make every run produce different IDs. Same
    # seed must give byte-identical files or the tests cannot assert on them.
    df["transaction_id"] = [str(uuid.UUID(bytes=bytes(rng.bytes(16)), version=4))
                            for _ in range(len(df))]
    df = df.sort_values("event_time").reset_index(drop=True)

    # Episodes can run past the final midnight; keep the window closed.
    df = df[df["event_time"] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]

    # Typology labels go to a separate file, not into the transaction schema.
    # A real pipeline has no such column, and putting it in the CSV would let
    # it leak into features by accident.
    ground_truth = df.loc[df["label"] == 1, ["transaction_id", "fraud_type"]]
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    ground_truth.to_csv(GROUND_TRUTH_PATH, index=False)

    df = df[COLUMNS]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for day, chunk in df.groupby(df["event_time"].dt.floor("D")):
        path = out_dir / f"transactions_{day:%Y%m%d}.csv"
        chunk.to_csv(path, index=False)
        written.append((path, len(chunk)))

    summarise(df)
    print(f"\nwrote {len(written)} daily files to {out_dir}")
    print(f"  {written[0][0].name} ... {written[-1][0].name}")
    print(f"wrote fraud typology labels to {GROUND_TRUTH_PATH.name} "
          f"({len(ground_truth)} rows)")


if __name__ == "__main__":
    main()
