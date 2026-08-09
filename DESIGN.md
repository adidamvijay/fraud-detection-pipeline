# Design

Status: 9 August 2026. Numbers in this document come from
`python models/evaluate.py` on this repository and are reproducible from
seed 42. Sections describing Airflow are marked as design, not as behaviour
that has been observed, because the DAGs have not been run.

## 1. What this system does

This is a batch pipeline that scores card transactions for fraud. Synthetic
transactions are generated as daily CSV files, validated against schema and
format rules that route bad rows to a quarantine directory rather than
dropping them, and normalised into a warehouse-ready layer. For each
transaction it computes seven features describing that user's recent
behaviour, four absolute and three expressed relative to the user's own
trailing baseline, then scores every transaction with an Isolation Forest and
writes the result to a table the dashboard reads. It is designed to run
hourly for scoring and daily for retraining under Apache Airflow, against
Snowflake. The interesting part is not the model, it is that the model is
measured against labelled ground truth with a held-out temporal split, and
the measurement says the model is only slightly better than sorting
transactions by amount.

## 2. Architecture as built

```
etl/generate_transactions.py   synthetic transactions -> data/outbox/
        |                      writes typology labels to a separate file
etl/validate_data.py           schema/type/format rules
        |                      -> data/validated/ and data/bad_records/
etl/ingest_local.py            normalise -> data/processed/
        |
models/features.py             seven per-user rolling features
        |
models/train_local.py          Isolation Forest -> models/artifacts/
        |                      scores -> data/scores/scores.csv
models/evaluate.py             held-out evaluation -> data/scores/evaluation.json
```

`run_pipeline.py` runs the local chain end to end. Measured: 29,178
transactions through four stages in 9.9 seconds.

### Real and verified

Everything above. The local path runs on Windows with no warehouse
connection, which is deliberate: the model and its evaluation must stay
reproducible after a Snowflake trial expires.

### Written but never executed

- `etl/ingest_to_snowflake.py`, `models/score_batch.py`,
  `models/feature_and_train.py` — the warehouse read and write paths. They
  compile and their column names now agree with `sql/schema.sql`, but no
  Snowflake account has been connected.
- `airflow/dags/*.py` — two DAGs, in Airflow 2 syntax on a 2.9.1 image.
  Never executed.
- `airflow/scripts/run_*.py` — five wrappers that import functions which do
  not exist. Four are dead code the DAGs never call.
- `models/check_fraud_alerts.py` — Slack alerting.
- `dashboard/app.py` — runs, but its time-series chart is still fabricated
  (see limitations).

### Not built

Model monitoring, backfill tooling, an API, CI.

## 3. Data model

Defined in `sql/schema.sql`. Three layers.

**`RAW_TRANSACTIONS`** — validated transactions as loaded. Keyed on
`TRANSACTION_ID VARCHAR(36)`, a UUID.

`AMOUNT` is `NUMBER(12,2)`, not `FLOAT`. Binary floating point cannot
represent 0.10 exactly, and the pipeline sums amounts over 24-hour windows,
so float error would accumulate into the features themselves.

`EVENT_TIME` is when the transaction happened; `LOADED_AT` defaults to
`CURRENT_TIMESTAMP()` and records when the row reached the warehouse. Keeping
both is what makes a backfill auditable: you can tell a transaction that
happened last Tuesday and arrived on time from one that happened last Tuesday
and arrived three days late.

`IP_ADDRESS` is `VARCHAR(45)` so an IPv6 address fits without a migration.
`LABEL` exists only because the data is synthetic; in a real system ground
truth arrives weeks later from chargebacks and would live in its own table
with its own timestamp.

**`FRAUD_SCORES`** — one row per scored transaction, keyed on
`TRANSACTION_ID`. It carries `USER_ID` and `EVENT_TIME` duplicated from
`RAW_TRANSACTIONS`. That denormalisation is deliberate: the dashboard groups
by event time, and without these columns every chart render would join back
to the raw table. Both columns are immutable, so the usual argument against
denormalisation does not apply.

`SCORE` is `FLOAT` because it is a model output, not money. `SCORED_AT`
records when the model ran, separately from `EVENT_TIME`; together they
answer "when did the fraud happen" and "how long did we take to see it".

**`FRAUD_SCORES_LOAD`** — a `TRANSIENT` staging table. The scoring job writes
here, MERGEs into `FRAUD_SCORES` on `TRANSACTION_ID`, then truncates.
Transient because it is emptied every run and never needs Snowflake's 7-day
fail-safe storage, which is billable.

All timestamps are `TIMESTAMP_NTZ` holding UTC. Conversion happens at the
display boundary only, so there is exactly one place to get it wrong.

Snowflake accepts `PRIMARY KEY` but does not enforce it. The constraints are
declared because they document intent and inform the optimiser; uniqueness is
actually enforced by the MERGE, not by the table.

## 4. The model

### Problem framing

Rank transactions by how unusual they are, so a limited review team looks at
the right ones first. This is a ranking problem with a severe class
imbalance, not a balanced classification problem. Fraud is 0.6% of rows.

### The seven features

Four absolute, describing the transaction and its trailing 24-hour window:

| Feature | What it is | Why |
|---|---|---|
| `total_amount_24h` | sum of amounts in the trailing 24h, inclusive | captures spending bursts in money terms |
| `txn_count_24h` | count in the same window | captures bursts in volume terms |
| `avg_amount_24h` | total / count | separates "many small" from "few large" |
| `hours_since_last_txn` | gap to this user's previous transaction | captures velocity; `-1.0` when there is no previous transaction |

Three relative, each an absolute quantity divided by that user's own trailing
baseline. These were added after measuring the first four:

| Feature | What it is | Why |
|---|---|---|
| `amount_vs_user_median` | amount / median of the user's *earlier* amounts | absolute amount cannot tell "large for this user" from "large" |
| `count_vs_user_typical` | `txn_count_24h` / mean of the user's earlier counts | a burst is defined relative to that user's normal rhythm |
| `gap_vs_user_typical` | `hours_since_last_txn` / median of the user's earlier gaps | ten minutes is routine for a daily user, extraordinary for a weekly one |

Baselines use a median for amount and gap so that one prior fraud does not
inflate the baseline and mask the next one, and an expanding window rather
than a fixed one so a burst cannot redefine the baseline it is measured
against.

Where a user has no prior history the ratio is set to `1.0`, meaning
"indistinguishable from this user's baseline". The alternative would flag
every new user by construction.

### Feature leakage: how each feature avoids it

Every baseline is built with `shift(1)` applied *before* the expanding
aggregation, in `models.features._trailing`. The current transaction's own
value therefore cannot enter its own baseline. That single call is the whole
causality guarantee.

Two properties are asserted in `tests/test_features.py`:

1. **No future information.** A transaction's features are identical whether
   computed over the full dataset or over only the transactions up to and
   including it. Tested for every prefix length, and again by appending a
   large later transaction and checking no earlier row changes.
2. **No self-reference.** Hand-computed fixtures where the causal and leaky
   baselines give different answers. For example, with prior amounts 100 and
   200 and a current amount of 10,000, the causal ratio is 10000/150 and the
   leaky one is 10000/200; the test asserts the former and explicitly asserts
   the value is *not* the latter.

These tests were verified by mutation: deleting the `shift(1)` makes two of
them fail, and restoring it makes them pass. A test that cannot fail is not
evidence.

There is one further leakage question worth naming. The evaluation computes
features over the whole dataset and *then* splits by time. That is safe
precisely because of property 1: a test-split row using history from the
training period is using its own past, which is what it would have in
production.

### Why Isolation Forest, given that labels exist

Honest answer: the labels exist only because the data is synthetic. In a real
fraud system, ground truth arrives weeks later via chargebacks, arrives only
for transactions someone bothered to dispute, and is therefore both delayed
and biased. An unsupervised detector can score a transaction the moment it
lands, and it can flag a fraud pattern nobody has labelled yet.

That is the argument. The measurement below partly undercuts it, and the
undercutting is the interesting part: a supervised model would very likely do
better on this data, and the evaluation shows exactly where the unsupervised
model fails.

Alternatives considered and rejected:

- **Supervised classifier (logistic regression, gradient boosting).** Would
  almost certainly beat this on the card-testing typology, because it can
  learn a conjunction of moderate deviations rather than relying on single-
  axis extremes. Rejected for the framing reason above, and because training
  on 175 positives invites overfitting. This is the alternative I would
  actually build next.
- **Local Outlier Factor.** Density-based and better at exactly the
  conjunction case Isolation Forest misses, but it does not produce a
  reusable model object for scoring new data without refitting.
- **Rule thresholds.** The `amount` baseline below is effectively this, and
  it performs nearly as well as the model, which is a real finding rather
  than a rhetorical device.
- **Deep learning.** Out of scope, and 29,000 rows with 175 positives does
  not justify it.

### How `contamination` was chosen

It was not. `contamination=0.02` was inherited, undocumented, and is wrong:
it flags 2% of whatever is scored regardless of content, so scoring a hundred
clean transactions still flags two. It is retained only as the *fitting*
parameter, and the decision threshold no longer comes from it. What it costs
is shown in the comparison rows below.

### Evaluation

Protocol, fixed before any numbers were seen:

- Temporal split, first 70% of days train, last 30% test. A random split
  would let the model learn from a user's future.
- The model never sees a label. Labels choose the threshold on the training
  split and measure on the test split.
- Threshold rule declared in advance: the point on the training-split
  precision/recall curve that maximises F1.
- Both feature sets run under this identical protocol.

Test split: 8,897 rows, **50 fraud** (0.562%). Fifty positives is a small
number and every test figure below carries real sampling noise; differences
of a few points are not meaningful.

#### Headline: PR-AUC on the test split

| Ranking | PR-AUC |
|---|---|
| random ordering | 0.0054 |
| **absolute features only (the original 4)** | **0.0232** |
| rank by transaction amount, largest first | 0.1477 |
| **absolute + user-relative (7 features)** | **0.1719** |

#### Before and after, at the chosen threshold

| | 4 absolute features | 7 features |
|---|---|---|
| PR-AUC | 0.0232 | **0.1719** |
| ROC-AUC | 0.7040 | **0.8882** |
| Precision | 0.042 | **0.320** |
| Recall | 0.020 | **0.160** |
| F1 | 0.027 | **0.213** |
| Caught | 1 of 50 | 8 of 50 |
| Alert rate | 0.27% | 0.28% |

At roughly the same alert volume, the relative features take precision from
4% to 32% and PR-AUC up 7.4x.

#### Recall by fraud typology, at the chosen threshold

| Typology | n | 4 features | 7 features |
|---|---|---|---|
| account takeover | 26 | 3.8% | **30.8%** |
| card testing | 14 | 0.0% | **0.0%** |
| subtle | 10 | 0.0% | 0.0% |

**The card-testing prediction failed.** `count_vs_user_typical` and
`gap_vs_user_typical` were added specifically to catch bursts, and at the
operating threshold they caught none. Per the rule set for this work, the
features were fixed in advance and not revised after seeing this.

It is not a bug in the feature. Card-testing rows have a median
`count_vs_user_typical` of 2.21 against 1.04 for legitimate traffic, and 67%
of them exceed the legitimate 95th percentile on that feature. The feature
computes what it claims.

The reason is the model. Card testing is a *conjunction* of moderate
deviations: somewhat elevated count, unusually small amounts, very short
gaps. No single axis is extreme — its `amount_vs_user_median` of 0.17 is low,
which is the "normal" direction, and its elevated count of 2.21 barely clears
the legitimate 95th percentile of 2.00. Isolation Forest isolates points
using axis-parallel splits and is strongest on single-axis extremes, which is
exactly why it does well on account takeover, whose
`amount_vs_user_median` of 14.09 towers over the legitimate 95th percentile
of 2.83. A conjunction of mild deviations is the case it handles worst.

This is the clearest argument in the repository for a supervised model, and
it is an argument produced by measurement rather than assumption.

#### The threshold, and what it costs

| Threshold rule | Precision | Recall | F1 | Alert rate |
|---|---|---|---|---|
| max F1 on the training split (chosen) | 0.320 | 0.160 | 0.213 | 0.28% |
| `contamination=0.02` (the old way) | 0.079 | 0.280 | 0.123 | 2.00% |

The old cutoff buys 12 points of recall by flagging seven times as many
transactions, at a quarter of the precision. Neither is obviously correct;
the right choice depends on how many alerts a review team can absorb. What
matters is that the chosen threshold is now a decision with a stated rule
behind it, rather than a number nobody picked.

The full precision/recall tradeoff is printed by `models/evaluate.py`. On the
seven-feature model, precision is 1.000 at 6% recall, 0.318 at 14%, and
collapses to 0.076 by 26%.

#### Against the trivial baseline

This is the number to be honest about. Sorting transactions by amount, with
no model at all, scores PR-AUC **0.1477**. The seven-feature model scores
**0.1719** — about 16% better.

The model earns its place, but narrowly, and mostly on the typology that
amount alone already finds. Anyone claiming this model is the valuable part
of the project has not read the baseline. The valuable part is the
measurement apparatus that makes this statement possible.

### Circularity: what this evaluation does not prove

I wrote the fraud typologies in the generator, and then wrote features
intended to detect those typologies. The evaluation therefore measures
whether the pipeline recovers a signal that was deliberately planted in it.
That is a legitimate and useful thing to measure — it verifies the features,
the model, the threshold logic and the metrics all work end to end — but it
is not evidence about real-world fraud detection performance, and no number
in this document should be read that way.

Three specific things it cannot tell you. It cannot tell you the model would
detect a fraud pattern I did not think to generate, since every pattern in
the data is one I invented. It cannot tell you the feature set is
well-chosen, because the features were designed against the same typologies
they are evaluated on. And it cannot tell you anything about calibration on
real transaction distributions, because real spending is not lognormal with
the parameters I picked.

The strongest honest claim is narrower and still worth making: given labelled
data, this repository measures a model correctly, on a held-out temporal
split, against a trivial baseline, with a threshold chosen by a stated rule.
That apparatus would work unchanged on real data. Two facts support the claim
that it is not merely reflecting its own assumptions: the model fails on one
of the three typologies I planted, and it barely beats sorting by amount.
Circular reasoning does not usually produce results that inconvenient.

## 5. The pipeline

### Schedules

Design, not observed behaviour. Two DAGs:

- `fraud_detection_daily_pipeline`, `@daily` — validate, load, update feature
  store, retrain, score, alert.
- `fraud_hourly_scoring_pipeline`, hourly — score new transactions, check
  alerts.

Both are written in Airflow 2 syntax (`schedule_interval`) on the 2.9.1
image. Migration to Airflow 3 is pending.

### Idempotency

Re-running a window must not double-count. Each stage achieves this
differently:

- **Generation** is seeded. The same seed writes byte-identical files, and
  output filenames are derived from the date, so a rerun overwrites rather
  than accumulates.
- **Validation and ingestion** move their input to `data/archive/` only after
  the outputs are written. A rerun finds an empty input directory and does
  nothing. Output filenames are deterministic, so reprocessing the same input
  overwrites in place.
- **Warehouse load and scoring** are keyed on `TRANSACTION_ID`. Scores go to
  the staging table, then `MERGE INTO FRAUD_SCORES ... ON
  tgt.TRANSACTION_ID = src.TRANSACTION_ID` updates matched rows and inserts
  the rest. Running the same window twice produces the same table.

The previous approach read the entire `alerts.csv` on every run to filter
already-scored IDs, which is O(all history) per run and fails as soon as two
jobs run concurrently. The MERGE moves that responsibility to the warehouse.

### Concurrency

If ingestion and scoring run at once, scoring may read a partially loaded
window and score fewer rows than exist. It will not corrupt anything, because
the MERGE is idempotent and the next run picks up what was missed. The daily
DAG sequences the two so this cannot happen within a run; it can happen
between the hourly scoring job and a long-running daily load. The clean fix
is an Airflow pool or a sensor on load completion, which is not built.

Two scoring jobs running concurrently is safe: both MERGE on the same key,
and the later write wins with the same value.

### Failure partway through

Stages are separate processes and separate Airflow tasks. A crash leaves
earlier stages' output on disk and later stages unrun. Because each stage is
idempotent, recovery is to rerun the DAG from the failed task; no manual
cleanup is required.

The failure mode that is *not* handled: a crash between writing the validated
output and moving the input to archive would cause that input to be validated
twice on the next run. Output names are deterministic so the result is
overwritten rather than duplicated, but the row would be counted twice in the
run's log output. Making that atomic needs a marker file or a transaction,
which is not built.

### Retries

`retries: 2` with a 5-minute delay on the daily DAG, `retries: 1` on the
hourly one. Appropriate because the likely failures are transient warehouse
connection errors. Retrying is only safe because the tasks are idempotent;
without the MERGE, a retry after a partial write would duplicate rows.

## 6. Interview questions

### Data and ML

**Q. Explain the precision/recall tradeoff when fraud is 0.1% of traffic.**
At that prevalence almost everything you flag is wrong unless the model is
very strong. On this data at 0.56% prevalence, the seven-feature model holds
precision 1.000 at 6% recall, 0.318 at 14%, and 0.076 by 26%. Precision falls
off a cliff as you reach for recall, because each additional true positive
costs an increasing number of false ones. The business question is not "what
is the best F1" but "how many alerts can the review team process per day",
which fixes the alert rate, and the threshold follows from that.

**Q. Why is accuracy useless here?**
Because a model that flags nothing scores 99.44% accuracy on this test split
and catches zero fraud. Accuracy is dominated by the majority class. Every
metric used here — precision, recall, F1, PR-AUC — is computed with respect
to the positive class specifically.

**Q. How did you handle class imbalance?**
Mostly by not needing to: the model is unsupervised and never sees labels, so
there is no loss function to reweight. Imbalance is handled in the
*evaluation* instead, by using PR-AUC rather than accuracy or ROC-AUC, by
reporting the trivial baselines the metrics must beat, and by reporting the
absolute count of positives in the test split (50) so the reader can judge
the noise.

**Q. Is there feature leakage in this pipeline?**
Not that I have found, and I tested for it rather than assuming. All three
relative features apply `shift(1)` before their expanding aggregation, so no
transaction is in its own baseline. `tests/test_features.py` asserts that a
row's features are unchanged when all later transactions are removed, and
uses hand-computed fixtures where the causal and leaky answers differ. I
verified the tests can fail by deleting the shift and watching two of them
break. The one design decision worth flagging is that features are computed
over the whole dataset before the temporal split, which is safe only because
of that first property.

**Q. Unsupervised or supervised, when you have labels?**
I chose unsupervised because real labels arrive weeks late via chargebacks,
only cover disputed transactions, and cannot catch a pattern nobody has
labelled. But my own evaluation argues against me: the model catches 30.8% of
account takeover and 0% of card testing, because card testing is a
conjunction of moderate deviations and Isolation Forest splits on single
axes. A supervised model can learn conjunctions. Given these labels I would
build one next and compare, and I would expect it to win on this data.

**Q. How would you detect model degradation in production?**
Three signals, none of which needs labels. The distribution of scores over
time — if the mean anomaly score drifts, either the traffic or the model has
changed. The alert rate at a fixed threshold, which should be roughly stable.
And the input feature distributions, particularly the relative features,
since a shift in what "normal" means for users invalidates the baselines.
Label-dependent metrics can only be computed once chargebacks arrive, so
they lag by weeks and are a backstop rather than an alarm.

**Q. PR-AUC or ROC-AUC at this imbalance?**
PR-AUC. ROC-AUC uses the false positive rate, whose denominator is the
enormous negative class, so large numbers of false positives barely move it.
The gap is visible in my own numbers: the four-feature model has a
respectable-looking ROC-AUC of 0.7040 while its PR-AUC is 0.0232, barely
above the random floor of 0.0054. ROC-AUC made a nearly useless model look
mediocre rather than useless.

**Q. What does evaluating on synthetic data not prove?**
That it works on real fraud. I wrote the typologies and then wrote features
to catch them, so the evaluation measures whether the pipeline recovers a
planted signal. It validates the apparatus, not the model's real-world skill.
It cannot show the model would catch a pattern I did not invent, and the
feature set was designed against the same typologies it is scored on. Two
results suggest it is not purely self-confirming: the model completely fails
on one planted typology, and it beats sorting by amount by only 16%.

### Engineering

**Q. Why this schema?**
Three layers separating raw, scored and staging. `AMOUNT` is `NUMBER(12,2)`
so money arithmetic is exact, since amounts are summed over rolling windows.
`FRAUD_SCORES` is keyed on `TRANSACTION_ID` and carries `USER_ID` and
`EVENT_TIME` denormalised from the raw table so the dashboard's time series
does not join on every render; both are immutable so the usual objection does
not apply. `EVENT_TIME` and `LOADED_AT` are kept separately so late-arriving
data is distinguishable from on-time data, which is what makes backfills
auditable.

**Q. What happens if ingestion and scoring run concurrently?**
Scoring may see a partially loaded window and score fewer rows than exist.
Nothing corrupts, because the write path MERGEs on `TRANSACTION_ID`, and the
next run picks up the rest. The daily DAG orders the tasks so it cannot
happen inside one run, but the hourly scoring job can overlap a slow daily
load. The proper fix is an Airflow pool or a sensor on load completion; it is
not built.

**Q. Is each task idempotent, and how?**
Yes, by three different mechanisms. Generation is seeded and writes
date-derived filenames. Validation and ingestion move inputs to an archive
only after outputs land, so a rerun finds nothing to do. The warehouse writes
MERGE on `TRANSACTION_ID`. The gap I know about: a crash between writing
validated output and archiving the input causes that file to be reprocessed,
which overwrites rather than duplicates but is logged twice.

**Q. What breaks at 100x volume?**
Roughly 3 million rows. Feature computation is fine — it is 0.07s for 29,000
rows and is a sort plus a linear pass. The first thing to break is memory:
`train_local.py` concatenates every processed CSV into one frame. The fix is
to push feature computation into SQL window functions and let Snowflake do
it, which is the natural home for it anyway. Second is `write_pandas`, which
should become a stage-and-COPY. Third, Isolation Forest training on 3 million
rows is unnecessary — it can be fitted on a sample, since it is estimating
the shape of normal behaviour rather than memorising rows.

**Q. Old feature loop versus the new one — what is the complexity?**
The old one grouped by user, iterated rows, and rebuilt a boolean mask over
that user's whole history for each row, so O(k²) per user with k
transactions. The new one sorts and uses a time-based rolling window: O(n log
n) then O(n). But I benchmarked it and the asymptotic story is not why it was
slow. Going from 250 to 2,000 transactions per user moved the loop's cost per
row only from 0.654ms to 0.699ms, so the O(k) term contributes almost
nothing at realistic depth. What costs 0.67ms per row is the fixed overhead
of doing anything per-row in Python and pandas. The measured speedup is 344x
and it is a constant-factor win. `benchmarks/feature_complexity.py`
reproduces it, and asserts both implementations agree on all four original
features.

**Q. How would you expose scores as an API?**
For lookups of already-scored transactions, a read endpoint over
`FRAUD_SCORES` keyed on `TRANSACTION_ID` — the table is already keyed for it.
Scoring a *new* transaction on request is the harder problem, because the
features need that user's trailing 24-hour history, so the request path needs
a low-latency store of per-user aggregates rather than a warehouse query.
That is what `FEATURE_STORE` is a sketch of. I have not built any of this,
and I would not claim the current design is ready for it.

**Q. Retry semantics?**
Two retries five minutes apart on the daily DAG, one on the hourly. The
expected failures are transient warehouse connection errors, which retries
fix. This is only safe because the tasks are idempotent — with the old
append-to-CSV write, a retry after a partial write would have duplicated
rows. Retries and idempotency are the same design decision.

**Q. How would you backfill?**
Generate or source the historical files, drop them in the outbox, and run the
DAG per day. `catchup=False` is set, so a backfill is deliberate rather than
something that happens by accident when the scheduler is restarted. The
MERGE means re-running a day that was already loaded is safe. The thing to
watch is that the relative features use expanding baselines over each user's
history, so backfilling *older* data than what is already scored would change
the correct feature values for later transactions, and those would need
rescoring. `LOADED_AT` versus `EVENT_TIME` is what lets you find them.

## 7. Limitations

**1. The model barely beats sorting by amount.** PR-AUC 0.1719 against
0.1477 for the trivial baseline, and it catches 0% of card-testing bursts.
The diagnosis is specific: Isolation Forest uses axis-parallel splits and
card testing is a conjunction of moderate deviations rather than a single
extreme. The fix is to train a supervised classifier on the same features and
compare honestly, accepting that it needs labels and will not generalise to
unlabelled patterns. That is the next thing I would build.

**2. The dashboard's time-series chart is still fabricated.** It spreads
every scoring timestamp evenly across 60 seconds with `np.linspace` and plots
the result as "transactions per minute". The data now has genuine time
spread, so the fix is to delete the hack and plot `EVENT_TIME`. Not yet done,
and it is listed as a known defect in the README rather than quietly left.

**3. Nothing has been run against Snowflake or Airflow.** The warehouse
scripts compile and their columns agree with the schema, and the DAGs are
defined, but no part of that path has been executed. Until it has, the
architecture diagram is a design and the local pipeline is the only thing
that has been observed working. Verifying it needs a Snowflake account and an
Airflow instance, and the DAGs need migrating to Airflow 3 syntax first.
