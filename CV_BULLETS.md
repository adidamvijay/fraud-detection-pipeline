# CV bullets

Every bullet below was checked against the repository before it was written.
The evidence table further down names the file, test or captured output
behind each one. If you cannot point at the evidence in an interview, do not
use the bullet.

Two things to know before you paste any of this anywhere:

**The transaction data is synthetic.** Row counts like 29,178 describe a
generated dataset, not real payment volume. Say "synthetic" the first time
you mention a number. An interviewer who discovers it themselves will assume
you were hoping they wouldn't.

**Do not claim to have run Airflow.** The DAGs are written in Airflow 3
syntax and they compile, but neither has ever been executed. See the gaps
section at the end.

---

## Variant 1 — general software / backend

- Made every stage of a batch pipeline idempotent so a retry cannot corrupt
  data: both Snowflake loaders write to a transient staging table and `MERGE`
  on `TRANSACTION_ID`. Proved it by running each twice against identical
  input, getting `(29,178 inserted, 0 updated)` then `(0 inserted, 29,178
  updated)` with row counts unchanged.
- Wrote 51 tests covering validation rules, feature causality and the scoring
  path, then verified the suite could actually fail by mutating 11 behaviours
  one at a time in a temporary copy of the source. All 11 mutations were
  caught.
- Audited an inherited codebase and fixed what the audit found: three
  mutually incompatible definitions of one table, a `MERGE` matching on a
  column its own staging table did not have, five wrapper scripts that raised
  `ImportError` on execution, and absolute POSIX paths that made the pipeline
  unrunnable on the development machine. Each fix is a separate commit
  explaining the defect.
- Replaced a per-row pandas loop in feature computation with a vectorised
  time-based rolling window, and asserted the rewrite was behaviour
  preserving by diffing all four features across 29,178 rows — which
  surfaced a window-boundary discrepancy affecting one row that a benchmark
  alone would have missed.

## Variant 2 — data engineering

- Built and verified a batch pipeline from CSV to Snowflake to dashboard
  against a live account: 29,178 synthetic transactions through validation,
  normalisation, `RAW_TRANSACTIONS`, feature computation and `FRAUD_SCORES`,
  with no rows lost or duplicated at any stage.
- Made both warehouse loaders idempotent using a transient staging table and
  `MERGE` on the primary key, demonstrated by replaying identical input and
  showing 0 rows inserted and 29,178 updated on the second run. This is what
  makes an orchestrator retry safe.
- Designed a three-layer schema separating raw, staging and serving, with
  money as `NUMBER(12,2)` rather than `FLOAT` because amounts are summed over
  rolling windows, and both `EVENT_TIME` and `LOADED_AT` retained so
  late-arriving data stays distinguishable from on-time data.
- Wrote data-quality validation that routes rejected rows to a quarantine
  directory with a `failure_reason` column rather than dropping them, across
  six rules including timestamp parsing, negative amounts, duplicate keys and
  IP octet ranges.

## Variant 3 — data science

- Evaluated an unsupervised fraud detector properly and published the result
  even though it went against the model I had built: on a held-out temporal
  split, Isolation Forest reached PR-AUC 0.1719 against 0.1477 for simply
  sorting transactions by amount, while a random forest on identical features
  reached 0.7221.
- Diagnosed the failure structurally rather than guessing: the model caught
  30.8% of account-takeover fraud but 0% of card testing, because card
  testing is a conjunction of moderate deviations and axis-parallel isolation
  cannot express one. Predicted a tree ensemble would fix it, then measured
  85.7% card-testing recall from the random forest, and 7.1% from logistic
  regression, confirming the conjunction explanation rather than "supervised
  is just better".
- Improved features from measurement, not intuition: the original four were
  absolute quantities, so adding three expressed relative to each user's own
  trailing baseline took PR-AUC from 0.0232 to 0.1719 and precision from
  0.042 to 0.320 at a matched alert rate of roughly 25 alerts per 8,897
  transactions.
- Prevented feature leakage by construction and tested for it: every
  user-relative baseline applies `shift(1)` before its expanding
  aggregation, and tests assert a transaction's features are unchanged when
  all later transactions are removed, using fixtures where the causal and
  leaky answers differ. Deleting the shift makes two tests fail.

---

## Evidence for each bullet

| Claim | Where it comes from |
|---|---|
| 29,178 rows end to end, no loss or duplication | `docs/verified-snowflake.md`, row counts section |
| Idempotency `(29178, 0)` then `(0, 29178)` | `docs/verified-snowflake.md`, idempotency and query-history sections |
| Staging table + `MERGE` on primary key | `etl/ingest_to_snowflake.py`, `models/score_batch.py`, `sql/schema.sql` |
| 51 tests | `python -m unittest discover -s tests` — 11 features, 23 validation, 17 scoring |
| 11 of 11 mutations caught | `python tests/mutation_check.py` |
| Three incompatible table definitions, broken `MERGE`, five broken wrappers | commits `2c1903b`, `83374f4`; README "known defects" history |
| Behaviour-preserving vectorisation, one-row boundary discrepancy | `benchmarks/feature_complexity.py` (asserts all four features match), `models/features.py` `closed="both"` comment |
| Three-layer schema, `NUMBER(12,2)`, `EVENT_TIME` vs `LOADED_AT` | `sql/schema.sql` with the reasoning in comments |
| Six validation rules, quarantine with `failure_reason` | `etl/validate_data.py`, `tests/test_validation.py` |
| PR-AUC 0.1719 / 0.1477 / 0.7221, per-typology recall | `python models/evaluate.py`; stored in `data/scores/evaluation.json`; tabulated in `DESIGN.md` section 4 |
| 0.0232 → 0.1719, precision 0.042 → 0.320 | `DESIGN.md`, "Before and after the relative features" |
| `shift(1)` causality and its tests | `models/features.py` `_trailing`, `tests/test_features.py` |
| Fresh-install reproducibility | `docs/verified-install.md` |

---

## What an interviewer can ask you to prove on the spot

Ranked by how likely someone is to actually ask. For each, the exact thing to
open or run.

**1. "Show me the evaluation."** Very likely — it is the strongest claim.
Run `python models/evaluate.py`. Takes about 30 seconds and prints every
number in the data-science variant. If you have no machine, open
`DESIGN.md` section 4.

**2. "Prove the tests can fail."** Likely if you mention mutation testing,
and it is the sort of claim people enjoy attacking. Run
`python tests/mutation_check.py`. It prints 11 mutations and whether each was
caught. Say up front that it mutates a temporary copy, never the working
tree.

**3. "How do you know the loaders are idempotent?"** Open
`docs/verified-snowflake.md` and show the query-history table: four MERGE
statements, alternating 29,178 and 0 rows inserted. If the Snowflake trial is
still alive you can re-run `etl/ingest_to_snowflake.py` live.

**4. "Where is the leakage prevention?"** Open `models/features.py` and point
at `_trailing`: the `shift(1)` is one line. Then
`tests/test_features.py::TestNoSelfReference` for the fixture where causal
gives 10000/150 and leaky gives 10000/200.

**5. "Why Isolation Forest if the random forest is four times better?"**
This is the one you should want. `DESIGN.md` section 4, "So why is the
unsupervised model still the one that ships?" — labels arrive 30 to 90 days
late via chargebacks, are biased toward disputed transactions, and cannot
exist for a novel pattern. Then volunteer the counter-argument yourself: your
own evaluation flatters the supervised models by handing them labels
instantly, and low-value card testing is exactly the fraud least likely to be
disputed and therefore least likely to be labelled in real data.

**6. "Show me the speedup."** Careful here. Run
`benchmarks/feature_complexity.py`, but the ratio is noisy: repeated runs on
the same machine gave 107x, 110x, 319x and 344x, because the vectorised pass
takes 0.07 to 0.15 seconds and background load swings it. **Quote the stable
number instead**: the loop costs 0.49 to 0.70 ms per row and that barely
moves as history deepens, which is the actual finding. If you quote a single
multiple and they re-run it, you will be off by a factor of three.

**7. "Is the O(k²) claim real?"** Related trap, and the honest answer is
better than the obvious one. Going from 250 to 2,000 transactions per user
moves cost per row only from about 0.49 to 0.52 ms, so the quadratic term
contributes almost nothing at realistic depth. The win is a constant factor —
removing per-row Python and pandas overhead — not a change in growth rate.
Saying "I made it linear" is the version an interviewer can disprove with
your own benchmark.

**8. "Walk me through the schema."** `sql/schema.sql`. The reasoning is in
the comments: `NUMBER(12,2)` because amounts are summed over rolling windows
and binary floats accumulate error; `USER_ID` and `EVENT_TIME` denormalised
into `FRAUD_SCORES` so the dashboard does not join on every render, safe
because both are immutable; staging tables `TRANSIENT` because they are
truncated after each merge and never need fail-safe storage.

**9. "What does evaluating on synthetic data prove?"** `DESIGN.md`, the
circularity section. You wrote the fraud typologies and then wrote features
to detect them, so it measures whether the pipeline recovers a planted
signal. The evidence that it is not purely self-confirming: the shipped model
fails completely on one planted typology, beats sorting by amount by only
16%, and the supervised comparison concluded against the model you built.

**10. "Run it."** `python run_pipeline.py` — about 14 seconds, no credentials
needed. Then `python -m streamlit run dashboard/app.py`, which falls back to
local scores and says which source it used.

---

## Claims I could not support, and gaps you should know about

These are things that would have made good bullets and are not in the list,
because the repository does not back them.

**Airflow.** The DAGs are Airflow 3 syntax, they compile, and the Compose
YAML parses. Neither has ever been executed, because Airflow does not run
natively on Windows. You hold the Airflow 3 Fundamentals certification and
Airflow is on your CV, so the honest phrasing if you want to mention it at
all is "wrote Airflow 3 DAGs" — never "orchestrated with Airflow" or
"scheduled hourly scoring in production". If asked, say plainly that the DAGs
are written but you have not run them yet, and that the pipeline has so far
been run by invoking the stages directly. That answer costs you very little.
Running them for real is the single highest-value remaining task.

**Slack alerting.** `models/check_fraud_alerts.py` exists and has never been
run. Do not mention it.

**The feature store.** `FEATURE_STORE` exists in the schema and is empty.
Nothing reads it. Do not mention it.

**Any model-performance claim framed as business impact.** There is no
fraud-loss figure, no cost saving, no false-positive reduction against a
baseline system, because there is no baseline system and no real money. If a
CV template asks for impact, the honest substitute is the measurement itself:
"measured against a trivial baseline and two supervised alternatives".

**A single speedup multiple.** As above — the ratio moves by a factor of
three between runs. This was discovered by re-running the benchmark during a
verification pass, after `344x` had already been written into `DESIGN.md`.

**Scale.** Nothing here has been run above 29,178 rows. `DESIGN.md` has a
reasoned answer about what breaks at 100x, but it is reasoning, not
measurement, and should be presented that way.
