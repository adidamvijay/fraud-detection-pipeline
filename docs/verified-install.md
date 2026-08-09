# Verified install

The setup instructions in the README were checked by following them exactly,
from an empty virtual environment against a fresh clone of the repository.
This file records what happened, so the claim "it installs and runs" is
something a reader can check rather than take on trust.

Verified 9 August 2026, Windows 11, Python 3.11.3. No `.env` file, no
Snowflake account, no Docker, no network access after `pip install`.

## What was run

A fresh `git clone` into an empty directory, then only the commands the
README gives, in the order it gives them.

### 1. Create the environment and install

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Installed 48 packages. The seven direct dependencies resolved to:

```
joblib==1.5.3
numpy==2.4.6
pandas==3.0.5
plotly==6.9.0
python-dotenv==1.2.2
scikit-learn==1.9.0
streamlit==1.61.1
```

No build failures and no dependency conflicts.

### 2. Run the pipeline

```
.venv\Scripts\python.exe run_pipeline.py
```

```
Pipeline completed
  Generate transactions           4.99s
  Validate                        2.02s
  Ingest to processed             1.84s
  Train and score                14.57s
  total                          23.42s
```

Slower than the 14s quoted elsewhere in the repository because this run
included first-import overhead in a cold environment. The row counts were
identical.

### 3. Run the tests

```
.venv\Scripts\python.exe -m unittest discover -s tests
```

```
Ran 11 tests in 0.369s

OK
```

### 4. Run the evaluation

```
.venv\Scripts\python.exe models\evaluate.py
```

```
SUMMARY: PR-AUC on the test split
  random ordering                          0.0054
  Isolation Forest, 4 absolute features    0.0232
  rank by amount, no model                 0.1477
  Isolation Forest, 7 features             0.1719
  Logistic regression, 7 features          0.5188
  Random forest, 7 features                0.7221
```

Every figure matches the numbers quoted in README.md and DESIGN.md to four
decimal places, from a clean clone on a different environment. That is the
point of seeding the generator and the models.

### 5. Launch the dashboard

```
.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Loaded and rendered:

```
Source: local file scores.csv. Figures reflect the last completed scoring run.

Transactions scored   29,178
Flagged               105
Alert rate            0.36%
Max score             1.0000

Transactions and flags over time
daily buckets, plotted on transaction event time. 30 buckets covering 720 hours.

Score distribution
```

No console errors and no server errors.

## What this does not cover

The Snowflake path and the Airflow DAGs. Neither has been run against a live
warehouse or scheduler, and neither is exercised by anything above. They are
listed as unverified in the README status table.
