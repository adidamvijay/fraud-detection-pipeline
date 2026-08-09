# Verified against a live Snowflake account

Evidence that the warehouse path in this repository actually ran, captured
at the time it ran. Kept so the claim stays checkable after the trial
account expires and the tables are gone.

- Date: 09 August 2026, run from Windows 11, Python 3.11.3
- Snowflake version: 10.27.101
- Region: AWS_AP_SOUTHEAST_1
- Warehouse: COMPUTE_WH (X-Small)
- Role: ACCOUNTADMIN
- Database: FRAUD_DB, schema PUBLIC

The account is a personal trial. Nobody else can connect to it, so these
numbers cannot be re-derived by a reader; that is the reason for writing
them down rather than saying 'it works'.

## Tables created by sql/schema.sql

```
TABLE_NAME                TYPE                ROWS        BYTES     
------------------------  ------------------  ----------  ----------
FEATURE_STORE             BASE TABLE          0           0         
FRAUD_SCORES              BASE TABLE          29178       1410560   
FRAUD_SCORES_LOAD         BASE TABLE          0           0         
RAW_TRANSACTIONS          BASE TABLE          29178       1701888   
RAW_TRANSACTIONS_LOAD     BASE TABLE          0           0         
```

## Row counts at each stage

Local stages, from the pipeline run that produced this data:

```
generated       29,178 transactions across 500 users, 30 days, seed 42
validated       29,178 valid, 0 rejected
normalised      29,178 rows written to data/processed as 30 daily files
```

Warehouse stages:

```
RAW_TRANSACTIONS        29,178 rows
FRAUD_SCORES            29,178 rows
RAW_TRANSACTIONS_LOAD        0 rows   (truncated after MERGE)
FRAUD_SCORES_LOAD            0 rows   (truncated after MERGE)
```

Every row that was generated reached the warehouse and every row in the
warehouse was scored. No row was lost or duplicated between stages.

## Idempotency, demonstrated

Both loaders were run twice against identical input. Snowflake's MERGE
returns (rows inserted, rows updated).

```
etl/ingest_to_snowflake.py
  first run    MERGE into RAW_TRANSACTIONS: (29178, 0)
  second run   MERGE into RAW_TRANSACTIONS: (0, 29178)
  RAW_TRANSACTIONS still holds 29,178 rows

models/score_batch.py
  first run    MERGE into FRAUD_SCORES: (29178, 0)
  second run   MERGE into FRAUD_SCORES: (0, 29178)
  FRAUD_SCORES still holds 29,178 rows
```

This is the property that makes an Airflow retry safe. The previous
implementation called write_pandas straight into the target table, which
appends, so a retry after a partial failure would have duplicated rows.

## FRAUD_SCORES contents

Summary:

```
TOTAL       FLAGGED    ALERT_PCT    MIN_SCORE    MAX_SCORE    MODELS  
----------  ---------  -----------  -----------  -----------  --------
29178       105        0.360        0.0          1.0          1       
```

Highest scoring transactions, joined back to the raw layer:

```
TRANSACTION_ID                          USER_ID    EVENT_TIME          AMOUNT      SCORE     FLAGGED  LABEL
--------------------------------------  ---------  ------------------  ----------  --------  -------  -----
8c3d75ea-c4d1-458e-83d8-6752b6c39952    U00034     2026-07-14 03:06    2575.99     1.0       True     1    
024a7bca-9097-4e1d-89c5-1af600388eed    U00059     2026-07-21 04:57    3879.80     0.925     True     1    
bae05c25-a875-4a5c-be55-3095559c05a0    U00034     2026-07-14 03:22    3387.97     0.9173    True     1    
1e9c5014-bc15-4be5-a09a-96ec18ca05c2    U00194     2026-08-08 04:30    3552.04     0.9148    True     1    
b3e16038-22d7-4f5f-9e06-f4ded3b1aab8    U00059     2026-07-21 04:06    4585.93     0.8995    True     1    
76d47ecb-c52e-4acf-97c1-6bea5fc42dfd    U00244     2026-07-11 19:52    127.36      0.8984    True     0    
645a8f2c-6beb-4730-b98b-87560563335a    U00346     2026-07-30 11:55    179.25      0.897     True     0    
a7241f40-f95d-4815-989f-92a9ff140793    U00346     2026-07-20 18:01    99.85       0.8885    True     0    
9ee6f273-3b76-4bf6-ac1f-22fe5cb8738b    U00342     2026-07-29 07:05    3674.55     0.8798    True     0    
e443c980-cc9a-4092-8fc8-812552f06649    U00346     2026-07-23 11:29    92.66       0.8797    True     0    
```

LABEL is the ground truth carried through from the generator. It is never
an input to the model.

Flagged transactions against ground truth, in-sample across all 30 days:

```
LABEL    N           FLAGGED   
-------  ----------  ----------
0        29003       81        
1        175         24        
```

These are in-sample figures over the full period and are not the
evaluation. The held-out metrics are in DESIGN.md.

Daily volume, showing the scored data has real time spread:

```
DAY           TXNS      FLAGGED  
------------  --------  ---------
2026-07-11    1116      3        
2026-07-12    977       4        
2026-07-13    802       3        
2026-07-14    844       4        
2026-07-15    924       0        
2026-07-16    958       3        
2026-07-17    1055      6        
2026-07-18    1112      3        
2026-07-19    1101      3        
2026-07-20    883       2        
```

(First 10 of 30 days.)

## Query history

From SNOWFLAKE.ACCOUNT_USAGE is not available on a trial without a delay,
so this comes from INFORMATION_SCHEMA.QUERY_HISTORY, which covers the
current session's recent activity.

```
QUERY_TYPE              N       SECONDS   
----------------------  ------  ----------
SELECT                  36      5.18      
TRUNCATE_TABLE          8       1.20      
CREATE_TABLE            5       0.94      
USE                     5       0.18      
CREATE                  5       0.43      
MERGE                   4       2.20      
SHOW                    4       0.22      
ALTER                   2       0.10      
CALL                    2       3.87      
GRANT                   2       0.16      
```

The MERGE statements, most recent first. ROWS_INSERTED counts new rows;
a MERGE that only updated shows zero here, which is the idempotent replay.

```
AT          INSERTED    PRODUCED    SECS     TARGET                        
----------  ----------  ----------  -------  ------------------------------
10:04:30    0           29178       0.51     MERGE INTO FRAUD_SCORES       
10:04:00    29178       29178       0.48     MERGE INTO FRAUD_SCORES       
10:03:35    0           29178       0.55     MERGE INTO RAW_TRANSACTIONS   
10:03:14    29178       29178       0.67     MERGE INTO RAW_TRANSACTIONS   
```

## What this does not show

- The Airflow DAGs. Neither has been executed; Airflow does not run
  natively on Windows and the Compose stack has never been started.
- Slack alerting. models/check_fraud_alerts.py has never been run.
- models/update_feature_store.py. FEATURE_STORE exists but is empty and
  nothing reads it.
