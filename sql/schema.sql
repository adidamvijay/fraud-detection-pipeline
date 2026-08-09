-- Canonical Snowflake schema for the fraud detection pipeline.
--
-- Three layers:
--   RAW_TRANSACTIONS   raw, validated transactions as loaded from CSV
--   FRAUD_SCORES       one scored row per transaction (the serving table)
--   FRAUD_SCORES_LOAD  staging table for the MERGE into FRAUD_SCORES
--
-- Run this once against your Snowflake database before the first load:
--   snowsql -f sql/schema.sql
-- or paste it into a Snowflake worksheet.
--
-- Note on constraints: Snowflake accepts PRIMARY KEY and UNIQUE but does not
-- enforce them (NOT NULL is enforced). They are declared here because they
-- document intent and inform the query optimiser. Uniqueness of
-- TRANSACTION_ID is enforced in the pipeline by the MERGE, not by the table.

-- ---------------------------------------------------------------------------
-- Raw layer
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS RAW_TRANSACTIONS (
    TRANSACTION_ID    VARCHAR(36)    NOT NULL,
    USER_ID           VARCHAR(16)    NOT NULL,
    EVENT_TIME        TIMESTAMP_NTZ  NOT NULL,
    AMOUNT            NUMBER(12,2)   NOT NULL,
    MERCHANT_ID       VARCHAR(64),
    MERCHANT_COUNTRY  VARCHAR(2),
    TXN_TYPE          VARCHAR(16),
    DEVICE_ID         VARCHAR(16),
    IP_ADDRESS        VARCHAR(45),
    LABEL             NUMBER(1)      NOT NULL,
    LOADED_AT         TIMESTAMP_NTZ  NOT NULL DEFAULT CURRENT_TIMESTAMP(),

    CONSTRAINT PK_RAW_TRANSACTIONS PRIMARY KEY (TRANSACTION_ID)
);

-- AMOUNT is NUMBER(12,2), not FLOAT: money needs exact decimal arithmetic.
-- Binary floating point cannot represent 0.10 exactly, and summing float
-- amounts over a 24-hour window accumulates error.
--
-- IP_ADDRESS is VARCHAR(45) so an IPv6 address would fit without a migration.
--
-- LABEL is the ground-truth fraud flag from the generator. It exists only
-- because this is synthetic data. In a real system it would arrive later,
-- from chargebacks or manual review, and would live in its own table with
-- its own timestamp.
--
-- EVENT_TIME is when the transaction happened. LOADED_AT is when this row
-- reached the warehouse. Keeping both is what makes backfills auditable.

-- ---------------------------------------------------------------------------
-- Scored layer
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS FRAUD_SCORES (
    TRANSACTION_ID    VARCHAR(36)    NOT NULL,
    USER_ID           VARCHAR(16)    NOT NULL,
    EVENT_TIME        TIMESTAMP_NTZ  NOT NULL,
    SCORE             FLOAT          NOT NULL,
    FLAGGED           BOOLEAN        NOT NULL,
    MODEL_VERSION     VARCHAR(64)    NOT NULL,
    SCORED_AT         TIMESTAMP_NTZ  NOT NULL,

    CONSTRAINT PK_FRAUD_SCORES PRIMARY KEY (TRANSACTION_ID)
);

-- USER_ID and EVENT_TIME are duplicated from RAW_TRANSACTIONS on purpose.
-- The dashboard's time series is grouped by EVENT_TIME; carrying it here
-- means the chart does not have to join back to RAW_TRANSACTIONS on every
-- render. This is a deliberate denormalisation of two immutable columns.
--
-- SCORE is FLOAT: it is a model output, not money.
--
-- EVENT_TIME is when the transaction happened; SCORED_AT is when the model
-- scored it. Both are needed: EVENT_TIME answers "when did fraud occur",
-- SCORED_AT answers "how long did we take to catch it".
--
-- All timestamps are TIMESTAMP_NTZ holding UTC. Timezone conversion happens
-- at the display boundary only, so there is exactly one place to get it wrong.

-- ---------------------------------------------------------------------------
-- Staging table for idempotent writes
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TRANSIENT TABLE FRAUD_SCORES_LOAD (
    TRANSACTION_ID    VARCHAR(36)    NOT NULL,
    USER_ID           VARCHAR(16)    NOT NULL,
    EVENT_TIME        TIMESTAMP_NTZ  NOT NULL,
    SCORE             FLOAT          NOT NULL,
    FLAGGED           BOOLEAN        NOT NULL,
    MODEL_VERSION     VARCHAR(64)    NOT NULL,
    SCORED_AT         TIMESTAMP_NTZ  NOT NULL
);

-- TRANSIENT because this table is truncated after every MERGE. It never needs
-- Snowflake's 7-day fail-safe storage, and transient tables do not pay for it.
--
-- The scoring job writes here, MERGEs into FRAUD_SCORES on TRANSACTION_ID,
-- then truncates. That is what makes re-running a window safe: a second run
-- updates the existing rows instead of inserting duplicates.

-- ---------------------------------------------------------------------------
-- Feature store
-- ---------------------------------------------------------------------------
-- Written by models/update_feature_store.py. As of this commit that script is
-- not wired into the scoring path and has never been run against a live
-- warehouse. The DDL is here so the schema is defined in one place.
CREATE TABLE IF NOT EXISTS FEATURE_STORE (
    USER_ID               VARCHAR(16)    NOT NULL,
    LAST_24H_TOTAL        NUMBER(12,2),
    LAST_24H_TXN_COUNT    NUMBER(9,0),
    LAST_24H_AVG_AMOUNT   NUMBER(12,2),
    UPDATED_AT            TIMESTAMP_NTZ  NOT NULL DEFAULT CURRENT_TIMESTAMP(),

    CONSTRAINT PK_FEATURE_STORE PRIMARY KEY (USER_ID)
);
