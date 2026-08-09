# Running the DAGs under Airflow 3

Status: not yet done. This is the procedure to follow, written before the
attempt so that the setup and Airflow are not being debugged at the same time.

Time budget: 6 hours, hard stop. Phase timings below add to roughly 4h30 with
slack. If phase 3 is not finished by the 3-hour mark, stop — the repository
already says the DAGs have been written but never run, and that stays true and
honest.

Everything runs from `airflow/` unless stated otherwise.

---

## Before the day

Two things that take minutes now and save an hour later.

**1. Give Docker enough memory.** Airflow's own init container warns below
4GB and the scheduler becomes unreliable. Docker Desktop → Settings →
Resources → Memory: **at least 4GB**, 6GB if the machine has 16GB.

**2. Check the Snowflake trial is still alive**, since the DAGs write to it:

```bash
.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'.');import snowflake.connector as s;from config import snowflake_args;c=s.connect(**snowflake_args());print('trial alive');c.close()"
```

Run that from the repository root. If it fails, there is no point starting —
the DAGs would run but every warehouse task would fail, and the evidence would
prove nothing.

---

## Phase 1 — Docker Desktop (30–60 min)

1. Install Docker Desktop for Windows from docker.com. Take the **WSL2
   backend** option when offered; it is the default and the Hyper-V backend
   is slower and more awkward.
2. Reboot when it asks. It will ask.
3. If it complains the WSL2 kernel is out of date, run in PowerShell as
   administrator:

```bash
wsl --update
```

4. Confirm it works before going further:

```bash
docker run --rm hello-world
```

If that prints a greeting, Docker is fine and any later problem is Airflow's.
That distinction is worth having.

**Checkpoint:** `docker --version` and `docker compose version` both answer.

---

## Phase 2 — Configure (15 min)

Everything below is in `airflow/`.

**1. Create `airflow/.env`.** This is separate from the repository root
`.env`, and the difference matters:

- **`airflow/.env`** is read by docker compose itself, to fill in `${...}`
  placeholders in `docker-compose.yaml`.
- **the root `.env`** is mounted into the containers by `env_file:` and holds
  the Snowflake credentials.

Two mechanisms, two files. Putting Snowflake credentials in `airflow/.env`
would not reach the tasks, and putting `AIRFLOW_UID` in the root one would not
reach compose.

Create `airflow/.env` with exactly this:

```
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
AIRFLOW_JWT_SECRET=local-development-not-a-secret
```

It is gitignored by the root `.gitignore` rule, which matches `.env` at any
depth.

**2. Create the directories compose expects to mount:**

```bash
mkdir logs config plugins
```

**3. Confirm compose can read the file and resolve everything:**

```bash
docker compose config
```

This renders the fully resolved configuration without starting anything. If a
variable is unset you get a warning here rather than a mystery later. Check in
the output that `env_file` resolves to the repository root `.env` and that the
`/project` mount points at the repository.

**Checkpoint:** `docker compose config` prints YAML with no warnings.

---

## Phase 3 — Build and start (30–90 min, mostly waiting)

```bash
docker compose build
```

First build pulls the ~1.5GB Airflow image and installs the project
dependencies into it. Ten to twenty minutes on a normal connection. It only
happens once.

```bash
docker compose up -d
```

Then watch it settle:

```bash
docker compose ps
```

You want `postgres`, `airflow-apiserver`, `airflow-scheduler`,
`airflow-dag-processor` and `airflow-triggerer` all `running`, most of them
`(healthy)`, and `airflow-init` `exited (0)`. Healthchecks have a 30-second
start period, so give it two minutes before concluding anything.

If something is not up:

```bash
docker compose logs airflow-init
docker compose logs airflow-scheduler --tail 50
```

`airflow-init` is the one to read first. It runs before everything else, and
if it failed the others never had a chance.

**Checkpoint:** http://localhost:8080 shows a login page. Log in with
`airflow` / `airflow`. Both DAGs appear in the list, paused, with no import
errors banner.

**This is the 3-hour decision point.** If the UI is not up by now, stop.

---

## Phase 4 — Run the DAGs (45–90 min)

The DAGs are created paused deliberately. Unpausing a `@daily` DAG makes the
scheduler start a run immediately, which you do not want while still checking
the stack.

### 4a. The hourly scoring DAG first

It is the simpler one — two tasks, no local file handling — so it fails in
fewer ways and tells you whether the container can reach Snowflake at all.

In the UI, open `fraud_hourly_scoring_pipeline` and use **Trigger DAG**
(leave it paused; a manual trigger runs regardless).

Watch `score_new_transactions`. Its log should show the model version, the
row count read from `RAW_TRANSACTIONS`, and the MERGE result.

Expect `MERGE into FRAUD_SCORES: (0, 29178)` — zero inserted, because the
rows are already there from the runs done outside Airflow. That is itself
evidence the MERGE is behaving.

`check_fraud_alerts` will fail unless `SLACK_WEBHOOK_URL` is set. That is
expected and fine; the repository has never claimed Slack works. Note it and
move on.

### 4b. The daily pipeline DAG

This one needs input, because `validate_data` reads `data/outbox` and the
outbox is empty — earlier runs archived everything.

From the repository root, on the host:

```bash
.venv\Scripts\python.exe etl\generate_transactions.py
```

That writes 30 files into `data/outbox`, which the container sees through the
`/project` mount. Then trigger `fraud_detection_daily_pipeline` in the UI.

Expect: `validate_data` 29,178 valid / 0 invalid, `normalise_to_processed`
30 files, `load_raw_transactions` a MERGE result, `train_model` a new
artifact.

### 4c. The second run — this is the one that matters

Regenerate with the same seed. The generator is deterministic, so the
transaction IDs are **identical** to the first run:

```bash
.venv\Scripts\python.exe etl\generate_transactions.py
```

Trigger `fraud_detection_daily_pipeline` again.

`load_raw_transactions` should now log `MERGE into RAW_TRANSACTIONS:
(0, 29178)` — nothing inserted, everything updated — and
`RAW_TRANSACTIONS now holds 29,178 rows`, unchanged.

That is the proof: **the same window ran twice under the scheduler and did
not duplicate a single row.** Almost no portfolio project demonstrates this,
and it is the single most valuable artifact of the whole exercise. Capture it
carefully.

---

## Phase 5 — Capture evidence (30 min)

See the checklist below. Then:

```bash
docker compose down
```

Add `-v` only if you want the Airflow metadata database gone too; without it,
a later `up` keeps your DAG run history.

---

## The four failures most likely to happen

Ordered by how likely they are. Each includes what it looks like, so it is
recognised rather than diagnosed.

### 1. Permission denied writing to logs

**Looks like:** `airflow-init` exits non-zero, or tasks fail immediately with
`PermissionError: [Errno 13]` on a path under `/opt/airflow/logs`.

**Cause:** `AIRFLOW_UID` unset, so compose defaults to 50000 while the
mounted host directories are owned by something else.

**Fix:** confirm `airflow/.env` exists and contains `AIRFLOW_UID=50000`, then:

```bash
docker compose down
docker compose up -d
```

If it persists, delete `logs/` on the host and let `airflow-init` recreate it.

### 2. DAG import error in the UI

**Looks like:** a red banner on the DAGs page, `ModuleNotFoundError: No
module named 'airflow.providers.standard'` or similar, and one or both DAGs
missing from the list.

**Cause:** the DAGs import `airflow.providers.standard.operators.bash`, which
is where `BashOperator` lives from Airflow 2.10 onward. It ships with the
3.0.3 image, so this should not happen — but if the image was overridden to
an older tag it will.

**Fix:** check `docker compose config` shows `apache/airflow:3.0.3` as the
Dockerfile base. To see the real error rather than the truncated banner:

```bash
docker compose logs airflow-dag-processor --tail 80
```

### 3. Tasks fail with a Snowflake authentication or "missing credentials" error

**Looks like:** the task log shows `Snowflake credentials are not set.
Missing: SNOW_USER, SNOW_PWD, SNOW_ACCOUNT`, or a 250001 authentication
failure.

**Cause:** two possibilities. Either `env_file: ../.env` did not resolve —
compose was run from the wrong directory — or the root `.env` picked up
Windows line endings from an editor, so the password arrives with a trailing
carriage return.

**Fix:** run compose from `airflow/`, not the repository root. Then confirm
what the container actually sees:

```bash
docker compose run --rm airflow-cli bash -c "env | grep SNOW_ | sed 's/=.*/=<set>/'"
```

That prints which variables are present without printing their values. If
`SNOW_PWD` is absent, it is the env_file path. If present but authentication
still fails, it is line endings — the file was LF-only when this runbook was
written, so it would have been changed since.

### 4. Tasks fail writing to /project/data

**Looks like:** `PermissionError` or `Read-only file system` on a path under
`/project/data/`, in `validate_data` or `normalise_to_processed`.

**Cause:** the container runs as UID 50000 and the repository lives on a
Windows filesystem surfaced through WSL2. Usually permissive, occasionally
not.

**Fix:** first check Docker Desktop → Settings → Resources → File Sharing
includes the drive. If it does and the error persists, the pragmatic
workaround is to run the file-handling stages on the host and let Airflow own
only the warehouse stages — the hourly scoring DAG needs no local writes at
all, so 4a still produces good evidence even if 4b cannot.

### Also possible, less likely

- **Out of memory.** The scheduler restarts in a loop, `docker compose ps`
  shows it cycling. Raise Docker's memory allocation to 6GB.
- **Port 8080 already in use.** `docker compose up` fails immediately with a
  bind error. Something else is on 8080; change the left-hand side of the
  ports mapping to `8081:8080`.

---

## Evidence checklist for `docs/verified-airflow.md`

Capture as you go, not at the end. The point is that this survives the
Snowflake trial expiring and the containers being deleted.

**1. The stack running**

```bash
docker compose ps
```

Paste the table. It shows the Airflow 3 service topology — `api-server`,
`dag-processor`, `triggerer` as separate services — which is itself evidence
you ran Airflow 3 rather than 2.

**2. Airflow version, from inside the container**

```bash
docker compose run --rm airflow-cli airflow version
```

**3. Grid view screenshot, both DAGs, all tasks green.** The one image that
communicates most. Include the DAG name and the run timestamps.

**4. Task log: `load_raw_transactions`, first run.** The lines showing
`staged 29,178 rows`, the MERGE tuple with 29,178 inserted, and
`RAW_TRANSACTIONS now holds 29,178 rows`.

**5. Task log: `load_raw_transactions`, second run — the important one.**
The same lines, showing `(0, 29178)` and the row count unchanged at 29,178.
Put these two side by side in the document. This is the artifact worth the
weekend.

**6. Task log: `score_new_transactions`.** Shows the model version and
threshold picked up from the artifact metadata, proving the scheduler-run
task used the same operating point as the measured evaluation.

**7. Snowflake query history, showing the MERGEs came from the container.**
Run in Snowsight after the DAG runs:

```sql
SELECT START_TIME, QUERY_TYPE, ROWS_INSERTED, ROWS_PRODUCED,
       LEFT(QUERY_TEXT, 40) AS QUERY
FROM TABLE(FRAUD_DB.INFORMATION_SCHEMA.QUERY_HISTORY(
     END_TIME_RANGE_START => DATEADD('hour', -6, CURRENT_TIMESTAMP()),
     RESULT_LIMIT => 200))
WHERE QUERY_TYPE = 'MERGE'
ORDER BY START_TIME DESC;
```

**8. What still did not work.** If `check_fraud_alerts` failed for want of a
webhook, write that down. A document that records only successes is the kind
this repository has spent its history removing.

Then update the README status table and the DESIGN.md limitation about
Airflow never having been run, and delete this line from the known-defects
list.

---

## If the six hours run out

Stop and change nothing else. The repository is already honest: the README
says the DAGs have never been executed, `DESIGN.md` limitation 3 says the
orchestration is a design rather than observed behaviour, and
`CV_BULLETS.md` tells you to say "wrote Airflow 3 DAGs" rather than
"orchestrated with Airflow".

That is a survivable position, and a far better one than a half-finished
Compose stack you would have to explain.
