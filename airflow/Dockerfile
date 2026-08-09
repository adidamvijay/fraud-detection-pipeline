# Airflow image with this project's Python dependencies baked in.
#
# The alternative is _PIP_ADDITIONAL_REQUIREMENTS in docker-compose.yaml,
# which the Airflow docs explicitly warn is for quick checks only. It
# reinstalls every package every time any container starts, which with
# snowflake-connector-python means several minutes per service on every
# `docker compose up`, and it turns a transient network failure into a
# service that will not start. Building once is faster and removes a whole
# class of failure.
#
# Build context is this directory, so requirements.txt below is
# airflow/requirements.txt.

FROM apache/airflow:3.0.3

COPY requirements.txt /tmp/airflow-project-requirements.txt

# --no-cache-dir keeps the image smaller; the constraint is that pip must
# resolve against the versions Airflow's image already pins, so install as
# the airflow user rather than root.
USER airflow
RUN pip install --no-cache-dir -r /tmp/airflow-project-requirements.txt
