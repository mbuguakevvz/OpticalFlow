# orchestration/pipeline_flow.py

import subprocess
import sys
import os
import duckdb
from datetime import datetime, timedelta

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON       = sys.executable
DB_PATH      = os.path.join(PROJECT_ROOT, "data", "opticalflow.duckdb")
DBT_DIR      = os.path.join(PROJECT_ROOT, "dbt_models")
DBT_EXE      = os.path.join(os.path.dirname(PYTHON), "dbt.exe")
UTF8_ENV     = {**os.environ, "PYTHONIOENCODING": "utf-8"}


# ──────────────────────────────────────────
# TASKS
# ──────────────────────────────────────────
@task(
    name                = "Generate Mock Data",
    description         = "Generates fresh supplier, inventory and shipment CSVs",
    retries             = 2,
    retry_delay_seconds = 10,
    cache_key_fn        = task_input_hash,
    cache_expiration    = timedelta(hours=1),
)
def generate_data():
    logger = get_run_logger()
    logger.info("Generating mock data...")

    result = subprocess.run(
        [PYTHON, os.path.join(PROJECT_ROOT, "ingestion", "generate_mock_data.py")],
        capture_output = True,
        text           = True,
        cwd            = PROJECT_ROOT,
        encoding       = "utf-8",
        env            = UTF8_ENV,
    )
    logger.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Data generation failed:\n{result.stderr}")

    logger.info("Mock data generated successfully")
    return True


@task(
    name                = "Ingest to DuckDB",
    description         = "Loads raw CSVs into DuckDB raw schema",
    retries             = 2,
    retry_delay_seconds = 10,
)
def ingest_to_duckdb(data_generated: bool):
    logger = get_run_logger()
    logger.info("Ingesting data into DuckDB...")

    result = subprocess.run(
        [PYTHON, os.path.join(PROJECT_ROOT, "ingestion", "load_to_duckdb.py")],
        capture_output = True,
        text           = True,
        cwd            = PROJECT_ROOT,
        encoding       = "utf-8",
        env            = UTF8_ENV,
    )
    logger.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Ingestion failed:\n{result.stderr}")

    logger.info("Ingestion complete")
    return True


@task(
    name                = "Run dbt Transformations",
    description         = "Runs dbt models: raw -> staging -> marts",
    retries             = 1,
    retry_delay_seconds = 15,
)
def run_dbt(ingestion_done: bool):
    logger = get_run_logger()
    logger.info("Running dbt transformations...")

    result = subprocess.run(
        [DBT_EXE, "run", "--profiles-dir", "."],
        capture_output = True,
        text           = True,
        cwd            = DBT_DIR,
        encoding       = "utf-8",
        env            = UTF8_ENV,
    )
    logger.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"dbt run failed:\n{result.stderr}")

    logger.info("dbt transformations complete")
    return True


@task(
    name                = "Run ML Risk Pipeline",
    description         = "Trains model and scores supplier disruption risk",
    retries             = 1,
    retry_delay_seconds = 15,
)
def run_ml_pipeline(dbt_done: bool):
    logger = get_run_logger()
    logger.info("Running ML disruption risk pipeline...")

    result = subprocess.run(
        [PYTHON, os.path.join(PROJECT_ROOT, "ml_pipeline", "disruption_predictor.py")],
        capture_output = True,
        text           = True,
        cwd            = PROJECT_ROOT,
        encoding       = "utf-8",
        env            = UTF8_ENV,
    )
    logger.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"ML pipeline failed:\n{result.stderr}")

    logger.info("ML pipeline complete")
    return True


@task(
    name        = "Pipeline Health Check",
    description = "Validates all tables exist and have data",
)
def health_check(ml_done: bool):
    logger = get_run_logger()
    logger.info("Running pipeline health check...")

    con = duckdb.connect(DB_PATH, read_only=True)

    checks = {
        "raw.raw_suppliers"                : "SELECT COUNT(*) FROM raw.raw_suppliers",
        "raw.raw_shipments"                : "SELECT COUNT(*) FROM raw.raw_shipments",
        "raw.raw_inventory"                : "SELECT COUNT(*) FROM raw.raw_inventory",
        "predictions.supplier_risk_scores" : "SELECT COUNT(*) FROM predictions.supplier_risk_scores",
    }

    all_passed = True
    for name, query in checks.items():
        try:
            count  = con.execute(query).fetchone()[0]
            status = "PASS" if count > 0 else "FAIL"
            if count == 0:
                all_passed = False
            logger.info(f"  [{status}] {name} -> {count} rows")
        except Exception as e:
            logger.error(f"  [FAIL] {name} -> ERROR: {e}")
            all_passed = False

    con.close()

    if not all_passed:
        raise Exception("Health check failed — one or more tables are empty or missing")

    logger.info("All health checks passed")
    return {
        "status"    : "SUCCESS",
        "timestamp" : datetime.utcnow().isoformat(),
        "checks"    : len(checks),
    }


# ──────────────────────────────────────────
# MAIN FLOW
# ──────────────────────────────────────────
@flow(
    name        = "OpticalFlow Daily Pipeline",
    description = "End-to-end supply chain pipeline: ingest -> transform -> predict",
    log_prints  = True,
)
def opticalflow_pipeline():
    logger = get_run_logger()
    logger.info("==================================================")
    logger.info("  OpticalFlow Pipeline Starting")
    logger.info(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info("==================================================")

    data_ready  = generate_data()
    ingest_done = ingest_to_duckdb(data_ready)
    dbt_done    = run_dbt(ingest_done)
    ml_done     = run_ml_pipeline(dbt_done)
    result      = health_check(ml_done)

    logger.info("==================================================")
    logger.info("  Pipeline completed successfully!")
    logger.info("==================================================")
    return result


if __name__ == "__main__":
    opticalflow_pipeline()